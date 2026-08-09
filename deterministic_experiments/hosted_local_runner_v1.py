#!/usr/bin/env python3
"""
hosted_local_runner_v1.py

Deterministic self-hosted committee runner for open-weight models.

This module reuses the prompt suite and parsing logic from src/llm_dynamics_v1.py,
but replaces the API call path with a single-process Hugging Face Transformers
backend suitable for the hosted deterministic experiments.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_dynamics_v1 import (  # noqa: E402
    BALLOT_DEV_RULES,
    BALLOT_TEMPLATE,
    CLERK_DEV_RULES,
    CLERK_TEMPLATE,
    GLOBAL_RULES,
    ROUND_TEMPLATE,
    Condition,
    Run,
    Turn,
    add_usage,
    init_state_by_role,
    parse_agent_output,
    parse_json_with_retry_hint,
    role_mandate_text,
    roles_for_n,
    format_state_table,
    format_window,
    update_window,
)


def configure_global_determinism(seed: int = 0) -> None:
    """Pin common deterministic settings before model execution."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["PYTHONHASHSEED"] = str(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_hash(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(canonical)


def normalize_agent_output(text: str) -> str:
    """
    Split a collapsed ARGUMENT/STATE response into the format expected by
    parse_agent_output().
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if any(line.lstrip().startswith("STATE:") for line in lines):
        return text
    idx = text.rfind("STATE:")
    if idx == -1:
        return text
    return text[:idx].rstrip() + "\n" + text[idx:].lstrip()


_RELAXED_STATE_RE = re.compile(
    r'^STATE:\s*pref=\[([0-9.%]+),([0-9.%]+),([0-9.%]+)\]'
    r'(?:[;\s]+conf=(\d{1,3}))?'
    r'(?:[;\s]+tags=\["([a-zA-Z0-9_-]+)","([a-zA-Z0-9_-]+)"\])?.*$'
)
_FULL_TAGS_RE = re.compile(r'tags=\["([a-zA-Z0-9_-]+)","([a-zA-Z0-9_-]+)"\]')
_PREF_ONLY_RE = re.compile(r'pref=\[(\d+\.\d+),(\d+\.\d+),(\d+\.\d+)\]')
_PREF_FLEX_RE = re.compile(r'pref=\[([^\]]+)\]')
_CONF_ONLY_RE = re.compile(r'conf=(\d{1,3})')


def _normalize_pref_triplet(raw_pref: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Renormalize a positive preference triplet into a valid rounded simplex point.

    Hosted runs occasionally emit rounded values that do not exactly sum to 1.00.
    We preserve the relative weights as long as the triplet is nonnegative and has
    positive mass, then deterministically force the rounded total back to 1.00.
    """
    if any(x < 0 for x in raw_pref):
        raise ValueError(f"pref contains negative entry: {raw_pref}")
    pref_sum = sum(raw_pref)
    if pref_sum <= 0:
        raise ValueError(f"pref sum must be positive: {raw_pref} sum={pref_sum}")

    pref = tuple(round(x / pref_sum, 2) for x in raw_pref)
    rounded_sum = round(sum(pref), 2)
    if rounded_sum != 1.00:
        pref = (pref[0], pref[1], round(1.00 - pref[0] - pref[1], 2))
    if any(x < 0 for x in pref):
        raise ValueError(f"rounded pref contains negative entry: {pref}")
    return pref


def _coerce_pref_token(token: str) -> float:
    """
    Parse a single pref token, allowing small-model near-misses like `25`
    where `0.25` was almost certainly intended.
    """
    tok = token.strip()
    if not tok:
        raise ValueError("empty pref token")
    has_percent = tok.endswith("%")
    if has_percent:
        tok = tok[:-1].strip()
        if not tok:
            raise ValueError("empty percent pref token")
    value = float(tok)
    if has_percent:
        return value / 100.0
    if "." not in tok and 1 < value <= 100:
        return value / 100.0
    return value


def _parse_pref_triplet_flex(state_line: str) -> tuple[float, float, float]:
    """
    Recover a pref triplet from a less strictly formatted STATE line.
    """
    pref_match = _PREF_FLEX_RE.search(state_line)
    if pref_match is None:
        raise ValueError(f"Could not salvage pref from STATE line: {state_line}")
    raw_tokens = [tok.strip() for tok in pref_match.group(1).split(",")]
    if len(raw_tokens) != 3:
        raise ValueError(f"Expected 3 pref tokens, got {len(raw_tokens)} in: {state_line}")
    raw_pref = tuple(_coerce_pref_token(tok) for tok in raw_tokens)
    return _normalize_pref_triplet(raw_pref)  # type: ignore[arg-type]


def _parse_agent_output_relaxed(text: str) -> tuple[str, tuple[float, float, float], int, tuple[str, str]]:
    """
    Hosted-model fallback parser for shortened STATE lines.

    Some local generations emit a valid preference vector but omit `conf` and/or
    `tags`. We keep the recorded preference trajectory and backfill the missing
    metadata with deterministic defaults.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty model output")

    state_line = lines[-1]
    m = _RELAXED_STATE_RE.match(state_line)
    if not m:
        raise ValueError(f"Malformed relaxed STATE line: {state_line}")

    raw_pref = (
        _coerce_pref_token(m.group(1)),
        _coerce_pref_token(m.group(2)),
        _coerce_pref_token(m.group(3)),
    )
    pref = _normalize_pref_triplet(raw_pref)

    conf = int(m.group(4)) if m.group(4) is not None else 50
    if not (0 <= conf <= 100):
        raise ValueError(f"conf out of range: {conf}")

    tags_match = _FULL_TAGS_RE.search(state_line)
    if tags_match is not None:
        tags = (tags_match.group(1), tags_match.group(2))
    else:
        tags = ("fallback", "fallback")

    arg = "\n".join(lines[:-1]).strip()
    if not arg:
        raise ValueError("ARGUMENT part empty (STATE present but no text above)")

    return arg, pref, conf, tags


def _parse_agent_output_salvage(text: str) -> tuple[str, tuple[float, float, float], int, tuple[str, str]]:
    """
    Last-resort hosted parser for badly truncated STATE lines.

    This path only requires that the final line still contain a parseable
    `pref=[...]` triple, and optionally `conf=NN`. Any broken or partial `tags`
    content is ignored and replaced with fallback values.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("Not enough content to salvage agent output")

    state_line = lines[-1]
    pref_match = _PREF_ONLY_RE.search(state_line)
    if pref_match is not None:
        raw_pref = tuple(float(pref_match.group(i)) for i in range(1, 4))
        pref = _normalize_pref_triplet(raw_pref)
    else:
        pref = _parse_pref_triplet_flex(state_line)

    conf_match = _CONF_ONLY_RE.search(state_line)
    conf = int(conf_match.group(1)) if conf_match is not None else 50
    if not (0 <= conf <= 100):
        raise ValueError(f"salvaged conf out of range: {conf}")

    arg = "\n".join(lines[:-1]).strip()
    if not arg:
        raise ValueError("ARGUMENT part empty after salvage")

    return arg, pref, conf, ("fallback", "fallback")


def parse_agent_output_with_retry(
    backend: "HostedDeterministicBackend",
    *,
    base_prompt: str,
    initial_output: str,
    initial_usage: Dict[str, int],
    max_new_tokens: int,
) -> tuple[str, tuple[float, float, float], int, tuple[str, str], Dict[str, int]]:
    """
    Parse agent output, retrying with a targeted formatting repair instruction if
    the model returns an invalid STATE line.
    """
    usage_total = dict(initial_usage)
    out_text = initial_output

    for attempt in range(3):
        try:
            argument, pref, conf, tags = parse_agent_output(normalize_agent_output(out_text))
            return argument, pref, conf, tags, usage_total
        except Exception as exc:
            try:
                argument, pref, conf, tags = _parse_agent_output_relaxed(normalize_agent_output(out_text))
                return argument, pref, conf, tags, usage_total
            except Exception:
                try:
                    argument, pref, conf, tags = _parse_agent_output_salvage(normalize_agent_output(out_text))
                    return argument, pref, conf, tags, usage_total
                except Exception:
                    pass
            if attempt == 2:
                raise ValueError(f"Failed to parse agent output after retries. Last output: {out_text!r}") from exc
            repair_prompt = (
                base_prompt
                + "\n\nRETRY INSTRUCTION: Your previous output had an invalid STATE line."
                + " Re-output the full response in exactly two parts: ARGUMENT on one or more lines,"
                + " then a final STATE line with exactly 3 preference values that sum to 1.00."
                + ' The STATE format must be exactly: STATE: pref=[pA,pB,pC]; conf=NN; tags=["tag1","tag2"]'
            )
            out_text, retry_usage = backend.generate(
                GLOBAL_RULES,
                repair_prompt,
                max_new_tokens=max_new_tokens,
            )
            add_usage(usage_total, retry_usage)


HOSTED_CLERK_CONCISION_RULES = """
HOSTED CONCISION RULES:
- Keep principle to at most 12 words.
- Keep each top_reasons entry to at most 8 words.
- Keep each case_applications value to at most 10 words.
- Keep dissent to at most 10 words.
- Keep uncertainty to at most 10 words.
- Prefer short noun phrases over explanations.
- Output JSON only.
""".strip()


def parse_final_decision_with_retry(
    backend: "HostedDeterministicBackend",
    *,
    clerk_prompt: str,
    max_new_tokens: int,
) -> tuple[Dict[str, Any], Dict[str, int]]:
    """
    Parse clerk output, retrying with stricter formatting prompts when needed.

    Smaller hosted models can otherwise overproduce free-text fields, causing
    the final JSON object to be truncated before closing braces are emitted.
    """
    usage_total: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    out_text = ""
    retry_suffixes = [
        "",
        "\n\nRETRY INSTRUCTION: Output one valid JSON object only. Keep every free-text field extremely short.",
        (
            "\n\nRETRY INSTRUCTION: Recompute the FINAL GROUP DECISION JSON from the ballots. "
            "Keep principle, dissent, uncertainty, and each case application very short. "
            "No markdown. No prose. Output exactly one valid JSON object."
        ),
    ]

    for suffix in retry_suffixes:
        prompt = clerk_prompt if not suffix else clerk_prompt + suffix
        out_text, usage = backend.generate(
            CLERK_DEV_RULES,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        add_usage(usage_total, usage)
        parsed = parse_json_with_retry_hint(out_text)
        if isinstance(parsed, dict):
            return parsed, usage_total

    raise ValueError(f"Failed to parse final decision JSON. Last output: {out_text!r}")


class HostedDeterministicBackend:
    """Single-process deterministic text generation wrapper."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 220,
        seed: int = 0,
    ) -> None:
        configure_global_determinism(seed=seed)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype '{dtype}'. Use one of {sorted(dtype_map)}.")

        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.dtype_name = dtype
        self.dtype = dtype_map[dtype]
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.generation_use_cache = "phi" not in model_name.lower()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(device)
        if not self.generation_use_cache:
            self.model.config.use_cache = False
        self.model.eval()

    def metadata(self) -> Dict[str, Any]:
        import torch
        import transformers

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        return {
            "model_name": self.model_name,
            "model_revision": self.revision,
            "device": self.device,
            "dtype": self.dtype_name,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "generation_use_cache": self.generation_use_cache,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gpu_name": gpu_name,
            "cuda_available": torch.cuda.is_available(),
        }

    def generate(self, developer_rules: str, user_prompt: str, *, max_new_tokens: Optional[int] = None) -> Tuple[str, Dict[str, int]]:
        import torch

        messages = [
            {"role": "system", "content": developer_rules},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=self.generation_use_cache,
            )

        full_ids = outputs[0]
        new_ids = full_ids[input_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        usage = {
            "input_tokens": input_len,
            "output_tokens": int(new_ids.shape[0]),
            "total_tokens": input_len + int(new_ids.shape[0]),
        }
        return text, usage


def run_hosted_ws_committee(
    scenario_id: str,
    scenario_text: str,
    condition: Condition,
    backend: HostedDeterministicBackend,
    *,
    protocol_variant: str = "ws",
    role_overrides: Optional[Dict[str, str]] = None,
    round_max_new_tokens: int = 220,
    ballot_max_new_tokens: int = 120,
    clerk_max_new_tokens: int = 520,
    agent_labels: Optional[list[str]] = None,
) -> Run:
    """Hosted deterministic analogue of run_ws_committee()."""

    if protocol_variant not in {"ws", "no_feedback", "one_shot"}:
        raise ValueError(f"Unsupported protocol_variant '{protocol_variant}'.")

    roles = list(agent_labels) if agent_labels is not None else roles_for_n(condition.n_agents)
    if len(roles) != condition.n_agents:
        raise ValueError(f"agent_labels length {len(roles)} does not match n_agents={condition.n_agents}")
    role_order = ", ".join(roles)
    total_rounds = 1 if protocol_variant == "one_shot" else condition.rounds
    use_transcript_feedback = protocol_variant == "ws"

    state_by_role = init_state_by_role(roles)
    window_messages: list[tuple[int, str, str]] = []
    turns: list[Turn] = []
    ballots: list[dict[str, Any]] = []
    usage_total: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for rr in range(1, total_rounds + 1):
        for role in roles:
            state_table = format_state_table(state_by_role, roles)
            window_str = format_window(window_messages) if use_transcript_feedback else ""

            extra = ""
            if rr == total_rounds:
                extra = (
                    "In this final round, clearly state which option you currently favor "
                    "and name the single strongest remaining uncertainty."
                )

            user_prompt = ROUND_TEMPLATE.format(
                scenario=scenario_text,
                role_order=role_order,
                state_table=state_table,
                window=window_str,
                role=role,
                round_num=rr,
                k=condition.k_window,
                extra_round_instructions=extra,
            )

            if condition.roles_enabled:
                if role_overrides is not None and role in role_overrides:
                    mandate = role_overrides[role]
                else:
                    mandate = role_mandate_text(role)
                if mandate:
                    user_prompt = mandate + "\n\n" + user_prompt

            out_text, usage = backend.generate(
                GLOBAL_RULES,
                user_prompt,
                max_new_tokens=round_max_new_tokens,
            )
            argument, pref, conf, tags, parsed_usage = parse_agent_output_with_retry(
                backend,
                base_prompt=user_prompt,
                initial_output=out_text,
                initial_usage=usage,
                max_new_tokens=round_max_new_tokens,
            )
            add_usage(usage_total, parsed_usage)

            state_by_role[role] = {"pref": pref, "conf": conf, "tags": tags}
            if use_transcript_feedback:
                window_messages = update_window(window_messages, (rr, role, argument), condition.k_window)

            turns.append(
                Turn(
                    round_num=rr,
                    role=role,
                    argument=argument,
                    pref=pref,
                    conf=conf,
                    tags=tags,
                    usage=usage,
                )
            )

    state_table = format_state_table(state_by_role, roles)
    window_str = format_window(window_messages) if use_transcript_feedback else ""

    for role in roles:
        ballot_prompt = BALLOT_TEMPLATE.format(
            role=role,
            scenario_id=scenario_id,
            state_table=state_table,
            window=window_str,
            k=condition.k_window,
        )
        if condition.roles_enabled:
            mandate = role_mandate_text(role)
            if mandate:
                ballot_prompt = mandate + "\n\n" + ballot_prompt

        parsed_ballot: Optional[Dict[str, Any]] = None
        retry_prompt = ballot_prompt
        for _ in range(3):
            out_text, usage = backend.generate(
                BALLOT_DEV_RULES,
                retry_prompt,
                max_new_tokens=ballot_max_new_tokens,
            )
            add_usage(usage_total, usage)
            parsed_ballot = parse_json_with_retry_hint(out_text)
            if parsed_ballot is not None:
                break
            retry_prompt = ballot_prompt + "\n\nRETRY INSTRUCTION: Output one-line valid JSON only, matching the schema exactly."
        if parsed_ballot is None:
            raise ValueError(f"Failed to parse ballot JSON for role={role}. Last output: {out_text!r}")
        ballots.append(parsed_ballot)

    ballots_text = "\n".join(json.dumps(b, ensure_ascii=False) for b in ballots)
    clerk_prompt = CLERK_TEMPLATE.format(
        scenario_id=scenario_id,
        ballots=ballots_text,
        state_table=state_table,
        window=window_str,
        k=condition.k_window,
    )
    clerk_prompt = clerk_prompt + "\n\n" + HOSTED_CLERK_CONCISION_RULES
    final_decision, clerk_usage = parse_final_decision_with_retry(
        backend,
        clerk_prompt=clerk_prompt,
        max_new_tokens=clerk_max_new_tokens,
    )
    add_usage(usage_total, clerk_usage)

    return Run(
        scenario_id=scenario_id,
        condition=condition,
        turns=turns,
        ballots=ballots,
        final_decision=final_decision,
        usage_total=usage_total,
    )


def run_to_canonical_record(run: Run, *, backend_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"run": asdict(run)}
    if backend_metadata is not None:
        payload["backend"] = backend_metadata
    return payload
