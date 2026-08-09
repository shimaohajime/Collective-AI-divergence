#!/usr/bin/env python3
"""
branching_entropy_v1.py — Branching entropy and local expansion estimation.

Implements Sections A, B, D, E from llm_discussion_rds_theory_metrics.tex:

  A. Empirical branching entropy Ĥ_K(x_t): at each state visited by a reference
     run, sample K independent next-step responses and measure stance diversity.
  B. Recurrence estimate μ̂(D): fraction of states in expansive region D̂.
  D. Conditional expansion ĉ_in vs ĉ_out.
  E. Certificate statistic Γ̂ = μ̂(D) · ĉ_in.

Usage (from project root):
    # First run chaos_run_v1.py to produce a JSONL, then:
    python src/branching_entropy_v1.py --runs-file data/raw/chaos_v1/IM-01__T0.7__N5__rolesTrue.jsonl
    python src/branching_entropy_v1.py --runs-file data/raw/chaos_v1/IM-01__T0.7__N5__rolesTrue.jsonl --k-samples 10 --rep-index 0
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from llm_dynamics_v1 import (
    DEFAULT_STATE, GLOBAL_RULES, ROUND_TEMPLATE,
    call_model, format_state_table, format_window,
    parse_agent_output, role_mandate_text, roles_for_n, update_window,
)
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

K_SAMPLES    = 10    # independent samples per state (Section A: K ∈ [20,50] ideal)
H0_THRESHOLD = 0.5   # entropy threshold defining D̂ (nats); ~midpoint of [0, log3≈1.1]
MAX_WORKERS  = 10    # concurrent API calls for K-sampling
MODEL        = "gpt-4.1-mini"
TRUNCATION   = "disabled"


# ── Prompt reconstruction ─────────────────────────────────────────────────────

def reconstruct_prompts(
    run_record: dict[str, Any],
    scenario_text: str,
    roles: list[str],
    k_window: int,
    rounds: int,
    roles_enabled: bool,
) -> list[tuple[int, str, str, str]]:
    """
    Replay the saved run turn-by-turn to reconstruct the exact (developer_rules,
    user_prompt) that was fed to the model at each (round_num, role) turn.

    Returns list of (round_num, role, developer_rules, user_prompt).
    State updates use the *saved* outputs so the reconstructed context matches
    the original run exactly.
    """
    role_order = ", ".join(roles)
    state_by_role: dict[str, dict[str, Any]] = {
        r: {
            "pref": list(DEFAULT_STATE["pref"]),
            "conf": DEFAULT_STATE["conf"],
            "tags": list(DEFAULT_STATE["tags"]),
        }
        for r in roles
    }
    window_messages: list[tuple[int, str, str]] = []

    # Index saved turns for fast lookup
    turn_index: dict[tuple[int, str], dict[str, Any]] = {}
    for turn in run_record["run"]["turns"]:
        turn_index[(turn["round_num"], turn["role"])] = turn

    prompts = []
    for rr in range(1, rounds + 1):
        for role in roles:
            state_table = format_state_table(state_by_role, roles)
            window_str  = format_window(window_messages)

            extra = (
                "In this final round, clearly state which option you currently "
                "favor and name the single strongest remaining uncertainty."
                if rr == rounds else ""
            )
            user_prompt = ROUND_TEMPLATE.format(
                scenario=scenario_text,
                role_order=role_order,
                state_table=state_table,
                window=window_str,
                role=role,
                round_num=rr,
                k=k_window,
                extra_round_instructions=extra,
            )
            if roles_enabled:
                mandate = role_mandate_text(role)
                if mandate:
                    user_prompt = mandate + "\n\n" + user_prompt

            prompts.append((rr, role, GLOBAL_RULES, user_prompt))

            # Advance state using saved outputs (mirrors original run exactly)
            saved = turn_index.get((rr, role))
            if saved:
                state_by_role[role] = {
                    "pref": saved["pref"],
                    "conf": saved["conf"],
                    "tags": saved["tags"],
                }
                window_messages = update_window(
                    window_messages, (rr, role, saved["argument"]), k_window
                )

    return prompts


# ── K-sampling at one state ────────────────────────────────────────────────────

def sample_state(
    client: OpenAI,
    developer_rules: str,
    user_prompt: str,
    temperature: float,
    k: int,
    max_workers: int,
) -> list[list[float]]:
    """
    Call the model K times independently from the same prompt.
    Returns list of K pref vectors (each [pA, pB, pC]).
    Failed parses are silently dropped; caller checks len(result).
    """
    def _one_call(_: int) -> list[float] | None:
        try:
            text, _ = call_model(
                client=client,
                model=MODEL,
                temperature=temperature,
                developer_rules=developer_rules,
                user_prompt=user_prompt,
                truncation=TRUNCATION,
            )
            _, pref, _, _ = parse_agent_output(text)
            return list(pref)
        except Exception:
            return None

    prefs = []
    with ThreadPoolExecutor(max_workers=min(max_workers, k)) as ex:
        futures = {ex.submit(_one_call, i): i for i in range(k)}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                prefs.append(result)
    return prefs


# ── Branching entropy metrics ─────────────────────────────────────────────────

def stance_label(pref: list[float]) -> int:
    """0=A, 1=B, 2=C based on argmax of pref."""
    return int(np.argmax(pref))


def branching_entropy(prefs: list[list[float]]) -> float:
    """
    Ĥ_K(x) = -Σ p̂_m log p̂_m over stance labels {A,B,C}.
    Uses the discrete label map ℓ = argmax(pref) as in Section A.
    """
    if not prefs:
        return float("nan")
    labels = [stance_label(p) for p in prefs]
    counts = np.bincount(labels, minlength=3).astype(float)
    freqs  = counts / counts.sum()
    return float(-np.sum(freqs[freqs > 0] * np.log(freqs[freqs > 0])))


def pref_variance(prefs: list[list[float]]) -> float:
    """Mean variance across pref dimensions — continuous expansion proxy."""
    if len(prefs) < 2:
        return float("nan")
    arr = np.array(prefs)  # (K, 3)
    return float(arr.var(axis=0).mean())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="branching_entropy_v1: estimate Ĥ_K, μ̂(D), and Γ̂ per state"
    )
    parser.add_argument("--runs-file",    required=True, help="JSONL from chaos_run_v1.py")
    parser.add_argument("--scenarios-file", default="src/ai_chaos_scenarios_v1.json")
    parser.add_argument("--rep-index",    type=int, default=0,
                        help="Which replicate to use as the reference trajectory (default: 0)")
    parser.add_argument("--k-samples",   type=int, default=K_SAMPLES,
                        help=f"K independent samples per state (default: {K_SAMPLES})")
    parser.add_argument("--h0",          type=float, default=H0_THRESHOLD,
                        help=f"Entropy threshold H0 for D̂ (default: {H0_THRESHOLD} nats)")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--out-file",    default=None,
                        help="Optional JSONL to save per-state results")
    args = parser.parse_args()

    # Load runs
    runs_path = Path(args.runs_file)
    if not runs_path.exists():
        print(f"ERROR: {runs_path} not found")
        sys.exit(1)
    runs_data = []
    with runs_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                runs_data.append(json.loads(line))

    if args.rep_index >= len(runs_data):
        print(f"ERROR: rep-index {args.rep_index} out of range (have {len(runs_data)} runs)")
        sys.exit(1)

    ref_run = runs_data[args.rep_index]
    cond    = ref_run["run"]["condition"]
    sid     = ref_run["scenario_id"]
    temp    = cond["temperature"]
    n_agents = cond["n_agents"]
    k_window = cond["k_window"]
    rounds   = cond["rounds"]
    roles_enabled = cond["roles_enabled"]

    # Load scenario text
    scenarios_path = Path(args.scenarios_file)
    scenario_texts = json.loads(scenarios_path.read_text(encoding="utf-8"))
    scenario_map   = {s["scenario_id"]: s["text"] for s in scenario_texts["scenarios"]}
    scenario_text  = scenario_map[sid]

    roles = roles_for_n(n_agents)
    if not roles_enabled:
        role_set = sorted({t["role"] for t in ref_run["run"]["turns"]})
        roles = role_set

    print(f"branching_entropy_v1  |  {sid}  temp={temp}  N={n_agents}  roles={roles_enabled}")
    print(f"Reference replicate: {args.rep_index}  |  K={args.k_samples}  H0={args.h0} nats")
    print(f"States to probe: {rounds} rounds × {n_agents} agents = {rounds * n_agents}")
    print(f"Total API calls: {rounds * n_agents * args.k_samples}")
    print()

    prompts = reconstruct_prompts(
        ref_run, scenario_text, roles, k_window, rounds, roles_enabled
    )

    client = OpenAI()
    state_results = []

    for idx, (rr, role, dev_rules, user_prompt) in enumerate(prompts):
        print(f"  [{idx+1:3d}/{len(prompts)}]  round={rr:02d}  role={role:<12} sampling K={args.k_samples}...",
              end="", flush=True)
        t0 = time.time()

        prefs = sample_state(
            client=client,
            developer_rules=dev_rules,
            user_prompt=user_prompt,
            temperature=temp,
            k=args.k_samples,
            max_workers=args.max_workers,
        )

        H  = branching_entropy(prefs)
        V  = pref_variance(prefs)
        in_D = H >= args.h0

        # Record the original pref from the reference run for comparison
        orig_pref = None
        for t in ref_run["run"]["turns"]:
            if t["round_num"] == rr and t["role"] == role:
                orig_pref = t["pref"]
                break

        print(f"  Ĥ={H:.3f}  var={V:.4f}  D̂={'YES' if in_D else 'no '}  {time.time()-t0:.1f}s")

        state_results.append({
            "round_num": rr,
            "role": role,
            "H_K": H,
            "pref_var": V,
            "in_D": in_D,
            "n_valid": len(prefs),
            "orig_pref": orig_pref,
            "sample_prefs": prefs,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = [s for s in state_results if not math.isnan(s["H_K"])]
    in_D  = [s for s in valid if s["in_D"]]
    out_D = [s for s in valid if not s["in_D"]]

    mu_D    = len(in_D) / len(valid) if valid else float("nan")
    c_in    = float(np.mean([s["H_K"] for s in in_D]))  if in_D  else float("nan")
    c_out   = float(np.mean([s["H_K"] for s in out_D])) if out_D else float("nan")
    gamma   = mu_D * c_in if not math.isnan(mu_D) and not math.isnan(c_in) else float("nan")

    print()
    print("=" * 72)
    print("BRANCHING ENTROPY REPORT")
    print("=" * 72)

    print(f"\n[A] Branching entropy Ĥ_K(x_t) — per round (averaged over agents):")
    print(f"    {'Round':<8} {'Mean Ĥ':>8}  {'|'}")
    for rr in range(1, rounds + 1):
        round_states = [s for s in valid if s["round_num"] == rr]
        if round_states:
            mean_H = np.mean([s["H_K"] for s in round_states])
            max_H  = math.log(3)  # max entropy for 3-label system
            bar_len = int(40 * mean_H / max_H)
            bar = "█" * bar_len
            in_d_count = sum(1 for s in round_states if s["in_D"])
            print(f"    {rr:<8} {mean_H:>8.3f}  {bar}  [{in_d_count}/{len(round_states)} in D̂]")

    print(f"\n[B] Recurrence estimate:")
    print(f"    μ̂(D̂) = {len(in_D)}/{len(valid)} = {mu_D:.3f}  (H0={args.h0} nats)")

    print(f"\n[D] Conditional expansion (using Ĥ as expansion proxy):")
    print(f"    ĉ_in  = {c_in:.3f} nats   (mean Ĥ | x ∈ D̂)")
    print(f"    ĉ_out = {c_out:.3f} nats   (mean Ĥ | x ∉ D̂)")
    print(f"    ĉ_in > ĉ_out: {'YES ✓' if c_in > c_out else 'NO'}")

    print(f"\n[E] Certificate statistic:")
    print(f"    Γ̂ = μ̂(D̂) × ĉ_in = {mu_D:.3f} × {c_in:.3f} = {gamma:.4f}")
    print(f"    Γ̂ > 0: {'YES ✓ — supports λ₁ > 0' if gamma > 0 else 'NO'}")

    print(f"\n[Role breakdown] Mean Ĥ per role across all rounds:")
    for role in roles:
        role_states = [s for s in valid if s["role"] == role]
        if role_states:
            mean_H = np.mean([s["H_K"] for s in role_states])
            in_d   = sum(1 for s in role_states if s["in_D"])
            print(f"    {role:<12}  Ĥ={mean_H:.3f}  in_D={in_d}/{len(role_states)}")

    # Optional save
    if args.out_file:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "scenario_id": sid,
            "temperature": temp,
            "n_agents": n_agents,
            "roles_enabled": roles_enabled,
            "rep_index": args.rep_index,
            "k_samples": args.k_samples,
            "h0": args.h0,
            "mu_D": mu_D,
            "c_in": c_in,
            "c_out": c_out,
            "gamma": gamma,
            "states": state_results,
        }
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved full results → {out_path}")


if __name__ == "__main__":
    main()
