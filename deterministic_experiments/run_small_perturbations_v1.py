#!/usr/bin/env python3
"""
run_small_perturbations_v1.py

Run deterministic hosted committee deliberation across small scenario-text
perturbations and compute perturbation-based divergence metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hosted_local_runner_v1 import (
    HostedDeterministicBackend,
    canonical_json_hash,
    run_hosted_ws_committee,
)
from llm_dynamics_v1 import Condition, roles_for_n


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_variant_set(path: Path, scenario_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload["variant_sets"]:
        if item["scenario_id"] == scenario_id:
            return item
    raise KeyError(f"Scenario '{scenario_id}' not found in {path}")


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def committee_mean_trajectory(run: dict[str, Any], roles: list[str], rounds: int | None = None) -> np.ndarray:
    pref_map: dict[tuple[int, str], np.ndarray] = {}
    for turn in run["turns"]:
        pref_map[(turn["round_num"], turn["role"])] = np.array(turn["pref"], dtype=float)

    if rounds is None:
        rounds = max(turn["round_num"] for turn in run["turns"])
    traj = np.zeros((rounds, 3), dtype=float)
    for t in range(rounds):
        prefs = [pref_map[(t + 1, role)] for role in roles]
        traj[t] = np.mean(prefs, axis=0)
    return traj


def pairwise_distance_curves(mean_trajs: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n_runs, n_rounds, _ = mean_trajs.shape
    if n_runs < 2:
        return np.zeros(n_rounds, dtype=float), []

    pair_curves = []
    ensemble = np.zeros(n_rounds, dtype=float)
    n_pairs = 0
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            curve = np.linalg.norm(mean_trajs[i] - mean_trajs[j], axis=1)
            pair_curves.append({"i": i, "j": j, "curve": curve})
            ensemble += curve
            n_pairs += 1
    ensemble /= n_pairs
    return ensemble, pair_curves


def estimate_lambda_pert(distance_curve: np.ndarray, start_round: int = 3) -> float:
    rounds = np.arange(1, len(distance_curve) + 1, dtype=float)
    mask = (rounds >= start_round) & (distance_curve > 1e-10)
    if mask.sum() < 3:
        return float("nan")
    return float(np.polyfit(rounds[mask], np.log(distance_curve[mask]), 1)[0])


def decision_fragility(decisions: list[str]) -> tuple[dict[str, int], float]:
    counts = dict(Counter(decisions))
    if not counts:
        return counts, float("nan")
    return counts, 1.0 - max(counts.values()) / len(decisions)


def branching_metrics(pair_curves: list[dict[str, Any]], threshold_frac: float = 0.25) -> dict[str, Any]:
    branching_rounds: list[int] = []
    branched_pairs = 0
    persistent_pairs = 0
    pair_rows: list[dict[str, Any]] = []

    for row in pair_curves:
        curve = row["curve"]
        final_dist = float(curve[-1])
        if final_dist <= 1e-12:
            pair_rows.append(
                {
                    "i": row["i"],
                    "j": row["j"],
                    "final_distance": final_dist,
                    "branched": False,
                    "branching_round": None,
                    "persistent_end_branch": False,
                }
            )
            continue

        threshold = threshold_frac * final_dist
        branch_round = None
        for idx, dist in enumerate(curve, start=1):
            if dist >= threshold:
                branch_round = idx
                break

        persistent = bool(np.all(curve[-3:] >= threshold))
        branched_pairs += 1
        if persistent:
            persistent_pairs += 1
        if branch_round is not None:
            branching_rounds.append(branch_round)

        pair_rows.append(
            {
                "i": row["i"],
                "j": row["j"],
                "final_distance": final_dist,
                "branched": True,
                "branching_round": branch_round,
                "persistent_end_branch": persistent,
            }
        )

    return {
        "median_branching_round": float(np.median(branching_rounds)) if branching_rounds else None,
        "reconvergence_index": (persistent_pairs / branched_pairs) if branched_pairs else None,
        "branched_pairs": branched_pairs,
        "pair_metrics": pair_rows,
    }


def variant_summary_rows(variant_items: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item, run in zip(variant_items, runs):
        rows.append(
            {
                "variant_id": item["variant_id"],
                "label": item["label"],
                "family": item["family"],
                "decision": (run["final_decision"] or {}).get("decision", "?"),
                "artifact_hash": canonical_json_hash(run),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic hosted prompt-perturbation experiments.")
    parser.add_argument("--scenario-id", default="IM-01")
    parser.add_argument("--roles-enabled", action="store_true", help="Enable role mandate text. Default is roles=False for AAAI-facing runs.")
    parser.add_argument("--no-roles", action="store_true", help="Disable role mandate text while keeping the same named turn order.")
    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument("--k-window", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants-file", default="deterministic_experiments/perturbation_variants_v1.json")
    parser.add_argument("--variant-limit", type=int, default=None)
    parser.add_argument("--out-dir", default="data/raw/deterministic_hosted/perturbation_v1")
    parser.add_argument("--resume", action="store_true", help="Resume from existing JSONL in out-dir if present.")
    parser.add_argument(
        "--protocol-variant",
        default="ws",
        choices=["ws", "no_feedback", "one_shot"],
        help="Committee protocol variant: full windowed-summary, iterative without transcript feedback, or one-shot voting.",
    )
    args = parser.parse_args()

    roles_enabled = bool(args.roles_enabled and not args.no_roles)
    condition = Condition(
        temperature=0.0,
        roles_enabled=roles_enabled,
        n_agents=args.n_agents,
        k_window=args.k_window,
        rounds=args.rounds,
    )
    roles = roles_for_n(args.n_agents)

    variant_set = load_variant_set(Path(args.variants_file), args.scenario_id)
    variants = list(variant_set["variants"])
    if args.variant_limit is not None:
        variants = variants[: args.variant_limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.scenario_id}__roles{roles_enabled}__N{args.n_agents}__R{args.rounds}__{args.protocol_variant}"
    jsonl_path = out_dir / f"{tag}.jsonl"
    summary_path = out_dir / f"{tag}__summary.json"

    backend = HostedDeterministicBackend(
        args.model_name,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    )
    backend_meta = backend.metadata()

    run_records: list[dict[str, Any]] = []
    mean_trajs: list[np.ndarray] = []
    existing_rows = load_existing_rows(jsonl_path) if args.resume else []
    existing_by_variant = {row["variant_id"]: row for row in existing_rows}

    if args.resume and existing_rows:
        for variant in variants:
            row = existing_by_variant.get(variant["variant_id"])
            if row is None:
                continue
            run_dict = row["run"]
            run_records.append(run_dict)
            mean_trajs.append(committee_mean_trajectory(run_dict, roles))

    print(
        f"run_small_perturbations_v1  |  scenario={args.scenario_id}  roles={roles_enabled}"
        f"  protocol={args.protocol_variant}"
    )
    print(f"model={args.model_name}  variants={len(variants)}  output → {jsonl_path}")
    if args.resume and existing_rows:
        print(f"resume=True  already_completed={len(existing_rows)}")

    open_mode = "a" if args.resume and jsonl_path.exists() else "w"
    with jsonl_path.open(open_mode, encoding="utf-8") as handle:
        for idx, variant in enumerate(variants, start=1):
            if variant["variant_id"] in existing_by_variant:
                print(f"  {idx:02d}/{len(variants)}  {variant['variant_id']}  already_done")
                continue
            run = run_hosted_ws_committee(
                scenario_id=args.scenario_id,
                scenario_text=variant["text"],
                condition=condition,
                backend=backend,
                protocol_variant=args.protocol_variant,
            )
            run_dict = asdict(run)
            row = {
                "saved_utc": now_utc(),
                "scenario_id": args.scenario_id,
                "variant_id": variant["variant_id"],
                "variant_label": variant["label"],
                "family": variant["family"],
                "protocol_variant": args.protocol_variant,
                "backend": backend_meta,
                "run": run_dict,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            run_records.append(run_dict)
            mean_trajs.append(committee_mean_trajectory(run_dict, roles))
            decision = (run.final_decision or {}).get("decision", "?")
            print(f"  {idx:02d}/{len(variants)}  {variant['variant_id']}  decision={decision}")

    mean_trajs_arr = np.stack(mean_trajs, axis=0)
    ensemble_curve, pair_curves = pairwise_distance_curves(mean_trajs_arr)
    lambda_pert = estimate_lambda_pert(ensemble_curve, start_round=3)
    variant_rows = variant_summary_rows(variants, run_records)
    decisions = [row["decision"] for row in variant_rows]
    decision_counts, fragility = decision_fragility(decisions)
    branching = branching_metrics(pair_curves, threshold_frac=0.25)

    summary = {
        "saved_utc": now_utc(),
        "scenario_id": args.scenario_id,
        "condition": {
            "roles_enabled": roles_enabled,
            "n_agents": args.n_agents,
            "k_window": args.k_window,
            "rounds": args.rounds,
            "temperature": 0.0,
        },
        "protocol_variant": args.protocol_variant,
        "backend": backend_meta,
        "variant_count": len(variants),
        "variant_rows": variant_rows,
        "decision_counts": decision_counts,
        "decision_fragility": fragility,
        "lambda_pert": lambda_pert,
        "ensemble_distance_curve": [float(x) for x in ensemble_curve],
        "branching_metrics": branching,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
