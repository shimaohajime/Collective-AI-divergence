#!/usr/bin/env python3
"""
semantic_perturbation_v1.py — Semantic perturbation experiment.

Implements Section C/F from llm_discussion_rds_theory_metrics.tex at temperature=0:
  - Run each semantically equivalent variant of IM-01 once (deterministic / greedy)
  - Measure trajectory divergence between variants
  - At temp=0 all runs are deterministic, so divergence is purely from semantic
    initial-state perturbation (x_0' = x_0 + δv, v = semantic direction)

This distinguishes structural sensitivity from stochastic noise.

Usage (from project root):
    python src/semantic_perturbation_v1.py
    python src/semantic_perturbation_v1.py --temperature 0.1   # low-noise comparison
    python src/semantic_perturbation_v1.py --n-replicates 5    # repeated at low temp

Output:
    data/raw/semantic_v1/{tag}.jsonl  — one run per variant
    Analysis printed to stdout
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from llm_dynamics_v1 import Condition, run_ws_committee
from chaos_run_v1 import (
    extract_trajectories, extract_mean_trajectories,
    mean_pairwise_distance, estimate_divergence_rate,
    decision_flip_rate, per_agent_switches, time_to_majority,
)

# ── Configuration ─────────────────────────────────────────────────────────────

N_AGENTS      = 5
ROLES_ENABLED = True
K_WINDOW      = 15
ROUNDS        = 20
MODEL         = "gpt-4.1-mini"
TRUNCATION    = "disabled"
VARIANTS_FILE = "src/im01_variants_v1.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_variants(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["variants"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="semantic_perturbation_v1: divergence across semantically equivalent IM-01 variants"
    )
    parser.add_argument("--temperature",    type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 = deterministic)")
    parser.add_argument("--n-replicates",   type=int,   default=1,
                        help="Runs per variant (default: 1; use >1 for low but nonzero temp)")
    parser.add_argument("--n-agents",       type=int,   default=N_AGENTS)
    parser.add_argument("--roles-enabled",  action="store_true", default=ROLES_ENABLED)
    parser.add_argument("--no-roles",       action="store_true")
    parser.add_argument("--variants-file",  default=VARIANTS_FILE)
    parser.add_argument("--out-dir",        default="data/raw/semantic_v1")
    args = parser.parse_args()

    roles_enabled = args.roles_enabled and not args.no_roles
    temperature   = args.temperature

    # temp=0 is valid for gpt-4.1-mini (greedy decoding)
    # For the model registry, temp=0 is still "supports temperature" (it's a valid value)

    variants_path = Path(args.variants_file)
    variants      = load_variants(variants_path)

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag       = f"IM-01__T{temperature}__N{args.n_agents}__roles{roles_enabled}"
    jsonl_path = out_dir / f"{tag}.jsonl"

    print(f"semantic_perturbation_v1  |  {len(variants)} variants  temp={temperature}  N={args.n_agents}  roles={roles_enabled}")
    print(f"n_replicates_per_variant={args.n_replicates}  model={MODEL}")
    print(f"output → {jsonl_path}")
    print()

    cond = Condition(
        temperature=temperature,
        roles_enabled=roles_enabled,
        n_agents=args.n_agents,
        k_window=K_WINDOW,
        rounds=ROUNDS,
    )

    all_records: list[dict[str, Any]] = []

    for v in variants:
        vid   = v["variant_id"]
        label = v["label"]
        text  = v["text"]

        for rep in range(args.n_replicates):
            run_tag = f"{vid}_r{rep:02d}"
            print(f"  {run_tag}  ({label}) ...", end="", flush=True)
            t0 = time.time()
            try:
                run = run_ws_committee(
                    scenario_id=vid,
                    scenario_text=text,
                    condition=cond,
                    model=MODEL,
                    truncation=TRUNCATION,
                )
                elapsed  = round(time.time() - t0, 1)
                decision = (run.final_decision or {}).get("decision", "?")
                tokens   = run.usage_total.get("total_tokens", 0)
                print(f"  decision={decision}  tokens={tokens}  {elapsed}s")

                record: dict[str, Any] = {
                    "run_tag":    run_tag,
                    "variant_id": vid,
                    "variant_label": label,
                    "replicate":  rep,
                    "model":      MODEL,
                    "status":     "ok",
                    "saved_utc":  now_utc(),
                    "elapsed_s":  elapsed,
                    "condition":  asdict(cond),
                    "run":        asdict(run),
                }
            except Exception as exc:
                elapsed = round(time.time() - t0, 1)
                print(f"  ERROR: {exc}  {elapsed}s")
                record = {
                    "run_tag":    run_tag,
                    "variant_id": vid,
                    "variant_label": label,
                    "replicate":  rep,
                    "model":      MODEL,
                    "status":     "error",
                    "saved_utc":  now_utc(),
                    "error":      str(exc),
                }

            all_records.append(record)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Analysis ──────────────────────────────────────────────────────────────
    ok_records = [r for r in all_records if r.get("status") == "ok"]
    if not ok_records:
        print("No successful runs.")
        return

    from llm_dynamics_v1 import roles_for_n
    roles  = roles_for_n(args.n_agents) if roles_enabled else \
             sorted({t["role"] for t in ok_records[0]["run"]["turns"]})

    trajs      = extract_trajectories(ok_records, ROUNDS, roles)
    mean_trajs = extract_mean_trajectories(trajs, args.n_agents)
    full_dists = mean_pairwise_distance(trajs)
    mean_dists = mean_pairwise_distance(mean_trajs)

    lam_full = estimate_divergence_rate(full_dists)
    lam_mean = estimate_divergence_rate(mean_dists)

    print()
    print("=" * 72)
    print(f"SEMANTIC PERTURBATION ANALYSIS  |  temp={temperature}  {len(ok_records)} runs")
    print("=" * 72)

    print("\n[Decisions per variant]")
    for v in variants:
        vid   = v["variant_id"]
        label = v["label"]
        decs  = [(r["run"]["final_decision"] or {}).get("decision", "?")
                 for r in ok_records if r["variant_id"] == vid]
        print(f"  {vid} ({label:<25}): {' '.join(decs)}")

    counts, flip_rate = decision_flip_rate(ok_records)
    print(f"\n  Overall decision distribution: {counts}")
    print(f"  Flip rate across variants: {flip_rate:.3f}")

    print(f"\n[Trajectory divergence across variants]")
    print(f"  {'Round':<8} {'Dist':>8}  {'|'}")
    max_d = mean_dists.max() if mean_dists.max() > 0 else 1.0
    for t in range(ROUNDS):
        bar_len = int(40 * mean_dists[t] / max_d)
        bar = "█" * bar_len
        print(f"  {t+1:<8} {mean_dists[t]:>8.4f}  {bar}")

    print(f"\n[Divergence rate λ]")
    print(f"  Full state:     λ = {lam_full:+.4f}  {'[DIVERGING]' if lam_full > 0 else '[converging/stable]'}")
    print(f"  Committee mean: λ = {lam_mean:+.4f}  {'[DIVERGING]' if lam_mean > 0 else '[converging/stable]'}")

    sw = per_agent_switches(ok_records, roles)
    print(f"\n[Preference switches per agent (mean across {len(ok_records)} variant-runs)]")
    for role in roles:
        vals = sw[role]
        print(f"  {role:<12}  mean={np.mean(vals):.2f}  std={np.std(vals):.2f}")

    ttm = time_to_majority(ok_records, args.n_agents)
    print(f"\n[Time-to-majority]  mean={np.mean(ttm):.1f}  median={np.median(ttm):.0f}")

    if temperature == 0.0:
        print("\n[Note] At temp=0, all runs are deterministic. Divergence above reflects")
        print("       purely semantic sensitivity — the structural Lyapunov signal")
        print("       without any sampling noise contribution.")


if __name__ == "__main__":
    main()
