#!/usr/bin/env python3
"""
chaos_landscape_v1.py — Cross-scenario chaos landscape analysis.

Reads all JSONL files in data/raw/chaos_v1/ and outputs a λ summary table
across all scenarios × conditions. Designed for Section E of the paper.

Focus conditions for publication:
  - T=0.0, roles=True   (primary: chaos with no sampling noise)
  - T=0.0, roles=False  (hardest case: chaos with no noise AND no roles)
  - T=0.1, roles=True   (secondary)
  - T=0.1, roles=False  (secondary)

Usage (from project root):
    python src/chaos_landscape_v1.py
    python src/chaos_landscape_v1.py --temps 0.0 0.1 --n-boot 500
    python src/chaos_landscape_v1.py --fast        # skip bootstrap/perm, λ only
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from chaos_run_v1 import (
    load_runs,
    extract_trajectories,
    extract_mean_trajectories,
    mean_pairwise_distance,
    estimate_divergence_rate,
    decision_flip_rate,
    time_to_majority,
)
from chaos_analysis_v2 import (
    bootstrap_lambda,
    permutation_null_lambda,
    parse_filename,
    infer_roles,
    ok_runs,
)


ROUNDS = 20
N_BOOT = 500
N_PERM = 500


SCENARIO_LABELS = {
    "IM-01": "Immigration Asylum Allocation",
    "IM-02": "City Safety & Services",
    "HL-01": "Health Financing",
    "HL-02": "Hospital Ethics (ICU)",
    "IN-01": "Income Policy Cabinet",
    "IN-02": "Welfare Program Oversight",
    "CL-01": "Climate Policy Council",
    "CL-04": "Adaptation Funding",
    "SP-01": "Platform Integrity",
    "SP-03": "Recommender Systems",
    "AI-01": "Frontier Model Release",
    "AI-02": "Algorithmic Accountability",
}

SCENARIO_ORDER = list(SCENARIO_LABELS.keys())


def compute_row(runs_data, cond_meta, n_boot, n_perm, rng, fast=False):
    """Compute λ, CI, p-value for one JSONL file."""
    runs  = ok_runs(runs_data)
    if len(runs) < 4:
        return None
    roles = infer_roles(runs)
    trajs = extract_trajectories(runs, ROUNDS, roles)
    N     = cond_meta["n_agents"]

    mt      = extract_mean_trajectories(trajs, N)
    lam_obs = estimate_divergence_rate(mean_pairwise_distance(mt))
    counts, flip = decision_flip_rate(runs)
    ttm     = time_to_majority(runs, N)

    if fast:
        return dict(
            lam_obs=lam_obs, ci_lo=float("nan"), ci_hi=float("nan"),
            lam_null=float("nan"), p_val=float("nan"), sig="?",
            flip=flip, ttm_mean=float(np.mean(ttm)),
            n_reps=len(runs),
        )

    lam_obs, ci_lo, ci_hi = bootstrap_lambda(trajs, N, n_boot, rng)
    _, p_val, lam_null, _ = permutation_null_lambda(trajs, N, n_perm, rng)
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    return dict(
        lam_obs=lam_obs, ci_lo=ci_lo, ci_hi=ci_hi,
        lam_null=lam_null, p_val=p_val, sig=sig,
        flip=flip, ttm_mean=float(np.mean(ttm)),
        n_reps=len(runs),
    )


def sig_star(p_val):
    if np.isnan(p_val):
        return "?"
    return "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))


def main():
    parser = argparse.ArgumentParser(
        description="chaos_landscape_v1: cross-scenario λ table"
    )
    parser.add_argument("--data-dir",  default="data/raw/chaos_v1")
    parser.add_argument("--n-boot",    type=int, default=N_BOOT)
    parser.add_argument("--n-perm",    type=int, default=N_PERM)
    parser.add_argument("--temps",     type=float, nargs="+", default=[0.0, 0.1],
                        help="Temperatures to include (default: 0.0 0.1)")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Scenario IDs to include (default: all found)")
    parser.add_argument("--fast",      action="store_true",
                        help="Skip bootstrap and permutation (λ only)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rng      = np.random.default_rng(42)

    # ── Discover files ──────────────────────────────────────────────────────────
    all_files  = sorted(data_dir.glob("*.jsonl"))
    conditions = [c for f in all_files if (c := parse_filename(f)) is not None]

    # Filter by requested temps and scenarios
    target_temps = set(args.temps)
    conditions = [c for c in conditions if c["temperature"] in target_temps]
    if args.scenarios:
        conditions = [c for c in conditions if c["scenario_id"] in args.scenarios]

    scenarios_found = sorted({c["scenario_id"] for c in conditions},
                             key=lambda s: SCENARIO_ORDER.index(s) if s in SCENARIO_ORDER else 99)

    print(f"chaos_landscape_v1  |  temps={args.temps}  fast={args.fast}")
    print(f"scenarios={scenarios_found}  files={len(conditions)}")
    print(f"N_boot={args.n_boot}  N_perm={args.n_perm}")
    print()

    # ── Compute λ for all conditions ────────────────────────────────────────────
    results: dict[tuple, dict] = {}  # (scenario_id, temp, roles) -> row
    for cond in conditions:
        key = (cond["scenario_id"], cond["temperature"], cond["roles_enabled"])
        print(f"  computing {cond['path'].name} ...", end=" ", flush=True)
        try:
            runs_data = load_runs(cond["path"])
            row = compute_row(runs_data, cond, args.n_boot, args.n_perm, rng, fast=args.fast)
            if row is None:
                print("SKIP (too few runs)")
                continue
            results[key] = row
            print(f"λ={row['lam_obs']:+.4f} {row['sig']}")
        except Exception as e:
            print(f"ERROR: {e}")

    # ── Primary table: T=0.0 roles=True vs roles=False ─────────────────────────
    for temp in sorted(target_temps):
        for roles_val, roles_label in [(True, "roles=True"), (False, "roles=False")]:
            print()
            print("=" * 100)
            print(f"CHAOS LANDSCAPE  |  T={temp}  {roles_label}  |  λ (bootstrap 95% CI)  |  p-val")
            print("=" * 100)
            print(f"  {'Scenario ID':<8}  {'Description':<32}  {'n':>3}  {'λ_obs':>7}  {'95% CI':^20}  {'flip':>5}  {'TTM':>4}  p-val   sig")
            print(f"  {'-'*96}")

            any_printed = False
            for sid in scenarios_found:
                key = (sid, temp, roles_val)
                if key not in results:
                    desc = SCENARIO_LABELS.get(sid, sid)
                    print(f"  {sid:<8}  {desc:<32}  {'—':>3}  {'—':>7}  {'—':^20}  {'—':>5}  {'—':>4}  —       —")
                    continue
                r    = results[key]
                desc = SCENARIO_LABELS.get(sid, sid)
                ci_str = f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]" if not np.isnan(r["ci_lo"]) else "  [skipped]   "
                p_str  = f"{r['p_val']:.4f}" if not np.isnan(r["p_val"]) else "—"
                print(f"  {sid:<8}  {desc:<32}  {r['n_reps']:>3}  {r['lam_obs']:>+7.4f}  {ci_str:^20}  {r['flip']:>5.3f}  {r['ttm_mean']:>4.1f}  {p_str}  {r['sig']}")
                any_printed = True

            if not any_printed:
                print(f"  (no data found for T={temp} {roles_label})")

    # ── Cross-condition comparison mini-table ────────────────────────────────────
    print()
    print("=" * 100)
    print("SUMMARY  |  λ by scenario × condition  (+ = chaotic, - = convergent)")
    print("=" * 100)

    col_keys = [(t, r) for t in sorted(target_temps) for r in [True, False]]
    col_labels = [f"T={t} r={'T' if r else 'F'}" for t, r in col_keys]
    header = f"  {'Scenario':<8}  {'Description':<28}" + "".join(f"  {lb:>14}" for lb in col_labels)
    print(header)
    print("  " + "-" * (40 + 16 * len(col_keys)))

    for sid in scenarios_found:
        desc = SCENARIO_LABELS.get(sid, sid)[:28]
        row_str = f"  {sid:<8}  {desc:<28}"
        for t, roles_val in col_keys:
            key = (sid, t, roles_val)
            if key in results:
                r = results[key]
                row_str += f"  {r['lam_obs']:>+7.4f} {r['sig']:>3}"
            else:
                row_str += f"  {'—':>14}"
        print(row_str)

    # ── Count chaotic vs stable ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("CHAOS COUNT  |  scenarios with λ > 0 and p < 0.05")
    print("=" * 60)
    for t, roles_val in col_keys:
        rl = "roles=True" if roles_val else "roles=False"
        chaotic   = [s for s in scenarios_found if (s, t, roles_val) in results
                     and results[(s, t, roles_val)]["p_val"] < 0.05
                     and results[(s, t, roles_val)]["lam_obs"] > 0]
        stable    = [s for s in scenarios_found if (s, t, roles_val) in results
                     and results[(s, t, roles_val)]["p_val"] >= 0.05]
        print(f"  T={t} {rl}: chaotic={len(chaotic)}/{len(scenarios_found)}  "
              f"stable={len(stable)}  chaotic_ids={chaotic}")


if __name__ == "__main__":
    main()
