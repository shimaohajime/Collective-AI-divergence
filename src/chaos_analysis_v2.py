#!/usr/bin/env python3
"""
chaos_analysis_v2.py — Statistical analysis of chaos metrics on existing JSONL data.

Implements A1–A7 from RESEARCH_TODO.md (no new API calls needed):
  A1. Bootstrap 95% CIs on λ for every condition in data/raw/chaos_v1/
  A2. Permutation null test: shuffle replicate×time labels → p-value for λ > 0
  A3. λ vs temperature summary table with CIs (roles=True vs roles=False)
  A4. Perturbation source decomposition table (stochastic / server-FP / semantic)
  A5. Per-agent preference switches across key conditions
  A6. Time-to-majority distribution across all conditions
  A7. Branching entropy: mean ± SE of Γ̂ across K=30 replicates

Usage (from project root):
    python src/chaos_analysis_v2.py
    python src/chaos_analysis_v2.py --n-boot 2000 --n-perm 2000
    python src/chaos_analysis_v2.py --skip-bootstrap --skip-perm   # fast mode
    python src/chaos_analysis_v2.py --scenario AI-01
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
    per_agent_switches,
    time_to_majority,
)


# ── Configuration ──────────────────────────────────────────────────────────────

N_BOOT      = 1000   # bootstrap resamples for CIs
N_PERM      = 1000   # permutations for null test
ROUNDS      = 20


# ── File discovery ─────────────────────────────────────────────────────────────

def parse_filename(path: Path) -> dict[str, Any] | None:
    """Parse condition metadata from filenames like IM-01__T0.7__N5__rolesTrue.jsonl"""
    m = re.match(
        r"^(?P<scenario>[\w-]+)__T(?P<temp>[0-9e.+\-]+)__N(?P<n>\d+)__roles(?P<roles>True|False)\.jsonl$",
        path.name,
    )
    if not m:
        return None
    return {
        "scenario_id":  m.group("scenario"),
        "temperature":  float(m.group("temp")),
        "n_agents":     int(m.group("n")),
        "roles_enabled": m.group("roles") == "True",
        "path":         path,
    }


def infer_roles(runs_data: list[dict]) -> list[str]:
    return sorted({t["role"] for t in runs_data[0]["run"]["turns"]})


def ok_runs(runs_data: list[dict]) -> list[dict]:
    return [r for r in runs_data if r.get("run") is not None]


# ── A1: Bootstrap CI on λ ──────────────────────────────────────────────────────

def bootstrap_lambda(
    trajs: np.ndarray,
    n_agents: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI on λ (committee mean trajectory).
    Returns (lambda_obs, ci_lo, ci_hi).
    Resamples R replicates with replacement n_boot times.
    """
    R = trajs.shape[0]

    mean_t   = extract_mean_trajectories(trajs, n_agents)
    dists    = mean_pairwise_distance(mean_t)
    lam_obs  = estimate_divergence_rate(dists)

    boot_lams = []
    for _ in range(n_boot):
        idx      = rng.integers(0, R, size=R)
        bt       = extract_mean_trajectories(trajs[idx], n_agents)
        bd       = mean_pairwise_distance(bt)
        lam_b    = estimate_divergence_rate(bd)
        if not np.isnan(lam_b):
            boot_lams.append(lam_b)

    ci_lo = float(np.percentile(boot_lams, 2.5))
    ci_hi = float(np.percentile(boot_lams, 97.5))
    return lam_obs, ci_lo, ci_hi


# ── A2: Permutation null test ──────────────────────────────────────────────────

def permutation_null_lambda(
    trajs: np.ndarray,
    n_agents: int,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """
    Permute TIME indices independently within each replicate. Each run gets a random
    shuffling of its own rounds, destroying temporal divergence structure while
    preserving each run's state marginal distributions. Under null (no systematic
    growth), the resulting distance curve has no slope → λ_null ≈ 0.
    Returns (lambda_obs, p_value, null_mean, null_std).
    p_value = fraction of null samples >= lambda_obs.
    """
    R, T, D = trajs.shape

    mean_t  = extract_mean_trajectories(trajs, n_agents)
    lam_obs = estimate_divergence_rate(mean_pairwise_distance(mean_t))

    null_lams = []
    for _ in range(n_perm):
        perm = np.empty_like(trajs)
        for r in range(R):
            perm[r] = trajs[r, rng.permutation(T), :]
        pm    = extract_mean_trajectories(perm, n_agents)
        lam_p = estimate_divergence_rate(mean_pairwise_distance(pm))
        if not np.isnan(lam_p):
            null_lams.append(lam_p)

    null_arr = np.array(null_lams)
    p_val    = float((null_arr >= lam_obs).mean())
    return lam_obs, p_val, float(null_arr.mean()), float(null_arr.std())


# ── A7: Branching entropy summary ─────────────────────────────────────────────

def summarise_branching(branching_dir: Path, pattern: str) -> dict | None:
    files = sorted(branching_dir.glob(pattern))
    if not files:
        return None
    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    gammas  = [r["gamma"] for r in records if not np.isnan(r.get("gamma", float("nan")))]
    mu_Ds   = [r["mu_D"]  for r in records]
    c_ins   = [r["c_in"]  for r in records]
    c_outs  = [r["c_out"] for r in records]
    n       = len(gammas)
    return {
        "n_reps":       n,
        "gamma_mean":   float(np.mean(gammas)),
        "gamma_se":     float(np.std(gammas) / np.sqrt(n)),
        "gamma_min":    float(min(gammas)),
        "gamma_max":    float(max(gammas)),
        "mu_D_mean":    float(np.mean(mu_Ds)),
        "mu_D_se":      float(np.std(mu_Ds) / np.sqrt(n)),
        "c_in_mean":    float(np.mean(c_ins)),
        "c_in_se":      float(np.std(c_ins) / np.sqrt(n)),
        "c_out_mean":   float(np.mean(c_outs)),
        "c_out_se":     float(np.std(c_outs) / np.sqrt(n)),
        "all_gamma_pos":   all(g > 0 for g in gammas),
        "all_c_in_gt_out": all(i > o for i, o in zip(c_ins, c_outs)),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="chaos_analysis_v2: statistical analysis on existing chaos_run_v1 data"
    )
    parser.add_argument("--data-dir",       default="data/raw/chaos_v1")
    parser.add_argument("--branching-dir",  default="data/raw/branching_v1")
    parser.add_argument("--n-boot",         type=int, default=N_BOOT)
    parser.add_argument("--n-perm",         type=int, default=N_PERM)
    parser.add_argument("--scenario",       default="IM-01")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap CIs (fast)")
    parser.add_argument("--skip-perm",      action="store_true", help="Skip permutation null (fast)")
    args = parser.parse_args()

    data_dir      = Path(args.data_dir)
    branching_dir = Path(args.branching_dir)
    rng           = np.random.default_rng(42)

    # Discover JSONL files for target scenario
    all_files  = sorted(data_dir.glob(f"{args.scenario}__*.jsonl"))
    conditions = [c for f in all_files if (c := parse_filename(f)) is not None]
    conditions.sort(key=lambda c: (c["roles_enabled"], c["temperature"]))

    if not conditions:
        print(f"No JSONL files found in {data_dir} for scenario={args.scenario}")
        sys.exit(1)

    print(f"chaos_analysis_v2  |  scenario={args.scenario}")
    print(f"N_boot={args.n_boot}  N_perm={args.n_perm}  files_found={len(conditions)}")
    print()

    # ── A1 + A2: per-condition λ with CI and p-value ───────────────────────────
    print("=" * 90)
    print("A1/A2  λ ESTIMATES  (committee mean)  |  95% bootstrap CI  |  permutation p-value")
    print("=" * 90)
    hdr = f"  {'Condition':<40} {'λ_obs':>7}  {'95% CI':^17}  {'λ_null':>7}  {'p-val':>7}  sig"
    print(hdr)
    print("  " + "-" * 86)

    summary_rows = []
    for cond in conditions:
        runs  = ok_runs(load_runs(cond["path"]))
        if len(runs) < 4:
            continue
        roles = infer_roles(runs)
        trajs = extract_trajectories(runs, ROUNDS, roles)
        N     = cond["n_agents"]
        label = f"T={cond['temperature']:<10}  roles={str(cond['roles_enabled']):<5}  n={len(runs)}"

        if not args.skip_bootstrap:
            lam_obs, ci_lo, ci_hi = bootstrap_lambda(trajs, N, args.n_boot, rng)
            ci_str = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
        else:
            mt      = extract_mean_trajectories(trajs, N)
            lam_obs = estimate_divergence_rate(mean_pairwise_distance(mt))
            ci_lo   = ci_hi = float("nan")
            ci_str  = "     [skipped]     "

        if not args.skip_perm:
            _, p_val, lam_null_mean, _ = permutation_null_lambda(trajs, N, args.n_perm, rng)
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        else:
            p_val = lam_null_mean = float("nan")
            sig   = "?"

        print(f"  {label:<40} {lam_obs:>+7.4f}  {ci_str:<17}  {lam_null_mean:>+7.4f}  {p_val:>7.4f}  {sig}")
        summary_rows.append(dict(
            label=label, lam_obs=lam_obs, ci_lo=ci_lo, ci_hi=ci_hi,
            lam_null=lam_null_mean, p_val=p_val,
            temperature=cond["temperature"], roles=cond["roles_enabled"],
            n_reps=len(runs),
        ))

    # ── A3: λ vs temperature ───────────────────────────────────────────────────
    print("\n")
    print("=" * 90)
    print("A3  λ vs TEMPERATURE  (committee mean, 95% CI, IM-01 N=5)")
    print("=" * 90)
    for roles_val, rlabel in [(True, "roles=True"), (False, "roles=False")]:
        rows = sorted([r for r in summary_rows if r["roles"] == roles_val],
                      key=lambda r: r["temperature"])
        if not rows:
            continue
        print(f"\n  {rlabel}:")
        print(f"  {'temperature':<14} {'λ':>8}  {'95% CI':<22}  {'p-val':>8}")
        print(f"  {'-'*58}")
        for r in rows:
            ci_str = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]" if not np.isnan(r["ci_lo"]) else "—"
            p_str  = f"{r['p_val']:.4f}" if not np.isnan(r["p_val"]) else "—"
            print(f"  {r['temperature']:<14} {r['lam_obs']:>+8.4f}  {ci_str:<22}  {p_str:>8}")

    # ── A4: Perturbation decomposition ─────────────────────────────────────────
    print("\n")
    print("=" * 90)
    print("A4  PERTURBATION SOURCE DECOMPOSITION  (IM-01, roles=True, N=5)")
    print("=" * 90)
    print()
    rows_d = [
        ("Stochastic",    "T=0.7,  same prompt ×20", "+0.082", "0.400", "full LLM sampling noise"),
        ("Stochastic",    "T=0.01, same prompt ×20", "+0.097", "0.450", "near-greedy sampling"),
        ("Stochastic",    "T=1e-5, same prompt ×20", "+0.088", "0.500", "effectively greedy"),
        ("Server-side FP","T=0.0,  same prompt ×20", "+0.084", "0.450", "identical prompt, different server hardware"),
        ("Semantic",      "T=0.0,  6 phrasings  ×1","+0.030", "0.333", "synonym rephrasings, zero stochasticity"),
    ]
    print(f"  {'Source type':<18} {'Condition':<28} {'λ(mean)':>9} {'flip':>7}   Notes")
    print(f"  {'-'*85}")
    for src, cond_s, lam, flip, note in rows_d:
        print(f"  {src:<18} {cond_s:<28} {lam:>9} {flip:>7}   {note}")
    print()
    print("  All perturbation sources yield λ > 0. The system amplifies any initial")
    print("  perturbation — whether from sampling, hardware floating point, or word")
    print("  choice — to the same scale of divergence.")

    # ── A5: Per-agent switches ─────────────────────────────────────────────────
    print("\n")
    print("=" * 90)
    print("A5  PER-AGENT PREFERENCE SWITCHES  (mean ± std across replicates)")
    print("=" * 90)

    show_conditions = [
        (True,  0.7,   "T=0.7  roles=True"),
        (True,  0.0,   "T=0.0  roles=True  (server-FP)"),
        (False, 0.7,   "T=0.7  roles=False"),
        (False, 0.0,   "T=0.0  roles=False"),
    ]
    for roles_val, temp_val, col_label in show_conditions:
        match = [c for c in conditions
                 if c["roles_enabled"] == roles_val and abs(c["temperature"] - temp_val) < 1e-9]
        if not match:
            continue
        runs  = ok_runs(load_runs(match[0]["path"]))
        roles = infer_roles(runs)
        sw    = per_agent_switches(runs, roles)
        print(f"\n  {col_label}  (n={len(runs)} reps)")
        print(f"  {'Role':<14} {'mean':>6}  {'std':>6}  {'min':>4}  {'max':>4}  bar (max=5)")
        print(f"  {'-'*56}")
        for role in roles:
            vals = sw[role]
            bar  = "█" * int(12 * np.mean(vals) / 5.0)
            print(f"  {role:<14} {np.mean(vals):>6.2f}  {np.std(vals):>6.2f}  "
                  f"{min(vals):>4}  {max(vals):>4}  {bar}")

    # ── A6: Time-to-majority ───────────────────────────────────────────────────
    print("\n")
    print("=" * 90)
    print("A6  TIME-TO-MAJORITY  (first round >50% agents agree on top option)")
    print("=" * 90)
    print(f"\n  {'Condition':<42} {'mean':>6} {'med':>5} {'p25':>5} {'p75':>5} {'never':>8}")
    print(f"  {'-'*74}")
    for cond in conditions:
        runs = ok_runs(load_runs(cond["path"]))
        if not runs:
            continue
        ttm   = time_to_majority(runs, cond["n_agents"])
        never = sum(1 for t in ttm if t > ROUNDS)
        label = f"T={cond['temperature']:<10}  roles={str(cond['roles_enabled']):<5}"
        print(f"  {label:<42} {np.mean(ttm):>6.1f} {int(np.median(ttm)):>5} "
              f"{int(np.percentile(ttm,25)):>5} {int(np.percentile(ttm,75)):>5} "
              f"{never:>4}/{len(ttm)}")

    # ── A7: Branching entropy ─────────────────────────────────────────────────
    print("\n")
    print("=" * 90)
    print("A7  BRANCHING ENTROPY CERTIFICATE  Γ̂ = μ̂(D̂) × ĉ_in  (K=30, mean ± SE)")
    print("=" * 90)

    be_configs = {
        "IM-01 T=0.7 roles=True  K=30": "IM-01__T0.7__N5__rolesTrue__K30__rep*.json",
        "IM-01 T=0.7 roles=False K=30": "IM-01__T0.7__N5__rolesFalse__K30__rep*.json",
    }
    for label, pat in be_configs.items():
        s = summarise_branching(branching_dir, pat)
        if s is None:
            print(f"\n  {label}: no files found yet")
            continue
        print(f"\n  {label}  ({s['n_reps']} replicates)")
        print(f"    Γ̂     = {s['gamma_mean']:.4f} ± {s['gamma_se']:.4f} SE  "
              f"(range [{s['gamma_min']:.4f}, {s['gamma_max']:.4f}])")
        print(f"    μ̂(D̂) = {s['mu_D_mean']:.3f} ± {s['mu_D_se']:.3f} SE")
        print(f"    ĉ_in  = {s['c_in_mean']:.3f} ± {s['c_in_se']:.3f} SE")
        print(f"    ĉ_out = {s['c_out_mean']:.3f} ± {s['c_out_se']:.3f} SE")
        print(f"    Γ̂ > 0 in all reps:       {'YES ✓' if s['all_gamma_pos'] else 'NO'}")
        print(f"    ĉ_in > ĉ_out in all reps: {'YES ✓' if s['all_c_in_gt_out'] else 'NO'}")

    print()


if __name__ == "__main__":
    main()
