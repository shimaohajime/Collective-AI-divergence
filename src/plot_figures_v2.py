#!/usr/bin/env python3
"""
plot_figures_v2.py — Updated publication-quality main figures.

Outputs:
  publication_v2/fig1_twopaths.pdf
  publication_v2/fig2_chair_mechanism_crossscenario.pdf
  publication_v2/fig3_intervention_falsification.pdf
  publication_v2/fig4_landscape_2x2_full.pdf
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


SCIENCE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
}

C_ROLES = "#1565C0"
C_NOROLES = "#757575"
C_MIXED = "#2E7D32"
C_CHAIR = "#E65100"
C_INTERVENTION = "#6A1B9A"
C_FALSIFICATION = "#C62828"

ROUNDS = 20
N_BOOT = 500
FIG1_SCENARIO = "HL-01"

RAW_DIR = Path("data/raw/chaos_v1")
VM_SYNC = Path("data/vm_sync")
MIXED_DIRS = [
    VM_SYNC / "instance-20260308-011204/chaos_v1_one_grok_t0_matrix",
    VM_SYNC / "instance-20260308-183507/chaos_v1_one_grok_t0_matrix",
    VM_SYNC / "instance-20260308-183552/chaos_v1_one_grok_t0_matrix",
    RAW_DIR,
    Path("data/raw/chaos_v1_one_grok"),
]
CHAIR_REP_DIR = VM_SYNC / "instance-20260307-022228/chaos_v1_chair_rep_t0"
INTERVENTION_DIR = VM_SYNC / "instance-20260307-022228/chaos_v1_intervention_kw3_t0"
FALSIFICATION_DIR = VM_SYNC / "instance-20260307-022228/chaos_v1_falsification_kw1_t0"

FIG_DIR = Path("publication_v2")


def load_runs(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def best_existing_path(filename: str, roots: Sequence[Path]) -> Optional[Path]:
    best_path = None
    best_n = -1
    for root in roots:
        p = root / filename
        if not p.exists():
            continue
        n = sum(1 for _ in p.open(encoding="utf-8"))
        if n > best_n:
            best_n = n
            best_path = p
    return best_path


def committee_mean_prefs(rec: dict, n_rounds: int) -> np.ndarray:
    turns = rec["run"]["turns"]
    result = np.full((n_rounds, 3), np.nan)
    for r in range(1, n_rounds + 1):
        rprefs = [t["pref"] for t in turns if t["round_num"] == r]
        if rprefs:
            result[r - 1] = np.mean(rprefs, axis=0)
    return result


def get_trajs(path: Path, n_rounds: int = ROUNDS) -> Optional[np.ndarray]:
    runs = load_runs(path)
    if len(runs) < 3:
        return None
    trajs = []
    for rec in runs:
        mp = committee_mean_prefs(rec, n_rounds)
        if not np.isnan(mp).all():
            trajs.append(mp)
    if not trajs:
        return None
    return np.array(trajs)


def mean_pairwise_dist(trajs: np.ndarray) -> np.ndarray:
    r, t, _ = trajs.shape
    if r < 2:
        return np.zeros(t)
    dists = []
    for i in range(r):
        for j in range(i + 1, r):
            dists.append(np.linalg.norm(trajs[i] - trajs[j], axis=1))
    return np.mean(dists, axis=0)


def fit_lambda(dists: np.ndarray, skip: int = 2) -> float:
    t = np.arange(len(dists))[skip:]
    y = dists[skip:]
    valid = y > 0
    if valid.sum() < 2:
        return float("nan")
    lam, _ = np.polyfit(t[valid], np.log(y[valid]), 1)
    return float(lam)


def bootstrap_lambda(trajs: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    r = trajs.shape[0]
    lam_obs = fit_lambda(mean_pairwise_dist(trajs))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, r, size=r)
        boots.append(fit_lambda(mean_pairwise_dist(trajs[idx])))
    boots = np.array([b for b in boots if not np.isnan(b)])
    if len(boots) == 0:
        return lam_obs, float("nan"), float("nan")
    return lam_obs, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_lambda(path: Optional[Path], rng: Optional[np.random.Generator] = None, n_boot: int = 0) -> Tuple[float, float, float, int]:
    if path is None:
        return float("nan"), float("nan"), float("nan"), 0
    trajs = get_trajs(path, n_rounds=ROUNDS)
    if trajs is None:
        return float("nan"), float("nan"), float("nan"), 0
    if n_boot > 0 and rng is not None:
        lam, lo, hi = bootstrap_lambda(trajs, n_boot, rng)
    else:
        lam = fit_lambda(mean_pairwise_dist(trajs))
        lo = hi = float("nan")
    return lam, lo, hi, trajs.shape[0]


def fig1(rng: np.random.Generator) -> None:
    plt.rcParams.update(SCIENCE_RC)
    fig = plt.figure(figsize=(7.5, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    sid = FIG1_SCENARIO
    conds = [
        (f"{sid}__T0.0__N5__rolesFalse.jsonl", [RAW_DIR], C_NOROLES, "--", "Uniform, no roles"),
        (f"{sid}__T0.0__N5__rolesTrue.jsonl", [RAW_DIR], C_ROLES, "-", "Uniform, with roles"),
        (f"{sid}__T0.0__N5__rolesFalse__multimodel.jsonl", MIXED_DIRS, C_MIXED, "-", "Mixed, no roles"),
    ]
    ann = []
    for fname, roots, color, ls, label in conds:
        p = best_existing_path(fname, roots)
        trajs = get_trajs(p, ROUNDS) if p else None
        if trajs is None:
            continue
        d = mean_pairwise_dist(trajs)
        lam, _, _ = bootstrap_lambda(trajs, N_BOOT, rng)
        x = np.arange(1, len(d) + 1)
        ax_a.plot(x, d, color=color, ls=ls, lw=1.6, label=label)
        ann.append((label, lam, color))

    ax_a.set_xlim(1, ROUNDS)
    ax_a.set_ylim(bottom=0)
    ax_a.set_xlabel("Round t")
    ax_a.set_ylabel("Mean pairwise distance D(t)")
    ax_a.set_title(f"Architecture modulates deployed instability ({sid}, T=0)")
    ax_a.legend(frameon=False, fontsize=6.5, loc="upper left")
    for i, (_, lam, color) in enumerate(ann):
        ax_a.text(0.98, 0.95 - 0.09 * i, f"$\\hat{{\\lambda}}={lam:+.3f}$",
                  transform=ax_a.transAxes, ha="right", va="top", color=color, fontsize=6.5)
    ax_a.text(-0.12, 1.03, "A", transform=ax_a.transAxes, fontsize=11, fontweight="bold")

    rows = ["Uniform", "Mixed"]
    cols = ["No roles", "With roles"]
    grid_files = {
        ("Uniform", "No roles"): (f"{sid}__T0.0__N5__rolesFalse.jsonl", [RAW_DIR]),
        ("Uniform", "With roles"): (f"{sid}__T0.0__N5__rolesTrue.jsonl", [RAW_DIR]),
        ("Mixed", "No roles"): (f"{sid}__T0.0__N5__rolesFalse__multimodel.jsonl", MIXED_DIRS),
        ("Mixed", "With roles"): (f"{sid}__T0.0__N5__rolesTrue__multimodel.jsonl", MIXED_DIRS),
    }
    lam_grid = np.full((2, 2), np.nan)
    n_grid = np.zeros((2, 2), dtype=int)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            fname, roots = grid_files[(r, c)]
            p = best_existing_path(fname, roots)
            lam, _, _, n = compute_lambda(p)
            lam_grid[i, j] = lam
            n_grid[i, j] = n

    norm = mcolors.TwoSlopeNorm(vmin=-0.02, vcenter=0.0, vmax=0.12)
    im = ax_b.imshow(lam_grid, aspect="auto", cmap="RdBu_r", norm=norm)
    for i in range(2):
        for j in range(2):
            if np.isnan(lam_grid[i, j]):
                ax_b.text(j, i, "—", ha="center", va="center", color="white", fontsize=9)
                continue
            txtc = "white" if abs(lam_grid[i, j]) > 0.04 else "#333333"
            ax_b.text(j, i - 0.1, f"{lam_grid[i, j]:+.3f}", ha="center", va="center", color=txtc, fontsize=7, fontweight="bold")
            ax_b.text(j, i + 0.20, f"n={n_grid[i, j]}", ha="center", va="center", color=txtc, fontsize=5.5)
    ax_b.set_xticks([0, 1], cols)
    ax_b.set_yticks([0, 1], rows)
    ax_b.set_title(f"2×2 governance matrix ({sid}, T=0)")
    cb = fig.colorbar(im, ax=ax_b, shrink=0.78, pad=0.03)
    cb.set_label(r"$\hat{\lambda}$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax_b.text(-0.20, 1.03, "B", transform=ax_b.transAxes, fontsize=11, fontweight="bold")

    out = FIG_DIR / "fig1_twopaths.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def fig2(rng: np.random.Generator) -> None:
    plt.rcParams.update(SCIENCE_RC)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.1, 1.1]})

    ax = axes[0]
    sid_a = "HL-01"
    p_full = best_existing_path(f"{sid_a}__T0.0__N5__rolesTrue.jsonl", [RAW_DIR, CHAIR_REP_DIR])
    tr_full = get_trajs(p_full, ROUNDS) if p_full else None
    if tr_full is None:
        raise RuntimeError(f"Missing/invalid full-roles trajectory for {sid_a}")

    labels = ["Ablate Chair", "Ablate Equity", "Ablate Rights", "Ablate Welfare", "Ablate Security", "No roles"]
    keys = ["Chair", "Equity", "Rights", "Welfare", "Security", "NoRoles"]
    deltas, lo_arr, hi_arr = [], [], []
    for k in keys:
        if k == "Chair":
            p_ab = best_existing_path(f"{sid_a}__T0.0__N5__rolesTrue__ablate-Chair.jsonl", [CHAIR_REP_DIR, RAW_DIR])
        elif k == "NoRoles":
            p_ab = best_existing_path(f"{sid_a}__T0.0__N5__rolesFalse.jsonl", [RAW_DIR])
        else:
            p_ab = best_existing_path(f"{sid_a}__T0.0__N5__rolesTrue__ablate-{k}.jsonl", [RAW_DIR, CHAIR_REP_DIR])

        tr_ab = get_trajs(p_ab, ROUNDS) if p_ab else None
        if tr_ab is None:
            deltas.append(np.nan)
            lo_arr.append(np.nan)
            hi_arr.append(np.nan)
            continue

        d_obs = fit_lambda(mean_pairwise_dist(tr_full)) - fit_lambda(mean_pairwise_dist(tr_ab))
        boots = []
        for _ in range(N_BOOT):
            idx_f = rng.integers(0, tr_full.shape[0], size=tr_full.shape[0])
            idx_a = rng.integers(0, tr_ab.shape[0], size=tr_ab.shape[0])
            boots.append(
                fit_lambda(mean_pairwise_dist(tr_full[idx_f]))
                - fit_lambda(mean_pairwise_dist(tr_ab[idx_a]))
            )
        boots = np.array([b for b in boots if not np.isnan(b)])
        if len(boots) == 0:
            lo = hi = np.nan
        else:
            lo, hi = np.percentile(boots, [2.5, 97.5]).tolist()
        deltas.append(d_obs)
        lo_arr.append(lo)
        hi_arr.append(hi)

    y = np.arange(len(labels))
    deltas = np.array(deltas, dtype=float)
    lo_arr = np.array(lo_arr, dtype=float)
    hi_arr = np.array(hi_arr, dtype=float)
    colors = [C_CHAIR, "#CFD8DC", "#B0BEC5", "#90A4AE", "#78909C", C_NOROLES]
    ax.barh(y, deltas, color=colors, height=0.62, edgecolor="none", zorder=3)
    xerr_lo = np.where(np.isnan(lo_arr), 0, deltas - lo_arr)
    xerr_hi = np.where(np.isnan(hi_arr), 0, hi_arr - deltas)
    ax.errorbar(deltas, y, xerr=[xerr_lo, xerr_hi], fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2, zorder=4)
    ax.axvline(0, color="#333333", lw=0.7)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta\hat{\lambda}$ = full roles - ablated")
    ax.set_title(f"{sid_a} role-ablation effects (T=0)")
    ax.text(-0.16, 1.03, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax = axes[1]
    scenarios = ["IM-01", "HL-01", "CL-01", "SP-03", "AI-01"]
    rows = []
    for sid in scenarios:
        p_full = best_existing_path(f"{sid}__T0.0__N5__rolesTrue.jsonl", [CHAIR_REP_DIR, RAW_DIR])
        p_ab = best_existing_path(f"{sid}__T0.0__N5__rolesTrue__ablate-Chair.jsonl", [CHAIR_REP_DIR, RAW_DIR])
        tr_f = get_trajs(p_full, ROUNDS) if p_full else None
        tr_a = get_trajs(p_ab, ROUNDS) if p_ab else None
        if tr_f is None or tr_a is None:
            continue
        lam_f = fit_lambda(mean_pairwise_dist(tr_f))
        lam_a = fit_lambda(mean_pairwise_dist(tr_a))
        delta_obs = lam_f - lam_a
        # Bootstrap CI directly on Δλ to avoid over-conservative error propagation.
        boots = []
        for _ in range(N_BOOT):
            idx_f = rng.integers(0, tr_f.shape[0], size=tr_f.shape[0])
            idx_a = rng.integers(0, tr_a.shape[0], size=tr_a.shape[0])
            boots.append(
                fit_lambda(mean_pairwise_dist(tr_f[idx_f]))
                - fit_lambda(mean_pairwise_dist(tr_a[idx_a]))
            )
        boots = np.array([b for b in boots if not np.isnan(b)])
        if len(boots) == 0:
            lo = hi = np.nan
        else:
            lo, hi = np.percentile(boots, [2.5, 97.5]).tolist()
        rows.append((sid, delta_obs, lo, hi))

    rows.sort(key=lambda x: x[1], reverse=True)
    labels_b = [r[0] for r in rows]
    delta_arr = np.array([r[1] for r in rows], dtype=float)
    lo_arr = np.array([r[2] for r in rows], dtype=float)
    hi_arr = np.array([r[3] for r in rows], dtype=float)
    x = np.arange(len(labels_b))
    err_lo = np.where(np.isnan(lo_arr), 0, delta_arr - lo_arr)
    err_hi = np.where(np.isnan(hi_arr), 0, hi_arr - delta_arr)
    sig_pos = (~np.isnan(lo_arr)) & (lo_arr > 0)
    bar_colors = [C_CHAIR if s else "#B0BEC5" for s in sig_pos]
    ax.bar(x, delta_arr, color=bar_colors, width=0.65, alpha=0.92)
    ax.errorbar(x, delta_arr, yerr=[err_lo, err_hi], fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2, zorder=4)
    ax.axhline(0, color="#333333", lw=0.7)
    ax.set_xticks(x, labels_b)
    ax.set_ylabel(r"$\Delta\hat{\lambda}$ = full roles - ablate Chair")
    ax.set_title("Chair effect heterogeneity across scenarios")
    n_pos = int((delta_arr > 0).sum())
    n_sig = int(sig_pos.sum())
    ax.text(0.98, 0.98, f"Positive: {n_pos}/{len(delta_arr)}\n95% CI > 0: {n_sig}/{len(delta_arr)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color="#333333")
    ax.text(-0.12, 1.03, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.tight_layout(w_pad=1.8)
    out = FIG_DIR / "fig2_chair_mechanism_crossscenario.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def fig3(rng: np.random.Generator) -> None:
    plt.rcParams.update(SCIENCE_RC)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    ax = axes[0]
    scenarios = ["AI-01", "CL-01", "HL-01", "SP-03"]
    base_lam, int_lam = [], []
    for sid in scenarios:
        p_base = best_existing_path(f"{sid}__T0.0__N5__rolesTrue.jsonl", [RAW_DIR, CHAIR_REP_DIR])
        p_int = best_existing_path(f"{sid}__T0.0__N5__rolesTrue.jsonl", [INTERVENTION_DIR])
        lam_b, _, _, _ = compute_lambda(p_base, rng=rng, n_boot=0)
        lam_i, _, _, _ = compute_lambda(p_int, rng=rng, n_boot=0)
        base_lam.append(lam_b)
        int_lam.append(lam_i)
    x = np.arange(len(scenarios))
    ax.plot(x, base_lam, marker="o", color=C_ROLES, label="Baseline roles (k=15)")
    ax.plot(x, int_lam, marker="s", color=C_INTERVENTION, label="Intervention (k=3)")
    ax.set_xticks(x, scenarios)
    ax.set_ylabel(r"$\hat{\lambda}$")
    ax.set_title("Intervention test: reduced memory window")
    ax.axhline(0, color="#333333", lw=0.7)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.text(-0.14, 1.03, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax = axes[1]
    scenarios_f = ["IM-01", "CL-01"]
    base_f, f1 = [], []
    for sid in scenarios_f:
        p_base = best_existing_path(f"{sid}__T0.0__N5__rolesTrue.jsonl", [RAW_DIR, CHAIR_REP_DIR])
        p_f = best_existing_path(f"{sid}__T0.0__N5__rolesTrue.jsonl", [FALSIFICATION_DIR])
        lam_b, _, _, _ = compute_lambda(p_base, rng=rng, n_boot=0)
        lam_f, _, _, _ = compute_lambda(p_f, rng=rng, n_boot=0)
        base_f.append(lam_b)
        f1.append(lam_f)
    x2 = np.arange(len(scenarios_f))
    w = 0.34
    ax.bar(x2 - w / 2, base_f, width=w, color=C_ROLES, label="Baseline roles (k=15)")
    ax.bar(x2 + w / 2, f1, width=w, color=C_FALSIFICATION, label="Falsification (k=1)")
    ax.axhline(0, color="#333333", lw=0.7)
    ax.set_xticks(x2, scenarios_f)
    ax.set_ylabel(r"$\hat{\lambda}$")
    ax.set_title("Falsification target: near-memoryless update")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.text(-0.12, 1.03, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.tight_layout(w_pad=1.8)
    out = FIG_DIR / "fig3_intervention_falsification.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def fig4(rng: np.random.Generator) -> None:
    del rng
    plt.rcParams.update(SCIENCE_RC)
    scenario_labels = {
        "IM-01": "Immigration Asylum",
        "IM-02": "City Safety",
        "HL-01": "Health Financing",
        "HL-02": "Hospital Ethics",
        "IN-01": "Income Policy",
        "IN-02": "Welfare Oversight",
        "CL-01": "Climate Policy",
        "CL-04": "Adaptation Fund",
        "SP-01": "Platform Integrity",
        "SP-03": "Recommender Systems",
        "AI-01": "Model Release",
        "AI-02": "Algorithmic Accountability",
    }
    scenarios = list(scenario_labels.keys())
    col_defs = [
        ("Uniform\nNo roles", False, False),
        ("Uniform\nWith roles", True, False),
        ("Mixed\nNo roles", False, True),
        ("Mixed\nWith roles", True, True),
    ]
    grid = np.full((len(scenarios), len(col_defs)), np.nan)
    n_grid = np.zeros_like(grid, dtype=int)
    for i, sid in enumerate(scenarios):
        for j, (_, roles, mixed) in enumerate(col_defs):
            if mixed:
                fname = f"{sid}__T0.0__N5__roles{roles}__multimodel.jsonl"
                p = best_existing_path(fname, MIXED_DIRS)
            else:
                fname = f"{sid}__T0.0__N5__roles{roles}.jsonl"
                p = best_existing_path(fname, [CHAIR_REP_DIR, RAW_DIR]) if roles else best_existing_path(fname, [RAW_DIR])
            lam, _, _, n = compute_lambda(p, n_boot=0)
            grid[i, j] = lam
            n_grid[i, j] = n

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    vmax = max(0.10, np.nanmax(np.abs(grid)) * 1.05)
    vmin = min(-0.02, np.nanmin(grid) * 1.05)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")

    for x in np.arange(-0.5, len(col_defs), 1):
        ax.axvline(x, color="white", lw=0.8)
    for y in np.arange(-0.5, len(scenarios), 1):
        ax.axhline(y, color="white", lw=0.5)
    ax.axvline(1.5, color="#CCCCCC", lw=1.3)

    for i in range(len(scenarios)):
        for j in range(len(col_defs)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#AAAAAA")
                continue
            txtc = "white" if abs(v) > 0.04 else "#333333"
            ax.text(j, i - 0.08, f"{v:+.3f}", ha="center", va="center", fontsize=5.8, color=txtc, fontweight="bold")
            ax.text(j, i + 0.20, f"n={int(n_grid[i, j])}", ha="center", va="center", fontsize=4.8, color=txtc)

    ax.set_xticks(range(len(col_defs)))
    ax.set_xticklabels([c[0] for c in col_defs], fontsize=7)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([f"{sid} — {scenario_labels[sid]}" for sid in scenarios], fontsize=7)
    ax.set_title("Full cross-scenario 2×2 matrix (T=0, N=5)")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$\hat{\lambda}$", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.tight_layout()
    out = FIG_DIR / "fig4_landscape_2x2_full.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build updated publication main figures")
    parser.add_argument("--fig", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    globals()["N_BOOT"] = args.n_boot

    FIG_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    builders = {1: fig1, 2: fig2, 3: fig3, 4: fig4}
    for fnum in args.fig:
        fn = builders.get(fnum)
        if fn is not None:
            print(f"Building figure {fnum} ...")
            fn(rng)
    print(f"Done. Saved outputs in {FIG_DIR}")


if __name__ == "__main__":
    main()
