#!/usr/bin/env python3
"""
plot_si_v1.py — Supplementary figures for "AI Society is a Chaos"

Produces (into publication_v1/):
  sfig1_temp_sweep_full.pdf   — Full temperature sweep, both role conditions
  sfig2_per_agent_trajs.pdf   — Per-agent preference traces (Chair vs. others)
  sfig3_switch_counts.pdf     — Per-agent switch count distributions by role
  sfig4_time_to_majority.pdf  — Time-to-majority ECDF, roles vs. no-roles
  sfig5_permutation_null.pdf  — Permutation null test for H0: λ=0
  sfig6_branching_entropy.pdf — Branching entropy Γ̂ per replicate
  sfig7_semantic_perturb.pdf  — Semantic perturbation simplex (6 phrasings)

Usage (from project root):
    python src/plot_si_v1.py
    python src/plot_si_v1.py --fig 1 3 5
    python src/plot_si_v1.py --n-boot 0   # fast, no CIs
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Shared style ───────────────────────────────────────────────────────────────

SCIENCE_RC = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          8,
    "axes.titlesize":     8.5,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "lines.linewidth":    1.2,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,
}

C_ROLES   = "#1565C0"
C_NOROLES = "#C62828"

ROLE_COLORS = {
    "Chair":    "#E65100",
    "Welfare":  "#1565C0",
    "Rights":   "#2E7D32",
    "Equity":   "#6A1B9A",
    "Security": "#795548",
}

DATA_DIR   = Path("data/raw/chaos_v1")
SEM_DIR    = Path("data/raw/semantic_v1")
BRANCH_DIR = Path("data/raw/branching_v1")
FIG_DIR    = Path("publication_v1")
ROUNDS     = 20

# ── Data helpers ───────────────────────────────────────────────────────────────

def load_runs(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def committee_mean_prefs(rec: dict, n_rounds: int) -> np.ndarray:
    turns = rec["run"]["turns"]
    result = np.full((n_rounds, 3), np.nan)
    for r in range(1, n_rounds + 1):
        rprefs = [t["pref"] for t in turns if t["round_num"] == r]
        if rprefs:
            result[r - 1] = np.mean(rprefs, axis=0)
    return result


def mean_pairwise_dist(trajs: np.ndarray) -> np.ndarray:
    R, T, _ = trajs.shape
    if R < 2:
        return np.zeros(T)
    dists = []
    for i in range(R):
        for j in range(i + 1, R):
            dists.append(np.linalg.norm(trajs[i] - trajs[j], axis=1))
    return np.mean(dists, axis=0)


def fit_lambda(dists: np.ndarray, skip: int = 2) -> float:
    T = len(dists)
    if T - skip < 3:
        return float("nan")
    t = np.arange(T)[skip:]
    d = dists[skip:]
    valid = d > 0
    if valid.sum() < 2:
        return float("nan")
    lam, _ = np.polyfit(t[valid], np.log(d[valid]), 1)
    return float(lam)


def bootstrap_lambda(trajs: np.ndarray, n_boot: int,
                     rng: np.random.Generator) -> Tuple[float, float, float]:
    R = trajs.shape[0]
    lam_obs = fit_lambda(mean_pairwise_dist(trajs))
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, R, size=R)
        boot.append(fit_lambda(mean_pairwise_dist(trajs[idx])))
    boot = np.array([b for b in boot if not np.isnan(b)])
    if len(boot) == 0:
        return lam_obs, float("nan"), float("nan")
    return lam_obs, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def get_trajs(path: Path, n_rounds: int = ROUNDS) -> Optional[np.ndarray]:
    runs = load_runs(path)
    if len(runs) < 2:
        return None
    trajs = []
    for rec in runs:
        mp = committee_mean_prefs(rec, n_rounds)
        if not np.isnan(mp).all():
            trajs.append(mp)
    return np.array(trajs) if trajs else None


def compute_lambda(path: Path, n_rounds: int = ROUNDS, n_boot: int = 0,
                   rng: Optional[np.random.Generator] = None
                   ) -> Tuple[float, float, float, int]:
    trajs = get_trajs(path, n_rounds)
    if trajs is None:
        return float("nan"), float("nan"), float("nan"), 0
    if n_boot > 0 and rng is not None:
        lam, ci_lo, ci_hi = bootstrap_lambda(trajs, n_boot, rng)
    else:
        lam = fit_lambda(mean_pairwise_dist(trajs))
        ci_lo = ci_hi = float("nan")
    return lam, ci_lo, ci_hi, len(trajs)


# ── Per-agent helpers ──────────────────────────────────────────────────────────

def agent_prefs_by_role(rec: dict, n_rounds: int) -> Dict[str, np.ndarray]:
    """Return {role: (n_rounds, 3)} per-agent preference matrices."""
    turns = rec["run"]["turns"]
    result: Dict[str, np.ndarray] = {}
    for t in turns:
        role = t["role"]
        r    = t["round_num"]
        if not (1 <= r <= n_rounds):
            continue
        if role not in result:
            result[role] = np.full((n_rounds, 3), np.nan)
        result[role][r - 1] = t["pref"]
    return result


def count_switches(arr: np.ndarray) -> int:
    """Count argmax changes across rounds, ignoring NaN rows."""
    valid = arr[~np.isnan(arr[:, 0])]
    if len(valid) < 2:
        return 0
    choices = np.argmax(valid, axis=1)
    return int(np.sum(choices[1:] != choices[:-1]))


def time_to_majority(rec: dict, n_rounds: int, n_agents: int = 5) -> Optional[int]:
    """First round where majority of agents share the same top-pref option."""
    turns = rec["run"]["turns"]
    threshold = (n_agents // 2) + 1
    for r in range(1, n_rounds + 1):
        rprefs = [t["pref"] for t in turns if t["round_num"] == r]
        if len(rprefs) < threshold:
            continue
        choices = [int(np.argmax(p)) for p in rprefs]
        if Counter(choices).most_common(1)[0][1] >= threshold:
            return r
    return None


# ── Simplex helpers ────────────────────────────────────────────────────────────

def bary_to_cart(pA, pB, pC):
    """Barycentric → Cartesian. A=top, B=bottom-left, C=bottom-right."""
    x = pA * 0.5 + pC * 1.0
    y = pA * (np.sqrt(3) / 2)
    return x, y


def draw_simplex_bg(ax, fontsize: float = 9):
    verts = np.array([[0.5, np.sqrt(3)/2], [0.0, 0.0], [1.0, 0.0],
                      [0.5, np.sqrt(3)/2]])
    ax.plot(verts[:, 0], verts[:, 1], "k-", lw=0.8, zorder=2)
    pad = 0.07
    ax.text(0.5, np.sqrt(3)/2 + pad, "A", ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold")
    ax.text(-0.06, -pad, "B", ha="center", va="top",
            fontsize=fontsize, fontweight="bold")
    ax.text(1.06, -pad, "C", ha="center", va="top",
            fontsize=fontsize, fontweight="bold")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.20)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Figure S1: Full temperature sweep (both role conditions) ───────────────────

def sfig1(rng: np.random.Generator, n_boot: int = 500):
    print("Building SFig 1: Full temperature sweep...")
    plt.rcParams.update(SCIENCE_RC)

    temps_str = ["0.0", "1e-05", "0.0001", "0.01", "0.05", "0.1", "0.2", "0.7"]
    temps_val = [0.0, 1e-5, 1e-4, 0.01, 0.05, 0.1, 0.2, 0.7]
    xlabels   = ["0", r"$10^{-5}$", r"$10^{-4}$", "0.01", "0.05", "0.1", "0.2", "0.7"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)

    for ax, roles, color, label, panel in [
        (axes[0], True,  C_ROLES,   "Roles",    "A"),
        (axes[1], False, C_NOROLES, "No Roles", "B"),
    ]:
        lams, ci_los, ci_his, xs = [], [], [], []
        for i, tstr in enumerate(temps_str):
            tag = "True" if roles else "False"
            fname = f"IM-01__T{tstr}__N5__roles{tag}.jsonl"
            lam, ci_lo, ci_hi, n = compute_lambda(
                DATA_DIR / fname, n_boot=n_boot, rng=rng)
            if not np.isnan(lam):
                lams.append(lam); ci_los.append(ci_lo)
                ci_his.append(ci_hi); xs.append(i)

        xs = np.array(xs)
        lams, ci_los, ci_his = map(np.array, [lams, ci_los, ci_his])

        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.35)
        if n_boot > 0:
            ax.fill_between(xs, ci_los, ci_his, color=color, alpha=0.18)
        ax.plot(xs, lams, "o-", color=color, ms=5, lw=1.4, label=label)
        ax.set_xticks(range(len(temps_str)))
        ax.set_xticklabels(xlabels, rotation=35, ha="right")
        ax.set_xlabel("Temperature $T$")
        ax.set_title(f"({panel}) {label}", loc="left", fontweight="bold", pad=5)

    axes[0].set_ylabel(r"Empirical Lyapunov exponent $\hat{\lambda}$")
    fig.suptitle("Full temperature sweep — IM-01, $N=5$, 20 replicates",
                 fontsize=8.5, y=1.01)
    fig.tight_layout()
    _save(fig, "sfig1_temp_sweep_full")


# ── Figure S2: Per-agent preference traces ────────────────────────────────────

def sfig2():
    print("Building SFig 2: Per-agent preference trajectories...")
    plt.rcParams.update(SCIENCE_RC)

    path = DATA_DIR / "IM-01__T0.0__N5__rolesTrue.jsonl"
    runs = load_runs(path)
    if not runs:
        print("  No data — skipping."); return

    sample = runs[:6]
    role_order = ["Chair", "Welfare", "Rights", "Equity", "Security"]
    rounds_ax  = np.arange(1, ROUNDS + 1)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    axes = axes.flatten()

    for ax_idx, rec in enumerate(sample):
        ax = axes[ax_idx]
        by_role = agent_prefs_by_role(rec, ROUNDS)

        for role in role_order:
            if role not in by_role:
                continue
            arr   = by_role[role]
            valid = ~np.isnan(arr[:, 0])
            c     = ROLE_COLORS.get(role, "gray")
            # Plot pref for top option: pA
            ax.plot(rounds_ax[valid], arr[valid, 0],
                    color=c, lw=1.0, alpha=0.85,
                    label=role if ax_idx == 0 else "_")
            ax.plot(rounds_ax[valid], arr[valid, 1],
                    color=c, lw=0.7, alpha=0.45, ls="--")

        ax.axhline(1/3, color="gray", lw=0.5, ls=":", alpha=0.5)
        ax.set_xlim(1, ROUNDS)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Run {ax_idx + 1}", fontsize=8)
        if ax_idx >= 3:
            ax.set_xlabel("Round")
        if ax_idx % 3 == 0:
            ax.set_ylabel(r"$p_A$ (solid), $p_B$ (dashed)")

    handles = [Line2D([0], [0], color=ROLE_COLORS.get(r, "gray"),
                      lw=1.5, label=r) for r in role_order]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(r"Per-agent preferences ($p_A$ solid, $p_B$ dashed) — "
                 "IM-01, $T=0$, roles=True; 6 representative runs",
                 fontsize=7.5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, "sfig2_per_agent_trajs")


# ── Figure S3: Per-agent switch count distributions ───────────────────────────

def sfig3():
    print("Building SFig 3: Per-agent switch count distributions...")
    plt.rcParams.update(SCIENCE_RC)

    temps_str  = ["0.0", "0.01", "0.05", "0.1", "0.2", "0.7"]
    role_order = ["Chair", "Welfare", "Rights", "Equity", "Security"]
    switch_data: Dict[str, List[int]] = {r: [] for r in role_order}

    for tstr in temps_str:
        path = DATA_DIR / f"IM-01__T{tstr}__N5__rolesTrue.jsonl"
        for rec in load_runs(path):
            by_role = agent_prefs_by_role(rec, ROUNDS)
            for role in role_order:
                if role in by_role:
                    switch_data[role].append(count_switches(by_role[role]))

    rng_jitter = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    for i, role in enumerate(role_order):
        vals = np.array(switch_data[role])
        if len(vals) == 0:
            continue
        c      = ROLE_COLORS.get(role, "gray")
        jitter = rng_jitter.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=c, alpha=0.30, s=9, zorder=2, linewidths=0)
        mean_v = np.mean(vals)
        sem_v  = np.std(vals, ddof=1) / np.sqrt(len(vals))
        ax.bar(i, mean_v, width=0.42, color=c, alpha=0.80, zorder=3,
               edgecolor="white", linewidth=0.5)
        ax.errorbar(i, mean_v, yerr=sem_v, fmt="none",
                    color="k", capsize=3, lw=1.0, zorder=4)

    n_agent_runs = sum(len(v) for v in switch_data.values())
    ax.set_xticks(range(len(role_order)))
    ax.set_xticklabels(role_order)
    ax.set_ylabel("Preference switches per run")
    ax.set_title(f"Per-agent switch counts by role\n"
                 f"(IM-01, $T\\in\\{{0,0.01,0.05,0.1,0.2,0.7\\}}$, roles=True; "
                 f"{n_agent_runs} agent-runs)",
                 fontsize=7.5)
    _save(fig, "sfig3_switch_counts")


# ── Figure S4: Time-to-majority ECDF ─────────────────────────────────────────

def sfig4():
    print("Building SFig 4: Time-to-majority distribution...")
    plt.rcParams.update(SCIENCE_RC)

    conditions = [
        ("IM-01__T0.0__N5__rolesTrue.jsonl",  C_ROLES,   "-",  "Roles ($T=0$)"),
        ("IM-01__T0.7__N5__rolesTrue.jsonl",  C_ROLES,   "--", "Roles ($T=0.7$)"),
        ("IM-01__T0.0__N5__rolesFalse.jsonl", C_NOROLES, "-",  "No Roles ($T=0$)"),
        ("IM-01__T0.7__N5__rolesFalse.jsonl", C_NOROLES, "--", "No Roles ($T=0.7$)"),
    ]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    for fname, color, ls, label in conditions:
        runs = load_runs(DATA_DIR / fname)
        ttms = []
        for rec in runs:
            t = time_to_majority(rec, ROUNDS)
            ttms.append(t if t is not None else ROUNDS + 1)
        ttms_s = np.sort(ttms)
        ecdf   = np.arange(1, len(ttms_s) + 1) / len(ttms_s)
        ax.step(ttms_s, ecdf, where="post", color=color, ls=ls,
                lw=1.4, label=label)

    ax.axvline(ROUNDS + 0.5, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.text(ROUNDS + 0.8, 0.4, "never", color="gray",
            fontsize=6.5, rotation=90, va="center")
    ax.set_xlabel("Round of first majority")
    ax.set_ylabel("Cumulative proportion")
    ax.set_xlim(0.5, ROUNDS + 2)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Time-to-majority: ECDF across replicates (IM-01)", fontsize=8)
    _save(fig, "sfig4_time_to_majority")


# ── Figure S5: Permutation null test ─────────────────────────────────────────

def sfig5(n_perm: int = 2000):
    print("Building SFig 5: Permutation null distribution...")
    plt.rcParams.update(SCIENCE_RC)

    path  = DATA_DIR / "IM-01__T0.0__N5__rolesTrue.jsonl"
    trajs = get_trajs(path)
    if trajs is None:
        print("  No data — skipping."); return

    lam_obs = fit_lambda(mean_pairwise_dist(trajs))
    rng     = np.random.default_rng(0)
    null_lams = []
    for _ in range(n_perm):
        pt = trajs.copy()
        for r in range(pt.shape[0]):
            pt[r] = pt[r][rng.permutation(ROUNDS)]
        null_lams.append(fit_lambda(mean_pairwise_dist(pt)))
    null_lams = np.array([x for x in null_lams if not np.isnan(x)])

    p_val = np.mean(null_lams >= lam_obs)
    p_str = f"$p < 0.001$" if p_val < 0.001 else f"$p = {p_val:.3f}$"

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.hist(null_lams, bins=45, color="gray", alpha=0.70,
            edgecolor="white", linewidth=0.3,
            label=r"Permuted $\hat{\lambda}_{null}$")
    ax.axvline(lam_obs, color=C_ROLES, lw=2.0,
               label=fr"Observed $\hat{{\lambda}} = {lam_obs:.3f}$")
    ax.text(0.97, 0.94, f"{p_str}\n(permutation, $n={n_perm}$)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=C_ROLES)
    ax.set_xlabel(r"$\hat{\lambda}$ (permuted)")
    ax.set_ylabel("Count")
    ax.set_title(r"Permutation null: $H_0{:}\ \hat{\lambda} = 0$"
                 "\n(IM-01, $T=0$, roles=True)", fontsize=8)
    ax.legend(fontsize=7)
    _save(fig, "sfig5_permutation_null")


# ── Figure S6: Branching entropy ─────────────────────────────────────────────

def sfig6():
    print("Building SFig 6: Branching entropy per replicate...")
    plt.rcParams.update(SCIENCE_RC)

    rep_gammas, rep_cin, rep_cout = [], [], []
    for rep in range(5):
        fpath = BRANCH_DIR / f"IM-01__T0.7__N5__rolesTrue__K30__rep{rep}.json"
        if not fpath.exists():
            continue
        with fpath.open() as f:
            data = json.load(f)
        g = data.get("gamma_hat", data.get("Gamma_hat", data.get("gamma", None)))
        if g is not None:
            rep_gammas.append(float(g))
        c_in  = data.get("c_in",  data.get("c_within",  None))
        c_out = data.get("c_out", data.get("c_between", None))
        if c_in  is not None: rep_cin.append(float(c_in))
        if c_out is not None: rep_cout.append(float(c_out))

    if not rep_gammas:
        print("  No branching data — skipping."); return

    gam   = np.array(rep_gammas)
    g_mu  = np.mean(gam)
    g_se  = np.std(gam, ddof=1) / np.sqrt(len(gam))

    n_panels = 2 if (rep_cin and rep_cout) else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.2))
    if n_panels == 1:
        axes = [axes]

    # Panel A: Γ̂ per replicate
    ax = axes[0]
    xs = np.arange(len(gam))
    ax.bar(xs, gam, color=C_ROLES, alpha=0.75,
           edgecolor="white", linewidth=0.5)
    ax.axhline(g_mu, color=C_ROLES, lw=1.5, ls="--",
               label=fr"$\bar{{\hat{{\Gamma}}}}={g_mu:.3f}\pm{g_se:.3f}$ SE")
    ax.fill_between([-0.5, len(gam) - 0.5],
                    g_mu - g_se, g_mu + g_se,
                    color=C_ROLES, alpha=0.15)
    ax.axhline(0, color="k", lw=0.6, ls=":", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Rep {i}" for i in range(len(gam))])
    ax.set_ylabel(r"Branching entropy $\hat{\Gamma}$")
    ax.set_title("(A) $\\hat{\\Gamma}$ per replicate\n"
                 "(IM-01, $T=0.7$, $K=30$)", loc="left",
                 fontweight="bold", fontsize=8)
    ax.legend(fontsize=7)

    # Panel B: c_in vs c_out (if available)
    if n_panels == 2:
        ax2 = axes[1]
        c_in_a  = np.array(rep_cin)
        c_out_a = np.array(rep_cout)
        x2 = np.arange(len(c_in_a))
        ax2.plot(x2, c_in_a,  "o-", color=C_ROLES,   lw=1.4, label=r"$\hat{c}_{in}$")
        ax2.plot(x2, c_out_a, "s-", color=C_NOROLES, lw=1.4, label=r"$\hat{c}_{out}$")
        ax2.set_xticks(x2)
        ax2.set_xticklabels([f"Rep {i}" for i in range(len(c_in_a))])
        ax2.set_ylabel("Cosine similarity")
        ax2.set_title(r"(B) Within- vs. between-branch similarity", loc="left",
                      fontweight="bold", fontsize=8)
        ax2.legend(fontsize=7)

    fig.suptitle("Branching entropy certificate — IM-01, $T=0.7$, $K=30$, 5 replicates",
                 fontsize=8.5, y=1.01)
    fig.tight_layout()
    _save(fig, "sfig6_branching_entropy")


# ── Figure S7: Semantic perturbation simplex ──────────────────────────────────

def sfig7():
    print("Building SFig 7: Semantic perturbation simplex...")
    plt.rcParams.update(SCIENCE_RC)

    sem_path = SEM_DIR / "IM-01__T0.0__N5__rolesTrue.jsonl"
    runs = load_runs(sem_path)
    if not runs:
        print("  No semantic data — skipping."); return

    variant_labels = [
        "Canonical",
        "Formal register",
        "Conversational",
        "Legalistic",
        "Passive reframe",
        "Synonym swap",
    ][:len(runs)]

    variant_colors = [
        "#1565C0", "#C62828", "#2E7D32",
        "#6A1B9A", "#E65100", "#795548",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: simplex
    ax_simp = axes[0]
    draw_simplex_bg(ax_simp, fontsize=9)
    for rec, label, color in zip(runs, variant_labels, variant_colors):
        mp    = committee_mean_prefs(rec, ROUNDS)
        valid = ~np.isnan(mp[:, 0])
        if valid.sum() < 2:
            continue
        mp = mp[valid]
        xs, ys = bary_to_cart(mp[:, 0], mp[:, 1], mp[:, 2])
        ax_simp.plot(xs, ys, color=color, lw=1.4, alpha=0.90, label=label)
        ax_simp.scatter(xs[0],  ys[0],  s=14, color=color, alpha=0.50, zorder=4)
        ax_simp.scatter(xs[-1], ys[-1], s=24, color=color, alpha=1.0,
                        zorder=5, edgecolors="white", linewidths=0.5)
    ax_simp.legend(fontsize=6.5, loc="upper right", frameon=True, framealpha=0.9)
    ax_simp.set_title("(A) Committee trajectories in preference simplex",
                      loc="left", fontweight="bold", fontsize=8)

    # Right: divergence over rounds
    ax_div = axes[1]
    trajs = get_trajs(sem_path)
    if trajs is not None:
        D = mean_pairwise_dist(trajs)
        rounds_ax = np.arange(1, ROUNDS + 1)
        lam = fit_lambda(D)
        ax_div.plot(rounds_ax, D, color="#6A1B9A", lw=1.5)
        ax_div.set_xlabel("Round")
        ax_div.set_ylabel(r"Mean pairwise $L_2$ distance $D(t)$")
        ax_div.set_title(f"(B) Divergence across 6 variants\n"
                         fr"$\hat{{\lambda}} = {lam:.3f}$ (vs. $0.084$ for FP noise)",
                         loc="left", fontweight="bold", fontsize=8)

    fig.suptitle("Semantic perturbation — IM-01, $T=0$, roles=True\n"
                 "6 synonym/register rephrasings of the same scenario",
                 fontsize=8, y=1.01)
    fig.tight_layout()
    _save(fig, "sfig7_semantic_perturb")


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / stem}.pdf")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate SI figures")
    parser.add_argument("--fig", nargs="+", type=int,
                        default=list(range(1, 8)),
                        help="Which SI figures to build (1–7)")
    parser.add_argument("--n-boot", type=int, default=500,
                        help="Bootstrap iterations for λ CIs in SFig 1 (0=skip)")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    fig_map = {
        1: lambda: sfig1(rng, n_boot=args.n_boot),
        2: sfig2,
        3: sfig3,
        4: sfig4,
        5: sfig5,
        6: sfig6,
        7: sfig7,
    }

    for i in sorted(args.fig):
        if i in fig_map:
            fig_map[i]()

    print(f"\nDone. SI figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
