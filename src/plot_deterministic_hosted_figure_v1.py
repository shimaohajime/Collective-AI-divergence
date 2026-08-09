#!/usr/bin/env python3
"""
plot_deterministic_hosted_figure_v1.py

Build a compact main-text figure for the deterministic hosted extension:
  A. HL-01 perturbation divergence curves D(t) under deterministic hosting
  B. Cross-scenario decision fragility under perturbations
  C. Cross-scenario perturbation divergence slope

Outputs:
  paper/figures/fig5_deterministic_hosted.pdf
  paper/figures/fig5_deterministic_hosted.png
  paper/figures/fig5_deterministic_hosted_metrics.csv
  paper/figures/fig5_deterministic_family_metrics.csv
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/llm_chaos_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/llm_chaos_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    "lines.linewidth": 1.4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
}

C_ROLES = "#1565C0"
C_NOROLES = "#616161"
C_CONNECT = "#D5D9DE"
C_GRID = "#EAECEF"
C_TEXT = "#333333"

ROOT = Path(__file__).resolve().parent.parent
ROLES_DIR = ROOT / "deterministic_experiments" / "fetched_vm1_roles"
VM1_NOROLES_DIR = ROOT / "deterministic_experiments" / "fetched_vm1_noroles"
VM2_NOROLES_DIR = ROOT / "deterministic_experiments" / "fetched_vm2_noroles"
FIG_DIR = ROOT / "paper" / "figures"

SCENARIOS = [
    "IM-01", "IM-02", "HL-01", "HL-02", "IN-01", "IN-02",
    "CL-01", "CL-04", "SP-01", "SP-03", "AI-01", "AI-02",
]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def choose_noroles_path(scenario_id: str) -> Path:
    vm2 = VM2_NOROLES_DIR / f"{scenario_id}__rolesFalse__N5__R20.jsonl"
    vm1 = VM1_NOROLES_DIR / f"{scenario_id}__rolesFalse__N5__R20.jsonl"
    n_vm2 = count_lines(vm2)
    n_vm1 = count_lines(vm1)

    if n_vm2 >= 20:
        return vm2
    if n_vm1 >= 20:
        return vm1
    return vm2 if n_vm2 >= n_vm1 else vm1


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def committee_mean_trajectory(row: dict) -> np.ndarray:
    turns = row["run"]["turns"]
    n_rounds = max(turn["round_num"] for turn in turns)
    out = np.zeros((n_rounds, 3), dtype=float)
    for round_num in range(1, n_rounds + 1):
        prefs = [turn["pref"] for turn in turns if turn["round_num"] == round_num]
        out[round_num - 1] = np.mean(np.asarray(prefs, dtype=float), axis=0)
    return out


def mean_pairwise_distance(trajs: np.ndarray) -> np.ndarray:
    n_runs = trajs.shape[0]
    pair_curves: list[np.ndarray] = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            pair_curves.append(np.linalg.norm(trajs[i] - trajs[j], axis=1))
    return np.mean(np.vstack(pair_curves), axis=0)


def estimate_lambda(distance_curve: np.ndarray, start_round: int = 3) -> float:
    rounds = np.arange(1, len(distance_curve) + 1, dtype=float)
    mask = (rounds >= start_round) & (distance_curve > 1e-10)
    if mask.sum() < 3:
        return float("nan")
    return float(np.polyfit(rounds[mask], np.log(distance_curve[mask]), 1)[0])


def decision_counts(rows: Iterable[dict]) -> Counter:
    return Counter(row["run"]["final_decision"]["decision"] for row in rows)


def decision_fragility(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return 1.0 - max(counts.values()) / total


def build_metrics() -> list[dict]:
    metrics: list[dict] = []
    for scenario_id in SCENARIOS:
        paths = {
            "roles": ROLES_DIR / f"{scenario_id}__rolesTrue__N5__R20.jsonl",
            "noroles": choose_noroles_path(scenario_id),
        }
        for condition, path in paths.items():
            rows = load_jsonl(path)
            trajs = np.asarray([committee_mean_trajectory(row) for row in rows], dtype=float)
            distance_curve = mean_pairwise_distance(trajs)
            counts = decision_counts(rows)
            metrics.append(
                {
                    "scenario_id": scenario_id,
                    "condition": condition,
                    "source_file": path.name,
                    "n_variants": len(rows),
                    "lambda_pert": estimate_lambda(distance_curve),
                    "decision_fragility": decision_fragility(counts),
                    "decision_A": counts.get("A", 0),
                    "decision_B": counts.get("B", 0),
                    "decision_C": counts.get("C", 0),
                    "distance_curve": distance_curve.tolist(),
                }
            )
    return metrics


def build_family_metrics() -> list[dict]:
    metrics: list[dict] = []
    for scenario_id in SCENARIOS:
        paths = {
            "roles": ROLES_DIR / f"{scenario_id}__rolesTrue__N5__R20.jsonl",
            "noroles": choose_noroles_path(scenario_id),
        }
        for condition, path in paths.items():
            rows = load_jsonl(path)
            for family in ("surface_rephrasing", "formatting_changes"):
                subrows = [row for row in rows if row["family"] == family]
                trajs = np.asarray([committee_mean_trajectory(row) for row in subrows], dtype=float)
                distance_curve = mean_pairwise_distance(trajs)
                counts = decision_counts(subrows)
                metrics.append(
                    {
                        "scenario_id": scenario_id,
                        "condition": condition,
                        "family": family,
                        "source_file": path.name,
                        "n_variants": len(subrows),
                        "lambda_pert": estimate_lambda(distance_curve),
                        "decision_fragility": decision_fragility(counts),
                        "decision_A": counts.get("A", 0),
                        "decision_B": counts.get("B", 0),
                        "decision_C": counts.get("C", 0),
                    }
                )
    return metrics


def plot_panel_a(ax: plt.Axes, metric_rows: list[dict]) -> None:
    hl_rows = {(row["condition"]): row for row in metric_rows if row["scenario_id"] == "HL-01"}
    x = np.arange(1, 21)

    for condition, color, label, marker in [
        ("roles", C_ROLES, "Roles", "o"),
        ("noroles", C_NOROLES, "No roles", "s"),
    ]:
        row = hl_rows[condition]
        y = np.asarray(row["distance_curve"], dtype=float)
        frag = row["decision_fragility"]
        lam = row["lambda_pert"]
        ax.plot(x, y, color=color, marker=marker, ms=2.6, label=label, zorder=3)
        ax.fill_between(x, 0, y, color=color, alpha=0.10, zorder=1)
        ax.text(
            x[-1] + 0.15,
            y[-1],
            f"{label}\n$\\hat{{\\lambda}}_{{pert}}$={lam:.3f}\nfrag.={frag:.2f}",
            color=color,
            fontsize=6.8,
            va="center",
        )

    ax.set_xlim(1, 22.2)
    ax.set_ylim(0, None)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Mean pairwise distance D(t)")
    ax.set_title("Nearby initial conditions diverge (HL-01)")
    ax.grid(axis="y", color=C_GRID, linewidth=0.7)
    ax.text(-0.08, 1.03, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")


def paired_metric_panel(
    ax: plt.Axes,
    metric_rows: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    sort_key: str,
    panel_label: str,
) -> None:
    by_scenario: dict[str, dict[str, dict]] = {}
    for row in metric_rows:
        by_scenario.setdefault(row["scenario_id"], {})[row["condition"]] = row

    ordered = sorted(
        SCENARIOS,
        key=lambda sid: max(
            by_scenario[sid]["roles"][sort_key],
            by_scenario[sid]["noroles"][sort_key],
        ),
        reverse=True,
    )

    x = np.arange(len(ordered))
    dx = 0.15
    roles_vals = [by_scenario[sid]["roles"][metric_key] for sid in ordered]
    noroles_vals = [by_scenario[sid]["noroles"][metric_key] for sid in ordered]

    for i, (rv, nv) in enumerate(zip(roles_vals, noroles_vals)):
        ax.plot([i - dx, i + dx], [rv, nv], color=C_CONNECT, lw=1.0, zorder=1)

    ax.scatter(x - dx, roles_vals, s=26, color=C_ROLES, zorder=3, label="Roles")
    ax.scatter(x + dx, noroles_vals, s=26, color=C_NOROLES, marker="s", zorder=3, label="No roles")

    ax.set_xticks(x, ordered, rotation=42, ha="right")
    ax.tick_params(axis="x", labelsize=6.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color=C_GRID, linewidth=0.7)
    ax.text(-0.12, 1.03, panel_label, transform=ax.transAxes, fontsize=11, fontweight="bold")


def write_metrics_csv(metric_rows: list[dict], path: Path) -> None:
    fields = [
        "scenario_id",
        "condition",
        "source_file",
        "n_variants",
        "lambda_pert",
        "decision_fragility",
        "decision_A",
        "decision_B",
        "decision_C",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow({k: row[k] for k in fields})


def write_family_metrics_csv(metric_rows: list[dict], path: Path) -> None:
    fields = [
        "scenario_id",
        "condition",
        "family",
        "source_file",
        "n_variants",
        "lambda_pert",
        "decision_fragility",
        "decision_A",
        "decision_B",
        "decision_C",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow({k: row[k] for k in fields})


def main() -> None:
    plt.rcParams.update(SCIENCE_RC)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = build_metrics()
    family_rows = build_family_metrics()

    fig = plt.figure(figsize=(7.45, 4.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], hspace=0.48, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    plot_panel_a(ax_a, metric_rows)
    paired_metric_panel(
        ax=ax_b,
        metric_rows=metric_rows,
        metric_key="decision_fragility",
        ylabel="Decision fragility",
        title="Decision fragility",
        sort_key="decision_fragility",
        panel_label="B",
    )
    paired_metric_panel(
        ax=ax_c,
        metric_rows=metric_rows,
        metric_key="lambda_pert",
        ylabel=r"Slope $\hat{\lambda}_{pert}$",
        title="Divergence slope",
        sort_key="decision_fragility",
        panel_label="C",
    )

    out = FIG_DIR / "fig5_deterministic_hosted.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)

    write_metrics_csv(metric_rows, FIG_DIR / "fig5_deterministic_hosted_metrics.csv")
    write_family_metrics_csv(family_rows, FIG_DIR / "fig5_deterministic_family_metrics.csv")
    print(f"Saved {out}")
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
