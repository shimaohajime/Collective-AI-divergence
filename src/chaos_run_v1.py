#!/usr/bin/env python3
"""
chaos_run_v1.py — Run N replicates of one setting in parallel, then compute chaos metrics.

Usage (from project root):
    python src/chaos_run_v1.py
    python src/chaos_run_v1.py --n-replicates 20 --max-workers 5
    python src/chaos_run_v1.py --scenario-id AI-01 --temperature 1.2
    python src/chaos_run_v1.py --analyze-only   # re-run analysis on existing data

Chaos metrics computed:
    1. Decision distribution and flip rate
    2. Per-round mean pairwise divergence of committee state trajectories
    3. Empirical divergence growth rate (proxy for Lyapunov exponent)
    4. Per-agent preference entropy and switch count
    5. Time-to-majority distribution
"""
import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from llm_dynamics_v1 import Condition, run_replicates_parallel, run_ws_committee

# ── Default configuration ─────────────────────────────────────────────────────

SCENARIO_ID   = "IM-01"
TEMPERATURE   = 0.7
N_AGENTS      = 5
ROLES_ENABLED = True
K_WINDOW      = 15
ROUNDS        = 20
MODEL         = "gpt-4.1-mini"
TRUNCATION    = "disabled"

# Default heterogeneous multi-model committee assignment (5 agents, each a different model).
# Maps role name → (provider, model_name).  Edit here to change the line-up.
DEFAULT_MULTIMODEL: dict[str, tuple[str, str]] = {
    "Chair":    ("openai",    "gpt-4.1"),
    "Welfare":  ("anthropic", "claude-sonnet-4-6"),
    "Rights":   ("gemini",    "gemini-2.5-flash"),
    "Equity":   ("grok",      "grok-3-mini"),
    "Security": ("openai",    "gpt-4.1-mini"),
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_scenarios(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {s["scenario_id"]: s["text"] for s in data["scenarios"]}


def save_runs(out_path: Path, runs_data: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in runs_data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_completed_replicates(path: Path) -> set[int]:
    """Return set of replicate indices already saved in JSONL (for resuming)."""
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    done.add(rec["replicate"])
                except Exception:
                    pass
    return done


def append_run(path: Path, rec: dict[str, Any]) -> None:
    """Append a single replicate record to JSONL (atomic enough for our use case)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Trajectory extraction ─────────────────────────────────────────────────────

def extract_trajectories(runs_data: list[dict[str, Any]], n_rounds: int, roles: list[str]) -> np.ndarray:
    """
    Return array of shape (R, T, 3*N) where:
        R = number of replicates
        T = number of rounds
        3*N = concatenated pref vectors of all N agents in role order

    State at round t = [pref_role0_t, pref_role1_t, ..., pref_roleN-1_t]
    Each pref is (pA, pB, pC) → 3 floats.
    """
    R = len(runs_data)
    N = len(roles)
    trajs = np.full((R, n_rounds, 3 * N), np.nan)

    for r_idx, rec in enumerate(runs_data):
        turns = rec["run"]["turns"]
        # Build {(round_num, role) -> pref}
        pref_map: dict[tuple[int, str], list[float]] = {}
        for turn in turns:
            key = (turn["round_num"], turn["role"])
            pref_map[key] = turn["pref"]

        for t in range(n_rounds):
            round_num = t + 1
            state = []
            for role in roles:
                pref = pref_map.get((round_num, role), [1/3, 1/3, 1/3])
                state.extend(pref)
            trajs[r_idx, t, :] = state

    return trajs


def extract_mean_trajectories(trajs: np.ndarray, n_agents: int) -> np.ndarray:
    """
    Reduce each state to the committee mean pref: shape (R, T, 3).
    """
    R, T, D = trajs.shape
    mean_trajs = np.zeros((R, T, 3))
    for n in range(n_agents):
        mean_trajs += trajs[:, :, 3*n:3*n+3]
    mean_trajs /= n_agents
    return mean_trajs


# ── Chaos metrics ─────────────────────────────────────────────────────────────

def mean_pairwise_distance(trajs: np.ndarray) -> np.ndarray:
    """
    Per-round mean pairwise L2 distance across all replicate pairs.
    Returns array of shape (T,).
    trajs shape: (R, T, D)
    """
    R, T, D = trajs.shape
    dists = np.zeros(T)
    n_pairs = 0
    for i in range(R):
        for j in range(i + 1, R):
            diff = trajs[i] - trajs[j]  # (T, D)
            dists += np.sqrt((diff ** 2).sum(axis=1))  # (T,)
            n_pairs += 1
    if n_pairs > 0:
        dists /= n_pairs
    return dists


def estimate_divergence_rate(dists: np.ndarray, skip_rounds: int = 2) -> float:
    """
    Fit log(dist_t) ~ lambda * t on rounds [skip_rounds:] to estimate
    an empirical divergence rate (proxy for max Lyapunov exponent).
    Returns lambda (positive = diverging, negative = converging).
    """
    T = len(dists)
    t_vals = np.arange(T)[skip_rounds:]
    d_vals = dists[skip_rounds:]
    valid = d_vals > 1e-10
    if valid.sum() < 3:
        return float("nan")
    t_fit = t_vals[valid].astype(float)
    d_fit = np.log(d_vals[valid])
    # linear regression: d_fit = lam * t_fit + c
    lam = float(np.polyfit(t_fit, d_fit, 1)[0])
    return lam


def decision_flip_rate(runs_data: list[dict[str, Any]]) -> tuple[dict[str, int], float]:
    """
    Returns (decision_counts, flip_rate).
    flip_rate = 1 - (plurality_count / total).
    """
    from collections import Counter
    decisions = [(rec["run"]["final_decision"] or {}).get("decision", "?") for rec in runs_data]
    counts = dict(Counter(decisions))
    total = len(decisions)
    plurality = max(counts.values()) if counts else 0
    flip_rate = 1.0 - plurality / total if total > 0 else float("nan")
    return counts, flip_rate


def time_to_majority(runs_data: list[dict[str, Any]], n_agents: int) -> list[int]:
    """
    Per replicate: first round where > n_agents/2 agents share the same top-pref option.
    Returns list of round numbers (or ROUNDS+1 if never reached).
    """
    threshold = n_agents // 2 + 1
    results = []
    for rec in runs_data:
        turns = rec["run"]["turns"]
        rounds_grouped: dict[int, list[list[float]]] = {}
        for turn in turns:
            rn = turn["round_num"]
            rounds_grouped.setdefault(rn, []).append(turn["pref"])

        ttm = rec["run"]["condition"]["rounds"] + 1
        for rn in sorted(rounds_grouped):
            prefs = rounds_grouped[rn]
            top_options = [p.index(max(p)) for p in prefs]
            from collections import Counter
            mc = Counter(top_options).most_common(1)[0][1]
            if mc >= threshold:
                ttm = rn
                break
        results.append(ttm)
    return results


def per_agent_switches(runs_data: list[dict[str, Any]], roles: list[str]) -> dict[str, list[int]]:
    """
    Per role: list (across replicates) of how many times top-pref option changes across rounds.
    """
    switches: dict[str, list[int]] = {r: [] for r in roles}
    for rec in runs_data:
        turns = rec["run"]["turns"]
        # Build {role -> [pref_r1, pref_r2, ...]}
        role_prefs: dict[str, list[list[float]]] = {r: [] for r in roles}
        for turn in sorted(turns, key=lambda x: x["round_num"]):
            if turn["role"] in role_prefs:
                role_prefs[turn["role"]].append(turn["pref"])
        for role in roles:
            prefs = role_prefs[role]
            sw = sum(
                1 for i in range(1, len(prefs))
                if prefs[i].index(max(prefs[i])) != prefs[i-1].index(max(prefs[i-1]))
            )
            switches[role].append(sw)
    return switches


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(
    runs_data: list[dict[str, Any]],
    trajs: np.ndarray,
    n_agents: int,
    roles: list[str],
    n_rounds: int,
    scenario_id: str,
    condition_label: str,
) -> None:
    R = len(runs_data)
    print()
    print("=" * 72)
    print(f"CHAOS ANALYSIS  |  scenario={scenario_id}  {condition_label}  N_reps={R}")
    print("=" * 72)

    # 1. Decision distribution
    counts, flip_rate = decision_flip_rate(runs_data)
    print(f"\n[1] Decision distribution: {counts}")
    print(f"    Flip rate (1 - plurality/N): {flip_rate:.3f}")

    # 2. Trajectory divergence
    mean_trajs = extract_mean_trajectories(trajs, n_agents)  # (R, T, 3)
    full_dists = mean_pairwise_distance(trajs)               # full state (R, T, 3N)
    mean_dists = mean_pairwise_distance(mean_trajs)          # committee mean (R, T, 3)

    print(f"\n[2] Mean pairwise divergence (committee mean pref trajectory, L2):")
    print(f"    {'Round':<8} {'Dist':>8}  {'|'}")
    max_d = mean_dists.max() if mean_dists.max() > 0 else 1.0
    for t in range(n_rounds):
        bar_len = int(40 * mean_dists[t] / max_d)
        bar = "█" * bar_len
        print(f"    {t+1:<8} {mean_dists[t]:>8.4f}  {bar}")

    # 3. Divergence rate
    lam_full = estimate_divergence_rate(full_dists)
    lam_mean = estimate_divergence_rate(mean_dists)
    print(f"\n[3] Empirical divergence rate λ (log-linear fit, skip first 2 rounds):")
    print(f"    Full state (3×N dim):     λ = {lam_full:+.4f}  {'[DIVERGING]' if lam_full > 0 else '[CONVERGING]'}")
    print(f"    Committee mean (3 dim):   λ = {lam_mean:+.4f}  {'[DIVERGING]' if lam_mean > 0 else '[CONVERGING]'}")

    # 4. Per-agent switches
    sw = per_agent_switches(runs_data, roles)
    print(f"\n[4] Preference switches per agent (mean ± std across {R} replicates):")
    for role in roles:
        vals = sw[role]
        print(f"    {role:<12}  mean={np.mean(vals):.2f}  std={np.std(vals):.2f}  min={min(vals)}  max={max(vals)}")

    # 5. Time-to-majority
    ttm = time_to_majority(runs_data, n_agents)
    never = sum(1 for t in ttm if t > n_rounds)
    print(f"\n[5] Time-to-majority (first round with >{n_agents//2+1} agents agreeing on top option):")
    print(f"    mean={np.mean(ttm):.1f}  median={np.median(ttm):.0f}  never={never}/{R}")
    print(f"    distribution: {sorted(ttm)}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="chaos_run_v1: N replicates of one setting → chaos metrics"
    )
    parser.add_argument("--scenario-id",    default=SCENARIO_ID)
    parser.add_argument("--temperature",    type=float, default=TEMPERATURE)
    parser.add_argument("--n-agents",       type=int,   default=N_AGENTS)
    parser.add_argument("--roles-enabled",  action="store_true", default=ROLES_ENABLED)
    parser.add_argument("--no-roles",       action="store_true", help="Override to disable roles")
    parser.add_argument("--k-window",       type=int,   default=K_WINDOW)
    parser.add_argument("--rounds",         type=int,   default=ROUNDS)
    parser.add_argument("--n-replicates",   type=int,   default=20)
    parser.add_argument("--max-workers",    type=int,   default=5)
    parser.add_argument("--model",          default=MODEL)
    parser.add_argument("--out-dir",        default="data/raw/chaos_v1")
    parser.add_argument("--scenarios-file", default="src/ai_chaos_scenarios_v1.json")
    parser.add_argument("--analyze-only",   action="store_true",
                        help="Skip running; load existing JSONL and re-run analysis")
    parser.add_argument("--provider",       default="openai",
                        help="API provider: openai, anthropic, gemini, grok, perplexity")
    parser.add_argument("--protocol-variant", default="ws",
                        choices=["ws", "no_feedback", "one_shot"],
                        help="Committee protocol variant: full windowed-summary, iterative without transcript feedback, or one-shot voting.")
    parser.add_argument("--ablate-role",    default=None, nargs="+",
                        help="Role(s) to ablate (remove mandate). E.g. --ablate-role Chair")
    parser.add_argument("--multimodel",     action="store_true",
                        help="Use heterogeneous multi-model committee (DEFAULT_MULTIMODEL map)")
    parser.add_argument("--multimodel-config", default=None,
                        help="JSON string overriding DEFAULT_MULTIMODEL, e.g. "
                             "'{\"Chair\":[\"openai\",\"gpt-4.1\"]}'")
    args = parser.parse_args()

    roles_enabled = args.roles_enabled and not args.no_roles
    scenario_id   = args.scenario_id
    condition_label = f"temp={args.temperature} N={args.n_agents} roles={roles_enabled} protocol={args.protocol_variant}"

    # Resolve multi-model assignment
    agent_models: dict[str, tuple[str, str]] | None = None
    if args.multimodel:
        agent_models = dict(DEFAULT_MULTIMODEL)
        if args.multimodel_config:
            overrides = json.loads(args.multimodel_config)
            for role, prov_mdl in overrides.items():
                agent_models[role] = tuple(prov_mdl)  # type: ignore[assignment]

    out_dir  = Path(args.out_dir)
    provider_tag   = args.provider if args.provider != "openai" else ""
    model_tag      = args.model.replace("/", "-") if args.model != MODEL else ""
    rounds_tag     = f"R{args.rounds}" if args.rounds != ROUNDS else ""
    protocol_tag   = args.protocol_variant if args.protocol_variant != "ws" else ""
    ablate_tag     = "ablate-" + "-".join(sorted(args.ablate_role)) if args.ablate_role else ""
    multimodel_tag = "multimodel" if args.multimodel else ""
    tag_suffix   = "__".join(filter(None, [rounds_tag, protocol_tag, ablate_tag, multimodel_tag, provider_tag, model_tag]))
    run_tag  = f"{scenario_id}__T{args.temperature}__N{args.n_agents}__roles{roles_enabled}" + (f"__{tag_suffix}" if tag_suffix else "")

    # Build role_overrides dict: each ablated role → "" (no mandate)
    role_overrides = {r: "" for r in args.ablate_role} if args.ablate_role else None
    jsonl_path = out_dir / f"{run_tag}.jsonl"

    # ── Run phase ──────────────────────────────────────────────────────────────
    if not args.analyze_only:
        scenarios = load_scenarios(Path(args.scenarios_file))
        if scenario_id not in scenarios:
            print(f"ERROR: scenario '{scenario_id}' not found in {args.scenarios_file}")
            sys.exit(1)

        cond = Condition(
            temperature=args.temperature,
            roles_enabled=roles_enabled,
            n_agents=args.n_agents,
            k_window=args.k_window,
            rounds=args.rounds,
        )

        # Resume: find already-completed replicates
        done_reps = load_completed_replicates(jsonl_path)
        missing   = [i for i in range(args.n_replicates) if i not in done_reps]

        print(f"chaos_run_v1  |  {scenario_id}  {condition_label}")
        if agent_models:
            mm_summary = "  ".join(f"{r}→{p}/{m}" for r, (p, m) in agent_models.items())
            print(f"multimodel: {mm_summary}")
        else:
            print(f"model={args.model}  provider={args.provider}")
        print(f"n_replicates={args.n_replicates}  max_workers={args.max_workers}")
        print(f"output → {jsonl_path}")
        if done_reps:
            print(f"Resuming: {len(done_reps)} already done, {len(missing)} remaining.")
        print()

        runs_data = []
        if not missing:
            print("All replicates already complete. Loading existing data.")
            runs_data = load_runs(jsonl_path)
        else:
            scenario_text = scenarios[scenario_id]
            n_errors = 0

            def _worker(rep_idx: int):
                return rep_idx, run_ws_committee(
                    scenario_id=scenario_id,
                    scenario_text=scenario_text,
                    condition=cond,
                    model=args.model,
                    truncation=TRUNCATION,
                    sleep_s=0.0,
                    provider=args.provider,
                    role_overrides=role_overrides,
                    agent_models=agent_models,
                    protocol_variant=args.protocol_variant,
                )

            with ThreadPoolExecutor(max_workers=min(args.max_workers, len(missing))) as ex:
                futures = {ex.submit(_worker, i): i for i in missing}
                for fut in as_completed(futures):
                    rep_idx = futures[fut]
                    try:
                        rep_idx, run = fut.result()
                        rec = {
                            "replicate":   rep_idx,
                            "scenario_id": scenario_id,
                            "model":       args.model,
                            "protocol_variant": args.protocol_variant,
                            "saved_utc":   now_utc(),
                            "run":         asdict(run),
                        }
                        append_run(jsonl_path, rec)
                        decision = (run.final_decision or {}).get("decision", "?")
                        n_done = len(load_completed_replicates(jsonl_path))
                        print(f"  rep {rep_idx:02d} done  decision={decision}  "
                              f"({n_done}/{args.n_replicates} total)")
                    except Exception as e:
                        n_errors += 1
                        print(f"  ERROR rep {rep_idx}: {e}")

            print(f"\nCompleted run phase. errors={n_errors}  output → {jsonl_path}")
            runs_data = load_runs(jsonl_path)
    else:
        if not jsonl_path.exists():
            print(f"ERROR: no data file found at {jsonl_path}")
            sys.exit(1)
        runs_data = load_runs(jsonl_path)
        print(f"Loaded {len(runs_data)} runs from {jsonl_path}")

    # ── Analysis phase ─────────────────────────────────────────────────────────
    if not runs_data:
        print("No successful runs to analyze.")
        return

    # Infer roles from first run
    first_run  = runs_data[0]["run"]
    n_rounds   = first_run["condition"]["rounds"]
    n_agents   = first_run["condition"]["n_agents"]
    roles_flag = first_run["condition"]["roles_enabled"]

    from llm_dynamics_v1 import roles_for_n
    roles = roles_for_n(n_agents)
    if not roles_flag:
        roles = [f"Agent{i+1}" for i in range(n_agents)]
        # Re-map role names from turns (they may be Chair/Welfare/... even with roles_enabled=False)
        turn_roles = sorted({t["role"] for t in first_run["turns"]})
        roles = turn_roles  # use actual role names from data

    trajs = extract_trajectories(runs_data, n_rounds, roles)
    print_report(runs_data, trajs, n_agents, roles, n_rounds, scenario_id, condition_label)


if __name__ == "__main__":
    main()
