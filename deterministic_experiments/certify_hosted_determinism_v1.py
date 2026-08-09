#!/usr/bin/env python3
"""
certify_hosted_determinism_v1.py

Run the exact same hosted committee configuration multiple times and certify
that the full serialized artifact hash is identical across reruns.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chaos_run_v1 import load_scenarios  # noqa: E402
from llm_dynamics_v1 import Condition  # noqa: E402

from hosted_local_runner_v1 import (  # noqa: E402
    HostedDeterministicBackend,
    canonical_json_hash,
    run_hosted_ws_committee,
    run_to_canonical_record,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify full-pipeline determinism for hosted committee runs.")
    parser.add_argument("--scenario-id", default="IM-01")
    parser.add_argument("--roles-enabled", action="store_true", help="Enable role mandate text. Default is roles=False for AAAI-facing runs.")
    parser.add_argument("--no-roles", action="store_true", help="Disable role mandate text while keeping the same named turn order.")
    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument("--k-window", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scenarios-file", default="src/ai_chaos_scenarios_v1.json")
    parser.add_argument("--out-dir", default="data/raw/deterministic_hosted/certification_v1")
    args = parser.parse_args()

    roles_enabled = bool(args.roles_enabled and not args.no_roles)
    scenario_text = load_scenarios(Path(args.scenarios_file))[args.scenario_id]
    condition = Condition(
        temperature=0.0,
        roles_enabled=roles_enabled,
        n_agents=args.n_agents,
        k_window=args.k_window,
        rounds=args.rounds,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.scenario_id}__roles{roles_enabled}__N{args.n_agents}__R{args.rounds}__seed{args.seed}"
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

    rows: list[dict[str, Any]] = []
    hashes: list[str] = []

    print(f"certify_hosted_determinism_v1  |  scenario={args.scenario_id}  roles={roles_enabled}")
    print(f"model={args.model_name}  dtype={args.dtype}  device={args.device}  runs={args.n_runs}")
    print(f"output → {jsonl_path}")

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for rep in range(args.n_runs):
            run = run_hosted_ws_committee(
                scenario_id=args.scenario_id,
                scenario_text=scenario_text,
                condition=condition,
                backend=backend,
            )
            payload = run_to_canonical_record(run, backend_metadata=backend_meta)
            artifact_hash = canonical_json_hash(payload)
            hashes.append(artifact_hash)
            row = {
                "saved_utc": now_utc(),
                "replicate": rep,
                "artifact_hash": artifact_hash,
                "decision": (run.final_decision or {}).get("decision", "?"),
                **payload,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  rep {rep + 1:03d}/{args.n_runs}  hash={artifact_hash[:12]}  decision={row['decision']}")

    counts = Counter(hashes)
    exact_match_rate = max(counts.values()) if counts else 0
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
        "backend": backend_meta,
        "n_runs": args.n_runs,
        "n_unique_hashes": len(counts),
        "exact_full_pipeline_match_rate": f"{exact_match_rate}/{args.n_runs}",
        "all_hashes_identical": len(counts) == 1,
        "hash_counts": dict(counts),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
