#!/usr/bin/env python3
"""
test_providers_v1.py — Run 1 replicate per model to verify API integration.

Tests that each provider:
  1. Accepts the temperature parameter correctly
  2. Returns output that parses through our STATE: format
  3. Returns valid ballot JSON
  4. Returns valid clerk decision JSON

Usage (from project root):
    python src/test_providers_v1.py                     # test all configured providers
    python src/test_providers_v1.py --providers openai anthropic
    python src/test_providers_v1.py --providers gemini grok perplexity
    python src/test_providers_v1.py --list              # show configured models
"""

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_dynamics_v1 import (
    Condition, run_ws_committee, PROVIDER_REGISTRY, _get_api_key,
)

# ── Model to test per provider ────────────────────────────────────────────────

PROVIDER_MODELS = {
    "openai":     "gpt-4.1-mini",      # cheap, already validated
    "anthropic":  "claude-sonnet-4-6",
    "gemini":     "models/gemini-2.5-flash",
    "grok":       "grok-3",
    "perplexity": "sonar-pro",
}

# Minimal 3-round condition for fast testing
TEST_CONDITION = Condition(
    temperature=0.1,
    roles_enabled=True,
    n_agents=5,
    k_window=5,
    rounds=3,          # 3 rounds instead of 20 — fast smoke test
)

SCENARIO_ID = "IM-01"
SCENARIO_TEXT = """SCENARIO_ID: IM-01
Institution: National Asylum Allocation Board
Constraint: You have 10,000 asylum grants this year (cannot exceed). Processing budget fixed.
Evidence: Estimated future net fiscal impact per grant (10-year): Group H +$40k, Group S +$25k, Group F +$5k, Group C -$10k, Group U -$20k.
Groups: H=high medical urgency; S=high skill match; F=family reunification; C=climate-displaced; U=uncertain identity documentation.
Options:
A) Prioritize humanitarian: allocate >=50% to H + C.
B) Balanced points system: weights (H 35, S 35, F 20, C 10); allocate by score.
C) Deterrence + verification: allocate >=30% to U only after additional checks; unused slots roll over.
Principle vote: Which decision rule should govern selection?
Cases:
1) A family with urgent medical need (H) but weak documentation (U risk).
2) A single applicant with rare skills (S) and strong documentation.
3) A climate-displaced applicant (C) with moderate documentation gaps.
Discrete endpoint: Choose A/B/C.
"""


def test_provider(provider: str, model: str) -> dict:
    """Run one 3-round committee with the given provider. Returns result dict."""
    result = {
        "provider": provider,
        "model": model,
        "status": "FAIL",
        "decision": None,
        "n_turns": 0,
        "error": None,
        "notes": [],
    }

    try:
        key = _get_api_key(provider)
        result["notes"].append("API key found")
    except Exception as e:
        result["error"] = f"No API key: {e}"
        return result

    try:
        run = run_ws_committee(
            scenario_id=SCENARIO_ID,
            scenario_text=SCENARIO_TEXT,
            condition=TEST_CONDITION,
            model=model,
            truncation="disabled",
            sleep_s=0.0,
            provider=provider,
        )
        result["n_turns"]  = len(run.turns)
        result["decision"] = (run.final_decision or {}).get("decision", "?")
        result["n_ballots"] = len(run.ballots)
        result["usage"]     = run.usage_total
        result["status"]    = "OK"

        # Sanity checks
        if result["decision"] not in ("A", "B", "C"):
            result["notes"].append(f"WARNING: unexpected decision '{result['decision']}'")
        if result["n_turns"] != TEST_CONDITION.rounds * 5:
            result["notes"].append(f"WARNING: expected {TEST_CONDITION.rounds * 5} turns, got {result['n_turns']}")

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="test_providers_v1: smoke-test all model APIs")
    parser.add_argument("--providers", nargs="+", default=list(PROVIDER_MODELS.keys()),
                        help="Providers to test (default: all)")
    parser.add_argument("--models",    nargs="+", default=None,
                        help="Override model IDs (parallel list to --providers)")
    parser.add_argument("--list",      action="store_true", help="Show configured models and exit")
    parser.add_argument("--rounds",    type=int, default=3,
                        help="Rounds per test run (default: 3, faster)")
    args = parser.parse_args()

    if args.list:
        print("Configured provider → model mappings:")
        for p, m in PROVIDER_MODELS.items():
            print(f"  {p:<12} {m}")
        return

    TEST_CONDITION.rounds = args.rounds

    provider_model_pairs = list(zip(
        args.providers,
        args.models if args.models else [PROVIDER_MODELS.get(p, "UNKNOWN") for p in args.providers],
    ))

    print(f"test_providers_v1  |  scenario={SCENARIO_ID}  rounds={args.rounds}")
    print(f"Testing {len(provider_model_pairs)} provider(s): {[p for p,_ in provider_model_pairs]}")
    print()

    results = []
    for provider, model in provider_model_pairs:
        print(f"  [{provider}]  model={model} ... ", end="", flush=True)
        r = test_provider(provider, model)
        results.append(r)
        status_str = f"{r['status']}"
        if r["status"] == "OK":
            status_str += f"  decision={r['decision']}  turns={r['n_turns']}  ballots={r['n_ballots']}"
            if r.get("usage"):
                u = r["usage"]
                status_str += f"  tokens={u.get('total_tokens', '?')}"
        else:
            status_str += f"  ERROR: {r['error']}"
        print(status_str)
        for note in r.get("notes", []):
            print(f"      {note}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = [r for r in results if r["status"] == "OK"]
    failed = [r for r in results if r["status"] != "OK"]
    print(f"  PASSED: {len(passed)}/{len(results)}")
    for r in passed:
        print(f"    OK   {r['provider']:<12} {r['model']}")
    for r in failed:
        print(f"    FAIL {r['provider']:<12} {r['model']}  — {r['error']}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
