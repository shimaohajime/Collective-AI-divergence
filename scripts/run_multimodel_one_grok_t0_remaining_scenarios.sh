#!/bin/bash
# P0-1: One-Grok multimodel 2x2 at T=0 over remaining 11 scenarios (excluding IM-01).
# Runs sequentially for cost/rate safety; each condition is resumable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".env" ]; then
  set -a; source .env; set +a
else
  echo "ERROR: no .env file found. Copy .env.example to .env and add API keys."
  exit 1
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required in .env}"
: "${CLAUDE_API_KEY:?CLAUDE_API_KEY is required in .env}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY is required in .env}"
if [ -z "${GROK_API_KEY:-}" ] && [ -z "${GROK_API_KE:-}" ]; then
  echo "ERROR: GROK_API_KEY (or GROK_API_KE) is required in .env"
  exit 1
fi

MAX_WORKERS=1
N_REPS=20
OUT_DIR="data/raw/chaos_v1_one_grok_t0_matrix"
MM_CFG='{"Chair":["openai","gpt-4.1"],"Welfare":["anthropic","claude-sonnet-4-6"],"Rights":["gemini","gemini-2.5-flash"],"Equity":["grok","grok-3-mini"],"Security":["openai","gpt-4.1-mini"]}'

SCENARIOS=(
  IM-02
  HL-01
  HL-02
  IN-01
  IN-02
  CL-01
  CL-04
  SP-01
  SP-03
  AI-01
  AI-02
)

mkdir -p /tmp

echo "=== P0-1 start: one-grok multimodel T=0 matrix on remaining scenarios ==="
echo "Output dir: ${OUT_DIR}"
echo "Scenarios: ${SCENARIOS[*]}"
echo "n_reps=${N_REPS}, max_workers=${MAX_WORKERS}"
echo

for sid in "${SCENARIOS[@]}"; do
  echo "--- ${sid} roles=False ---"
  python3.11 src/chaos_run_v1.py \
    --scenario-id "${sid}" --temperature 0.0 --no-roles --multimodel \
    --multimodel-config "$MM_CFG" \
    --n-replicates "${N_REPS}" --max-workers "${MAX_WORKERS}" \
    --out-dir "${OUT_DIR}" \
    > "/tmp/P0_ONEGROK_${sid}_rolesFalse_T0.log" 2>&1
  tail -n 5 "/tmp/P0_ONEGROK_${sid}_rolesFalse_T0.log" || true

  echo "--- ${sid} roles=True ---"
  python3.11 src/chaos_run_v1.py \
    --scenario-id "${sid}" --temperature 0.0 --roles-enabled --multimodel \
    --multimodel-config "$MM_CFG" \
    --n-replicates "${N_REPS}" --max-workers "${MAX_WORKERS}" \
    --out-dir "${OUT_DIR}" \
    > "/tmp/P0_ONEGROK_${sid}_rolesTrue_T0.log" 2>&1
  tail -n 5 "/tmp/P0_ONEGROK_${sid}_rolesTrue_T0.log" || true

  echo
done

echo "=== P0-1 done ==="
