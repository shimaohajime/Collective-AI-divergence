#!/bin/bash
# (7/F1) Direct falsification test at T=0:
# Preserve roles but collapse memory to k_window=1.
# If early exponential divergence weakens materially, this supports
# feedback-memory dependence of instability amplification.
# Scenarios: IM-01, CL-01

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".env" ]; then
  set -a; source .env; set +a
else
  echo "ERROR: no .env file found. Copy .env.example to .env and add API keys."
  exit 1
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"

MAX_WORKERS=1
N_REPS=20
OUT_DIR="data/raw/chaos_v1_falsification_kw1_t0"
SCENARIOS=(IM-01 CL-01)

mkdir -p /tmp
echo "=== (7/F1) Falsification (k_window=1) start ==="
echo "Output dir: ${OUT_DIR}"
echo "Scenarios: ${SCENARIOS[*]}"
echo

for sid in "${SCENARIOS[@]}"; do
  echo "--- ${sid} roles=True k_window=1 ---"
  python3.11 src/chaos_run_v1.py \
    --scenario-id "${sid}" --temperature 0.0 --roles-enabled \
    --k-window 1 \
    --n-replicates "${N_REPS}" --max-workers "${MAX_WORKERS}" \
    --out-dir "${OUT_DIR}" \
    > "/tmp/FALSIFY_KW1_${sid}.log" 2>&1
  tail -n 5 "/tmp/FALSIFY_KW1_${sid}.log" || true
  echo
done

echo "=== (7/F1) Falsification (k_window=1) done ==="
