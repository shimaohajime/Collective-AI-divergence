#!/bin/bash
# (4) Stabilization intervention at T=0:
# roles=True with reduced memory window (k_window=3)
# Scenarios: HL-01, CL-01, SP-03, AI-01

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
OUT_DIR="data/raw/chaos_v1_intervention_kw3_t0"
SCENARIOS=(HL-01 CL-01 SP-03 AI-01)

mkdir -p /tmp
echo "=== (4) Intervention (k_window=3) start ==="
echo "Output dir: ${OUT_DIR}"
echo "Scenarios: ${SCENARIOS[*]}"
echo

for sid in "${SCENARIOS[@]}"; do
  echo "--- ${sid} roles=True k_window=3 ---"
  python3.11 src/chaos_run_v1.py \
    --scenario-id "${sid}" --temperature 0.0 --roles-enabled \
    --k-window 3 \
    --n-replicates "${N_REPS}" --max-workers "${MAX_WORKERS}" \
    --out-dir "${OUT_DIR}" \
    > "/tmp/INTERV_KW3_${sid}.log" 2>&1
  tail -n 5 "/tmp/INTERV_KW3_${sid}.log" || true
  echo
done

echo "=== (4) Intervention (k_window=3) done ==="
