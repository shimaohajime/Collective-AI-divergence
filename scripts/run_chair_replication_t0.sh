#!/bin/bash
# (3) Cross-scenario Chair mechanism replication at T=0
# Scenarios: HL-01, CL-01, SP-03, AI-01
# Conditions per scenario: roles=True baseline, roles=True + ablate Chair

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
OUT_DIR="data/raw/chaos_v1_chair_rep_t0"
SCENARIOS=(HL-01 CL-01 SP-03 AI-01)

mkdir -p /tmp
echo "=== (3) Chair replication start ==="
echo "Output dir: ${OUT_DIR}"
echo "Scenarios: ${SCENARIOS[*]}"
echo

for sid in "${SCENARIOS[@]}"; do
  echo "--- ${sid} baseline roles=True ---"
  python3.11 src/chaos_run_v1.py \
    --scenario-id "${sid}" --temperature 0.0 --roles-enabled \
    --n-replicates "${N_REPS}" --max-workers "${MAX_WORKERS}" \
    --out-dir "${OUT_DIR}" \
    > "/tmp/CHAIR_REP_${sid}_baseline.log" 2>&1
  tail -n 5 "/tmp/CHAIR_REP_${sid}_baseline.log" || true

  echo "--- ${sid} ablate Chair roles=True ---"
  python3.11 src/chaos_run_v1.py \
    --scenario-id "${sid}" --temperature 0.0 --roles-enabled \
    --ablate-role Chair \
    --n-replicates "${N_REPS}" --max-workers "${MAX_WORKERS}" \
    --out-dir "${OUT_DIR}" \
    > "/tmp/CHAIR_REP_${sid}_ablateChair.log" 2>&1
  tail -n 5 "/tmp/CHAIR_REP_${sid}_ablateChair.log" || true
  echo
done

echo "=== (3) Chair replication done ==="
