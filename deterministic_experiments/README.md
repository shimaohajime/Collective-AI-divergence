# `deterministic_experiments/` Code Map

This folder contains the self-hosted deterministic extension for the paper.

## Main Scripts

- `hosted_local_runner_v1.py`
  Single-process Hugging Face backend plus hosted committee runner.
  Reuses the prompt suite and parsing logic from
  [`src/llm_dynamics_v1.py`](../src/llm_dynamics_v1.py).

- `certify_hosted_determinism_v1.py`
  Runs the exact same full 20-round committee pipeline multiple times and checks whether the full serialized artifact hash is identical across reruns.

- `run_small_perturbations_v1.py`
  Runs deterministic committee deliberation across small scenario-text perturbations and computes:
  - perturbation divergence slope `lambda_pert`
  - decision fragility
  - median branching round
  - reconvergence index

  It now also supports baseline protocol variants via `--protocol-variant`:
  - `ws` for the full windowed-summary deliberation
  - `no_feedback` for iterative rounds without transcript feedback
  - `one_shot` for immediate private ballots without iterative dialogue

## Variant Data

- `perturbation_variants_v1.json`
  Starter scenario-text-only perturbations for:
  - `IM-01`
  - `HL-01`
  - `AI-01`
  - `CL-01`

Allowed perturbation families in this file:
- `surface_rephrasing`
- `formatting_changes`

## Example Commands

Run the full-pipeline determinism certification:

```bash
source ~/deterministic-env/bin/activate
python deterministic_experiments/certify_hosted_determinism_v1.py \
  --scenario-id IM-01 \
  --roles-enabled \
  --n-runs 100 \
  --model-name Qwen/Qwen2.5-7B-Instruct
```

Run the first perturbation experiment:

```bash
source ~/deterministic-env/bin/activate
python deterministic_experiments/run_small_perturbations_v1.py \
  --scenario-id IM-01 \
  --roles-enabled \
  --model-name Qwen/Qwen2.5-7B-Instruct
```

Run the no-feedback baseline on the same perturbation set:

```bash
source ~/deterministic-env/bin/activate
python deterministic_experiments/run_small_perturbations_v1.py \
  --scenario-id IM-01 \
  --roles-enabled \
  --protocol-variant no_feedback \
  --model-name Qwen/Qwen2.5-7B-Instruct
```
