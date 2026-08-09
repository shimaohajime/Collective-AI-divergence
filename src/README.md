# `src/` Code Map

This folder contains the core code for the multi-LLM chaos experiments, analyses, and figure generation.

## Quick Start (Typical Pipeline)

1. Run experiment replicates  
`python3.11 src/chaos_run_v1.py --scenario-id IM-01 --temperature 0.0 --roles-enabled`

2. Compute analysis tables/statistics  
`python3.11 src/chaos_analysis_v2.py`

3. Build cross-scenario landscape summary  
`python3.11 src/chaos_landscape_v1.py`

4. Generate paper figures  
Main: `python3.11 src/plot_figures_v2.py`  
SI: `python3.11 src/plot_si_v1.py`

## File-by-File

### Core execution engine

- `llm_dynamics_v1.py`  
  Core runtime for committee deliberation.  
  Handles provider/model calls, prompt templates, state parsing, ballots, and parallel replicate execution.

- `chaos_run_v1.py`  
  Main experiment runner for one condition (scenario, temperature, roles, etc.).  
  Produces JSONL run artifacts and computes condition-level chaos metrics.

### Data and scenario configuration

- `ai_chaos_scenarios_v1.json`  
  Full scenario packets used by the experiment runner.

- `ai_chaos_experiment_matrix_v1.json`  
  Large matrix spec for broad scheduled runs.

- `im01_variants_v1.json`  
  Semantic-perturbation variants for IM-01.

### Analysis scripts

- `chaos_analysis_v2.py`  
  Statistical post-processing (bootstrap CIs, permutation tests, decomposition summaries, switch/TTM metrics).

- `chaos_landscape_v1.py`  
  Cross-scenario landscape summarization of empirical Lyapunov estimates across key conditions.

- `branching_entropy_v1.py`  
  Branching/local-expansion analysis from existing run files.

- `semantic_perturbation_v1.py`  
  Runs and analyzes semantic-perturbation experiments (typically IM-01 variants).

## Deterministic Hosted Extension

The self-hosted deterministic extension is documented in
[`deterministic_experiments/README.md`](../deterministic_experiments/README.md).

That folder contains:

- a single-process hosted runner for open-weight models
- a full-pipeline determinism certification script
- a small-perturbation runner for scenario-text-only variants

### Figure generation

- `plot_figures_v2.py`  
  Current main-text figure builder (publication-v2 outputs).

- `plot_si_v1.py`  
  SI figure builder.

### Utility

- `test_providers_v1.py`  
  Smoke test for provider connectivity and output-format compatibility.

## Output/Data Expectations

- Raw run outputs are generally written under `data/raw/...`
- Some multi-VM synced artifacts are under `data/vm_sync/...`
- Plot scripts write figure assets to publication figure folders (`publication_v1/`, `publication_v2/`) and/or `paper/figures` depending on workflow.

## Notes

- If you add a new script, update this README and (if relevant) the top-level README runbook section.
- Avoid storing secrets in `src/` code/config files.
