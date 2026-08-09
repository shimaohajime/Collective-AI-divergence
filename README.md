# Collective AI divergence

Code, input materials, raw run artifacts, and figure sources for the manuscript
*Collective AI can amplify tiny perturbations into divergent decisions*.

## What is included

The repository is organized to preserve the paths used by the analysis scripts.

- `src/` contains the deployed-API committee runner, statistical analyses, and
  main and supplementary figure scripts.
- `data/raw/` contains the raw successful records for the deployed API
  experiments: baseline and robustness runs (`chaos_v1`), the API
  surface-perturbation precursor (`semantic_v1`), and branching artifacts
  (`branching_v1`).
- `data/vm_sync/` contains the synchronized raw records for the mixed-model
  matrix, Chair-ablation replication, and memory-window experiments.
- `deterministic_experiments/` contains the deterministic hosted runner,
  certification and perturbation scripts, all screened scenario-text variants,
  and the 12-scenario hosted benchmark records.
- `paper/` contains the manuscript, Supporting Information, figures, and
  bibliography sources.
- `scripts/` contains the shell launch scripts for the principal deployed
  cross-scenario, Chair, and memory-window experiments.

`DATA_MANIFEST.md` maps every headline manuscript result to its exact raw
artifact path.

## Reproduce the display items

The included raw artifacts are sufficient to regenerate the principal figures
without API access. From the repository root, create a Python 3.11 environment
and install the analysis dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/plot_figures_v2.py
python src/plot_si_v1.py
python src/plot_deterministic_hosted_figure_v1.py
```

The first two commands write derived figures to `publication_v2/` and
`publication_v1/`, respectively. The final command updates the deterministic
hosted figure and its metric tables in `paper/figures/`. The source figure files
used by the submitted manuscript are already included in `paper/figures/`.

## Run a new experiment

Deployed API experiments require provider credentials. Set them in the
environment or in an untracked `.env` file; never commit credentials.

```bash
export OPENAI_API_KEY='...'
python src/chaos_run_v1.py --scenario-id IM-01 --temperature 0.0 --roles-enabled
```

The deployed mixed-model configuration additionally requires credentials for
Anthropic, Google Gemini, and xAI. The hosted deterministic extension requires
a CUDA-capable machine, PyTorch, Transformers, and access to the specified
open-weight model; see `deterministic_experiments/README.md`.

