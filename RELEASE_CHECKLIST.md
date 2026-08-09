# Remaining archive gaps

This release contains all manuscript-relevant code and raw data located in the
working project on 2026-08-09. The following artifacts were not present locally
and therefore cannot honestly be represented as included:

- The per-rerun artifact hashes and full records for the reported 100-run
  deterministic-hosted certification.
- Structured historical failure logs for the API runs. The existing JSONL files
  retain successful records only.
- A frozen environment lockfile or container specification for the exact hosted
  environment. The successful hosted records do retain model name, device,
  dtype, seed, PyTorch version, Transformers version, and GPU name; their model
  revision field is null.

Before submission, retrieve these artifacts if available, add a versioned
release archive (for example, Zenodo) to obtain a DOI, and replace the GitHub
URL in `paper/main.tex` with the versioned archive DOI and repository URL.
