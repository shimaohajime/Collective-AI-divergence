# Data manifest

All paths below are relative to the repository root. The records are JSON Lines
files: one serialized committee run per line.

| Reported component | Raw artifacts |
| --- | --- |
| Uniform HL-01 2x2 baseline | `data/raw/chaos_v1/HL-01__T0.0__N5__roles{False,True}.jsonl` |
| Mixed-model HL-01 2x2 | `data/vm_sync/instance-20260308-011204/chaos_v1_one_grok_t0_matrix/HL-01__T0.0__N5__roles{False,True}__multimodel.jsonl` |
| Uniform IM-01 baseline | `data/raw/chaos_v1/IM-01__T0.0__N5__roles{False,True}.jsonl` |
| Mixed-model IM-01 | `data/raw/chaos_v1/IM-01__T0.0__N5__roles{False,True}__multimodel.jsonl` |
| Full 12-scenario mixed-model matrix | `data/vm_sync/instance-20260308-*/chaos_v1_one_grok_t0_matrix/*__multimodel.jsonl` |
| HL-01 role ablations | `data/raw/chaos_v1/HL-01__T0.0__N5__rolesTrue__ablate-*.jsonl` |
| Chair replication panel | `data/vm_sync/instance-20260307-022228/chaos_v1_chair_rep_t0/*.jsonl` |
| Reduced-memory intervention (`k=3`) | `data/vm_sync/instance-20260307-022228/chaos_v1_intervention_kw3_t0/*.jsonl` |
| Near-memoryless test (`k=1`) | `data/vm_sync/instance-20260307-022228/chaos_v1_falsification_kw1_t0/*.jsonl` |
| Temperature robustness and provider runs | `data/raw/chaos_v1/` |
| API surface-perturbation precursor | `data/raw/semantic_v1/IM-01__T0.0__N5__rolesTrue.jsonl` |
| Exploratory branching analysis | `data/raw/branching_v1/*.json` |
| Hosted deterministic roles benchmark | `deterministic_experiments/fetched_vm1_roles/*__rolesTrue__N5__R20.jsonl` |
| Hosted deterministic no-role benchmark | `deterministic_experiments/fetched_vm{1,2}_noroles/*__rolesFalse__N5__R20.jsonl` |
| Hosted scenario-text variants | `deterministic_experiments/perturbation_variants*.json` |

The figure scripts choose the populated synchronized raw file when a condition
exists in more than one location. This avoids silently replacing a complete
20-replicate file with an earlier partial copy.

Historical files contain successful runs only. See the Supporting Information,
Table S8, for the resulting failure-log limitation.
