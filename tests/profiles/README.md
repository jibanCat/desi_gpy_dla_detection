# Test profiles (Paper-1 code review, 2026-08-26)

One environment does not run everything, by design: the HBI stack (jax/numpyro) and the
finder/training stack (desispec, desiutil, healpy, torch) live in different conda envs.
Each profile has an environment of record and a known green / expected-skip state.

| profile | file list | environment | what it covers |
|---|---|---|---|
| `hbi` | `hbi.txt` | `gpdla-hbi` | the Paper-1 HBI / forward-model / calibration / pooling / guard path |
| `finder` | `finder.txt` | `gpdla` | the GP-DLA finder, catalogue post-processing, mock/injection tooling, packaging |
| `training` | `training.txt` | `gpdla` (needs torch) | GP null-model training, parity vs MATLAB, jacobians (heavy/optional) |
| legacy | `legacy.txt` + `tests/parity/`, `tests/profile/`, `tests/matlab/`, `archive/` | none | `test_selection.py`, `test_zestimation.py`: exercise the pre-refactor finder driver API (`from run_bayes_select import process_qso`; `process_qso` has been a nested function of the driver since the FILTER-flag/writer refactor) and have been un-importable since — documented, not run; plus reference implementations and archived one-offs; never collected |

Run one profile:  `tools/paper1/run_tests.sh <profile>`  (picks the environment; `--continue-on-collection-errors` is NOT used --
a collection error in the right environment is a red result).
