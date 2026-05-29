# `training_v3/` — debug rebuild from v1

Started 2026-05-06 after the `_corrected` v2 retrain regression
(see `docs/notes/2026-05-06_corrected_model_validation/REPORT.md`).

## Provenance of files in this directory

The four `.py` files in this directory are **verbatim copies** of the v1
reference at the moment this directory was created:

| this dir | copied from | purpose |
|---|---|---|
| `learn_qso_model.py` | `gpy_dla_detection/learn_qso_model.py` | model class + PCA init + Trainer |
| `objective.py` | `gpy_dla_detection/objective.py` | hand-coded analytic-gradient loss |
| `desi_learn_qsos_model.py` | `desi_learn_qsos_model.py` (repo root; removed 2026-05-20 — see git history) | DESI-specific entry point + prepare_data |
| `__init__.py` | (new, empty) | package marker |

The original v1 files at the source paths above are **frozen** — not
modified by this PR. They serve as the diff reference for any change
landed here. (The root `desi_learn_qsos_model.py` was removed in the
2026-05-20 housekeeping pass; the verbatim copy here is now the sole
in-tree v1 reference.)

The broken v2 modules at `gpy_dla_detection/training/{model_v2,
objective_v2, trainer_v2}.py` are also frozen (kept for diff inspection
during root-cause analysis); this directory will eventually replace
them as the canonical trainer.

## Plan

See `docs/notes/2026-05-06_trainer_debug_plan.md`.

In short:
1. Step A — verbatim retrain. Get v1 (these copies) running on a small
   2lpt subset. Confirm it produces a physically-plausible μ, ω, and
   corr(M·M^T) and detects canonical TID 120046865 with p_DLA ≈ 1.
   Cross-check against MATLAB on the same input batch.
2. Step B — vectorize the per-spectrum loop in `objective.py:44`
   across the batch dimension WITHOUT changing the analytic-gradient
   math. Land behind a numeric-equivalence test against the
   per-spectrum baseline.
3. Step C — production retrain on full 2lpt + LOA, replacing v2.

## Hard rules (see `feedback_training_code_discipline.md` in memory)

1. Initial values for `M`, `μ`, `log_ω` come from PCA / population
   statistics of the trainset. Never `torch.randn` or `torch.zeros`.
   If PCA on the trainset fails, abort training.
2. All gradients are hand-coded analytic, matching the MATLAB
   reference at `https://github.com/jibanCat/gp_dla_detection_dr16q_public`.
   No `loss.backward()` for parameters with hand-derivable gradients.
3. Modifications happen by diffing against the verbatim copies in this
   directory. Don't reimagine the algorithm.
