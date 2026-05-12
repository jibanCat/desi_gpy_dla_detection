# GP training overview

> Last updated 2026-05-11 (mid-PR #6). This file maps the multiple
> generations of GP training code that have accumulated in this repo.
> Read this before adding a new trainer or refactoring an existing one.
>
> See also: **[`test_results_overview.md`](test_results_overview.md)** —
> running record of correctness tests + perf benchmarks (per-iter rates,
> speedup ratios, SLURM job ledger, storage paths). Update that file
> when a new test lands or a benchmark changes.

## TL;DR — which trainer should I use?

- **For SDSS DR16 retrain (validation rig)**: `tests/phase2_train_dr16.py`
  (PCA init + hand-coded gradient + Adam, vectorized loss path). This
  is the **corrected** trainer from PR #6.
- **For DESI v2 preload retrain (production target)**: `train_gp.py`
  → `gpy_dla_detection/training/trainer_v2.py`. **WARNING**: this is
  the broken trainer (randn-init M + autograd loss); see
  `docs/notes/2026-05-06_trained_corrected_v2_models_validation/REPORT.md`
  for the regression. **Step C of PR #6 will replace this with a DESI
  port of `phase2_train_dr16.py` (`tests/phase2_train_desi.py`).**
- **DO NOT** invoke `desi_learn_qsos_model.py` for new work — legacy
  v1 entry point, kept only as a frozen reference.

## Glossary of "v" generations

| label | what | status |
|---|---|---|
| v1 | Original MATLAB-faithful reference: `gpy_dla_detection/learn_qso_model.py` + `gpy_dla_detection/objective.py`. Per-spectrum loop, hand-coded gradients, MATLAB-equivalent math. | Frozen reference. Has the documented `objective.py:53` `zqso_1pz` bug. |
| v2 | First refactor: `gpy_dla_detection/training/{model_v2,objective_v2,trainer_v2,dataset}.py`. Vectorized NLL via einsum, Adam loop, GPU-capable, autograd backward, randn-init M. | **Broken** — the autograd + randn-init combination produced the 2026-05-06 _corrected retrains regression (all 6 retrains miss canonical TID). |
| v3 | PR #6 working area: `gpy_dla_detection/training_v3/`. Verbatim copy of v1 (for diff hygiene during Step A) plus `objective_vectorized.py` (Step B vectorization with hand-coded gradients). | Active — used by the corrected trainer. |
| v3.5 | `gpy_dla_detection/training_v3_5/`. Strict-`dlog_β` variant of v3 used only for Step A.2 comparative testing. | Comparative only; not production. |

## 1. Production trainer entry points

| Path | Purpose | Backend | Status |
|---|---|---|---|
| `train_gp.py` | Streamlined v2 CLI; loads `gp_interp_trainset.h5`, calls `trainer_v2.train_v2`. The current production GP trainer for DESI on NERSC/GreatLakes. | `training.trainer_v2` (broken — autograd + randn-init) | **Broken — do not run for production until rebuilt.** |
| `desi_learn_qsos_model.py` | Legacy v1 entry point. Wraps `learn_qso_model.Trainer` (per-spectrum loop). | v1 | Deprecated. Kept as reference; will be removed end-of-PR. |
| `tests/phase2_train_dr16.py` | The **corrected** trainer (PR #6 deliverable). PCA init + hand-coded gradient via `training_v3/objective_vectorized.py` + Adam. CPU only. Hardcoded to MATLAB DR16 paths. | training_v3 (vectorized, hand-coded grad) | **Production-grade math.** Used for the MATLAB DR16 cross-validation. To be ported to DESI as `phase2_train_desi.py` (Step C). |

## 2. Trainer libraries

### v1 reference (frozen)

| Path | Purpose |
|---|---|
| `gpy_dla_detection/learn_qso_model.py` | `GaussianProcessModel` + `Trainer`. Per-spectrum loop, MATLAB-equivalent. |
| `gpy_dla_detection/objective.py` | Per-spectrum `spectrum_loss` + hand-coded gradients. Contains the documented `:53` `zqso_1pz` bug; bypassed by the new trainers. |

### v2 (broken — to be replaced)

| Path | Purpose | Status |
|---|---|---|
| `gpy_dla_detection/training/__init__.py` | Subpackage init. | OK |
| `gpy_dla_detection/training/dataset.py` | Loads `gp_interp_trainset.h5`, applies mask + de-forest + center. | OK — used by Step C trainer |
| `gpy_dla_detection/training/model_v2.py` | `GPModelV2` parameter container. Initializes `M` with `torch.randn` instead of PCA. | **Broken** (randn-init M) |
| `gpy_dla_detection/training/objective_v2.py` | Vectorized NLL via einsum + autograd backward. | **Broken** (autograd, not analytic gradient) |
| `gpy_dla_detection/training/trainer_v2.py` | Adam loop, GPU-capable, checkpoint+resume, HDF5 saver. | **Broken** (uses `model_v2` + `objective_v2`) |

### v3 (Step B working area — to become canonical)

| Path | Purpose |
|---|---|
| `gpy_dla_detection/training_v3/learn_qso_model.py` | Verbatim copy of v1 (Step A diff hygiene). |
| `gpy_dla_detection/training_v3/objective.py` | Verbatim copy of v1. |
| `gpy_dla_detection/training_v3/objective_vectorized.py` | **Step B**: batched `spectrum_loss_batch` with hand-coded gradients (numerically equiv to v1 per-spectrum to ~1e-10). This is the loss path used by `phase2_train_dr16.py`. |

### v3.5 (Step A.2 comparative, not production)

| Path | Purpose |
|---|---|
| `gpy_dla_detection/training_v3_5/objective.py` | v3 + strict `dlog_β` variant. Used only for `tests/test_v3_5_vs_matlab.py` lane comparisons. **Cleanup candidate end-of-PR.** |

## 3. Tests / scaffold trainers

### Active in PR #6

| Path | Type | Purpose |
|---|---|---|
| `tests/phase2_train_dr16.py` | Full retrain | A.5 / Step B end-to-end MATLAB DR16 validation. Production-grade math. |
| `tests/phase2_npz_to_h5.py` | Converter | npz → DESI h5 schema for the inference loader. |
| `tests/short_retrain_2lpt.py` | Small retrain | Step A.3 v1 ≈ v3.5 parity on 1300-spectrum 2lpt fixture. |

### Parity / regression tests (keep)

| Path | What it guards |
|---|---|
| `tests/test_v3_train_step_parity.py` | per-spectrum ≡ batched vectorized to ~1e-11 (Step B) |
| `tests/test_v3_objective_vectorized_parity.py` | spectrum_loss_batch ≡ per-spectrum loop |
| `tests/test_v3_objective_vectorized_jacobian.py` | Numeric Jacobian on spectrum_loss_batch |
| `tests/test_objective_v2_parity.py` | vectorized_nll ≡ legacy per-spectrum loop (v2 layer) |
| `tests/test_objective_v2_jitter.py` | Jitter keyword stability for ill-conditioned matrices |
| `tests/test_normalize_by_rest_median.py` | Preprocessing pipeline regression (NaN/centering) |
| `tests/test_train_gp_end_to_end.py` | v2 stack on synthetic preload schema |
| `tests/test_trainer_v2_smoke.py` | v2 stack wiring |
| `tests/test_v1_spectrum_loss_jacobian.py` | A.1 numeric Jacobian on v1 |
| `tests/test_v1_matches_matlab.py` | A.2.a v1 ≡ MATLAB on spectrum_loss |

### Step A debug artifacts (cleanup candidates after PR #6)

| Path | Use | Cleanup? |
|---|---|---|
| `tests/test_v3_5_vs_matlab.py` | A.2.b v3.5 ≡ MATLAB | Maybe — superseded after Step C |
| `tests/test_v3_5_spectrum_loss_jacobian.py` | A.1 Jacobian on v3.5 | Maybe — depends on whether v3.5 stays |
| `tests/plot_corr_dr16_comparison.py` | A.5 corr-matrix diagnostic | Keep as reference |
| `tests/plot_phase2_paths_comparison.py` | Phase 2 path diagnostics | Keep — documents PR #6 verdict |
| `tests/plot_vec_vs_perspec_kernels.py` | Kernel overlay (PR #6) | Keep — comparison artifact |
| `tests/plot_vec_vs_perspec_corr.py` | Correlation matrix overlay (PR #6) | Keep |
| `tests/profile/profile_training.py` | Layer 3 benchmarking | Cleanup candidate |

### Legacy (keep only if v1 stays)

`tests/test_learn_qso.py`, `tests/test_learn_qso_100spec.py`,
`tests/test_training_loss.py` — all exercise the v1 `Trainer`. Retain
while `learn_qso_model.py` is the frozen reference.

## 4. Preload pipeline (`preload_spectra/`)

Training input prep — separate from training itself, but related.

| Path | Purpose | Status |
|---|---|---|
| `preload_spectra/preload_2lpt_simple.py` | Streamlined 2lpt mock preload (zcat + spectra → trainset.h5). HCD/BAL filter via truth catalog. | Preferred |
| `preload_spectra/preload_loa_real.py` | Streamlined LOA real preload. z-range, ZWARN, BAL filters. | Preferred |
| `preload_spectra/prepare_trainset.py` | Legacy: preloaded HDF5 + FITS catalog → gp_interp_trainset.h5. | OK |
| `preload_spectra/desi-preload.py` | Legacy heavy preload (uses DLAHolder + run_bayes_select). | OK but heavyweight |
| `preload_spectra/trainset_catalog.py`, `qsopreload.py`, `_dataset_readme.py`, `recover_dataset_readme.py` | Helpers / readme writers. | OK |

## 5. SLURM training scripts (`slurm/greatlakes/`)

| Path | Purpose | Backend |
|---|---|---|
| `slurm/greatlakes/train_gp_v2_2lpt.sh` | Production v2 train on 2lpt (uses train_gp.py — broken trainer) | trainer_v2 (CUDA) |
| `slurm/greatlakes/preload_train_2lpt.sh` | Coupled preload + train (broken trainer) | trainer_v2 (CUDA) |
| `slurm/greatlakes/train_only_gpu.sh` | Train-only, assumes preload exists | trainer_v2 (CUDA) |
| `slurm/greatlakes/smoke_e2e_train_loa.sh` | E2E smoke on LOA real subset | trainer_v2 (CUDA) |
| `slurm/greatlakes/preload_2lpt_only.sh`, `preload_mock_only.sh` | Preload-only stages | n/a |
| `slurm/greatlakes/profile_training_gpu.sh` | Profiling | n/a |
| `slurm/greatlakes/phase2_dr16_retrain.sh` | **PR #6 validation: DR16 retrain** | phase2_train_dr16.py (CPU) |
| `slurm/greatlakes/phase2_dr16_vectorized_smoke.sh` | DR16 vec smoke | phase2_train_dr16.py (CPU) |

The SLURM scripts that target `train_gp.py` will need to be updated
(or replaced) once Step C lands — the `trainer_v2` path they use
produces the broken model.

## 6. Cleanup candidates at end of PR #6

(Subject to user confirmation before removal.)

**Probably remove:**
- All `.ipynb_checkpoints/` directories under `gpy_dla_detection/`,
  `tests/`, `docs/notes/` (Jupyter artifacts).
- `tests/profile/profile_training.py` (Layer 3 benchmarking artifact).

**Maybe remove (depends on whether v3.5 stays):**
- `gpy_dla_detection/training_v3_5/` (Step A.2 comparative variant).
- `tests/test_v3_5_vs_matlab.py`, `tests/test_v3_5_spectrum_loss_jacobian.py`.

**Replace then remove (after Step C lands):**
- `gpy_dla_detection/training/{model_v2,objective_v2,trainer_v2}.py`
  — the broken trainer. Once the corrected `phase2_train_desi.py` is
  in production, the v2 modules can be deprecated.
- `desi_learn_qsos_model.py` (legacy entry point).

**Keep (do not delete):**
- `gpy_dla_detection/learn_qso_model.py`, `objective.py` — frozen v1 reference.
- `gpy_dla_detection/training/dataset.py` — used by the new DESI port.
- All parity / regression tests in §3 marked "keep".
- `tests/phase2_train_dr16.py`, `tests/phase2_npz_to_h5.py` — PR #6
  deliverables.

## Status of PR #6 (this file written mid-PR)

| Step | What | Status |
|---|---|---|
| A | DR16 PCA-init verify | done |
| A.5 | Phase 2 DR16 retrain (validation rig) | done |
| B | Vectorize loss path | done — vec ≡ per-spec at corr 0.95% Frobenius, p_DLA Δ ~3e-3 |
| **C** | **DESI 1500-ep retrain (production target)** | **not started — needs phase2_train_desi.py port** |
| D | Production decision | blocked on C |
| Cleanup | Remove litter per §6 | end of PR |
