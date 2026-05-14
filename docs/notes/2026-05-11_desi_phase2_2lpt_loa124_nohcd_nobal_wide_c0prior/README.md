# Phase 2 DESI trained GP — model card

This directory contains a GP model trained by `tests/phase2_train_desi.py`
(PR #6 corrected trainer; PCA init + hand-coded gradient via
`gpy_dla_detection/training_v3/objective_vectorized.spectrum_loss_batch`).

> **2026-05-14**: re-emitted to fix the original auto-template's hard-coded
> norm band; the norm band below now reflects what the model was actually
> trained on (from the .h5 manifest or the SLURM log header).

## Files

| File | Purpose |
|---|---|
| `phase2_result.h5` | **Learned model** in DESI schema. Production-loadable by `gpy_dla_detection.null_gp.NullGPMAT(learned_file=...)`. |
| `phase2_result.npz` | Training-history record. Not loaded by the inference pipeline. |
| `README.md` | This file. |

## Training config

| Parameter | Value |
|---|---|
| n_spectra | -1 |
| n_pix (rest) | 5662 |
| rest grid | [850.75, 1700.00] Å, dλ=0.1500 |
| n_iters (Adam) | -1 |
| lr | nan |
| normalize | per-spectrum median in **[1310.00, 1325.00] Å rest** (Garnett+2017 convention) |
| log_c_0 prior σ | (none) |
| SLURM job | 50021381 |

## Endpoint scalars

| Parameter | Value |
|---|---:|
| c_0 | 0.019821 |
| τ_0 | 0.001530 |
| β | 2.5668 |
| log p(D \| Adam endpoint) | 573377984.0000 |

## Provenance

- norm band source: `.h5` manifest
- corr-noise debug arc: see `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md`
- DLA-recovery test on canonical TID: see `docs/notes/2026-05-13_step_c_dla_recovery/findings.md`
