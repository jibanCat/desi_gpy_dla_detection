# Phase 2 DESI trained GP — model card

> **STATUS: ⚠ SUPERSEDED**. pre-reorder pipeline; **passes DLA recovery** (p_DLA=0.70 on canonical TID). Post-reorder retrain in flight (50212866) — see `docs/notes/2026-05-14_desi_phase2_2lpt_loa0_wide_m_normmask/`.
>
> See `docs/CURRENT_MODELS.md` for the current top pick per use case.

> ⚠ **Pre-reorder caveat**: this model was trained BEFORE the
> 2026-05-13 `dataset.py` reorder (commit aa36205). It carries
> corr(M·M^T) mean-adj-diff ≈ 0.004, ~7× rougher than v1
> production. Inference impact: the `_m` variants (norm
> [1425, 1475]) pass DLA recovery on the canonical strong-DLA
> target with p_DLA > 0.7 and MAP log_NHI within 0.25 dex of
> truth. Post-reorder retrains landing 2026-05-15 AM will
> supersede. See
> `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md`.

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
| n_spectra | 236,755 |
| n_pix (rest) | 5662 |
| rest grid | [850.75, 1700.00] Å, dλ=0.1500 |
| n_iters (Adam) | 1500 |
| lr | 0.005 |
| normalize | per-spectrum median in **[1425.00, 1475.00] Å rest** (MATLAB DR16 convention) |
| log_c_0 prior σ | (none) |
| SLURM job | (not tracked) |

## Endpoint scalars

| Parameter | Value |
|---|---:|
| c_0 | 0.022995 |
| τ_0 | 0.001733 |
| β | 3.0862 |
| log p(D \| Adam endpoint) | 883427904.0000 |

## Provenance

- norm band source: `.h5` manifest
- corr-noise debug arc: see `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md`
- DLA-recovery test on canonical TID: see `docs/notes/2026-05-13_step_c_dla_recovery/findings.md`
