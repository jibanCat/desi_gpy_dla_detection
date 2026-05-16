# Phase 2 DESI trained GP — model card

> **STATUS: 🚫 NOT PREFERRED**. outlier on canonical TID; equivalent to `_m` on 10-target sample (7/10 match). Prefer `_m`. See `docs/notes/2026-05-14_c0prior_failure_investigation/`.
>
> See `docs/CURRENT_MODELS.md` for the current top pick per use case.

> ⚠ **Not preferred for production — use `_m` instead.** On a
> 10-target random sample of strong DLAs in 2lpt loa-124 this
> model performed identically to `_m` (7/10 detected, same 3/10
> missed). But on canonical TID 120046865 (truth log_NHI=21.26)
> it gave p_DLA = 0.042 vs `_m`'s 0.755 — an outlier in the gap
> between the two models' detection thresholds. Root cause: the
> log_c_0 prior anchoring failed (c_0 still drifted to 0.020),
> but the slower drift allowed M to balloon — ‖M‖² is 13×
> larger than `_m`'s (21,317 vs 1,648), which widens the
> truncated-marginal QMC prior envelope and drags borderline
> evidences below null. Multi-DLA NaN posteriors are the
> production code's deliberate early-stop (`dla_gp.py:790-810`),
> NOT a Cholesky failure — both models hit the same NaN for
> k≥3 on this target. Full analysis:
> `docs/notes/2026-05-14_c0prior_failure_investigation/findings.md`.

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
| n_spectra | 203,984 |
| n_pix (rest) | 5662 |
| rest grid | [850.75, 1700.00] Å, dλ=0.1500 |
| n_iters (Adam) | 1500 |
| lr | 0.005 |
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
