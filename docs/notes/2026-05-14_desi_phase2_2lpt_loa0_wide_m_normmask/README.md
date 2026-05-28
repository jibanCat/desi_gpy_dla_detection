# Phase 2 DESI trained GP — model card

> **STATUS: ✓ post-reorder pipeline** (`dataset.py` normalize→mask order,
> `|med| < 1e-2` threshold). Training freshly completed.
> See `docs/CURRENT_MODELS.md` for the current top pick per use case;
> update that doc + re-run `examples/reemit_step_c_readmes.py` once
> this model has been DLA-recovery-validated to mark it CURRENT.

This directory contains a GP model trained by `tests/phase2_train_desi.py`
(PR #6 corrected trainer; PCA init + hand-coded gradient via
`gpy_dla_detection/training_v3/objective_vectorized.spectrum_loss_batch`).

## Files

| File | Purpose |
|---|---|
| `phase2_result.h5` | **Learned model** in DESI schema. Production-loadable by `gpy_dla_detection.null_gp.NullGPMAT(learned_file=...)`. |
| `phase2_result.npz` | Training-history record (loss + log_*_history per iter, n_spectra, n_iters, lr). Not loaded by the inference pipeline. |
| `README.md` | This file. |

## Training config

| Parameter | Value | Source |
|---|---|---|
| preload source | `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_wide_v2_1778186324/trainset.h5` | `--preload` |
| n_spectra (after filter) | 236755 | post z/SNR/cap |
| n_pix (rest) | 5662 | preload |
| rest grid | [850.75, 1700.00] Å, dλ=0.1500 | preload |
| k (PCA components) | 30 | `--k` |
| n_iters (Adam) | 1500 | `--n-iters` |
| lr | 0.005 | `--lr` |
| chunk_size (vec) | 5000 | `--chunk-size` |
| device | cuda | `--device` |
| optimizer | `torch.optim.Adam` | trainer |
| loss path | `spectrum_loss_batch` (training_v3, vectorized, hand-coded grad) | trainer |
| τ_0 prior | N(0.00246, 0.00014²) — Turner+2024 | trainer |
| β prior | N(3.62, 0.04²) — Turner+2024 | trainer |
| log_c_0 prior σ | (none) | `--log-c-0-prior-sigma` |
| de-forest | τ_0=0.00246, β=3.62, num_lines=31 | dataset.py |
| normalize | per-spectrum median in [1425.00, 1475.00] Å rest (MATLAB DR16 convention) | `--norm-min-lambda` / `--norm-max-lambda` |
| max_noise_variance | 9.0 | dataset.py |
| mask order | normalize-then-mask (MATLAB-faithful since 2026-05-13) | `dataset.py:356-372` |
| PCA init `random_state` | 0 (pinned, ac7bed8) | `_pca_init` |

## Endpoint scalars

| Parameter | Value |
|---|---:|
| c_0 | 0.025862 |
| τ_0 | 0.004170 |
| β | 3.3113 |
| log p(D \| Adam endpoint) | 567632384.0000 |

## DESI .h5 schema (`phase2_result.h5`)

The DESI inference loader (`null_gp.NullGPMAT.__init__`, `null_gp.py:440-503`)
reads these keys at top level:

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `M` | (n_pix, k) | float64 | GP low-rank basis. `K = M·M^T + diag(omega²)` is the GP prior covariance. |
| `mu` | (n_pix,) | float64 | GP mean function (per-pixel inverse-variance-weighted training mean). |
| `log_omega` | (n_pix,) | float64 | log of per-pixel variance addition. ω² adds to the noise diagonal. |
| `log_c_0` | scalar | float64 | log of mean-flux scale c_0. Reconstructed flux = c_0 × A_lyα(z) × (μ + Mη). |
| `log_tau_0` | scalar | float64 | log Lyα optical-depth normalization. Final τ_eff(z) = τ_0 × (1+z)^β. |
| `log_beta` | scalar | float64 | log Lyα optical-depth power-law index. |
| `rest_wavelengths` | (n_pix,) | float64 | Rest-wavelength grid (Å). |
| `max_noise_variance` | scalar | float64 | Pixel-mask threshold used during preprocessing. |
| `normalization_min_lambda` | scalar | float64 | Per-spectrum normalization band (Å rest), lower edge. |
| `normalization_max_lambda` | scalar | float64 | Per-spectrum normalization band (Å rest), upper edge. |

This schema matches `learnlogs/model_epoch_*.h5` from production runs.
Detection mode is set automatically by `NullGPMAT` based on
`log_tau_0.ndim == 0` (DESI = scalar; SDSS = (1,1)).

## How to load this model for inference

```python
from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.set_parameters import Parameters

# Build Parameters with the inference-time settings YOU want. The
# trained model's `normalization_min/max_lambda` will be auto-applied
# by NullGPMAT's loader (overrides params.normalization in place).
# k and rest range MUST match the trained model — k=30, rest=[851, 1700] Å.
params = Parameters(
    k=30,
    min_lambda=850.75,
    max_lambda=1700.00,
    dlambda=0.1500,
    max_noise_variance=9.0,
    num_lines=3,
    num_forest_lines=31,
    # ... other params per your inference setup
)
prior = ...  # PriorCatalog instance
gp = NullGPMAT(params, prior, learned_file="phase2_result.h5")
```

## Training provenance

Trained on commit `35c3fe9`.
