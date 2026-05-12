# Training pipeline audit: corrected trainer vs MATLAB DR16 vs v1 Python

**Date**: 2026-05-12
**Branch**: `claude/debug-trainer-from-v1` (PR #6)
**Files audited**:
- MATLAB (gold): `/home/mfho/MATLAB/gp_dla_detection_dr16q_public/{set_parameters,preload_qsos,learn_qso_model,objective,spectrum_loss,effective_optical_depth}.m`
- v1 Python (silver): `/home/mfho/desi_gpy_dla_detection/gpy_dla_detection/{learn_qso_model,objective,effective_optical_depth}.py`, `desi_learn_qsos_model.py`
- New trainer (under audit): `tests/phase2_train_desi.py`, `tests/phase2_train_dr16.py`, `gpy_dla_detection/training/dataset.py`, `gpy_dla_detection/training_v3/objective_vectorized.py`

Legend: ✓ MATCH · ⚠ DIFFERS BY DESIGN · 🔴 BUG · 🟡 NOTE/MINOR

## 1. Data loading + per-spectrum filters

| Item | MATLAB | New trainer | Verdict |
|---|---|---|---|
| z range | `z_qso_cut=2.15` (`set_parameters.m:25`) | `z_min=2.15, z_max=4.25` (`phase2_train_desi.py:507-508`) | ⚠ z_max extended for DESI Y3 |
| ZWARN cut | upstream in `build_catalogs.m` | upstream in preload + TARGETID join (`dataset.py:298`) | ✓ |
| SNR / per-spectrum cut | **none** | `min_snr=0.0` default — reverted from 2.0 in commit `750699e` | ✓ MATCH |
| DLA / BAL exclusion | upstream `train_ind` filter | upstream `nonBAL-nonDLA` catalog | ✓ |

## 2. Per-pixel filters

| Item | MATLAB | New trainer | Verdict |
|---|---|---|---|
| `max_noise_variance` | `9` (`set_parameters.m:38`); applied `learn_qso_model.m:128` | `9.0` (`dataset.py:222`); applied `_mask_high_noise_pixels` | ✓ |
| NaN/inf masking | implicit via NaN propagation | `valid_masks = isfinite(centered) & isfinite(nv) & (nv>0)`, sanitized to 0/1 for vec path | ✓ (different mechanism, identical result) |

## 3. Normalization

| Item | MATLAB | New trainer | Verdict |
|---|---|---|---|
| Band | `[1425, 1475]` (`set_parameters.m:30-31`) | `[1425, 1475]` (`phase2_train_desi.py:538`, post-`750699e`) | ✓ |
| Method | per-spectrum `nanmedian(flux[band ∩ ~mask])` (`preload_qsos.m:41`) | `np.nanmedian(fluxes[:, norm_mask], axis=1)` (`dataset.py:162`) | ✓ |
| Edge cases | only `isnan` rejected; negatives slip through | also rejects `≤0` and `\|·\|<1e-3` (`dataset.py:169-184`) | ⚠ tightening; documented in `3e76056` |
| Divides flux & nv | `flux/m, nv/m²` | `fluxes/safe_med, nv/safe_med²` | ✓ |

## 4. De-forest

| Item | MATLAB | New trainer | Verdict |
|---|---|---|---|
| τ₀, β (DR16) | `0.00554, 3.182` (`learn_qso_model.m:141-142`) | `0.00554, 3.182` (`phase2_train_dr16.py:67`) | ✓ |
| τ₀, β (DESI) | n/a | `0.00246, 3.62` (`phase2_train_desi.py:75-78`) | ⚠ DESI Y3 (Turner+2024) |
| `num_lines` | `31` (`set_parameters.m:85`) | `31` (`phase2_train_dr16.py:66`, `phase2_train_desi.py:66`) | ✓ — **fixes v1 inconsistency** (v1 used 3 lines at training, 31 at inference) |

## 5. Centering (inverse-variance weighted mean)

MATLAB `learn_qso_model.m:175-176`:
```
sum_inverse_variance = nansum(1 ./ rest_noise_variances_exp1pz);
mu = nansum(rest_fluxes_div_exp1pz ./ rest_noise_variances_exp1pz) ./ sum_inverse_variance;
```
(The `_exp1pz` suffix is misleading — these arrays were divided by `lya_absorption` and `lya_absorption²` respectively, i.e. de-forested.)

New `dataset.py:_center_fluxes_inverse_variance` and `phase2_train_dr16.py:168-172`: identical formula on de-forested arrays.

**Verdict**: ✓ MATCH

## 6. PCA initialization

| Item | MATLAB | New trainer | Verdict |
|---|---|---|---|
| Method | row-median NaN fill → `pca(...)` → `M = coeff·sqrt(eigval)` (`learn_qso_model.m:194-219`) | row-median fill → `PCA(k, random_state=0)` → same product (`phase2_train_dr16.py:_pca_init`) | ✓ |
| k (DR16) | `20` | `20` | ✓ |
| k (DESI) | n/a | `30` | ⚠ DESI Y3 |
| Random state | `rng('default')` | `random_state=0` (pinned in `ac7bed8`) | ✓ deterministic |

## 7. Optimizer

| Item | MATLAB | New trainer | Verdict |
|---|---|---|---|
| Optimizer | `minFunc` L-BFGS, MaxIter=2000 | `torch.optim.Adam` | ⚠ design — Adam validated empirically against MATLAB endpoint in DR16 trainer |
| Initial c_0 | `0.1` | `0.1` | ✓ |
| Initial τ_0, β (DR16) | `0.00554, 3.182` | `0.00554, 3.182` | ✓ |
| Initial τ_0, β (DESI) | n/a | `0.00246, 3.62` | ⚠ DESI Y3 |
| Initial log_omega | `log(nanstd(centered))` | `np.log(np.nanstd(centered, axis=0) + 1e-12)` | ✓ |

## 8. Loss function math (most critical) — **all ✓ MATCH except v1 bug fix**

Line-by-line comparison of `objective_vectorized.spectrum_loss_batch` vs MATLAB `spectrum_loss.m`:

- **Lyα optical depth** (`spectrum_loss.m:23-28` ↔ `objective_vectorized.py:101-102`): ✓
- **Lyman series i ≥ 2** (`spectrum_loss.m:30-41` ↔ `objective_vectorized.py:105-111`): ✓
- **Absorption noise** (`spectrum_loss.m:42-46` ↔ `objective_vectorized.py:113-117`): ✓
- **Woodbury / Cholesky** (`spectrum_loss.m:50-67` ↔ `objective_vectorized.py:118-142`): ✓
- **`nlog_p`** with `n*log_2pi` using count of valid pixels: ✓
- **dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta** (`spectrum_loss.m:74-95` ↔ `objective_vectorized.py:149-179`): ✓

🔴 **v1 zqso_1pz bug fixed** (`gpy_dla_detection/objective.py:53` passes `z_qsos[i]` instead of `z_qsos[i] + 1`):
- MATLAB `objective.m:47`: `zqso_1pz = z_qsos(i) + 1;` ✓
- New `phase2_train_dr16.py:259`: `zqso_1pz_t = torch.tensor(np.asarray(z_qsos) + 1.0, ...)` ✓ **fixed**
- New `phase2_train_desi.py:138`: `zqso_1pz_cpu = ((z_qsos + 1.0)...)` ✓ **fixed**

## 9. Priors

| Item | MATLAB DR12Q | v1 Python (Turner+24) | New DR16 | New DESI | Verdict |
|---|---|---|---|---|---|
| τ_0 μ | 0.00554 | 0.00246 | 0.00554 ✓ | 0.00246 | ⚠ DESI uses Turner mean |
| τ_0 σ | 0.00064 | 0.00014 | 0.00064 ✓ | **0.00064** | ⚠ DESI uses BOSS σ — **see open question below** |
| β μ | 3.182 | 3.62 | 3.182 ✓ | 3.62 | ⚠ DESI uses Turner mean |
| β σ | 0.074 | 0.04 | 0.074 ✓ | **0.074** | ⚠ DESI uses BOSS σ — **see open question below** |

🟡 **Open question — DESI prior σ choice**: Currently we use Turner+2024 *means* but BOSS DR12Q *sigmas* (wider). The published Turner sigmas are `(0.00014, 0.04)` — much tighter. v1 production used strict Turner sigmas. The widened σ allows the data to dominate (which the 2lpt drift suggests is happening — see Step C jobs converging to τ_0~0.0006, β~1.3, well below prior μ).

## 10. Output schema

MATLAB saves: `training_release, train_ind, max_noise_variance, minFunc_options, rest_wavelengths, mu, initial_M, initial_log_omega, initial_log_c_0, initial_tau_0, initial_beta, M, log_omega, log_c_0, log_tau_0, log_beta, log_likelihood, minFunc_output`

New `phase2_train_desi.py::_save_h5` saves: trained kernel + `mu` + initial_* + `loss_history, log_*_history` + `max_noise_variance, normalization_min_lambda, normalization_max_lambda, rest_wavelengths` + attrs `n_spectra, n_iters, lr, vectorized, preload_source`.

✓ MATCH (superset).

🟡 **`phase2_train_dr16.py:512-513` writes `normalization_min_lambda=1310/1325`** but the DR16 trainer reads pre-normalized MATLAB data and never calls `_normalize_by_rest_median`. The saved value is technically wrong but harmless. **Action: change to 1425/1475 for accuracy.**

## Remaining issues

### High (consider before long retrain)
**None.** All loss-math gradients match MATLAB exactly; the v1 zqso_1pz bug is correctly fixed.

### Medium (verify with science lead before commit)
1. **DESI prior σ** (§9): currently `(0.00064, 0.074)` (BOSS sigmas with Turner means). v1 production used strict Turner `(0.00014, 0.04)`. Choose deliberately and document.

### Low (housekeeping)
2. **DR16 trainer saves wrong norm band** (§10): change `phase2_train_dr16.py:512-513` from `1310/1325` to `1425/1475`, or omit those keys.
3. **Persist `train_targetids`** in `_save_h5` for reproducibility.

### Confirmations of intentional differences
- ⚠ Adam vs L-BFGS (validated against MATLAB endpoint in DR16 trainer)
- ⚠ k=30 (DESI) vs k=20 (DR16/MATLAB) — DESI Y3 production
- ⚠ z_max=4.25 — DESI Y3
- ⚠ Turner+2024 priors and de-forest values for DESI — DESI Y3
- ⚠ Tighter median rejection (≤0, |·|<1e-3) — defense in depth
- ⚠ `normalization_min/max_lambda` saved in .h5 — improvement over MATLAB

## Recommendation

The corrected trainer pipeline (`tests/phase2_train_desi.py` + `dataset.py` + `objective_vectorized.py`) is **clean and ready for a long production retrain**. The loss-function math is line-for-line equivalent to MATLAB and v1, with the v1 zqso_1pz bug correctly fixed in both new trainers. The data preprocessing matches MATLAB after the recent `750699e` switch to [1425, 1475] normalization band.

The only substantive question worth raising before kicking off the production retrain is the **prior σ choice for τ_0 / β** in the DESI trainer (§9).
