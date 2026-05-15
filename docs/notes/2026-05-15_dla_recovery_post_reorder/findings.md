# DLA-recovery test: Step C 2lpt + post-reorder models on canonical TID

Date: 2026-05-15. Target: TID 120046865, log_NHI = 21.263.

## Setup

- Target spectrum: `/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits`
- Redshift catalog: `/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits`
- Truth: 2lpt loa-124 mock-0, TID 120046865, log_NHI = 21.263.
- DLAHolder config: max_dlas=4, single_absorber_model=False, filter_low_likelihood=True, num_dla_samples=100000, k_lines=3, num_forest_lines=31, max_noise_variance=9.0.
- v1 reference (`canonical_tid_summary.md`): p_DLA = 0.9897, MAP log NHI = 21.628.
- Inference loader picks up `normalization_{min,max}_lambda` and the rest grid from each `.h5`; the trained-on grid is used directly.

## Per-model results

| model | status | p_DLA | MAP z | MAP log NHI | Δ log NHI | elapsed (s) |
|---|---|---:|---:|---:|---:|---:|
| `v1_production_epoch920` | ok | 0.519535 | 2.774825 | 21.529067 | +0.266 | 22.7 |
| `stepc_2lpt_loa0_wide_m` | ok | 0.703063 | 2.774962 | 21.516874 | +0.254 | 34.3 |
| `stepc_2lpt_loa124_nohcd_nobal_wide_m` | ok | 0.755377 | 2.774962 | 21.516874 | +0.254 | 40.2 |
| `stepc_2lpt_loa124_nohcd_nobal_wide_c0prior` | ok | 0.041597 | nan | nan | +nan | 18.0 |
| `stepc_2lpt_loa0_wide_m_normmask` | ok | 0.724413 | 2.774962 | 21.516874 | +0.254 | 35.8 |
| `stepc_2lpt_loa0_wide_g_normmask` | ok | 0.104131 | nan | nan | +nan | 20.6 |
| `stepc_2lpt_loa124_nohcd_nobal_wide_m_normmask` | ok | 0.761871 | 2.774962 | 21.516874 | +0.254 | 30.7 |
| `stepc_2lpt_loa124_nohcd_nobal_wide_g_normmask` | ok | 0.135090 | nan | nan | +nan | 22.7 |
| `stepc_loa_no_dla_no_bal_wide_m_normmask_3000iter` | ok | 0.503114 | 2.774962 | 21.516874 | +0.254 | 42.0 |
| `stepc_loa_no_hcd_with_bal_wide_m_normmask_3000iter` | ok | 0.215485 | nan | nan | +nan | 17.0 |
| `smoke_postreorder_50iter` | ok | 0.847102 | 2.774321 | 21.628475 | +0.365 | 29.2 |

### model_posteriors (columns: noDLA, subDLA, 1DLA, 2DLA, 3DLA, 4DLA)

| model | noDLA | subDLA | 1DLA | 2DLA | 3DLA | 4DLA |
|---|---:|---:|---:|---:|---:|---:|
| `v1_production_epoch920` | 4.805e-01 | 3.183e-06 | 5.195e-01 | 1.339e-06 | nan | nan |
| `stepc_2lpt_loa0_wide_m` | 2.969e-01 | 4.600e-06 | 7.031e-01 | 1.808e-06 | nan | nan |
| `stepc_2lpt_loa124_nohcd_nobal_wide_m` | 2.446e-01 | 4.128e-06 | 7.554e-01 | 1.737e-06 | nan | nan |
| `stepc_2lpt_loa124_nohcd_nobal_wide_c0prior` | 9.584e-01 | 6.922e-06 | 4.160e-02 | nan | nan | nan |
| `stepc_2lpt_loa0_wide_m_normmask` | 2.756e-01 | 4.402e-06 | 7.244e-01 | 1.958e-06 | nan | nan |
| `stepc_2lpt_loa0_wide_g_normmask` | 8.959e-01 | 8.697e-06 | 1.041e-01 | nan | nan | nan |
| `stepc_2lpt_loa124_nohcd_nobal_wide_m_normmask` | 2.381e-01 | 3.986e-06 | 7.619e-01 | 1.752e-06 | nan | nan |
| `stepc_2lpt_loa124_nohcd_nobal_wide_g_normmask` | 8.649e-01 | 8.597e-06 | 1.351e-01 | nan | nan | nan |
| `stepc_loa_no_dla_no_bal_wide_m_normmask_3000iter` | 4.969e-01 | 3.321e-06 | 5.031e-01 | 1.341e-06 | nan | nan |
| `stepc_loa_no_hcd_with_bal_wide_m_normmask_3000iter` | 7.845e-01 | 4.660e-06 | 2.155e-01 | nan | nan | nan |
| `smoke_postreorder_50iter` | 1.529e-01 | 7.032e-07 | 8.471e-01 | 2.886e-06 | nan | nan |

### Model metadata (read from each `.h5`)

| model | k | rest_min | rest_max | n_pix | dλ | norm_min | norm_max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v1_production_epoch920` | 30 | 850.90 | 1420.60 | 3798 | 0.1500 | None | None |
| `stepc_2lpt_loa0_wide_m` | 30 | 850.75 | 1700.00 | 5662 | 0.1500 | 1425.0 | 1475.0 |
| `stepc_2lpt_loa124_nohcd_nobal_wide_m` | 30 | 850.75 | 1700.00 | 5662 | 0.1500 | 1425.0 | 1475.0 |
| `stepc_2lpt_loa124_nohcd_nobal_wide_c0prior` | 30 | 850.75 | 1700.00 | 5662 | 0.1500 | 1310.0 | 1325.0 |
| `stepc_2lpt_loa0_wide_m_normmask` | 30 | 850.75 | 1700.00 | 5662 | 0.1500 | 1425.0 | 1475.0 |
| `stepc_2lpt_loa0_wide_g_normmask` | 30 | 850.75 | 1700.00 | 5662 | 0.1500 | 1310.0 | 1325.0 |
| `stepc_2lpt_loa124_nohcd_nobal_wide_m_normmask` | 30 | 850.75 | 1700.00 | 5662 | 0.1500 | 1425.0 | 1475.0 |
| `stepc_2lpt_loa124_nohcd_nobal_wide_g_normmask` | 30 | 850.75 | 1700.00 | 5662 | 0.1500 | 1310.0 | 1325.0 |
| `stepc_loa_no_dla_no_bal_wide_m_normmask_3000iter` | 30 | 850.75 | 1700.05 | 5663 | 0.1500 | 1425.0 | 1475.0 |
| `stepc_loa_no_hcd_with_bal_wide_m_normmask_3000iter` | 30 | 850.75 | 1700.05 | 5663 | 0.1500 | 1425.0 | 1475.0 |
| `smoke_postreorder_50iter` | 30 | 850.75 | 1700.00 | 5662 | 0.1500 | 1425.0 | 1475.0 |

## Verdict (corr-noise debug arc impact on inference)

- FAIL (strict): 7 of 7 Step C 2lpt models miss the p_DLA > 0.9 bar: [('stepc_2lpt_loa0_wide_m', '0.7031'), ('stepc_2lpt_loa124_nohcd_nobal_wide_m', '0.7554'), ('stepc_2lpt_loa124_nohcd_nobal_wide_c0prior', '0.0416'), ('stepc_2lpt_loa0_wide_m_normmask', '0.7244'), ('stepc_2lpt_loa0_wide_g_normmask', '0.1041'), ('stepc_2lpt_loa124_nohcd_nobal_wide_m_normmask', '0.7619'), ('stepc_2lpt_loa124_nohcd_nobal_wide_g_normmask', '0.1351')].
- FAIL (operational): 3 of 7 Step C 2lpt models below the p_DLA = 0.5 threshold: [('stepc_2lpt_loa124_nohcd_nobal_wide_c0prior', '0.0416'), ('stepc_2lpt_loa0_wide_g_normmask', '0.1041'), ('stepc_2lpt_loa124_nohcd_nobal_wide_g_normmask', '0.1351')].
- PARTIAL: Step C models with |Δ NHI| > 0.5 dex or NaN: [('stepc_2lpt_loa124_nohcd_nobal_wide_c0prior', nan), ('stepc_2lpt_loa0_wide_g_normmask', nan), ('stepc_2lpt_loa124_nohcd_nobal_wide_g_normmask', nan)].
- INFO: v1 production p_DLA = 0.5195, MAP log NHI = 21.529 (Δ = +0.266 dex). The brief's reference p_DLA = 0.9897 is from a short-retrain v1-trainer replica (`tests/fixtures/.../short_retrain/v1.npz`), not literal `model_epoch_920.h5`; the production model gives a different number here (-0.4702 from the reference). The MAP log NHI bias (+0.27 dex) matches the historical +0.34-0.37 dex v1 bias documented in the τ-EB notes.
- INFO: smoke (50 iter, post-reorder) p_DLA = 0.8471, MAP log NHI = 21.628 (Δ = +0.365 dex). This is undertrained by design — agreement with v1 reference is a happy accident, not pass/fail signal for the corr-noise fix.
- INFO (LOA OOD): LOA-trained `_normmask_3000iter` models on this 2lpt canonical TID — 1/2 cross p_DLA>0.5, 1/2 pass both p_DLA>0.5 AND |Δ NHI|≤0.5dex. This is an OUT-of-distribution test (real-LOA model, mock-2lpt target). For in-distribution recovery on real LOA data, see future LOA-target recovery tests.
-   - `stepc_loa_no_dla_no_bal_wide_m_normmask_3000iter`: p_DLA=0.5031, MAP log NHI=21.517 (Δ=+0.254 dex), posteriors[noDLA,subDLA,1DLA,2DLA,...]=[0.4968830433741989, 3.3209967779238763e-06, 0.5031122941296698, 1.3414994986031574e-06, nan, nan]
-   - `stepc_loa_no_hcd_with_bal_wide_m_normmask_3000iter`: p_DLA=0.2155, MAP log NHI=nan (Δ=+nan dex), posteriors[noDLA,subDLA,1DLA,2DLA,...]=[0.7845102597405776, 4.6595380359583475e-06, 0.215485080721293, nan, nan, nan]
- OVERALL: corr-noise debug arc has PARTIALLY degraded inference. 4 of 7 Step C models pass both criteria; 3 fail at least one.

## Caveats

- The smoke run (`2026-05-13_desi_smoke_normmask`) used only 50 Adam
  iterations; it is undertrained by design (sanity check of the
  post-reorder pipeline end-to-end). Detection at p_DLA > 0.9 is not
  expected and any number here is reported for completeness, not as a
  pass/fail signal for the corr-noise fix.
- The `2026-05-11_*` Step C 2lpt models (kinds `stepc_2lpt`, `stepc_2lpt_c0prior`)
  are PRE-reorder; they share the corr(M·M^T) roughness caveat described in
  `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md` (mean adj-diff
  ≈ 0.004 vs v1 production's 0.0006). The 2026-05-14 `*_normmask` retrains
  (kinds `stepc_2lpt_normmask`, `stepc_loa_normmask`) are POST-reorder
  (dataset.py normalize→mask order + `|med| < 1e-2` threshold, commit aa36205+);
  these supersede the 2026-05-11 batch.
- v1 production was trained on real DESI Y3 LOA spectra (different rest
  grid: [850.90, 1420.60]); inference on a 2lpt mock is still well-
  defined because the loader truncates/extends to the trained grid and
  picks up the normalization band from the `.h5`.
- `model_posteriors` columns: see `CLAUDE.md` §11 — for
  `single_absorber_model=False, max_dlas=4` the column count is 6
  (noDLA, subDLA, 1DLA, 2DLA, 3DLA, 4DLA).

