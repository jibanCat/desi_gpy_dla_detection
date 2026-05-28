# DLA-recovery test: Step C 2lpt models on canonical TID

Date: 2026-05-13. Target: TID 120046865, log_NHI = 21.263.

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
| `v1_production_epoch920` | ok | 0.519535 | 2.774825 | 21.529067 | +0.266 | 27.8 |
| `stepc_2lpt_loa0_wide_m` | ok | 0.703063 | 2.774962 | 21.516874 | +0.254 | 29.7 |
| `stepc_2lpt_loa124_nohcd_nobal_wide_m` | ok | 0.755377 | 2.774962 | 21.516874 | +0.254 | 32.8 |
| `stepc_2lpt_loa124_nohcd_nobal_wide_c0prior` | ok | 0.041597 | nan | nan | +nan | 18.7 |
| `smoke_postreorder_50iter` | ok | 0.847102 | 2.774321 | 21.628475 | +0.365 | 27.3 |

### model_posteriors (columns: noDLA, subDLA, 1DLA, 2DLA, 3DLA, 4DLA)

| model | noDLA | subDLA | 1DLA | 2DLA | 3DLA | 4DLA |
|---|---:|---:|---:|---:|---:|---:|
| `v1_production_epoch920` | 4.805e-01 | 3.183e-06 | 5.195e-01 | 1.329e-06 | nan | nan |
| `stepc_2lpt_loa0_wide_m` | 2.969e-01 | 4.600e-06 | 7.031e-01 | 1.824e-06 | nan | nan |
| `stepc_2lpt_loa124_nohcd_nobal_wide_m` | 2.446e-01 | 4.128e-06 | 7.554e-01 | 1.746e-06 | nan | nan |
| `stepc_2lpt_loa124_nohcd_nobal_wide_c0prior` | 9.584e-01 | 6.922e-06 | 4.160e-02 | nan | nan | nan |
| `smoke_postreorder_50iter` | 1.529e-01 | 7.032e-07 | 8.471e-01 | 2.990e-06 | nan | nan |

### Model metadata (read from each `.h5`)

| model | k | rest_min | rest_max | n_pix | dλ | norm_min | norm_max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v1_production_epoch920` | 30 | 850.90 | 1420.60 | 3798 | 0.1500 | None | None |
| `stepc_2lpt_loa0_wide_m` | 30 | 850.75 | 1700.00 | 5662 | 0.1500 | 1425.0 | 1475.0 |
| `stepc_2lpt_loa124_nohcd_nobal_wide_m` | 30 | 850.75 | 1700.00 | 5662 | 0.1500 | 1425.0 | 1475.0 |
| `stepc_2lpt_loa124_nohcd_nobal_wide_c0prior` | 30 | 850.75 | 1700.00 | 5662 | 0.1500 | 1310.0 | 1325.0 |
| `smoke_postreorder_50iter` | 30 | 850.75 | 1700.00 | 5662 | 0.1500 | 1425.0 | 1475.0 |

## Verdict (corr-noise debug arc impact on inference)

- FAIL (strict): 3 of 3 Step C 2lpt models miss the p_DLA > 0.9 bar: [('stepc_2lpt_loa0_wide_m', '0.7031'), ('stepc_2lpt_loa124_nohcd_nobal_wide_m', '0.7554'), ('stepc_2lpt_loa124_nohcd_nobal_wide_c0prior', '0.0416')].
- FAIL (operational): 1 of 3 Step C 2lpt models below the p_DLA = 0.5 threshold: [('stepc_2lpt_loa124_nohcd_nobal_wide_c0prior', '0.0416')].
- PARTIAL: Step C models with |Δ NHI| > 0.5 dex or NaN: [('stepc_2lpt_loa124_nohcd_nobal_wide_c0prior', nan)].
- INFO: v1 production p_DLA = 0.5195, MAP log NHI = 21.529 (Δ = +0.266 dex). The brief's reference p_DLA = 0.9897 is from a short-retrain v1-trainer replica (`tests/fixtures/.../short_retrain/v1.npz`), not literal `model_epoch_920.h5`; the production model gives a different number here (-0.4702 from the reference). The MAP log NHI bias (+0.27 dex) matches the historical +0.34-0.37 dex v1 bias documented in the τ-EB notes.
- INFO: smoke (50 iter, post-reorder) p_DLA = 0.8471, MAP log NHI = 21.628 (Δ = +0.365 dex). This is undertrained by design — agreement with v1 reference is a happy accident, not pass/fail signal for the corr-noise fix.
- OVERALL: corr-noise debug arc has PARTIALLY degraded inference. 2 of 3 Step C models pass both criteria; 1 fail at least one.

## Caveats

- The smoke run (`2026-05-13_desi_smoke_normmask`) used only 50 Adam
  iterations; it is undertrained by design (sanity check of the
  post-reorder pipeline end-to-end). Detection at p_DLA > 0.9 is not
  expected and any number here is reported for completeness, not as a
  pass/fail signal for the corr-noise fix.
- All Step C 2lpt models are pre-reorder; they share the corr(M·M^T)
  roughness caveat described in
  `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md` (mean adj-diff
  ≈ 0.004 vs v1 production's 0.0006). This is a kernel-level effect; the
  per-spectrum recovery test here probes whether it propagates to p_DLA
  on a strong in-domain DLA.
- v1 production was trained on real DESI Y3 LOA spectra (different rest
  grid: [850.90, 1420.60]); inference on a 2lpt mock is still well-
  defined because the loader truncates/extends to the trained grid and
  picks up the normalization band from the `.h5`.
- `model_posteriors` columns: see `CLAUDE.md` §11 — for
  `single_absorber_model=False, max_dlas=4` the column count is 6
  (noDLA, subDLA, 1DLA, 2DLA, 3DLA, 4DLA).

