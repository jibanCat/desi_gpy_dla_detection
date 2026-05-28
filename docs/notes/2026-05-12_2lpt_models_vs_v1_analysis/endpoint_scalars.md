# 2lpt trained models vs v1 production — endpoint scalars

v1 production: `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5`  
  Trained on real DESI Y3 LOA spectra (Y3 pipeline; epoch 920).

2lpt models: from PR #6 Step C, trained 2026-05-11 on the v2 wide preload
(rest grid [850.75, 1700] @ dλ=0.15, k=30, 1500 Adam iter, Turner+2024 priors).

| Param | v1 production | 2lpt loa-0 wide | 2lpt loa-124 nohcd-nobal wide |
|---|---:|---:|---:|
| c_0 | 0.173766 | 0.003962 | 0.006006 |
| τ_0 | 0.002099 | 0.000541 | 0.000695 |
| β | 2.407414 | 1.279256 | 1.451302 |
| log_c_0 | -1.750046 | -5.531059 | -5.114985 |
| log_τ_0 | -6.166138 | -7.522962 | -7.272199 |
| log_β | 0.878553 | 0.246279 | 0.372461 |

## Rest grid

| Model | n_pix | min_lambda | max_lambda | dlambda |
|---|---:|---:|---:|---:|
| v1_production | 3798 | 850.90 | 1420.60 | 0.1500 |
| 2lpt_loa0_wide | 5662 | 850.75 | 1700.00 | 0.1500 |
| 2lpt_loa124_nohcd_nobal_wide | 5662 | 850.75 | 1700.00 | 0.1500 |

## Observations

- **c_0 differs by ~30-40×** between v1 and 2lpt models. v1 c_0 ≈ 0.17 (continuum scale around Lyα). 2lpt c_0 ≈ 0.004-0.006. Likely reflects different absolute flux normalization in the lyacolore mocks vs real DESI spectra.
- **τ_0 and β** in 2lpt models are ~3-4× below Turner+2024 priors (0.00246, 3.62), while v1 also lands below prior but less aggressively. The 2lpt mocks may have been constructed with weaker effective optical depth than Turner+2024.
- **Rest grid** is wider for the 2lpt models ([850.75, 1700] vs v1's [~911, ~1500]), reflecting the v2 wide_v2 preload format — captures more red-side continuum.

**Caveat**: scalar differences alone do not say the 2lpt models are wrong. The trained model fit the *2lpt mock data*, which is its own statistical realization. The DLA-detection capability is the actual test, see the inference comparison.
