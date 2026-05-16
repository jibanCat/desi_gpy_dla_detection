# Phase 2: trained scalars — ours vs MATLAB DR16

## Setup
- training subset: 89408 of 89408 train_ind QSOs
- iterations: 200
- optimizer: Adam, lr=0.01
- priors: BOSS DR12Q (τ_0 ~ N(0.00554, 0.00064²); β ~ N(3.182, 0.074²))

| param | ours (trained) | MATLAB (trained) | Δ |
|---|---:|---:|---:|
| c_0 | 0.106198 | 0.145989 | -0.039792 |
| τ_0 | 0.004488 | 0.000119 | +0.004369 |
| β | 3.026339 | 5.153660 | -2.127321 |
