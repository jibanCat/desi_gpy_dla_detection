# Phase 2: trained scalars — ours vs MATLAB DR16

## Setup
- training subset: 5000 of 89408 train_ind QSOs
- iterations: 50
- optimizer: Adam, lr=0.01
- priors: BOSS DR12Q (τ_0 ~ N(0.00554, 0.00064²); β ~ N(3.182, 0.074²))

| param | ours (trained) | MATLAB (trained) | Δ |
|---|---:|---:|---:|
| c_0 | 0.108490 | 0.145989 | -0.037499 |
| τ_0 | 0.005111 | 0.000119 | +0.004992 |
| β | 3.013064 | 5.153660 | -2.140596 |
