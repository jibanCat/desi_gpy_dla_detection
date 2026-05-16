# Phase 2: trained scalars — ours vs MATLAB DR16

## Setup
- training subset: 89408 of 89408 train_ind QSOs
- iterations: 200
- optimizer: Adam, lr=0.01
- priors: BOSS DR12Q (τ_0 ~ N(0.00554, 0.00064²); β ~ N(3.182, 0.074²))

| param | ours (trained) | MATLAB (trained) | Δ |
|---|---:|---:|---:|
| c_0 | 0.105925 | 0.145989 | -0.040064 |
| τ_0 | 0.004489 | 0.000119 | +0.004370 |
| β | 3.027178 | 5.153660 | -2.126482 |
