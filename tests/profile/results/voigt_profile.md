# Voigt-variant profiling

- n_pix = 600 (DESI rest grid, ~600 pix per spectrum is typical)
- num_eval = 1000 per variant
- All times reported in **microseconds per Voigt call**.

## Kernel comparison (num_lines=3)

| variant | kernel | num_lines | output len | mean (μs) | std (μs) | note |
|---|---|---:|---:|---:|---:|---|
| `v1_voigt_fast (C ext)` | `boss-log-r2000` | 3 | 594 | **90.5** | 29.8 |  |
| `v2 (boss-log-r2000)` | `boss-log-r2000` | 3 | 594 | **127.2** | 2.5 |  |
| `v2 (desi-linear-r3000)` | `desi-linear-r3000` | 3 | 594 | **140.3** | 7.3 |  |
| `v2 (desi-linear-r5000)` | `desi-linear-r5000` | 3 | 594 | **140.1** | 2.1 |  |
| `v2 (none)` | `none` | 3 | 600 | **120.5** | 1.9 |  |
| `v2 (none)` | `none` | 1 | 600 | **57.2** | 1.2 |  |
| `v2 (none)` | `none` | 6 | 600 | **215.8** | 39.6 |  |
| `v2 (none)` | `none` | 12 | 600 | **400.1** | 2.7 |  |
| `v2 (none)` | `none` | 31 | 600 | **882.0** | 72.7 |  |
| `v2_torch_cpu (none)` | `none` | 3 | 600 | **235.6** | 7.2 | torch wraps numpy/scipy wofz — bottleneck unchanged |

## Per-spectrum projection (multi-DLA mode, FILTER=1)

Production inference per spectrum: roughly 100,000 QMC samples × 1–4 DLAs × num_lines Voigt evaluations. 
With FILTER=1 the truncated set is ~10–20 % of full QMC. 
Effective Voigt evaluations per spectrum: 10,000 × ~3 × num_lines ≈ 30k (low estimate, 1 DLA + filter) to 1.2M (4 DLAs no filter).

| variant | μs/call | s/spectrum (30k Voigt) | s/spectrum (300k Voigt) |
|---|---:|---:|---:|
| `v2 (none)` | 120.5 | 3.6 | 36.2 |
| `v2 (none)` | 57.2 | 1.7 | 17.2 |
| `v2 (none)` | 215.8 | 6.5 | 64.8 |
| `v2 (none)` | 400.1 | 12.0 | 120.0 |
| `v2 (none)` | 882.0 | 26.5 | 264.6 |
| `v2_torch_cpu (none)` | 235.6 | 7.1 | 70.7 |
