# Purity / completeness analysis

Batch: `out/smoke/batch/eboss_filter1_n10000_targets100`
Targets file: `out/smoke/targets100.tsv`
Match thresholds: Δz_truth-match = 0.005, Δz_lybeta = 0.005

## Completeness — fraction of truth DLAs matched

| log N_HI bin | total truth DLAs | matched | completeness |
|:------------:|----------------:|---------:|:-------------:|
| [20.3, 20.6) | 73 | 58 | 79.5% |
| [20.6, 21.0) | 59 | 50 | 84.7% |
| [21.0, 21.5) | 55 | 51 | 92.7% |
| [21.5, 23.5) | 51 | 43 | 84.3% |
| **all** | **238** | **202** | **84.9%** |

## ΔlogN_HI on matched DLAs

- N matched = 202
- median ΔlogN_HI = **+0.029**
- σ ΔlogN_HI = 0.125
- median Δz = -0.00016
- σ Δz = 0.00176

| log N_HI bin | N matched | median ΔlogN_HI | σ |
|:------------:|---------:|:---------------:|:---:|
| [20.3, 20.6) | 58 | +0.022 | 0.135 |
| [20.6, 21.0) | 50 | +0.022 | 0.112 |
| [21.0, 21.5) | 51 | +0.032 | 0.142 |
| [21.5, 23.5) | 43 | +0.030 | 0.098 |

## Purity — what fraction of MAP DLAs match a truth DLA?

- N MAP DLAs total = 292
- N matched to truth = 202  (purity = **69.2%**)
- N spurious (no truth match) = 90
  - of which **17 (18.9% of spurious)** match the Lyβ-shifted z of a truth DLA on the same LOS.
  - This suggests that the model misidentifies Lyβ absorption from a real DLA as an additional Lyα DLA.

- Spurious MAP NHI distribution: median=20.74, min=20.01, max=22.56

