# Purity / completeness analysis

Batch: `out/smoke/batch/eboss_filter1_n100000_targets100`
Targets file: `out/smoke/targets100.tsv`
Match thresholds: Δz_truth-match = 0.005, Δz_lybeta = 0.005

## Completeness — fraction of truth DLAs matched

| log N_HI bin | total truth DLAs | matched | completeness |
|:------------:|----------------:|---------:|:-------------:|
| [20.3, 20.6) | 73 | 57 | 78.1% |
| [20.6, 21.0) | 59 | 51 | 86.4% |
| [21.0, 21.5) | 55 | 50 | 90.9% |
| [21.5, 23.5) | 51 | 48 | 94.1% |
| **all** | **238** | **206** | **86.6%** |

## ΔlogN_HI on matched DLAs

- N matched = 206
- median ΔlogN_HI = **+0.049**
- σ ΔlogN_HI = 0.105
- median Δz = -0.00024
- σ Δz = 0.00160

| log N_HI bin | N matched | median ΔlogN_HI | σ |
|:------------:|---------:|:---------------:|:---:|
| [20.3, 20.6) | 57 | +0.031 | 0.132 |
| [20.6, 21.0) | 51 | +0.055 | 0.108 |
| [21.0, 21.5) | 50 | +0.055 | 0.096 |
| [21.5, 23.5) | 48 | +0.046 | 0.066 |

## Purity — what fraction of MAP DLAs match a truth DLA?

- N MAP DLAs total = 285
- N matched to truth = 206  (purity = **72.3%**)
- N spurious (no truth match) = 79
  - of which **22 (27.8% of spurious)** match the Lyβ-shifted z of a truth DLA on the same LOS.
  - This suggests that the model misidentifies Lyβ absorption from a real DLA as an additional Lyα DLA.

- Spurious MAP NHI distribution: median=20.67, min=20.00, max=22.49

