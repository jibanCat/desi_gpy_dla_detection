# Purity / completeness analysis

Batch: `out/smoke/batch/eboss_filter0_n10000_targets100`
Targets file: `out/smoke/targets100.tsv`
Match thresholds: Δz_truth-match = 0.005, Δz_lybeta = 0.005

## Completeness — fraction of truth DLAs matched

| log N_HI bin | total truth DLAs | matched | completeness |
|:------------:|----------------:|---------:|:-------------:|
| [20.3, 20.6) | 73 | 65 | 89.0% |
| [20.6, 21.0) | 59 | 51 | 86.4% |
| [21.0, 21.5) | 55 | 51 | 92.7% |
| [21.5, 23.5) | 51 | 42 | 82.4% |
| **all** | **238** | **209** | **87.8%** |

## ΔlogN_HI on matched DLAs

- N matched = 209
- median ΔlogN_HI = **+0.029**
- σ ΔlogN_HI = 0.135
- median Δz = -0.00015
- σ Δz = 0.00173

| log N_HI bin | N matched | median ΔlogN_HI | σ |
|:------------:|---------:|:---------------:|:---:|
| [20.3, 20.6) | 65 | +0.008 | 0.162 |
| [20.6, 21.0) | 51 | +0.010 | 0.109 |
| [21.0, 21.5) | 51 | +0.059 | 0.146 |
| [21.5, 23.5) | 42 | +0.036 | 0.096 |

## Purity — what fraction of MAP DLAs match a truth DLA?

- N MAP DLAs total = 434
- N matched to truth = 209  (purity = **48.2%**)
- N spurious (no truth match) = 225
  - of which **49 (21.8% of spurious)** match the Lyβ-shifted z of a truth DLA on the same LOS.
  - This suggests that the model misidentifies Lyβ absorption from a real DLA as an additional Lyα DLA.

- Spurious MAP NHI distribution: median=20.22, min=20.00, max=22.56

