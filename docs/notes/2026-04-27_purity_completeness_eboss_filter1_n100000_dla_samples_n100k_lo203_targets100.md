# Purity / completeness analysis

Batch: `out/smoke/batch/eboss_filter1_n100000_dla_samples_n100k_lo203_targets100`
Targets file: `out/smoke/targets100.tsv`
Match thresholds: Δz_truth-match = 0.005, Δz_lybeta = 0.005

## Completeness — fraction of truth DLAs matched

| log N_HI bin | total truth DLAs | matched | completeness |
|:------------:|----------------:|---------:|:-------------:|
| [20.3, 20.6) | 63 | 51 | 81.0% |
| [20.6, 21.0) | 48 | 45 | 93.8% |
| [21.0, 21.5) | 28 | 25 | 89.3% |
| [21.5, 23.5) | 25 | 25 | 100.0% |
| **all** | **164** | **146** | **89.0%** |

## ΔlogN_HI on matched DLAs

- N matched = 146
- median ΔlogN_HI = **+0.039**
- σ ΔlogN_HI = 0.115
- median Δz = -0.00009
- σ Δz = 0.00170

| log N_HI bin | N matched | median ΔlogN_HI | σ |
|:------------:|---------:|:---------------:|:---:|
| [20.3, 20.6) | 51 | +0.037 | 0.146 |
| [20.6, 21.0) | 45 | +0.050 | 0.099 |
| [21.0, 21.5) | 25 | +0.042 | 0.097 |
| [21.5, 23.5) | 25 | +0.023 | 0.074 |

## Purity — what fraction of MAP DLAs match a truth DLA?

- N MAP DLAs total = 183
- N matched to truth = 146  (purity = **79.8%**)
- N spurious (no truth match) = 37
  - of which **10 (27.0% of spurious)** match the Lyβ-shifted z of a truth DLA on the same LOS.
  - This suggests that the model misidentifies Lyβ absorption from a real DLA as an additional Lyα DLA.

- Spurious MAP NHI distribution: median=20.80, min=20.30, max=22.48

