# London production catalog — purity / completeness

- Multi-DLA catalog: `/nfs/turbo/lsa-cavestru/mfho/DESI/DLA/desi-mock-gpdla-20250912-y3-learned-epoch920-filter`
- LLS catalog:       `/nfs/turbo/lsa-cavestru/mfho/DESI/DLA/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172`
- Truth:             `/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat_mask_20.30.fits` (N truth DLAs with NHI≥20.3 = 110641)
- Match metric: |Δz|/(1+z_truth) ≤ 0.01
- p(DLA) cut: 0.5
- N MAP DLAs after cut: raw=173588
- N MAP DLAs after Lyβ veto: 172416  (removed 1172)
- N MAP DLAs after Lyβ + LLS xref: 163760  (removed 9828 total)

## Headline numbers

| stage                       | completeness (all) | purity |
|:----------------------------|:-----------------:|:------:|
| raw catalog                 | 39.8% | 25.4% |
| + Lyβ veto                  | 39.8% | 25.5% |
| + LLS cross-reference       | 39.4% | 26.6% |

## Completeness — raw

| bin | total | matched | rate |
|:-:|:-:|:-:|:-:|
| [20.3, 20.6) | 47830 | 16111 | 0.337 |
| [20.6, 21.0) | 39579 | 16660 | 0.421 |
| [21.0, 21.5) | 19519 | 9301 | 0.477 |
| [21.5, 23.5) | 3713 | 1956 | 0.527 |
| all | 110641 | 44028 | 0.398 |

## Completeness — after Lyβ veto

| bin | total | matched | rate |
|:-:|:-:|:-:|:-:|
| [20.3, 20.6) | 47830 | 16105 | 0.337 |
| [20.6, 21.0) | 39579 | 16657 | 0.421 |
| [21.0, 21.5) | 19519 | 9301 | 0.477 |
| [21.5, 23.5) | 3713 | 1956 | 0.527 |
| all | 110641 | 44019 | 0.398 |

## Completeness — after Lyβ + LLS cross-reference

| bin | total | matched | rate |
|:-:|:-:|:-:|:-:|
| [20.3, 20.6) | 47830 | 15678 | 0.328 |
| [20.6, 21.0) | 39579 | 16647 | 0.421 |
| [21.0, 21.5) | 19519 | 9300 | 0.476 |
| [21.5, 23.5) | 3713 | 1956 | 0.527 |
| all | 110641 | 43581 | 0.394 |
