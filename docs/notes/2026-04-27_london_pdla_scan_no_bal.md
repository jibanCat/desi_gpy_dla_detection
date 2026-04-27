# P_DLA cut sweep (excluding BAL LOS)

- Catalog dir: `/nfs/turbo/lsa-cavestru/mfho/DESI/DLA/desi-mock-gpdla-20250912-y3-learned-epoch920-filter`
- Truth DLA: `/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat_mask_20.30.fits` (NHI ≥ 20.3, 43983 on processed TIDs)
- BAL excluded: True
- Match: |Δz|/(1+z_truth) ≤ 0.01

| P_DLA cut | N MAP | N matched | completeness | strict purity | recovery rate (anything real) |
|:--------:|------:|---------:|:-----------:|:-------------:|:-----------------------------:|
| ≥ 0.500 | 114,280 | 81,294 | 0.927 | 0.711 | 0.932 |
| ≥ 0.900 | 103,542 | 76,656 | 0.874 | 0.740 | 0.931 |
| ≥ 0.990 | 94,458 | 71,792 | 0.818 | 0.760 | 0.930 |
| ≥ 0.999 | 87,044 | 67,396 | 0.768 | 0.774 | 0.929 |
