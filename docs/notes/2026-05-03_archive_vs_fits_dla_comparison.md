# FITS vs LoaArchive DLA search comparison

Archive: `/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5`
LOA root: `/nfs/turbo/lsa-cavestru/mfho/DESI/loa/`
Production model: `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5`
num_dla_samples=10000, max_dlas=1, single_absorber_model=False

| TID | z_qso | wave | flux | nv | mask | p_dla FITS | p_dla ARCH | Δp_dla | MAP_z FITS | MAP_z ARCH | Δmp_max |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 39633010785519257 | 2.004 | ✓ | ✓ | ✓ | ✓ | 0.0000 | 0.0000 | 5.97e-15 | nan | nan | 3.13e-13 |
| 39633067924522971 | 2.548 | ✓ | ✓ | ✓ | ✓ | 0.9333 | 0.9333 | 2.57e-06 | 2.4689 | 2.4689 | 2.57e-06 |
| 39628512230899761 | 2.821 | ✓ | ✓ | ✓ | ✓ | 0.0000 | 0.0000 | 8.75e-13 | nan | nan | 3.32e-10 |
