# `slurm/configs/` — modular production launch

Each `.env` file is a self-contained recipe for one (data, mode) pair. The
outer driver is `slurm/launch.sh`; the inner sbatch scripts
(`slurm/submit_desi_loa.sh`, `slurm/submit_desi_mock.sh`) are unchanged.

## Quick reference

```bash
# Dry-run (prints sbatch commands, does NOT submit, does NOT mkdir).
# Always do this first.
bash slurm/launch.sh slurm/configs/london0_y3.env --dry-run --no-sleep | head

# Submit the whole production
bash slurm/launch.sh slurm/configs/london0_y3.env

# Restart partway through
bash slurm/launch.sh slurm/configs/london0_y3.env --start 704

# One-off OUTDIR for a debug pass
bash slurm/launch.sh slurm/configs/loa_y3.env \
    --outdir /pscratch/sd/j/jibancat/loa-debug-$(date +%s)/ \
    --start 0 --end 1664 --no-sleep
```

The launcher will **refuse** to submit if the resolved `OUTDIR` falls outside
`/pscratch/sd/j/jibancat/`, `/global/homes/j/jibancat/`, or
`/global/cfs/cdirs/desicollab/users/jibancat/`. See
[`docs/nersc_write_permissions.md`](../../docs/nersc_write_permissions.md).
On `--dry-run` the launcher also skips `mkdir` so it leaves no trace.

## Flavour matrix

Each flavour has three modes:
- `*_y3.env`         — multi-DLA (MAX_DLAS=3, SINGLE_ABSORBER_MODEL=0, FILTER_LOW_LIKELIHOOD=1)
- `*_y3_lls172.env`  — LLS single-absorber, NHI floor 17.2 (`pw_samples_a3_172_220_50000.mat`)
- `*_y3_lls190.env`  — LLS single-absorber, NHI floor 19.0 (`pw_samples_a3_190_220_50000.mat`)

| Flavour    | Data                                                                                                       | Outer loop |
|------------|------------------------------------------------------------------------------------------------------------|-----------:|
| `loa`      | `/global/cfs/cdirs/desi/science/lya/y3/loa/.../QSO_cat_loa_main_dark_healpix_v2-altbal-20241115.fits`      | 0..16519 step 1664 |
| `london0`  | `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/`                  | 0..1150 step 64    |
| `saclay0`  | `/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/`       | 0..1127 step 64    |
| `2lpt0`    | `/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/`   | 0..1150 step 64    |

All twelve `.env` files share `_base.env`, which encodes the Y3 production
defaults (model_epoch_920, dlambda=0.15, k=30, num_forest_lines=3, Turner+2024
tau/beta, BAL included). Override any of those by setting the var in the
flavour config before sourcing, or in the environment when calling `launch.sh`.

## After submission — combining outputs

Each sbatch job writes per-healpix HDF5 chunks into `OUTDIR/`. After all jobs
finish:

```bash
# Mock
python combine_processed_h5.py \
    --processed_dir "$OUTDIR" \
    --output_file   "$OUTDIR/combined.h5" \
    --mock

# Real LOA
python combine_processed_h5.py \
    --processed_dir "$OUTDIR" \
    --output_file   "$OUTDIR/combined.h5"
```

Then run purity/completeness with `examples/analyze_production_catalog.py`
(see CLAUDE.md §3 and `examples/README.md`).

## Migration from the old scripts

The legacy outer drivers (`run_loa_desi_y3_learned.sh`,
`run_referece_mock_desi_y3_learned.sh`, `run_saclay_mock_desi_y3_learned.sh`,
all of `slurm/lls_runs/run_*.sh`) hardcoded the same paths and looped the
same way. Each maps to a single `launch.sh <config>` invocation:

| Old script | Replacement |
|------------|-------------|
| `slurm/run_loa_desi_y3_learned.sh`            | `bash slurm/launch.sh slurm/configs/loa_y3.env`            |
| `slurm/run_referece_mock_desi_y3_learned.sh`  | `bash slurm/launch.sh slurm/configs/london0_y3.env`        |
| `slurm/run_saclay_mock_desi_y3_learned.sh`    | `bash slurm/launch.sh slurm/configs/saclay0_y3.env`        |
| `slurm/lls_runs/run_reference_mock_nhi172.sh` | `bash slurm/launch.sh slurm/configs/london0_y3_lls172.env` |
| `slurm/lls_runs/run_reference_mock_nhi19.sh`  | `bash slurm/launch.sh slurm/configs/london0_y3_lls190.env` |
| `slurm/lls_runs/run_saclay_mock0_nhi172.sh`   | `bash slurm/launch.sh slurm/configs/saclay0_y3_lls172.env` |
| `slurm/lls_runs/run_saclay_mock0_nhi19.sh`    | `bash slurm/launch.sh slurm/configs/saclay0_y3_lls190.env` |
| `slurm/lls_runs/run_loa_nhi172.sh`            | `bash slurm/launch.sh slurm/configs/loa_y3_lls172.env`     |
| `slurm/lls_runs/run_loa_nhi19.sh`             | `bash slurm/launch.sh slurm/configs/loa_y3_lls190.env`     |
| *(new — 2LPT not in old tree)*                | `bash slurm/launch.sh slurm/configs/2lpt0_y3.env`          |

The old scripts still work; the launcher just removes the copy-paste.
