# LLS Run Scripts — `slurm/lls_runs/`

All scripts here run GP-DLA in **single-absorber mode** (`SINGLE_ABSORBER_MODEL=1`,
`MAX_DLAS=1`) to build separate LLS, sub-DLA, and DLA absorber catalogs for
calibration analysis. This is distinct from the standard multi-DLA production
runs in `slurm/` which use `MAX_DLAS=3/4`.

---

## Common model settings (all scripts below)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `LEARNED_FILE` | `model_epoch_920.h5` | Y3 GP trained model |
| `PREV_TAU_0` | 0.00246 | Turner+2024 mean-flux prior |
| `PREV_BETA` | 3.62 | Turner+2024 mean-flux prior |
| `DLAMBDA` | 0.15 Å | Pixel spacing |
| `K` | 30 | GP rank |
| `MAX_DLAS` | 1 | Single absorber per spectrum |
| `SINGLE_ABSORBER_MODEL` | 1 | Enables single-absorber mode |
| `NUM_FOREST_LINES` | 3 | Lyα + Lyβ + Lyγ |
| `FILTER_LOW_LIKELIHOOD` | 0 | Keep all detections |
| `BATCH_SIZE` | 6250 | QSOs per SLURM task |
| `MAX_WORKERS` | 8 | CPU workers per task |
| `NUM_DLA_SAMPLES` | 50 000 | QMC integration samples |
| `NUM_SUBDLA_SAMPLES` | 100 000 | Sub-DLA QMC samples |
| `SUB_DLA_SAMPLES_FILE` | `subdla_samples_a03_191_200_100000.mat` | Fixed for all runs |

---

## NHI range variants

Two DLA sample files are used, selecting different NHI floors:

| Suffix | `DLA_SAMPLES_FILE` | NHI range | Targets |
|--------|--------------------|-----------|---------|
| `nhi172` | `pw_samples_a3_172_220_50000.mat` | 17.2–22.0 | LLS + subDLA + DLA |
| `nhi19` | `pw_samples_a3_190_220_50000.mat` | 19.0–22.0 | subDLA + DLA |

Run **both variants** to get full coverage from LLS to DLA.  The nhi172 run
catches all LLS; nhi19 is a separate pass with denser sampling in the
subDLA/DLA regime.

---

## Script inventory

### London mock runs (reference mock, `v5.9.5`)

Data root: `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/`  
Batching: `MAX_START_INDEX=1150`, `STEP=64` → **19 SLURM jobs per script**

| Script | Mock | NHI range | Output directory |
|--------|------|-----------|-----------------|
| `run_reference_mock_nhi172.sh` | mock-0 | 17.2–22.0 | `desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/` |
| `run_reference_mock_nhi19.sh` | mock-0 | 19.0–22.0 | `desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi190/` |
| `run_reference_mock1_nhi172.sh` | mock-1 | 17.2–22.0 | `desi-mock-1-gpdla-20260119-y3-learned-epoch920-lls_run-nhi172/` |

> **Note:** `run_reference_mock1_nhi172.sh` was partially run; it now starts from
> `START_INDEX=702` to avoid re-submitting completed jobs.

### Saclay mock runs (`v4.7.5`)

Data root: `/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/`  
Batching: `MAX_START_INDEX=1127`, `STEP=64` → **18 SLURM jobs per script**

| Script | Mock | NHI range | Output directory |
|--------|------|-----------|-----------------|
| `run_saclay_mock0_nhi172.sh` | mock-0 | 17.2–22.0 | `desi-mock-saclay-0-gpdla-20260415-y3-learned-epoch920-lls_run-nhi172/` |
| `run_saclay_mock0_nhi19.sh` | mock-0 | 19.0–22.0 | `desi-mock-saclay-0-gpdla-20260415-y3-learned-epoch920-lls_run-nhi190/` |
| `run_saclay_mock1_nhi172.sh` | mock-1 | 17.2–22.0 | `desi-mock-saclay-1-gpdla-20260415-y3-learned-epoch920-lls_run-nhi172/` |
| `run_saclay_mock1_nhi19.sh` | mock-1 | 19.0–22.0 | `desi-mock-saclay-1-gpdla-20260415-y3-learned-epoch920-lls_run-nhi190/` |

### Real DESI LOA runs

Data: `/global/cfs/cdirs/desi/science/lya/y3/loa/catalogs/QSO_cat_loa_main_dark_healpix_v2-altbal-20241115.fits`  
Batching: healpix-based, `MAX_HPX_INDEX=16519`, `STEP=1664` → **10 SLURM jobs per script**

| Script | Data | NHI range | Output directory |
|--------|------|-----------|-----------------|
| `run_loa_nhi172.sh` | DESI Y3 LOA | 17.2–22.0 | `desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/` |
| `run_loa_nhi19.sh` | DESI Y3 LOA | 19.0–22.0 | `desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi190/` |

### Debug / development

| Script | Purpose |
|--------|---------|
| `debug_submit_desi_loa_nhi172.sh` | Single-node debug run on LOA data; uses SBATCH debug queue (30 min limit); runs `srun` directly instead of `sbatch` |

---

## Workflow

Each output directory holds per-healpix (mock) or per-batch (LOA) HDF5 files.
After all jobs complete, combine with:

```bash
# Mock runs
python combine_processed_h5.py \
    --processed_dir /pscratch/sd/j/jibancat/<OUTDIR>/ \
    --output_file /pscratch/sd/j/jibancat/<OUTDIR>/combined.h5 \
    --mock

# LOA runs
python combine_processed_h5.py \
    --processed_dir /pscratch/sd/j/jibancat/<OUTDIR>/ \
    --output_file /pscratch/sd/j/jibancat/<OUTDIR>/combined.h5
```

Then pass the combined HDF5 to `CDDF_analysis/cddf_mock.py` for dN/dX and
f(N,z) computation.  See `docs/tutorial_population_statistics.md` for the
full calibration workflow.

---

## Relationship to standard DLA runs

The scripts in the parent `slurm/` directory (e.g. `run_loa_desi_y3_learned.sh`,
`run_saclay_mock_desi_y3_learned.sh`) run in **multi-DLA mode** (`MAX_DLAS=3`,
`SINGLE_ABSORBER_MODEL=0`) to produce the main DLA catalog. Those runs cover
only the DLA NHI range (20.3+).

The scripts here produce **separate single-absorber catalogs** covering the full
NHI range down to LLS (17.2+), used for calibration and the CDDF f(N,z)
analysis in `CDDF_analysis/notebooks/`.
