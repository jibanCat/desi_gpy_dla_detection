# NERSC ↔ GreatLakes path mapping

Goal: run the GP-DLA pipeline on either cluster by swapping a small number of
paths. NERSC remains the production cluster; GreatLakes is for testing,
analysis, and pre-flight validation.

> **Status**: living document. Last updated 2026-04-25.

## 1. Compute account & queues

| Item | NERSC Perlmutter | GreatLakes (UMich) |
|---|---|---|
| Account flag | `-A desi` | `-A cavestru0` |
| Default queue | `-q regular` | `-p standard` (TBD per first submission) |
| Debug queue  | `-q debug` | `-p debug` |
| GPU queue    | `-q gpu`   | `-p gpu --gres=gpu:1` |

## 2. Software stack

| Item | NERSC | GreatLakes |
|---|---|---|
| DESI env load | `source /global/cfs/cdirs/desi/software/desi_environment.sh main` | activate `gpdla` conda env (built per `docs/greatlakes_setup.md`) |
| Python | site-provided | 3.11.15 in `~/.conda/envs/gpdla/` |
| `desispec` | from desi_environment.sh | `pip install desispec` (0.70.0) |
| libcerf | site-provided / built | built from source under `~/.local/usr/local/lib64/` |
| `LD_LIBRARY_PATH` for libcerf | already set | append `$HOME/.local/usr/local/lib64` |
| GPU node visibility | login nodes have GPUs | login nodes do **not** — request via SLURM |

## 3. Filesystem roots

| Resource | NERSC path | GreatLakes path |
|---|---|---|
| Personal scratch | `/pscratch/sd/j/jibancat/` | `/scratch/cavestru_root/cavestru0/mfho/` (TBD) and turbo storage at `/nfs/turbo/lsa-cavestru/mfho/` |
| Code mirror      | `/pscratch/sd/j/jibancat/desi_gpy_dla_detection/` | `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/` |
| Trained models (Y3)   | `<code>/learnlogs/model_epoch_920.h5` | `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5` |
| Trained models (London-mock) | `<code>/learnlogs_london/model_epoch_NNN.h5` | `…/learnlogs_london/model_epoch_199.h5` (latest available on GL) |
| QMC sample grids | `<code>/data/dr12q/processed/*.mat` | same relative path under the GreatLakes code mirror |
| DLA / LOS catalogs | `<code>/data/dla_catalogs/dr9q_concordance/processed/{dla,los}_catalog` | same |

## 4. Mock spectra

| Mock | NERSC path | GreatLakes path | Notes |
|---|---|---|---|
| London v5.9.5 mock-0 contaminated | `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/` | **not mirrored** | London is NERSC-only. |
| London v5.9.5 mock-1 contaminated | `…/london/.../mock-1/jura-124/` | **not mirrored** | NERSC only. |
| Saclay v4.7.5 mock-0 uncontaminated | `…/saclay/qq_desi_y3/v4.7.5/mock-0/jura-0/` | `/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/jura-0/` | |
| Saclay v4.7.5 mock-0 contaminated   | `…/saclay/.../mock-0/juraLy8-124/` | `…/saclay/.../mock-0/juraLy8-124/` | NB: `juraLy8-124`, not `jura-124` (CLAUDE.md gotcha) |
| Saclay v4.7.5 mock-1 contaminated   | `…/saclay/.../mock-1/jura-124/` | not present (only mock-0 is on GL) | |
| 2LPT v2.8.5 mock-0 uncontaminated   | (NERSC location TBD) | `/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-0/` | uncontaminated baseline for Voigt injection-recovery tests |
| 2LPT v2.8.5 mock-0 contaminated     | (NERSC location TBD) | `…/loa-124/` | DLA + metals + BAL contamination |

## 5. Real DESI LOA

| Resource | NERSC | GreatLakes |
|---|---|---|
| LOA QSO/BAL catalog (altbal) | `/global/cfs/projectdirs/desi/users/.../QSO_cat_loa_main_dark_healpix_v3-altbal.fits` (etc.) | `/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v{2,3}-altbal.fits` |
| LOA spectra healpix root | `/global/cfs/cdirs/desi/spectro/redux/<release>/healpix/main/dark/` | private — see `private/loa_paths.md` (gitignored) |

The production script `desi-DLAGP.py:518` hardcodes the NERSC healpix root.
On GreatLakes, do **not** edit the production script — instead use the
GreatLakes runner / wrapper that reads the path from
`private/loa_paths.md` (TODO when first real-data run is needed).

## 6. Inference outputs already on GreatLakes turbo

These are NERSC pscratch outputs that have been copied over for analysis
(CDDF, calibration, purity/completeness):

| Output dir on GreatLakes | NERSC origin | Status |
|---|---|---|
| `/nfs/turbo/lsa-cavestru/mfho/DESI/DLA/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/` | `/pscratch/sd/j/jibancat/desi-loa-gpdla-…-nhi172/` | Real LOA LLS 17.2 — done |
| `…/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi190/` | NERSC equivalent | Real LOA LLS 19.0 — done |
| `…/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/` | NERSC equivalent | London mock-0 LLS 17.2 — done (~578 per-spectrum FITS, source for CDDF analysis) |
| `…/desi-mock-gpdla-2025{0912,0929,1001}-…-filter*-*` | NERSC | older mock filter runs |

## 7. SLURM scripts

| NERSC script (current) | GreatLakes counterpart | Notes |
|---|---|---|
| `slurm/submit_desi_mock.sh` | TODO (e.g. `slurm/greatlakes/submit_desi_mock_gl.sh`) | Strip `desi_environment.sh`; activate `gpdla`; `-A cavestru0`; rebase `MOCKDIR/OUTDIR` paths |
| `slurm/submit_desi_loa.sh` | TODO | Same; plus path override for the hardcoded healpix root in `desi-DLAGP.py` |
| `slurm/lls_runs/run_*.sh` | TODO | LLS variants — each needs the same NERSC→GL substitutions |

## 8. Inferred sample-file naming convention

Once duplicates are renamed cluster-side, both clusters use the same names:

- `pw_samples_a3_<NHIlo>_<NHIhi>_<N>.mat` — Prochaska–Wolfe LLS prior, log NHI ∈ [`<NHIlo>/10`, `<NHIhi>/10`], `N` QMC samples.
- `subdla_samples_a03_<NHIlo>_<NHIhi>_<N>.mat` — sub-DLA prior, same convention.
- `dla_samples_a03_<N>.mat` — Ho+2020 DLA prior across the full DLA range, `N` samples.

## 9. Production parameter parity

The **byte-stable production parameter set** for parity smoke tests (LLS-mode and multi-DLA-mode) is documented in `slurm/lls_runs/README.md` at the NERSC repo and reproduced in this repo's docs once the parallel SLURM scripts land. Values like `PREV_TAU_0=0.00246`, `DLAMBDA=0.15`, `K=30`, `NUM_DLA_SAMPLES=50000` (LLS) or `100000` (multi-DLA) must be matched between clusters whenever the goal is parity validation.
