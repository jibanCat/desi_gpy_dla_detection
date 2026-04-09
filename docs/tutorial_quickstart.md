# Quickstart: Running the GP-DLA Pipeline on London Mock Data

This tutorial walks you through running the full GP-DLA detection pipeline
on a small subset of London mock DESI spectra, and inspecting the output.

**Time estimate**: 15–30 minutes once data files are in place.

---

## Prerequisites

### 1. Python environment

This codebase requires the `desispec` conda environment:

```bash
conda activate desispec
# or use the full path:
# /Users/jibanmac/Documents/conda/desispec/bin/python
```

### 2. C extension for Voigt profiles

The production Voigt profile code requires the `libcerf` C library:

```bash
# macOS (Homebrew)
brew install libcerf

# Linux (Ubuntu/Debian)
sudo apt-get install libcerf-dev

# Then build the extension from the repo root:
python setup.py build_ext --inplace
```

If the C extension is unavailable, the pipeline falls back to a slower
pure-Python Voigt implementation automatically (with a warning).

### 3. Download required data files

The pipeline needs a trained null-GP model, QMC sample grids, and spectra:

```bash
cd data/scripts

# Download trained GP model matrices (.mat files)
bash download_gp_files.sh

# Download London mock spectra + catalogs
bash download_spectra.sh

# Download reference DLA catalogs (for priors)
bash download_catalogs.sh
```

After downloading, you should have:
```
data/
├── dr12q/processed/
│   └── learned_qso_model.mat       # Trained null GP model
├── dr9q/
│   └── dr9q_concordance_*.mat      # Prior catalogs
├── london/
│   ├── dla_cat.fits                # Mock absorber truth catalog
│   ├── zcat.fits                   # Mock QSO catalog
│   └── spectra-16-*/               # Mock coadded spectra (FITS)
└── dla_catalogs/
    └── dla_samples_a03.mat         # QMC sample grid for DLA params
```

---

## Step 1: Run DLA detection on one London mock healpix

The simplest run on London mock data uses a single healpix tile:

```bash
python desi-DLAGP.py \
    --mocks \
    --mockdir data/london \
    --learned_file data/dr12q/processed/learned_qso_model.mat \
    --dla_samples_file data/dla_catalogs/dla_samples_a03.mat \
    --outdir output/test_run \
    --max_dlas 3 \
    --min_lambda 911.75 \
    --max_lambda 1216.75 \
    --dlambda 0.25 \
    --nproc 1
```

For LLS or sub-DLA mode, add `--single_absorber_model` and change the NHI range:

```bash
# LLS run (log NHI 17.2–19)
python desi-DLAGP.py \
    --mocks \
    --mockdir data/london \
    --learned_file data/dr12q/processed/learned_qso_model.mat \
    --dla_samples_file data/dla_catalogs/subdla_samples.mat \
    --single_absorber_model \
    --min_log_nhi 17.2 \
    --max_log_nhi 19.0 \
    --outdir output/test_run_lls \
    --nproc 1
```

---

## Step 2: Inspect the HDF5 output

The pipeline writes per-healpix HDF5 files:

```python
import h5py
import numpy as np

f = h5py.File("output/test_run/processed-mock-dark-705.h5", "r")
print("HDF5 keys:", list(f.keys()))
```

Expected output:
```
HDF5 keys: ['target_ids', 'z_qsos', 'min_z_dlas', 'max_z_dlas', 'snrs',
            'model_posteriors', 'p_dlas', 'p_no_dlas',
            'MAP_z_dlas', 'MAP_log_nhis',
            'sample_log_likelihoods_dla', 'base_sample_inds',
            'detection_flags']
```

### Key arrays explained

```python
# Which spectra were processed
target_ids = f["target_ids"][:]     # shape (N,), int64
z_qsos     = f["z_qsos"][:]        # shape (N,), float64 — QSO redshifts

# Detection probabilities
p_dlas    = f["p_dlas"][:]         # shape (N,) — total P(at least 1 DLA)
p_no_dlas = f["p_no_dlas"][:]      # shape (N,) — P(null)

# Model comparison posteriors
model_posteriors = f["model_posteriors"][:]
# shape (N, 1 + num_subdla + K)
# DLA run: [:, 0]=Null, [:, 1]=SubDLA, [:, 2]=DLA(1), [:, 3]=DLA(2), [:, 4]=DLA(3)
# LLS/subDLA run: [:, 0]=Null, [:, 1]=single absorber

# MAP absorber parameters
MAP_z_dlas    = f["MAP_z_dlas"][:]      # shape (N, K) — best-fit z_DLA per absorber
MAP_log_nhis  = f["MAP_log_nhis"][:]    # shape (N, K) — best-fit log NHI per absorber

f.close()
```

### Find confident DLA detections

```python
# Spectra with P(DLA) > 0.9
high_confidence = np.where(p_dlas > 0.9)[0]
print(f"N spectra with P(DLA) > 0.9: {len(high_confidence)}")

# Their MAP parameters
for i in high_confidence[:5]:
    print(f"  TARGETID={target_ids[i]}, z_QSO={z_qsos[i]:.3f}, "
          f"z_DLA={MAP_z_dlas[i,0]:.3f}, logNHI={MAP_log_nhis[i,0]:.2f}, "
          f"P(DLA)={p_dlas[i]:.3f}")
```

---

## Step 3: Combine per-healpix outputs

If you ran multiple healpix tiles, combine them:

```bash
python combine_processed_h5.py \
    --indir output/test_run \
    --outfile output/processed-mock-dark.h5

python combine_dlakibo.py \
    --indir output/test_run \
    --outfile output/dlacat-mock-dark.fits
```

---

## Step 4: Quick population statistics check

```python
from astropy.table import Table
import numpy as np
from CDDF_analysis.cddf_mock import compute_dndx, zbins_from_zmid_uniform

# Load detection results as catalog
# (combine_dlakibo.py produces a FITS catalog of confident detections)
dla_cat = Table.read("output/dlacat-mock-dark.fits")
qso_cat = Table.read("data/london/zcat.fits")

# Simple z bins
zbins = np.array([2.0, 2.5, 3.0, 3.5, 4.0])

out = compute_dndx(
    dla_cat, qso_cat,
    zbins=zbins,
    logNHImin=20.3,
    logNHImax=23.0,
    v_prox_kms=3000.0,
    lambda_obs_min=3700.0,
    blue_limit_mode="max",
)

print("z_mid:", out["z_mid"])
print("dN/dX:", out["dndx"])
print("N_abs:", out["N_abs"])
```

---

## Step 5: Run the demo script

For a single-spectrum demo without running the full pipeline:

```bash
python examples/demo_desi_spectrum.py
```

This script loads one London mock spectrum, runs GP-DLA inference on it,
and plots the posterior distribution over DLA parameters.

---

## Common Issues

### `ModuleNotFoundError: No module named 'desispec'`

Activate the correct conda environment:
```bash
conda activate desispec
```

### `LinAlgError: Matrix is not positive definite`

This occasionally occurs for very noisy or unusual spectra. The pipeline
catches these errors and writes `detection_flags = BAD_FIT` for that spectrum.
Investigate with `fitwarning.DLAFLAG`.

### `RuntimeWarning: voigt_fast C extension not available`

The C extension for Voigt profiles failed to load. The pipeline will use
the slower Python fallback. Build the extension:
```bash
python setup.py build_ext --inplace
```

### `model_posteriors` has unexpected shape

Check which run mode was used. DLA run (`single_absorber_model=False`) produces
5 columns; sub-DLA/LLS runs produce 2 columns. See `docs/architecture.md`.

---

## What to read next

- `docs/architecture.md` — Full pipeline code flow
- `docs/tutorial_population_statistics.md` — Computing dN/dX, CDDF, Omega_HI
- `notebooks/CDDF_dNdX_all.ipynb` — Mock validation and calibration in practice
- `gpy_dla_detection/process_helpers.py` — Full HDF5 output schema documentation
