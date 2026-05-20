# Quickstart: Running the GP-DLA Pipeline

This tutorial walks you through running the full GP-DLA detection pipeline
on London mock DESI spectra, and inspecting the output.

> **Data access note**: Real DESI spectra are not publicly released. This tutorial
> uses London mock spectra, which are available to DESI collaboration members at NERSC.
> For public users without NERSC access, SDSS DR12 spectra can be downloaded
> via `data/scripts/download_spectra.sh` and used with `tests/test_selection.py`
> as a reference check (see the note at the end of this guide).

---

## Prerequisites

### 1. Python environment

The pipeline requires `desispec` and `desiutil` for FITS I/O and camera coadding.
At NERSC, source the environment before running:

```bash
source /global/cfs/cdirs/desi/software/desi_environment.sh main
```

On a local machine, the recommended setup is the `desispec` conda environment.

### 2. C extension for Voigt profiles

The Voigt profile computation uses a C helper that requires `libcerf`, compiled
from source. There is no Homebrew or apt package — you must build it yourself.

**Step 1: Compile `libcerf` from source**

```bash
cd $HOME
git clone https://jugit.fz-juelich.de/mlz/libcerf.git
cd libcerf
mkdir build && cd build
cmake ..
make
ctest
make install DESTDIR=~/.local/
```

**Step 2: Compile the C helper**

```bash
cd /path/to/desi_gpy_dla_detection/gpy_dla_detection
cc -fPIC -shared -o _voigt.so ctypes_voigt.c \
    -I$HOME/.local/usr/local/include \
    -L$HOME/.local/usr/local/lib64 -lcerf
```

**Step 3: Add the library path to your shell**

```bash
echo 'export LD_LIBRARY_PATH=$HOME/.local/usr/local/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

If `_voigt.so` is not found at runtime, the pipeline falls back automatically to a
slower pure-Python Voigt implementation with a warning. Results are numerically
identical; only speed differs.

### 3. Download required data files

```bash
cd data/scripts

# Trained null GP model and DR12Q processed files
./download_gp_files.sh

# DR9Q concordance prior catalogs (used for DLA existence priors)
./download_catalogs.sh
```

After downloading, the expected layout is:

```
data/
├── dr12q/processed/
│   ├── learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat
│   ├── catalog.mat
│   ├── dla_samples_a03.mat
│   └── subdla_samples.mat
└── dla_catalogs/dr9q_concordance/processed/
    ├── los_catalog
    └── dla_catalog
```

For the DESI Y3 trained model (`model_epoch_920.h5`) and 100k-sample grids
(`dla_samples_a03_100000.mat`, `subdla_samples_a03_191_200_100000.mat`),
these are stored on NERSC scratch and must be set via `--learned_file` and
`--dla_samples_file` flags directly.

### 4. London mock spectra

London mock DESI spectra (v5.9.5, jura-124) are available at NERSC:

```
/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/
```

The key files are:
- `zcat.fits` — QSO catalog (TARGETID, Z, etc.)
- `spectra-16-{HPX}.fits` — coadded mock spectra per healpix

---

## Step 1: Run DLA detection on London mock spectra

The production DESI Y3 run uses these flags (derived from `slurm/run_reference_mock_desi_y3_learned.sh`):

```bash
MOCKDIR="/path/to/london/mock-0/jura-124"
OUTDIR="output/test_mock_run"
LEARNED_FILE="/path/to/model_epoch_920.h5"
DLA_SAMPLES_FILE="/path/to/dla_samples_a03_100000.mat"
SUB_DLA_SAMPLES_FILE="/path/to/subdla_samples_a03_191_200_100000.mat"

python desi-DLAGP.py \
    --mocks \
    --mockdir "$MOCKDIR" \
    --qsocat "$MOCKDIR/zcat.fits" \
    --learned_file "$LEARNED_FILE" \
    --dla_samples_file "$DLA_SAMPLES_FILE" \
    --sub_dla_samples_file "$SUB_DLA_SAMPLES_FILE" \
    --outdir "$OUTDIR" \
    --prev_tau_0 0.00246 \
    --prev_beta 3.62 \
    --dlambda 0.15 \
    --k 30 \
    --max_dlas 4 \
    --num_forest_lines 3 \
    --filter_low_likelihood 1 \
    --num_dla_samples 100000 \
    --num_subdla_samples 100000 \
    --batch_size 12500 \
    --max_workers 8 \
    --level2_start 0 \
    --level2_end 10
```

**Flag reference** (all values from production Y3 runs):

| Flag | Value | Meaning |
|------|-------|---------|
| `--prev_tau_0` | 0.00246 | Lyman-forest optical depth τ₀ (Turner+2024, DESI Y1) |
| `--prev_beta` | 3.62 | Optical depth power-law index β (Turner+2024) |
| `--dlambda` | 0.15 | GP wavelength grid spacing [Å] |
| `--k` | 30 | Rank of the GP covariance low-rank component |
| `--max_dlas` | 4 | Maximum DLAs per spectrum to model |
| `--num_forest_lines` | 3 | Lyman series lines included in mean-flux model (Lyα, Lyβ, Lyγ) |
| `--filter_low_likelihood` | 1 | Enable stopping criterion: skip DLA(k) if evidence < null |
| `--num_dla_samples` | 100000 | QMC samples for DLA model integration |
| `--num_subdla_samples` | 100000 | QMC samples for sub-DLA model integration |
| `--batch_size` | 12500 | Samples per parallel worker batch |
| `--max_workers` | 8 | Parallel workers per healpix |

For a quick debug run on a laptop (slower but doesn't need the large sample files),
use the SDSS DR12 defaults:

```bash
python desi-DLAGP.py \
    --mocks \
    --mockdir "$MOCKDIR" \
    --qsocat "$MOCKDIR/zcat.fits" \
    --learned_file data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat \
    --dla_samples_file data/dr12q/processed/dla_samples_a03.mat \
    --sub_dla_samples_file data/dr12q/processed/subdla_samples.mat \
    --outdir output/test_debug_run \
    --prev_tau_0 0.00554 \
    --prev_beta 3.182 \
    --dlambda 0.25 \
    --k 20 \
    --max_dlas 3 \
    --level2_start 0 \
    --level2_end 2 \
    --max_workers 4
```

This uses the SDSS DR12Q-trained model with Kamble+2020 optical depth priors,
which will produce different (less accurate for DESI data) posterior values
but is convenient for testing the pipeline machinery.

---

## Step 2: Inspect the HDF5 output

The pipeline writes per-healpix HDF5 files to `$OUTDIR/processed/`:

```python
import h5py
import numpy as np

f = h5py.File("output/test_mock_run/processed/processed-16-705.h5", "r")
print("HDF5 keys:", list(f.keys()))
```

Expected keys:
```
['target_ids', 'z_qsos', 'min_z_dlas', 'max_z_dlas', 'snrs', 'snrs_blue',
 'model_posteriors', 'p_dlas', 'p_no_dlas',
 'MAP_z_dlas', 'MAP_log_nhis', 'z_dla_errs', 'log_nhi_errs',
 'sample_log_likelihoods_dla', 'base_sample_inds',
 'detection_flags', 'log_posteriors_dla', 'log_posteriors_no_dla']
```

### Key arrays

```python
# QSO identifiers and redshifts
target_ids = f["target_ids"][:]   # shape (N,), int64
z_qsos     = f["z_qsos"][:]       # shape (N,), float64

# Detection probabilities
p_dlas    = f["p_dlas"][:]        # shape (N,) — total P(≥1 DLA)
p_no_dlas = f["p_no_dlas"][:]     # shape (N,) — P(Null)

# Model posteriors — layout depends on run mode:
# DLA run (single_absorber_model=False, max_dlas=4):
#   col 0 → P(Null),  col 1 → P(SubDLA),
#   col 2 → P(DLA(1)), col 3 → P(DLA(2)), col 4 → P(DLA(3)), col 5 → P(DLA(4))
# LLS/subDLA run (single_absorber_model=True):
#   col 0 → P(Null),  col 1 → P(absorber)
model_posteriors = f["model_posteriors"][:]

# MAP absorber parameters
MAP_z_dlas   = f["MAP_z_dlas"][:]    # shape (N, max_dlas)
MAP_log_nhis = f["MAP_log_nhis"][:]  # shape (N, max_dlas)

f.close()
```

### Find confident DLA detections

```python
# Spectra with P(DLA) > 0.9
high_confidence = np.where(p_dlas > 0.9)[0]
print(f"N spectra with P(DLA) > 0.9: {len(high_confidence)}")

for i in high_confidence[:5]:
    print(f"  TARGETID={target_ids[i]}, z_QSO={z_qsos[i]:.3f}, "
          f"z_DLA={MAP_z_dlas[i,0]:.3f}, logNHI={MAP_log_nhis[i,0]:.2f}, "
          f"P(DLA)={p_dlas[i]:.3f}")
```

---

## Step 3: Combine per-healpix outputs

After running over multiple healpix tiles, combine results:

```bash
# Combine HDF5 intermediate files (one per healpix → one combined)
python combine_processed_h5.py \
    --processed_dir output/test_mock_run/processed \
    --output_file output/test_mock_run/processed-main-dark.h5 \
    --catalog "$MOCKDIR/zcat.fits" \
    --survey spectra \
    --program 16 \
    --mock

# Combine per-healpix DLA catalog FITS files → single catalog
# (--initial/--end/--step are the healpix range used in the inference run)
python combine_dlakibo.py \
    --release mock \
    --survey spectra \
    --program 16 \
    --outdir output/test_mock_run \
    --initial 0 \
    --end 1150 \
    --step 64
```

---

## Step 4: Quick population statistics check

```python
from astropy.table import Table
import numpy as np
import sys
sys.path.insert(0, "CDDF_analysis")
from cddf_mock import compute_dndx, zbins_from_zmid_uniform

# Load results
dla_cat = Table.read("output/test_mock_run/dlacat-mock.fits")
qso_cat = Table.read(f"{MOCKDIR}/zcat.fits")

# Define redshift bins
z_mid = np.array([2.3, 2.7, 3.1, 3.5])
zbins = zbins_from_zmid_uniform(z_mid)

# Compute dN/dX
out = compute_dndx(
    dla_cat, qso_cat,
    zbins=zbins,
    logNHImin=20.3,
    logNHImax=23.0,
    v_prox_kms=3000.0,
    lambda_obs_min=3700.0,
    blue_limit_mode="max",
)

print("z_mid:  ", out["z_mid"])
print("dN/dX:  ", out["dndx"])
print("N_abs:  ", out["N_abs"])
```

---

## BAL treatment note

By default, production runs **include BAL quasars without pixel masking**:

- `no_bal = False` in `constants.py`: BAL QSOs are **not excluded** from the catalog.
- `BALMASK=false` (the default): BAL pixel masking is **disabled**. BAL columns
  (NCIV_450, VMIN_CIV_450, VMAX_CIV_450) are not read from the catalog.
- The `POTENTIAL_BAL` detection flag is never set unless `--balmask` is passed.

To exclude BAL QSOs entirely, set `no_bal = True` in `constants.py`.
To pixel-mask BAL features, pass `--balmask` and use a catalog containing BAL columns
(e.g., the `*-altbal.fits` catalogs from Paul Martini's team at NERSC).

---

## Note for public users: SDSS spectra

For users without DESI data access, SDSS DR12 spectra can be downloaded publicly:

```bash
cd data/scripts
./download_spectra.sh    # rsync SDSS DR12 BOSS spectra from data.sdss.org
./download_gp_files.sh   # DR12Q trained GP model
./download_catalogs.sh   # DR9Q concordance prior catalogs
```

The test suite in `tests/test_selection.py` contains 100 SDSS spectra
(`spec-{plate}-{mjd}-{fiber}.fits`) with known reference p_DLA values
from Ho, Bird & Garnett (2020). These spectra cover the same DLA population
used to validate the GP model. Note that `test_selection.py` uses a legacy
SDSS API (`process_qso` function); if you encounter an ImportError, the
SDSS spectrum reader may need to be wired up again via `gpy_dla_detection/read_spec.py`.

---

## Common issues

### `ModuleNotFoundError: No module named 'desispec'`

The `desispec` package is required. Source the DESI environment:
```bash
source /global/cfs/cdirs/desi/software/desi_environment.sh main
```

### `RuntimeWarning: Could not load the compiled C Voigt extension (_voigt.so)`

The C Voigt extension is missing or `LD_LIBRARY_PATH` is not set.
Build `_voigt.so` per the instructions above, then:
```bash
export LD_LIBRARY_PATH=$HOME/.local/usr/local/lib64:$LD_LIBRARY_PATH
```
The pipeline continues with a slower pure-Python fallback if the `.so` is missing.

### `LinAlgError: Matrix is not positive definite`

Occasionally raised for very noisy or unusual spectra. The pipeline catches this
and sets `detection_flags = BAD_ZFIT` for that spectrum. Check `fitwarning.DLAFLAG`.

### `model_posteriors` has unexpected shape

Check which run mode was used:
- DLA run (`single_absorber_model=False`, `max_dlas=4`): shape `(N, 6)` = [Null, SubDLA, DLA(1)–DLA(4)]
- Sub-DLA/LLS run (`single_absorber_model=True`): shape `(N, 2)` = [Null, absorber]

When loading results into `DLACatalogue` (`CDDF_analysis/calc_cddf.py`),
set `sub_dla=True` for DLA runs and `sub_dla=False` for sub-DLA/LLS runs.

---

## What to read next

- `docs/architecture.md` — full pipeline code flow and module map
- `docs/tutorial_population_statistics.md` — computing dN/dX, CDDF, Omega_HI
- `README.md` — run modes (DLA/sub-DLA/LLS), flag reference, libcerf compilation
- `gpy_dla_detection/process_helpers.py` — complete HDF5 output schema
