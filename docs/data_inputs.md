# Data Inputs Reference

This document describes every external file the GP-DLA pipeline requires,
their formats, required keys, and how to obtain them.

---

## 1. Trained null GP model

**Default filenames:**
- SDSS DR12Q (legacy, public): `data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat`
- DESI Y3 (production): `learnlogs/model_epoch_920.h5`

**Format:** HDF5 (both `.mat` MATLAB v7.3 and `.h5` are HDF5 internally).

**How to get it:**
- SDSS DR12Q model: `cd data/scripts && ./download_gp_files.sh` (clones `github.com/jibanCat/gp_dr12_trained`)
- DESI Y3 model: produced by `desi_learn_qsos_model.py` on NERSC; stored at `/pscratch/sd/j/jibancat/desi_gpy_dla_detection/learnlogs/`

**Required HDF5 keys** (loaded in `gpy_dla_detection/null_gp.py:NullGPMAT.__init__`):

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `rest_wavelengths` | `(W,)` | float64 | GP model rest-frame wavelength grid [Å] |
| `mu` | `(W,)` | float64 | GP prior mean flux at each wavelength |
| `M` | `(W, K)` | float64 | Rank-K factor loading matrix (low-rank component of covariance) |
| `log_omega` | `(W,)` | float64 | Log diagonal noise amplitude per wavelength |
| `log_c_0` | scalar | float64 | Log amplitude of pixel noise floor |
| `log_tau_0` | scalar | float64 | Log mean optical depth amplitude τ₀ |
| `log_beta` | scalar | float64 | Log power-law index β for τ_eff(z) = τ₀(1+z)^β |

**DESI vs SDSS format distinction** (detected automatically at load time):
- DESI `.h5`: `log_tau_0.ndim == 0` (scalar)
- SDSS `.mat`: `log_tau_0.ndim > 0` (1D array from MATLAB)

**Key pipeline parameters tied to the model:**

| CLI flag | Default (SDSS) | Production (DESI Y3) | Effect |
|----------|---------------|----------------------|--------|
| `--prev_tau_0` | 0.00554 | 0.00246 | Lyman-forest prior τ₀ (Kamble+2020 vs Turner+2024) |
| `--prev_beta` | 3.182 | 3.62 | Lyman-forest prior β |
| `--dlambda` | 0.25 | 0.15 | GP wavelength grid spacing [Å] |
| `--k` | 20 | 30 | Rank of GP covariance low-rank component |

---

## 2. DLA QMC sample file

**Default filename:** `data/dr12q/processed/dla_samples_a03.mat` (legacy 10k, Ho+2020 generator)
**Production (100k PW14 samples):** `dla_samples_pw14_100000.mat`

**Format:** MATLAB v7.3 HDF5 (`.mat`).

**How to get it:**
- 10k samples (`a03`): included in the `gp_dr12_trained` repo (via `download_gp_files.sh`); uses the original Ho+2020 logNHI prior generator
- 100k PW14-based samples: generated with `gpy_dla_detection/generate_samples.py --mode dla` — note this uses the Prochaska+2014 (PW14) logNHI prior mixture, **not** the legacy `a03` generator; name the output `dla_samples_pw14_100000.mat` to avoid confusion with the legacy `a03` files

**Required HDF5 keys** (loaded in `gpy_dla_detection/dla_samples.py:DLASamplesMAT`):

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `offset_samples` | `(N,)` | float64 | Low-discrepancy [0,1] offsets for z_DLA sampling |
| `log_nhi_samples` | `(N,)` | float64 | log₁₀(N_HI / cm⁻²) samples from prior |
| `nhi_samples` | `(N,)` | float64 | 10^log_nhi_samples (linear N_HI) |
| `alpha` | `(1,1)` or scalar | float64 | Mixture weight for data-driven logNHI prior |
| `fit_min_log_nhi` | `(1,1)` or scalar | float64 | Lower boundary of power-law prior region |
| `uniform_min_log_nhi` | `(1,1)` or scalar | float64 | Lower bound of uniform logNHI sampling region |
| `uniform_max_log_nhi` | `(1,1)` or scalar | float64 | Upper bound of uniform logNHI sampling region |

**How z_DLA samples are used:** At inference time, z_DLA_i = z_min + (z_max - z_min) × offset_samples_i, where z_min/z_max are the per-spectrum search window bounds.

**NHI range by run mode:**
| Run mode | logNHI range | Sample file |
|----------|-------------|-------------|
| DLA | [20.3, 23] | `dla_samples_a03.mat` (legacy 10k) / `dla_samples_pw14_100000.mat` (PW14 100k) |
| Sub-DLA | [19, 20.3] | `subdla_samples_a03_191_200_100000.mat` |
| LLS | [17.2, 19] | Generated with `generate_samples.py --mode lls` |

---

## 3. Sub-DLA / LLS QMC sample file

**Default filename:** `data/dr12q/processed/subdla_samples.mat`
**Production:** `subdla_samples_a03_191_200_100000.mat`

**Format:** MATLAB v7.3 HDF5 (`.mat`).

**How to get it:**
- Default: included in `gp_dr12_trained` repo
- Production: `python -m gpy_dla_detection.generate_samples --mode subdla --output subdla_samples.mat`

**Required keys:** same as DLA sample file (see above).

Loaded in `gpy_dla_detection/subdla_samples.py:SubDLASamplesMAT`.

---

## 4. DR9Q concordance prior catalogs

These three files provide the data-driven prior on DLA existence, P(DLA | z_QSO).

### 4a. `catalog.mat`

**Path:** `data/dr12q/processed/catalog.mat`

**How to get it:** `download_gp_files.sh`

**Required HDF5 keys** (loaded in `gpy_dla_detection/model_priors.py:PriorCatalog`):

| Key | Shape | Description |
|-----|-------|-------------|
| `z_qsos` | `(1, N)` | QSO redshifts from the training catalog |
| `in_dr9` | `(1, N)` | Boolean mask: QSO is in DR9Q concordance catalog |
| `in_dr10` | `(1, N)` | Boolean mask: QSO is in DR10Q |
| `filter_flags` | `(1, N)` | Quality flags (0 = clean) |

### 4b. `los_catalog` (text file)

**Path:** `data/dla_catalogs/dr9q_concordance/processed/los_catalog`

**Format:** Plain text. Each line is a SDSS `thingID` (int64) — one entry per QSO line-of-sight searched for DLAs in the DR9Q concordance catalog.

### 4c. `dla_catalog` (text file)

**Path:** `data/dla_catalogs/dr9q_concordance/processed/dla_catalog`

**Format:** Plain text with 3 columns (no header):

| Column | Type | Description |
|--------|------|-------------|
| 0 | int64 | SDSS `thingID` of the QSO sightline |
| 1 | float64 | Absorber redshift z_DLA |
| 2 | float64 | log₁₀(N_HI / cm⁻²) |

---

## 5. QSO catalog (DESI production)

**Filename:** `QSO_cat_loa_main_dark_healpix_v2-altbal-20241115.fits`

**Format:** FITS binary table.

**How to get it:** Available at NERSC:
```
/global/cfs/cdirs/desi/science/lya/y3/loa/catalogs/
```

**Required columns** (when `--balmask` is passed; see `utilities/read_catalogs.py`):

| Column | Type | Description |
|--------|------|-------------|
| `TARGETID` | int64 | Unique fiber target ID |
| `TARGET_RA` | float64 | Right ascension [deg] |
| `TARGET_DEC` | float64 | Declination [deg] |
| `Z` | float64 | QSO spectroscopic redshift |
| `HPXPIXEL` | int64 | Healpix pixel (nside=64) |
| `SPECTYPE` | str | Spectral classification (e.g., "QSO") |
| `ZWARN` | int64 | Redshift warning bitmask (0 = clean) |
| `AI_CIV` | float64 | Absorption index for CIV BAL detection |
| `NCIV_450` | int32 | Number of CIV systems with v > 450 km/s |
| `VMIN_CIV_450` | float32 array | Minimum velocity per CIV system [km/s] |
| `VMAX_CIV_450` | float32 array | Maximum velocity per CIV system [km/s] |

Without `--balmask`, only TARGETID, RA/DEC, Z, HPXPIXEL, SPECTYPE, ZWARN are read.

**The "altbal" suffix** means this catalog was built using an alternative BAL catalog from Paul Martini's team at NERSC (`/global/cfs/cdirs/desi/users/martini/bal-catalogs/`).

**QSO selection cuts** (applied by `read_catalog()`, controlled by `constants.py`):

| Constant | Value | Effect |
|----------|-------|--------|
| `zmin_qso` | 2.0 | Lower redshift cut |
| `zmax_qso` | 4.25 | Upper redshift cut |
| `no_bal` | False | If True, exclude QSOs with NCIV_450 > 0 |
| `zwarning` | False | If True, require ZWARN == 0 |
| `is_qso` | False | If True, require SPECTYPE == "QSO" |

---

## 6. QSO catalog (London mock)

**Filename:** `zcat.fits` (from the London mock data directory)

**Location at NERSC:**
```
/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/zcat.fits
```

**Required columns:** TARGETID, RA, DEC, Z (no BAL columns in mock catalog).

---

## 7. DESI coadded spectra (real data)

**Format:** DESI FITS format, read by `desispec.io.read_spectra()`.

**Filename pattern:** `coadd-{survey}-{program}-{healpix}.fits`

**Example:** `coadd-main-dark-9000.fits`

**Location:** Depends on release; at NERSC:
```
/global/cfs/cdirs/desi/spectro/redux/{release}/healpix/{survey}/{program}/{hpx//100}/{hpx}/
```

**Key HDUs used by the pipeline:**
| HDU | Name | Content |
|-----|------|---------|
| FIBERMAP | | Per-fiber catalog (TARGETID, RA/DEC, etc.) |
| B_WAVELENGTH | | Blue camera wavelength array [Å] |
| B_FLUX | | Blue camera flux array |
| B_IVAR | | Blue camera inverse variance |
| B_MASK | | Blue camera pixel mask |
| R_WAVELENGTH | | Red camera |
| R_FLUX | | |
| R_IVAR | | |
| R_MASK | | |
| Z_WAVELENGTH | | Near-IR camera |
| Z_FLUX | | |
| Z_IVAR | | |
| Z_MASK | | |

After loading, the three cameras (b/r/z) are coadded into a single `brz` grid via `desispec.coaddition.coadd_cameras()`. Resolution data (`B_RESOLUTION`, etc.) are required for `coadd_cameras()`; London mock spectra without these fall back to `resample_spectra_lin_or_log()` with a truth file.

---

## 8. HDF5 output file (per-healpix intermediate)

**Filename pattern:** `processed-{survey}-{program}-{healpix}.h5`

**Written by:** `gpy_dla_detection/process_helpers.py:save_results_to_hdf5()`

Full schema is documented in `process_helpers.py`. Summary:

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `target_ids` | `(N,)` | int64 | TARGETID for each processed spectrum |
| `z_qsos` | `(N,)` | float64 | QSO redshifts |
| `min_z_dlas` | `(N,)` | float64 | Search window lower edge per spectrum |
| `max_z_dlas` | `(N,)` | float64 | Search window upper edge per spectrum |
| `snrs` | `(N,)` | float64 | SNR in red continuum window (1420–1480 Å rest) |
| `snrs_blue` | `(N,)` | float64 | SNR in Lyman forest (1040–1205 Å rest) |
| `model_posteriors` | `(N, 1+num_subdla+K)` | float64 | Posterior per model (see layout below) |
| `p_dlas` | `(N,)` | float64 | Total P(≥1 DLA \| D) |
| `p_no_dlas` | `(N,)` | float64 | P(Null \| D) |
| `MAP_z_dlas` | `(N, K)` | float64 | Best-fit z_DLA per absorber slot |
| `MAP_log_nhis` | `(N, K)` | float64 | Best-fit log₁₀(N_HI) per absorber slot |
| `z_dla_errs` | `(N, K)` | float64 | 1σ error on z_DLA |
| `log_nhi_errs` | `(N, K)` | float64 | 1σ error on log N_HI |
| `log_posteriors_no_dla` | `(N,)` | float64 | log P(Null \| D) |
| `log_posteriors_dla` | `(N, K)` | float64 | log P(DLA(k) \| D) for each absorber |
| `detection_flags` | `(N,)` | bool | True if any DLAFLAG was set |

**`model_posteriors` column layout:**

```
DLA run (single_absorber_model=False, max_dlas=K):
  col 0          → P(Null | D)
  col 1          → P(SubDLA | D)
  col 2 .. K+1   → P(DLA(1)..DLA(K) | D)

Sub-DLA / LLS run (single_absorber_model=True):
  col 0  → P(Null | D)
  col 1  → P(absorber | D)
```

`N` = number of spectra processed; `K` = `max_dlas` (typically 3 or 4).
NaN indicates the k-th absorber slot was not populated (fewer DLAs than K).

---

## 9. Combined HDF5 output (full-survey)

**Filename:** `processed-{survey}-{program}.h5`

Produced by `combine_processed_h5.py`. Same schema as per-healpix file, concatenated across all healpix tiles, filtered to target IDs in the QSO catalog.

---

## 10. DLA catalog (FITS)

**Filename:** `dlacat-{release}-{survey}-{program}.fits`

Produced by `combine_dlakibo.py`. One row per detected absorber.

**Key columns:**

| Column | Type | Description |
|--------|------|-------------|
| `TARGETID` | int64 | QSO TARGETID |
| `RA`, `DEC` | float64 | QSO coordinates [deg] |
| `Z_QSO` | float64 | QSO redshift |
| `SNR_FOREST` | float64 | SNR in Lyman forest (blue side) |
| `SNR_REDSIDE` | float64 | SNR in red continuum |
| `DLAID` | str | Unique absorber ID (`{TARGETID}00{n}`) |
| `Z_DLA` | float64 | MAP absorber redshift |
| `Z_DLA_ERR` | float64 | 1σ error on z_DLA |
| `NHI` | float64 | MAP log₁₀(N_HI) |
| `NHI_ERR` | float64 | 1σ error on log N_HI |
| `DLAFLAG` | int | Bitmask from `fitwarning.DLAFLAG` |
| `P_DLA` | float64 | Total P(≥1 DLA \| D) |
| `P_NULL` | float64 | P(Null \| D) |
| `LOGP_DLA` | float64 | log P(DLA(n) \| D) for this absorber |
| `LOGP_NULL` | float64 | log P(Null \| D) |
| `MODEL_P` | float64 | Posterior of the k-DLA model for this absorber |

---

## Quick download reference

```bash
cd data/scripts

# Public, ~100 MB: SDSS DR12Q trained GP model, DLA samples, prior catalogs
./download_gp_files.sh    # clones github.com/jibanCat/gp_dr12_trained
./download_catalogs.sh    # downloads DR9Q concordance catalogs

# Public, requires SDSS rsync access: SDSS DR12 test spectra (spec-*.fits)
./download_spectra.sh     # rsync from data.sdss.org/dr12/boss/spectro/redux/

# NERSC only: DESI Y3 production files
# - Spectra: /global/cfs/cdirs/desi/spectro/redux/
# - QSO catalog: /global/cfs/cdirs/desi/science/lya/y3/loa/catalogs/
# - DESI Y3 model: /pscratch/sd/j/jibancat/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5
# - 100k samples: /pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/
```
