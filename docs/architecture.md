# GP-DLA Codebase Architecture

> **Addendum (2026-08-26):** a fourth workflow is missing below — the Paper-1 statistics path `extract_pack_real → contract_guards_check → cc_real_posterior (NUTS) → cc_pool_posterior → hbi_reduction (paper repo)`, with the BH arm `track_c_tf_hz → bh_ratify_stamp`. `desi_cddf.py` is the pre-HBI statistics entry point and is superseded for Paper 1. See `docs/PAPER1_REPRODUCTION.md`.


This document describes the high-level structure of the DESI GP-DLA detection pipeline,
its main workflows, and how the modules relate to each other.

---

## Overview

The codebase has three distinct workflows:

| Workflow | Entry point | Purpose |
|----------|-------------|---------|
| **Training** | `tests/phase2_train_desi.py` | Learn the null-GP model from QSO spectra (see `docs/training_overview.md`) |
| **Inference** | `desi-DLAGP.py` | Detect DLA/sub-DLA/LLS absorbers per spectrum |
| **Statistics** | `desi_cddf.py` or CDDF notebooks | Compute population statistics from catalogs |

---

## 1. End-to-End Inference Pipeline

```
DESI coadded spectra (FITS)
  + QSO catalog (TARGETID, Z, HPXPIXEL, BAL flags)
  + Trained null GP model (.mat: mu, M, log_omega)
  + DLA/SubDLA QMC sample grids (.mat)
  + DR9Q prior catalogs (.mat)
          │
          ▼
desi-DLAGP.py  ── CLI (792 lines)
  Parses 60+ arguments: survey, program, release, hpx range, absorber mode, paths
  Filters QSO catalog (redshift, SNR, ZWARN, BAL flags)
          │
          ▼
dlasearch.dlasearch_hpx()  ── per healpix (parallel via ProcessPoolExecutor)
  Loads DESI coadded spectra (desispec.io.read_spectra)
  Coadds b/r/z camera bands (coadd_cameras)
  Applies BAL masking (CIV velocity windows, optional)
          │
          ▼
run_bayes_select.DLAHolder  ── initialized once per healpix
  Loads GP model matrices + QMC sample grids
  Loads DLA existence priors (model_priors.py)
          │
          ▼  (per spectrum)
process_single_spectrum()
  null_gp.NullGPMAT.set_data(wavelengths, flux, noise)
    ↳ normalizes flux, applies Lyman-forest mean-flux suppression
    ↳ builds covariance matrix K + Ω via Woodbury identity (O(nk²))
  dla_gp.DLAGPMAT.set_data(...)
    ↳ adds Voigt absorption profile per QMC sample
    ↳ voigt_fast.voigt_absorption()  [C extension, PRODUCTION]
  bayesian_model_selection.BayesModelSelect.model_selection()
    ↳ computes log-evidence for each model via 10,000 QMC samples
    ↳ applies DLA existence priors
    ↳ multi-DLA stopping criterion (Bayes factor threshold)
  compute_1sigma_errors.compute_1sigma_errors_fast()
    ↳ MAP estimates of z_DLA, log_NHI from max-likelihood sample
          │
          ▼
process_helpers.save_results_to_hdf5()
  Writes per-healpix HDF5: model_posteriors, z_DLA, log_NHI, etc.
          │
          ▼
combine_processed_h5.py  →  processed-{survey}-{program}.h5
combine_dlakibo.py       →  dlacat-{release}-{survey}-{program}.fits
```

### Three absorber run modes

| Mode | Flag | log NHI | Multi-DLA | `model_posteriors` layout |
|------|------|---------|-----------|--------------------------|
| DLA (default) | *(not set)* | 20.3–23 | Yes (up to k=3) | `[:,0]`=Null, `[:,1]`=SubDLA, `[:,2]`=DLA(1), `[:,3]`=DLA(2), `[:,4]`=DLA(3) |
| Sub-DLA | `--single_absorber_model` | 19–20.3 | No | `[:,0]`=Null, `[:,1]`=SubDLA |
| LLS | `--single_absorber_model` | 17.2–19 | No | `[:,0]`=Null, `[:,1]`=LLS |

**Critical**: the column index offset in `model_posteriors` changes with mode.
When reading HDF5 output, always check which mode was used.
`DLACatalogue(sub_dla=True)` assumes the DLA run (columns 2+); `sub_dla=False` for single-absorber runs.

---

## 2. Training Workflow (offline, one-time)

```
Raw DESI QSO spectra (HDF5 preloaded)
  → preload_spectra/desi-preload.py
      → preloaded-{survey}-{program}-*.h5   [cached, chunked by healpix]
  → preload_spectra/prepare_trainset.py
      → training set subset (DLA-free spectra from DR9Q concordance)
  → tests/phase2_train_desi.py   (PR #6 corrected trainer; see docs/training_overview.md)
      → gpy_dla_detection.training_v3.objective_vectorized + Adam loop
          → PCA decomposition of mean spectra
          → hand-coded-gradient minimization of GP log-likelihood
          → Turner+2024 optical depth priors: τ₀=0.00246 ± 0.00014, β=3.62 ± 0.04
      → checkpoint + final export: `model_epoch_*.h5` (HDF5 with mu, M, log_omega)
      (v1 frozen reference for diffing: gpy_dla_detection/training_v3/desi_learn_qsos_model.py)
```

---

## 3. Population Statistics: Two Pathways

After inference, the pipeline supports **two distinct pathways** for computing
CDDF, dN/dX, and Omega_DLA statistics.

### Pathway A — Bayesian model-posterior pathway

Uses the full Bayesian information from the HDF5 inference output.

```
processed-*.h5 (model_posteriors per spectrum)
  → desi_cddf.py  (entry point)
      → CDDF_analysis/calc_cddf.DLACatalogue
          → loads model_posteriors from HDF5
          → sums P(DLA|D) per (z, logN) bin
          → Poisson-binomial CI via DFT (Fernandez & Williams 2010)
          → Bayesian credible intervals on CDDF, dN/dX, Omega_DLA
```

**When to use**: When you want the full Bayesian posterior with credible intervals
that account for individual detection probabilities.

### Pathway B — Direct catalog pathway (for mock validation and calibration)

Takes absorber catalogs (FITS tables) directly. Used for mock truth comparison
and calibration of the real-data measurements.

```
dla_cat.fits (absorbers from inference or mock truth)
  + qso_cat.fits (QSO redshifts)
  → CDDF_analysis/cddf_mock.build_qso_windows()
      → per-QSO search windows [z_lo, z_hi]
  → cddf_mock.AbsorptionDistance
      → X(z) grid: trapezoidal integration of dX/dz, WMAP9 cosmology
  → cddf_mock.compute_dndx()    → dN/dX with Poisson + bootstrap CI
  → cddf_mock.compute_cddf_fN() → 2D f(N,z) array
  → cddf_mock.omega_hi_from_cddf() → Omega_HI(z)
  → cddf_mock.compute_calibration_alpha()  → alpha(z) = measured/truth
  → cddf_mock.apply_calibration()          → calibrated dN/dX for real data
```

**When to use**: For mock validation workflows, and when working with
absorber catalogs rather than raw model posteriors.

---

## 4. Module Map

### Science-sensitive modules (handle with care)

| Module | Role | Sensitivity |
|--------|------|-------------|
| `gpy_dla_detection/null_gp.py` | QSO continuum GP; Woodbury inversion; O(nk²) | HIGH — core GP math |
| `gpy_dla_detection/dla_gp.py` | DLA GP with Voigt absorption; multi-DLA recursion | HIGH — core inference |
| `gpy_dla_detection/subdla_gp.py` | Sub-DLA / LLS variant | HIGH |
| `gpy_dla_detection/bayesian_model_selection.py` | Bayes factors; stopping criterion; model priors | HIGH |
| `gpy_dla_detection/voigt_fast.py` | Voigt profile (C extension, PRODUCTION) | HIGH — physics |
| `gpy_dla_detection/effective_optical_depth.py` | IGM mean flux model (τ_eff power law) | HIGH — physics |
| `gpy_dla_detection/objective.py` | PyTorch training loss (Turner+2024 priors) | HIGH — training |
| `CDDF_analysis/cddf_mock.py` | Population statistics engine (dN/dX, CDDF, Omega) | MEDIUM-HIGH |
| `CDDF_analysis/calc_cddf.py` | Bayesian CDDF via model posteriors + Poisson-binomial | MEDIUM-HIGH |

### Infrastructure modules (safer to modify)

| Module | Role |
|--------|------|
| `desi-DLAGP.py` | CLI entry point, argument parsing, job dispatch |
| `dlasearch.py` | Healpix/tile/mock orchestration, BAL masking |
| `run_bayes_select.py` | DLAHolder: model initialization + per-spectrum dispatch |
| `gpy_dla_detection/process_helpers.py` | HDF5 result schema init + write |
| `gpy_dla_detection/desi_spectrum_reader.py` | DESI FITS I/O, camera coadding |
| `utilities/read_catalogs.py` | QSO catalog loading + filtering |
| `constants.py` | Pipeline-level thresholds (z range, SNR, BAL config) |
| `gpy_dla_detection/set_parameters.py` | All GP hyperparameters and defaults |
| `fitwarning.py` | DLAFLAG bitmask definitions |

### Experimental / legacy modules (do not use in production)

| Module | Status |
|--------|--------|
| `gpy_dla_detection/voigt.py` | DEPRECATED: pure-Python Voigt, slow |
| `gpy_dla_detection/voigt_jit.py` | EXPERIMENTAL: JIT Voigt, slow, not production |
| `gpy_dla_detection/null_meanflux_gp.py` | EXPERIMENTAL: mean-flux marginalization |
| `gpy_dla_detection/dla_meanflux_gp.py` | EXPERIMENTAL |
| `gpy_dla_detection/subdla_meanflux_gp.py` | EXPERIMENTAL |
| `gpy_dla_detection/read_spec.py` | LEGACY: SDSS DR12Q/DR14Q reader |

---

## 5. Key Data Files (not in git, must be downloaded)

| File | Purpose | Source |
|------|---------|--------|
| `learned_qso_model.mat` / `model_epoch_*.h5` | Trained null GP (mu, M, log_omega) | `tests/phase2_train_desi.py` (PR #6); see `docs/training_overview.md` |
| `dla_samples_a03.mat` | QMC grid for DLA params (Ho+2020) | `data/scripts/download_gp_files.sh` |
| `subdla_samples.mat` | QMC grid for sub-DLA/LLS params | `gpy_dla_detection/generate_samples.py` |
| `data/london/dla_cat.fits` | London mock absorber catalog | `data/scripts/download_spectra.sh` |
| `data/london/zcat.fits` | London mock QSO catalog | same |
| `data/loa/preloaded-main-dark-0.h5` | Preloaded real DESI LOA spectra | `preload_spectra/desi-preload.py` |

---

## 6. Scientific Invariants

These must be preserved across any refactor:

1. **Bayesian model comparison** via QMC integration over (z_DLA, log_NHI) joint space
2. **Voigt profile physics** — Lyman series wavelengths, oscillator strengths, damping constants
3. **Effective optical depth** — Turner+2024 priors: τ₀=0.00246, β=3.62
4. **Woodbury-form GP likelihood** — O(nk²) inversion must match MATLAB reference
5. **CDDF formula** — f(N) = d²n/(dN dX) with Bird+ 2016 normalization convention
6. **Path length** — dX/dz = (1+z)² H₀/H(z) with WMAP9 Ω_m=0.279
7. **`model_posteriors` column layout** — must be consistent between inference and CDDF analysis
8. **Prochaska+2014 CDDF spline** — used as truth for mock calibration

---

## 7. Entry Points for Common Tasks

| Task | Command |
|------|---------|
| Run DLA detection on real DESI healpix | `python desi-DLAGP.py --survey main --program dark --release loa ...` |
| Run on London mock data | `python desi-DLAGP.py --mocks --mockdir data/london ...` |
| Train null GP model | `python tests/phase2_train_desi.py ...` (see `docs/training_overview.md`) |
| Compute Bayesian CDDF | `python desi_cddf.py ...` |
| Run population stats from catalog | Use `CDDF_analysis/cddf_mock.py` directly (see notebooks) |
| Demo on one spectrum | `python examples/demo_desi_spectrum.py` |
| Generate sub-DLA/LLS QMC samples | `python -m gpy_dla_detection.generate_samples --mode lls --output ...` |
