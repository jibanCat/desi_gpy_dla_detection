# Tutorial: Population Statistics Workflow

This tutorial explains how to compute population statistics — dN/dX, CDDF f(N,z),
and Omega_HI — from GP-DLA inference outputs, and how to calibrate real-data
measurements using London mock spectra.

**Audience**: A collaborator who has already run the inference pipeline
(`desi-DLAGP.py`) and has absorber catalogs in hand.

---

## Overview of the Two Analysis Pathways

The codebase offers two ways to compute CDDF / dN/dX statistics:

| Pathway | Module | Input | CI method |
|---------|--------|-------|-----------|
| **A: Bayesian** | `CDDF_analysis/calc_cddf.py` | HDF5 model posteriors (`processed-*.h5`) | Poisson-binomial via DFT |
| **B: Direct catalog** | `CDDF_analysis/cddf_mock.py` | FITS absorber catalog (`dla_cat.fits`) | Bootstrap over QSO sightlines |

**Use Pathway A** when you want the full Bayesian posterior propagation
(individual detection probabilities are preserved in the credible intervals).

**Use Pathway B** when working directly with absorber catalogs,
for mock validation, or for calibration workflows.

The CDDF notebooks (`notebooks/CDDF_dNdX_all.ipynb`, `notebooks/CDDF_fN_z.ipynb`)
use **Pathway B** with mock validation and calibration.

---

## Part 1: Direct Catalog Statistics (Pathway B)

### 1.1 Required inputs

```python
from astropy.table import Table

# QSO sightline catalog
qso_cat = Table.read("data/london/zcat.fits")
# Required columns: TARGETID, Z

# Absorber catalog (from GP-DLA inference, or mock truth)
dla_cat = Table.read("data/london/dla_cat.fits")
# Required columns: TARGETID, Z_DLA, NHI
# NHI should be log10(NHI/cm^-2) when assume_logNHI=True (default)
```

### 1.2 Define search window parameters

These parameters define which absorbers fall within the valid search window
for each QSO sightline.

```python
import numpy as np

# Standard DESI DLA search window parameters
window_params = dict(
    zmin           = 2.15,         # Global floor on absorber redshift
    zmax_global    = None,         # No global ceiling
    v_prox_kms     = 3000.0,       # Proximity-zone velocity cut [km/s]
    absorber_rest  = 1215.67,      # Lyα rest wavelength [Å] — defines z_DLA
    blue_rest      = 1025.72,      # Lyβ rest wavelength [Å] — defines QSO blue edge
    blue_limit_mode= "max",        # Blue edge = max of {zmin, lambda_obs_min, QSO Lyβ}
    lambda_obs_min = 3700.0,       # DESI instrument blue cutoff [Å]
    lambda_obs_max = None,
    Omega_m        = 0.279,        # WMAP9 cosmology
)
```

The per-QSO search window `[z_lo, z_hi]` is:

```
z_lo = max(
    2.15,                                           # zmin
    3700/1215.67 - 1 ≈ 2.045,                      # from lambda_obs_min
    (1 + z_qso) * (1025.72 / 1215.67) - 1          # QSO Lyβ edge
)
z_hi = z_qso - (1 + z_qso) * (3000 / 299792.458)  # proximity zone
```

### 1.3 Define redshift bins

```python
# Uniform z-bins matching the London mock z_cent values
zbins = np.array([2.10, 2.64, 3.08, 3.54, 4.00, 4.50])  # example
# Or derive from z_mid using cddf_mock.zbins_from_zmid_uniform()
```

### 1.4 Compute dN/dX

```python
from CDDF_analysis.cddf_mock import compute_dndx

# DLA dN/dX (log NHI > 20.3)
out_dla = compute_dndx(
    dla_cat, qso_cat,
    zbins=zbins,
    logNHImin=20.3,
    logNHImax=23.0,
    n_boot=200,           # bootstrap error over QSO sightlines
    **window_params,
)

# Sub-DLA dN/dX (19 < log NHI < 20.3)
out_subdla = compute_dndx(
    dla_cat, qso_cat,
    zbins=zbins,
    logNHImin=19.0,
    logNHImax=20.3,
    n_boot=200,
    **window_params,
)

# LLS dN/dX (17.2 < log NHI < 19.0)
out_lls = compute_dndx(
    dla_cat, qso_cat,
    zbins=zbins,
    logNHImin=17.2,
    logNHImax=19.0,
    n_boot=200,
    **window_params,
)
```

**Output dict keys:**

| Key | Shape | Description |
|-----|-------|-------------|
| `z_mid` | (nbins,) | Redshift bin centers |
| `dndx` | (nbins,) | dN/dX per bin |
| `err_poisson` | (nbins,) | Poisson error = sqrt(N_abs) / ΔX |
| `err_boot` | (nbins,) | Bootstrap std over QSO sightlines |
| `N_abs` | (nbins,) | Raw absorber counts |
| `X_tot` | (nbins,) | Total absorption distance ΔX per bin |
| `meta` | dict | All parameters used |

### 1.5 Compute CDDF f(N,z)

```python
from CDDF_analysis.cddf_mock import compute_cddf_fN

logN_bins = np.arange(17.2, 22.2, 0.2)   # 0.2 dex bins in log10 N

out_cddf = compute_cddf_fN(
    dla_cat, qso_cat,
    zbins=zbins,
    logN_bins=logN_bins,
    logNHImin=17.2,
    logNHImax=22.0,
    n_boot=200,
    **window_params,
)
```

**Output dict keys:**

| Key | Shape | Description |
|-----|-------|-------------|
| `fN` | (nbins_z, nbins_logN) | CDDF f(N,z) [cm²] |
| `err_poisson` | same | Poisson error |
| `err_boot` | same | Bootstrap error |
| `logN_mid` | (nbins_logN,) | log10 N bin centers |
| `N_mid` | (nbins_logN,) | Linear N bin centers (geometric mean) [cm⁻²] |
| `dN` | (nbins_logN,) | Linear bin widths ΔN [cm⁻²] |
| `X_tot` | (nbins_z,) | Total ΔX per z-bin |

### 1.6 Compute Omega_HI

```python
from CDDF_analysis.cddf_mock import omega_hi_from_cddf

out_omega = omega_hi_from_cddf(
    out_cddf,
    logN_min=17.2,
    logN_max=22.0,
    H0_km_s_Mpc=70.0,
)

print("Omega_HI(z):", out_omega["omega_hi"])
print("Omega_HI err:", out_omega["omega_hi_err"])
```

---

## Part 2: Mock Validation and Calibration

The calibration workflow uses London mock spectra to estimate the
detection efficiency of the GP-DLA pipeline. The ratio of measured-to-truth
statistics gives a redshift-dependent correction factor alpha(z).

### 2.1 Understand the calibration logic

```
alpha(z) = dNdX_truth(z) / dNdX_measured_mock(z)

where:
  dNdX_truth         = integral of Prochaska+2014 CDDF spline over log NHI range
  dNdX_measured_mock = GP-DLA result on London mock spectra (Pipeline B)

Then for real DESI data:
  dNdX_calibrated(z) = alpha(z) × dNdX_real(z)
  err_calibrated     = sqrt((alpha × err_real)² + (dNdX_real × err_alpha)²)
```

`alpha(z) > 1` means the pipeline is missing absorbers (incompleteness — measured < truth).
`alpha(z) < 1` means the pipeline is over-counting (false positives or bias — measured > truth).

### 2.2 Compute the truth dN/dX

```python
from CDDF_analysis.cddf_mock import truth_cddf_prochaska2014, truth_dndx_prochaska2014

# Evaluate truth CDDF at a grid of logN
logN_grid = np.linspace(17.2, 22, 500)
log10_fN_truth = truth_cddf_prochaska2014(logN_grid)
fN_truth = 10 ** log10_fN_truth  # in cm^2

# Integrate to get truth dN/dX for each absorber class
dndx_truth_lls    = truth_dndx_prochaska2014(17.2, 19.0)
dndx_truth_subdla = truth_dndx_prochaska2014(19.0, 20.3)
dndx_truth_dla    = truth_dndx_prochaska2014(20.3, 23.0)
```

### 2.3 Load London mock truth catalog and compute truth dN/dX

If you have the London mock truth catalog (injected absorbers before noise),
you can directly compute the truth dN/dX from it rather than the spline:

```python
mock_truth_cat = Table.read("data/london/dla_cat.fits")  # truth absorbers
qso_cat = Table.read("data/london/zcat.fits")

out_truth_lls = compute_dndx(
    mock_truth_cat, qso_cat,
    zbins=zbins,
    logNHImin=17.2, logNHImax=19.0,
    n_boot=200,
    **window_params,
)
```

### 2.4 Run GP-DLA on mock and compute measured dN/dX

Run `desi-DLAGP.py --mocks --mockdir data/london ...` to detect absorbers
in the London mock spectra, then compute statistics from the detected catalog:

```python
detected_mock_cat = Table.read("path/to/mock_detections/dla_cat.fits")

out_measured_lls = compute_dndx(
    detected_mock_cat, qso_cat,
    zbins=zbins,
    logNHImin=17.2, logNHImax=19.0,
    n_boot=200,
    **window_params,
)
```

### 2.5 Compute calibration factor alpha(z)

```python
from CDDF_analysis.cddf_mock import compute_calibration_alpha, apply_calibration

# Compare measured mock to truth
cal_lls = compute_calibration_alpha(out_truth_lls, out_measured_lls)

print("z:          ", cal_lls["z"])
print("alpha(z):   ", cal_lls["alpha"])   # ideally close to 1.0
print("err_alpha:  ", cal_lls["alpha_err"])
```

### 2.6 Apply calibration to real DESI data

```python
# Compute raw statistics on real DESI data
real_dla_cat = Table.read("path/to/real/desi/dla_cat.fits")
real_qso_cat = Table.read("path/to/real/desi/qso_cat.fits")

out_real_lls = compute_dndx(
    real_dla_cat, real_qso_cat,
    zbins=zbins,
    logNHImin=17.2, logNHImax=19.0,
    n_boot=200,
    **window_params,
)

# Apply calibration
result_lls = apply_calibration(out_real_lls, cal_lls)

print("z:                  ", result_lls["z"])
print("dN/dX raw:          ", result_lls["dndx_raw"])
print("dN/dX calibrated:   ", result_lls["dndx_calibrated"])
print("err calibrated:     ", result_lls["err_calibrated"])
```

### 2.7 Save calibrated results to text files

```python
import numpy as np

z = result_lls["z"]
data = np.column_stack([
    z,
    result_lls["dndx_raw"],
    result_lls["err_raw"],
    result_lls["dndx_calibrated"],
    result_lls["err_calibrated"],
])
np.savetxt(
    "lls_dndx_calibrated.txt",
    data,
    header="z_mid  dndx_raw  err_raw  dndx_cal  err_cal",
)
```

---

## Part 3: Plotting

```python
from CDDF_analysis.cddf_mock import plot_dndx, plot_cddf_slice_fN, plot_omega_hi
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plot_dndx(out_lls,    label="LLS (17.2-19.0)",   ax=axes[0], show=False)
plot_dndx(out_subdla, label="subDLA (19.0-20.3)", ax=axes[0], show=False)
plot_dndx(out_dla,    label="DLA (20.3-23.0)",    ax=axes[0], show=False)
axes[0].set_title("dN/dX vs redshift")

# CDDF slice at second z-bin
plot_cddf_slice_fN(out_cddf, zbin_index=1, label="z ≈ 2.5", ax=axes[1], show=False)
axes[1].set_title("f(N) at z ≈ 2.5")

plot_omega_hi(out_omega, label="Omega_HI", ax=axes[2], show=False)
axes[2].set_title("Omega_HI(z)")

plt.tight_layout()
plt.savefig("population_statistics.pdf")
```

---

## Part 4: Bayesian CDDF Pathway (for reference)

If you want to use the Bayesian model-posterior approach instead:

```python
# CLI usage
# python desi_cddf.py \
#   --processed_file processed-main-dark.h5 \
#   --sample_file subdla_samples.mat \
#   --catalog_file dlacat-loa-main-dark.fits \
#   --snr 6.0 \
#   --sub_dla         # include sub-DLA model in posteriors

from CDDF_analysis.calc_cddf import DLACatalogue

cat = DLACatalogue(
    processed_file="processed-main-dark.h5",
    sample_file="subdla_samples.mat",
    catalog_file="dlacat-loa-main-dark.fits",
    snr=6.0,
    sub_dla=True,     # set True for DLA run mode (includes sub-DLA column)
)

# Compute dN/dX with Bayesian credible intervals
z_mid, dndx, lower_68, upper_68 = cat.line_density(
    z_min=2.0, z_max=4.0,
    lnhi_min=20.3, lnhi_max=23.0,
)
```

See `CDDF_analysis/calc_cddf.py` module docstring for full details.

---

## Hardcoded Parameters Reference

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `zmin` | 2.15 | Lower redshift floor (DESI coverage) |
| `v_prox_kms` | 3000.0 | Proximity-zone velocity cutoff |
| `lambda_obs_min` | 3700.0 Å | DESI blue wavelength limit |
| `absorber_rest` | 1215.67 Å | Lyα — defines z_DLA |
| `blue_rest` | 1025.72 Å | Lyβ — defines QSO blue edge |
| `Omega_m` | 0.279 | WMAP9 matter density |
| `H0` | 70.0 km/s/Mpc | Hubble constant (Omega_HI only) |
| LLS logNHI | [17.2, 19.0] | Lyman Limit System range |
| sub-DLA logNHI | [19.0, 20.3] | sub-DLA range |
| DLA logNHI | [20.3, 23.0] | DLA range |
| SNR cut (DLA) | > 4 | Signal-to-noise threshold |
| SNR cut (LLS/subDLA) | > 6 | Signal-to-noise threshold |
| `n_boot` | 200 | Bootstrap samples |
| logN bin width | 0.2 dex | CDDF binning resolution |

---

## See Also

- `docs/architecture.md` — Code flow diagram and module map
- `docs/tutorial_quickstart.md` — Running the inference pipeline
- `CDDF_analysis/cddf_mock.py` — Module docstring with full API reference
- `notebooks/CDDF_dNdX_all.ipynb` — Full calibration workflow in practice (76 cells)
- `notebooks/CDDF_fN_z.ipynb` — 2D CDDF f(N,z) with calibration (31 cells)
