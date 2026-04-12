"""
cddf_calibration.py — Calibration factor computation for dN/dX and CDDF f(N,z).

Overview
--------
After running GP-DLA on London mock spectra, the pipeline computes a
calibration factor alpha(z) [for dN/dX] or r(logN, z) [for CDDF f(N,z)]
that accounts for detection incompleteness and false positives.  These
factors are then applied to real DESI measurements.

Two calibration pathways implemented here:

Pathway A — dN/dX calibration (from CDDF_analysis/notebooks/CDDF_dNdX_all.ipynb)
    Uses absolute 68%/95% bounds from bootstrap.

    calibration_factor_alpha(z_meas, y_meas, bounds68_meas, out_truth, ...)
        alpha(z) = y_true(z) / y_meas(z)
        Returned as a dict with fields: z, alpha, alpha_err, y_corr, etc.

    apply_alpha_to_bounds(z, y, y68_lo, y68_hi, y95_lo, y95_hi, cal, ...)
        Propagate asymmetric uncertainties with alpha(z) correction.

    apply_alpha_to_dndx_bounds(...)     — alias for apply_alpha_to_bounds
    apply_alpha_to_omegahi_real(...)    — alias for apply_alpha_to_bounds

Pathway B — CDDF f(N,z) calibration (from CDDF_analysis/notebooks/CDDF_fN_z.ipynb)
    Per-logN-bin correction ratio.

    correction_ratio_with_uncertainty(f_true, sig_true, f_meas_mock, sig_meas_mock)
        r = f_true / f_meas_mock with fractional error propagation.

    apply_correction_with_uncertainty(f_apply, sig_apply, r, sig_r, nsig=1.0)
        f_corr = r * f_apply.

Usage
-----
Both pathways operate on outputs from ``cddf_mock.compute_dndx()`` and
``cddf_mock.compute_cddf_fN()`` respectively.  See those functions for
the dict layouts used as inputs here.

Reference
---------
Ho, Bird & Garnett (2020), arXiv:2003.11036
DESI Y3 GP-DLA calibration notebooks (internal):
  CDDF_analysis/notebooks/CDDF_dNdX_all.ipynb
  CDDF_analysis/notebooks/CDDF_fN_z.ipynb
"""
import numpy as np


# ------------------------------------------------------------------ #
# Pathway A — dN/dX (and Omega_HI) calibration with asymmetric bounds
# ------------------------------------------------------------------ #

def sym_err_from_bounds(y, bounds68):
    """
    Convert absolute 68% bounds into a symmetric ~1σ uncertainty around y.

    Parameters
    ----------
    y : array-like, shape (N,)
        Central values.
    bounds68 : array-like, shape (N, 2)
        Absolute [low, high] bounds (NOT ± offsets).

    Returns
    -------
    sigma : ndarray, shape (N,)
        Approximate symmetric 1σ: 0.5 * ((y - low) + (high - y)).
    """
    y = np.asarray(y, float)
    bounds68 = np.asarray(bounds68, float)
    low = bounds68[:, 0]
    high = bounds68[:, 1]
    return 0.5 * ((y - low) + (high - y))


def _get_truth_z(out_truth):
    """Return the truth redshift grid from either 'z_mid' or 'z'."""
    if "z_mid" in out_truth:
        return np.asarray(out_truth["z_mid"], float)
    if "z" in out_truth:
        return np.asarray(out_truth["z"], float)
    raise KeyError("out_truth must contain 'z_mid' or 'z'.")


def _get_truth_y_and_err(out_truth, *, y_key, err_kind="boot"):
    """
    Get truth y and 1σ error arrays for a given y_key.

    Supported dict layouts (from ``compute_dndx()`` and ``omega_hi_from_cddf()``):
      - dN/dX:  y_key='dndx', err in 'err_boot' or 'err_poisson'
      - Omega_HI: y_key='omega_hi', err in 'omega_hi_err'
    """
    y = np.asarray(out_truth[y_key], float)

    # Specific overrides (in case user adds named error keys later)
    if err_kind == "boot":
        k = f"{y_key}_err_boot"
        if k in out_truth and out_truth[k] is not None:
            return y, np.asarray(out_truth[k], float)
    else:
        k = f"{y_key}_err_poisson"
        if k in out_truth and out_truth[k] is not None:
            return y, np.asarray(out_truth[k], float)

    # dN/dX convention
    if "err_boot" in out_truth and out_truth["err_boot"] is not None and err_kind == "boot":
        return y, np.asarray(out_truth["err_boot"], float)
    if "err_poisson" in out_truth and out_truth["err_poisson"] is not None and err_kind != "boot":
        return y, np.asarray(out_truth["err_poisson"], float)

    # Omega_HI convention
    if f"{y_key}_err" in out_truth and out_truth[f"{y_key}_err"] is not None:
        return y, np.asarray(out_truth[f"{y_key}_err"], float)

    # Fallback
    if "err_boot" in out_truth and out_truth["err_boot"] is not None:
        return y, np.asarray(out_truth["err_boot"], float)
    if "err_poisson" in out_truth and out_truth["err_poisson"] is not None:
        return y, np.asarray(out_truth["err_poisson"], float)

    raise KeyError(
        f"Cannot find error array for y_key='{y_key}'. "
        "Expected: err_boot/err_poisson, '{y_key}_err', or '{y_key}_err_boot/poisson'."
    )


def eval_truth_at_z(z_eval, out_truth, *, y_key, err_kind="boot"):
    """
    Interpolate truth y(z) and its 1σ error onto z_eval.

    Parameters
    ----------
    z_eval : array-like
        Redshift points to interpolate to.
    out_truth : dict
        Output of ``compute_dndx()`` or ``omega_hi_from_cddf()`` on truth catalog.
    y_key : str
        Key for the quantity to interpolate (e.g. 'dndx', 'omega_hi').
    err_kind : str
        'boot' or 'poisson'.

    Returns
    -------
    y_interp : ndarray
    err_interp : ndarray
    """
    zt = _get_truth_z(out_truth)
    yt, et = _get_truth_y_and_err(out_truth, y_key=y_key, err_kind=err_kind)
    z_eval = np.asarray(z_eval, float)
    return np.interp(z_eval, zt, yt), np.interp(z_eval, zt, et)


def calibration_factor_alpha(
    z_meas, y_meas, bounds68_meas,
    out_truth, *,
    truth_y_key,
    truth_err_kind="boot",
    clip=None,
    eps=1e-30,
):
    """
    Compute alpha(z) = y_true(z) / y_meas(z) from mock and truth dN/dX (or Omega_HI).

    The calibration factor corrects for detection incompleteness and false positives.
    Error propagation (independent fractional errors):

        (σ_alpha / alpha)^2 = (σ_true / y_true)^2 + (σ_meas / y_meas)^2

    Parameters
    ----------
    z_meas : array-like, shape (N,)
        Redshift bin centers from the mock measurement.
    y_meas : array-like, shape (N,)
        Measured dN/dX (or Omega_HI) from GP-DLA on mock spectra.
    bounds68_meas : array-like, shape (N, 2)
        Absolute [low, high] 68% bounds on y_meas.
    out_truth : dict
        Output of ``compute_dndx()`` on the truth absorber catalog, OR
        output of ``omega_hi_from_cddf()`` for Omega_HI calibration.
    truth_y_key : str
        Key for the truth quantity in out_truth ('dndx' or 'omega_hi').
    truth_err_kind : str
        'boot' or 'poisson'.
    clip : tuple (lo, hi) or None
        Optional clipping range for alpha.
    eps : float
        Floor to prevent division by zero.

    Returns
    -------
    dict with keys:
        z            : redshift bin centers
        alpha        : calibration factor alpha(z)
        alpha_err    : 1σ error on alpha
        y_meas       : input y_meas
        bounds68_meas: input bounds68_meas
        yerr_meas_sym: symmetric error derived from bounds68_meas
        y_true       : truth values interpolated to z_meas
        yerr_true    : truth 1σ error interpolated to z_meas
        y_corr       : y_meas * alpha (calibrated central value)
        y_corr_err   : error on y_corr
    """
    z_meas = np.asarray(z_meas, float)
    y_meas = np.asarray(y_meas, float)
    bounds68_meas = np.asarray(bounds68_meas, float)

    yerr_meas = sym_err_from_bounds(y_meas, bounds68_meas)
    y_true, yerr_true = eval_truth_at_z(
        z_meas, out_truth, y_key=truth_y_key, err_kind=truth_err_kind
    )

    A = np.maximum(y_true, eps)
    B = np.maximum(y_meas, eps)

    alpha = A / B
    frac = np.sqrt((yerr_true / A) ** 2 + (yerr_meas / B) ** 2)
    alpha_err = alpha * frac

    if clip is not None:
        lo, hi = clip
        alpha = np.clip(alpha, lo, hi)

    y_corr = y_meas * alpha
    y_corr_err = y_corr * frac

    return {
        "z": z_meas,
        "y_meas": y_meas,
        "bounds68_meas": bounds68_meas,
        "yerr_meas_sym": yerr_meas,
        "y_true": y_true,
        "yerr_true": yerr_true,
        "alpha": alpha,
        "alpha_err": alpha_err,
        "y_corr": y_corr,
        "y_corr_err": y_corr_err,
        "meta": {
            "truth_y_key": truth_y_key,
            "truth_err_kind": truth_err_kind,
        },
    }


def bounds_to_asym_sigma(y, low, high):
    """
    Convert absolute bounds [low, high] into asymmetric (σ_minus, σ_plus).

    Parameters
    ----------
    y : array-like
        Central values.
    low, high : array-like
        Absolute lower and upper bounds.

    Returns
    -------
    sigma_minus : ndarray   (y - low)
    sigma_plus  : ndarray   (high - y)
    """
    y = np.asarray(y, float)
    return y - np.asarray(low, float), np.asarray(high, float) - y


def _interp_alpha_at_z(z_cent, cal, *, clip_alpha=None):
    """Interpolate alpha and alpha_err from a calibration dict onto z_cent."""
    z_cent = np.asarray(z_cent, float)
    alpha = np.interp(z_cent, np.asarray(cal["z"], float),
                      np.asarray(cal["alpha"], float))
    alpha_err = np.interp(z_cent, np.asarray(cal["z"], float),
                          np.asarray(cal["alpha_err"], float))
    if clip_alpha is not None:
        lo, hi = clip_alpha
        alpha = np.clip(alpha, lo, hi)
    return alpha, alpha_err


def apply_alpha_to_bounds(
    z_cent, y, y68_low, y68_high, y95_low, y95_high,
    cal, *,
    include_alpha_uncertainty=True,
    clip_alpha=None,
):
    """
    Apply alpha(z) correction to a measurement stored with asymmetric bounds.

    Propagates uncertainties separately for lower and upper bounds:

        frac_±²  = (σ_±/y)² + (σ_alpha/alpha)²
        y_corr_± = y_corr ± y_corr * frac_±

    Parameters
    ----------
    z_cent : array-like, shape (N,)
    y : array-like, shape (N,)
        Central measurement values.
    y68_low, y68_high : array-like, shape (N,)
        Absolute 68% bounds.
    y95_low, y95_high : array-like, shape (N,)
        Absolute 95% bounds.
    cal : dict
        Output of ``calibration_factor_alpha()``, containing 'z', 'alpha', 'alpha_err'.
    include_alpha_uncertainty : bool
        If True (default), include alpha_err in error propagation.
    clip_alpha : tuple (lo, hi) or None
        Optional clipping range for alpha.

    Returns
    -------
    dict with keys: z_cent, alpha, alpha_err, y_raw, y_corr,
        y68_low_corr, y68_high_corr, y95_low_corr, y95_high_corr
    """
    z_cent = np.asarray(z_cent, float)
    y = np.asarray(y, float)
    y68_low = np.asarray(y68_low, float)
    y68_high = np.asarray(y68_high, float)
    y95_low = np.asarray(y95_low, float)
    y95_high = np.asarray(y95_high, float)

    alpha, alpha_err = _interp_alpha_at_z(z_cent, cal, clip_alpha=clip_alpha)
    y_corr = alpha * y

    sig68_m, sig68_p = bounds_to_asym_sigma(y, y68_low, y68_high)
    sig95_m, sig95_p = bounds_to_asym_sigma(y, y95_low, y95_high)

    eps = 1e-30
    frac_alpha = (alpha_err / np.maximum(alpha, eps)) if include_alpha_uncertainty else 0.0

    def propagate(sig):
        return y_corr * np.sqrt((sig / np.maximum(y, eps)) ** 2 + frac_alpha ** 2)

    sig68_m_c = propagate(sig68_m)
    sig68_p_c = propagate(sig68_p)
    sig95_m_c = propagate(sig95_m)
    sig95_p_c = propagate(sig95_p)

    return {
        "z_cent": z_cent,
        "alpha": alpha,
        "alpha_err": alpha_err,
        "y_raw": y,
        "y_corr": y_corr,
        "y68_low_corr": y_corr - sig68_m_c,
        "y68_high_corr": y_corr + sig68_p_c,
        "y95_low_corr": y_corr - sig95_m_c,
        "y95_high_corr": y_corr + sig95_p_c,
    }


def apply_alpha_to_dndx_bounds(z_cent, y, y68_low, y68_high, y95_low, y95_high,
                                cal, **kwargs):
    """Alias for ``apply_alpha_to_bounds`` for dN/dX measurements."""
    return apply_alpha_to_bounds(z_cent, y, y68_low, y68_high, y95_low, y95_high,
                                 cal, **kwargs)


def apply_alpha_to_omegahi_real(z_cent, omega, omega68_low, omega68_high,
                                 omega95_low, omega95_high, cal_omega, **kwargs):
    """Alias for ``apply_alpha_to_bounds`` for Omega_HI measurements."""
    return apply_alpha_to_bounds(z_cent, omega, omega68_low, omega68_high,
                                 omega95_low, omega95_high, cal_omega, **kwargs)


# ------------------------------------------------------------------ #
# Pathway B — CDDF f(N,z) calibration with per-logN-bin correction
# ------------------------------------------------------------------ #

def correction_ratio_with_uncertainty(f_true, sig_true, f_meas_mock, sig_meas_mock):
    """
    Compute the per-logN-bin CDDF correction ratio r = f_true / f_meas_mock.

    Error propagation (fractional, independent):

        (σ_r / r)² = (σ_true / f_true)² + (σ_meas_mock / f_meas_mock)²

    Parameters
    ----------
    f_true : array-like
        Truth CDDF values f(N) [from Prochaska+2014 spline or mock truth catalog].
    sig_true : array-like
        1σ uncertainty on f_true.
    f_meas_mock : array-like
        Measured CDDF from GP-DLA on mock spectra.
    sig_meas_mock : array-like
        1σ uncertainty on f_meas_mock.

    Returns
    -------
    r : ndarray
        Correction ratio (NaN where f_true or f_meas_mock is non-positive).
    sig_r : ndarray
        1σ error on r.
    """
    f_true = np.asarray(f_true, dtype=float)
    sig_true = np.asarray(sig_true, dtype=float)
    f_meas_mock = np.asarray(f_meas_mock, dtype=float)
    sig_meas_mock = np.asarray(sig_meas_mock, dtype=float)

    r = np.full_like(f_true, np.nan)
    sig_r = np.full_like(f_true, np.nan)

    m = (
        np.isfinite(f_true) & np.isfinite(sig_true) &
        np.isfinite(f_meas_mock) & np.isfinite(sig_meas_mock) &
        (f_true > 0) & (f_meas_mock > 0)
    )

    r[m] = f_true[m] / f_meas_mock[m]
    frac_true = np.where(m, sig_true / np.where(f_true > 0, f_true, 1.0), 0.0)
    frac_meas = np.where(m, sig_meas_mock / np.where(f_meas_mock > 0, f_meas_mock, 1.0), 0.0)
    sig_r[m] = r[m] * np.sqrt(frac_true[m] ** 2 + frac_meas[m] ** 2)

    return r, sig_r


def apply_correction_with_uncertainty(f_apply, sig_apply, r, sig_r, nsig=1.0):
    """
    Apply a per-logN-bin correction factor to a CDDF measurement.

    Corrected CDDF:
        f_corr = r * f_apply

    Error propagation (fractional, independent):
        (σ_corr / f_corr)² = (σ_apply / f_apply)² + (σ_r / r)²

    Parameters
    ----------
    f_apply : array-like
        CDDF values to calibrate (e.g. from GP-DLA on real DESI data).
    sig_apply : array-like
        1σ uncertainty on f_apply.
    r : array-like
        Correction ratio from ``correction_ratio_with_uncertainty()``.
    sig_r : array-like
        1σ error on r.
    nsig : float
        Interval half-width in units of sigma (default 1.0 for 68% symmetric interval).

    Returns
    -------
    f_corr : ndarray
        Calibrated CDDF values.
    sig_corr : ndarray
        1σ error on f_corr.
    interval : ndarray, shape (N, 2)
        Symmetric [low, high] interval: f_corr ± nsig × sig_corr.
        Lower bound is clipped at 0.
    """
    f_apply = np.asarray(f_apply, dtype=float)
    sig_apply = np.asarray(sig_apply, dtype=float)
    r = np.asarray(r, dtype=float)
    sig_r = np.asarray(sig_r, dtype=float)

    f_corr = np.full_like(f_apply, np.nan)
    sig_corr = np.full_like(f_apply, np.nan)

    m = (
        np.isfinite(f_apply) & np.isfinite(sig_apply) &
        np.isfinite(r) & np.isfinite(sig_r) &
        (f_apply > 0) & (r > 0)
    )

    f_corr[m] = r[m] * f_apply[m]
    frac_apply = np.where(m, sig_apply / np.where(f_apply > 0, f_apply, 1.0), 0.0)
    frac_r = np.where(m, sig_r / np.where(r > 0, r, 1.0), 0.0)
    sig_corr[m] = f_corr[m] * np.sqrt(frac_apply[m] ** 2 + frac_r[m] ** 2)

    lo = f_corr - nsig * sig_corr
    hi = f_corr + nsig * sig_corr
    lo = np.where(np.isfinite(lo), np.maximum(lo, 0.0), np.nan)

    return f_corr, sig_corr, np.column_stack([lo, hi])
