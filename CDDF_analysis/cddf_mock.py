"""
CDDF_analysis/cddf_mock.py
==========================
DLA / LLS / sub-DLA population statistics: dN/dX, CDDF f(N), and Omega_HI.

This module implements the **direct-catalog** population statistics pathway.
It takes an absorber catalog (from GP-DLA inference or a mock truth catalog)
plus a QSO sightline catalog, applies per-QSO search windows, and computes:

  - dN/dX (line density) in redshift bins
  - CDDF f(N, z) = d²N_abs / (dN dX)  (Bird+ 2016 convention)
  - Omega_HI(z) from CDDF integration
  - Mock validation calibration: alpha(z) = f_measured_mock / f_truth

This is **distinct from** ``CDDF_analysis/calc_cddf.py``, which propagates
full Bayesian model posteriors (from ``process_helpers.py`` HDF5 output)
through a Poisson-binomial distribution to obtain credible intervals.
Use this module when working directly with absorber catalogs (FITS tables),
and use ``calc_cddf.py`` when working with the HDF5 model-posterior files.

CDDF convention (Bird+ 2016 / arXiv:1610.01165):
-------------------------------------------------
    f(N) = d²N_abs / (dN dX)

where N is **linear** column density in cm⁻² and X is the dimensionless
comoving absorption distance:

    dX/dz = (1+z)² H₀/H(z)   (Bahcall & Peebles 1969)

Bins are defined in log₁₀(N) space, but the normalization divides by
ΔN = 10^{logN_hi} - 10^{logN_lo}  (linear width), so f(N) has units of cm².

Search-window logic:
--------------------
Absorber redshift z_abs is defined using ``absorber_rest`` (default Lyα = 1215.67 Å).

The blue edge of the per-QSO window can be set by:
  - a global ``zmin`` floor
  - the instrument blue limit: ``lambda_obs_min / absorber_rest - 1``
  - the QSO Lyβ edge: ``(1+z_qso) * blue_rest / absorber_rest - 1``
  - all three combined via ``blue_limit_mode="max"``

The red edge is set by:
  - proximity-zone cut: ``z_qso - (1+z_qso) * v_prox_kms / c``
  - optional ``lambda_obs_max`` or ``zmax_global``

Typical DESI DLA-style parameters:
  - absorber_rest    = 1215.67  (Lyα)
  - blue_rest        = 1025.72  (Lyβ)
  - blue_limit_mode  = "max"
  - lambda_obs_min   = 3700.0   (DESI blue cutoff in Å)
  - v_prox_kms       = 3000.0
  - Omega_m          = 0.279    (WMAP9)
  - zmin             = 2.15

Input catalog requirements:
  - QSO catalog:      TARGETID, Z
  - Absorber catalog: TARGETID, Z_DLA, NHI  (NHI in log10 if assume_logNHI=True)

Mock calibration workflow:
--------------------------
To calibrate real-data statistics using London mock spectra:

1. Compute ``dNdX_truth`` from the Prochaska+2014 CDDF spline
   (use ``truth_cddf_prochaska2014()`` and integrate over logN range).
2. Compute ``dNdX_measured_mock`` by running GP-DLA on London mock spectra
   and calling ``compute_dndx()`` on the detected mock absorbers.
3. Compute calibration: ``alpha(z) = dNdX_measured_mock(z) / dNdX_truth(z)``
   using ``compute_calibration_alpha()``.
4. Apply to real data: ``dNdX_calibrated = alpha(z) × dNdX_real``
   using ``apply_calibration()``.

References:
-----------
- Bird, Garnett & Ho (2017), MNRAS 466, 2111 [arXiv:1610.01165]
  CDDF convention and Poisson-binomial CI method.
- Prochaska, Worseck & O'Meara (2009), ApJL 705, L113
  Source of the wide-logN CDDF spline used as calibration truth.
- Bahcall & Peebles (1969), ApJL 156, L7
  Comoving absorption distance dX/dz formula.
"""
# ============================================================
# DLA/LLS/subDLA summary statistics: dN/dX and CDDF
#
# Correct CDDF convention (Bird+ 2016 / arXiv:1610.01165):
#   f(N) = d^2 N_abs / (dN dX)
# where N is LINEAR column density in cm^-2, and X is dimensionless
# ============================================================

import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import cm

# ----------------------------
# Constants
# ----------------------------
C_KMS = 299792.458
LYA_REST = 1215.67
LYB_REST = 1025.72


# ----------------------------
# 1) Cosmology / path length
# ----------------------------

def HubbleByH0(z, Omega_m=0.279):
    """
    H(z)/H0 for flat LCDM with Omega_m, Omega_L = 1 - Omega_m
    """
    z = np.asarray(z, dtype=float)
    return np.sqrt(Omega_m * (1.0 + z) ** 3 + (1.0 - Omega_m))


def path_length_int(z, Omega_m=0.279):
    """
    Exact integrand:
      dX/dz = (1+z)^2 / (H(z)/H0)
    """
    z = np.asarray(z, dtype=float)
    return (1.0 + z) ** 2 / HubbleByH0(z, Omega_m)


class AbsorptionDistance:
    """
    Fast helper to compute X(z)=∫ dX/dz dz and ΔX via grid + interpolation.
    Uses the exact dX/dz definition above.
    """
    def __init__(self, zmax, Omega_m=0.279, ngrid=40001):
        self.Omega_m = float(Omega_m)
        self.zgrid = np.linspace(0.0, float(zmax), int(ngrid))
        integrand = path_length_int(self.zgrid, Omega_m=self.Omega_m)

        dz = np.diff(self.zgrid)
        X = np.empty_like(self.zgrid)
        X[0] = 0.0
        X[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz)
        self.Xgrid = X

    def X(self, z):
        z = np.asarray(z, dtype=float)
        return np.interp(z, self.zgrid, self.Xgrid)

    def deltaX(self, z1, z2):
        z1 = np.asarray(z1, dtype=float)
        z2 = np.asarray(z2, dtype=float)
        return self.X(z2) - self.X(z1)


# ----------------------------
# 2) QSO searchable windows
# ----------------------------

def zmax_nonprox(z_qso, v_prox_kms=10000.0):
    """
    Proximate cut:
      z_max = z_qso - (1+z_qso) * v/c
    """
    z_qso = np.asarray(z_qso, dtype=float)
    return z_qso - (1.0 + z_qso) * (v_prox_kms / C_KMS)


def observed_lambda_to_z_abs(lambda_obs, absorber_rest=LYA_REST):
    """
    Convert observed wavelength to absorber redshift using absorber_rest.

    For DLA searches absorber_rest should be Lyα = 1215.67 Å.
    """
    return np.asarray(lambda_obs, dtype=float) / float(absorber_rest) - 1.0


def qso_blue_edge_to_z_abs(z_qso, blue_rest=LYB_REST, absorber_rest=LYA_REST):
    """
    Convert the QSO rest-frame blue cutoff (e.g. Lyβ) into absorber redshift.

    If the search region blue edge is QSO Lyβ, then:
      lambda_obs,edge = (1+z_qso) * blue_rest

    But absorber redshift is defined through absorber_rest (usually Lyα):
      z_abs = lambda_obs,edge / absorber_rest - 1
    """
    z_qso = np.asarray(z_qso, dtype=float)
    return (1.0 + z_qso) * (float(blue_rest) / float(absorber_rest)) - 1.0


def build_qso_windows(
    qso_cat,
    *,
    zmin=None,
    zmax_global=None,
    v_prox_kms=10000.0,
    absorber_rest=LYA_REST,
    blue_limit_mode="global",
    blue_rest=LYB_REST,
    lambda_obs_min=None,
    lambda_obs_max=None,
):
    """
    Build per-QSO absorber windows [z_lo, z_hi].

    Parameters
    ----------
    qso_cat : table-like
        Must provide columns:
          - TARGETID
          - Z   (QSO redshift)

    zmin : float or None
        Optional global floor on absorber redshift.

    zmax_global : float or None
        Optional global ceiling on absorber redshift.

    v_prox_kms : float
        Proximity-zone cut for the red edge:
          z_hi = z_qso - (1+z_qso) * v/c

    absorber_rest : float
        Rest wavelength used to define absorber redshift.
        For DLA searches this should be Lyα = 1215.67 Å.

    blue_limit_mode : {"global", "lyb", "max"}
        How to define the lower edge:
          - "global": use only zmin and/or lambda_obs_min
          - "lyb"   : use only the QSO blue_rest edge
          - "max"   : use max of all available lower-bound candidates
                      among {zmin, lambda_obs_min, QSO blue_rest edge}

    blue_rest : float
        QSO rest-frame blue edge of the search region.
        For Lyβ cutoff use 1025.72 Å.

    lambda_obs_min, lambda_obs_max : float or None
        Instrument observed wavelength limits in Å. These are converted into
        absorber redshift using absorber_rest.
    """
    tid = np.asarray(qso_cat["TARGETID"])
    zq = np.asarray(qso_cat["Z"], dtype=float)

    # ----- lower edge candidates -----
    lower_candidates = []

    if zmin is not None:
        lower_candidates.append(np.full_like(zq, float(zmin), dtype=float))

    if lambda_obs_min is not None:
        z_from_obsmin = observed_lambda_to_z_abs(lambda_obs_min, absorber_rest=absorber_rest)
        lower_candidates.append(np.full_like(zq, z_from_obsmin, dtype=float))

    if blue_limit_mode in ("lyb", "max"):
        z_from_blue = qso_blue_edge_to_z_abs(
            zq, blue_rest=blue_rest, absorber_rest=absorber_rest
        )
        lower_candidates.append(z_from_blue)

    if blue_limit_mode == "global":
        if len(lower_candidates) == 0:
            raise ValueError(
                "blue_limit_mode='global' requires at least one of zmin or lambda_obs_min."
            )
        z_lo = np.maximum.reduce(lower_candidates)

    elif blue_limit_mode == "lyb":
        z_lo = qso_blue_edge_to_z_abs(zq, blue_rest=blue_rest, absorber_rest=absorber_rest)

    elif blue_limit_mode == "max":
        if len(lower_candidates) == 0:
            raise ValueError(
                "blue_limit_mode='max' requires at least one lower-bound candidate."
            )
        z_lo = np.maximum.reduce(lower_candidates)

    else:
        raise ValueError("blue_limit_mode must be one of {'global', 'lyb', 'max'}.")

    # ----- upper edge candidates -----
    upper_candidates = [zmax_nonprox(zq, v_prox_kms=v_prox_kms)]

    if lambda_obs_max is not None:
        z_from_obsmax = observed_lambda_to_z_abs(lambda_obs_max, absorber_rest=absorber_rest)
        upper_candidates.append(np.full_like(zq, z_from_obsmax, dtype=float))

    if zmax_global is not None:
        upper_candidates.append(np.full_like(zq, float(zmax_global), dtype=float))

    z_hi = np.minimum.reduce(upper_candidates)

    ok = np.isfinite(z_lo) & np.isfinite(z_hi) & (z_hi > z_lo)
    return tid[ok], z_lo[ok], z_hi[ok]


# ----------------------------
# 3) Filter absorbers to selected QSO sample + windows
# ----------------------------

def filter_absorbers_to_qsos(
    abs_cat,
    qso_tid, qso_zlo, qso_zhi,
    *,
    logNHImin=None,
    logNHImax=None,
    assume_logNHI=True,
):
    """
    Keep only absorbers that:
      - have TARGETID in the provided QSO catalog
      - satisfy z_lo <= z_abs <= z_hi for that QSO
      - satisfy logNHI cuts (if provided)

    abs_cat must provide:
      - TARGETID
      - Z_DLA (absorber redshift)
      - NHI (log10NHI if assume_logNHI=True else linear NHI)
    """
    tid_abs = np.asarray(abs_cat["TARGETID"])
    z_abs = np.asarray(abs_cat["Z_DLA"], dtype=float)
    NHIcol = np.asarray(abs_cat["NHI"], dtype=float)

    logN = NHIcol if assume_logNHI else np.log10(NHIcol)

    # map TARGETID -> QSO index using sorted search
    sort_idx = np.argsort(qso_tid)
    qso_tid_sorted = qso_tid[sort_idx]

    pos = np.searchsorted(qso_tid_sorted, tid_abs)
    in_bounds = (pos >= 0) & (pos < len(qso_tid_sorted))
    match = in_bounds & (qso_tid_sorted[pos.clip(0, len(qso_tid_sorted) - 1)] == tid_abs)

    qso_idx = sort_idx[pos[match]]
    z_m = z_abs[match]
    logN_m = logN[match]
    tid_m = tid_abs[match]

    # window cut
    zlo = qso_zlo[qso_idx]
    zhi = qso_zhi[qso_idx]
    w = (z_m >= zlo) & (z_m <= zhi)

    z_w = z_m[w]
    logN_w = logN_m[w]
    tid_w = tid_m[w]
    qso_idx_w = qso_idx[w]

    # column density cuts
    if logNHImin is not None:
        m = logN_w >= float(logNHImin)
        z_w, logN_w, tid_w, qso_idx_w = z_w[m], logN_w[m], tid_w[m], qso_idx_w[m]
    if logNHImax is not None:
        m = logN_w <= float(logNHImax)
        z_w, logN_w, tid_w, qso_idx_w = z_w[m], logN_w[m], tid_w[m], qso_idx_w[m]

    return z_w, logN_w, tid_w, qso_idx_w


# ----------------------------
# 4) Total ΔX in z bins
# ----------------------------

def total_DeltaX_in_zbins(zbins, qso_zlo, qso_zhi, Xcalc):
    """
    Total absorption distance ΔX in each z-bin, summed over all QSO sightlines.

    For each redshift bin [z_k, z_{k+1}] and each QSO with window [z_lo_i, z_hi_i]:

        ΔX_k = Σ_i X(min(z_hi_i, z_{k+1})) - X(max(z_lo_i, z_k))

    where X(z) is the comoving absorption distance and the overlap is taken
    only for sightlines that intersect bin k.

    Parameters
    ----------
    zbins : array of shape (nbins+1,)
        Bin edges in absorber redshift.
    qso_zlo, qso_zhi : arrays of shape (nqso,)
        Per-QSO search window lower and upper edges.
    Xcalc : AbsorptionDistance
        Precomputed comoving distance calculator.

    Returns
    -------
    X_tot : array of shape (nbins,)
        Total absorption distance per z-bin. Zero for bins with no coverage.
    """
    zbins = np.asarray(zbins, dtype=float)
    nb = len(zbins) - 1
    X_tot = np.zeros(nb, dtype=float)

    for k in range(nb):
        lo, hi = zbins[k], zbins[k + 1]
        o_lo = np.maximum(qso_zlo, lo)
        o_hi = np.minimum(qso_zhi, hi)
        m = o_hi > o_lo
        if np.any(m):
            X_tot[k] = np.sum(Xcalc.deltaX(o_lo[m], o_hi[m]))
    return X_tot


# ----------------------------
# 5) dN/dX
# ----------------------------

def compute_dndx(
    abs_cat, qso_cat,
    *,
    zbins,
    zmin=None,
    zmax_global=None,
    v_prox_kms=10000.0,
    Omega_m=0.279,
    logNHImin=20.3,
    logNHImax=23.0,
    assume_logNHI=True,
    n_boot=0,
    rng=None,
    absorber_rest=LYA_REST,
    blue_limit_mode="global",
    blue_rest=LYB_REST,
    lambda_obs_min=None,
    lambda_obs_max=None,
):
    """
    Compute the DLA line density dN/dX in redshift bins.

    Line density is defined as:

        dN/dX(z) = N_abs(z) / ΔX_tot(z)

    where N_abs(z) is the number of absorbers in the z-bin and ΔX_tot(z) is
    the total comoving absorption distance across all QSO sightlines in that bin.

    All search-window parameters (``zmin``, ``v_prox_kms``, ``blue_limit_mode``,
    ``lambda_obs_min``, etc.) are passed through to ``build_qso_windows()``.

    Parameters
    ----------
    abs_cat : table-like
        Absorber catalog with columns: TARGETID, Z_DLA, NHI.
        NHI is log10(NHI/cm⁻²) if ``assume_logNHI=True``, else linear.
    qso_cat : table-like
        QSO sightline catalog with columns: TARGETID, Z.
    zbins : array-like
        Redshift bin edges for the output. Shape (nbins+1,).
    zmin : float or None
        Global lower floor on absorber redshift.
    zmax_global : float or None
        Global upper ceiling on absorber redshift.
    v_prox_kms : float
        Proximity-zone velocity cut for the red window edge [km/s].
    Omega_m : float
        Matter density parameter for LCDM path length (WMAP9: 0.279).
    logNHImin, logNHImax : float
        Column density range for absorber selection [log10 cm⁻²].
    assume_logNHI : bool
        If True, the NHI column in abs_cat is already log10-scaled.
    n_boot : int
        Number of bootstrap samples for error estimation (0 = skip bootstrap).
    rng : numpy Generator or None
        Random number generator for bootstrap. If None, uses default_rng().
    absorber_rest : float
        Rest wavelength defining absorber redshift (Lyα = 1215.67 Å for DLAs).
    blue_limit_mode : {"global", "lyb", "max"}
        Strategy for setting the blue edge of each QSO search window.
    blue_rest : float
        QSO blue-edge rest wavelength (Lyβ = 1025.72 Å).
    lambda_obs_min, lambda_obs_max : float or None
        Instrument observed wavelength limits [Å].

    Returns
    -------
    dict with keys:
        z_mid       : (nbins,)  Bin center redshifts.
        zbins       : (nbins+1,) Input bin edges.
        dndx        : (nbins,)  dN/dX per bin (NaN where ΔX = 0).
        err_poisson : (nbins,)  Poisson error = sqrt(N_abs) / ΔX.
        err_boot    : (nbins,) or None  Bootstrap std over QSO sightlines.
        N_abs       : (nbins,)  Raw absorber counts per bin.
        X_tot       : (nbins,)  Total absorption distance per bin.
        meta        : dict      All window/cosmology parameters used.
    """
    zbins = np.asarray(zbins, dtype=float)
    z_mid = 0.5 * (zbins[:-1] + zbins[1:])

    # QSO windows
    qso_tid, qso_zlo, qso_zhi = build_qso_windows(
        qso_cat,
        zmin=zmin,
        zmax_global=zmax_global,
        v_prox_kms=v_prox_kms,
        absorber_rest=absorber_rest,
        blue_limit_mode=blue_limit_mode,
        blue_rest=blue_rest,
        lambda_obs_min=lambda_obs_min,
        lambda_obs_max=lambda_obs_max,
    )

    if len(qso_tid) == 0:
        raise ValueError("No QSOs left after applying cuts; cannot compute dN/dX.")

    # Absorption distance
    Xcalc = AbsorptionDistance(zmax=float(np.max(qso_zhi)), Omega_m=Omega_m)

    # Total ΔX per z-bin
    X_tot = total_DeltaX_in_zbins(zbins, qso_zlo, qso_zhi, Xcalc)

    # Filter absorbers to selected QSOs + windows + logNHI cuts
    z_abs, logN, tid_abs, qso_idx_abs = filter_absorbers_to_qsos(
        abs_cat, qso_tid, qso_zlo, qso_zhi,
        logNHImin=logNHImin, logNHImax=logNHImax,
        assume_logNHI=assume_logNHI,
    )

    # Counts per z-bin
    N_abs, _ = np.histogram(z_abs, bins=zbins)

    dndx = np.where(X_tot > 0, N_abs / X_tot, np.nan)
    err_pois = np.where(X_tot > 0, np.sqrt(N_abs) / X_tot, np.nan)

    # Bootstrap over QSOs (sightlines)
    err_boot = None
    if n_boot and n_boot > 0:
        rng = np.random.default_rng() if rng is None else rng
        nq = len(qso_tid)
        nb = len(z_mid)

        zbin = np.digitize(z_abs, zbins) - 1
        valid = (zbin >= 0) & (zbin < nb)
        zbin = zbin[valid]
        qso_idx_abs = qso_idx_abs[valid]

        # per-QSO counts in each zbin
        per_qso_counts = np.zeros((nq, nb), dtype=int)
        np.add.at(per_qso_counts, (qso_idx_abs, zbin), 1)

        # per-QSO ΔX contributions in each zbin
        per_qso_X = np.zeros((nq, nb), dtype=float)
        for k in range(nb):
            lo, hi = zbins[k], zbins[k + 1]
            o_lo = np.maximum(qso_zlo, lo)
            o_hi = np.minimum(qso_zhi, hi)
            m = o_hi > o_lo
            if np.any(m):
                per_qso_X[m, k] = Xcalc.deltaX(o_lo[m], o_hi[m])

        boot = np.empty((n_boot, nb), dtype=float)
        for b in range(n_boot):
            draw = rng.integers(0, nq, size=nq)
            Nb = per_qso_counts[draw].sum(axis=0)
            Xb = per_qso_X[draw].sum(axis=0)
            boot[b] = np.where(Xb > 0, Nb / Xb, np.nan)

        err_boot = np.nanstd(boot, axis=0, ddof=1)

    return {
        "z_mid": z_mid,
        "zbins": zbins,
        "dndx": dndx,
        "err_poisson": err_pois,
        "err_boot": err_boot,
        "N_abs": N_abs,
        "X_tot": X_tot,
        "meta": {
            "logNHImin": float(logNHImin),
            "logNHImax": float(logNHImax),
            "Omega_m": float(Omega_m),
            "v_prox_kms": float(v_prox_kms),
            "absorber_rest": float(absorber_rest),
            "blue_limit_mode": str(blue_limit_mode),
            "blue_rest": float(blue_rest),
            "lambda_obs_min": None if lambda_obs_min is None else float(lambda_obs_min),
            "lambda_obs_max": None if lambda_obs_max is None else float(lambda_obs_max),
            "zmin": None if zmin is None else float(zmin),
            "zmax_global": None if zmax_global is None else float(zmax_global),
        },
    }


# ----------------------------
# 6) CDDF f(N): d^2N_abs / (dN dX)
# ----------------------------

def compute_cddf_fN(
    abs_cat, qso_cat,
    *,
    zbins,
    logN_bins,
    zmin=None,
    zmax_global=None,
    v_prox_kms=10000.0,
    Omega_m=0.279,
    logNHImin=None,
    logNHImax=None,
    assume_logNHI=True,
    n_boot=0,
    rng=None,
    absorber_rest=LYA_REST,
    blue_limit_mode="global",
    blue_rest=LYB_REST,
    lambda_obs_min=None,
    lambda_obs_max=None,
):
    """
    Compute CDDF in Bird+ convention:
        f(N) = d^2 N_abs / (dN dX)
    using log10N bins but dividing by ΔN (linear).

    Returns arrays:
      - z_mid (nbz,)
      - logN_mid (nbn,)
      - N_mid (nbn,) linear column density at bin center (geometric mean)
      - fN (nbz, nbn)
      - err_poisson (nbz, nbn)
      - err_boot (nbz, nbn) or None
    """
    zbins = np.asarray(zbins, dtype=float)
    logN_bins = np.asarray(logN_bins, dtype=float)

    z_mid = 0.5 * (zbins[:-1] + zbins[1:])
    nbz = len(z_mid)
    nbn = len(logN_bins) - 1

    # Linear bin edges and widths ΔN (cm^-2)
    N_edges = 10.0 ** logN_bins
    dN = np.diff(N_edges)

    # Geometric-mean bin center (standard for log bins)
    N_mid = np.sqrt(N_edges[:-1] * N_edges[1:])
    logN_mid = np.log10(N_mid)

    # QSO windows
    qso_tid, qso_zlo, qso_zhi = build_qso_windows(
        qso_cat,
        zmin=zmin,
        zmax_global=zmax_global,
        v_prox_kms=v_prox_kms,
        absorber_rest=absorber_rest,
        blue_limit_mode=blue_limit_mode,
        blue_rest=blue_rest,
        lambda_obs_min=lambda_obs_min,
        lambda_obs_max=lambda_obs_max,
    )

    if len(qso_tid) == 0:
        raise ValueError("No QSOs left after applying cuts; cannot compute CDDF.")

    # Absorption distance and ΔX per z-bin
    Xcalc = AbsorptionDistance(zmax=float(np.max(qso_zhi)), Omega_m=Omega_m)
    X_tot = total_DeltaX_in_zbins(zbins, qso_zlo, qso_zhi, Xcalc)

    # Filter absorbers to selected QSOs + windows (+ optional logN truncation)
    z_abs, logN, tid_abs, qso_idx_abs = filter_absorbers_to_qsos(
        abs_cat, qso_tid, qso_zlo, qso_zhi,
        logNHImin=logNHImin, logNHImax=logNHImax,
        assume_logNHI=assume_logNHI,
    )

    # 2D binning
    zbin = np.digitize(z_abs, zbins) - 1
    nbin = np.digitize(logN, logN_bins) - 1
    valid = (zbin >= 0) & (zbin < nbz) & (nbin >= 0) & (nbin < nbn)

    zbin = zbin[valid]
    nbin = nbin[valid]
    qso_idx_abs = qso_idx_abs[valid]

    counts = np.zeros((nbz, nbn), dtype=int)
    np.add.at(counts, (zbin, nbin), 1)

    # f(N) normalization: counts / (ΔX * ΔN)
    fN = np.full_like(counts, np.nan, dtype=float)
    err_pois = np.full_like(counts, np.nan, dtype=float)
    for k in range(nbz):
        if X_tot[k] > 0:
            fN[k] = counts[k] / (X_tot[k] * dN)
            err_pois[k] = np.sqrt(counts[k]) / (X_tot[k] * dN)

    # Bootstrap over QSOs (sightlines)
    err_boot = None
    if n_boot and n_boot > 0:
        rng = np.random.default_rng() if rng is None else rng
        nq = len(qso_tid)

        # per-QSO ΔX contributions in each zbin
        per_qso_X = np.zeros((nq, nbz), dtype=float)
        for k in range(nbz):
            lo, hi = zbins[k], zbins[k + 1]
            o_lo = np.maximum(qso_zlo, lo)
            o_hi = np.minimum(qso_zhi, hi)
            m = o_hi > o_lo
            if np.any(m):
                per_qso_X[m, k] = Xcalc.deltaX(o_lo[m], o_hi[m])

        # per-QSO 2D counts in (zbin, nbin)
        per_qso_counts = np.zeros((nq, nbz, nbn), dtype=int)
        np.add.at(per_qso_counts, (qso_idx_abs, zbin, nbin), 1)

        boot = np.empty((n_boot, nbz, nbn), dtype=float)
        for b in range(n_boot):
            draw = rng.integers(0, nq, size=nq)
            Cb = per_qso_counts[draw].sum(axis=0)
            Xb = per_qso_X[draw].sum(axis=0)

            fb = np.full((nbz, nbn), np.nan, dtype=float)
            for k in range(nbz):
                if Xb[k] > 0:
                    fb[k] = Cb[k] / (Xb[k] * dN)
            boot[b] = fb

        err_boot = np.nanstd(boot, axis=0, ddof=1)

    return {
        "z_mid": z_mid,
        "zbins": zbins,
        "logN_mid": logN_mid,
        "logN_bins": logN_bins,
        "N_mid": N_mid,
        "N_edges": N_edges,
        "dN": dN,
        "fN": fN,
        "err_poisson": err_pois,
        "err_boot": err_boot,
        "counts": counts,
        "X_tot": X_tot,
        "meta": {
            "Omega_m": float(Omega_m),
            "v_prox_kms": float(v_prox_kms),
            "logNHImin": None if logNHImin is None else float(logNHImin),
            "logNHImax": None if logNHImax is None else float(logNHImax),
            "absorber_rest": float(absorber_rest),
            "blue_limit_mode": str(blue_limit_mode),
            "blue_rest": float(blue_rest),
            "lambda_obs_min": None if lambda_obs_min is None else float(lambda_obs_min),
            "lambda_obs_max": None if lambda_obs_max is None else float(lambda_obs_max),
            "zmin": None if zmin is None else float(zmin),
            "zmax_global": None if zmax_global is None else float(zmax_global),
        },
    }


# ----------------------------
# 7) Optional: plotting utilities
# ----------------------------

def plot_dndx(out, *, label=None, prefer_boot=True, ax=None, show=True):
    """
    Plot dN/dX vs redshift from the output of ``compute_dndx()``.

    Parameters
    ----------
    out : dict
        Output dict from ``compute_dndx()``.
    label : str or None
        Legend label for this series.
    prefer_boot : bool
        If True, use bootstrap error bars when available; fall back to Poisson.
    ax : matplotlib Axes or None
        Axes to plot on; creates a new figure if None.
    show : bool
        If True, call plt.show() after plotting.

    Returns
    -------
    ax : matplotlib Axes
    """
    z = out["z_mid"]
    y = out["dndx"]
    yerr = out["err_boot"] if (prefer_boot and out.get("err_boot") is not None) else out["err_poisson"]

    zb = out["zbins"]
    xerr = 0.5 * (zb[1:] - zb[:-1])

    if ax is None:
        fig, ax = plt.subplots()

    ax.errorbar(z, y, yerr=yerr, xerr=xerr, fmt="o", capsize=2, label=label)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$dN/dX$")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    if show:
        plt.show()
    return ax


def plot_cddf_slice_fN(out, zbin_index, *, label=None, prefer_boot=True, ax=None, show=True, ylog=True):
    """
    Plot f(N) vs log10 N at a chosen z-bin index.
    """
    x = out["logN_mid"]
    y = out["fN"][zbin_index]
    yerr = out["err_boot"][zbin_index] if (prefer_boot and out.get("err_boot") is not None) else out["err_poisson"][zbin_index]

    if ax is None:
        fig, ax = plt.subplots()

    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=2, label=label)
    ax.set_xlabel(r"$\log_{10} N_{\rm HI}$")
    ax.set_ylabel(r"$f(N)=d^2N/(dN\,dX)$  [cm$^2$]")
    ax.grid(True, alpha=0.3)
    if ylog:
        ax.set_yscale("log")
    if label:
        ax.legend()
    if show:
        plt.show()
    return ax


# ----------------------------
# 8) Omega_HI from CDDF
# ----------------------------

def omega_hi_prefactor(H0_km_s_Mpc=70.0):
    """
    Return prefactor:
        K = H0 * m_H / (c * rho_c0)
    such that:
        Omega_HI(z) = K * ∫ N f(N,z) dN

    Units:
      - N in cm^-2
      - f(N) in cm^2
      - so N*f(N)*dN is dimensionless
      - K is dimensionless

    Uses cgs constants.
    """
    c_cms = 2.99792458e10
    mH_g  = 1.6735575e-24
    G_cgs = 6.67430e-8

    Mpc_cm = 3.085677581e24
    H0_s = (H0_km_s_Mpc * 1.0e5) / Mpc_cm

    rho_c0 = 3.0 * H0_s**2 / (8.0 * np.pi * G_cgs)

    K = (H0_s * mH_g) / (c_cms * rho_c0)
    return K


def omega_hi_from_cddf(
    out_cddf,
    *,
    zbin_index=None,
    zmin=None,
    zmax=None,
    logN_min=None,
    logN_max=None,
    H0_km_s_Mpc=70.0,
    prefer_boot=True,
):
    """
    Compute the neutral hydrogen mass density Omega_HI(z) from CDDF output.

    Integrates the CDDF over column density:

        Omega_HI(z) = K × Σ_j N_HI,j * f(N_HI,j, z) * ΔN_HI,j

    where K = H₀ m_H / (c ρ_c) (dimensionless, ~2.8×10⁻²⁸ for H₀=70).
    Error propagation assumes independence between N bins (Gaussian quadrature).

    Parameters
    ----------
    out_cddf : dict
        Output from ``compute_cddf_fN()``.
    zbin_index : int or None
        If set, compute Omega_HI only for this single z-bin.
    zmin, zmax : float or None
        Redshift range for selecting z-bins (ignored if zbin_index is set).
    logN_min, logN_max : float or None
        Column density integration limits [log10 cm⁻²].
    H0_km_s_Mpc : float
        Hubble constant in km/s/Mpc. Used in the prefactor K.
    prefer_boot : bool
        If True, propagate bootstrap errors; fall back to Poisson.

    Returns
    -------
    dict with keys:
        z           : (nz,)   Redshift bin centers selected.
        omega_hi    : (nz,)   Omega_HI(z).
        omega_hi_err: (nz,)   Propagated 1-sigma error on Omega_HI(z).
        meta        : dict    H0, logN range, error kind used.
    """
    z_mid = np.asarray(out_cddf["z_mid"], float)
    zbins = np.asarray(out_cddf["zbins"], float)
    N_edges = np.asarray(out_cddf["N_edges"], float)
    dN = np.asarray(out_cddf["dN"], float)
    fN = np.asarray(out_cddf["fN"], float)

    if prefer_boot and (out_cddf.get("err_boot") is not None):
        fN_err = np.asarray(out_cddf["err_boot"], float)
        err_kind = "boot"
    else:
        fN_err = np.asarray(out_cddf["err_poisson"], float)
        err_kind = "poisson"

    # select N-range
    logN_edges = np.log10(N_edges)
    n_lo = 0 if (logN_min is None) else int(np.searchsorted(logN_edges, logN_min, side="left"))
    n_hi = len(dN) if (logN_max is None) else int(np.searchsorted(logN_edges, logN_max, side="right") - 1)

    n_lo = max(0, min(n_lo, len(dN)))
    n_hi = max(0, min(n_hi, len(dN)))
    if n_hi <= n_lo:
        raise ValueError("Empty N range after applying logN_min/logN_max.")

    N_mid = np.sqrt(N_edges[:-1] * N_edges[1:])
    K = omega_hi_prefactor(H0_km_s_Mpc=H0_km_s_Mpc)

    # choose z bins
    if zbin_index is not None:
        z_sel = np.array([z_mid[int(zbin_index)]])
        f_sel = fN[int(zbin_index)][None, :]
        e_sel = fN_err[int(zbin_index)][None, :]
    else:
        m = np.ones_like(z_mid, dtype=bool)
        if (zmin is not None) or (zmax is not None):
            if zmin is not None:
                m &= (z_mid >= float(zmin))
            if zmax is not None:
                m &= (z_mid <= float(zmax))
        z_sel = z_mid[m]
        f_sel = fN[m]
        e_sel = fN_err[m]
        if len(z_sel) == 0:
            raise ValueError("No z-bins selected for Omega_HI.")

    Nm = N_mid[n_lo:n_hi]
    dNj = dN[n_lo:n_hi]

    omega = K * np.sum((Nm * f_sel[:, n_lo:n_hi]) * dNj[None, :], axis=1)

    # approximate error propagation assuming independence between N bins
    omega_err = K * np.sqrt(np.sum(((Nm * e_sel[:, n_lo:n_hi]) * dNj[None, :]) ** 2, axis=1))

    return {
        "z": z_sel,
        "omega_hi": omega,
        "omega_hi_err": omega_err,
        "meta": {
            "H0_km_s_Mpc": float(H0_km_s_Mpc),
            "err_kind": err_kind,
            "logN_min": None if logN_min is None else float(logN_min),
            "logN_max": None if logN_max is None else float(logN_max),
        },
    }


def plot_omega_hi(out_omega, *, ax=None, label=None, show=True):
    """
    Plot Omega_HI vs redshift from the output of ``omega_hi_from_cddf()``.

    The y-axis is scaled by 1000 (i.e., plotted as 10⁻³ Omega_HI) for readability.

    Parameters
    ----------
    out_omega : dict
        Output dict from ``omega_hi_from_cddf()``.
    ax : matplotlib Axes or None
        Axes to plot on; creates a new figure if None.
    label : str or None
        Legend label.
    show : bool
        If True, call plt.show().

    Returns
    -------
    ax : matplotlib Axes
    """
    z = out_omega["z"]

    # scale only for plotting
    y = 1e3 * out_omega["omega_hi"]
    yerr = 1e3 * out_omega["omega_hi_err"]

    if ax is None:
        fig, ax = plt.subplots()

    ax.errorbar(z, y, yerr=yerr, fmt="o", capsize=2, label=label)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$10^{-3}\,\Omega_{\rm HI}$")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    if show:
        plt.show()
    return ax


# ----------------------------
# 9) Prochaska+2014 truth CDDF and mock calibration
# ----------------------------

def truth_cddf_prochaska2014(logN):
    """
    Evaluate the Prochaska+2014 wide-logN CDDF spline at log10(N) values.

    This is the literature "truth" CDDF used as the calibration reference
    when comparing GP-DLA measured statistics to mock truth.  The spline
    is a piecewise cubic Hermite (PchipInterpolator) fit to nodes spanning
    logN ∈ [12, 22], covering LLS, sub-DLA, and DLA regimes.

    Source: Prochaska, Worseck & O'Meara (2009), ApJL 705, L113;
    spline nodes as used in the DESI Y3 GP-DLA calibration notebooks.

    Parameters
    ----------
    logN : array-like
        log10(N_HI / cm⁻²) values at which to evaluate the CDDF.

    Returns
    -------
    log10_fN : ndarray
        log10 of f(N) [cm²] at the requested logN values.
        Values outside [12, 22] are clipped to the spline boundary.

    Notes
    -----
    The returned quantity is log10 f(N), not f(N) itself.
    To get f(N): ``fN = 10 ** truth_cddf_prochaska2014(logN)``.
    """
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        raise ImportError("scipy is required for truth_cddf_prochaska2014().")

    # Spline nodes from Prochaska+2014 / DESI Y3 calibration notebooks
    _logN_nodes = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    _logf_nodes = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    _spline = PchipInterpolator(_logN_nodes, _logf_nodes)

    logN = np.asarray(logN, dtype=float)
    logN_clip = np.clip(logN, _logN_nodes[0], _logN_nodes[-1])
    return _spline(logN_clip)


def truth_dndx_prochaska2014(logNHImin, logNHImax, n_points=1000):
    """
    Compute the truth dN/dX by numerically integrating the Prochaska+2014
    CDDF spline over a logN range.

    dN/dX = ∫_{logNHImin}^{logNHImax} f(N) dN

    Parameters
    ----------
    logNHImin, logNHImax : float
        Integration limits in log10(N_HI / cm⁻²).
    n_points : int
        Number of quadrature points for trapezoidal integration.

    Returns
    -------
    dndx_truth : float
        Integrated line density (scalar, dimensionless).
    """
    logN_grid = np.linspace(logNHImin, logNHImax, int(n_points))
    N_grid = 10.0 ** logN_grid                                    # cm⁻²
    fN_grid = 10.0 ** truth_cddf_prochaska2014(logN_grid)        # cm²

    # f(N) dN in linear N: dN = N * ln(10) * d(logN)
    dlogN = logN_grid[1] - logN_grid[0]
    dndx_truth = np.trapz(fN_grid * N_grid * np.log(10), logN_grid) * dlogN / dlogN
    # Simpler: trapz over logN with integrand = f(N) * N * ln(10)
    dndx_truth = np.trapz(fN_grid * N_grid * np.log(10.0), logN_grid)
    return float(dndx_truth)


def compute_calibration_alpha(out_truth, out_measured_mock, *, kind="linear"):
    """
    Compute the calibration factor alpha(z) = dNdX_measured_mock / dNdX_truth.

    This factor corrects for detection incompleteness and false positives:
    the GP-DLA pipeline measured on London mock spectra is compared to the
    known truth dN/dX to estimate the redshift-dependent bias alpha(z).

    Parameters
    ----------
    out_truth : dict
        Output of ``compute_dndx()`` on truth absorbers (or a dict with
        keys ``z_mid``, ``dndx``, ``err_poisson``, and optionally ``err_boot``).
    out_measured_mock : dict
        Output of ``compute_dndx()`` on GP-DLA-detected mock absorbers.
        Must be on the same z grid as ``out_truth``.
    kind : str
        Interpolation kind for scipy.interpolate.interp1d ('linear', 'cubic', etc.).
        Only used if the z grids differ; for same-grid inputs, no interpolation.

    Returns
    -------
    dict with keys:
        z       : (n,)  Redshift bin centers (from out_truth).
        alpha   : (n,)  alpha(z) = measured / truth.
        alpha_err : (n,)  Propagated 1-sigma error on alpha(z).

    Notes
    -----
    Relative error propagation:
        sigma_alpha / alpha = sqrt( (sigma_meas/meas)^2 + (sigma_truth/truth)^2 )
    """
    z_truth = np.asarray(out_truth["z_mid"], dtype=float)
    y_truth = np.asarray(out_truth["dndx"], dtype=float)
    e_truth = (
        np.asarray(out_truth["err_boot"], dtype=float)
        if out_truth.get("err_boot") is not None
        else np.asarray(out_truth["err_poisson"], dtype=float)
    )

    z_meas = np.asarray(out_measured_mock["z_mid"], dtype=float)
    y_meas = np.asarray(out_measured_mock["dndx"], dtype=float)
    e_meas = (
        np.asarray(out_measured_mock["err_boot"], dtype=float)
        if out_measured_mock.get("err_boot") is not None
        else np.asarray(out_measured_mock["err_poisson"], dtype=float)
    )

    # Interpolate measured mock to truth z grid if needed
    if not np.allclose(z_truth, z_meas, atol=1e-6):
        from scipy.interpolate import interp1d
        y_meas = interp1d(z_meas, y_meas, kind=kind, bounds_error=False, fill_value=np.nan)(z_truth)
        e_meas = interp1d(z_meas, e_meas, kind=kind, bounds_error=False, fill_value=np.nan)(z_truth)

    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.where(y_truth > 0, y_meas / y_truth, np.nan)
        rel_err = np.sqrt(
            np.where(y_meas > 0, (e_meas / y_meas) ** 2, 0.0)
            + np.where(y_truth > 0, (e_truth / y_truth) ** 2, 0.0)
        )
        alpha_err = alpha * rel_err

    return {"z": z_truth, "alpha": alpha, "alpha_err": alpha_err}


def apply_calibration(out_real, out_calibration):
    """
    Apply the calibration factor alpha(z) to real GP-DLA dN/dX measurements.

    Calibrated result:
        dNdX_calibrated = alpha(z) * dNdX_real

    Error propagation (assuming alpha and dNdX_real are independent):
        err_calibrated = sqrt( (alpha * err_real)^2 + (dNdX_real * err_alpha)^2 )

    Parameters
    ----------
    out_real : dict
        Output of ``compute_dndx()`` on real DESI data.
    out_calibration : dict
        Output of ``compute_calibration_alpha()``, with keys z, alpha, alpha_err.
        The z grid of the calibration will be interpolated to match out_real["z_mid"].

    Returns
    -------
    dict with keys:
        z               : (n,)  Redshift bin centers from out_real.
        dndx_raw        : (n,)  Uncalibrated dN/dX.
        dndx_calibrated : (n,)  Calibrated dN/dX.
        err_raw         : (n,)  Error on raw dN/dX (bootstrap if available).
        err_calibrated  : (n,)  Propagated error on calibrated dN/dX.
        alpha           : (n,)  Alpha(z) values used.
        alpha_err       : (n,)  Error on alpha(z) used.
    """
    z_real = np.asarray(out_real["z_mid"], dtype=float)
    y_real = np.asarray(out_real["dndx"], dtype=float)
    e_real = (
        np.asarray(out_real["err_boot"], dtype=float)
        if out_real.get("err_boot") is not None
        else np.asarray(out_real["err_poisson"], dtype=float)
    )

    z_cal = np.asarray(out_calibration["z"], dtype=float)
    alpha = np.asarray(out_calibration["alpha"], dtype=float)
    alpha_err = np.asarray(out_calibration["alpha_err"], dtype=float)

    # Interpolate calibration to real data z grid
    if not np.allclose(z_real, z_cal, atol=1e-6):
        from scipy.interpolate import interp1d
        alpha = interp1d(z_cal, alpha, kind="linear", bounds_error=False, fill_value=np.nan)(z_real)
        alpha_err = interp1d(z_cal, alpha_err, kind="linear", bounds_error=False, fill_value=np.nan)(z_real)

    dndx_cal = alpha * y_real
    err_cal = np.sqrt((alpha * e_real) ** 2 + (y_real * alpha_err) ** 2)

    return {
        "z": z_real,
        "dndx_raw": y_real,
        "dndx_calibrated": dndx_cal,
        "err_raw": e_real,
        "err_calibrated": err_cal,
        "alpha": alpha,
        "alpha_err": alpha_err,
    }


def zbins_from_zmid_uniform(z_mid, pad=True):
    """
    Reconstruct uniform bin edges from bin center positions.

    Assumes uniformly spaced centers: zbins[k] = z_mid[k] - dz/2.

    Parameters
    ----------
    z_mid : array-like
        Bin centers, assumed to be uniformly spaced.
    pad : bool
        If True, extend the last edge by half a bin width (default True).

    Returns
    -------
    zbins : array of shape (len(z_mid)+1,)
        Bin edges.
    """
    z_mid = np.asarray(z_mid, dtype=float)
    dz = np.diff(z_mid)
    if not np.allclose(dz, dz[0], rtol=1e-4):
        raise ValueError("z_mid does not appear to be uniformly spaced.")
    hw = dz[0] / 2.0
    zbins = np.empty(len(z_mid) + 1)
    zbins[:-1] = z_mid - hw
    zbins[-1] = z_mid[-1] + hw
    return zbins


# ----------------------------
# 10) Example usage
# ----------------------------
#
# Example DESI-style DLA/LLS search window:
#
# out_lls = compute_cddf_fN(
#     abs_cat,
#     qso_cat,
#     zbins=np.array([2.0, 2.5, 3.0, 3.5, 4.0]),
#     logN_bins=np.array([17.2, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0]),
#     zmin=None,
#     zmax_global=None,
#     v_prox_kms=3000.0,
#     Omega_m=0.279,
#     logNHImin=17.2,
#     logNHImax=20.3,
#     assume_logNHI=True,
#     n_boot=200,
#     absorber_rest=LYA_REST,       # z_DLA defined using Lyα
#     blue_limit_mode="max",        # use max of {zmin, lambda_obs_min, Lyβ edge}
#     blue_rest=LYB_REST,           # QSO blue edge is Lyβ
#     lambda_obs_min=3600.0,        # DESI blue cutoff
#     lambda_obs_max=None,
# )
#
# out_dla_dndx = compute_dndx(
#     abs_cat,
#     qso_cat,
#     zbins=np.array([2.0, 2.5, 3.0, 3.5, 4.0]),
#     zmin=None,
#     zmax_global=None,
#     v_prox_kms=3000.0,
#     Omega_m=0.279,
#     logNHImin=20.3,
#     logNHImax=23.0,
#     assume_logNHI=True,
#     n_boot=200,
#     absorber_rest=LYA_REST,
#     blue_limit_mode="max",
#     blue_rest=LYB_REST,
#     lambda_obs_min=3600.0,
#     lambda_obs_max=None,
# )