# ============================================================
# DLA/LLS/subDLA summary statistics: dN/dX and CDDF
#
# Correct CDDF convention (Bird+ 2016 / arXiv:1610.01165):
#   f(N) = d^2 N_abs / (dN dX)
# where N is LINEAR column density in cm^-2, and X is dimensionless
#
# This file computes:
#   - dN/dX in redshift bins (for any logNHI range)
#   - CDDF f(N) in (z-bin, logN-bin) grids, where the binning is in log10 N
#     but normalization divides by ΔN = 10^{logN_hi} - 10^{logN_lo}
#
# Search-window logic:
#   - absorber redshift z_abs is defined using absorber_rest (default Lyα 1215.67 Å)
#   - the blue edge of the search window can be set by:
#       * a global zmin
#       * instrument lambda_obs_min converted to z_abs
#       * the QSO Lyβ edge converted into absorber redshift
#   - the red edge of the search window is set by:
#       * proximity-zone cut
#       * optional lambda_obs_max converted to z_abs
#       * optional global zmax_global
#
# Typical DLA-style use:
#   - absorber_rest = 1215.67  (Lyα)
#   - blue_rest     = 1025.72  (Lyβ)
#   - blue_limit_mode = "max"
#   - lambda_obs_min = 3600.0
#   - v_prox_kms = 3000.0
#
# Assumptions:
#   - QSO catalog contains TARGETID, Z
#   - absorber catalog contains TARGETID, Z_DLA, NHI
#   - NHI column is log10(NHI/cm^2) if assume_logNHI=True, else linear NHI
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
    ΔX_k = sum_i ∫_{W_i ∩ [z_k,z_{k+1}]} dX
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
    dN/dX in z-bins for absorbers with logNHI in [logNHImin, logNHImax].

    Search-window settings are passed through to build_qso_windows().
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
    Compute Omega_HI(z) from CDDF output of compute_cddf_fN().
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
# 9) Example usage
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