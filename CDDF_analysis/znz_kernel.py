"""znz_kernel.py — Track-C Stage-0b: measure + fit the (xhat, z) bias/scatter model
and the (N, z) completeness model from a truth-matched mock catalog.

Used by Stage-1 to build the 2-D posterior kernel that replaces the frozen
broaden012 kernel with a properly prior-edge-corrected version.

Interfaces
----------
ZNZModel   : b(xhat, z) bias + sigma(xhat, z) scatter polynomial model
CNZModel   : g(j_nhi_cell, kz) completeness model (smooth monotone)

Functions
---------
measure_znz_response   measure (xhat, z, dx) from a truth-matched cat_cut
fit_znz_model          2-D polynomial fit -> ZNZModel
measure_c_nz           count-ratio completeness grid from cat_cut + truth_cut
fit_c_nz_model         smooth + normalize -> CNZModel
save_znz / load_znz    NPZ serialization for both dataclasses
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.polynomial.polynomial import polyvander2d
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ZNZModel:
    """Polynomial model for conditional bias b(xhat, z) and scatter sigma(xhat, z).

    The polynomial is centred at (xhat_ref, z_ref) for numerical stability:
        b(xhat, z)     = polyvander2d(xhat - xhat_ref, z - z_ref, [deg_xhat, deg_z]) @ b_coef
        sigma(xhat, z) = clip(polyvander2d(...) @ sig_coef, 1e-4, inf)

    Attributes
    ----------
    b_coef : shape ((deg_xhat+1)*(deg_z+1),)
        Flat coefficient array for the bias surface.
    sig_coef : shape ((deg_xhat+1)*(deg_z+1),)
        Flat coefficient array for the scatter surface.
    xhat_ref : float
        Reference xhat (median of training set).
    z_ref : float
        Reference z (median of training set).
    b_ref : float
        b(xhat_ref, z_ref) evaluated at the reference point.
    sig_ref : float
        sigma(xhat_ref, z_ref).
    z_covariate : str
        Column used as z; "z_dla" (Phase 1).
    """
    b_coef: np.ndarray
    sig_coef: np.ndarray
    xhat_ref: float
    z_ref: float
    b_ref: float
    sig_ref: float
    z_covariate: str

    def _design(self, xhat: np.ndarray, z: np.ndarray) -> np.ndarray:
        xhat = np.asarray(xhat, float).ravel()
        z = np.asarray(z, float).ravel()
        deg_xhat = int(round(np.sqrt(len(self.b_coef)) - 1))
        deg_z = int(round(len(self.b_coef) / (deg_xhat + 1))) - 1
        return polyvander2d(xhat - self.xhat_ref, z - self.z_ref,
                            [deg_xhat, deg_z])

    def b(self, xhat: np.ndarray, z: np.ndarray) -> np.ndarray:
        """E[xhat - xtrue | xhat, z] at given (xhat, z) points."""
        return self._design(xhat, z) @ self.b_coef

    def sigma(self, xhat: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Conditional scatter (> 0) at given (xhat, z) points."""
        return np.clip(self._design(xhat, z) @ self.sig_coef, 1e-4, None)


@dataclass
class CNZModel:
    """Smooth (N, z) completeness model: g(N-cell index j, z-bin index kz).

    g_grid[j, kz] is normalised so that g(j, z_ref_col) = 1 at the reference
    z column (closest to median z).  Values are in (0, ~2].

    Attributes
    ----------
    g_grid : shape (n_nhi_cell, n_zf)
        Smoothed completeness grid, normalised at z_ref.
    nhi_edges : shape (n_nhi_cell + 1,)
        NHI cell edges (from MollyMatrix).
    z_edges_fine : shape (n_zf + 1,)
        Fine z-bin edges (from _fine_z_grid).
    """
    g_grid: np.ndarray
    nhi_edges: np.ndarray
    z_edges_fine: np.ndarray

    def g(self, j_nhi_cell: int, kz: int) -> float:
        """Completeness at NHI cell j and fine z-bin kz. Returns float in (0, ~2]."""
        j = int(np.clip(j_nhi_cell, 0, self.g_grid.shape[0] - 1))
        k = int(np.clip(kz, 0, self.g_grid.shape[1] - 1))
        return float(self.g_grid[j, k])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deg_from_coef(coef: np.ndarray, deg_xhat: int) -> int:
    """Recover deg_z from flat coef length and known deg_xhat."""
    n_total = len(coef)
    return n_total // (deg_xhat + 1) - 1


def _poly_fit_2d(x: np.ndarray, z: np.ndarray, y: np.ndarray,
                 x_ref: float, z_ref: float,
                 deg_x: int, deg_z: int) -> np.ndarray:
    """Least-squares 2-D polynomial fit of y ~ poly(x-x_ref, z-z_ref).

    Returns flat coefficient array of shape ((deg_x+1)*(deg_z+1),).
    """
    V = polyvander2d(x - x_ref, z - z_ref, [deg_x, deg_z])
    coef, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
    return coef


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------

def measure_znz_response(cat_cut, good_mask, cfg, mm, fine_grid,
                         z_covariate: str = "z_dla",
                         host_col: str = "NHI_TILT_HOST") -> dict:
    """Measure per-detection (xhat, z, dx) arrays from a truth-matched catalog.

    Parameters
    ----------
    cat_cut : astropy Table
        Output of load_and_cut_catalog — carries NHI, Z_DLA, S2N_RED, P_DLA,
        NHI_TRUE / NHI_TILT_HOST.
    good_mask : np.ndarray[bool]
        Per-row good-geometry mask (already on cat_cut).
    cfg : HBIConfig
        Pipeline config (snr_min, p_dla_min).
    mm : MollyMatrix
        Molly matrix (not used here directly; reserved for future SNR cell logic).
    fine_grid : tuple
        (logN_lo, logN_hi, N_b, dN_b) from build_fine_grid(cfg).
    z_covariate : str
        Which z to use; "z_dla" maps to "Z_DLA" column.
    host_col : str
        Column name carrying the true NHI of the matched host absorber.

    Returns
    -------
    dict with keys: "xhat", "z", "dx", "z_covariate"
        All arrays are float64 of the same length (TPs only).
    """
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask

    # predicted NHI (xhat) and matched true NHI (xtrue)
    xhat_pred = np.asarray(cat_cut["NHI"], float)[op]
    true_col = host_col if host_col in cat_cut.colnames else "NHI_TRUE"
    xtrue = np.asarray(cat_cut[true_col], float)[op]

    # z covariate
    z_col_map = {"z_dla": "Z_DLA", "z_qso": "Z_QSO"}
    z_col = z_col_map.get(z_covariate, z_covariate.upper())
    z_all = np.asarray(cat_cut[z_col], float)[op]

    # TPs only: finite true NHI
    tp = np.isfinite(xtrue)
    xhat_tp = xhat_pred[tp]
    xtrue_tp = xtrue[tp]
    z_tp = z_all[tp]
    dx = xhat_tp - xtrue_tp

    return {"xhat": xhat_tp, "z": z_tp, "dx": dx, "z_covariate": z_covariate}


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_znz_model(meas: dict, deg_z: int = 2, deg_xhat: int = 1) -> ZNZModel:
    """Fit a 2-D polynomial model for bias b(xhat, z) and scatter sigma(xhat, z).

    Parameters
    ----------
    meas : dict
        Output of measure_znz_response (or hand-constructed test dict) with keys:
        "xhat", "z", "dx", "z_covariate".
    deg_z : int
        Polynomial degree in z (default 2).
    deg_xhat : int
        Polynomial degree in xhat (default 1).

    Returns
    -------
    ZNZModel
    """
    xhat = np.asarray(meas["xhat"], float)
    z = np.asarray(meas["z"], float)
    dx = np.asarray(meas["dx"], float)
    z_covariate = meas.get("z_covariate", "z_dla")

    xhat_ref = float(np.median(xhat))
    z_ref = float(np.median(z))

    # --- fit bias surface b(xhat, z) ---
    b_coef = _poly_fit_2d(xhat, z, dx, xhat_ref, z_ref, deg_xhat, deg_z)

    # --- fit scatter surface: |dx - b_pred| ---
    V = polyvander2d(xhat - xhat_ref, z - z_ref, [deg_xhat, deg_z])
    b_pred = V @ b_coef
    abs_resid = np.abs(dx - b_pred)
    sig_coef = _poly_fit_2d(xhat, z, abs_resid, xhat_ref, z_ref, deg_xhat, deg_z)

    # evaluate at reference point
    V_ref = polyvander2d(np.array([0.0]), np.array([0.0]), [deg_xhat, deg_z])
    b_ref = float((V_ref @ b_coef)[0])
    sig_ref = float(np.clip((V_ref @ sig_coef)[0], 1e-4, None))

    return ZNZModel(
        b_coef=b_coef, sig_coef=sig_coef,
        xhat_ref=xhat_ref, z_ref=z_ref,
        b_ref=b_ref, sig_ref=sig_ref,
        z_covariate=z_covariate,
    )


# ---------------------------------------------------------------------------
# Completeness model
# ---------------------------------------------------------------------------

def measure_c_nz(cat_cut, truth_cut, cfg, mm, z_edges_fine: np.ndarray) -> dict:
    """Measure empirical completeness grid g_raw[j_nhi, kz] = n_rec / n_true.

    Parameters
    ----------
    cat_cut : astropy Table
        GP catalog (truth-matched; carries NHI_TRUE, Z_DLA, S2N_RED, P_DLA).
    truth_cut : astropy Table
        Truth absorber catalog (carries NHI and Z_DLA / Z_DLA_NO_RSD / Z).
    cfg : HBIConfig
        Pipeline config (snr_min, p_dla_min, zbins).
    mm : MollyMatrix
        Molly matrix — provides nhi_edges.
    z_edges_fine : np.ndarray
        Fine z-bin edges from _fine_z_grid(cfg).

    Returns
    -------
    dict with keys: "g_raw", "n_true", "n_rec", "nhi_edges", "z_edges_fine"
    """
    nhi_edges = mm.nhi_edges
    n_nhi = len(nhi_edges) - 1
    n_zf = len(z_edges_fine) - 1

    # --- truth side: count true absorbers per (nhi-cell, z-bin) ---
    t_nhi = np.asarray(truth_cut["NHI"], float)
    z_col = next((c for c in ("Z_DLA", "Z_DLA_NO_RSD", "Z") if c in truth_cut.colnames), None)
    t_z = np.asarray(truth_cut[z_col], float) if z_col else np.zeros(len(truth_cut))

    j_true = np.searchsorted(nhi_edges, t_nhi, side="right") - 1
    k_true = np.searchsorted(z_edges_fine, t_z, side="right") - 1
    j_true = np.clip(j_true, 0, n_nhi - 1)
    k_true = np.clip(k_true, 0, n_zf - 1)

    # only count truth in the z range of the fine grid
    in_zrange = (t_z >= z_edges_fine[0]) & (t_z < z_edges_fine[-1])
    n_true = np.zeros((n_nhi, n_zf), dtype=float)
    for ii in range(len(t_nhi)):
        if in_zrange[ii]:
            n_true[j_true[ii], k_true[ii]] += 1.0

    # --- detected side: recovered TPs among the operating set ---
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    # good_mask not passed here — use a permissive mask based on operational cuts
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min)

    # true NHI of matched TPs
    nhi_true_all = np.asarray(cat_cut["NHI_TRUE"], float)
    z_dla_col = next((c for c in ("Z_DLA", "Z_QSO") if c in cat_cut.colnames), None)
    z_cat = np.asarray(cat_cut[z_dla_col], float) if z_dla_col else np.zeros(len(cat_cut))

    tp_op = op & np.isfinite(nhi_true_all)

    j_rec = np.searchsorted(nhi_edges, nhi_true_all[tp_op], side="right") - 1
    k_rec = np.searchsorted(z_edges_fine, z_cat[tp_op], side="right") - 1
    j_rec = np.clip(j_rec, 0, n_nhi - 1)
    k_rec = np.clip(k_rec, 0, n_zf - 1)

    in_zrange_cat = (z_cat[tp_op] >= z_edges_fine[0]) & (z_cat[tp_op] < z_edges_fine[-1])
    n_rec = np.zeros((n_nhi, n_zf), dtype=float)
    for ii in range(int(np.sum(tp_op))):
        if in_zrange_cat[ii]:
            n_rec[j_rec[ii], k_rec[ii]] += 1.0

    # --- raw completeness ratio ---
    with np.errstate(invalid="ignore", divide="ignore"):
        g_raw = np.where(n_true > 0, n_rec / n_true, np.nan)

    return {
        "g_raw": g_raw,
        "n_true": n_true,
        "n_rec": n_rec,
        "nhi_edges": nhi_edges,
        "z_edges_fine": z_edges_fine,
    }


def fit_c_nz_model(meas_c: dict, smooth: float = 1.0) -> CNZModel:
    """Smooth and normalise the raw completeness grid to produce CNZModel.

    Parameters
    ----------
    meas_c : dict
        Output of measure_c_nz.
    smooth : float
        Gaussian smoothing sigma in z-bin pixels (applied along z axis).

    Returns
    -------
    CNZModel
    """
    g_raw = np.asarray(meas_c["g_raw"], float)
    nhi_edges = np.asarray(meas_c["nhi_edges"], float)
    z_edges_fine = np.asarray(meas_c["z_edges_fine"], float)

    n_nhi, n_zf = g_raw.shape

    # fill NaN cells with row median (or 1.0 if entire row is NaN)
    g_filled = g_raw.copy()
    for j in range(n_nhi):
        row = g_raw[j]
        valid = row[np.isfinite(row)]
        fill = float(np.median(valid)) if len(valid) > 0 else 1.0
        g_filled[j, ~np.isfinite(row)] = fill

    # smooth along z axis
    if smooth > 0:
        g_smooth = gaussian_filter1d(g_filled, sigma=smooth, axis=1,
                                     mode="nearest")
    else:
        g_smooth = g_filled.copy()

    # normalise each row so that g at the reference z column = 1
    # reference column = index of z closest to the median of z_edges midpoints
    z_mids = 0.5 * (z_edges_fine[:-1] + z_edges_fine[1:])
    z_ref = float(np.median(z_mids))
    kz_ref = int(np.argmin(np.abs(z_mids - z_ref)))

    norms = g_smooth[:, kz_ref].copy()
    norms[norms <= 0] = 1.0  # guard against zero
    g_norm = g_smooth / norms[:, np.newaxis]

    # safety clip
    g_norm = np.clip(g_norm, 0.01, 10.0)

    return CNZModel(g_grid=g_norm, nhi_edges=nhi_edges, z_edges_fine=z_edges_fine)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_znz(path: str, znz: ZNZModel, cnz: CNZModel) -> None:
    """Save both models to a single NPZ file.

    Keys: b_coef, sig_coef, xhat_ref, z_ref, b_ref, sig_ref, z_covariate,
          g_grid, nhi_edges, z_edges_fine
    """
    np.savez(
        path,
        b_coef=znz.b_coef,
        sig_coef=znz.sig_coef,
        xhat_ref=np.array(znz.xhat_ref),
        z_ref=np.array(znz.z_ref),
        b_ref=np.array(znz.b_ref),
        sig_ref=np.array(znz.sig_ref),
        z_covariate=np.array(znz.z_covariate),
        g_grid=cnz.g_grid,
        nhi_edges=cnz.nhi_edges,
        z_edges_fine=cnz.z_edges_fine,
    )


def load_znz(path: str) -> tuple:
    """Load (ZNZModel, CNZModel) from a NPZ file written by save_znz.

    Returns
    -------
    (ZNZModel, CNZModel)
    """
    d = np.load(path, allow_pickle=True)
    znz = ZNZModel(
        b_coef=d["b_coef"],
        sig_coef=d["sig_coef"],
        xhat_ref=float(d["xhat_ref"]),
        z_ref=float(d["z_ref"]),
        b_ref=float(d["b_ref"]),
        sig_ref=float(d["sig_ref"]),
        z_covariate=str(d["z_covariate"]),
    )
    cnz = CNZModel(
        g_grid=d["g_grid"],
        nhi_edges=d["nhi_edges"],
        z_edges_fine=d["z_edges_fine"],
    )
    return znz, cnz
