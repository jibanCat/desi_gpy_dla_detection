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
build_cache            CLI entrypoint to reproducibly build the stage-0 NPZ cache

Note on b(xhat, z):
    b fits the MEAN of the dx = xhat - xtrue distribution (right-skewed due to
    the prior-edge pile-up at log N_HI ~ 20.3).  b RISES with both xhat and z —
    larger x̂ sits closer to the prior edge (more up-migration) and higher z has
    denser forest (more blending pushes absorbers toward the edge).
    Do NOT interpret b(20.5) > b(21.0) — the measured direction is the opposite:
    b increases monotonically with xhat and with z.

Note on g(j_nhi_cell, kz):
    g lives on the molly nhi_edges grid whose top edge is +inf.  Stage-1 must
    map g onto the fine-N axis and must NOT index the +inf top cell for any
    finite N value.  g is smaller than the (N,z) kernel shift but non-negligible:
    it must be carried, not dropped.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
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
    deg_xhat : int
        Polynomial degree in xhat dimension (stored for robust _design recovery).
    deg_z : int
        Polynomial degree in z dimension (stored for robust _design recovery).
    """
    b_coef: np.ndarray
    sig_coef: np.ndarray
    xhat_ref: float
    z_ref: float
    b_ref: float
    sig_ref: float
    z_covariate: str
    deg_xhat: int = 1
    deg_z: int = 2

    def _design(self, xhat: np.ndarray, z: np.ndarray) -> np.ndarray:
        xhat = np.asarray(xhat, float).ravel()
        z = np.asarray(z, float).ravel()
        # Use stored degrees — robust for any (deg_xhat, deg_z) combination.
        # The old sqrt(len(b_coef))-1 formula only worked for perfect squares.
        return polyvander2d(xhat - self.xhat_ref, z - self.z_ref,
                            [self.deg_xhat, self.deg_z])

    def b(self, xhat: np.ndarray, z: np.ndarray) -> np.ndarray:
        """E[xhat - xtrue | xhat, z] at given (xhat, z) points.

        b is the mean of a right-skewed dx distribution driven by the prior-edge
        pile-up at log N_HI ~ 20.3.  b RISES with xhat (closer to prior edge →
        more up-migration) and RISES with z (denser forest → more blending).
        """
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
    """Recover deg_z from flat coef length and known deg_xhat.

    Used as a cross-check when loading old NPZ files that pre-date the stored-
    degree fields.  ZNZModel now stores deg_xhat/deg_z directly; _design uses
    them, not this function.
    """
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
        deg_xhat=deg_xhat, deg_z=deg_z,
    )


# ---------------------------------------------------------------------------
# Completeness model
# ---------------------------------------------------------------------------

def measure_c_nz(cat_cut, truth_cut, cfg, mm, z_edges_fine: np.ndarray,
                 good_mask: Optional[np.ndarray] = None) -> dict:
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
    good_mask : np.ndarray[bool] or None
        Per-row good-geometry mask (same as passed to measure_znz_response).
        Must be included to make the op-set IDENTICAL to the b-measurement;
        if None, a permissive all-True mask is used (backward-compat only).

    Returns
    -------
    dict with keys: "g_raw", "n_true", "n_rec", "nhi_edges", "z_edges_fine"

    Note on g:
        g lives on the molly nhi_edges grid whose top edge is +inf.  Stage-1
        must map g onto the fine-N axis and must NOT index the +inf top cell for
        any finite N value.  g is smaller than the (N,z) kernel shift but
        non-negligible — it must be carried, not dropped.
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
    # good_mask must match the b-measurement's op-set exactly (same as measure_znz_response).
    # If not provided, fall back to all-True (backward-compat only — prefer passing it).
    if good_mask is None:
        good_mask = np.ones(len(cat_cut), dtype=bool)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask

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
          deg_xhat, deg_z,
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
        deg_xhat=np.array(znz.deg_xhat),
        deg_z=np.array(znz.deg_z),
        g_grid=cnz.g_grid,
        nhi_edges=cnz.nhi_edges,
        z_edges_fine=cnz.z_edges_fine,
    )


def load_znz(path: str) -> tuple:
    """Load (ZNZModel, CNZModel) from a NPZ file written by save_znz.

    Returns
    -------
    (ZNZModel, CNZModel)

    Backward-compatible: if deg_xhat/deg_z are absent (old NPZ), they are
    recovered from the coef length using _deg_from_coef with a default deg_xhat=1.
    """
    d = np.load(path, allow_pickle=True)
    b_coef = d["b_coef"]
    # Recover degrees: prefer stored fields; fall back to _deg_from_coef for old files.
    if "deg_xhat" in d:
        deg_xhat = int(d["deg_xhat"])
        deg_z = int(d["deg_z"])
    else:
        deg_xhat = 1  # production default
        deg_z = _deg_from_coef(b_coef, deg_xhat)
    znz = ZNZModel(
        b_coef=b_coef,
        sig_coef=d["sig_coef"],
        xhat_ref=float(d["xhat_ref"]),
        z_ref=float(d["z_ref"]),
        b_ref=float(d["b_ref"]),
        sig_ref=float(d["sig_ref"]),
        z_covariate=str(d["z_covariate"]),
        deg_xhat=deg_xhat,
        deg_z=deg_z,
    )
    cnz = CNZModel(
        g_grid=d["g_grid"],
        nhi_edges=d["nhi_edges"],
        z_edges_fine=d["z_edges_fine"],
    )
    return znz, cnz


# ---------------------------------------------------------------------------
# build_cache — reproducible Stage-0 NPZ entrypoint
# ---------------------------------------------------------------------------

def build_cache(argv=None):
    """CLI entrypoint: build (or rebuild) the stage-0 znz NPZ cache deterministically.

    Op-set used here is IDENTICAL to the b-measurement in measure_znz_response:
      (S2N_RED > snr_min) & (P_DLA > p_dla_min) & good_mask
    with NHI_TILT_HOST as the host-truth column (host_truth_floor=19.0).

    The cache is written to --out.  The exact N + b_ref + b(20.5, [2.25,2.75,3.25])
    are printed for verification.

    Usage
    -----
    python -m CDDF_analysis.znz_kernel build-cache \\
        --catalog-dir /scratch/.../gl_prod_2lpt0_v1_20260526/combined_catalog/ \\
        --truth       /nfs/.../hcd_truth_cat.fits \\
        --bal-cat     /nfs/.../bal_cat.fits \\
        --molly-tsv   /scratch/.../figures_molly_nhi195/lya_only/molly_matrix.tsv \\
        --out         /scratch/.../track_c/stage0/znz_2lpt0.npz

    All defaults match the documented WALL-1 calibrated configuration used by
    ab_loa0_fp_baseline.py (figures_molly_nhi195, host_truth_floor=19.0,
    NHI_TILT_HOST, snr_min=2.0, p_dla_min=0.99).
    """
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

    # Import here to avoid hard dependency at module-import time
    from CDDF_analysis.ab_loa0_fp_baseline import (
        build_ingredients, DEF_CAT, DEF_TRUTH, DEF_BAL,
        DEF_KERNEL, DEF_LOA0_PRODUCT,
    )
    from CDDF_analysis.cddf_catalog_hbi import build_fine_grid

    p = argparse.ArgumentParser(
        description="Build stage-0 znz NPZ cache (reproducible, documented op-set).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--catalog-dir", default=DEF_CAT)
    p.add_argument("--truth", default=DEF_TRUTH)
    p.add_argument("--bal-cat", default=DEF_BAL)
    p.add_argument("--molly-tsv", default=None,
                   help="Lyα-only nhi195 molly matrix (auto-resolved if not given)")
    p.add_argument("--kernel", default=DEF_KERNEL)
    p.add_argument("--loa0-product", default=DEF_LOA0_PRODUCT)
    p.add_argument("--out",
                   default=("/scratch/cavestru_root/cavestru0/mfho/"
                            "cddf_o3_realdata/track_c/stage0/znz_2lpt0.npz"))
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3,20.6")
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--fit-ceil", type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes", type=int, default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0,
                   help="host_truth_floor for load_and_cut_catalog (default 19.0)")
    p.add_argument("--deg-xhat", type=int, default=1)
    p.add_argument("--deg-z", type=int, default=2)
    p.add_argument("--z-fine-step", type=float, default=0.1)
    args = p.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print("[build_cache] loading ingredients (same op-set as ab_loa0_fp_baseline)...")
    ing = build_ingredients(args, fp_estimator="purity_mixture")
    cfg = ing["cfg"]
    cat_cut = ing["cat_cut"]
    truth_cut = ing["truth_cut"]
    good_mask = ing["good_mask"]
    mm = ing["mm"]
    fine_grid = build_fine_grid(cfg)

    print("[build_cache] measuring b(xhat, z) ...")
    meas = measure_znz_response(
        cat_cut, good_mask, cfg, mm, fine_grid,
        z_covariate="z_dla", host_col="NHI_TILT_HOST")

    N_tp = len(meas["xhat"])
    print(f"[build_cache] N (truth-matched TPs in op-set) = {N_tp:,}")

    znz = fit_znz_model(meas, deg_z=args.deg_z, deg_xhat=args.deg_xhat)
    print(f"[build_cache] b_ref = {znz.b_ref:.4f} at "
          f"(xhat_ref={znz.xhat_ref:.3f}, z_ref={znz.z_ref:.3f})")
    for z_eval in [2.25, 2.75, 3.25]:
        bval = float(znz.b(np.array([20.5]), np.array([z_eval]))[0])
        print(f"[build_cache] b(20.5, z={z_eval}) = {bval:.4f}")

    print("[build_cache] measuring g(N,z) completeness ...")
    zbins = np.asarray(cfg.zbins, float)
    z_lo, z_hi = float(zbins[0]), float(zbins[-1])
    z_edges_fine = np.arange(z_lo, z_hi + args.z_fine_step * 0.5, args.z_fine_step)
    meas_c = measure_c_nz(cat_cut, truth_cut, cfg, mm, z_edges_fine,
                           good_mask=good_mask)
    cnz = fit_c_nz_model(meas_c)

    print(f"[build_cache] saving -> {args.out}")
    save_znz(args.out, znz, cnz)

    # verify round-trip
    znz2, cnz2 = load_znz(args.out)
    assert np.allclose(znz2.b_coef, znz.b_coef), "round-trip b_coef mismatch"
    assert float(znz2.b(np.array([20.5]), np.array([2.75]))[0]) == \
           float(znz.b(np.array([20.5]), np.array([2.75]))[0]), "round-trip b() mismatch"
    print("[build_cache] round-trip verified OK.")

    print("\n[build_cache] STAMP:")
    print(f"  N           = {N_tp:,}")
    print(f"  b_ref       = {znz.b_ref:.4f}  (at xhat_ref={znz.xhat_ref:.4f}, z_ref={znz.z_ref:.4f})")
    for z_eval in [2.25, 2.75, 3.25]:
        bval = float(znz.b(np.array([20.5]), np.array([z_eval]))[0])
        print(f"  b(20.5,{z_eval}) = {bval:.4f}")
    print(f"  deg_xhat    = {znz.deg_xhat},  deg_z = {znz.deg_z}")
    print(f"  host_col    = NHI_TILT_HOST,  host_truth_floor = {args.host_truth_floor}")
    print(f"  op-cut      = (S2N_RED>{cfg.snr_min}) & (P_DLA>{cfg.p_dla_min}) & good_mask")
    print(f"  molly       = {cfg.molly_tsv}")
    return znz, cnz


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "build-cache":
        _sys.argv.pop(1)
        build_cache()
    else:
        print("Usage: python -m CDDF_analysis.znz_kernel build-cache [options]")
        print("       python znz_kernel.py build-cache [options]")
        _sys.exit(1)
