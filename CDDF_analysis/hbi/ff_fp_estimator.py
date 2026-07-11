"""Queue-2 FF+FP estimator (R2+): exposure-matched loa-0 FP subtraction followed by a
matched-real completeness-only correction, evaluated on the stamped Q1 calc_cddf
artifacts. Spec: desi_gpy_dla_notes/notes/2026-07-11_q2_spec.md (the contract).

Estimator per cell b = (N-bin, z-bin | SNR-stratum):

    n_FP,b   = w_b * n_FP,b^loa0          # exposure matching
    n_real,b = n_obs,b - n_FP,b           # NO clipping (negatives kept + flagged)
    f_b      = n_real,b / (C_eff,b * dX_b * dN_b)

Design decisions pinned here (discovered interfaces, 2026-07-11):

  * INPUT counts n_obs are the Q1 calccddf_{mock}_{closure,splits}.json arrays:
    LITERAL calc_cddf posterior-weighted expected counts (soft estimand R1), 52
    0.1-dex N-bins (17.2..22.4), z in [2, 3.5], SNR_REDSIDE > 2. Splits: 3 z-bins
    (full-SNR) + one z-integrated snr_gt4 stratum; snr (2,4] = full - snr_gt4 by
    differencing (the artifacts do NOT cross z with SNR).
  * FP = the committed loa-0 product (build_loa0_fp_product.py npz): fine (52,3)
    (logN, z) counts + molly (8 SNR, 12 NHI) counts, n_sl_loa0/n_sl_prod scalars,
    band-eta host occlusion. FIX-3c resample semantics REPLICATED as a pure
    function (single Jeffreys 1/2 at the LOWEST-N edge; NEVER per-cell +1/2;
    empty tiers draw EXACTLY 0), cross-checked against Loa0FP.resample in tests.
  * EXPOSURE WEIGHTING: the product does NOT persist a loa-0 per-z/SNR pathlength
    -> the sightline-count-anchored GLOBAL ratio is the PRIMARY weighting:
        w = (n_sl_prod/n_sl_loa0) * (dX_total^mock / dX_total^calib)
    (for mock == calib this is EXACTLY the committed Loa0FP vol_scale). The
    "dx_shape" VARIANT rescales per (z-bin | SNR-stratum) by the calib mock's dX
    shape (twin approximation: loa-0 sightlines are drawn from the calib
    2LPT-0 loa-124 population). A TRUE per-z dX-ratio weighting needs the loa-0
    run's own pathlength persisted in the product — documented FOLLOW-UP.
  * COMPLETENESS: the committed 2LPT-0 lya_only-nhi195 molly matrix (truth-indexed
    matched-real C; numerator = truth-matched TPs only -> NO purity embedded ->
    no double-count with the FP subtraction; purity is GUARDED so any access
    raises). C is z-marginal: this is the z-DIAGONAL variant (C per SNR-stratum
    only); per-z closure residuals MEASURE the missing z-resolution. C is
    truth-N-indexed but applied on N-hat bins: the diagonal (no-migration)
    approximation, labeled; migration is Model A's job.
    C_eff per (N-col, stratum) = truth-occupancy (cmp_nfid) weighted C over the
    stratum's SNR cells. Jeffreys-Beta(n_det+1/2, n_tot-n_det+1/2) MC draws from
    the regenerated (cmp_nfound, cmp_nfid) counts (cached; --build-molly-counts).
    C is undefined below N=19.5 (the nhi195 matrix floor): those bins report the
    FP-subtracted counts but f = NaN, flagged.
  * KNOWN estimand mismatch (stated, not hidden): n_obs is the SOFT posterior-
    weighted expected count, while C and the loa-0 FP are calibrated at the HARD
    P_DLA>0.99 operating point. The closure ratios measure the combined residual.

CLI:
  conda run -n gpdla python -m CDDF_analysis.hbi.ff_fp_estimator \
      --mock {2lpt0,saclay0,london0} --calib-mock 2lpt0 --out <json>
  [--ndraw 10000 --seed 0 --molly-counts <npz> --build-molly-counts]

MOCKS ONLY. This module never touches loa_main_dark_v1 (real LOA).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# constants / committed input paths (mirroring ab_loa0_fp_baseline defaults)
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HBI_DIR = os.path.join(_REPO, "CDDF_analysis", "hbi")

# the committed loa-0 forest-FP product (lya_only 1025 rebin — FIX 2 canonical)
DEF_LOA0_PRODUCT = ("/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/"
                    "outputs/loa0_fp_product_lyaonly1025.npz")
# the canonical 2LPT-0 lya_only-nhi195 molly C matrix (ab_loa0_fp_baseline
# DEF_LYAONLY_MOLLY — the matrix the broaden012 headline was calibrated against)
DEF_MOLLY_TSV = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
                 "figures_molly_nhi195/lya_only/molly_matrix.tsv")
# molly (n_det, n_tot) count cache (TSV stores ratios only). Built by
# --build-molly-counts from the 2LPT-0 production catalog + truth (15 s).
DEF_SCRATCH = os.environ.get(
    "FF_FP_CACHE_DIR",
    "/tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/"
    "000bee07-19b0-4a65-a031-f4078712a3e1/scratchpad")
DEF_MOLLY_COUNTS = os.path.join(DEF_SCRATCH, "molly_counts_2lpt0_lyaonly195.npz")

N_EDGES = np.round(np.arange(17.2, 22.40001, 0.1), 3)        # 52 bins
N_CENT = 0.5 * (N_EDGES[:-1] + N_EDGES[1:])
DN_LIN = 10.0 ** N_EDGES[1:] - 10.0 ** N_EDGES[:-1]
Z_EDGES = (2.0, 2.5, 3.0, 3.5)
Z_TAGS = ["z_2.0_2.5", "z_2.5_3.0", "z_3.0_3.5"]
SNR_HI = 4.0

TIERS = {                       # closure/reporting tiers (all >= the C floor 19.5)
    "dla_20.3": (20.3, np.inf),
    "dla_20.0": (20.0, np.inf),
    "subdla_195_203": (19.5, 20.3),
}
# molly SNR rows (edges 0,1,2,3,4,5,6,7,inf) covered by each stratum (SNR>2 sample)
STRATUM_SNR_ROWS = {
    "full": [2, 3, 4, 5, 6, 7],
    "snr_gt4": [4, 5, 6, 7],
    "snr_2_4": [2, 3],
}

# C2 blended-alpha reference (Q1 gate note 2026-07-11): held-out FF transfer
# residuals of the per-bin mock_recovery_ratio alpha calibrated on 2LPT-0.
C2_REFERENCE = {
    "saclay0": {"dla_20.3": -0.018, "dla_20.0": -0.022, "subdla_195_203": -0.072},
    "london0": {"dla_20.3": -0.024, "dla_20.0": -0.029, "subdla_195_203": -0.098},
}

Z_SHAPE_LABEL = ("DIAGONAL-IN-Z (C per SNR-stratum only; per-z closure residual "
                 "MEASURES the missing z-resolution)")


# ---------------------------------------------------------------------------
# no-double-count guard: purity (rho) must never be readable in the FF+FP path
# ---------------------------------------------------------------------------
class RhoAccessError(RuntimeError):
    """Raised on ANY attempt to read purity (rho) inside the FF+FP path."""


class RhoGuard:
    """Sentinel replacing every purity array: any access raises RhoAccessError.

    The FF+FP estimator subtracts the EXTERNAL loa-0 FP; molly C's numerator is
    truth-matched TPs only, so no purity is embedded. Using rho anywhere on top
    of that would double-count the FP removal — hence a loud guard, not a
    convention."""

    def __init__(self, name="purity"):
        object.__setattr__(self, "_name", name)

    def _raise(self):
        raise RhoAccessError(
            f"{object.__getattribute__(self, '_name')} (rho) must never be used "
            "in the FF+FP path — no-double-count guard (Q2 spec, input 3)")

    def __getattr__(self, key):
        self._raise()

    def __getitem__(self, key):
        self._raise()

    def __array__(self, dtype=None, copy=None):
        self._raise()

    def __iter__(self):
        self._raise()

    def __len__(self):
        self._raise()

    def __call__(self, *a, **k):
        self._raise()


# ---------------------------------------------------------------------------
# pure estimator functions
# ---------------------------------------------------------------------------
def subtract_fp(n_obs, n_fp_loa0, w):
    """n_real = n_obs - w * n_fp_loa0. NO clipping: negative bins are the
    diagnostic (flagged downstream), never zeroed. `w` scalar or per-bin."""
    return np.asarray(n_obs, float) - np.asarray(w, float) * np.asarray(n_fp_loa0, float)


def apply_completeness(n_real, C_eff, dX, dN):
    """f = n_real / (C_eff * dX * dN); NaN where C is undefined (NaN or <= 0).
    Negative n_real passes through (no clipping)."""
    n_real = np.asarray(n_real, float)
    C = np.asarray(C_eff, float)
    dN = np.asarray(dN, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = n_real / (C * float(dX) * dN)
    return np.where(np.isfinite(C) & (C > 0), f, np.nan)


def fp_gamma_draws(n_counts, floor_axis, ell_eff, rng, ndraw):
    """FIX-3c loa-0 FP rate draws -> effective perturbed loa-0 COUNTS, shape
    (ndraw, *n_counts.shape). Replicates Loa0FP.resample._neff semantics:

      lam_cell ~ Gamma(shape_cell, scale=1/ell_eff),  n_eff = lam * ell_eff
      shape_cell = n_cell + (1/2 / n_orth  iff cell is in the LOWEST row/col of
                             `floor_axis`), else n_cell.

    The SINGLE Jeffreys 1/2 for the whole inference sits at the lowest-N edge
    (split across the orthogonal axis); Gamma(0) = 0 exactly for every other
    empty cell -> an empty tier above any floor draws EXACTLY 0 FP (no phantom
    mass, no per-cell +1/2 — the FIX-3c defect this construction replaced)."""
    n = np.asarray(n_counts, float)
    shape = n.copy()
    if floor_axis == 0:
        shape[0, :] = shape[0, :] + 0.5 / shape.shape[1]
    elif floor_axis == 1:
        shape[:, 0] = shape[:, 0] + 0.5 / shape.shape[0]
    else:
        raise ValueError(f"floor_axis must be 0 or 1, got {floor_axis}")
    out = np.zeros((int(ndraw),) + shape.shape, dtype=float)
    pos = shape > 0.0
    k = int(pos.sum())
    if k:
        lam = rng.gamma(shape=shape[pos], scale=1.0 / float(ell_eff),
                        size=(int(ndraw), k))
        out[:, pos] = lam * float(ell_eff)
    return out


def beta_c_draws(n_found, n_fid, rng, ndraw):
    """Jeffreys-Beta completeness draws per molly cell:
        C_cell ~ Beta(n_det + 1/2, n_tot - n_det + 1/2)
    from the regenerated (cmp_nfound, cmp_nfid) counts. Cells with n_fid == 0
    have NO calibration -> NaN (excluded by occupancy weighting downstream)."""
    nf = np.asarray(n_found, float)
    nt = np.asarray(n_fid, float)
    if np.any(nf > nt + 1e-9):
        raise ValueError("n_found > n_fid: C numerator must be truth-matched only")
    out = np.full((int(ndraw),) + nf.shape, np.nan)
    valid = nt > 0
    k = int(valid.sum())
    if k:
        a = nf[valid] + 0.5
        b = (nt[valid] - nf[valid]) + 0.5
        out[:, valid] = rng.beta(a, b, size=(int(ndraw), k))
    return out


def c_eff_occupancy(C_cells, occ, rows):
    """Occupancy-weighted C per NHI column over the given SNR rows.

    C_cells: (..., n_snr, n_nhi) — leading axes (e.g. MC draws) broadcast.
    occ:     (n_snr, n_nhi) truth occupancy (cmp_nfid) — held FIXED across draws.
    rows:    SNR-row indices of the stratum.
    NaN C cells are excluded from numerator AND denominator; a column with no
    valid occupancy is NaN (undefined C_eff)."""
    C = np.asarray(C_cells, float)
    occ = np.asarray(occ, float)
    rows = np.asarray(rows, int)
    Cr = C[..., rows, :]
    wr = np.broadcast_to(occ[rows, :], Cr.shape)
    valid = np.isfinite(Cr)
    w = np.where(valid, wr, 0.0)
    num = np.where(valid, Cr, 0.0) * w
    den = w.sum(axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num.sum(axis=-2) / den
    return np.where(den > 0, out, np.nan)


def mc_band(rng, ndraw, n_obs, fp_point, fp_draws, w, C_point, C_draws, dX, dN):
    """Correlated MC band for one stratum (per-bin arrays).

    Draws: n_obs ~ Poisson(n_obs_point) (independent per bin); fp_draws and
    C_draws are SHARED across strata by the caller (correlation carrier); w is
    FIXED per exposure model. NO recenter-on-point anywhere. Negative n_real
    kept; bins with |n_real_point| < 2*sigma(n_real draws) flagged
    "consistent with zero"."""
    n_obs = np.asarray(n_obs, float)
    ndraw = int(ndraw)
    obs_draws = rng.poisson(lam=np.maximum(n_obs, 0.0),
                            size=(ndraw, n_obs.size)).astype(float)
    n_real_point = subtract_fp(n_obs, fp_point, w)
    n_real_draws = obs_draws - np.asarray(w, float) * fp_draws
    sigma = n_real_draws.std(axis=0)
    flag = np.abs(n_real_point) < 2.0 * sigma

    C_point = np.asarray(C_point, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_true_point = np.where(np.isfinite(C_point) & (C_point > 0),
                                n_real_point / C_point, np.nan)
        n_true_draws = np.where(np.isfinite(C_draws) & (C_draws > 0),
                                n_real_draws / C_draws, np.nan)
    f_point = apply_completeness(n_real_point, C_point, dX, dN)
    f_draws = n_true_draws / (float(dX) * np.asarray(dN, float))

    qs = np.nanpercentile(f_draws, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0) \
        if np.isfinite(f_draws).any() else np.full((5, n_obs.size), np.nan)
    return dict(
        n_real_point=n_real_point, n_real_sigma=sigma,
        flag_zero_consistent=flag,
        n_true_point=n_true_point, n_true_draws=n_true_draws,
        f_point=f_point, f_std=np.nanstd(f_draws, axis=0),
        f_q2p5=qs[0], f_q16=qs[1], f_q50=qs[2], f_q84=qs[3], f_q97p5=qs[4],
    )


# ---------------------------------------------------------------------------
# input loaders
# ---------------------------------------------------------------------------
def load_molly_completeness(tsv_path=DEF_MOLLY_TSV):
    """Load the committed molly matrix; return ONLY completeness. The purity
    matrix is replaced by a RhoGuard that raises on any access."""
    from CDDF_analysis.hbi.cddf_catalog_hbi import load_molly_matrix
    mm = load_molly_matrix(tsv_path)
    return dict(
        snr_edges=np.asarray(mm.snr_edges, float),
        nhi_edges=np.asarray(mm.nhi_edges, float),
        completeness=np.asarray(mm.completeness, float),
        purity=RhoGuard("purity"),
        tsv_path=tsv_path,
    )


def load_molly_counts(path=DEF_MOLLY_COUNTS):
    """Load the cached molly (n_det, n_tot) completeness counts. Returns None if
    the cache does not exist. Purity count arrays are NEVER exposed (RhoGuard)."""
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return dict(
        snr_edges=np.asarray(d["snr_edges"], float),
        nhi_edges=np.asarray(d["nhi_edges"], float),
        cmp_nfound=np.asarray(d["cmp_nfound"], float),
        cmp_nfid=np.asarray(d["cmp_nfid"], float),
        pur_ntp=RhoGuard("pur_ntp"), pur_ntot=RhoGuard("pur_ntot"),
        max_c_diff=float(d["max_c_diff"]),
        path=path,
    )


def build_molly_counts_cache(out_path=DEF_MOLLY_COUNTS, molly_tsv=DEF_MOLLY_TSV):
    """Regenerate the 2LPT-0 lya_only-nhi195 molly (n_det, n_tot) counts from the
    production catalog + truth (the TSV stores ratios only) and cache to npz.
    ~15 s. Guards: regenerated ratios must reproduce the committed TSV to 5e-3
    (the run_pipeline hard-guard threshold)."""
    from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
    from CDDF_analysis.hbi.cddf_catalog_hbi import (
        HBIConfig, load_molly_matrix, regenerate_molly_counts,
        load_and_cut_catalog, _build_qso_lookup)
    t0 = time.time()
    cfg = HBIConfig(
        catalog_dir=AB.DEF_CAT, truth_path=AB.DEF_TRUTH, bal_cat_path=AB.DEF_BAL,
        molly_tsv=molly_tsv, out_dir=os.path.dirname(out_path) or ".",
        mockdir=os.path.dirname(AB.DEF_TRUTH),
        lam_rf_min=1025.0,      # lya_only window (molly_summary: lam_rf_min 1025)
        no_bal=True)
    mm = load_molly_matrix(molly_tsv)
    truth_floor = float(mm.nhi_edges[0])
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, is_TP, good_mask, _meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(19.0, truth_floor))
    mm = regenerate_molly_counts(mm, cat_cut, is_TP, truth_cut, good_mask, cfg)
    if mm._max_c_diff > 5e-3:
        raise SystemExit(
            f"regenerated completeness ratios deviate from the committed TSV by "
            f"{mm._max_c_diff:.2e} > 5e-3 — cut bundle no longer replicates molly")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path,
             snr_edges=mm.snr_edges, nhi_edges=mm.nhi_edges,
             cmp_nfound=mm.cmp_nfound, cmp_nfid=mm.cmp_nfid,
             pur_ntp=mm.pur_ntp, pur_ntot=mm.pur_ntot,   # stored, never exposed
             max_p_diff=mm._max_p_diff, max_c_diff=mm._max_c_diff,
             molly_tsv=molly_tsv, lam_rf_min=1025.0)
    print(f"[ff_fp] molly counts cached -> {out_path} "
          f"({time.time()-t0:.0f}s; max_c_diff={mm._max_c_diff:.1e})")
    return out_path


def load_fp_product(path=DEF_LOA0_PRODUCT):
    d = np.load(path, allow_pickle=True)
    prod = dict(
        n_fp_fine=np.asarray(d["n_fp_fine"], float),         # (52, 3) counts
        n_fp_molly=np.asarray(d["n_fp_molly"], float),       # (8, 12) counts
        snr_edges=np.asarray(d["snr_edges"], float),
        nhi_edges=np.asarray(d["nhi_edges"], float),         # nhi172 grid
        logN_lo=np.asarray(d["logN_lo"], float),
        logN_hi=np.asarray(d["logN_hi"], float),
        zbins=np.asarray(d["zbins"], float),
        band_eta_per_nbin=np.asarray(d["band_eta_per_nbin"], float),
        n_sl_loa0=float(d["n_sl_loa0"]), n_sl_prod=float(d["n_sl_prod"]),
        ell_eff=float(d["ell_eff"]), path=path,
    )
    # grid contract with the Q1 artifacts
    if not (np.allclose(prod["logN_lo"], N_EDGES[:-1])
            and np.allclose(prod["logN_hi"], N_EDGES[1:])):
        raise SystemExit("loa-0 product fine N grid != Q1 artifact N grid")
    if not np.allclose(prod["zbins"], Z_EDGES):
        raise SystemExit("loa-0 product zbins != (2.0, 2.5, 3.0, 3.5)")
    return prod


def load_artifact(mock, kind, art_dir=_HBI_DIR):
    path = os.path.join(art_dir, f"calccddf_{mock}_{kind}.json")
    with open(path) as f:
        d = json.load(f)
    d["_path"] = path
    return d


# ---------------------------------------------------------------------------
# strata assembly from the Q1 artifacts
# ---------------------------------------------------------------------------
def build_strata(closure, splits):
    """{name: {n_obs(52), n_truth(52), dX}} for full, 3 z-bins, snr_gt4 and the
    differenced snr_2_4 = full - snr_gt4 (counts AND dX)."""
    full_obs = np.asarray(closure["counts_calccddf_N"], float)
    full_tru = np.asarray(closure["counts_truth_N"], float)
    dX_full = float(closure["dX_total"])
    out = {"full": dict(n_obs=full_obs, n_truth=full_tru, dX=dX_full)}
    sp = splits["splits"]
    for tag in Z_TAGS:
        out[tag] = dict(n_obs=np.asarray(sp[tag]["counts_est"], float),
                        n_truth=np.asarray(sp[tag]["counts_truth"], float),
                        dX=float(sp[tag]["dX"]))
    g4 = sp[f"snr_gt{SNR_HI:g}"]
    out["snr_gt4"] = dict(n_obs=np.asarray(g4["counts_est"], float),
                          n_truth=np.asarray(g4["counts_truth"], float),
                          dX=float(g4["dX"]))
    # splits' own full-sample accumulation (same run as the splits)
    sp_full_obs = np.asarray(splits["counts_calccddf_N"], float)
    sp_full_tru = np.asarray(splits["counts_truth_N"], float)
    out["snr_2_4"] = dict(
        n_obs=sp_full_obs - out["snr_gt4"]["n_obs"],
        n_truth=sp_full_tru - out["snr_gt4"]["n_truth"],
        dX=float(splits["dX_total"]) - out["snr_gt4"]["dX"])
    return out


def check_z_additivity(strata, rtol=1e-6):
    """The gate note pins z-split additivity as bit-level; enforce it softly."""
    s_obs = sum(strata[t]["n_obs"] for t in Z_TAGS)
    s_dx = sum(strata[t]["dX"] for t in Z_TAGS)
    ok_counts = np.allclose(s_obs, strata["full"]["n_obs"], rtol=rtol, atol=1e-6)
    ok_dx = np.isclose(s_dx, strata["full"]["dX"], rtol=rtol)
    return bool(ok_counts and ok_dx)


# ---------------------------------------------------------------------------
# FP expectation per stratum (point + draws), in loa-0 counts (pre-weighting)
# ---------------------------------------------------------------------------
def _fine_bin_to_molly_col(nhi_edges):
    """Map each fine N bin (by center) to its molly NHI column."""
    j = np.searchsorted(nhi_edges, N_CENT, side="right") - 1
    return np.clip(j, 0, len(nhi_edges) - 2)


def fp_stratum_counts(prod, fine, molly):
    """loa-0 FP counts per fine N bin for every stratum, from a fine-grid array
    `fine` (..., 52, 3) and a molly-grid array `molly` (..., 8, 12) (point counts
    or MC draws). Host-occlusion (1 - band_eta) applied per fine bin (mirrors
    Loa0FP.mu_fp_grid).

    SNR strata: the fine grid is NOT SNR-resolved -> the z-integrated fine counts
    are split by the molly grid's per-NHI-column SNR-stratum SHARE (rows 2..7 =
    the SNR>2 sample). This keeps the fine grid's z-window (the molly grid also
    counts ~2.6% z-outside FPs) and uses molly only for the SNR shape."""
    one_m_eta = 1.0 - prod["band_eta_per_nbin"]                     # (52,)
    base = fine * one_m_eta[..., :, None]                           # (...,52,3)
    z_int = base.sum(axis=-1)                                       # (...,52)
    out = {"full": z_int}
    for k, tag in enumerate(Z_TAGS):
        out[tag] = base[..., :, k]
    col_of_bin = _fine_bin_to_molly_col(prod["nhi_edges"])          # (52,)
    tot = molly[..., 2:8, :].sum(axis=-2)                           # (...,12)
    for s in ("snr_gt4", "snr_2_4"):
        rows = STRATUM_SNR_ROWS[s]
        num = molly[..., rows, :].sum(axis=-2)                      # (...,12)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(tot > 0, num / tot, 0.0)
        out[s] = z_int * share[..., col_of_bin]
    return out


# ---------------------------------------------------------------------------
# exposure weights
# ---------------------------------------------------------------------------
def exposure_weights(prod, strata_m, strata_c):
    """Two exposure models (their spread = the exposure-model systematic):

    PRIMARY 'global' (sightline-count anchored): w = (n_sl_prod/n_sl_loa0) *
      (dX_total^mock / dX_total^calib), identical for all strata. For
      mock == calib this is EXACTLY the committed Loa0FP vol_scale.
    VARIANT 'dx_shape': per-stratum w_s = (n_sl_prod/n_sl_loa0) *
      (dX_s^mock / dX_s^calib) — the calib mock's per-(z, SNR) dX shape stands in
      for the loa-0 shape (twin approximation). A TRUE dX-ratio weighting needs
      the loa-0 per-z pathlength, which the product does not persist (FOLLOW-UP).
    """
    W0 = prod["n_sl_prod"] / prod["n_sl_loa0"]
    w_global = W0 * strata_m["full"]["dX"] / strata_c["full"]["dX"]
    out = {"global": {s: float(w_global) for s in strata_m},
           "dx_shape": {s: float(W0 * strata_m[s]["dX"] / strata_c[s]["dX"])
                        for s in strata_m}}
    return out


# ---------------------------------------------------------------------------
# completeness per stratum
# ---------------------------------------------------------------------------
def c_eff_per_bin(C_cells, occ, molly195_edges, stratum):
    """Expand the per-NHI-column occupancy-weighted C_eff onto the 52 fine bins.
    Bins below the matrix floor (19.5) have NO calibrated C -> NaN."""
    rows = STRATUM_SNR_ROWS["full" if stratum in ("full", *Z_TAGS) else stratum]
    per_col = c_eff_occupancy(C_cells, occ, rows)          # (..., n_nhi195)
    col = np.searchsorted(molly195_edges, N_CENT, side="right") - 1
    valid = col >= 0
    colc = np.clip(col, 0, len(molly195_edges) - 2)
    out = per_col[..., colc]
    out = np.where(valid, out, np.nan)
    return out


# ---------------------------------------------------------------------------
# closure reductions
# ---------------------------------------------------------------------------
def _tier_sel(lo, hi):
    return (N_EDGES[:-1] >= lo - 1e-9) & (N_EDGES[1:] <= hi + 1e-9)


def tier_closures(st, band, w, fp_point, fp_draws, n_truth):
    """Per-tier closure ratios: raw (detected-space), subtracted-only, and the
    FF+FP point + MC band, plus the FP totals (point and band)."""
    out = {}
    tier_R_draws = {}
    for tier, (lo, hi) in TIERS.items():
        sel = _tier_sel(lo, hi)
        tru = float(n_truth[sel].sum())
        n_obs_t = float(st["n_obs"][sel].sum())
        fp_t = float(w * fp_point[sel].sum())
        nreal_t = float(band["n_real_point"][sel].sum())
        ntrue_t = float(np.nansum(band["n_true_point"][sel]))
        R_draws = np.nansum(band["n_true_draws"][:, sel], axis=1) / tru \
            if tru > 0 else np.full(band["n_true_draws"].shape[0], np.nan)
        fp_band = w * fp_draws[:, sel].sum(axis=1)
        q = np.nanpercentile(R_draws, [2.5, 16, 50, 84, 97.5]) \
            if np.isfinite(R_draws).any() else [np.nan] * 5
        out[tier] = dict(
            truth_total=tru, obs_total=n_obs_t,
            fp_subtracted_total=fp_t,
            fp_band_lo95=float(np.percentile(fp_band, 2.5)),
            fp_band_hi95=float(np.percentile(fp_band, 97.5)),
            R_raw=(n_obs_t / tru if tru > 0 else np.nan),
            R_subtracted_only=(nreal_t / tru if tru > 0 else np.nan),
            R_point=(ntrue_t / tru if tru > 0 else np.nan),
            R_q2p5=float(q[0]), R_q16=float(q[1]), R_q50=float(q[2]),
            R_q84=float(q[3]), R_q97p5=float(q[4]),
            R_std=float(np.nanstd(R_draws)),
        )
        tier_R_draws[tier] = R_draws
    # full bin-bin covariance is huge; report the tier-level covariance (§9.5)
    names = list(TIERS)
    M = np.stack([tier_R_draws[t] for t in names])
    cov = np.cov(M) if np.isfinite(M).all() else np.full((3, 3), np.nan)
    return out, dict(tiers=names, cov=cov.tolist())


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run_estimator(mock, calib_mock="2lpt0", ndraw=10_000, seed=0,
                  loa0_product=DEF_LOA0_PRODUCT, molly_tsv=DEF_MOLLY_TSV,
                  molly_counts_path=DEF_MOLLY_COUNTS, art_dir=_HBI_DIR,
                  build_counts_if_missing=True):
    """FF+FP estimator on one mock. Returns a JSON-serializable result dict."""
    t0 = time.time()
    rng = np.random.default_rng(seed)

    closure = load_artifact(mock, "closure", art_dir)
    splits = load_artifact(mock, "splits", art_dir)
    prod = load_fp_product(loa0_product)
    mc = load_molly_completeness(molly_tsv)
    counts = load_molly_counts(molly_counts_path)
    if counts is None:
        if not build_counts_if_missing:
            raise SystemExit(f"molly counts cache missing: {molly_counts_path} "
                             "(run --build-molly-counts)")
        build_molly_counts_cache(molly_counts_path, molly_tsv)
        counts = load_molly_counts(molly_counts_path)
    if not (np.allclose(counts["snr_edges"], mc["snr_edges"])
            and np.allclose(counts["nhi_edges"], mc["nhi_edges"])):
        raise SystemExit("molly counts cache grid != committed TSV grid")
    # double-count audit (analytic contract of matched-real C): the numerator is
    # truth-matched TPs only -> n_found <= n_fid everywhere; rho never read.
    assert np.all(counts["cmp_nfound"] <= counts["cmp_nfid"] + 1e-9), \
        "C numerator contains unmatched detections — matched-real contract broken"

    strata_m = build_strata(closure, splits)
    if mock == calib_mock:
        strata_c = strata_m
    else:
        strata_c = build_strata(load_artifact(calib_mock, "closure", art_dir),
                                load_artifact(calib_mock, "splits", art_dir))
    z_add_ok = check_z_additivity(strata_m)
    weights = exposure_weights(prod, strata_m, strata_c)

    # ---- FP: point + shared MC draws (FIX-3c semantics) ----
    fp_point = fp_stratum_counts(prod, prod["n_fp_fine"], prod["n_fp_molly"])
    fine_draws = fp_gamma_draws(prod["n_fp_fine"], 0, prod["ell_eff"], rng, ndraw)
    molly_draws = fp_gamma_draws(prod["n_fp_molly"], 1, prod["ell_eff"], rng, ndraw)
    fp_draws = fp_stratum_counts(prod, fine_draws, molly_draws)

    # ---- completeness: point + shared Jeffreys-Beta draws ----
    occ = counts["cmp_nfid"]                       # truth occupancy, held fixed
    C_cell_draws = beta_c_draws(counts["cmp_nfound"], counts["cmp_nfid"],
                                rng, ndraw)
    C_point_cells = mc["completeness"]

    # ---- per-stratum estimator ----
    strata_out = {}
    for s, st in strata_m.items():
        w = weights["global"][s]                   # PRIMARY exposure model
        C_pt = c_eff_per_bin(C_point_cells, occ, mc["nhi_edges"], s)
        C_dr = c_eff_per_bin(C_cell_draws, occ, mc["nhi_edges"], s)
        band = mc_band(rng, ndraw, st["n_obs"], fp_point[s], fp_draws[s],
                       w, C_pt, C_dr, st["dX"], DN_LIN)
        closures, tier_cov = tier_closures(st, band, w, fp_point[s],
                                           fp_draws[s], st["n_truth"])
        # dx_shape exposure variant (point-only; the spread = the systematic)
        w2 = weights["dx_shape"][s]
        nreal2 = subtract_fp(st["n_obs"], fp_point[s], w2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ntrue2 = np.where(np.isfinite(C_pt) & (C_pt > 0), nreal2 / C_pt,
                              np.nan)
        var_cl = {}
        for tier, (lo, hi) in TIERS.items():
            sel = _tier_sel(lo, hi)
            tru = float(st["n_truth"][sel].sum())
            var_cl[tier] = float(np.nansum(ntrue2[sel]) / tru) if tru > 0 \
                else float("nan")
        strata_out[s] = dict(
            dX=st["dX"], w_primary=w, w_dx_shape=w2,
            n_obs=st["n_obs"].tolist(), n_truth=st["n_truth"].tolist(),
            n_fp_weighted=(w * fp_point[s]).tolist(),
            n_real=band["n_real_point"].tolist(),
            n_real_sigma=band["n_real_sigma"].tolist(),
            flag_zero_consistent=band["flag_zero_consistent"].tolist(),
            f_point=band["f_point"].tolist(),
            f_std=band["f_std"].tolist(),
            f_q2p5=band["f_q2p5"].tolist(), f_q16=band["f_q16"].tolist(),
            f_q50=band["f_q50"].tolist(), f_q84=band["f_q84"].tolist(),
            f_q97p5=band["f_q97p5"].tolist(),
            closure=closures, tier_R_covariance=tier_cov,
            closure_dx_shape_variant=var_cl,
        )

    # ---- C2 comparison (held-out transfer residuals of the blended alpha) ----
    # C2's residuals are TRANSFER residuals (alpha forces the calib mock to 1 by
    # construction), so the apples-to-apples FF+FP number is R_mock / R_calib - 1
    # (the calib mock's own FF+FP closure computed with the SAME external C+FP).
    W0 = prod["n_sl_prod"] / prod["n_sl_loa0"]
    C_pt_full = c_eff_per_bin(C_point_cells, occ, mc["nhi_edges"], "full")
    nreal_c = subtract_fp(strata_c["full"]["n_obs"], fp_point["full"], W0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ntrue_c = np.where(np.isfinite(C_pt_full) & (C_pt_full > 0),
                           nreal_c / C_pt_full, np.nan)
    calib_R = {}
    for tier, (lo, hi) in TIERS.items():
        sel = _tier_sel(lo, hi)
        tru_c = float(strata_c["full"]["n_truth"][sel].sum())
        calib_R[tier] = float(np.nansum(ntrue_c[sel]) / tru_c) if tru_c > 0 \
            else float("nan")
    c2 = dict(
        reference_residuals=C2_REFERENCE.get(mock),
        note=("C2 = per-bin blended mock_recovery_ratio alpha calibrated on "
              "2LPT-0, applied held-out (Q1 gate note); its residuals are "
              "transfer residuals (calib == 1 by construction). FF+FP separates "
              "the additive FP before a completeness-only correction; compare "
              "the sub-DLA one-sidedness of the TRANSFER residuals."),
        fffp_absolute_residuals={
            t: strata_out["full"]["closure"][t]["R_point"] - 1.0 for t in TIERS},
        calib_R_point=calib_R,
        fffp_transfer_residuals={
            t: strata_out["full"]["closure"][t]["R_point"] / calib_R[t] - 1.0
            for t in TIERS},
    )

    res = dict(
        mock=mock, calib_mock=calib_mock, ndraw=int(ndraw), seed=int(seed),
        estimator="FF+FP (R2+): exposure-matched loa-0 FP subtraction + "
                  "matched-real completeness-only correction",
        N_edges=N_EDGES.tolist(), z_edges=list(Z_EDGES),
        strata=strata_out,
        checks=dict(z_split_additivity_ok=z_add_ok,
                    molly_counts_reproduce_tsv_max_diff=counts["max_c_diff"],
                    c_numerator_matched_real_only=True),
        c2_comparison=c2,
        provenance=dict(
            routine="CDDF_analysis/hbi/ff_fp_estimator.py",
            code_commit=_git_commit(),
            code_state="UNCOMMITTED (Queue-2 implementation, pre-review)",
            date=time.strftime("%Y-%m-%d"),
            rederive=(f"conda run -n gpdla python -m CDDF_analysis.hbi."
                      f"ff_fp_estimator --mock {mock} --calib-mock {calib_mock} "
                      f"--ndraw {ndraw} --seed {seed} --out <out>"),
            artifacts=[closure["_path"], splits["_path"]],
            loa0_product=prod["path"],
            molly_tsv=molly_tsv,
            molly_counts_cache=counts["path"],
            weighting=dict(
                primary="global (sightline-count anchored): w = "
                        "(n_sl_prod/n_sl_loa0) * (dX_tot^mock/dX_tot^calib)",
                variant="dx_shape: per-(z|SNR)-stratum dX ratio vs the calib "
                        "mock (twin approximation for the loa-0 shape)",
                followup="TRUE per-z dX-ratio weighting needs the loa-0 run's "
                         "own pathlength persisted in the FP product — not "
                         "available; NOT fabricated.",
                n_sl_loa0=prod["n_sl_loa0"], n_sl_prod=prod["n_sl_prod"]),
            z_shape=Z_SHAPE_LABEL,
            n_migration=("C is truth-N-indexed, applied on N-hat bins: diagonal "
                         "(no-migration) approximation — Model A carries "
                         "migration via the forward kernel"),
            estimand_note=("n_obs = LITERAL calc_cddf posterior-weighted "
                           "expected counts (soft); C and loa-0 FP calibrated "
                           "at the hard P_DLA>0.99 operating point — closure "
                           "measures the combined residual"),
            fp_resample="FIX-3c: single Jeffreys 1/2 at the lowest-N edge; "
                        "empty tiers draw exactly 0; never per-cell +1/2",
            recenter_band_on_point=False,
            rho_used=False,
            wallclock_s=time.time() - t0,
        ),
    )
    return _py(res)


def _py(o):
    """Recursively convert numpy scalars/arrays to JSON-serializable python."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_tables(res):
    m = res["mock"]
    print(f"\n===== FF+FP closure — {m} (calib={res['calib_mock']}, "
          f"ndraw={res['ndraw']}, seed={res['seed']}) =====")
    hdr = (f"{'stratum':<12} {'tier':<15} {'R_raw':>7} {'R_sub':>7} "
           f"{'R_FF+FP':>8} {'q16':>7} {'q84':>7} {'FP_sub':>8} {'dxvar':>7}")
    print(hdr)
    print("-" * len(hdr))
    for s in ("full", *Z_TAGS, "snr_gt4", "snr_2_4"):
        for tier in TIERS:
            c = res["strata"][s]["closure"][tier]
            v = res["strata"][s]["closure_dx_shape_variant"][tier]
            print(f"{s:<12} {tier:<15} {c['R_raw']:>7.3f} "
                  f"{c['R_subtracted_only']:>7.3f} {c['R_point']:>8.3f} "
                  f"{c['R_q16']:>7.3f} {c['R_q84']:>7.3f} "
                  f"{c['fp_subtracted_total']:>8.1f} {v:>7.3f}")
    c2 = res["c2_comparison"]
    print("\nC2 blended-alpha comparison (full stratum):")
    print(f"  {'tier':<15} {'FF+FP abs':>10} {'FF+FP transfer':>15} {'C2 (transfer)':>14}")
    for t in TIERS:
        ref = (c2["reference_residuals"] or {}).get(t)
        ref_s = f"{ref:+.3f}" if ref is not None else "n/a"
        print(f"  {t:<15} {c2['fffp_absolute_residuals'][t]:>+10.3f} "
              f"{c2['fffp_transfer_residuals'][t]:>+15.3f} {ref_s:>14}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mock", choices=["2lpt0", "saclay0", "london0"],
                   required=False)
    p.add_argument("--calib-mock", default="2lpt0",
                   choices=["2lpt0", "saclay0", "london0"])
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--ndraw", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loa0-product", default=DEF_LOA0_PRODUCT)
    p.add_argument("--molly-tsv", default=DEF_MOLLY_TSV)
    p.add_argument("--molly-counts", default=DEF_MOLLY_COUNTS)
    p.add_argument("--build-molly-counts", action="store_true",
                   help="(re)build the molly count cache and exit")
    args = p.parse_args(argv)

    if args.build_molly_counts:
        build_molly_counts_cache(args.molly_counts, args.molly_tsv)
        return 0
    if not args.mock:
        p.error("--mock is required (unless --build-molly-counts)")

    res = run_estimator(
        mock=args.mock, calib_mock=args.calib_mock, ndraw=args.ndraw,
        seed=args.seed, loa0_product=args.loa0_product,
        molly_tsv=args.molly_tsv, molly_counts_path=args.molly_counts)
    _print_tables(res)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\n[ff_fp] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
