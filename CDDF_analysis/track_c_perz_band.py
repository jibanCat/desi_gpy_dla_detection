#!/usr/bin/env python
"""track_c_perz_band.py — Track-C PER-Z (redshift-resolved) forward-kernel coverage.

The Track-C headline deliverable (track_c_td_band.py) reports the z-MARGINALIZED
dN/dX, Ω and f(N) recovered by the forward-empirical kernel on the 2LPT-0 mock. This
standalone diagnostic resolves that SAME result IN REDSHIFT: per coarse z bin
(z≈2.25 / 2.75 / 3.25) it reports

  * dN/dX(z) and Ω(z) integrated at NHI ≥ 20.0 / ≥ 20.3  (MAP + recentered 68/95 band
    + 2LPT-0 truth + cover?),
  * the differential f(N | z) over the DLA range (HBI band + MAP + truth),

so the PI can see whether the forward kernel FLATTENS the z-dependent over-/under-
recovery that the earlier kappa kernel showed (a rising 20.3–21.5 shoulder skew
~0.87 → ~1.18 across z).

It IMPORTS the committed estimator at HEAD (the forward kernel + recenter + slope-
extrap); it does NOT edit cddf_catalog_hbi.py or track_c_td_band.py. The forward band
is built with the EXACT track_c_td_band recipe:

  resp_kind=forward, resp_family=empirical, kernel_forward_model=<forward NPZ>,
  Stage I = mc_inner=laplace, Stage II = mc_nuisance=shared_boot,
  Stage III = mc_response=marginalize (per-draw forward refit on the shared boot_mult),
  band_recenter=True, omega_slope_extrap=True.

The per-z machinery is the canonical estimator's: the genuine 2-D f(N | z_coarse)
``f_bk_coarse`` (the SAME pathlength-weighted reduction _v2_reduce uses for dndx_z,
tied by construction — see cddf_catalog_hbi._coarse_z_differential_f), collected per MC
draw by v3x_joint_mc as ``_f_bk_coarse_samples``. We reduce each draw's f_bk_coarse to
per-z dN/dX(z) and Ω(z), recenter on the MAP point per z, and (for Ω) splice the high-N
slope-extrapolation per z column.

Reduce-only / analysis-side. NO GP inference (gpy_dla_detection/ byte-frozen). conda
gpdla; BLAS pinned; few workers.

Usage:
  python CDDF_analysis/track_c_perz_band.py --n-mc 120 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis import ab_loa0_fp_baseline as AB
from CDDF_analysis.ab_loa0_fp_baseline import build_ingredients, run_baseline
from CDDF_analysis.cddf_catalog_hbi import (
    joint_mc_errors, make_v3x_refit_fn, v3x_reduce, build_truth_match_resample,
    omega_hi_prefactor, recenter_band_on_point,
    omega_integrated_slope_extrap_samples,
    _bin_index_logN, _zbin_index,
)

_DEF_FORWARD = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                "track_c/stage0/forward_response_2lpt0.npz")
_DEF_OUT = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/perz_band")


# -----------------------------------------------------------------------------
# per-z reductions from the genuine 2-D f(N | z_coarse)
# -----------------------------------------------------------------------------
def perz_dndx_from_fbk(f_bk, logN_lo, dN_b, lim):
    """dN/dX(z_coarse) at NHI >= lim from the genuine per-coarse-z differential f.
    Σ_{N≥lim} f_bk[:, k]·dN_b — tied by construction to _v2_reduce's dndx_z[lim]."""
    sel = np.asarray(logN_lo, float) >= lim - 1e-9
    return np.nansum(np.asarray(f_bk, float)[sel, :] * np.asarray(dN_b, float)[sel, None],
                     axis=0)


def perz_omega_from_fbk(f_bk, logN_lo, N_b, dN_b, K, lim):
    """Ω(z_coarse) at NHI >= lim from the per-coarse-z f. K·Σ_{N≥lim} N_b·f_bk[:, k]·dN_b."""
    sel = np.asarray(logN_lo, float) >= lim - 1e-9
    N = np.asarray(N_b, float)[sel, None]
    dN = np.asarray(dN_b, float)[sel, None]
    return K * np.nansum(N * np.asarray(f_bk, float)[sel, :] * dN, axis=0)


# -----------------------------------------------------------------------------
# truth f(N,z) per (logN bin, coarse z bin) — mirrors hbi_fNz_coverage.truth_fNz
# -----------------------------------------------------------------------------
def truth_fNz(cfg, truth_cut, logN_lo, logN_hi, dN_b, X_tot):
    """f_truth[b,k] = (truth count in logN bin b AND z bin k) / (dN_b[b]·X_tot[k]).
    SAME snr cut (S2N_RED > snr_min) + ceiling (build_fine_grid drops top bin) as the
    estimator's fine grid. X_tot is the per-coarse-z pathlength."""
    zbins = np.asarray(cfg.zbins, float)
    n_zc = len(zbins) - 1
    n_nbins = len(logN_lo)
    X = np.asarray(X_tot, float)
    t_nhi = np.asarray(truth_cut["NHI"], float)
    t_z = np.asarray(truth_cut["Z_DLA"], float)
    t_snr = np.asarray(truth_cut["S2N_RED"], float)
    keep = t_snr > cfg.snr_min
    t_nhi, t_z = t_nhi[keep], t_z[keep]
    t_nidx = _bin_index_logN(t_nhi, logN_lo, logN_hi)   # -1 outside [floor, ceil]
    t_zidx = _zbin_index(t_z, zbins)                     # -1 outside zbins
    counts = np.zeros((n_nbins, n_zc))
    valid = (t_nidx >= 0) & (t_zidx >= 0)
    np.add.at(counts, (t_nidx[valid], t_zidx[valid]), 1.0)
    f = np.full((n_nbins, n_zc), np.nan)
    for k in range(n_zc):
        if X[k] > 0:
            f[:, k] = counts[:, k] / (dN_b * X[k])
    return dict(f_truth=f, counts=counts, X_tot=X)


def truth_perz_integrals(cfg, f_truth, logN_lo, N_b, dN_b, limits):
    """Truth dN/dX(z) and Ω(z) at each limit, integrated from the truth f(N,z)."""
    K = omega_hi_prefactor(cfg.H0)
    out = dict(dndx={}, omega={})
    for l in limits:
        out["dndx"][l] = perz_dndx_from_fbk(f_truth, logN_lo, dN_b, l)
        out["omega"][l] = perz_omega_from_fbk(f_truth, logN_lo, N_b, dN_b, K, l)
    return out


# -----------------------------------------------------------------------------
# band quantiles + coverage
# -----------------------------------------------------------------------------
def _band(samp, point=None, recenter=False):
    """68/95 quantile band. When recenter and a finite point, the samples are first
    shifted so their median sits at the point (FIX 1; spread preserved)."""
    s = np.asarray(samp, float)
    if recenter and point is not None and np.isfinite(point):
        s = recenter_band_on_point(s, point)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return dict(q025=np.nan, q16=np.nan, q50=np.nan, q84=np.nan, q975=np.nan,
                    std=np.nan, n=0)
    return dict(q025=float(np.percentile(s, 2.5)), q16=float(np.percentile(s, 16.0)),
                q50=float(np.percentile(s, 50.0)), q84=float(np.percentile(s, 84.0)),
                q975=float(np.percentile(s, 97.5)), std=float(np.std(s)), n=int(s.size))


def _cover(band, truth):
    if not np.isfinite(truth):
        return dict(in68=None, in95=None)
    return dict(in68=bool(band["q16"] <= truth <= band["q84"]),
                in95=bool(band["q025"] <= truth <= band["q975"]))


# -----------------------------------------------------------------------------
# the forward per-z band run
# -----------------------------------------------------------------------------
def _set_forward_cfg(cfg, args):
    """Configure the forward-empirical marginalized band on cfg (mirrors
    track_c_td_band._set_forward_band_cfg, carry=True)."""
    cfg.resp_kind = "forward"
    cfg.resp_family = args.resp_family
    cfg.kernel_forward_model = args.forward_model
    cfg.mc_inner = "laplace"
    cfg.mc_nuisance = "shared_boot"
    cfg.mc_response = "marginalize"
    cfg.band_recenter = bool(args.band_recenter)
    cfg.omega_slope_extrap = bool(args.omega_slope_extrap)
    cfg.omega_slope_extrap_edge = float(args.slope_edge)
    cfg.omega_slope_extrap_fit_dex = float(args.slope_fit_dex)
    cfg.omega_slope_extrap_sigma = float(args.sigma_slope)
    cfg.omega_slope_extrap_integrated = bool(args.omega_slope_extrap_integrated)


def run_perz(args, limits, seed):
    """Build the PM forward-empirical per-z marginalized band; return point + samples."""
    t0 = time.time()
    ing = build_ingredients(args, "purity_mixture")
    cfg = ing["cfg"]
    cfg.report_logN_limits = limits
    cfg._wall1_estimator = "v3"
    cfg.n_mc = args.n_mc
    _set_forward_cfg(cfg, args)

    logN_lo = ing["logN_lo"]; logN_hi = ing["logN_hi"]
    N_b = ing["N_b"]; dN_b = ing["dN_b"]
    zbins = np.asarray(cfg.zbins, float)
    n_zc = len(zbins) - 1
    K = omega_hi_prefactor(cfg.H0)

    # ---- single-source point: the forward MAP headline (e0) + per-z MAP f ----
    base = run_baseline(ing)
    e0 = base["e0"]
    fwd = e0["_v3x"]["fwd"]; family = e0["_v3x"]["family"]; fine = e0["_v3x"]["fine"]
    M_meta = e0["_v3x"]["M_meta"]; theta_map = e0["_v3x"]["theta_map"]
    # genuine MAP per-coarse-z f (NOT the np.repeat filler in e0['f_bk'])
    rr_map = v3x_reduce(cfg, theta_map, fine, family, M_meta)
    map_fbk = np.asarray(rr_map["f_bk_coarse"], float)        # (n_nbins, n_zc)
    # MAP per-z integrals (tied to e0['dndx_z']; assert below)
    map_dndx = {l: perz_dndx_from_fbk(map_fbk, logN_lo, dN_b, l) for l in limits}
    map_omega = {l: perz_omega_from_fbk(map_fbk, logN_lo, N_b, dN_b, K, l) for l in limits}
    # consistency gate: MAP per-z dN/dX from f_bk == e0['dndx_z']
    cerr = 0.0
    for l in limits:
        a = map_dndx[l]; b = np.asarray(e0["dndx_z"][l], float)
        good = np.isfinite(b) & (np.abs(b) > 0)
        if good.any():
            cerr = max(cerr, float(np.max(np.abs(a[good] - b[good]) / np.abs(b[good]))))
    if cerr >= 1e-7:
        raise AssertionError(
            f"MAP per-z dN/dX from f_bk_coarse vs e0['dndx_z'] mismatch: {cerr:.2e}")

    # ---- the per-z marginalized band (forward Stage III via joint_mc_errors + the
    # per-draw v3x refit_fn — the EXACT recipe track_c_td_band.run_forward_band uses).
    # The SHARED tmr ties the C/ρ/g draws to the Stage-III forward refit. joint_mc_errors
    # is serial (no pickling); it collects the genuine per-z f_bk_coarse + dndx_z per draw.
    tmr = build_truth_match_resample(
        ing["mm"], ing["cat_cut"], ing["is_TP"], ing["truth_cut"], ing["good_mask"], cfg)
    refit_fn = make_v3x_refit_fn(cfg, e0["_v3x"], ing["mm"],
                                 cat_cut=ing["cat_cut"], good_mask=ing["good_mask"], tmr=tmr)
    cfg.n_mc = args.n_mc
    mc = joint_mc_errors(
        ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["mm"], ing["fp_model"],
        ing["X_tot"], logN_lo, logN_hi, N_b, dN_b, ing["truth_cut"],
        cfg, np.random.default_rng(seed + 4), refit_fn=refit_fn, tmr=tmr)
    fbk_samp = np.asarray(mc["_samples"]["f_bk_coarse"], float)   # (n_mc, n_nbins, n_zc)
    fb_samp = np.asarray(mc["_samples"]["f_b"], float)            # (n_mc, n_nbins) z-marg
    dndx_z_samp = {l: np.asarray(mc["_samples"]["dndx_z"][l], float) for l in limits}
    print(f"    per-z band done ({time.time()-t0:.0f}s, n_mc={fbk_samp.shape[0]})")

    # per-z integral samples from the genuine 2-D f (dN/dX from f_bk MUST equal the
    # canonical per-draw dndx_z the estimator stored; assert it — the per-z analogue of
    # hbi_fNz_coverage's consistency gate). Ω(z) has no stored per-draw analogue, so it
    # is built from the SAME f_bk (tied by construction to dN/dX).
    n_draw = fbk_samp.shape[0]
    dndx_samp = {l: np.stack([perz_dndx_from_fbk(fbk_samp[m], logN_lo, dN_b, l)
                              for m in range(n_draw)], axis=0)
                 for l in limits}                                # each (n_mc, n_zc)
    omega_samp = {l: np.stack([perz_omega_from_fbk(fbk_samp[m], logN_lo, N_b, dN_b, K, l)
                               for m in range(n_draw)], axis=0)
                  for l in limits}
    band_cerr = 0.0
    for l in limits:
        a = dndx_samp[l]; b = dndx_z_samp[l]
        good = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-30)
        if good.any():
            band_cerr = max(band_cerr, float(np.max(np.abs(a[good] - b[good])
                                                    / np.abs(b[good]))))
    if band_cerr >= 1e-7:
        raise AssertionError(
            f"per-draw dN/dX(z) from f_bk_coarse vs stored dndx_z mismatch: {band_cerr:.2e}")
    cerr = max(cerr, band_cerr)

    # truth f(N,z) + truth per-z integrals
    tf = truth_fNz(cfg, ing["truth_cut"], logN_lo, logN_hi, dN_b, ing["X_tot"])
    f_truth = tf["f_truth"]
    tr = truth_perz_integrals(cfg, f_truth, logN_lo, N_b, dN_b, limits)

    return dict(
        cfg=cfg, H0=cfg.H0, K=K, zbins=zbins, n_zc=n_zc,
        logN_lo=logN_lo, logN_hi=logN_hi, N_b=N_b, dN_b=dN_b,
        mid=0.5 * (logN_lo + logN_hi),
        map_fbk=map_fbk, map_dndx=map_dndx, map_omega=map_omega,
        dndx_samp=dndx_samp, omega_samp=omega_samp,
        fbk_samp=fbk_samp, fb_samp=fb_samp,
        f_truth=f_truth, truth_dndx=tr["dndx"], truth_omega=tr["omega"],
        consistency_err=float(cerr), n_mc=int(fbk_samp.shape[0]),
    )


# -----------------------------------------------------------------------------
# assemble per-z coverage (dN/dX(z), Ω(z) at each limit) + Ω slope-extrap
# -----------------------------------------------------------------------------
def assemble_coverage(res, args, limits, seed):
    cfg = res["cfg"]; n_zc = res["n_zc"]
    recenter = bool(getattr(cfg, "band_recenter", False))
    slope_extrap = bool(getattr(cfg, "omega_slope_extrap", False))
    omega_se_int = (slope_extrap
                    and bool(getattr(cfg, "omega_slope_extrap_integrated", False)))
    cov = dict(dndx={}, omega={})
    for l in limits:
        cov["dndx"][str(l)] = []
        cov["omega"][str(l)] = []
        for k in range(n_zc):
            # dN/dX(z): recentered band on the MAP per-z point
            pt = float(res["map_dndx"][l][k])
            band = _band(res["dndx_samp"][l][:, k], point=pt, recenter=recenter)
            tv = float(res["truth_dndx"][l][k])
            c = _cover(band, tv)
            r0 = (pt / tv) if tv > 0 else np.nan
            cov["dndx"][str(l)].append(dict(
                z_idx=k, MAP=pt, MAP_R0=float(r0), truth=tv,
                band68=[band["q16"], band["q84"]], band95=[band["q025"], band["q975"]],
                cover68=c["in68"], cover95=c["in95"]))

            # Ω(z): recentered band; optionally splice the high-N slope-extrap per z
            pt_o = float(res["map_omega"][l][k])
            samp_o = res["omega_samp"][l][:, k]
            se_note = False
            if omega_se_int:
                # per-z column of the f_bk samples (n_mc, n_nbins) -> integrated-Ω
                # slope-extrap shoulder, same machinery as the headline, applied per z.
                col_samp = res["fbk_samp"][:, :, k]               # (n_mc, n_nbins)
                col_point = res["map_fbk"][:, k]
                samp_o2, pt_in_data = omega_integrated_slope_extrap_samples(
                    col_samp, col_point, res["logN_lo"], res["logN_hi"],
                    res["N_b"], res["dN_b"], cfg,
                    np.random.default_rng(seed + 77 + 100 * k + int(round(l * 10))), l)
                # the in-data integral must reproduce the per-z MAP Ω (byte-identical point)
                if np.isfinite(pt_in_data) and np.isfinite(pt_o):
                    assert np.isclose(pt_in_data, pt_o, rtol=1e-7), (
                        f"per-z Ω(>={l}) in-data integral {pt_in_data} != MAP {pt_o} "
                        f"(z bin {k})")
                samp_o = samp_o2
                se_note = True
            band_o = _band(samp_o, point=pt_o, recenter=recenter)
            tv_o = float(res["truth_omega"][l][k])
            c_o = _cover(band_o, tv_o)
            r0_o = (pt_o / tv_o) if tv_o > 0 else np.nan
            cov["omega"][str(l)].append(dict(
                z_idx=k, MAP=pt_o, MAP_R0=float(r0_o), truth=tv_o,
                band68=[band_o["q16"], band_o["q84"]],
                band95=[band_o["q025"], band_o["q975"]],
                cover68=c_o["in68"], cover95=c_o["in95"],
                slope_extrap_shoulder=bool(se_note)))
    cov["_meta"] = dict(band_recenter=recenter, omega_slope_extrap=slope_extrap,
                        omega_slope_extrap_integrated=omega_se_int)
    return cov


# -----------------------------------------------------------------------------
# the figure
# -----------------------------------------------------------------------------
def make_figure(out_path, res, cov, limits, args):
    """fig_track_c_perz.png — columns = coarse z bins; rows = {dN/dX(z), Ω(z),
    f(N | z)}. Each panel: the MAP point, the 68/95 recentered band, the 2LPT-0 truth,
    coverage clear. Style matched to the headline (CDDF + dN/dX + Ω), resolved in z."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zbins = res["zbins"]; n_zc = res["n_zc"]
    zmid = 0.5 * (zbins[:-1] + zbins[1:])
    mid = res["mid"]
    f_truth = res["f_truth"]; map_fbk = res["map_fbk"]; fbk_samp = res["fbk_samp"]
    recenter = bool(getattr(res["cfg"], "band_recenter", False))
    C_MAP = "#1f77b4"; C_BAND68 = "#1f77b4"; C_BAND95 = "#aec7e8"; C_TRUTH = "k"
    C_DLA = {20.0: "#2ca02c", 20.3: "#d62728"}

    fig, axes = plt.subplots(3, n_zc, figsize=(4.6 * n_zc, 11.5), squeeze=False)

    # ---- ROW 0: dN/dX(z) integrated bars at each limit ----
    for k in range(n_zc):
        ax = axes[0, k]
        xs = np.arange(len(limits))
        for j, l in enumerate(limits):
            c = cov["dndx"][str(l)][k]
            color = C_DLA.get(l, "#555555")
            lo68, hi68 = c["band68"]; lo95, hi95 = c["band95"]
            ax.vlines(xs[j], lo95, hi95, color=color, lw=2.5, alpha=0.45,
                      label=("95% band" if (k == 0 and j == 0) else None))
            ax.vlines(xs[j], lo68, hi68, color=color, lw=7.0, alpha=0.55,
                      label=("68% band" if (k == 0 and j == 0) else None))
            ax.plot(xs[j], c["MAP"], "o", color=color, ms=8, mec="k", mew=0.8,
                    label=(f"MAP" if (k == 0 and j == 0) else None), zorder=5)
            ax.plot(xs[j], c["truth"], "*", color=C_TRUTH, ms=15, zorder=6,
                    label=("2LPT-0 truth" if (k == 0 and j == 0) else None))
            tag = "cover" if c["cover68"] else ("95" if c["cover95"] else "MISS")
            ax.annotate(f"R0={c['MAP_R0']:.2f}\n{tag}",
                        (xs[j], max(hi68, c["truth"])), textcoords="offset points",
                        xytext=(0, 8), fontsize=7.5, ha="center",
                        color=("k" if c["cover68"] else "red"))
        ax.set_xticks(xs)
        ax.set_xticklabels([rf"$\geq{l:.1f}$" for l in limits])
        ax.set_xlim(-0.6, len(limits) - 0.4)
        ax.set_title(rf"$z \approx {zmid[k]:.2f}$  ([{zbins[k]:.1f},{zbins[k+1]:.1f}])",
                     fontsize=12)
        if k == 0:
            ax.set_ylabel(r"$dN/dX\,(z)$", fontsize=12)
            ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
        ax.margins(y=0.25)

    # ---- ROW 1: Ω(z) integrated bars at each limit ----
    for k in range(n_zc):
        ax = axes[1, k]
        xs = np.arange(len(limits))
        for j, l in enumerate(limits):
            c = cov["omega"][str(l)][k]
            color = C_DLA.get(l, "#555555")
            lo68, hi68 = c["band68"]; lo95, hi95 = c["band95"]
            ax.vlines(xs[j], lo95, hi95, color=color, lw=2.5, alpha=0.45)
            ax.vlines(xs[j], lo68, hi68, color=color, lw=7.0, alpha=0.55)
            ax.plot(xs[j], c["MAP"], "o", color=color, ms=8, mec="k", mew=0.8, zorder=5)
            ax.plot(xs[j], c["truth"], "*", color=C_TRUTH, ms=15, zorder=6)
            tag = "cover" if c["cover68"] else ("95" if c["cover95"] else "MISS")
            ax.annotate(f"R0={c['MAP_R0']:.2f}\n{tag}",
                        (xs[j], max(hi68, c["truth"])), textcoords="offset points",
                        xytext=(0, 8), fontsize=7.5, ha="center",
                        color=("k" if c["cover68"] else "red"))
        ax.set_xticks(xs)
        ax.set_xticklabels([rf"$\geq{l:.1f}$" for l in limits])
        ax.set_xlim(-0.6, len(limits) - 0.4)
        if k == 0:
            ax.set_ylabel(r"$\Omega_{\rm HI}\,(z)$", fontsize=12)
        ax.grid(alpha=0.25)
        ax.margins(y=0.25)

    # ---- ROW 2: differential f(N | z) over the DLA range ----
    xlo, xhi = 20.0, 22.0
    pmask = (mid >= xlo - 1e-9) & (mid <= xhi + 1e-9)
    lo68 = np.nanpercentile(fbk_samp, 16, axis=0)   # (n_nbins, n_zc)
    hi68 = np.nanpercentile(fbk_samp, 84, axis=0)
    lo95 = np.nanpercentile(fbk_samp, 2.5, axis=0)
    hi95 = np.nanpercentile(fbk_samp, 97.5, axis=0)
    med = np.nanpercentile(fbk_samp, 50, axis=0)
    for k in range(n_zc):
        ax = axes[2, k]
        # per-bin recenter on the MAP f(N|z)
        l68, h68, l95, h95 = lo68[:, k].copy(), hi68[:, k].copy(), lo95[:, k].copy(), hi95[:, k].copy()
        if recenter:
            shift = map_fbk[:, k] - med[:, k]
            l68 += shift; h68 += shift; l95 += shift; h95 += shift
        m = pmask & np.isfinite(map_fbk[:, k]) & (map_fbk[:, k] > 0)
        ax.fill_between(mid[m], np.clip(l95[m], 1e-30, None), np.clip(h95[m], 1e-30, None),
                        color=C_BAND95, alpha=0.55, lw=0,
                        label=("HBI 95%" if k == 0 else None))
        ax.fill_between(mid[m], np.clip(l68[m], 1e-30, None), np.clip(h68[m], 1e-30, None),
                        color=C_BAND68, alpha=0.30, lw=0,
                        label=("HBI 68%" if k == 0 else None))
        ax.plot(mid[m], np.clip(map_fbk[m, k], 1e-30, None), "-", color=C_MAP, lw=1.6,
                label=("HBI MAP" if k == 0 else None))
        ft = f_truth[:, k]
        mt = pmask & np.isfinite(ft) & (ft > 0)
        ax.plot(mid[mt], ft[mt], "*", color=C_TRUTH, ms=11, ls="none",
                label=("2LPT-0 truth" if k == 0 else None))
        ax.plot(mid[mt], ft[mt], "-", color=C_TRUTH, lw=0.7, alpha=0.45)
        ax.set_yscale("log")
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel(r"$\log_{10} N_{\rm HI}$", fontsize=11)
        if k == 0:
            ax.set_ylabel(r"$f(N\,|\,z)$", fontsize=12)
            ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.25, which="both")
        # annotate the 20.3-21.5 shoulder skew (MAP/truth) — the z-dependence the PI asked
        sh = []
        for b in np.where(mt)[0]:
            if 20.3 <= mid[b] <= 21.5 and ft[b] > 0 and np.isfinite(map_fbk[b, k]):
                sh.append(map_fbk[b, k] / ft[b])
        if sh:
            ax.text(0.97, 0.95, rf"shoulder MAP/truth$={np.nanmedian(sh):.2f}$",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="w", ec="0.6", alpha=0.85))

    fig.suptitle(
        "Track-C forward-kernel recovery resolved in redshift — 2LPT-0 mock\n"
        "(forward-empirical kernel, recentered 68/95 band;  ● MAP   ★ 2LPT-0 truth)",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    print(f"[perz] figure -> {out_path}")


# -----------------------------------------------------------------------------
# report (the per-z coverage table) + cross-z shoulder summary
# -----------------------------------------------------------------------------
def write_report(out_path, res, cov, limits, args, wallclock, headline_limit=20.3):
    zbins = res["zbins"]; n_zc = res["n_zc"]
    zmid = 0.5 * (zbins[:-1] + zbins[1:])
    mid = res["mid"]; f_truth = res["f_truth"]; map_fbk = res["map_fbk"]
    L = []
    L.append("# Track-C per-z forward-kernel band — 2LPT-0 mock")
    L.append("")
    L.append(f"- Status: COMPLETE.  n_mc = {res['n_mc']}   wallclock = {wallclock:.0f}s "
             f"({wallclock/60:.1f} min)   workers = {args.workers}")
    L.append(f"- Forward model: `{args.forward_model}`  (resp_family={args.resp_family})")
    L.append(f"- Stages: I=laplace, II=shared_boot, III=marginalize(forward refit); "
             f"band_recenter={bool(getattr(res['cfg'],'band_recenter',False))}, "
             f"omega_slope_extrap={bool(getattr(res['cfg'],'omega_slope_extrap',False))} "
             f"(integrated_shoulder={cov['_meta']['omega_slope_extrap_integrated']})")
    L.append(f"- Consistency gate (MAP per-z dN/dX from f_bk_coarse vs e0.dndx_z): "
             f"max rel err = {res['consistency_err']:.2e}  (< 1e-7)")
    L.append("- Inference (gpy_dla_detection/) byte-FROZEN; estimator (cddf_catalog_hbi.py) "
             "and band driver (track_c_td_band.py) NOT edited; this script IMPORTS them.")
    L.append("")
    L.append("## Per-z coverage table (dN/dX(z) & Ω(z) at NHI ≥ 20.3)")
    L.append("")
    L.append("| reduction | z bin | z≈ | MAP | truth | MAP R0 | 68% band | cover68? |")
    L.append("|---|---|---|---|---|---|---|---|")
    hl = headline_limit
    for kind in ("dndx", "omega"):
        for k in range(n_zc):
            c = cov[kind][str(hl)][k]
            cv = {True: "yes", False: "**MISS**", None: "—"}[c["cover68"]]
            name = "dN/dX(z)" if kind == "dndx" else "Ω_HI(z)"
            L.append(f"| {name} | [{zbins[k]:.1f},{zbins[k+1]:.1f}] | {zmid[k]:.2f} | "
                     f"{c['MAP']:.4g} | {c['truth']:.4g} | {c['MAP_R0']:.3f} | "
                     f"[{c['band68'][0]:.4g}, {c['band68'][1]:.4g}] | {cv} |")
    L.append("")
    # also the >=20.0 table for completeness
    L.append("## Per-z coverage table (dN/dX(z) & Ω(z) at NHI ≥ 20.0)")
    L.append("")
    L.append("| reduction | z bin | z≈ | MAP | truth | MAP R0 | 68% band | cover68? |")
    L.append("|---|---|---|---|---|---|---|---|")
    for kind in ("dndx", "omega"):
        for k in range(n_zc):
            c = cov[kind]["20.0"][k]
            cv = {True: "yes", False: "**MISS**", None: "—"}[c["cover68"]]
            name = "dN/dX(z)" if kind == "dndx" else "Ω_HI(z)"
            L.append(f"| {name} | [{zbins[k]:.1f},{zbins[k+1]:.1f}] | {zmid[k]:.2f} | "
                     f"{c['MAP']:.4g} | {c['truth']:.4g} | {c['MAP_R0']:.3f} | "
                     f"[{c['band68'][0]:.4g}, {c['band68'][1]:.4g}] | {cv} |")
    L.append("")
    # cross-z shoulder skew (does the forward kernel flatten the 0.87->1.18 kappa trend?)
    L.append("## z-dependence of the 20.3–21.5 shoulder over-recovery (MAP/truth)")
    L.append("")
    L.append("The kappa kernel showed a RISING shoulder skew across z (~0.87 → ~1.18). "
             "Below is the forward-kernel shoulder MAP/truth per z bin (median over the "
             "[20.3,21.5] f(N|z) bins) and the high-N tail [≥21.0].")
    L.append("")
    L.append("| z≈ | shoulder[20.3,21.5] MAP/truth | tail[≥21.0] MAP/truth | dN/dX R0(≥20.3) |")
    L.append("|---|---|---|---|")
    shoulder_vals = []
    for k in range(n_zc):
        sh, tl = [], []
        ft = f_truth[:, k]
        for b in range(len(mid)):
            if not (np.isfinite(ft[b]) and ft[b] > 0 and np.isfinite(map_fbk[b, k])):
                continue
            r = map_fbk[b, k] / ft[b]
            if 20.3 <= mid[b] <= 21.5:
                sh.append(r)
            if mid[b] >= 21.0:
                tl.append(r)
        shv = (np.nanmedian(sh) if sh else np.nan)
        tlv = (np.nanmedian(tl) if tl else np.nan)
        shoulder_vals.append(shv)
        r0 = cov["dndx"][str(hl)][k]["MAP_R0"]
        L.append(f"| {zmid[k]:.2f} | {shv:.3f} | {tlv:.3f} | {r0:.3f} |")
    L.append("")
    # flatten verdict
    finite_sh = [v for v in shoulder_vals if np.isfinite(v)]
    if len(finite_sh) >= 2:
        spread = max(finite_sh) - min(finite_sh)
        kappa_spread = 1.18 - 0.87
        L.append(f"- Forward-kernel shoulder MAP/truth spread across z = "
                 f"**{spread:.3f}** (min {min(finite_sh):.3f} → max {max(finite_sh):.3f}); "
                 f"the kappa kernel's was ~{kappa_spread:.2f} (0.87 → 1.18). "
                 + ("The forward kernel **flattens** the z-trend."
                    if spread < 0.6 * kappa_spread else
                    "The forward kernel does **not** materially flatten the z-trend."))
    L.append("")
    L.append(f"- Figure: `{os.path.join(args.out, 'fig_track_c_perz.png')}`")
    L.append(f"- JSON:   `{os.path.join(args.out, 'track_c_perz_coverage.json')}`")
    with open(out_path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[perz] report -> {out_path}")
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir", default=AB.DEF_CAT)
    p.add_argument("--truth", default=AB.DEF_TRUTH)
    p.add_argument("--bal-cat", default=AB.DEF_BAL)
    p.add_argument("--molly-tsv", default=AB.DEF_LYAONLY_MOLLY)
    p.add_argument("--kernel", default=AB.DEF_KERNEL)
    p.add_argument("--loa0-product", default=AB.DEF_LOA0_PRODUCT)
    p.add_argument("--forward-model", default=_DEF_FORWARD)
    p.add_argument("--resp-family", default="empirical", choices=["skewnorm", "empirical"])
    p.add_argument("--out", default=_DEF_OUT)
    p.add_argument("--report-out", default=".superpowers/sdd/track_c_perz_report.md",
                   help="path for the markdown report (also written under --out)")
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3")
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--fit-ceil", type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes", type=int, default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    p.add_argument("--n-mc", type=int, default=120)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    # Track-C BAND-FINALIZE knobs (default ON — the central recipe at HEAD)
    p.add_argument("--band-recenter", dest="band_recenter", action="store_true", default=True)
    p.add_argument("--no-band-recenter", dest="band_recenter", action="store_false")
    p.add_argument("--omega-slope-extrap", dest="omega_slope_extrap",
                   action="store_true", default=True)
    p.add_argument("--no-omega-slope-extrap", dest="omega_slope_extrap",
                   action="store_false")
    p.add_argument("--omega-slope-extrap-integrated", dest="omega_slope_extrap_integrated",
                   action="store_true", default=True,
                   help="splice the per-z high-N slope-extrap into the integrated Ω(z) band")
    p.add_argument("--no-omega-slope-extrap-integrated",
                   dest="omega_slope_extrap_integrated", action="store_false")
    p.add_argument("--slope-edge", type=float, default=21.2)
    p.add_argument("--slope-fit-dex", type=float, default=0.6)
    p.add_argument("--sigma-slope", type=float, default=0.5)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    limits = tuple(float(x) for x in args.report_limits.split(","))

    t0 = time.time()
    print("=" * 78)
    print("TRACK-C PER-Z — forward-empirical kernel band, resolved in redshift (2LPT-0)")
    print(f"  forward model: {args.forward_model}  family={args.resp_family}")
    print(f"  z bins: {args.zbins}   limits: {args.report_limits}   n_mc={args.n_mc}")
    print(f"  Stage I=laplace  Stage II=shared_boot  Stage III=marginalize(forward refit)")
    print(f"  band_recenter={args.band_recenter}  omega_slope_extrap={args.omega_slope_extrap}"
          f"  integrated_shoulder={args.omega_slope_extrap_integrated}")
    print("=" * 78)

    res = run_perz(args, limits, args.seed)
    cov = assemble_coverage(res, args, limits, args.seed)
    wallclock = time.time() - t0

    # JSON
    json_path = os.path.join(args.out, "track_c_perz_coverage.json")
    out_json = dict(
        metadata=dict(
            forward_model=args.forward_model, resp_family=args.resp_family,
            n_mc=res["n_mc"], seed=args.seed, workers=args.workers,
            limits=list(limits), zbins=list(map(float, res["zbins"])),
            wallclock_s=float(wallclock), consistency_err=res["consistency_err"],
            band_recenter=cov["_meta"]["band_recenter"],
            omega_slope_extrap=cov["_meta"]["omega_slope_extrap"],
            omega_slope_extrap_integrated=cov["_meta"]["omega_slope_extrap_integrated"],
            stages="I=laplace, II=shared_boot, III=marginalize(forward refit)",
            kernel=args.kernel, molly=args.molly_tsv, truth=args.truth),
        coverage=dict(dndx=cov["dndx"], omega=cov["omega"]))
    with open(json_path, "w") as fh:
        json.dump(out_json, fh, indent=2)
    print(f"[perz] json -> {json_path}")

    # figure
    fig_path = os.path.join(args.out, "fig_track_c_perz.png")
    make_figure(fig_path, res, cov, limits, args)

    # report (under --out AND at --report-out)
    rep_under_out = os.path.join(args.out, "track_c_perz_report.md")
    txt = write_report(rep_under_out, res, cov, limits, args, wallclock)
    if args.report_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
        with open(args.report_out, "w") as fh:
            fh.write(txt + "\n")
        print(f"[perz] report -> {args.report_out}")

    print("\n" + txt)
    print(f"\n[perz] DONE in {wallclock:.0f}s  (consistency {res['consistency_err']:.2e})")
    return dict(res=res, cov=cov)


if __name__ == "__main__":
    main()
