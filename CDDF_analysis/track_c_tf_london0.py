#!/usr/bin/env python
"""track_c_tf_london0.py — Track-C T-F LEG 2: the cross-RECIPE non-circular proof.

THE TEST.  Apply the FROZEN 2LPT-0 Track-C recipe (forward-response kernel +
z-resolved completeness C·g(N,z) + band machinery, CALIBRATED ONLY on 2LPT-0) to
the HELD-OUT london-0 catalog WITHOUT any refit, and check whether the recovered
dN/dX(z) / Ω(z) match london-0's OWN truth (R0 = recovered / london-0-truth ≈ 1).

LEG 2 vs LEG 1.  Leg 1 (track_c_tf_2lpt1.py) held out 2lpt-1 — the SAME mock recipe,
a different realization (the gentlest transfer test). Leg 2 holds out london-0 — a
DIFFERENT mock recipe (CoLoRe/london v5.9.5 vs lyacolore_2lpt v2.8.5). The forest
mean-flux normalization, the HCD/DLA injection prescription, and the noise model all
differ between recipes, so this is the STRONGEST transfer test: it probes whether the
2LPT-0-calibrated forward response generalizes across the recipe boundary, not just
across realizations.

EXPECTATION (stated up front, honest).  The forward-response kernel is a GP-SET
property (it encodes how the GP maps a true (N,z) absorber into a detected (N̂,ẑ) +
P_DLA), and the GP model is IDENTICAL across these catalogs (same learned_file, same
inference config — see the held-out dataset memory). So the kernel SHAPE should
transfer. BUT the tail WEIGHT (how many high-N absorbers exist, and how the forest
mean-flux modulates detectability) is recipe-dependent. A clean, expected outcome is
therefore: R0(z) ≈ 1 up to an OVERALL mean-flux amplitude offset — i.e. the shape
transfers and a single scalar rescale restores R0 ≈ 1. A large UNEXPLAINED per-z
deviation (or a shape mismatch the A/B decomposition pins on the kernel) would be a
real finding.

What is FROZEN vs HELD-OUT (the crux — strict, IDENTICAL to leg 1):
  * Forward-response kernel  → FROZEN  (forward_response_2lpt0.npz; never re-fit).
  * z-resolved completeness g(N,z)  → FROZEN  (built ONCE on 2LPT-0's truth-match;
    stashed onto cfg._cnz_resolved BEFORE the london-0 cut so ensure_cnz_resolved
    returns it unchanged — NOT rebuilt on london-0).
  * molly C/ρ ratio matrix (the .tsv)  → FROZEN  (the same 2LPT-0 lya_only-nhi195
    matrix is the default for BOTH mocks; it is a file, not a per-catalog rebuild).
  * The MC-resampling C/ρ COUNT denominators (mm.pur_ntp, cmp_nfound, …):
      - VARIANT A (fully frozen): FROZEN from 2LPT-0 (real-data-applicable — real
        LOA has no truth to rebuild the count denominators).
      - VARIANT B (kernel+g frozen, molly counts rebuilt on london-0): regenerated
        on london-0 (keeps the band's Poisson C/ρ jitter matched to the held-out
        catalog's own cell occupancy). A/B localizes whether the COMPLETENESS
        transfers vs only the kernel.
  * Band machinery (recenter, slope-extrap, Stage I/II/III) — as in the headline.
  * The london-0 CATALOG, TRUTH (R0 scoring ONLY) and QSO/pathlength — held-out's own.

DATA NOTES (london-0 specifics — adapted vs leg 1):
  * london truth file is `dla_cat.fits` (NOT `hcd_truth_cat.fits`), cols
    [NHI, Z_DLA, TARGETID, DLAID]. The HBI truth loader already aliases Z_DLA→Z_TRUTH
    and does NOT require a per-DLA truth SNR column, so it threads unchanged.
  * london mockdir has NO snr_cat.fits. The production-correct SNR_REDSIDE (the cut
    variable) lives in the catalog's processed-spectra-16-*.h5 `snrs` field (verified
    byte-identical to the dlacat SNR_REDSIDE, 20000/20000). A consolidated snr_cat.fits
    is pre-built from those h5 via examples/make_snr_cat_from_processed.py and staged in
    the held-out mockdir; it covers EXACTLY the 926,122 zcat QSOs in z∈(2.0,4.25) =
    100% of the pathlength population (the ~290k "missing" zcat rows are out-of-z-range
    QSOs the pathlength excludes anyway). So the pathlength scope is COMPLETE.

The cached row-indexed posterior kappa (posterior_kernel_2lpt0.npz) is NOT used:
resp_kind='forward' builds A_ib from the forward model (cell-keyed, transfers), and
it is row-aligned to the 2LPT-0 catalog so it CANNOT be attached to london-0. We
deliberately do NOT set cfg._posterior_kernel_2d (else the op-row-count assert fires).

Reduce-only / analysis-side. NO GP inference (gpy_dla_detection/ byte-FROZEN).
No estimator-logic edit (cddf_catalog_hbi.py untouched). conda gpdla; BLAS pinned.

Usage:
  python CDDF_analysis/track_c_tf_london0.py --variant both --n-mc 120 --workers 4
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis import ab_loa0_fp_baseline as AB
from CDDF_analysis import track_c_perz_band as PZ
from CDDF_analysis.cddf_catalog_hbi import (
    HBIConfig, load_molly_matrix, load_and_cut_catalog, build_fine_grid,
    regenerate_molly_counts, make_C_interpolator, build_pathlength,
    make_fp_model, make_rho_interpolator, _build_qso_lookup, v3x_refit,
    build_cnz_resolved,
)
import functools

# ---------------------------------------------------------------------------
# FROZEN 2LPT-0 recipe artifacts (the calibration; never re-fit on london-0)
# ---------------------------------------------------------------------------
_DEF_FORWARD = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                "track_c/stage0/forward_response_2lpt0.npz")

# 2LPT-0 (CALIBRATION mock) — the catalog/truth/bal the frozen g + molly counts come from
_C0_CAT = AB.DEF_CAT
_C0_TRUTH = AB.DEF_TRUTH
_C0_BAL = AB.DEF_BAL

# london-0 (HELD-OUT mock) — the catalog/truth/QSO scored against, NO refit.
_L0_BASE = "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/london0_jura124_v1"
# load_catalog_dir globs dlacat-*.fits inside the DIRECTORY (not a file path).
_L0_CAT = _L0_BASE
# the london mockdir holds the truth (dla_cat.fits) + zcat; snr_cat.fits is the
# consolidated SNR catalog built from the catalog's processed h5 and staged here (the
# london mock has no native snr_cat). zcat/dla_cat/bal_cat are symlinks to the read-only
# london mock; snr_cat.fits is a real file on writable scratch.
_L0_MOCKDIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "track_c/tf_london0/mockdir")
_L0_TRUTH = _L0_MOCKDIR + "/dla_cat.fits"        # london truth (NOT hcd_truth_cat.fits)
_L0_BAL = _L0_MOCKDIR + "/bal_cat.fits"

_DEF_OUT = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_london0"


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _exists(p):
    """A catalog target may be a FITS file or a combined-catalog directory."""
    return os.path.exists(p)


# ---------------------------------------------------------------------------
# build the FROZEN 2LPT-0 completeness g(N,z) + (optionally) molly counts ONCE
# ---------------------------------------------------------------------------
def build_frozen_calibration(args):
    """Build the 2LPT-0 ingredients ONCE and extract the FROZEN recipe pieces:

      * g_cnz  : the z-resolved CNZModel g(N,z) (build_cnz_resolved on 2LPT-0's
                 truth-match — this is the SAME object Track-C #39 stashes).
      * molly_counts : the 2LPT-0 (pur_ntp, pur_ntot, cmp_nfound, cmp_nfid) count
                 denominators (for VARIANT A — the band's C/ρ Poisson jitter source).

    The molly C/ρ RATIO matrix and the forward kernel are files (already frozen);
    this only captures the two pieces that a naive rerun would otherwise rebuild on
    london-0. Returns a dict; this function reads ONLY 2LPT-0 inputs.
    """
    print("[T-F] building FROZEN 2LPT-0 calibration (g(N,z) + molly counts) ...")
    molly_tsv = AB._resolve_molly(args)            # the 2LPT-0 lya_only-nhi195 matrix
    cfg = HBIConfig(
        catalog_dir=_C0_CAT, truth_path=_C0_TRUTH, bal_cat_path=_C0_BAL,
        molly_tsv=molly_tsv, out_dir=args.out,
        mockdir=os.path.dirname(_C0_TRUTH),
        zbins=tuple(float(x) for x in args.zbins.split(",")),
        report_logN_limits=tuple(float(x) for x in args.report_limits.split(",")),
        fp_estimator="purity_mixture", no_bal=True,
        v3_family=args.family, v3_logN_fit_floor=args.fit_floor,
        v3_logN_fit_ceil=args.fit_ceil, v3_lambda_bspbody=args.lambda_bspbody,
        v3_mc_n_restart=2, lam_rf_min=args.lam_rf_min,
        v3_bspbody_edge_slope_lam=args.edge_slope_lam,
        v3_fine_density_gl_nodes=args.gl_nodes,
        v2_z_fit_lo=2.0, v2_z_fit_hi=3.5, v2_z_fit_step=0.1, rng_seed=0,
        completeness_z_resolved=True, completeness_z_min_count=float(args.cz_min_count),
    )
    mm = load_molly_matrix(molly_tsv)
    truth_floor = float(mm.nhi_edges[0])
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(args.host_truth_floor, truth_floor))
    mm = regenerate_molly_counts(mm, cat_cut, is_TP, truth_cut, good_mask, cfg)
    # the FROZEN z-resolved completeness g(N,z) (Track-C #39 object)
    g_cnz = build_cnz_resolved(cfg, cat_cut, truth_cut, good_mask, mm)
    molly_counts = dict(
        pur_ntp=np.array(mm.pur_ntp, float), pur_ntot=np.array(mm.pur_ntot, float),
        cmp_nfound=np.array(mm.cmp_nfound, float), cmp_nfid=np.array(mm.cmp_nfid, float),
        nhi_edges=np.array(mm.nhi_edges, float), snr_edges=np.array(mm.snr_edges, float),
    )
    print(f"[T-F] frozen g(N,z) shape={g_cnz.g_grid.shape}; molly counts captured "
          f"(C ratio matrix + forward kernel are frozen files).")
    return dict(g_cnz=g_cnz, molly_counts=molly_counts, molly_tsv=molly_tsv,
                c0_truth_floor=truth_floor)


# ---------------------------------------------------------------------------
# build the london-0 (HELD-OUT) ingredients with the FROZEN recipe injected
# ---------------------------------------------------------------------------
def build_heldout_ingredients(args, frozen, variant):
    """Mirror ab_loa0_fp_baseline.build_ingredients but:
      - point catalog/truth/bal/QSO at london-0,
      - inject the FROZEN g(N,z) onto cfg._cnz_resolved (NOT rebuilt on london-0),
      - VARIANT A: also overwrite the molly COUNT denominators with the frozen
        2LPT-0 ones (fully frozen); VARIANT B: regenerate them on london-0,
      - do NOT attach the row-indexed posterior kappa (forward path ignores it; the
        op-row-count assert would otherwise fire on the held-out catalog).
    """
    molly_tsv = frozen["molly_tsv"]               # frozen 2LPT-0 C/ρ ratio matrix
    cfg = HBIConfig(
        catalog_dir=args.heldout_cat, truth_path=args.heldout_truth,
        bal_cat_path=args.heldout_bal, molly_tsv=molly_tsv, out_dir=args.out,
        mockdir=args.heldout_mockdir or os.path.dirname(args.heldout_truth),
        zbins=tuple(float(x) for x in args.zbins.split(",")),
        report_logN_limits=tuple(float(x) for x in args.report_limits.split(",")),
        fp_estimator="purity_mixture", no_bal=True,
        v3_family=args.family, v3_logN_fit_floor=args.fit_floor,
        v3_logN_fit_ceil=args.fit_ceil, v3_lambda_bspbody=args.lambda_bspbody,
        v3_mc_n_restart=2, lam_rf_min=args.lam_rf_min,
        v3_bspbody_edge_slope_lam=args.edge_slope_lam,
        v3_fine_density_gl_nodes=args.gl_nodes,
        v2_z_fit_lo=2.0, v2_z_fit_hi=3.5, v2_z_fit_step=0.1, rng_seed=0,
        # the z-resolved completeness IS used (the headline recipe); g is FROZEN below.
        completeness_z_resolved=True, completeness_z_min_count=float(args.cz_min_count),
    )
    # IMPORTANT: do NOT set cfg._posterior_kernel_2d — the forward path does not read
    # it, and the 2LPT-0 kappa is row-aligned to the 2LPT-0 catalog (would mis-index).
    mm = load_molly_matrix(molly_tsv)
    truth_floor = float(mm.nhi_edges[0])
    qso_lookup = _build_qso_lookup(cfg)            # reads london-0 mockdir snr/zcat
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(args.host_truth_floor, truth_floor))

    # mm_resample carries london-0's OWN regenerated COUNTS — used ONLY by the MC band's
    # shared-resample basis (build_truth_match_resample), whose unit-weight reconstruction
    # MUST equal the counts derived from london-0's own cat_cut/truth_cut (the validate
    # assertion). The per-TID sightline bootstrap is over LONDON-0 sightlines, so its count
    # denominators are intrinsically london-0's — they CANNOT be the 2LPT-0 grafted counts
    # (those have a different TID structure). The frozen-recipe C/ρ RATIO still enters the
    # POINT via C_interp/rho_interp below; the band recenters on that point (band_recenter).
    mm_resample = regenerate_molly_counts(
        load_molly_matrix(molly_tsv), cat_cut, is_TP, truth_cut, good_mask, cfg)

    if variant == "A":
        # FULLY FROZEN (the POINT C/ρ ratio): take the molly count denominators from 2LPT-0.
        # This sets the POINT's C/ρ interpolators to the frozen 2LPT-0 RATIO matrix. The band
        # jitter uses mm_resample (london occupancy) and recenters on this frozen point.
        mc0 = frozen["molly_counts"]
        if not (np.allclose(mc0["nhi_edges"], mm.nhi_edges)
                and np.allclose(mc0["snr_edges"], mm.snr_edges)):
            raise SystemExit("frozen molly count grid != held-out molly grid (same TSV "
                             "expected) — cannot graft counts.")
        mm.pur_ntp = mc0["pur_ntp"].copy(); mm.pur_ntot = mc0["pur_ntot"].copy()
        mm.cmp_nfound = mc0["cmp_nfound"].copy(); mm.cmp_nfid = mc0["cmp_nfid"].copy()
        mm._max_p_diff = 0.0; mm._max_c_diff = 0.0
    else:
        # VARIANT B: regenerate the count denominators on the held-out london-0 catalog
        # (kernel + g still frozen). Localizes whether the COMPLETENESS COUNTS transfer.
        # The POINT C/ρ ratio now reflects london-0's own counts == mm_resample's counts.
        mm = mm_resample

    C_interp = make_C_interpolator(mm)
    rho_interp = make_rho_interpolator(mm)
    X_tot, n_sl, qzl, qzh, qsn, Xcalc = build_pathlength(
        cfg, qso_lookup=qso_lookup, return_per_sl=True)
    cfg.n_sl_prod = int(n_sl)
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)

    # FREEZE the completeness g(N,z): stash the 2LPT-0 model so ensure_cnz_resolved
    # (called by v3x_build_forward via cfg) returns it unchanged — NOT rebuilt here.
    # The frozen g grid lives on the molly nhi-cell × fine-z grid; both are identical
    # for 2LPT-0 and london-0 (same molly TSV, same cfg fine-z step) so it threads cleanly.
    g0 = frozen["g_cnz"]
    if not np.allclose(np.asarray(g0.nhi_edges, float), np.asarray(mm.nhi_edges, float)):
        raise SystemExit("frozen g(N,z) nhi_edges != held-out molly nhi_edges.")
    cfg._cnz_resolved = copy.deepcopy(g0)

    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op_mask = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    fp_model, _ = make_fp_model(cfg, cat_cut, op_mask, rho_interp)

    estimator_fn = functools.partial(
        v3x_refit, mm=mm, qso_per_sl=(qzl, qzh, qsn), Xcalc=Xcalc,
        rng=np.random.default_rng(0))
    print(f"  [variant {variant}] held-out london-0: n_op_sl={n_sl}, "
          f"frozen g shape={cfg._cnz_resolved.g_grid.shape}, kappa NOT attached.")
    return dict(cfg=cfg, mm=mm, mm_resample=mm_resample,
                cat_cut=cat_cut, truth_cut=truth_cut, is_TP=is_TP,
                good_mask=good_mask, C_interp=C_interp, fp_model=fp_model,
                X_tot=X_tot, n_sl=n_sl, logN_lo=logN_lo, logN_hi=logN_hi,
                N_b=N_b, dN_b=dN_b, estimator_fn=estimator_fn, meta=meta)


# ---------------------------------------------------------------------------
# the forward per-z band on the HELD-OUT ingredients (reuses PZ machinery)
# ---------------------------------------------------------------------------
def run_tf_variant(args, ing, limits, seed):
    """Run the forward-empirical per-z band on the held-out ingredients and score R0
    against london-0's OWN truth. Reuses the EXACT PZ (track_c_perz_band) band path so
    the recipe is bit-identical — only the ingredients (catalog/truth/QSO) differ and
    g/kernel are frozen. Returns (res, cov)."""
    cfg = ing["cfg"]
    cfg.report_logN_limits = limits
    cfg._wall1_estimator = "v3"
    cfg.n_mc = args.n_mc
    PZ._set_forward_cfg(cfg, args)

    logN_lo = ing["logN_lo"]; logN_hi = ing["logN_hi"]
    N_b = ing["N_b"]; dN_b = ing["dN_b"]
    from CDDF_analysis.cddf_catalog_hbi import (
        joint_mc_errors, make_v3x_refit_fn, v3x_reduce, build_truth_match_resample,
        omega_hi_prefactor,
    )
    from CDDF_analysis.ab_loa0_fp_baseline import run_baseline
    zbins = np.asarray(cfg.zbins, float)
    n_zc = len(zbins) - 1
    K = omega_hi_prefactor(cfg.H0)

    base = run_baseline(ing)
    e0 = base["e0"]
    # CANONICAL integrated (z-marginal) MAP R0 — tied to the headline ab_loa0 baseline:
    # e0/t0 dndx_total + omega and the R0_* ratios are the SAME reductions the headline
    # reports (truth_reductions for t0, v3x_reduce for e0). No hand reconstruction.
    integrated_point = dict(
        R0_dndx={l: float(base["R0_dndx_total"][l]) for l in limits},
        R0_omega={l: float(base["R0_omega"][l]) for l in limits},
        dndx_map={l: float(e0["dndx_total"][l]) for l in limits},
        dndx_truth={l: float(base["t0"]["dndx_total"][l]) for l in limits},
        omega_map={l: float(e0["omega"][l]) for l in limits},
        omega_truth={l: float(base["t0"]["omega"][l]) for l in limits},
    )
    fwd = e0["_v3x"]["fwd"]; family = e0["_v3x"]["family"]; fine = e0["_v3x"]["fine"]
    M_meta = e0["_v3x"]["M_meta"]; theta_map = e0["_v3x"]["theta_map"]
    rr_map = v3x_reduce(cfg, theta_map, fine, family, M_meta)
    map_fbk = np.asarray(rr_map["f_bk_coarse"], float)
    map_dndx = {l: PZ.perz_dndx_from_fbk(map_fbk, logN_lo, dN_b, l) for l in limits}
    map_omega = {l: PZ.perz_omega_from_fbk(map_fbk, logN_lo, N_b, dN_b, K, l)
                 for l in limits}
    cerr = 0.0
    for l in limits:
        a = map_dndx[l]; b = np.asarray(e0["dndx_z"][l], float)
        good = np.isfinite(b) & (np.abs(b) > 0)
        if good.any():
            cerr = max(cerr, float(np.max(np.abs(a[good] - b[good]) / np.abs(b[good]))))
    if cerr >= 1e-7:
        raise AssertionError(f"MAP per-z dN/dX vs e0.dndx_z mismatch: {cerr:.2e}")

    # The MC band's resample basis (tmr), per-draw response refit (refit_fn) and the
    # C/ρ-derivation (joint_mc_errors) form ONE coherent system over london-0's OWN cell
    # occupancy — so they ALL take mm_resample (london counts), NOT the grafted point mm.
    # The frozen-recipe C/ρ RATIO already entered the POINT via run_baseline(ing) above
    # (ing["C_interp"] = grafted-2LPT-0 in variant A); band_recenter=True then recenters the
    # band on that frozen point. Using ing["mm"] (grafted) here would fail the
    # build_truth_match_resample validate (a per-TID london bootstrap can't reproduce the
    # 2LPT-0 count denominators) and would mis-key the per-draw ρ matrix.
    mm_band = ing.get("mm_resample", ing["mm"])
    tmr = build_truth_match_resample(
        mm_band, ing["cat_cut"], ing["is_TP"], ing["truth_cut"], ing["good_mask"], cfg)
    refit_fn = make_v3x_refit_fn(cfg, e0["_v3x"], mm_band,
                                 cat_cut=ing["cat_cut"], good_mask=ing["good_mask"], tmr=tmr)
    cfg.n_mc = args.n_mc
    mc = joint_mc_errors(
        ing["cat_cut"], ing["is_TP"], ing["good_mask"], mm_band, ing["fp_model"],
        ing["X_tot"], logN_lo, logN_hi, N_b, dN_b, ing["truth_cut"],
        cfg, np.random.default_rng(seed + 4), refit_fn=refit_fn, tmr=tmr)
    fbk_samp = np.asarray(mc["_samples"]["f_bk_coarse"], float)
    fb_samp = np.asarray(mc["_samples"]["f_b"], float)
    dndx_z_samp = {l: np.asarray(mc["_samples"]["dndx_z"][l], float) for l in limits}

    n_draw = fbk_samp.shape[0]
    dndx_samp = {l: np.stack([PZ.perz_dndx_from_fbk(fbk_samp[m], logN_lo, dN_b, l)
                              for m in range(n_draw)], axis=0) for l in limits}
    omega_samp = {l: np.stack([PZ.perz_omega_from_fbk(fbk_samp[m], logN_lo, N_b, dN_b, K, l)
                               for m in range(n_draw)], axis=0) for l in limits}
    band_cerr = 0.0
    for l in limits:
        a = dndx_samp[l]; b = dndx_z_samp[l]
        good = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-30)
        if good.any():
            band_cerr = max(band_cerr,
                            float(np.max(np.abs(a[good] - b[good]) / np.abs(b[good]))))
    if band_cerr >= 1e-7:
        raise AssertionError(f"per-draw dN/dX(z) vs stored mismatch: {band_cerr:.2e}")
    cerr = max(cerr, band_cerr)

    # london-0's OWN truth f(N,z) + integrals (the R0 denominator)
    tf = PZ.truth_fNz(cfg, ing["truth_cut"], logN_lo, logN_hi, dN_b, ing["X_tot"])
    f_truth = tf["f_truth"]
    tr = PZ.truth_perz_integrals(cfg, f_truth, logN_lo, N_b, dN_b, limits)

    res = dict(
        cfg=cfg, H0=cfg.H0, K=K, zbins=zbins, n_zc=n_zc,
        logN_lo=logN_lo, logN_hi=logN_hi, N_b=N_b, dN_b=dN_b,
        mid=0.5 * (logN_lo + logN_hi),
        map_fbk=map_fbk, map_dndx=map_dndx, map_omega=map_omega,
        dndx_samp=dndx_samp, omega_samp=omega_samp,
        fbk_samp=fbk_samp, fb_samp=fb_samp,
        f_truth=f_truth, truth_dndx=tr["dndx"], truth_omega=tr["omega"],
        consistency_err=float(cerr), n_mc=int(fbk_samp.shape[0]),
        integrated_point=integrated_point,
    )
    cov = PZ.assemble_coverage(res, args, limits, seed)
    return res, cov


# ---------------------------------------------------------------------------
# mean-flux amplitude rescale diagnostic (the "expected story" probe)
# ---------------------------------------------------------------------------
def fit_meanflux_rescale(variants, limits):
    """The EXPECTED clean outcome (stated in the module docstring): the forward-response
    SHAPE transfers but the tail WEIGHT may need a single overall mean-flux amplitude
    rescale. Probe it: for each variant/limit, fit the SCALAR s that minimizes
    Σ_z (s·R0_dndx(z) − 1)^2 over the per-z dN/dX R0 (the amplitude that would restore
    R0≈1), then report the RESIDUAL per-z spread AFTER applying s. If a single scalar
    collapses the deviation to ~the band, that is the clean, expected cross-recipe story.

    Returns {variant: {limit: {s, pre_spread, post_spread, pre_rms, post_rms}}}.
    """
    out = {}
    for vk, V in variants.items():
        cov = V["cov"]; res = V["res"]; n_zc = res["n_zc"]
        out[vk] = {}
        for lim in limits:
            r0 = np.array([cov["dndx"][str(lim)][k]["MAP_R0"] for k in range(n_zc)],
                          float)
            good = np.isfinite(r0) & (r0 > 0)
            if good.sum() < 2:
                out[vk][lim] = dict(s=np.nan, pre_spread=np.nan, post_spread=np.nan,
                                    pre_rms=np.nan, post_rms=np.nan)
                continue
            rg = r0[good]
            # scalar s minimizing Σ (s·R0 − 1)^2  →  s = Σ R0 / Σ R0^2
            s = float(np.sum(rg) / np.sum(rg ** 2))
            post = s * rg
            out[vk][lim] = dict(
                s=s,
                pre_spread=float(rg.max() - rg.min()),
                post_spread=float(post.max() - post.min()),
                pre_rms=float(np.sqrt(np.mean((rg - 1.0) ** 2))),
                post_rms=float(np.sqrt(np.mean((post - 1.0) ** 2))),
            )
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(out_path, variants, args, wallclock, rescale):
    L = []
    L.append("# Track-C T-F LEG 2 — frozen-recipe CROSS-RECIPE proof on held-out london-0")
    L.append("")
    L.append(f"- Status: COMPLETE.  n_mc={args.n_mc}  seed={args.seed}  "
             f"wallclock={wallclock:.0f}s ({wallclock/60:.1f} min)")
    L.append(f"- Held-out mock RECIPE: london CoLoRe v5.9.5 (DIFFERENT recipe from the "
             f"2LPT-0/2lpt-1 lyacolore_2lpt v2.8.5 calibration — the STRONGEST transfer "
             f"test).")
    L.append(f"- Forward kernel (FROZEN, 2LPT-0): `{args.forward_model}`")
    L.append(f"- Completeness g(N,z) (FROZEN, 2LPT-0): built on 2LPT-0 truth-match, "
             f"stashed on cfg._cnz_resolved (NOT rebuilt on london-0).")
    L.append(f"- molly C/ρ ratio matrix (FROZEN, 2LPT-0): `{args.molly_tsv}`")
    L.append(f"- Held-out catalog (london-0): `{args.heldout_cat}`")
    L.append(f"- Held-out truth (R0 scoring only): `{args.heldout_truth}`  "
             f"(london `dla_cat.fits` — cols NHI/Z_DLA/TARGETID/DLAID)")
    L.append(f"- Held-out QSO/pathlength: `{args.heldout_mockdir}` "
             f"(snr_cat.fits built from the catalog's processed h5 `snrs`=SNR_REDSIDE, "
             f"100% of the z∈(2.0,4.25) pathlength population).")
    L.append(f"- Inference (gpy_dla_detection/) byte-FROZEN; no estimator-logic edit; "
             f"posterior kappa NOT attached (forward path). STAMP code_commit="
             f"`{_git_commit()}`.")
    L.append("")
    L.append("## VERDICT")
    L.append("")
    for vk, V in variants.items():
        cov = V["cov"]; res = V["res"]; itr = V["int_truth"]; ir0 = V["int_R0"]
        zbins = res["zbins"]; zmid = 0.5 * (zbins[:-1] + zbins[1:])
        label = ("A (fully frozen: 2LPT-0 molly counts + g + kernel)" if vk == "A"
                 else "B (kernel+g frozen; molly counts rebuilt on london-0)")
        L.append(f"### Variant {label}")
        L.append("")
        for lim in args._limits:
            L.append(f"**NHI ≥ {lim:.1f}** — R0 = recovered / london-0-truth")
            L.append("")
            L.append("| reduction | z bin | z≈ | MAP | london-0 truth | R0 | cover68? |")
            L.append("|---|---|---|---|---|---|---|")
            for kind, name in (("dndx", "dN/dX(z)"), ("omega", "Ω_HI(z)")):
                for k in range(res["n_zc"]):
                    c = cov[kind][str(lim)][k]
                    cv = {True: "yes", False: "**MISS**", None: "—"}[c["cover68"]]
                    L.append(f"| {name} | [{zbins[k]:.1f},{zbins[k+1]:.1f}] | "
                             f"{zmid[k]:.2f} | {c['MAP']:.4g} | {c['truth']:.4g} | "
                             f"{c['MAP_R0']:.3f} | {cv} |")
                # integrated row
                ir = ir0[kind][lim]
                L.append(f"| {name} | INTEGRATED | all | {ir['map']:.4g} | "
                         f"{itr[kind][lim]:.4g} | **{ir['R0']:.3f}** | — |")
            L.append("")
            # spread
            r0s = [cov["dndx"][str(lim)][k]["MAP_R0"] for k in range(res["n_zc"])]
            r0s = [x for x in r0s if np.isfinite(x)]
            if r0s:
                L.append(f"- dN/dX(z) R0 spread (≥{lim:.1f}): "
                         f"{max(r0s)-min(r0s):.3f} (min {min(r0s):.3f} → max {max(r0s):.3f})")
            # mean-flux rescale diagnostic
            rs = rescale.get(vk, {}).get(lim)
            if rs and np.isfinite(rs["s"]):
                L.append(f"- single mean-flux amplitude rescale s={rs['s']:.3f}: "
                         f"per-z dN/dX(z) R0 RMS-from-1 {rs['pre_rms']:.3f} → "
                         f"{rs['post_rms']:.3f}; spread {rs['pre_spread']:.3f} → "
                         f"{rs['post_spread']:.3f} (if post≪pre, the SHAPE transfers and a "
                         f"single scalar restores R0≈1 — the clean expected cross-recipe story).")
            L.append("")
    L.append("## A/B decomposition (kernel vs completeness transfer)")
    L.append("")
    if "A" in variants and "B" in variants:
        L.append("Variant A freezes EVERYTHING from 2LPT-0 (real-data-applicable). "
                 "Variant B rebuilds only the molly COUNT denominators on london-0. If A ≈ B, "
                 "the completeness COUNTS transfer (the recipe is mock-agnostic). If A "
                 "deviates from 1 but B recovers, the 2LPT-0 completeness COUNTS mis-transfer "
                 "(g shape OK, count level off). If BOTH deviate together, it is the KERNEL "
                 "(forward response) that is recipe-dependent — and the mean-flux rescale "
                 "diagnostic above tells whether a single scalar restores R0≈1.")
    else:
        L.append("(only one variant run — pass --variant both for the decomposition.)")
    L.append("")
    L.append(f"- Figure: `{os.path.join(args.out, 'fig_tf_london0.png')}`")
    with open(out_path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[T-F] report -> {out_path}")
    return "\n".join(L)


def make_figure(out_path, variants, args):
    """fig_tf_london0.png — 3-panel: dN/dX, Ω, CDDF; recovered band + london-0 truth.
    One column per variant; the headline variant A on top, B below if present."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vkeys = list(variants.keys())
    nrow = len(vkeys)
    fig, axes = plt.subplots(nrow, 3, figsize=(15, 4.6 * nrow), squeeze=False)
    hl = args._limits[-1] if args._limits else 20.3
    C_MAP = "#1f77b4"; C95 = "#aec7e8"; C68 = "#1f77b4"; CT = "k"
    for r, vk in enumerate(vkeys):
        res = variants[vk]["res"]; cov = variants[vk]["cov"]
        zbins = res["zbins"]; zmid = 0.5 * (zbins[:-1] + zbins[1:])
        mid = res["mid"]; n_zc = res["n_zc"]
        # panel 0: dN/dX(z) at the headline limit
        ax = axes[r, 0]
        for k in range(n_zc):
            c = cov["dndx"][str(hl)][k]
            lo68, hi68 = c["band68"]
            ax.vlines(zmid[k], lo68, hi68, color=C68, lw=7, alpha=0.5)
            ax.plot(zmid[k], c["MAP"], "o", color=C_MAP, ms=8, mec="k", zorder=5)
            ax.plot(zmid[k], c["truth"], "*", color=CT, ms=15, zorder=6)
            ax.annotate(f"R0={c['MAP_R0']:.2f}", (zmid[k], max(hi68, c["truth"])),
                        textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
        ax.set_xlabel("z"); ax.set_ylabel(rf"$dN/dX\,(z)\ (\geq{hl:.1f})$")
        ax.set_title(f"variant {vk}: dN/dX  (● MAP  ★ london-0 truth)")
        ax.grid(alpha=0.25); ax.margins(y=0.3)
        # panel 1: Ω(z)
        ax = axes[r, 1]
        for k in range(n_zc):
            c = cov["omega"][str(hl)][k]
            lo68, hi68 = c["band68"]
            ax.vlines(zmid[k], lo68, hi68, color=C68, lw=7, alpha=0.5)
            ax.plot(zmid[k], c["MAP"], "o", color=C_MAP, ms=8, mec="k", zorder=5)
            ax.plot(zmid[k], c["truth"], "*", color=CT, ms=15, zorder=6)
            ax.annotate(f"R0={c['MAP_R0']:.2f}", (zmid[k], max(hi68, c["truth"])),
                        textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
        ax.set_xlabel("z"); ax.set_ylabel(rf"$\Omega_{{\rm HI}}(z)\ (\geq{hl:.1f})$")
        ax.set_title(f"variant {vk}: Ω_HI"); ax.grid(alpha=0.25); ax.margins(y=0.3)
        # panel 2: CDDF f(N) z-marginal (sum over z of MAP/truth f_bk)
        ax = axes[r, 2]
        fbk_samp = res["fbk_samp"]; map_fbk = res["map_fbk"]; f_truth = res["f_truth"]
        # pathlength-weighted z-marginal f: Σ_k f[:,k]·X_k / ΣX_k
        X = np.asarray(variants[vk]["X_tot"], float)
        Xn = X / np.nansum(X)
        map_fb = np.nansum(map_fbk * Xn[None, :], axis=1)
        ft_fb = np.nansum(f_truth * Xn[None, :], axis=1)
        lo68 = np.nanpercentile(np.nansum(fbk_samp * Xn[None, None, :], axis=2), 16, axis=0)
        hi68 = np.nanpercentile(np.nansum(fbk_samp * Xn[None, None, :], axis=2), 84, axis=0)
        m = (mid >= 20.0) & (mid <= 22.0) & np.isfinite(map_fb) & (map_fb > 0)
        ax.fill_between(mid[m], np.clip(lo68[m], 1e-30, None),
                        np.clip(hi68[m], 1e-30, None), color=C95, alpha=0.5, label="HBI 68%")
        ax.plot(mid[m], np.clip(map_fb[m], 1e-30, None), "-", color=C_MAP, lw=1.6,
                label="HBI MAP")
        mt = (mid >= 20.0) & (mid <= 22.0) & np.isfinite(ft_fb) & (ft_fb > 0)
        ax.plot(mid[mt], ft_fb[mt], "*", color=CT, ms=10, ls="none", label="london-0 truth")
        ax.set_yscale("log"); ax.set_xlim(20.0, 22.0)
        ax.set_xlabel(r"$\log_{10} N_{\rm HI}$"); ax.set_ylabel(r"$f(N)$")
        ax.set_title(f"variant {vk}: CDDF"); ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")
    fig.suptitle("Track-C T-F LEG 2 — FROZEN 2LPT-0 recipe applied to held-out london-0 "
                 "(DIFFERENT recipe, no refit)\nR0 = recovered / london-0 truth; "
                 "R0≈1 (up to a single mean-flux rescale) = cross-recipe generalization",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[T-F] figure -> {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # frozen 2LPT-0 calibration
    p.add_argument("--catalog-dir", default=_C0_CAT, help="2LPT-0 catalog (calibration)")
    p.add_argument("--truth", default=_C0_TRUTH, help="2LPT-0 truth (calibration)")
    p.add_argument("--bal-cat", default=_C0_BAL)
    p.add_argument("--molly-tsv", default=None,
                   help="frozen lya_only-nhi195 molly C/ρ matrix (2LPT-0 default).")
    p.add_argument("--kernel", default=AB.DEF_KERNEL,
                   help="(unused for forward path; kept for _resolve_molly parity)")
    p.add_argument("--forward-model", default=_DEF_FORWARD)
    p.add_argument("--resp-family", default="empirical", choices=["skewnorm", "empirical"])
    # held-out london-0
    p.add_argument("--heldout-cat", default=_L0_CAT)
    p.add_argument("--heldout-truth", default=_L0_TRUTH)
    p.add_argument("--heldout-bal", default=_L0_BAL)
    p.add_argument("--heldout-mockdir", default=_L0_MOCKDIR)
    # run knobs (match the headline perz recipe)
    p.add_argument("--variant", default="both", choices=["A", "B", "both"])
    p.add_argument("--out", default=_DEF_OUT)
    p.add_argument("--report-out", default=".superpowers/sdd/track_c_TF_london0_report.md")
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
    p.add_argument("--cz-min-count", type=float, default=30.0)
    # band-finalize knobs (default ON — central recipe at HEAD)
    p.add_argument("--band-recenter", dest="band_recenter", action="store_true", default=True)
    p.add_argument("--no-band-recenter", dest="band_recenter", action="store_false")
    p.add_argument("--omega-slope-extrap", dest="omega_slope_extrap",
                   action="store_true", default=True)
    p.add_argument("--no-omega-slope-extrap", dest="omega_slope_extrap",
                   action="store_false")
    p.add_argument("--omega-slope-extrap-integrated", dest="omega_slope_extrap_integrated",
                   action="store_true", default=True)
    p.add_argument("--no-omega-slope-extrap-integrated",
                   dest="omega_slope_extrap_integrated", action="store_false")
    p.add_argument("--slope-edge", type=float, default=21.2)
    p.add_argument("--slope-fit-dex", type=float, default=0.6)
    p.add_argument("--sigma-slope", type=float, default=0.5)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    limits = tuple(float(x) for x in args.report_limits.split(","))
    args._limits = limits

    # ---- pre-flight: the held-out truth + QSO catalog must exist (R0 needs truth) ----
    missing = []
    if not _exists(args.heldout_cat):
        missing.append(f"held-out catalog: {args.heldout_cat}")
    if not os.path.exists(args.heldout_truth):
        missing.append(f"held-out london-0 TRUTH (R0 denominator): {args.heldout_truth}")
    # the QSO/snr catalog (pathlength + truth-match SNR) lives in the mockdir
    md = args.heldout_mockdir
    has_zcat = any(os.path.exists(os.path.join(md, f))
                   for f in ("zcat.fits", "seed_zcat.fits"))
    has_snr = os.path.exists(os.path.join(md, "snr_cat.fits"))
    if not (has_zcat and has_snr):
        missing.append(f"held-out london-0 QSO/snr catalog (pathlength): "
                       f"{md}/{{zcat.fits|seed_zcat.fits, snr_cat.fits}}")
    if missing:
        msg = ("\n[T-F] DATA BLOCKER — cannot score R0 without the held-out london-0 "
               "truth + QSO catalog. Missing:\n  - " + "\n  - ".join(missing) +
               "\n\nThe london-0 dlacat is present. R0 = recovered/london-0-truth needs "
               "the london mock `dla_cat.fits` (the R0 denominator) AND a `snr_cat.fits` "
               "+ `zcat.fits` (the pathlength X_tot covers ALL sightlines). Build the "
               "snr_cat from the catalog's processed h5:\n  python "
               "examples/make_snr_cat_from_processed.py --processed-dir "
               "<catalog>/processed --out-prefix <mockdir>/snr_cat\nthen symlink the "
               "london dla_cat.fits/zcat.fits/bal_cat.fits into <mockdir>.\n")
        print(msg)
        with open(os.path.abspath(args.report_out), "w") as fh:
            fh.write("# Track-C T-F LEG 2 — BLOCKED (held-out truth not staged)\n\n"
                     + msg + "\n")
        return dict(status="blocked", missing=missing)

    t0 = time.time()
    print("=" * 78)
    print("TRACK-C T-F LEG 2 — FROZEN 2LPT-0 recipe on held-out london-0 (no refit)")
    print(f"  forward kernel: {args.forward_model}")
    print(f"  held-out catalog: {args.heldout_cat}")
    print(f"  held-out truth:   {args.heldout_truth}")
    print("=" * 78)

    frozen = build_frozen_calibration(args)
    args.molly_tsv = frozen["molly_tsv"]
    variants = {}
    vlist = (["A", "B"] if args.variant == "both" else [args.variant])
    for vk in vlist:
        print(f"\n[T-F] === VARIANT {vk} ===")
        ing = build_heldout_ingredients(args, frozen, vk)
        res, cov = run_tf_variant(args, ing, limits, args.seed)
        ip = res["integrated_point"]
        itr = {"dndx": {l: ip["dndx_truth"][l] for l in limits},
               "omega": {l: ip["omega_truth"][l] for l in limits}}
        ir0 = {"dndx": {}, "omega": {}}
        for l in limits:
            ir0["dndx"][l] = dict(map=ip["dndx_map"][l], truth=ip["dndx_truth"][l],
                                  R0=ip["R0_dndx"][l])
            ir0["omega"][l] = dict(map=ip["omega_map"][l], truth=ip["omega_truth"][l],
                                   R0=ip["R0_omega"][l])
        variants[vk] = dict(res=res, cov=cov, int_truth=itr, int_R0=ir0,
                            X_tot=ing["X_tot"])
        print(f"  [variant {vk}] integrated R0 dN/dX(≥{limits[-1]:.1f})="
              f"{ir0['dndx'][limits[-1]]['R0']:.3f}, "
              f"Ω(≥{limits[-1]:.1f})={ir0['omega'][limits[-1]]['R0']:.3f}")

    wallclock = time.time() - t0
    rescale = fit_meanflux_rescale(variants, limits)
    fig_path = os.path.join(args.out, "fig_tf_london0.png")
    make_figure(fig_path, variants, args)
    rep = write_report(os.path.abspath(args.report_out), variants, args, wallclock, rescale)

    # JSON dump
    out_json = dict(metadata=dict(
        n_mc=args.n_mc, seed=args.seed, limits=list(limits),
        forward_model=args.forward_model, molly_tsv=args.molly_tsv,
        heldout_cat=args.heldout_cat, heldout_truth=args.heldout_truth,
        heldout_mockdir=args.heldout_mockdir,
        wallclock_s=float(wallclock), code_commit=_git_commit()),
        variants={vk: dict(coverage=V["cov"], integrated_R0={
            kind: {str(l): V["int_R0"][kind][l] for l in limits}
            for kind in ("dndx", "omega")},
            meanflux_rescale={str(l): rescale.get(vk, {}).get(l) for l in limits})
            for vk, V in variants.items()})
    with open(os.path.join(args.out, "track_c_tf_london0.json"), "w") as fh:
        json.dump(out_json, fh, indent=2, default=float)
    print("\n" + rep)
    print(f"\n[T-F] DONE in {wallclock:.0f}s")
    return dict(status="complete", variants=variants)


if __name__ == "__main__":
    main()
