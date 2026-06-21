#!/usr/bin/env python
"""track_c_tf_loa.py — Track-C T-F LEG 3: the SCIENCE measurement on REAL DESI LOA.

THE PAYOFF.  Apply the FROZEN 2LPT-0 Track-C recipe (forward-response kernel +
z-resolved completeness C·g(N,z) + molly C/ρ + band machinery, CALIBRATED ONLY on
2LPT-0, cross-validated cross-recipe on held-out london-0) to the REAL DESI Y3 LOA
GP-DLA catalog, producing the actual dN/dX(z) / Ω_HI(z) / CDDF f(N) measurement with
honest bands, and overlaying the published literature (Ho+2021, N12, PW09, …).

WHY FROZEN-C IS THE RIGHT DESIGN FOR REAL DATA.  Real survey data has NO truth
catalog, so the completeness C, purity ρ, and forward-response kernel CANNOT be
rebuilt on it — there is nothing to match against.  This is exactly the regime the
frozen-recipe design was built for: the kernel + g(N,z) + C/ρ are GP-SET properties
(the SAME learned_file + inference config produced 2LPT-0, london-0, and this LOA
catalog), so they transfer.  The london-0 leg quantified the transfer: dN/dX
recovers to ~1–2 % (restored to <1 % by a single mean-flux amplitude rescale), and
Ω to within ~12 % (the known high-N tail-weight recipe-dependence).  We carry those
as STATED SYSTEMATICS on the real-LOA measurement.

WHAT IS FROZEN vs DATA (the crux — strict):
  * Forward-response kernel  → FROZEN  (forward_response_2lpt0.npz; never re-fit).
  * z-resolved completeness g(N,z)  → FROZEN  (built ONCE on 2LPT-0's truth-match,
    stashed on cfg._cnz_resolved so ensure_cnz_resolved returns it unchanged).
  * molly C/ρ ratio matrix (the .tsv) + the COUNT denominators → FROZEN from 2LPT-0
    (real LOA has NO truth to rebuild them — and london-0 proved Variant A≡B, i.e.
    rebuilding the counts on a held-out catalog does NOT move the answer).
  * Band machinery (recenter, slope-extrap) — as in the headline.
  * The real-LOA CATALOG, the QSO/pathlength (production zcat + processed-h5 SNR),
    and the per-object NHI_ERR — the DATA.

THE BAND — TRUTH-FREE BY CONSTRUCTION (reviewed; see report).  The headline mock
band uses Stage-II shared_boot + Stage-III per-draw forward-refit, both of which
need a truth-match resample (tmr) we cannot build on real data.  We therefore use
the `indep` band (cfg.mc_nuisance='indep', cfg.mc_response='frozen', refit_fn=None,
tmr=None): the per-molly-cell C/ρ Wilson/Jeffreys-Beta jitter is drawn from the
FROZEN 2LPT-0 count denominators, the sightline bootstrap is over the REAL LOA op
sightlines (the genuine Poisson sightline-counting variance of the data), and the
σ_i width uses the REAL per-object NHI_ERR.  The forward-response kernel + C/ρ
calibration are held FROZEN across draws — consistent with the frozen-kernel point.
This is justified empirically: on london-0, grafting the frozen 2LPT-0 counts
(Variant A) vs rebuilding them on the mock's own truth (Variant B) gave identical
points AND identical bands to every printed digit (the count level / Stage-III
response-refit is NOT a transfer lever).  The band reflects C/ρ-calibration,
real-sightline-bootstrap and NHI-measurement variance about a frozen calibration;
it does NOT include the calibration-TRANSFER uncertainty (the ~1–2 % dN/dX / ~12 %
Ω mock→data offset), which is assessed separately by the london-0 closure and
carried as a stated systematic.

PER-z Ω BAND (frozen-kernel transport; reviewed).  indep joint_mc_errors stores
per-z dN/dX(z) samples + z-marginal Ω/CDDF samples but NOT the genuine 2-D per-z f
(f_bk_coarse is filled only on the forward-refit path).  The genuine 2-D MAP f
(rr_map['f_bk_coarse']) IS available.  Because the N-shape within each z-bin is
FROZEN by the kernel + g(N,z), Ω(z) and dN/dX(z) share one per-z multiplicative
fluctuation; we transport it onto the genuine MAP Ω(z) and recenter on the MAP via
the established recenter_band_on_point primitive (same band recipe the headline
uses).  The result brackets the MAP and is consistent in width with the z-marginal Ω
band that exists natively in indep mode.

NO truth, NO R0 (real data — the MEASUREMENT *is* the deliverable, not a recovery
ratio).  Reduce-only / analysis-side.  NO GP inference (gpy_dla_detection/
byte-FROZEN).  No estimator-logic edit (cddf_catalog_hbi.py untouched).  conda gpdla;
BLAS pinned.

🔴 REAL-DATA PRIVACY: this driver writes the AGGREGATE measurement (dN/dX/Ω/CDDF
numbers + figure) to scratch + the private notes repo.  It does NOT write any raw
real-data products (catalog rows, per-object arrays) to the public code repo.

Usage:
  python CDDF_analysis/track_c_tf_loa.py --n-mc 120 --workers 4
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
    make_C_interpolator, build_pathlength, make_fp_model, make_rho_interpolator,
    _build_qso_lookup, v3x_refit, build_cnz_resolved, regenerate_molly_counts,
    v3x_reduce, joint_mc_errors, omega_hi_prefactor, recenter_band_on_point,
)
import functools

# ---------------------------------------------------------------------------
# FROZEN 2LPT-0 recipe artifacts (the calibration; never re-fit on real LOA)
# ---------------------------------------------------------------------------
_DEF_FORWARD = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                "track_c/stage0/forward_response_2lpt0.npz")

# 2LPT-0 (CALIBRATION mock) — the catalog/truth/bal the frozen g + molly counts come from
_C0_CAT = AB.DEF_CAT
_C0_TRUTH = AB.DEF_TRUTH
_C0_BAL = AB.DEF_BAL

# REAL DESI LOA (the DATA — no truth).
_LOA_CAT = "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1"
# staged mockdir holds: snr_cat.fits (= zcat.fits symlink; both built from the catalog's
# processed-main-dark-*.h5 `snrs`/`z_qsos` — SNR_REDSIDE byte-identical to the dlacat,
# verified 20000/20000); bal_cat.fits (TARGETID where BI_CIV>0 in the production QSO
# catalog — reproduces the dlacat BAL_FLAG bit-for-bit, 77007/77007); dla_cat.fits is an
# EMPTY placeholder truth (real data has none — never read for the point or the indep band).
_LOA_MOCKDIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                "track_c/tf_loa/mockdir")
_LOA_TRUTH = _LOA_MOCKDIR + "/dla_cat.fits"      # EMPTY placeholder (no real truth)
_LOA_BAL = _LOA_MOCKDIR + "/bal_cat.fits"

_DEF_OUT = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_loa"
_DLA_DATA_DIR = "/home/mfho/DLA_data"            # the literature module (dla_data.py)


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# build the FROZEN 2LPT-0 completeness g(N,z) + molly counts ONCE
# ---------------------------------------------------------------------------
def build_frozen_calibration(args):
    """Build the 2LPT-0 ingredients ONCE and extract the FROZEN recipe pieces:
      * g_cnz : the z-resolved CNZModel g(N,z) (build_cnz_resolved on 2LPT-0).
      * molly_counts : the 2LPT-0 (pur_ntp, pur_ntot, cmp_nfound, cmp_nfid) count
                 denominators (the band's C/ρ Wilson/Jeffreys-Beta jitter source).
    Reads ONLY 2LPT-0 inputs."""
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
        v2_z_fit_lo=2.0, v2_z_fit_hi=args.v2_z_fit_hi, v2_z_fit_step=0.1, rng_seed=0,
        completeness_z_resolved=True, completeness_z_min_count=float(args.cz_min_count),
    )
    mm = load_molly_matrix(molly_tsv)
    truth_floor = float(mm.nhi_edges[0])
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(args.host_truth_floor, truth_floor))
    mm = regenerate_molly_counts(mm, cat_cut, is_TP, truth_cut, good_mask, cfg)
    g_cnz = build_cnz_resolved(cfg, cat_cut, truth_cut, good_mask, mm)

    # CALIBRATION-SUPPORT FLAG: per coarse-z report bin, the number of 2LPT-0 TRUTH
    # DLAs (the calibration mock) with NHI >= the headline limit.  A coarse z-bin with
    # ZERO (or <cz_min_count) truth has NO z-resolved completeness support — there
    # build_cnz_resolved's occupancy shrinkage (w=n_true/(n_true+k)) sends g(N,z)->1,
    # i.e. the z-resolved completeness collapses back to the z-MARGINAL molly C
    # (EXTRAPOLATION).  The indep statistical band holds g FROZEN, so it will NOT
    # capture that completeness-extrapolation bias — such bins are FLAGGED, not folded
    # into any calibrated headline.
    zbins = np.asarray(cfg.zbins, float)
    n_zc = len(zbins) - 1
    t_nhi_all = np.asarray(truth_cut["NHI"], float)
    t_z_all = np.asarray(truth_cut["Z_DLA"], float)
    hl = float(max(args._limits))                 # headline limit for the support flag
    truth_counts_perz = np.zeros(n_zc, dtype=int)
    for k in range(n_zc):
        sel = (t_z_all >= zbins[k]) & (t_z_all < zbins[k + 1]) & (t_nhi_all >= hl - 1e-9)
        truth_counts_perz[k] = int(np.count_nonzero(sel))
    max_truth_z = float(np.nanmax(t_z_all)) if t_z_all.size else float("nan")
    print(f"[T-F] 2LPT-0 truth z-support (NHI>={hl:.1f}) per coarse z-bin "
          f"{[f'[{zbins[k]:.2f},{zbins[k+1]:.2f})={truth_counts_perz[k]}' for k in range(n_zc)]}; "
          f"max truth z_DLA={max_truth_z:.3f}")

    molly_counts = dict(
        pur_ntp=np.array(mm.pur_ntp, float), pur_ntot=np.array(mm.pur_ntot, float),
        cmp_nfound=np.array(mm.cmp_nfound, float), cmp_nfid=np.array(mm.cmp_nfid, float),
        nhi_edges=np.array(mm.nhi_edges, float), snr_edges=np.array(mm.snr_edges, float),
    )
    print(f"[T-F] frozen g(N,z) shape={g_cnz.g_grid.shape}; molly counts captured "
          f"(C ratio matrix + forward kernel are frozen files).")
    return dict(g_cnz=g_cnz, molly_counts=molly_counts, molly_tsv=molly_tsv,
                c0_truth_floor=truth_floor,
                truth_counts_perz=truth_counts_perz, max_truth_z=max_truth_z,
                cz_min_count=float(args.cz_min_count), support_limit=hl)


# ---------------------------------------------------------------------------
# build the REAL-LOA ingredients with the FROZEN recipe injected (truth-free)
# ---------------------------------------------------------------------------
def build_loa_ingredients(args, frozen):
    """Mirror the london-0 Variant-A ingredients, but for the REAL LOA catalog and
    with NO truth: graft the FROZEN 2LPT-0 molly COUNT denominators (the band's C/ρ
    jitter source), inject the FROZEN g(N,z) onto cfg._cnz_resolved, and never rebuild
    the molly counts on the real catalog (there is no truth to do so, and london-0
    proved it is not a transfer lever).  The point + indep band read C/ρ from the
    FROZEN matrix; the empty placeholder truth threads through load_and_cut_catalog
    harmlessly (is_TP all False, NHI_TRUE all NaN — unused by the point or indep band).
    """
    molly_tsv = frozen["molly_tsv"]               # frozen 2LPT-0 C/ρ ratio matrix
    cfg = HBIConfig(
        catalog_dir=args.loa_cat, truth_path=args.loa_truth,
        bal_cat_path=args.loa_bal, molly_tsv=molly_tsv, out_dir=args.out,
        mockdir=args.loa_mockdir,
        zbins=tuple(float(x) for x in args.zbins.split(",")),
        report_logN_limits=tuple(float(x) for x in args.report_limits.split(",")),
        fp_estimator="purity_mixture", no_bal=True,
        v3_family=args.family, v3_logN_fit_floor=args.fit_floor,
        v3_logN_fit_ceil=args.fit_ceil, v3_lambda_bspbody=args.lambda_bspbody,
        v3_mc_n_restart=2, lam_rf_min=args.lam_rf_min,
        v3_bspbody_edge_slope_lam=args.edge_slope_lam,
        v3_fine_density_gl_nodes=args.gl_nodes,
        v2_z_fit_lo=2.0, v2_z_fit_hi=args.v2_z_fit_hi, v2_z_fit_step=0.1, rng_seed=0,
        completeness_z_resolved=True, completeness_z_min_count=float(args.cz_min_count),
    )
    mm = load_molly_matrix(molly_tsv)
    truth_floor = float(mm.nhi_edges[0])
    qso_lookup = _build_qso_lookup(cfg)            # reads the real LOA mockdir snr/zcat
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(args.host_truth_floor, truth_floor))

    # GRAFT the FROZEN 2LPT-0 molly COUNT denominators (TRAP A: regenerate_molly_counts
    # ran inside load? no — we call it here with the EMPTY real truth and OVERWRITE).
    # The C/ρ ratio matrix is already the frozen file (load_molly_matrix on molly_tsv);
    # the COUNT arrays (used only by the indep band's Jeffreys-Beta jitter) come from
    # 2LPT-0 — real LOA has no truth to rebuild them, and Variant A≡B on london-0.
    mc0 = frozen["molly_counts"]
    if not (np.allclose(mc0["nhi_edges"], mm.nhi_edges)
            and np.allclose(mc0["snr_edges"], mm.snr_edges)):
        raise SystemExit("frozen molly count grid != real-LOA molly grid (same TSV "
                         "expected) — cannot graft counts.")
    mm.pur_ntp = mc0["pur_ntp"].copy(); mm.pur_ntot = mc0["pur_ntot"].copy()
    mm.cmp_nfound = mc0["cmp_nfound"].copy(); mm.cmp_nfid = mc0["cmp_nfid"].copy()
    mm._max_p_diff = 0.0; mm._max_c_diff = 0.0

    C_interp = make_C_interpolator(mm)
    rho_interp = make_rho_interpolator(mm)
    X_tot, n_sl, qzl, qzh, qsn, Xcalc = build_pathlength(
        cfg, qso_lookup=qso_lookup, return_per_sl=True)
    cfg.n_sl_prod = int(n_sl)
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)

    # FREEZE g(N,z): stash the 2LPT-0 model so ensure_cnz_resolved returns it unchanged.
    g0 = frozen["g_cnz"]
    if not np.allclose(np.asarray(g0.nhi_edges, float), np.asarray(mm.nhi_edges, float)):
        raise SystemExit("frozen g(N,z) nhi_edges != real-LOA molly nhi_edges.")
    cfg._cnz_resolved = copy.deepcopy(g0)

    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op_mask = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    fp_model, _ = make_fp_model(cfg, cat_cut, op_mask, rho_interp)

    # ---- FEED-FORWARD (kappa) mode: attach the PER-DETECTION GP-posterior kernel ----
    # The kappa2d posterior kernel p(N_true,z | x̂_i) is a per-OBJECT property, so unlike
    # the frozen-from-2LPT-0 forward kernel it is built ON the REAL-LOA detections (the GP
    # ran on this very catalog; the posterior samples are in its processed-h5). v2_kernel
    # ='posterior' routes build_A_ib -> _build_A_ib_kappa2d (the pre-Track-C path that
    # over-counts high-N on-mock — exactly what Track-C's forward kernel replaces).
    kernel_built_path = None
    if getattr(args, "resp_kind", "forward") == "kappa":
        from CDDF_analysis.cddf_catalog_hbi import (
            build_posterior_kernel, build_targetid_backlink)
        cfg.v2_kernel = "posterior"
        loa_kernel = getattr(args, "loa_kernel", None)
        if loa_kernel and os.path.exists(loa_kernel):
            d = np.load(loa_kernel, allow_pickle=True)
            cfg._posterior_kernel_2d = d["kappa"].astype(np.float32)
            kernel_built_path = loa_kernel
            print(f"  [LOA-kappa] loaded posterior kernel {cfg._posterior_kernel_2d.shape} "
                  f"<- {loa_kernel}")
        else:
            out_npz = (loa_kernel or
                       os.path.join(args.out, "posterior_kernel_loa.npz"))
            print(f"  [LOA-kappa] building per-detection posterior kernel on the REAL-LOA "
                  f"catalog (glob={args.loa_processed_glob}) ...")
            backlink, files = build_targetid_backlink(args.loa_processed_glob)
            kappa, _ess = build_posterior_kernel(
                cfg, cat_cut, good_mask, (logN_lo, logN_hi, N_b, dN_b),
                processed_glob=args.loa_processed_glob,
                pw_samples_path=args.loa_pw_samples,
                backlink=backlink, files=files, out_npz=out_npz,
                n_jobs=max(1, int(args.workers)), verbose=False)
            cfg._posterior_kernel_2d = np.asarray(kappa, np.float32)
            kernel_built_path = out_npz
            print(f"  [LOA-kappa] posterior kernel {cfg._posterior_kernel_2d.shape} "
                  f"built -> {out_npz}")

    estimator_fn = functools.partial(
        v3x_refit, mm=mm, qso_per_sl=(qzl, qzh, qsn), Xcalc=Xcalc,
        rng=np.random.default_rng(0))
    _kmsg = ("kappa NOT attached (forward path)"
             if getattr(args, "resp_kind", "forward") != "kappa"
             else f"kappa2d ATTACHED {cfg._posterior_kernel_2d.shape} (feed-forward)")
    print(f"  [LOA] real catalog: n_op_sl={n_sl}, n_cat_cut={len(cat_cut)}, "
          f"frozen g shape={cfg._cnz_resolved.g_grid.shape}, {_kmsg}.")
    meta["n_op_detections"] = int(op_mask.sum())
    return dict(cfg=cfg, mm=mm, cat_cut=cat_cut, truth_cut=truth_cut, is_TP=is_TP,
                good_mask=good_mask, C_interp=C_interp, fp_model=fp_model,
                X_tot=X_tot, n_sl=n_sl, logN_lo=logN_lo, logN_hi=logN_hi,
                N_b=N_b, dN_b=dN_b, estimator_fn=estimator_fn, meta=meta,
                op_mask=op_mask, kernel_built_path=kernel_built_path)


# ---------------------------------------------------------------------------
# the MEASUREMENT: MAP point + truth-free indep band (NO R0, NO truth scoring)
# ---------------------------------------------------------------------------
def run_measurement(args, ing, limits, seed, frozen=None):
    """MAP point (frozen kernel) + indep MC band (truth-free).  Returns a `res` dict
    with per-z + integrated dN/dX(z) / Ω(z) MAP + 68/95 bands and the z-marginal CDDF
    f(N) MAP + band.  NO truth, NO R0 — the measurement IS the deliverable."""
    cfg = ing["cfg"]
    cfg.report_logN_limits = limits
    cfg._wall1_estimator = "v3"
    cfg.n_mc = args.n_mc
    # the truth-free band knobs (reviewed): indep C/ρ jitter from FROZEN counts +
    # real-sightline bootstrap + real NHI_ERR width; response held FROZEN across draws.
    PZ._set_forward_cfg(cfg, args)        # sets resp_kind=forward, kernel, recenter, slope-extrap
    # KERNEL MODE OVERRIDE (default 'forward' = byte-identical to the committed run). In
    # 'kappa' (feed-forward) mode the forward-kernel dispatch is turned OFF and the
    # estimator consumes cfg._posterior_kernel_2d via v2_kernel='posterior' (attached in
    # build_loa_ingredients). The band recipe (indep C/ρ jitter, frozen response, recenter,
    # slope-extrap) is otherwise IDENTICAL, so the only thing that differs between the two
    # measurements is the kernel object — isolating exactly the +9% Track-C correction.
    if getattr(args, "resp_kind", "forward") == "kappa":
        cfg.resp_kind = "kappa"
    cfg.mc_nuisance = "indep"             # OVERRIDE: no shared_boot (needs truth tmr)
    cfg.mc_response = "frozen"            # OVERRIDE: no Stage-III refit (needs truth tmr)

    logN_lo = ing["logN_lo"]; logN_hi = ing["logN_hi"]
    N_b = ing["N_b"]; dN_b = ing["dN_b"]
    zbins = np.asarray(cfg.zbins, float)
    n_zc = len(zbins) - 1
    K = omega_hi_prefactor(cfg.H0)
    mid = 0.5 * (logN_lo + logN_hi)

    # ---- MAP point: the forward estimator (truth-free) + genuine 2-D MAP f ----
    e0 = ing["estimator_fn"](
        ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["C_interp"],
        ing["fp_model"], ing["X_tot"], logN_lo, logN_hi, N_b, dN_b,
        ing["truth_cut"], cfg, clip_negative=False)
    theta_map = e0["_v3x"]["theta_map"]; fine = e0["_v3x"]["fine"]
    family = e0["_v3x"]["family"]; M_meta = e0["_v3x"]["M_meta"]
    rr_map = v3x_reduce(cfg, theta_map, fine, family, M_meta)
    map_fbk = np.asarray(rr_map["f_bk_coarse"], float)            # genuine (n_nbins, n_zc)
    map_dndx_z = {l: PZ.perz_dndx_from_fbk(map_fbk, logN_lo, dN_b, l) for l in limits}
    map_omega_z = {l: PZ.perz_omega_from_fbk(map_fbk, logN_lo, N_b, dN_b, K, l)
                   for l in limits}
    # integrated (z-marginal) MAP, tied to the estimator's own reductions
    map_dndx_tot = {l: float(e0["dndx_total"][l]) for l in limits}
    map_omega_tot = {l: float(e0["omega"][l]) for l in limits}
    # consistency gate: MAP per-z dN/dX from f_bk == e0['dndx_z']
    cerr = 0.0
    for l in limits:
        a = map_dndx_z[l]; b = np.asarray(e0["dndx_z"][l], float)
        good = np.isfinite(b) & (np.abs(b) > 0)
        if good.any():
            cerr = max(cerr, float(np.max(np.abs(a[good] - b[good]) / np.abs(b[good]))))
    if cerr >= 1e-7:
        raise AssertionError(f"MAP per-z dN/dX vs e0.dndx_z mismatch: {cerr:.2e}")
    map_fb = np.asarray(e0["f_b"], float)                        # z-marginal MAP CDDF

    # ---- truth-free indep band (NO tmr, NO refit_fn) ----
    cfg.n_mc = args.n_mc
    mc = joint_mc_errors(
        ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["mm"], ing["fp_model"],
        ing["X_tot"], logN_lo, logN_hi, N_b, dN_b, ing["truth_cut"],
        cfg, np.random.default_rng(seed + 4), refit_fn=None, tmr=None)
    samp = mc["_samples"]
    fb_samp = np.asarray(samp["f_b"], float)                     # (n_mc, n_nbins) z-marg CDDF
    dndx_z_samp = {l: np.asarray(samp["dndx_z"][l], float) for l in limits}   # (n_mc, n_zc)
    dndx_tot_samp = {l: np.asarray(samp["dndx_total"][l], float) for l in limits}  # (n_mc,)
    omega_tot_samp = {l: np.asarray(samp["omega"][l], float) for l in limits}      # (n_mc,)
    n_draw = fb_samp.shape[0]

    # per-z Ω band (recentered-C1 frozen-kernel transport — reviewed): transport the
    # per-z dN/dX multiplicative fluctuation onto the genuine MAP Ω(z), recentered on
    # the MAP by the established recenter_band_on_point primitive.  Faithful because the
    # N-shape within a z-bin is FROZEN by the kernel + g, so Ω(z) and dN/dX(z) share one
    # per-z normalization fluctuation.
    omega_z_samp = {}
    for l in limits:
        oz = np.full((n_draw, n_zc), np.nan)
        for k in range(n_zc):
            mz = map_dndx_z[l][k]
            if np.isfinite(mz) and mz > 0:
                raw = map_omega_z[l][k] * (dndx_z_samp[l][:, k] / mz)
                oz[:, k] = recenter_band_on_point(raw, map_omega_z[l][k])
        omega_z_samp[l] = oz

    def _band(samp_arr, point):
        b = PZ._band(np.asarray(samp_arr, float), point=point, recenter=cfg.band_recenter)
        return dict(MAP=float(point), q16=b["q16"], q84=b["q84"],
                    q025=b["q025"], q975=b["q975"], std=b["std"])

    # CALIBRATION-SUPPORT FLAG per coarse-z bin (from the FROZEN 2LPT-0 truth occupancy).
    # A bin is EXTRAPOLATED (flagged, excluded from headline) when the 2LPT-0 truth count
    # at the headline NHI limit falls to ZERO (g(N,z)->z-marginal C there; the indep band
    # holds g frozen so cannot capture that completeness-extrapolation bias).  Also flag
    # bins whose lower edge sits above the max 2LPT-0 truth z_DLA (no truth above the cap).
    z_extrapolated = np.zeros(n_zc, dtype=bool)
    z_thin = np.zeros(n_zc, dtype=bool)
    truth_counts_perz = None
    if frozen is not None:
        tcz = np.asarray(frozen.get("truth_counts_perz", np.full(n_zc, -1)), int)
        max_tz = float(frozen.get("max_truth_z", np.nan))
        kmin = float(frozen.get("cz_min_count", 30.0))
        for k in range(n_zc):
            cnt = tcz[k] if k < len(tcz) else -1
            if cnt == 0 or (np.isfinite(max_tz) and zbins[k] >= max_tz):
                z_extrapolated[k] = True            # NO calibration support -> EXTRAPOLATED
            elif 0 < cnt < kmin:
                z_thin[k] = True                    # THIN support -> calibrated but wide band
        truth_counts_perz = tcz.tolist()

    # assemble per-z + integrated band records (MAP-recentered, same as headline)
    out = dict(
        cfg=cfg, H0=cfg.H0, K=K, zbins=zbins, n_zc=n_zc, mid=mid,
        logN_lo=logN_lo, logN_hi=logN_hi, N_b=N_b, dN_b=dN_b,
        n_mc=int(n_draw), consistency_err=float(cerr),
        n_op_detections=int(ing["meta"].get("n_op_detections", -1)),
        n_op_sl=int(ing["n_sl"]), X_tot=np.asarray(ing["X_tot"], float),
        map_fb=map_fb, fb_samp=fb_samp,
        z_extrapolated=z_extrapolated, z_thin=z_thin,
        truth_counts_perz=truth_counts_perz,
        support_limit=(float(frozen.get("support_limit", max(limits)))
                       if frozen is not None else float(max(limits))),
        max_truth_z=(float(frozen.get("max_truth_z", np.nan))
                     if frozen is not None else float("nan")),
        dndx=dict(), omega=dict(),
    )
    for l in limits:
        out["dndx"][l] = dict(
            perz=[_band(dndx_z_samp[l][:, k], map_dndx_z[l][k]) for k in range(n_zc)],
            integrated=_band(dndx_tot_samp[l], map_dndx_tot[l]),
        )
        out["omega"][l] = dict(
            perz=[_band(omega_z_samp[l][:, k], map_omega_z[l][k]) for k in range(n_zc)],
            integrated=_band(omega_tot_samp[l], map_omega_tot[l]),
        )
    return out


# ---------------------------------------------------------------------------
# literature comparison helpers (import dla_data; read tables, not pyplot)
# ---------------------------------------------------------------------------
def _load_literature():
    """Import the literature module and pull the numeric tables we overlay:
    Ho+2021 CDDF f(N) (z=3-4 + all-z), dN/dX(z), Ω(z); N12 dN/dX/Ω; PW09 dN/dX/Ω.
    Returns a dict of numpy arrays (no pyplot side effects)."""
    if _DLA_DATA_DIR not in sys.path:
        sys.path.insert(0, _DLA_DATA_DIR)
    import dla_data as DD            # noqa: E402
    lit = {}
    dd = DD.datadir
    import os.path as P
    # Ho+2021 CDDF (all-z and z=3-4): files are 6-row blocks (logN, f, lo68, hi68, lo95, hi95)
    for tag, fn in (("ho21_all", "ho21/cddf_all.txt"), ("ho21_z34", "ho21/cddf_z34.txt"),
                    ("ho21_z225", "ho21/cddf_z225.txt"), ("ho21_z253", "ho21/cddf_z253.txt")):
        p = P.join(dd, fn)
        if P.exists(p):
            d = np.loadtxt(p)
            lit[tag] = dict(logN=d[0], f=d[1], lo95=d[4], hi95=d[5], lo68=d[2], hi68=d[3])
    # Ho+2021 dN/dX(z): (6, nz) block — row0=z, row1=dN/dX, rows2-5 = 16/84/2.5/97.5
    p = P.join(dd, "ho21/dndx_all.txt")
    if P.exists(p):
        d = np.loadtxt(p)
        lit["ho21_dndx"] = dict(z=d[0], val=d[1], lo68=d[2], hi68=d[3])
    # Ho+2021 Ω(z): (6, nz) block — row0=z, row1 = 10^3·Ω, rows2-5 = raw-Ω quantiles
    p = P.join(dd, "ho21/omega_dla_all.txt")
    if P.exists(p):
        d = np.loadtxt(p)
        # row1 is already 10^3·Ω; rows2/3 are raw-Ω 16/84 → scale to 10^3
        lit["ho21_omega"] = dict(z=d[0], val1e3=d[1],
                                 lo68_1e3=d[2] * 1000.0, hi68_1e3=d[3] * 1000.0)
    # N12 dN/dX + Ω (hard-coded in dla_data.dndx_not / omegahi_not)
    lit["n12_z"] = np.array([2.15, 2.45, 2.75, 3.05, 3.35])
    dndz = np.array([0.2, 0.2, 0.25, 0.29, 0.36])
    dzdx = np.array([3690 / 11625., 4509 / 14841., 2867 / 9900.,
                     1620 / 5834., 789 / 2883.])
    lit["n12_dndx"] = dndz * dzdx
    lit["n12_omega"] = np.array([0.99, 0.87, 1.04, 1.1, 1.27]) * 0.76   # 10^3 Ω scale
    lit["n12_omega_err"] = np.array([0.05, 0.04, 0.05, 0.08, 0.13])
    # PW09 dN/dX + Ω (dndx.txt)
    p = P.join(dd, "dndx.txt")
    if P.exists(p):
        d = np.loadtxt(p)
        zcen = (d[1:-1, 0] + d[1:-1, 1]) / 2.
        lit["pw09_z"] = zcen
        lit["pw09_dndx"] = d[1:-1, 2]
        lit["pw09_dndx_err"] = d[1:-1, 3]
        rhohi = d[1:-1, 4]
        rho_crit = 9.3125685124148235e-30
        conv = 6.7699111782945424e-33
        lit["pw09_omega"] = rhohi * conv / rho_crit * 1000   # 10^3 Ω scale
        lit["pw09_omega_err"] = d[1:-1, 5]
    # N12 CDDF f(N) (not_2012.dat): logN, logf, dlogN, dlogf
    p = P.join(dd, "not_2012.dat")
    if P.exists(p):
        d = np.loadtxt(p)
        lit["n12_fN"] = dict(logN=d[:, 0], logf=d[:, 1], dlogf=d[:, 2:4])
    return lit


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(out_path, res, args, lit):
    """fig_tf_loa.png — 3-panel: (0) dN/dX(z), (1) 10^3·Ω(z), (2) CDDF f(N).
    Real-LOA HBI MAP + 68 % band, with literature overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hl = args._limits[-1] if args._limits else 20.3        # headline limit for z-panels
    zbins = res["zbins"]; zmid = 0.5 * (zbins[:-1] + zbins[1:])
    n_zc = res["n_zc"]; mid = res["mid"]
    C_MAP = "#1f77b4"; C68 = "#1f77b4"
    C_FLAG = "#7f7f7f"                                      # grey = extrapolated
    z_extrap = np.asarray(res.get("z_extrapolated", np.zeros(n_zc, bool)), bool)
    z_thin = np.asarray(res.get("z_thin", np.zeros(n_zc, bool)), bool)
    sup_lim = float(res.get("support_limit", hl))

    def _plot_perz(ax, kind, sc):
        """Plot per-z points: calibrated (filled, blue, with HBI band), thin-but-
        calibrated (filled blue, lighter, with wide band), extrapolated (open grey,
        no headline)."""
        lab_cal = lab_thin = lab_ext = False
        for k in range(n_zc):
            c = res[kind][hl]["perz"][k]
            if not np.isfinite(c["MAP"]):
                continue                                   # empty bin (no fine-z support)
            lo, hi = c["q16"] * sc, c["q84"] * sc
            if z_extrap[k]:
                ax.vlines(zmid[k], lo, hi, color=C_FLAG, lw=8, alpha=0.30)
                ax.plot(zmid[k], c["MAP"] * sc, "o", color="white", ms=10,
                        mec=C_FLAG, mew=2.0, zorder=7,
                        label=("EXTRAP. (beyond 2LPT-0 truth; C=z-marg)"
                               if not lab_ext else None))
                lab_ext = True
            elif z_thin[k]:
                ax.vlines(zmid[k], lo, hi, color=C68, lw=8, alpha=0.30)
                ax.plot(zmid[k], c["MAP"] * sc, "o", color=C_MAP, ms=9, mec="k",
                        alpha=0.7, zorder=6,
                        label=("this work (thin calib.)" if not lab_thin else None))
                lab_thin = True
            else:
                ax.vlines(zmid[k], lo, hi, color=C68, lw=8, alpha=0.45)
                ax.plot(zmid[k], c["MAP"] * sc, "o", color=C_MAP, ms=9, mec="k",
                        zorder=6,
                        label=("DESI-LOA GP-DLA (this work)" if not lab_cal else None))
                lab_cal = True

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # panel 0: dN/dX(z) at the headline limit + literature
    ax = axes[0]
    _plot_perz(ax, "dndx", 1.0)
    # annotate the extrapolated bin(s)
    for k in range(n_zc):
        if z_extrap[k] and np.isfinite(res["dndx"][hl]["perz"][k]["MAP"]):
            ax.annotate("extrapolated\n(beyond 2LPT-0 truth\nsupport; C=z-marg.)",
                        xy=(zmid[k], res["dndx"][hl]["perz"][k]["MAP"]),
                        xytext=(0.62, 0.78), textcoords="axes fraction", fontsize=7,
                        color=C_FLAG, ha="left",
                        arrowprops=dict(arrowstyle="->", color=C_FLAG, lw=0.8))
            break
    if "n12_z" in lit:
        ax.plot(lit["n12_z"], lit["n12_dndx"], "s", color="black", ms=6, label="N12")
    if "pw09_z" in lit:
        ax.errorbar(lit["pw09_z"], lit["pw09_dndx"], yerr=lit["pw09_dndx_err"],
                    fmt="o", color="magenta", ms=6, label="PW09")
    if "ho21_dndx" in lit:
        h = lit["ho21_dndx"]
        ax.errorbar(h["z"], h["val"], yerr=[h["val"] - h["lo68"], h["hi68"] - h["val"]],
                    fmt="^", color="green", ms=5, label="Ho21", alpha=0.8)
    ax.set_xlabel("z"); ax.set_ylabel(rf"$dN/dX\,(\geq {hl:.1f})$")
    ax.set_title(f"dN/dX(z), $\\log N_{{HI}}\\geq{hl:.1f}$  (real DESI LOA)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8); ax.margins(y=0.25)

    # panel 1: 10^3 Ω(z) at headline limit + literature
    ax = axes[1]
    SCALE = 1000.0
    _plot_perz(ax, "omega", SCALE)
    for k in range(n_zc):
        if z_extrap[k] and np.isfinite(res["omega"][hl]["perz"][k]["MAP"]):
            ax.annotate("extrapolated\n(beyond 2LPT-0 truth\nsupport; C=z-marg.)",
                        xy=(zmid[k], res["omega"][hl]["perz"][k]["MAP"] * SCALE),
                        xytext=(0.62, 0.78), textcoords="axes fraction", fontsize=7,
                        color=C_FLAG, ha="left",
                        arrowprops=dict(arrowstyle="->", color=C_FLAG, lw=0.8))
            break
    if "n12_z" in lit:
        ax.errorbar(lit["n12_z"], lit["n12_omega"], yerr=lit["n12_omega_err"],
                    fmt="s", color="black", ms=6, label="N12")
    if "pw09_z" in lit:
        ax.errorbar(lit["pw09_z"], lit["pw09_omega"], yerr=lit["pw09_omega_err"],
                    fmt="o", color="magenta", ms=6, label="PW09")
    if "ho21_omega" in lit:
        h = lit["ho21_omega"]
        ax.errorbar(h["z"], h["val1e3"],
                    yerr=[h["val1e3"] - h["lo68_1e3"], h["hi68_1e3"] - h["val1e3"]],
                    fmt="^", color="green", ms=5, label="Ho21", alpha=0.8)
    ax.set_xlabel("z"); ax.set_ylabel(rf"$10^3\,\Omega_{{\rm DLA}}\,(\geq {hl:.1f})$")
    ax.set_title(f"$\\Omega_{{\\rm DLA}}(z)$, $\\log N_{{HI}}\\geq{hl:.1f}$  (real DESI LOA)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8); ax.margins(y=0.25)

    # panel 2: CDDF f(N) z-marginal + literature (Ho21 z3-4, N12)
    ax = axes[2]
    map_fb = res["map_fb"]; fb_samp = res["fb_samp"]
    # recenter-on-point (Track-C #34) for the DIFFERENTIAL f(N) band — panels 0/1
    # (dN/dX(z), Ω(z)) already recenter via recenter_band_on_point; panel 2 must match.
    # Per-bin additive median->point shift (width-preserving); without it the convex-
    # bspline-MAP Jensen offset puts the band ~17.5% above the plotted MAP line.
    if getattr(args, "band_recenter", False):
        _med = np.nanmedian(fb_samp, axis=0)
        _sh = np.where(np.isfinite(_med) & np.isfinite(map_fb), map_fb - _med, 0.0)
        fb_samp = fb_samp + _sh[None, :]
    lo68 = np.nanpercentile(fb_samp, 16, axis=0)
    hi68 = np.nanpercentile(fb_samp, 84, axis=0)
    m = (mid >= 20.0) & (mid <= 22.0) & np.isfinite(map_fb) & (map_fb > 0)
    ax.fill_between(mid[m], np.clip(lo68[m], 1e-30, None), np.clip(hi68[m], 1e-30, None),
                    color="#aec7e8", alpha=0.55, label="HBI 68%")
    ax.plot(mid[m], np.clip(map_fb[m], 1e-30, None), "-", color=C_MAP, lw=1.8,
            label="DESI-LOA GP-DLA (this work)")
    if "ho21_z34" in lit:
        h = lit["ho21_z34"]
        sel = (h["logN"] >= 20.0) & (h["logN"] <= 22.0) & (h["f"] > 0)
        ax.errorbar(h["logN"][sel], h["f"][sel],
                    yerr=[h["f"][sel] - h["lo95"][sel], h["hi95"][sel] - h["f"][sel]],
                    fmt="o", color="black", ms=4, label="Ho21 (z3-4)", alpha=0.8)
    if "n12_fN" in lit:
        n = lit["n12_fN"]
        f = 10 ** n["logf"]
        sel = (n["logN"] >= 20.0) & (n["logN"] <= 22.0)
        ax.plot(n["logN"][sel], f[sel], "^", color="darkred", ms=6, label="N12", alpha=0.8)
    ax.set_yscale("log"); ax.set_xlim(20.0, 22.0)
    ax.set_xlabel(r"$\log_{10} N_{\rm HI}$"); ax.set_ylabel(r"$f(N_{\rm HI})$")
    ax.set_title("CDDF f(N) (z-marginal, real DESI LOA)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")

    fig.suptitle("Track-C T-F LEG 3 — REAL DESI LOA CDDF measurement "
                 "(FROZEN 2LPT-0 recipe, cross-validated on london-0; no truth, no refit)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[T-F] figure -> {out_path}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(out_path, res, args, wallclock, lit):
    L = []
    L.append("# Track-C T-F LEG 3 — REAL DESI LOA CDDF measurement (frozen recipe)")
    L.append("")
    L.append(f"- Status: COMPLETE.  n_mc={args.n_mc}  seed={args.seed}  "
             f"wallclock={wallclock:.0f}s ({wallclock/60:.1f} min)")
    L.append(f"- DATA: REAL DESI Y3 LOA GP-DLA catalog `{args.loa_cat}` "
             f"(no truth → the MEASUREMENT is the deliverable, not an R0).")
    L.append(f"- n_op detections (SNR>2 & P_DLA>0.99 & DLAFLAG==0): "
             f"{res['n_op_detections']};  n_op sightlines: {res['n_op_sl']}.")
    L.append(f"- Forward kernel (FROZEN, 2LPT-0): `{args.forward_model}`")
    L.append(f"- Completeness g(N,z) (FROZEN, 2LPT-0); molly C/ρ ratio + counts "
             f"(FROZEN, 2LPT-0): `{args.molly_tsv}`")
    L.append(f"- Band: TRUTH-FREE indep (C/ρ Wilson jitter on FROZEN 2LPT-0 counts + "
             f"REAL-LOA sightline bootstrap + REAL NHI_ERR width; response FROZEN). "
             f"MAP-recentered (band_recenter={args.band_recenter}).")
    L.append(f"- Inference (gpy_dla_detection/) byte-FROZEN; no estimator-logic edit; "
             f"posterior kappa NOT attached (forward path). code_commit="
             f"`{_git_commit()}`.")
    L.append("")
    L.append("## The measurement")
    L.append("")
    z_extrap = np.asarray(res.get("z_extrapolated", np.zeros(res["n_zc"], bool)), bool)
    z_thin = np.asarray(res.get("z_thin", np.zeros(res["n_zc"], bool)), bool)
    tcz = res.get("truth_counts_perz", None)
    for l in args._limits:
        L.append(f"### NHI ≥ {l:.1f}")
        L.append("")
        L.append("| reduction | z bin | z≈ | support | MAP | 68% band | 95% band |")
        L.append("|---|---|---|---|---|---|---|")
        zbins = res["zbins"]; zmid = 0.5 * (zbins[:-1] + zbins[1:])
        for kind, name, sc in (("dndx", "dN/dX(z)", 1.0), ("omega", "10³·Ω(z)", 1000.0)):
            for k in range(res["n_zc"]):
                c = res[kind][l]["perz"][k]
                if z_extrap[k]:
                    flag = "🔴 EXTRAP"
                elif z_thin[k]:
                    flag = "⚠ thin"
                else:
                    flag = "calibrated"
                if tcz is not None and k < len(tcz):
                    flag += f" (n_truth={tcz[k]})"
                if not np.isfinite(c["MAP"]):
                    L.append(f"| {name} | [{zbins[k]:.2f},{zbins[k+1]:.2f}] | {zmid[k]:.2f} "
                             f"| {flag} | — (empty: no fine-z support) | — | — |")
                    continue
                L.append(f"| {name} | [{zbins[k]:.2f},{zbins[k+1]:.2f}] | {zmid[k]:.2f} | "
                         f"{flag} | {c['MAP']*sc:.4g} | "
                         f"[{c['q16']*sc:.4g}, {c['q84']*sc:.4g}] | "
                         f"[{c['q025']*sc:.4g}, {c['q975']*sc:.4g}] |")
            ci = res[kind][l]["integrated"]
            L.append(f"| {name} | INTEGRATED (z-marg) | all | calibrated | "
                     f"**{ci['MAP']*sc:.4g}** | "
                     f"[{ci['q16']*sc:.4g}, {ci['q84']*sc:.4g}] | "
                     f"[{ci['q025']*sc:.4g}, {ci['q975']*sc:.4g}] |")
        L.append("")
    L.append("## High-z calibration support (the z>3.5 extension)")
    L.append("")
    zbins = res["zbins"]
    sup_lim = float(res.get("support_limit", max(args._limits)))
    max_tz = float(res.get("max_truth_z", float("nan")))
    L.append(f"- 2LPT-0 truth (the calibration mock) max z_DLA = {max_tz:.3f}; truth counts "
             f"(NHI≥{sup_lim:.1f}) per coarse z-bin: "
             f"{tcz if tcz is not None else 'n/a'}.")
    cal_bins = [f"[{zbins[k]:.2f},{zbins[k+1]:.2f})" for k in range(res["n_zc"])
                if not z_extrap[k] and not z_thin[k]]
    thin_bins = [f"[{zbins[k]:.2f},{zbins[k+1]:.2f})" for k in range(res["n_zc"])
                 if z_thin[k]]
    ext_bins = [f"[{zbins[k]:.2f},{zbins[k+1]:.2f})" for k in range(res["n_zc"])
                if z_extrap[k]]
    L.append(f"- **Calibrated** z-bins (full truth support): {cal_bins or 'none'}.")
    if thin_bins:
        L.append(f"- **Thinly calibrated** z-bins (0<n_truth<{res['cfg'].completeness_z_min_count:.0f}; "
                 f"reported with their WIDE HBI band, but the z-resolved completeness g(N,z) "
                 f"is occupancy-shrunk toward the z-marginal there): {thin_bins}.")
    if ext_bins:
        L.append(f"- 🔴 **EXTRAPOLATED** z-bins (ZERO 2LPT-0 truth → g(N,z)→1 → completeness "
                 f"= the z-MARGINAL molly C): {ext_bins}.  These are FLAGGED (open grey "
                 f"marker on the figure) and EXCLUDED from any calibrated headline.  The "
                 f"indep statistical band holds g FROZEN, so it does NOT capture the "
                 f"completeness-extrapolation bias here — the plotted band is a LOWER bound "
                 f"on the true uncertainty.  Real-LOA detections DO extend into these bins, "
                 f"but the mock provides no completeness calibration to correct them.")
    L.append("")
    L.append("## Stated systematics (carried from the london-0 cross-recipe closure)")
    L.append("")
    L.append("- **Mean-flux recipe-dependence (dN/dX ~1–2%)**: the forward-response SHAPE "
             "transfers across the mock→data recipe boundary; on london-0 a single overall "
             "mean-flux amplitude rescale s≈1.01–1.02 restored R0 to <1%.  Real LOA's forest "
             "mean-flux differs from 2LPT-0, so the absolute dN/dX normalization carries a "
             "~1–2% recipe systematic (NOT applied — stated).")
    L.append("- **Ω deep-tail (~12%)**: Ω weights the high-N tail (N·f(N)), where the forest "
             "mean-flux normalization + HCD prescription differ most between recipes; on "
             "london-0 the integrated Ω R0 was ~0.88 (Ω ~12% low under the frozen recipe). "
             "The real-LOA Ω therefore carries a ~12% downward tail systematic.")
    L.append("- **Low-N edge**: the differential f(N) below the [19.5,19.7) fit floor is "
             "edge-migration-limited (non-identifiable on a 19.5-floored catalog); the "
             "headline integrals are reported at ≥20.0 and ≥20.3, above the edge.")
    L.append("- **Band scope**: the indep band reflects C/ρ-calibration + real-sightline "
             "bootstrap + NHI-measurement variance about a FROZEN calibration.  It does NOT "
             "propagate the calibration-TRANSFER uncertainty (the systematics above), which "
             "is assessed by the london-0 closure, not by the band.")
    L.append("")
    L.append("## Literature comparison")
    L.append("")
    # quick numeric proximity at the headline z and limit
    hl = args._limits[-1]
    zmid = 0.5 * (res["zbins"][:-1] + res["zbins"][1:])
    L.append(f"Headline NHI≥{hl:.1f}, z≈{zmid[1]:.2f} (mid bin):")
    cd = res["dndx"][hl]["perz"][1]; co = res["omega"][hl]["perz"][1]
    L.append(f"- this work dN/dX = {cd['MAP']:.4g}; 10³·Ω = {co['MAP']*1000:.4g}")
    if "n12_z" in lit:
        # nearest N12 z
        j = int(np.argmin(np.abs(lit["n12_z"] - zmid[1])))
        L.append(f"- N12 (z={lit['n12_z'][j]:.2f}): dN/dX = {lit['n12_dndx'][j]:.4g}  "
                 f"(ratio this/N12 = {cd['MAP']/lit['n12_dndx'][j]:.2f}); "
                 f"10³·Ω = {lit['n12_omega'][j]:.4g} "
                 f"(ratio = {co['MAP']*1000/lit['n12_omega'][j]:.2f}).")
    if "ho21_dndx" in lit:
        hz = lit["ho21_dndx"]["z"]; j = int(np.argmin(np.abs(hz - zmid[1])))
        ho_dndx = lit["ho21_dndx"]["val"][j]
        ho_om = lit["ho21_omega"]["val1e3"][j] if "ho21_omega" in lit else np.nan
        L.append(f"- Ho+2021 (this group's OWN measurement; nearest z={hz[j]:.2f}): "
                 f"dN/dX = {ho_dndx:.4g} (ratio this/Ho21 = {cd['MAP']/ho_dndx:.2f}); "
                 f"10³·Ω = {ho_om:.4g} (ratio = {co['MAP']*1000/ho_om:.2f}).")
    L.append("- The CDDF f(N) panel compares the z-marginal f(N) to Ho21 (z3-4) and N12.")
    L.append("")
    L.append(f"- Figure: `{os.path.join(args.out, 'fig_tf_loa.png')}`")
    L.append("")
    L.append("🔴 REAL-DATA PRIVACY: only AGGREGATE numbers (dN/dX/Ω/f(N) + figure) are "
             "written.  No raw real-data rows/per-object arrays are committed to the public "
             "code repo; measurement artifacts live on scratch + the private notes repo.")
    with open(out_path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[T-F] report -> {out_path}")
    return "\n".join(L)


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
    # KERNEL MODE (default 'forward' = Track-C, byte-identical to the committed run).
    # 'kappa' = the PRE-Track-C GP-posterior deconvolution kernel (the "feed-forward"
    # baseline that over-counts high-N on-mock; what Track-C's forward kernel replaces).
    # In 'kappa' mode the per-detection GP-posterior kappa2d is built ON the real-LOA
    # detections (it is a per-object posterior property, NOT a frozen population
    # property) via build_posterior_kernel, or loaded from --loa-kernel if present.
    p.add_argument("--resp-kind", default="forward", choices=["forward", "kappa"],
                   help="forward = Track-C (DEFAULT, byte-identical); kappa = pre-Track-C "
                        "GP-posterior deconvolution kernel (feed-forward baseline).")
    p.add_argument("--loa-kernel", default=None,
                   help="(kappa mode) path to a real-LOA posterior_kernel NPZ "
                        "(build_posterior_kernel output). If missing it is built in-driver.")
    p.add_argument("--loa-processed-glob", default=(
        "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1/"
        "processed/processed-main-dark-*.h5"),
                   help="(kappa mode) real-LOA processed-h5 glob for the kernel build.")
    p.add_argument("--loa-pw-samples", default=(
        "/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection/"
        "data/dr12q/processed/pw_samples_a3_172_225_50000.mat"),
                   help="(kappa mode) the 50k pw_samples grid matching the real-LOA "
                        "inference (NUM_DLA_SAMPLES=50000 — must match the h5 sample axis).")
    # the REAL LOA data
    p.add_argument("--loa-cat", default=_LOA_CAT)
    p.add_argument("--loa-truth", default=_LOA_TRUTH)
    p.add_argument("--loa-bal", default=_LOA_BAL)
    p.add_argument("--loa-mockdir", default=_LOA_MOCKDIR)
    # run knobs (match the headline perz recipe)
    p.add_argument("--out", default=_DEF_OUT)
    p.add_argument("--report-out", default=".superpowers/sdd/track_c_TF_loa_report.md")
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    # Fine z-fit-grid upper edge.  DEFAULT 3.5 = the committed/byte-identical behavior
    # (the fine-z reduction grid stops at 3.5, so coarse report bins above 3.5 are empty
    # by construction).  Raise it (e.g. 4.25) WHEN extending --zbins above 3.5 so the
    # per-bin BINNED reduction (_v2_reduce / _coarse_z_differential_f) actually has fine
    # z-columns covering [3.5, hi] — i.e. the z>3.5 detections + path-length + truth are
    # folded into the high-z report bins (a genuine binned measurement, not an empty bin).
    # The estimator's own HBIConfig.v2_z_fit_hi default (3.5) is NOT changed.
    p.add_argument("--v2-z-fit-hi", dest="v2_z_fit_hi", type=float, default=3.5,
                   help="fine z-fit-grid upper edge (default 3.5 = byte-identical). Raise "
                        "to match --zbins when reporting bins above z=3.5.")
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

    # ---- pre-flight: the real-LOA catalog + QSO/SNR + BAL must exist ----
    missing = []
    if not os.path.exists(args.loa_cat):
        missing.append(f"real-LOA catalog dir: {args.loa_cat}")
    md = args.loa_mockdir
    if not os.path.exists(os.path.join(md, "snr_cat.fits")):
        missing.append(f"real-LOA snr_cat.fits (pathlength+SNR): {md}/snr_cat.fits")
    if not (os.path.exists(os.path.join(md, "zcat.fits"))):
        missing.append(f"real-LOA zcat.fits (Z_QSO): {md}/zcat.fits")
    if not os.path.exists(args.loa_bal):
        missing.append(f"real-LOA bal_cat.fits (BAL exclusion): {args.loa_bal}")
    if not os.path.exists(args.loa_truth):
        missing.append(f"placeholder truth (empty; structural): {args.loa_truth}")
    if missing:
        msg = ("\n[T-F] DATA BLOCKER — real-LOA inputs not staged:\n  - "
               + "\n  - ".join(missing) +
               "\n\nStage: build snr_cat.fits (= zcat.fits) from the catalog's "
               "processed-main-dark-*.h5 via examples/make_snr_cat_from_processed.py "
               "(--pattern 'processed-main-dark-*.h5'); build bal_cat.fits from the "
               "production QSO catalog BI_CIV>0; write an empty placeholder dla_cat.fits.\n")
        print(msg)
        with open(os.path.abspath(args.report_out), "w") as fh:
            fh.write("# Track-C T-F LEG 3 — BLOCKED (real-LOA inputs not staged)\n\n"
                     + msg + "\n")
        return dict(status="blocked", missing=missing)

    t0 = time.time()
    print("=" * 78)
    print("TRACK-C T-F LEG 3 — REAL DESI LOA CDDF measurement (FROZEN 2LPT-0 recipe)")
    print(f"  forward kernel: {args.forward_model}")
    print(f"  real-LOA catalog: {args.loa_cat}")
    print("=" * 78)

    frozen = build_frozen_calibration(args)
    args.molly_tsv = frozen["molly_tsv"]
    ing = build_loa_ingredients(args, frozen)
    res = run_measurement(args, ing, limits, args.seed, frozen=frozen)
    wallclock = time.time() - t0

    lit = _load_literature()
    fig_path = os.path.join(args.out, "fig_tf_loa.png")
    make_figure(fig_path, res, args, lit)
    rep = write_report(os.path.abspath(args.report_out), res, args, wallclock, lit)

    # JSON dump (aggregate only)
    out_json = dict(metadata=dict(
        n_mc=args.n_mc, seed=args.seed, limits=list(limits),
        resp_kind=getattr(args, "resp_kind", "forward"),
        loa_kernel=ing.get("kernel_built_path"),
        forward_model=args.forward_model, molly_tsv=args.molly_tsv,
        loa_cat=args.loa_cat, n_op_detections=res["n_op_detections"],
        n_op_sl=res["n_op_sl"], consistency_err=res["consistency_err"],
        v2_z_fit_hi=float(args.v2_z_fit_hi),
        z_extrapolated=[bool(x) for x in np.asarray(res.get("z_extrapolated", []))],
        z_thin=[bool(x) for x in np.asarray(res.get("z_thin", []))],
        truth_counts_perz=res.get("truth_counts_perz"),
        max_truth_z=float(res.get("max_truth_z", float("nan"))),
        support_limit=float(res.get("support_limit", max(limits))),
        wallclock_s=float(wallclock), code_commit=_git_commit()),
        measurement={
            str(l): dict(
                dndx=dict(
                    perz=[res["dndx"][l]["perz"][k] for k in range(res["n_zc"])],
                    integrated=res["dndx"][l]["integrated"]),
                omega=dict(
                    perz=[res["omega"][l]["perz"][k] for k in range(res["n_zc"])],
                    integrated=res["omega"][l]["integrated"]),
            ) for l in limits},
        zbins=list(map(float, res["zbins"])))
    with open(os.path.join(args.out, "track_c_tf_loa.json"), "w") as fh:
        json.dump(out_json, fh, indent=2, default=float)
    print("\n" + rep)
    print(f"\n[T-F] DONE in {wallclock:.0f}s")
    return dict(status="complete", res=res)


if __name__ == "__main__":
    main()
