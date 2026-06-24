#!/usr/bin/env python
"""run_remp_kernel.py — EMPIRICAL TRUTH-MATCH RESPONSE kernel (R_emp) experiment.

THE PRINCIPLED WIDTH FIX (analysis-side only; NEVER touches dla_gp.py / inference).

The per-object SIR posterior kernel (build_posterior_kernel -> posterior_kernel_2lpt0.npz)
is too NARROW (isolated-TP PIT cov68=0.46, cov95=0.75 -> overconfident), so the v3
forward fit under-recovers dN/dX>=20.3 by ~2x (R0~0.52 at fit-floor 19.5; ~0.16 at
floor 20.3) and misses WALL-1. The cure is to replace each op object's narrow
per-object posterior with the EMPIRICAL truth-match response kernel

    R_emp(x_true | x_hat, SNR)

measured ON the 2LPT-0 truth-match itself (the GW-style catalog-HBI response): for
every truth-matched TP detection we have BOTH the predicted N_hat (cat NHI) AND the
truth-host logN (cat NHI_TILT_HOST, 19.0-floored so sub-DLA up-migrants keep their true
host). Histogramming (x_hat, x_true) per molly SNR cell gives a 2-D response per SNR
bin that carries the REAL N-scatter width + the prior-edge skew + the up/down-migration
asymmetry BY CONSTRUCTION (not a fitted sigma). Then each op object i (detected at
x_hat_i, SNR cell s_i) is assigned the column-normalized empirical conditional
p(x_true | x_hat-bin(x_hat_i), s_i) as its N-kernel, and a near-delta z-kernel at
z_hat_i (z is well measured; sigma_z ~ 0.0014 << 0.1, EXACTLY as the SIR kernel does).

This is FROZEN like C/rho/b_FP and is NON-CIRCULAR with WALL-1: WALL-1 reweights
truth + detections by a tilt mark, never R_emp. R_emp uses ONLY the (x_hat, x_true)
joint, which is slope-agnostic (the tilt reweights how many absorbers sit at each
x_true, not how x_hat scatters around a given x_true).

The cube is built ONCE in the EXACT op_base order v3x_build_forward rebuilds
((S2N_RED>snr_min)&(P_DLA>p_dla_min)&good_mask on cat_cut row order), cached as
kappa[n_op_base, n_Nbins, n_zf] (the SAME key/shape build_posterior_kernel writes),
so the consumption path is UNCHANGED: _load_kernel_into_cfg attaches it to
cfg._posterior_kernel_2d and v3x_build_forward -> build_A_ib -> _build_A_ib_kappa2d
consumes it directly. No re-inference; reuses cddf_catalog_hbi kernels only.

Stages (mirrors run_phase3d_postkernel.py):
  build  build R_emp cube + cache <out>/posterior_kernel_2lpt0.npz (key 'kappa').
  2      v3 bspbody MAP fit WITH R_emp -> PIT cov68 + dN/dX + Omega.
  3      WALL-1 tilt-closure (estimator=v3, closure_R0_mode=divide, dalpha=0.5).

OUT default: /scratch/.../cddf_o3_realdata/phase3d_experiments/r_emp/
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import cddf_catalog_hbi as H
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    HBIConfig, load_molly_matrix, load_and_cut_catalog, build_fine_grid,
    _build_qso_lookup, _op_mask_and_slots, _cell_index, _fine_z_grid,
)

# the SAME ESS tiers build_posterior_kernel writes (consumed by the gate band-ESS KILL)
ESS_TIERS = (20.3, 20.6, 21.0)


def _make_cfg(args) -> HBIConfig:
    zbins = tuple(float(x) for x in args.zbins.split(","))
    report_limits = tuple(float(x) for x in args.report_limits.split(","))
    cfg = HBIConfig(
        catalog_dir=args.catalog_dir, truth_path=args.truth,
        bal_cat_path=args.bal_cat, molly_tsv=args.molly_tsv, out_dir=args.out,
        mockdir=args.mockdir or os.path.dirname(args.truth),
        zbins=zbins, n_mc=args.n_mc, rng_seed=args.seed,
        fp_estimator="purity_mixture", no_bal=True,
        report_logN_limits=report_limits,
        v3_family="bspbody",
        v3_logN_fit_floor=args.fit_floor,
        v3_lambda_bspbody=args.lambda_bspbody,
        lam_rf_min=args.lam_rf_min,
        v2_z_fit_lo=zbins[0], v2_z_fit_hi=zbins[-1], v2_z_fit_step=0.1,
    )
    return cfg


def compute_R_response(cfg, cat_cut, good_mask, fine_grid, mm,
                       smooth_bins=1.0, n_floor=20, host_col="NHI_TILT_HOST",
                       verbose=False):
    """Measure the UNTILTED 2-D (x_hat, x_true) response matrix R[s, jhat, jtru].

    This is the SLOPE-AGNOSTIC, population-FROZEN operator (design §2): it records
    only how x_hat scatters around a given x_true per SNR cell, NOT how many absorbers
    sit at each x_true. Built ONCE on the untilted truth-match; ``assign_R_emp_to_catalog``
    then re-binds it onto ANY (re-inferred, possibly-tilted) catalog without rebuilding.

    R definition (the 2-D (x_hat,x_true) response per SNR bin):
      training set = truth-matched TPs (finite host_col) passing the op_base mask
        (S2N_RED>snr_min & P_DLA>p_dla_min & good_mask) on cat_cut — the response is
        measured on the very detections we forward-model.
      bin x_hat and x_true on the SAME fine logN grid; bin SNR by molly snr_edges.
      cube R[s, jhat, jN_true] = count of training TPs in (SNR cell s, x_hat-bin jhat,
        x_true-bin jN_true). 2-D Gaussian smoothing (smooth_bins fine bins, in BOTH
        x_hat and x_true) regularizes the response before column-normalization. The
        all-SNR marginal R_par and the PRE-smoothing per-cell / per-x_hat-bin occupancy
        (occ_cell, occ_par) are kept for the SNR-pool shrinkage at re-bind time.

    Returns a dict ``R_response`` carrying everything ``assign_R_emp_to_catalog`` needs:
      R          float64 [n_snr, n_Nbins, n_Nbins]  -- smoothed per-SNR response counts
      R_par      float64 [n_Nbins, n_Nbins]         -- smoothed all-SNR marginal
      occ_cell   float64 [n_snr, n_Nbins]           -- PRE-smoothing per-cell occupancy
      occ_par    float64 [n_Nbins]                  -- PRE-smoothing per-x_hat-bin occ
      edges_N, n_Nbins, n_snr, n_train, host_col    -- grid/provenance metadata
    """
    logN_lo, logN_hi, N_b, dN_b = fine_grid
    n_Nbins = len(logN_lo)
    edges_N = np.concatenate([logN_lo, [logN_hi[-1]]])
    n_snr = len(mm.snr_edges) - 1

    # op_base set (only its size + the host match define the training TPs here).
    op_mask, _slot_op, _tid_op, _dlaid_op = _op_mask_and_slots(cat_cut, good_mask, cfg)
    n_op = int(op_mask.sum())

    # ---- training TP set for the response (truth-matched, op-cut) ----
    host = np.asarray(cat_cut[host_col], float)
    is_tp_train = np.isfinite(host) & op_mask
    xhat_tr = np.asarray(cat_cut["NHI"], float)[is_tp_train]
    xtru_tr = host[is_tp_train]
    snr_tr = np.asarray(cat_cut["S2N_RED"], float)[is_tp_train]
    i_snr_tr = _cell_index(mm, xhat_tr, snr_tr)[0]
    jhat_tr = np.searchsorted(edges_N, xhat_tr, side="right") - 1
    jtru_tr = np.searchsorted(edges_N, xtru_tr, side="right") - 1
    valid_tr = ((jhat_tr >= 0) & (jhat_tr < n_Nbins)
                & (jtru_tr >= 0) & (jtru_tr < n_Nbins))
    n_train = int(valid_tr.sum())
    if verbose:
        print(f"[R_emp] op rows = {n_op}; training TP pairs (finite {host_col}) "
              f"= {n_train} ({100.0*n_train/max(n_op,1):.1f}% of op)")

    # cube R[s, jhat, jtru] -- the 2-D (x_hat,x_true) response per SNR bin
    R = np.zeros((n_snr, n_Nbins, n_Nbins), dtype=np.float64)
    np.add.at(R, (i_snr_tr[valid_tr], jhat_tr[valid_tr], jtru_tr[valid_tr]), 1.0)
    R_par = R.sum(axis=0)                                       # all-SNR marginal

    # 2-D Gaussian smoothing in (jhat, jtru) before column-normalization
    if smooth_bins and smooth_bins > 0:
        try:
            from scipy.ndimage import gaussian_filter
            for s in range(n_snr):
                R[s] = gaussian_filter(R[s], sigma=smooth_bins, mode="constant")
            R_par = gaussian_filter(R_par, sigma=smooth_bins, mode="constant")
        except Exception as e:                                 # noqa: BLE001
            print(f"[R_emp] WARN: scipy gaussian_filter unavailable ({e}); no smoothing")

    # SNR-pool shrinkage: response cells (s,jhat) with < n_floor pairs borrow R_par
    occ_cell = np.zeros((n_snr, n_Nbins))                      # PRE-smoothing occupancy
    np.add.at(occ_cell, (i_snr_tr[valid_tr], jhat_tr[valid_tr]), 1.0)
    occ_par = occ_cell.sum(axis=0)                             # per x_hat-bin parent occ

    return dict(R=R, R_par=R_par, occ_cell=occ_cell, occ_par=occ_par,
                edges_N=edges_N, n_Nbins=n_Nbins, n_snr=n_snr, n_train=n_train,
                host_col=host_col, smooth_bins=smooth_bins, n_floor=n_floor)


def assign_R_emp_to_catalog(R_response, cfg, cat_cut, good_mask, fine_grid, mm,
                            verbose=False):
    """Re-bind a FROZEN untilted ``R_response`` (from :func:`compute_R_response`) onto an
    ARBITRARY catalog's op detections, producing the per-op-object kappa cube.

    This is the load-bearing WALL-1 full-injection mechanism (design §5.3): each op
    detection i of ``cat_cut`` (which may be a GENUINELY RE-INFERRED, tilted catalog) is
    assigned the UNTILTED response row p(x_true | x_hat-bin(x_hat_i), SNR-cell s_i) from
    ``R_response``, with SNR-pool shrinkage in starved cells, and a near-delta z-kernel at
    z_hat_i. NOTHING about the response is re-measured from ``cat_cut`` — the operator is
    frozen at the untilted slope and merely BINNED to the new detections' (x_hat, SNR).

    On the SAME catalog/grid that built ``R_response``, this reproduces
    :func:`build_R_emp` byte-for-byte (TDD-gated, tests/test_remp_rebind.py).

    Returns (kappa, ess, info) with the SAME shapes/keys as :func:`build_R_emp`:
      kappa : float32 [n_op, n_Nbins, n_zf]  -- per-op p(x_true,z_true|det i).
      ess   : dict tier->float32[n_op]       -- per-object response ESS >= tier.
      info  : n_op / n_train / n_snr / fallback counts / tid_op / slot_op / dlaid_op.
    """
    logN_lo, logN_hi, N_b, dN_b = fine_grid
    z_edges_fine = _fine_z_grid(cfg)
    n_zf = len(z_edges_fine) - 1

    R = R_response["R"]
    R_par = R_response["R_par"]
    occ_cell = R_response["occ_cell"]
    occ_par = R_response["occ_par"]
    edges_N = R_response["edges_N"]
    n_Nbins = R_response["n_Nbins"]
    n_floor = R_response["n_floor"]

    # ---- op_base set of THIS catalog (the EXACT order v3x_build_forward rebuilds) ----
    op_mask, slot_op, tid_op, dlaid_op = _op_mask_and_slots(cat_cut, good_mask, cfg)
    n_op = int(op_mask.sum())
    xhat_op = np.asarray(cat_cut["NHI"], float)[op_mask]
    zhat_op = np.asarray(cat_cut["Z_DLA"], float)[op_mask]
    snr_op = np.asarray(cat_cut["S2N_RED"], float)[op_mask]
    i_snr_op = _cell_index(mm, xhat_op, snr_op)[0]             # molly SNR cell per op
    jhat_op = np.searchsorted(edges_N, xhat_op, side="right") - 1
    jhat_op = np.clip(jhat_op, 0, n_Nbins - 1)
    # fine z-bin of each op object (near-delta z-kernel)
    kz_op = np.searchsorted(z_edges_fine, zhat_op, side="right") - 1
    kz_valid = (kz_op >= 0) & (kz_op < n_zf)

    # ---- assemble per-op-object kappa ----
    kappa = np.zeros((n_op, n_Nbins, n_zf), dtype=np.float32)
    ess = {t: np.zeros(n_op, dtype=np.float32) for t in ESS_TIERS}
    n_fallback_par = 0
    n_fallback_delta = 0
    tier_lo = {t: np.searchsorted(edges_N, t, side="right") - 1 for t in ESS_TIERS}

    for i in range(n_op):
        if not kz_valid[i]:
            continue                                           # z outside grid -> all-zero
        s = int(i_snr_op[i]); jh = int(jhat_op[i])
        resp = R[s, jh, :]                                     # p(x_true | x_hat-bin jh, s)
        occ = occ_cell[s, jh]
        # SNR-pool shrinkage to the parent response in starved cells
        if occ < n_floor:
            resp = R_par[jh, :]
            occ = occ_par[jh]
            n_fallback_par += 1
        tot = resp.sum()
        if tot <= 0:
            # no empirical support at this x_hat-bin even after pooling -> delta at x_hat
            row_N = np.zeros(n_Nbins, dtype=np.float64)
            row_N[jh] = 1.0
            n_fallback_delta += 1
        else:
            row_N = resp / tot
        kappa[i, :, kz_op[i]] = row_N.astype(np.float32)       # z-delta x p(x_true)
        # per-object ESS by report tier: the EMPIRICAL response mass-weighted N_eff of
        # the training pairs feeding this cell, restricted to x_true>=tier.
        for t in ESS_TIERS:
            jlo = tier_lo[t]
            w = resp[jlo:]                                     # response counts >=tier
            sw = w.sum()
            if sw > 0:
                ess[t][i] = float(occ * (sw / tot) if tot > 0 else 0.0)

    if verbose:
        # wall-truncate above drop_top is automatic: the fine grid already drops >22.4
        nz = np.asarray([kappa[i].sum() for i in range(min(n_op, 5))])
        print(f"[R_emp] kappa {kappa.shape} float32; fallback(SNR-pool)={n_fallback_par}, "
              f"fallback(delta)={n_fallback_delta}, sample row-sums={nz}")
        for t in ESS_TIERS:
            e = ess[t]; pos = e[e > 0]
            print(f"[R_emp] ESS(>={t}): median={np.median(pos) if pos.size else 0:.1f} "
                  f"frac<30={np.mean(e < 30):.3f}")
        # response-width sanity: the column-normalized response std at a few x_hat bins
        for xh in (20.0, 20.3, 20.6, 21.0):
            jh = int(np.clip(np.searchsorted(edges_N, xh, side="right") - 1,
                             0, n_Nbins - 1))
            col = R_par[jh, :]
            if col.sum() > 0:
                p = col / col.sum()
                mids = 0.5 * (logN_lo + logN_hi)
                mean = float((p * mids).sum())
                std = float(np.sqrt((p * (mids - mean) ** 2).sum()))
                print(f"[R_emp]  x_hat={xh:.1f}: E[x_true]={mean:.3f} "
                      f"(bias {mean-xh:+.3f}), sd(x_true)={std:.3f}")
    info = dict(n_op=n_op, n_train=int(R_response["n_train"]), n_snr=int(R_response["n_snr"]),
                n_fallback_par=n_fallback_par, n_fallback_delta=n_fallback_delta,
                tid_op=tid_op, slot_op=slot_op,
                dlaid_op=np.array(dlaid_op, dtype=object))
    return kappa, ess, info


def build_R_emp(cfg, cat_cut, good_mask, fine_grid, mm,
                smooth_bins=1.0, n_floor=20, host_col="NHI_TILT_HOST",
                verbose=True):
    """Build the empirical truth-match response kernel cube in op_base order.

    Now a thin composition of :func:`compute_R_response` (measure the slope-agnostic
    untilted response R[s,jhat,jtru]) + :func:`assign_R_emp_to_catalog` (re-bind it onto
    THIS cat's op detections). Byte-identical to the pre-refactor monolith (TDD-gated,
    tests/test_remp_rebind.py): the decomposition exists so the full-injection test can
    freeze the response on the untilted cat and re-bind it onto the re-inferred tilted
    catalog (design §5.3) — never re-measuring the response from the tilted population.

    Returns (kappa, ess, info) — see :func:`assign_R_emp_to_catalog` for the schema.
    """
    R_response = compute_R_response(
        cfg, cat_cut, good_mask, fine_grid, mm,
        smooth_bins=smooth_bins, n_floor=n_floor, host_col=host_col, verbose=verbose)
    return assign_R_emp_to_catalog(
        R_response, cfg, cat_cut, good_mask, fine_grid, mm, verbose=verbose)


def stage_build(cfg, args):
    print("=" * 70)
    print("[build] R_emp empirical truth-match response kernel (op_base order)")
    print("=" * 70)
    t0 = time.time()
    mm = load_molly_matrix(cfg.molly_tsv)
    truth_floor = float(mm.nhi_edges[0])
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(args.host_truth_floor, truth_floor))
    print(f"    cat_cut meta: {meta}")
    fine = build_fine_grid(cfg)
    logN_lo, logN_hi, N_b, dN_b = fine
    z_edges_fine = _fine_z_grid(cfg)
    kappa, ess, info = build_R_emp(
        cfg, cat_cut, good_mask, fine, mm,
        smooth_bins=args.smooth_bins, n_floor=args.n_floor,
        host_col=args.host_col, verbose=True)
    out_npz = os.path.join(cfg.out_dir, "posterior_kernel_2lpt0.npz")
    os.makedirs(cfg.out_dir, exist_ok=True)
    np.savez_compressed(
        out_npz, kappa=kappa,
        ess_203=ess[20.3], ess_206=ess[20.6], ess_210=ess[21.0],
        n_Nbins=len(logN_lo), n_zf=len(z_edges_fine) - 1,
        logN_lo=logN_lo, logN_hi=logN_hi, z_edges_fine=z_edges_fine,
        tid_op=info["tid_op"], slot_op=info["slot_op"],
        dlaid_op=info["dlaid_op"],
        n_no_support=info["n_fallback_delta"], n_unmatched=0,
        norm=np.zeros(info["n_op"]),
        kernel_kind="R_emp", smooth_bins=args.smooth_bins,
        n_floor=args.n_floor, host_col=args.host_col)
    print(f"[build] DONE: R_emp kappa {kappa.shape} cached -> {out_npz} "
          f"({time.time()-t0:.0f}s)")
    return out_npz


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["build", "2", "3", "all"], default="all")
    p.add_argument("--out", default=("/scratch/cavestru_root/cavestru0/mfho/"
                                     "cddf_o3_realdata/phase3d_experiments/r_emp"))
    p.add_argument("--catalog-dir",
                   default=("/scratch/cavestru_root/cavestru0/mfho/"
                            "gl_prod_2lpt0_v1_20260526/combined_catalog/"))
    p.add_argument("--truth",
                   default=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                            "qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"))
    p.add_argument("--bal-cat",
                   default=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                            "qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"))
    p.add_argument("--molly-tsv",
                   default=("/scratch/cavestru_root/cavestru0/mfho/"
                            "gl_prod_2lpt0_v1_20260526/figures_molly/molly_matrix.tsv"))
    p.add_argument("--mockdir", default=None)
    p.add_argument("--processed-glob", default=H.DEF_PROCESSED_GLOB)  # unused by R_emp
    p.add_argument("--pw-samples", default=H.DEF_PW_SAMPLES)          # unused by R_emp
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3,20.6")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=911.0,
                   help="rest-frame blue edge: 911.0=full Lyα+Lyβ (default); 1025.0="
                        "Lyα-only forest (pair with the lya_only molly matrix). Restricts "
                        "the catalog/truth cut + R_emp training pairs consistently.")
    p.add_argument("--dalpha", type=float, default=0.5)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    p.add_argument("--n-mc", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=16)
    # R_emp build knobs
    p.add_argument("--smooth-bins", type=float, default=1.0,
                   help="2-D Gaussian smoothing of the response (fine bins, 0.1 dex)")
    p.add_argument("--n-floor", type=int, default=20,
                   help="SNR-pool shrinkage occupancy floor (deep-tail cells)")
    p.add_argument("--host-col", default="NHI_TILT_HOST",
                   help="truth-host column for x_true (NHI_TILT_HOST=19.0-floored, "
                        "captures sub-DLA up-migrants; NHI_TRUE=matrix-floored)")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    cfg = _make_cfg(args)
    out_npz = os.path.join(cfg.out_dir, "posterior_kernel_2lpt0.npz")

    if args.stage in ("build", "all"):
        out_npz = stage_build(cfg, args)
    if args.stage in ("2", "all"):
        if not os.path.exists(out_npz):
            raise SystemExit(f"stage2 needs the R_emp cube; {out_npz} missing "
                             "(run --stage build first)")
        # reuse run_phase3d_postkernel.stage2 (it just attaches cfg._posterior_kernel_2d
        # and runs the v3 point fit + PIT diagnostic on whatever kappa is cached).
        from CDDF_analysis.hbi.run_phase3d_postkernel import stage2_v3_fit
        stage2_v3_fit(cfg, args, out_npz)
    if args.stage in ("3", "all"):
        if not os.path.exists(out_npz):
            raise SystemExit(f"stage3 needs the R_emp cube; {out_npz} missing "
                             "(run --stage build first)")
        from CDDF_analysis.hbi.run_phase3d_postkernel import stage3_wall1
        stage3_wall1(cfg, args, out_npz)


if __name__ == "__main__":
    main()
