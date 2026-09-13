#!/usr/bin/env python
"""rebuild_a0_products.py — A0v2: close support-mismatch instance #8 on the
VALIDATION side, without touching the frozen ``load_and_cut_catalog``.

WHY (``A0_SUPPORT_REPORT.md`` §5).  ``build_scan_packs.py`` selects the pack's
``counts`` with the λ/z window applied to **``Z_DLA`` alone** (observable-only:
"truth never enters the selection", ckpt-10.8), while
``cddf_catalog_hbi.load_and_cut_catalog`` step 6 calls
``make_lambda_z_BAL_cuts(..., use_truth_z=True)``, which applies the window to
**``min/max(Z_DLA, Z_TRUE)``**.  The FP census and the empirical operators are
therefore on a TRUTH-LEAKING selection that differs from the pack's ``counts``
by 877 / 965 / 861 rows (1.00 / 1.10 / 0.99 %) at fixed collar.

WHAT THIS MODULE DOES.  It re-implements ONLY step 6 of
``load_and_cut_catalog`` — the λ_rf + z_qso + BAL cut — with

    * ``z_cut_columns = "zdla_only"``  (the pack's own convention), and
    * an arbitrary ``collar_kms`` (3300 for the A0 support),

and calls the COMMITTED machinery for everything else: ``load_catalog_dir``,
``_build_qso_lookup``, the sentinel filter, ``match_truth_to_cat_molly``
(steps 1-5), ``good_mask``/``is_TP`` (steps 7-8),
``track_c_tf_saclay._snap_off_molly_edges``, and ``extract_pack.bin_counts_cks``.
Nothing under ``CDDF_analysis/`` is imported-and-monkeypatched, modified or
written; ``validation/fp_ladder/build_fp_census.py`` and
``validation/absorber_diag/build_matched_ops.py`` are left untouched (both are
tracked on this branch).

FIDELITY IS GATED, NOT ASSERTED BY FIAT.  Run with the ORIGINAL convention
(``z_cut_columns="minmax"``, ``collar_kms=3000``) this re-implementation must
reproduce, BIT-EXACTLY:
  * all seven 17.2-floor census blocks of ``fp_census_<fam>.npz``;
  * the adopted pack's ``truth_counts`` / ``truth_counts_bks``;
  * the six contract arrays of ``empirical_ops_<fam>.npz``.
Only then are the A0v2 products (Z_DLA-only, collar 3300) written.

PRIMARY GATE.  ``counts_all`` of the A0v2 census must equal the A0 pack's
``counts`` array EXACTLY; any residual is reported, not tolerated silently.

NO SAMPLER IS RUN.  ENV: ``gpdla`` (jax-free).

Usage
-----
    python validation/absorber_ladder/support/rebuild_a0_products.py \
        --family 2lpt0 london0 saclay0 \
        --out /scratch/.../absorber_ladder_2026-09-13/support
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import platform
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_ABSDIAG = os.path.join(_REPO, "validation", "absorber_diag")
sys.path.insert(0, _HERE)
sys.path.insert(0, _ABSDIAG)
sys.path.insert(0, _REPO)

from binning import bin_index, coarse_block_sum, safe_ratio      # noqa: E402
from support_contract import (                                   # noqa: E402
    NO_TRUTH_SIDE, ROW_SELECTION_FIELDS, SUPPORT_FIELDS, Z_CUT_ZDLA_ONLY,
    Z_CUT_ZDLA_OR_ZTRUE, SupportContractError, assert_same_support,
    check_support_consistency, file_sha256, stamp, stamp_array, support_id,
)
import build_a0_support as A0                                    # noqa: E402

FAMILIES = A0.FAMILIES
LYA = 1215.67
C_KMS = 299792.458
BASIS_FLOOR = A0.BASIS_FLOOR          # 19.0
CENSUS_FLOOR = A0.CENSUS_FLOOR        # 17.2
SLOT_EPS = A0.SLOT_EPS
HOST_SLOT_NAMES = A0.HOST_SLOT_NAMES
CENSUS_BLOCKS = A0.CENSUS_BLOCKS
ADOPTED_COLLAR = A0.ADOPTED_COLLAR    # 3000.0
A0_COLLAR = A0.SCANPACK_COLLAR        # 3300.0

#: the six arrays ``run_ladder.py --fix`` reads out of an ops NPZ
OPS_CONTRACT_KEYS = ("C_true_bKs", "C_true_bs", "M_true_sKcb", "E_true_cKsb",
                     "N_match_cksb", "P6b_cks")

OPS_OF_RECORD = os.path.join(_ABSDIAG, "empirical_ops_{fam}.npz")


# ---------------------------------------------------------------------------
# step 6, re-implemented
# ---------------------------------------------------------------------------
def lambda_z_bal_mask(z_dla, z_true, z_qso, tids, bal_tids, *, lam_rf_min,
                      lam_rf_max, z_qso_min, z_qso_max, collar_kms,
                      z_cut_columns):
    """``make_lambda_z_BAL_cuts`` with the collar AND the z column(s) as inputs.

    ``z_cut_columns``:
      * ``"minmax"``     — the committed behaviour (``use_truth_z=True``): the
        window must contain BOTH ``Z_DLA`` and ``Z_TRUE`` (NaN ``Z_TRUE``
        replaced by ``Z_DLA``, exactly as cell 19 does);
      * ``"zdla_only"``  — the pack's observable-only behaviour
        (``use_truth_z=False``): the window is applied to ``Z_DLA`` alone.
    """
    z_dla = np.asarray(z_dla, float)
    z_qso = np.asarray(z_qso, float)
    coll = float(collar_kms) / C_KMS
    z_lo = np.maximum(3600.0 / LYA - 1.0,
                      lam_rf_min * (1 + z_qso) / LYA - 1.0 + coll)
    z_hi = np.minimum(z_qso - coll,
                      lam_rf_max * (1 + z_qso) / LYA - 1.0 - coll)
    if z_cut_columns == "minmax":
        if z_true is None:
            raise ValueError("z_cut_columns='minmax' needs a Z_TRUE column")
        zt = np.asarray(z_true, float)
        zt = np.where(np.isnan(zt), z_dla, zt)
        z_min = np.minimum(z_dla, zt)
        z_max = np.maximum(z_dla, zt)
    elif z_cut_columns == "zdla_only":
        z_min = z_max = z_dla
    else:
        raise ValueError(f"unknown z_cut_columns {z_cut_columns!r}")
    m = (z_max < z_hi) & (z_min > z_lo)
    m &= (z_qso > z_qso_min) & (z_qso < z_qso_max)
    if bal_tids is not None:
        m &= ~np.isin(np.asarray(tids, np.int64),
                      np.fromiter(bal_tids, dtype=np.int64))
    return m


def selection_pass(ep, family, work_dir, floor, *, collar_kms, z_cut_columns):
    """Steps 1-5 + 7-8 of ``load_and_cut_catalog`` verbatim (COMMITTED calls),
    with step 6 replaced by ``lambda_z_bal_mask``."""
    import fitsio
    from astropy.table import Table
    from CDDF_analysis.hbi.cddf_catalog_hbi import _build_qso_lookup
    from CDDF_analysis.hbi import track_c_tf_saclay as TS
    sys.path.insert(0, os.path.join(_REPO, "examples"))
    from gp_native_pc_plots import load_catalog_dir
    from molly_faithful_pc_plots import match_truth_to_cat_molly

    t0 = time.time()
    cfg = ep._make_cfg(family, work_dir)
    from CDDF_analysis.hbi.cddf_catalog_hbi import load_molly_matrix
    mm = load_molly_matrix(cfg.molly_tsv)
    qso_lookup = _build_qso_lookup(cfg)

    # 1. detection catalog + S2N_RED rename
    cat = load_catalog_dir(cfg.catalog_dir)
    cat["S2N_RED"] = np.asarray(cat["SNR_REDSIDE"], dtype=float)
    n_loaded = len(cat)

    # 2. truth load, floor, per-QSO (S2N_RED, Z_QSO) attach
    truth = Table(fitsio.read(cfg.truth_path, ext=1))
    z_col = next((c for c in ("Z_DLA", "Z_DLA_NO_RSD", "Z")
                  if c in truth.colnames), None)
    if z_col != "Z_DLA":
        truth.rename_column(z_col, "Z_DLA")
    truth["Z_TRUTH"] = np.asarray(truth["Z_DLA"], dtype=float)
    truth = truth[np.asarray(truth["NHI"], dtype=float) >= float(floor)]
    t_tids = np.asarray(truth["TARGETID"], dtype=np.int64)
    t_s2n = np.full(len(truth), np.nan)
    t_zq = np.full(len(truth), np.nan)
    for i, t in enumerate(t_tids):
        v = qso_lookup.get(int(t))
        if v is not None:
            t_s2n[i], t_zq[i] = v
    truth["S2N_RED"] = t_s2n
    truth["Z_QSO"] = t_zq
    truth = truth[~np.isnan(t_s2n) & ~np.isnan(t_zq)]

    # 3. sentinel filter BEFORE matching
    nhi_err = np.asarray(cat["NHI_ERR"], dtype=float)
    zdla_err = np.asarray(cat["Z_DLA_ERR"], dtype=float)
    sentinel = (nhi_err == -1) | (zdla_err == -1)
    n_sentinel = int(sentinel.sum())
    cat = cat[~sentinel]

    # 4. BAL set: ALL bal_cat TIDs
    bal_tids = None
    if cfg.no_bal:
        bal = fitsio.read(cfg.bal_cat_path, ext=1, columns=["TARGETID"])
        bal_tids = set(int(r["TARGETID"]) for r in bal)

    # 5. primary truth match BEFORE cuts (host_floor == floor here, so the
    #    hierarchical NHI_TILT_HOST branch is never taken — same as the census
    #    builder's and build_matched_ops' two passes)
    iter_order = "input" if cfg.molly_input_order else "nhi_desc"
    _tp, cat_NHI_TR, cat_Z_TR, _tm = match_truth_to_cat_molly(
        cat, truth, cfg.dz_rel, cat_iter_order=iter_order)
    cat["NHI_TRUE"] = cat_NHI_TR
    cat["Z_TRUE"] = cat_Z_TR
    cat["NHI_TILT_HOST"] = np.asarray(cat_NHI_TR, dtype=float).copy()

    # 6. THE ONLY RE-IMPLEMENTED STEP
    kw = dict(lam_rf_min=float(cfg.lam_rf_min), lam_rf_max=float(cfg.lam_rf_max),
              z_qso_min=float(cfg.z_qso_min), z_qso_max=float(cfg.z_qso_max),
              collar_kms=float(collar_kms))
    m_cat = lambda_z_bal_mask(cat["Z_DLA"], cat["Z_TRUE"], cat["Z_QSO"],
                              cat["TARGETID"], bal_tids,
                              z_cut_columns=z_cut_columns, **kw)
    cat_cut = cat[m_cat]
    # the truth table has no Z_TRUE: the committed call is use_truth_z=False,
    # i.e. ALWAYS Z_DLA-only, for BOTH conventions
    m_tru = lambda_z_bal_mask(truth["Z_DLA"], None, truth["Z_QSO"],
                              truth["TARGETID"], bal_tids,
                              z_cut_columns="zdla_only", **kw)
    truth_cut = truth[m_tru]

    # 7./8. + the committed interior-edge tie-break
    TS._snap_off_molly_edges(cat_cut, truth_cut, mm)
    good_mask = (np.asarray(cat_cut["DLAFLAG"], dtype=int) == 0)
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask

    det = dict(nhat=np.asarray(cat_cut["NHI"], float)[op],
               zobs=np.asarray(cat_cut["Z_DLA"], float)[op],
               snr=np.asarray(cat_cut["S2N_RED"], float)[op],
               nhi_true=np.asarray(cat_cut["NHI_TRUE"], float)[op],
               z_true=np.asarray(cat_cut["Z_TRUE"], float)[op],
               zqso=np.asarray(cat_cut["Z_QSO"], float)[op])
    tru = dict(nhi=np.asarray(truth_cut["NHI"], float),
               z=np.asarray(truth_cut["Z_DLA"], float),
               snr=np.asarray(truth_cut["S2N_RED"], float),
               zqso=np.asarray(truth_cut["Z_QSO"], float))
    return dict(det=det, truth=tru, cfg=cfg, mm=mm,
                n_loaded=int(n_loaded), n_sentinel_dropped=n_sentinel,
                n_cat_cut=int(len(cat_cut)), n_op=int(op.sum()),
                n_truth_cut=int(len(truth_cut)), floor=float(floor),
                collar_kms=float(collar_kms), z_cut_columns=z_cut_columns,
                seconds=round(time.time() - t0, 1))


# ---------------------------------------------------------------------------
def slot_masks(nhi_true):
    n = np.asarray(nhi_true, float)
    host = np.isfinite(n)
    out = {"hostless": ~host}
    for name in HOST_SLOT_NAMES:
        lo, hi = A0._slot_bounds(name)
        out[name] = (host & (n >= lo - SLOT_EPS)
                     & ((n < hi - SLOT_EPS) if np.isfinite(hi)
                        else np.ones_like(host)))
    stacked = np.vstack([out[k] for k in ("hostless",) + HOST_SLOT_NAMES])
    if not np.all(stacked.sum(axis=0) == 1):
        raise SystemExit("SLOT PARTITION FAILED")
    return out


def census_blocks(ep, det):
    masks = slot_masks(det["nhi_true"])
    out = {}
    out["counts_all"], _ = ep.bin_counts_cks(det["nhat"], det["zobs"],
                                             det["snr"])
    for name in ("hostless",) + HOST_SLOT_NAMES:
        m = masks[name]
        out[name], _ = ep.bin_counts_cks(det["nhat"][m], det["zobs"][m],
                                         det["snr"][m])
    return out


def build_ops(bmo, pk, det19, det172, tru19, ntrue, nhat, zf, snr_e, kz,
              snr_min):
    """The operator block of ``build_matched_ops.build``, on a given pass pair.

    Structurally identical to the committed builder (same helpers, same axis
    order, same fallback convention); only the pass that feeds it changes.
    """
    B, C, Kf, S = len(ntrue) - 1, len(nhat) - 1, len(zf) - 1, len(snr_e) - 1
    tc_bks, _, _ = bmo.truth_hist_bks(tru19, ntrue, zf, snr_e, snr_min)
    N_match_cksb, _ = bmo.bin4(det19["nhat"], det19["zobs"], det19["snr"],
                               det19["nhi_true"], nhat, zf, snr_e, ntrue)
    N_match_true_z, _ = bmo.bin4(det19["nhat"], det19["z_true"], det19["snr"],
                                 det19["nhi_true"], nhat, zf, snr_e, ntrue)
    _bq = bin_index(ntrue, det19["nhi_true"])
    _kq = bin_index(zf, det19["z_true"])
    _sq = np.clip(bin_index(snr_e, det19["snr"]), 0, S - 1)
    _okq = ((_bq >= 0) & (_bq < B) & (_kq >= 0) & (_kq < Kf)
            & np.isfinite(det19["nhi_true"]))
    N_det_all_bks = np.zeros((B, Kf, S), float)
    np.add.at(N_det_all_bks, (_bq[_okq], _kq[_okq], _sq[_okq]), 1.0)

    host = np.isfinite(det19["nhi_true"])
    above_top = host & (det19["nhi_true"] >= ntrue[-1] - SLOT_EPS)
    host_above_cks, _ = bmo.bin3(det19["nhat"][above_top],
                                 det19["zobs"][above_top],
                                 det19["snr"][above_top], nhat, zf, snr_e)
    lo_host = (host & (det19["nhi_true"] >= BASIS_FLOOR - SLOT_EPS)
               & (det19["nhi_true"] < 19.5 - SLOT_EPS))
    hi_host = (host & (det19["nhi_true"] >= 19.5 - SLOT_EPS)
               & (det19["nhi_true"] < ntrue[-1] - SLOT_EPS))
    N_match_lo_cks, _ = bmo.bin3(det19["nhat"][lo_host], det19["zobs"][lo_host],
                                 det19["snr"][lo_host], nhat, zf, snr_e)
    N_match_hi_cks, _ = bmo.bin3(det19["nhat"][hi_host], det19["zobs"][hi_host],
                                 det19["snr"][hi_host], nhat, zf, snr_e)
    cnt_obs, _ = bmo.bin3(det19["nhat"], det19["zobs"], det19["snr"],
                          nhat, zf, snr_e)

    # the census-floor classes
    n2 = det172["nhi_true"]
    h2 = np.isfinite(n2)
    m_p6b = h2 & (n2 >= CENSUS_FLOOR - SLOT_EPS) & (n2 < BASIS_FLOOR - SLOT_EPS)
    P6b_cks, _ = bmo.bin3(det172["nhat"][m_p6b], det172["zobs"][m_p6b],
                          det172["snr"][m_p6b], nhat, zf, snr_e)
    hostless_cks, _ = bmo.bin3(det172["nhat"][~h2], det172["zobs"][~h2],
                               det172["snr"][~h2], nhat, zf, snr_e)

    # operators
    N_det_bks_true = N_match_true_z.sum(axis=0).transpose(2, 0, 1)
    N_det_bks_obs = N_match_cksb.sum(axis=0).transpose(2, 0, 1)
    tc_bKs = coarse_block_sum(tc_bks, kz, axis=1)
    Nd_bKs_true = coarse_block_sum(N_det_bks_true, kz, axis=1)
    Nd_bKs_obs = coarse_block_sum(N_det_bks_obs, kz, axis=1)
    C_true_bKs = safe_ratio(Nd_bKs_true, tc_bKs, fill=0.0)
    C_true_bKs_obsz = safe_ratio(Nd_bKs_obs, tc_bKs, fill=0.0)
    C_true_bs = safe_ratio(Nd_bKs_true.sum(axis=1), tc_bKs.sum(axis=1))
    C_true_bk = safe_ratio(N_det_bks_true.sum(axis=2), tc_bks.sum(axis=2))

    N_cKsb = coarse_block_sum(N_match_cksb, kz, axis=1)
    M_counts_sKcb = np.transpose(N_cKsb, (2, 1, 0, 3))
    tot = M_counts_sKcb.sum(axis=2)
    had_mass = tot > 0
    M_true_sKcb = safe_ratio(M_counts_sKcb,
                             np.broadcast_to(tot[:, :, None, :],
                                             M_counts_sKcb.shape))
    Mg_norm_sKcb = bmo._model_kernel_rows(pk, nhat, kz)
    M_true_sKcb = np.where(had_mass[:, :, None, :], M_true_sKcb, Mg_norm_sKcb)

    N_cKsb_truez = coarse_block_sum(N_match_true_z, kz, axis=1)
    E_true_cKsb = safe_ratio(
        N_cKsb_truez, np.broadcast_to(
            np.transpose(tc_bKs, (1, 2, 0))[None, :, :, :], N_cKsb_truez.shape))
    E_true_cksb_fine = safe_ratio(
        N_match_true_z, np.broadcast_to(
            np.transpose(tc_bks, (1, 2, 0))[None, :, :, :],
            N_match_true_z.shape))
    return dict(
        C_true_bKs=C_true_bKs, C_true_bs=C_true_bs, M_true_sKcb=M_true_sKcb,
        E_true_cKsb=E_true_cKsb, N_match_cksb=N_match_cksb, P6b_cks=P6b_cks,
        M_counts_sKcb=M_counts_sKcb, M_had_mass_sKb=had_mass,
        E_true_cksb_fine=E_true_cksb_fine,
        N_match_true_z_cksb=N_match_true_z,
        N_det_bks_true_z=N_det_bks_true, N_det_bks_obs_z=N_det_bks_obs,
        N_det_all_bks_true_z=N_det_all_bks,
        C_true_bKs_obsz=C_true_bKs_obsz, C_true_bk=C_true_bk,
        hostless_cks=hostless_cks, host_above_top_cks=host_above_cks,
        N_match_host_19p0_19p5_cks=N_match_lo_cks,
        N_match_host_ge19p5_cks=N_match_hi_cks,
        counts_obs_cks=cnt_obs, truth_counts_bks=tc_bks)


# ---------------------------------------------------------------------------
def build(family, out_dir, work_dir=None):
    t_start = time.time()
    bmo = A0._bmo()
    ep = bmo._load_ep()
    out_dir = os.path.abspath(out_dir)
    work_dir = work_dir or os.path.join(out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)
    rec: dict = dict(family=family)

    a0_pack = os.path.join(out_dir, f"scanpack_{family}_b300_A0.npz")
    if not os.path.exists(a0_pack):
        raise SystemExit(f"A0 pack missing: {a0_pack} — run build_a0_support.py first")
    pk = np.load(a0_pack, allow_pickle=False)
    ntrue = np.asarray(pk["ntrue_edges"], float)
    nhat = np.asarray(pk["nhat_edges"], float)
    zf = np.asarray(pk["zf_edges"], float)
    snr_e = np.asarray(pk["snr_edges"], float)
    kz = np.asarray(pk["kz_to_K"], int)
    pack_counts = np.asarray(pk["counts"], np.int64)
    pack_tcb = np.asarray(pk["truth_counts_bks"], float)
    cen_rec_path = A0.census_of_record(family)
    ops_rec_path = OPS_OF_RECORD.format(fam=family)

    # =====================================================================
    # FIDELITY GATES — the ORIGINAL convention must be reproduced bit-exactly
    # =====================================================================
    f19_old = selection_pass(ep, family, work_dir, BASIS_FLOOR,
                             collar_kms=ADOPTED_COLLAR, z_cut_columns="minmax")
    f172_old = selection_pass(ep, family, work_dir, CENSUS_FLOOR,
                              collar_kms=ADOPTED_COLLAR, z_cut_columns="minmax")
    snr_min = float(f19_old["cfg"].snr_min)

    cen_old = census_blocks(ep, f172_old["det"])
    cz = np.load(cen_rec_path, allow_pickle=True)
    g_cen = {k: bool(np.array_equal(cen_old[k].astype(float),
                                    np.asarray(cz[k], float)))
             for k in CENSUS_BLOCKS}
    tcb_old, _, _ = bmo.truth_hist_bks(f19_old["truth"], ntrue, zf, snr_e,
                                       snr_min)
    ad = np.load(A0.adopted_pack(family), allow_pickle=False)
    g_truth = bool(np.array_equal(tcb_old,
                                  np.asarray(ad["truth_counts_bks"], float)))
    ops_old = build_ops(bmo, pk, f19_old["det"], f172_old["det"],
                        f19_old["truth"], ntrue, nhat, zf, snr_e, kz, snr_min)
    oz = np.load(ops_rec_path, allow_pickle=True)
    g_ops = {k: bool(np.array_equal(ops_old[k], np.asarray(oz[k], float)))
             for k in OPS_CONTRACT_KEYS}
    rec["fidelity_gates_original_convention"] = dict(
        note="this module's re-implementation of load_and_cut_catalog step 6, "
             "run with the COMMITTED convention (minmax z columns, collar "
             "3000), must reproduce the objects of record BIT-EXACTLY",
        census_blocks=g_cen, census_of_record=cen_rec_path,
        truth_counts_bks_vs_adopted_pack=g_truth,
        ops_contract_arrays=g_ops, ops_of_record=ops_rec_path,
        n_loaded=f172_old["n_loaded"],
        n_sentinel_dropped=f172_old["n_sentinel_dropped"],
        n_cat_cut_17p2=f172_old["n_cat_cut"], n_op_17p2=f172_old["n_op"],
        n_cat_cut_19p0=f19_old["n_cat_cut"], n_op_19p0=f19_old["n_op"])
    bad = ([k for k, v in g_cen.items() if not v]
           + [k for k, v in g_ops.items() if not v]
           + ([] if g_truth else ["truth_counts_bks"]))
    if bad:
        print(json.dumps(rec["fidelity_gates_original_convention"], indent=1),
              file=sys.stderr)
        raise SystemExit(f"FIDELITY GATE FAILED on {bad} — nothing written")
    print(f"[{family}] FIDELITY GATES PASSED (7 census blocks + truth + 6 ops "
          f"contract arrays, all bit-exact)", flush=True)

    # =====================================================================
    # A0v2 — Z_DLA-only window at collar 3300
    # =====================================================================
    f19_new = selection_pass(ep, family, work_dir, BASIS_FLOOR,
                             collar_kms=A0_COLLAR, z_cut_columns="zdla_only")
    f172_new = selection_pass(ep, family, work_dir, CENSUS_FLOOR,
                              collar_kms=A0_COLLAR, z_cut_columns="zdla_only")
    cen_new = census_blocks(ep, f172_new["det"])
    ops_new = build_ops(bmo, pk, f19_new["det"], f172_new["det"],
                        f19_new["truth"], ntrue, nhat, zf, snr_e, kz, snr_min)

    # under Z_DLA-only the truth floor can no longer perturb the detection rows
    # (that coupling ran through Z_TRUE in the window) — a free consistency check
    rec["floor_independence_under_zdla_only"] = dict(
        counts_all_floor_17p2=int(cen_new["counts_all"].sum()),
        counts_obs_floor_19p0=int(ops_new["counts_obs_cks"].sum()),
        IDENTICAL=bool(np.array_equal(cen_new["counts_all"].astype(np.int64),
                                      ops_new["counts_obs_cks"].astype(np.int64))))

    # ---- PRIMARY GATE: counts_all == the A0 pack's counts -----------------
    diff = cen_new["counts_all"].astype(np.int64) - pack_counts
    rec["counts_gate_vs_A0_pack"] = dict(
        pack=a0_pack, pack_counts_total=int(pack_counts.sum()),
        census_counts_all_total=int(cen_new["counts_all"].sum()),
        total_difference=int(diff.sum()),
        n_cells_differing=int(np.count_nonzero(diff)),
        max_abs_cell_difference=int(np.abs(diff).max()),
        EXACT=bool(np.array_equal(cen_new["counts_all"].astype(np.int64),
                                  pack_counts)),
        residual_explanation=(
            "the only remaining selection difference is load_and_cut_catalog "
            "step 3, the sentinel filter (NHI_ERR == -1 OR Z_DLA_ERR == -1), "
            "which build_scan_packs.py does not apply; "
            f"{f172_new['n_sentinel_dropped']} sentinel rows were dropped "
            "before matching here"),
        n_sentinel_dropped_before_matching=f172_new["n_sentinel_dropped"],
        difference_per_coarse_z_K=[int(diff[:, K * 5:(K + 1) * 5, :].sum())
                                   for K in range(3)],
        difference_per_snr_s=[int(diff[:, :, s].sum())
                              for s in range(diff.shape[2])])
    print(f"[{family}] counts gate vs A0 pack: "
          f"{int(cen_new['counts_all'].sum())} vs {int(pack_counts.sum())} "
          f"(EXACT={rec['counts_gate_vs_A0_pack']['EXACT']}, "
          f"{rec['counts_gate_vs_A0_pack']['n_cells_differing']} cells)",
          flush=True)

    # ---- deltas vs the truth-aware versions -------------------------------
    rec["deltas_vs_truth_aware"] = dict(
        rows=dict(
            n_op_17p2_minmax_c3000=f172_old["n_op"],
            n_op_17p2_zdla_only_c3300=f172_new["n_op"],
            n_cat_cut_17p2_minmax_c3000=f172_old["n_cat_cut"],
            n_cat_cut_17p2_zdla_only_c3300=f172_new["n_cat_cut"]),
        census={k: dict(truth_aware_c3000=int(cen_old[k].sum()),
                        zdla_only_c3300=int(cen_new[k].sum()),
                        delta=int(cen_new[k].sum() - cen_old[k].sum()))
                for k in CENSUS_BLOCKS},
        truth_counts_bks_used_by_E_and_C=dict(
            truth_aware_pass_c3000=float(ops_old["truth_counts_bks"].sum()),
            zdla_only_pass_c3300=float(ops_new["truth_counts_bks"].sum()),
            delta=float(ops_new["truth_counts_bks"].sum()
                        - ops_old["truth_counts_bks"].sum()),
            equals_A0_pack_truth_counts_bks=bool(np.array_equal(
                ops_new["truth_counts_bks"], pack_tcb))),
        N_match_cksb=dict(
            truth_aware_c3000=float(ops_old["N_match_cksb"].sum()),
            zdla_only_c3300=float(ops_new["N_match_cksb"].sum()),
            delta=float(ops_new["N_match_cksb"].sum()
                        - ops_old["N_match_cksb"].sum())),
        C_true_bs_mean_shift=float(np.nanmean(
            np.where(ops_old["C_true_bs"] > 0,
                     ops_new["C_true_bs"] / np.where(ops_old["C_true_bs"] > 0,
                                                     ops_old["C_true_bs"], np.nan),
                     np.nan))))

    # =====================================================================
    # supports + write
    # =====================================================================
    m = ep.MOCKS[family]
    cfg = f19_new["cfg"]

    def _sup(floor):
        return A0.family_support(cfg, m["catalog_dir"], str(cfg.truth_path),
                                 m["bal_cat_path"], collar_kms=A0_COLLAR,
                                 truth_host_floor=floor,
                                 z_cut_columns=Z_CUT_ZDLA_ONLY)

    sup_data = _sup(NO_TRUTH_SIDE)
    sup_truth = _sup(BASIS_FLOOR)
    sup_census = _sup(CENSUS_FLOOR)
    rec["support_ids"] = {
        "A0v2 census (all blocks) @3300 Z_DLA-only floor 17.2": sup_census.sha256,
        "A0v2 ops (basis operators) @3300 Z_DLA-only floor 19.0": sup_truth.sha256,
        "A0 pack data plane @3300 Z_DLA-only": sup_data.sha256,
        "row_support_shared_by_all_three": sup_census.row_sha256,
    }
    rec["row_support_equal_across_pack_truth_census_ops"] = bool(
        sup_data.row_sha256 == sup_truth.row_sha256 == sup_census.row_sha256)

    prov = dict(
        role="A0v2: FP-truth census + empirical operators rebuilt on the "
             "pack's OWN observable-only selection (Z_DLA-only lambda/z window) "
             "at collar 3300 — closes support-mismatch instance #8 on the "
             "validation side; VALIDATION-ONLY, no sampler",
        family=family,
        recipe="validation/absorber_ladder/support/rebuild_a0_products.py",
        z_cut_columns=Z_CUT_ZDLA_ONLY, collar_kms=A0_COLLAR,
        reimplemented="ONLY load_and_cut_catalog step 6 (lambda_rf + z_qso + "
                      "BAL cut) is re-implemented, parameterised on collar and "
                      "z column(s); steps 1-5 and 7-8, _snap_off_molly_edges "
                      "and bin_counts_cks are COMMITTED calls",
        frozen_code_untouched=["CDDF_analysis/hbi/cddf_catalog_hbi.py",
                               "validation/fp_ladder/build_fp_census.py",
                               "validation/absorber_diag/build_matched_ops.py"],
        code=A0._git_head(),
        env=dict(python=platform.python_version(), numpy=np.__version__,
                 conda_prefix=os.environ.get("CONDA_PREFIX"),
                 host=platform.node()),
        catalog_dir=str(m["catalog_dir"]), truth_path=str(cfg.truth_path),
        bal_cat_path=str(m["bal_cat_path"]), molly_tsv=str(cfg.molly_tsv),
        snr_min=snr_min, p_dla_min=float(cfg.p_dla_min),
        lam_rf_min=float(cfg.lam_rf_min), lam_rf_max=float(cfg.lam_rf_max),
        z_qso_window=[float(cfg.z_qso_min), float(cfg.z_qso_max)],
        a0_pack=a0_pack, a0_pack_sha256=file_sha256(a0_pack),
        census_of_record=cen_rec_path,
        census_of_record_sha256=file_sha256(cen_rec_path),
        ops_of_record=ops_rec_path,
        ops_of_record_sha256=file_sha256(ops_rec_path),
        built_utc=datetime.datetime.utcnow().isoformat() + "Z",
        findings=rec)

    written = []
    # ---- the A0v2 census --------------------------------------------------
    cen_path = os.path.join(out_dir, f"fp_census_{family}_A0v2.npz")
    np.savez(cen_path,
             **{k: cen_new[k] for k in CENSUS_BLOCKS},
             **{k + "_truth_aware_collar3000": cen_old[k] for k in CENSUS_BLOCKS},
             nhat_edges=ep.NHAT_EDGES, zf_edges=ep.ZF_EDGES,
             snr_edges=ep.SNR_EDGES, zc_edges=ep.ZC_EDGES, kz_to_K=ep.KZ_TO_K,
             host_slot_names=np.array(HOST_SLOT_NAMES),
             support_id=stamp_array(sup_census),
             provenance=np.array(json.dumps(prov, default=str)))
    with open(cen_path.replace(".npz", ".provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1, default=str)
    with open(cen_path.replace(".npz", ".json"), "w") as fh:
        json.dump(dict(family=family, collar_kms=A0_COLLAR,
                       z_cut_columns=Z_CUT_ZDLA_ONLY,
                       totals={k: int(cen_new[k].sum()) for k in CENSUS_BLOCKS},
                       totals_truth_aware_collar3000={
                           k: int(cen_old[k].sum()) for k in CENSUS_BLOCKS},
                       counts_gate_vs_A0_pack=rec["counts_gate_vs_A0_pack"],
                       hostless_per_coarse_z_K=[
                           int(cen_new["hostless"][:, K * 5:(K + 1) * 5, :].sum())
                           for K in range(3)],
                       hostless_per_snr_s=[int(cen_new["hostless"][:, :, s].sum())
                                           for s in range(8)],
                       support_id=sup_census.sha256, provenance=prov),
                  fh, indent=1, default=str)
    stamp(cen_path, sup_census, extra=dict(
        note="A0v2 census: every block on the pack's own observable-only "
             "Z_DLA-only selection at collar 3300 (instance #8 closed)",
        planes={k: dict(sup_census.fields) for k in CENSUS_BLOCKS}))
    written.append(cen_path)

    # ---- the A0 operators -------------------------------------------------
    ops_path = os.path.join(out_dir, f"empirical_ops_{family}_A0.npz")
    np.savez_compressed(
        ops_path, **ops_new,
        truth_counts_bks_truth_aware_collar3000=ops_old["truth_counts_bks"],
        ntrue_edges=ntrue, nhat_edges=nhat, zf_edges=zf, snr_edges=snr_e,
        zc_edges=np.asarray(pk["zc_edges"], float), kz_to_K=kz,
        support_id=stamp_array(sup_truth),
        provenance=np.array(json.dumps(prov, default=str), dtype=object))
    with open(ops_path.replace(".npz", ".provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1, default=str)
    stamp(ops_path, sup_truth, extra=dict(
        note="A0 empirical operators on the pack's own observable-only "
             "Z_DLA-only selection at collar 3300; same key contract as "
             "validation/absorber_diag/empirical_ops_<fam>.npz",
        contract_keys=list(OPS_CONTRACT_KEYS),
        planes={k: dict(sup_truth.fields) for k in OPS_CONTRACT_KEYS}))
    written.append(ops_path)

    rec["wall_s"] = round(time.time() - t_start, 1)
    print(f"[{family}] wrote {len(written)} A0v2 products "
          f"({rec['wall_s']:.0f}s)", flush=True)
    return rec, written


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", nargs="+", default=list(FAMILIES),
                    choices=list(FAMILIES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    a = ap.parse_args(argv)
    summary = {}
    for fam in a.family:
        summary[fam], _ = build(fam, a.out, a.work)
    with open(os.path.join(a.out, "A0V2_BUILD_SUMMARY.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)

    # the ladder gate, at the COMMANDER-ADJUDICATED 'row' level, on the A0v2 set
    gates = {}
    for fam in a.family:
        pkp = os.path.join(a.out, f"scanpack_{fam}_b300_A0.npz")
        cen = os.path.join(a.out, f"fp_census_{fam}_A0v2.npz")
        ops = os.path.join(a.out, f"empirical_ops_{fam}_A0.npz")
        for label, lvl in (("row", ROW_SELECTION_FIELDS),
                           ("full", SUPPORT_FIELDS)):
            key = f"{fam}:pack+A0v2census+A0ops.{label}"
            try:
                gates[key] = check_support_consistency(pkp, cen, ops, fields=lvl)
            except SupportContractError as exc:
                gates[key] = dict(status="FAIL", error=str(exc))
            print(f"\n=== {key} -> {gates[key]['status']} ===")
            if gates[key]["status"] == "PASS":
                print(" support_id", gates[key]["support_id_short"],
                      "planes", len(gates[key]["planes"]),
                      "\n truth_host_floor", json.dumps(
                          sorted(set(map(str, gates[key]["truth_host_floor"]
                                         .values())))))
            else:
                print(" ".join(gates[key]["error"].split("\n\n")[1:2]))
    with open(os.path.join(a.out, "A0V2_SUPPORT_GATES.json"), "w") as fh:
        json.dump(gates, fh, indent=1, default=str)
    print("\nSHA256SUMS ->", A0.write_sha256sums(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
