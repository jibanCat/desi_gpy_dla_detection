#!/usr/bin/env python
"""build_matched_ops.py — EMPIRICAL absorber-side operators from the mocks' own
matched truth (VALIDATION-ONLY; no sampler; nothing under ``CDDF_analysis/`` is
modified, read or written).

For each mock family (2LPT-0 / London-0 / Saclay-0) this rebuilds the matched
per-detection table through the SAME committed machinery the census builder uses
(``extract_pack.load_mock_bundle`` / ``cddf_catalog_hbi.load_and_cut_catalog``)
and reduces it to the operators the frozen fold factorises into:

    mu_TP[c,k,s] = dX[k,s] * sum_b Mg[s,k,c,b] * C[s,b] * g[b,k] * f[b,k] * dN_b

  * ``N_match_cksb``  (C, Kf, S, B) matched in-basis detections: latent true-N
    bin b (host NHI_TRUE on the pack's ``ntrue_edges``) -> observed cell (c,k,s)
    with (c,k,s) on the OBSERVED axes, exactly as ``counts`` is binned;
  * ``N_match_true_z`` (C, Kf, S, B) the same, with k taken from the host's
    TRUE z (``Z_TRUE``) — the fold has no z-migration term, so the difference
    between the two IS the z-migration the fold cannot represent;
  * ``C_true_bKs`` / ``C_true_bs`` completeness = detections / truth_counts_bks;
  * ``M_true_sKcb`` migration conditional on detection (normalised over c);
  * ``E_true_cKsb`` the full empirical TP transfer at coarse-K resolution;
  * ``P6b_cks`` host below the basis floor ([17.2, 19.0)) — the class the fold
    has NO term for — and ``hostless_cks`` (the census's P4+P6c).

Two matching passes are run per family because ``is_TP`` is a property of the
(catalogue, truth-floor) bundle, not of a detection:
  * floor **19.0** — the floor the pack's own ``truth_counts`` (basis pad
    19.0) was built at.  This pass is the one every in-basis operator uses, and
    its truth histogram is GATED elementwise against the pack's
    ``truth_counts_bks``.
  * floor **17.2** — the census floor, reproducing
    ``validation/fp_ladder/build_fp_census.py`` exactly, for ``P6b`` /
    ``hostless``.  Gated elementwise against the census NPZ on disk.

COLLAR.  ``load_and_cut_catalog`` applies the 3000 km/s proximity collar
(``molly_faithful_pc_plots.make_lambda_z_BAL_cuts``).  The adopted packs are on
that collar; the collar-SCAN packs (``scanpack_*_b300.npz``) have ``counts``,
``dX`` and ``fp_E_alloc`` rebuilt at **3300 km/s** while ``truth_counts`` /
``truth_counts_bks`` are copied byte-identically from the 3000 km/s source
(``build_scan_packs.py`` docstring: "EVERY OTHER ARRAY is copied
byte-identically").  This builder therefore works on the ADOPTED pack (collar
matched to the matched-pair tables) and, separately, rebuilds the truth
histogram AT 3300 km/s so the scan packs' collar mismatch can be quantified
(``truth_counts_bks_collar3300`` and ``collar`` in the provenance).

ENV: ``gpdla`` (jax-free).  ``extract_pack.py`` is loaded FILE-DIRECTLY for the
same reason as the census builder (the package ``__init__`` imports jax).

Usage
-----
    python validation/absorber_diag/build_matched_ops.py --family 2lpt0 \
        --out validation/absorber_diag [--work DIR]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util as ilu
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from binning import bin_index, coarse_block_sum, safe_ratio  # noqa: E402

_EXTRACT_PACK = os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc",
                             "extract_pack.py")
FAMILIES = ("2lpt0", "london0", "saclay0")

ADOPTED_DIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "adopted_packs_v2p2_20260821")
SCANPACK_DIR = ("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/"
                "packs")
CENSUS_DIR = ("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/"
              "census")

LYA = 1215.67
C_KMS = 299792.458
BASIS_FLOOR = 19.0          # ntrue_edges[0]; the latent basis floor
CENSUS_FLOOR = 17.2
SLOT_EPS = 1e-9             # the census's half-open tolerance, same sign


def adopted_pack(fam):
    return os.path.join(ADOPTED_DIR,
                        f"modelA_pack_{fam}_bw0p2_pad19p0_molly172_v2.npz")


def scan_pack(fam):
    return os.path.join(SCANPACK_DIR, f"scanpack_{fam}_b300.npz")


def census_npz(fam):
    return os.path.join(CENSUS_DIR, f"fp_census_{fam}.npz")


# ---------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo=_REPO):
    try:
        out = {}
        out["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            stderr=subprocess.DEVNULL).decode().strip()
        out["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo,
            stderr=subprocess.DEVNULL).decode().strip()
        out["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
            stderr=subprocess.DEVNULL).decode().strip())
        return out
    except Exception as exc:                                # pragma: no cover
        return dict(commit="unknown", error=str(exc))


def _load_ep():
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    spec = ilu.spec_from_file_location("_absdiag_ep", _EXTRACT_PACK)
    mod = ilu.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _assert_index_convention(ep, nhat, zhat, snr):
    """The local pure helper MUST agree with the committed ``extract_pack._idx``."""
    for edges, x, label in ((ep.NHAT_EDGES, nhat, "NHAT"),
                            (ep.ZF_EDGES, zhat, "ZF"),
                            (ep.SNR_EDGES, snr, "SNR")):
        a = bin_index(edges, x)
        b = ep._idx(np.asarray(edges, float), np.asarray(x, float))
        if not np.array_equal(a, b):
            raise AssertionError(f"INDEX CONVENTION DRIFT on {label}")


# ---------------------------------------------------------------------------
def matching_pass(ep, family, work_dir, floor):
    """Cut + match the catalogue against a truth table floored at ``floor``.

    Mirrors ``build_fp_census.pass_17p2`` verbatim except for the floor; the
    op mask, the molly edge tie-break and the column set are identical.
    """
    from CDDF_analysis.hbi.cddf_catalog_hbi import (
        load_and_cut_catalog, load_molly_matrix, _build_qso_lookup)
    from CDDF_analysis.hbi import track_c_tf_saclay as TS

    t0 = time.time()
    cfg = ep._make_cfg(family, work_dir)
    mm = load_molly_matrix(cfg.molly_tsv)
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, _is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=float(floor), qso_lookup=qso_lookup,
        host_truth_floor=float(floor))
    TS._snap_off_molly_edges(cat_cut, truth_cut, mm)
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    det = dict(
        nhat=np.asarray(cat_cut["NHI"], float)[op],
        zobs=np.asarray(cat_cut["Z_DLA"], float)[op],
        snr=np.asarray(cat_cut["S2N_RED"], float)[op],
        nhi_true=np.asarray(cat_cut["NHI_TRUE"], float)[op],
        z_true=np.asarray(cat_cut["Z_TRUE"], float)[op],
        zqso=np.asarray(cat_cut["Z_QSO"], float)[op])
    _assert_index_convention(ep, det["nhat"], det["zobs"], det["snr"])
    tru = dict(
        nhi=np.asarray(truth_cut["NHI"], float),
        z=np.asarray(truth_cut["Z_DLA"], float),
        snr=np.asarray(truth_cut["S2N_RED"], float),
        zqso=np.asarray(truth_cut["Z_QSO"], float))
    return dict(det=det, truth=tru, cfg=cfg, meta=meta,
                n_cat_cut=int(len(cat_cut)), n_op=int(op.sum()),
                n_truth_cut=int(len(truth_cut)), floor=float(floor),
                seconds=time.time() - t0)


def truth_hist_bks(tru, ntrue_edges, zf_edges, snr_edges, snr_min, keep=None):
    """``build_truth_counts`` semantics: SNR>snr_min strict, half-open bins."""
    n_b = len(ntrue_edges) - 1
    n_k = len(zf_edges) - 1
    n_s = len(snr_edges) - 1
    m = np.asarray(tru["snr"], float) > float(snr_min)
    if keep is not None:
        m = m & np.asarray(keep, bool)
    b = bin_index(ntrue_edges, tru["nhi"][m])
    k = bin_index(zf_edges, tru["z"][m])
    s = np.clip(bin_index(snr_edges, tru["snr"][m]), 0, n_s - 1)
    ok = (b >= 0) & (b < n_b) & (k >= 0) & (k < n_k)
    out = np.zeros((n_b, n_k, n_s), float)
    np.add.at(out, (b[ok], k[ok], s[ok]), 1.0)
    return out, int(ok.sum()), int(m.sum())


def collar_keep(z, zqso, collar_kms, lam_rf_min, lam_rf_max):
    """The ``make_lambda_z_BAL_cuts`` window geometry at an arbitrary collar."""
    coll = float(collar_kms) / C_KMS
    z_lo = np.maximum(3600.0 / LYA - 1.0,
                      lam_rf_min * (1 + zqso) / LYA - 1.0 + coll)
    z_hi = np.minimum(zqso - coll,
                      lam_rf_max * (1 + zqso) / LYA - 1.0 - coll)
    return (np.asarray(z, float) < z_hi) & (np.asarray(z, float) > z_lo)


def bin4(nhat, z, snr, nhi_true, nhat_edges, zf_edges, snr_edges, ntrue_edges):
    """(c, k, s, b) histogram of matched detections; b from the host's true N."""
    C = len(nhat_edges) - 1
    K = len(zf_edges) - 1
    S = len(snr_edges) - 1
    B = len(ntrue_edges) - 1
    c = bin_index(nhat_edges, nhat)
    k = bin_index(zf_edges, z)
    s = np.clip(bin_index(snr_edges, snr), 0, S - 1)
    b = bin_index(ntrue_edges, nhi_true)
    ok = ((c >= 0) & (c < C) & (k >= 0) & (k < K)
          & (b >= 0) & (b < B) & np.isfinite(nhi_true))
    out = np.zeros((C, K, S, B), float)
    np.add.at(out, (c[ok], k[ok], s[ok], b[ok]), 1.0)
    return out, ok


def bin3(nhat, z, snr, nhat_edges, zf_edges, snr_edges):
    C = len(nhat_edges) - 1
    K = len(zf_edges) - 1
    S = len(snr_edges) - 1
    c = bin_index(nhat_edges, nhat)
    k = bin_index(zf_edges, z)
    s = np.clip(bin_index(snr_edges, snr), 0, S - 1)
    ok = (c >= 0) & (c < C) & (k >= 0) & (k < K)
    out = np.zeros((C, K, S), float)
    np.add.at(out, (c[ok], k[ok], s[ok]), 1.0)
    return out, int(ok.sum())


def _model_kernel_rows(pk, nhat_edges, kz):
    """The ADOPTED kernel's unit-mass rows on the (s, K, c, b) grid.

    ``count_conserving_fold`` is pure numpy/scipy but lives in the jax-importing
    package, so it is loaded FILE-DIRECTLY like ``extract_pack``.  Returns
    masses normalised to unit mass over the observed grid — the fallback used
    for (s, K, b) rows with no matched detection at all.
    """
    spec = ilu.spec_from_file_location(
        "_absdiag_ccf", os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc",
                                     "count_conserving_fold.py"))
    ccf = ilu.module_from_spec(spec)
    sys.modules[spec.name] = ccf
    spec.loader.exec_module(ccf)

    class _P:
        pass
    q = _P()
    for k in ("ntrue_edges", "resp_N_ref", "resp_skew_ramp", "resp_sig_floor"):
        setattr(q, k, pk[k])
    masses, _ = ccf.surface_masses(
        q, np.asarray(pk["adopted_resp_mu_coef"], float),
        np.asarray(pk["adopted_resp_sig_coef"], float),
        np.asarray(pk["adopted_resp_skew_coef"], float),
        np.asarray(pk["adopted_resp_fit_range"], float),
        np.asarray(nhat_edges, float))                       # (SR, ZR, C, B)
    snr_e = np.asarray(pk["snr_edges"], float)
    rse = np.asarray(pk["resp_snr_edges"], float)
    s2sr = np.clip(np.digitize(snr_e[:-1] + 1e-9, rse) - 1, 0,
                   masses.shape[0] - 1)
    zc = np.asarray(pk["zc_edges"], float)
    rze = np.asarray(pk["resp_z_edges"], float)
    K2zr = np.digitize(0.5 * (zc[:-1] + zc[1:]), rze) - 1
    g = masses[s2sr[:, None], K2zr[None, :], :, :]           # (S, KK, C, B)
    return g / np.maximum(g.sum(axis=2, keepdims=True), 1e-300)


# ---------------------------------------------------------------------------
def build(family, out_dir, work_dir=None, also_copy_to=None):
    t_start = time.time()
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    work_dir = work_dir or os.path.join(
        "/scratch/cavestru_root/cavestru0/mfho/absorber_diag_2026-09-13",
        "_work")
    os.makedirs(work_dir, exist_ok=True)

    ep = _load_ep()
    if not np.array_equal(np.asarray(ep.KZ_TO_K), np.repeat([0, 1, 2], 5)):
        raise AssertionError("KZ_TO_K changed — the coarse-K slicing is stale")

    pk_path = adopted_pack(family)
    pk = np.load(pk_path, allow_pickle=True)
    ntrue = np.asarray(pk["ntrue_edges"], float)
    nhat = np.asarray(pk["nhat_edges"], float)
    zf = np.asarray(pk["zf_edges"], float)
    snr_e = np.asarray(pk["snr_edges"], float)
    kz = np.asarray(pk["kz_to_K"], int)
    B, C, Kf, S = len(ntrue) - 1, len(nhat) - 1, len(zf) - 1, len(snr_e) - 1
    KK = int(kz.max()) + 1
    tc_bks_pack = np.asarray(pk["truth_counts_bks"], float)
    counts_pack = np.asarray(pk["counts"], np.int64)

    # ---- pass at the BASIS floor 19.0 (the pack's truth_counts floor) ------
    p19 = matching_pass(ep, family, work_dir, BASIS_FLOOR)
    snr_min = float(p19["cfg"].snr_min)
    lam_lo = float(p19["cfg"].lam_rf_min)
    lam_hi = float(p19["cfg"].lam_rf_max)
    print(f"[{family}] floor 19.0: n_cat_cut={p19['n_cat_cut']} "
          f"n_op={p19['n_op']} n_truth_cut={p19['n_truth_cut']} "
          f"({p19['seconds']:.0f}s)", flush=True)

    tc_bks, n_truth_in, n_truth_snr = truth_hist_bks(
        p19["truth"], ntrue, zf, snr_e, snr_min)
    truth_gate = dict(
        pack=pk_path, pack_sha256=_sha256(pk_path),
        pack_total=float(tc_bks_pack.sum()), rebuilt_total=float(tc_bks.sum()),
        max_abs_cell_difference=float(np.abs(tc_bks - tc_bks_pack).max()),
        EQUAL=bool(np.array_equal(tc_bks, tc_bks_pack)))
    if not truth_gate["EQUAL"]:
        print(json.dumps(truth_gate, indent=1), file=sys.stderr)
        raise SystemExit("TRUTH GATE FAILED — nothing written")
    print(f"[{family}] TRUTH GATE PASSED ({tc_bks.sum():.0f} truth systems)",
          flush=True)

    det = p19["det"]
    cnt_obs, n_on_grid = bin3(det["nhat"], det["zobs"], det["snr"],
                              nhat, zf, snr_e)
    counts_gate = dict(
        pack_total=int(counts_pack.sum()), rebuilt_total=int(n_on_grid),
        max_abs_cell_difference=float(np.abs(cnt_obs - counts_pack).max()),
        EQUAL=bool(np.array_equal(cnt_obs.astype(np.int64), counts_pack)))

    # in-basis matched detections (host NHI_TRUE >= 19.0, inside the basis)
    N_match_cksb, ok_m = bin4(det["nhat"], det["zobs"], det["snr"],
                              det["nhi_true"], nhat, zf, snr_e, ntrue)
    N_match_true_z, _ = bin4(det["nhat"], det["z_true"], det["snr"],
                             det["nhi_true"], nhat, zf, snr_e, ntrue)
    # matched detections with NO observed-grid restriction: separates the
    # DETECTION probability from the kernel's in-grid COUNTING probability phi
    _bq = bin_index(ntrue, det["nhi_true"])
    _kq = bin_index(zf, det["z_true"])
    _sq = np.clip(bin_index(snr_e, det["snr"]), 0, S - 1)
    _okq = ((_bq >= 0) & (_bq < B) & (_kq >= 0) & (_kq < Kf)
            & np.isfinite(det["nhi_true"]))
    N_det_all_bks = np.zeros((B, Kf, S), float)
    np.add.at(N_det_all_bks, (_bq[_okq], _kq[_okq], _sq[_okq]), 1.0)
    # host classes the basis does NOT carry
    host = np.isfinite(det["nhi_true"])
    above_top = host & (det["nhi_true"] >= ntrue[-1] - SLOT_EPS)
    host_above_cks, _ = bin3(det["nhat"][above_top], det["zobs"][above_top],
                             det["snr"][above_top], nhat, zf, snr_e)
    # split of the in-basis matched set used by the report
    lo_host = host & (det["nhi_true"] >= BASIS_FLOOR - SLOT_EPS) \
        & (det["nhi_true"] < 19.5 - SLOT_EPS)
    hi_host = host & (det["nhi_true"] >= 19.5 - SLOT_EPS) \
        & (det["nhi_true"] < ntrue[-1] - SLOT_EPS)
    N_match_lo_cks, _ = bin3(det["nhat"][lo_host], det["zobs"][lo_host],
                             det["snr"][lo_host], nhat, zf, snr_e)
    N_match_hi_cks, _ = bin3(det["nhat"][hi_host], det["zobs"][hi_host],
                             det["snr"][hi_host], nhat, zf, snr_e)

    # ---- pass at the CENSUS floor 17.2 (P6b / hostless) -------------------
    p172 = matching_pass(ep, family, work_dir, CENSUS_FLOOR)
    d2 = p172["det"]
    n2 = d2["nhi_true"]
    h2 = np.isfinite(n2)
    m_p6b = h2 & (n2 >= CENSUS_FLOOR - SLOT_EPS) & (n2 < BASIS_FLOOR - SLOT_EPS)
    P6b_cks, _ = bin3(d2["nhat"][m_p6b], d2["zobs"][m_p6b], d2["snr"][m_p6b],
                      nhat, zf, snr_e)
    hostless_cks, _ = bin3(d2["nhat"][~h2], d2["zobs"][~h2], d2["snr"][~h2],
                           nhat, zf, snr_e)
    # the same two classes under the SCAN packs' 3300 km/s collar, so the
    # ORACLE FP pin (a collar-3000 census against collar-3300 counts) can be
    # costed
    k33_172 = collar_keep(d2["zobs"], d2["zqso"], 3300.0,
                          float(p172["cfg"].lam_rf_min),
                          float(p172["cfg"].lam_rf_max))
    hostless_3300, _ = bin3(d2["nhat"][(~h2) & k33_172],
                            d2["zobs"][(~h2) & k33_172],
                            d2["snr"][(~h2) & k33_172], nhat, zf, snr_e)
    P6b_3300, _ = bin3(d2["nhat"][m_p6b & k33_172], d2["zobs"][m_p6b & k33_172],
                       d2["snr"][m_p6b & k33_172], nhat, zf, snr_e)

    census_gate = None
    cpath = census_npz(family)
    if os.path.exists(cpath):
        cz = np.load(cpath, allow_pickle=True)
        census_gate = dict(
            census=cpath, census_sha256=_sha256(cpath),
            P6b_EQUAL=bool(np.array_equal(P6b_cks,
                                          np.asarray(cz["host_17p2_19p0"],
                                                     float))),
            hostless_EQUAL=bool(np.array_equal(hostless_cks,
                                               np.asarray(cz["hostless"],
                                                          float))),
            P6b_total=float(P6b_cks.sum()),
            census_P6b_total=float(np.asarray(cz["host_17p2_19p0"]).sum()),
            hostless_total=float(hostless_cks.sum()),
            census_hostless_total=float(np.asarray(cz["hostless"]).sum()))
        if not (census_gate["P6b_EQUAL"] and census_gate["hostless_EQUAL"]):
            print(json.dumps(census_gate, indent=1), file=sys.stderr)
            raise SystemExit("CENSUS GATE FAILED — nothing written")
        print(f"[{family}] CENSUS GATE PASSED "
              f"(P6b={P6b_cks.sum():.0f}, hostless={hostless_cks.sum():.0f})",
              flush=True)

    # ---- collar 3300 rebuild of the truth histogram (scan-pack mismatch) ---
    keep33 = collar_keep(p19["truth"]["z"], p19["truth"]["zqso"], 3300.0,
                         lam_lo, lam_hi)
    tc_bks_3300, _, _ = truth_hist_bks(p19["truth"], ntrue, zf, snr_e, snr_min,
                                       keep=keep33)
    det_keep33 = collar_keep(det["zobs"], det["zqso"], 3300.0, lam_lo, lam_hi) \
        & collar_keep(np.where(np.isfinite(det["z_true"]), det["z_true"],
                               det["zobs"]), det["zqso"], 3300.0, lam_lo,
                      lam_hi)
    cnt_3300, _ = bin3(det["nhat"][det_keep33], det["zobs"][det_keep33],
                       det["snr"][det_keep33], nhat, zf, snr_e)
    sp = np.load(scan_pack(family), allow_pickle=True)
    collar_rec = dict(
        adopted_collar_kms=3000.0, scanpack_collar_kms=3300.0,
        truth_total_3000=float(tc_bks.sum()),
        truth_total_3300=float(tc_bks_3300.sum()),
        truth_ratio_3300_over_3000=float(tc_bks_3300.sum() / tc_bks.sum()),
        counts_total_adopted_pack=int(counts_pack.sum()),
        counts_total_scanpack=int(np.asarray(sp["counts"]).sum()),
        counts_total_rebuilt_3300=int(cnt_3300.sum()),
        dX_total_adopted=float(np.asarray(pk["dX"]).sum()),
        dX_total_scanpack=float(np.asarray(sp["dX"]).sum()),
        hostless_total_3000=float(hostless_cks.sum()),
        hostless_total_3300=float(hostless_3300.sum()),
        P6b_total_3000=float(P6b_cks.sum()),
        P6b_total_3300=float(P6b_3300.sum()),
        scanpack_truth_is_byte_identical_to_adopted=bool(
            np.array_equal(np.asarray(sp["truth_counts_bks"], float),
                           tc_bks_pack)),
        note=("build_scan_packs.py rebuilds counts/dX/fp_E_alloc at collar "
              "3000+b and copies EVERY other array byte-identically, so the "
              "scan packs carry a 3000 km/s truth histogram against a 3300 "
              "km/s data/exposure pair."))

    # ---- the operators ----------------------------------------------------
    # detections per (b, k, s): TRUE-z binning is the completeness-correct one
    # (truth_counts_bks is binned on the truth's own z), OBSERVED-z is what the
    # fold's own k axis sees.  Both are stored.
    N_det_bks_true = N_match_true_z.sum(axis=0).transpose(2, 0, 1)   # (B,Kf,S)
    N_det_bks_obs = N_match_cksb.sum(axis=0).transpose(2, 0, 1)
    tc_bKs = coarse_block_sum(tc_bks, kz, axis=1)                    # (B,KK,S)
    Nd_bKs_true = coarse_block_sum(N_det_bks_true, kz, axis=1)
    Nd_bKs_obs = coarse_block_sum(N_det_bks_obs, kz, axis=1)
    C_true_bKs = safe_ratio(Nd_bKs_true, tc_bKs, fill=0.0)
    C_true_bKs_obsz = safe_ratio(Nd_bKs_obs, tc_bKs, fill=0.0)
    C_true_bs = safe_ratio(Nd_bKs_true.sum(axis=1), tc_bKs.sum(axis=1))
    C_true_bk = safe_ratio(N_det_bks_true.sum(axis=2), tc_bks.sum(axis=2))

    # migration conditional on detection, per (s, K, b), normalised over c
    N_cKsb = coarse_block_sum(N_match_cksb, kz, axis=1)              # (C,KK,S,B)
    M_counts_sKcb = np.transpose(N_cKsb, (2, 1, 0, 3))               # (S,KK,C,B)
    tot = M_counts_sKcb.sum(axis=2)                                  # (S,KK,B)
    had_mass = tot > 0
    M_true_sKcb = safe_ratio(M_counts_sKcb,
                             np.broadcast_to(tot[:, :, None, :],
                                             M_counts_sKcb.shape))
    # rows with NO matched detection fall back to the model's own adopted
    # kernel row, renormalised to unit mass on the observed grid (the
    # count-conserving convention); `M_had_mass_sKb` records which rows are
    # MEASURED so a fallback row is never mistaken for a measurement.
    Mg_norm_sKcb = _model_kernel_rows(pk, nhat, kz)
    M_true_sKcb = np.where(had_mass[:, :, None, :], M_true_sKcb, Mg_norm_sKcb)
    # the empirical transfer at coarse-K (truth on its own z axis)
    N_cKsb_truez = coarse_block_sum(N_match_true_z, kz, axis=1)
    E_true_cKsb = safe_ratio(
        N_cKsb_truez, np.broadcast_to(
            np.transpose(tc_bKs, (1, 2, 0))[None, :, :, :],
            N_cKsb_truez.shape))
    E_true_cksb_fine = safe_ratio(
        N_match_true_z, np.broadcast_to(
            np.transpose(tc_bks, (1, 2, 0))[None, :, :, :],
            N_match_true_z.shape))

    prov = dict(
        role=("EMPIRICAL absorber-side operators from the mock's own matched "
              "truth; VALIDATION-ONLY, no sampler, CDDF_analysis untouched"),
        family=family,
        built_utc=datetime.datetime.utcnow().isoformat() + "Z",
        recipe="validation/absorber_diag/build_matched_ops.py",
        git=_git_head(), python=platform.python_version(),
        numpy=np.__version__,
        catalog_dir=str(ep.MOCKS[family]["catalog_dir"]),
        truth_path=str(p19["cfg"].truth_path),
        bal_cat_path=str(ep.MOCKS[family]["bal_cat_path"]),
        molly_tsv=str(p19["cfg"].molly_tsv),
        snr_min=snr_min, p_dla_min=float(p19["cfg"].p_dla_min),
        lam_rf_min=lam_lo, lam_rf_max=lam_hi,
        pack=pk_path, pack_sha256=_sha256(pk_path),
        scanpack=scan_pack(family), scanpack_sha256=_sha256(scan_pack(family)),
        census=cpath if os.path.exists(cpath) else None,
        census_sha256=(_sha256(cpath) if os.path.exists(cpath) else None),
        matching_floor_in_basis=BASIS_FLOOR,
        matching_floor_census=CENSUS_FLOOR,
        n_cat_cut_19p0=p19["n_cat_cut"], n_op_19p0=p19["n_op"],
        n_cat_cut_17p2=p172["n_cat_cut"], n_op_17p2=p172["n_op"],
        n_truth_cut_19p0=p19["n_truth_cut"],
        n_truth_on_grid=int(n_truth_in), n_truth_snr_kept=int(n_truth_snr),
        n_matched_in_basis=int(N_match_cksb.sum()),
        n_matched_above_basis_top=int(host_above_cks.sum()),
        n_P6b=int(P6b_cks.sum()), n_hostless=int(hostless_cks.sum()),
        truth_gate=truth_gate, counts_gate=counts_gate,
        census_gate=census_gate, collar=collar_rec,
        wall_s=round(time.time() - t_start, 1))

    out = os.path.join(out_dir, f"empirical_ops_{family}.npz")
    np.savez_compressed(
        out,
        # --- the contract keys the ladder runner reads -------------------
        C_true_bKs=C_true_bKs, C_true_bs=C_true_bs,
        M_true_sKcb=M_true_sKcb, E_true_cKsb=E_true_cKsb,
        N_match_cksb=N_match_cksb, P6b_cks=P6b_cks,
        # --- everything else ---------------------------------------------
        M_counts_sKcb=M_counts_sKcb, M_had_mass_sKb=had_mass,
        E_true_cksb_fine=E_true_cksb_fine,
        N_match_true_z_cksb=N_match_true_z,
        N_det_bks_true_z=N_det_bks_true, N_det_bks_obs_z=N_det_bks_obs,
        N_det_all_bks_true_z=N_det_all_bks,
        C_true_bKs_obsz=C_true_bKs_obsz, C_true_bk=C_true_bk,
        hostless_cks=hostless_cks, host_above_top_cks=host_above_cks,
        hostless_cks_collar3300=hostless_3300, P6b_cks_collar3300=P6b_3300,
        N_match_host_19p0_19p5_cks=N_match_lo_cks,
        N_match_host_ge19p5_cks=N_match_hi_cks,
        counts_obs_cks=cnt_obs,
        truth_counts_bks=tc_bks,
        truth_counts_bks_collar3300=tc_bks_3300,
        counts_obs_cks_collar3300=cnt_3300,
        ntrue_edges=ntrue, nhat_edges=nhat, zf_edges=zf, snr_edges=snr_e,
        zc_edges=np.asarray(pk["zc_edges"], float), kz_to_K=kz,
        provenance=np.array(json.dumps(prov, indent=1), dtype=object))
    print(f"[{family}] wrote {out} ({os.path.getsize(out)/1e6:.1f} MB, "
          f"{prov['wall_s']:.0f}s)", flush=True)
    with open(out.replace(".npz", ".provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1)
    if also_copy_to:
        os.makedirs(also_copy_to, exist_ok=True)
        import shutil
        shutil.copy2(out, os.path.join(also_copy_to, os.path.basename(out)))
        shutil.copy2(out.replace(".npz", ".provenance.json"),
                     os.path.join(also_copy_to,
                                  os.path.basename(out).replace(
                                      ".npz", ".provenance.json")))
    return out, prov


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", nargs="+", default=list(FAMILIES))
    ap.add_argument("--out", default=_HERE)
    ap.add_argument("--work", default=None)
    ap.add_argument("--copy-to", default=("/scratch/cavestru_root/cavestru0/"
                                          "mfho/absorber_diag_2026-09-13/ops"))
    a = ap.parse_args(argv)
    for fam in a.family:
        build(fam, a.out, a.work, a.copy_to)


if __name__ == "__main__":
    main()
