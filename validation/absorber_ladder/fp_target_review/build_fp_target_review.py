#!/usr/bin/env python
"""build_fp_target_review.py — FP CALIBRATION TARGET DEFINITION REVIEW
(PI ruling 2026-09-13c §7).  VALIDATION-ONLY; calibration / mock information
only; NO sampler, NO real survey product, nothing committed, no tracked file
modified.

It builds three families of products in one place so the two estimands can be
compared cell by cell:

  A. ``loa0_fp_target.npz`` — the HCD-FREE TWIN side.  The 89-event
     ``fp_counts`` block re-derived from the raw loa-0 dlacat (GATED
     bit-exactly against the pack's own ``fp_counts``), plus the same selection
     with the N-hat floor removed (the 2,378-event product), plus every
     intermediate rung of the selection ladder, plus the collar/z_qso/BAL
     counterfactuals (R-015).

  B. ``fp_target_decomposition_<fam>.npz`` — the MOCK side.  The A0v2 census
     rebuilt (GATED bit-exactly against ``fp_census_<fam>_A0v2.npz``) with the
     ``hostless`` class SPLIT by the mechanism that produced it:
       hostless_taken        a host was inside the matching window but the
                             greedy one-to-one rule gave it to another
                             detection (blend / matching-definition artefact);
       hostless_unresolvable the only host(s) were dropped from the matcher's
                             pool before matching;
       hostless_no_absorber  no truth absorber of ANY N_HI inside the window
                             — the ONLY class the HCD-free twin can produce.
     The five host slots are carried through unchanged, so the eight classes
     PARTITION the pack's own ``counts``.

  C. ``fp_target_shapes.npz`` — the four (N-hat x S/N) shapes the memo compares
     (89-event Perks template; mock hostless census; mock
     ``hostless_no_absorber``; the 2,378-event twin product's implied
     template), with row/column KL and cell-level Poisson consistency.

Every product carries a ``support_id`` (``support_contract.py``) and the mock
products FAIL CLOSED unless their support equals the A0v2 census's.

ENV: ``gpdla-hbi`` (jax present; only used for the committed ``fp_ladder``
cross-check) or ``gpdla``.  Login node, ~4 min.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_SUPPORT = os.path.join(_REPO, "validation", "absorber_ladder", "support")
for _p in (_HERE, _SUPPORT, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import selection as SEL                                          # noqa: E402
from support_contract import (                                   # noqa: E402
    NO_TRUTH_SIDE, ROW_SELECTION_FIELDS, Z_CUT_ZDLA_ONLY, SupportContractError,
    file_sha256, read_stamp, stamp, stamp_array, support_id,
)
import rebuild_a0_products as R                                   # noqa: E402

FAMILIES = ("2lpt0", "london0", "saclay0")

LOA0_DIR = ("/nfs/turbo/lsa-cavestru/mfho/paper1_durable_inputs/"
            "loa0_fp_v1_20260615_outputs")
A0_SUPPORT_DIR = ("/scratch/cavestru_root/cavestru0/mfho/"
                  "absorber_ladder_2026-09-13/support")
COLLAR = 3300.0
Z_CUT = "zdla_only"
CENSUS_FLOOR = 17.2


# ---------------------------------------------------------------------------
def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z")


def _git():
    import subprocess
    try:
        return dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=_REPO).decode().strip(),
            branch=subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=_REPO).decode().strip(),
            dirty=bool(subprocess.check_output(
                ["git", "status", "--porcelain", "-uno"],
                cwd=_REPO).decode().strip()))
    except Exception as exc:                                   # pragma: no cover
        return dict(error=str(exc))


# ---------------------------------------------------------------------------
# A.  the loa-0 HCD-free twin
# ---------------------------------------------------------------------------
def load_loa0_catalogue(loa0_dir=LOA0_DIR):
    import fitsio
    from astropy.table import Table, vstack
    files = sorted(glob.glob(os.path.join(loa0_dir, "dlacat-*.fits")))
    if not files:
        raise SystemExit(f"no dlacat-*.fits in {loa0_dir}")
    cat = vstack([Table(fitsio.read(f, ext=1)) for f in files])
    return cat, files


def build_loa0(out_dir, pack_path, loa0_dir=LOA0_DIR):
    cat, files = load_loa0_catalogue(loa0_dir)
    snr = np.asarray(cat["SNR_REDSIDE"], float)
    pdla = np.asarray(cat["P_DLA"], float)
    nhi = np.asarray(cat["NHI"], float)
    zd = np.asarray(cat["Z_DLA"], float)
    zq = np.asarray(cat["Z_QSO"], float)
    flag = np.asarray(cat["DLAFLAG"], int)
    nerr = np.asarray(cat["NHI_ERR"], float)
    zerr = np.asarray(cat["Z_DLA_ERR"], float)
    lam = SEL.lam_rest(zd, zq)

    ladder = {}
    ladder["raw_detections"] = int(len(cat))
    m_op = SEL.loa0_op_mask(snr, pdla, zd, zq, lam_rf_min=None,
                            z_lo=None, z_hi=None)
    ladder["snr_gt2_and_pdla_gt0p99"] = int(m_op.sum())
    m_lya = SEL.loa0_op_mask(snr, pdla, zd, zq, z_lo=None, z_hi=None)
    ladder["plus_lam_rest_ge_1025"] = int(m_lya.sum())
    m_lya_red = m_lya & (lam <= 1216.0)
    ladder["plus_lam_rest_le_1216"] = int(m_lya_red.sum())
    m_z = SEL.loa0_op_mask(snr, pdla, zd, zq)
    ladder["plus_Z_DLA_in_2p0_3p5"] = int(m_z.sum())            # 2,378 -> 2,318
    m_89 = SEL.loa0_op_mask(snr, pdla, zd, zq, nhi=nhi, nhat_lo=19.5)
    ladder["plus_Nhat_ge_19p5_(fp_counts)"] = int(m_89.sum())
    ladder["of_which_Nhat_ge_22p4_offgrid"] = int((m_89 & (nhi >= 22.4)).sum())

    # counterfactual support fields the fp_counts block does NOT apply
    cf = {}
    for coll in (3000.0, 3300.0):
        z_lo, z_hi = SEL.collar_window(zq, collar_kms=coll)
        cf[f"collar_{int(coll)}_kms"] = int(
            (m_89 & (zd > z_lo) & (zd < z_hi)).sum())
    cf["z_qso_window_2p0_4p25"] = int(
        (m_89 & (zq > 2.0) & (zq < 4.25)).sum())
    cf["quality_DLAFLAG_eq_0"] = int((m_89 & (flag == 0)).sum())
    cf["sentinel_filter"] = int((m_89 & ~((nerr == -1) | (zerr == -1))).sum())
    z_lo33, z_hi33 = SEL.collar_window(zq, collar_kms=3300.0)
    cf["all_of_the_above_collar3300"] = int(
        (m_89 & (zd > z_lo33) & (zd < z_hi33) & (zq > 2.0) & (zq < 4.25)
         & (flag == 0) & ~((nerr == -1) | (zerr == -1))).sum())
    cf["bal_veto"] = "loa-0 is BAL-free by construction (no bal_cat); inert"

    # grids
    g89 = SEL.grid_cs(nhi[m_89], snr[m_89])
    # the 2,378-event product, on the SAME event definition but with the
    # observed-N floor removed: it needs its own N-hat grid (17.2 up)
    nhat_edges_full = np.round(np.arange(17.2, 22.4 + 1e-9, 0.1), 3)
    g2378 = SEL.grid_cs(nhi[m_lya], snr[m_lya], nhat_edges=nhat_edges_full)
    g2318 = SEL.grid_cs(nhi[m_z], snr[m_z], nhat_edges=nhat_edges_full)

    pack = np.load(pack_path, allow_pickle=True)
    fp_counts_pack = np.asarray(pack["fp_counts"], np.int64)
    gate_exact = bool(np.array_equal(g89, fp_counts_pack))
    if not gate_exact:
        raise SystemExit(
            "loa-0 GATE FAILED: re-derived (29,8) fp_counts != the pack's "
            f"fp_counts ({int(g89.sum())} vs {int(fp_counts_pack.sum())})")

    out = dict(
        fp_counts_89=g89,
        nhat_edges=SEL.NHAT_EDGES, snr_edges=SEL.SNR_EDGES,
        grid_2378_lyaonly=g2378, grid_2318_lyaonly_zwin=g2318,
        nhat_edges_full=nhat_edges_full,
        nhi_89=nhi[m_89], snr_89=snr[m_89], z_89=zd[m_89], zqso_89=zq[m_89],
        nhi_2378=nhi[m_lya], snr_2378=snr[m_lya], z_2378=zd[m_lya],
        selection_ladder=json.dumps(ladder),
        counterfactual_support=json.dumps(cf),
    )
    prov = dict(
        role=("loa-0 HCD-FREE TWIN false-positive estimand: the fp_counts "
              "block of record re-derived from the raw dlacat, its selection "
              "ladder, and the same selection without the N-hat floor "
              "(the 2,378-event product)"),
        recipe="validation/absorber_ladder/fp_target_review/build_fp_target_review.py",
        loa0_catalogue=[os.path.abspath(f) for f in files],
        loa0_catalogue_sha256=[file_sha256(f) for f in files],
        gate_fp_counts_equals_pack=gate_exact, pack=os.path.abspath(pack_path),
        selection_ladder=ladder, counterfactual_support=cf,
        code=_git(), built_utc=_now(),
        note=("EVERY loa-0 detection is a false positive by construction (the "
              "twin contains no HCDs); the estimand is a pure SELECTION "
              "statement. 'hostless' is not a meaningful sub-class there."),
    )
    out["provenance"] = json.dumps(prov)

    sup = support_id(
        collar_kms=0.0, snr_min=2.0, z_window=(0.0, 99.0), p_dla_min=0.99,
        lya_only_lam_min=1025.0, lam_rf_max=99999.0,
        z_cut_columns="Z_DLA(fine-grid window only)", quality_cut="none",
        truth_host_floor=NO_TRUTH_SIDE,
        bal_policy="loa-0 HCD-free twin: no BAL catalogue, veto inert",
        catalogue_id="loa0:" + ",".join(
            file_sha256(f)[:12] for f in files),
        truth_catalogue_sha256=NO_TRUTH_SIDE)
    out["support_id"] = stamp_array(sup)
    path = os.path.join(out_dir, "loa0_fp_target.npz")
    np.savez(path, **out)
    stamp(path, sup, extra=dict(planes={
        "fp_counts_89": "the calibration block of record (89 events)",
        "grid_2378_lyaonly": "same selection, no N-hat floor, no z window",
        "grid_2318_lyaonly_zwin": "same selection, no N-hat floor, z window"}))
    return path, out, prov


# ---------------------------------------------------------------------------
# B.  the mock hostless census, decomposed
# ---------------------------------------------------------------------------
def candidate_counts(cat_tid, cat_z, t_tid, t_z, dz_rel, t_nhi=None):
    """Per cat row: (n_cand, dz_nearest, nhi_nearest) against a truth pool.

    ``n_cand`` = # truth rows with the same TARGETID and
    |z_cat - z_truth| / (1 + z_truth) < dz_rel — the number of hosts an
    'ANY host within the window' definition would accept, ignoring whether the
    greedy one-to-one matcher gave that host to somebody else.
    ``dz_nearest`` = the same distance to the CLOSEST truth row in the
    sightline (inf if the sightline has none), and ``nhi_nearest`` its N_HI:
    these measure how much of the hostless class is a WINDOW-SIZE artefact.
    Also returns per-truth-row candidate multiplicity (the blend diagnostic).
    """
    by = defaultdict(list)
    for j, t in enumerate(np.asarray(t_tid, np.int64)):
        by[int(t)].append(j)
    t_z = np.asarray(t_z, float)
    t_nhi = None if t_nhi is None else np.asarray(t_nhi, float)
    gidx = {k: np.asarray(v, int) for k, v in by.items()}
    cat_tid = np.asarray(cat_tid, np.int64)
    cat_z = np.asarray(cat_z, float)
    n = len(cat_tid)
    n_cand = np.zeros(n, np.int64)
    dz_near = np.full(n, np.inf)
    nhi_near = np.full(n, np.nan)
    contested = np.zeros(len(t_z), np.int64)
    for i in range(n):
        idx = gidx.get(int(cat_tid[i]))
        if idx is None:
            continue
        tz = t_z[idx]
        d = np.abs(cat_z[i] - tz) / (1.0 + tz)
        a = int(np.argmin(d))
        dz_near[i] = d[a]
        if t_nhi is not None:
            nhi_near[i] = t_nhi[idx[a]]
        close = d < dz_rel
        k = int(close.sum())
        if k:
            n_cand[i] = k
            contested[idx[close]] += 1
    return n_cand, dz_near, nhi_near, contested


def mock_pass(ep, family, work_dir):
    """``rebuild_a0_products.selection_pass`` (A0v2 convention) with the
    candidate bookkeeping added.  Steps 1-5 and 7-8 are COMMITTED calls; only
    step 6 is the A0v2 re-implementation (imported, not re-typed)."""
    import fitsio
    from astropy.table import Table
    from CDDF_analysis.hbi.cddf_catalog_hbi import (_build_qso_lookup,
                                                    load_molly_matrix)
    from CDDF_analysis.hbi import track_c_tf_saclay as TS
    sys.path.insert(0, os.path.join(_REPO, "examples"))
    from gp_native_pc_plots import load_catalog_dir
    from molly_faithful_pc_plots import match_truth_to_cat_molly

    t0 = time.time()
    cfg = ep._make_cfg(family, work_dir)
    mm = load_molly_matrix(cfg.molly_tsv)
    qso_lookup = _build_qso_lookup(cfg)

    cat = load_catalog_dir(cfg.catalog_dir)
    cat["S2N_RED"] = np.asarray(cat["SNR_REDSIDE"], dtype=float)

    truth_raw = Table(fitsio.read(cfg.truth_path, ext=1))
    z_col = next((c for c in ("Z_DLA", "Z_DLA_NO_RSD", "Z")
                  if c in truth_raw.colnames), None)
    if z_col != "Z_DLA":
        truth_raw.rename_column(z_col, "Z_DLA")
    truth_raw["Z_TRUTH"] = np.asarray(truth_raw["Z_DLA"], float)
    n_truth_raw = len(truth_raw)
    nhi_raw_min = float(np.asarray(truth_raw["NHI"], float).min())

    truth = truth_raw[np.asarray(truth_raw["NHI"], float) >= CENSUS_FLOOR]
    n_after_floor = len(truth)
    t_tids = np.asarray(truth["TARGETID"], np.int64)
    t_s2n = np.full(len(truth), np.nan)
    t_zq = np.full(len(truth), np.nan)
    for i, t in enumerate(t_tids):
        v = qso_lookup.get(int(t))
        if v is not None:
            t_s2n[i], t_zq[i] = v
    truth["S2N_RED"] = t_s2n
    truth["Z_QSO"] = t_zq
    keep_t = ~np.isnan(t_s2n) & ~np.isnan(t_zq)
    n_truth_unresolvable = int((~keep_t).sum())
    truth = truth[keep_t]

    nhi_err = np.asarray(cat["NHI_ERR"], float)
    zdla_err = np.asarray(cat["Z_DLA_ERR"], float)
    sentinel = (nhi_err == -1) | (zdla_err == -1)
    n_sentinel = int(sentinel.sum())
    cat = cat[~sentinel]

    bal_tids = None
    if cfg.no_bal:
        bal = fitsio.read(cfg.bal_cat_path, ext=1, columns=["TARGETID"])
        bal_tids = set(int(r["TARGETID"]) for r in bal)

    iter_order = "input" if cfg.molly_input_order else "nhi_desc"
    _tp, cat_NHI_TR, cat_Z_TR, _tm = match_truth_to_cat_molly(
        cat, truth, cfg.dz_rel, cat_iter_order=iter_order)
    cat["NHI_TRUE"] = cat_NHI_TR
    cat["Z_TRUE"] = cat_Z_TR
    cat["NHI_TILT_HOST"] = np.asarray(cat_NHI_TR, float).copy()

    # --- the candidate bookkeeping (the new part) -------------------------
    n_cand_pool, dz_near, nhi_near, contested_pool = candidate_counts(
        cat["TARGETID"], cat["Z_DLA"], truth["TARGETID"], truth["Z_TRUTH"],
        cfg.dz_rel, t_nhi=truth["NHI"])
    if len(truth) == len(truth_raw):
        n_cand_raw = n_cand_pool            # the pools are identical
    else:
        n_cand_raw, _, _, _ = candidate_counts(
            cat["TARGETID"], cat["Z_DLA"], truth_raw["TARGETID"],
            truth_raw["Z_TRUTH"], cfg.dz_rel)

    kw = dict(lam_rf_min=float(cfg.lam_rf_min), lam_rf_max=float(cfg.lam_rf_max),
              z_qso_min=float(cfg.z_qso_min), z_qso_max=float(cfg.z_qso_max),
              collar_kms=COLLAR)
    m_cat = R.lambda_z_bal_mask(cat["Z_DLA"], cat["Z_TRUE"], cat["Z_QSO"],
                                cat["TARGETID"], bal_tids,
                                z_cut_columns=Z_CUT, **kw)
    m_tru = R.lambda_z_bal_mask(truth["Z_DLA"], None, truth["Z_QSO"],
                                truth["TARGETID"], bal_tids,
                                z_cut_columns="zdla_only", **kw)
    cat_cut = cat[m_cat]
    truth_cut = truth[m_tru]
    n_cand_pool_cut = n_cand_pool[m_cat]
    n_cand_raw_cut = n_cand_raw[m_cat]

    TS._snap_off_molly_edges(cat_cut, truth_cut, mm)
    good = (np.asarray(cat_cut["DLAFLAG"], int) == 0)
    op = ((np.asarray(cat_cut["S2N_RED"], float) > cfg.snr_min)
          & (np.asarray(cat_cut["P_DLA"], float) > cfg.p_dla_min) & good)

    det = dict(nhat=np.asarray(cat_cut["NHI"], float)[op],
               zobs=np.asarray(cat_cut["Z_DLA"], float)[op],
               snr=np.asarray(cat_cut["S2N_RED"], float)[op],
               nhi_true=np.asarray(cat_cut["NHI_TRUE"], float)[op],
               n_cand_pool=n_cand_pool_cut[op],
               n_cand_raw=n_cand_raw_cut[op],
               dz_near=dz_near[m_cat][op],
               nhi_near=nhi_near[m_cat][op])
    meta = dict(n_truth_raw=int(n_truth_raw), nhi_truth_min=nhi_raw_min,
                n_truth_after_floor_17p2=int(n_after_floor),
                n_truth_unresolvable_dropped=n_truth_unresolvable,
                n_sentinel_dropped=n_sentinel,
                n_cat_cut=int(len(cat_cut)), n_op=int(op.sum()),
                dz_rel=float(cfg.dz_rel),
                n_truth_contested_ge2=int((contested_pool >= 2).sum()),
                seconds=round(time.time() - t0, 1))
    return det, meta, cfg


def build_mock(family, out_dir, ep, work_dir=None):
    det, meta, cfg = mock_pass(ep, family, work_dir)
    labels = SEL.classify_host_association(
        det["nhi_true"], det["n_cand_pool"], det["n_cand_raw"])

    def binner(nhat, zobs, snr):
        return ep.bin_counts_cks(nhat, zobs, snr)[0]      # COMMITTED binner

    grids = SEL.class_grids(labels, binner, det["nhat"], det["zobs"],
                            det["snr"])                   # (C,K,S) each
    total = grids.pop("_total")

    # --- FIDELITY GATE: reproduce the A0v2 census blocks bit-exactly -------
    cen_path = os.path.join(A0_SUPPORT_DIR, f"fp_census_{family}_A0v2.npz")
    cen = np.load(cen_path, allow_pickle=True)
    hostless = (grids["hostless_no_absorber"] + grids["hostless_taken"]
                + grids["hostless_unresolvable"])
    gate = {"counts_all": bool(np.array_equal(total, np.asarray(cen["counts_all"]))),
            "hostless": bool(np.array_equal(hostless, np.asarray(cen["hostless"])))}
    for slot in ("host_17p2_19p0", "host_19p0_19p5", "host_19p5_19p7",
                 "host_19p7_21p6", "host_ge_21p6"):
        gate[slot] = bool(np.array_equal(grids[slot], np.asarray(cen[slot])))
    if not all(gate.values()):
        raise SystemExit(f"A0v2 CENSUS GATE FAILED for {family}: {gate}")

    # --- SUPPORT GATE: identical support to the A0v2 census ---------------
    cen_sup = read_stamp(cen_path)
    sup = support_id(
        collar_kms=COLLAR, snr_min=float(cfg.snr_min),
        z_window=(float(cfg.z_qso_min), float(cfg.z_qso_max)),
        p_dla_min=float(cfg.p_dla_min), lya_only_lam_min=float(cfg.lam_rf_min),
        lam_rf_max=float(cfg.lam_rf_max), z_cut_columns=Z_CUT_ZDLA_ONLY,
        quality_cut="DLAFLAG==0", truth_host_floor=CENSUS_FLOOR,
        bal_policy=cen_sup.fields["bal_policy"],
        catalogue_id=cen_sup.fields["catalogue_id"],
        truth_catalogue_sha256=cen_sup.fields["truth_catalogue_sha256"])
    if sup.sha256 != cen_sup.sha256:
        raise SystemExit(
            f"SUPPORT GATE FAILED for {family}: {sup.sha256[:16]} != "
            f"{cen_sup.sha256[:16]} (A0v2 census)")

    out = {f"cks_{k}": v for k, v in grids.items()}
    out["cks_total"] = total
    out["cks_hostless"] = hostless
    for k in list(out):
        out["cs_" + k[4:]] = out[k].sum(axis=1)
    out["nhat_edges"] = SEL.NHAT_EDGES
    out["snr_edges"] = SEL.SNR_EDGES
    out["zf_edges"] = np.asarray(cen["zf_edges"])
    out["zc_edges"] = np.asarray(cen["zc_edges"])
    out["kz_to_K"] = np.asarray(cen["kz_to_K"])
    # blend diagnostics on the selected rows
    out["n_cand_pool_hist"] = np.bincount(
        np.asarray(det["n_cand_pool"], int), minlength=6)[:6]
    # window-size probe: for the hostless class, how far is the NEAREST truth
    # absorber in the same sightline, and how strong is it?
    hl = labels == "hostless_no_absorber"
    for tag, m in (("hostless", hl), ("taken", labels == "hostless_taken")):
        out[f"{tag}_dz_near"] = det["dz_near"][m].astype(np.float32)
        out[f"{tag}_nhi_near"] = det["nhi_near"][m].astype(np.float32)
        out[f"{tag}_nhat"] = det["nhat"][m].astype(np.float32)
        out[f"{tag}_zobs"] = det["zobs"][m].astype(np.float32)
        out[f"{tag}_snr"] = det["snr"][m].astype(np.float32)
    out["support_id"] = stamp_array(sup)
    prov = dict(
        role=("MOCK hostless-census estimand with the hostless class split by "
              "the mechanism that produced it (matching definition / "
              "unresolvable host / no absorber at all)"),
        family=family,
        recipe="validation/absorber_ladder/fp_target_review/build_fp_target_review.py",
        reuses=("rebuild_a0_products.lambda_z_bal_mask (A0v2 step 6) and "
                "extract_pack.bin_counts_cks; every other step is a COMMITTED "
                "call"),
        a0v2_census=os.path.abspath(cen_path),
        a0v2_census_sha256=file_sha256(cen_path),
        fidelity_gates_vs_A0v2_census=gate,
        support_gate_equals_A0v2_census=True,
        support_id=sup.sha256, meta=meta, code=_git(), built_utc=_now())
    out["provenance"] = json.dumps(prov)
    path = os.path.join(out_dir, f"fp_target_decomposition_{family}.npz")
    np.savez(path, **out)
    stamp(path, sup, extra=dict(planes={k: "(C,K,S) counts" for k in out
                                        if k.startswith("cks_")}))
    return path, out, prov, meta


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", nargs="+", default=list(FAMILIES))
    p.add_argument("--out", required=True)
    p.add_argument("--work", default=None)
    p.add_argument("--skip-loa0", action="store_true")
    p.add_argument("--skip-mock", action="store_true")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    summary = {"built_utc": _now(), "code": _git()}
    if not a.skip_loa0:
        pack = os.path.join(A0_SUPPORT_DIR, "scanpack_2lpt0_b300_A0.npz")
        path, _, prov = build_loa0(a.out, pack)
        print(f"[loa0] -> {path}")
        print(json.dumps(prov["selection_ladder"], indent=1))
        print(json.dumps(prov["counterfactual_support"], indent=1))
        summary["loa0"] = prov

    if not a.skip_mock:
        ep = R.A0._load_ep() if hasattr(R.A0, "_load_ep") else None
        if ep is None:
            import importlib.util as ilu
            spec = ilu.spec_from_file_location(
                "_fptr_ep",
                os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc",
                             "extract_pack.py"))
            ep = ilu.module_from_spec(spec)
            sys.modules[spec.name] = ep
            spec.loader.exec_module(ep)
        summary["mock"] = {}
        for fam in a.family:
            path, out, prov, meta = build_mock(fam, a.out, ep, a.work)
            n = {k[4:]: int(v.sum()) for k, v in out.items()
                 if k.startswith("cks_")}
            print(f"[{fam}] -> {path}\n  {json.dumps(n)}\n  {json.dumps(meta)}")
            summary["mock"][fam] = dict(counts=n, meta=meta, provenance=prov)

    with open(os.path.join(a.out, "FP_TARGET_REVIEW_SUMMARY.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("[done]", a.out)


if __name__ == "__main__":
    main()
