#!/usr/bin/env python
"""build_pack_v3.py — BOUNDED CORRECTED MOCK-PACK REBUILD (scan packs v3).

PI ruling 2026-09-13c §8 ("Mock packs of record"):
  "Bounded corrected rebuild for validation; old packs immutable; new version.
   Correct at least: collar/support mismatch; truth-aware z-cut inconsistency;
   sentinel handling; support metadata; FP census support.  Molly denominator
   window: repair only if the intended support is unambiguous, else escalate.
   Every corrected product carries a machine-readable ``support_id`` and fails
   closed on mismatch."

THE FIVE CORRECTIONS (and the counting argument that closes each)

 (1) COLLAR / SUPPORT MISMATCH  [A0_SUPPORT_REPORT.md §1, §4].
     ``build_scan_packs.py`` rebuilds ``counts`` / ``dX`` /
     ``dX_coarse_committed`` / ``fp_E_alloc`` at collar 3000 + b and copies
     ``truth_counts`` / ``truth_counts_bks`` BYTE-IDENTICALLY from the adopted
     collar-3000 pack.  v3 puts every plane at collar 3300.
     Counting argument: the collar-3000 mask is a NO-OP on an already
     collar-3000-cut truth table (0 of ~2e5 rows dropped) so the 3300 selection
     is strictly nested; the collar-3000 rebuild reproduces the adopted pack
     BIT-EXACTLY before anything is written.

 (2) TRUTH-AWARE z-CUT INCONSISTENCY  [A0 §5, instance #8].
     ``build_scan_packs`` applies the lambda/z window to ``Z_DLA`` alone;
     ``load_and_cut_catalog`` -> ``make_lambda_z_BAL_cuts(use_truth_z=True)``
     applies it to ``min/max(Z_DLA, Z_TRUE)``, leaking truth into the selection
     (877 / 965 / 861 rows = 1.00 / 1.10 / 0.99 % at fixed collar).  v3 puts
     counts, truth, census AND operators on the observable-only window.

 (3) SENTINEL HANDLING  [A0 §5.1].
     ``load_and_cut_catalog`` step 3 drops ``NHI_ERR == -1 OR Z_DLA_ERR == -1``
     BEFORE matching; ``build_scan_packs`` applies no sentinel filter at all, so
     the packs of record's ``counts`` carry 4 / 5 / 5 detections no matched
     object can ever carry.  v3 applies the filter to ``counts``.
     THE DECISIVE COUNTING ARGUMENT (this module's primary gate):
         build_scan_packs recipe @3300, Z_DLA-only, + sentinel filter
           ==  the A0v2/v3 census ``counts_all``      BIT-EXACTLY, 0 cells,
     i.e. the pack's numerator and the matched-object census are provably ONE
     row set.  Without the filter the same comparison leaves exactly the
     4 / 5 / 5 sentinel rows, all in SNR stratum s = 7.

 (4) SUPPORT METADATA.  Every product carries an 11-field row ``support_id``
     (``support_contract.py``), a ``.support.json`` stamp recording the
     product's own sha256, and an entry in ``support/MANIFEST.json``.
     ``pack.load_pack`` is a CLOSED schema and rejects an embedded
     ``support_id`` NPZ key, so the v3 PACK's support lives in the sidecar +
     manifest; the one-line ``pack._OPTIONAL_KEYS`` change that would fix that
     is DOCUMENTED, NOT APPLIED (see ``PROPOSED_PACK_PY_CHANGE``).

 (5) FP CENSUS SUPPORT.  The census and the empirical operators are rebuilt on
     the pack's own row support and stamped with the SAME row ``support_id``.
     The pack's ``fp_counts`` block is NOT on the survey support by nature
     (loa-0 HCD-free twin catalogue; op cut SNR>2 & P_DLA>0.99 & lam_rest>=1025
     ONLY -- no collar, no z_qso window, no BAL veto), so it is extracted as a
     SEPARATE product with its OWN support_id (a second contract instance) and
     the 89-vs-87 collar difference is DISCLOSED, never silently chosen.

 (6) MOLLY DENOMINATOR WINDOW -- studied, NOT repaired.  See
     ``molly_window_study()``: the intended support is AMBIGUOUS, so per the
     ruling this module escalates with both candidates and the counts under
     each.  Nothing in the v3 packs changes.

IMMUTABILITY.  The packs of record, the adopted packs, the census of record and
the operators of record are only ever READ; their sha256 is recorded before and
after.  Nothing under ``CDDF_analysis/`` is modified.  NO SAMPLER IS RUN.

ENV: ``gpdla`` for the build (jax-free); ``gpdla-hbi`` for the ``load_pack`` /
``build_cc_tensors`` acceptance in ``verify_pack_v3.py``.

Usage
-----
    python validation/absorber_ladder/support/build_pack_v3.py \
        --family 2lpt0 london0 saclay0 \
        --out /scratch/.../absorber_ladder_2026-09-13/support_v3
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
for _p in (_HERE, _ABSDIAG, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from support_contract import (                                   # noqa: E402
    NO_TRUTH_SIDE, ROW_SELECTION_FIELDS, SUPPORT_FIELDS, Z_CUT_ZDLA_ONLY,
    SupportContractError, check_support_consistency, file_sha256, stamp,
    stamp_array, support_id,
)
import build_a0_support as A0                                    # noqa: E402
import rebuild_a0_products as RB                                 # noqa: E402

FAMILIES = A0.FAMILIES
LYA = A0.LYA
C_KMS = A0.C_KMS
BASIS_FLOOR = A0.BASIS_FLOOR              # 19.0
CENSUS_FLOOR = A0.CENSUS_FLOOR            # 17.2
ADOPTED_COLLAR = A0.ADOPTED_COLLAR        # 3000.0
V3_COLLAR = A0.SCANPACK_COLLAR            # 3300.0
CENSUS_BLOCKS = A0.CENSUS_BLOCKS
HOST_SLOT_NAMES = A0.HOST_SLOT_NAMES
OPS_CONTRACT_KEYS = RB.OPS_CONTRACT_KEYS

#: the A0 products this build reproduces and then extends
A0_DIR = ("/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/"
          "support")

VERSION = "scanpack/v3-support-corrected"

#: the change that would let a pack carry its own support (NOT APPLIED HERE)
PROPOSED_PACK_PY_CHANGE = dict(
    file="CDDF_analysis/hbi_mcmc/pack.py",
    anchor='_OPTIONAL_KEYS = ("resp_fitcov_diag", "resp_N_ref", '
           '"truth_counts_bks",',
    proposed='_OPTIONAL_KEYS = ("support_id", "resp_fitcov_diag", '
             '"resp_N_ref", "truth_counts_bks",',
    effect="admits a 0-d '<U64' NPZ key 'support_id' to the closed schema so a "
           "pack's support travels WITH the pack instead of only in a sidecar "
           "that a careless copy can separate from it; load_pack ignores the "
           "key otherwise (it is not a dataclass field, so ModelAPack is "
           "unchanged and every existing pack still loads).",
    status="PROPOSED, NOT APPLIED — pack.py is a frozen production module "
           "(PI decision required; A0_SUPPORT_REPORT.md §7, calibration report "
           "F.4.3).",
    verified="probed 2026-09-13: PackSchemaError: unknown keys ['support_id'] "
             "(schema v1 is a closed contract)")

_SHA: dict = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def a0_pack(fam):
    return os.path.join(A0_DIR, f"scanpack_{fam}_b300_A0.npz")


def a0v2_census(fam):
    return os.path.join(A0_DIR, f"fp_census_{fam}_A0v2.npz")


def a0_ops(fam):
    return os.path.join(A0_DIR, f"empirical_ops_{fam}_A0.npz")


def _raw_catalogue(ep, fam):
    """The detection catalogue exactly as ``build_scan_packs.build_family``
    reads it (raw fitsio concatenation; NO astropy Table, NO matching)."""
    import fitsio
    m = ep.MOCKS[fam]
    catf = sorted(glob.glob(os.path.join(m["catalog_dir"], "dlacat*.fits")))
    cat = (np.concatenate([fitsio.read(f, ext=1) for f in catf])
           if len(catf) > 1 else fitsio.read(catf[0], ext=1))
    bal = np.unique(fitsio.read(m["bal_cat_path"], ext=1,
                                columns=["TARGETID"])["TARGETID"]
                    .astype(np.int64))
    return cat, bal


def counts_ladder(ep, fam, collar_kms):
    """``build_scan_packs``'s counts block + THE SENTINEL FILTER, with the
    row count after every single cut (the counting argument the recurring
    one-sided-support bug class demands).

    Returns ``(counts_no_sentinel, counts_sentinel_filtered, ladder)``.
    """
    cat, bal = _raw_catalogue(ep, fam)
    tid = cat["TARGETID"].astype(np.int64)
    zqc = np.asarray(cat["Z_QSO"], float)
    zd = np.asarray(cat["Z_DLA"], float)
    nhi = np.asarray(cat["NHI"], float)
    snr = np.asarray(cat["SNR_REDSIDE"], float)
    flag_ok = np.asarray(cat["DLAFLAG"], int) == 0
    p_ok = np.asarray(cat["P_DLA"], float) > 0.99
    snr_ok = snr > 2.0
    bal_ok = ~np.isin(tid, bal)
    zq_ok = (zqc > 2.0) & (zqc < 4.25)
    # load_and_cut_catalog step 3
    sentinel = ((np.asarray(cat["NHI_ERR"], float) == -1)
                | (np.asarray(cat["Z_DLA_ERR"], float) == -1))

    coll = float(collar_kms) / C_KMS
    z_lo = np.maximum(3600.0 / LYA - 1.0, 1025.0 * (1 + zqc) / LYA - 1.0 + coll)
    z_hi = np.minimum(zqc - coll, 1216.0 * (1 + zqc) / LYA - 1.0 - coll)
    win = (zd > z_lo) & (zd < z_hi)

    ladder = [("catalogue rows loaded", int(len(cat)))]
    m = flag_ok.copy()
    ladder.append(("& DLAFLAG == 0", int(m.sum())))
    m &= p_ok
    ladder.append(("& P_DLA > 0.99", int(m.sum())))
    m &= snr_ok
    ladder.append(("& SNR_REDSIDE > 2.0", int(m.sum())))
    m &= bal_ok
    ladder.append(("& not in bal_cat", int(m.sum())))
    m &= zq_ok
    ladder.append(("& 2.0 < Z_QSO < 4.25", int(m.sum())))
    m &= win
    ladder.append((f"& lambda/z window @collar {collar_kms:.0f} (Z_DLA only)",
                   int(m.sum())))
    op = m.copy()
    m_s = m & ~sentinel
    ladder.append(("& NOT sentinel (NHI_ERR == -1 or Z_DLA_ERR == -1)",
                   int(m_s.sum())))

    c_nos, _ = ep.bin_counts_cks(nhi[op], zd[op], snr[op])
    c_sen, _ = ep.bin_counts_cks(nhi[m_s], zd[m_s], snr[m_s])
    ladder.append(("binned onto (c, k, s) [no sentinel filter]",
                   int(c_nos.sum())))
    ladder.append(("binned onto (c, k, s) [sentinel filter APPLIED]",
                   int(c_sen.sum())))

    drop = op & sentinel
    rec = dict(
        ladder=[dict(step=s, n_rows=n) for s, n in ladder],
        n_sentinel_rows_in_catalogue=int(sentinel.sum()),
        n_sentinel_rows_surviving_the_op_cut=int(drop.sum()),
        n_sentinel_rows_that_land_on_the_grid=int(c_nos.sum() - c_sen.sum()),
        off_grid_note=("sentinel rows carry fit-failure NHI/Z values, so most "
                       "fall outside the (c, k, s) grid and never entered "
                       "`counts` in the first place; only the on-grid ones "
                       "move"))
    if drop.any():
        cs, _ = ep.bin_counts_cks(nhi[drop], zd[drop], snr[drop])
        rec["sentinel_on_grid_per_snr_stratum"] = [int(cs[:, :, s].sum())
                                                   for s in range(cs.shape[2])]
        rec["sentinel_on_grid_per_coarse_K"] = [int(cs[:, K * 5:(K + 1) * 5, :].sum())
                                                for K in range(3)]
    return c_nos.astype(np.int64), c_sen.astype(np.int64), rec


# ---------------------------------------------------------------------------
# the molly denominator window study (correction 6) — DIAGNOSIS ONLY
# ---------------------------------------------------------------------------
def molly_window_study(bmo, ep, work_dir, fam="2lpt0"):
    """Is the molly completeness denominator's INTENDED support unambiguous?

    The frozen ``molly_n_det`` / ``molly_n_tot`` in EVERY mock pack (and in the
    real pack) is ONE 2LPT-0 object — verified byte-identical across the three
    mock packs here.  It is built by
    ``ff_fp_estimator.build_molly_counts_cache`` ->
    ``cddf_catalog_hbi.regenerate_molly_counts`` ->
    ``molly_faithful_pc_plots.completeness_snr_nhi_bins``, whose denominator is
    ``mock_mask.sum()`` over ``truth_cut`` in (SNR stratum, molly N cell) with
    NO z-grid restriction beyond the lambda/z window.

    This routine reproduces the frozen matrix from the catalogues and then
    counts the SAME truth table under the pack's own (b, k, s) restrictions, so
    the 5.7 % discrepancy of the calibration report §D.8 is DECOMPOSED rather
    than attributed.
    """
    pk = np.load(A0.pack_of_record(fam), allow_pickle=False)
    mn_tot = np.asarray(pk["molly_n_tot"], float)
    mn_det = np.asarray(pk["molly_n_det"], float)
    me = np.asarray(pk["molly_nhi_edges"], float)
    ms = np.asarray(pk["molly_snr_edges"], float)
    ntrue = np.asarray(pk["ntrue_edges"], float)
    zf = np.asarray(pk["zf_edges"], float)

    same = {}
    for f2 in FAMILIES:
        p2 = np.load(A0.pack_of_record(f2), allow_pickle=False)
        same[f2] = dict(
            n_tot_identical_to_2lpt0=bool(np.array_equal(
                np.asarray(p2["molly_n_tot"], float), mn_tot)),
            n_det_identical_to_2lpt0=bool(np.array_equal(
                np.asarray(p2["molly_n_det"], float), mn_det)))

    p = bmo.matching_pass(ep, fam, work_dir, CENSUS_FLOOR)
    tru = p["truth"]
    n, z, s = tru["nhi"], tru["z"], tru["snr"]
    keep33 = bmo.collar_keep(z, tru["zqso"], V3_COLLAR,
                             float(p["cfg"].lam_rf_min),
                             float(p["cfg"].lam_rf_max))

    def hist(mask=None):
        out = np.zeros((len(ms) - 1, len(me) - 1))
        base = np.ones(len(n), bool) if mask is None else np.asarray(mask, bool)
        for i in range(len(ms) - 1):
            mi = base & (s > ms[i]) & (s < ms[i + 1])
            for j in range(len(me) - 1):
                out[i, j] = np.count_nonzero(mi & (n > me[j]) & (n < me[j + 1]))
        return out

    h_full = hist()
    reproduced = bool(np.array_equal(h_full, mn_tot))

    zok = (z >= zf[0]) & (z < zf[-1])
    nok = (n >= ntrue[0]) & (n < ntrue[-1])
    J = slice(int(np.searchsorted(me, BASIS_FLOOR)), len(me) - 1)  # cells >= 19.0
    I = slice(2, len(ms) - 1)                                      # live strata
    variants = {
        "A_of_record (molly's own lambda/z window, collar 3000, all z)":
            dict(mask=None),
        "B1 (+ pack fine-z grid: 2.0 <= Z_DLA < 3.5)": dict(mask=zok),
        "B1c (+ pack fine-z grid + collar 3300 = the v3 row support)":
            dict(mask=zok & keep33),
        "B2 (+ pack fine-z grid + pack latent range 19.0 <= N < 22.4) "
        "[conflates the sub-floor pad cells — shown for completeness only]":
            dict(mask=zok & nok),
        "B3 (+ pack fine-z grid + latent range + collar 3300) "
        "[idem]": dict(mask=zok & nok & keep33),
    }
    tot_counts = float(np.asarray(pk["counts"], float).sum())
    out = {}
    for lbl, v in variants.items():
        hh = hist(v["mask"])
        out[lbl] = dict(
            n_tot_all_cells=float(hh.sum()),
            n_tot_ge19p0_live_strata=float(hh[I, J].sum()),
            farr_ratio_vs_pack_counts=round(float(hh.sum() / tot_counts), 4))
    rec = dict(
        role="DIAGNOSIS ONLY — nothing in the v3 packs changes",
        family_the_matrix_was_built_from=fam,
        frozen_matrix_is_one_shared_2lpt0_object=same,
        matrix_reproduced_bit_exactly_from_the_catalogues=reproduced,
        frozen_total_n_tot=float(mn_tot.sum()),
        rebuilt_total_n_tot=float(h_full.sum()),
        frozen_ge19p0_live=dict(
            n_tot=float(mn_tot[I, J].sum()), n_det=float(mn_det[I, J].sum()),
            C=round(float(mn_det[I, J].sum() / mn_tot[I, J].sum()), 6)),
        pack_truth_counts_total=float(np.asarray(pk["truth_counts"],
                                                 float).sum()),
        variants=out,
        decomposition=(
            "the ENTIRE discrepancy is the z-grid restriction, not a "
            "collar/window mismatch: the frozen denominator reproduces "
            "bit-exactly from the collar-3000 truth pass, and restricting the "
            "SAME truth rows to the pack's fine-z grid [2.0, 3.5) takes "
            "n_tot(>=19.0, live strata) from the frozen value to the pack's "
            "own truth_counts total EXACTLY"),
        farr_gate_note=(
            "model_a.run hard-fails when sum(molly_n_tot) < 4 * sum(counts) "
            "(the Farr N_eff gate). The ratio under each candidate is listed "
            "above; candidate B drops it and MAY cross the 4.0 threshold — a "
            "frozen production guard is therefore load-bearing on this "
            "choice."),
        ambiguity=dict(
            candidate_A="molly's own marginalisation window (the object of "
                        "record). Recorded class: matching_contract.py "
                        "Quantity('molly_n_det / molly_n_tot') = "
                        "FIXED_CALIBRATION_PRODUCT, 'frozen 2LPT-0; identical "
                        "in all three mock packs' — and the Population note "
                        "says the denominator is deliberately 'NOT the pack's "
                        "basis bins'. The same object is carried by the REAL "
                        "pack, which has no truth side at all.",
            candidate_B="the pack window's population: the fold applies C at "
                        "cells (b, k, s) with k restricted to [2.0, 3.5), and "
                        "completeness is measurably z-dependent (C0 K-residual "
                        "+2.4 / -7.5 / -8.5 %), so a denominator marginalised "
                        "over a WIDER z range than the grid it is applied on "
                        "is a support mismatch of the recurring class.",
            verdict="AMBIGUOUS — no object on disk and no line of code records "
                    "the INTENDED z-marginalisation range; the recorded intent "
                    "('NOT the pack's basis bins') speaks only to the N-cell "
                    "granularity and the open/half-open convention. Per PI "
                    "ruling 2026-09-13c §8 this is ESCALATED, not chosen. "
                    "Note also that every C1 completeness variant removes the "
                    "mismatch BY CONSTRUCTION (re-fit on the pack window), so "
                    "the ladder does not depend on the ruling."))
    return rec, dict(molly_n_tot_frozen=mn_tot, molly_n_det_frozen=mn_det,
                     molly_n_tot_rebuilt_A=h_full,
                     molly_n_tot_rebuilt_B1=hist(zok),
                     molly_n_tot_rebuilt_B1c=hist(zok & keep33),
                     molly_n_tot_rebuilt_B2=hist(zok & nok),
                     molly_n_tot_rebuilt_B3=hist(zok & nok & keep33),
                     molly_nhi_edges=me, molly_snr_edges=ms)


# ---------------------------------------------------------------------------
def build(family, out_dir, work_dir=None):
    t0 = time.time()
    bmo = A0._bmo()
    ep = bmo._load_ep()
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "support"), exist_ok=True)
    work_dir = work_dir or os.path.join(out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)
    rec: dict = dict(family=family, version=VERSION)

    sp_path = A0.pack_of_record(family)
    ad_path = A0.adopted_pack(family)
    cen_rec = A0.census_of_record(family)
    ops_rec = RB.OPS_OF_RECORD.format(fam=family)
    a0p, a0c, a0o = a0_pack(family), a0v2_census(family), a0_ops(family)
    for p in (sp_path, ad_path, cen_rec, ops_rec, a0p, a0c, a0o):
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}")

    inputs = {k: dict(path=v, sha256=file_sha256(v, _SHA)) for k, v in dict(
        pack_of_record=sp_path, adopted_pack=ad_path, census_of_record=cen_rec,
        ops_of_record=ops_rec, a0_pack=a0p, a0v2_census=a0c,
        a0_ops=a0o).items()}
    rec["inputs"] = inputs

    sp = np.load(sp_path, allow_pickle=False)
    ad = np.load(ad_path, allow_pickle=False)
    ntrue = np.asarray(ad["ntrue_edges"], float)
    nhat = np.asarray(ad["nhat_edges"], float)
    zf = np.asarray(ad["zf_edges"], float)
    snr_e = np.asarray(ad["snr_edges"], float)
    kz = np.asarray(ad["kz_to_K"], int)

    # =====================================================================
    # (1)+(2)+(3) the COUNTS plane
    # =====================================================================
    c_nos, c_sen, cl = counts_ladder(ep, family, V3_COLLAR)
    rec["counts_counting_argument"] = cl
    pack_counts = np.asarray(sp["counts"], np.int64)
    rec["counts_gates"] = dict(
        recipe_reproduces_pack_of_record_bit_exactly=bool(
            np.array_equal(c_nos, pack_counts)),
        pack_of_record_total=int(pack_counts.sum()),
        v3_total=int(c_sen.sum()),
        delta=int(c_sen.sum() - pack_counts.sum()),
        n_cells_differing=int(np.count_nonzero(c_sen - pack_counts)),
        max_abs_cell_difference=int(np.abs(c_sen - pack_counts).max()),
        per_snr_stratum=[int((c_sen - pack_counts)[:, :, s].sum())
                         for s in range(c_sen.shape[2])],
        per_coarse_K=[int((c_sen - pack_counts)[:, K * 5:(K + 1) * 5, :].sum())
                      for K in range(3)])
    if not rec["counts_gates"]["recipe_reproduces_pack_of_record_bit_exactly"]:
        raise SystemExit("COUNTS REGRESSION GATE FAILED: the build_scan_packs "
                         "recipe does not reproduce the pack of record — the "
                         "selection premise is wrong; nothing written")
    print(f"[{family}] counts: {int(pack_counts.sum())} -> {int(c_sen.sum())} "
          f"(sentinel filter, {rec['counts_gates']['n_cells_differing']} cells)",
          flush=True)

    # =====================================================================
    # (1)+(2) the TRUTH planes (the A0 rebuild, re-derived and re-gated here)
    # =====================================================================
    p19 = bmo.matching_pass(ep, family, work_dir, BASIS_FLOOR)
    cfg = p19["cfg"]
    snr_min = float(cfg.snr_min)
    lam_lo, lam_hi = float(cfg.lam_rf_min), float(cfg.lam_rf_max)
    tru = p19["truth"]
    tcb_3000, n_on_grid_3000, n_snr_3000 = bmo.truth_hist_bks(
        tru, ntrue, zf, snr_e, snr_min)
    if not np.array_equal(tcb_3000, np.asarray(ad["truth_counts_bks"], float)):
        raise SystemExit("COLLAR-3000 TRUTH GATE FAILED — nothing written")
    keep30 = bmo.collar_keep(tru["z"], tru["zqso"], ADOPTED_COLLAR,
                             lam_lo, lam_hi)
    if not keep30.all():
        raise SystemExit("SELF-TEST FAILED: the collar-3000 mask drops rows "
                         "from an already-collar-3000 truth table")
    keep33 = bmo.collar_keep(tru["z"], tru["zqso"], V3_COLLAR, lam_lo, lam_hi)
    tcb_3300, n_on_grid_3300, n_snr_3300 = bmo.truth_hist_bks(
        tru, ntrue, zf, snr_e, snr_min, keep=keep33)
    tc_3300 = tcb_3300.sum(axis=2)
    rec["truth_counting_argument"] = dict(ladder=[
        dict(step="truth rows after load_and_cut_catalog @floor 19.0, "
                  "collar 3000, Z_DLA-only window",
             n_rows=int(p19["n_truth_cut"])),
        dict(step="& SNR_REDSIDE > 2.0 (strict)", n_rows=int(n_snr_3000)),
        dict(step="& on the (b, k, s) grid [collar 3000]",
             n_rows=int(n_on_grid_3000)),
        dict(step="& collar-3000 mask re-applied (idempotence self-test)",
             n_rows=int(keep30.sum()),
             NOOP=bool(keep30.all())),
        dict(step="& collar-3300 window", n_rows=int(keep33.sum())),
        dict(step="& SNR > 2 & on the (b, k, s) grid [collar 3300] = v3 truth",
             n_rows=int(n_on_grid_3300))],
        collar_ratio=round(float(tcb_3300.sum() / tcb_3000.sum()), 6),
        n_truth_systems_removed=int(tcb_3000.sum() - tcb_3300.sum()))
    rec["truth_gates"] = dict(
        collar3000_rebuild_equals_adopted_pack_bit_exactly=True,
        v3_truth_equals_A0_pack_truth_bit_exactly=None,   # filled below
        of_record_truth_total=float(np.asarray(sp["truth_counts"],
                                               float).sum()),
        v3_truth_total=float(tc_3300.sum()))

    a0 = np.load(a0p, allow_pickle=False)
    same_tc = bool(np.array_equal(tc_3300, np.asarray(a0["truth_counts"], float)))
    same_tcb = bool(np.array_equal(tcb_3300,
                                   np.asarray(a0["truth_counts_bks"], float)))
    rec["truth_gates"]["v3_truth_equals_A0_pack_truth_bit_exactly"] = dict(
        truth_counts=same_tc, truth_counts_bks=same_tcb)
    if not (same_tc and same_tcb):
        raise SystemExit("A0 REPRODUCTION GATE FAILED on the truth planes")
    print(f"[{family}] truth: {float(np.asarray(sp['truth_counts']).sum()):.0f} "
          f"-> {tc_3300.sum():.0f} (collar 3300; == A0 bit-exactly)", flush=True)

    # =====================================================================
    # (2)+(5) the CENSUS and the OPERATORS, on the pack's own row support
    # =====================================================================
    f19_old = RB.selection_pass(ep, family, work_dir, BASIS_FLOOR,
                                collar_kms=ADOPTED_COLLAR,
                                z_cut_columns="minmax")
    f172_old = RB.selection_pass(ep, family, work_dir, CENSUS_FLOOR,
                                 collar_kms=ADOPTED_COLLAR,
                                 z_cut_columns="minmax")
    cen_old = RB.census_blocks(ep, f172_old["det"])
    cz = np.load(cen_rec, allow_pickle=True)
    g_cen = {k: bool(np.array_equal(cen_old[k].astype(float),
                                    np.asarray(cz[k], float)))
             for k in CENSUS_BLOCKS}
    ops_old = RB.build_ops(bmo, a0, f19_old["det"], f172_old["det"],
                           f19_old["truth"], ntrue, nhat, zf, snr_e, kz,
                           snr_min)
    oz = np.load(ops_rec, allow_pickle=True)
    g_ops = {k: bool(np.array_equal(ops_old[k], np.asarray(oz[k], float)))
             for k in OPS_CONTRACT_KEYS}
    rec["fidelity_gates_committed_convention"] = dict(
        note="the step-6 re-implementation, run with the COMMITTED convention "
             "(minmax z columns, collar 3000), must reproduce the objects of "
             "record BIT-EXACTLY before anything is written",
        census_blocks=g_cen, ops_contract_arrays=g_ops,
        n_sentinel_dropped_before_matching=f172_old["n_sentinel_dropped"])
    bad = ([k for k, v in g_cen.items() if not v]
           + [k for k, v in g_ops.items() if not v])
    if bad:
        raise SystemExit(f"FIDELITY GATE FAILED on {bad} — nothing written")

    f19_new = RB.selection_pass(ep, family, work_dir, BASIS_FLOOR,
                                collar_kms=V3_COLLAR,
                                z_cut_columns="zdla_only")
    f172_new = RB.selection_pass(ep, family, work_dir, CENSUS_FLOOR,
                                 collar_kms=V3_COLLAR,
                                 z_cut_columns="zdla_only")
    cen_new = RB.census_blocks(ep, f172_new["det"])
    ops_new = RB.build_ops(bmo, a0, f19_new["det"], f172_new["det"],
                           f19_new["truth"], ntrue, nhat, zf, snr_e, kz,
                           snr_min)

    # ---- THE DECISIVE COUNTING IDENTITY (correction 3) --------------------
    ca = cen_new["counts_all"].astype(np.int64)
    d_id = c_sen - ca
    rec["counting_identity_counts_vs_census"] = dict(
        claim="build_scan_packs recipe @3300 Z_DLA-only WITH the "
              "load_and_cut_catalog sentinel filter == the v3 census "
              "counts_all, bit-exactly; the pack's numerator and the matched "
              "census are ONE row set",
        v3_counts_total=int(c_sen.sum()),
        census_counts_all_total=int(ca.sum()),
        EXACT=bool(np.array_equal(c_sen, ca)),
        n_cells_differing=int(np.count_nonzero(d_id)),
        control_without_the_sentinel_filter=dict(
            total=int(c_nos.sum()),
            residual_vs_census=int(c_nos.sum() - ca.sum()),
            n_cells_differing=int(np.count_nonzero(c_nos - ca)),
            note="this residual IS the A0v2 -4 / -5 / -5, all in SNR "
                 "stratum s = 7 — the defect correction (3) removes"))
    if not rec["counting_identity_counts_vs_census"]["EXACT"]:
        raise SystemExit("COUNTING IDENTITY FAILED: v3 counts != census "
                         "counts_all — the sentinel correction does NOT close "
                         "the support; nothing written")
    print(f"[{family}] COUNTING IDENTITY: v3 counts == census counts_all "
          f"({int(c_sen.sum())}), 0 cells differing", flush=True)

    # ---- v3 census / ops must reproduce A0v2 / A0 bit-exactly -------------
    a0cz = np.load(a0c, allow_pickle=True)
    a0oz = np.load(a0o, allow_pickle=True)
    rec["A0_reproduction_gates"] = dict(
        census_blocks={k: bool(np.array_equal(cen_new[k].astype(float),
                                              np.asarray(a0cz[k], float)))
                       for k in CENSUS_BLOCKS},
        ops_contract_arrays={k: bool(np.array_equal(
            ops_new[k], np.asarray(a0oz[k], float)))
            for k in OPS_CONTRACT_KEYS})
    bad = ([k for k, v in rec["A0_reproduction_gates"]["census_blocks"].items()
            if not v]
           + [k for k, v in rec["A0_reproduction_gates"][
               "ops_contract_arrays"].items() if not v])
    if bad:
        raise SystemExit(f"A0 REPRODUCTION GATE FAILED on {bad}")

    rec["deltas_vs_of_record"] = dict(
        census={k: dict(of_record_truth_aware_c3000=int(cen_old[k].sum()),
                        v3=int(cen_new[k].sum()),
                        delta=int(cen_new[k].sum() - cen_old[k].sum()))
                for k in CENSUS_BLOCKS},
        N_match_cksb=dict(of_record=float(ops_old["N_match_cksb"].sum()),
                          v3=float(ops_new["N_match_cksb"].sum()),
                          delta=float(ops_new["N_match_cksb"].sum()
                                      - ops_old["N_match_cksb"].sum())),
        truth_counts_bks_inside_the_operators=dict(
            of_record=float(ops_old["truth_counts_bks"].sum()),
            v3=float(ops_new["truth_counts_bks"].sum()),
            equals_v3_pack_truth_counts_bks=bool(np.array_equal(
                ops_new["truth_counts_bks"], tcb_3300))))
    if not rec["deltas_vs_of_record"]["truth_counts_bks_inside_the_operators"][
            "equals_v3_pack_truth_counts_bks"]:
        raise SystemExit("the operators' truth plane is not the pack's truth "
                         "plane — the division is still one-sided")

    # =====================================================================
    # (4) supports
    # =====================================================================
    m = ep.MOCKS[family]

    def _sup(floor):
        return A0.family_support(cfg, m["catalog_dir"], str(cfg.truth_path),
                                 m["bal_cat_path"], collar_kms=V3_COLLAR,
                                 truth_host_floor=floor,
                                 z_cut_columns=Z_CUT_ZDLA_ONLY)

    sup_data = _sup(NO_TRUTH_SIDE)          # counts / dX / fp_E_alloc
    sup_truth = _sup(BASIS_FLOOR)           # truth planes + operators
    sup_census = _sup(CENSUS_FLOOR)         # census
    # the loa-0 FP calibration block: a SEPARATE contract instance
    loa0_cat = _loa0_identity()
    sup_fp = support_id(
        collar_kms="none (extract_pack.build_fp_block applies NO collar)",
        snr_min=2.0, z_window="none (no Z_QSO admission window)",
        p_dla_min=0.99, lya_only_lam_min=1025.0,
        lam_rf_max="none (no red edge; lam_rest >= 1025 A only)",
        z_cut_columns="Z_DLA in [2.0, 3.5) (the fine mu_FP grid only)",
        quality_cut="none (no DLAFLAG cut)",
        truth_host_floor="n/a (loa-0 is HCD-free by construction: every "
                         "detection is a forest FP)",
        bal_policy="none (no BAL veto on the loa-0 twin)",
        catalogue_id=loa0_cat["catalogue_id"],
        truth_catalogue_sha256=loa0_cat["product_sha256"])
    if len({sup_data.row_sha256, sup_truth.row_sha256,
            sup_census.row_sha256}) != 1:
        raise SystemExit("row support differs across the v3 planes")
    if sup_fp.row_sha256 == sup_data.row_sha256:
        raise SystemExit("fp_counts must NOT share the survey row support — "
                         "it is a different selection by nature")
    rec["support_ids"] = dict(
        row_support_shared_by_pack_truth_census_ops=sup_data.row_sha256,
        row_support_short=sup_data.row_sha256[:16],
        full_data_plane=sup_data.sha256, full_truth_plane=sup_truth.sha256,
        full_census=sup_census.sha256,
        fp_counts_OWN_support=sup_fp.sha256,
        fp_counts_row_support=sup_fp.row_sha256,
        fp_counts_note="DIFFERENT BY NATURE — a second contract instance, "
                       "never comparable to the survey support")
    rec["support_fields_row"] = {k: sup_data.fields[k]
                                 for k in ROW_SELECTION_FIELDS}
    rec["truth_host_floor_by_plane"] = dict(
        counts_dX_fp_E_alloc=NO_TRUTH_SIDE, truth_counts=BASIS_FLOOR,
        operators=BASIS_FLOOR, census=CENSUS_FLOOR)

    # =====================================================================
    # write
    # =====================================================================
    prov = dict(
        version=VERSION, role=__doc__.splitlines()[2].strip(),
        family=family,
        recipe="validation/absorber_ladder/support/build_pack_v3.py",
        corrections={
            "1_collar": "counts / dX / dX_coarse_committed / fp_E_alloc / "
                        "truth_counts / truth_counts_bks / census / operators "
                        f"ALL at collar {V3_COLLAR:.0f} km/s",
            "2_z_cut_columns": "observable-only (Z_DLA) lambda/z window on "
                               "counts AND truth AND census AND operators",
            "3_sentinel": "load_and_cut_catalog step 3 (NHI_ERR == -1 OR "
                          "Z_DLA_ERR == -1) applied to `counts`",
            "4_support_metadata": "row support_id on every product + sidecar + "
                                  "support/MANIFEST.json",
            "5_fp_census_support": "census + operators on the pack's own row "
                                   "support; fp_counts extracted with its OWN "
                                   "support_id",
            "6_molly_window": "STUDIED, NOT REPAIRED — escalated (see "
                              "MOLLY_WINDOW_STUDY.json)"},
        machinery="validation/absorber_diag/build_matched_ops.py + "
                  "validation/absorber_ladder/support/rebuild_a0_products.py "
                  "(step 6 only) + CDDF_analysis/hbi_mcmc/extract_pack.py "
                  "(bin_counts_cks) — committed code called, not reimplemented",
        frozen_code_untouched=[
            "CDDF_analysis/hbi_mcmc/build_scan_packs.py",
            "CDDF_analysis/hbi_mcmc/extract_pack.py",
            "CDDF_analysis/hbi_mcmc/pack.py",
            "CDDF_analysis/hbi_mcmc/model_cc*",
            "CDDF_analysis/hbi_mcmc/perz_gate.py",
            "CDDF_analysis/hbi/adopted_response/*",
            "CDDF_analysis/hbi/cddf_catalog_hbi.py",
            "validation/fp_ladder/build_fp_census.py",
            "validation/absorber_diag/build_matched_ops.py"],
        proposed_pack_py_change=PROPOSED_PACK_PY_CHANGE,
        code=A0._git_head(),
        env=dict(python=platform.python_version(), numpy=np.__version__,
                 conda_prefix=os.environ.get("CONDA_PREFIX"),
                 host=platform.node()),
        inputs=inputs,
        catalog_dir=str(m["catalog_dir"]), truth_path=str(cfg.truth_path),
        bal_cat_path=str(m["bal_cat_path"]), molly_tsv=str(cfg.molly_tsv),
        snr_min=snr_min, p_dla_min=float(cfg.p_dla_min),
        lam_rf_min=lam_lo, lam_rf_max=lam_hi,
        z_qso_window=[float(cfg.z_qso_min), float(cfg.z_qso_max)],
        collar_kms=V3_COLLAR, z_cut_columns=Z_CUT_ZDLA_ONLY,
        built_utc=datetime.datetime.utcnow().isoformat() + "Z",
        findings=rec)

    written = []

    # ---- the v3 pack ------------------------------------------------------
    raw = dict(np.load(sp_path, allow_pickle=False))
    raw["counts"] = c_sen.astype(np.asarray(sp["counts"]).dtype)
    raw["truth_counts"] = tc_3300.astype(np.asarray(sp["truth_counts"]).dtype)
    raw["truth_counts_bks"] = tcb_3300.astype(
        np.asarray(sp["truth_counts_bks"]).dtype)
    changed = sorted(k for k in raw
                     if not np.array_equal(np.asarray(raw[k]),
                                           np.asarray(sp[k])))
    if changed != ["counts", "truth_counts", "truth_counts_bks"]:
        raise SystemExit(f"v3 pack changed unexpected keys: {changed}")
    pk_v3 = os.path.join(out_dir, f"scanpack_{family}_b300_v3.npz")
    np.savez_compressed(pk_v3, **raw)
    pprov = dict(prov)
    pprov.update(product=pk_v3, source_pack=sp_path,
                 keys_changed=changed,
                 keys_byte_identical_to_the_pack_of_record=sorted(
                     set(raw) - set(changed)),
                 support_id_storage="sidecar .support.json + "
                                    "support/MANIFEST.json ONLY — "
                                    "pack.load_pack is a CLOSED schema and "
                                    "raises PackSchemaError on an extra NPZ "
                                    "key (see proposed_pack_py_change)")
    with open(pk_v3.replace(".npz", ".provenance.json"), "w") as fh:
        json.dump(pprov, fh, indent=1, default=str)
    stamp(pk_v3, sup_data, extra=dict(
        note="v3 pack: counts (sentinel-filtered), dX, fp_E_alloc and "
             "truth_counts / truth_counts_bks all on ONE collar-3300, "
             "Z_DLA-only row support",
        version=VERSION,
        planes={"counts": dict(sup_data.fields), "dX": dict(sup_data.fields),
                "fp_E_alloc": dict(sup_data.fields),
                "truth_counts": dict(sup_truth.fields),
                "truth_counts_bks": dict(sup_truth.fields)},
        excluded_planes={
            "fp_counts": "NOT on the survey support by nature — stamped "
                         "separately as fp_counts_<fam>_v3.npz with its own "
                         "support_id",
            "molly_n_det/molly_n_tot/g_grid/resp_*": "frozen 2LPT-0 "
                                                     "calibration products, "
                                                     "not row selections"}))
    written.append(pk_v3)

    # ---- the v3 census ----------------------------------------------------
    cen_v3 = os.path.join(out_dir, f"fp_census_{family}_v3.npz")
    np.savez(cen_v3,
             **{k: cen_new[k] for k in CENSUS_BLOCKS},
             **{k + "_of_record_truth_aware_collar3000": cen_old[k]
                for k in CENSUS_BLOCKS},
             nhat_edges=ep.NHAT_EDGES, zf_edges=ep.ZF_EDGES,
             snr_edges=ep.SNR_EDGES, zc_edges=ep.ZC_EDGES, kz_to_K=ep.KZ_TO_K,
             host_slot_names=np.array(HOST_SLOT_NAMES),
             support_id=stamp_array(sup_census),
             provenance=np.array(json.dumps(prov, default=str)))
    with open(cen_v3.replace(".npz", ".provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1, default=str)
    with open(cen_v3.replace(".npz", ".json"), "w") as fh:
        json.dump(dict(family=family, version=VERSION, collar_kms=V3_COLLAR,
                       z_cut_columns=Z_CUT_ZDLA_ONLY,
                       totals={k: int(cen_new[k].sum()) for k in CENSUS_BLOCKS},
                       totals_of_record={k: int(cen_old[k].sum())
                                         for k in CENSUS_BLOCKS},
                       counting_identity=rec[
                           "counting_identity_counts_vs_census"],
                       support_id=sup_census.sha256), fh, indent=1, default=str)
    stamp(cen_v3, sup_census, extra=dict(
        note="v3 census: every block on the pack's own collar-3300 Z_DLA-only "
             "row support (instance #8 closed)",
        version=VERSION,
        planes={k: dict(sup_census.fields) for k in CENSUS_BLOCKS}))
    written.append(cen_v3)

    # ---- the v3 operators -------------------------------------------------
    ops_v3 = os.path.join(out_dir, f"empirical_ops_{family}_v3.npz")
    np.savez_compressed(
        ops_v3, **ops_new,
        truth_counts_bks_of_record_truth_aware_collar3000=ops_old[
            "truth_counts_bks"],
        ntrue_edges=ntrue, nhat_edges=nhat, zf_edges=zf, snr_edges=snr_e,
        zc_edges=np.asarray(a0["zc_edges"], float), kz_to_K=kz,
        support_id=stamp_array(sup_truth),
        provenance=np.array(json.dumps(prov, default=str), dtype=object))
    with open(ops_v3.replace(".npz", ".provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1, default=str)
    stamp(ops_v3, sup_truth, extra=dict(
        note="v3 empirical operators; the truth_counts_bks they divide by is "
             "BIT-IDENTICAL to the v3 pack's truth plane",
        version=VERSION, contract_keys=list(OPS_CONTRACT_KEYS),
        planes={k: dict(sup_truth.fields) for k in OPS_CONTRACT_KEYS}))
    written.append(ops_v3)

    # ---- the fp_counts block, on its OWN support --------------------------
    fp_v3 = os.path.join(out_dir, f"fp_counts_{family}_v3.npz")
    np.savez(fp_v3,
             fp_counts=np.asarray(sp["fp_counts"]),
             fp_ell_eff=np.asarray(sp["fp_ell_eff"]),
             fp_w_sightline_ratio=np.asarray(sp["fp_w_sightline_ratio"]),
             nhat_edges=nhat, snr_edges=snr_e,
             support_id=stamp_array(sup_fp),
             provenance=np.array(json.dumps(dict(
                 version=VERSION, family=family,
                 role="the loa-0 forest-FP calibration block, COPIED "
                      "UNCHANGED from the pack of record and stamped with its "
                      "OWN support_id",
                 source_pack=sp_path,
                 source_pack_sha256=inputs["pack_of_record"]["sha256"],
                 fp_counts_total=float(np.asarray(sp["fp_counts"]).sum()),
                 disclosed_difference=_FP_DISCLOSURE,
                 loa0_identity=loa0_cat), default=str)))
    stamp(fp_v3, sup_fp, extra=dict(
        note="SECOND CONTRACT INSTANCE — the loa-0 FP calibration block is on "
             "a DIFFERENT support from the survey planes BY NATURE; it must "
             "never be entered into the survey-support gate",
        version=VERSION, disclosed_difference=_FP_DISCLOSURE,
        planes={"fp_counts": dict(sup_fp.fields)}))
    written.append(fp_v3)

    rec["fp_counts_disclosure"] = _FP_DISCLOSURE
    rec["fp_counts_total"] = float(np.asarray(sp["fp_counts"]).sum())
    rec["wall_s"] = round(time.time() - t0, 1)

    # immutability re-check
    rec["inputs_unchanged_after_build"] = {
        k: bool(file_sha256(v["path"]) == v["sha256"])
        for k, v in inputs.items()}
    if not all(rec["inputs_unchanged_after_build"].values()):
        raise SystemExit("AN INPUT OF RECORD CHANGED DURING THE BUILD")

    print(f"[{family}] wrote {len(written)} v3 products "
          f"({rec['wall_s']:.0f}s)", flush=True)
    return rec, written


_FP_DISCLOSURE = dict(
    what="the pack's fp_counts block is read by extract_pack.build_fp_block "
         "from the loa-0 HCD-free twin catalogue under SNR > 2 & P_DLA > 0.99 "
         "& lam_rest >= 1025 A & Z_DLA in [2.0, 3.5) ONLY",
    not_applied=["proximity collar", "Z_QSO admission window", "BAL veto",
                 "DLAFLAG quality cut"],
    collar_difference="the real pack's selection_contract.json records 89 "
                      "events on support vs 87 under the c = 3300 collar "
                      "(R-015, 2026-08-26)",
    status="DISCLOSED, NOT CHOSEN — 89 vs 87 is a PI decision (the modular-Λ "
           "cut-feedback question of PI ruling 2026-09-13c §7/§11); v3 copies "
           "the block UNCHANGED and stamps it with its own support_id so the "
           "difference can never again be silent")


def _loa0_identity():
    """Identity of the loa-0 FP calibration inputs (for the fp_counts stamp)."""
    from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
    from CDDF_analysis.hbi import build_loa0_fp_product as BL
    prod = AB.DEF_LOA0_PRODUCT
    out = BL.DEF_LOA0_OUT
    cat_id = "loa-0 catalogue dir absent at stamp time"
    try:
        from support_contract import catalogue_identity
        cat_id = catalogue_identity(out, cache=_SHA)
    except Exception as exc:                                # pragma: no cover
        cat_id = f"loa0:{os.path.basename(str(out))}:unhashable({exc})"
    return dict(catalogue_id=cat_id, product_path=prod,
                product_sha256=(file_sha256(prod, _SHA)
                                if os.path.exists(prod)
                                else "loa0-product-absent"),
                loa0_out=str(out))


# ---------------------------------------------------------------------------
def write_manifest(out_dir, families, summary):
    """``support/MANIFEST.json`` — every product, its sha256, its support_id
    and the support fields of every one of its stamped planes.

    This is the machine-readable support metadata that does NOT depend on the
    pack schema staying closed: it travels with the directory, names the
    product's sha256, and is what ``verify_pack_v3.py`` re-derives.
    """
    from support_contract import read_stamp, stamp_path
    man = dict(schema="pack_v3_support_manifest/v1", version=VERSION,
               built_utc=datetime.datetime.utcnow().isoformat() + "Z",
               directory=os.path.abspath(out_dir), products={})
    for fam in families:
        for kind, name in (("pack", f"scanpack_{fam}_b300_v3.npz"),
                           ("census", f"fp_census_{fam}_v3.npz"),
                           ("ops", f"empirical_ops_{fam}_v3.npz"),
                           ("fp_counts", f"fp_counts_{fam}_v3.npz")):
            p = os.path.join(out_dir, name)
            if not os.path.exists(p):
                continue
            with open(stamp_path(p)) as fh:
                st = json.load(fh)
            sid = read_stamp(p)
            man["products"][name] = dict(
                family=fam, kind=kind, sha256=file_sha256(p),
                support_id=sid.sha256, support_id_short=sid.sha256[:16],
                row_support_id=sid.row_sha256,
                on_the_survey_support=(kind != "fp_counts"),
                fields=st["fields"],
                planes=sorted((st.get("extra", {}) or {}).get("planes", {})))
    man["survey_row_support_by_family"] = {
        fam: summary[fam]["support_ids"][
            "row_support_shared_by_pack_truth_census_ops"]
        for fam in families if fam in summary}
    man["proposed_pack_py_change"] = PROPOSED_PACK_PY_CHANGE
    p = os.path.join(out_dir, "support", "MANIFEST.json")
    with open(p, "w") as fh:
        json.dump(man, fh, indent=1, default=str)
    return p


def write_sha256sums(out_dir, skip=("_work", "SHA256SUMS")):
    """Recursive SHA256SUMS (so ``support/MANIFEST.json`` is covered too)."""
    lines = []
    for root, dirs, files in os.walk(out_dir):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        for f in sorted(files):
            if f in skip:
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, out_dir)
            lines.append(f"{file_sha256(p)}  {rel}")
    p = os.path.join(out_dir, "SHA256SUMS")
    with open(p, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", nargs="+", default=list(FAMILIES),
                    choices=list(FAMILIES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--skip-molly", action="store_true")
    ap.add_argument("--molly-only", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    summary = {}
    if a.molly_only:
        with open(os.path.join(a.out, "V3_BUILD_SUMMARY.json")) as fh:
            summary = json.load(fh)
    else:
        for fam in a.family:
            summary[fam], _ = build(fam, a.out, a.work)
        with open(os.path.join(a.out, "V3_BUILD_SUMMARY.json"), "w") as fh:
            json.dump(summary, fh, indent=1, default=str)

    # correction (6): the molly window study (one shared 2LPT-0 object)
    if not a.skip_molly:
        bmo = A0._bmo()
        ep = bmo._load_ep()
        work = a.work or os.path.join(a.out, "_work")
        mrec, marr = molly_window_study(bmo, ep, work)
        np.savez(os.path.join(a.out, "molly_window_study.npz"), **marr)
        with open(os.path.join(a.out, "MOLLY_WINDOW_STUDY.json"), "w") as fh:
            json.dump(mrec, fh, indent=1, default=str)
        print("\nMOLLY WINDOW: " + mrec["ambiguity"]["verdict"][:70] + " ...")

    # the fail-closed gate, at the row level, on the v3 set
    gates = {}
    for fam in a.family:
        pkp = os.path.join(a.out, f"scanpack_{fam}_b300_v3.npz")
        cen = os.path.join(a.out, f"fp_census_{fam}_v3.npz")
        ops = os.path.join(a.out, f"empirical_ops_{fam}_v3.npz")
        for label, lvl in (("row", ROW_SELECTION_FIELDS),
                           ("full", SUPPORT_FIELDS)):
            key = f"{fam}:pack+census+ops.{label}"
            try:
                gates[key] = check_support_consistency(pkp, cen, ops,
                                                       fields=lvl)
            except SupportContractError as exc:
                gates[key] = dict(status="FAIL", error=str(exc))
            print(f"=== {key} -> {gates[key]['status']} ===")
            if gates[key]["status"] == "PASS":
                print("   support_id", gates[key]["support_id_short"],
                      "planes", len(gates[key]["planes"]),
                      "truth_host_floor",
                      sorted(set(map(str,
                                     gates[key]["truth_host_floor"].values()))))
    with open(os.path.join(a.out, "V3_SUPPORT_GATES.json"), "w") as fh:
        json.dump(gates, fh, indent=1, default=str)

    print("\nMANIFEST ->", write_manifest(a.out, a.family, summary))
    print("SHA256SUMS ->", write_sha256sums(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
