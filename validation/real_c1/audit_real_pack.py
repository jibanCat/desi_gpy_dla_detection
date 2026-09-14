#!/usr/bin/env python
"""audit_real_pack.py — SUPPORT AUDIT of the REAL low-z C1 pack against the v3
contract, and the row-level ``support_id`` for the real pack.

CLASSIFICATION: **VALIDATION / PREPARATION ONLY.**  Nothing here evaluates the
real likelihood, runs a sampler or computes a posterior.  It loads the real
pack, re-derives its ``counts`` selection from the DESI catalogue (a counting
argument), declares the pack's 12-field support under
``validation/absorber_ladder/support/support_contract.py`` and writes a
``.support.json`` sidecar into a NEW directory.  The pack of record is NEVER
touched.

PRIVACY (feedback_real_data_privacy): every count/total this script produces is
a REAL-DATA aggregate.  It is written to the scratch JSON only; the operator
transcribes it to the PRIVATE notes repo.  Nothing is printed that is not
needed for the audit, and nothing goes into a commit message.

Usage:
    python validation/real_c1/audit_real_pack.py --out-dir <real_c1_inputs>
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "validation", "absorber_ladder", "support"))

import support_contract as SC  # noqa: E402

C_KMS = 299792.458
LYA = 1215.67

#: the pack of record (posterior campaign 2026-09-12; the pack the L13 real
#: battery and the C1 pool of record were run on)
REAL_PACK = ("/home/mfho/lowz_clean_work_2026-09-12/posterior_campaign/"
             "packs/C1_pack.npz")
REAL_CAT_DIR = "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1"
REAL_QSOCAT = ("/nfs/turbo/lsa-cavestru/mfho/DESI/loa/"
               "QSO_cat_loa_main_dark_healpix_v2-altbal.fits")

#: the frozen real cut bundle (CDDF_analysis/hbi_mcmc/extract_pack_real.py:
#: COLLAR_KMS + make_cfg + HBIConfig defaults)
COLLAR_KMS = 3300.0
SNR_MIN = 2.0
P_DLA_MIN = 0.99
Z_QSO_WINDOW = (2.0, 4.25)
LAM_RF_MIN = 1025.0
LAM_RF_MAX = 1216.0
QUALITY_CUT = "DLAFLAG==0"
#: the real BAL policy of record (contract v1.1): drop every TARGETID with
#: BI_CIV > 0 in the loa QSO catalogue.
BAL_POLICY_FORM = "real:drop_all_TARGETID_with_BI_CIV>0_in_loa_qsocat"
#: real data has NO truth catalogue; the schema forbids None, so the absence is
#: declared explicitly (a string is a legal support value).
NO_TRUTH_CATALOGUE = "n/a (REAL DATA: no truth catalogue exists)"

MOCK_V3_DIR = ("/scratch/cavestru_root/cavestru0/mfho/"
               "absorber_ladder_2026-09-13/support_v3")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def real_support(*, truth_host_floor=SC.NO_TRUTH_SIDE, collar_kms=COLLAR_KMS,
                 cache=None) -> SC.SupportID:
    """The 12-field support declaration of the REAL C1 data plane."""
    cache = {} if cache is None else cache
    return SC.support_id(
        collar_kms=float(collar_kms),
        snr_min=float(SNR_MIN),
        z_window=(float(Z_QSO_WINDOW[0]), float(Z_QSO_WINDOW[1])),
        p_dla_min=float(P_DLA_MIN),
        lya_only_lam_min=float(LAM_RF_MIN),
        lam_rf_max=float(LAM_RF_MAX),
        z_cut_columns=SC.Z_CUT_ZDLA_ONLY,
        quality_cut=QUALITY_CUT,
        truth_host_floor=truth_host_floor,
        bal_policy=(BAL_POLICY_FORM + ";qso_cat_sha12="
                    + SC.file_sha256(REAL_QSOCAT, cache)[:12]),
        catalogue_id=SC.catalogue_identity(REAL_CAT_DIR, cache=cache),
        truth_catalogue_sha256=NO_TRUTH_CATALOGUE,
    )


def counts_ladder(collar_kms=COLLAR_KMS):
    """Re-derive the real pack's ``counts`` selection from the catalogue.

    Returns ``(counts_no_sentinel, counts_sentinel_filtered, ladder)`` — the
    same shape of counting argument ``build_pack_v3.counts_ladder`` runs on the
    mocks, so the two supports can be compared cut by cut.
    """
    import fitsio
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "_ep_audit", os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc",
                                  "extract_pack.py"))
    ep = ilu.module_from_spec(spec)
    sys.modules["_ep_audit"] = ep
    spec.loader.exec_module(ep)

    catf = sorted(glob.glob(os.path.join(REAL_CAT_DIR, "dlacat*.fits")))
    cat = (np.concatenate([fitsio.read(f, ext=1) for f in catf])
           if len(catf) > 1 else fitsio.read(catf[0], ext=1))
    qso = fitsio.read(REAL_QSOCAT, ext=1, columns=["TARGETID", "BI_CIV"])
    bal = np.unique(qso["TARGETID"][qso["BI_CIV"] > 0].astype(np.int64))

    tid = cat["TARGETID"].astype(np.int64)
    zqc = np.asarray(cat["Z_QSO"], float)
    zd = np.asarray(cat["Z_DLA"], float)
    nhi = np.asarray(cat["NHI"], float)
    snr = np.asarray(cat["SNR_REDSIDE"], float)
    sentinel = ((np.asarray(cat["NHI_ERR"], float) == -1)
                | (np.asarray(cat["Z_DLA_ERR"], float) == -1))

    ladder = [("all catalogue rows", int(len(tid)))]
    m = np.asarray(cat["DLAFLAG"], int) == 0
    ladder.append(("& DLAFLAG == 0", int(m.sum())))
    m &= np.asarray(cat["P_DLA"], float) > P_DLA_MIN
    ladder.append((f"& P_DLA > {P_DLA_MIN}", int(m.sum())))
    m &= snr > SNR_MIN
    ladder.append((f"& SNR_REDSIDE > {SNR_MIN}", int(m.sum())))
    m &= ~np.isin(tid, bal)
    ladder.append(("& NOT BAL (BI_CIV > 0)", int(m.sum())))
    m &= (zqc > Z_QSO_WINDOW[0]) & (zqc < Z_QSO_WINDOW[1])
    ladder.append((f"& z_qso in {Z_QSO_WINDOW} (strict)", int(m.sum())))

    coll = float(collar_kms) / C_KMS
    z_lo = np.maximum(3600.0 / LYA - 1.0,
                      LAM_RF_MIN * (1 + zqc) / LYA - 1.0 + coll)
    z_hi = np.minimum(zqc - coll, LAM_RF_MAX * (1 + zqc) / LYA - 1.0 - coll)
    m &= (zd > z_lo) & (zd < z_hi)
    ladder.append((f"& lambda/z window @collar {collar_kms:.0f} (Z_DLA only)",
                   int(m.sum())))

    m_s = m & ~sentinel
    ladder.append(("& NOT sentinel (NHI_ERR == -1 or Z_DLA_ERR == -1)",
                   int(m_s.sum())))
    c_nos, n_win_nos = ep.bin_counts_cks(nhi[m], zd[m], snr[m])
    c_sen, _ = ep.bin_counts_cks(nhi[m_s], zd[m_s], snr[m_s])
    ladder.append(("binned onto (c, k, s) [no sentinel filter]",
                   int(c_nos.sum())))
    ladder.append(("binned onto (c, k, s) [sentinel filter APPLIED]",
                   int(c_sen.sum())))
    rec = dict(
        ladder=[list(x) for x in ladder],
        n_sentinel_rows_in_catalogue=int(sentinel.sum()),
        n_sentinel_rows_surviving_the_op_cut=int((m & sentinel).sum()),
        n_sentinel_rows_that_land_on_the_grid=int(c_nos.sum() - c_sen.sum()),
        n_bal_targetids=int(len(bal)))
    return c_nos.astype(np.int64), c_sen.astype(np.int64), rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=REAL_PACK)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mock-v3-dir", default=MOCK_V3_DIR)
    ap.add_argument("--skip-catalogue", action="store_true",
                    help="skip the catalogue-side counting argument (fast path)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    cache: dict = {}

    z = np.load(a.pack, allow_pickle=True)
    prov = json.load(open(a.pack[:-4] + ".provenance.json"))
    rec: dict = dict(
        schema="absorber_ladder/real_c1_support_audit/v1",
        pack=os.path.abspath(a.pack),
        pack_sha256=sha256_file(a.pack),
        pack_provenance=prov,
        checked_utc=datetime.datetime.utcnow().isoformat() + "Z")

    # ---- real-mode gate (the same one cc_real_posterior enforces) ---------
    tc = np.asarray(z["truth_counts"])
    rec["real_mode_gate"] = dict(
        provenance_real_data=bool(prov.get("real_data")),
        truth_counts_sentinel=prov.get("truth_counts_sentinel"),
        truth_counts_all_zero=bool(tc.size > 0 and not np.any(tc != 0)),
        status=("PASS" if (prov.get("real_data")
                           and prov.get("truth_counts_sentinel") == "ZEROS_NO_TRUTH"
                           and tc.size > 0 and not np.any(tc != 0)) else "FAIL"))

    # ---- grid / strata equality with the mock v3 packs --------------------
    grid_keys = ("nhat_edges", "ntrue_edges", "zf_edges", "zc_edges",
                 "snr_edges", "kz_to_K", "nhat_masked_bins")
    frozen_keys = ("molly_n_det", "molly_n_tot", "molly_nhi_edges",
                   "molly_snr_edges", "g_grid", "g_occupancy", "t_sigma",
                   "fp_eta_c", "fp_counts", "contract_id", "tp_convention_id",
                   "resp_N_fit_range", "resp_N_ref", "resp_fitcov_diag",
                   "resp_mu_coef", "resp_sig_coef", "resp_sig_floor",
                   "resp_skew_coef", "resp_skew_ramp", "resp_snr_edges",
                   "resp_z_edges", "adopted_carrier_mu", "adopted_carrier_sig",
                   "adopted_carrier_skew", "adopted_carrier_shared3",
                   "adopted_phi_ref", "adopted_resp_fit_range",
                   "adopted_resp_mu_coef", "adopted_resp_sig_coef",
                   "adopted_resp_skew_coef", "adopted_resp_version")
    grids = {}
    for fam in ("2lpt0", "london0", "saclay0"):
        mp = os.path.join(a.mock_v3_dir, f"scanpack_{fam}_b300_v3.npz")
        if not os.path.exists(mp):
            grids[fam] = dict(status="UNVERIFIED", reason="mock v3 pack absent")
            continue
        zm = np.load(mp, allow_pickle=True)
        grids[fam] = dict(
            mock_pack=mp, mock_pack_sha256=sha256_file(mp),
            grid_identical={k: bool(np.array_equal(np.asarray(z[k]),
                                                   np.asarray(zm[k])))
                            for k in grid_keys},
            frozen_calibration_identical={
                k: bool(np.array_equal(np.asarray(z[k]), np.asarray(zm[k])))
                for k in frozen_keys},
            survey_plane_differs={
                k: bool(not np.array_equal(np.asarray(z[k]), np.asarray(zm[k])))
                for k in ("counts", "dX", "dX_coarse_committed", "fp_E_alloc",
                          "fp_ell_eff", "fp_w_sightline_ratio")})
    rec["grid_equality_vs_mock_v3"] = grids

    # ---- fp_counts: the shared loa-0 calibration block --------------------
    fpc = np.asarray(z["fp_counts"])
    fpc_std = {}
    for fam in ("2lpt0", "london0", "saclay0"):
        p = os.path.join(a.mock_v3_dir, f"fp_counts_{fam}_v3.npz")
        if os.path.exists(p):
            b = np.asarray(np.load(p, allow_pickle=True)["fp_counts"])
            fpc_std[fam] = dict(file=p, array_sha256=array_sha256(b),
                                identical_to_real_pack=bool(
                                    np.array_equal(b, fpc)))
    rec["fp_counts_block"] = dict(
        shape=list(fpc.shape), dtype=str(fpc.dtype), total=int(fpc.sum()),
        array_sha256=array_sha256(fpc), v3_standalone=fpc_std,
        R_015_disclosure=(
            "the loa-0 fp_counts block carries 89 events on its OWN support "
            "(no proximity collar, no z_qso admission window, no BAL veto, no "
            "DLAFLAG cut); rebuilt under the survey collar c = 3300 km/s it is "
            "87 events (-2.2 %). R-015, disclosure pending PI. The block is "
            "copied UNCHANGED into the real pack, exactly as into the mock v3 "
            "packs, so the real run inherits the same disclosed defect and no "
            "new one."))

    # ---- internal collar/consistency of the real data plane ---------------
    dX = np.asarray(z["dX"], float)
    fpE = np.asarray(z["fp_E_alloc"], float)
    col = dX.sum(axis=0)
    exp = np.zeros_like(dX)
    nz = col > 0
    exp[:, nz] = dX[:, nz] / col[nz]
    counts = np.asarray(z["counts"])
    dead = ~(dX.sum(axis=0) > 0)
    rec["internal_consistency"] = dict(
        counts_shape=list(counts.shape), dX_shape=list(dX.shape),
        counts_finite=bool(np.all(np.isfinite(counts))),
        dX_finite=bool(np.all(np.isfinite(dX))),
        dX_nonnegative=bool(np.all(dX >= 0)),
        fp_E_alloc_reproduced_from_dX=bool(
            np.allclose(fpE, exp, rtol=0, atol=1e-15)),
        fp_E_alloc_max_abs_dev=float(np.abs(fpE - exp).max()),
        n_dead_strata=int(dead.sum()),
        counts_zero_on_dead_strata=bool(counts[:, :, dead].sum() == 0),
        n_live_cells=int((dX > 0).sum()),
        note=("counts, dX and fp_E_alloc are produced by ONE call to "
              "extract_pack_real.build_data_plane under ONE collar_kms "
              "variable; fp_E_alloc is dX renormalised per fine-z column, so "
              "reproducing it from dX proves the two planes are the same "
              "object (collar included)."))

    # ---- the catalogue-side counting argument (real-data aggregates) ------
    if a.skip_catalogue:
        rec["counting_argument"] = dict(status="SKIPPED")
    else:
        c_nos, c_sen, lad = counts_ladder(COLLAR_KMS)
        rec["counting_argument"] = dict(
            collar_kms=COLLAR_KMS, **lad,
            pack_counts_total=int(counts.sum()),
            v3_recipe_total_sentinel_filtered=int(c_sen.sum()),
            no_sentinel_total=int(c_nos.sum()),
            pack_equals_no_sentinel_recipe=bool(np.array_equal(
                np.asarray(counts, np.int64), c_nos)),
            pack_equals_v3_sentinel_filtered_recipe=bool(np.array_equal(
                np.asarray(counts, np.int64), c_sen)),
            n_cells_differing_vs_v3_recipe=int(np.count_nonzero(
                np.asarray(counts, np.int64) - c_sen)),
            verdict=("the real pack's counts reproduce the collar-3300, "
                     "Z_DLA-only cut bundle; whether the v3 sentinel filter is "
                     "ALSO applied is reported by the two booleans above"))

    # ---- the support declaration + sidecar --------------------------------
    sup_data = real_support(truth_host_floor=SC.NO_TRUTH_SIDE, cache=cache)
    sup_3000 = real_support(truth_host_floor=SC.NO_TRUTH_SIDE,
                            collar_kms=3000.0, cache=cache)
    rec["support"] = dict(
        support_id=sup_data.sha256, support_id_short=sup_data.short,
        row_support_id=sup_data.row_sha256,
        canonical=sup_data.canonical,
        fields={k: SC._json_safe(sup_data.fields[k]) for k in SC.SUPPORT_FIELDS},
        mutation_control_collar_3000_support_id=sup_3000.sha256,
        mutation_control_differs=bool(sup_3000.sha256 != sup_data.sha256))
    mock_rows = {}
    for fam in ("2lpt0", "london0", "saclay0"):
        sp = os.path.join(a.mock_v3_dir, f"scanpack_{fam}_b300_v3.support.json")
        if os.path.exists(sp):
            st = json.load(open(sp))
            mock_rows[fam] = dict(row_support_id=st.get("row_support_id"),
                                  support_id=st.get("support_id"))
    rec["support"]["mock_v3_row_supports"] = mock_rows
    rec["support"]["note"] = (
        "the REAL row support is EXPECTED to differ from the mocks' — a "
        "different detection catalogue, a different BAL policy and no truth "
        "catalogue. What must agree is the GEOMETRY of the cut bundle: "
        "collar 3300, Z_DLA-only z cut, SNR > 2, P_DLA > 0.99, DLAFLAG == 0, "
        "z_qso in (2.0, 4.25), lambda_rf in [1025, 1216].")

    # A READ-ONLY working copy of the pack of record is staged next to the
    # stamp, because support_contract.read_stamp resolves the sidecar from the
    # pack's own path.  The pack of record is opened read-only and never
    # modified; the copy is verified byte-for-byte by sha256.
    import shutil
    out_pack = os.path.join(a.out_dir, os.path.basename(a.pack))
    if (not os.path.exists(out_pack)
            or sha256_file(out_pack) != rec["pack_sha256"]):
        shutil.copy2(a.pack, out_pack)
    copy_sha = sha256_file(out_pack)
    if copy_sha != rec["pack_sha256"]:
        raise SystemExit("the staged working copy does not match the pack of "
                         "record by sha256 — fail-closed")
    for ext in (".provenance.json", ".contract_guards.json"):
        src = a.pack[:-4] + ext
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(a.out_dir,
                                           os.path.basename(a.pack)[:-4] + ext))
    rec["working_copy"] = dict(path=out_pack, sha256=copy_sha,
                               identical_to_pack_of_record=True)
    SC.stamp(out_pack, sup_data, hash_product=True, extra=dict(
        note=("support stamp for the REAL low-z C1 pack of record. "
              "'product' is a byte-identical READ-ONLY working copy staged "
              "beside this stamp; 'source_pack' / 'source_pack_sha256' "
              "identify the file of record, which was not modified."),
        source_pack=os.path.abspath(a.pack),
        source_pack_sha256=rec["pack_sha256"],
        planes={"counts": dict(sup_data.fields), "dX": dict(sup_data.fields),
                "fp_E_alloc": dict(sup_data.fields)},
        excluded_planes={
            "truth_counts": "the all-zero REAL-DATA sentinel; no truth side "
                            "exists, so it carries no support",
            "fp_counts": "the loa-0 calibration block on its OWN support "
                         "(see fp_counts_<fam>_v3.support.json; R-015)",
            "molly_n_det/molly_n_tot/g_grid/resp_*/adopted_*":
                "frozen 2LPT-0 calibration products, not row selections"}))
    rec["support_sidecar"] = SC.stamp_path(out_pack)

    with open(os.path.join(a.out_dir, "REAL_PACK_SUPPORT_AUDIT.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print("support_id      ", sup_data.short)
    print("row_support_id  ", sup_data.row_sha256[:16])
    print("real-mode gate  ", rec["real_mode_gate"]["status"])
    print("sidecar         ", rec["support_sidecar"])
    print("audit JSON      ", os.path.join(a.out_dir,
                                           "REAL_PACK_SUPPORT_AUDIT.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
