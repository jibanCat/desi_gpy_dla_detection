#!/usr/bin/env python
"""sample_accounting.py — R-039 (a)–(f) and R-036: sample and absorption-path accounting
of both Paper-1 arms, from frozen artifacts and the COMMITTED window geometry.

Paper-1 request package 2026-08-28. These are CATALOG-ACCOUNTING quantities (how many
sightlines, how much path, how many accepted candidates above a reported N-hat); they
are not the HBI estimand and the manuscript uses them only to expose counting leverage.

Low-z arm (frozen real pack v2 + its provenance sidecar): the sums the paper lane asked
for are read from the pack's integer `counts` grid and its `dX` plane; the semantics of
`n_sl`, `n_op_rows`, `counts_in_window`, `n_bal_excluded` are those of
extract_pack_real.build_data_plane (the producer), restated here and, where cheap,
re-derived from the same population inputs (the archive QSO population + the BAL list).

High-z arm: the sightline set and the per-sightline windows are REBUILT with the same
geometry as cddf_catalog_hbi.build_pathlength (SNR_REDSIDE > 2 strict, z_QSO in
(4.25, 7.0) strict, BI_CIV>0 dropped, lam_rf in [1025, 1216] with a 3000 km/s collar on
both edges, 3600 A floor, Omega_m = 0.279), so that the two recorded counts
(`n_op_sl` in the RATIFIED artifact; `op_sightlines` in tail_ge5_audit.json) can be
DEFINED rather than guessed, and the path in [3.8, 5.0) and its path-weighted effective
redshift (R-036) are recomputed from the same windows. The accepted-candidate counts of
tail_ge5_audit.json are re-derived from the catalogue rows under the recorded contract.

Nothing here modifies a frozen artifact. Real-data VALUES never enter this file.

Usage (explicit inputs; no frozen-path defaults):
  python -m CDDF_analysis.hbi_mcmc.sample_accounting \
      --pack <real pack v2 npz> --pack-provenance <...provenance.json> \
      --selection-contract <...selection_contract.json> --paper-dndx-npz <fig_hbi_dndx.data.npz> \
      --archive-npy <src_archive_catalog.npy> --real-qsocat <QSO catalogue fits> \
      --hz-cat <dir of dlacat-*.fits> --hz-mockdir <dir with snr_cat/zcat/bal_cat> \
      --bh-artifact <RATIFIED json> --tail-audit <tail_ge5_audit.json> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

LYA = 1215.67
C_KMS = 299792.458
OMEGA_M = 0.279
HZ_ZQSO = (4.25, 7.0)
HZ_LAM_RF = (1025.0, 1216.0)
HZ_COLLAR_KMS = 3000.0
HZ_BIN = (3.8, 5.0)
HZ_SUBBINS = [3.8, 4.25, 4.5, 5.0]
LOWZ_ZQSO = (2.0, 4.25)
LOWZ_COLLAR_KMS = 3300.0
SNR_MIN = 2.0
P_DLA_MIN = 0.99
LOWZ_BINS = [("B1", 2.15, 2.35), ("B2", 2.35, 2.56), ("B3", 2.56, 2.96), ("B4", 2.96, 3.40), ("B5", 3.40, 3.80)]


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here).decode().strip()
    except Exception:
        return "unknown"


def windows(zq, lam_rf_min, lam_rf_max, collar_kms):
    """Per-sightline absorber-z window: identical geometry to build_pathlength /
    extract_pack_real.build_data_plane (3600 A floor, rest-frame band, symmetric collar)."""
    coll = collar_kms / C_KMS
    zlo = np.maximum(3600.0 / LYA - 1.0, lam_rf_min * (1 + zq) / LYA - 1.0 + coll)
    zhi = np.minimum(zq - coll, lam_rf_max * (1 + zq) / LYA - 1.0 - coll)
    return zlo, zhi


# ---------------------------------------------------------------------------------------
def lowz(a):
    from CDDF_analysis.hbi_mcmc.extract_pack_real import REAL_QSOCAT  # noqa: F401 (documented producer)
    import fitsio
    P = np.load(a.pack, allow_pickle=True)
    prov = json.load(open(a.pack_provenance))
    sel = json.load(open(a.selection_contract))
    counts = np.asarray(P["counts"], int)
    nh = np.asarray(P["nhat_edges"], float)
    zf = np.asarray(P["zf_edges"], float)
    dX = np.asarray(P["dX"], float)
    out = {"inputs": {"pack": {"path": a.pack, "sha256": _sha(a.pack)},
                      "provenance": {"path": a.pack_provenance, "sha256": _sha(a.pack_provenance)},
                      "selection_contract": {"path": a.selection_contract, "sha256": _sha(a.selection_contract)}},
           "provenance_values_as_recorded": {k: prov[k] for k in ("n_sl", "n_op_rows", "counts_in_window", "n_bal_excluded", "bal_policy", "catalog", "catalog_sha256", "qso_population_source", "code_commit")},
           "contract": {"collar_kms": LOWZ_COLLAR_KMS, "z_qso_window_strict": list(LOWZ_ZQSO), "SNR_REDSIDE_min_strict": SNR_MIN, "P_DLA_min_strict": P_DLA_MIN,
                        "DLAFLAG": 0, "BAL_policy": prov["bal_policy"], "lam_rf": [1025.0, 1216.0], "floor_A": 3600.0,
                        "z_support": [float(zf[0]), float(zf[-1])], "nhat_grid": nh.tolist(), "nhat_masked_bins": np.asarray(P["nhat_masked_bins"]).tolist(),
                        "selection_contract_sidecar_version": sel.get("version", sel.get("schema"))}}
    # (d) semantics of n_sl / n_bal_excluded, restated from the producer and re-derived
    sem = {
        "n_sl": "number of quasar SIGHTLINES in the pack's path plane: archive QSO population (qso_population_source) with finite SNR_REDSIDE, SNR_REDSIDE > 2 (strict), 2.0 < z_QSO < 4.25 (strict), TARGETID NOT in the BAL list (BI_CIV > 0), and a non-empty Lyα window (3600 A floor, lam_rf 1025-1216 A, symmetric 3300 km/s collar). It is POST-BAL-exclusion. The pack's dX (per fine-z cell x SNR stratum) is integrated over exactly these sightlines' windows (build_M_b), so it IS the likelihood's path support: the hierarchical measurement runs on these sightlines.",
        "n_bal_excluded": "the SIZE of the BAL exclusion list = number of unique TARGETIDs with BI_CIV > 0 in the WHOLE real QSO catalogue (all redshifts, all SNR), NOT the number removed from the n_sl population. The number actually removed from the low-z operating population is re-derived below (n_bal_removed_from_lowz_population).",
        "n_op_rows": "catalogue DETECTION rows passing the contract row mask (z_QSO window, z_DLA inside the sightline window at the 3300 km/s collar, non-BAL) AND DLAFLAG == 0 AND P_DLA > 0.99 AND SNR_REDSIDE > 2 — at ANY reported N-hat (rows below the 19.5 grid floor and above 22.4 included).",
        "counts_in_window": "the subset of n_op_rows inside the (N-hat, z, SNR) data-plane grid: 19.5 <= N-hat < 22.4 and 2.0 <= z_DLA < 3.5 — equal to counts.sum(); the two lowest N-hat bins [19.5,19.7) are MASKED in the likelihood (nuisance support), so the detections the likelihood actually fits are counts_in_window minus the masked-bin counts.",
    }
    out["semantics"] = sem
    tot = int(counts.sum())
    masked = np.asarray(P["nhat_masked_bins"], bool)
    out["derived_from_pack"] = {
        "counts_sum_equals_counts_in_window": bool(tot == prov["counts_in_window"]),
        "counts_in_masked_bins_[19.5,19.7)": int(counts[masked].sum()),
        "counts_fitted_by_likelihood_(unmasked)": int(counts[~masked].sum()),
        "detections_nhat_ge_20p3": int(counts[nh[:-1] >= 20.3 - 1e-9].sum()),
        "detections_nhat_ge_20p0": int(counts[nh[:-1] >= 20.0 - 1e-9].sum()),
        "edge_20p3_on_grid": bool(np.any(np.isclose(nh, 20.3))), "edge_20p0_on_grid": bool(np.any(np.isclose(nh, 20.0))),
        "dX_total_[2.0,3.5]": float(dX.sum()),
        "dX_coarse_committed_sum": float(np.asarray(P["dX_coarse_committed"], float).sum()),
        "dX_per_fine_cell": dX.sum(axis=1).tolist(), "zf_edges": zf.tolist(),
        "detections_nhat_ge_20p3_per_fine_cell": counts[nh[:-1] >= 20.3 - 1e-9].sum(axis=(0, 2)).tolist(),
        "detections_nhat_ge_20p0_per_fine_cell": counts[nh[:-1] >= 20.0 - 1e-9].sum(axis=(0, 2)).tolist()}
    # (f) per reporting bin: path by overlap (continuous) exists; integer counts do not
    per = []
    for name, lo, hi in LOWZ_BINS:
        ov = np.clip(np.minimum(zf[1:], hi) - np.maximum(zf[:-1], lo), 0, None) / np.diff(zf)
        aligned = bool(np.any(np.isclose(zf, lo)) and np.any(np.isclose(zf, hi)))
        per.append({"bin": name, "z": [lo, hi], "edges_on_native_grid": aligned,
                    "dX_by_overlap": float((dX.sum(axis=1) * ov).sum()),
                    "integer_count_defined": aligned,
                    "note": ("no exact integer count exists: the bin edges are not on the 0.1-wide native z grid of the committed counts, and the "
                             "per-cell counts are integers on that grid only; a fractional-overlap 'count' would be a paper-lane reduction, not a catalogue count")})
    out["per_reporting_bin"] = per
    if a.paper_dndx_npz:
        npz = np.load(a.paper_dndx_npz, allow_pickle=True)
        out["per_reporting_bin_dX_closure_vs_paper_npz_max_abs"] = float(np.max(np.abs(np.array([p["dX_by_overlap"] for p in per]) - np.asarray(npz["dX_per_bin"], float)[:5])))
    # re-derive n_sl and the BAL removal from the population inputs
    arch = np.load(a.archive_npy)
    qso = fitsio.read(a.real_qsocat, ext=1, columns=["TARGETID", "BI_CIV"])
    bal_tids = np.unique(qso["TARGETID"][qso["BI_CIV"] > 0].astype(np.int64))
    tid = arch["TARGETID"].astype(np.int64)
    snr = arch["RED_SNR"].astype(float)
    zq = arch["Z"].astype(float)
    ok = np.isfinite(snr) & (snr > SNR_MIN) & (zq > LOWZ_ZQSO[0]) & (zq < LOWZ_ZQSO[1])
    zlo, zhi = windows(zq, 1025.0, 1216.0, LOWZ_COLLAR_KMS)
    okw = ok & np.isfinite(zlo) & np.isfinite(zhi) & (zhi > zlo)
    isbal = np.isin(tid, bal_tids)
    out["rederived_population"] = {
        "archive_rows": int(tid.size), "n_bal_list_(whole catalogue)": int(bal_tids.size),
        "n_sl_pre_bal_exclusion": int(okw.sum()), "n_bal_removed_from_lowz_population": int((okw & isbal).sum()),
        "n_sl_post_bal_exclusion": int((okw & ~isbal).sum()),
        "matches_provenance_n_sl": bool(int((okw & ~isbal).sum()) == prov["n_sl"]),
        "matches_provenance_n_bal_excluded": bool(int(bal_tids.size) == prov["n_bal_excluded"]),
        "inputs": {"archive_npy": {"path": a.archive_npy, "sha256": _sha(a.archive_npy)}, "real_qsocat": {"path": a.real_qsocat, "sha256": _sha(a.real_qsocat)}}}
    return out


# ---------------------------------------------------------------------------------------
def highz(a):
    import fitsio
    from CDDF_analysis.cddf_mock import AbsorptionDistance, total_DeltaX_in_zbins, path_length_int
    from CDDF_analysis.hbi.cddf_catalog_hbi import build_per_qso_snr
    lookup = build_per_qso_snr(a.hz_cat, snr_cat_path=None, zcat_path=None, mockdir=a.hz_mockdir, restrict_to_processed=False)
    bal = fitsio.read(os.path.join(a.hz_mockdir, "bal_cat.fits"), ext=1, columns=["TARGETID"])
    bal_tids = set(int(t) for t in bal["TARGETID"])
    tids, zqs, snrs = [], [], []
    n_pop = 0
    for t, (snr, zq) in lookup.items():
        n_pop += 1
        # EXACT build_pathlength semantics: `snr <= snr_min` skips; a NaN SNR_REDSIDE therefore
        # PASSES this test (NaN <= 2 is False) and is counted in n_sl_used.
        if (snr <= SNR_MIN) or not (HZ_ZQSO[0] < zq < HZ_ZQSO[1]) or int(t) in bal_tids:
            continue
        tids.append(int(t)); zqs.append(float(zq)); snrs.append(float(snr))
    tids, zq, snr = np.asarray(tids, np.int64), np.asarray(zqs, float), np.asarray(snrs, float)
    zlo, zhi = windows(zq, HZ_LAM_RF[0], HZ_LAM_RF[1], HZ_COLLAR_KMS)
    n_pass_cuts = int(zq.size)                                     # build_pathlength's n_sl_used
    ok = np.isfinite(zlo) & np.isfinite(zhi) & (zhi > zlo)
    n_valid_window = int(ok.sum())
    nan_snr = ~np.isfinite(snr)
    n_nan_snr = int(nan_snr.sum())
    nan_rows = [{"z_qso": float(zq[i]), "window_lo": float(zlo[i]), "window_hi": float(zhi[i]), "SNR_REDSIDE": None,
                 "window_entirely_above_5p0": bool(zlo[i] >= HZ_BIN[1])} for i in np.where(nan_snr)[0]]
    empty = [{"z_qso": float(zq[i]), "window_lo": float(zlo[i]), "window_hi": float(zhi[i])} for i in np.where(~ok)[0]]
    # the in-bin / finite-SNR population (the tail audit's op_sightlines and the row-level op mask both require SNR > 2)
    fin = ok & np.isfinite(snr) & (snr > SNR_MIN)
    n_finite_snr = int(fin.sum())
    tids, zq, snr, zlo, zhi = tids[fin], zq[fin], snr[fin], zlo[fin], zhi[fin]
    Xcalc = AbsorptionDistance(zmax=float(zhi.max()), Omega_m=OMEGA_M)
    X_sub = total_DeltaX_in_zbins(np.asarray(HZ_SUBBINS, float), zlo, zhi, Xcalc)
    X_bin = float(X_sub.sum())
    X_tail = float(total_DeltaX_in_zbins(np.array([HZ_BIN[1], float(zhi.max()) + 1e-9]), zlo, zhi, Xcalc)[0])
    inbin = (np.minimum(zhi, HZ_BIN[1]) > np.maximum(zlo, HZ_BIN[0]))
    n_inbin = int(inbin.sum())
    not_inbin = ~inbin
    # path-weighted effective redshift over the reported bin (R-036)
    zg = Xcalc.zgrid
    integ = path_length_int(zg, Omega_m=OMEGA_M)
    dz = np.diff(zg)
    ZX = np.concatenate([[0.0], np.cumsum(0.5 * (zg[:-1] * integ[:-1] + zg[1:] * integ[1:]) * dz)])
    def _zx(z):
        return np.interp(z, zg, ZX)
    lo_c, hi_c = np.maximum(zlo, HZ_BIN[0]), np.minimum(zhi, HZ_BIN[1])
    m = hi_c > lo_c
    zeff = float((_zx(hi_c[m]) - _zx(lo_c[m])).sum() / (Xcalc.X(hi_c[m]) - Xcalc.X(lo_c[m])).sum())
    lo_t, hi_t = np.maximum(zlo, 4.45), np.minimum(zhi, HZ_BIN[1])
    mt = hi_t > lo_t
    zeff_ge445 = float((_zx(hi_t[mt]) - _zx(lo_t[mt])).sum() / (Xcalc.X(hi_t[mt]) - Xcalc.X(lo_t[mt])).sum())
    X_ge445 = float((Xcalc.X(hi_t[mt]) - Xcalc.X(lo_t[mt])).sum())
    # accepted candidates under the recorded contract, from the catalogue rows
    rows = []
    for fn in sorted(glob.glob(os.path.join(a.hz_cat, "dlacat-*.fits"))):
        rows.append(fitsio.read(fn, ext=1))
    cat = np.concatenate(rows)
    cols = cat.dtype.names
    snrcol = "SNR_REDSIDE" if "SNR_REDSIDE" in cols else "S2N_RED"
    ctid = cat["TARGETID"].astype(np.int64)
    idx = {t: i for i, t in enumerate(tids.tolist())}
    sl = np.array([idx.get(int(t), -1) for t in ctid])
    insl = sl >= 0
    zd = np.asarray(cat["Z_DLA"], float)
    nhi = np.asarray(cat["NHI"], float)
    op = insl & (np.asarray(cat["DLAFLAG"], int) == 0) & (np.asarray(cat["P_DLA"], float) > P_DLA_MIN) & (np.asarray(cat[snrcol], float) > SNR_MIN)
    win = np.zeros_like(op)
    win[insl] = (zd[insl] > zlo[sl[insl]]) & (zd[insl] < zhi[sl[insl]])
    op &= win
    inb = op & (zd >= HZ_BIN[0]) & (zd < HZ_BIN[1])
    tail = op & (zd >= HZ_BIN[1])
    counts = {thr: {"n_in_[3.8,5.0)": int((inb & (nhi >= thr)).sum()), "n_tail_ge5": int((tail & (nhi >= thr)).sum()),
                    "raw_ratio_in_bin": float((inb & (nhi >= thr)).sum() / X_bin)} for thr in (20.0, 20.3)}
    bh = json.load(open(a.bh_artifact))
    ta = json.load(open(a.tail_audit))
    md = bh["metadata"]
    out = {
        "inputs": {"hz_cat": a.hz_cat, "hz_mockdir": a.hz_mockdir,
                   "bh_artifact": {"path": a.bh_artifact, "sha256": _sha(a.bh_artifact)},
                   "tail_audit": {"path": a.tail_audit, "sha256": _sha(a.tail_audit)},
                   "catalog_files": [{"path": fn, "sha256": _sha(fn)} for fn in sorted(glob.glob(os.path.join(a.hz_cat, "dlacat-*.fits")))],
                   "mockdir_files": [{"path": os.path.join(a.hz_mockdir, f), "sha256": _sha(os.path.join(a.hz_mockdir, f))} for f in ("snr_cat.fits", "zcat.fits", "bal_cat.fits")]},
        "contract": {"sample": "P1_PRIMARY_LYA", "z_qso_window_strict": list(HZ_ZQSO), "SNR_REDSIDE_min_strict": SNR_MIN, "P_DLA_min_strict": P_DLA_MIN, "DLAFLAG": 0,
                     "BAL_policy": "BI_CIV>0 dropped (mockdir/bal_cat.fits)", "lam_rf": list(HZ_LAM_RF), "collar_kms": HZ_COLLAR_KMS, "collar_type": "constant Delta z (3000 km/s / c) on both window edges",
                     "floor_A": 3600.0, "Omega_m": OMEGA_M, "reported_bin": list(HZ_BIN), "sub_bins_(construction only, PI #44)": HZ_SUBBINS,
                     "note_collar_differs_from_lowz": "low-z HBI pack uses 3300 km/s; the BH arm 3000 km/s (documented, immaterial: NEARQSO_STRIP_SENSITIVITY +0.07 %)"},
        "recorded": {"n_op_sl_(RATIFIED metadata)": md["n_op_sl"], "n_op_detections_(RATIFIED metadata)": md["n_op_detections"],
                     "loa0_n_sl_prod_(RATIFIED calibration.loa0)": md["calibration"]["loa0"]["n_sl_prod"],
                     "op_sightlines_(tail_ge5_audit)": ta["op_sightlines"], "dX_bh_(tail_ge5_audit)": ta["dX_bh"], "dX_tail_(tail_ge5_audit)": ta["dX_tail"],
                     "tail_audit_counts": ta["results"]},
        "rederived": {"qso_population_rows_in_lookup": int(n_pop),
                      "n_pass_cuts_(SNR>2, 4.25<z_QSO<7.0, non-BAL; = build_pathlength n_sl_used, window validity NOT yet applied)": n_pass_cuts,
                      "n_with_non_empty_window": n_valid_window,
                      "n_with_NaN_SNR_REDSIDE_(pass the <= test, carry NO detections: the row-level op mask requires SNR > 2)": n_nan_snr,
                      "NaN_SNR_sightlines": nan_rows,
                      "sightlines_with_EMPTY_window": empty,
                      "n_finite_SNR_gt2_with_non_empty_window": n_finite_snr,
                      "n_sightlines_with_path_inside_[3.8,5.0)": n_inbin,
                      "n_sightlines_in_arm_with_NO_path_inside_[3.8,5.0)": int(not_inbin.sum()),
                      "those_sightlines": [{"z_qso": float(zq[i]), "window": [float(zlo[i]), float(zhi[i])], "SNR_REDSIDE": float(snr[i])} for i in np.where(not_inbin)[0]],
                      "dX_[3.8,5.0)": X_bin, "dX_sub_bins": X_sub.tolist(), "path_fraction_sub_bins": (X_sub / X_bin).tolist(),
                      "dX_tail_ge5": X_tail, "tail_path_fraction_of_(bin+tail)": float(X_tail / (X_bin + X_tail)),
                      "z_eff_path_weighted_[3.8,5.0)": zeff, "z_eff_path_weighted_[4.45,5.0)": zeff_ge445, "dX_[4.45,5.0)": X_ge445,
                      "path_fraction_inside_[4.45,5.0)": float(X_ge445 / X_bin),
                      "z_qso_median_of_arm": float(np.median(zq)), "z_qso_max_of_arm": float(zq.max()),
                      "accepted_candidates_under_contract": counts, "snr_column_used": snrcol,
                      "n_op_rows_in_window_any_nhat": int(op.sum()), "n_op_rows_in_[3.8,5.0)_any_nhat": int(inb.sum())},
        "matches": {}}
    r = out["rederived"]
    out["matches"] = {"n_pass_cuts == RATIFIED n_op_sl": bool(n_pass_cuts == md["n_op_sl"]),
                      "n_finite_SNR_gt2 == tail_audit op_sightlines": bool(n_finite_snr == ta["op_sightlines"]),
                      "n_sightlines_with_path_inside_bin == tail_audit op_sightlines": bool(n_inbin == ta["op_sightlines"]),
                      "dX_bin_rel_diff_vs_tail_audit": float(X_bin / ta["dX_bh"] - 1.0), "dX_tail_rel_diff_vs_tail_audit": float(X_tail / ta["dX_tail"] - 1.0),
                      "n_ge20p3_in_bin == tail_audit": bool(counts[20.3]["n_in_[3.8,5.0)"] == ta["results"]["ge20.3"]["n_bh"]),
                      "n_ge20p0_in_bin == tail_audit": bool(counts[20.0]["n_in_[3.8,5.0)"] == ta["results"]["ge20.0"]["n_bh"]),
                      "n_tail_ge20p3 == tail_audit": bool(counts[20.3]["n_tail_ge5"] == ta["results"]["ge20.3"]["n_tail"]),
                      "n_tail_ge20p0 == tail_audit": bool(counts[20.0]["n_tail_ge5"] == ta["results"]["ge20.0"]["n_tail"]),
                      "n_op_rows_any_nhat vs RATIFIED n_op_detections": [int(op.sum()), md["n_op_detections"]]}
    out["semantics"] = {
        "n_op_sl": "the high-z sample as build_pathlength counts it (n_sl_used, taken BEFORE the window-validity mask): quasars with 4.25 < z_QSO < 7.0, not BAL, and NOT (SNR_REDSIDE <= 2). Because that test is written as `snr <= snr_min`, a quasar whose SNR_REDSIDE is NaN passes it. It is the number the loa-0 FP background is volume-scaled to (loa0 n_sl_prod = n_op_sl).",
        "op_sightlines": "the quasars with a FINITE SNR_REDSIDE > 2 (the same strict test the row-level detection mask applies), 4.25 < z_QSO < 7.0, not BAL, and a non-empty window. Every one of them carries path inside [3.8, 5.0). This is the population that carries the bin's path dX_bh and therefore the LIKELIHOOD SUPPORT of the reported one-bin measurement, and the population on which the accepted candidates are counted.",
        "the difference": "quasars in the z_QSO window with NaN SNR_REDSIDE (see NaN_SNR_sightlines: their z_QSO and windows). They pass build_pathlength's sightline test but carry NO accepted candidate (the detection mask requires SNR > 2), and their Lyα windows lie ENTIRELY above z_abs = 5.0 (z_QSO > 6.1: the 1025 A edge of the window is already above 5.0), so they contribute ZERO path to [3.8, 5.0) and zero to any reported quantity; they enter only the loa-0 FP volume scale through n_sl_prod (the relative difference of the two counts is the shift of the FP-background normalisation, and the effect on dN/dX is bounded by that fraction times the FP share of the >=20.3 detections; disclosed, not corrected).",
        "which count goes with the published line density": "op_sightlines (the in-bin population) with dX_bh; n_op_sl is the arm's sample size. Both are legitimate statements with different subjects; the manuscript should not use n_op_sl as 'sightlines supporting the [3.8,5.0) measurement'.",
        "dX carrier": "dX_bh in tail_ge5_audit.json (role record) is the path of the reported bin; the RATIFIED artifact does not store X_tot (it feeds the estimator in memory). This product re-derives it from the committed geometry and is the citable machine-readable carrier together with the sidecar.",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--pack-provenance", required=True)
    ap.add_argument("--selection-contract", required=True)
    ap.add_argument("--paper-dndx-npz", default=None)
    ap.add_argument("--archive-npy", required=True)
    ap.add_argument("--real-qsocat", required=True)
    ap.add_argument("--hz-cat", required=True)
    ap.add_argument("--hz-mockdir", required=True)
    ap.add_argument("--bh-artifact", required=True)
    ap.add_argument("--tail-audit", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out = {"role": "R-039 / R-036 sample and absorption-path accounting of both Paper-1 arms — CATALOG-ACCOUNTING quantities (not the HBI estimand); sits BESIDE the frozen products, supersedes nothing",
           "status": "publication-ready as catalogue accounting (each value is a read-through or an exact re-derivation of a frozen artifact under the recorded contract; PI quotability per number still applies)",
           "written_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
           "generator": {"module": "CDDF_analysis/hbi_mcmc/sample_accounting.py", "commit": _git_commit(), "argv": sys.argv, "python": sys.version.split()[0], "numpy": np.__version__, "conda_env": os.environ.get("CONDA_DEFAULT_ENV")},
           "lowz": lowz(a), "highz": highz(a)}
    jp = os.path.join(a.out_dir, "R039_R036_sample_accounting.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(a.out_dir, "SHA256SUMS"), "w") as fh:
        fh.write(f"{_sha(jp)}  {os.path.basename(jp)}\n")
    print(json.dumps(out["highz"]["matches"], indent=1))
    print(json.dumps(out["lowz"]["rederived_population"], indent=1)[:800])
    print("wrote", jp, _sha(jp))


if __name__ == "__main__":
    main()
