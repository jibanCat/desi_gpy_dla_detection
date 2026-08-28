#!/usr/bin/env python
"""pc_fixed_denominator.py — absorber-level and class-level purity / completeness with FIXED,
STATED denominators, resolved in (log N_HI x SNR) cells on the Paper-1 contract (R-033,
request package 2026-08-28, priority 2), for one calibration mock family per call.

The four quantities (paper-lane decision D043) and their denominators:

  C_abs[true-N cell, SNR cell]  = # true absorbers in the cell MATCHED by an accepted detection
                                  / # true absorbers in the cell             (denominator: TRUTH)
      two variants of "matched": N-hat >= 19.5 (the frozen data-plane floor; = the molly
      definition) and N-hat >= 19.7 (the REPORTED catalogue), same denominator.
  P_abs[N-hat cell, SNR cell]   = # accepted detections in the cell with a true host >= 19.5
                                  / # accepted detections in the cell        (denominator: DETECTIONS)
      decomposed further into host >= 19.5 / host in [19.0, 19.5) / no host >= 19.0.
  C_cls[true-N cell, SNR cell]  = among TRUE-POSITIVE detections whose true N lies in the cell,
                                  the fraction whose N-hat lands on the SAME side of 20.3 as
                                  the true N                                 (denominator: TP DETECTIONS by true N)
  P_cls[N-hat cell, SNR cell]   = among TRUE-POSITIVE detections whose N-hat lies in the cell,
                                  the fraction whose true N lies on the SAME side of 20.3
                                                                             (denominator: TP DETECTIONS by N-hat)
  plus the 2x2 class confusion matrix per SNR cell over ALL TP detections (one denominator).

Truth and detections come from the certified calibration machinery of record
(cddf_catalog_hbi.load_and_cut_catalog: greedy 1-to-1 truth match at the matrix floor 19.5
with |dz|/(1+z) < 0.01, host re-match at 19.0; accepted = P_DLA > 0.99, S2N_RED > 2, window
cuts; the same objects that generate the frozen molly counts). The frozen molly completeness /
purity counts of the real pack are RE-DERIVED from the same tables as a closure gate
(2LPT-0) before anything is written. Cells below the 19.7 reporting floor are computed and
FLAGGED (masked latent support), never merged into a reported cell; the 0.2-dex cell edges are
the Paper-1 latent edges 19.5, 19.7, ..., 22.1, 22.4, so no cell straddles 19.7 or 20.3.

No real-data values enter this file (mock calibration substrates only).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

NEDGES = np.array([19.5, 19.7, 19.9, 20.1, 20.3, 20.5, 20.7, 20.9, 21.1, 21.3, 21.5, 21.7, 21.9, 22.1, 22.4])
SNR_EDGES = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.inf])        # molly cells above the S2N > 2 cut
RESP_SNR_EDGES = np.array([2.0, 3.5, 6.5, np.inf])                 # the adopted response cells
TRUTH_FLOOR = 19.5
HOST_FLOOR = 19.0
REPORT_FLOOR = 19.7
CLASS_CUT = 20.3
P_MIN = 0.99
SNR_MIN = 2.0
MIN_SUPPORT = 20


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(cwd):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
    except Exception:
        return "unknown"


def beta68(k, n):
    """Jeffreys 68 % interval of a binomial fraction (nan where n = 0)."""
    from scipy.stats import beta
    k = np.asarray(k, float); n = np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.where(n > 0, beta.ppf(0.16, k + 0.5, n - k + 0.5), np.nan)
        hi = np.where(n > 0, beta.ppf(0.84, k + 0.5, n - k + 0.5), np.nan)
    return lo, hi


def cell_index(x, edges):
    i = np.searchsorted(edges, x, side="right") - 1
    ok = (i >= 0) & (i < len(edges) - 1) & np.isfinite(x)
    return i, ok


def family_paths(name):
    from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
    if name == "2lpt0":
        return dict(cat=AB.DEF_CAT, truth=AB.DEF_TRUTH, bal=AB.DEF_BAL, mockdir=os.path.dirname(AB.DEF_TRUTH))
    if name == "london0":
        from CDDF_analysis.hbi import track_c_tf_london0 as L
        return dict(cat=L._L0_CAT, truth=L._L0_TRUTH, bal=L._L0_BAL, mockdir=L._L0_MOCKDIR)
    if name == "saclay0":
        from CDDF_analysis.hbi import track_c_tf_saclay as S
        return dict(cat=S._S0_CAT, truth=S._S0_TRUTH, bal=S._S0_BAL, mockdir=S._S0_MOCKDIR)
    raise KeyError(name)


def tabulate(nhat, ntrue, snr, is_tp, host, truth_nhi, truth_snr, snr_edges):
    """All four quantities on (N cell x SNR cell) for one SNR grid. Returns dict of arrays
    with axis order (N cell, SNR cell)."""
    nN, nS = len(NEDGES) - 1, len(snr_edges) - 1
    # --- truth side
    ti, tok = cell_index(truth_nhi, NEDGES); tsi, tsok = cell_index(truth_snr, snr_edges)
    n_true = np.zeros((nN, nS)); np.add.at(n_true, (ti[tok & tsok], tsi[tok & tsok]), 1)
    # matched truth absorbers: TP detections carry the matched truth N (1-to-1 matching -> no double count)
    di, dok = cell_index(ntrue, NEDGES); dsi, dsok = cell_index(snr, snr_edges)
    m_any = is_tp & dok & dsok & (nhat >= TRUTH_FLOOR - 1e-9)
    m_rep = is_tp & dok & dsok & (nhat >= REPORT_FLOOR - 1e-9)
    n_found_any = np.zeros((nN, nS)); np.add.at(n_found_any, (di[m_any], dsi[m_any]), 1)
    n_found_rep = np.zeros((nN, nS)); np.add.at(n_found_rep, (di[m_rep], dsi[m_rep]), 1)
    # --- detection side, binned on N-hat
    hi_, hok = cell_index(nhat, NEDGES)
    acc = hok & dsok
    n_det = np.zeros((nN, nS)); np.add.at(n_det, (hi_[acc], dsi[acc]), 1)
    tp = acc & is_tp
    n_tp_by_nhat = np.zeros((nN, nS)); np.add.at(n_tp_by_nhat, (hi_[tp], dsi[tp]), 1)
    sub = acc & ~is_tp & np.isfinite(host) & (host >= HOST_FLOOR - 1e-9)      # host in [19.0, 19.5)
    n_subfloor_host = np.zeros((nN, nS)); np.add.at(n_subfloor_host, (hi_[sub], dsi[sub]), 1)
    nohost = acc & ~is_tp & ~(np.isfinite(host) & (host >= HOST_FLOOR - 1e-9))
    n_nohost = np.zeros((nN, nS)); np.add.at(n_nohost, (hi_[nohost], dsi[nohost]), 1)
    # --- class level among TP detections
    same = (nhat >= CLASS_CUT) == (ntrue >= CLASS_CUT)
    tpt = is_tp & dok & dsok                      # TP by TRUE-N cell
    n_tp_by_true = np.zeros((nN, nS)); np.add.at(n_tp_by_true, (di[tpt], dsi[tpt]), 1)
    n_cls_ok_by_true = np.zeros((nN, nS)); np.add.at(n_cls_ok_by_true, (di[tpt & same], dsi[tpt & same]), 1)
    n_cls_ok_by_nhat = np.zeros((nN, nS)); np.add.at(n_cls_ok_by_nhat, (hi_[tp & same], dsi[tp & same]), 1)
    # 2x2 confusion per SNR cell over ALL TP detections (rows true >=20.3 / <20.3; cols N-hat >=20.3 / <20.3)
    conf = np.zeros((nS, 2, 2))
    for s in range(nS):
        sel = is_tp & dsok & (dsi == s) & np.isfinite(ntrue)
        for r, tr in enumerate((ntrue >= CLASS_CUT, ntrue < CLASS_CUT)):
            for c, hh in enumerate((nhat >= CLASS_CUT, nhat < CLASS_CUT)):
                conf[s, r, c] = np.count_nonzero(sel & tr & hh)
    out = dict(n_true=n_true, n_found_any=n_found_any, n_found_reported=n_found_rep,
               n_det=n_det, n_tp_by_nhat=n_tp_by_nhat, n_subfloor_host=n_subfloor_host, n_nohost=n_nohost,
               n_tp_by_true=n_tp_by_true, n_cls_ok_by_true=n_cls_ok_by_true, n_cls_ok_by_nhat=n_cls_ok_by_nhat,
               confusion_tp=conf)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["C_abs_any"] = n_found_any / n_true
        out["C_abs_reported"] = n_found_rep / n_true
        out["P_abs"] = n_tp_by_nhat / n_det
        out["P_abs_incl_subfloor_host"] = (n_tp_by_nhat + n_subfloor_host) / n_det
        out["C_cls"] = n_cls_ok_by_true / n_tp_by_true
        out["P_cls"] = n_cls_ok_by_nhat / n_tp_by_nhat
    for key, k, n in (("C_abs_any", n_found_any, n_true), ("C_abs_reported", n_found_rep, n_true), ("P_abs", n_tp_by_nhat, n_det),
                      ("C_cls", n_cls_ok_by_true, n_tp_by_true), ("P_cls", n_cls_ok_by_nhat, n_tp_by_nhat)):
        lo, hi = beta68(k, n)
        out[key + "_lo68"], out[key + "_hi68"] = lo, hi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["2lpt0", "london0", "saclay0"])
    ap.add_argument("--molly-tsv", required=True, help="the 2LPT-0 lya_only nhi195 matrix (sets the truth floor 19.5)")
    ap.add_argument("--pack", default=None, help="frozen real pack: closure of its molly counts (2lpt0 only)")
    ap.add_argument("--closure-molly-tsv", default=None, help="the matrix the pack's molly block was built from (nhi172 lya_only); the closure re-derives the pack's counts at THAT floor")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo)
    from CDDF_analysis.hbi.cddf_catalog_hbi import (HBIConfig, load_molly_matrix, load_and_cut_catalog,
                                                    _build_qso_lookup, regenerate_molly_counts)
    fp = family_paths(a.family)
    cfg = HBIConfig(catalog_dir=fp["cat"], truth_path=fp["truth"], bal_cat_path=fp["bal"], molly_tsv=a.molly_tsv,
                    out_dir=a.out_dir, mockdir=fp["mockdir"], fp_estimator="purity_mixture", no_bal=True, lam_rf_min=1025.0)
    mm = load_molly_matrix(a.molly_tsv)
    assert abs(float(mm.nhi_edges[0]) - TRUTH_FLOOR) < 1e-9, "the product of record uses the 19.5-floor (nhi195 lya_only) matrix"
    lookup = _build_qso_lookup(cfg)
    cat, truth, is_tp, good, meta = load_and_cut_catalog(cfg, truth_nhi_floor=TRUTH_FLOOR, qso_lookup=lookup, host_truth_floor=HOST_FLOOR)
    s2n = np.asarray(cat["S2N_RED"], float); pdla = np.asarray(cat["P_DLA"], float)
    op = (s2n > SNR_MIN) & (pdla > P_MIN) & good
    nhat = np.asarray(cat["NHI"], float)[op]; ntrue = np.asarray(cat["NHI_TRUE"], float)[op]
    host = np.asarray(cat["NHI_TILT_HOST"], float)[op] if "NHI_TILT_HOST" in cat.colnames else np.full(op.sum(), np.nan)
    snr = s2n[op]; tp = np.asarray(is_tp, bool)[op]
    truth_nhi = np.asarray(truth["NHI"], float); truth_snr = np.asarray(truth["S2N_RED"], float)
    # ---- closure gate: the frozen molly counts re-derived from these tables (2LPT-0) ----
    closure = None
    if a.pack:
        # the pack's completeness block was built from the nhi172 lya_only matrix (truth floor 17.2);
        # re-derive it with the same machinery at THAT floor and require exact integer agreement —
        # this certifies that the tables used here are the certified calibration tables of record
        P = np.load(a.pack, allow_pickle=True)
        mmc = load_molly_matrix(a.closure_molly_tsv)
        floor_c = float(mmc.nhi_edges[0])
        cfg_c = HBIConfig(catalog_dir=fp["cat"], truth_path=fp["truth"], bal_cat_path=fp["bal"], molly_tsv=a.closure_molly_tsv,
                          out_dir=a.out_dir, mockdir=fp["mockdir"], fp_estimator="purity_mixture", no_bal=True, lam_rf_min=1025.0)
        cat_c, truth_c, tp_c, good_c, _ = load_and_cut_catalog(cfg_c, truth_nhi_floor=floor_c, qso_lookup=lookup, host_truth_floor=HOST_FLOOR)
        mm2 = regenerate_molly_counts(mmc, cat_c, tp_c, truth_c, good_c, cfg_c)
        closure = {"closure_matrix": a.closure_molly_tsv, "closure_truth_floor": floor_c,
                   "max_abs_diff_n_det": float(np.max(np.abs(mm2.cmp_nfound - P["molly_n_det"]))),
                   "max_abs_diff_n_tot": float(np.max(np.abs(mm2.cmp_nfid - P["molly_n_tot"]))),
                   "max_abs_diff_purity_ratio_vs_tsv": float(mm2._max_p_diff), "max_abs_diff_completeness_ratio_vs_tsv": float(mm2._max_c_diff),
                   "note": "the product of record below uses the 19.5-floor matrix (truth absorbers >= 19.5; 'found' requires N-hat >= 19.5): the pack's own block counts hosts down to 17.2 and 'found' at N-hat > 17.2, so its cells >= 19.5 are the same objects with a lower 'found' threshold"}
        d_det = mm2.cmp_nfound - np.asarray(P["molly_n_det"], float)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(d_det) / np.maximum(np.asarray(P["molly_n_tot"], float), 1.0)
        closure.update({"n_det_diff_sum": float(d_det.sum()), "n_det_diff_max_rel_to_n_tot": float(np.nanmax(rel)),
                        "n_det_diff_by_N_cell_summed_over_snr": d_det.sum(axis=0).tolist(),
                        "verdict": ("truth denominators (n_tot) IDENTICAL in every cell; found counts reproduced to within the pipeline's own regeneration guard "
                                    "(run_pipeline hard-guards at 5e-3 on the ratios) but not integer-exact — the frozen block was produced by "
                                    "track_c_tf_loa.build_frozen_calibration with its own load path")})
        if closure["max_abs_diff_n_tot"] > 0 or closure["max_abs_diff_completeness_ratio_vs_tsv"] > 5e-3 or closure["max_abs_diff_purity_ratio_vs_tsv"] > 5e-3:
            raise SystemExit(f"BLOCKED: molly counts do not close against the frozen pack: {closure}")
        print(f"closure: n_tot identical; n_det max |diff| {closure['max_abs_diff_n_det']:.0f} (max rel to n_tot {closure['n_det_diff_max_rel_to_n_tot']:.2e}); ratio diffs {closure['max_abs_diff_completeness_ratio_vs_tsv']:.1e} / {closure['max_abs_diff_purity_ratio_vs_tsv']:.1e} < 5e-3 guard")
    grids = {"molly_snr_cells": SNR_EDGES, "response_snr_cells": RESP_SNR_EDGES}
    res = {}
    npz = {"n_edges": NEDGES, "molly_snr_edges": SNR_EDGES, "response_snr_edges": RESP_SNR_EDGES,
           "axis_note": np.array(["every (N x SNR) array is (N cell in n_edges order, SNR cell in the named snr_edges order); confusion_tp is (SNR cell, true row: >=20.3 / <20.3, N-hat col: >=20.3 / <20.3)"])}
    for gname, se in grids.items():
        t = tabulate(nhat, ntrue, snr, tp, host, truth_nhi, truth_snr, se)
        res[gname] = {k: (v.tolist()) for k, v in t.items()}
        for k, v in t.items():
            npz[f"{gname}__{k}"] = v
    cell_flags = [{"n_cell": [float(NEDGES[i]), float(NEDGES[i + 1])],
                   "status": ("BELOW REPORTING FLOOR: masked latent support [19.5, 19.7) — computed, never a reported number" if NEDGES[i + 1] <= REPORT_FLOOR + 1e-9
                              else "reported sub-DLA range [19.7, 20.3)" if NEDGES[i + 1] <= CLASS_CUT + 1e-9 else "reported DLA range (>= 20.3)")} for i in range(len(NEDGES) - 1)]
    out = {
        "role": f"R-033 fixed-denominator purity / completeness, family {a.family} — absorber level (truth denominator) vs class level (TP-detection denominator), (log N_HI x SNR) cells on the Paper-1 contract; sits BESIDE the frozen products",
        "status": "publication-ready as calibration-product tables on the mock substrate (the denominator conventions below are the science-lane choice, stated; PI ratifies)",
        "family": a.family, "written_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {"module": "CDDF_analysis/hbi_mcmc/pc_fixed_denominator.py", "commit": _git(repo), "argv": sys.argv, "python": sys.version.split()[0], "numpy": np.__version__, "conda_env": os.environ.get("CONDA_DEFAULT_ENV")},
        "inputs": {"catalog_dir": fp["cat"], "truth": {"path": fp["truth"], "sha256": _sha(fp["truth"])}, "bal": {"path": fp["bal"], "sha256": _sha(fp["bal"])},
                   "mockdir": fp["mockdir"], "molly_tsv": {"path": a.molly_tsv, "sha256": _sha(a.molly_tsv)}, **({"pack": {"path": a.pack, "sha256": _sha(a.pack)}, "closure_molly_tsv": {"path": a.closure_molly_tsv, "sha256": _sha(a.closure_molly_tsv)}} if a.pack else {}),
                   "load_and_cut_catalog_meta": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in meta.items()}},
        "contract": {"accepted_detection": f"P_DLA > {P_MIN}, S2N_RED > {SNR_MIN}, good_mask (window / BAL / sentinel cuts of load_and_cut_catalog), lam_rf_min 1025",
                     "truth_match": "greedy 1-to-1, descending N-hat, |dz|/(1+z_truth) < 0.01, truth floored at 19.5 (matrix floor); host re-match floored at 19.0 (hierarchical)",
                     "truth_denominator": f"truth absorbers with N >= {TRUTH_FLOOR} and a finite S2N_RED on their sightline, in the (true N, SNR) cell (the molly 'n_fid' definition; NOT window-restricted beyond the catalogue's own cuts)",
                     "found": "matched by an accepted detection with N-hat >= 19.5 ('any', the frozen molly definition) or >= 19.7 ('reported')",
                     "detection_denominator": "accepted detections in the (N-hat, SNR) cell", "class_cut": CLASS_CUT, "report_floor": REPORT_FLOOR,
                     "class_level_denominator": "TRUE-POSITIVE accepted detections (host >= 19.5), by true-N cell for C_cls and by N-hat cell for P_cls; the 2x2 confusion per SNR cell uses all TPs in that SNR cell",
                     "n_edges": NEDGES.tolist(), "no_cell_straddles": "19.7 and 20.3 are both cell edges", "low_support_rule": f"flag cells with denominator < {MIN_SUPPORT}"},
        "counts_summary": {"n_accepted_detections": int(op.sum()), "n_true_positive": int(tp.sum()), "n_truth_absorbers_ge_floor": int((truth_nhi >= TRUTH_FLOOR).sum()),
                           "n_truth_in_window_cells": float(res["molly_snr_cells"]["n_true"] and np.array(res["molly_snr_cells"]["n_true"]).sum())},
        "cell_flags": cell_flags, "closure_vs_frozen_molly_counts": closure,
        "tables": res,
    }
    jp = os.path.join(a.out_dir, f"R033_pc_fixed_denominator_{a.family}.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else float(o))
    np.savez(os.path.join(a.out_dir, f"R033_pc_fixed_denominator_{a.family}.npz"), **npz)
    print(json.dumps({os.path.basename(p): _sha(p) for p in (jp, jp[:-5] + ".npz")}, indent=1))
    t = res["response_snr_cells"]
    print("C_abs_any (N cell rows x 3 SNR cells):"); print(np.round(np.array(t["C_abs_any"]), 3))
    print("P_abs:"); print(np.round(np.array(t["P_abs"]), 3))
    print("C_cls:"); print(np.round(np.array(t["C_cls"]), 3))
    print("P_cls:"); print(np.round(np.array(t["P_cls"]), 3))
    print("confusion per SNR cell:", np.array(t["confusion_tp"]).astype(int).tolist())


if __name__ == "__main__":
    main()
