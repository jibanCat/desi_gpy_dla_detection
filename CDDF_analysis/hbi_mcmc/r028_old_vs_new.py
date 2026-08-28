#!/usr/bin/env python
"""r028_old_vs_new.py — FIG-04 / R-028 part 2: old-finder vs new-finder purity / completeness on a
MATCHED sightline set of one mock family, recomputed from the raw candidate catalogues with the
certified R-033 machinery (cddf_catalog_hbi.load_and_cut_catalog + pc_fixed_denominator.tabulate),
plus the per-object matched table for the "molly plot".

Matched set: sightlines the OLD run actually processed (its per-spectrum output TARGET_IDs) that
are inside the Paper-1 operational population (zcat z window, S2N_RED > 2, non-BAL); the SAME set
bounds truth, detections and the completeness denominators of BOTH runs. Headline P/C at the
thresholds 20.0 and 20.3 (absorber level; truth denominators; TP = |dz|/(1+z) < 0.01 greedy 1-1
match at the 19.5 floor, accepted = P_DLA > 0.99 & S2N_RED > 2 & window). Caveat (stated in the
product): NOT a controlled A/B — the two runs differ in the trained model, the code version and
the search configuration (see the run records); the comparison is "the previous finder as run"
vs "the new finder as run". No real-data values (mock substrates only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pc_fixed_denominator as PC  # noqa: E402


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def run_one(label, cat_dir, fp, molly_tsv, out_dir, restrict):
    from CDDF_analysis.hbi.cddf_catalog_hbi import HBIConfig, load_and_cut_catalog, _build_qso_lookup
    cfg = HBIConfig(catalog_dir=cat_dir, truth_path=fp["truth"], bal_cat_path=fp["bal"], molly_tsv=molly_tsv,
                    out_dir=out_dir, mockdir=fp["mockdir"], fp_estimator="purity_mixture", no_bal=True, lam_rf_min=1025.0)
    lookup = _build_qso_lookup(cfg)
    if restrict is not None:
        lookup = {t: v for t, v in lookup.items() if t in restrict}
    cat, truth, is_tp, good, meta = load_and_cut_catalog(cfg, truth_nhi_floor=PC.TRUTH_FLOOR, qso_lookup=lookup, host_truth_floor=PC.HOST_FLOOR)
    ctid = np.asarray(cat["TARGETID"], np.int64); ttid = np.asarray(truth["TARGETID"], np.int64)
    if restrict is not None:
        keep_c = np.isin(ctid, list(restrict)); keep_t = np.isin(ttid, list(restrict))
        cat = cat[keep_c]; is_tp = np.asarray(is_tp, bool)[keep_c]; good = np.asarray(good, bool)[keep_c]; truth = truth[keep_t]
    s2n = np.asarray(cat["S2N_RED"], float); pdla = np.asarray(cat["P_DLA"], float)
    op = (s2n > PC.SNR_MIN) & (pdla > PC.P_MIN) & np.asarray(good, bool)
    nhat = np.asarray(cat["NHI"], float)[op]; ntrue = np.asarray(cat["NHI_TRUE"], float)[op]
    host = np.asarray(cat["NHI_TILT_HOST"], float)[op] if "NHI_TILT_HOST" in cat.colnames else np.full(int(op.sum()), np.nan)
    snr = s2n[op]; tp = np.asarray(is_tp, bool)[op]; tid_op = np.asarray(cat["TARGETID"], np.int64)[op]
    zd = np.asarray(cat["Z_DLA"], float)[op] if "Z_DLA" in cat.colnames else np.full(int(op.sum()), np.nan)
    truth_nhi = np.asarray(truth["NHI"], float); truth_snr = np.asarray(truth["S2N_RED"], float)
    tabs = {g: PC.tabulate(nhat, ntrue, snr, tp, host, truth_nhi, truth_snr, se) for g, se in (("molly_snr_cells", PC.SNR_EDGES), ("response_snr_cells", PC.RESP_SNR_EDGES))}
    # headline P / C at thresholds over all SNR > 2 (absorber level)
    head = {}
    for T in (20.0, 20.3):
        n_true = int(np.sum((truth_nhi >= T - 1e-9) & (truth_snr > PC.SNR_MIN)))
        n_found = int(np.sum(tp & (ntrue >= T - 1e-9) & (nhat >= PC.REPORT_FLOOR - 1e-9)))
        n_det = int(np.sum(nhat >= T - 1e-9)); n_tp = int(np.sum(tp & (nhat >= T - 1e-9)))
        # class-consistent variants: found AND reported on the same side; TP AND true on the same side
        n_found_cls = int(np.sum(tp & (ntrue >= T - 1e-9) & (nhat >= T - 1e-9))); n_tp_cls = int(np.sum(tp & (nhat >= T - 1e-9) & (ntrue >= T - 1e-9)))
        head[f"T{T}"] = dict(n_true=n_true, n_found=n_found, C=(n_found / n_true if n_true else None), n_det=n_det, n_tp=n_tp, P=(n_tp / n_det if n_det else None),
                             C_class=(n_found_cls / n_true if n_true else None), P_class=(n_tp_cls / n_det if n_det else None),
                             C_lo68_hi68=[float(np.ravel(x)[0]) for x in PC.beta68(np.array([n_found]), np.array([n_true]))] if n_true else None,
                             P_lo68_hi68=[float(np.ravel(x)[0]) for x in PC.beta68(np.array([n_tp]), np.array([n_det]))] if n_det else None)
    per_obj = dict(TARGETID=tid_op[tp], NHI_TRUE=ntrue[tp], NHI_HAT=nhat[tp], S2N_RED=snr[tp], Z_DLA=zd[tp])
    return dict(label=label, cat_dir=cat_dir, n_sightlines_population=len(lookup), n_accepted=int(op.sum()), n_tp=int(tp.sum()),
                n_truth=int(truth_nhi.size), headline=head, tables=tabs, meta={k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in meta.items()}), per_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["2lpt0", "london0", "saclay0"])
    ap.add_argument("--old-cat-dir", required=True, help="dir holding the OLD finder's dlacat-*.fits")
    ap.add_argument("--old-label", required=True)
    ap.add_argument("--old-processed-fits", default=None, help="OLD run per-spectrum output (TARGET_ID column) = the processed set")
    ap.add_argument("--new-cat-dir", default=None, help="default: the family's catalogue of record")
    ap.add_argument("--new-label", default="new finder (catalogue of record)")
    ap.add_argument("--molly-tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    import fitsio
    fp = PC.family_paths(a.family)
    new_dir = a.new_cat_dir or fp["cat"]
    restrict = None
    if a.old_processed_fits:
        ps = fitsio.read(a.old_processed_fits, ext=1)
        col = "TARGET_ID" if "TARGET_ID" in ps.dtype.names else "TARGETID"
        restrict = set(np.asarray(ps[col], np.int64).tolist())
    res_old, po = run_one(a.old_label, a.old_cat_dir, fp, a.molly_tsv, a.out_dir, restrict)
    res_new, pn = run_one(a.new_label, new_dir, fp, a.molly_tsv, a.out_dir, restrict)
    # per-object matched table: truth absorbers matched (TP) in BOTH runs, keyed by (TARGETID, truth N)
    ko = {(int(t), round(float(n), 4)): i for i, (t, n) in enumerate(zip(po["TARGETID"], po["NHI_TRUE"]))}
    rows = []
    for j, (t, n) in enumerate(zip(pn["TARGETID"], pn["NHI_TRUE"])):
        i = ko.get((int(t), round(float(n), 4)))
        if i is not None:
            rows.append((int(t), float(n), float(po["NHI_HAT"][i]), float(pn["NHI_HAT"][j]), float(pn["S2N_RED"][j]), float(pn["Z_DLA"][j])))
    both = np.array(rows, float) if rows else np.zeros((0, 6))
    out = dict(role="R-028 part 2 (FIG-04 old->new): purity / completeness of the previous finder vs the new finder on a matched sightline set, from raw candidate catalogues",
               family=a.family, matched_set=dict(source=a.old_processed_fits, n_tids=(len(restrict) if restrict else None)),
               caveat="NOT a controlled A/B: model, code version and search configuration differ between the runs (see run records); values are 'as run'",
               old=res_old, new=res_new,
               per_object=dict(n_tp_old=int(len(po["TARGETID"])), n_tp_new=int(len(pn["TARGETID"])), n_tp_both=int(both.shape[0]),
                               columns=["TARGETID", "NHI_TRUE", "NHI_HAT_old", "NHI_HAT_new", "S2N_RED", "Z_DLA"]),
               inputs=dict(truth=dict(path=fp["truth"], sha256=_sha(fp["truth"])), bal=dict(path=fp["bal"], sha256=_sha(fp["bal"])), molly_tsv=dict(path=a.molly_tsv, sha256=_sha(a.molly_tsv)),
                           old_cat=[dict(path=f, sha256=_sha(f)) for f in sorted(__import__("glob").glob(os.path.join(a.old_cat_dir, "dlacat-*.fits")))][:5],
                           new_cat_dir=new_dir),
               generator=dict(module="CDDF_analysis/hbi_mcmc/r028_old_vs_new.py", commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(), argv=sys.argv))
    for r in (out["old"], out["new"]):
        r["tables"] = {g: {k: np.asarray(v).tolist() for k, v in t.items()} for g, t in r["tables"].items()}
    jp = os.path.join(a.out_dir, f"r028_old_vs_new_{a.family}.json"); json.dump(out, open(jp, "w"), indent=1)
    np.savez(os.path.join(a.out_dir, f"r028_old_vs_new_{a.family}.npz"), per_object_both=both, old_tp_TARGETID=po["TARGETID"], old_tp_NHI_TRUE=po["NHI_TRUE"], old_tp_NHI_HAT=po["NHI_HAT"],
             new_tp_TARGETID=pn["TARGETID"], new_tp_NHI_TRUE=pn["NHI_TRUE"], new_tp_NHI_HAT=pn["NHI_HAT"], n_edges=PC.NEDGES, molly_snr_edges=PC.SNR_EDGES, response_snr_edges=PC.RESP_SNR_EDGES)
    print(json.dumps({k: {"headline": out[k]["headline"], "n_accepted": out[k]["n_accepted"], "n_tp": out[k]["n_tp"], "n_truth": out[k]["n_truth"], "n_pop": out[k]["n_sightlines_population"]} for k in ("old", "new")}, indent=1))
    print("per-object both:", both.shape[0])


if __name__ == "__main__":
    main()
