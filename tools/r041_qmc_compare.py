#!/usr/bin/env python3
"""r041_qmc_compare.py — paired QMC spot check (PI ruling 2026-09-01 late, §2): two finder runs on IDENTICAL spectra and
configuration except the QMC sample count. Per TARGETID compare: rows per TID, MAP absorber multiplicity (from the processed h5
model_posteriors argmax when available, else the dlacat row count), accepted-candidate status (P_DLA > thr & DLAFLAG == 0 &
SNR_REDSIDE > snr_min), P_DLA threshold crossings, and the matched (z_DLA, N_HI) of accepted rows paired across arms by nearest |Δz|.
Optionally, with --truth, per-injection detection in each arm. Writes a JSON summary + a per-TID CSV. No science verdict.

    python tools/r041_qmc_compare.py --a-dir OUT_50k --b-dir OUT_100k --a-label 50k --b-label 100k --out cmp.json [--truth truth.csv] [--population r041_population.csv]
"""
import argparse, csv, glob, json, os, sys
import numpy as np


def load_cat(d):
    from astropy.io import fits
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "dlacat-*.fits"))):
        t = fits.open(f)[1].data
        for r in t:
            rows.append(dict(TARGETID=int(r["TARGETID"]), Z_QSO=float(r["Z_QSO"]), SNR=float(r["SNR_REDSIDE"]), DLAID=str(r["DLAID"]), Z=float(r["Z_DLA"]), NHI=float(r["NHI"]), P=float(r["P_DLA"]), FLAG=int(r["DLAFLAG"])))
    return rows


def load_mult(d):
    """MAP multiplicity per TID from processed h5 (argmax of model_posteriors over [Null, 1..K])."""
    out = {}
    try:
        import h5py
    except ImportError:
        return out
    for f in sorted(glob.glob(os.path.join(d, "figures", "processed", "processed-*.h5"))):
        with h5py.File(f, "r") as h:
            if "model_posteriors" not in h or "target_ids" not in h:
                continue
            mp = h["model_posteriors"][...]; tids = h["target_ids"][...]
            for tid, row in zip(tids, mp):
                out[int(tid)] = int(np.nanargmax(row)) if np.isfinite(row).any() else -1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True); ap.add_argument("--b-dir", required=True)
    ap.add_argument("--a-label", default="A"); ap.add_argument("--b-label", default="B")
    ap.add_argument("--p-thr", type=float, default=0.99); ap.add_argument("--snr-min", type=float, default=2.0)
    ap.add_argument("--dz-match", type=float, default=0.01, help="|dz|/(1+z) tolerance for pairing accepted rows across arms")
    ap.add_argument("--truth", default=None); ap.add_argument("--population", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    A, B = load_cat(a.a_dir), load_cat(a.b_dir); mA, mB = load_mult(a.a_dir), load_mult(a.b_dir)
    tids = sorted(set(r["TARGETID"] for r in A) | set(r["TARGETID"] for r in B))
    pop = None
    if a.population:
        pop = {int(r["TARGETID"]) for r in csv.DictReader(open(a.population))}
    acc = lambda r: (r["P"] > a.p_thr) and (r["FLAG"] == 0) and (r["SNR"] > a.snr_min)
    per, S = [], dict(n_tids=len(tids), n_rows_A=len(A), n_rows_B=len(B), rows_per_tid_equal=0, mult_equal=0, accepted_status_equal=0, n_accepted_A=0, n_accepted_B=0,
                      pairs_matched=0, pairs_unmatched_A=0, pairs_unmatched_B=0, dz_abs=[], dN_abs=[], p_cross_diff=0, in_population=0)
    for tid in tids:
        ra = [r for r in A if r["TARGETID"] == tid]; rb = [r for r in B if r["TARGETID"] == tid]
        aa = [r for r in ra if acc(r)]; ab = [r for r in rb if acc(r)]
        S["n_accepted_A"] += len(aa); S["n_accepted_B"] += len(ab)
        S["rows_per_tid_equal"] += int(len(ra) == len(rb)); S["accepted_status_equal"] += int((len(aa) > 0) == (len(ab) > 0))
        ma, mb = mA.get(tid), mB.get(tid); S["mult_equal"] += int(ma == mb)
        pa = any(r["P"] > a.p_thr for r in ra); pb = any(r["P"] > a.p_thr for r in rb); S["p_cross_diff"] += int(pa != pb)
        if pop is not None and tid in pop:
            S["in_population"] += 1
        # pair accepted rows by nearest |dz|
        used = set(); matched = []
        for x in aa:
            best, bd = None, 1e9
            for j, y in enumerate(ab):
                if j in used:
                    continue
                d = abs(x["Z"] - y["Z"]) / (1 + x["Z"])
                if d < bd:
                    best, bd = j, d
            if best is not None and bd < a.dz_match:
                used.add(best); y = ab[best]; matched.append((x, y)); S["pairs_matched"] += 1; S["dz_abs"].append(abs(x["Z"] - y["Z"])); S["dN_abs"].append(abs(x["NHI"] - y["NHI"]))
            else:
                S["pairs_unmatched_A"] += 1
        S["pairs_unmatched_B"] += len(ab) - len(used)
        per.append(dict(TARGETID=tid, in_population=int(pop is not None and tid in pop), rows_A=len(ra), rows_B=len(rb), map_mult_A=ma, map_mult_B=mb,
                        n_accepted_A=len(aa), n_accepted_B=len(ab), any_accepted_A=int(bool(aa)), any_accepted_B=int(bool(ab)), p_cross_A=int(pa), p_cross_B=int(pb),
                        pmax_A=max([r["P"] for r in ra], default=None), pmax_B=max([r["P"] for r in rb], default=None),
                        accepted_A=";".join(f"{r['Z']:.4f}:{r['NHI']:.3f}:{r['P']:.4f}" for r in aa), accepted_B=";".join(f"{r['Z']:.4f}:{r['NHI']:.3f}:{r['P']:.4f}" for r in ab),
                        matched_pairs=len(matched), max_dz_matched=(max(abs(x["Z"] - y["Z"]) for x, y in matched) if matched else None),
                        max_dN_matched=(max(abs(x["NHI"] - y["NHI"]) for x, y in matched) if matched else None)))
    S["dz_abs"] = dict(n=len(S["dz_abs"]), max=(max(S["dz_abs"]) if S["dz_abs"] else None), median=(float(np.median(S["dz_abs"])) if S["dz_abs"] else None))
    S["dN_abs"] = dict(n=len(S["dN_abs"]), max=(max(S["dN_abs"]) if S["dN_abs"] else None), median=(float(np.median(S["dN_abs"])) if S["dN_abs"] else None), p90=(float(np.percentile(S["dN_abs"], 90)) if S["dN_abs"] else None))
    S["selection_relevant_discrepancies"] = dict(accepted_status_differs=S["n_tids"] - S["accepted_status_equal"], rows_per_tid_differs=S["n_tids"] - S["rows_per_tid_equal"],
                                                 map_multiplicity_differs=S["n_tids"] - S["mult_equal"], p_threshold_crossing_differs=S["p_cross_diff"], unmatched_accepted_rows=S["pairs_unmatched_A"] + S["pairs_unmatched_B"])
    if a.truth:
        truth = list(csv.DictReader(open(a.truth))); det = dict(A=0, B=0, both=0, neither=0, only_A=0, only_B=0); rows = []
        for t in truth:
            tid = int(t["TARGETID"]); zt = float(t["z_inj"]); ok = {}
            for lab, cat in (("A", A), ("B", B)):
                c = [r for r in cat if r["TARGETID"] == tid and acc(r) and abs(r["Z"] - zt) / (1 + zt) < a.dz_match]
                ok[lab] = bool(c); rows.append(dict(TARGETID=tid, arm=lab, logN=float(t["logN"]), z_inj=zt, detected=int(bool(c)), nhat=(c[0]["NHI"] if c else None)))
            det["A"] += ok["A"]; det["B"] += ok["B"]; det["both"] += ok["A"] and ok["B"]; det["neither"] += (not ok["A"]) and (not ok["B"]); det["only_A"] += ok["A"] and not ok["B"]; det["only_B"] += ok["B"] and not ok["A"]
        S["injections"] = dict(n=len(truth), **det); S["per_injection"] = rows
    S.update(a_dir=a.a_dir, b_dir=a.b_dir, a_label=a.a_label, b_label=a.b_label, p_thr=a.p_thr, snr_min=a.snr_min, dz_match=a.dz_match)
    json.dump(S, open(a.out, "w"), indent=1)
    with open(a.out[:-5] + "_per_tid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0].keys())); w.writeheader(); w.writerows(per)
    print(json.dumps({k: S[k] for k in ("n_tids", "n_rows_A", "n_rows_B", "n_accepted_A", "n_accepted_B", "pairs_matched", "dz_abs", "dN_abs", "selection_relevant_discrepancies", "in_population")} | ({"injections": {k: v for k, v in S["injections"].items() if k != "per_injection"}} if "injections" in S else {}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
