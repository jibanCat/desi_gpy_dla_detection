#!/usr/bin/env python
"""r041_analyze.py — recovery analysis of one or more R-041 waves under the MEASUREMENT's
canonical contract (the same one the high-z catalogue and R-039 use):

  accepted candidate: P_DLA > 0.99, DLAFLAG == 0, SNR_REDSIDE > 2 (the dlacat column), the
  candidate's z inside the sightline's Lyα window; matching to the injected truth: nearest
  candidate with |dz| / (1 + z_true) < 0.01 (the canonical matcher tolerance; under MAX_DLAS = 1
  at most one candidate per sightline exists, under MAX_DLAS = 4 up to four — the match is the
  nearest-|dz| accepted row either way).

Multiplicity (MAX4 repair cycle, 2026-09-01; PI ruling 2026-08-28 item 3): per injection the
analyzer also records n_accepted (accepted rows on the sightline), accepted_nhats / accepted_zs
(all of them, ';'-joined, file order) and accepted_nhat_max. The legacy accepted_nhat / accepted_z
columns are the accepted row with the LARGEST N-hat — with a single accepted row (every MAX_DLAS = 1
output) this is exactly the old first-row value, so MAX1 products re-analysed here reproduce their
old columns bit-for-bit; with several rows it is order-independent (the old cands[0] was file-order
dependent).

Per injection: detected (matched accepted candidate), reported N-hat, P_DLA, dz. Per cell
(log N point | molly N cell, z sub-bin, SNR stratum): trials n, recoveries k, DETECTION
completeness C = k/n with Jeffreys 68 % intervals, REPORTING completeness at T = 20.0 / 20.3
(N-hat >= T among truth >= T), the N-hat response moments (mean/std of N-hat - N_true), and
the class outcome (N-hat >= 20.3 vs N_true >= 20.3). Marginalisations: per (molly N cell,
stratum) over z with the injected (path-proportional) z mixture; per molly N cell over
strata with the population's path weights per stratum. Pair mode (R-041D): both absorbers of
a pair are matched; outcomes by accepted rows on the sightline (n_accepted 0 / 1 / >=2) and by
accepted rows matched to DISTINCT absorbers within the tolerance (n_matched 0 / 1 / 2 — two is only
possible with MAX_DLAS > 1; frac_two is that fraction), which absorber wins, merged-N-hat bias,
threshold migration.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LYA = 1215.67
C_KMS = 299792.458
DZ_REL = 0.01
P_MIN = 0.99
SNR_MIN = 2.0
MOLLY_N_EDGES = [19.5, 20.0, 20.3, 20.5, 21.0, 21.5, 22.0, np.inf]
ZBINS = [3.8, 4.25, 4.5, 5.0]
STRATA_EDGES = [2.0, 3.0, 4.0, 5.0, 7.0, np.inf]


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def jeffreys(k, n):
    from scipy.stats import beta
    if n == 0:
        return (None, None)
    return (float(beta.ppf(0.16, k + 0.5, n - k + 0.5)), float(beta.ppf(0.84, k + 0.5, n - k + 0.5)))


def _parse_list(s):
    """';'-joined float list (the accepted_nhats / accepted_zs CSV columns) -> list of floats."""
    return [float(x) for x in str(s).split(";") if x not in ("", "nan")]


def cell(rows, key):
    n = len(rows); k = sum(1 for r in rows if r["detected"])
    lo, hi = jeffreys(k, n)
    det = [r for r in rows if r["detected"]]
    dN = np.array([r["nhat"] - r["logN"] for r in det]) if det else np.array([])
    out = dict(key=key, n=n, k=k, C=(k / n if n else None), C_lo68=lo, C_hi68=hi,
               dlogN_mean=(float(dN.mean()) if dN.size else None), dlogN_std=(float(dN.std()) if dN.size else None),
               n_reported_ge20p3=sum(1 for r in det if r["nhat"] >= 20.3), n_reported_ge20p0=sum(1 for r in det if r["nhat"] >= 20.0))
    for T in (20.0, 20.3):
        dom = [r for r in rows if r["logN"] >= T - 1e-9]
        kk = sum(1 for r in dom if r["detected"] and r["nhat"] >= T)
        out[f"reporting_C_T{T}"] = (kk / len(dom) if dom else None); out[f"reporting_n_T{T}"] = len(dom); out[f"reporting_k_T{T}"] = kk
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", nargs="+", required=True, help="truth CSVs of the waves")
    ap.add_argument("--outputs", nargs="+", required=True, help="finder output dirs (dlacat-*.fits) matching --truth order")
    ap.add_argument("--population", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    import fitsio
    pop = {int(r["TARGETID"]): r for r in csv.DictReader(open(a.population))}
    dX_strat = np.zeros(len(STRATA_EDGES) - 1)
    for r in pop.values():
        dX_strat[int(r["stratum"])] += float(r["dX_bin"])
    w_strat = dX_strat / dX_strat.sum()
    inj = []
    prov = []
    for tpath, odir in zip(a.truth, a.outputs):
        truth = list(csv.DictReader(open(tpath)))
        files = sorted(glob.glob(os.path.join(odir, "dlacat-*.fits")))
        if not files:
            raise SystemExit(f"no dlacat outputs in {odir}")
        cat = np.concatenate([fitsio.read(f, ext=1) for f in files])
        prov.append(dict(truth=tpath, truth_sha256=_sha(tpath), outputs=odir, n_output_files=len(files), n_rows=int(len(cat)),
                         output_sha256=[_sha(f) for f in files]))
        by_tid = {}
        for row in cat:
            by_tid.setdefault(int(row["TARGETID"]), []).append(row)
        for t in truth:
            tid = int(t["TARGETID"]); zt = float(t["z_inj"]); pr = pop.get(tid)
            zlo, zhi = (float(pr["zlo"]), float(pr["zhi"])) if pr else (-np.inf, np.inf)
            cands = [c for c in by_tid.get(tid, []) if c["P_DLA"] > P_MIN and int(c["DLAFLAG"]) == 0 and c["SNR_REDSIDE"] > SNR_MIN and zlo < c["Z_DLA"] < zhi]
            best, bdz = None, None
            for c in cands:
                dz = abs(float(c["Z_DLA"]) - zt) / (1.0 + zt)
                if dz < DZ_REL and (bdz is None or dz < bdz):
                    best, bdz = c, dz
            allrows = by_tid.get(tid, [])
            acc_nhats = [float(c["NHI"]) for c in cands]           # every accepted row, file order
            acc_zs = [float(c["Z_DLA"]) for c in cands]
            imax = int(np.argmax(acc_nhats)) if cands else None     # the accepted row of largest N-hat
            rec = dict(TARGETID=tid, wave=int(t["wave"]), inj_idx=int(t["inj_idx"]), logN=float(t["logN"]), z_inj=zt, stratum=int(t["stratum"]),
                       snr=float(t["snr"]), has_cand_ge20=int(t.get("has_cand_ge20", 0) or 0), pair_class=t.get("pair_class", ""), dv_kms=t.get("dv_kms", ""),
                       pair_logN=t.get("pair_logN", ""), method=t.get("method", ""), meanflux_model=t.get("meanflux_model", ""),
                       detected=best is not None, nhat=(float(best["NHI"]) if best is not None else np.nan), p=(float(best["P_DLA"]) if best is not None else np.nan),
                       dz=(float(bdz) if best is not None else np.nan), n_rows_sightline=len(allrows),
                       any_accepted=(len(cands) > 0),
                       # legacy columns = the accepted row of largest N-hat (== the old cands[0] whenever one row is accepted)
                       accepted_nhat=(acc_nhats[imax] if cands else np.nan), accepted_z=(acc_zs[imax] if cands else np.nan),
                       zbin=int(np.searchsorted(ZBINS, zt, side="right") - 1),
                       # multiplicity columns (MAX4 repair cycle; appended so the old column order is a prefix)
                       n_accepted=len(cands), accepted_nhat_max=(acc_nhats[imax] if cands else np.nan),
                       accepted_nhats=";".join(repr(x) for x in acc_nhats), accepted_zs=";".join(repr(x) for x in acc_zs))
            inj.append(rec)
    # ---- tables
    logn_points = sorted({r["logN"] for r in inj})
    ns = len(STRATA_EDGES) - 1
    tables = {"per_logN_point_x_stratum": [], "per_logN_point_x_zbin_x_stratum": [], "per_molly_cell_x_stratum": [], "per_molly_cell_x_zbin_x_stratum": [],
              "per_molly_cell_zmarginal_pathweighted": [], "per_molly_cell_x_zbin_pathweighted": [], "per_logN_point_all": [], "by_has_candidate": []}
    single = [r for r in inj if not r["pair_class"]]
    for ln in logn_points:
        rows = [r for r in single if r["logN"] == ln]
        tables["per_logN_point_all"].append(cell(rows, dict(logN=ln)))
        for s in range(ns):
            rs = [r for r in rows if r["stratum"] == s]
            tables["per_logN_point_x_stratum"].append(cell(rs, dict(logN=ln, stratum=s)))
            for zb in range(3):
                tables["per_logN_point_x_zbin_x_stratum"].append(cell([r for r in rs if r["zbin"] == zb], dict(logN=ln, stratum=s, zbin=zb)))
        for hc in (0, 1):
            tables["by_has_candidate"].append(cell([r for r in rows if r["has_cand_ge20"] == hc], dict(logN=ln, has_cand_ge20=hc)))
    def mcell(ln):
        return int(np.searchsorted(MOLLY_N_EDGES, ln, side="right") - 1)
    for j in range(len(MOLLY_N_EDGES) - 1):
        rows = [r for r in single if mcell(r["logN"]) == j]
        key = dict(molly_cell=j, n_lo=MOLLY_N_EDGES[j], n_hi=(MOLLY_N_EDGES[j + 1] if np.isfinite(MOLLY_N_EDGES[j + 1]) else "inf"))
        per_s = []
        for s in range(ns):
            c = cell([r for r in rows if r["stratum"] == s], {**key, "stratum": s}); tables["per_molly_cell_x_stratum"].append(c); per_s.append(c)
            for zb in range(3):
                tables["per_molly_cell_x_zbin_x_stratum"].append(cell([r for r in rows if r["stratum"] == s and r["zbin"] == zb], {**key, "stratum": s, "zbin": zb}))
        # path-weighted marginal over strata (weights = the population's dX per stratum)
        Cs = np.array([c["C"] if c["C"] is not None else np.nan for c in per_s]); ok = np.isfinite(Cs)
        Cm = float((Cs[ok] * w_strat[ok]).sum() / w_strat[ok].sum()) if ok.any() else None
        tables["per_molly_cell_zmarginal_pathweighted"].append({**key, "C_pathweighted_over_strata": Cm, "n_total": len(rows), "k_total": sum(1 for r in rows if r["detected"]),
                                                                "stratum_weights_dX": w_strat.tolist()})
        for zb in range(3):
            rz = [r for r in rows if r["zbin"] == zb]
            Cs = np.array([cell([r for r in rz if r["stratum"] == s], {})["C"] or np.nan for s in range(ns)], float); ok = np.isfinite(Cs)
            tables["per_molly_cell_x_zbin_pathweighted"].append({**key, "zbin": zb, "z": [ZBINS[zb], ZBINS[zb + 1]], "C_pathweighted_over_strata": (float((Cs[ok] * w_strat[ok]).sum() / w_strat[ok].sum()) if ok.any() else None), "n_total": len(rz)})
    pairs = [r for r in inj if r["pair_class"]]
    pair_tab = []
    if pairs:
        groups = {}
        for r in pairs:
            groups.setdefault((r["TARGETID"], r["wave"]), []).append(r)
        for (tid, wv), g in groups.items():
            g = sorted(g, key=lambda r: r["inj_idx"])
            if len(g) != 2:
                continue
            a0, a1 = g
            acc_z = _parse_list(a0["accepted_zs"]); acc_n = _parse_list(a0["accepted_nhats"])
            n_acc = len(acc_z)                                   # accepted rows on the sightline (MAX1: 0 / 1)
            # every accepted row -> the nearer absorber; at most one row per absorber (the nearest)
            assign = {}
            for i, zr in enumerate(acc_z):
                d0 = abs(zr - a0["z_inj"]) / (1 + a0["z_inj"]); d1 = abs(zr - a1["z_inj"]) / (1 + a1["z_inj"])
                j, d = (0, d0) if d0 < d1 else (1, d1)
                if d < DZ_REL and (j not in assign or d < assign[j][1]):
                    assign[j] = (i, d)
            n_matched = len(assign)                              # accepted rows matched to DISTINCT absorbers (2 needs MAX_DLAS > 1)
            # legacy single-row summaries: the accepted row of largest N-hat (== the only row under MAX1)
            zacc = a0["accepted_z"]; nacc = a0["accepted_nhat"]
            win = None; matched = False
            if n_acc:
                d0 = abs(zacc - a0["z_inj"]) / (1 + a0["z_inj"]); d1 = abs(zacc - a1["z_inj"]) / (1 + a1["z_inj"])
                win = 0 if d0 < d1 else 1
                matched = min(d0, d1) < DZ_REL
            pair_tab.append(dict(TARGETID=tid, wave=wv, pair_class=a0["pair_class"], dv_kms=float(a0["dv_kms"]), pair_logN=a0["pair_logN"], stratum=a0["stratum"],
                                 logN=[a0["logN"], a1["logN"]], z=[a0["z_inj"], a1["z_inj"]], n_accepted=n_acc, winner=win, matched_within_tol=(bool(matched) if n_acc else False),
                                 accepted_nhat=(float(nacc) if n_acc else None), max_true_logN=max(a0["logN"], a1["logN"]),
                                 n_matched=n_matched,
                                 nhat_abs0=(acc_n[assign[0][0]] if 0 in assign else None), nhat_abs1=(acc_n[assign[1][0]] if 1 in assign else None),
                                 log_sum_true_N=float(np.log10(10 ** a0["logN"] + 10 ** a1["logN"])),
                                 merged_bias_vs_max=(float(nacc - max(a0["logN"], a1["logN"])) if n_acc else None),
                                 merged_bias_vs_sum=(float(nacc - np.log10(10 ** a0["logN"] + 10 ** a1["logN"])) if n_acc else None),
                                 reported_ge20p3=(bool(nacc >= 20.3) if n_acc else False), any_true_ge20p3=bool(max(a0["logN"], a1["logN"]) >= 20.3)))
        summ = {}
        for cls in sorted({p["pair_class"] for p in pair_tab}):
            ps = [p for p in pair_tab if p["pair_class"] == cls]
            summ[cls] = dict(n_pairs=len(ps), frac_zero=float(np.mean([p["n_accepted"] == 0 for p in ps])), frac_one=float(np.mean([p["n_accepted"] == 1 for p in ps])),
                             frac_two=float(np.mean([p["n_matched"] == 2 for p in ps])),
                             frac_ge2_accepted=float(np.mean([p["n_accepted"] >= 2 for p in ps])),
                             frac_one_matched=float(np.mean([p["n_matched"] == 1 for p in ps])),
                             frac_winner_is_higher_N=float(np.mean([p["winner"] == int(np.argmax(p["logN"])) for p in ps if p["n_accepted"]])) if any(p["n_accepted"] for p in ps) else None,
                             merged_bias_vs_max_mean=float(np.nanmean([p["merged_bias_vs_max"] for p in ps if p["n_accepted"]])) if any(p["n_accepted"] for p in ps) else None,
                             merged_bias_vs_sum_mean=float(np.nanmean([p["merged_bias_vs_sum"] for p in ps if p["n_accepted"]])) if any(p["n_accepted"] for p in ps) else None,
                             frac_reported_ge20p3_given_any_true_ge20p3=(float(np.mean([p["reported_ge20p3"] for p in ps if p["any_true_ge20p3"]])) if any(p["any_true_ge20p3"] for p in ps) else None),
                             frac_reported_ge20p3_given_both_true_lt20p3=(float(np.mean([p["reported_ge20p3"] for p in ps if not p["any_true_ge20p3"]])) if any(not p["any_true_ge20p3"] for p in ps) else None))
        tables["pairs_summary"] = summ
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        commit = "unknown"
    out = dict(label=a.label, contract=dict(P_DLA_min=P_MIN, DLAFLAG=0, SNR_REDSIDE_min=SNR_MIN, dz_rel=DZ_REL, window="the sightline's Lyα window (population CSV)",
                                            molly_n_edges=[x if np.isfinite(x) else "inf" for x in MOLLY_N_EDGES], zbins=ZBINS, strata_edges=[x if np.isfinite(x) else "inf" for x in STRATA_EDGES],
                                            stratum_path_weights=w_strat.tolist()),
               n_injections=len(inj), n_single=len(single), n_pairs=len(pairs) // 2, provenance=prov, population=dict(path=a.population, sha256=_sha(a.population)),
               code_commit=commit, tables=tables)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else (o.item() if hasattr(o, "item") else str(o)))
    with open(a.out[:-5] + "_per_injection.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(inj[0])); w.headerow = None; w.writeheader(); w.writerows(inj)
    if pairs:
        with open(a.out[:-5] + "_pairs.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pair_tab[0])); w.writeheader(); w.writerows(pair_tab)
    print(f"{a.label}: {len(inj)} injections; overall detection C = {np.mean([r['detected'] for r in single]) if single else float('nan'):.3f}")
    def _f(x, fmt=".3f"):                     # None-safe (a cell with no single injections, e.g. a pairs-only wave)
        return "nan" if x is None else format(x, fmt)
    for c in tables["per_logN_point_all"]:
        print(f"  logN {c['key']['logN']:.2f}: n {c['n']:4d} C {_f(c['C'])} [{_f(c['C_lo68'])},{_f(c['C_hi68'])}] rep20.3 {c['reporting_C_T20.3']} dlogN {c['dlogN_mean']} +- {c['dlogN_std']}")
    if pairs:
        print(json.dumps(tables["pairs_summary"], indent=1))


if __name__ == "__main__":
    main()
