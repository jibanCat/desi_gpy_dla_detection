#!/usr/bin/env python
"""r041_subset_tables.py — rebuild the analyzer's per_molly_cell_x_stratum (k, n, C) table from a
per-injection CSV restricted by a predicate (e.g. candidate-free sightlines only), so the
alternative population convention can be run through track_c_tf_hz --variant r041cal
unchanged. Same cell / stratum definitions as tools/r041_analyze.py."""
from __future__ import annotations
import argparse, csv, json, hashlib, os, subprocess
import numpy as np
MOLLY_N_EDGES = [19.5, 20.0, 20.3, 20.5, 21.0, 21.5, 22.0, np.inf]
NS = 5


def jeffreys(k, n):
    from scipy.stats import beta
    if n == 0:
        return None, None
    return float(beta.ppf(0.16, k + 0.5, n - k + 0.5)), float(beta.ppf(0.84, k + 0.5, n - k + 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-injection", required=True)
    ap.add_argument("--where", required=True, help="python expression over the row dict r, e.g. \"int(r['has_cand_ge20'])==0\"")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.per_injection)) if not r.get("pair_class") and eval(a.where, {}, {"r": r})]
    tab = []
    for j in range(len(MOLLY_N_EDGES) - 1):
        lo, hi = MOLLY_N_EDGES[j], MOLLY_N_EDGES[j + 1]
        for s in range(NS):
            rs = [r for r in rows if lo <= float(r["logN"]) < hi and int(r["stratum"]) == s]
            n = len(rs); k = sum(1 for r in rs if r["detected"] == "True"); lo68, hi68 = jeffreys(k, n)
            tab.append(dict(key=dict(molly_cell=j, n_lo=lo, n_hi=(hi if np.isfinite(hi) else "inf"), stratum=s), n=n, k=k, C=(k / n if n else None), C_lo68=lo68, C_hi68=hi68))
    out = dict(label=a.label, subset_where=a.where, n_injections=len(rows), source=a.per_injection, source_sha256=hashlib.sha256(open(a.per_injection, "rb").read()).hexdigest(),
               generator_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip(),
               tables={"per_molly_cell_x_stratum": tab})
    json.dump(out, open(a.out, "w"), indent=1)
    print(a.label, len(rows), "injections;", {f"[{c['key']['n_lo']},{c['key']['n_hi']})": round(sum(x['k'] for x in tab if x['key']['molly_cell']==c['key']['molly_cell'])/max(1,sum(x['n'] for x in tab if x['key']['molly_cell']==c['key']['molly_cell'])),3) for c in tab if c['key']['stratum']==0})


if __name__ == "__main__":
    main()
