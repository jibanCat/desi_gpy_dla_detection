#!/usr/bin/env python
"""r041_cell_compare.py — CELL-LEVEL comparison of two independent injection samples (different sightlines / draws), for
the P1 legs that cannot be paired by injection: mean-flux arms vs the P0 fiducial, London vs 2LPT random
(MAX4_P1_GATES_2026-09-02.md §B-§C). Cells = log N point x S/N stratum (the analyzer's per_injection tables); per cell
dC = C_B - C_A with a two-sample bootstrap (B = 4000, seed 20260902) and Jeffreys intervals; PRIMARY = production-weighted
dC_w over the leveraged points (20.3, 20.5, 21.0 -> molly cells) with the frozen weights g(N) x s(stratum) (candidate
factor 1); tiers with the 3 / 5 pp anchors: INSENSITIVE |dC_w| <= 0.03 & 95% bound < 0.05; BOUNDED |dC_w| <= thr_b
(default 0.05) & 95% within +-0.08; else MATERIAL."""
from __future__ import annotations
import argparse, csv, json, os
import numpy as np

LEV = {20.3: "1", 20.5: "2", 21.0: "3"}     # log N point -> molly cell key of gate_weights.json


def load(paths):
    rows = []
    for p in paths:
        for r in csv.DictReader(open(p)):
            rows.append(dict(logN=float(r["logN"]), stratum=int(r["stratum"]), y=1 if r["detected"] == "True" else 0,
                             dN=(float(r["nhat"]) - float(r["logN"])) if r["detected"] == "True" and r["nhat"] not in ("", "nan") else np.nan))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True); ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--a-label", default="A"); ap.add_argument("--b-label", default="B")
    ap.add_argument("--weights", required=True); ap.add_argument("--n-boot", type=int, default=4000); ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--bounded-thr", type=float, default=0.05); ap.add_argument("--out", required=True); ap.add_argument("--label", default="cell")
    a = ap.parse_args(argv)
    A, B = load(a.a), load(a.b); W = json.load(open(a.weights)); g = W["g_cell"]; s_str = W["s_stratum"]
    rng = np.random.default_rng(a.seed)
    cells = sorted({(r["logN"], r["stratum"]) for r in A} & {(r["logN"], r["stratum"]) for r in B})
    ya = {c: np.array([r["y"] for r in A if (r["logN"], r["stratum"]) == c]) for c in cells}
    yb = {c: np.array([r["y"] for r in B if (r["logN"], r["stratum"]) == c]) for c in cells}
    da = {c: np.array([r["dN"] for r in A if (r["logN"], r["stratum"]) == c and r["dN"] == r["dN"]]) for c in cells}
    db = {c: np.array([r["dN"] for r in B if (r["logN"], r["stratum"]) == c and r["dN"] == r["dN"]]) for c in cells}
    lev = [c for c in cells if c[0] in LEV and len(ya[c]) >= 5 and len(yb[c]) >= 5]
    w = {c: float(g[LEV[c[0]]]) * float(s_str[c[1]]) for c in lev}
    def dcw(sample=False):
        num = den = 0.0
        for c in lev:
            xa, xb = ya[c], yb[c]
            if sample:
                xa = xa[rng.integers(0, xa.size, xa.size)]; xb = xb[rng.integers(0, xb.size, xb.size)]
            num += w[c] * (xb.mean() - xa.mean()); den += w[c]
        return num / den if den else np.nan
    point = dcw(); boots = np.array([dcw(True) for _ in range(a.n_boot)])
    ci = [float(np.percentile(boots, q)) for q in (16, 84, 2.5, 97.5)]
    b95 = max(abs(ci[2]), abs(ci[3]))
    tier = "INSENSITIVE" if abs(point) <= 0.03 and b95 < 0.05 else ("BOUNDED" if abs(point) <= a.bounded_thr and b95 <= 0.08 else "MATERIAL")
    per = {}
    for c in cells:
        xa, xb = ya[c], yb[c]
        bs = np.array([xb[rng.integers(0, xb.size, xb.size)].mean() - xa[rng.integers(0, xa.size, xa.size)].mean() for _ in range(1000)])
        per[f"logN={c[0]}|stratum={c[1]}"] = dict(nA=int(xa.size), nB=int(xb.size), C_A=float(xa.mean()), C_B=float(xb.mean()), dC=float(xb.mean() - xa.mean()),
                                                  ci68=[float(np.percentile(bs, 16)), float(np.percentile(bs, 84))], dN_A=float(np.mean(da[c])) if da[c].size else None,
                                                  dN_B=float(np.mean(db[c])) if db[c].size else None, leveraged=c in lev)
    # per log N point (all strata pooled)
    pts = {}
    for p in sorted({c[0] for c in cells}):
        xa = np.concatenate([ya[c] for c in cells if c[0] == p]); xb = np.concatenate([yb[c] for c in cells if c[0] == p])
        pts[str(p)] = dict(nA=int(xa.size), nB=int(xb.size), C_A=float(xa.mean()), C_B=float(xb.mean()), dC=float(xb.mean() - xa.mean()))
    out = dict(label=a.label, a=a.a_label, b=a.b_label, production_weighted=dict(dC_w=float(point), ci68=ci[:2], ci95=ci[2:], tier=tier, n_leveraged_cells=len(lev), n_boot=a.n_boot, seed=a.seed),
               per_point=pts, per_cell=per, weights=dict(g_cell=g, s_stratum=s_str))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(dict(production_weighted=out["production_weighted"], per_point=pts), indent=1))


if __name__ == "__main__":
    main()
