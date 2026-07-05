#!/usr/bin/env python
"""track_c_shape_diag.py — DIAGNOSIS ONLY (read/reduce; no inference, no commits).

PI hypothesis: the recovered f(N) under-recovers at the LOW-N edge (~logN 20.0) AND the
HIGH-N tail (>21) because the bspbody shape prior (the penalized B-spline 2nd-difference
curvature penalty + the floor-edge slope anchor) imposes a curvature that bends the fit
DOWNWARD at the edges.

This rebuilds the POINT (MAP) f(N) ONLY (no MC band) and sweeps:
  (2) v3_lambda_bspbody  : 30 (default) -> 10 -> 3 -> 0     [curvature penalty]
  (3) v3_bspbody_edge_slope_target / _lam : -1.4 / measured / OFF  [edge-slope anchor]
  (4) v3_family          : bspbody vs bplcut vs plawcut vs plaw

For each setting it prints the per-0.1-dex-bin f(N) MAP/truth ratio over [20.0,22.0],
plus per-edge summary ratios (low-N [20.0,20.2), shoulder [21.0,21.5), deep [21.5,22.0)).

Build the ingredients ONCE (expensive: catalog load + molly + pathlength); re-fit MAP
per setting (cheap).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
from CDDF_analysis.hbi.ab_loa0_fp_baseline import build_ingredients, run_baseline


def _point_fN(ing):
    """Re-build forward + re-fit MAP at the current cfg; return (mid, f_map, f_truth)."""
    base = run_baseline(ing)
    e0 = base["e0"]; t0 = base["t0"]
    mid = 0.5 * (ing["logN_lo"] + ing["logN_hi"])
    return (mid, np.asarray(e0["f_b"], float), np.asarray(t0["f_truth"], float),
            base)


def _edge_summary(mid, fmap, ftru):
    """Geometric-mean ratio (and per-bin) over named edges (avoids huge-bin dominance)."""
    def gmean_ratio(lo, hi):
        sel = (mid >= lo - 1e-9) & (mid < hi - 1e-9) & (ftru > 0) & np.isfinite(fmap)
        if not sel.any():
            return np.nan, 0
        r = fmap[sel] / ftru[sel]
        r = r[r > 0]
        if r.size == 0:
            return np.nan, 0
        return float(np.exp(np.mean(np.log(r)))), int(r.size)
    bands = [("lowedge[20.0,20.2)", 20.0, 20.2),
             ("body[20.2,21.0)", 20.2, 21.0),
             ("shoulder[21.0,21.5)", 21.0, 21.5),
             ("deep[21.5,22.0)", 21.5, 22.0)]
    return {name: gmean_ratio(lo, hi) for name, lo, hi in bands}


def _measure_truth_lowN_slope(mid, ftru, lo=19.7, hi=20.3):
    """Local d(log10 f)/d(logN) of TRUTH over [lo,hi] — the data-implied edge slope to
    compare against the -1.4 anchor target."""
    sel = (mid >= lo - 1e-9) & (mid < hi + 1e-9) & (ftru > 0)
    x = mid[sel]; y = np.log10(ftru[sel])
    if x.size < 2:
        return np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)


def _print_table(label, mid, fmap, ftru):
    print(f"\n----- {label} -----")
    print(f"{'logN':>6} {'fmap':>11} {'ftru':>11} {'ratio':>7}")
    for x, fm, ft in zip(mid, fmap, ftru):
        if not (20.0 - 1e-9 <= x <= 22.0 + 1e-9):
            continue
        r = (fm / ft) if ft > 0 else np.nan
        flag = ""
        if np.isfinite(r):
            if r < 0.90:
                flag = " LOW"
            elif r > 1.10:
                flag = " HIGH"
        print(f"{x:6.2f} {fm:11.4e} {ft:11.4e} {r:7.3f}{flag}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--catalog-dir", default=AB.DEF_CAT)
    p.add_argument("--truth", default=AB.DEF_TRUTH)
    p.add_argument("--bal-cat", default=AB.DEF_BAL)
    p.add_argument("--molly-tsv", default=AB.DEF_LYAONLY_MOLLY)
    p.add_argument("--kernel", default=AB.DEF_KERNEL)
    p.add_argument("--loa0-product", default=AB.DEF_LOA0_PRODUCT)
    p.add_argument("--out", default="/tmp/track_c_shape_diag")
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3")
    # headline defaults
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--fit-ceil", type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes", type=int, default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    print("=" * 78)
    print("TRACK-C f(N) SHAPE-PRIOR DIAGNOSIS — POINT (MAP) only, purity_mixture")
    print("=" * 78)
    ing = build_ingredients(args, "purity_mixture", loa0_product=args.loa0_product)
    cfg = ing["cfg"]
    print(f"[ingredients built in {time.time()-t0:.0f}s]")

    results = {}

    # ---- 0. HEADLINE (reproduce the JSON point) ----
    mid, fmap0, ftru, base = _point_fN(ing)
    tru_slope = _measure_truth_lowN_slope(mid, ftru, 19.7, 20.3)
    tru_slope_2002 = _measure_truth_lowN_slope(mid, ftru, 20.0, 20.4)
    print(f"\n[truth low-N slope d log10 f / d logN]  [19.7,20.3]={tru_slope:.3f}  "
          f"[20.0,20.4]={tru_slope_2002:.3f}   (anchor target = "
          f"{cfg.v3_bspbody_edge_slope_target})")
    _print_table("HEADLINE bspbody lam=30 edge_lam=40 target=-1.4", mid, fmap0, ftru)
    results["headline"] = _edge_summary(mid, fmap0, ftru)
    print("edge summary:", {k: f"{v[0]:.3f}(n={v[1]})" for k, v in results["headline"].items()})

    # ---- 2. CURVATURE PENALTY sweep ----
    print("\n" + "#" * 78)
    print("# TEST 2 — curvature penalty v3_lambda_bspbody sweep (edge anchor at default 40)")
    print("#" * 78)
    for lam in (30.0, 10.0, 3.0, 0.0):
        cfg.v3_lambda_bspbody = lam
        mid, fmap, ftru, _ = _point_fN(ing)
        es = _edge_summary(mid, fmap, ftru)
        results[f"lam{lam}"] = es
        print(f"\nlam={lam:>5}:  " +
              "  ".join(f"{k.split('[')[0]}={v[0]:.3f}" for k, v in es.items()))
        _print_table(f"lam={lam}", mid, fmap, ftru)
    cfg.v3_lambda_bspbody = 30.0  # restore

    # ---- 3. EDGE-SLOPE ANCHOR sweep ----
    print("\n" + "#" * 78)
    print("# TEST 3 — edge-slope anchor (target / lam) sweep (lam_bspbody at default 30)")
    print("#" * 78)
    anchor_settings = [
        ("target=-1.4 lam=40 (default)", -1.4, 40.0),
        ("target=meas lam=40", float(tru_slope), 40.0),
        ("target=-1.4 lam=10", -1.4, 10.0),
        ("anchor OFF lam=0", -1.4, 0.0),
    ]
    for name, tgt, elam in anchor_settings:
        cfg.v3_bspbody_edge_slope_target = tgt
        cfg.v3_bspbody_edge_slope_lam = elam
        mid, fmap, ftru, _ = _point_fN(ing)
        es = _edge_summary(mid, fmap, ftru)
        results[f"anchor:{name}"] = es
        print(f"\n{name}:  " +
              "  ".join(f"{k.split('[')[0]}={v[0]:.3f}" for k, v in es.items()))
        _print_table(name, mid, fmap, ftru)
    cfg.v3_bspbody_edge_slope_target = -1.4   # restore
    cfg.v3_bspbody_edge_slope_lam = 40.0

    # ---- 4. FAMILY comparison ----
    print("\n" + "#" * 78)
    print("# TEST 4 — family comparison (bspbody vs bplcut vs plawcut vs plaw)")
    print("#" * 78)
    for fam in ("bspbody", "bplcut", "plawcut", "plaw"):
        cfg.v3_family = fam
        mid, fmap, ftru, _ = _point_fN(ing)
        es = _edge_summary(mid, fmap, ftru)
        results[f"family:{fam}"] = es
        print(f"\nfamily={fam:>8}:  " +
              "  ".join(f"{k.split('[')[0]}={v[0]:.3f}" for k, v in es.items()))
        _print_table(f"family={fam}", mid, fmap, ftru)
    cfg.v3_family = "bspbody"  # restore

    out = os.path.join(args.out, "shape_diag.json")
    with open(out, "w") as fh:
        json.dump({"truth_lowN_slope_19.7_20.3": tru_slope,
                   "truth_lowN_slope_20.0_20.4": tru_slope_2002,
                   "edge_summaries": {k: {kk: list(vv) for kk, vv in v.items()}
                                      for k, v in results.items()}}, fh, indent=2)
    print(f"\n[done {time.time()-t0:.0f}s] saved -> {out}")


if __name__ == "__main__":
    main()
