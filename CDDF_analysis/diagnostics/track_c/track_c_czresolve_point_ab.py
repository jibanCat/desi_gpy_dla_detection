#!/usr/bin/env python
"""track_c_czresolve_point_ab.py — fast POINT-only A/B for the z-resolved completeness.

Builds the Track-C forward MAP point with completeness_z_resolved OFF (the byte-identical
z-marginalized molly C) and ON (the z-resolved C·g(N,z)). Reports the per-coarse-z
dN/dX(z) and Ω(z) at >=20.0/>=20.3 vs the 2LPT-0 truth, so we can see whether the per-z
amplitude tilt FLATTENS toward 1 BEFORE paying for the full n_mc band.

NO MC band (point MAP only) — fast (~1-2 min/arm). NO GP inference. Reduce-only.

Usage:
  python CDDF_analysis/track_c_czresolve_point_ab.py [--cz-min-count 30]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    v3x_reduce, omega_hi_prefactor,
)
from CDDF_analysis.hbi.track_c_perz_band import (
    _set_forward_cfg, perz_dndx_from_fbk, perz_omega_from_fbk, truth_fNz,
    truth_perz_integrals,
)

_DEF_FORWARD = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                "track_c/stage0/forward_response_2lpt0.npz")


class _A:
    catalog_dir = AB.DEF_CAT; truth = AB.DEF_TRUTH; bal_cat = AB.DEF_BAL
    molly_tsv = AB.DEF_LYAONLY_MOLLY; kernel = AB.DEF_KERNEL
    loa0_product = AB.DEF_LOA0_PRODUCT; out = "/tmp/track_c_czresolve_ab"
    mockdir = None; zbins = "2.0,2.5,3.0,3.5"; report_limits = "20.0,20.3"
    family = "bspbody"; fit_floor = 19.5; fit_ceil = 99.0; lambda_bspbody = 30.0
    lam_rf_min = 1025.0; edge_slope_lam = 40.0; gl_nodes = 1; host_truth_floor = 19.0
    # forward band cfg knobs (point uses them via _set_forward_cfg)
    forward_model = _DEF_FORWARD; resp_family = "empirical"
    band_recenter = True; omega_slope_extrap = True; omega_slope_extrap_integrated = True
    slope_edge = 21.2; slope_fit_dex = 0.6; sigma_slope = 0.5
    cz_resolved = False; cz_min_count = 30.0


def run_point(cz_resolved, cz_min_count, limits):
    A = _A()
    A.cz_resolved = bool(cz_resolved)
    A.cz_min_count = float(cz_min_count)
    t0 = time.time()
    ing = AB.build_ingredients(A, "purity_mixture")
    cfg = ing["cfg"]
    cfg.report_logN_limits = limits
    cfg._wall1_estimator = "v3"
    _set_forward_cfg(cfg, A)
    logN_lo = ing["logN_lo"]; logN_hi = ing["logN_hi"]
    N_b = ing["N_b"]; dN_b = ing["dN_b"]
    K = omega_hi_prefactor(cfg.H0)
    base = AB.run_baseline(ing)
    e0 = base["e0"]
    fwd = e0["_v3x"]["fwd"]; family = e0["_v3x"]["family"]
    M_meta = e0["_v3x"]["M_meta"]; theta_map = e0["_v3x"]["theta_map"]
    rr = v3x_reduce(cfg, theta_map, fwd["fine"], family, M_meta)
    map_fbk = np.asarray(rr["f_bk_coarse"], float)
    dndx = {l: perz_dndx_from_fbk(map_fbk, logN_lo, dN_b, l) for l in limits}
    omega = {l: perz_omega_from_fbk(map_fbk, logN_lo, N_b, dN_b, K, l) for l in limits}
    # z-marginal (integrated headline) dN/dX & Ω
    dndx_tot = {l: float(np.sum(perz_dndx_from_fbk(map_fbk, logN_lo, dN_b, l))) for l in limits}
    # truth
    tf = truth_fNz(cfg, ing["truth_cut"], logN_lo, logN_hi, dN_b, ing["X_tot"])
    tr = truth_perz_integrals(cfg, tf["f_truth"], logN_lo, N_b, dN_b, limits)
    cnz_built = getattr(cfg, "_cnz_resolved", None) is not None
    print(f"  [{'ON ' if cz_resolved else 'OFF'}] point done {time.time()-t0:.0f}s; "
          f"cnz_built={cnz_built}")
    return dict(dndx=dndx, omega=omega, dndx_tot=dndx_tot, truth=tr,
                zc=(0.5*(np.asarray(cfg.zbins[:-1])+np.asarray(cfg.zbins[1:]))).tolist())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cz-min-count", type=float, default=30.0)
    p.add_argument("--out", default="/tmp/track_c_czresolve_ab/point_ab.json")
    args = p.parse_args(argv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    limits = (20.0, 20.3)

    print("=== OFF (z-marginalized molly C — byte-identical default) ===")
    off = run_point(False, args.cz_min_count, limits)
    print("=== ON  (z-resolved C·g(N,z)) ===")
    on = run_point(True, args.cz_min_count, limits)

    zc = off["zc"]
    for lim in limits:
        print(f"\n##### dN/dX(z) at >= {lim} #####")
        print(f"  z       truth      OFF_MAP   OFF_R0    ON_MAP    ON_R0")
        td = off["truth"]["dndx"][lim]
        for k in range(len(zc)):
            t = td[k]; o = off["dndx"][lim][k]; n = on["dndx"][lim][k]
            print(f"  {zc[k]:.2f}   {t:.5f}   {o:.5f}  {o/t:6.3f}   {n:.5f}  {n/t:6.3f}")
        r0_off = np.array([off["dndx"][lim][k]/td[k] for k in range(len(zc))])
        r0_on = np.array([on["dndx"][lim][k]/td[k] for k in range(len(zc))])
        print(f"  spread(R0): OFF {r0_off.max()-r0_off.min():.3f}   "
              f"ON {r0_on.max()-r0_on.min():.3f}")
        # integrated headline
        tt = float(np.sum(td))
        print(f"  INTEGRATED dN/dX(>= {lim}): truth {tt:.5f}  "
              f"OFF {off['dndx_tot'][lim]:.5f} (R0 {off['dndx_tot'][lim]/tt:.3f})  "
              f"ON {on['dndx_tot'][lim]:.5f} (R0 {on['dndx_tot'][lim]/tt:.3f})")

    for lim in limits:
        print(f"\n##### Omega(z) at >= {lim} #####")
        print(f"  z       truth        OFF_MAP     OFF_R0    ON_MAP      ON_R0")
        to = off["truth"]["omega"][lim]
        for k in range(len(zc)):
            t = to[k]; o = off["omega"][lim][k]; n = on["omega"][lim][k]
            print(f"  {zc[k]:.2f}   {t:.6e}  {o:.6e}  {o/t:6.3f}   {n:.6e}  {n/t:6.3f}")

    with open(args.out, "w") as fh:
        json.dump(dict(off=_jsonify(off), on=_jsonify(on)), fh, indent=2)
    print(f"\nwrote {args.out}")


def _jsonify(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {str(kk): (vv.tolist() if hasattr(vv, "tolist") else
                                {str(k2): (v2.tolist() if hasattr(v2, "tolist") else v2)
                                 for k2, v2 in vv.items()} if isinstance(vv, dict) else vv)
                      for kk, vv in v.items()}
        elif hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
