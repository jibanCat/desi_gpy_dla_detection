"""track_c_bref_r0check.py — PART 4 (AFTER-THE-FACT, separately labeled):
R0(z) check under each NON-CIRCULAR re-center recipe.

We build the Stage-0 ZNZ model from the truth-match ONLY (measure_znz_response +
fit_znz_model), then form recipe variants that DIFFER ONLY in which functional of
the conditional dx distribution the re-center targets:

  MEAN_bref0   : b = E[dx|xhat,z]   (lstsq fit, b_ref=0)   -> m_tgt = xhat - mean(dx)
  MEDIAN_bref0 : b = median(dx|.)   (quantile fit, b_ref=0)-> m_tgt = xhat - median(dx)
  MEAN_brefmed : b = E[dx|.], b_ref = median-at-ref        -> partial mean shift
  OFF          : no znz (broaden012 frozen headline)

All three are FIXED from the truth-match alone. R0(z) is computed ONLY as an
after-the-fact CHECK and is NOT used to choose anything.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
from numpy.polynomial.polynomial import polyvander2d

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.ab_loa0_fp_baseline import (
    build_ingredients, DEF_CAT, DEF_TRUTH, DEF_BAL, DEF_KERNEL, DEF_LOA0_PRODUCT,
)
from CDDF_analysis.cddf_tilt_closure import baseline_recovery
from CDDF_analysis.cddf_catalog_hbi import build_fine_grid, _fine_z_grid
from CDDF_analysis import znz_kernel as Z


def _fit_quantile_2d(xhat, z, dx, x_ref, z_ref, deg_x, deg_z, q=0.5, n_iter=30):
    """IRLS quantile (median when q=0.5) polynomial fit of dx ~ poly(xhat,z)."""
    V = polyvander2d(xhat - x_ref, z - z_ref, [deg_x, deg_z])
    # init with OLS
    coef, _, _, _ = np.linalg.lstsq(V, dx, rcond=None)
    for _ in range(n_iter):
        r = dx - V @ coef
        w = np.where(r >= 0, q, 1.0 - q) / np.maximum(np.abs(r), 1e-3)
        Vw = V * w[:, None]
        coef, _, _, _ = np.linalg.lstsq(Vw, dx * w, rcond=None)
    return coef


def build_models(ing, cfg, deg_xhat=1, deg_z=2):
    """Return dict of ZNZModel variants fitted from the truth-match ONLY."""
    cat_cut = ing["cat_cut"]; good_mask = ing["good_mask"]
    mm = ing["mm"]
    fine_grid = (ing["logN_lo"], ing["logN_hi"], ing["N_b"], ing["dN_b"])
    meas = Z.measure_znz_response(cat_cut, good_mask, cfg, mm, fine_grid,
                                  z_covariate="z_dla", host_col="NHI_TILT_HOST")
    xhat = np.asarray(meas["xhat"], float)
    z = np.asarray(meas["z"], float)
    dx = np.asarray(meas["dx"], float)
    x_ref = float(np.median(xhat)); z_ref = float(np.median(z))

    # --- scatter surface shared (fit to |dx - mean_pred|), used by all variants ---
    Vfull = polyvander2d(xhat - x_ref, z - z_ref, [deg_xhat, deg_z])
    b_mean = np.linalg.lstsq(Vfull, dx, rcond=None)[0]
    sig_coef = np.linalg.lstsq(Vfull, np.abs(dx - Vfull @ b_mean), rcond=None)[0]
    V_ref = polyvander2d(np.array([0.0]), np.array([0.0]), [deg_xhat, deg_z])
    sig_ref = float(np.clip((V_ref @ sig_coef)[0], 1e-4, None))

    b_median = _fit_quantile_2d(xhat, z, dx, x_ref, z_ref, deg_xhat, deg_z, q=0.5)

    def _mk(b_coef, b_ref):
        return Z.ZNZModel(b_coef=b_coef, sig_coef=sig_coef, xhat_ref=x_ref,
                          z_ref=z_ref, b_ref=float(b_ref), sig_ref=sig_ref,
                          z_covariate="z_dla", deg_xhat=deg_xhat, deg_z=deg_z)

    b_ref_mean = float((V_ref @ b_mean)[0])       # mean-at-ref
    b_ref_med = float((V_ref @ b_median)[0])      # median-at-ref
    print(f"  fit: x_ref={x_ref:.3f} z_ref={z_ref:.3f}  "
          f"mean-at-ref={b_ref_mean:+.4f}  median-at-ref={b_ref_med:+.4f}  "
          f"sig_ref={sig_ref:.4f}")
    return {
        "MEAN_bref0":   _mk(b_mean, 0.0),
        "MEDIAN_bref0": _mk(b_median, 0.0),
        "MEAN_brefmed": _mk(b_mean, b_ref_mean),   # = subtract only (mean - mean_ref)
    }


def _save_tmp(m, cnz_src_path, out, name):
    """Save a ZNZ NPZ carrying model m and a passthrough CNZ (g==1 so completeness
    z-correction is OFF, isolating the re-center effect)."""
    # build a unity CNZModel on the matching molly+fine-z grid
    d = np.load(cnz_src_path, allow_pickle=True) if cnz_src_path and os.path.exists(cnz_src_path) else None
    if d is not None and "g_grid" in d:
        cnz = Z.CNZModel(g_grid=np.ones_like(d["g_grid"]), nhi_edges=d["nhi_edges"],
                         z_edges_fine=d["z_edges_fine"])
    else:
        cnz = None
    path = os.path.join(out, f"znz_{name}.npz")
    if cnz is None:
        # save a minimal stub: still needs cnz for load_znz; build trivial grid
        raise SystemExit("need a cnz source npz to build the unity g grid")
    Z.save_znz(path, m, cnz)
    return path


def run_recipe(args, recipe_name, znz_path):
    """Run baseline_recovery with cfg.kernel_znz_model set; return R0_dndx(z) at 20.3."""
    ing = build_ingredients(args, "loa0", loa0_product=args.loa0_product)
    cfg = ing["cfg"]; cfg._wall1_estimator = "v3"
    cfg.report_logN_limits = tuple(float(x) for x in args.report_limits.split(","))
    if znz_path is not None:
        cfg.kernel_znz_model = znz_path
        cfg.c_nz_model = None  # isolate re-center; completeness z-corr OFF
    base = baseline_recovery(
        cfg, ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["truth_cut"],
        ing["C_interp"], ing["fp_model"], ing["X_tot"], ing["logN_lo"],
        ing["logN_hi"], ing["N_b"], ing["dN_b"], estimator_fn=ing["estimator_fn"])
    return base


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--catalog-dir", default=DEF_CAT)
    p.add_argument("--truth", default=DEF_TRUTH)
    p.add_argument("--bal-cat", default=DEF_BAL)
    p.add_argument("--molly-tsv", default=None)
    p.add_argument("--kernel", default=DEF_KERNEL)
    p.add_argument("--loa0-product", default=DEF_LOA0_PRODUCT)
    p.add_argument("--out", default="/tmp/track_c_bref_r0")
    p.add_argument("--cnz-src", default=None,
                   help="an existing znz NPZ to copy the cnz grid shape from (g set to 1)")
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3,20.6")
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--fit-ceil", type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes", type=int, default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    p.add_argument("--recipes", default="OFF,MEAN_bref0,MEDIAN_bref0,MEAN_brefmed")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    limits = tuple(float(x) for x in args.report_limits.split(","))
    zbins = tuple(float(x) for x in args.zbins.split(","))
    zc = 0.5 * (np.array(zbins[:-1]) + np.array(zbins[1:]))

    # build models once
    ing0 = build_ingredients(args, "loa0", loa0_product=args.loa0_product)
    cfg0 = ing0["cfg"]
    models = build_models(ing0, cfg0)

    # need a cnz source: build_znz cache? fall back to building a unity grid from molly+fine-z
    if args.cnz_src is None:
        # construct a unity cnz directly from molly + fine-z grid
        mm = ing0["mm"]; zef = _fine_z_grid(cfg0)
        n_nhi = len(mm.nhi_edges) - 1; n_zf = len(zef) - 1
        unity = Z.CNZModel(g_grid=np.ones((n_nhi, n_zf)), nhi_edges=mm.nhi_edges,
                           z_edges_fine=zef)
        cnz_use = unity
    else:
        d = np.load(args.cnz_src, allow_pickle=True)
        cnz_use = Z.CNZModel(g_grid=np.ones_like(d["g_grid"]), nhi_edges=d["nhi_edges"],
                             z_edges_fine=d["z_edges_fine"])

    requested = args.recipes.split(",")
    results = {}
    for name in requested:
        if name == "OFF":
            znz_path = None
        else:
            znz_path = os.path.join(args.out, f"znz_{name}.npz")
            Z.save_znz(znz_path, models[name], cnz_use)
        print("\n" + "=" * 80)
        print(f"[recipe] {name}  (znz={'OFF' if znz_path is None else os.path.basename(znz_path)})")
        print("=" * 80)
        base = run_recipe(args, name, znz_path)
        LIM = 20.3
        R0z = np.asarray(base["R0_dndx_z"][LIM], float)
        results[name] = dict(
            R0_dndx={lim: float(base["R0_dndx_total"][lim]) for lim in limits},
            R0_omega={lim: float(base["R0_omega"][lim]) for lim in limits},
            R0z_dndx_203=R0z.tolist())
        print(f"  R0_dndx 20.0/20.3/20.6 = " +
              "/".join(f"{results[name]['R0_dndx'][l]:.4f}" for l in limits))
        print(f"  R0_omega 20.0/20.3/20.6 = " +
              "/".join(f"{results[name]['R0_omega'][l]:.4f}" for l in limits))
        print(f"  R0(dN/dX>=20.3) by z " +
              ", ".join(f"z{zc[k]:.2f}={R0z[k]:.4f}" for k in range(len(zc))))

    # summary table
    print("\n" + "#" * 80)
    print("# AFTER-THE-FACT R0 CHECK SUMMARY (NOT used to choose the recipe)")
    print("#" * 80)
    print(f"{'recipe':>14} | {'R0dndx20.3':>10} {'R0om20.3':>9} | " +
          " ".join(f"R0z_{z:.2f}" for z in zc))
    for name in requested:
        r = results[name]
        print(f"{name:>14} | {r['R0_dndx'][20.3]:>10.4f} {r['R0_omega'][20.3]:>9.4f} | " +
              " ".join(f"{v:8.4f}" for v in r["R0z_dndx_203"]))
    with open(os.path.join(args.out, "part4_r0check.tsv"), "w") as fh:
        fh.write("recipe\tR0dndx_20.0\tR0dndx_20.3\tR0dndx_20.6\t"
                 "R0om_20.0\tR0om_20.3\tR0om_20.6\t" +
                 "\t".join(f"R0z_{z:.2f}" for z in zc) + "\n")
        for name in requested:
            r = results[name]
            fh.write(name + "\t" +
                     "\t".join(f"{r['R0_dndx'][l]:.6g}" for l in limits) + "\t" +
                     "\t".join(f"{r['R0_omega'][l]:.6g}" for l in limits) + "\t" +
                     "\t".join(f"{v:.6g}" for v in r["R0z_dndx_203"]) + "\n")
    print(f"\n[done] -> {os.path.join(args.out, 'part4_r0check.tsv')}")
    return results


if __name__ == "__main__":
    main()
