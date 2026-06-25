"""decompose_r0_zstructure.py — Phase-1 DRILL into the z-trend (H4): WHY does
R0(dN/dX>=20.3) rise 1.045 -> 1.236 -> 1.422 with z? Decompose the z-structure
into (a) z-dependent up-migration/NHI-bias [H2 x H4], (b) z-flat-C vs z-dependent
true recovery [H1 x H4], (c) the kernel's z-residual.

Reduce-only, cached kernel, NO inference. Builds the calibrated WALL-1 loa0
ingredients ONCE.

Key decomposition of the recovered dN/dX(z)>=20.3:
  recovered_dndx(z)  = [ sum_{det pred>=20.3 in z} 1/C(det) - mu_FP(z) ] / X(z)
                       (then kernel-deconvolved through the v3 parametric f(N|theta))
  truth_dndx(z)      = N_true(>=20.3, z) / X(z)

So the per-z over-recovery factor is driven by:
  - n_pred>=20.3 / n_true>=20.3  (the RAW count migration ratio, per z)  [H2]
  - the 1/C inflation per z (z-flat C vs z-true recovery)               [H1]
  - whether the kernel removes the net promotion at that z              [kernel]
We measure each at the COUNT level (pre-parametric) so the v3 fit is bypassed
and the raw selection arithmetic is exposed; then compare to the v3 recovered
dN/dX(z) to attribute the residual to the kernel.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi.ab_loa0_fp_baseline import (
    build_ingredients, DEF_CAT, DEF_TRUTH, DEF_BAL, DEF_KERNEL, DEF_LOA0_PRODUCT,
)
from CDDF_analysis.hbi.cddf_tilt_closure import baseline_recovery
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    _zbin_index, make_C_interpolator, C_FLOOR,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir", default=DEF_CAT)
    p.add_argument("--truth", default=DEF_TRUTH)
    p.add_argument("--bal-cat", default=DEF_BAL)
    p.add_argument("--molly-tsv", default=None)
    p.add_argument("--kernel", default=DEF_KERNEL)
    p.add_argument("--loa0-product", default=DEF_LOA0_PRODUCT)
    p.add_argument("--out", default="/tmp/decompose_r0")
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
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    limits = tuple(float(x) for x in args.report_limits.split(","))
    LIM = 20.3

    ing = build_ingredients(args, "loa0", loa0_product=args.loa0_product)
    cfg = ing["cfg"]; cfg.report_logN_limits = limits; cfg._wall1_estimator = "v3"
    cat_cut = ing["cat_cut"]; truth_cut = ing["truth_cut"]
    mm = ing["mm"]; X_tot = np.asarray(ing["X_tot"], float)
    zbins = np.asarray(cfg.zbins, float); n_zb = len(zbins) - 1
    zc = 0.5 * (zbins[:-1] + zbins[1:])
    C_interp = make_C_interpolator(mm)

    # v3 recovered dN/dX(z) (the actual estimator, kernel ON)
    base = baseline_recovery(
        cfg, cat_cut, ing["is_TP"], ing["good_mask"], truth_cut,
        ing["C_interp"], ing["fp_model"], X_tot, ing["logN_lo"], ing["logN_hi"],
        ing["N_b"], ing["dN_b"], estimator_fn=ing["estimator_fn"])
    e0 = base["e0"]; t0 = base["t0"]
    v3_ez = np.asarray(e0["dndx_z"][LIM], float)
    truth_ez = np.asarray(t0["dndx_z"][LIM], float)
    v3_R0 = np.where(truth_ez > 0, v3_ez / truth_ez, np.nan)

    # operating set
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & ing["good_mask"]
    nhi_pred = np.asarray(cat_cut["NHI"], float)[op]
    nhi_true = np.asarray(cat_cut["NHI_TRUE"], float)[op]
    z_pred = np.asarray(cat_cut["Z_DLA"], float)[op]
    snr_op = s2n[op]
    is_tp = np.isfinite(nhi_true)
    zidx_pred = _zbin_index(z_pred, zbins)

    # truth
    t_nhi = np.asarray(truth_cut["NHI"], float)
    t_z = np.asarray(truth_cut["Z_DLA"], float)
    t_snr = np.asarray(truth_cut["S2N_RED"], float)
    tk = t_snr > cfg.snr_min
    t_nhi, t_z = t_nhi[tk], t_z[tk]
    t_zidx = _zbin_index(t_z, zbins)

    print("=" * 92)
    print("Z-STRUCTURE DECOMPOSITION of R0(dN/dX>=20.3) — count-level arithmetic per z")
    print("=" * 92)
    print(f"  v3 recovered R0(z): " + ", ".join(f"z{zc[k]:.2f}={v3_R0[k]:.4f}" for k in range(n_zb)))
    print()
    hdr = (f"  {'z':>5} {'N_true':>7} {'N_pred':>7} {'rawN_ratio':>10} "
           f"{'sum1/C':>9} {'1/C_R0':>8} {'<C_app>':>8} {'true_rec':>8} "
           f"{'NHIbias':>8} {'v3_R0':>7}")
    print(hdr)
    rows = []
    for k in range(n_zb):
        # truth counts >=20.3 in z
        N_true = int(((t_nhi >= LIM) & (t_zidx == k)).sum())
        # detections pred>=20.3 in z (ALL op, TP+FP — same set the estimator sums)
        sel_det = (nhi_pred >= LIM) & (zidx_pred == k)
        N_pred = int(sel_det.sum())
        rawN_ratio = N_pred / N_true if N_true > 0 else np.nan
        # the 1/C-weighted count the estimator forms (pre-FP, pre-kernel):
        # sum_{det pred>=20.3 in z} 1/C(N_pred, SNR)
        C_app = C_interp(nhi_pred[sel_det], snr_op[sel_det])
        sum_invC = float(np.sum(1.0 / np.clip(C_app, C_FLOOR, None)))
        # the pure 1/C selection-corrected count-rate R0 (NO kernel, NO FP subtraction):
        # (sum 1/C) / N_true  -- isolates raw migration x 1/C inflation
        invC_R0 = sum_invC / N_true if N_true > 0 else np.nan
        mean_C = float(np.mean(C_app)) if sel_det.sum() else np.nan
        # empirical true recovery: matched TP whose TRUE host>=20.3 in this z / N_true
        tp_true_ge = (nhi_true >= LIM) & is_tp & (_zbin_index(z_pred, zbins) == k)
        n_rec = int(tp_true_ge.sum())
        true_rec = n_rec / N_true if N_true > 0 else np.nan
        # NHI bias on TP true>=20.3 in this z
        sel_b = (nhi_true >= LIM) & is_tp & (_zbin_index(z_pred, zbins) == k)
        bias = float(np.median(nhi_pred[sel_b] - nhi_true[sel_b])) if sel_b.sum() else np.nan
        rows.append((zc[k], N_true, N_pred, rawN_ratio, sum_invC, invC_R0,
                     mean_C, true_rec, bias, v3_R0[k]))
        print(f"  {zc[k]:>5.2f} {N_true:>7d} {N_pred:>7d} {rawN_ratio:>10.4f} "
              f"{sum_invC:>9.1f} {invC_R0:>8.4f} {mean_C:>8.4f} {true_rec:>8.4f} "
              f"{bias:>+8.4f} {v3_R0[k]:>7.4f}")

    print("\n  LEGEND:")
    print("   rawN_ratio = N_detected(pred>=20.3) / N_true(>=20.3)  [pure count migration, H2]")
    print("   1/C_R0     = (sum 1/C over pred>=20.3 det) / N_true   [migration x 1/C inflation, no kernel/FP]")
    print("   true_rec   = matched TP w/ true host>=20.3 / N_true   [the EMPIRICAL completeness]")
    print("   <C_app>    = mean molly C applied to pred>=20.3 det   [z-FLAT by construction]")
    print("   v3_R0      = the actual recovered/truth (kernel ON)   [should < 1/C_R0 IF kernel removes up-migration]")

    print("\n" + "=" * 92)
    print("ATTRIBUTION per z (decompose 1/C_R0 and the kernel residual):")
    print("=" * 92)
    for r in rows:
        zk, N_true, N_pred, rawN, sinvC, invC_R0, meanC, truerec, bias, v3R0 = r
        # raw count migration component
        mig = rawN                       # N_pred/N_true
        # 1/C inflation ON TOP of raw migration: invC_R0 / rawN = <1/C>_eff
        invC_infl = invC_R0 / rawN if rawN > 0 else np.nan
        # kernel residual = v3_R0 / invC_R0 (how much the kernel deconvolution changed the
        # pure 1/C count-rate). <1 means kernel REMOVED some up-migration; ~1 means it didn't.
        kern = v3R0 / invC_R0 if invC_R0 > 0 else np.nan
        print(f"  z={zk:.2f}: v3_R0={v3R0:.3f} = rawN_mig({mig:.3f}) "
              f"x 1/C_infl({invC_infl:.3f}) x kernel_resid({kern:.3f})")
    with open(os.path.join(args.out, "h4_zstructure.tsv"), "w") as fh:
        fh.write("z_center\tN_true\tN_pred\trawN_ratio\tsum_invC\tinvC_R0\tmean_C\t"
                 "true_rec\tNHI_bias\tv3_R0\n")
        for r in rows:
            fh.write("\t".join(f"{v:.6g}" for v in r) + "\n")
    print(f"\n  TSV -> {os.path.join(args.out, 'h4_zstructure.tsv')}")
    return rows


if __name__ == "__main__":
    main()
