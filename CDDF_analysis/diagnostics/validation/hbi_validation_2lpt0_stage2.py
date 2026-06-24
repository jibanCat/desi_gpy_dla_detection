"""Stage II validation on 2LPT-0: independent-Beta vs shared-bootstrap calibration
nuisance band, at mc_inner=laplace (the full faithful inner structure), n_mc=200,
on BOTH the loa0 (frozen-background) and purity_mixture FP paths.

Reduce-only / analysis-side. NO GP inference. Builds the calibrated WALL-1 ingredient
bundle ONCE per FP estimator (the exact partA bundle) and runs each band under
cfg.mc_nuisance in {'indep','shared_boot'}; reports the band shape/width change and
truth coverage, and writes the comparison figure + an npz.

Usage (env: conda activate gpdla; BLAS pinned; <=4 workers):
  python CDDF_analysis/hbi_validation_2lpt0_stage2.py --n-mc 200 --out <dir>
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import cddf_catalog_hbi as H
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    truth_reductions, joint_mc_errors, make_v3x_refit_fn,
)
from CDDF_analysis.hbi.ab_loa0_fp_baseline import build_ingredients
from CDDF_analysis.hbi.wall1_explain_partA import loa0_full_posterior_mc


def _band(samples, lo=16, hi=84):
    return (float(np.nanpercentile(samples, lo)),
            float(np.nanpercentile(samples, 50)),
            float(np.nanpercentile(samples, hi)),
            float(np.nanpercentile(samples, 2.5)),
            float(np.nanpercentile(samples, 97.5)))


def run_loa0(args, limits, seed):
    print("=" * 70)
    print("[stage2-val] loa0 ingredients (calibrated WALL-1 bundle, kernel ON)")
    ing = build_ingredients(args, "loa0", loa0_product=args.loa0_product)
    cfg = ing["cfg"]; cfg.report_logN_limits = limits
    cfg._wall1_estimator = "v3"; cfg.mc_inner = "laplace"
    logN_lo, logN_hi = ing["logN_lo"], ing["logN_hi"]
    N_b, dN_b, X_tot = ing["N_b"], ing["dN_b"], ing["X_tot"]
    point = ing["estimator_fn"](
        ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["C_interp"],
        ing["fp_model"], X_tot, logN_lo, logN_hi, N_b, dN_b, ing["truth_cut"], cfg)
    tr = truth_reductions(cfg, ing["truth_cut"], logN_lo, logN_hi, N_b, dN_b, X_tot)
    out = {"point": point, "truth": tr, "limits": limits}
    for mc_nuisance in ("indep", "shared_boot"):
        cfg.mc_nuisance = mc_nuisance
        t0 = time.time()
        full = loa0_full_posterior_mc(cfg, ing, point, args.n_mc,
                                      np.random.default_rng(seed + 3))
        print(f"    loa0 {mc_nuisance:12s} band done ({time.time()-t0:.0f}s)")
        out[mc_nuisance] = full
    return out


def run_pm(args, limits, seed):
    print("=" * 70)
    print("[stage2-val] purity_mixture ingredients")
    ing = build_ingredients(args, "purity_mixture")
    cfg = ing["cfg"]; cfg.report_logN_limits = limits
    cfg._wall1_estimator = "v3"; cfg.mc_inner = "laplace"; cfg.n_mc = args.n_mc
    logN_lo, logN_hi = ing["logN_lo"], ing["logN_hi"]
    N_b, dN_b, X_tot = ing["N_b"], ing["dN_b"], ing["X_tot"]
    point = ing["estimator_fn"](
        ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["C_interp"],
        ing["fp_model"], X_tot, logN_lo, logN_hi, N_b, dN_b, ing["truth_cut"], cfg)
    tr = truth_reductions(cfg, ing["truth_cut"], logN_lo, logN_hi, N_b, dN_b, X_tot)
    out = {"point": point, "truth": tr, "limits": limits}
    for mc_nuisance in ("indep", "shared_boot"):
        cfg.mc_nuisance = mc_nuisance
        refit_fn = make_v3x_refit_fn(cfg, point["_v3x"], ing["mm"])
        t0 = time.time()
        mc = joint_mc_errors(
            ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["mm"],
            ing["fp_model"], X_tot, logN_lo, logN_hi, N_b, dN_b, ing["truth_cut"],
            cfg, np.random.default_rng(seed + 4), refit_fn=refit_fn)
        print(f"    PM   {mc_nuisance:12s} band done ({time.time()-t0:.0f}s)")
        out[mc_nuisance] = dict(
            f_b_samples=mc["_samples"]["f_b"],
            **{f"dndx_{l}_samples": mc["_samples"]["dndx_total"][l] for l in limits},
            **{f"omega_{l}_samples": mc["_samples"]["omega"][l] for l in limits})
    return out


def report(tag, res, limits):
    print(f"\n===== {tag}: band (indep vs shared_boot), point, truth =====")
    pt = res["point"]; tr = res["truth"]
    lines = []
    for q in ("dndx", "omega"):
        for l in limits:
            key = f"{q}_{l}_samples"
            bi = _band(res["indep"][key]); bs = _band(res["shared_boot"][key])
            pv = (pt["dndx_total"][l] if q == "dndx" else pt["omega"][l])
            tv = (tr["dndx_total"][l] if q == "dndx" else tr["omega"][l])
            wi = bi[2] - bi[0]; ws = bs[2] - bs[0]
            # ALERT THRESHOLD RULE: if w68(shared_boot) < w68(indep)/2 at any reported
            # limit, Stage II tightens enough to force a plan re-eval before Stage III.
            # A w_ratio < 0.5 means the shared correlation removes >50% of the indep
            # band width, which is a sign that the double-counted D_t noise was dominant
            # rather than the genuine systematic; this would imply the indep band was
            # substantially miscalibrated and the Stage III kernel marginalization plan
            # should be reassessed. Flag clearly in the report output.
            cov_i = bi[0] <= tv <= bi[2]; cov_s = bs[0] <= tv <= bs[2]
            w_ratio = ws / wi if wi > 0 else float("nan")
            alert = " *** ALERT: w_ratio<0.5 -> plan re-eval before Stage III ***" \
                if (wi > 0 and w_ratio < 0.5) else ""
            ln = (f"  {q:5s}>={l}: point={pv:.4e} truth={tv:.4e}\n"
                  f"        indep   q16/q50/q84 = {bi[0]:.4e}/{bi[1]:.4e}/{bi[2]:.4e}  "
                  f"w68={wi:.3e}  cover68={cov_i}\n"
                  f"        shared  q16/q50/q84 = {bs[0]:.4e}/{bs[1]:.4e}/{bs[2]:.4e}  "
                  f"w68={ws:.3e}  cover68={cov_s}  (w_ratio={w_ratio:.3f}){alert}")
            print(ln); lines.append(ln)
    return lines


def make_figure(loa0, pm, limits, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = [("loa0 dN/dX", loa0, "dndx"), ("loa0 Omega_HI", loa0, "omega"),
              ("PM dN/dX", pm, "dndx"), ("PM Omega_HI", pm, "omega")]
    for ax, (title, res, q) in zip(axes.ravel(), panels):
        pt = res["point"]; tr = res["truth"]
        xs = np.arange(len(limits))
        ptv = [pt["dndx_total"][l] if q == "dndx" else pt["omega"][l] for l in limits]
        for off, mode, col in ((-0.12, "indep", "C0"), (0.12, "shared_boot", "C1")):
            cen = []; lo = []; hi = []
            for j, l in enumerate(limits):
                b = _band(res[mode][f"{q}_{l}_samples"])
                # RECENTER the HBI marked-Poisson MC band on the MAP point:
                # the convex-MAP Jensen offset drifts the raw-percentile median b[1]
                # off the plug-in MAP point ptv[j]. Apply the per-bin additive shift
                # (point - median) (== cddf_catalog_hbi.recenter_band_on_point) so the
                # band's center lands on the MAP point while the 68% half-widths
                # (b[1]-b[0], b[2]-b[1]) are preserved.
                cen.append(ptv[j]); lo.append(b[1] - b[0]); hi.append(b[2] - b[1])
            ax.errorbar(xs + off, cen, yerr=[lo, hi], fmt="o", color=col, capsize=3,
                        label=f"{mode} (68%, recentered)")
        trv = [tr["dndx_total"][l] if q == "dndx" else tr["omega"][l] for l in limits]
        ax.plot(xs, ptv, "kx", ms=9, mew=2, label="MAP point")
        ax.plot(xs, trv, "s", color="C3", ms=7, label="truth")
        ax.set_xticks(xs); ax.set_xticklabels([f"≥{l}" for l in limits])
        ax.set_title(title); ax.set_yscale("log"); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Stage II — independent-Beta vs shared-bootstrap calibration band "
                 "(mc_inner=laplace, 2LPT-0)")
    fig.savefig(out_png, dpi=130)
    print(f"[stage2-val] figure -> {out_png}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir",
                   default=("/scratch/cavestru_root/cavestru0/mfho/"
                            "gl_prod_2lpt0_v1_20260526/combined_catalog/"))
    p.add_argument("--truth",
                   default=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                            "qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"))
    p.add_argument("--bal-cat",
                   default=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                            "qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"))
    p.add_argument("--molly-tsv", default=None)
    p.add_argument("--kernel",
                   default=("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                            "phase3d_experiments/mollynhi195_lyaonly1025_broaden012/"
                            "posterior_kernel_2lpt0.npz"))
    p.add_argument("--loa0-product",
                   default=("/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/"
                            "outputs/loa0_fp_product_lyaonly1025.npz"))
    p.add_argument("--out", default="/tmp/hbi_stage2_validation")
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
    p.add_argument("--n-mc", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-pm", action="store_true")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    limits = tuple(float(x) for x in args.report_limits.split(","))

    t0 = time.time()
    loa0 = run_loa0(args, limits, args.seed)
    loa0_lines = report("loa0", loa0, limits)
    pm = None; pm_lines = []
    if not args.skip_pm:
        pm = run_pm(args, limits, args.seed)
        pm_lines = report("purity_mixture", pm, limits)

    # save npz
    sav = {}
    for tag, res in (("loa0", loa0), ("pm", pm)):
        if res is None:
            continue
        for mode in ("indep", "shared_boot"):
            for l in limits:
                sav[f"{tag}_{mode}_dndx_{l}"] = np.asarray(res[mode][f"dndx_{l}_samples"])
                sav[f"{tag}_{mode}_omega_{l}"] = np.asarray(res[mode][f"omega_{l}_samples"])
        for l in limits:
            sav[f"{tag}_point_dndx_{l}"] = float(res["point"]["dndx_total"][l])
            sav[f"{tag}_point_omega_{l}"] = float(res["point"]["omega"][l])
            sav[f"{tag}_truth_dndx_{l}"] = float(res["truth"]["dndx_total"][l])
            sav[f"{tag}_truth_omega_{l}"] = float(res["truth"]["omega"][l])
    npz = os.path.join(args.out, "stage2_band_compare.npz")
    np.savez(npz, **sav, report_limits=np.asarray(limits), n_mc=args.n_mc)
    print(f"[stage2-val] npz -> {npz}")

    if pm is not None:
        make_figure(loa0, pm, limits, os.path.join(args.out, "fig_correlated_band.png"))
    with open(os.path.join(args.out, "band_report.txt"), "w") as fh:
        fh.write("\n".join(loa0_lines + pm_lines) + "\n")
    print(f"[stage2-val] done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
