"""Stage III validation on 2LPT-0: the response (θ_K) marginalization coverage test.

The faithful marginalized band composes Stage I (mc_inner=laplace, inner Laplace draw),
Stage II (mc_nuisance=shared_boot, shared D_t bootstrap), and Stage III (mc_response, the
per-draw response re-fit). Stage III is THE dominant coverage lever: the truth–band gap
after I+II is the FROZEN response θ_K (the kernel re-center held at one functional). This
driver runs, on loa0 (+ optional PM cross-check):

  * FROZEN        : mc_response='frozen'  — the response is fixed at the cached functional
                    (the broaden012 + znz mean-shift); A built once. The pre-Stage-III band.
  * STEP-1 (param): mc_response='marginalize', α∈[1,1] — only the b/σ PARAMETER scatter
                    (re-fit per shared resample) + the mean↔median FORM-mix q vary. The
                    b_ref note predicts this is too NARROW to bracket truth (mean↔median is
                    ~0.035 dex). MEASURE it.
  * STEP-2 (form) : mc_response='marginalize', α∈[0,1] — the response STRENGTH (OFF↔full)
                    enters too. The b_ref note shows OFF↔corrected spans R0≈1.11↔0.79,
                    which BRACKETS truth (R0=1). The genuine response-form uncertainty.

Reports, per limit (≥20.0/20.3/20.6) and per z-bin, dN/dX(z) & Ω: does truth fall in the
marginalized 68% / 95% band? Writes the coverage figure + an npz.

Reduce-only / analysis-side. NO GP inference. conda gpdla; BLAS pinned; <=4 workers.
The per-draw A-rebuild dominates the cost (Stage III), so n_mc is small (the coverage
answer needs the band SHAPE, not n_mc=200 precision).

Usage:
  python CDDF_analysis/hbi_validation_2lpt0_stage3.py --n-mc 100 --kernel-znz <znz.npz> \
      --out <dir> [--skip-pm] [--workers 4]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import functools

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis import cddf_catalog_hbi as H
from CDDF_analysis.cddf_catalog_hbi import truth_reductions
from CDDF_analysis.ab_loa0_fp_baseline import build_ingredients
from CDDF_analysis.wall1_explain_partA import loa0_full_posterior_mc

DEF_ZNZ = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/stage0/"
           "znz_2lpt0.npz")
DEF_KERNEL = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phase3d_experiments/"
              "mollynhi195_lyaonly1025_broaden012/posterior_kernel_2lpt0.npz")
DEF_LOA0 = ("/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/outputs/"
            "loa0_fp_product_lyaonly1025.npz")
DEF_CAT = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
           "combined_catalog/")
DEF_TRUTH = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
             "mock-0/loa-124/hcd_truth_cat.fits")
DEF_BAL = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
           "mock-0/loa-124/bal_cat.fits")

MODES = ("frozen", "step1", "step2")
MODE_CFG = {
    "frozen": dict(mc_response="frozen"),
    "step1": dict(mc_response="marginalize", mc_response_q_lo=0.0, mc_response_q_hi=1.0,
                  mc_response_alpha_lo=1.0, mc_response_alpha_hi=1.0),
    "step2": dict(mc_response="marginalize", mc_response_q_lo=0.0, mc_response_q_hi=1.0,
                  mc_response_alpha_lo=0.0, mc_response_alpha_hi=1.0),
}


def _cov(samp, t):
    s = np.asarray(samp, float)
    lo68, hi68 = np.nanpercentile(s, 16), np.nanpercentile(s, 84)
    lo95, hi95 = np.nanpercentile(s, 2.5), np.nanpercentile(s, 97.5)
    med = np.nanpercentile(s, 50)
    return dict(lo68=lo68, hi68=hi68, lo95=lo95, hi95=hi95, med=med,
                cov68=bool(lo68 <= t <= hi68), cov95=bool(lo95 <= t <= hi95))


def run_loa0(args, limits, seed):
    ing = build_ingredients(args, "loa0", loa0_product=args.loa0_product)
    cfg = ing["cfg"]; cfg.report_logN_limits = limits
    cfg._wall1_estimator = "v3"
    cfg.mc_inner = "laplace"            # Stage I
    cfg.mc_nuisance = "shared_boot"     # Stage II (required for Stage III shared boot_mult)
    cfg.kernel_znz_model = args.kernel_znz   # response transform ON (so it is marginalizable)
    logN_lo, logN_hi = ing["logN_lo"], ing["logN_hi"]
    N_b, dN_b, X_tot = ing["N_b"], ing["dN_b"], ing["X_tot"]
    point = ing["estimator_fn"](
        ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["C_interp"],
        ing["fp_model"], X_tot, logN_lo, logN_hi, N_b, dN_b, ing["truth_cut"], cfg)
    tr = truth_reductions(cfg, ing["truth_cut"], logN_lo, logN_hi, N_b, dN_b, X_tot)
    # truth dN/dX(z) per zbin per limit (for the per-z-bin coverage)
    zbins = np.asarray(cfg.zbins, float)
    from CDDF_analysis.cddf_catalog_hbi import _zbin_index
    t_nhi = np.asarray(ing["truth_cut"]["NHI"], float)
    t_z = np.asarray(ing["truth_cut"]["Z_DLA"], float)
    t_snr = np.asarray(ing["truth_cut"]["S2N_RED"], float)
    keep = t_snr > cfg.snr_min
    t_nhi, t_z = t_nhi[keep], t_z[keep]
    t_zidx = _zbin_index(t_z, zbins)
    Xz = np.asarray(X_tot, float)
    truth_dndx_z = {}
    for l in limits:
        dz = np.zeros(len(zbins) - 1)
        for k in range(len(zbins) - 1):
            sel = (t_nhi >= l - 1e-9) & (t_nhi < cfg.drop_top_bin_above) & (t_zidx == k)
            dz[k] = sel.sum() / Xz[k] if Xz[k] > 0 else np.nan
        truth_dndx_z[l] = dz

    pool = None
    if args.workers > 1:
        import multiprocessing as mp
        pool = mp.Pool(args.workers)
    out = {"point": point, "truth": tr, "truth_dndx_z": truth_dndx_z,
           "limits": limits, "zbins": zbins, "bands": {}}
    try:
        for mode in MODES:
            for k, v in MODE_CFG[mode].items():
                setattr(cfg, k, v)
            t0 = time.time()
            band = loa0_full_posterior_mc(cfg, ing, point, args.n_mc,
                                          np.random.default_rng(seed + 3))
            out["bands"][mode] = band
            print(f"    loa0 {mode:7s} band done ({time.time()-t0:.0f}s)")
    finally:
        if pool is not None:
            pool.close(); pool.join()
    return out


def report(out, label):
    limits = out["limits"]; tr = out["truth"]; point = out["point"]
    lines = []
    lines.append("=" * 78)
    lines.append(f"STAGE III COVERAGE — {label}  (frozen vs step1[param] vs step2[form])")
    lines.append("=" * 78)
    for kind, key, tk in (("dN/dX", "dndx", "dndx_total"), ("Omega", "omega", "omega")):
        for l in limits:
            t = tr[tk][l]
            lines.append(f"{kind} >={l}:  truth={t:.5e}  MAP={point[tk][l]:.5e}")
            for mode in MODES:
                samp = out["bands"][mode][f"{key}_{l}_samples"]
                c = _cov(samp, t)
                tag = "  <== COVERS68" if c["cov68"] else ("  (cov95)" if c["cov95"] else "")
                lines.append(
                    f"    {mode:7s}: med={c['med']:.4e} "
                    f"68[{c['lo68']:.4e},{c['hi68']:.4e}] "
                    f"95[{c['lo95']:.4e},{c['hi95']:.4e}]{tag}")
    # per-z-bin dN/dX coverage (step2)
    zbins = out["zbins"]; zmid = 0.5 * (zbins[:-1] + zbins[1:])
    lines.append("-" * 78)
    lines.append("Per-z-bin dN/dX coverage (step2 form-marginalized band):")
    for l in limits:
        tz = out["truth_dndx_z"][l]
        s2z = out["bands"]["step2"][f"dndx_z_{l}_samples"]   # (n_mc, n_zbins)
        row = [f">={l}:"]
        for k in range(len(zmid)):
            c = _cov(s2z[:, k], tz[k])
            row.append(f"z{zmid[k]:.2f}={'IN' if c['cov68'] else 'out'}")
        lines.append("    " + "  ".join(row))
    return "\n".join(lines)


def make_figure(out_loa0, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    limits = out_loa0["limits"]; tr = out_loa0["truth"]; point = out_loa0["point"]
    fig, axes = plt.subplots(2, len(limits), figsize=(4.6 * len(limits), 8.4))
    colors = {"frozen": "#888888", "step1": "#1f77b4", "step2": "#d62728"}
    for ri, (kind, key, tk, ylab) in enumerate(
            (("dN/dX", "dndx", "dndx_total", r"$dN/dX$"),
             ("Omega", "omega", "omega", r"$\Omega_{\rm HI}$"))):
        for ci, l in enumerate(limits):
            ax = axes[ri, ci]
            t = tr[tk][l]
            for j, mode in enumerate(MODES):
                samp = out_loa0["bands"][mode][f"{key}_{l}_samples"]
                c = _cov(samp, t)
                x = j
                ax.fill_between([x - 0.32, x + 0.32], [c["lo95"]] * 2, [c["hi95"]] * 2,
                                color=colors[mode], alpha=0.18, lw=0)
                ax.fill_between([x - 0.32, x + 0.32], [c["lo68"]] * 2, [c["hi68"]] * 2,
                                color=colors[mode], alpha=0.45, lw=0)
                ax.plot([x - 0.32, x + 0.32], [c["med"]] * 2, color=colors[mode], lw=2)
                if c["cov68"]:
                    ax.plot(x, t, "*", color="k", ms=13, zorder=5)
            ax.axhline(t, color="k", ls="--", lw=1.2, label="truth")
            ax.axhline(point[tk][l], color="green", ls=":", lw=1.2, label="MAP")
            ax.set_xticks(range(len(MODES)))
            ax.set_xticklabels(["frozen", "step1\n(param)", "step2\n(form)"])
            ax.set_title(f"{kind} (>={l})")
            if ci == 0:
                ax.set_ylabel(ylab)
            if ri == 0 and ci == len(limits) - 1:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle("Stage III: response-θ_K marginalization vs 2LPT-0 truth "
                 "(loa0; ★ = truth inside 68% band)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[stage3-val] figure -> {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir", default=DEF_CAT)
    p.add_argument("--truth", default=DEF_TRUTH)
    p.add_argument("--bal-cat", default=DEF_BAL)
    p.add_argument("--molly-tsv", default=None)
    p.add_argument("--kernel", default=DEF_KERNEL)
    p.add_argument("--kernel-znz", default=DEF_ZNZ)
    p.add_argument("--loa0-product", default=DEF_LOA0)
    p.add_argument("--out", default="/scratch/cavestru_root/cavestru0/mfho/"
                                    "cddf_o3_realdata/faithful_stage3")
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
    p.add_argument("--n-mc", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    limits = tuple(float(x) for x in args.report_limits.split(","))

    t0 = time.time()
    print("=" * 78)
    print(f"[stage3-val] loa0 (mc_inner=laplace + mc_nuisance=shared_boot + mc_response)")
    print(f"             kernel_znz={args.kernel_znz}  n_mc={args.n_mc}")
    out_loa0 = run_loa0(args, limits, args.seed)
    print(f"[stage3-val] loa0 bands done ({time.time()-t0:.0f}s)")

    rep = report(out_loa0, "loa0 (2LPT-0)")
    print("\n" + rep)
    with open(os.path.join(args.out, "stage3_coverage_report.txt"), "w") as fh:
        fh.write(rep + "\n")

    fig_path = os.path.join(args.out, "fig_coverage.png")
    make_figure(out_loa0, fig_path)

    savez = {}
    for mode in MODES:
        for l in limits:
            savez[f"{mode}_dndx_{l}"] = out_loa0["bands"][mode][f"dndx_{l}_samples"]
            savez[f"{mode}_omega_{l}"] = out_loa0["bands"][mode][f"omega_{l}_samples"]
            savez[f"{mode}_dndx_z_{l}"] = out_loa0["bands"][mode][f"dndx_z_{l}_samples"]
    for l in limits:
        savez[f"truth_dndx_{l}"] = float(out_loa0["truth"]["dndx_total"][l])
        savez[f"truth_omega_{l}"] = float(out_loa0["truth"]["omega"][l])
        savez[f"truth_dndx_z_{l}"] = out_loa0["truth_dndx_z"][l]
        savez[f"map_dndx_{l}"] = float(out_loa0["point"]["dndx_total"][l])
        savez[f"map_omega_{l}"] = float(out_loa0["point"]["omega"][l])
    savez["zbins"] = out_loa0["zbins"]; savez["limits"] = np.asarray(limits)
    np.savez(os.path.join(args.out, "stage3_bands.npz"), **savez)
    print(f"[stage3-val] saved npz + report -> {args.out}  ({time.time()-t0:.0f}s)")
    return out_loa0


if __name__ == "__main__":
    main()
