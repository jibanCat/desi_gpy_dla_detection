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
from CDDF_analysis.cddf_catalog_hbi import (
    truth_reductions, make_v3x_refit_fn,
    _draw_beta_cell, _rescale_unitC_active, _apply_C_to_M, _cell_index,
    _slice_active_unitC, C_FLOOR, make_rho_interpolator,
    build_truth_match_resample, draw_shared_boot, draw_shared_boot_with_mult,
    v3x_response_setup, v3x_response_rebuild_unitC, draw_response_params,
    v3x_fit_map, v3x_mc_inner_theta, v3x_reduce,
)
from CDDF_analysis.ab_loa0_fp_baseline import build_ingredients
from CDDF_analysis.wall1_explain_partA import loa0_full_posterior_mc
from CDDF_analysis.znz_kernel import refit_znz_from_resample

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


def _hpd68(samp):
    """Highest-Posterior-Density 68% interval: the shortest contiguous interval
    containing 68% of the sorted draws. Returns (lo, hi)."""
    s = np.sort(np.asarray(samp, float)[np.isfinite(np.asarray(samp, float))])
    n = len(s)
    if n == 0:
        return (np.nan, np.nan)
    width = int(np.ceil(0.68 * n))
    if width >= n:
        return (float(s[0]), float(s[-1]))
    intervals = s[width - 1:] - s[:n - width + 1]
    idx = int(np.argmin(intervals))
    return (float(s[idx]), float(s[idx + width - 1]))


def _cov(samp, t):
    s = np.asarray(samp, float)
    lo68, hi68 = np.nanpercentile(s, 16), np.nanpercentile(s, 84)
    lo95, hi95 = np.nanpercentile(s, 2.5), np.nanpercentile(s, 97.5)
    med = np.nanpercentile(s, 50)
    hlo68, hhi68 = _hpd68(s)
    return dict(lo68=lo68, hi68=hi68, lo95=lo95, hi95=hi95, med=med,
                hlo68=hlo68, hhi68=hhi68,
                cov68=bool(lo68 <= t <= hi68), cov95=bool(lo95 <= t <= hi95),
                covhpd68=bool(hlo68 <= t <= hhi68))


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


def pm_full_posterior_mc(cfg, ing, point, n_mc, rng):
    """Stage I+II+III MC band for the purity_mixture (PM) headline estimator.

    Mirrors loa0_full_posterior_mc exactly for Stage I (mc_inner=laplace),
    Stage II (mc_nuisance=shared_boot, shared D_t), and Stage III (mc_response
    marginalization of the kernel response transform), but replaces the frozen
    loa0 FP path with the purity-mixture per-draw FP:
        lam_fp_i = (1 - rho_i) * boot_w_i    (coherent with rho draw)

    This is the correct PM band because rho_i is drawn from the SAME shared
    resample as C and boot_w (Stage II correlation), and the unitC rebuild
    (Stage III) uses the same boot_mult. The PM FP is therefore JOINTLY
    correlated with both the completeness and the response, as in the PM
    MAP estimate.
    """
    mm = ing["mm"]; cat_cut = ing["cat_cut"]; family = point["_v3x"]["family"]
    fwd = point["_v3x"]["fwd"]; theta_map = point["_v3x"]["theta_map"]
    A_meta = fwd["A_meta"]; M_meta = fwd["M_meta"]; cat_op = fwd["cat_op"]
    fine = fwd["fine"]
    logN_lo, logN_hi, N_b, dN_b, z_edges_fine = fine
    n_flat = len(logN_lo) * (len(z_edges_fine) - 1)
    unitC = _slice_active_unitC(A_meta, np.arange(n_flat),
                                np.ones(A_meta["n_obs"], bool))
    xhat = cat_op["xhat"]; snr_op = cat_op["snr"]; i_snr0 = cat_op["i_snr"]
    active_flat = fwd["active_flat"]
    op = fwd["op_mask"]
    nhi_err_op = np.asarray(cat_cut["NHI_ERR"], float)[op]
    nhi_err_op = np.where(np.isfinite(nhi_err_op) & (nhi_err_op > 0), nhi_err_op, 0.0)
    tids_op = np.asarray(cat_cut["TARGETID"], np.int64)[op]
    uniq, inv = np.unique(tids_op, return_inverse=True)
    n_uniq = len(uniq)
    keep_in_base = fwd["keep_in_base"]
    limits = cfg.report_logN_limits
    n_zc = len(np.asarray(cfg.zbins, float)) - 1

    # Stage II: shared D_t resample (same as loa0)
    mc_nuisance = getattr(cfg, "mc_nuisance", "indep")
    tmr = None
    if mc_nuisance == "shared_boot":
        tmr = build_truth_match_resample(
            mm, cat_cut, ing["is_TP"], ing["truth_cut"], ing["good_mask"], cfg)

    # Stage III: response (θ_K) marginalization (same setup as loa0)
    mc_response = getattr(cfg, "mc_response", "frozen")
    rctx = None
    if mc_response == "marginalize":
        if mc_nuisance != "shared_boot":
            raise ValueError("mc_response='marginalize' requires mc_nuisance='shared_boot'.")
        rctx = v3x_response_setup(cfg, cat_cut, ing["good_mask"], mm, fwd, tmr)
        if rctx is None:
            raise ValueError(
                "mc_response='marginalize' requires cfg.kernel_znz_model.")

    f_bs = []
    dndx = {l: [] for l in limits}; omega = {l: [] for l in limits}
    dndx_z = {l: [] for l in limits}
    seeds = rng.integers(0, 2**31 - 1, size=n_mc)
    for s in seeds:
        rg = np.random.default_rng(int(s))
        boot_mult = None
        if mc_nuisance == "shared_boot":
            if mc_response == "marginalize":
                C_draw, rho_draw, boot_w_base, boot_mult = \
                    draw_shared_boot_with_mult(rg, tmr)
            else:
                C_draw, rho_draw, boot_w_base = draw_shared_boot(rg, tmr)
            boot_w = boot_w_base[keep_in_base]
            nhi_m = xhat + rg.normal(0, 1, len(xhat)) * nhi_err_op
        else:
            C_draw = _draw_beta_cell(rg, mm.cmp_nfound, mm.cmp_nfid)
            rho_draw = _draw_beta_cell(rg, mm.pur_ntp, mm.pur_ntot)
            C_draw = np.where(mm.cmp_nfid > 0, C_draw, C_FLOOR)
            rho_draw = np.where(mm.pur_ntot > 0, rho_draw, 0.0)
            nhi_m = xhat + rg.normal(0, 1, len(xhat)) * nhi_err_op
            boot_w = rg.multinomial(n_uniq, np.full(n_uniq, 1.0 / n_uniq)).astype(float)[inv]

        # Stage III unitC rebuild (or frozen)
        unitC_draw = unitC
        if mc_response == "marginalize":
            q_draw, alpha_draw = draw_response_params(rg, cfg)
            znz_draw = refit_znz_from_resample(rctx["rfr"], boot_mult,
                                               b_mix=q_draw, corr_strength=alpha_draw)
            unitC_draw = v3x_response_rebuild_unitC(cfg, rctx, znz_draw)
        A_draw = _rescale_unitC_active(unitC_draw, C_draw)
        M_draw = np.where(active_flat, _apply_C_to_M(M_meta, C_draw), 0.0)

        # PM FP: (1 - rho_i) * boot_w — coherent with the shared rho_draw
        j_nhi = _cell_index(mm, nhi_m, snr_op)[1]
        rho_i = rho_draw[i_snr0, j_nhi]
        lam_fp = (1.0 - rho_i) * boot_w
        mu_fp = float(np.sum(lam_fp))

        fit = v3x_fit_map(A_draw, M_draw, lam_fp, mu_fp, fine, family, cfg,
                          obj_weights=boot_w, theta0=theta_map, n_restart=2, rng=rg,
                          lit_start=False)
        # Stage I: laplace draw within this draw's ψ
        theta_inner = v3x_mc_inner_theta(cfg, fit, A_draw, M_draw, lam_fp, mu_fp,
                                         fine, family, boot_w, rg)
        rr = v3x_reduce(cfg, theta_inner, fine, family, M_meta)
        f_bs.append(rr["f_b"])
        for l in limits:
            dndx[l].append(rr["dndx_total"][l]); omega[l].append(rr["omega"][l])
            dndx_z[l].append(rr["dndx_z"][l])

    out = dict(f_b_samples=np.array(f_bs), n_mc=int(n_mc))
    for l in limits:
        out[f"dndx_{l}_samples"] = np.array(dndx[l])
        out[f"omega_{l}_samples"] = np.array(omega[l])
        out[f"dndx_z_{l}_samples"] = np.array(dndx_z[l])   # (n_mc, n_zbins)
    return out


def run_pm(args, limits, seed):
    """Build PM ingredients and run frozen/step1/step2 MC bands (Stage I+II+III)."""
    ing = build_ingredients(args, "purity_mixture")
    cfg = ing["cfg"]; cfg.report_logN_limits = limits
    cfg._wall1_estimator = "v3"
    cfg.mc_inner = "laplace"            # Stage I
    cfg.mc_nuisance = "shared_boot"     # Stage II
    cfg.kernel_znz_model = args.kernel_znz   # Stage III response transform ON
    logN_lo, logN_hi = ing["logN_lo"], ing["logN_hi"]
    N_b, dN_b, X_tot = ing["N_b"], ing["dN_b"], ing["X_tot"]
    point = ing["estimator_fn"](
        ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["C_interp"],
        ing["fp_model"], X_tot, logN_lo, logN_hi, N_b, dN_b, ing["truth_cut"], cfg)
    tr = truth_reductions(cfg, ing["truth_cut"], logN_lo, logN_hi, N_b, dN_b, X_tot)
    # per-z-bin truth dN/dX (same pattern as run_loa0)
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

    out = {"point": point, "truth": tr, "truth_dndx_z": truth_dndx_z,
           "limits": limits, "zbins": zbins, "bands": {}}
    for mode in MODES:
        for k, v in MODE_CFG[mode].items():
            setattr(cfg, k, v)
        t0 = time.time()
        band = pm_full_posterior_mc(cfg, ing, point, args.n_mc,
                                    np.random.default_rng(seed + 7))
        out["bands"][mode] = band
        print(f"    PM   {mode:7s} band done ({time.time()-t0:.0f}s)")
    return out


def report(out, label):
    """Print coverage table for one estimator (loa0 or PM).

    Per quantity/limit/mode reports:
      - truth, MAP point
      - equal-tailed 68% (q16/q84) and 95% (q2.5/q97.5) bands + cover booleans
      - HPD 68% interval (shortest contiguous 68% mass) + cover boolean
    """
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
                cov_tag = []
                if c["cov68"]:   cov_tag.append("COV68")
                if c["cov95"]:   cov_tag.append("COV95")
                if c["covhpd68"]: cov_tag.append("COV-HPD68")
                tag = ("  <== " + "+".join(cov_tag)) if cov_tag else ""
                lines.append(
                    f"    {mode:7s}: med={c['med']:.4e} "
                    f"EQ68[{c['lo68']:.4e},{c['hi68']:.4e}] cov68={c['cov68']} "
                    f"EQ95[{c['lo95']:.4e},{c['hi95']:.4e}] cov95={c['cov95']} "
                    f"HPD68[{c['hlo68']:.4e},{c['hhi68']:.4e}] covHPD68={c['covhpd68']}"
                    f"{tag}")
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
            row.append(f"z{zmid[k]:.2f}={'IN68' if c['cov68'] else ('IN95' if c['cov95'] else 'out')}")
        lines.append("    " + "  ".join(row))
    return "\n".join(lines)


def _draw_estimator_panels(axes_row_loa0, axes_row_pm, out_loa0, out_pm, kind, key, tk):
    """Helper: draw one row of panels for a given quantity (dN/dX or Omega).
    axes_row_loa0 / axes_row_pm are lists of axes for loa0 / PM respectively.
    Pass out_pm=None to skip PM panels."""
    colors = {"frozen": "#888888", "step1": "#1f77b4", "step2": "#d62728"}
    for panel_axes, out, estimator_label in (
            (axes_row_loa0, out_loa0, "loa0"),
            (axes_row_pm,   out_pm,   "PM")):
        if out is None or panel_axes is None:
            continue
        limits = out["limits"]; tr = out["truth"]; point = out["point"]
        for ci, l in enumerate(limits):
            ax = panel_axes[ci]
            t = tr[tk][l]
            for j, mode in enumerate(MODES):
                samp = out["bands"][mode][f"{key}_{l}_samples"]
                c = _cov(samp, t)
                x = j
                ax.fill_between([x - 0.32, x + 0.32], [c["lo95"]] * 2, [c["hi95"]] * 2,
                                color=colors[mode], alpha=0.18, lw=0)
                ax.fill_between([x - 0.32, x + 0.32], [c["lo68"]] * 2, [c["hi68"]] * 2,
                                color=colors[mode], alpha=0.45, lw=0)
                ax.plot([x - 0.32, x + 0.32], [c["med"]] * 2, color=colors[mode], lw=2)
                # ★ = truth inside equal-tail 68% band
                if c["cov68"]:
                    ax.plot(x, t, "*", color="k", ms=13, zorder=5)
                # ✕ = MAP point (offset x slightly for clarity)
                ax.plot(x, point[tk][l], "x", color="green", ms=9, mew=2, zorder=4)
            ax.axhline(t, color="k", ls="--", lw=1.2, label="truth ★")
            ax.axhline(point[tk][l], color="green", ls=":", lw=1.2, label="MAP ✕")
            ax.set_xticks(range(len(MODES)))
            ax.set_xticklabels(["frozen", "step1\n(param)", "step2\n(form)"])
            ax.set_title(f"{estimator_label} {kind} (>={l})")


def make_figure(out_loa0, out_path, out_pm=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    limits = (out_loa0 or out_pm)["limits"]
    n_rows = 0
    if out_loa0 is not None: n_rows += 2  # dN/dX + Omega for loa0
    if out_pm  is not None: n_rows += 2   # dN/dX + Omega for PM
    n_rows = max(n_rows, 1)
    fig, axes = plt.subplots(n_rows, len(limits), figsize=(4.6 * len(limits), 4.2 * n_rows),
                             squeeze=False)
    row = 0
    for kind, key, tk, ylab in (
            ("dN/dX", "dndx", "dndx_total", r"$dN/dX$"),
            ("Omega", "omega", "omega", r"$\Omega_{\rm HI}$")):
        loa0_row = list(axes[row]) if out_loa0 is not None else None
        pm_row   = None
        if out_loa0 is not None:
            for ci in range(len(limits)): axes[row, ci].set_ylabel(f"loa0 {ylab}")
            row += 1
        if out_pm is not None:
            pm_row = list(axes[row])
            for ci in range(len(limits)): axes[row, ci].set_ylabel(f"PM {ylab}")
            row += 1
        _draw_estimator_panels(loa0_row, pm_row, out_loa0, out_pm, kind, key, tk)
    # legend on first occupied axis
    for ax in axes.ravel():
        if ax.lines or ax.collections:
            ax.legend(fontsize=7, loc="best"); break
    fig.suptitle("Stage III: response-θ_K marginalization vs 2LPT-0 truth\n"
                 "(★ = truth inside eq-tail 68% band;  ✕ = MAP)", fontsize=11)
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
    p.add_argument("--skip-loa0", action="store_true",
                   help="Skip loa0 estimator run (run PM only). Default: run both.")
    p.add_argument("--skip-pm", action="store_true",
                   help="Skip purity_mixture estimator run (run loa0 only). Default: run both.")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    limits = tuple(float(x) for x in args.report_limits.split(","))

    t0 = time.time()
    out_loa0 = None; out_pm = None
    reports = []

    if not args.skip_loa0:
        print("=" * 78)
        print(f"[stage3-val] loa0 (mc_inner=laplace + mc_nuisance=shared_boot + mc_response)")
        print(f"             kernel_znz={args.kernel_znz}  n_mc={args.n_mc}")
        out_loa0 = run_loa0(args, limits, args.seed)
        print(f"[stage3-val] loa0 bands done ({time.time()-t0:.0f}s)")
        rep_loa0 = report(out_loa0, "loa0 (2LPT-0)")
        print("\n" + rep_loa0)
        reports.append(rep_loa0)

    if not args.skip_pm:
        print("=" * 78)
        print(f"[stage3-val] PM (mc_inner=laplace + mc_nuisance=shared_boot + mc_response)")
        print(f"             kernel_znz={args.kernel_znz}  n_mc={args.n_mc}")
        t1 = time.time()
        out_pm = run_pm(args, limits, args.seed)
        print(f"[stage3-val] PM bands done ({time.time()-t1:.0f}s)")
        rep_pm = report(out_pm, "purity_mixture/PM (2LPT-0)")
        print("\n" + rep_pm)
        reports.append(rep_pm)

    full_report = "\n\n".join(reports)
    with open(os.path.join(args.out, "stage3_coverage_report.txt"), "w") as fh:
        fh.write(full_report + "\n")

    fig_path = os.path.join(args.out, "fig_coverage.png")
    make_figure(out_loa0, fig_path, out_pm=out_pm)

    savez = {}
    for tag, out in (("loa0", out_loa0), ("pm", out_pm)):
        if out is None:
            continue
        for mode in MODES:
            for l in limits:
                savez[f"{tag}_{mode}_dndx_{l}"] = out["bands"][mode][f"dndx_{l}_samples"]
                savez[f"{tag}_{mode}_omega_{l}"] = out["bands"][mode][f"omega_{l}_samples"]
                savez[f"{tag}_{mode}_dndx_z_{l}"] = out["bands"][mode][f"dndx_z_{l}_samples"]
        for l in limits:
            savez[f"{tag}_truth_dndx_{l}"] = float(out["truth"]["dndx_total"][l])
            savez[f"{tag}_truth_omega_{l}"] = float(out["truth"]["omega"][l])
            savez[f"{tag}_truth_dndx_z_{l}"] = out["truth_dndx_z"][l]
            savez[f"{tag}_map_dndx_{l}"] = float(out["point"]["dndx_total"][l])
            savez[f"{tag}_map_omega_{l}"] = float(out["point"]["omega"][l])
        savez[f"{tag}_zbins"] = out["zbins"]
    # backward-compat: keep flat loa0-only keys if loa0 ran
    if out_loa0 is not None:
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
        savez["zbins"] = out_loa0["zbins"]
    savez["limits"] = np.asarray(limits)
    np.savez(os.path.join(args.out, "stage3_bands.npz"), **savez)
    print(f"[stage3-val] saved npz + report -> {args.out}  ({time.time()-t0:.0f}s)")
    return dict(loa0=out_loa0, pm=out_pm)


if __name__ == "__main__":
    main()
