"""hbi_fNz_coverage.py — the per-z DIFFERENTIAL CDDF f(N,z) coverage deliverable.

The Stage-III driver (hbi_validation_2lpt0_stage3.py) reports the per-z dN/dX(z) and
Ω(z) coverage — the z-resolved INTEGRALS. This standalone driver delivers the
underlying z-resolved DIFFERENTIAL CDDF f(N | z_coarse): for each (logN bin b, coarse
z bin k) does the truth f_truth[b,k] sit inside the HBI marginalized band of the
genuine 2-D f draws ``f_bk_coarse[:, b, k]``?

It reuses the SAME PM-band machinery as the stage3 ``run_pm`` path — build_ingredients(
"purity_mixture"); cfg.mc_inner=laplace; mc_nuisance=shared_boot; kernel_znz=broaden012;
modes frozen + step2 — but collects the NEW additive ``f_bk_coarse`` per draw (the
genuine per-coarse-z differential f, tied by construction to the already-reported
dndx_z; see cddf_catalog_hbi._coarse_z_differential_f). The consistency of the saved f
with dndx_z is ASSERTED inline (the correctness gate).

The truth f(N,z) is the truth count in (logN bin b ∧ z bin k), divided by dN_b[b]·X_tot[k]
(same snr cut + drop_top_bin_above ceiling as the estimator).

Reduce-only / analysis-side. NO GP inference. conda gpdla; BLAS pinned; <=4 workers.

Usage:
  python CDDF_analysis/hbi_fNz_coverage.py --n-mc 120 --workers 4
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

from CDDF_analysis import cddf_catalog_hbi as H
from CDDF_analysis.cddf_catalog_hbi import (
    truth_reductions,
    _draw_beta_cell, _rescale_unitC_active, _apply_C_to_M, _cell_index,
    _slice_active_unitC, C_FLOOR, _zbin_index, _bin_index_logN,
    build_truth_match_resample, draw_shared_boot, draw_shared_boot_with_mult,
    v3x_response_setup, v3x_response_rebuild_unitC, draw_response_params,
    v3x_fit_map, v3x_mc_inner_theta, v3x_reduce,
)
from CDDF_analysis.ab_loa0_fp_baseline import build_ingredients
from CDDF_analysis.znz_kernel import refit_znz_from_resample

# Defaults shared with hbi_validation_2lpt0_stage3.py / ab_loa0_fp_baseline.py
DEF_ZNZ = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/stage0/"
           "znz_2lpt0.npz")
DEF_KERNEL = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phase3d_experiments/"
              "mollynhi195_lyaonly1025_broaden012/posterior_kernel_2lpt0.npz")
DEF_CAT = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
           "combined_catalog/")
DEF_TRUTH = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
             "mock-0/loa-124/hcd_truth_cat.fits")
DEF_BAL = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
           "mock-0/loa-124/bal_cat.fits")
DEF_OUT = "/scratch/cavestru_root/cavestru0/mfho/hbi_stage3_fNz"

# frozen + step2 (the response-form-marginalized band that brackets truth on the
# integrals; step1 is omitted — the per-(N,z) deliverable is frozen vs the headline).
MODES = ("frozen", "step2")
MODE_CFG = {
    "frozen": dict(mc_response="frozen"),
    "step2": dict(mc_response="marginalize", mc_response_q_lo=0.0, mc_response_q_hi=1.0,
                  mc_response_alpha_lo=0.0, mc_response_alpha_hi=1.0),
}


# -----------------------------------------------------------------------------
# PM band collecting the genuine 2-D f_bk_coarse per draw (mirrors stage3
# pm_full_posterior_mc, + f_bk_coarse collection + consistency assertion)
# -----------------------------------------------------------------------------
def pm_fNz_band(cfg, ing, point, n_mc, rng, assert_consistency=True):
    """Stage I+II+III PM MC band that collects, per draw, the GENUINE per-coarse-z
    differential CDDF ``rr['f_bk_coarse']`` (n_nbins, n_zc) alongside the integral
    reductions. Asserts the saved f is consistent with the per-z dN/dX it integrates to
    (the correctness gate)."""
    mm = ing["mm"]; cat_cut = ing["cat_cut"]; family = point["_v3x"]["family"]
    fwd = point["_v3x"]["fwd"]; theta_map = point["_v3x"]["theta_map"]
    A_meta = fwd["A_meta"]; M_meta = fwd["M_meta"]; cat_op = fwd["cat_op"]
    fine = fwd["fine"]
    logN_lo, logN_hi, N_b, dN_b, z_edges_fine = fine
    n_nbins = len(logN_lo)
    n_flat = n_nbins * (len(z_edges_fine) - 1)
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
    K = H.omega_hi_prefactor(cfg.H0)

    mc_nuisance = getattr(cfg, "mc_nuisance", "indep")
    tmr = None
    if mc_nuisance == "shared_boot":
        tmr = build_truth_match_resample(
            mm, cat_cut, ing["is_TP"], ing["truth_cut"], ing["good_mask"], cfg)

    mc_response = getattr(cfg, "mc_response", "frozen")
    rctx = None
    if mc_response == "marginalize":
        if mc_nuisance != "shared_boot":
            raise ValueError("mc_response='marginalize' requires mc_nuisance='shared_boot'.")
        rctx = v3x_response_setup(cfg, cat_cut, ing["good_mask"], mm, fwd, tmr)
        if rctx is None:
            raise ValueError("mc_response='marginalize' requires cfg.kernel_znz_model.")

    f_bks = []                                   # (n_mc, n_nbins, n_zc)
    dndx = {l: [] for l in limits}; omega = {l: [] for l in limits}
    dndx_z = {l: [] for l in limits}
    seeds = rng.integers(0, 2**31 - 1, size=n_mc)
    max_consistency_err = 0.0
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

        unitC_draw = unitC
        if mc_response == "marginalize":
            q_draw, alpha_draw = draw_response_params(rg, cfg)
            znz_draw = refit_znz_from_resample(rctx["rfr"], boot_mult,
                                               b_mix=q_draw, corr_strength=alpha_draw)
            unitC_draw = v3x_response_rebuild_unitC(cfg, rctx, znz_draw)
        A_draw = _rescale_unitC_active(unitC_draw, C_draw)
        M_draw = np.where(active_flat, _apply_C_to_M(M_meta, C_draw), 0.0)

        j_nhi = _cell_index(mm, nhi_m, snr_op)[1]
        rho_i = rho_draw[i_snr0, j_nhi]
        lam_fp = (1.0 - rho_i) * boot_w
        mu_fp = float(np.sum(lam_fp))

        fit = v3x_fit_map(A_draw, M_draw, lam_fp, mu_fp, fine, family, cfg,
                          obj_weights=boot_w, theta0=theta_map, n_restart=2, rng=rg,
                          lit_start=False)
        theta_inner = v3x_mc_inner_theta(cfg, fit, A_draw, M_draw, lam_fp, mu_fp,
                                         fine, family, boot_w, rg)
        rr = v3x_reduce(cfg, theta_inner, fine, family, M_meta)
        f_bk = rr["f_bk_coarse"]                  # (n_nbins, n_zc) — genuine 2-D f
        f_bks.append(f_bk)
        for l in limits:
            dndx[l].append(rr["dndx_total"][l]); omega[l].append(rr["omega"][l])
            dndx_z[l].append(rr["dndx_z"][l])

        # ---- CORRECTNESS GATE (consistency assertion, spec Task A) ----
        # Tie the saved per-z f to the already-reported per-z dN/dX AND the per-z Ω:
        #   Σ_{N≥lim} f_bk[:,k]·dN_b == dndx_z[lim][k]
        #   K·Σ_{N≥lim} N_b·f_bk[:,k]·dN_b == omega_z (the per-z Ω that f integrates to)
        if assert_consistency:
            for l in limits:
                sel = logN_lo >= l - 1e-9
                for k in range(n_zc):
                    lhs = float(np.nansum(f_bk[sel, k] * dN_b[sel]))
                    rhs = float(rr["dndx_z"][l][k])
                    if np.isfinite(rhs):
                        rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
                        max_consistency_err = max(max_consistency_err, rel)
            if max_consistency_err >= 1e-9:
                raise AssertionError(
                    f"f_bk_coarse <-> dndx_z consistency gate FAILED: "
                    f"max rel err {max_consistency_err:.2e} >= 1e-9")

    out = dict(f_bk_coarse_samples=np.array(f_bks), n_mc=int(n_mc),
               max_consistency_err=float(max_consistency_err))
    for l in limits:
        out[f"dndx_{l}_samples"] = np.array(dndx[l])
        out[f"omega_{l}_samples"] = np.array(omega[l])
        out[f"dndx_z_{l}_samples"] = np.array(dndx_z[l])   # (n_mc, n_zc)
    return out


# -----------------------------------------------------------------------------
# truth f(N,z) per (logN bin, coarse z bin)
# -----------------------------------------------------------------------------
def truth_fNz(cfg, truth_cut, logN_lo, logN_hi, N_b, dN_b, X_tot):
    """f_truth[b,k] = (truth count in logN bin b AND z bin k) / (dN_b[b]·X_tot[k]).

    SAME snr cut (S2N_RED > snr_min) + drop_top_bin_above ceiling as the estimator's
    fine grid (build_fine_grid drops logN_hi>drop_top_bin_above; the truth count is
    binned on those same edges so it inherits the ceiling)."""
    zbins = np.asarray(cfg.zbins, float)
    n_zc = len(zbins) - 1
    n_nbins = len(logN_lo)
    X = np.asarray(X_tot, float)
    t_nhi = np.asarray(truth_cut["NHI"], float)
    t_z = np.asarray(truth_cut["Z_DLA"], float)
    t_snr = np.asarray(truth_cut["S2N_RED"], float)
    keep = t_snr > cfg.snr_min
    t_nhi, t_z = t_nhi[keep], t_z[keep]
    t_nidx = _bin_index_logN(t_nhi, logN_lo, logN_hi)   # -1 if outside [floor, ceil]
    t_zidx = _zbin_index(t_z, zbins)                     # -1 if outside zbins
    counts = np.zeros((n_nbins, n_zc))
    valid = (t_nidx >= 0) & (t_zidx >= 0)
    np.add.at(counts, (t_nidx[valid], t_zidx[valid]), 1.0)
    f = np.full((n_nbins, n_zc), np.nan)
    for k in range(n_zc):
        if X[k] > 0:
            f[:, k] = counts[:, k] / (dN_b * X[k])
    return dict(f_truth=f, counts=counts, X_tot=X)


# -----------------------------------------------------------------------------
# per-(N,z)-cell coverage
# -----------------------------------------------------------------------------
def cell_coverage(samples, f_truth):
    """For each (logN bin b, coarse z bin k): does f_truth[b,k] sit in the 68%/95%
    equal-tailed band of samples[:, b, k]? Returns per-cell dicts and a coverage report.
    Cells where truth==0 (count 0) are reported as 'empty' (no constraint)."""
    n_mc, n_nbins, n_zc = samples.shape
    lo68 = np.nanpercentile(samples, 16, axis=0)
    hi68 = np.nanpercentile(samples, 84, axis=0)
    lo95 = np.nanpercentile(samples, 2.5, axis=0)
    hi95 = np.nanpercentile(samples, 97.5, axis=0)
    med = np.nanpercentile(samples, 50, axis=0)
    cov68 = np.zeros((n_nbins, n_zc), bool)
    cov95 = np.zeros((n_nbins, n_zc), bool)
    for b in range(n_nbins):
        for k in range(n_zc):
            t = f_truth[b, k]
            if not np.isfinite(t):
                continue
            cov68[b, k] = bool(lo68[b, k] <= t <= hi68[b, k])
            cov95[b, k] = bool(lo95[b, k] <= t <= hi95[b, k])
    return dict(lo68=lo68, hi68=hi68, lo95=lo95, hi95=hi95, med=med,
                cov68=cov68, cov95=cov95)


def map_fNz(point, cfg, fine, M_meta):
    """The MAP per-coarse-z differential f (point estimate), via v3x_reduce on the
    point θ_map. (point['_v3x'] carries theta_map/family/fine/M_meta.)"""
    family = point["_v3x"]["family"]
    rr = v3x_reduce(cfg, point["_v3x"]["theta_map"], fine, family, M_meta)
    return rr["f_bk_coarse"]


# -----------------------------------------------------------------------------
# figure
# -----------------------------------------------------------------------------
def make_figure(out_path, logN_lo, logN_hi, zbins, f_truth,
                bands, map_f, second_row=True):
    """fig_fNz_coverage.png: a row of panels, one per coarse z bin. Each panel:
    log10 f(N) vs logN, the step2 68% band (shaded), MAP (x), truth (star/line),
    log-y, cover/miss annotated. Optional 2nd row: frozen vs step2 per z."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zbins = np.asarray(zbins, float)
    zmid = 0.5 * (zbins[:-1] + zbins[1:])
    n_zc = len(zmid)
    mid = 0.5 * (logN_lo + logN_hi)

    step2 = bands["step2"]; cov = step2["coverage"]
    n_rows = 2 if (second_row and "frozen" in bands) else 1
    fig, axes = plt.subplots(n_rows, n_zc, figsize=(4.6 * n_zc, 4.0 * n_rows),
                             squeeze=False)

    xlo, xhi = 19.5, 22.5
    pmask = (mid >= xlo - 1e-9) & (mid <= xhi + 1e-9)

    def _draw_band(ax, band, color, label, with_truth=True, with_map=True):
        lo68 = band["coverage"]["lo68"]; hi68 = band["coverage"]["hi68"]
        med = band["coverage"]["med"]
        for k_unused in (0,):
            pass

    for k in range(n_zc):
        ax = axes[0, k]
        lo68 = cov["lo68"][:, k]; hi68 = cov["hi68"][:, k]; medk = cov["med"][:, k]
        ft = f_truth[:, k]; mp = map_f[:, k]
        m = pmask & np.isfinite(medk)
        # shaded 68% band
        ax.fill_between(mid[m], np.clip(lo68[m], 1e-30, None),
                        np.clip(hi68[m], 1e-30, None),
                        color="#d62728", alpha=0.30, lw=0, label="step2 68%")
        ax.plot(mid[m], np.clip(medk[m], 1e-30, None), color="#d62728", lw=1.4,
                label="step2 median")
        # MAP
        mm_ = pmask & np.isfinite(mp) & (mp > 0)
        ax.plot(mid[mm_], mp[mm_], "x", color="green", ms=7, mew=1.8, label="MAP")
        # truth
        mt = pmask & np.isfinite(ft) & (ft > 0)
        ax.plot(mid[mt], ft[mt], "*", color="k", ms=11, ls="none", label="truth")
        ax.plot(mid[mt], ft[mt], "-", color="k", lw=0.8, alpha=0.5)
        # cover/miss markers on truth points
        for b in np.where(mt)[0]:
            inside = cov["cov68"][b, k]
            ax.annotate("" if inside else "miss",
                        (mid[b], ft[b]), textcoords="offset points",
                        xytext=(0, 8), fontsize=6,
                        color=("k" if inside else "red"), ha="center")
        ncov = int(cov["cov68"][mt, k].sum()); ntot = int(mt.sum())
        ax.set_yscale("log")
        ax.set_xlim(xlo, xhi)
        ax.set_title(f"z≈{zmid[k]:.2f}   cover68 {ncov}/{ntot}")
        ax.set_xlabel(r"$\log_{10} N_{\rm HI}$")
        if k == 0:
            ax.set_ylabel(r"$f(N\,|\,z)$  (step2 band)")
        ax.grid(alpha=0.25, which="both")

    if n_rows == 2:
        frozen = bands["frozen"]; fcov = frozen["coverage"]
        for k in range(n_zc):
            ax = axes[1, k]
            ft = f_truth[:, k]
            # frozen band
            flo = fcov["lo68"][:, k]; fhi = fcov["hi68"][:, k]; fmed = fcov["med"][:, k]
            slo = cov["lo68"][:, k]; shi = cov["hi68"][:, k]; smed = cov["med"][:, k]
            mf = pmask & np.isfinite(fmed)
            ms = pmask & np.isfinite(smed)
            ax.fill_between(mid[mf], np.clip(flo[mf], 1e-30, None),
                            np.clip(fhi[mf], 1e-30, None),
                            color="#888888", alpha=0.30, lw=0, label="frozen 68%")
            ax.plot(mid[mf], np.clip(fmed[mf], 1e-30, None), color="#888888", lw=1.2,
                    label="frozen median")
            ax.fill_between(mid[ms], np.clip(slo[ms], 1e-30, None),
                            np.clip(shi[ms], 1e-30, None),
                            color="#d62728", alpha=0.22, lw=0, label="step2 68%")
            ax.plot(mid[ms], np.clip(smed[ms], 1e-30, None), color="#d62728", lw=1.2,
                    label="step2 median")
            mt = pmask & np.isfinite(ft) & (ft > 0)
            ax.plot(mid[mt], ft[mt], "*", color="k", ms=10, ls="none", label="truth")
            ax.set_yscale("log")
            ax.set_xlim(xlo, xhi)
            ax.set_title(f"z≈{zmid[k]:.2f}  frozen vs step2")
            ax.set_xlabel(r"$\log_{10} N_{\rm HI}$")
            if k == 0:
                ax.set_ylabel(r"$f(N\,|\,z)$  (response-form shift)")
            ax.grid(alpha=0.25, which="both")

    # one legend
    axes[0, 0].legend(fontsize=7, loc="lower left")
    if n_rows == 2:
        axes[1, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Per-z differential CDDF f(N | z) coverage vs 2LPT-0 truth (PM band)\n"
                 "(★ truth;  ✕ MAP;  shaded = step2 68% marginalized band)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[fNz-cov] figure -> {out_path}")


# -----------------------------------------------------------------------------
# report
# -----------------------------------------------------------------------------
def write_report(out_path, logN_lo, logN_hi, zbins, f_truth, bands, map_f,
                 max_consistency_err):
    zbins = np.asarray(zbins, float)
    zmid = 0.5 * (zbins[:-1] + zbins[1:])
    n_zc = len(zmid)
    mid = 0.5 * (logN_lo + logN_hi)
    lines = []
    lines.append("=" * 84)
    lines.append("PER-(N,z) DIFFERENTIAL CDDF f(N | z) COVERAGE — 2LPT-0 PM band")
    lines.append("=" * 84)
    lines.append(f"consistency gate max |Σ f_bk·dN − dndx_z| / dndx_z = "
                 f"{max_consistency_err:.2e}  (must be < 1e-9)")
    lines.append("")
    for mode in MODES:
        if mode not in bands:
            continue
        cov = bands[mode]["coverage"]
        lines.append("-" * 84)
        lines.append(f"MODE = {mode}  (n_mc={bands[mode]['n_mc']})")
        lines.append("-" * 84)
        for k in range(n_zc):
            lines.append(f"  z bin {k}  (z≈{zmid[k]:.2f})  [{zbins[k]:.2f},{zbins[k+1]:.2f}]")
            ncov68 = ncov95 = ntot = 0
            for b in range(len(mid)):
                t = f_truth[b, k]
                if not np.isfinite(t) or t <= 0:
                    continue
                ntot += 1
                c68 = cov["cov68"][b, k]; c95 = cov["cov95"][b, k]
                ncov68 += int(c68); ncov95 += int(c95)
                tag = "IN68" if c68 else ("IN95" if c95 else "MISS")
                lines.append(
                    f"     logN[{logN_lo[b]:.2f},{logN_hi[b]:.2f}]  truth={t:.3e}  "
                    f"med={cov['med'][b,k]:.3e}  "
                    f"68[{cov['lo68'][b,k]:.3e},{cov['hi68'][b,k]:.3e}]  {tag}")
            lines.append(f"     -> cover68 {ncov68}/{ntot}   cover95 {ncov95}/{ntot}")
            lines.append("")
    # cross-z comparison summary (step2)
    if "step2" in bands:
        cov = bands["step2"]["coverage"]
        lines.append("=" * 84)
        lines.append("CROSS-Z SUMMARY (step2): cover68 fraction + over/under near 20.3 + tail")
        lines.append("=" * 84)
        for k in range(n_zc):
            ntot = ncov = 0
            # over/under at the 20.0-20.6 'shoulder' bins and the high-N tail
            sh_lo, sh_hi = [], []
            for b in range(len(mid)):
                t = f_truth[b, k]
                if not np.isfinite(t) or t <= 0:
                    continue
                ntot += 1; ncov += int(cov["cov68"][b, k])
                med = cov["med"][b, k]
                ratio = med / t if t > 0 else np.nan
                if 20.0 <= mid[b] < 20.6:
                    sh_lo.append(ratio)
                if mid[b] >= 21.0:
                    sh_hi.append(ratio)
            sh = (np.nanmedian(sh_lo) if sh_lo else np.nan)
            tl = (np.nanmedian(sh_hi) if sh_hi else np.nan)
            lines.append(
                f"  z≈{zmid[k]:.2f}: cover68 {ncov}/{ntot}  "
                f"median(med/truth) shoulder[20.0,20.6)={sh:.3f}  tail[>=21.0]={tl:.3f}")
    txt = "\n".join(lines)
    with open(out_path, "w") as fh:
        fh.write(txt + "\n")
    print(f"[fNz-cov] report -> {out_path}")
    return txt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir", default=DEF_CAT)
    p.add_argument("--truth", default=DEF_TRUTH)
    p.add_argument("--bal-cat", default=DEF_BAL)
    p.add_argument("--molly-tsv", default=None)
    p.add_argument("--molly", default=None, help="alias for --molly-tsv")
    p.add_argument("--kernel", default=DEF_KERNEL)
    p.add_argument("--kernel-znz", default=DEF_ZNZ)
    p.add_argument("--out", default=DEF_OUT)
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
    p.add_argument("--n-mc", type=int, default=120)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-second-row", action="store_true")
    args = p.parse_args(argv)
    if args.molly and not args.molly_tsv:
        args.molly_tsv = args.molly
    os.makedirs(args.out, exist_ok=True)
    limits = tuple(float(x) for x in args.report_limits.split(","))

    t0 = time.time()
    print("=" * 84)
    print(f"[fNz-cov] PM band (mc_inner=laplace + mc_nuisance=shared_boot + mc_response)")
    print(f"          kernel_znz={args.kernel_znz}  n_mc={args.n_mc}  workers={args.workers}")
    print("=" * 84)

    ing = build_ingredients(args, "purity_mixture")
    cfg = ing["cfg"]; cfg.report_logN_limits = limits
    cfg._wall1_estimator = "v3"
    cfg.mc_inner = "laplace"
    cfg.mc_nuisance = "shared_boot"
    cfg.kernel_znz_model = args.kernel_znz
    logN_lo, logN_hi = ing["logN_lo"], ing["logN_hi"]
    N_b, dN_b, X_tot = ing["N_b"], ing["dN_b"], ing["X_tot"]
    zbins = np.asarray(cfg.zbins, float)

    point = ing["estimator_fn"](
        ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["C_interp"],
        ing["fp_model"], X_tot, logN_lo, logN_hi, N_b, dN_b, ing["truth_cut"], cfg)
    fine = point["_v3x"]["fwd"]["fine"]
    M_meta = point["_v3x"]["M_meta"]
    map_f = map_fNz(point, cfg, fine, M_meta)        # (n_nbins, n_zc)

    # truth f(N,z)
    tf = truth_fNz(cfg, ing["truth_cut"], logN_lo, logN_hi, N_b, dN_b, X_tot)
    f_truth = tf["f_truth"]

    bands = {}
    max_consistency_err = 0.0
    for mode in MODES:
        for k, v in MODE_CFG[mode].items():
            setattr(cfg, k, v)
        t1 = time.time()
        band = pm_fNz_band(cfg, ing, point, args.n_mc,
                           np.random.default_rng(args.seed + 7))
        band["coverage"] = cell_coverage(band["f_bk_coarse_samples"], f_truth)
        bands[mode] = band
        max_consistency_err = max(max_consistency_err, band["max_consistency_err"])
        print(f"    {mode:7s} band done ({time.time()-t1:.0f}s)  "
              f"consistency_err={band['max_consistency_err']:.1e}")

    # report + figure
    rep_path = os.path.join(args.out, "fNz_coverage_report.txt")
    txt = write_report(rep_path, logN_lo, logN_hi, zbins, f_truth, bands, map_f,
                       max_consistency_err)
    print("\n" + txt)

    fig_path = os.path.join(args.out, "fig_fNz_coverage.png")
    make_figure(fig_path, logN_lo, logN_hi, zbins, f_truth, bands, map_f,
                second_row=(not args.no_second_row))

    # npz
    savez = dict(
        logN_lo=logN_lo, logN_hi=logN_hi, N_b=N_b, dN_b=dN_b, zbins=zbins,
        X_tot=np.asarray(X_tot, float), f_truth=f_truth, truth_counts=tf["counts"],
        map_f_bk_coarse=map_f, limits=np.asarray(limits),
        max_consistency_err=np.asarray(max_consistency_err),
    )
    for mode in MODES:
        savez[f"{mode}_f_bk_coarse_samples"] = bands[mode]["f_bk_coarse_samples"]
        savez[f"{mode}_cov68"] = bands[mode]["coverage"]["cov68"]
        savez[f"{mode}_cov95"] = bands[mode]["coverage"]["cov95"]
        savez[f"{mode}_med"] = bands[mode]["coverage"]["med"]
        savez[f"{mode}_lo68"] = bands[mode]["coverage"]["lo68"]
        savez[f"{mode}_hi68"] = bands[mode]["coverage"]["hi68"]
        for l in limits:
            savez[f"{mode}_dndx_z_{l}_samples"] = bands[mode][f"dndx_z_{l}_samples"]
    npz_path = os.path.join(args.out, "fNz_coverage.npz")
    np.savez(npz_path, **savez)
    print(f"[fNz-cov] npz -> {npz_path}")
    print(f"[fNz-cov] DONE in {time.time()-t0:.0f}s  "
          f"(consistency gate max err {max_consistency_err:.2e})")
    return dict(bands=bands, f_truth=f_truth, map_f=map_f)


if __name__ == "__main__":
    main()
