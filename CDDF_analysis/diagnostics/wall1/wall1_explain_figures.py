"""wall1_explain_figures.py — render the 5 explanatory figures for the WALL-1
tilt-closure + recovered-CDDF-posterior doc, from the partA/partB npz products.
Reduce-only (reads npz, plots). NO inference, NO MC.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

DPI = 145


def _band(samp, axis=0):
    return (np.nanpercentile(samp, 2.5, axis=axis), np.nanpercentile(samp, 16, axis=axis),
            np.nanpercentile(samp, 50, axis=axis), np.nanpercentile(samp, 84, axis=axis),
            np.nanpercentile(samp, 97.5, axis=axis))


def fig1_tilt_schematic(B, out):
    mid = B["mid"]; pivot = float(B["pivot"])
    f0 = B["f_truth_zero"]; fp = B["f_truth_plus"]; fm = B["f_truth_minus"]
    fr = B["f_truth_real"]
    show = (mid >= 19.5 - 1e-9) & (f0 > 0)
    fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
    ax.plot(mid[show], f0[show], "k-", lw=2.2, label=r"2LPT truth $f(N)$  ($\Delta\alpha=0$)")
    ax.plot(mid[show], fp[show], color="C3", lw=1.8, ls="--",
            label=r"$+\Delta\alpha=+0.5$  (steepen tail up)")
    ax.plot(mid[show], fm[show], color="C0", lw=1.8, ls="--",
            label=r"$-\Delta\alpha=-0.5$  (flatten tail down)")
    ax.plot(mid[show], fr[show], color="C2", lw=1.4, ls=":",
            label=r"realistic $\Delta\alpha=0.015$  (operating pt; ~truth)")
    ax.axvline(pivot, color="0.4", lw=1.0, ls="-.")
    ax.text(pivot + 0.02, 3e-22, rf"pivot $\log N={pivot:.1f}$", rotation=90,
            va="center", fontsize=8.5, color="0.3")
    ax.set_yscale("log"); ax.set_xlim(19.5, 22.4)
    ax.set_xlabel(r"$\log_{10} N_{\rm HI}\ [{\rm cm}^{-2}]$")
    ax.set_ylabel(r"$f(N_{\rm HI}, X)$")
    ax.set_title("WALL-1 tilt mechanism: re-tilt the TRUE CDDF slope about 20.3")
    ax.annotate(r"$w(N)=10^{\,\Delta\alpha\,(\log N - 20.3)}$" "\n"
                r"applied to EACH true absorber & to each detection's true host",
                xy=(0.04, 0.06), xycoords="axes fraction", fontsize=9.5,
                bbox=dict(boxstyle="round", fc="0.95", ec="0.6"))
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    p = os.path.join(out, "fig1_tilt_schematic.png")
    fig.savefig(p, dpi=DPI); plt.close(fig); return p


def fig2_wall1_closure(B, out):
    """recovered-vs-injected under tilt: per-bin recovered f_b (95% MC) overlaid on
    the injected tilted truth & the closure target R0*truth^tilt, both tilts;
    lower panel = closure pulls + ±3 band. Annotate integrated pulls + verdict."""
    mid = B["mid"]; gated = B["cached_gated"].astype(bool)
    R0 = B["cached_R0"]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.0), height_ratios=[2.2, 1.0],
                             sharex=True, constrained_layout=True)
    ax, axp = axes
    show = (mid >= 19.5 - 1e-9) & (mid <= 22.0 + 1e-9)
    # +tilt
    ft_p = B["cached_ftrue_plus"]; pr_p = B["cached_fpred_plus"]
    fe_p = B["cached_fest_plus"]; fs_p = B["cached_fstd_plus"]
    ft_m = B["cached_ftrue_minus"]; pr_m = B["cached_fpred_minus"]
    fe_m = B["cached_fest_minus"]; fs_m = B["cached_fstd_minus"]
    s = show & (ft_p > 0)
    ax.errorbar(mid[s], fe_p[s], yerr=1.96 * fs_p[s], fmt="o", color="C3", ms=4,
                capsize=2, label=r"recovered $f^{tilt}$ ($+0.5$, 95% MC)")
    ax.plot(mid[s], pr_p[s], "^--", color="C3", ms=4, alpha=0.55,
            label=r"closure target $R_0\,$truth$^{+0.5}$")
    s = show & (ft_m > 0)
    ax.errorbar(mid[s], fe_m[s], yerr=1.96 * fs_m[s], fmt="s", color="C0", ms=4,
                capsize=2, label=r"recovered $f^{tilt}$ ($-0.5$, 95% MC)")
    ax.plot(mid[s], pr_m[s], "v--", color="C0", ms=4, alpha=0.55,
            label=r"closure target $R_0\,$truth$^{-0.5}$")
    ax.set_yscale("log"); ax.set_ylabel(r"$f(N_{\rm HI}, X)$")
    ax.axvline(20.3, color="0.5", lw=0.8, ls="-.")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("WALL-1 closure: recovered $f^{tilt}$ vs the gated target "
                 r"$R_0\cdot$truth$^{tilt}$  (2LPT-0, $\Delta\alpha=\pm0.5$)")
    # lower: closure pulls
    pc_p = B["cached_pull_closure_plus"]; pc_m = B["cached_pull_closure_minus"]
    axp.plot(mid[show & np.isfinite(pc_p)], pc_p[show & np.isfinite(pc_p)],
             "o-", color="C3", ms=3.5, label=r"$\Delta\alpha=+0.5$")
    axp.plot(mid[show & np.isfinite(pc_m)], pc_m[show & np.isfinite(pc_m)],
             "s-", color="C0", ms=3.5, label=r"$\Delta\alpha=-0.5$")
    axp.axhline(0, color="k", lw=0.6)
    axp.axhspan(-3, 3, color="0.85", alpha=0.6, label=r"$|{\rm pull}|\leq3$")
    axp.axhline(3, color="r", lw=0.7, ls="--"); axp.axhline(-3, color="r", lw=0.7, ls="--")
    axp.axvline(20.3, color="0.5", lw=0.8, ls="-.")
    axp.set_ylim(-30, 30)
    axp.set_xlabel(r"$\log_{10} N_{\rm HI}$")
    axp.set_ylabel(r"closure pull $\frac{f^{tilt}_{\rm est}-R_0 f^{tilt}_{\rm tr}}{\sigma_{\rm MC}}$")
    axp.legend(fontsize=8, loc="upper right")
    axp.grid(alpha=0.3)
    # integrated pulls + verdict annotation (DLA tier, >=20.3)
    pp = B["cached_dndx_total_closure_pull_20.3_plus"]
    pm = B["cached_dndx_total_closure_pull_20.3_minus"]
    op = B["cached_omega_closure_pull_20.3_plus"]
    om = B["cached_omega_closure_pull_20.3_minus"]
    verdict = str(B["cached_verdict"])
    txt = (f"INTEGRATED (>=20.3): dN/dX pull +0.5={pp:+.1f}, -0.5={pm:+.1f}\n"
           f"                      Omega pull +0.5={op:+.1f}, -0.5={om:+.1f}\n"
           f"WALL-1 = {verdict}  (opposite-sign coherent -> V3_KERNEL_SLOPE_DEPENDENCE)")
    axp.annotate(txt, xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8,
                 bbox=dict(boxstyle="round", fc="#fff3f0", ec="C3"))
    p = os.path.join(out, "fig2_wall1_closure.png")
    fig.savefig(p, dpi=DPI); plt.close(fig); return p


def fig3_cddf_posterior_vs_truth(A, out):
    logN_lo = A["logN_lo"]; logN_hi = A["logN_hi"]; mid = 0.5 * (logN_lo + logN_hi)
    f_pt = A["f_b_point"]; f_tr = A["f_truth"]
    full = A["full_f_b_samples"]; lap = A["lap_f_b_samples"]
    lo95, lo68, med, hi68, hi95 = _band(full)
    llo95, _, _, _, lhi95 = _band(lap)
    show = (mid >= 19.5 - 1e-9) & (f_tr > 0)
    fig, ax = plt.subplots(figsize=(7.8, 5.6), constrained_layout=True)
    ax.fill_between(mid[show], np.clip(lo95[show], 1e-300, None),
                    np.clip(hi95[show], 1e-300, None), color="C0", alpha=0.18,
                    label="95% full posterior (θ⊕nuisance)")
    ax.fill_between(mid[show], np.clip(lo68[show], 1e-300, None),
                    np.clip(hi68[show], 1e-300, None), color="C0", alpha=0.34,
                    label="68% full posterior")
    ax.plot(mid[show], np.clip(llo95[show], 1e-300, None), color="C1", lw=0.9, ls=":",
            label="95% θ-only (Laplace, nuisance frozen)")
    ax.plot(mid[show], np.clip(lhi95[show], 1e-300, None), color="C1", lw=0.9, ls=":")
    ax.plot(mid[show], np.clip(f_pt[show], 1e-300, None), color="C0", lw=1.6,
            label="recovered MAP $f(N)$")
    ax.plot(mid[show], np.clip(f_tr[show], 1e-300, None), "k-", lw=2.0,
            label="2LPT injected truth")
    ax.axvspan(21.0, 21.5, color="gold", alpha=0.18)
    ax.text(21.25, ax.get_ylim()[0] * 1.0 if False else 2e-23, "[21,21.5]\ntail",
            ha="center", fontsize=8, color="0.3")
    ax.axvline(20.3, color="0.5", lw=0.8, ls="-.")
    ax.set_yscale("log"); ax.set_xlim(19.5, 22.4)
    ax.set_xlabel(r"$\log_{10} N_{\rm HI}\ [{\rm cm}^{-2}]$")
    ax.set_ylabel(r"$f(N_{\rm HI}, X)$")
    ax.set_title("Recovered $f(N)$ with full population-posterior band vs 2LPT truth")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    p = os.path.join(out, "fig3_cddf_posterior_vs_truth.png")
    fig.savefig(p, dpi=DPI); plt.close(fig); return p


def fig4_dndx_z_posterior(A, out):
    zbins = A["zbins"]; limits = A["report_limits"]
    zmid = 0.5 * (zbins[:-1] + zbins[1:])
    fig, ax = plt.subplots(figsize=(7.8, 5.4), constrained_layout=True)
    colors = {20.0: "C0", 20.3: "C3", 20.6: "C2"}
    for l in limits:
        c = colors.get(float(l), "C4")
        samp = A[f"full_dndx_z_{l}_samples"]   # (n_mc, n_zbins)
        lo95, lo68, med, hi68, hi95 = _band(samp, axis=0)
        # MAP per-z dN/dX from the point estimate's dndx_z is not stored; use the band
        # median as the recovered curve (the MC median; the doc notes the +drift). We
        # plot the truth and the 68/95 band.
        tr = A[f"truth_dndx_z_{l}"]
        ax.fill_between(zmid, lo95, hi95, color=c, alpha=0.15)
        ax.fill_between(zmid, lo68, hi68, color=c, alpha=0.30)
        ax.plot(zmid, med, "-o", color=c, ms=4, label=fr"recovered $\geq{l}$ (MC median, 68/95%)")
        ax.plot(zmid, tr, "--s", color=c, ms=5, mfc="none",
                label=fr"truth $\geq{l}$")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$dN/dX\,(\geq \log N)$")
    ax.set_title("Recovered $dN/dX(z)$ with full-posterior band vs 2LPT truth")
    ax.set_xticks(zmid); ax.set_xticklabels([f"{a:.2f}–{b:.2f}" for a, b in
                                             zip(zbins[:-1], zbins[1:])])
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3)
    p = os.path.join(out, "fig4_dndx_z_posterior.png")
    fig.savefig(p, dpi=DPI); plt.close(fig); return p


def fig5_omega_posterior(A, out):
    limits = [float(x) for x in A["report_limits"]]
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    xs = np.arange(len(limits))
    for i, l in enumerate(limits):
        samp = A[f"full_omega_{l}_samples"]
        lo95, lo68, med, hi68, hi95 = (np.nanpercentile(samp, q) for q in
                                       (2.5, 16, 50, 84, 97.5))
        pt = float(A[f"omega_{l}_point"]); tr = float(A[f"omega_{l}_truth"])
        ax.vlines(xs[i], lo95, hi95, color="C0", lw=2.0, alpha=0.5)
        ax.vlines(xs[i], lo68, hi68, color="C0", lw=6.0, alpha=0.5)
        ax.plot(xs[i], med, "o", color="C0", ms=7,
                label=("recovered MC median (68/95% band)" if i == 0 else None))
        ax.plot(xs[i], pt, "x", color="C1", ms=10, mew=2,
                label=("recovered MAP" if i == 0 else None))
        ax.plot(xs[i], tr, "*", color="k", ms=14,
                label=("2LPT truth" if i == 0 else None))
    ax.set_xticks(xs); ax.set_xticklabels([fr"$\geq{l}$" for l in limits])
    ax.set_xlabel(r"$\log_{10} N_{\rm HI}$ integration limit")
    ax.set_ylabel(r"$\Omega_{\rm HI}$ (per limit)")
    ax.set_title("Recovered $\\Omega_{\\rm HI}$ with full-posterior band vs 2LPT truth")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    p = os.path.join(out, "fig5_omega_posterior.png")
    fig.savefig(p, dpi=DPI); plt.close(fig); return p


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--partA",
                   default="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                           "wall1_explain_partA/partA_posterior.npz")
    p.add_argument("--partB",
                   default="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                           "wall1_explain_partB/partB_tilt.npz")
    p.add_argument("--out",
                   default="/home/mfho/desi_gpy_dla_notes/notes/figures/"
                           "2026-06-17_wall1_explain")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    A = dict(np.load(args.partA, allow_pickle=True))
    B = dict(np.load(args.partB, allow_pickle=True))
    paths = []
    paths.append(fig1_tilt_schematic(B, args.out))
    paths.append(fig2_wall1_closure(B, args.out))
    paths.append(fig3_cddf_posterior_vs_truth(A, args.out))
    paths.append(fig4_dndx_z_posterior(A, args.out))
    paths.append(fig5_omega_posterior(A, args.out))
    for pp in paths:
        print("  saved", pp)
    return paths


if __name__ == "__main__":
    main()
