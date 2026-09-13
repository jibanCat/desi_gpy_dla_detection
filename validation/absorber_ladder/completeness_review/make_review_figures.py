#!/usr/bin/env python
"""make_review_figures.py — the PI-facing C1 completeness visual review
(figures C1-C5) plus the low-S/N forensic numbers.

PI ruling 2026-09-13c §6.  VALIDATION-ONLY, CALIBRATION/MOCK INFORMATION ONLY:
no sampler is run, no real data is read, no existing tracked file is modified,
nothing is committed.  The only products are PNGs under
``figures/2026-09-13_completeness_review/`` in the notes repo and a JSON of the
numbers quoted in the memo.

Run (login node, ~1 minute):

    conda activate gpdla-hbi
    PYTHONPATH=/home/mfho/wt_abs_diag_2026-09 JAX_PLATFORMS=cpu \
        python validation/absorber_ladder/completeness_review/make_review_figures.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from validation.absorber_ladder.completeness.cal_fit import (          # noqa: E402
    design_2d_additive, irls_binomial, expit)
from validation.absorber_ladder.completeness_review.review_lib import (  # noqa: E402
    wilson_interval, standardised_residual, additive_logit_C, phi_measured,
    effective_completeness, fold_tp_2d, fold_tp_3d, snr_marginal, pooled_rate)

ROOT = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13"
DIAG = "/scratch/cavestru_root/cavestru0/mfho/absorber_diag_2026-09-13"
FIGDIR = "/home/mfho/desi_gpy_dla_notes/figures/2026-09-13_completeness_review"
FAMS = ("2lpt0", "london0", "saclay0")
FAMLAB = {"2lpt0": "2LPT-0", "london0": "London-0", "saclay0": "Saclay-0"}
CAL_FAM = "2lpt0"
X0 = 20.0
DEG_N, DEG_U = 3, 2

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.2, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 130, "savefig.dpi": 130,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "mathtext.fontset": "dejavuserif", "lines.linewidth": 1.2,
})
CLR = {"raw": "#22223b", "C1nsadd": "#1b7837", "C1n": "#2166ac",
       "C1g": "#8073ac", "C0": "#b2182b", "Ctrue": "#e08214", "phi": "#4d4d4d"}


# --------------------------------------------------------------------------
def load_all():
    d = {}
    d["cal"] = {f: np.load(f"{ROOT}/completeness/cal_table_{f}.npz", allow_pickle=True)
                for f in FAMS}
    d["C"] = {v: np.load(f"{ROOT}/completeness/C_{v}_{CAL_FAM}.npz", allow_pickle=True)
              for v in ("C0", "C1g", "C1n", "C1ns", "C1nsadd")}
    d["ops"] = {f: np.load(f"{ROOT}/support/empirical_ops_{f}_A0.npz", allow_pickle=True)
                for f in FAMS}
    d["prov"] = json.loads(str(d["C"]["C1nsadd"]["provenance"].item()))
    return d


def grid_from(cal):
    ne = np.asarray(cal["ntrue_edges"], float)
    Nc = 0.5 * (ne[:-1] + ne[1:])
    tot = np.asarray(cal["truth_bks"], float)
    live = tot.sum(axis=(0, 1)) > 0
    live_idx = np.where(live)[0]
    logmed = np.asarray(cal["snr_logmed_s"], float)
    w = tot.sum(axis=(0, 1))[live_idx]
    u0 = float((logmed[live_idx] * w).sum() / w.sum())
    u_s = np.where(live, logmed - u0, 0.0)
    return dict(Nc=Nc, edges=ne, x_b=Nc - X0, live=live, live_idx=live_idx,
                u0=u0, u_s=u_s, snr_med=np.asarray(cal["snr_med_s"], float),
                snr_edges=np.asarray(cal["snr_edges"], float))


def zpool(a):
    """(B, Kf, S) count table -> (S, B) z-pooled."""
    return np.asarray(a, float).sum(axis=1).T


def fit_additive(det_sb, tot_sb, x_b, u_live):
    """Refit C1nsadd (poly_3(x) + poly_2(u)) on a (S_live, B) count pair."""
    X = design_2d_additive(x_b, u_live, DEG_N, DEG_U)
    st = irls_binomial(X, det_sb.ravel(), tot_sb.ravel(), ridge=1e-6)
    return st, expit(X @ st["beta"]).reshape(det_sb.shape)



def rate_and_yerr(n_det, n_tot, z=1.0):
    """Cell rate plus the Wilson error bars, clipped at 0.

    The clip is float-rounding hygiene only: at an observed rate of exactly 1
    the Wilson upper limit is 1 up to ~2e-16, and matplotlib refuses a negative
    yerr.  It never hides a real asymmetry (the largest clip is < 1e-15).
    """
    d = np.asarray(n_det, float)
    t = np.asarray(n_tot, float)
    ok = t > 0
    p = np.where(ok, d / np.maximum(t, 1.0), np.nan)
    lo, hi = wilson_interval(d, t, z=z)
    return p, np.vstack([np.maximum(p - lo, 0.0), np.maximum(hi - p, 0.0)])


# ==========================================================================
# FIGURE C1 — completeness vs true N, per S/N stratum
# ==========================================================================
def fig_C1(D, G, out):
    cal = D["cal"][CAL_FAM]
    det, tot = zpool(cal["det_bks"]), zpool(cal["truth_bks"])
    Nc, x_b = G["Nc"], G["x_b"]
    Cadd = np.asarray(D["C"]["C1nsadd"]["C_fixed"], float)
    Cn = np.asarray(D["C"]["C1n"]["C_fixed"], float)
    Cg = np.asarray(D["C"]["C1g"]["C_fixed"], float)
    C0 = np.asarray(D["C"]["C0"]["C_fixed"], float)
    beta = np.asarray(D["prov"]["coefficients"], float)


    show = [2, 3, 5, 7]
    Ngrid = np.linspace(Nc[0] - 0.05, Nc[-1] + 0.05, 400)
    fig, ax = plt.subplots(2, len(show), figsize=(13.0, 6.1), sharey="row")
    for j, s in enumerate(show):
        u = G["u_s"][s]
        curve = additive_logit_C(Ngrid - X0, u, beta, DEG_N, DEG_U)
        rr, ee = rate_and_yerr(det[s], tot[s])
        for row in (0, 1):
            a = ax[row, j]
            a.errorbar(Nc, rr, yerr=ee, fmt="o", ms=3.0, lw=0.9, color=CLR["raw"], zorder=5,
                       label=r"calibration $n_{\rm det}/n_{\rm tot}$ (Wilson $1\sigma$)")
            # C0 is constant inside each molly cell, so stepping it on the latent
            # bin edges reproduces the molly step structure exactly
            a.step(G["edges"], np.r_[C0[s], C0[s][-1]], where="post",
                   color=CLR["C0"], lw=1.0, alpha=0.85,
                   label=r"frozen molly $C_0$ (step)")
            a.plot(Ngrid, curve, color=CLR["C1nsadd"], lw=1.7, label=r"C1nsadd (6 coef.)")
            a.plot(Nc, Cn[s], color=CLR["C1n"], lw=1.0, ls="--", label=r"C1n (12 coef.)")
            a.plot(Nc, Cg[s], color=CLR["C1g"], lw=0.9, ls=":", label=r"C1g (96 cells)")
            a.set_xlabel(r"true $\log_{10} N_{\rm HI}$")
            if j == 0:
                a.set_ylabel("completeness $C_{\\rm det}$")
        ax[0, j].set_title(rf"S/N stratum $s={s}$  ({G['snr_edges'][s]:.0f}"
                           + (r"$-\infty$" if not np.isfinite(G['snr_edges'][s + 1])
                              else f"$-${G['snr_edges'][s+1]:.0f}")
                           + rf"),  median $={G['snr_med'][s]:.2f}$")
        ax[0, j].set_xlim(18.95, 22.45)
        ax[0, j].set_ylim(-0.02, 1.05)
        ax[1, j].set_xlim(19.45, 20.75)
        ax[1, j].set_ylim(0.40, 1.02)
        ax[1, j].axvline(20.3, color="0.5", lw=0.7, ls="-.")
        ax[1, j].set_title("zoom 19.5--20.7", fontsize=8)
    ax[0, 0].legend(loc="lower right", framealpha=0.92)
    fig.suptitle("C1 — fixed completeness vs true $N_{\\rm HI}$, calibration family 2LPT-0 "
                 "(top: full reported range; bottom: zoom).  CANDIDATE, NOT ADOPTED.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(f"{out}/C1_completeness_vs_N_by_stratum.png")
    plt.close(fig)


def fig_C1b(D, G, out):
    """Transfer families on the same axes — fit is 2LPT-0 only."""
    Nc = G["Nc"]
    beta = np.asarray(D["prov"]["coefficients"], float)
    show = [2, 7]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    Ngrid = np.linspace(19.0, 22.4, 400)
    mk = {"2lpt0": "o", "london0": "s", "saclay0": "^"}
    for j, s in enumerate(show):
        for f in FAMS:
            cal = D["cal"][f]
            det, tot = zpool(cal["det_bks"]), zpool(cal["truth_bks"])
            r, ee = rate_and_yerr(det[s], tot[s])
            ax[j].errorbar(Nc, r, yerr=ee, fmt=mk[f], ms=3.0, lw=0.8,
                           alpha=0.85, label=FAMLAB[f])
        ax[j].plot(Ngrid, additive_logit_C(Ngrid - X0, G["u_s"][s], beta, DEG_N, DEG_U),
                   color=CLR["C1nsadd"], lw=1.7, label="C1nsadd (fit: 2LPT-0)")
        ax[j].set_title(rf"$s={s}$, median S/N $={G['snr_med'][s]:.2f}$")
        ax[j].set_xlabel(r"true $\log_{10} N_{\rm HI}$")
        ax[j].set_xlim(19.4, 21.8)
    ax[0].set_ylabel(r"completeness $C_{\rm det}$")
    ax[0].legend(loc="lower right")
    fig.suptitle("C1b — transfer: the 2LPT-0 object against the two held-out mock families",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{out}/C1b_transfer_families.png")
    plt.close(fig)


# ==========================================================================
# FIGURE C2 — completeness vs S/N at fixed true N
# ==========================================================================
def fig_C2(D, G, out, numbers):
    cal = D["cal"][CAL_FAM]
    det, tot = zpool(cal["det_bks"]), zpool(cal["truth_bks"])
    beta = np.asarray(D["prov"]["coefficients"], float)
    C0 = np.asarray(D["C"]["C0"]["C_fixed"], float)
    Cadd = np.asarray(D["C"]["C1nsadd"]["C_fixed"], float)
    li = G["live_idx"]
    med = G["snr_med"][li]
    targets = [19.7, 20.0, 20.3, 20.6, 21.0]
    # the additive quadratic in u = log10(S/N) - u0 turns over at
    # u* = -beta_u / (2 beta_u2): above it the fitted C DECREASES with S/N
    u_turn = -beta[DEG_N + 1] / (2.0 * beta[DEG_N + 2])
    snr_turn = 10.0 ** (u_turn + G["u0"])
    snr_grid = np.geomspace(1.6, 25.0, 300)
    u_grid = np.log10(snr_grid) - G["u0"]
    fig, ax = plt.subplots(1, len(targets), figsize=(14.2, 3.4), sharey=True)
    numbers["C2_extrapolation"] = dict(
        snr_median_live=[float(v) for v in med],
        calibrated_log10_snr_range=[float(np.log10(med).min()), float(np.log10(med).max())],
        logSNR_turning_point_u=float(u_turn), snr_turning_point=float(snr_turn),
        note=("the quadratic in u = log10(S/N) - u0 is calibrated ONLY at the six "
              "stratum medians; anything off those six points is interpolation, and "
              "below 2.44 / above 10.68 it is extrapolation"))
    for j, Nt in enumerate(targets):
        b = int(np.argmin(np.abs(G["Nc"] - Nt)))
        a = ax[j]
        a.plot(snr_grid, additive_logit_C(Nt - X0, u_grid, beta, DEG_N, DEG_U),
               color=CLR["C1nsadd"], lw=1.7, label="C1nsadd (continuous)")
        a.plot(snr_grid, additive_logit_C(G["Nc"][b] - X0, u_grid, beta, DEG_N, DEG_U),
               color=CLR["C1nsadd"], lw=0.8, ls="--", alpha=0.7,
               label=rf"C1nsadd at bin centre {G['Nc'][b]:.2f}")
        r, ee = rate_and_yerr(det[li, b], tot[li, b])
        a.errorbar(med, r, yerr=ee, fmt="o", ms=4, lw=1.0,
                   color=CLR["raw"], zorder=5, label="calibration cells (Wilson)")
        a.step(np.r_[med, 30.0], np.r_[C0[li, b], C0[li[-1], b]], where="post",
               color=CLR["C0"], lw=1.0, label=r"frozen molly $C_0$")
        a.axvline(snr_turn, color="0.35", lw=0.9, ls=(0, (4, 2)))
        a.axvspan(1.6, med[0], color="0.85", alpha=0.55, lw=0)
        a.axvspan(med[-1], 25.0, color="0.85", alpha=0.55, lw=0)
        a.set_xscale("log")
        a.set_xlim(1.6, 25.0)
        a.set_xticks([2, 3, 5, 10, 20])
        a.set_xticklabels(["2", "3", "5", "10", "20"])
        a.set_xlabel("median S/N of stratum")
        a.set_title(rf"true $\log_{{10}} N_{{\rm HI}} = {Nt:.1f}$  (bin {G['Nc'][b]:.2f})")
        numbers.setdefault("C2_curve", {})[f"N={Nt}"] = dict(
            C_at_snr={f"{v:.2f}": float(additive_logit_C(Nt - X0, np.log10(v) - G["u0"],
                                                         beta, DEG_N, DEG_U))
                      for v in (2.0, 2.44, 3.45, 5.47, 10.68, 20.0)})
    ax[0].set_ylabel(r"completeness $C_{\rm det}$")
    ax[0].set_ylim(0.0, 1.05)
    ax[0].legend(loc="lower right", framealpha=0.92)
    fig.suptitle("C2 — completeness vs S/N at fixed true $N_{\\rm HI}$.  Grey bands = "
                 "OUTSIDE the calibrated stratum-median range (2.44--10.68): extrapolation "
                 "of the quadratic in $\\log_{10}$ S/N.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(f"{out}/C2_completeness_vs_snr.png")
    plt.close(fig)


# ==========================================================================
# FIGURE C3 — the 2-D surface with the calibration support
# ==========================================================================
def fig_C3(D, G, out, numbers):
    cal = D["cal"][CAL_FAM]
    tot = zpool(cal["truth_bks"])
    beta = np.asarray(D["prov"]["coefficients"], float)
    li, med = G["live_idx"], G["snr_med"][G["live_idx"]]
    Ngrid = np.linspace(19.0, 22.4, 260)
    snr_grid = np.geomspace(1.6, 25.0, 240)
    S = additive_logit_C(Ngrid[:, None] - X0, (np.log10(snr_grid) - G["u0"])[None, :],
                         beta, DEG_N, DEG_U)
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.2))
    im = ax[0].pcolormesh(Ngrid, snr_grid, S.T, cmap="viridis", vmin=0, vmax=1,
                          shading="auto")
    cs = ax[0].contour(Ngrid, snr_grid, S.T, levels=[0.2, 0.5, 0.8, 0.9, 0.95, 0.99],
                       colors="w", linewidths=0.8)
    ax[0].clabel(cs, fmt="%.2f", fontsize=6.5)
    fig.colorbar(im, ax=ax[0], label=r"$C_{\rm C1nsadd}(N,\,{\rm S/N})$")
    # calibration support: truth systems per (b, s), drawn at the cell centres
    n_bs = tot[li, :]                                        # (S_live, B)
    ax[0].scatter(np.tile(G["Nc"], len(li)), np.repeat(med, len(G["Nc"])),
                  s=np.clip(np.sqrt(n_bs.ravel()) * 0.9, 0.0, 40.0),
                  facecolor="none", edgecolor="w", lw=0.7, alpha=0.9)
    ax[0].set_yscale("log")
    ax[0].set_yticks([2, 3, 5, 10, 20])
    ax[0].set_yticklabels(["2", "3", "5", "10", "20"])
    ax[0].set_xlabel(r"true $\log_{10} N_{\rm HI}$")
    ax[0].set_ylabel("median S/N of stratum")
    ax[0].set_title("C1nsadd surface; circles $\\propto\\sqrt{n_{\\rm truth}}$ per cell")
    ax[0].axhspan(1.6, med[0], color="k", alpha=0.20, lw=0)
    ax[0].axhspan(med[-1], 25.0, color="k", alpha=0.20, lw=0)
    ax[0].axvspan(21.7, 22.4, color="k", alpha=0.20, lw=0)

    lg = np.log10(np.maximum(n_bs, 0.5))
    im2 = ax[1].pcolormesh(G["edges"], np.arange(len(li) + 1) - 0.5, lg,
                           cmap="magma", shading="auto")
    fig.colorbar(im2, ax=ax[1], label=r"$\log_{10} n_{\rm truth}$ per $(b,s)$ cell")
    for i in range(len(li)):
        for b in range(len(G["Nc"])):
            if n_bs[i, b] < 50:
                ax[1].text(G["Nc"][b], i, f"{int(n_bs[i,b])}", ha="center", va="center",
                           fontsize=5.2, color="w")
    ax[1].set_yticks(range(len(li)))
    ax[1].set_yticklabels([f"s={s} ({G['snr_med'][s]:.1f})" for s in li])
    ax[1].set_xlabel(r"true $\log_{10} N_{\rm HI}$")
    ax[1].set_title("calibration support (counts printed where $n<50$)")
    ax[1].axvline(21.7, color="c", lw=1.2)
    numbers["C3_support"] = dict(
        n_live_cells=int(n_bs.size),
        cells_lt_50=int((n_bs < 50).sum()), cells_lt_20=int((n_bs < 20).sum()),
        cells_zero=int((n_bs == 0).sum()),
        min_N_of_sparse_cells=float(G["Nc"][np.where((n_bs < 50).any(axis=0))[0].min()]),
        total_truth=float(n_bs.sum()))
    fig.suptitle("C3 — where the 6-coefficient surface is data-constrained.  Shaded = "
                 "outside the calibrated S/N range or above $N=21.7$ (no per-cell support).",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{out}/C3_surface_and_support.png")
    plt.close(fig)


# ==========================================================================
# FIGURE C4 — held-out standardised residuals (E/O parity folds)
# ==========================================================================
def fig_C4(D, G, out, numbers):
    cal = D["cal"][CAL_FAM]
    li, x_b = G["live_idx"], G["x_b"]
    u_live = G["u_s"][li]
    dE, tE = zpool(cal["det_bks_E"])[li], zpool(cal["truth_bks_E"])[li]
    dO, tO = zpool(cal["det_bks_O"])[li], zpool(cal["truth_bks_O"])[li]
    stE, pE = fit_additive(dE, tE, x_b, u_live)      # fitted on even
    stO, pO = fit_additive(dO, tO, x_b, u_live)      # fitted on odd
    rEO = standardised_residual(pE, dO, tO)          # fit even, score odd
    rOE = standardised_residual(pO, dE, tE)
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 3.7),
                           gridspec_kw=dict(width_ratios=[1, 1, 0.85]))
    nrm = TwoSlopeNorm(vmin=-4, vcenter=0.0, vmax=4)
    for j, (r, lab) in enumerate([(rEO, "fit even $\\to$ score odd"),
                                  (rOE, "fit odd $\\to$ score even")]):
        im = ax[j].pcolormesh(G["edges"], np.arange(len(li) + 1) - 0.5, r,
                              cmap="RdBu_r", norm=nrm, shading="auto")
        fig.colorbar(im, ax=ax[j], label=r"$(C_{\rm pred}-C_{\rm held})/\sigma_{\rm binom}$")
        ax[j].set_yticks(range(len(li)))
        ax[j].set_yticklabels([f"s={s} ({G['snr_med'][s]:.1f})" for s in li])
        ax[j].set_xlabel(r"true $\log_{10} N_{\rm HI}$")
        ax[j].set_title(lab)
        ax[j].axhline(-0.5, color="k", lw=2.0)
        ax[j].axhline(0.5, color="k", lw=2.0)
        ax[j].text(19.05, 0.0, "S/N 2--3", fontsize=7, va="center", color="k")
    both = np.concatenate([rEO[np.isfinite(rEO)], rOE[np.isfinite(rOE)]])
    ax[2].hist(both, bins=np.linspace(-5, 5, 41), color="0.65", edgecolor="k", lw=0.4,
               density=True, label="all live cells")
    s2 = np.concatenate([rEO[0][np.isfinite(rEO[0])], rOE[0][np.isfinite(rOE[0])]])
    ax[2].hist(s2, bins=np.linspace(-5, 5, 41), histtype="step", color=CLR["C0"], lw=1.6,
               density=True, label="S/N 2--3 only")
    xx = np.linspace(-5, 5, 200)
    ax[2].plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.0,
               label=r"$\mathcal{N}(0,1)$")
    ax[2].set_xlabel("standardised held-out residual")
    ax[2].set_ylabel("density")
    ax[2].legend()
    numbers["C4_heldout"] = dict(
        rms_all=float(np.sqrt(np.mean(both ** 2))), mean_all=float(both.mean()),
        rms_snr2_3=float(np.sqrt(np.mean(s2 ** 2))), mean_snr2_3=float(s2.mean()),
        frac_abs_gt_3=float(np.mean(np.abs(both) > 3.0)),
        beta_even=[float(v) for v in stE["beta"]],
        beta_odd=[float(v) for v in stO["beta"]],
        heldout_pooled_ratio_by_stratum=[
            float(v) for v in ((pE * tO + pO * tE).sum(axis=1)
                               / np.maximum((dO + dE).sum(axis=1), 1e-12))],
        heldout_pooled_ratio_snr2_3=float(
            ((pE * tO + pO * tE)[0].sum()) / max((dO + dE)[0].sum(), 1e-12)))
    fig.suptitle("C4 — held-out standardised residuals of the 6-coefficient surface "
                 "(TARGETID-parity sightline halves, both directions).", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{out}/C4_heldout_residuals.png")
    plt.close(fig)


# ==========================================================================
# FIGURE C5 — the low-S/N forensic
# ==========================================================================
def fig_C5(D, G, out, numbers):
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors

    cal = D["cal"][CAL_FAM]
    ops = D["ops"][CAL_FAM]
    det, tot = zpool(cal["det_bks"]), zpool(cal["truth_bks"])
    tc = np.asarray(ops["truth_counts_bks"], float)
    ndt = np.asarray(ops["N_det_bks_true_z"], float)
    ndo = np.asarray(ops["N_det_bks_obs_z"], float)
    nda = np.asarray(ops["N_det_all_bks_true_z"], float)
    Ctrue_bs = np.asarray(ops["C_true_bs"], float)                 # (B,S)
    Ctrue_bKs = np.asarray(ops["C_true_bKs"], float)               # (B,KK,S)
    kz = np.asarray(ops["kz_to_K"], int)
    host = np.asarray(ops["hostless_cks"], float)
    obs = np.asarray(ops["counts_obs_cks"], float)
    phi = phi_measured(ndt, nda)                                   # (B,S)
    Cadd = np.asarray(D["C"]["C1nsadd"]["C_fixed"], float)
    C0 = np.asarray(D["C"]["C0"]["C_fixed"], float)
    li, Nc = G["live_idx"], G["Nc"]

    pk = load_pack(f"{ROOT}/support/scanpack_{CAL_FAM}_b300_A0.npz")
    consts, Mg = build_cc_tensors(pk)
    Mg = np.asarray(Mg)
    g_bk = np.asarray(consts.g_bk)
    phi_mod_bks = Mg.sum(axis=2).transpose(2, 1, 0)                # (B,Kf,S)
    wgt = tc
    phi_mod = np.where(wgt.sum(1) > 0,
                       (phi_mod_bks * wgt).sum(1) / np.maximum(wgt.sum(1), 1e-12), 1.0)

    folds = {
        "C1nsadd x g": fold_tp_2d(Mg, Cadd, g_bk, tc),
        "C0 x g (frozen)": fold_tp_2d(Mg, C0, g_bk, tc),
        "C_true_bs x g  [--fix Cz]": fold_tp_2d(Mg, Ctrue_bs.T, g_bk, tc),
        "C_true_bKs, no g  [--fix C]": fold_tp_3d(Mg, Ctrue_bKs[:, kz, :], tc),
        "C1nsadd x phi_meas x g  [double-phi control]":
            fold_tp_2d(Mg, Cadd * phi.T, g_bk, tc),
        "C1nsadd x (phi_meas/phi_model) x g  [phi-corrected]":
            fold_tp_2d(Mg, Cadd * (phi / np.maximum(phi_mod, 1e-12)).T, g_bk, tc),
    }
    marg = {}
    for k, tp in folds.items():
        r, t = snr_marginal(tp + host, obs)
        marg[k] = dict(shape=[float(v) for v in r], pooled_total=float(t))

    fig, ax = plt.subplots(2, 2, figsize=(12.4, 7.4))
    # (a)/(b): the four objects at s = 2 and s = 3
    for j, s in enumerate([2, 3]):
        a = ax[0, j]
        r, ee = rate_and_yerr(det[s], tot[s])
        a.errorbar(Nc, r, yerr=ee, fmt="o", ms=3.5, lw=0.9,
                   color=CLR["raw"], zorder=5,
                   label=r"(ii) matched table $n_{\rm det}/n_{\rm tot}=C_{\rm det}$")
        a.plot(Nc, Ctrue_bs[:, s], "s-", ms=3.0, color=CLR["Ctrue"], lw=1.2,
               label=r"(i) truth-pinned $C_{\rm true}$ = $C_{\rm det}\,\phi$")
        a.plot(Nc, Cadd[s], color=CLR["C1nsadd"], lw=1.7, label="(iii) C1nsadd")
        a.plot(Nc, C0[s], color=CLR["C0"], lw=1.0, ls="--", label=r"(iv) frozen molly $C_0$")
        a.plot(Nc, phi[:, s], color=CLR["phi"], lw=1.0, ls=":",
               label=r"measured in-grid fraction $\phi$")
        a.set_xlim(18.95, 21.9)
        a.set_ylim(-0.02, 1.05)
        a.set_xlabel(r"true $\log_{10} N_{\rm HI}$")
        a.set_title(rf"S/N stratum $s={s}$ (median {G['snr_med'][s]:.2f})")
        if j == 0:
            a.set_ylabel("completeness / fraction")
            a.legend(loc="lower right", framealpha=0.92)
    # (c) phi measured vs phi model
    a = ax[1, 0]
    for s in li:
        a.plot(Nc, phi[:, s], "-o", ms=2.5, lw=1.0,
               label=rf"$\phi_{{\rm meas}}$ $s={s}$" if s in (2, 7) else None,
               color=plt.cm.viridis((s - 2) / 5.0))
        a.plot(Nc, phi_mod[:, s], "--", lw=1.0, color=plt.cm.viridis((s - 2) / 5.0),
               label=rf"$\phi_{{\rm model}}$ $s={s}$" if s in (2, 7) else None)
    a.set_xlim(18.95, 20.3)
    a.set_ylim(0.0, 1.08)
    a.set_xlabel(r"true $\log_{10} N_{\rm HI}$")
    a.set_ylabel(r"in-grid counting fraction $\phi$")
    a.set_title(r"(c) $\phi$: measured (solid) vs the deployed kernel's $\phi_{\rm ref}$ "
                r"(dashed); colour = stratum")
    a.legend(loc="lower right", fontsize=6.5, ncol=2)
    # (d) the S/N marginal of each fold at truth f
    a = ax[1, 1]
    xs = np.arange(len(li))
    styles = {"C1nsadd x g": ("-o", CLR["C1nsadd"]),
              "C0 x g (frozen)": ("--s", CLR["C0"]),
              "C_true_bs x g  [--fix Cz]": ("-^", CLR["Ctrue"]),
              "C_true_bKs, no g  [--fix C]": (":v", "#7f3b08"),
              "C1nsadd x phi_meas x g  [double-phi control]": ("-D", "#01665e"),
              "C1nsadd x (phi_meas/phi_model) x g  [phi-corrected]": ("-.*", "#542788")}
    for k, v in marg.items():
        st, c = styles[k]
        a.plot(xs, v["shape"][2:], st, color=c, ms=4, lw=1.2,
               label=f"{k}  (total {v['pooled_total']:.3f})")
    a.axhline(1.0, color="k", lw=0.8)
    a.set_xticks(xs)
    a.set_xticklabels([f"{s}\n{G['snr_med'][s]:.1f}" for s in li])
    a.set_xlabel("S/N stratum (median S/N below)")
    a.set_ylabel(r"shape of $\mu/{\rm obs}$ (level divided out)")
    a.set_title("(d) deterministic fold at truth $f$: S/N marginal by completeness object")
    a.legend(loc="upper left", fontsize=6.0, framealpha=0.92)
    fig.suptitle("C5 — the S/N 2--3 forensic: the truth-pinned object is $C_{\\rm det}\\,"
                 "\\phi$, and $\\phi$ is already carried by the kernel.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{out}/C5_low_snr_forensic.png")
    plt.close(fig)

    # ---------------- the forensic numbers ---------------------------------
    numbers["C5"] = dict(
        snr_marginal_shape_at_truth_f=marg,
        identity_checks=dict(
            truth_bks_vs_truth_counts_bks=dict(
                cal_total=float(np.asarray(cal["truth_bks"]).sum()),
                ops_total=float(tc.sum()),
                n_cells_differing=int((np.asarray(cal["truth_bks"], float) != tc).sum()),
                ratio_by_stratum=[float(np.asarray(cal["truth_bks"], float)[:, :, s].sum()
                                        / max(tc[:, :, s].sum(), 1e-12)) for s in li]),
            det_bks_vs_N_det_all_true_z=dict(
                cal_total=float(np.asarray(cal["det_bks"]).sum()),
                ops_total=float(nda.sum()),
                max_abs_cell_diff=float(np.abs(np.asarray(cal["det_bks"], float) - nda).max())),
            N_det_true_z_vs_obs_z=dict(true_z=float(ndt.sum()), obs_z=float(ndo.sum()),
                                       max_abs_cell_diff=float(np.abs(ndt - ndo).max())),
            C_true_bs_identity=dict(
                formula="C_true_bs == sum_k N_det_bks_true_z / sum_k truth_counts_bks",
                max_abs_diff=float(np.nanmax(np.abs(
                    pooled_rate(ndt, tc, axis=1) - Ctrue_bs)))),
            phi_pooled=float(ndt.sum() / nda.sum()),
            kernel_phi_is_carried=dict(
                statement="sum_c Mg[s,k,c,b] == phi_ref(b,cell) < 1 at b<=2",
                sum_c_Mg_b0_s2=float(phi_mod[0, 2]), sum_c_Mg_b0_s7=float(phi_mod[0, 7]),
                sum_c_Mg_b3_s2=float(phi_mod[3, 2])),
        ),
        phi_measured_by_b_s={f"b={b} (N={Nc[b]:.2f})":
                             {f"s={s}": float(phi[b, s]) for s in li} for b in range(4)},
        phi_model_by_b_s={f"b={b} (N={Nc[b]:.2f})":
                          {f"s={s}": float(phi_mod[b, s]) for s in li} for b in range(4)},
        example_cells=[
            dict(b=b, N_bin=[float(G["edges"][b]), float(G["edges"][b + 1])], s=s,
                 cal_truth=float(np.asarray(cal["truth_bks"], float)[b, :, s].sum()),
                 cal_det=float(np.asarray(cal["det_bks"], float)[b, :, s].sum()),
                 C_det=float(det[s, b] / max(tot[s, b], 1e-12)),
                 ops_truth=float(tc[b, :, s].sum()),
                 ops_det_all=float(nda[b, :, s].sum()),
                 ops_det_ingrid=float(ndt[b, :, s].sum()),
                 C_true=float(Ctrue_bs[b, s]), phi_meas=float(phi[b, s]),
                 phi_model=float(phi_mod[b, s]),
                 C1nsadd=float(Cadd[s, b]), C0=float(C0[s, b]))
            for (b, s) in [(0, 2), (0, 7), (1, 2), (2, 2), (4, 2), (6, 2)]],
        C_times_g_vs_C_det_true_by_K=_cg_vs_ctrue(Cadd, g_bk, ops, tc, kz, li),
        effective_completeness_residual_pct_by_stratum_and_K=_eff_resid_sK(
            Cadd, g_bk, D["cal"][CAL_FAM], kz, li),
        low_N_attribution=_low_N_attribution(Mg, Cadd, g_bk, phi, tc, host, obs, li),
    )
    return numbers



def _eff_resid_sK(Cadd, g_bk, cal, kz, li):
    """Held-out-free truth-weighted residual of the fold's effective completeness
    ``C x g`` against the matched detections, resolved by (stratum, coarse z).

    Addendum A reports this pooled over strata (+0.36 / -0.36 / -0.22 %).  The
    pooled number hides a large, SIGN-FLIPPING S/N dependence, which this table
    exposes: the frozen ``g`` carries one z-shape per N with no S/N dependence,
    but the truth's z-shape at S/N 2-3 is very different from the one at S/N >= 7.
    """
    E = effective_completeness(Cadd, g_bk)
    tb = np.asarray(cal["truth_bks"], float)
    db = np.asarray(cal["det_bks"], float)
    out = {}
    for s in li:
        row = {}
        for K in range(int(kz.max()) + 1):
            m = kz == K
            den = db[:, m, s].sum()
            row[f"K{K}"] = float(100.0 * ((E[:, m, s] * tb[:, m, s]).sum() / max(den, 1e-12) - 1.0))
        row["all_K"] = float(100.0 * ((E[:, :, s] * tb[:, :, s]).sum() / max(db[:, :, s].sum(), 1e-12) - 1.0))
        out[f"s={s}"] = row
    pooled = {}
    for K in range(int(kz.max()) + 1):
        m = kz == K
        num = (E[:, m, :][:, :, li] * tb[:, m, :][:, :, li]).sum()
        pooled[f"K{K}"] = float(100.0 * (num / max(db[:, m, :][:, :, li].sum(), 1e-12) - 1.0))
    out["pooled_over_live_strata"] = pooled
    return out


def _low_N_attribution(Mg, Cadd, g_bk, phi, tc, host, obs, li):
    """Where the S/N gradient of the fold lives: the migrated sub-19.7 latent
    population.  Removing latent bins b <= 2 REVERSES the gradient, so the
    gradient is entirely a property of how much sub-19.7 mass the kernel puts
    inside the reported grid (its phi), not of the completeness level."""
    mask = np.zeros(Mg.shape[-1])
    mask[:3] = 1.0
    base = fold_tp_2d(Mg, Cadd, g_bk, tc)
    low = fold_tp_2d(Mg * mask[None, None, None, :], Cadd, g_bk, tc)
    dbl = fold_tp_2d(Mg, Cadd * phi.T, g_bk, tc)
    low_d = fold_tp_2d(Mg * mask[None, None, None, :], Cadd * phi.T, g_bk, tc)
    r_no_low, t_no_low = snr_marginal(base - low + host, obs)
    return dict(
        latent_bins_removed="b <= 2, i.e. true log10 N_HI < 19.7",
        share_of_predicted_TP_from_b_le_2=dict(
            C1nsadd=[float(v) for v in (low.sum((0, 1)) / base.sum((0, 1)))[li]],
            double_phi=[float(v) for v in (low_d.sum((0, 1)) / dbl.sum((0, 1)))[li]]),
        snr_shape_with_b_le_2_removed=[float(v) for v in r_no_low[li]],
        pooled_total_with_b_le_2_removed=float(t_no_low))


def _cg_vs_ctrue(Cadd, g_bk, ops, tc, kz, li):
    """Is the fitted z-pooled ``C x g`` equal to the truth completeness by coarse z?

    The comparison is made on the DETECTION-probability convention on both
    sides: the truth side is rebuilt as
    ``C_det_true[b,K,s] = sum_{k in K} N_det_all / sum_{k in K} truth_counts``
    (i.e. WITHOUT the in-grid restriction that ``C_true_bKs`` carries), so the
    only thing being tested is the z-shape, not the phi convention.  Reported
    truth-weighted over b for the two lowest live strata.
    """
    E = effective_completeness(Cadd, g_bk)                   # (B,Kf,S)
    nda = np.asarray(ops["N_det_all_bks_true_z"], float)
    out = {}
    for s in li[:2]:
        blk = {}
        for K in range(int(kz.max()) + 1):
            m = (kz == K)
            w = tc[:, m, s]
            if w.sum() <= 0:
                continue
            cg = float((E[:, m, s] * w).sum() / w.sum())
            c_det_true_b = np.where(w.sum(1) > 0,
                                    nda[:, m, s].sum(1) / np.maximum(w.sum(1), 1e-12), 0.0)
            ct = float((c_det_true_b * w.sum(1)).sum() / w.sum())
            blk[f"K{K}"] = dict(C_times_g=cg, C_det_true=ct, ratio=cg / max(ct, 1e-12))
        out[f"s={s}"] = blk
    return out


# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default=FIGDIR)
    a = ap.parse_args(argv)
    os.makedirs(a.figdir, exist_ok=True)
    D = load_all()
    G = grid_from(D["cal"][CAL_FAM])
    numbers = dict(
        role="PI-facing C1 completeness visual review (PI ruling 2026-09-13c §6); "
             "VALIDATION-ONLY, calibration/mock information only, no sampler, no real data",
        calibration_family=CAL_FAM,
        form="logit C = poly_3(x) + poly_2(u), x = N - 20, u = log10(SNR_med) - u0",
        coefficients=D["prov"]["coefficients"],
        u0=D["prov"]["u_pivot_log10_snr"], x_pivot=D["prov"]["x_pivot"],
        snr_logmed_s=D["prov"]["snr_logmed_s"], live_strata=D["prov"]["live_strata"],
        status="CANDIDATE, NOT ADOPTED")
    fig_C1(D, G, a.figdir)
    fig_C1b(D, G, a.figdir)
    fig_C2(D, G, a.figdir, numbers)
    fig_C3(D, G, a.figdir, numbers)
    fig_C4(D, G, a.figdir, numbers)
    fig_C5(D, G, a.figdir, numbers)
    # the run-reported S/N marginals, for the record
    runs = {}
    for tag, p in [("A0", f"{ROOT}/runs/A0/RUN_A0_%s_s20260811.json"),
                   ("A0+C1nsadd", f"{ROOT}/runs/A0+C1nsadd/RUN_A0+C1nsadd_%s_s20260811.json"),
                   ("OP  [fix P]", f"{DIAG}/OP/RUN_OP_%s_s20260811.json"),
                   ("OC  [fix P,C]", f"{DIAG}/OC/RUN_OC_%s_s20260811.json"),
                   ("OCz [fix P,Cz]", f"{DIAG}/OCz/RUN_OCz_%s_s20260811.json"),
                   ("OM  [fix P,M]", f"{DIAG}/OM/RUN_OM_%s_s20260811.json"),
                   ("OCM [fix P,C,M]", f"{DIAG}/OCM/RUN_OCM_%s_s20260811.json")]:
        for f in FAMS:
            try:
                j = json.load(open(p % f))
            except OSError:
                continue
            runs.setdefault(tag, {})[f] = [
                float(v) for v in
                j["diagnostics"]["predictive_marginals"]["mu_over_obs_by_snr"]]
    numbers["run_reported_mu_over_obs_by_snr"] = runs
    with open(os.path.join(a.figdir, "completeness_review_numbers.json"), "w") as fh:
        json.dump(numbers, fh, indent=1)
    print("wrote figures + numbers to", a.figdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
