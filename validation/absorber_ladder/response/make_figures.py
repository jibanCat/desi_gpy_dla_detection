#!/usr/bin/env python
"""make_figures.py — figures for the response-variant ladder.

  respvar_fig1_rows_<fam>.png   kernel rows: adopted (R0) vs variants vs the
                                MEASURED row, for the alternating reporting
                                bins, per S/N stratum and coarse K
  respvar_fig2_cv_residuals.png held-out (2-fold sightline CV) width / mean /
                                skew / up-leakage residuals vs true-N bin
  respvar_fig3_width_cause.png  the width-defect decomposition: what the
                                midpoint quadrature drops

VALIDATION-ONLY.  ENV: gpdla.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import respfit as R                                             # noqa: E402
import opmetrics as M                                           # noqa: E402

ORDER = ["R0", "R1a", "R1b", "R1c", "R1d"]
COL = {"R0": "#440154", "R1a": "#31688e", "R1b": "#35b779",
       "R1c": "#d95f02", "R1d": "#999999"}
ALT_BINS = [(19.7, 19.9), (20.1, 20.3), (21.1, 21.3), (21.3, 21.5),
            (21.5, 21.7)]


def _load_masses(out_dir, name, fam, pk):
    z = np.load(os.path.join(out_dir, f"Mg_{name}_{fam}.npz"),
                allow_pickle=True)
    mg = np.asarray(z["Mg"], float)             # (S, Kf, C, B), count-conserving
    return mg / np.maximum(mg.sum(axis=2, keepdims=True), 1e-300)


def fig_rows(out_dir, fig_dir, fam, pk, variants):
    ops = os.path.join(_REPO, "validation", "absorber_diag",
                       f"empirical_ops_{fam}.npz")
    Mt = np.asarray(np.load(ops, allow_pickle=True)["M_true_sKcb"], float)
    kz = np.asarray(pk["kz_to_K"], int)
    ne = np.asarray(pk["nhat_edges"], float)
    nt = np.asarray(pk["ntrue_edges"], float)
    cen = 0.5 * (ne[:-1] + ne[1:])
    rows = {n: _load_masses(out_dir, n, fam, pk) for n in variants}
    strata = [(1, "S/N 2-3"), (7, r"S/N $\geq$ 7")]
    fig, ax = plt.subplots(len(ALT_BINS), len(strata) * 3,
                           figsize=(19, 2.5 * len(ALT_BINS)), sharex=True)
    for ib, (lo, hi) in enumerate(ALT_BINS):
        b = int(np.argmin(np.abs(0.5 * (nt[:-1] + nt[1:])
                                 - 0.5 * (lo + hi))))
        col = 0
        for s, slab in strata:
            for K in range(3):
                a = ax[ib, col]
                m = Mt[s, K, :, b]
                if m.sum() > 0:
                    a.step(cen, m / m.sum(), where="mid", color="k", lw=2.0,
                           label="measured")
                kf = int(np.where(kz == K)[0][0])
                for n in variants:
                    a.step(cen, rows[n][s, kf, :, b], where="mid",
                           color=COL[n], lw=1.2, ls="--" if n == "R0" else "-",
                           label=n)
                a.axvspan(lo, hi, color="0.85", zorder=0)
                a.set_yscale("log"); a.set_ylim(1e-4, 1.0)
                a.set_xlim(19.4, 22.4)
                if ib == 0:
                    a.set_title(f"{slab}, K{K}", fontsize=9)
                if col == 0:
                    a.set_ylabel(f"b=[{lo},{hi})\nrow mass", fontsize=8)
                if ib == len(ALT_BINS) - 1:
                    a.set_xlabel(r"$\log N_{\rm HI}$ (observed)", fontsize=8)
                a.tick_params(labelsize=7)
                col += 1
    ax[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Response rows: adopted vs variants vs measured — {fam}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(fig_dir, f"respvar_fig1_rows_{fam}.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_cv(out_dir, fig_dir, pk, variants):
    nt = np.asarray(pk["ntrue_edges"], float)
    Nc = 0.5 * (nt[:-1] + nt[1:])
    keys = [("r_sd", "width residual  $\\sigma_{\\rm mod}/\\sigma_{\\rm emp}-1$"),
            ("d_mean", "mean residual (dex)"),
            ("d_skew", "row-skew residual"),
            ("d_up", "up-leakage residual")]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    for i, (k, lab) in enumerate(keys):
        a = ax[i // 2, i % 2]
        for n in variants:
            z = np.load(os.path.join(out_dir, f"cv_records_{n}.npz"))
            b = z["b"].astype(int); v = z[k]; w = z["n"]
            xs, ys = [], []
            for bb in np.unique(b):
                m = (b == bb) & np.isfinite(v)
                if m.sum() == 0:
                    continue
                xs.append(Nc[bb])
                ys.append(np.sum(w[m] * v[m]) / np.sum(w[m]))
            a.plot(xs, ys, "o-", color=COL[n], ms=4, lw=1.3, label=n)
        a.axhline(0.0, color="k", lw=0.8)
        a.axvline(19.7, color="0.6", lw=0.8, ls=":")
        a.axvline(21.3, color="0.6", lw=0.8, ls=":")
        a.set_xlabel(r"true $\log N_{\rm HI}$ bin centre")
        a.set_ylabel(lab, fontsize=9)
        a.tick_params(labelsize=8)
    ax[0, 0].legend(fontsize=8, ncol=3)
    fig.suptitle("Held-out (2-fold TARGETID-parity CV) calibration-side "
                 "residuals, count-weighted per true-N bin", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(fig_dir, "respvar_fig2_cv_residuals.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def fig_cause(events, fig_dir, pk, adopted):
    """The diagnosed CAUSE of the 8-25% width narrowing."""
    ev = R.load_events(events)
    isr, izr = R.cell_index(ev["snr"], ev["zqso"])
    nt = np.asarray(pk["ntrue_edges"], float)
    Nc = 0.5 * (nt[:-1] + nt[1:])
    snre = np.asarray(pk["snr_edges"], float)
    zc = np.asarray(pk["zc_edges"], float)
    rse = np.asarray(pk["resp_snr_edges"], float)
    rze = np.asarray(pk["resp_z_edges"], float)
    s2sr = np.clip(np.searchsorted(rse, snre[:-1] + 1e-9, "right") - 1, 0, 2)
    K2zr = np.searchsorted(rze, 0.5 * (zc[:-1] + zc[1:]), "right") - 1
    b_i = np.clip(np.digitize(ev["N_true"], nt) - 1, 0, len(Nc) - 1)
    s_i = np.clip(np.digitize(ev["snr"], snre) - 1, 0, len(snre) - 2)
    K_i = np.clip(np.digitize(ev["zdla"], zc) - 1, 0, len(zc) - 2)
    ad = np.load(adopted, allow_pickle=True)
    surf = {"mu": ad["mu_coef"], "sig": ad["sig_coef"],
            "skew": ad["skew_coef"]}
    rngA = ad["fit_rng"]; N_ref = float(ad["N_ref"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for ia, (s_sel, slab) in enumerate([(2, "S/N 2-3"), (7, r"S/N $\geq$ 7")]):
        xs, y_mod, y_mix, y_full, y_emp = [], [], [], [], []
        for b in range(len(Nc)):
            m = (b_i == b) & (s_i == s_sel) & (K_i == 1)
            if m.sum() < 40:
                continue
            _, sg0, _ = R.eval_moments(surf, rngA, N_ref,
                                       np.array([Nc[b]]),
                                       np.array([s2sr[s_sel]]),
                                       np.array([K2zr[1]]),
                                       skew_ramp=(21.0, 0.5))
            mu_i, sg_i, _ = R.eval_moments(surf, rngA, N_ref,
                                           ev["N_true"][m], isr[m], izr[m],
                                           skew_ramp=(21.0, 0.5))
            xs.append(Nc[b])
            y_mod.append(float(sg0[0]))
            y_mix.append(float(np.sqrt(np.mean(sg_i ** 2))))
            y_full.append(float(np.sqrt(np.mean(sg_i ** 2)
                                        + np.var(ev["N_true"][m] + mu_i))))
            y_emp.append(float(np.std(ev["xhat"][m], ddof=1)))
        a = ax[ia]
        a.plot(xs, y_emp, "ko-", ms=5, label="measured row sd")
        a.plot(xs, y_mod, "s--", color="#440154", ms=4,
               label=r"model: $\sigma$ at the bin CENTRE (R0)")
        a.plot(xs, y_mix, "^-", color="#31688e", ms=4,
               label=r"$\langle\sigma^2\rangle^{1/2}$ over the block's covariates")
        a.plot(xs, y_full, "v-", color="#d95f02", ms=4,
               label=r"$(\langle\sigma^2\rangle+{\rm Var}[N+\mu])^{1/2}$")
        a.set_title(f"{slab}, K1", fontsize=10)
        a.set_xlabel(r"true $\log N_{\rm HI}$ bin centre")
        a.set_ylabel("row width (dex)")
        a.tick_params(labelsize=8)
        if ia == 0:
            a.legend(fontsize=7.5)
    fig.suptitle("Cause of the 8-25% kernel narrowing: the midpoint quadrature "
                 "drops the within-block covariate spread", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(fig_dir, "respvar_fig3_width_cause.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--families", default="2lpt0,london0,saclay0")
    ap.add_argument("--adopted", default=(
        "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/"
        "stage0/adopted_response_v1p1.npz"))
    a = ap.parse_args(argv)
    os.makedirs(a.fig_dir, exist_ok=True)
    rep = json.load(open(os.path.join(a.out_dir, "variants_report.json")))
    variants = [n for n in ORDER if n in rep["variants"]]
    made = []
    for fam in a.families.split(","):
        pk = np.load("/scratch/cavestru_root/cavestru0/mfho/"
                     f"fp_ladder_2026-09-12/packs/scanpack_{fam}_b300.npz",
                     allow_pickle=True)
        made.append(fig_rows(a.out_dir, a.fig_dir, fam, pk, variants))
    pk = np.load("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/"
                 "packs/scanpack_2lpt0_b300.npz", allow_pickle=True)
    made.append(fig_cv(a.out_dir, a.fig_dir, pk, variants))
    made.append(fig_cause(a.events, a.fig_dir, pk, a.adopted))
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
