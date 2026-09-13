#!/usr/bin/env python
"""ORACLE absorber-recovery diagnostic plots (read-only on existing runs).

Reads the FP-ladder ORACLE diagnostic runs (mu_FP pinned to the mock FP truth)
plus the wave-1 (M0/M1/M2) and wave-2 (M3) runs, and emits the PI-requested
figure set + a markdown table of every number plotted.

NOT COMMITTED.  Outputs go to the private notes repo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12")
DIRS = {
    "ORACLE": ROOT / "diag_oracle",
    "M0": ROOT / "wave1",
    "M1": ROOT / "wave1",
    "M2": ROOT / "wave1",
    "M3": ROOT / "wave2_M3",
}
PACKS = ROOT / "packs"
OUT = Path("/home/mfho/desi_gpy_dla_notes/figures/2026-09-13_absorber_diag")
OUT.mkdir(parents=True, exist_ok=True)

FAMS = ["2lpt0", "london0", "saclay0"]
SEEDS = [20260811, 20260812]
COMPARE = ["M0", "M2", "M3"]

# gate windows (sealed)
GATE_200 = (-0.5, 0.5)     # |bias| <= 0.5 % at >=20.0 all-z
GATE_203 = (-0.5, 3.0)     # [-0.5, +3.0] % at >=20.3

MISSING: list[str] = []

# ---------------------------------------------------------------- style
sys.path.insert(0, "/home/mfho/Latex/gp_dla_desi_y3/paper_figures")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

USETEX = True
try:
    import common  # type: ignore

    common.style()
    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.xlabel(r"$\log N_{\rm HI}$")
    fig.savefig("/tmp/_usetex_probe.png")
    plt.close(fig)
    CYCLE = list(common.CYCLE)
    STYLE_NOTE = "paper_figures/common.py style() (usetex serif, LaTeX math)"
except Exception as exc:  # pragma: no cover
    USETEX = False
    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm",
                         "font.family": "serif", "figure.dpi": 160,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    CYCLE = ["#31688e", "#35b779", "#20a386", "#e16462", "#440154"]
    STYLE_NOTE = f"FALLBACK mathtext (usetex failed: {type(exc).__name__}: {exc})"
    print("STYLE FALLBACK:", STYLE_NOTE)

C_ORACLE = "#440154"
C_MODEL = {"M0": "#31688e", "M2": "#35b779", "M3": "#e16462"}
SEED_LS = {20260811: "-", 20260812: "--"}
SEED_MK = {20260811: "o", 20260812: "s"}


def nlab(s: str) -> str:
    return s


# ---------------------------------------------------------------- loading
def run_json(model: str, fam: str, seed: int):
    tag = "ORACLE" if model == "ORACLE" else model
    p = DIRS[model] / f"RUN_{tag}_{fam}_s{seed}.json"
    if not p.is_file():
        MISSING.append(str(p))
        return None
    return json.loads(p.read_text())


def run_fdraws(model: str, fam: str, seed: int):
    tag = "ORACLE" if model == "ORACLE" else model
    p = DIRS[model] / f"RUN_{tag}_{fam}_s{seed}_fdraws.npz"
    if not p.is_file():
        MISSING.append(str(p))
        return None
    return np.load(p)


def run_ppc(model: str, fam: str, seed: int):
    tag = "ORACLE" if model == "ORACLE" else model
    p = DIRS[model] / f"RUN_{tag}_{fam}_s{seed}_ppc.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


PACK = {f: np.load(PACKS / f"scanpack_{f}_b300.npz") for f in FAMS}
KZ = {f: PACK[f]["kz_to_K"] for f in FAMS}
NHAT_EDGES = {f: PACK[f]["nhat_edges"] for f in FAMS}
SNR_EDGES = {f: PACK[f]["snr_edges"] for f in FAMS}

TABLE: dict[str, list] = {}


# ---------------------------------------------------------------- helpers
def pathweighted(f, dX, sel=None):
    """f (D,B,Kf) or (B,Kf) -> path-weighted over the selected z cells."""
    w = np.array(dX, float).copy()
    if sel is not None:
        w = np.where(sel, w, 0.0)
    w = w / w.sum()
    return np.tensordot(f, w, axes=([-1], [0]))


def qs(a, axis=0):
    return np.percentile(a, [2.5, 16, 50, 84, 97.5], axis=axis)


# ================================================================ FIG 1
def fig1():
    fg, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), sharex=True,
                            gridspec_kw={"height_ratios": [2.0, 1.35]})
    rows = []
    for j, fam in enumerate(FAMS):
        ax, axr = axes[0, j], axes[1, j]
        for seed in SEEDS:
            z = run_fdraws("ORACLE", fam, seed)
            if z is None:
                continue
            f, tf, dX = z["f"], z["truth_f"], z["dX_k"]
            e = z["ntrue_edges"]
            ctr = 0.5 * (e[:-1] + e[1:])
            fpw = pathweighted(f, dX)          # (D,B)
            tpw = pathweighted(tf, dX)         # (B,)
            q = qs(fpw)
            r = (fpw / tpw[None, :] - 1.0) * 100.0
            qr = qs(r)
            dx = 0.0 if seed == SEEDS[0] else 0.022
            if seed == SEEDS[0]:
                ax.plot(ctr, tpw, color="k", lw=1.6, zorder=5,
                        label="truth $f_{\\rm true}(N)$" if USETEX else "truth f_true(N)")
                ax.plot(ctr, tpw, "k.", ms=3.5, zorder=6)
            ax.fill_between(ctr + dx, q[0], q[4], color=C_ORACLE, alpha=0.14, lw=0)
            ax.fill_between(ctr + dx, q[1], q[3], color=C_ORACLE, alpha=0.32, lw=0)
            ax.plot(ctr + dx, q[2], color=C_ORACLE, ls=SEED_LS[seed], lw=1.2,
                    marker=SEED_MK[seed], ms=2.6, label=f"ORACLE s{seed}")
            axr.fill_between(ctr + dx, qr[0], qr[4], color=C_ORACLE, alpha=0.14, lw=0)
            axr.fill_between(ctr + dx, qr[1], qr[3], color=C_ORACLE, alpha=0.32, lw=0)
            axr.plot(ctr + dx, qr[2], color=C_ORACLE, ls=SEED_LS[seed], lw=1.2,
                     marker=SEED_MK[seed], ms=2.6)
            for b in range(len(ctr)):
                rows.append(dict(family=fam, seed=seed, b=b,
                                 nlo=float(e[b]), nhi=float(e[b + 1]),
                                 truth=float(tpw[b]), med=float(q[2, b]),
                                 p16=float(q[1, b]), p84=float(q[3, b]),
                                 p2p5=float(q[0, b]), p97p5=float(q[4, b]),
                                 ratio_med=float(qr[2, b]),
                                 ratio_p16=float(qr[1, b]), ratio_p84=float(qr[3, b])))
        ax.set_yscale("log")
        ax.set_title(f"{fam} (ORACLE)")
        ax.axvline(20.3, color="0.5", lw=0.7, ls=":")
        axr.axvline(20.3, color="0.5", lw=0.7, ls=":")
        axr.axhline(0, color="k", lw=0.8)
        axr.axhspan(GATE_203[0], GATE_203[1], color="#35b779", alpha=0.13, lw=0, zorder=0)
        axr.set_ylim(-45, 45)
        axr.set_xlabel(r"true $\log_{10} N_{\rm HI}$" if USETEX else "true log10 N_HI")
        if j == 0:
            ax.set_ylabel(r"$f(N)$, path-weighted all $z$ [dex$^{-1}$ per unit $X$]"
                          if USETEX else "f(N), path-weighted all z")
            axr.set_ylabel(r"recovered/truth $-\,1$ [\%]" if USETEX
                           else "recovered/truth - 1 [%]")
            ax.legend(frameon=False, loc="lower left")
        ax.tick_params(labelbottom=False)
    hnd = [Patch(facecolor="#35b779", alpha=0.13,
                 label=r"$[-0.5,+3.0]\,\%$ gate window ($\geq 20.3$ all-$z$)"
                 if USETEX else "[-0.5,+3.0] % gate window")]
    axes[1, 2].legend(handles=hnd, frameon=False, loc="lower left", fontsize=6.5)
    fg.suptitle("Fig. 1 — ORACLE: truth vs recovered absorber density by TRUE $N$ bin "
                "(path-weighted over all $z$)" if USETEX else
                "Fig. 1 - ORACLE: truth vs recovered f(N) by true N bin", y=0.98)
    fg.tight_layout(rect=(0, 0, 1, 0.96))
    fg.savefig(OUT / "fig1_truth_vs_recovered_byN.png")
    plt.close(fg)
    TABLE["fig1"] = rows


# ================================================================ FIG 2
def fig2():
    fg, axes = plt.subplots(3, 3, figsize=(11.0, 8.2), sharex=True, sharey=True)
    rows = []
    zblk = [(0, "K0  $2.0\\le z<2.5$"), (1, "K1  $2.5\\le z<3.0$"), (2, "K2  $3.0\\le z<3.5$")]
    for j, fam in enumerate(FAMS):
        kz = KZ[fam]
        for i, (K, lab) in enumerate(zblk):
            ax = axes[i, j]
            sel = kz == K
            for seed in SEEDS:
                z = run_fdraws("ORACLE", fam, seed)
                if z is None:
                    continue
                f, tf, dX = z["f"], z["truth_f"], z["dX_k"]
                e = z["ntrue_edges"]
                ctr = 0.5 * (e[:-1] + e[1:])
                fpw = pathweighted(f, dX, sel)
                tpw = pathweighted(tf, dX, sel)
                r = (fpw / tpw[None, :] - 1.0) * 100.0
                qr = qs(r)
                dx = 0.0 if seed == SEEDS[0] else 0.022
                ax.fill_between(ctr + dx, qr[1], qr[3], color=C_ORACLE, alpha=0.28, lw=0)
                ax.plot(ctr + dx, qr[2], color=C_ORACLE, ls=SEED_LS[seed], lw=1.2,
                        marker=SEED_MK[seed], ms=2.8, label=f"s{seed}")
                for b in range(len(ctr)):
                    rows.append(dict(family=fam, seed=seed, block=f"K{K}", b=b,
                                     nlo=float(e[b]), nhi=float(e[b + 1]),
                                     truth=float(tpw[b]), ratio_med=float(qr[2, b]),
                                     ratio_p16=float(qr[1, b]), ratio_p84=float(qr[3, b])))
            ax.axhline(0, color="k", lw=0.8)
            ax.axhspan(GATE_203[0], GATE_203[1], color="#35b779", alpha=0.13, lw=0, zorder=0)
            ax.axvline(20.3, color="0.5", lw=0.7, ls=":")
            ax.set_ylim(-60, 60)
            if i == 0:
                ax.set_title(f"{fam} (ORACLE)")
            if j == 0:
                ax.set_ylabel((lab if USETEX else lab.replace("$", "").replace("\\le", "<="))
                              + "\n" + (r"rec/truth $-1$ [\%]" if USETEX else "rec/truth -1 [%]"))
            if i == 2:
                ax.set_xlabel(r"true $\log_{10} N_{\rm HI}$" if USETEX else "true log10 N_HI")
            if i == 0 and j == 0:
                ax.legend(frameon=False, loc="upper left", fontsize=7)
    fg.suptitle("Fig. 2 — ORACLE: recovered/truth ratio vs true $N$, within each coarse $z$ block "
                "(68\\,\\% bands)" if USETEX else
                "Fig. 2 - ORACLE: ratio vs true N within each coarse z block (68% bands)", y=0.985)
    fg.tight_layout(rect=(0, 0, 1, 0.965))
    fg.savefig(OUT / "fig2_ratio_vs_N_by_block.png")
    plt.close(fg)
    TABLE["fig2"] = rows


# ================================================================ FIG 3
def fig3():
    fg, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharey=True)
    rows = []
    for j, fam in enumerate(FAMS):
        ax = axes[j]
        for model in COMPARE:
            for seed in SEEDS:
                d = run_json(model, fam, seed)
                if d is None:
                    continue
                rb = d["reporting_bins"]
                ctr = [0.5 * (b["bin"][0] + b["bin"][1]) for b in rb]
                y = [b["median_bias_pct"] for b in rb]
                ax.plot(ctr, y, color=C_MODEL[model], ls=SEED_LS[seed], lw=0.9,
                        alpha=0.45, zorder=2,
                        label=(f"{model}" if seed == SEEDS[0] else None))
                for b, c in zip(rb, ctr):
                    rows.append(dict(family=fam, model=model, seed=seed,
                                     bin_lo=b["bin"][0], bin_hi=b["bin"][1],
                                     median_bias_pct=b["median_bias_pct"],
                                     truth_in_68=b["truth_in_68"],
                                     truth_in_95=b["truth_in_95"]))
        for seed in SEEDS:
            d = run_json("ORACLE", fam, seed)
            if d is None:
                continue
            rb = d["reporting_bins"]
            ctr = np.array([0.5 * (b["bin"][0] + b["bin"][1]) for b in rb])
            y = np.array([b["median_bias_pct"] for b in rb])
            in68 = np.array([b["truth_in_68"] for b in rb])
            in95 = np.array([b["truth_in_95"] for b in rb])
            ax.plot(ctr, y, color=C_ORACLE, ls=SEED_LS[seed], lw=1.8, zorder=6,
                    label=f"ORACLE s{seed}")
            ax.scatter(ctr[in68], y[in68], s=30, facecolor=C_ORACLE, edgecolor=C_ORACLE,
                       zorder=7, marker=SEED_MK[seed])
            ax.scatter(ctr[~in68], y[~in68], s=32, facecolor="white", edgecolor=C_ORACLE,
                       linewidths=1.1, zorder=7, marker=SEED_MK[seed])
            bad95 = ~in95
            if bad95.any():
                ax.scatter(ctr[bad95], y[bad95], s=120, facecolor="none", edgecolor="#e16462",
                           linewidths=1.3, zorder=8, marker="o")
            for b, c in zip(rb, ctr):
                rows.append(dict(family=fam, model="ORACLE", seed=seed,
                                 bin_lo=b["bin"][0], bin_hi=b["bin"][1],
                                 median_bias_pct=b["median_bias_pct"],
                                 truth_in_68=b["truth_in_68"],
                                 truth_in_95=b["truth_in_95"]))
        ax.axhline(0, color="k", lw=0.8)
        ax.axhspan(GATE_203[0], GATE_203[1], color="#35b779", alpha=0.13, lw=0, zorder=0)
        ax.axvline(20.3, color="0.5", lw=0.7, ls=":")
        ax.set_title(fam)
        ax.set_xlabel(r"0.2-dex reporting bin centre, $\log_{10} N_{\rm HI}$"
                      if USETEX else "0.2-dex reporting bin centre")
        if j == 0:
            ax.set_ylabel(r"median bias [\%]" if USETEX else "median bias [%]")
    hnd = [Line2D([], [], color=C_ORACLE, lw=1.8, label="ORACLE (s20260811 solid, s20260812 dashed)"),
           Line2D([], [], color=C_MODEL["M0"], lw=0.9, alpha=0.5, label="M0"),
           Line2D([], [], color=C_MODEL["M2"], lw=0.9, alpha=0.5, label="M2"),
           Line2D([], [], color=C_MODEL["M3"], lw=0.9, alpha=0.5, label="M3"),
           Line2D([], [], color=C_ORACLE, marker="o", ls="none", label="truth in 68\\%" if USETEX else "truth in 68%"),
           Line2D([], [], color=C_ORACLE, marker="o", mfc="white", ls="none",
                  label="truth OUTSIDE 68\\%" if USETEX else "truth OUTSIDE 68%"),
           Line2D([], [], color="#e16462", marker="o", mfc="none", ms=9, ls="none",
                  label="truth OUTSIDE 95\\%" if USETEX else "truth OUTSIDE 95%"),
           Patch(facecolor="#35b779", alpha=0.13, label=r"$[-0.5,+3.0]\,\%$ window" if USETEX else "[-0.5,+3.0] % window")]
    fg.legend(handles=hnd, frameon=False, fontsize=7.0, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, -0.02))
    fg.suptitle("Fig. 3 — 0.2-dex reporting-bin zigzag: ORACLE (bold) vs M0/M2/M3 (faint)"
                if USETEX else "Fig. 3 - 0.2-dex reporting-bin zigzag", y=0.99)
    fg.tight_layout(rect=(0, 0.09, 1, 0.94))
    fg.savefig(OUT / "fig3_reporting_bin_zigzag.png")
    plt.close(fg)
    TABLE["fig3"] = rows


# ================================================================ FIG 4
def fig4():
    rows = []
    have = {}
    for fam in FAMS:
        for seed in SEEDS:
            p = run_ppc("ORACLE", fam, seed)
            if p is not None:
                have[(fam, seed)] = p
    ppc_seeds = sorted({s for (_, s) in have})
    fg, axes = plt.subplots(3, 3, figsize=(11.5, 8.4))
    spec = [("marginal_by_snr", "S/N stratum", 0),
            ("marginal_by_nhat", r"observed $\hat N$ bin" if USETEX else "observed Nhat bin", 1),
            ("marginal_by_z", r"$z$ cell" if USETEX else "z cell", 2)]
    for j, fam in enumerate(FAMS):
        for key, xlab, i in spec:
            ax = axes[i, j]
            for seed in ppc_seeds:
                p = have.get((fam, seed))
                if p is None:
                    continue
                m = p["ppc_block"][key]
                idx = np.array([r["i"] for r in m])
                obs = np.array([r["obs"] for r in m], float)
                mu = np.array([r["mu_median"] for r in m], float)
                pts = np.array([r["p_two_sided"] for r in m], float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    res = np.where(obs > 0, (mu / obs - 1.0) * 100.0, np.nan)
                if key == "marginal_by_snr":
                    ed = SNR_EDGES[fam]
                    x = 0.5 * (ed[:-1] + ed[1:])
                    x = np.where(np.isfinite(x), x, ed[:-1] + 0.5)
                elif key == "marginal_by_nhat":
                    ed = NHAT_EDGES[fam]
                    x = 0.5 * (ed[:-1] + ed[1:])
                else:
                    x = 2.0 + 0.05 + 0.1 * idx
                bad = pts <= p["ppc_block"]["pval_threshold_two_sided"]
                ax.plot(x, res, color=C_ORACLE, ls=SEED_LS[seed], lw=1.1,
                        marker=SEED_MK[seed], ms=3.2, label=f"ORACLE s{seed}")
                if bad.any():
                    ax.scatter(x[bad], res[bad], s=80, facecolor="none",
                               edgecolor="#e16462", linewidths=1.3, zorder=7)
                for k in range(len(idx)):
                    rows.append(dict(family=fam, seed=seed, marginal=key, i=int(idx[k]),
                                     x=float(x[k]), obs=float(obs[k]), mu_median=float(mu[k]),
                                     resid_pct=(None if not np.isfinite(res[k]) else float(res[k])),
                                     p_two_sided=float(pts[k])))
            ax.axhline(0, color="k", lw=0.8)
            if key == "marginal_by_snr":
                ax.axvspan(2, 3, color="#e16462", alpha=0.15, lw=0, zorder=0)
                ax.set_xlim(-0.2, 8.4)
            ax.set_xlabel(xlab)
            if j == 0:
                ax.set_ylabel(r"$\mu_{\rm median}/{\rm obs}-1$ [\%]" if USETEX
                              else "mu_median/obs - 1 [%]")
            if i == 0:
                ax.set_title(f"{fam} (ORACLE PPC)")
            if i == 0 and j == 0:
                hnd = [Line2D([], [], color=C_ORACLE, lw=1.1, marker="o", label=f"ORACLE s{ppc_seeds[0]}"),
                       Patch(facecolor="#e16462", alpha=0.15, label="S/N 2–3 stratum"),
                       Line2D([], [], color="#e16462", marker="o", mfc="none", ms=8, ls="none",
                              label="PPC cell fail")]
                ax.legend(handles=hnd, frameon=False, fontsize=6.5, loc="lower right")
    note = ("PPC files exist for seed %s only" % ", ".join(str(s) for s in ppc_seeds))
    fg.suptitle("Fig. 4 — ORACLE posterior-predictive residuals by S/N, observed $\\hat N$ and $z$ "
                "(%s)" % note if USETEX else "Fig. 4 - ORACLE PPC residuals (%s)" % note, y=0.985)
    fg.tight_layout(rect=(0, 0, 1, 0.965))
    fg.savefig(OUT / "fig4_ppc_residuals.png")
    plt.close(fg)
    TABLE["fig4"] = rows
    TABLE["fig4_seeds"] = ppc_seeds
    # snr_ramp summary
    ramp = []
    for (fam, seed), p in sorted(have.items()):
        r = p.get("snr_ramp")
        if r:
            ramp.append(dict(family=fam, seed=seed, strata=r["strata"], ratios=r["ratios"],
                             amplitude=r["amplitude"], flag_threshold=r["flag_threshold"],
                             flag=r["flag"]))
    TABLE["snr_ramp"] = ramp


# ================================================================ FIG 5
BINKEYS = [("paper1_bins", ["B1", "B2", "B3", "B4", "B5"]),
           ("coarse_blocks", ["block0", "block1", "block2"])]


def fig5():
    rows = []
    models = ["ORACLE"] + COMPARE
    fg, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharey="row")
    labels = ["B1", "B2", "B3", "B4", "B5", "K0", "K1", "K2"]
    for j, fam in enumerate(FAMS):
        for i, est in enumerate(["ge20.0", "ge20.3"]):
            ax = axes[i, j]
            nm = len(models)
            width = 0.8 / nm
            for mi, model in enumerate(models):
                vals = {s: [] for s in SEEDS}
                for seed in SEEDS:
                    d = run_json(model, fam, seed)
                    if d is None:
                        vals[seed] = [np.nan] * 8
                        continue
                    e = d["perz_recovery"]["estimand"][est]
                    v = []
                    for key, names in BINKEYS:
                        lut = {r["bin"]: r for r in e[key]}
                        for n in names:
                            r = lut.get(n)
                            v.append(np.nan if r is None else r["median_bias_pct"])
                            if r is not None:
                                rows.append(dict(family=fam, model=model, seed=seed,
                                                 estimand=est, bin=n,
                                                 z_lo=r["z"][0], z_hi=r["z"][1],
                                                 truth=r["truth"],
                                                 median_bias_pct=r["median_bias_pct"],
                                                 truth_in_68=r["truth_in_68"],
                                                 truth_in_95=r["truth_in_95"]))
                    vals[seed] = v
                x = np.arange(8) + (mi - (nm - 1) / 2.0) * width
                col = C_ORACLE if model == "ORACLE" else C_MODEL[model]
                ax.bar(x, vals[SEEDS[0]], width=width * 0.92, color=col,
                       alpha=0.92 if model == "ORACLE" else 0.55,
                       edgecolor="k" if model == "ORACLE" else "none", linewidth=0.4,
                       label=model if (i == 0 and j == 0) else None, zorder=3)
                ax.plot(x, vals[SEEDS[1]], ls="none", marker="_", ms=5.5, mew=1.1,
                        color="k", zorder=5,
                        label=("2nd seed (s%d)" % SEEDS[1]) if (i == 0 and j == 0 and mi == 0) else None)
            lo, hi = GATE_200 if est == "ge20.0" else GATE_203
            ax.axhspan(lo, hi, color="#35b779", alpha=0.16, lw=0, zorder=0)
            ax.axhline(0, color="k", lw=0.8, zorder=2)
            ax.set_xticks(np.arange(8))
            ax.set_xticklabels(labels)
            ax.axvline(4.5, color="0.6", lw=0.8, ls=":")
            if j == 0:
                ax.set_ylabel((r"$\geq 20.0$" if est == "ge20.0" else r"$\geq 20.3$")
                              + "\n" + (r"d$N$/d$X$ median bias [\%]" if USETEX
                                        else "dN/dX median bias [%]"))
            if i == 0:
                ax.set_title(fam)
            if i == 1:
                ax.set_xlabel("Paper-1 reporting bin  |  coarse $z$ block"
                              if USETEX else "Paper-1 bin | coarse z block")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
            gp = Patch(facecolor="#35b779", alpha=0.16,
                       label=(r"$|b|\leq 0.5\,\%$" if est == "ge20.0" else r"$[-0.5,+3.0]\,\%$")
                       + " gate window" if USETEX else "gate window")
            if j == 2:
                ax.legend(handles=[gp], frameon=False, fontsize=6.2, loc="upper right")
    fg.suptitle("Fig. 5 — d$N$/d$X$ recovery bias per Paper-1 bin and coarse $z$ block: "
                "ORACLE vs M0/M2/M3 (bars = s%d, ticks = s%d)" % (SEEDS[0], SEEDS[1])
                if USETEX else "Fig. 5 - dN/dX bias per bin: ORACLE vs M0/M2/M3", y=0.985)
    fg.tight_layout(rect=(0, 0, 1, 0.955))
    fg.savefig(OUT / "fig5_dndx_bias_bars.png")
    plt.close(fg)
    TABLE["fig5"] = rows


# ================================================================ all-z table
def allz_table():
    rows = []
    for fam in FAMS:
        for model in ["ORACLE"] + COMPARE:
            for seed in SEEDS:
                d = run_json(model, fam, seed)
                if d is None:
                    continue
                for est in ["ge20.0", "ge20.3"]:
                    t = d["thresholds"][est]
                    rows.append(dict(family=fam, model=model, seed=seed, estimand=est,
                                     truth=t["truth"],
                                     median=t["post_p16_50_84"][1],
                                     p16=t["post_p16_50_84"][0], p84=t["post_p16_50_84"][2],
                                     median_bias_pct=t["median_bias_pct"],
                                     truth_in_68=t["truth_in_68"],
                                     truth_in_95=t["truth_in_95"]))
    TABLE["allz"] = rows


def main():
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    fns = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5}
    for k in which:
        fns[k]()
    if set(which) == {"1", "2", "3", "4", "5"}:
        allz_table()
        (OUT / "_plotted_numbers.json").write_text(json.dumps(
            {"style": STYLE_NOTE, "usetex": USETEX, "missing": MISSING, **TABLE},
            indent=1, default=str))
    print("USETEX:", USETEX, "|", STYLE_NOTE)
    print("MISSING:", MISSING)
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
