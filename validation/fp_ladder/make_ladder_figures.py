"""Key figures for the first absorber-side ladder return packet (PI 2026-09-13b §18 item 6).

VALIDATION-ONLY plots of MOCK runs. Reads RUN_<variant>_<fam>_s<seed>.json (+ _fdraws.npz) and draws:
  fig1  headline biases (>=20.0, >=20.3) per variant and family with 68 % posterior half-widths and the frozen gate bands
  fig2  0.2-dex reporting-bin median bias vs N (the zigzag) for the four anchor variants
  fig3  latent-bin recovered/truth ratio (z-integrated with dX weights) for the same variants -- shows the top-bin blow-up under R1c
  fig4  predictive mu/obs marginal by S/N stratum for the same variants (the S/N 2-3 residual)
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAMS = ("2lpt0", "london0", "saclay0")
ANCHORS = ("A0", "A0+C1nsadd", "A0+R1c", "A0+R1c+C1nsadd")
_ANCHORS_DEFAULT = ANCHORS


def style():
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False})


def load_run(runs, variant, fam, seed):
    p = os.path.join(runs, variant, f"RUN_{variant}_{fam}_s{seed}.json")
    if not os.path.exists(p):
        cands = sorted(glob.glob(os.path.join(runs, variant, f"RUN_{variant}_{fam}_s{seed}*.json")))
        if not cands:
            return None, None
        p = cands[0]
    j = json.load(open(p))
    fd = p.replace(".json", "_fdraws.npz")
    return j, (np.load(fd) if os.path.exists(fd) else None)


def fig1_headline(runs, variants, seed, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    x = np.arange(len(variants)); w = 0.26
    for ax, key, band in zip(axes, ("ge20.0", "ge20.3"), ((-0.5, 0.5), (-0.5, 3.0))):
        ax.axhspan(band[0], band[1], color="0.85", zorder=0, label="frozen gate (all-z)")
        for i, fam in enumerate(FAMS):
            vals, errs = [], []
            for v in variants:
                j, _ = load_run(runs, v, fam, seed)
                if j is None:
                    vals.append(np.nan); errs.append(0); continue
                t = j["thresholds"][key]; tr = t["truth"]; p16, p50, p84 = t["post_p16_50_84"]
                vals.append(100 * (p50 / tr - 1)); errs.append(100 * 0.5 * (p84 - p16) / tr)
            ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=2, label=fam)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(variants, rotation=35, ha="right")
        ax.set_ylabel(f"dN/dX({key.replace('ge', '≥')}) median bias [%]")
        ax.set_title(f"{key.replace('ge', 'N ≥ ')} (seed {seed}; bars = 68 % half-width)", fontsize=9)
    axes[0].legend(ncol=4, fontsize=7.5)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig2_zigzag(runs, seed, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, fam in zip(axes, FAMS):
        for v in ANCHORS:
            j, _ = load_run(runs, v, fam, seed)
            if j is None:
                continue
            rb = [r for r in j["reporting_bins"] if r.get("median_bias_pct") is not None]
            c = [0.5 * (r["bin"][0] + r["bin"][1]) for r in rb]
            ax.plot(c, [r["median_bias_pct"] for r in rb], marker="o", ms=3, lw=1, label=v)
        ax.axhline(0, color="k", lw=0.8); ax.set_title(fam); ax.set_xlabel("log N_HI (0.2-dex reporting bin centre)")
    axes[0].set_ylabel("per-bin median bias [%]"); axes[0].legend(fontsize=7.5)
    fig.suptitle("Reporting-bin recovery (0.2-dex bins)", y=1.02, fontsize=9)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig3_latent(runs, seed, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, fam in zip(axes, FAMS):
        for v in ANCHORS:
            j, fd = load_run(runs, v, fam, seed)
            if fd is None:
                continue
            f = fd["f"]; tf = fd["truth_f"]; dX = fd["dX_k"]; e = fd["ntrue_edges"]
            num = (f * dX[None, None, :]).sum(-1); den = (tf * dX[None, :]).sum(-1)
            r = num / den[None, :]
            c = 0.5 * (e[:-1] + e[1:])
            ax.errorbar(c, np.median(r, 0), yerr=[np.median(r, 0) - np.percentile(r, 16, 0), np.percentile(r, 84, 0) - np.median(r, 0)],
                        marker="o", ms=3, lw=1, capsize=2, label=v)
        ax.axhline(1, color="k", lw=0.8); ax.set_yscale("log"); ax.set_title(fam); ax.set_xlabel("latent log N_HI bin centre")
    axes[0].set_ylabel("recovered / truth (z-integrated, dX-weighted)"); axes[0].legend(fontsize=7.5)
    fig.suptitle("Latent 0.2-dex bins: recovered / truth", y=1.02, fontsize=9)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig4_snr(runs, seed, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, fam in zip(axes, FAMS):
        for v in ANCHORS:
            j, _ = load_run(runs, v, fam, seed)
            if j is None:
                continue
            r = np.asarray(j["diagnostics"]["predictive_marginals"]["mu_over_obs_by_snr"], float)
            idx = np.flatnonzero(r > 0)
            ax.plot(idx, r[idx], marker="o", ms=3, lw=1, label=v)
        ax.axhline(1, color="k", lw=0.8); ax.set_title(fam); ax.set_xlabel("S/N stratum index (live strata)")
    axes[0].set_ylabel("posterior-median μ / observed counts"); axes[0].legend(fontsize=7.5)
    fig.suptitle("Predictive S/N marginal (posterior-median μ / observed)", y=1.02, fontsize=9)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--anchors", nargs="+", default=None, help="variants for figs 2-4 (default: first-ladder anchors)")
    ap.add_argument("--prefix", default="ladder")
    ap.add_argument("--variants", nargs="+", default=["A0", "A0-P6bcal", "A0+R1a", "A0+R1b", "A0+R1c", "A0+C1g", "A0+C1n", "A0+C1ns", "A0+C1nsadd", "A0+R1c+C1nsadd"])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); style()
    global ANCHORS
    if a.anchors: ANCHORS = tuple(a.anchors)
    fig1_headline(a.runs, a.variants, a.seed, os.path.join(a.out, f"{a.prefix}_fig1_headline_bias_by_variant.png"))
    fig2_zigzag(a.runs, a.seed, os.path.join(a.out, f"{a.prefix}_fig2_reporting_bin_zigzag.png"))
    fig3_latent(a.runs, a.seed, os.path.join(a.out, f"{a.prefix}_fig3_latent_ratio.png"))
    fig4_snr(a.runs, a.seed, os.path.join(a.out, f"{a.prefix}_fig4_snr_marginal.png"))
    print("wrote 4 figures to", a.out)


if __name__ == "__main__":
    main()
