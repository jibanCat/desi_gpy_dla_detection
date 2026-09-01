#!/usr/bin/env python3
"""cross_run_pairs.py — the cross-run comparison the PI must be able to read (kickoff §24): the key scientific pairs of every run
of the matrix on IDENTICAL axes (R0's), one colour slot per run in fixed order, contours (68 %) per run plus the R0 cloud.
PRIVATE outputs.

    python tools/hbi_validation/cross_run_pairs.py --pack PACK --run R0=POOLED_R0.json --run R1=... --out-dir DIR
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from CDDF_analysis.hbi_mcmc.pack import load_pack                             # noqa: E402
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors  # noqa: E402
from tools.hbi_validation.sci_corner import sci_coords                        # noqa: E402
from tools.hbi_validation.atlas import load_arms                              # noqa: E402
from tools.hbi_validation.viz_common import plt, SLOTS, INK2                  # noqa: E402

PAIRS = [("log Λ", "t0"), ("t0", "C̄ subDLA"), ("t0", "f subDLA 19.7–20.3"), ("t0", "dN/dX ≥20.0"), ("t0", "dN/dX ≥20.3"),
         ("f subDLA 19.7–20.3", "dN/dX ≥20.0"), ("dN/dX ≥20.0", "dN/dX ≥20.3"), ("dN/dX ≥20.3", "10³ Ω[20.3,21.6]"), ("C̄ subDLA", "f subDLA 19.7–20.3")]


def contour68(ax, x, y, color, bins=40, lims=None):
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=lims)
    Hn = H.T / H.sum(); srt = np.sort(Hn.ravel())[::-1]; cs = np.cumsum(srt)
    lev = srt[np.searchsorted(cs, 0.68)]
    if lev > 0:
        ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), Hn, levels=[lev], colors=[color], linewidths=1.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True); ap.add_argument("--run", action="append", required=True, help="LABEL=POOLED.json (first = R0 baseline)")
    ap.add_argument("--geometry-r0", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    pk = load_pack(a.pack); consts, _ = build_cc_tensors(pk)
    lev = [e["cell"] for e in json.load(open(a.geometry_r0))["leverage_psi_c"]["z0"][:3]]
    runs = {}
    for spec in a.run:
        lab, p = spec.split("=", 1); _, chain, arm, tags, paths = load_arms(p, None)
        Y, names, _ = sci_coords(pk, consts, [np.load(q) for q in paths], lev); runs[lab] = Y
    labs = list(runs); base = runs[labs[0]]; ni = {n: i for i, n in enumerate(names)}
    lims = {}
    for n, i in ni.items():
        allv = np.concatenate([runs[l][:, i] for l in labs]); lo, hi = np.percentile(allv, [0.2, 99.8]); pad = 0.06 * (hi - lo) if hi > lo else 1e-3
        lims[n] = (lo - pad, hi + pad)
    fig, axs = plt.subplots(3, 3, figsize=(12, 10.5))
    for ax, (xn, yn) in zip(axs.ravel(), PAIRS):
        i, j = ni[xn], ni[yn]
        ax.scatter(base[:, i], base[:, j], s=1.5, alpha=0.12, color="#9aa0a6", linewidths=0, rasterized=True)
        for k, l in enumerate(labs):
            Y = runs[l]
            if Y[:, i].std() > 0 and Y[:, j].std() > 0:
                contour68(ax, Y[:, i], Y[:, j], SLOTS[k % len(SLOTS)], lims=[lims[xn], lims[yn]])
            else:   # a fixed coordinate: draw the run's marginal in the other coordinate as a rug at the fixed value
                fx = float(Y[0, i]) if Y[:, i].std() == 0 else None; fy = float(Y[0, j]) if Y[:, j].std() == 0 else None
                if fx is not None and Y[:, j].std() > 0:
                    q = np.percentile(Y[:, j], [16, 84]); ax.plot([fx, fx], q, color=SLOTS[k % len(SLOTS)], lw=2.5, solid_capstyle="butt")
                elif fy is not None and Y[:, i].std() > 0:
                    q = np.percentile(Y[:, i], [16, 84]); ax.plot(q, [fy, fy], color=SLOTS[k % len(SLOTS)], lw=2.5, solid_capstyle="butt")
            ax.plot(np.median(Y[:, i]), np.median(Y[:, j]), marker="o", ms=4, color=SLOTS[k % len(SLOTS)], mec=INK2, mew=0.4, ls="none", label=l if ax is axs[0, 0] else None)
        ax.set_xlim(lims[xn]); ax.set_ylim(lims[yn]); ax.set_xlabel(xn, fontsize=7); ax.set_ylabel(yn, fontsize=7); ax.tick_params(labelsize=6)
    axs[0, 0].legend(fontsize=6, loc="best")
    fig.suptitle("Cross-run comparison on identical axes — grey: R0 draws; contours: 68 % per run; dots: medians; bars: 16–84 % at a fixed coordinate", fontsize=8, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "cross_run_pairs.png"), dpi=140); plt.close(fig)
    json.dump({l: {n: [float(np.percentile(runs[l][:, i], q)) for q in (16, 50, 84)] for n, i in ni.items()} for l in labs}, open(os.path.join(a.out_dir, "cross_run_medians.json"), "w"), indent=1)
    print("written", os.path.join(a.out_dir, "cross_run_pairs.png"), "runs", labs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
