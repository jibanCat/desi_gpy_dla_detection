# -*- coding: utf-8 -*-
"""Figure: differential f(N) calc_cddf vs HBI vs truth, + ratio panel (est/truth), per mock.
UNTRACKED. MOCK-ONLY. Writes PNGs to the private notes repo (not the code repo)."""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = "/home/mfho/desi_gpy_dla_notes/notes/figures/2026-07-10_calccddf"

# HBI kernel-based per-0.1dex differential in the band [19.5,20.3), 2LPT-0 loa0
# (CDDF_analysis/hbi/subdla_mock_validation.json  per_bin.loa0). dndx per bin.
HBI_BAND_2LPT0 = {  # blo -> (dndx_tru, dndx_est_loa0)
    19.5: (0.0131187, 0.00596205), 19.6: (0.0132386, 0.00948884),
    19.7: (0.0130268, 0.0116299), 19.8: (0.012823, 0.0119525),
    19.9: (0.0124474, 0.0112131), 20.0: (0.0115663, 0.0104513),
    20.1: (0.0112706, 0.0104033), 20.2: (0.0104634, 0.0107529),
}
# HBI forward-path cumulative R0 (crossmock_transfer_loa0.json self baseline / legs)
HBI_CUM = {
    "2lpt0": dict(dndx={"20.3": 1.0381, "20.0": 1.0041, "19.5": 0.9189, "band": 0.8490},
                  omega={"20.3": 0.9593, "20.0": 0.9547, "19.5": 0.9377, "band": 0.8220}),
    "london0": dict(dndx={"20.3": 1.0347, "20.0": 1.0190, "19.5": 0.8829, "band": 0.7892},
                    omega={"20.3": 0.9464, "20.0": 0.9469, "19.5": 0.9270, "band": 0.8168}),
    "saclay0": dict(dndx={"20.3": 1.0452, "20.0": 1.0165, "19.5": 0.8898, "band": 0.7977},
                    omega={"20.3": 0.9420, "20.0": 0.9423, "19.5": 0.9228, "band": 0.8183}),
}


def plot_mock(mock, jpath, ax_top, ax_bot):
    d = json.load(open(jpath))
    N = np.array(d["N_centers"])
    fN_calc = np.array(d["fN_calccddf"])
    fN_tru = np.array(d["fN_truth"])
    edges = np.round(np.arange(17.2, 22.40001, 0.1), 3)
    dN = 10.0 ** edges[1:] - 10.0 ** edges[:-1]

    Nlin = 10.0 ** N
    good = fN_tru > 0
    # top: f(N) log-log
    ax_top.plot(Nlin[good], fN_tru[good], "k-", lw=1.6, label="truth (injected)")
    gc = fN_calc > 0
    ax_top.plot(Nlin[gc], fN_calc[gc], "o-", color="#2166ac", ms=3, lw=1.0, label="calc_cddf (literal Bird-2017)")
    # HBI band differential (2lpt0 only)
    if mock == "2lpt0":
        xb, yb = [], []
        for blo, (dt, de) in HBI_BAND_2LPT0.items():
            c = blo + 0.05
            i = np.argmin(np.abs(N - c))
            xb.append(10.0 ** c)
            yb.append(de / dN[i])
        ax_top.plot(xb, yb, "s", color="#b2182b", ms=5, label="HBI (kernel, band)")
    ax_top.axvspan(10 ** 19.5, 10 ** 20.3, color="orange", alpha=0.08)
    ax_top.axvline(10 ** 20.3, color="grey", ls=":", lw=0.8)
    ax_top.set_yscale("log"); ax_top.set_xscale("log")
    ax_top.set_ylabel(r"$f(N_{\rm HI})$  [cm$^2$]")
    ax_top.set_title(f"{mock}  (MOCK; z$\\in$[2,3.5], SNR>2, Ly$\\alpha$-only)", fontsize=10)
    ax_top.legend(fontsize=7.5, loc="upper right")
    ax_top.set_xlim(10 ** 18.8, 10 ** 22.4)
    ymid = np.median(fN_tru[good])
    ax_top.set_ylim(ymid * 1e-4, fN_tru[good].max() * 5)

    # bottom: ratio est/truth
    r_calc = np.where(good, fN_calc / np.where(good, fN_tru, 1), np.nan)
    ax_bot.plot(Nlin[good], r_calc[good], "o-", color="#2166ac", ms=3, lw=1.0, label="calc_cddf / truth")
    if mock == "2lpt0":
        xb, rb = [], []
        for blo, (dt, de) in HBI_BAND_2LPT0.items():
            xb.append(10.0 ** (blo + 0.05)); rb.append(de / dt)
        ax_bot.plot(xb, rb, "s-", color="#b2182b", ms=4, lw=0.8, label="HBI / truth (band)")
    ax_bot.axhline(1.0, color="k", lw=0.8, ls="--")
    ax_bot.axvspan(10 ** 19.5, 10 ** 20.3, color="orange", alpha=0.08)
    ax_bot.axvline(10 ** 20.3, color="grey", ls=":", lw=0.8)
    ax_bot.set_xscale("log")
    ax_bot.set_ylim(0, 1.6)
    ax_bot.set_xlim(10 ** 18.8, 10 ** 22.4)
    ax_bot.set_ylabel("est / truth")
    ax_bot.set_xlabel(r"$N_{\rm HI}$  [cm$^{-2}$]")
    ax_bot.legend(fontsize=7.5, loc="lower right")

    cum = d["cumulative"]
    txt = (f"cumulative R0 (calc/truth):\n"
           f"  >=20.3: {cum['R0_calccddf']['dndx']['20.3']:.3f}  (HBI {HBI_CUM[mock]['dndx']['20.3']:.3f})\n"
           f"  >=20.0: {cum['R0_calccddf']['dndx']['20.0']:.3f}  (HBI {HBI_CUM[mock]['dndx']['20.0']:.3f})\n"
           f"  band[19.5,20.3): {cum['R0_calccddf']['dndx']['band_195_203']:.3f}  (HBI {HBI_CUM[mock]['dndx']['band']:.3f})")
    ax_bot.text(0.02, 0.97, txt, transform=ax_bot.transAxes, fontsize=6.8, va="top",
                family="monospace", bbox=dict(fc="white", ec="grey", alpha=0.8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", nargs="+", required=True, help="mock=jsonpath ...")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    for spec in args.jsons:
        mock, jp = spec.split("=", 1)
        if not os.path.exists(jp):
            print("skip (missing):", jp); continue
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.4, 6.4), height_ratios=[2.2, 1],
                                     sharex=True, gridspec_kw=dict(hspace=0.06))
        plot_mock(mock, jp, a0, a1)
        out = os.path.join(FIGDIR, f"calccddf_vs_hbi_{mock}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
