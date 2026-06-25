#!/usr/bin/env python3
"""make_compare_figs.py -- regenerate the two catalog-HBI mock-validation figures
shown in ``CDDF_analysis/hbi/README.md``:

  * ``fig_compare_integrated.png`` -- integrated dN/dX and 10^3*Omega_DLA at the
    **logN_HI >= 20.3 DLA headline**, for the raw feed-forward, both HBI FP
    variants (purity_mixture, loa0), vs the injected mock truth, with R0
    (= method/truth) annotated.
  * ``fig_compare_fN.png`` -- the differential f(N_HI): the raw feed-forward tail
    is too flat (drives the Omega over-statement) and HBI's kernel deconvolution
    re-steepens it back onto the injected truth; the one-sided HBI MC band is
    shown for the loa0 variant.

SELF-CONTAINED & REPRODUCIBLE: reads ONLY the committed sibling table
``compare_mock_data.csv`` (small MOCK-injection numbers, truth known -- no
real-survey values). No private-repo or scratch-cache dependency at run time.

These are MOCK numbers (2LPT-0 injection validation) and are fine to commit
publicly.  Matplotlib Agg (headless).

Run:
  HDF5_USE_FILE_LOCKING=FALSE conda run -n gpdla python \
      CDDF_analysis/hbi/figures/make_compare_figs.py
"""
from __future__ import annotations
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "compare_mock_data.csv")

# Headline DLA threshold. The catalog in compare_mock_data.csv is the >=20.3
# integrated block; the f(N) differential block spans the full reported range.
THRESHOLD = "20.3"


def load_csv(path=CSV):
    """Parse the multi-section CSV into {section_name: list[dict]}."""
    sections: dict[str, list[dict]] = {}
    cur_name = None
    cur_header = None
    cur_rows: list[dict] = []
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("section,"):
                if cur_name is not None:
                    sections[cur_name] = cur_rows
                cur_name = line.split(",", 1)[1].strip()
                cur_header = None
                cur_rows = []
                continue
            if cur_header is None:
                cur_header = next(csv.reader([line]))
                continue
            vals = next(csv.reader([line]))
            cur_rows.append(dict(zip(cur_header, vals)))
        if cur_name is not None:
            sections[cur_name] = cur_rows
    return sections


# ---------------------------------------------------------------------------
# Figure 1: integrated dN/dX & Omega at >= 20.3
# ---------------------------------------------------------------------------
def fig_integrated(sections, out=None):
    out = out or os.path.join(HERE, "fig_compare_integrated.png")
    rows = sections["integrated"]
    # index by (method, quantity)
    tab = {(r["method"], r["quantity"]): r for r in rows}

    methods = [("raw_feedforward", "Raw feed-forward", "#2980b9"),
               ("HBI_purity_mixture", "HBI (purity_mixture)", "#e67e22"),
               ("HBI_loa0", "HBI (loa0 FP)", "#c0392b")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    panels = [
        (axes[0], "dndx", r"$dN/dX$ $(\geq 20.3)$",
         r"$dN/dX$", 1.0),
        (axes[1], "omega", r"$10^{3}\,\Omega_{\rm DLA}$ $(\geq 20.3)$",
         r"$10^{3}\,\Omega_{\rm DLA}$", 1e3),
    ]
    for ax, q, ylab, title, scale in panels:
        t = float(tab[(methods[0][0], q)]["truth"]) * scale
        ax.axhline(t, color="k", lw=2, ls="-", label="injected truth", zorder=1)
        for i, (key, lab, col) in enumerate(methods):
            r = tab[(key, q)]
            val = float(r["value"]) * scale
            if r["err_kind"] == "std":
                err = float(r["err_lo"]) * scale
            else:  # ci68 -> asymmetric [lo, hi]
                lo = float(r["err_lo"]) * scale
                hi = float(r["err_hi"]) * scale
                err = np.array([[val - lo], [hi - val]])
            ax.errorbar([i], [val], yerr=err, fmt="o", color=col, ms=9,
                        capsize=4, lw=2, label=lab, zorder=3)
            ax.annotate(f"R0={float(r['R0']):.3f}", (i, val),
                        textcoords="offset points", xytext=(8, 6),
                        fontsize=8, color=col)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m[1].replace(" ", "\n") for m in methods], fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xlim(-0.5, len(methods) - 0.5)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Integrated DLA recovery on mock injection "
                 r"($\log N_{\rm HI}\geq 20.3$; $z\in[2.0,3.5]$); "
                 "MC std (HBI) / PB-68 (raw)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: differential f(N_HI)
# ---------------------------------------------------------------------------
def fig_fN(sections, out=None):
    out = out or os.path.join(HERE, "fig_compare_fN.png")
    fN = sections["fN"]
    logN = np.array([float(r["logN_mid"]) for r in fN])
    f_truth = np.array([float(r["f_truth"]) for r in fN])
    f_loa0 = np.array([float(r["f_hbi_loa0"]) for r in fN])
    f_pm = np.array([float(r["f_hbi_pm"]) for r in fN])
    band_lo = np.array([float(r["f_loa0_band_lo"]) for r in fN])
    band_hi = np.array([float(r["f_loa0_band_hi"]) for r in fN])

    rff = sections["fN_rawff"]
    rN = np.array([float(r["logN_mid"]) for r in rff])
    rf = np.array([float(r["f_rawff"]) for r in rff])
    rlo = np.array([float(r["f_rawff_lo"]) for r in rff])
    rhi = np.array([float(r["f_rawff_hi"]) for r in rff])

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.plot(logN, f_truth, "k-", lw=2.4, label="injected truth", zorder=5)
    ax.fill_between(logN, band_lo, band_hi, color="#c0392b", alpha=0.18,
                    zorder=2, label="HBI (loa0) MC 68%")
    ax.plot(logN, f_loa0, "o-", color="#c0392b", lw=1.8, ms=4,
            label="HBI (loa0 FP)", zorder=4)
    ax.plot(logN, f_pm, "s--", color="#e67e22", lw=1.4, ms=3,
            label="HBI (purity_mixture)", zorder=3)
    m = rf > 0
    ax.plot(rN[m], rf[m], "^-", color="#2980b9", lw=1.6, ms=4,
            label="Raw feed-forward (calc_cddf)", zorder=2)
    ax.fill_between(rN[m], rlo[m], rhi[m], color="#2980b9", alpha=0.15, zorder=1)
    for x in (20.0, 20.3):
        ax.axvline(x, color="0.6", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlim(19.5, 22.0)
    ax.set_xlabel(r"$\log_{10} N_{\rm HI}\ [{\rm cm^{-2}}]$")
    ax.set_ylabel(r"$f(N_{\rm HI})\ [{\rm cm^{2}}]$")
    ax.set_title("Mock-injection DLA CDDF: HBI vs raw feed-forward vs injected truth")
    ax.legend(frameon=False, fontsize=9)
    ax.text(20.32, ax.get_ylim()[1] * 0.4, "DLA edge 20.3", fontsize=7,
            color="0.4", rotation=90, va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    sections = load_csv()
    p_int = fig_integrated(sections)
    p_fN = fig_fN(sections)
    # echo the headline R0 numbers so the run is self-checking against the README
    tab = {(r["method"], r["quantity"]): r for r in sections["integrated"]}
    print("wrote:", p_int)
    print("wrote:", p_fN)
    print(f"integrated >= {THRESHOLD} R0:")
    for key in ("raw_feedforward", "HBI_purity_mixture", "HBI_loa0"):
        rd = tab[(key, "dndx")]
        ro = tab[(key, "omega")]
        print(f"  {key:20s} dN/dX R0={float(rd['R0']):.3f}  "
              f"Omega R0={float(ro['R0']):.3f}")


if __name__ == "__main__":
    main()
