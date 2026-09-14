"""Shared helpers for the PI-facing calibration inspection figures (SEC 5/6/9).

READ-ONLY on every frozen object: nothing here fits, refits or retunes.  All
numbers are read from released products or are deterministic evaluations of
released coefficients / tensors.
"""
from __future__ import annotations

import hashlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REL = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/release"
LAD = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13"
FIGDIR = "/home/mfho/desi_gpy_dla_notes/figures/2026-09-14_pi_inspection"

PALETTE = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725",
           "#b5367a", "#f26d5b"]


def style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.family": "serif",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.1,
        "axes.linewidth": 0.6,
        "savefig.bbox": "tight",
    })


def sha8(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def wilson(k, n, z=1.0):
    """Wilson interval half-widths (lo, hi) for a binomial fraction."""
    import numpy as np
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1.0 + z * z / np.maximum(n, 1)
    centre = (p + z * z / (2 * np.maximum(n, 1))) / den
    half = (z / den) * np.sqrt(p * (1 - p) / np.maximum(n, 1)
                               + z * z / (4 * np.maximum(n, 1) ** 2))
    lo = np.clip(p - (centre - half), 0, None)
    hi = np.clip((centre + half) - p, 0, None)
    return lo, hi


def manifest(paths, out):
    lines = []
    for p in paths:
        lines.append("%s  %s" % (sha8(p) if os.path.exists(p) else "MISSING", p))
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines
