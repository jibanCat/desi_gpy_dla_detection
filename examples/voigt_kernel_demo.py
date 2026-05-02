"""Generate a demonstration figure: what the LSF kernel does to a
forward-modelled DLA profile in the three NHI regimes.

The figure shows for log NHI ∈ {17.5 (LLS), 19.5 (sub-DLA), 21.0 (DLA),
21.5 (strong DLA)}:
  - bare Voigt profile (no LSF)
  - convolved with `boss-log-r2000` (production)
  - convolved with `desi-linear-r3000` (proposed fix)

Side-by-side plots show how the wrong LSF over- or under-broadens
the modelled trough relative to truth, which is what biases N_HI in
the inference.

Run::

    python examples/voigt_kernel_demo.py --out docs/notes/voigt_kernel_demo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpy_dla_detection.voigt_v2 import voigt_absorption


_LYA_AA = 1215.6701


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--z-dla", type=float, default=2.5)
    p.add_argument("--dlambda-A", type=float, default=0.15)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = args.z_dla
    centre_obs = (1 + z) * _LYA_AA   # ~4254.85 Å for z=2.5
    # Wide-enough window to see DLA wings.
    wave = np.arange(centre_obs - 100, centre_obs + 100, args.dlambda_A)

    nhi_regimes = [
        (17.5, "LLS",        "C2"),
        (19.5, "sub-DLA",    "C1"),
        (21.0, "DLA",        "C0"),
        (21.5, "strong DLA", "C3"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.ravel()

    for ax, (log_nhi, regime, colour) in zip(axes, nhi_regimes):
        for kernel, ls, label in [
            ("none",            "-",  "bare Voigt (no LSF)"),
            ("boss-log-r2000",  ":",  "BOSS-log-R2000 (production)"),
            ("desi-linear-r3000","--", "DESI-linear-R3000 (proposed)"),
        ]:
            profile = voigt_absorption(
                wave, log_nhi=log_nhi, z_dla=z, num_lines=3,
                kernel=kernel, dlambda_A=args.dlambda_A,
            )
            # Trim wave to match output length when convolution drops edges.
            half = (len(wave) - len(profile)) // 2
            wave_plot = wave[half:half + len(profile)]
            ax.plot(wave_plot, profile, ls=ls, color=colour, alpha=0.85,
                    label=label, linewidth=1.5)
        ax.axvline(centre_obs, color="grey", ls=":", alpha=0.5)
        ax.set_title(f"{regime}: log NHI = {log_nhi}")
        ax.set_ylabel("flux fraction (Voigt absorption)")
        ax.set_ylim(-0.02, 1.1)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)
    axes[2].set_xlabel(r"$\lambda_{\rm obs}$ (Å)")
    axes[3].set_xlabel(r"$\lambda_{\rm obs}$ (Å)")
    fig.suptitle(
        f"DLA forward model under three LSF kernels   "
        f"(z_DLA = {z:.2f}, dλ = {args.dlambda_A} Å)",
        fontsize=13,
    )
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"[demo] wrote {args.out}")


if __name__ == "__main__":
    main()
