"""Plot 6 random spectra from the new LoaArchive
(/scratch/.../loa_archives/loa_full_z2_noR_v2.h5).

Quick visual sanity check: spectra have continuum + emission lines,
no obvious truncation/zero-fill artifacts. Each panel shows obs flux
+ rest-wavelength markers for Lyα/CIV/SiIV/Lyβ + reports z + B/R SNR.

Usage:
    python examples/plot_loa_archive_samples.py \\
        --archive /scratch/.../loa_archives/loa_full_z2_noR_v2.h5 \\
        --n 6 --out figs/archive_sample.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, help="Path to loa_full_*.h5")
    p.add_argument("--n", type=int, default=6, help="Number of random spectra to plot")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="Output PNG")
    args = p.parse_args()

    from gpy_dla_detection.loa_archive import LoaArchive

    print(f"[plot] loading {args.archive}", flush=True)
    ar = LoaArchive(args.archive)
    ar.open()
    print(f"  n_qsos: {ar.n_qsos}  n_pix: {ar.wavelength.shape[0]}  has_R: {ar.has_resolution}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    tids_all = np.array(list(ar._tid_to_idx.keys()))
    picks = rng.choice(tids_all, size=args.n, replace=False)
    print(f"  random TIDs: {picks.tolist()}", flush=True)

    # Mark canonical lines (rest wavelength)
    LINES = {"Lyα": 1215.67, "Lyβ": 1025.72, "OVI": 1031.93, "NV": 1240.0,
             "SiIV": 1394.0, "CIV": 1548.2, "CIII]": 1908.7}

    fig, axes = plt.subplots(args.n, 1, figsize=(12, 1.8 * args.n), sharex=False,
                             gridspec_kw=dict(hspace=0.30))
    if args.n == 1:
        axes = [axes]

    wave_obs = ar.wavelength

    for ax, tid in zip(axes, picks):
        spec = ar.get_spectrum(int(tid))
        flux = spec.flux
        z = spec.z
        # Plot observed flux
        ax.plot(wave_obs, flux, color="0.3", lw=0.4)
        ax.axhline(0, color="0.7", lw=0.4, ls="--")
        # Mark each line in observed frame
        for lname, lwave in LINES.items():
            obs_lwave = lwave * (1.0 + z)
            if wave_obs[0] < obs_lwave < wave_obs[-1]:
                ax.axvline(obs_lwave, color="C2", ls=":", lw=0.5, alpha=0.6)
                ax.text(obs_lwave, 0.92, lname, transform=ax.get_xaxis_transform(),
                        fontsize=7, ha="center", color="C2",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="C2", alpha=0.7))
        ax.set_xlim(wave_obs[0], wave_obs[-1])
        ax.set_ylabel(
            f"flux\nTID {tid}\nz={z:.3f}\nBSNR={spec.blue_snr:.1f}\nRSNR={spec.red_snr:.1f}",
            fontsize=7,
        )
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("observed wavelength [Å]")
    fig.suptitle(
        f"LoaArchive sample: {args.n} random spectra from {Path(args.archive).name}",
        fontsize=10, y=0.998,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[plot] wrote {out}  ({out.stat().st_size/1e6:.2f} MB)", flush=True)
    ar.close()


if __name__ == "__main__":
    main()
