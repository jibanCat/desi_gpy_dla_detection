"""Side-by-side spectrum plots for the 3 TIDs from
2026-05-03_archive_vs_fits_dla_comparison.md.

For each TID, two panels:
  - LEFT:  spectrum loaded via raw FITS pipeline (load_one_desi_spectrum)
           with MAP DLA Voigt overlaid if found, annotated with FITS result
  - RIGHT: same TID loaded via LoaArchive (LoaArchive.get_spectrum)
           with MAP DLA Voigt overlaid if found, annotated with archive result

If the two pipelines agree (which 2026-05-03 confirmed they do at
4-5 sig figs), the panels should look identical.

Reads the comparison results from the .md table (so we don't re-run
the expensive process_qso).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ARCHIVE = "/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5"
LOA_ROOT = "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/"

# Hardcoded results from the 2026-05-03 comparison run (so we don't re-run
# the ~5 min process_qso × 3 TIDs × 2 paths). Source:
# docs/notes/2026-05-03_archive_vs_fits_dla_comparison.md
RESULTS = [
    dict(tid=39633010785519257, z_qso=2.004,
         fits=dict(p_dla=0.0000, map_z=np.nan, map_log_nhi=np.nan),
         archive=dict(p_dla=0.0000, map_z=np.nan, map_log_nhi=np.nan),
         dp=5.97e-15),
    dict(tid=39633067924522971, z_qso=2.548,
         fits=dict(p_dla=0.9333, map_z=2.4689, map_log_nhi=20.149),
         archive=dict(p_dla=0.9333, map_z=2.4689, map_log_nhi=20.149),
         dp=2.57e-06),
    dict(tid=39628512230899761, z_qso=2.821,
         fits=dict(p_dla=0.0000, map_z=np.nan, map_log_nhi=np.nan),
         archive=dict(p_dla=0.0000, map_z=np.nan, map_log_nhi=np.nan),
         dp=8.75e-13),
]


def voigt_profile(obs_wave, z_dla, log_nhi, num_lines=3):
    """Voigt absorption profile (Lyα + higher Lyman lines), no broadening
    so output length matches input."""
    from gpy_dla_detection.voigt import voigt_absorption
    return voigt_absorption(obs_wave, 10.0 ** log_nhi, z_dla,
                            num_lines=num_lines, broadening=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="docs/notes/2026-05-03_archive_vs_fits_panels.png")
    args = p.parse_args()

    from examples.smoke_one_spectrum import load_one_desi_spectrum
    from gpy_dla_detection.loa_archive import LoaArchive

    ar = LoaArchive(ARCHIVE)
    ar.open()
    with h5py.File(ARCHIVE, "r") as f:
        cat = f["catalog"][:]
    cat_by_tid = {int(r["TARGETID"]): r for r in cat}

    n = len(RESULTS)
    fig, axes = plt.subplots(n, 2, figsize=(15, 3.0 * n), sharex=False,
                             gridspec_kw=dict(hspace=0.30, wspace=0.05))
    if n == 1:
        axes = axes[None, :]

    for row, r in enumerate(RESULTS):
        tid = int(r["tid"])
        z_qso = r["z_qso"]
        sf = cat_by_tid[tid]["SOURCE_FILE"].decode()
        fits_path = os.path.join(LOA_ROOT, sf)

        # FITS pipeline
        try:
            wave_f, flux_f, _, _ = load_one_desi_spectrum(fits_path, tid)
            fits_loaded = True
        except Exception as e:
            print(f"  TID {tid}: FITS load failed: {e}")
            wave_f = flux_f = None
            fits_loaded = False

        # Archive pipeline
        spec_a = ar.get_spectrum(tid)
        wave_a = ar.wavelength
        flux_a = spec_a.flux

        # Restrict x-axis to Lyα forest region (rest 850-1250 → obs)
        xmin_obs = 3500
        xmax_obs = 1250 * (1.0 + z_qso) + 200

        for col, (label, wave, flux, res) in enumerate([
            ("FITS pipeline", wave_f, flux_f, r["fits"]),
            ("LoaArchive", wave_a, flux_a, r["archive"]),
        ]):
            ax = axes[row, col]
            if wave is not None and flux is not None:
                ax.plot(wave, flux, color="0.3", lw=0.4, alpha=0.85)
                # smoothed median for visual reference
                from scipy.ndimage import median_filter
                ax.plot(wave, median_filter(flux, size=51), color="C0", lw=0.7,
                        label="51-pix median")
            ax.axhline(0, color="0.7", lw=0.4, ls="--")

            # Mark Lyα at z_qso
            ax.axvline(1215.67 * (1 + z_qso), color="C2", ls=":", lw=0.7,
                       alpha=0.7)
            ax.text(1215.67 * (1 + z_qso), 0.92, "Lyα(QSO)",
                    transform=ax.get_xaxis_transform(),
                    fontsize=7, ha="center", color="C2",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="C2", alpha=0.7))

            # MAP DLA Voigt overlay
            if not np.isnan(res["map_log_nhi"]):
                # Estimate continuum scale at red side of the spectrum for amplitude reference
                cont_med = np.nanmedian(flux[(wave > 5500) & (wave < 6500)]) if wave is not None else 1.0
                if not np.isfinite(cont_med) or cont_med <= 0:
                    cont_med = 1.0
                voigt = voigt_profile(wave_a, res["map_z"], res["map_log_nhi"])
                ax.plot(wave_a, cont_med * voigt, color="C3", lw=0.8, alpha=0.85,
                        label=f"MAP DLA z={res['map_z']:.4f} logNHI={res['map_log_nhi']:.2f}")
                ax.axvline(1215.67 * (1 + res["map_z"]), color="C3",
                           ls="--", lw=0.7, alpha=0.7)

            ax.set_xlim(xmin_obs, xmax_obs)
            ax.grid(alpha=0.3)
            verdict = (f"DLA detected: p={res['p_dla']:.4f}"
                       if not np.isnan(res["map_log_nhi"])
                       else f"no DLA: p_dla={res['p_dla']:.4f}")
            ax.set_title(f"{label}  →  {verdict}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"flux\nTID {tid}\nz_qso={z_qso:.3f}", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
        # Annotate Δp_dla in the row title space
        axes[row, 1].text(1.02, 0.5,
                          f"Δp_dla\n= {r['dp']:.2e}",
                          transform=axes[row, 1].transAxes,
                          fontsize=8, ha="left", va="center",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    fc="lightyellow", ec="0.5"))

    axes[-1, 0].set_xlabel("observed wavelength [Å]")
    axes[-1, 1].set_xlabel("observed wavelength [Å]")
    fig.suptitle(
        "FITS pipeline ↔ LoaArchive: DLA search comparison on 3 random TIDs\n"
        "(production model_epoch_920.h5, num_dla_samples=10000, max_dlas=1)",
        fontsize=10, y=0.998)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    ar.close()


if __name__ == "__main__":
    main()
