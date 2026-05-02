"""Canonical TID 120046865 overlay: spectrum + truth Voigt + per-model
GP μ + each model's MAP-DLA Voigt.

The MAP DLA values per model are read from
``docs/notes/2026-05-02_v2_canonical_tid_comparison.md`` (or hardcoded
below if the table changes). Truth absorbers come from the mock-0
``hcd_truth_cat.fits``.

Per-model μ comes from the trained model's stored ``mu(rest_lambda)``
shifted to obs frame (×(1+z_qso)) — this is the GP's *prior* mean
continuum shape, not the posterior fit, so the comparison reads as
"how does each trained continuum compare to the actual data?"

Output: 1 figure with N+1 panels stacked. Top panel = spectrum + truth.
Below = one panel per model with GP μ overlay + MAP DLA Voigt + label.
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

CANONICAL_TID = 120046865
CANONICAL_SPEC = (
    "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
    "mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits"
)
CANONICAL_ZCAT = (
    "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
    "mock-0/loa-124/zcat.fits"
)
TRUTH_CAT = (
    "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
    "mock-0/loa-124/hcd_truth_cat.fits"
)


# Per-model MAP results from docs/notes/2026-05-02_v2_canonical_tid_comparison.md.
# (label, model_path, p_dla, MAP_z, MAP_log_NHI). nan = no DLA found.
MODEL_RESULTS = [
    ("v1 baseline (epoch 920)",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5",
     0.920, 2.7735, 21.626),
    ("LOA noHCD-withBAL (normalized)",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/GP_trained/loa_no_hcd_with_bal_normalized/model_epoch_1499.h5",
     0.037, np.nan, np.nan),
    ("LOA noDLA-noBAL norm[1280,1300]",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/GP_trained/loa_no_dla_no_bal_norm1280/model_epoch_1499.h5",
     0.136, np.nan, np.nan),
    ("LOA noDLA-noBAL y1off",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/GP_trained/loa_no_dla_no_bal_y1off/model_epoch_1499.h5",
     0.997, 2.7735, 21.626),
    ("2lpt mock0 (normalized)",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_normalized/model_epoch_1499.h5",
     1.000, 2.7735, 21.626),
    ("2lpt loa124 noHCD-noBAL",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa124_nohcd_nobal_normalized/model_epoch_1499.h5",
     0.842, 2.7735, 21.626),
    ("saclay mock0 noHCD-noBAL",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/saclay_mock0_nohcd_nobal_normalized/model_epoch_1499.h5",
     0.965, 2.7759, 21.489),
]


def _load_canonical():
    from examples.smoke_one_spectrum import load_one_desi_spectrum, lookup_z_qso
    wave, flux, nv, mask = load_one_desi_spectrum(CANONICAL_SPEC, CANONICAL_TID)
    z_qso = float(lookup_z_qso(CANONICAL_ZCAT, CANONICAL_TID))
    return dict(wave=wave, flux=flux, nv=nv, mask=mask, z_qso=z_qso)


def _load_truth_dlas():
    import fitsio
    d = fitsio.read(TRUTH_CAT)
    rows = d[d["TARGETID"] == CANONICAL_TID]
    return [(float(r["Z"]), float(r["NHI"])) for r in rows]


def _load_mu(path):
    """Read mu(rest_lambda) from a trained .h5; handle v1/v2 schema."""
    with h5py.File(path, "r") as f:
        rw = f["rest_wavelengths"][...]
        rw = rw[:, 0] if rw.ndim == 2 else rw
        mu = f["mu"][...]
        mu = mu[:, 0] if mu.ndim == 2 else mu
    return rw, mu


def _voigt(obs_wave, z_dla, log_nhi, num_lines=3):
    """Voigt absorption profile, with broadening turned off so the output
    has the same length as obs_wave (broadening trims the trailing edge by
    2*width pixels)."""
    from gpy_dla_detection.voigt import voigt_absorption
    return voigt_absorption(obs_wave, 10.0**log_nhi, z_dla,
                            num_lines=num_lines, broadening=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Output PNG path")
    args = p.parse_args()

    print("[main] loading canonical spectrum")
    spec = _load_canonical()
    wave = spec["wave"]
    flux = spec["flux"]
    z_qso = spec["z_qso"]
    print(f"  z_qso={z_qso:.4f}  n_pix={len(wave)}")

    truth = _load_truth_dlas()
    print(f"[main] truth absorbers at TID {CANONICAL_TID}:")
    for z, nhi in truth:
        kind = "DLA" if nhi >= 20.3 else "subDLA" if nhi >= 19.5 else "LLS/sub-LLS"
        print(f"  z={z:.4f}  log_NHI={nhi:.3f}  ({kind})")

    n_models = len(MODEL_RESULTS)
    fig, axes = plt.subplots(
        n_models + 1, 1,
        figsize=(11, 2.0 * (n_models + 1)),
        sharex=True,
        gridspec_kw=dict(hspace=0.25),
    )

    # Wavelength range to show: focus on the Lyα forest where DLAs live
    lya_obs = 1215.67 * (1 + z_qso)
    xlim = (3700.0, max(lya_obs + 50, 6000.0))

    # Truth Voigt — combined (multiply both absorbers)
    truth_voigt_each = [_voigt(wave, z, nhi) for z, nhi in truth]
    truth_voigt_combined = np.ones_like(wave)
    for v in truth_voigt_each:
        truth_voigt_combined *= v

    # ===== Top panel: spectrum + truth Voigt =====
    ax = axes[0]
    ax.plot(wave, flux, color="0.4", lw=0.4, label="observed")
    # Show the unit-flux level for reference
    median_red = np.nanmedian(flux[(wave > 5500) & (wave < 6500)])
    if not np.isfinite(median_red) or median_red <= 0:
        median_red = 1.0
    # Plot truth as continuum × absorption (approximate normalization)
    ax.plot(wave, median_red * truth_voigt_combined, color="C2", lw=1.0,
            label=f"truth abs (×median_red={median_red:.2f})", alpha=0.8)
    for z, nhi in truth:
        x_pos = 1215.67 * (1 + z)
        kind = "DLA" if nhi >= 20.3 else "subDLA"
        ax.axvline(x_pos, color="C2", ls=":", lw=0.8, alpha=0.7)
        ax.text(x_pos, ax.get_ylim()[1] * 0.85,
                f"truth {kind}\nz={z:.3f}\nlogNHI={nhi:.2f}",
                fontsize=7, ha="center", color="C2",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="C2", alpha=0.7))
    ax.set_xlim(*xlim)
    ax.set_ylabel("flux\n[10⁻¹⁷ erg/s/cm²/Å]", fontsize=9)
    ax.set_title(f"Canonical TID {CANONICAL_TID}  z_qso={z_qso:.3f}  (2lpt mock-0/loa-124)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # ===== Per-model panels =====
    for i, (label, path, p_dla, map_z, map_nhi) in enumerate(MODEL_RESULTS):
        ax = axes[i + 1]
        # Spectrum (faded)
        ax.plot(wave, flux, color="0.7", lw=0.3, alpha=0.7)

        # GP μ overlay
        rw, mu = _load_mu(path)
        # Shift μ from rest to observed frame and rescale to median
        obs_lambda_for_mu = rw * (1 + z_qso)
        mu_obs_scaled = median_red * mu
        ax.plot(obs_lambda_for_mu, mu_obs_scaled, color="C0", lw=0.8,
                label=f"GP μ (×{median_red:.2f})", alpha=0.9)

        # Truth absorbers (dotted vertical lines)
        for z, nhi in truth:
            x_pos = 1215.67 * (1 + z)
            ax.axvline(x_pos, color="C2", ls=":", lw=0.6, alpha=0.5)

        # MAP DLA Voigt overlay (only if model found one)
        if not np.isnan(map_nhi):
            voigt_map = _voigt(wave, map_z, map_nhi)
            ax.plot(wave, median_red * voigt_map,
                    color="C3", lw=0.9, alpha=0.85,
                    label=f"MAP DLA z={map_z:.3f} logNHI={map_nhi:.2f}")
            x_map = 1215.67 * (1 + map_z)
            ax.axvline(x_map, color="C3", ls="--", lw=0.7, alpha=0.6)

        # Label
        verdict = "DLA" if not np.isnan(map_nhi) else "no DLA"
        title = f"{label}  →  p_dla={p_dla:.3f}  ({verdict})"
        if not np.isnan(map_nhi):
            true_dla_nhi = max(t[1] for t in truth)
            bias = map_nhi - true_dla_nhi
            title += f"  Δlog_NHI={bias:+.3f}"
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("flux", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlim(*xlim)

    axes[-1].set_xlabel("observed wavelength [Å]")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[main] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
