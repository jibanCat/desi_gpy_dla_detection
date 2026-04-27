"""
examples/plot_smoke_result.py
=============================
Plot a single-spectrum GP-DLA detection result from ``smoke_one_spectrum.py``.

Two-panel diagnostic plot:

  Panel A (top)   — observed flux vs wavelength, with
                     · vertical line at the truth DLA Lyα (if --truth-z given)
                     · vertical band at the MAP DLA Lyα ± z error
                     · QSO Lyα emission marker
                     · GP search-window edges (rest-frame [min_lambda, max_lambda]
                       converted to observed-frame at z_qso)

  Panel B (bottom) — (log N_HI, z_DLA) sample posterior from the
                     ``sample_log_likelihoods_dla[:, :, 0]`` (1-DLA model)
                     entry of the saved ``holder.results`` pickle, weighted by
                     exp(log_lik). Shows MAP point and truth point if known.

This is the minimal "did we actually find the DLA?" plot. It does NOT
overlay the full GP+DLA fit (that requires reinstantiating ``DLAGPMAT``);
that's a v2 enhancement.

Usage
-----
    python examples/plot_smoke_result.py \
        --pkl out/smoke/eboss_multidla_120046865.pkl \
        --specfile <2LPT spectra-16-789.fits> \
        --target-id 120046865 \
        --dla-samples /nfs/turbo/.../dla_samples_a03.mat \
        --truth-z 2.773 --truth-nhi 21.26 \
        --out figures/smoke/eboss_120046865.png
"""

from __future__ import annotations

import argparse
import os
import pickle

import h5py
import numpy as np
from astropy.constants import c as C_LIGHT


LYA = 1215.67  # Å, rest


def load_one_desi_spectrum(specfile: str, target_id: int):
    import fitsio
    from desispec.io import read_spectra
    from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log

    spectra = read_spectra(specfile, targetids=[target_id])
    try:
        spectra = coadd_cameras(spectra)
        band = "brz"
    except Exception:
        if spectra.resolution_data is None:
            truthfile = specfile.replace("spectra-16-", "truth-16-")
            spectra.resolution_data = {}
            for cam in ["b", "r", "z"]:
                tres = fitsio.read(truthfile, ext=f"{cam}_RESOLUTION")
                tresdata = np.empty(
                    [spectra.flux[cam].shape[0], tres.shape[0],
                     spectra.flux[cam].shape[1]], dtype=float)
                for i in range(spectra.flux[cam].shape[0]):
                    tresdata[i] = tres
                spectra.resolution_data[cam] = tresdata
        spectra = resample_spectra_lin_or_log(
            spectra, linear_step=0.8,
            wave_min=float(np.min(spectra.wave["b"])),
            wave_max=float(np.max(spectra.wave["z"])),
            fast=True,
        )
        band = "brz" if "brz" in spectra.wave else list(spectra.wave.keys())[0]

    fibermap = spectra.fibermap
    i = int(np.where(np.asarray(fibermap["TARGETID"]) == target_id)[0][0])
    wave = spectra.wave[band].astype(np.float64).copy()
    flux = spectra.flux[band][i].astype(np.float64)
    ivar = spectra.ivar[band][i].astype(np.float64)
    sigma = np.where(ivar > 0, 1.0 / np.sqrt(np.maximum(ivar, 1e-30)), np.inf)
    return wave, flux, sigma


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pkl", required=True, help="holder.results pickle")
    p.add_argument("--specfile", required=True,
                   help="Contaminated spec file used for inference (e.g. loa-124)")
    p.add_argument("--specfile-uncontaminated", default=None,
                   help="(optional) matching uncontaminated spec file (e.g. loa-0); "
                        "overlay shows the same TARGETID without DLA/metals/BAL")
    p.add_argument("--zcat", required=True)
    p.add_argument("--target-id", type=int, required=True)
    p.add_argument("--dla-samples", required=True,
                   help="dla_samples_a03.mat (.mat file the run used)")
    p.add_argument("--truth-z", type=float, default=None,
                   help="(optional) z_DLA from mock truth — drawn as a vertical line")
    p.add_argument("--truth-nhi", type=float, default=None,
                   help="(optional) log N_HI from mock truth — drawn as a marker")
    p.add_argument("--voigt-overlay", action="store_true",
                   help="Overlay analytical Voigt absorption at MAP (NHI,z) on top "
                        "of the uncontaminated continuum. Requires --specfile-uncontaminated.")
    p.add_argument("--title", default=None)
    p.add_argument("--out", required=True, help="output .png path")
    return p.parse_args()


def _voigt_absorption_curve(wave_obs: np.ndarray, log_nhi: float, z_dla: float):
    import sys
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    """Compute a smooth analytical Voigt absorption profile on the observed
    wavelength grid, using the project's compiled C extension if available
    (falls back to the pure-Python voigt). Used only for visualization —
    does NOT include instrumental smoothing beyond what's baked into the C
    extension's hard-coded kernel."""
    from gpy_dla_detection.voigt_fast import VoigtProfile
    v = VoigtProfile()
    # The C extension trims 3 pixels off each side of the input; pad with
    # nearest-neighbour to preserve length so we can plot on the original grid.
    pad = 3
    waves_padded = np.concatenate([
        wave_obs[:1].repeat(pad), wave_obs, wave_obs[-1:].repeat(pad),
    ])
    prof = v.compute_voigt_profile(waves_padded, nhi=10**log_nhi, z_dla=z_dla)
    if prof.size == wave_obs.size:
        return prof
    if prof.size == waves_padded.size - 2 * pad:
        return prof
    # Last-resort fallback
    n = min(prof.size, wave_obs.size)
    out = np.ones(wave_obs.size)
    out[:n] = prof[:n]
    return out


def main():
    args = parse_args()

    with open(args.pkl, "rb") as f:
        results = pickle.load(f)

    z_qso = float(results["z_qsos"][0])
    p_dla = float(results["p_dlas"][0])
    p_null = float(results["p_no_dlas"][0])
    map_z = float(results["MAP_z_dlas"][0, 0])
    map_nhi = float(results["MAP_log_nhis"][0, 0])
    z_err = float(results["z_dla_errs"][0, 0])
    nhi_err = float(results["log_nhi_errs"][0, 0])
    min_z_dla = float(results["min_z_dlas"][0])
    max_z_dla = float(results["max_z_dlas"][0])

    # 1-DLA model per-sample log-likelihoods
    sample_log_lik = np.asarray(results["sample_log_likelihoods_dla"])[0, :, 0]

    # Sample (logNHI, offset) → reconstruct z_DLA per sample
    with h5py.File(args.dla_samples, "r") as ds:
        offset_samples = ds["offset_samples"][:, 0]
        log_nhi_samples = ds["log_nhi_samples"][:, 0]

    # offset → z_DLA via min_z_dla + offset * (max_z_dla - min_z_dla)
    z_dla_samples = min_z_dla + offset_samples * (max_z_dla - min_z_dla)

    # Filter NaN / -inf likelihoods (filter_low_likelihood=1 sets some to NaN)
    finite = np.isfinite(sample_log_lik)
    log_lik = sample_log_lik[finite]
    z_s = z_dla_samples[finite]
    n_s = log_nhi_samples[finite]

    # Normalised posterior weights (subtract max for numerical stability)
    weights = np.exp(log_lik - log_lik.max())
    weights /= weights.sum() if weights.sum() > 0 else 1.0

    # Load spectrum (and optional uncontaminated companion)
    wave, flux, sigma = load_one_desi_spectrum(args.specfile, args.target_id)
    wave_u = flux_u = None
    if args.specfile_uncontaminated:
        wave_u, flux_u, _ = load_one_desi_spectrum(
            args.specfile_uncontaminated, args.target_id
        )

    # Plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.2], hspace=0.28)

    # ----- Panel A: spectrum -----
    ax = fig.add_subplot(gs[0])
    ax.step(wave, flux, where="mid", lw=0.7, color="0.20",
            label="contaminated (loa-124)")
    ax.fill_between(wave, flux - sigma, flux + sigma, step="mid",
                    color="0.7", alpha=0.35, lw=0, label="±σ")
    if wave_u is not None:
        ax.step(wave_u, flux_u, where="mid", lw=0.7, color="C0", alpha=0.75,
                label="uncontaminated (loa-0)")

    # Optional analytical Voigt overlay at MAP (NHI, z), referenced to the
    # local continuum level inferred from the uncontaminated spectrum.
    if args.voigt_overlay and (wave_u is not None) and np.isfinite(map_z):
        try:
            voigt_profile = _voigt_absorption_curve(wave, map_nhi, map_z)
            # Local continuum proxy: nearby uncontaminated flux median.
            map_lya = LYA * (1 + map_z)
            near = (wave_u > map_lya - 60) & (wave_u < map_lya + 60)
            cont = float(np.nanmedian(flux_u[near])) if near.sum() > 5 else 0.3
            ax.plot(wave[: voigt_profile.size], cont * voigt_profile,
                    color="C3", lw=1.5, alpha=0.9,
                    label=f"Voigt(MAP NHI={map_nhi:.2f}) × continuum proxy")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Voigt overlay failed: {exc}")

    qso_lya_obs = LYA * (1 + z_qso)
    ax.axvline(qso_lya_obs, color="goldenrod", lw=1.2, ls=":",
               label=f"QSO Lyα (z={z_qso:.3f})")

    if args.truth_z is not None:
        true_lya_obs = LYA * (1 + args.truth_z)
        ax.axvline(true_lya_obs, color="C2", lw=1.6, ls="--",
                   label=f"truth Lyα (z={args.truth_z:.3f}"
                         + (f", logNHI={args.truth_nhi:.2f})" if args.truth_nhi else ")"))

    if np.isfinite(map_z):
        map_lya_obs = LYA * (1 + map_z)
        ax.axvline(map_lya_obs, color="C3", lw=1.6,
                   label=f"MAP Lyα (z={map_z:.3f}, logNHI={map_nhi:.2f})")
        ax.axvspan(LYA * (1 + map_z - z_err), LYA * (1 + map_z + z_err),
                   color="C3", alpha=0.12, lw=0)

    # GP search window markers (rest [min, max] λ)
    # min/max were set by preset; just bracket using min_z_dla, max_z_dla
    lo = LYA * (1 + min_z_dla)
    hi = LYA * (1 + max_z_dla)
    ax.axvspan(lo, hi, color="C0", alpha=0.04, lw=0)
    ax.text(0.5 * (lo + hi), 0.92, "GP search window (Lyα)",
            transform=ax.get_xaxis_transform(), ha="center", color="C0", fontsize=9)

    ax.set_xlim(max(wave.min(), 0.85 * qso_lya_obs), 1.02 * qso_lya_obs)
    ax.set_ylim(np.percentile(flux, 1) - 1, np.percentile(flux, 99.5) + 1)
    ax.set_xlabel("observed wavelength [Å]")
    ax.set_ylabel("flux  [10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹]")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.text(0.99, 0.97, f"p(DLA)={p_dla:.3f}  p(null)={p_null:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(facecolor="white", edgecolor="0.7"))

    # ----- Panel B: (logNHI, z_DLA) sample posterior -----
    ax2 = fig.add_subplot(gs[1])
    sc = ax2.scatter(z_s, n_s, c=log_lik, s=4, cmap="viridis", lw=0,
                     alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax2, label="sample log-likelihood (1-DLA model)",
                 fraction=0.04, pad=0.01)

    # 2D weighted histogram contour
    if weights.sum() > 0 and (n_s.size > 50):
        # bin in offset (z) and logNHI
        hb, ye, xe = np.histogram2d(
            n_s, z_s, weights=weights, bins=[40, 40],
            range=[[n_s.min(), n_s.max()], [z_s.min(), z_s.max()]],
        )
        # contour lines at 68/95 cumulative-mass levels
        flat = hb.flatten()
        order = np.argsort(flat)[::-1]
        cum = np.cumsum(flat[order])
        cum /= cum[-1]
        levels = []
        for tgt in [0.95, 0.68]:
            idx = np.searchsorted(cum, tgt)
            if idx < flat.size:
                levels.append(flat[order][idx])
        if len(levels) >= 1:
            xc = 0.5 * (xe[1:] + xe[:-1])
            yc = 0.5 * (ye[1:] + ye[:-1])
            ax2.contour(xc, yc, hb, levels=sorted(levels),
                        colors=["C3"], linewidths=[1.5, 2.5][:len(levels)],
                        linestyles=["--", "-"][:len(levels)])

    if args.truth_z is not None and args.truth_nhi is not None:
        ax2.plot(args.truth_z, args.truth_nhi, "*", color="C2", ms=18,
                 mec="black", mew=1.0, label=f"truth ({args.truth_z:.3f}, {args.truth_nhi:.2f})")
    if np.isfinite(map_z) and np.isfinite(map_nhi):
        ax2.errorbar(map_z, map_nhi, xerr=z_err, yerr=nhi_err,
                     fmt="o", color="C3", ms=10, mec="black", mew=0.8,
                     label=f"MAP ({map_z:.3f}±{z_err:.3f}, {map_nhi:.2f}±{nhi_err:.2f})",
                     capsize=3)
    ax2.set_xlabel("z_DLA")
    ax2.set_ylabel("log N_HI [cm⁻²]")
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.85)

    title = args.title or f"TARGETID {args.target_id}  (z_qso={z_qso:.3f})"
    fig.suptitle(title, y=0.995, fontsize=12)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
