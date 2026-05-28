"""
examples/plot_smoke_v2.py
=========================
Unified diagnostic plot combining the project's GP+DLA fit overlay with the
custom (z_DLA, log N_HI) posterior contour, plus prominent truth markers.

Three panels:
  A — observed spectrum + GP mean continuum + GP+DLA fit + ±2σ band
        truth and MAP DLA Lyα marked. If --specfile-uncontaminated is
        provided, the matching loa-0 / jura-0 spectrum is overlaid.
  B — sample log-likelihood scatter on (z_DLA, log N_HI) plus 68/95 %
        contours, truth star, MAP errorbar.
  C — header strip with run metadata + truth/MAP comparison.

Reuses the saved smoke .pkl for MAP / sample_log_likelihoods, and re-runs
NullGPMAT.set_data + DLAGPMAT.this_dla_gp on the spectrum to get the model
mean. This is cheap (one matrix multiply per pixel; no log-evidence loop).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import h5py
import numpy as np


LYA = 1215.67


# --------------------------------------------------------------------------
# Pure-Python analytical Voigt absorption (no instrument convolution).
# Mirrors the constants in gpy_dla_detection/ctypes_voigt.c so the LSF-free
# answer here is a fair "what the DLA looks like physically" baseline that
# the LSF-convolved model overlay can be compared against visually.
# --------------------------------------------------------------------------
_C_CGS = 2.99792458e10            # cm/s
# Doppler sigma for T = 10^4 K, m_H proton mass; same as the C extension.
_SIGMA = 9.08537121627923800e5    # cm/s
# First three Lyman lines (cm), oscillator strengths, transition rates s^-1.
_LINES = [
    (1.2156701e-05, 0.416400, 6.265e+08),   # Lyα
    (1.0257223e-05, 0.079120, 1.897e+08),   # Lyβ
    (9.725368e-06,  0.029000, 8.127e+07),   # Lyγ
]


def raw_voigt_absorption(wave_obs_A: np.ndarray, log_nhi: float,
                         z_dla: float, num_lines: int = 3) -> np.ndarray:
    """Voigt absorption exp(-Nτ) on `wave_obs_A` [Å], with no LSF kernel.
    The optical depth sums over Lyα, Lyβ, Lyγ (or fewer if num_lines<3).
    """
    from scipy.special import wofz  # Faddeeva function w(z) = exp(-z^2)·erfc(-iz)
    wave_cm = wave_obs_A * 1e-8
    total = np.zeros_like(wave_obs_A, dtype=float)
    N = 10.0 ** log_nhi
    # The C extension uses leading_constants[i] = π e² f λ / (m_e c) [cm²].
    # Pre-computed for first three lines (matches the .c file exactly).
    leading_consts_first3 = [
        1.34347262962625339e-07,   # Lyα
        2.15386482180851912e-08,   # Lyβ
        7.48525170087141461e-09,   # Lyγ
    ]
    gammas_first3 = [
        6.06075804241938613e+02,   # Lyα Lorentzian half-width [cm/s]
        1.54841462408931704e+02,   # Lyβ
        6.28964942715328164e+01,   # Lyγ
    ]
    for i in range(min(num_lines, 3)):
        lam_line = _LINES[i][0]
        # velocity from QSO-Lyα frame: v = c·(wave_cm/(λ·(1+z)) - 1)
        vel = _C_CGS * (wave_cm / (lam_line * (1 + z_dla)) - 1.0)
        # Voigt profile via Faddeeva: V(v; σ, γ) = Re[w((v + iγ) / (σ√2))] / (σ√(2π))
        sigma = _SIGMA
        gamma = gammas_first3[i]
        z = (vel + 1j * gamma) / (sigma * np.sqrt(2.0))
        voigt_v = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
        total += leading_consts_first3[i] * voigt_v
    return np.exp(-N * total)


def _absorber_type(log_nhi: float) -> str:
    if log_nhi >= 20.3:
        return "DLA"
    if log_nhi >= 19.5:
        return "subDLA"
    return "LLS"


_TYPE_COLOR = {"DLA": "#2ca02c", "subDLA": "#ff7f0e", "LLS": "#8c564b"}


def read_truth_absorbers(truth_cat: str, target_id: int):
    """All true absorbers (z, log_nhi, type) for a TARGETID, strongest first."""
    import fitsio
    t = fitsio.read(truth_cat, ext=1)
    m = t["TARGETID"] == target_id
    zc = "Z" if "Z" in t.dtype.names else ("Z_DLA" if "Z_DLA" in t.dtype.names else "Z_QSO")
    return sorted(
        [(float(z), float(n), _absorber_type(float(n)))
         for z, n in zip(t[zc][m], t["NHI"][m])],
        key=lambda r: -r[1],
    )


def bal_civ_troughs(bal_cat: str, target_id: int, z_qso: float):
    """Observed-Å BAL trough bands for the main UV lines in the Lyα forest.
    Uses the AI CIV_450 trough velocities; order-agnostic (lo=min,hi=max v).
    Convention λ_obs = λ_rest·(1+z_qso)·(1 − v/c), positive v = blueward."""
    import fitsio
    b = fitsio.read(bal_cat, ext=1)
    idx = np.where(b["TARGETID"] == target_id)[0]
    if idx.size == 0:
        return []
    r = idx[0]
    vmin = np.atleast_1d(b["VMIN_CIV_450"][r]).astype(float)
    vmax = np.atleast_1d(b["VMAX_CIV_450"][r]).astype(float)
    good = np.isfinite(vmin) & np.isfinite(vmax) & ((vmin != 0) | (vmax != 0))
    vmin, vmax = vmin[good], vmax[good]
    if vmin.size == 0:
        return []
    c = 299792.458
    lines = [1215.67, 1240.81, 1206.50, 1031.93, 1037.62, 1025.72, 972.54]
    bands = []
    for v0, v1 in zip(vmin, vmax):
        lo_v, hi_v = min(v0, v1), max(v0, v1)
        for lam in lines:
            obs = lam * (1 + z_qso)
            bands.append((obs * (1 - hi_v / c), obs * (1 - lo_v / c)))
    return bands


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pkl", required=True)
    p.add_argument("--specfile", required=True)
    p.add_argument("--specfile-uncontaminated", default=None,
                   help="(optional) sibling un-contaminated mock for overlay")
    p.add_argument("--zcat", required=True)
    p.add_argument("--target-id", type=int, required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--preset", choices=["eboss", "y3", "london", "2lpt"], required=True)
    p.add_argument("--learned-file", default=None,
                   help="Absolute path to the trained GP model HDF5; overrides "
                        "the preset default (needed for the 2lpt model, which "
                        "lives outside --data-root).")
    p.add_argument("--dla-samples-file", default=None)
    p.add_argument("--sub-dla-samples-file", default=None)
    p.add_argument("--num-dla-samples", type=int, default=10000)
    p.add_argument("--num-subdla-samples", type=int, default=10000)
    p.add_argument("--truth-z", type=float, default=None)
    p.add_argument("--truth-nhi", type=float, default=None)
    p.add_argument("--truth-cat", default=None,
                   help="hcd_truth_cat.fits / dla_cat.fits: overlay ALL true "
                        "absorbers for this TARGETID, colored by class "
                        "(DLA/subDLA/LLS).")
    p.add_argument("--bal-cat", default=None,
                   help="bal_cat.fits: shade the CIV-velocity BAL troughs "
                        "(over the main UV lines) if this TARGETID is a BAL.")
    p.add_argument("--snr", type=float, default=None,
                   help="(optional) SNR_FOREST from snr_cat for the title")
    p.add_argument("--title", default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


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
            truth = specfile.replace("spectra-16-", "truth-16-")
            spectra.resolution_data = {}
            for cam in "brz":
                tres = fitsio.read(truth, ext=f"{cam}_RESOLUTION")
                td = np.empty([spectra.flux[cam].shape[0], tres.shape[0],
                               spectra.flux[cam].shape[1]], dtype=float)
                for i in range(spectra.flux[cam].shape[0]):
                    td[i] = tres
                spectra.resolution_data[cam] = td
        spectra = resample_spectra_lin_or_log(
            spectra, linear_step=0.8,
            wave_min=float(np.min(spectra.wave["b"])),
            wave_max=float(np.max(spectra.wave["z"])),
            fast=True,
        )
        band = "brz" if "brz" in spectra.wave else list(spectra.wave.keys())[0]
    i = int(np.where(np.asarray(spectra.fibermap["TARGETID"]) == target_id)[0][0])
    wave = spectra.wave[band].astype(np.float64)
    flux = spectra.flux[band][i].astype(np.float64)
    ivar = spectra.ivar[band][i].astype(np.float64)
    mask = spectra.mask[band][i].astype(bool)
    nv = np.full_like(flux, np.inf)
    good = ivar > 0
    nv[good] = 1.0 / ivar[good]
    return wave, flux, nv, mask


def main():
    args = parse_args()

    # Load smoke results
    with open(args.pkl, "rb") as f:
        results = pickle.load(f)

    z_qso = float(results["z_qsos"][0])
    p_dla = float(results["p_dlas"][0])
    p_null = float(results["p_no_dlas"][0])
    # MAP_z_dlas/MAP_log_nhis store the selected k-DLA solution: indices 0..k-1
    # are the k MAP values, and indices k..max_dlas-1 are NaN. Pull all finite
    # entries so multi-DLA selections are plotted in full.
    map_z_arr   = np.asarray(results["MAP_z_dlas"])[0]
    map_nhi_arr = np.asarray(results["MAP_log_nhis"])[0]
    z_err_arr   = np.asarray(results["z_dla_errs"])[0]
    nhi_err_arr = np.asarray(results["log_nhi_errs"])[0]
    finite = np.isfinite(map_z_arr) & np.isfinite(map_nhi_arr)
    map_z_all   = map_z_arr[finite]
    map_nhi_all = map_nhi_arr[finite]
    z_err_all   = z_err_arr[finite]
    nhi_err_all = nhi_err_arr[finite]
    n_selected = int(finite.sum())
    map_z   = float(map_z_all[0])   if n_selected else float("nan")
    map_nhi = float(map_nhi_all[0]) if n_selected else float("nan")
    z_err   = float(z_err_all[0])   if n_selected else float("nan")
    nhi_err = float(nhi_err_all[0]) if n_selected else float("nan")
    min_z_dla = float(results["min_z_dlas"][0])
    max_z_dla = float(results["max_z_dlas"][0])
    sample_log_lik = np.asarray(results["sample_log_likelihoods_dla"])[0, :, 0]

    # Re-instantiate the GP to get this_mu (continuum) and dla_mu (continuum×Voigt at MAP)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from gpy_dla_detection.set_parameters import Parameters
    from gpy_dla_detection.model_priors import PriorCatalog
    from gpy_dla_detection.dla_samples import DLASamplesMAT
    from gpy_dla_detection.subdla_samples import SubDLASamplesMAT
    from gpy_dla_detection.dla_gp import DLAGPMAT
    from examples.smoke_one_spectrum import PRESETS

    preset = PRESETS[args.preset]
    learned = args.learned_file if args.learned_file else os.path.join(args.data_root, preset.learned_file)
    catalog = os.path.join(args.data_root, "data/dr12q/processed/catalog.mat")
    los_cat = os.path.join(args.data_root, "data/dla_catalogs/dr9q_concordance/processed/los_catalog")
    dla_cat = os.path.join(args.data_root, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog")
    dla_samples_file = args.dla_samples_file or os.path.join(
        args.data_root, "data/dr12q/processed/dla_samples_a03.mat")
    sub_dla_samples_file = args.sub_dla_samples_file or os.path.join(
        args.data_root, "data/dr12q/processed/subdla_samples.mat")

    common = dict(
        loading_min_lambda=preset.loading_min_lambda,
        loading_max_lambda=preset.loading_max_lambda,
        normalization_min_lambda=preset.normalization_min_lambda,
        normalization_max_lambda=preset.normalization_max_lambda,
        min_lambda=preset.min_lambda, max_lambda=preset.max_lambda,
        dlambda=preset.dlambda, k=preset.k,
        max_noise_variance=9.0, num_lines=preset.num_lines,
        max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=preset.num_forest_lines,
    )
    params = Parameters(num_dla_samples=args.num_dla_samples, **common)
    prior = PriorCatalog(params, catalog, los_cat, dla_cat)
    dla_samples = DLASamplesMAT(params, prior, dla_samples_file)

    dla_gp = DLAGPMAT(params, prior, dla_samples, min_z_separation=3000.0,
                     learned_file=learned, broadening=True,
                     prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta)

    # Load spectrum and feed to dla_gp.set_data
    wave, flux, nv, mask = load_one_desi_spectrum(args.specfile, args.target_id)
    rest = params.emitted_wavelengths(wave, z_qso)
    dla_gp.set_data(rest, flux, nv, mask, z_qso=z_qso,
                    normalize=True, build_model=True)

    # Continuum (no DLA) and full DLA model at MAP (NHI, z) for ALL selected
    # DLAs (k can be > 1 for multi-DLA selections).
    this_mu = dla_gp.this_mu
    if n_selected > 0:
        dla_mu, _, _ = dla_gp.this_dla_gp(map_z_all.copy(),
                                          10.0 ** map_nhi_all)
    else:
        dla_mu = None

    # Optional un-contaminated overlay
    wave_u = flux_u = None
    if args.specfile_uncontaminated:
        try:
            wave_u, flux_u, _, _ = load_one_desi_spectrum(
                args.specfile_uncontaminated, args.target_id)
            # Normalise by the same median used for the contaminated run.
            flux_u = flux_u / dla_gp.normalization_median
        except Exception as exc:
            print(f"[warn] couldn't load uncontaminated overlay: {exc}")
            wave_u = flux_u = None

    # Reconstruct sample (logNHI, z_DLA) from .mat file + saved bounds
    with h5py.File(dla_samples_file, "r") as ds:
        offsets = ds["offset_samples"][:, 0]
        nhis = ds["log_nhi_samples"][:, 0]
    z_samples = min_z_dla + offsets * (max_z_dla - min_z_dla)
    finite = np.isfinite(sample_log_lik)
    log_lik = sample_log_lik[finite]
    z_s = z_samples[finite]
    n_s = nhis[finite]
    if log_lik.size:
        weights = np.exp(log_lik - log_lik.max())
        weights /= weights.sum() if weights.sum() > 0 else 1.0
    else:
        weights = np.zeros_like(log_lik)

    # ---- Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13, 9.5))
    gs = GridSpec(2, 1, height_ratios=[1.05, 1.15], hspace=0.27)

    # Panel A — observed spectrum + GP fit + truth/MAP markers
    ax = fig.add_subplot(gs[0])
    obs_x = dla_gp.x * (1 + z_qso)              # back to observed wavelength
    Y = dla_gp.Y                                # normalised flux
    sigma = np.sqrt(np.maximum(dla_gp.v, 0.0))

    ax.fill_between(obs_x, Y - 2 * sigma, Y + 2 * sigma,
                    color="C0", alpha=0.22, lw=0, label="±2σ instrumental")
    ax.step(obs_x, Y, where="mid", lw=0.8, color="0.20",
            label="contaminated flux (normalised)")
    if wave_u is not None and flux_u is not None:
        in_range = (wave_u >= obs_x[0]) & (wave_u <= obs_x[-1])
        ax.step(wave_u[in_range], flux_u[in_range], where="mid", lw=0.7,
                color="C0", alpha=0.6, label="uncontaminated (loa-0/jura-0)")

    ax.plot(obs_x, this_mu, lw=1.4, color="0.0", alpha=0.7,
            label="GP mean continuum (no DLA)")
    if dla_mu is not None:
        if n_selected == 1:
            map_lbl = f"GP × Voigt(MAP)  z={map_z:.4f}, logNHI={map_nhi:.2f}"
        else:
            entries = ", ".join(f"({z:.4f}, {n:.2f})"
                                for z, n in zip(map_z_all, map_nhi_all))
            map_lbl = f"GP × Voigt(MAP, k={n_selected})  [{entries}]"
        ax.plot(obs_x, dla_mu, lw=1.8, color="C3", label=map_lbl)
        # Vertical lines + small annotations for each MAP DLA.
        for i, (mz, mn) in enumerate(zip(map_z_all, map_nhi_all)):
            ax.axvline(LYA * (1 + mz), color="C3", lw=1.0, alpha=0.7,
                       ls=(":" if i > 0 else "-"))

    if args.truth_z is not None:
        true_lya = LYA * (1 + args.truth_z)
        ax.axvline(true_lya, color="C2", lw=1.6, ls="--",
                   label=(f"truth Lyα  z={args.truth_z:.4f}"
                          + (f", logNHI={args.truth_nhi:.2f}"
                             if args.truth_nhi is not None else "")))
        # Annotate truth at the line position (top of axes)
        ax.annotate(
            f"  truth\n  z={args.truth_z:.4f}\n  logNHI={args.truth_nhi:.2f}"
            if args.truth_nhi is not None else f"  truth\n  z={args.truth_z:.4f}",
            xy=(true_lya, 0.97), xycoords=("data", "axes fraction"),
            ha="left", va="top", color="C2", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="C2", alpha=0.85, pad=2),
        )

        # Unconvolved Voigt at truth (NHI, z), drawn on top of the GP continuum.
        # This shows what the DLA "should" look like physically without the
        # production LSF kernel. The difference between this and the red
        # GP×Voigt(MAP) overlay is exactly the LSF + N_HI bias contribution.
        if args.truth_nhi is not None:
            try:
                truth_abs = raw_voigt_absorption(obs_x, args.truth_nhi,
                                                 args.truth_z, num_lines=3)
                ax.plot(obs_x, this_mu * truth_abs, color="C2", lw=1.4,
                        ls=":",
                        label=f"continuum × raw Voigt(truth, no LSF)")
            except Exception as exc:
                print(f"[warn] truth Voigt overlay failed: {exc}")
    for mz, ze in zip(map_z_all, z_err_all):
        ax.axvspan(LYA * (1 + mz - ze), LYA * (1 + mz + ze),
                   color="C3", alpha=0.10, lw=0)

    # All true absorbers from --truth-cat, colored by class; logNHI annotated.
    if args.truth_cat:
        seen = set()
        for tz, tn, typ in read_truth_absorbers(args.truth_cat, args.target_id):
            col = _TYPE_COLOR[typ]
            ax.axvline(LYA * (1 + tz), color=col, lw=1.5, ls="--", alpha=0.9,
                       label=(f"true {typ}" if typ not in seen else None))
            seen.add(typ)
            ax.annotate(f"{tn:.1f}", xy=(LYA * (1 + tz), 0.02),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", color=col, fontsize=7)
    # BAL CIV troughs from --bal-cat (shaded purple bands over the UV lines).
    if args.bal_cat:
        first = True
        for lo, hi in bal_civ_troughs(args.bal_cat, args.target_id, z_qso):
            ax.axvspan(lo, hi, color="purple", alpha=0.10, lw=0,
                       label=("BAL CIV trough" if first else None))
            first = False

    ax.axvline(LYA * (1 + z_qso), color="goldenrod", lw=1.0, ls=":",
               label=f"QSO Lyα (z_qso={z_qso:.3f})")

    # Limit x-range to the GP search window plus a little padding
    xmin = LYA * (1 + min_z_dla) - 30
    xmax = LYA * (1 + max_z_dla) + 30
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(np.nanpercentile(Y, 1) - 0.3, np.nanpercentile(Y, 99) + 0.5)
    ax.set_xlabel("observed wavelength [Å]")
    ax.set_ylabel("normalised flux")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85, ncol=2)
    info = f"p(DLA)={p_dla:.3f}  p(null)={p_null:.3f}"
    if args.snr is not None:
        info = f"SNR_forest={args.snr:.2f}   " + info
    ax.text(0.99, 0.97, info,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(facecolor="white", edgecolor="0.7"))

    # Panel B — (z_DLA, log N_HI) sample posterior
    ax2 = fig.add_subplot(gs[1])
    sc = ax2.scatter(z_s, n_s, c=log_lik, s=4, cmap="viridis", lw=0,
                     alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax2, label="sample log-likelihood (1-DLA model)",
                 fraction=0.04, pad=0.01)

    if z_s.size > 50 and weights.sum() > 0:
        hb, ye, xe = np.histogram2d(
            n_s, z_s, weights=weights, bins=[40, 40],
            range=[[n_s.min(), n_s.max()], [z_s.min(), z_s.max()]],
        )
        flat = hb.flatten()
        order = np.argsort(flat)[::-1]
        cum = np.cumsum(flat[order])
        cum /= cum[-1]
        levels = []
        for tgt in [0.95, 0.68]:
            idx = np.searchsorted(cum, tgt)
            if idx < flat.size:
                levels.append(flat[order][idx])
        levels = sorted(set(levels))
        if levels:
            xc = 0.5 * (xe[1:] + xe[:-1])
            yc = 0.5 * (ye[1:] + ye[:-1])
            ax2.contour(xc, yc, hb, levels=levels,
                        colors=["C3"] * len(levels),
                        linewidths=[1.5, 2.5][:len(levels)],
                        linestyles=["--", "-"][:len(levels)])

    if args.truth_z is not None and args.truth_nhi is not None:
        ax2.plot(args.truth_z, args.truth_nhi, "*", color="C2", ms=20,
                 mec="black", mew=1.0, zorder=10,
                 label=f"truth ({args.truth_z:.3f}, {args.truth_nhi:.2f})")
    # All true absorbers as class-colored stars (DLA/subDLA in range; LLS may
    # sit below the 1-DLA sample band).
    if args.truth_cat:
        seen = set()
        for tz, tn, typ in read_truth_absorbers(args.truth_cat, args.target_id):
            ax2.plot(tz, tn, "*", color=_TYPE_COLOR[typ], ms=16, mec="black",
                     mew=0.8, zorder=10,
                     label=(f"true {typ}" if typ not in seen else None))
            seen.add(typ)
    for i, (mz, mn, ze, ne) in enumerate(zip(
            map_z_all, map_nhi_all, z_err_all, nhi_err_all)):
        lbl = (f"MAP DLA{i+1} ({mz:.3f}±{ze:.3f}, {mn:.2f}±{ne:.2f})"
               if n_selected > 1 else
               f"MAP ({mz:.3f}±{ze:.3f}, {mn:.2f}±{ne:.2f})")
        ax2.errorbar(mz, mn, xerr=ze, yerr=ne, fmt="o", color="C3",
                     ms=10, mec="black", mew=0.8, capsize=3, zorder=10,
                     label=lbl)
    ax2.set_xlabel("z_DLA")
    ax2.set_ylabel("log N_HI [cm⁻²]")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.85)

    # Title block
    delta_msg = ""
    if args.truth_nhi is not None and np.isfinite(map_nhi):
        delta_msg = f"   ΔlogNHI = {map_nhi - args.truth_nhi:+.3f}"
    title = args.title or (
        f"{args.preset.upper()} preset — TARGETID {args.target_id}"
        f" (z_qso={z_qso:.3f}){delta_msg}"
    )
    fig.suptitle(title, y=0.995, fontsize=12)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
