"""
examples/smoke_one_spectrum.py
==============================
Single-spectrum end-to-end GP-DLA smoke test runner for GreatLakes.

Loads ONE DESI spectrum (mock or real coadd) via ``desispec.io.read_spectra``,
runs ``run_bayes_select.DLAHolder.process_qso`` with an explicit parameter
preset, and writes results + optional diagnostic plots.

The runner does not change any production code; it sits beside the existing
``examples/demo_desi_spectrum.py`` and adds:
- explicit CLI control over LLS-mode vs multi-DLA-mode
- explicit FILTER_LOW_LIKELIHOOD=0/1 toggle (Q open in production)
- support for the three trained models (eBOSS DR16Q, DESI Y3, London-mock)
- support for both mock spectra-16 files AND real LOA coadd files
- writes HDF5 + optional .pkl with per-sample posterior arrays for plotting

Designed to be the canonical "is the env working / am I using the right
flags" smoke test on a fresh machine.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from dataclasses import dataclass
from typing import Optional

import fitsio
import numpy as np


# ---------------------------------------------------------------------------
# Parameter presets — keep in lockstep with the trained-model wavelength grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelPreset:
    name: str
    learned_file: str
    dlambda: float
    k: int
    min_lambda: float
    max_lambda: float
    loading_min_lambda: float = 910.0
    loading_max_lambda: float = 1550.0
    normalization_min_lambda: float = 1425.0
    normalization_max_lambda: float = 1475.0
    prev_tau_0: float = 0.00554
    prev_beta: float = 3.182
    num_forest_lines: int = 31
    num_lines: int = 3


PRESETS: dict[str, ModelPreset] = {
    # eBOSS DR16Q-trained — original Ho+2020 model, used for SDSS data and the
    # demo spectra. Grid: dlambda 0.25 Å, k=20, lambda 911-1217 Å rest.
    "eboss": ModelPreset(
        name="eboss",
        learned_file=(
            "data/dr12q/processed/"
            "learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat"
        ),
        dlambda=0.25,
        k=20,
        min_lambda=910.75,
        max_lambda=1216.75,
        prev_tau_0=0.0023,   # Kamble+2020 used in demo
        prev_beta=3.65,
        num_forest_lines=3,  # demo uses 3
    ),
    # DESI Y3 production model — trained for Y3 LOA. Grid: dlambda 0.15 Å,
    # k=30, NUM_FOREST_LINES=3, Turner+2024 mean-flux prior.
    "y3": ModelPreset(
        name="y3",
        learned_file="learnlogs/model_epoch_920.h5",
        dlambda=0.15,
        k=30,
        min_lambda=911.75,
        max_lambda=1216.75,
        prev_tau_0=0.00246,  # Turner+2024
        prev_beta=3.62,
        num_forest_lines=3,
    ),
    # London-mock-trained model — assumes same grid hyperparameters as Y3.
    # Latest available epoch in learnlogs_london/ is used.
    "london": ModelPreset(
        name="london",
        learned_file="learnlogs_london/model_epoch_199.h5",
        dlambda=0.15,
        k=30,
        min_lambda=911.75,
        max_lambda=1216.75,
        prev_tau_0=0.00246,
        prev_beta=3.62,
        num_forest_lines=3,
    ),
}


# ---------------------------------------------------------------------------
# Spectrum loader — works for both mock spectra-16 and real LOA coadd files
# ---------------------------------------------------------------------------
def load_one_desi_spectrum(specfile: str, target_id: int):
    """Return (wavelengths, flux, noise_var, mask) for a single TARGETID.

    Accepts either a mock ``spectra-16-XXX.fits`` or a real
    ``coadd-<survey>-<program>-<healpix>.fits``. Mirrors the band-coadd
    fallback chain in ``dlasearch.process_spectra_group`` (lines 318–365):

      1. ``coadd_cameras`` — works when grids align (DESI coadds, some mocks).
      2. If grids don't align AND we have ``resolution_data`` — resample to a
         linear grid then coadd.
      3. If grids don't align AND no ``resolution_data`` — pull per-camera
         resolution from the sibling ``truth-16-XXX.fits`` (mock truth file)
         and resample.
    """
    from desispec.io import read_spectra
    from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log

    spectra = read_spectra(specfile, targetids=[target_id])

    try:
        spectra = coadd_cameras(spectra)
        band = "brz"
    except Exception:
        if spectra.resolution_data is not None:
            wave_min = float(np.min(spectra.wave["b"]))
            wave_max = float(np.max(spectra.wave["z"]))
            spectra = resample_spectra_lin_or_log(
                spectra, linear_step=0.8,
                wave_min=wave_min, wave_max=wave_max, fast=True,
            )
            spectra = coadd_cameras(spectra)
        else:
            truthfile = specfile.replace("spectra-16-", "truth-16-")
            if not os.path.exists(truthfile):
                raise RuntimeError(
                    f"Cannot coadd {specfile}: bands disagree, no resolution_data, "
                    f"and no truth file at {truthfile}."
                )
            spectra.resolution_data = {}
            for cam in ["b", "r", "z"]:
                tres = fitsio.read(truthfile, ext=f"{cam}_RESOLUTION")
                tresdata = np.empty(
                    [spectra.flux[cam].shape[0], tres.shape[0],
                     spectra.flux[cam].shape[1]],
                    dtype=float,
                )
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
        # In the resolution-fallback path, post-coadd the bands key is "brz".
        band = "brz" if "brz" in spectra.wave else list(spectra.wave.keys())[0]

    fibermap = spectra.fibermap
    target_ids = np.asarray(fibermap["TARGETID"])
    matches = np.where(target_ids == target_id)[0]
    if matches.size == 0:
        raise ValueError(
            f"TARGETID {target_id} not found in {specfile}. "
            f"File contains {target_ids.size} targets, sample IDs: "
            f"{target_ids[:5].tolist()}..."
        )
    i = int(matches[0])

    wave = spectra.wave[band].astype(np.float64).copy()
    flux = spectra.flux[band][i].astype(np.float64)
    ivar = spectra.ivar[band][i].astype(np.float64)
    mask = spectra.mask[band][i].astype(bool)

    noise_var = np.full_like(flux, np.inf)
    good = ivar > 0
    noise_var[good] = 1.0 / ivar[good]

    return wave, flux, noise_var, mask


def lookup_z_qso(zcat_file: str, target_id: int) -> float:
    """Pull Z for one TARGETID from a DESI/mocks zcat.fits."""
    cols_try = [
        ["TARGETID", "Z", "ZWARN"],
    ]
    last_err = None
    for cols in cols_try:
        try:
            zcat = fitsio.read(zcat_file, ext=1, columns=cols)
            break
        except Exception as e:
            last_err = e
    else:
        raise last_err

    matches = np.where(zcat["TARGETID"] == target_id)[0]
    if matches.size == 0:
        raise ValueError(f"TARGETID {target_id} not in zcat {zcat_file}")
    row = zcat[int(matches[0])]
    if row["ZWARN"] != 0:
        print(f"[warn] TARGETID {target_id} has ZWARN={row['ZWARN']} (continuing)")
    return float(row["Z"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--specfile", required=True,
                   help="Path to spectra-16-XXX.fits OR coadd-...-<healpix>.fits")
    p.add_argument("--zcat", required=True,
                   help="Path to zcat.fits with the redshift for --target-id")
    p.add_argument("--target-id", type=int, required=True,
                   help="TARGETID of the spectrum to process")
    p.add_argument("--preset", choices=list(PRESETS), default="y3",
                   help="Model parameter preset (default y3)")
    p.add_argument("--data-root", required=True,
                   help="Root containing data/dr12q/processed/, learnlogs/, etc. "
                        "Typically /nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection")
    p.add_argument("--dla-samples-file", default=None,
                   help="Override DLA QMC samples (.mat). If unset, uses preset default.")
    p.add_argument("--sub-dla-samples-file", default=None,
                   help="Override sub-DLA QMC samples (.mat).")
    p.add_argument("--single-absorber-model", type=int, default=1,
                   choices=[0, 1],
                   help="LLS/sub-DLA mode (1) or multi-DLA mode (0). Default 1.")
    p.add_argument("--max-dlas", type=int, default=1,
                   help="MAX_DLAS. Default 1 (LLS mode); use 4 for multi-DLA.")
    p.add_argument("--filter-low-likelihood", type=int, default=0,
                   choices=[0, 1],
                   help="FILTER_LOW_LIKELIHOOD. **Run BOTH 0 and 1** for "
                        "comparison: production currently uses both.")
    p.add_argument("--num-dla-samples", type=int, default=50000)
    p.add_argument("--num-subdla-samples", type=int, default=100000)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=6250)
    p.add_argument("--plot", action="store_true",
                   help="Have DLAHolder dump diagnostic figures to --figure-dir")
    p.add_argument("--figure-dir", default="figures/smoke")
    p.add_argument("--output", default=None,
                   help="HDF5 path for results. Default: smoke_<target>_<preset>_filter<0|1>.h5")
    p.add_argument("--output-pkl", default=None,
                   help="Optional pickle of holder.results (with sample-level "
                        "posteriors needed for the (logNHI,z) contour plot).")
    return p.parse_args()


def main():
    args = parse_args()

    preset = PRESETS[args.preset]

    # Resolve data files relative to --data-root
    def under_root(rel: str) -> str:
        return os.path.join(args.data_root, rel)

    learned_file = under_root(preset.learned_file)
    catalog_name = under_root("data/dr12q/processed/catalog.mat")
    los_catalog = under_root("data/dla_catalogs/dr9q_concordance/processed/los_catalog")
    dla_catalog = under_root("data/dla_catalogs/dr9q_concordance/processed/dla_catalog")

    # Default sample files: production LLS-mode set if not overridden
    dla_samples_file = (
        args.dla_samples_file
        or under_root("data/dr12q/processed/pw_samples_a3_172_220_50000.mat")
    )
    sub_dla_samples_file = (
        args.sub_dla_samples_file
        or under_root("data/dr12q/processed/subdla_samples_a03_191_200_100000.mat")
    )

    missing = [p for p in [learned_file, catalog_name, los_catalog, dla_catalog,
                           dla_samples_file, sub_dla_samples_file]
               if not os.path.exists(p)]
    if missing:
        print("[error] missing input files:")
        for m in missing:
            print("  ", m)
        sys.exit(2)

    # Load one spectrum
    print(f"[load] {os.path.basename(args.specfile)} TARGETID={args.target_id}")
    wave, flux, noise_var, mask = load_one_desi_spectrum(args.specfile, args.target_id)
    z_qso = lookup_z_qso(args.zcat, args.target_id)
    print(f"  wave: {wave.size} pixels, {wave[0]:.1f}–{wave[-1]:.1f} Å")
    print(f"  flux: median={np.nanmedian(flux):.3f}")
    print(f"  z_qso = {z_qso:.4f}")

    # Build Parameters and DLAHolder
    sys.path.insert(0, os.getcwd())
    from gpy_dla_detection.set_parameters import Parameters
    from run_bayes_select import DLAHolder

    common = dict(
        loading_min_lambda=preset.loading_min_lambda,
        loading_max_lambda=preset.loading_max_lambda,
        normalization_min_lambda=preset.normalization_min_lambda,
        normalization_max_lambda=preset.normalization_max_lambda,
        min_lambda=preset.min_lambda,
        max_lambda=preset.max_lambda,
        dlambda=preset.dlambda,
        k=preset.k,
        max_noise_variance=9.0,
        num_lines=preset.num_lines,
        max_z_cut=3000.0,
        min_z_cut=3000.0,
        num_forest_lines=preset.num_forest_lines,
    )
    params = Parameters(num_dla_samples=args.num_dla_samples, **common)
    params_subdla = Parameters(num_dla_samples=args.num_subdla_samples, **common)

    print(f"[holder] preset={args.preset}  single_absorber={bool(args.single_absorber_model)}  "
          f"max_dlas={args.max_dlas}  filter_low_likelihood={bool(args.filter_low_likelihood)}")
    print(f"  learned_file = {os.path.relpath(learned_file, args.data_root)}")
    print(f"  dla_samples  = {os.path.relpath(dla_samples_file, args.data_root)}")
    print(f"  subdla       = {os.path.relpath(sub_dla_samples_file, args.data_root)}")

    holder = DLAHolder(
        learned_file=learned_file,
        catalog_name=catalog_name,
        los_catalog=los_catalog,
        dla_catalog=dla_catalog,
        dla_samples_file=dla_samples_file,
        sub_dla_samples_file=sub_dla_samples_file,
        params=params,
        params_subdla=params_subdla,
        min_z_separation=3000.0,
        prev_tau_0=preset.prev_tau_0,
        prev_beta=preset.prev_beta,
        max_dlas=args.max_dlas,
        broadening=True,
        plot_figures=args.plot,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        figure_dir=args.figure_dir,
        filter_low_likelihood=bool(args.filter_low_likelihood),
        single_absorber_model=bool(args.single_absorber_model),
    )
    holder.initialize_results(1)

    if args.plot:
        os.makedirs(args.figure_dir, exist_ok=True)

    t0 = time.time()
    holder.process_qso(
        idx=0,
        target_id=str(args.target_id),
        wavelengths=wave,
        flux=flux,
        noise_variance=noise_var,
        pixel_mask=mask,
        z_qso=z_qso,
    )
    dt = time.time() - t0
    print(f"[done] inference took {dt:.1f}s")

    # Summarise
    res = holder.results
    print("\n=== smoke result ===")
    print(f"TARGETID         {args.target_id}")
    print(f"z_qso            {z_qso:.4f}")
    print(f"p(no absorber)   {res['p_no_dlas'][0]:.4f}")
    print(f"p(>=1 absorber)  {res['p_dlas'][0]:.4f}")
    print(f"MAP z_dla[0]     {res['MAP_z_dlas'][0, 0]:.4f}")
    print(f"MAP logNHI[0]    {res['MAP_log_nhis'][0, 0]:.3f}")
    print(f"model_posteriors {res['model_posteriors'][0].tolist()}")

    out = args.output or (
        f"smoke_{args.target_id}_{args.preset}_filter{args.filter_low_likelihood}.h5"
    )
    holder.save_results(out)
    print(f"[saved] {out}")

    if args.output_pkl:
        with open(args.output_pkl, "wb") as f:
            pickle.dump(holder.results, f)
        print(f"[saved] {args.output_pkl} (pickle for plotting)")


if __name__ == "__main__":
    main()
