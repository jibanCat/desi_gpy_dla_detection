#!/usr/bin/env python
"""Streamlined 2LPT mock preload → gp_interp_trainset.h5.

Reads 2LPT mock spectra (loa-0 uncontaminated or loa-124 contaminated),
optionally filters out HCDs (DLA / sub-DLA / LLS) and BALs from the
truth catalogs, applies the legacy SpectrumProcessor preprocessing
(mask noisy pixels + interpolate to common rest grid), and writes an
HDF5 in the *legacy* ``gp_interp_trainset.h5`` schema readable by
``gpy_dla_detection.training.dataset.load_preprocessed_h5``.

This is a focused alternative to the full ``preload_spectra/desi-preload.py
+ prepare_trainset.py`` pipeline — it does NOT need DLAHolder /
``run_bayes_select`` and runs in a single Python process, which fits
inside one GreatLakes SLURM job.

Usage::

    # 2LPT loa-0, uncontaminated baseline
    python preload_spectra/preload_2lpt_simple.py \\
        --mock-dir /nfs/turbo/.../mock-0/loa-0 \\
        --output  /nfs/turbo/.../trainset_2lpt_loa0.h5 \\
        --max-spectra 50000

    # 2LPT loa-124, with HCDs/BALs FILTERED OUT
    python preload_spectra/preload_2lpt_simple.py \\
        --mock-dir /nfs/turbo/.../mock-0/loa-124 \\
        --output  /nfs/turbo/.../trainset_2lpt_loa124_nohcd_nobal.h5 \\
        --max-spectra 50000 \\
        --exclude-hcd --exclude-bal
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d

# repo root on path
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# DESI healpix layout: spectra-16/{healpix//100}/{healpix}/spectra-16-{healpix}.fits
# ---------------------------------------------------------------------------
def _healpix_for_radec(ra, dec, nside=16):
    """nside=16 NESTED healpix index from RA/DEC, in degrees."""
    import healpy as hp
    theta = np.deg2rad(90.0 - np.asarray(dec))
    phi = np.deg2rad(np.asarray(ra))
    return hp.ang2pix(nside, theta, phi, nest=True)


def _spec_path(mock_dir: Path, healpix: int) -> Path:
    """Return spectra-16-{healpix}.fits path, mirroring the 2LPT layout."""
    return (mock_dir / "spectra-16" / str(healpix // 100) / str(healpix)
            / f"spectra-16-{healpix}.fits")


# ---------------------------------------------------------------------------
# Filtering: HCD + BAL anti-joins
# ---------------------------------------------------------------------------
def _build_targetid_filter(zcat: Table, mock_dir: Path,
                           exclude_hcd: bool, exclude_bal: bool,
                           hcd_min_nhi: float = 17.0) -> np.ndarray:
    """Return boolean mask over zcat rows: True = keep this TARGETID.

    Reads truth catalogs from mock_dir if requested. For 2LPT loa-0
    (uncontaminated) the catalogs may not exist — caller should not
    pass --exclude-hcd / --exclude-bal in that case.
    """
    keep = np.ones(len(zcat), dtype=bool)
    tids = np.asarray(zcat["TARGETID"])

    if exclude_hcd:
        hcd_path = mock_dir / "hcd_truth_cat.fits"
        if not hcd_path.exists():
            print(f"[filter] WARN: --exclude-hcd requested but {hcd_path} not found; skipping")
        else:
            hcd = Table.read(hcd_path)
            mask_strong = hcd["NHI"] >= hcd_min_nhi
            bad_tids = set(int(x) for x in hcd["TARGETID"][mask_strong])
            print(f"[filter] HCD: {len(bad_tids)} unique TARGETIDs with logNHI ≥ {hcd_min_nhi}")
            in_bad = np.isin(tids, list(bad_tids))
            keep &= ~in_bad

    if exclude_bal:
        bal_path = mock_dir / "bal_cat.fits"
        if not bal_path.exists():
            print(f"[filter] WARN: --exclude-bal requested but {bal_path} not found; skipping")
        else:
            bal = Table.read(bal_path)
            if "BI_CIV" in bal.colnames:
                mask_bal = bal["BI_CIV"] > 0
            elif "ai_civ" in bal.colnames:
                mask_bal = bal["ai_civ"] > 0
            else:
                raise KeyError(f"bal_cat has neither BI_CIV nor ai_civ: {bal.colnames}")
            bad_bal_tids = set(int(x) for x in bal["TARGETID"][mask_bal])
            print(f"[filter] BAL: {len(bad_bal_tids)} unique TARGETIDs with positive BI/AI")
            in_bal = np.isin(tids, list(bad_bal_tids))
            keep &= ~in_bal

    return keep


# ---------------------------------------------------------------------------
# Spectrum reader (per-TARGETID, batched by healpix file)
# ---------------------------------------------------------------------------
def _read_one_healpix_file(specfile: Path, target_ids: list[int]):
    """Read multiple TARGETIDs from one spectra-16-XXX.fits file.

    Returns list of (target_id, wave, flux, ivar, mask). Wave is observed-frame
    Å on the band-coadded grid.

    Falls back through the same chain as ``examples/smoke_one_spectrum.py``:
      1) coadd_cameras directly (works when grids align — DESI coadds, some mocks)
      2) if grids don't align AND resolution_data is present → resample then coadd
      3) if grids don't align AND no resolution_data → pull per-camera resolution
         from truth-16-XXX.fits and resample (2LPT mocks fall here)
    """
    import fitsio
    from desispec.io import read_spectra
    from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log

    spectra = read_spectra(str(specfile), targetids=target_ids)

    coadd_succeeded = False
    band = "brz"
    try:
        spectra_co = coadd_cameras(spectra)
        # Some desispec versions only LOG and return; verify the result has the band key.
        if "brz" in spectra_co.wave or "b" in spectra_co.wave:
            spectra = spectra_co
            band = "brz" if "brz" in spectra.wave else list(spectra.wave.keys())[0]
            coadd_succeeded = True
    except Exception:
        coadd_succeeded = False

    if not coadd_succeeded:
        # Bands don't align — resample first, then coadd.
        if spectra.resolution_data is None:
            # 2LPT mocks: pull resolution from sibling truth-16-XXX.fits.
            truthfile = str(specfile).replace("spectra-16-", "truth-16-")
            if not os.path.exists(truthfile):
                return []
            spectra.resolution_data = {}
            for cam in ("b", "r", "z"):
                tres = fitsio.read(truthfile, ext=f"{cam}_RESOLUTION")
                tresdata = np.empty(
                    [spectra.flux[cam].shape[0], tres.shape[0],
                     spectra.flux[cam].shape[1]],
                    dtype=float,
                )
                for ii in range(spectra.flux[cam].shape[0]):
                    tresdata[ii] = tres
                spectra.resolution_data[cam] = tresdata

        wave_min = float(np.min(spectra.wave["b"]))
        wave_max = float(np.max(spectra.wave["z"]))
        spectra = resample_spectra_lin_or_log(
            spectra, linear_step=0.8,
            wave_min=wave_min, wave_max=wave_max, fast=True,
        )
        spectra = coadd_cameras(spectra)
        band = "brz" if "brz" in spectra.wave else list(spectra.wave.keys())[0]

    wave = spectra.wave[band].astype(np.float64)
    flux = spectra.flux[band].astype(np.float64)
    ivar = spectra.ivar[band].astype(np.float64)
    mask = spectra.mask[band].astype(bool)
    fibermap_tids = np.asarray(spectra.fibermap["TARGETID"])

    out = []
    for tid in target_ids:
        idx = np.where(fibermap_tids == tid)[0]
        if idx.size == 0:
            continue
        i = int(idx[0])
        out.append((tid, wave, flux[i], ivar[i], mask[i]))
    return out


def _to_noise_variance(ivar: np.ndarray) -> np.ndarray:
    """Convert ivar (1/var) to variance, with zero-ivar → NaN."""
    nv = np.full_like(ivar, np.nan)
    good = (ivar > 0) & np.isfinite(ivar)
    nv[good] = 1.0 / ivar[good]
    return nv


def _interpolate_to_rest_grid(wave_obs, flux, noise_variance, z_qso,
                              rest_grid: np.ndarray):
    """Interpolate (flux, noise_variance) onto a common rest-frame grid."""
    rest_wave = wave_obs / (1.0 + z_qso)
    valid = np.isfinite(wave_obs) & np.isfinite(flux) & np.isfinite(noise_variance)
    if valid.sum() < 50:
        return None, None
    f_interp = interp1d(rest_wave[valid], flux[valid], bounds_error=False,
                        fill_value=np.nan, kind="linear")
    nv_interp = interp1d(rest_wave[valid], noise_variance[valid], bounds_error=False,
                         fill_value=np.nan, kind="linear")
    return f_interp(rest_grid), nv_interp(rest_grid)


def _compute_redsnr(wave_obs, flux, noise_variance, z_qso,
                    rest_min=1425.0, rest_max=1475.0):
    """Median S/N in [1425, 1475] Å rest-frame (matches legacy redsnr column)."""
    rest = wave_obs / (1.0 + z_qso)
    region = (rest >= rest_min) & (rest <= rest_max)
    region &= np.isfinite(flux) & (noise_variance > 0)
    if region.sum() < 5:
        return 0.0
    snr = flux[region] / np.sqrt(noise_variance[region])
    return float(np.median(snr[np.isfinite(snr)])) if np.any(np.isfinite(snr)) else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mock-dir", required=True, type=Path,
                   help="Directory with zcat.fits + spectra-16/")
    p.add_argument("--output", required=True, type=Path,
                   help="Output HDF5 path")
    p.add_argument("--z-min", type=float, default=2.5)
    p.add_argument("--z-max", type=float, default=4.25)
    p.add_argument("--min-snr", type=float, default=0.0)
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--exclude-hcd", action="store_true",
                   help="Filter out TARGETIDs with HCDs in hcd_truth_cat.fits")
    p.add_argument("--exclude-bal", action="store_true",
                   help="Filter out TARGETIDs with BI_CIV>0 in bal_cat.fits")
    p.add_argument("--hcd-min-nhi", type=float, default=17.0)
    # Rest-frame grid (matches legacy slurm_train SLURM defaults).
    p.add_argument("--min-lambda", type=float, default=850.75)
    p.add_argument("--max-lambda", type=float, default=1420.75)
    p.add_argument("--dlambda", type=float, default=0.15)
    p.add_argument("--max-noise-variance", type=float, default=9.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nside", type=int, default=16,
                   help="DESI healpix nside (default 16)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    mock_dir: Path = args.mock_dir
    zcat_path = mock_dir / "zcat.fits"
    if not zcat_path.exists():
        sys.exit(f"[error] zcat.fits not found at {zcat_path}")

    # 1) Load + filter zcat.
    print(f"[step 1/5] reading {zcat_path}")
    zcat = Table.read(zcat_path)
    print(f"[step 1/5] zcat: {len(zcat)} rows")

    z_mask = (zcat["Z"] >= args.z_min) & (zcat["Z"] <= args.z_max)
    if "ZWARN" in zcat.colnames:
        z_mask &= (zcat["ZWARN"] == 0)
    keep = z_mask.copy()
    print(f"[step 1/5] z + ZWARN filter: {keep.sum()} kept of {len(keep)}")

    if args.exclude_hcd or args.exclude_bal:
        truth_keep = _build_targetid_filter(
            zcat, mock_dir, args.exclude_hcd, args.exclude_bal,
            hcd_min_nhi=args.hcd_min_nhi,
        )
        keep &= truth_keep
        print(f"[step 1/5] after HCD/BAL filter: {keep.sum()} kept")

    zcat = zcat[keep]
    if args.max_spectra is not None and len(zcat) > args.max_spectra:
        # Random subset (keep RNG-deterministic for reproducibility).
        idx = rng.choice(len(zcat), size=args.max_spectra, replace=False)
        idx.sort()
        zcat = zcat[idx]
        print(f"[step 1/5] capped at --max-spectra={args.max_spectra}")

    # 2) Compute healpix per row, group by healpix file.
    print("[step 2/5] grouping by healpix file")
    hpx = _healpix_for_radec(zcat["TARGET_RA"], zcat["TARGET_DEC"], nside=args.nside)
    by_hpx: dict[int, list[tuple[int, float]]] = {}
    for h, tid, z in zip(hpx, zcat["TARGETID"], zcat["Z"]):
        by_hpx.setdefault(int(h), []).append((int(tid), float(z)))
    print(f"[step 2/5] {len(by_hpx)} unique healpix files to read")

    # 3) Build the rest-frame grid.
    n_pix = int((args.max_lambda - args.min_lambda) / args.dlambda) + 1
    rest_grid = np.linspace(args.min_lambda, args.max_lambda, n_pix)
    print(f"[step 3/5] rest grid: {n_pix} pixels in [{args.min_lambda}, {args.max_lambda}] Å")

    # 4) Read each healpix file, process spectra in-memory.
    print("[step 4/5] reading + preprocessing spectra")
    out_tids: list[int] = []
    out_z: list[float] = []
    out_flux: list[np.ndarray] = []
    out_nv: list[np.ndarray] = []
    out_snr: list[float] = []
    skipped = 0
    t_start = time.time()

    for hp_idx, (healpix, target_pairs) in enumerate(sorted(by_hpx.items())):
        specfile = _spec_path(mock_dir, healpix)
        if not specfile.exists():
            skipped += len(target_pairs)
            if hp_idx % 100 == 0:
                print(f"[step 4/5] hpx {healpix}: file not found, skipping {len(target_pairs)} targets")
            continue
        target_ids = [tid for tid, _ in target_pairs]
        z_qsos_dict = {tid: z for tid, z in target_pairs}

        try:
            results = _read_one_healpix_file(specfile, target_ids)
        except Exception as e:
            print(f"[step 4/5] hpx {healpix}: read failed ({e}); skipping")
            skipped += len(target_pairs)
            continue

        for tid, wave, flux, ivar, mask_bool in results:
            z_qso = z_qsos_dict[tid]
            nv = _to_noise_variance(ivar)
            # Apply pipeline mask.
            flux_masked = np.where(mask_bool, np.nan, flux)
            nv_masked = np.where(mask_bool, np.nan, nv)
            # Apply max-noise-variance threshold.
            high_n = nv_masked > args.max_noise_variance
            flux_masked = np.where(high_n, np.nan, flux_masked)
            nv_masked = np.where(high_n, np.nan, nv_masked)

            # Interpolate to common rest grid.
            f_interp, nv_interp = _interpolate_to_rest_grid(
                wave, flux_masked, nv_masked, z_qso, rest_grid,
            )
            if f_interp is None:
                skipped += 1
                continue

            # Red-side SNR.
            snr = _compute_redsnr(wave, flux_masked, nv_masked, z_qso)

            out_tids.append(tid)
            out_z.append(z_qso)
            out_flux.append(f_interp.astype(np.float32))
            out_nv.append(nv_interp.astype(np.float32))
            out_snr.append(snr)

        if (hp_idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = len(out_tids) / max(elapsed, 1e-3)
            print(f"[step 4/5] processed {hp_idx + 1}/{len(by_hpx)} hpx files, "
                  f"{len(out_tids)} spectra ({rate:.1f}/s, skipped {skipped})")

    print(f"[step 4/5] done: {len(out_tids)} spectra, skipped {skipped}, "
          f"wall {(time.time() - t_start) / 60:.1f} min")

    if not out_tids:
        sys.exit("[error] no spectra processed")

    # 5) Write the legacy-schema HDF5.
    print(f"[step 5/5] writing {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Apply min_snr filter at write time.
    out_snr_arr = np.asarray(out_snr, dtype=np.float32)
    snr_keep = out_snr_arr >= args.min_snr
    n_keep = int(snr_keep.sum())
    print(f"[step 5/5] min_snr={args.min_snr}: {n_keep} of {len(out_tids)} kept")

    flux_arr = np.stack([f for f, k in zip(out_flux, snr_keep) if k]).astype(np.float32)
    nv_arr = np.stack([nv for nv, k in zip(out_nv, snr_keep) if k]).astype(np.float32)
    tids_arr = np.asarray([t for t, k in zip(out_tids, snr_keep) if k], dtype=np.int64)
    z_arr = np.asarray([z for z, k in zip(out_z, snr_keep) if k], dtype=np.float32)
    snr_kept = out_snr_arr[snr_keep]

    rest_wavelengths_per_spec = np.tile(rest_grid.astype(np.float32), (n_keep, 1))

    with h5py.File(args.output, "w") as f:
        # Use the LEGACY schema (older keys); load_preprocessed_h5 reads both.
        f.create_dataset("tids", data=tids_arr)
        f.create_dataset("rest_wavelengths", data=rest_wavelengths_per_spec, compression="gzip")
        f.create_dataset("fluxes", data=flux_arr, compression="gzip")
        f.create_dataset("noise_variance", data=nv_arr, compression="gzip")
        f.create_dataset("zqso", data=z_arr)
        f.create_dataset("redsnr", data=snr_kept)
        f.create_dataset("bluesnr", data=np.zeros_like(snr_kept))
        f.attrs["mock_dir"] = str(mock_dir)
        f.attrs["exclude_hcd"] = bool(args.exclude_hcd)
        f.attrs["exclude_bal"] = bool(args.exclude_bal)
        f.attrs["min_lambda"] = float(args.min_lambda)
        f.attrs["max_lambda"] = float(args.max_lambda)
        f.attrs["dlambda"] = float(args.dlambda)

    print(f"[step 5/5] wrote {args.output} ({n_keep} spectra × {n_pix} pixels)")

    # Companion README + JSON metadata for human / tooling consumption.
    # Sibling module — work regardless of whether the repo is pip-installed
    # (only `gpdla` env on GreatLakes has the editable install). Bare
    # `from _dataset_readme import …` works because sys.path[0] is the
    # script's directory under `python preload_spectra/preload_2lpt_simple.py`.
    from _dataset_readme import write_dataset_readme
    filter_pipeline = [
        f"z in [{args.z_min}, {args.z_max}] AND ZWARN==0 (if column exists)",
    ]
    if args.exclude_hcd:
        filter_pipeline.append(
            f"HCD anti-join: drop TARGETIDs with logNHI ≥ {args.hcd_min_nhi} "
            f"in mock's hcd_truth_cat.fits"
        )
    if args.exclude_bal:
        filter_pipeline.append(
            "BAL anti-join: drop TARGETIDs with BI_CIV > 0 in mock's bal_cat.fits"
        )
    if args.max_spectra is not None:
        filter_pipeline.append(
            f"Random subset to --max-spectra={args.max_spectra}"
        )
    suggested = (
        "python train_gp.py "
        f"--preloaded-file {args.output.name} "
        f"--z-min {args.z_min} --z-max {args.z_max} "
        f"--num-pca-components 30 "
        "--num-epochs 800 --batch-size 12500 --learning-rate 0.005 "
        "--num-forest-lines 3 "
        f"--output-dir <run_folder> --device cuda --save-every 25"
    )
    write_dataset_readme(
        args.output,
        dataset_kind="2lpt_mock",
        n_spectra=n_keep,
        n_pix=n_pix,
        rest_min=float(args.min_lambda),
        rest_max=float(args.max_lambda),
        dlambda=float(args.dlambda),
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        filter_pipeline=filter_pipeline,
        sources={
            "mock_dir": str(args.mock_dir),
        },
        cli_args={k: (str(v) if isinstance(v, Path) else v)
                  for k, v in vars(args).items()},
        suggested_train_command=suggested,
    )


if __name__ == "__main__":
    main()
