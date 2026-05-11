"""Adapter: LoaArchive → v2 trainset.h5 (rest-frame, masked, normalized-ready).

Converts a compressed LOA archive (`gpy_dla_detection.loa_archive.LoaArchive`,
observed-frame flux/ivar) to the v2 preload schema that
`gpy_dla_detection.training.dataset.load_preprocessed_h5` reads.

This is the analog of `preload_spectra/preload_loa_real.py` but reading
from a pre-built LoaArchive instead of raw DESI FITS files. Use this for
Step C training on real LOA when the archive is in place.

Filter pipeline (in order):
  1. catalog z-range filter (`z_min ≤ Z ≤ z_max`)
  2. ZWARN == 0 (only if ZWARN column present in archive)
  3. exclude_targetids set (BAL anti-join, HCD anti-join — built externally)
  4. cap to max_spectra (random subset, seeded)

For each surviving QSO:
  - Read flux, ivar from archive (observed frame, shared wavelength grid)
  - Convert to rest frame: λ_rest = λ_obs / (1 + z)
  - Interpolate to common rest grid [rest_min, rest_max] at dλ=rest_dlambda
  - Mask invalid pixels (mask != 0 OR ivar <= 0)
  - Output: flux, noise_variance per QSO at rest grid

OUTPUT SCHEMA (matches preload_loa_real.py legacy keys):
  tids                       (n_qso,)        int64
  rest_wavelengths           (n_qso, n_pix)  float32  (per-spectrum, all same)
  fluxes                     (n_qso, n_pix)  float32  NaN at invalid pixels
  noise_variance             (n_qso, n_pix)  float32  inf at invalid pixels
  zqso                       (n_qso,)        float32
  redsnr                     (n_qso,)        float32
  bluesnr                    (n_qso,)        float32
  attrs:
    rest_min, rest_max, rest_dlambda, source_archive, n_filtered, n_kept

The output is consumable by `load_preprocessed_h5` directly (legacy schema
branch), so the rest of the Step C pipeline (`tests/phase2_train_desi.py`)
needs no further changes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d

# Make the repo importable when invoked as a top-level script (e.g. SLURM).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpy_dla_detection.loa_archive import LoaArchive

LYA_AA = 1215.6701  # Å rest, Lyα — used only for SNR computation


def _interp_to_rest(wave_obs: np.ndarray, flux: np.ndarray,
                    noise_variance: np.ndarray, z_qso: float,
                    rest_grid: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Resample (flux, nv) from obs wavelengths to a shared rest grid.

    Returns (flux_rest, nv_rest) on `rest_grid`. Pixels outside the
    spectrum's rest coverage are NaN (flux) / inf (nv).
    """
    rest_wave = wave_obs / (1.0 + z_qso)
    valid = np.isfinite(flux) & np.isfinite(noise_variance) & (noise_variance > 0)
    if valid.sum() < 2:
        return (np.full_like(rest_grid, np.nan, dtype=np.float32),
                np.full_like(rest_grid, np.inf, dtype=np.float32))
    f_interp = interp1d(rest_wave[valid], flux[valid],
                        bounds_error=False, fill_value=np.nan)
    nv_interp = interp1d(rest_wave[valid], noise_variance[valid],
                         bounds_error=False, fill_value=np.inf)
    return (f_interp(rest_grid).astype(np.float32),
            nv_interp(rest_grid).astype(np.float32))


def _compute_redsnr(wave_obs: np.ndarray, flux: np.ndarray,
                    noise_variance: np.ndarray, z_qso: float,
                    *, lambda_min_rest: float = 1268.0,
                    lambda_max_rest: float = 1380.0) -> float:
    """Median-SNR in the red-side rest-frame window. Matches the
    convention in preload_loa_real.py for the `redsnr` field.
    """
    rest_wave = wave_obs / (1.0 + z_qso)
    band = (rest_wave >= lambda_min_rest) & (rest_wave <= lambda_max_rest)
    band &= np.isfinite(flux) & np.isfinite(noise_variance) & (noise_variance > 0)
    if band.sum() < 5:
        return float("nan")
    snr = flux[band] / np.sqrt(noise_variance[band])
    return float(np.nanmedian(snr))


def _compute_bluesnr(wave_obs: np.ndarray, flux: np.ndarray,
                     noise_variance: np.ndarray, z_qso: float,
                     *, lambda_min_rest: float = 1041.0,
                     lambda_max_rest: float = 1185.0) -> float:
    """Median-SNR in the Lyα forest rest-frame window."""
    rest_wave = wave_obs / (1.0 + z_qso)
    band = (rest_wave >= lambda_min_rest) & (rest_wave <= lambda_max_rest)
    band &= np.isfinite(flux) & np.isfinite(noise_variance) & (noise_variance > 0)
    if band.sum() < 5:
        return float("nan")
    snr = flux[band] / np.sqrt(noise_variance[band])
    return float(np.nanmedian(snr))


def loa_archive_to_trainset(
    archive_path: str | Path,
    output_path: str | Path,
    *,
    z_min: float = 2.15,
    z_max: float = 4.25,
    exclude_targetids: set[int] | None = None,
    rest_min: float = 850.75,
    rest_max: float = 1700.0,
    rest_dlambda: float = 0.15,
    max_spectra: int | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Build a v2 trainset.h5 from a LoaArchive.

    Parameters
    ----------
    archive_path : path
        LoaArchive HDF5 (output of `gpy_dla_detection.loa_archive.write_archive`).
    output_path : path
        trainset.h5 to write (legacy v2 schema).
    z_min, z_max : float
        QSO redshift filter applied to the archive's catalog Z column.
    exclude_targetids : set[int], optional
        TARGETIDs to drop. Build this externally from BAL and HCD catalogs
        (e.g. ``set(bal_cat["TARGETID"][bal_cat["BI_CIV"] > 0])``).
    rest_min, rest_max, rest_dlambda : float
        Rest-frame grid. Default (850.75 — 1700.0 at 0.15 Å) matches the
        2lpt v2 preload convention; trains on the same grid as the v2
        preload trainset.h5 files.
    max_spectra : int, optional
        Cap. If exceeded, draw a uniform random subset (seeded).
    seed : int
        RNG seed for the random cap.

    Returns
    -------
    dict
        Summary stats: n_archive, n_after_z, n_after_zwarn, n_after_exclude,
        n_kept, n_pix, output_path, elapsed_s.
    """
    archive_path = Path(archive_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rest_grid = np.arange(rest_min, rest_max + rest_dlambda * 0.5,
                          rest_dlambda, dtype=np.float64)
    n_pix = int(rest_grid.shape[0])
    if verbose:
        print(f"[loa_adapter] rest grid: {n_pix} pix in [{rest_min}, {rest_max}] Å, dλ={rest_dlambda}")

    with LoaArchive(archive_path) as ar:
        cat = ar.catalog()
        wave_obs = np.asarray(ar.wavelength, dtype=np.float64)
        n_archive = len(cat)
        if verbose:
            print(f"[loa_adapter] archive {archive_path.name}: {n_archive} QSOs, "
                  f"obs grid [{wave_obs[0]:.1f}, {wave_obs[-1]:.1f}] Å")

        # 1. z filter
        z_arr = np.asarray(cat["Z"], dtype=np.float64)
        keep = (z_arr >= z_min) & (z_arr <= z_max)
        n_after_z = int(keep.sum())
        if verbose:
            print(f"[loa_adapter] z-filter [{z_min}, {z_max}]: {n_after_z}/{n_archive} kept")

        # 2. ZWARN == 0
        zwarn = np.asarray(cat["ZWARN"], dtype=np.int64)
        keep &= (zwarn == 0)
        n_after_zwarn = int(keep.sum())
        if verbose:
            print(f"[loa_adapter] ZWARN==0 filter: {n_after_zwarn}/{n_archive} kept")

        # 3. exclude set
        if exclude_targetids:
            tids = np.asarray(cat["TARGETID"], dtype=np.int64)
            excl = np.isin(tids, np.fromiter(exclude_targetids, dtype=np.int64))
            keep &= ~excl
        n_after_exclude = int(keep.sum())
        if verbose:
            print(f"[loa_adapter] exclude-set ({len(exclude_targetids or [])} TIDs): "
                  f"{n_after_exclude}/{n_archive} kept")

        # 4. cap
        kept_idx = np.where(keep)[0]
        if max_spectra is not None and len(kept_idx) > max_spectra:
            rng = np.random.default_rng(seed)
            kept_idx = rng.choice(kept_idx, size=max_spectra, replace=False)
            kept_idx.sort()
        n_kept = int(len(kept_idx))
        if verbose:
            print(f"[loa_adapter] final cap: {n_kept} QSOs")

        # 5. Loop & resample.
        tids_out = np.empty(n_kept, dtype=np.int64)
        zqso_out = np.empty(n_kept, dtype=np.float32)
        redsnr_out = np.empty(n_kept, dtype=np.float32)
        bluesnr_out = np.empty(n_kept, dtype=np.float32)
        flux_out = np.empty((n_kept, n_pix), dtype=np.float32)
        nv_out = np.empty((n_kept, n_pix), dtype=np.float32)

        # Read flux/ivar/mask in one bulk slice for speed (avoids per-QSO HDF5 IO).
        flux_block = ar._h["flux"][kept_idx]      # (n_kept, n_obs_pix) f4
        ivar_block = ar._h["ivar"][kept_idx]      # (n_kept, n_obs_pix) f4
        mask_block = ar._h["mask"][kept_idx]      # (n_kept, n_obs_pix) u4

        for i, src_idx in enumerate(kept_idx):
            row = cat[src_idx]
            z = float(row["Z"])
            f_obs = flux_block[i].astype(np.float64)
            iv = ivar_block[i].astype(np.float64)
            m = mask_block[i]
            # Apply DESI mask: any nonzero → invalid
            bad = (m != 0) | (iv <= 0) | ~np.isfinite(iv) | ~np.isfinite(f_obs)
            f_obs[bad] = np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                nv = np.where(bad, np.inf, 1.0 / iv)

            f_rest, nv_rest = _interp_to_rest(wave_obs, f_obs, nv, z, rest_grid)
            tids_out[i] = int(row["TARGETID"])
            zqso_out[i] = np.float32(z)
            redsnr_out[i] = np.float32(_compute_redsnr(wave_obs, f_obs, nv, z))
            bluesnr_out[i] = np.float32(_compute_bluesnr(wave_obs, f_obs, nv, z))
            flux_out[i] = f_rest
            nv_out[i] = nv_rest

    # 6. Write v2 trainset.h5 (legacy schema — load_preprocessed_h5 reads it)
    rest_wave_2d = np.tile(rest_grid.astype(np.float32)[None, :], (n_kept, 1))
    with h5py.File(output_path, "w") as h:
        h.create_dataset("tids", data=tids_out)
        h.create_dataset("rest_wavelengths", data=rest_wave_2d)
        h.create_dataset("fluxes", data=flux_out)
        h.create_dataset("noise_variance", data=nv_out)
        h.create_dataset("zqso", data=zqso_out)
        h.create_dataset("redsnr", data=redsnr_out)
        h.create_dataset("bluesnr", data=bluesnr_out)
        h.attrs["rest_min"] = float(rest_min)
        h.attrs["rest_max"] = float(rest_max)
        h.attrs["rest_dlambda"] = float(rest_dlambda)
        h.attrs["source_archive"] = str(archive_path)
        h.attrs["n_archive"] = int(n_archive)
        h.attrs["n_after_z"] = int(n_after_z)
        h.attrs["n_after_zwarn"] = int(n_after_zwarn)
        h.attrs["n_after_exclude"] = int(n_after_exclude)
        h.attrs["n_kept"] = int(n_kept)
        h.attrs["z_min"] = float(z_min)
        h.attrs["z_max"] = float(z_max)
        if exclude_targetids:
            h.attrs["n_exclude_set"] = int(len(exclude_targetids))

    elapsed = time.time() - t0
    summary = dict(
        n_archive=n_archive, n_after_z=n_after_z, n_after_zwarn=n_after_zwarn,
        n_after_exclude=n_after_exclude, n_kept=n_kept, n_pix=n_pix,
        output_path=str(output_path), elapsed_s=elapsed,
    )
    if verbose:
        print(f"[loa_adapter] wrote {output_path} ({n_kept} × {n_pix}) in {elapsed:.1f}s")
    return summary


def _load_excludes_from_fits(path: Path | None, *, tid_col: str = "TARGETID",
                             bal_col: str | None = None,
                             bal_min: float = 0.0,
                             nhi_col: str | None = None,
                             nhi_min: float | None = None) -> set[int]:
    """Helper: build exclusion set from a FITS catalog.

    Filtering modes:
      - `bal_col` + `bal_min` → exclude rows with `bal_col > bal_min`.
      - `nhi_col` + `nhi_min` → exclude rows with `nhi_col >= nhi_min`
        (HCD anti-join with NHI threshold; matches preload_loa_real.py
        `--hcd-min-nhi` behaviour).
      - neither → exclude all rows in the catalog.
    """
    if path is None:
        return set()
    from astropy.table import Table
    tbl = Table.read(path)
    if bal_col is not None and nhi_col is not None:
        raise ValueError("pass at most one of bal_col / nhi_col")
    if bal_col is not None:
        mask = np.asarray(tbl[bal_col]) > bal_min
    elif nhi_col is not None and nhi_min is not None:
        mask = np.asarray(tbl[nhi_col]) >= nhi_min
    else:
        mask = np.ones(len(tbl), dtype=bool)
    return set(int(t) for t in np.asarray(tbl[tid_col])[mask])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--archive", type=Path, required=True,
                   help="LoaArchive HDF5 input")
    p.add_argument("--out", type=Path, required=True,
                   help="trainset.h5 output (v2 legacy schema)")
    p.add_argument("--z-min", type=float, default=2.15)
    p.add_argument("--z-max", type=float, default=4.25)
    p.add_argument("--rest-min", type=float, default=850.75)
    p.add_argument("--rest-max", type=float, default=1700.0)
    p.add_argument("--rest-dlambda", type=float, default=0.15)
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bal-cat", type=Path, default=None,
                   help="FITS catalog with BAL TARGETIDs to exclude. Pass "
                        "with --bal-col + --bal-min to filter rows.")
    p.add_argument("--bal-col", default="BI_CIV")
    p.add_argument("--bal-min", type=float, default=0.0)
    p.add_argument("--hcd-cat", type=Path, default=None,
                   help="FITS or HDF5 catalog with HCD TARGETIDs to exclude.")
    p.add_argument("--hcd-tid-col", default="TARGETID")
    p.add_argument("--hcd-nhi-col", default="NHI",
                   help="Column for the HCD NHI threshold filter (default 'NHI'). "
                        "Set --hcd-min-nhi to enable threshold filtering.")
    p.add_argument("--hcd-min-nhi", type=float, default=None,
                   help="If set, only exclude HCD rows with NHI ≥ this value. "
                        "Match preload_loa_real.py convention: 20.3 = DLA-only, "
                        "17.2 = DLAs + sub-DLAs / LLS.")
    args = p.parse_args()

    excludes: set[int] = set()
    if args.bal_cat:
        bal_excl = _load_excludes_from_fits(args.bal_cat, tid_col="TARGETID",
                                            bal_col=args.bal_col,
                                            bal_min=args.bal_min)
        print(f"[main] BAL exclude: {len(bal_excl)} TIDs")
        excludes |= bal_excl
    if args.hcd_cat:
        hcd_excl = _load_excludes_from_fits(
            args.hcd_cat,
            tid_col=args.hcd_tid_col,
            nhi_col=args.hcd_nhi_col if args.hcd_min_nhi is not None else None,
            nhi_min=args.hcd_min_nhi,
        )
        nhi_str = f" (NHI ≥ {args.hcd_min_nhi})" if args.hcd_min_nhi is not None else " (all rows)"
        print(f"[main] HCD exclude{nhi_str}: {len(hcd_excl)} TIDs")
        excludes |= hcd_excl
    print(f"[main] total exclude set: {len(excludes)} TIDs")

    summary = loa_archive_to_trainset(
        args.archive, args.out,
        z_min=args.z_min, z_max=args.z_max,
        exclude_targetids=excludes,
        rest_min=args.rest_min, rest_max=args.rest_max,
        rest_dlambda=args.rest_dlambda,
        max_spectra=args.max_spectra, seed=args.seed,
    )
    print(f"\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
