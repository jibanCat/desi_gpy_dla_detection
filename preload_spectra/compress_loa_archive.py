#!/usr/bin/env python
"""Compress raw LOA healpix coadds into a single LoaArchive HDF5.

Walks the LOA QSO catalog, groups TARGETIDs by HEALPIX, opens each
coadd via ``desispec.io.read_spectra`` and runs ``coadd_cameras``
(real-data fast path; falls back to ``resample_spectra_lin_or_log``
+ ``coadd_cameras`` if the bands aren't on a common grid). Writes the
result to a single concatenated HDF5 with offset-by-row indexing.

Usage:
    python preload_spectra/compress_loa_archive.py \\
        --qso-catalog /path/to/QSO_cat_loa_main_dark_healpix_v3-altbal.fits \\
        --loa-root    /path/to/loa \\
        --output      /path/to/loa_archive.h5 \\
        [--max-spectra 5000]    \\
        [--healpix-list 10902,10908,10952]    \\
        [--z-min 2.0] [--z-max 5.0]    \\
        [--with-resolution]    \\
        [--limit-fibers-per-coadd 50]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import fitsio
import numpy as np

# Make the repo importable when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpy_dla_detection.loa_archive import CoaddRecord, write_archive

LOG = logging.getLogger("compress_loa_archive")


def _setup_log(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)5s] %(message)s",
    )


def healpix_path(loa_root: Path, healpix: int, survey: str = "main",
                 program: str = "dark") -> Path:
    group = healpix // 100
    return (loa_root / "healpix" / survey / program / f"{group}" / f"{healpix}"
            / f"coadd-{survey}-{program}-{healpix}.fits")


def _read_filtered_catalog(qso_catalog: Path, *, z_min: float, z_max: float,
                           spectype: str | None, healpix_list: list[int] | None,
                           max_spectra: int | None) -> np.ndarray:
    """Read just the columns we need, then apply z + spectype + healpix cuts."""
    cols = ["TARGETID", "Z", "TARGET_RA", "TARGET_DEC", "HPXPIXEL",
            "ZWARN", "SPECTYPE", "SURVEY", "PROGRAM"]
    optional = ["BLUE_SNR", "RED_SNR", "SNR_REDSIDE", "SNR_FOREST"]
    with fitsio.FITS(qso_catalog) as f:
        all_cols = f[1].get_colnames()
    for c in optional:
        if c in all_cols:
            cols.append(c)
    LOG.info(f"Reading catalog columns: {cols}")
    cat = fitsio.read(qso_catalog, columns=cols)

    n0 = len(cat)
    keep = (cat["Z"] >= z_min) & (cat["Z"] <= z_max)
    if spectype is not None:
        spc = np.char.strip(cat["SPECTYPE"].astype(str))
        keep &= spc == spectype
    if healpix_list is not None:
        keep &= np.isin(cat["HPXPIXEL"], healpix_list)
    cat = cat[keep]
    LOG.info(f"Catalog: {n0} → {len(cat)} after z/spectype/healpix cuts")
    if max_spectra is not None and len(cat) > max_spectra:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(cat), size=max_spectra, replace=False)
        idx.sort()
        cat = cat[idx]
        LOG.info(f"Subsampled to {len(cat)} (--max-spectra)")
    return cat


def _extract_records_from_coadd(
    coadd_path: Path,
    cat_rows: np.ndarray,
    loa_root: Path,
    *,
    limit_fibers: int | None,
) -> Iterator[CoaddRecord]:
    """Run desispec read+coadd_cameras on one healpix; yield CoaddRecord rows."""
    import desispec.io
    from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log

    target_tids = list(cat_rows["TARGETID"])
    if limit_fibers is not None and len(target_tids) > limit_fibers:
        target_tids = target_tids[:limit_fibers]
        cat_rows = cat_rows[: limit_fibers]
    LOG.debug(f"  reading {len(target_tids)} fibers from {coadd_path.name}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = desispec.io.read_spectra(
            str(coadd_path),
            targetids=target_tids,
            skip_hdus=["EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG"],
        )
        try:
            spec = coadd_cameras(spec)
        except Exception as e:
            LOG.debug(f"  coadd_cameras direct failed ({e}); falling back to resample+coadd")
            spec = resample_spectra_lin_or_log(
                spec, linear_step=0.8,
                wave_min=spec.wave["b"].min(),
                wave_max=spec.wave["z"].max(),
                fast=True,
            )
            spec = coadd_cameras(spec)

    wave = np.asarray(spec.wave["brz"], dtype=np.float32)
    flux_all = spec.flux["brz"]
    ivar_all = spec.ivar["brz"]
    mask_all = spec.mask["brz"]
    res_all = spec.resolution_data["brz"]
    fmap_tids = np.asarray(spec.fibermap["TARGETID"])

    rel_path = str(coadd_path.relative_to(loa_root)) if coadd_path.is_relative_to(loa_root) else str(coadd_path)

    for cat_row in cat_rows:
        tid = int(cat_row["TARGETID"])
        idx_arr = np.where(fmap_tids == tid)[0]
        if len(idx_arr) == 0:
            LOG.warning(f"  TARGETID {tid} not in fibermap of {coadd_path.name}; skipping")
            continue
        idx = int(idx_arr[0])
        blue_snr = float(cat_row["BLUE_SNR"]) if "BLUE_SNR" in cat_row.dtype.names else np.nan
        red_snr = float(cat_row["RED_SNR"]) if "RED_SNR" in cat_row.dtype.names else (
            float(cat_row["SNR_REDSIDE"]) if "SNR_REDSIDE" in cat_row.dtype.names else np.nan
        )
        yield CoaddRecord(
            targetid=tid,
            z=float(cat_row["Z"]),
            ra=float(cat_row["TARGET_RA"]),
            dec=float(cat_row["TARGET_DEC"]),
            healpix=int(cat_row["HPXPIXEL"]),
            zwarn=int(cat_row["ZWARN"]),
            blue_snr=blue_snr,
            red_snr=red_snr,
            source_file=rel_path,
            fiber_idx=idx,
            flux=flux_all[idx],
            ivar=ivar_all[idx],
            mask=mask_all[idx],
            R=res_all[idx],
        ), wave  # also yield wave so caller can lock the grid on first use


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qso-catalog", required=True, type=Path)
    p.add_argument("--loa-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--healpix-list", default=None,
                   help="Comma-separated list of HEALPIX ids; default = all")
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--z-min", type=float, default=2.0)
    p.add_argument("--z-max", type=float, default=5.0)
    p.add_argument("--spectype", default="QSO")
    p.add_argument("--with-resolution", action="store_true",
                   help="Also store the full 11-band R matrix per QSO (~3× size)")
    p.add_argument("--limit-fibers-per-coadd", type=int, default=None,
                   help="Smoke-test mode: cap fibers per healpix")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    _setup_log(args.verbose)

    healpix_list = None
    if args.healpix_list:
        healpix_list = [int(s) for s in args.healpix_list.split(",")]

    cat = _read_filtered_catalog(
        args.qso_catalog,
        z_min=args.z_min, z_max=args.z_max,
        spectype=args.spectype,
        healpix_list=healpix_list,
        max_spectra=args.max_spectra,
    )

    # Group rows by (HEALPIX, SURVEY, PROGRAM)
    by_hp: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(cat):
        survey = row["SURVEY"].decode() if isinstance(row["SURVEY"], (bytes, np.bytes_)) else str(row["SURVEY"])
        program = row["PROGRAM"].decode() if isinstance(row["PROGRAM"], (bytes, np.bytes_)) else str(row["PROGRAM"])
        by_hp[(int(row["HPXPIXEL"]), survey.strip(), program.strip())].append(i)
    LOG.info(f"Grouped {len(cat)} QSOs into {len(by_hp)} healpixes")

    # Stream records — first pass to get the wavelength grid, then drive write_archive
    locked_wave: np.ndarray | None = None

    def record_stream() -> Iterator[CoaddRecord]:
        nonlocal locked_wave
        n_done = 0
        for (hp, survey, program), row_idx_list in sorted(by_hp.items()):
            cat_subset = cat[row_idx_list]
            cpath = healpix_path(args.loa_root, hp, survey, program)
            if not cpath.exists():
                LOG.warning(f"healpix coadd missing: {cpath}; skipping {len(row_idx_list)} QSOs")
                continue
            try:
                for rec, wave in _extract_records_from_coadd(
                    cpath, cat_subset, args.loa_root,
                    limit_fibers=args.limit_fibers_per_coadd,
                ):
                    if locked_wave is None:
                        locked_wave = wave
                    elif wave.shape != locked_wave.shape or not np.allclose(wave, locked_wave):
                        LOG.warning(
                            f"healpix {hp} has wavelength grid that differs from locked grid "
                            f"({wave.shape} vs {locked_wave.shape}); skipping")
                        break
                    yield rec
                    n_done += 1
                    if n_done % 200 == 0:
                        LOG.info(f"  wrote {n_done} QSOs...")
            except Exception as e:
                LOG.error(f"failed to process {cpath}: {e}")
                continue

    # Walk one healpix first to lock wave, then stream the rest
    first_hp_key = next(iter(sorted(by_hp.keys())))
    first_cat = cat[by_hp[first_hp_key]]
    first_cpath = healpix_path(args.loa_root, *first_hp_key)
    for _rec, wave in _extract_records_from_coadd(first_cpath, first_cat[:1], args.loa_root,
                                                  limit_fibers=1):
        locked_wave = wave
        break
    if locked_wave is None:
        LOG.error("could not lock wavelength grid from first healpix")
        return 2

    summary = write_archive(
        args.output,
        record_stream(),
        wavelength=locked_wave,
        source_root=str(args.loa_root),
        with_resolution=args.with_resolution,
    )
    LOG.info(f"DONE: {summary['n_qsos']} QSOs → {summary['out_path']} "
             f"({summary['size_bytes']/1e9:.2f} GB, n_pix={summary['n_pix']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
