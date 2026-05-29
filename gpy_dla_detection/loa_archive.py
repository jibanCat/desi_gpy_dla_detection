"""Compressed LOA archive — reader + writer.

Stores DESI healpix coadds as a single concatenated HDF5 with
offset+length indexing, so inference can slice any QSO without
opening a separate FITS file.

Storage schema (HDF5):

    attrs:
        schema_version    int    = 1
        wave_min          f4     observed Å, shared across all QSOs
        wave_max          f4
        wave_step         f4     0.8 for DESI brz
        n_pix             i4     length of the shared wavelength grid
        source_root       str    e.g. "/nfs/turbo/.../loa"
        produced_utc      str    ISO timestamp

    wavelength            (n_pix,)            float32  observed Å, shared
    catalog/              compound dataset    one row per QSO:
        TARGETID            i8
        Z                   f4
        RA                  f8
        DEC                 f8
        HEALPIX             i4
        ZWARN               i4
        BLUE_SNR            f4   may be NaN if absent in source catalog
        RED_SNR             f4
        SOURCE_FILE         S128 relative to source_root
        FIBER_IDX           i4   row in the source coadd
    flux                  (n_qso, n_pix)      float32  post coadd_cameras
    ivar                  (n_qso, n_pix)      float32
    mask                  (n_qso, n_pix)      uint32   DESI mask bits
    fwhm_pix              (n_qso, n_pix)      float32  per-pixel LSF FWHM in
                                                       pixel units (multiply
                                                       by ``wave_step`` for Å)
    resolution            (n_qso, 11, n_pix)  float32  optional, only when the
                                                       writer ran with
                                                       ``with_resolution=True``

The FWHM is derived from the post-resample R matrix as the second
moment of the 11-element banded kernel at each pixel; this matches what
``voigt_v2`` consumes as a Gaussian LSF (see ``voigt_v2._kernel_for``).
By default the full 11-band R is NOT stored — for 300k QSOs it adds
~100 GB while voigt_v2 only consumes the FWHM today. Pass
``with_resolution=True`` if you want the full R for a future pipeline
that does per-pixel R convolution.

Numerical precision
-------------------
``flux`` / ``ivar`` / ``resolution`` are stored at float32. ``desispec``
returns float64; the archive cast quantizes at ~1e-7 relative, well
below DESI's per-pixel noise floor (~0.5–1% in the forest). Inference
posteriors agree with the raw-FITS path to 4+ significant figures
(verified: tests/test_loa_archive.py + smoke run on 3 healpixes,
2026-05-01). If you ever need bit-exact f64, change the dtype on
the relevant create_dataset calls.

This module exposes:
    write_archive(...) — append-style writer
    LoaArchive       — read-side class with TARGETID lookup
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# FWHM extraction from DESI's banded R
# ---------------------------------------------------------------------------

def fwhm_pixels_from_resolution(R: np.ndarray) -> np.ndarray:
    """Convert DESI's 11-band R matrix to per-pixel FWHM in PIXEL units.

    Parameters
    ----------
    R : (11, n_pix) float — the banded resolution at one fiber, post coadd_cameras.
        Each column is the 11-element LSF kernel for that output pixel.

    Returns
    -------
    fwhm_pix : (n_pix,) float32 — per-pixel FWHM in pixel units. Multiply by
        the wavelength step (0.8 Å for DESI brz) to get FWHM in Å. Pixels
        with a degenerate kernel (zero norm or non-finite values) get NaN.
    """
    if R.ndim != 2 or R.shape[0] != 11:
        raise ValueError(f"expected R shape (11, n_pix), got {R.shape}")
    offsets = np.arange(-5, 6, dtype=np.float64)
    norm = R.sum(axis=0)
    bad = ~np.isfinite(norm) | (norm <= 0)
    safe = np.where(bad, 1.0, norm)
    Rn = R / safe[None, :]
    mu = (offsets[:, None] * Rn).sum(axis=0)
    var = (Rn * (offsets[:, None] - mu[None, :]) ** 2).sum(axis=0)
    var = np.maximum(var, 0.0)
    fwhm = 2.3548200450309493 * np.sqrt(var)
    fwhm = np.where(bad, np.nan, fwhm)
    return fwhm.astype(np.float32)


# ---------------------------------------------------------------------------
# Compound catalog dtype
# ---------------------------------------------------------------------------

CATALOG_DTYPE = np.dtype([
    ("TARGETID", "<i8"),
    ("Z", "<f4"),
    ("RA", "<f8"),
    ("DEC", "<f8"),
    ("HEALPIX", "<i4"),
    ("ZWARN", "<i4"),
    ("BLUE_SNR", "<f4"),
    ("RED_SNR", "<f4"),
    ("SOURCE_FILE", "S128"),
    ("FIBER_IDX", "<i4"),
])


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

@dataclass
class CoaddRecord:
    """One QSO's data, post coadd_cameras, ready to append to an archive."""
    targetid: int
    z: float
    ra: float
    dec: float
    healpix: int
    zwarn: int
    blue_snr: float
    red_snr: float
    source_file: str
    fiber_idx: int
    flux: np.ndarray       # (n_pix,) float
    ivar: np.ndarray       # (n_pix,) float
    mask: np.ndarray       # (n_pix,) int
    R: np.ndarray          # (11, n_pix) float — used to derive FWHM


def write_archive(
    out_path: str | Path,
    records: Iterable[CoaddRecord],
    *,
    wavelength: np.ndarray,
    source_root: str,
    with_resolution: bool = False,
    chunk_qsos: int = 256,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> dict:
    """Write a streaming archive from an iterable of CoaddRecord.

    Builds extendable HDF5 datasets and appends in chunks to keep peak
    memory low; suitable for streaming over many healpixes.

    Returns a dict with summary stats (n_qsos, total_bytes, ...).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wavelength = np.asarray(wavelength, dtype=np.float32)
    n_pix = wavelength.shape[0]
    if wavelength.ndim != 1 or n_pix < 2:
        raise ValueError("wavelength must be 1D with at least 2 entries")
    dlam = float(np.diff(wavelength).mean())

    # HDF5 chunk shape for the 2D datasets (flux/ivar/mask/resolution).
    # Use `chunk_qsos` rows per chunk so bulk reads of a healpix (~256 QSOs)
    # touch ~1 chunk, and gzip compresses across the whole chunk.
    # Lower-clamped to 1 to defend against `chunk_qsos=0`. The original
    # `min(chunk_qsos, 1)` was a typo (always 1 → 1-row chunks regardless
    # of input) and shipped in commit d4799c1; existing archives written
    # under that bug still read correctly via HDF5's transparent chunking,
    # just with ~256× more chunk fetches + worse gzip ratio.
    chunks_2d = (max(chunk_qsos, 1), n_pix)

    with h5py.File(out_path, "w") as h:
        h.attrs["schema_version"] = SCHEMA_VERSION
        h.attrs["wave_min"] = float(wavelength[0])
        h.attrs["wave_max"] = float(wavelength[-1])
        h.attrs["wave_step"] = dlam
        h.attrs["n_pix"] = n_pix
        h.attrs["source_root"] = source_root
        h.attrs["produced_utc"] = datetime.now(timezone.utc).isoformat()

        h.create_dataset("wavelength", data=wavelength)

        cat_d = h.create_dataset(
            "catalog", shape=(0,), maxshape=(None,), dtype=CATALOG_DTYPE,
            chunks=(min(chunk_qsos * 4, 4096),),
        )
        flux_d = h.create_dataset(
            "flux", shape=(0, n_pix), maxshape=(None, n_pix),
            dtype="f4", chunks=chunks_2d,
            compression=compression, compression_opts=compression_opts,
        )
        ivar_d = h.create_dataset(
            "ivar", shape=(0, n_pix), maxshape=(None, n_pix),
            dtype="f4", chunks=chunks_2d,
            compression=compression, compression_opts=compression_opts,
        )
        mask_d = h.create_dataset(
            "mask", shape=(0, n_pix), maxshape=(None, n_pix),
            dtype="u4", chunks=chunks_2d,
            compression=compression, compression_opts=compression_opts,
        )
        fwhm_d = h.create_dataset(
            "fwhm_pix", shape=(0, n_pix), maxshape=(None, n_pix),
            dtype="f4", chunks=chunks_2d,
            compression=compression, compression_opts=compression_opts,
        )
        if with_resolution:
            res_d = h.create_dataset(
                "resolution", shape=(0, 11, n_pix), maxshape=(None, 11, n_pix),
                dtype="f4", chunks=(min(chunk_qsos, 1), 11, n_pix),
                compression=compression, compression_opts=compression_opts,
            )

        # Buffer chunks
        buf_cat: list = []
        buf_flux: list = []
        buf_ivar: list = []
        buf_mask: list = []
        buf_fwhm: list = []
        buf_res: list = []
        n_total = 0

        def _flush() -> None:
            nonlocal n_total
            if not buf_cat:
                return
            n_new = len(buf_cat)
            old = cat_d.shape[0]
            cat_d.resize((old + n_new,))
            cat_d[old:] = np.array(buf_cat, dtype=CATALOG_DTYPE)
            for d, b in [(flux_d, buf_flux), (ivar_d, buf_ivar),
                         (mask_d, buf_mask), (fwhm_d, buf_fwhm)]:
                d.resize((old + n_new, n_pix))
                d[old:] = np.stack(b, axis=0)
            if with_resolution:
                res_d.resize((old + n_new, 11, n_pix))
                res_d[old:] = np.stack(buf_res, axis=0)
            buf_cat.clear(); buf_flux.clear(); buf_ivar.clear()
            buf_mask.clear(); buf_fwhm.clear(); buf_res.clear()
            n_total += n_new

        for r in records:
            if r.flux.shape != (n_pix,):
                raise ValueError(
                    f"flux for TARGETID {r.targetid} has shape {r.flux.shape}, "
                    f"expected ({n_pix},)")
            if r.R.shape != (11, n_pix):
                raise ValueError(
                    f"R for TARGETID {r.targetid} has shape {r.R.shape}, "
                    f"expected (11, {n_pix})")

            buf_cat.append((
                np.int64(r.targetid),
                np.float32(r.z),
                np.float64(r.ra),
                np.float64(r.dec),
                np.int32(r.healpix),
                np.int32(r.zwarn),
                np.float32(r.blue_snr),
                np.float32(r.red_snr),
                np.bytes_(r.source_file)[:128],
                np.int32(r.fiber_idx),
            ))
            buf_flux.append(r.flux.astype(np.float32, copy=False))
            buf_ivar.append(r.ivar.astype(np.float32, copy=False))
            buf_mask.append(r.mask.astype(np.uint32, copy=False))
            buf_fwhm.append(fwhm_pixels_from_resolution(r.R))
            if with_resolution:
                buf_res.append(r.R.astype(np.float32, copy=False))

            if len(buf_cat) >= chunk_qsos:
                _flush()
        _flush()

    return {
        "out_path": str(out_path),
        "n_qsos": n_total,
        "n_pix": n_pix,
        "size_bytes": os.path.getsize(out_path),
    }


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class LoaArchive:
    """Random-access reader for a compressed LOA archive.

    Usage:
        with LoaArchive("loa_archive.h5") as ar:
            spec = ar.get_spectrum(targetid)
            spec.flux, spec.ivar, spec.wavelength, spec.fwhm_angstrom
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._h: h5py.File | None = None
        self._tid_to_idx: dict[int, int] | None = None

    def __enter__(self) -> "LoaArchive":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._h is None:
            self._h = h5py.File(self.path, "r")
            cat = self._h["catalog"][:]
            self._tid_to_idx = {int(t): int(i) for i, t in enumerate(cat["TARGETID"])}

    def close(self) -> None:
        if self._h is not None:
            self._h.close()
            self._h = None
            self._tid_to_idx = None

    @property
    def wavelength(self) -> np.ndarray:
        assert self._h is not None
        return self._h["wavelength"][:]

    @property
    def wave_step(self) -> float:
        assert self._h is not None
        return float(self._h.attrs["wave_step"])

    @property
    def n_qsos(self) -> int:
        assert self._h is not None
        return int(self._h["catalog"].shape[0])

    def catalog(self) -> np.ndarray:
        assert self._h is not None
        return self._h["catalog"][:]

    @property
    def has_resolution(self) -> bool:
        assert self._h is not None
        return "resolution" in self._h

    def get_spectrum(self, targetid: int, *, with_resolution: bool = False) -> "Spectrum":
        if self._h is None or self._tid_to_idx is None:
            raise RuntimeError("LoaArchive not opened")
        if int(targetid) not in self._tid_to_idx:
            raise KeyError(f"TARGETID {targetid} not in archive")
        idx = self._tid_to_idx[int(targetid)]
        cat = self._h["catalog"][idx]
        wave = self._h["wavelength"][:]
        R = None
        if with_resolution:
            if "resolution" not in self._h:
                raise KeyError(
                    "this archive was written without --with-resolution; "
                    "no per-pixel R available")
            R = self._h["resolution"][idx]
        return Spectrum(
            targetid=int(cat["TARGETID"]),
            z=float(cat["Z"]),
            ra=float(cat["RA"]),
            dec=float(cat["DEC"]),
            healpix=int(cat["HEALPIX"]),
            zwarn=int(cat["ZWARN"]),
            blue_snr=float(cat["BLUE_SNR"]),
            red_snr=float(cat["RED_SNR"]),
            wavelength=wave,
            flux=self._h["flux"][idx],
            ivar=self._h["ivar"][idx],
            mask=self._h["mask"][idx],
            fwhm_pix=self._h["fwhm_pix"][idx],
            resolution=R,
            wave_step=self.wave_step,
        )


@dataclass
class Spectrum:
    targetid: int
    z: float
    ra: float
    dec: float
    healpix: int
    zwarn: int
    blue_snr: float
    red_snr: float
    wavelength: np.ndarray
    flux: np.ndarray
    ivar: np.ndarray
    mask: np.ndarray
    fwhm_pix: np.ndarray
    wave_step: float
    resolution: np.ndarray | None = None

    @property
    def fwhm_angstrom(self) -> np.ndarray:
        return self.fwhm_pix * self.wave_step

    @property
    def noise_variance(self) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.where(self.ivar > 0, 1.0 / self.ivar, np.inf)
