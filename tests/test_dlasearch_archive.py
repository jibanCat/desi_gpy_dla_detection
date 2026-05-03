"""Unit tests for the --archive PATH wiring in dlasearch.py.

Validates that ``dlasearch._load_group_spectra(coaddpath, catalog,
archive=...)`` correctly bypasses ``desispec.io.read_spectra`` and
serves spectra from a LoaArchive instead. Also verifies graceful
handling of missing TIDs and that the archive path is independent of
the FITS file existence.

Three scenarios:
1. Synthetic-archive shape + content correctness.
2. Missing-TID handling: not in archive → log warning + skip.
3. Independence from desispec: monkeypatch desispec to fail; archive
   path should still produce results.

Bit-equivalence FITS-vs-archive at the DLA-inference level is already
covered by ``tests/test_loa_archive.py::test_loa_archive_preserves_pdla_through_inference``
and the script-level check
``examples/compare_archive_vs_fits_dla_search.py`` (PR description's
"P(DLA) match to 4 sig figs" claim, validated 2026-05-03).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _make_synthetic_archive(tmp_path, tids, n_pix=200):
    """Build a tiny LoaArchive .h5 with the given TIDs for testing."""
    from gpy_dla_detection.loa_archive import CoaddRecord, write_archive

    rng = np.random.default_rng(int(tids[0]) & 0xFFFF)
    wave = np.linspace(3600.0, 3600.0 + 0.8 * (n_pix - 1), n_pix, dtype=np.float32)
    records = []
    for tid in tids:
        flux = (1.0 + 0.1 * rng.standard_normal(n_pix)).astype(np.float32)
        ivar = (3.0 + 1.0 * rng.uniform(size=n_pix)).astype(np.float32)
        mask = np.zeros(n_pix, dtype=np.uint32)
        # 11×n_pix R kernel (a Gaussian with sigma=1.2 px)
        sigma_pix = 1.2
        i = np.arange(-5, 6, dtype=float)
        k = np.exp(-0.5 * (i / sigma_pix) ** 2)
        k /= k.sum()
        R = np.tile(k[:, None], (1, n_pix))
        records.append(CoaddRecord(
            targetid=int(tid), z=2.5 + 0.001 * (int(tid) % 100),
            ra=180.0, dec=-15.0, healpix=12345, zwarn=0,
            blue_snr=2.0, red_snr=3.0,
            source_file=f"healpix/main/dark/123/12345/coadd-main-dark-12345.fits",
            fiber_idx=int(tid) % 500,
            flux=flux, ivar=ivar, mask=mask, R=R,
        ))
    out = tmp_path / "synthetic_archive.h5"
    write_archive(out, records, wavelength=wave, source_root="/loa")
    return out


def _make_synthetic_catalog(tids):
    """Minimal catalog row dict mimicking what dlasearch_hpx passes
    (astropy Table-like with TARGETID + HPXPIXEL keys)."""
    from astropy.table import Table
    return Table({
        "TARGETID": np.asarray(tids, dtype=np.int64),
        "TARGET_RA": np.full(len(tids), 180.0),
        "TARGET_DEC": np.full(len(tids), -15.0),
        "Z": np.full(len(tids), 2.55),
        "HPXPIXEL": np.full(len(tids), 12345, dtype=np.int64),
    })


def test_load_group_spectra_archive_shapes(tmp_path):
    """Archive path returns correctly-shaped (wave, flux, ivar, mask, tids)
    arrays for all TIDs in the catalog."""
    import dlasearch
    from gpy_dla_detection.loa_archive import LoaArchive

    tids = [10001, 10002, 10003]
    archive_path = _make_synthetic_archive(tmp_path, tids)
    catalog = _make_synthetic_catalog(tids)

    archive = LoaArchive(str(archive_path))
    archive.open()
    wave, flux, ivar, mask, fmap_tids = dlasearch._load_group_spectra(
        coaddpath=None, catalog=catalog, archive=archive
    )
    archive.close()

    assert wave.shape == (200,), f"wave shape {wave.shape}"
    assert flux.shape == (3, 200), f"flux shape {flux.shape}"
    assert ivar.shape == (3, 200), f"ivar shape {ivar.shape}"
    assert mask.shape == (3, 200), f"mask shape {mask.shape}"
    assert fmap_tids.tolist() == tids
    # flux around 1, ivar > 0 everywhere
    assert np.isclose(np.median(flux), 1.0, atol=0.5)
    assert (ivar > 0).all()
    assert (mask == 0).all()


def test_load_group_spectra_archive_skips_missing_tid(tmp_path, caplog):
    """A TID in the catalog but NOT in the archive is logged + skipped;
    the rest of the catalog is still served correctly."""
    import logging
    import dlasearch
    from gpy_dla_detection.loa_archive import LoaArchive

    archive_path = _make_synthetic_archive(tmp_path, [10001, 10002])
    # Catalog includes a TID that doesn't exist in the archive
    catalog = _make_synthetic_catalog([10001, 99999, 10002])

    archive = LoaArchive(str(archive_path))
    archive.open()
    with caplog.at_level(logging.WARNING):
        wave, flux, ivar, mask, fmap_tids = dlasearch._load_group_spectra(
            coaddpath=None, catalog=catalog, archive=archive
        )
    archive.close()

    assert fmap_tids.tolist() == [10001, 10002], (
        f"missing TID 99999 should be skipped, got {fmap_tids.tolist()}"
    )
    assert flux.shape == (2, 200)
    assert any("99999" in r.getMessage() for r in caplog.records), (
        f"no warning logged for missing TID 99999; logs: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_load_group_spectra_archive_independent_of_desispec(tmp_path,
                                                              monkeypatch):
    """When archive is provided, the FITS code path (desispec.io.read_spectra)
    must NOT be called. Monkeypatch it to raise; the test passes only if
    the function avoids it."""
    import dlasearch
    import desispec.io
    from gpy_dla_detection.loa_archive import LoaArchive

    def boom(*args, **kwargs):
        raise RuntimeError("desispec.io.read_spectra was called; archive "
                           "path should have bypassed this!")

    monkeypatch.setattr(desispec.io, "read_spectra", boom)

    tids = [10001, 10002]
    archive_path = _make_synthetic_archive(tmp_path, tids)
    catalog = _make_synthetic_catalog(tids)

    archive = LoaArchive(str(archive_path))
    archive.open()
    # Should NOT raise
    wave, flux, ivar, mask, fmap_tids = dlasearch._load_group_spectra(
        coaddpath="/nonexistent/coadd.fits",
        catalog=catalog,
        archive=archive,
    )
    archive.close()

    assert flux.shape == (2, 200)
    assert fmap_tids.tolist() == tids
