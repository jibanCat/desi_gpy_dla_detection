"""Unit tests for gpy_dla_detection.loa_archive.

Covers:
  * fwhm_pixels_from_resolution analytic correctness
  * write → read round-trip preserves flux/ivar/mask/wavelength bit-exactly
  * with_resolution=True round-trip preserves R bit-exactly
  * Catalog metadata round-trip
  * KeyError on missing TARGETID
"""
from __future__ import annotations

import numpy as np
import pytest

from gpy_dla_detection.loa_archive import (
    CoaddRecord,
    LoaArchive,
    fwhm_pixels_from_resolution,
    write_archive,
)


# ---------------------------------------------------------------------------
# fwhm_pixels_from_resolution
# ---------------------------------------------------------------------------

def _gaussian_kernel_11(sigma_pix: float) -> np.ndarray:
    i = np.arange(-5, 6, dtype=float)
    k = np.exp(-0.5 * (i / sigma_pix) ** 2)
    return k / k.sum()


def test_fwhm_for_known_gaussian() -> None:
    # σ=1.0 keeps the Gaussian inside the ±5-pixel band (5σ truncation
    # leaves ≪0.001% mass outside), so the computed FWHM should match
    # the analytic 2.3548σ to high precision. For larger σ the 11-pixel
    # kernel truncates the wings and the second moment runs slightly
    # short — a well-known effect of finite-band kernels.
    sigma_pix = 1.0
    n_pix = 50
    k = _gaussian_kernel_11(sigma_pix)
    R = np.tile(k[:, None], (1, n_pix))
    fwhm = fwhm_pixels_from_resolution(R)
    expected = 2.3548 * sigma_pix
    assert np.allclose(fwhm, expected, atol=2e-3), \
        f"FWHM={fwhm[0]:.4f}, expected {expected:.4f}"


def test_fwhm_truncation_effect_known() -> None:
    """At σ=1.5 px the 11-pixel band loses a small fraction of variance,
    so computed FWHM is ~0.15% short of analytic 2.3548σ. Document it."""
    sigma_pix = 1.5
    n_pix = 50
    k = _gaussian_kernel_11(sigma_pix)
    R = np.tile(k[:, None], (1, n_pix))
    fwhm = fwhm_pixels_from_resolution(R)
    expected = 2.3548 * sigma_pix
    rel_err = abs(fwhm[0] - expected) / expected
    assert rel_err < 0.01, f"truncation rel-err = {rel_err:.4f}"


def test_fwhm_handles_degenerate_kernels() -> None:
    n_pix = 10
    R = np.zeros((11, n_pix), dtype=float)
    R[5, :] = 1.0
    fwhm = fwhm_pixels_from_resolution(R)
    assert np.allclose(fwhm, 0.0)
    R2 = R.copy()
    R2[:, 3] = 0.0
    fwhm2 = fwhm_pixels_from_resolution(R2)
    assert np.isnan(fwhm2[3])
    assert np.allclose(fwhm2[~np.isnan(fwhm2)], 0.0)


def test_fwhm_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        fwhm_pixels_from_resolution(np.zeros((10, 5)))
    with pytest.raises(ValueError):
        fwhm_pixels_from_resolution(np.zeros(50))


# ---------------------------------------------------------------------------
# write/read round-trip (no resolution)
# ---------------------------------------------------------------------------

def _make_record(targetid: int, n_pix: int, sigma_pix: float = 1.2) -> CoaddRecord:
    rng = np.random.default_rng(int(targetid) & 0xFFFF)
    flux = rng.normal(1.0, 0.1, size=n_pix).astype(np.float32)
    ivar = rng.uniform(1.0, 5.0, size=n_pix).astype(np.float32)
    mask = rng.integers(0, 16, size=n_pix, dtype=np.uint32)
    R = np.tile(_gaussian_kernel_11(sigma_pix)[:, None], (1, n_pix))
    return CoaddRecord(
        targetid=targetid,
        z=2.5 + 0.001 * (targetid % 100),
        ra=180.0 + 0.01 * targetid,
        dec=-15.0 + 0.001 * targetid,
        healpix=10000 + (targetid % 100),
        zwarn=0,
        blue_snr=2.5 + 0.01 * (targetid % 10),
        red_snr=3.5,
        source_file=f"healpix/main/dark/100/{10000 + targetid % 100}/coadd.fits",
        fiber_idx=int(targetid % 500),
        flux=flux,
        ivar=ivar,
        mask=mask,
        R=R,
    )


def test_round_trip_basic(tmp_path) -> None:
    n_pix = 200
    wave = np.linspace(3600.0, 3600.0 + 0.8 * (n_pix - 1), n_pix, dtype=np.float32)
    records = [_make_record(t, n_pix) for t in [10001, 10002, 10003, 10004, 10005]]

    out = tmp_path / "archive.h5"
    summary = write_archive(out, records, wavelength=wave, source_root="/loa")
    assert summary["n_qsos"] == 5
    assert summary["n_pix"] == n_pix

    with LoaArchive(out) as ar:
        assert ar.n_qsos == 5
        assert not ar.has_resolution
        np.testing.assert_array_equal(ar.wavelength, wave)
        for rec in records:
            spec = ar.get_spectrum(rec.targetid)
            np.testing.assert_array_equal(spec.flux, rec.flux)
            np.testing.assert_array_equal(spec.ivar, rec.ivar)
            np.testing.assert_array_equal(spec.mask, rec.mask)
            np.testing.assert_array_equal(spec.wavelength, wave)
            assert spec.targetid == rec.targetid
            assert abs(spec.z - rec.z) < 1e-5
            assert spec.healpix == rec.healpix
            # FWHM derived from R should match analytic value
            assert np.allclose(spec.fwhm_pix, 2.3548 * 1.2, atol=1e-3)


def test_round_trip_with_resolution(tmp_path) -> None:
    n_pix = 100
    wave = np.linspace(3600.0, 3600.0 + 0.8 * (n_pix - 1), n_pix, dtype=np.float32)
    records = [_make_record(t, n_pix) for t in [20001, 20002]]

    out = tmp_path / "archive_with_R.h5"
    write_archive(out, records, wavelength=wave, source_root="/loa",
                  with_resolution=True)

    with LoaArchive(out) as ar:
        assert ar.has_resolution
        for rec in records:
            spec = ar.get_spectrum(rec.targetid, with_resolution=True)
            assert spec.resolution is not None
            # R is stored as float32; comparison must allow that quantization
            np.testing.assert_allclose(
                spec.resolution.astype(np.float64),
                rec.R.astype(np.float32).astype(np.float64),
                rtol=0, atol=0,
                err_msg="R should round-trip bit-exactly at float32",
            )

    # And without with_resolution=True, we don't load it
    with LoaArchive(out) as ar:
        spec = ar.get_spectrum(20001)
        assert spec.resolution is None


def test_get_spectrum_missing_targetid(tmp_path) -> None:
    n_pix = 50
    wave = np.arange(3600.0, 3600.0 + 0.8 * n_pix, 0.8, dtype=np.float32)[:n_pix]
    records = [_make_record(31337, n_pix)]
    out = tmp_path / "archive.h5"
    write_archive(out, records, wavelength=wave, source_root="/loa")
    with LoaArchive(out) as ar:
        with pytest.raises(KeyError):
            ar.get_spectrum(99999)


def test_with_resolution_required_for_R_access(tmp_path) -> None:
    """Asking for R from an archive that wasn't written with R must raise."""
    n_pix = 50
    wave = np.arange(3600.0, 3600.0 + 0.8 * n_pix, 0.8, dtype=np.float32)[:n_pix]
    records = [_make_record(42, n_pix)]
    out = tmp_path / "archive.h5"
    write_archive(out, records, wavelength=wave, source_root="/loa",
                  with_resolution=False)
    with LoaArchive(out) as ar:
        with pytest.raises(KeyError, match="with-resolution"):
            ar.get_spectrum(42, with_resolution=True)


def test_streaming_chunked_write(tmp_path) -> None:
    """100 records with a small flush chunk — verify all land correctly."""
    n_pix = 80
    wave = np.linspace(3600.0, 3600.0 + 0.8 * n_pix, n_pix, dtype=np.float32)
    records = [_make_record(t, n_pix) for t in range(50000, 50100)]
    out = tmp_path / "stream.h5"
    summary = write_archive(out, records, wavelength=wave, source_root="/loa",
                            chunk_qsos=7)  # awkward chunk to test partial flush
    assert summary["n_qsos"] == 100
    with LoaArchive(out) as ar:
        assert ar.n_qsos == 100
        np.testing.assert_array_equal(
            ar.get_spectrum(50050).flux,
            records[50].flux,
        )


def test_wavelength_shape_mismatch_raises(tmp_path) -> None:
    n_pix = 50
    wave = np.arange(3600.0, 3600.0 + 0.8 * n_pix, 0.8, dtype=np.float32)[:n_pix]
    bad_record = _make_record(7, n_pix=40)
    out = tmp_path / "bad.h5"
    with pytest.raises(ValueError, match="flux for TARGETID"):
        write_archive(out, [bad_record], wavelength=wave, source_root="/loa")
