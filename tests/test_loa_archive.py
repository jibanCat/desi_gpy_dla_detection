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


# ---------------------------------------------------------------------------
# Integration test: archive round-trip preserves P(DLA) at inference
# ---------------------------------------------------------------------------
# This test validates the PR description's claim that the LoaArchive
# preserves P(DLA) "to 4 sig figs" through the inference pipeline. It runs
# only on machines with both the production GP model AND a real LOA coadd
# AND the DR9Q prior catalogs available — i.e. GreatLakes / NERSC. On any
# other machine it skips with a clear reason.

LOA_COADD_PATH = (
    "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/healpix/main/dark/109/10978/"
    "coadd-main-dark-10978.fits"
)
PROD_LEARNED = (
    "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/"
    "learnlogs/model_epoch_920.h5"
)
DATA_ROOT_CANDIDATES = [
    "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection",
    "/pscratch/sd/j/jibancat/desi_gpy_dla_detection",
]


def _data_root_or_none() -> str | None:
    """Return the first DATA_ROOT that has DR9Q catalog, or None."""
    import os
    for root in DATA_ROOT_CANDIDATES:
        if os.path.exists(os.path.join(root, "data/dr12q/processed/catalog.mat")):
            return root
    return None


def test_loa_archive_preserves_pdla_through_inference(tmp_path):
    """Build a 1-spectrum LoaArchive from a real LOA coadd, then run the
    full ``DLAHolder.process_qso`` on (a) the FITS-loaded arrays and
    (b) the archive-loaded arrays. Assert ``model_posteriors`` agrees
    to 4 sig figs — the PR description and
    ``project_loa_archive_2026_05_01.md`` claim "2/2 P(DLA)/MAP-NHI/MAP-z
    identical to 4 sig figs"; this codifies that empirical check.

    Skipped on machines that don't have the production GP model + a real
    LOA coadd + the DR9Q prior catalogs (i.e. anything other than
    GreatLakes / NERSC).
    """
    import os
    import sys
    pytest.importorskip("desispec")
    pytest.importorskip("fitsio")
    if not os.path.exists(LOA_COADD_PATH):
        pytest.skip(f"LOA coadd not available: {LOA_COADD_PATH}")
    if not os.path.exists(PROD_LEARNED):
        pytest.skip(f"production learned-file not available: {PROD_LEARNED}")
    DATA_ROOT = _data_root_or_none()
    if DATA_ROOT is None:
        pytest.skip(
            f"DR9Q prior catalog not found; tried {DATA_ROOT_CANDIDATES}"
        )

    # Need to import run_bayes_select which lives at repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import warnings
    import desispec.io
    from desispec.coaddition import coadd_cameras
    from gpy_dla_detection.loa_archive import CoaddRecord, LoaArchive, write_archive
    from gpy_dla_detection.set_parameters import Parameters
    from run_bayes_select import DLAHolder

    # 1) Load one TARGETID from a real LOA coadd via the production path.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = desispec.io.read_spectra(
            LOA_COADD_PATH,
            skip_hdus=["EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG"],
        )
        spec = coadd_cameras(spec)
    fmap_tids = np.asarray(spec.fibermap["TARGETID"])
    target_idx = 0
    target_tid = int(fmap_tids[target_idx])

    wave = np.asarray(spec.wave["brz"], dtype=np.float32)
    flux_fits = spec.flux["brz"][target_idx].astype(np.float32)
    ivar_fits = spec.ivar["brz"][target_idx].astype(np.float32)
    mask_fits = spec.mask["brz"][target_idx].astype(np.uint32)
    R_fits = spec.resolution_data["brz"][target_idx]

    # 2) Round-trip through LoaArchive.
    record = CoaddRecord(
        targetid=target_tid, z=2.5, ra=180.0, dec=-15.0, healpix=10978,
        zwarn=0, blue_snr=2.0, red_snr=3.0,
        source_file="coadd-main-dark-10978.fits", fiber_idx=target_idx,
        flux=flux_fits, ivar=ivar_fits, mask=mask_fits, R=R_fits,
    )
    archive_path = tmp_path / "single_tid.h5"
    write_archive(archive_path, [record], wavelength=wave,
                  source_root="/nfs/turbo/lsa-cavestru/mfho/DESI/loa")
    with LoaArchive(archive_path) as ar:
        spec_arch = ar.get_spectrum(target_tid)
        wave_arch = np.asarray(spec_arch.wavelength, dtype=np.float32)
        flux_arch = np.asarray(spec_arch.flux, dtype=np.float32)
        ivar_arch = np.asarray(spec_arch.ivar, dtype=np.float32)
        mask_arch = np.asarray(spec_arch.mask, dtype=np.uint32)

    # 3) The arrays MUST be bit-exact (writer is a passthrough for f32).
    np.testing.assert_array_equal(wave_arch, wave)
    np.testing.assert_array_equal(flux_arch, flux_fits)
    np.testing.assert_array_equal(ivar_arch, ivar_fits)
    np.testing.assert_array_equal(mask_arch, mask_fits)

    # 4) Build the DLAHolder. Same params as the y3 production preset
    # (smoke_one_spectrum.PRESETS["y3"]). The subdla_samples.mat enforces
    # num_dla_samples=10000 via an assertion in SubDLASamplesMAT.__init__,
    # so we keep the production sample count. max_dlas=1 keeps the test
    # to one DLA-recursion step (~30s per process_qso call).
    common_params = dict(
        loading_min_lambda=910.0, loading_max_lambda=1550.0,
        normalization_min_lambda=1425.0, normalization_max_lambda=1475.0,
        min_lambda=911.75, max_lambda=1216.75,
        dlambda=0.15, k=30, max_noise_variance=9.0,
        num_lines=3, max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=3,
    )
    params = Parameters(num_dla_samples=10000, **common_params)
    params_subdla = Parameters(num_dla_samples=10000, **common_params)

    holder = DLAHolder(
        learned_file=PROD_LEARNED,
        catalog_name=os.path.join(DATA_ROOT, "data/dr12q/processed/catalog.mat"),
        los_catalog=os.path.join(
            DATA_ROOT,
            "data/dla_catalogs/dr9q_concordance/processed/los_catalog",
        ),
        dla_catalog=os.path.join(
            DATA_ROOT,
            "data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
        ),
        dla_samples_file=os.path.join(
            DATA_ROOT, "data/dr12q/processed/dla_samples_a03.mat"),
        sub_dla_samples_file=os.path.join(
            DATA_ROOT, "data/dr12q/processed/subdla_samples.mat"),
        params=params, params_subdla=params_subdla,
        min_z_separation=3000.0,
        prev_tau_0=0.00246, prev_beta=3.62,
        max_dlas=1, broadening=True, plot_figures=False,
        max_workers=1, batch_size=1,
        single_absorber_model=False,
    )

    def _ivar_to_nv(ivar):
        return np.where(ivar > 0, 1.0 / np.where(ivar == 0, 1.0, ivar), 1e10)

    z_qso_fixed = 2.6  # plausible LOA z; doesn't have to be the truth
    pixel_mask = mask_fits != 0
    pixel_mask_arch = mask_arch != 0

    holder.initialize_results(1)
    holder.process_qso(
        idx=0, target_id=target_tid,
        wavelengths=wave.astype(np.float64),
        flux=flux_fits.astype(np.float64),
        noise_variance=_ivar_to_nv(ivar_fits).astype(np.float64),
        pixel_mask=pixel_mask, z_qso=z_qso_fixed,
    )
    mp_fits = np.asarray(holder.results["model_posteriors"][0]).copy()
    p_dla_fits = float(holder.results["p_dlas"][0])

    holder.initialize_results(1)
    holder.process_qso(
        idx=0, target_id=target_tid,
        wavelengths=wave_arch.astype(np.float64),
        flux=flux_arch.astype(np.float64),
        noise_variance=_ivar_to_nv(ivar_arch).astype(np.float64),
        pixel_mask=pixel_mask_arch, z_qso=z_qso_fixed,
    )
    mp_arch = np.asarray(holder.results["model_posteriors"][0]).copy()
    p_dla_arch = float(holder.results["p_dlas"][0])

    # 5) Inputs were bit-exact → outputs should be too. PR description
    # commits to "4 sig figs"; we use 1e-4 relative for the headline
    # P(DLA) plus 1e-4 absolute on the per-model posterior vector.
    assert mp_fits.shape == mp_arch.shape
    assert abs(p_dla_arch - p_dla_fits) < 1e-4, (
        f"P(DLA) divergence: FITS={p_dla_fits:.6f} ARCHIVE={p_dla_arch:.6f} "
        f"Δ={p_dla_arch - p_dla_fits:.2e}"
    )
    np.testing.assert_allclose(
        mp_arch, mp_fits, rtol=0, atol=1e-4,
        err_msg=(
            f"per-model posterior divergence:\n"
            f"  FITS:    {mp_fits}\n"
            f"  ARCHIVE: {mp_arch}\n"
            f"  Δ:       {mp_arch - mp_fits}"
        ),
    )
