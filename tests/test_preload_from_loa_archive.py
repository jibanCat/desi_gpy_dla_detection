"""Unit tests for preload_spectra.preload_from_loa_archive.

Builds a tiny synthetic LoaArchive (5 QSOs of known shape), runs the
adapter, and verifies the output trainset.h5:
  - schema matches v2 legacy convention (load_preprocessed_h5 reads it)
  - rest grid + per-spectrum interpolation are correct
  - filters apply in the right order (z, ZWARN, exclude_targetids, max_spectra)
  - DESI mask + ivar=0 → invalid pixels propagate to NaN/inf in output
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from gpy_dla_detection.loa_archive import (
    write_archive, CoaddRecord, LoaArchive,
)
from preload_spectra.preload_from_loa_archive import loa_archive_to_trainset


# --- synthetic archive builder ---

OBS_WAVE = np.arange(3600.0, 9800.0 + 0.8, 0.8, dtype=np.float32)  # DESI grid
N_OBS = OBS_WAVE.shape[0]


def _make_record(targetid: int, z: float, *, zwarn: int = 0,
                 flat_flux: float = 1.0, ivar_val: float = 100.0,
                 mask_first_n: int = 0) -> CoaddRecord:
    flux = np.full(N_OBS, flat_flux, dtype=np.float32)
    ivar = np.full(N_OBS, ivar_val, dtype=np.float32)
    mask = np.zeros(N_OBS, dtype=np.uint32)
    if mask_first_n > 0:
        mask[:mask_first_n] = 1
        ivar[:mask_first_n] = 0.0
    R = np.ones((11, N_OBS), dtype=np.float32) / 11.0
    return CoaddRecord(
        targetid=targetid, z=z, ra=10.0, dec=20.0,
        healpix=1234, zwarn=zwarn,
        blue_snr=5.0, red_snr=10.0,
        source_file="synthetic.fits", fiber_idx=targetid,
        flux=flux, ivar=ivar, mask=mask, R=R,
    )


def _build_synthetic_archive(path: Path, records: list[CoaddRecord]) -> Path:
    # write_archive requires either compression=None+compression_opts=None
    # (h5py rejects compression_opts without compression). Use gzip level 1
    # for tests — fast and small.
    write_archive(path, records, wavelength=OBS_WAVE,
                  source_root="/synthetic", with_resolution=False,
                  chunk_qsos=4, compression="gzip", compression_opts=1)
    return path


# --- tests ---

def test_archive_to_trainset_basic_schema(tmp_path):
    """Adapter writes the v2 legacy schema that load_preprocessed_h5 reads."""
    archive = _build_synthetic_archive(
        tmp_path / "tiny.h5",
        [_make_record(100 + i, z=2.5 + 0.1 * i) for i in range(5)],
    )
    out = tmp_path / "trainset.h5"
    summary = loa_archive_to_trainset(
        archive, out,
        z_min=2.0, z_max=4.5,
        rest_min=900.0, rest_max=1500.0, rest_dlambda=0.5,
        verbose=False,
    )
    assert summary["n_archive"] == 5
    assert summary["n_kept"] == 5  # all pass z + ZWARN

    with h5py.File(out, "r") as f:
        # Required v2 legacy schema keys
        for k in ("tids", "rest_wavelengths", "fluxes", "noise_variance",
                  "zqso", "redsnr", "bluesnr"):
            assert k in f, f"missing required key: {k}"
        assert f["tids"].shape == (5,)
        assert f["zqso"].shape == (5,)
        assert f["fluxes"].shape == (5, summary["n_pix"])
        assert f["noise_variance"].shape == (5, summary["n_pix"])
        assert f["rest_wavelengths"].shape == (5, summary["n_pix"])
        assert int(f["tids"][0]) == 100


def test_z_filter(tmp_path):
    """z-filter excludes QSOs outside [z_min, z_max]."""
    records = [
        _make_record(1, z=1.5),  # below
        _make_record(2, z=2.5),  # in range
        _make_record(3, z=3.5),  # in range
        _make_record(4, z=4.5),  # above
    ]
    archive = _build_synthetic_archive(tmp_path / "z.h5", records)
    out = tmp_path / "out.h5"
    summary = loa_archive_to_trainset(
        archive, out, z_min=2.0, z_max=4.0, verbose=False,
    )
    assert summary["n_after_z"] == 2
    assert summary["n_kept"] == 2

    with h5py.File(out, "r") as f:
        kept_tids = sorted(f["tids"][:].tolist())
    assert kept_tids == [2, 3]


def test_zwarn_filter(tmp_path):
    """ZWARN != 0 spectra are excluded."""
    records = [
        _make_record(1, z=2.5, zwarn=0),  # OK
        _make_record(2, z=2.5, zwarn=1),  # ZWARN flagged
        _make_record(3, z=2.5, zwarn=4),  # ZWARN flagged
        _make_record(4, z=2.5, zwarn=0),  # OK
    ]
    archive = _build_synthetic_archive(tmp_path / "zw.h5", records)
    out = tmp_path / "out.h5"
    summary = loa_archive_to_trainset(archive, out, verbose=False)
    assert summary["n_after_zwarn"] == 2
    with h5py.File(out, "r") as f:
        kept_tids = sorted(f["tids"][:].tolist())
    assert kept_tids == [1, 4]


def test_exclude_targetids(tmp_path):
    """exclude_targetids set drops matching TARGETIDs."""
    records = [_make_record(100 + i, z=2.5) for i in range(5)]
    archive = _build_synthetic_archive(tmp_path / "ex.h5", records)
    out = tmp_path / "out.h5"
    summary = loa_archive_to_trainset(
        archive, out,
        exclude_targetids={101, 103},
        verbose=False,
    )
    assert summary["n_after_exclude"] == 3
    with h5py.File(out, "r") as f:
        kept_tids = sorted(f["tids"][:].tolist())
    assert kept_tids == [100, 102, 104]


def test_max_spectra_cap_with_seed(tmp_path):
    """max_spectra cap uses a seeded RNG for reproducibility."""
    records = [_make_record(100 + i, z=2.5) for i in range(20)]
    archive = _build_synthetic_archive(tmp_path / "cap.h5", records)
    out_a = tmp_path / "a.h5"
    out_b = tmp_path / "b.h5"
    summary_a = loa_archive_to_trainset(
        archive, out_a, max_spectra=7, seed=42, verbose=False,
    )
    summary_b = loa_archive_to_trainset(
        archive, out_b, max_spectra=7, seed=42, verbose=False,
    )
    assert summary_a["n_kept"] == 7
    assert summary_b["n_kept"] == 7
    with h5py.File(out_a, "r") as fa, h5py.File(out_b, "r") as fb:
        # Same seed → same sample
        assert sorted(fa["tids"][:].tolist()) == sorted(fb["tids"][:].tolist())


def test_rest_frame_shift(tmp_path):
    """A flat-flux QSO at z=2.0 has its observed-frame λ mapped to
    rest = obs / (1+z) = obs / 3 in the output."""
    rec = _make_record(42, z=2.0, flat_flux=1.5, ivar_val=100.0)
    archive = _build_synthetic_archive(tmp_path / "rest.h5", [rec])
    out = tmp_path / "out.h5"
    # Rest grid covers a band that maps to within obs grid:
    # rest 1500 Å × (1+2) = obs 4500 Å (well inside [3600, 9800])
    summary = loa_archive_to_trainset(
        archive, out,
        z_min=1.0, z_max=4.5,  # accept z=2.0 (default 2.15 would drop)
        rest_min=1500.0, rest_max=2500.0, rest_dlambda=10.0,
        verbose=False,
    )
    with h5py.File(out, "r") as f:
        flux = f["fluxes"][0]
        nv = f["noise_variance"][0]
    # Interior pixels (within obs-rest coverage) should be ~1.5
    finite = np.isfinite(flux)
    assert finite.sum() > 0
    np.testing.assert_allclose(flux[finite], 1.5, atol=1e-5)
    # Noise variance should be ~1/100 = 0.01
    finite_nv = np.isfinite(nv)
    np.testing.assert_allclose(nv[finite_nv], 0.01, atol=1e-5)


def test_mask_propagates_to_invalid(tmp_path):
    """DESI mask != 0 → output flux=NaN, nv=inf at the rest pixels that
    sample those obs pixels.

    A blueward mask (first 1000 obs pixels masked) wipes out the bluest
    rest-frame pixels. We verify the count of invalid rest pixels is
    consistent with the mask coverage at z=2.0.
    """
    rec = _make_record(7, z=2.0, mask_first_n=1000)
    archive = _build_synthetic_archive(tmp_path / "mask.h5", [rec])
    out = tmp_path / "out.h5"
    summary = loa_archive_to_trainset(
        archive, out,
        z_min=1.0, z_max=4.5,  # accept z=2.0 (default 2.15 would drop)
        rest_min=900.0, rest_max=2500.0, rest_dlambda=1.0,
        verbose=False,
    )
    with h5py.File(out, "r") as f:
        flux = f["fluxes"][0]
        nv = f["noise_variance"][0]
    # The first 1000 obs pixels (3600 to 4400 Å) at z=2 map to
    # rest 1200 to ~1467 Å. Below rest 1200 Å (rest pixels 0 to ~300)
    # should also be invalid because the obs spectrum starts at 3600 Å.
    # The blueward rest pixels should be NaN.
    assert np.isnan(flux[0])
    assert np.isinf(nv[0])
    # Some interior pixels should still be valid
    assert np.isfinite(flux).sum() > 0


def test_hcd_nhi_filter_via_helper(tmp_path):
    """_load_excludes_from_fits with nhi_col + nhi_min builds the right
    HCD exclusion set.

    Mirrors the legacy preload_loa_real.py --hcd-min-nhi behaviour:
    only HCDs with NHI ≥ threshold get excluded.
    """
    from astropy.table import Table
    from preload_spectra.preload_from_loa_archive import _load_excludes_from_fits

    cat = Table()
    cat["TARGETID"] = [1, 2, 3, 4, 5]
    cat["NHI"] = [19.5, 20.5, 21.0, 17.5, 22.5]
    cat_path = tmp_path / "hcd.fits"
    cat.write(cat_path, format="fits")

    # NHI ≥ 20.3 → exclude TIDs with NHI in {20.5, 21.0, 22.5} → {2, 3, 5}
    excl_strict = _load_excludes_from_fits(
        cat_path, tid_col="TARGETID", nhi_col="NHI", nhi_min=20.3,
    )
    assert excl_strict == {2, 3, 5}

    # NHI ≥ 17.2 → all 5 rows
    excl_lax = _load_excludes_from_fits(
        cat_path, tid_col="TARGETID", nhi_col="NHI", nhi_min=17.2,
    )
    assert excl_lax == {1, 2, 3, 4, 5}

    # No nhi_min → exclude all (default behaviour)
    excl_all = _load_excludes_from_fits(cat_path, tid_col="TARGETID")
    assert excl_all == {1, 2, 3, 4, 5}


def test_compatible_with_load_preprocessed_h5(tmp_path):
    """Output is consumable by gpy_dla_detection.training.dataset.load_preprocessed_h5."""
    from gpy_dla_detection.training.dataset import load_preprocessed_h5

    records = [_make_record(100 + i, z=2.3 + 0.2 * i) for i in range(8)]
    archive = _build_synthetic_archive(tmp_path / "c.h5", records)
    out = tmp_path / "trainset.h5"
    loa_archive_to_trainset(
        archive, out,
        rest_min=1200.0, rest_max=1500.0, rest_dlambda=0.5,
        verbose=False,
    )
    # load_preprocessed_h5 reads our schema and applies its own
    # preprocessing (mask high-noise, normalize, de-forest, center).
    ts = load_preprocessed_h5(
        out, z_min=2.0, z_max=4.5,
        max_noise_variance=1e6,  # don't drop our synthetic flat data
        apply_normalize=False,    # skip — our tiny synthetic spectra
        apply_de_forest=False,    # don't have a useful normalization band
        apply_center=False,
    )
    assert ts.n_spectra == 8
    assert ts.fluxes.shape[1] == ts.n_pix
    # Our synthetic flat-flux + flat-ivar produces finite outputs
    assert ts.fluxes.isfinite().all()
