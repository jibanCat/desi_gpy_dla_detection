"""
tests/test_generate_samples.py

Unit tests for gpy_dla_detection.generate_samples.

Verifies:
  1. Output arrays have the expected shape and dtype.
  2. log_NHI samples fall within the requested range.
  3. z-offset samples are in [0, 1].
  4. The saved HDF5 file has the correct keys and shapes (compatible with DLASamplesMAT).
  5. The normalized prior integrates to ~1.
  6. Presets (subdla, lls, dla) produce sensible ranges.
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from gpy_dla_detection.generate_samples import (
    build_pw14_prior,
    f_pw14,
    generate_pw14_samples,
    save_samples_to_mat,
)


# ---------------------------------------------------------------------------
# Tests for f_pw14 (CDDF)
# ---------------------------------------------------------------------------


def test_f_pw14_decreasing():
    """f_pw14 should be a decreasing function of log_NHI."""
    log_nhis = np.linspace(12.0, 22.0, 100)
    f = f_pw14(log_nhis)
    assert np.all(np.diff(f) < 0), "CDDF should be monotonically decreasing."


def test_f_pw14_positive():
    """f_pw14 should be positive everywhere in the node range."""
    log_nhis = np.linspace(12.0, 22.0, 100)
    f = f_pw14(log_nhis)
    assert np.all(f > 0), "CDDF values should be positive."


# ---------------------------------------------------------------------------
# Tests for build_pw14_prior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_lognhi,max_lognhi", [
    (19.0, 20.3),  # sub-DLA
    (17.2, 19.0),  # LLS
    (20.3, 23.0),  # DLA
])
def test_prior_integrates_to_one(min_lognhi, max_lognhi):
    """The prior PDF should integrate to approximately 1 over the requested range."""
    from scipy.integrate import quad

    pdf, _ = build_pw14_prior(min_lognhi, max_lognhi)
    integral, _ = quad(pdf, min_lognhi, max_lognhi)
    assert abs(integral - 1.0) < 1e-3, (
        f"Prior integral = {integral:.6f}, expected ~1.0 "
        f"for range [{min_lognhi}, {max_lognhi}]"
    )


def test_inverse_cdf_monotone():
    """The inverse CDF should be monotonically non-decreasing."""
    _, inv_cdf = build_pw14_prior(19.0, 20.3)
    u_vals = np.linspace(0.01, 0.99, 200)
    x_vals = inv_cdf(u_vals)
    assert np.all(np.diff(x_vals) >= 0), "Inverse CDF should be non-decreasing."


# ---------------------------------------------------------------------------
# Tests for generate_pw14_samples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_lognhi,max_lognhi", [
    (19.0, 20.3),  # sub-DLA
    (17.2, 19.0),  # LLS
])
def test_samples_in_range(min_lognhi, max_lognhi):
    """All log_NHI samples should lie within [min_log_nhi, max_log_nhi]."""
    n = 500
    result = generate_pw14_samples(num_samples=n, min_log_nhi=min_lognhi,
                                   max_log_nhi=max_lognhi, seed=0)
    log_nhis = result["log_nhi_samples"]
    assert log_nhis.min() >= min_lognhi - 1e-9
    assert log_nhis.max() <= max_lognhi + 1e-9


def test_offset_samples_in_unit_interval():
    """z-offset samples (offset_samples) must lie in [0, 1]."""
    result = generate_pw14_samples(num_samples=500, min_log_nhi=19.0,
                                   max_log_nhi=20.3, seed=0)
    offsets = result["offset_samples"]
    assert offsets.min() >= 0.0
    assert offsets.max() <= 1.0


def test_nhi_samples_consistent():
    """nhi_samples should equal 10^log_nhi_samples."""
    result = generate_pw14_samples(num_samples=200, min_log_nhi=19.0,
                                   max_log_nhi=20.3, seed=0)
    expected = 10.0 ** result["log_nhi_samples"]
    np.testing.assert_allclose(result["nhi_samples"], expected, rtol=1e-10)


def test_reproducibility():
    """Same seed should give identical results."""
    r1 = generate_pw14_samples(num_samples=100, min_log_nhi=19.0,
                               max_log_nhi=20.3, seed=7)
    r2 = generate_pw14_samples(num_samples=100, min_log_nhi=19.0,
                               max_log_nhi=20.3, seed=7)
    np.testing.assert_array_equal(r1["log_nhi_samples"], r2["log_nhi_samples"])
    np.testing.assert_array_equal(r1["offset_samples"],  r2["offset_samples"])


# ---------------------------------------------------------------------------
# Tests for save_samples_to_mat (HDF5 output format)
# ---------------------------------------------------------------------------

_EXPECTED_KEYS = {
    "log_nhi_samples",
    "nhi_samples",
    "offset_samples",
    "alpha",
    "fit_min_log_nhi",
    "fit_max_log_nhi",
    "uniform_min_log_nhi",
    "uniform_max_log_nhi",
}


def test_save_and_reload_keys():
    """Saved HDF5 file should contain all expected keys."""
    samples = generate_pw14_samples(num_samples=200, min_log_nhi=19.0,
                                    max_log_nhi=20.3, seed=0)
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
        path = f.name
    try:
        save_samples_to_mat(samples, path)
        with h5py.File(path, "r") as hf:
            assert _EXPECTED_KEYS.issubset(set(hf.keys())), (
                f"Missing keys: {_EXPECTED_KEYS - set(hf.keys())}"
            )
    finally:
        os.unlink(path)


def test_save_shapes():
    """Saved datasets should have shape (N, 1) for arrays and (1, 1) for scalars."""
    n = 300
    samples = generate_pw14_samples(num_samples=n, min_log_nhi=19.0,
                                    max_log_nhi=20.3, seed=0)
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
        path = f.name
    try:
        save_samples_to_mat(samples, path)
        with h5py.File(path, "r") as hf:
            for key in ("log_nhi_samples", "nhi_samples", "offset_samples"):
                assert hf[key].shape == (n, 1), (
                    f"{key} shape is {hf[key].shape}, expected ({n}, 1)"
                )
            for key in ("alpha", "fit_min_log_nhi", "fit_max_log_nhi",
                        "uniform_min_log_nhi", "uniform_max_log_nhi"):
                assert hf[key].shape == (1, 1), (
                    f"{key} shape is {hf[key].shape}, expected (1, 1)"
                )
    finally:
        os.unlink(path)


def test_save_values_preserved():
    """The saved values should round-trip correctly."""
    min_lognhi, max_lognhi = 17.2, 19.0
    n = 200
    samples = generate_pw14_samples(num_samples=n, min_log_nhi=min_lognhi,
                                    max_log_nhi=max_lognhi, seed=0)
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
        path = f.name
    try:
        save_samples_to_mat(samples, path)
        with h5py.File(path, "r") as hf:
            np.testing.assert_allclose(
                hf["log_nhi_samples"][:, 0],
                samples["log_nhi_samples"],
                rtol=1e-10,
            )
            assert float(hf["fit_min_log_nhi"][0, 0]) == pytest.approx(min_lognhi)
            assert float(hf["fit_max_log_nhi"][0, 0]) == pytest.approx(max_lognhi)
    finally:
        os.unlink(path)


def test_generate_with_output_path(tmp_path):
    """generate_pw14_samples should write the file when output_path is given."""
    out = str(tmp_path / "test_samples.mat")
    generate_pw14_samples(num_samples=100, min_log_nhi=19.0, max_log_nhi=20.3,
                          output_path=out, seed=0)
    assert os.path.exists(out)
    with h5py.File(out, "r") as hf:
        assert "log_nhi_samples" in hf
