"""Comprehensive tests for the v2 trainer + preload preprocessing pipeline.

The bug we found and are fixing:
  - v2 preload (`preload_loa_real.py`, `preload_2lpt_simple.py`) skipped
    the per-spectrum median normalization.
  - Without it, the trainset has 40+× per-spectrum dynamic range, so
    `_center_fluxes_inverse_variance` produces a bright-QSO-biased μ.
  - The fix adds `_normalize_by_rest_median` (Garnett+2017 [1310, 1325])
    BEFORE deforest + center, mirroring v1's SpectrumProcessor ordering.
  - The trained .h5 stores `normalization_{min,max}_lambda` so inference
    can pick up the right window; NullGPMAT/DLAGPMAT/SubDLAGPMAT mutate
    `params.normalization_*` in place so set_data uses the right window.

Tests below verify each step of the pipeline:
  1. `_mask_high_noise_pixels` — masks pixels with noise > threshold
  2. `_normalize_by_rest_median` — divides by per-spectrum median
  3. `_de_forest_batch` — applies Turner+2024 forest correction
  4. `_center_fluxes_inverse_variance` — subtracts ivar-weighted mean
  5. `load_preprocessed_h5` — orders the steps correctly
  6. `save_h5_model` — writes the new normalization fields to .h5
  7. `state_dict_for_h5` — exposes the fields
  8. NullGPMAT/DLAGPMAT/SubDLAGPMAT — mutate params on load when fields present
  9. Falls back to params for legacy v1 .h5 files (no fields)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


from gpy_dla_detection.training.dataset import (
    _mask_high_noise_pixels,
    _normalize_by_rest_median,
    _de_forest_batch,
    _center_fluxes_inverse_variance,
)


# ============================================================
# Step 1 — `_mask_high_noise_pixels`
# ============================================================
def test_mask_high_noise_pixels_basic():
    """Pixels with noise variance above threshold get NaN'd."""
    fluxes = np.ones((3, 5))
    noise = np.array([[1, 4, 9, 16, 25.],
                      [2, 4, 6, 8, 10.],
                      [0.5, 0.5, 0.5, 0.5, 0.5]])
    masked_f, masked_v = _mask_high_noise_pixels(fluxes, noise, max_noise_variance=9.0)
    # Pixel 3 (noise=16) and pixel 4 (noise=25) of spectrum 0 should be NaN
    assert np.isnan(masked_f[0, 3]) and np.isnan(masked_v[0, 3])
    assert np.isnan(masked_f[0, 4]) and np.isnan(masked_v[0, 4])
    # Spectrum 1: pixel 4 (noise=10) NaN'd
    assert np.isnan(masked_f[1, 4])
    # Spectrum 0 pixel 2 (noise=9) is at threshold and should be KEPT
    # (semantics: > max_noise_variance, not >=)
    assert not np.isnan(masked_f[0, 2])
    # Spectrum 2: all noise=0.5 < threshold → all kept
    assert not np.any(np.isnan(masked_f[2]))


# ============================================================
# Step 2 — `_normalize_by_rest_median`
# ============================================================
def test_normalize_synthetic_basic():
    """Each spectrum's median in window equals its known scale; after
    normalization, median in window = 1, noise scales by 1/median²."""
    rest = np.linspace(900, 1400, 1001)
    n_spec = 5
    scales = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    fluxes = np.broadcast_to(scales[:, None], (n_spec, len(rest))).copy()
    noise_variances = np.ones_like(fluxes) * 0.01

    fluxes_n, nv_n, meds = _normalize_by_rest_median(
        fluxes.copy(), noise_variances.copy(), rest,
        norm_min_lambda=1310.0, norm_max_lambda=1325.0,
    )
    np.testing.assert_allclose(meds, scales, rtol=1e-6)
    np.testing.assert_allclose(fluxes_n, 1.0, rtol=1e-6)
    # Each row should have a constant noise variance equal to 0.01/scale²
    for i, s in enumerate(scales):
        np.testing.assert_allclose(nv_n[i], 0.01 / (s ** 2), rtol=1e-6)


def test_normalize_preserves_snr():
    """SNR per pixel (flux / sqrt(noise_var)) is invariant under normalize."""
    rest = np.linspace(900, 1400, 1001)
    rng = np.random.default_rng(0)
    scales = np.array([1.0, 5.0])
    fluxes = scales[:, None] * (1.0 + 0.1 * rng.standard_normal((2, len(rest))))
    noise_variances = (scales[:, None] * 0.05) ** 2 * np.ones_like(fluxes)

    snr_before = fluxes / np.sqrt(noise_variances)
    fluxes_n, nv_n, _ = _normalize_by_rest_median(
        fluxes, noise_variances, rest,
        norm_min_lambda=1310.0, norm_max_lambda=1325.0,
    )
    snr_after = fluxes_n / np.sqrt(nv_n)
    np.testing.assert_allclose(snr_after, snr_before, rtol=1e-6)


def test_normalize_handles_bad_spectra():
    """All-NaN or zero-median spectra get NaN'd (not propagated as inf)."""
    rest = np.linspace(900, 1400, 1001)
    fluxes = np.ones((3, len(rest)))
    fluxes[1, :] = np.nan
    fluxes[2, :] = 0.0
    noise_variances = np.ones_like(fluxes) * 0.01

    fluxes_n, nv_n, meds = _normalize_by_rest_median(
        fluxes, noise_variances, rest,
        norm_min_lambda=1310.0, norm_max_lambda=1325.0,
    )
    assert np.allclose(fluxes_n[0], 1.0)
    assert np.all(np.isnan(fluxes_n[1]))
    assert np.all(np.isnan(fluxes_n[2]))
    assert np.isnan(meds[1])


def test_normalize_window_outside_grid_raises():
    """Window outside the rest grid → ValueError with clear message."""
    rest = np.linspace(900, 1300, 401)  # ends at 1300
    fluxes = np.ones((1, len(rest)))
    nv = np.ones_like(fluxes) * 0.01
    with pytest.raises(ValueError, match="Normalization window"):
        _normalize_by_rest_median(
            fluxes, nv, rest,
            norm_min_lambda=1425.0, norm_max_lambda=1475.0,
        )


def test_normalize_garnett_window_fits_v2_grid():
    """Garnett window [1310, 1325] fits in the v2 trainset rest grid
    [850.8, 1420.8] and contains > 50 pixels at dλ=0.15 Å.
    THIS IS THE PRIMARY REASON we picked Garnett over [1425, 1475]."""
    rest = np.arange(850.8, 1420.8 + 1e-6, 0.15)
    norm_mask = (rest >= 1310) & (rest <= 1325)
    assert norm_mask.sum() > 50


# ============================================================
# Step 3 — `_de_forest_batch`
# ============================================================
def test_de_forest_purely_multiplicative():
    """De-forest scales flux by 1/exp(-τ_eff) — no other transformation.
    Same multiplicative factor for both flux and noise²."""
    rest = np.array([1000., 1100., 1200.])
    z_qsos = np.array([2.5, 3.0])
    fluxes = np.array([[1.0, 1.0, 1.0],
                       [2.0, 2.0, 2.0]])
    nv = np.array([[0.01, 0.01, 0.01],
                   [0.04, 0.04, 0.04]])
    f_d, nv_d = _de_forest_batch(fluxes.copy(), nv.copy(), rest, z_qsos,
                                 tau_0=0.00246, beta=3.62, num_forest_lines=3)
    # Each per-spectrum, per-pixel correction should be a single multiplicative
    # factor; SNR (flux / sqrt(nv)) should be preserved.
    snr_before = fluxes / np.sqrt(nv)
    snr_after = f_d / np.sqrt(nv_d)
    np.testing.assert_allclose(snr_after, snr_before, rtol=1e-6)
    # Forest absorption: deforested flux should be > original flux for
    # pixels in the forest (rest < 1216 Å)
    assert np.all(f_d[0, :2] > fluxes[0, :2])  # both forest pixels for z=2.5
    assert np.all(f_d[1, :2] > fluxes[1, :2])


def test_de_forest_preserves_relative_brightness():
    """De-forest preserves ratio of flux between bright and faint QSOs
    (it's a per-pixel multiplicative correction; doesn't depend on flux scale)."""
    rest = np.linspace(950, 1200, 50)
    z_qsos = np.full(2, 2.5)
    fluxes = np.array([np.ones(50), 5 * np.ones(50)])
    nv = np.ones_like(fluxes) * 0.01
    f_d, _ = _de_forest_batch(fluxes.copy(), nv.copy(), rest, z_qsos,
                              tau_0=0.00246, beta=3.62, num_forest_lines=3)
    # Ratio bright/faint should be preserved at every pixel
    np.testing.assert_allclose(f_d[1] / f_d[0], 5.0, rtol=1e-6)


# ============================================================
# Step 4 — `_center_fluxes_inverse_variance`
# ============================================================
def test_center_fluxes_basic():
    """Centered flux per pixel should equal flux − inverse-variance-weighted mean."""
    fluxes = np.array([[1.0, 2.0, 3.0],
                       [2.0, 3.0, 4.0],
                       [3.0, 4.0, 5.0]])
    noise_variances = np.ones_like(fluxes) * 0.01  # equal weights
    centered, mean_flux = _center_fluxes_inverse_variance(fluxes.copy(), noise_variances)
    # Equal weights → mean is just np.mean axis=0
    expected_mean = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(mean_flux, expected_mean, rtol=1e-6)
    np.testing.assert_allclose(centered, fluxes - expected_mean, rtol=1e-6)


def test_center_inverse_variance_weighted():
    """Bright spectra with low noise dominate the mean."""
    fluxes = np.array([[1.0, 1.0, 1.0],
                       [10.0, 10.0, 10.0]])
    # Spectrum 1 has 100× lower noise → 100× higher weight
    noise_variances = np.array([[1.0, 1.0, 1.0],
                                [0.01, 0.01, 0.01]])
    _, mean_flux = _center_fluxes_inverse_variance(fluxes, noise_variances)
    # Weighted mean: (1*1 + 10*100) / (1 + 100) ≈ 9.91
    expected = (1*1 + 10*100) / (1 + 100)
    np.testing.assert_allclose(mean_flux, expected, rtol=1e-6)


# ============================================================
# Step 5 — `load_preprocessed_h5` end-to-end ordering
# ============================================================
def test_load_preprocessed_h5_normalize_path_smoke(tmp_path):
    """End-to-end smoke: synthetic trainset → load with apply_normalize=True
    produces μ near 1 in the normalization window (because every spectrum
    was constructed to have median = its scale, and normalize divides each
    by its own median)."""
    import h5py
    from gpy_dla_detection.training.dataset import load_preprocessed_h5

    rest = np.arange(850.8, 1420.8 + 1e-6, 0.15)
    n_pix = len(rest)
    n_spec = 50
    rng = np.random.default_rng(1)
    scales = rng.uniform(0.5, 10.0, size=n_spec)
    fluxes = scales[:, None] * np.ones((n_spec, n_pix), dtype=np.float32)
    fluxes += rng.standard_normal(fluxes.shape).astype(np.float32) * 0.05
    nv = np.ones_like(fluxes) * 0.01
    z_qsos = np.full(n_spec, 2.5, dtype=np.float32)
    tids = np.arange(n_spec, dtype=np.int64)
    redsnr = np.ones(n_spec, dtype=np.float32) * 5.0

    p = tmp_path / "trainset.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("tids", data=tids)
        f.create_dataset("zqso", data=z_qsos)
        f.create_dataset("redsnr", data=redsnr)
        f.create_dataset("rest_wavelengths",
                         data=np.tile(rest, (n_spec, 1)).astype(np.float32))
        f.create_dataset("fluxes", data=fluxes)
        f.create_dataset("noise_variance", data=nv)

    ts = load_preprocessed_h5(
        p, z_min=2.0, z_max=4.25,
        apply_normalize=True, apply_de_forest=False, apply_center=True,
        norm_min_lambda=1310.0, norm_max_lambda=1325.0,
    )
    mu = ts.mu.numpy()
    norm_mask = (rest >= 1310) & (rest <= 1325)
    mu_in_window = mu[norm_mask]
    assert np.isfinite(mu_in_window).all()
    assert abs(np.median(mu_in_window) - 1.0) < 0.1, (
        f"μ in window has median {np.median(mu_in_window):.3f}, expected ≈ 1.0")


def test_load_preprocessed_h5_skip_normalize(tmp_path):
    """Disabling apply_normalize gives the OLD (buggy) behavior:
    μ in window dominated by bright QSOs ≫ 1."""
    import h5py
    from gpy_dla_detection.training.dataset import load_preprocessed_h5

    rest = np.arange(850.8, 1420.8 + 1e-6, 0.15)
    n_pix = len(rest)
    n_spec = 50
    rng = np.random.default_rng(2)
    scales = rng.uniform(0.5, 10.0, size=n_spec)
    fluxes = scales[:, None] * np.ones((n_spec, n_pix), dtype=np.float32)
    nv = np.ones_like(fluxes) * 0.01
    z_qsos = np.full(n_spec, 2.5, dtype=np.float32)
    tids = np.arange(n_spec, dtype=np.int64)

    p = tmp_path / "trainset.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("tids", data=tids)
        f.create_dataset("zqso", data=z_qsos)
        f.create_dataset("redsnr", data=np.ones(n_spec, dtype=np.float32))
        f.create_dataset("rest_wavelengths",
                         data=np.tile(rest, (n_spec, 1)).astype(np.float32))
        f.create_dataset("fluxes", data=fluxes)
        f.create_dataset("noise_variance", data=nv)

    ts = load_preprocessed_h5(
        p, z_min=2.0, z_max=4.25,
        apply_normalize=False, apply_de_forest=False, apply_center=True,
    )
    mu = ts.mu.numpy()
    norm_mask = (rest >= 1310) & (rest <= 1325)
    mu_in_window = mu[norm_mask]
    # μ should be FAR from 1 (it's bright-QSO-biased mean of absolute scales)
    assert abs(np.median(mu_in_window) - 1.0) > 0.5, (
        f"Without normalization μ should NOT be near 1; got {np.median(mu_in_window):.3f}"
    )


# ============================================================
# Step 6+7 — `save_h5_model` writes normalization_* + state_dict exposes them
# ============================================================
def test_state_dict_contains_normalization_fields():
    """GPModelV2.state_dict_for_h5() exposes normalization_min/max_lambda."""
    from gpy_dla_detection.training.model_v2 import GPModelV2
    m = GPModelV2(num_pixels=100, k=10,
                  normalization_min_lambda=1310.0,
                  normalization_max_lambda=1325.0)
    sd = m.state_dict_for_h5()
    assert sd["normalization_min_lambda"] == 1310.0
    assert sd["normalization_max_lambda"] == 1325.0


def test_save_h5_model_writes_normalization_fields(tmp_path):
    """trainer_v2.save_h5_model writes the new normalization fields to the .h5."""
    from gpy_dla_detection.training.model_v2 import GPModelV2
    from gpy_dla_detection.training.trainer_v2 import save_h5_model
    import h5py

    m = GPModelV2(num_pixels=100, k=10,
                  normalization_min_lambda=1310.0,
                  normalization_max_lambda=1325.0)
    p = save_h5_model(m, tmp_path, epoch=42)
    with h5py.File(p, "r") as h:
        assert "normalization_min_lambda" in h
        assert "normalization_max_lambda" in h
        assert float(h["normalization_min_lambda"][()]) == 1310.0
        assert float(h["normalization_max_lambda"][()]) == 1325.0


# ============================================================
# Step 8+9 — Inference picks up normalization region from .h5
# ============================================================
def _make_fake_v2_model_h5(tmp_path, norm_min=1310.0, norm_max=1325.0,
                           include_norm_fields=True):
    """Helper: write a minimal v2-style .h5 the inference loader can read."""
    import h5py
    rest = np.arange(850.8, 1420.8 + 1e-6, 0.15).astype(np.float32)
    n_pix = len(rest)
    p = tmp_path / "fake_model.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("M", data=np.zeros((n_pix, 30), dtype=np.float32))
        f.create_dataset("log_omega", data=np.zeros(n_pix, dtype=np.float32))
        f.create_dataset("log_c_0", data=np.array(np.log(0.1)))
        f.create_dataset("log_tau_0", data=np.array(np.log(0.00246)))
        f.create_dataset("log_beta", data=np.array(np.log(3.62)))
        f.create_dataset("rest_wavelengths", data=rest)
        f.create_dataset("mu", data=np.ones(n_pix, dtype=np.float32))
        f.create_dataset("max_noise_variance", data=np.array(9.0))
        if include_norm_fields:
            f.create_dataset("normalization_min_lambda", data=np.array(norm_min))
            f.create_dataset("normalization_max_lambda", data=np.array(norm_max))
    return p


def _build_params(norm_min=1425.0, norm_max=1475.0):
    """Helper: build a Parameters instance with explicit normalization region."""
    from gpy_dla_detection.set_parameters import Parameters
    return Parameters(
        loading_min_lambda=910, loading_max_lambda=1550,
        normalization_min_lambda=norm_min,
        normalization_max_lambda=norm_max,
        min_lambda=850.75, max_lambda=1420.8,
        dlambda=0.15, k=30, max_noise_variance=9.0,
        num_dla_samples=10000,
    )


def _data_root_or_skip():
    DATA_ROOT = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"
    if not os.path.exists(os.path.join(DATA_ROOT, "data/dr12q/processed/catalog.mat")):
        pytest.skip("prior catalog files not present on this filesystem")
    return DATA_ROOT


def _build_prior(params, DATA_ROOT):
    from gpy_dla_detection.model_priors import PriorCatalog
    return PriorCatalog(
        params,
        os.path.join(DATA_ROOT, "data/dr12q/processed/catalog.mat"),
        os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/los_catalog"),
        os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog"),
    )


def test_null_gp_mat_picks_up_norm_from_v2_h5(tmp_path):
    """NullGPMAT mutates params.normalization_{min,max}_lambda in place
    when a v2 .h5 carries those fields."""
    from gpy_dla_detection.null_gp import NullGPMAT
    DATA_ROOT = _data_root_or_skip()

    p = _make_fake_v2_model_h5(tmp_path, norm_min=1310.0, norm_max=1325.0,
                               include_norm_fields=True)
    params = _build_params(norm_min=1425.0, norm_max=1475.0)  # wrong window
    prior = _build_prior(params, DATA_ROOT)
    NullGPMAT(params, prior, learned_file=str(p))
    # params should be mutated to match the v2 trained region
    assert params.normalization_min_lambda == 1310.0
    assert params.normalization_max_lambda == 1325.0


def test_null_gp_mat_falls_back_for_legacy_v1(tmp_path):
    """Legacy v1 .h5 (no normalization_* fields) → params NOT mutated."""
    from gpy_dla_detection.null_gp import NullGPMAT
    DATA_ROOT = _data_root_or_skip()

    p = _make_fake_v2_model_h5(tmp_path, include_norm_fields=False)
    params = _build_params(norm_min=1425.0, norm_max=1475.0)
    prior = _build_prior(params, DATA_ROOT)
    NullGPMAT(params, prior, learned_file=str(p))
    # params unchanged
    assert params.normalization_min_lambda == 1425.0
    assert params.normalization_max_lambda == 1475.0


def test_set_data_uses_overridden_norm_region(tmp_path):
    """End-to-end: load v2 model → set_data reads from the (mutated) params."""
    from gpy_dla_detection.null_gp import NullGPMAT
    DATA_ROOT = _data_root_or_skip()

    p = _make_fake_v2_model_h5(tmp_path, norm_min=1310.0, norm_max=1325.0)
    params = _build_params(norm_min=1425.0, norm_max=1475.0)
    prior = _build_prior(params, DATA_ROOT)
    g = NullGPMAT(params, prior, learned_file=str(p))

    # Build a synthetic spectrum with median 5.0 in [1310, 1325] obs frame
    z_qso = 2.5
    rest_w = np.linspace(900, 1400, 1001)
    obs_w = rest_w * (1 + z_qso)
    flux = np.full_like(rest_w, 5.0, dtype=np.float64)
    nv = np.full_like(flux, 0.01)
    mask = np.zeros_like(flux, dtype=bool)
    g.set_data(rest_w, flux, nv, mask, z_qso, normalize=True, build_model=False)
    # After normalize, median should be 1
    assert abs(g.normalization_median - 5.0) < 1e-6, (
        f"normalization should pick up the v2 window; "
        f"normalized median = {g.normalization_median} (expected 5.0)")


# ============================================================
# Cross-check with REAL trainset (only runs if path exists)
# ============================================================
def test_real_trainset_after_normalize_lands_at_unity():
    """On an actual v2 trainset.h5, applying the normalize step gives
    μ ≈ 1 in [1310, 1325]. Validates the fix on real data."""
    real_path = (
        "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/"
        "v2_runs/2lpt_loa124_nohcd_nobal_48938766/trainset.h5"
    )
    if not os.path.exists(real_path):
        pytest.skip(f"real trainset not on this fs: {real_path}")

    from gpy_dla_detection.training.dataset import load_preprocessed_h5
    ts = load_preprocessed_h5(
        real_path, z_min=2.0, z_max=4.25,
        max_spectra=2000,  # subset for speed
        apply_normalize=True, apply_de_forest=False, apply_center=True,
        norm_min_lambda=1310.0, norm_max_lambda=1325.0,
    )
    mu = ts.mu.numpy()
    rest = ts.rest_wavelengths.numpy()
    norm_mask = (rest >= 1310) & (rest <= 1325)
    mu_in_window = mu[norm_mask]
    finite = np.isfinite(mu_in_window)
    assert finite.any(), "no finite μ in normalization window — normalize broke?"
    median_mu = np.median(mu_in_window[finite])
    print(f"  real trainset μ median in [1310, 1325] = {median_mu:.4f} (expected ≈ 1)")
    assert abs(median_mu - 1.0) < 0.1, (
        f"μ in window has median {median_mu:.3f}, expected ≈ 1.0 — "
        f"normalize step may not be wired correctly on real data"
    )
