"""Tests added per 2026-05-07 PR-review feedback.

Each test corresponds to one missing-test bullet from the review:

  1. test_de_forest_matches_v1
        Catches divergence between our fixture's `_de_forest` and v1's
        SpectrumProcessor.de_forest_spectra when called with the same
        num_forest_lines.

  3. test_h5_round_trip
        Builds a synthetic GP model, saves via tests/a4_inference._save_h5,
        reloads via h5py, asserts every field round-trips bit-identically.

  4. test_prior_augmented_jacobian
        Extends Step A.1 to include the BOSS DR12Q Gaussian priors on
        log_τ_0 and log_β (matching tests/short_retrain_2lpt.py:48-51).
        FD over the prior-augmented loss should match the priors-included
        analytic gradient.

  5. test_chromatic_correction_regression
        Asserts v3.5's `chromatic_correction` accumulator equals the
        closed-form Σ_{k>1} τ_k · log(λ_α/λ_k) · 𝟙_{k-forest} on a
        synthetic input with known answer.

(Test 2 from the review — PCA-init reproducibility — is already covered
in practice by sklearn.decomposition.PCA being deterministic on identical
inputs; not added here.)
"""
from __future__ import annotations

import io
import sys
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------- 1. de-forest fidelity vs v1 -----------------------------------

def test_de_forest_matches_v1():
    """Our `_de_forest(num_forest_lines=3)` ≡ v1's SpectrumProcessor.de_forest_spectra
    on a synthetic input. v1 hardcodes num_forest_lines=3 internally; we
    pass it explicitly for an apples-to-apples comparison.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
    from build_2lpt_frozen_test_fixture import _de_forest
    from gpy_dla_detection.learn_qso_model import SpectrumProcessor

    rng = np.random.default_rng(0)
    rest = np.linspace(850.75, 1420.75, 200)
    n_spec = 5
    z_qsos = np.array([2.5, 2.8, 3.0, 3.4, 3.8])
    fluxes = rng.normal(loc=1.0, scale=0.2, size=(n_spec, 200))
    nv = np.full((n_spec, 200), 0.05)

    flux_ours, nv_ours = _de_forest(
        rest, fluxes.copy(), nv.copy(), z_qsos, num_forest_lines=3,
    )

    sp = SpectrumProcessor()
    flux_v1, nv_v1 = sp.de_forest_spectra(
        [rest] * n_spec, fluxes.copy(), nv.copy(), z_qsos,
    )
    flux_v1 = np.asarray(flux_v1)
    nv_v1 = np.asarray(nv_v1)

    assert flux_ours.shape == flux_v1.shape == fluxes.shape
    np.testing.assert_allclose(flux_ours, flux_v1, rtol=1e-12, atol=0,
                                err_msg="_de_forest flux ≠ v1 SpectrumProcessor.de_forest_spectra")
    np.testing.assert_allclose(nv_ours, nv_v1, rtol=1e-12, atol=0,
                                err_msg="_de_forest nv ≠ v1")


# ---------- 3. .h5 schema round-trip --------------------------------------

def test_h5_round_trip(tmp_path):
    """Synthetic params → _save_h5 → reload → all fields preserved bitwise."""
    from tests import a4_inference

    n_pix, k = 100, 5
    rng = np.random.default_rng(42)
    params = dict(
        rest_wavelengths=np.linspace(850.75, 1420.75, n_pix).astype(np.float64),
        mu=rng.normal(loc=1.0, scale=0.3, size=n_pix).astype(np.float64),
        log_omega=rng.normal(loc=0.0, scale=0.5, size=n_pix).astype(np.float64),
        M=rng.normal(scale=10, size=(n_pix, k)).astype(np.float64),
        log_c_0=float(np.log(0.07)),
        log_tau_0=float(np.log(0.0021)),
        log_beta=float(np.log(3.05)),
    )

    a4_inference.H5_DIR = tmp_path  # redirect output dir for the test
    out_path = a4_inference._save_h5("test_lane", params)
    assert out_path.exists()

    with h5py.File(out_path, "r") as f:
        for key in ("rest_wavelengths", "mu", "log_omega", "M"):
            np.testing.assert_array_equal(np.asarray(f[key]), params[key],
                                           err_msg=f"{key} not preserved in .h5")
        for key in ("log_c_0", "log_tau_0", "log_beta"):
            assert float(np.asarray(f[key])) == params[key], f"{key} not preserved"
        # max_noise_variance is added by _save_h5 with v1 preset value
        assert float(np.asarray(f["max_noise_variance"])) == 9.0
        # normalization_min/max_lambda should NOT be written (post-2026-05-07
        # review fix b — fall back to v1 production preset [1425, 1475])
        assert "normalization_min_lambda" not in f, \
            "normalization_min_lambda should NOT be in the .h5 (review fix b)"
        assert "normalization_max_lambda" not in f


# ---------- 4. Prior-augmented Jacobian ----------------------------------

def test_prior_augmented_jacobian():
    """Step A.1 + Gaussian priors on log_τ_0 and log_β.

    The prior-augmented gradient (matching short_retrain_2lpt.py:135-136):
        dlog_τ_0_full = dlog_τ_0_data + tau_0 * (tau_0 - μ_τ) / σ_τ²
        dlog_β_full   = dlog_β_data   + beta  * (beta  - μ_β) / σ_β²
    """
    from gpy_dla_detection.objective import spectrum_loss

    FIX = Path(__file__).resolve().parent / "fixtures" / "2lpt_frozen"
    init = np.load(FIX / "init_params.npz")
    spec = np.load(FIX / "120046865.npz")

    DTYPE = torch.float64
    EPS = 1e-5
    TAU_0_MU, TAU_0_SIGMA = 0.00554, 0.00064
    BETA_MU, BETA_SIGMA = 3.182, 0.074

    valid = np.asarray(spec["valid_mask"]).astype(bool)
    y = torch.tensor(np.asarray(spec["flux"])[valid], dtype=DTYPE)
    nv = torch.tensor(np.asarray(spec["noise_variance"])[valid], dtype=DTYPE)
    lya_1pz = torch.tensor(np.asarray(spec["lya_1pz"])[valid], dtype=DTYPE)
    M = torch.tensor(init["M"][valid], dtype=DTYPE)
    log_omega = torch.tensor(init["log_omega"][valid], dtype=DTYPE)
    log_c_0_base = torch.tensor(np.log(float(init["c_0"])), dtype=DTYPE)
    log_tau_0_base = torch.tensor(np.log(float(init["tau_0"])), dtype=DTYPE)
    log_beta_base = torch.tensor(np.log(float(init["beta"])), dtype=DTYPE)
    num_forest_lines = int(init["num_forest_lines"])
    TW = torch.tensor(init["all_transition_wavelengths"], dtype=DTYPE)
    OS = torch.tensor(init["all_oscillator_strengths"], dtype=DTYPE)
    zqso_1pz = torch.tensor(float(spec["zqso_1pz"]), dtype=DTYPE)

    def _eval_with_priors(log_tau_0, log_beta):
        omega2 = torch.exp(2 * log_omega)
        c_0 = torch.exp(log_c_0_base)
        tau_0 = torch.exp(log_tau_0)
        beta = torch.exp(log_beta)
        nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta = spectrum_loss(
            y, lya_1pz, nv, M, omega2, c_0, tau_0, beta,
            num_forest_lines, TW, OS, zqso_1pz,
        )
        prior_t = 0.5 * ((float(tau_0) - TAU_0_MU) / TAU_0_SIGMA) ** 2
        prior_b = 0.5 * ((float(beta) - BETA_MU) / BETA_SIGMA) ** 2
        return float(nlog_p) + prior_t + prior_b, float(dlog_tau_0), float(dlog_beta), \
               float(tau_0), float(beta)

    L0, dlog_tau_0_data, dlog_beta_data, tau_0, beta = _eval_with_priors(
        log_tau_0_base, log_beta_base)
    dlog_tau_0_analytic = dlog_tau_0_data + tau_0 * (tau_0 - TAU_0_MU) / TAU_0_SIGMA**2
    dlog_beta_analytic = dlog_beta_data + beta * (beta - BETA_MU) / BETA_SIGMA**2

    L_plus, *_ = _eval_with_priors(log_tau_0_base + EPS, log_beta_base)
    L_minus, *_ = _eval_with_priors(log_tau_0_base - EPS, log_beta_base)
    fd_tau = (L_plus - L_minus) / (2 * EPS)
    rel_tau = abs(fd_tau - dlog_tau_0_analytic) / max(abs(dlog_tau_0_analytic), 1e-12)
    assert rel_tau < 1e-4, f"prior-augmented dlog_τ_0 FD mismatch: rel_err={rel_tau:.2e}"

    L_plus, *_ = _eval_with_priors(log_tau_0_base, log_beta_base + EPS)
    L_minus, *_ = _eval_with_priors(log_tau_0_base, log_beta_base - EPS)
    fd_beta = (L_plus - L_minus) / (2 * EPS)
    # log_beta has the documented v1+MATLAB approximate gradient (term-A only).
    # Augmenting with the prior doesn't change the bias size — still ~5e-2.
    rel_beta = abs(fd_beta - dlog_beta_analytic) / max(abs(dlog_beta_analytic), 1e-12)
    assert rel_beta < 5e-2, f"prior-augmented dlog_β FD mismatch: rel_err={rel_beta:.2e}"


# ---------- 5. v3.5 chromatic correction regression -----------------------

def test_v3_5_chromatic_correction_closed_form():
    """v3.5's chromatic_correction = Σ_{k>1} τ_k · log(λ_α/λ_k) · 𝟙_{k-forest}.

    We can't reach into spectrum_loss to inspect chromatic_correction
    directly; instead we verify the OBSERVABLE consequence: dlog_β_v3_5
    minus dlog_β_v1 equals the contribution from chromatic_correction
    against (K_inv_y, diag_K_inv).

    Build a tiny synthetic spectrum with known params; both spectrum_losses
    expose dlog_β. The difference should equal:
        Σ_pixels [(diag_K_inv_i - K_inv_y_i²) * ω²_i * scaling_i * exp(-τ_tot_i) *
                  chromatic_correction_i * β]
    where chromatic_correction_i is the closed-form sum.
    """
    from gpy_dla_detection.objective import spectrum_loss as v1_loss
    from gpy_dla_detection.training_v3_5.objective import spectrum_loss as v35_loss
    from gpy_dla_detection.voigt import (
        transition_wavelengths as TW_phys, oscillator_strengths as OS_phys)

    DTYPE = torch.float64
    rng = np.random.default_rng(0)

    n, k = 50, 4
    num_forest_lines = 3
    z_qso = 3.0
    lya_rest = float(TW_phys[0]) * 1e8  # in Å (TW is in cm)
    rest = np.linspace(900.0, 1300.0, n)
    lya_1pz_np = (1.0 + z_qso) * rest / lya_rest

    y = torch.tensor(rng.normal(0, 0.3, size=n), dtype=DTYPE)
    nv = torch.tensor(np.full(n, 0.04), dtype=DTYPE)
    lya_1pz = torch.tensor(lya_1pz_np, dtype=DTYPE)
    M = torch.tensor(rng.normal(scale=2, size=(n, k)), dtype=DTYPE)
    omega2 = torch.tensor(np.full(n, 0.5), dtype=DTYPE)
    c_0, tau_0, beta = (
        torch.tensor(0.1, dtype=DTYPE),
        torch.tensor(0.00246, dtype=DTYPE),
        torch.tensor(3.62, dtype=DTYPE),
    )
    TW_t = torch.tensor(np.asarray(TW_phys), dtype=DTYPE)
    OS_t = torch.tensor(np.asarray(OS_phys), dtype=DTYPE)
    zqso_1pz = torch.tensor(1.0 + z_qso, dtype=DTYPE)

    _, _, _, _, _, dlog_beta_v1 = v1_loss(
        y, lya_1pz, nv, M, omega2, c_0, tau_0, beta,
        num_forest_lines, TW_t, OS_t, zqso_1pz,
    )
    _, _, _, _, _, dlog_beta_v35 = v35_loss(
        y, lya_1pz, nv, M, omega2, c_0, tau_0, beta,
        num_forest_lines, TW_t, OS_t, zqso_1pz,
    )
    diff = float(dlog_beta_v35) - float(dlog_beta_v1)

    # Closed-form chromatic correction at each pixel:
    # τ_k = τ_0 * (TW_k * OS_k / TW_α / OS_α) * (TW_α/TW_k * (1+z_α))^β * 𝟙_{k-forest}
    # chromatic_i = Σ_{k>1} τ_k(i) * log(TW_α / TW_k)
    TW_phys_arr = np.asarray(TW_phys)
    OS_phys_arr = np.asarray(OS_phys)
    chromatic = np.zeros(n)
    for k_idx in range(1, num_forest_lines):
        r_k = TW_phys_arr[0] / TW_phys_arr[k_idx]
        lyk_1pz = r_k * lya_1pz_np
        ind_k = (lyk_1pz <= 1.0 + z_qso).astype(np.float64)
        scale = float(tau_0) * TW_phys_arr[k_idx] * OS_phys_arr[k_idx] / (TW_phys_arr[0] * OS_phys_arr[0])
        tau_k = scale * lyk_1pz ** float(beta) * ind_k
        chromatic = chromatic + tau_k * np.log(r_k)

    # Now compute the closed-form difference contribution. We need
    # K_inv_y, diag_K_inv from v1's Woodbury — re-derive minimally.
    omega2_np = omega2.numpy().copy()
    M_np = M.numpy().copy()
    nv_np = nv.numpy().copy()
    y_np = y.numpy().copy()
    # τ_total under v1 (sum of all 3 lines)
    indicator = (lya_1pz_np <= 1.0 + z_qso).astype(np.float64)
    tau_lya = float(tau_0) * lya_1pz_np ** float(beta) * indicator
    tau_tot = tau_lya.copy()
    for k_idx in range(1, num_forest_lines):
        r_k = TW_phys_arr[0] / TW_phys_arr[k_idx]
        lyk_1pz = r_k * lya_1pz_np
        ind_k = (lyk_1pz <= 1.0 + z_qso).astype(np.float64)
        scale = float(tau_0) * TW_phys_arr[k_idx] * OS_phys_arr[k_idx] / (TW_phys_arr[0] * OS_phys_arr[0])
        tau_tot = tau_tot + scale * lyk_1pz ** float(beta) * ind_k
    lya_absorption = np.exp(-tau_tot)
    scaling = 1 - lya_absorption + float(c_0)
    absorption_noise = omega2_np * scaling ** 2
    d = nv_np + absorption_noise
    d_inv = 1.0 / d
    DiM = d_inv[:, None] * M_np
    B = M_np.T @ DiM + np.eye(k)
    Bi = np.linalg.inv(B)
    K_inv_y = d_inv * y_np - DiM @ (Bi @ (M_np.T @ (d_inv * y_np)))
    diag_K_inv = d_inv - np.einsum("ij,ji->i", DiM, Bi @ DiM.T)

    # Δ(da_β_v3.5 - da_β_v1) per pixel = ω² · scaling · exp(-τ) · chromatic · β · 𝟙
    delta_da_beta = (omega2_np * scaling * lya_absorption *
                      chromatic * float(beta) * indicator)
    closed_form_diff = float(
        -np.dot(K_inv_y * delta_da_beta, K_inv_y) + np.dot(diag_K_inv, delta_da_beta)
    )

    rel = abs(diff - closed_form_diff) / max(abs(closed_form_diff), 1e-12)
    assert rel < 1e-9, (
        f"v3.5 chromatic_correction does not match closed-form: "
        f"v3.5 - v1 = {diff:.6e}, closed_form = {closed_form_diff:.6e}, rel={rel:.2e}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
