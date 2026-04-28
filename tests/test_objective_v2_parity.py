"""Parity test: vectorized objective_v2 vs legacy spectrum_loss-loop.

Asserts that ``gpy_dla_detection.training.objective_v2.vectorized_nll``
produces the same loss and the **correct** gradients (i.e. the gradients
from autograd of the legacy ``spectrum_loss``-summed-in-a-Python-loop).

Layer 1 (``test_objective_math.py``) shows that the legacy
**analytical** ``dlog_beta`` formula matches autograd of legacy nlog_p
**only when all higher-order Lyman lines are masked above z_qso**.
When they aren't, the legacy analytical formula is approximate — it uses
``log(lya_1pz)`` for every line, but each Lyman line's true contribution
is ``log(lyman_1pz_i)``, which differs by ``log(λ_α/λ_i) ≈ 0.17–0.29``.

Both Python ``objective.py`` line 188 and DR16Q-public MATLAB
``spectrum_loss.m`` line 94 carry this same approximation, so Layer 4
parity passes byte-stably. v2 (autograd-based) computes the correct
gradient — so on inputs where the legacy formula's approximation is
non-trivial, **v2 will disagree with the legacy analytical dlog_beta by
design**. We therefore test v2 against autograd of legacy nlog_p (which
is correct), not against the legacy analytical formula.

Tested separately:
  - Likelihood-only loss (legacy spectrum_loss vs v2 vectorized_nll).
  - Likelihood-only gradients via autograd of legacy nlog_p vs autograd
    of v2 nlog_p.
  - Posterior gradients (Y1 prior added on v2 side; we add the
    prior-gradient block manually on the legacy side as objective.py does).

Pass criterion: rtol = atol = 1e-9 in float64.
"""

from __future__ import annotations

import math

import pytest
import torch

from gpy_dla_detection.objective import spectrum_loss
from gpy_dla_detection.training.objective_v2 import vectorized_nll
from gpy_dla_detection.voigt import (
    transition_wavelengths as TRANSITION_WAVELENGTHS_NP,
    oscillator_strengths as OSCILLATOR_STRENGTHS_NP,
)


DTYPE = torch.float64


def _make_batch(seed: int = 7, B: int = 8, n_pix: int = 64, k: int = 4,
                num_forest_lines: int = 3, nan_frac: float = 0.05):
    """Synthetic batch with realistic NaN-padded pixels."""
    g = torch.Generator().manual_seed(seed)

    fluxes = torch.randn(B, n_pix, generator=g, dtype=DTYPE) * 0.4
    noise_variances = 0.01 + 0.05 * torch.rand(B, n_pix, generator=g, dtype=DTYPE)

    # Inject NaN at random pixels.
    mask = torch.rand(B, n_pix, generator=g) < nan_frac
    fluxes[mask] = float("nan")
    noise_variances[mask] = float("nan")

    z_qsos = 2.5 + 1.0 * torch.rand(B, generator=g, dtype=DTYPE)  # (2.5, 3.5)

    # lya_1pz per pixel: assume rest_lambda 911..1216 mapped via observed = rest * (1+z_qso)
    rest_lambda = torch.linspace(911.0, 1216.0, n_pix, dtype=DTYPE)
    lya_wavelength = 1216.0
    one_plus_z_qso = (1.0 + z_qsos).unsqueeze(-1)
    lya_1pzs = 1.0 + (one_plus_z_qso * rest_lambda - lya_wavelength) / lya_wavelength

    # Hyperparameters
    M = torch.randn(n_pix, k, generator=g, dtype=DTYPE) * 0.1
    log_omega = torch.randn(n_pix, generator=g, dtype=DTYPE) * 0.3
    log_c_0 = torch.tensor(math.log(0.05), dtype=DTYPE)
    log_tau_0 = torch.tensor(math.log(0.0025), dtype=DTYPE)
    log_beta = torch.tensor(math.log(3.6), dtype=DTYPE)

    transition_wavelengths = torch.tensor(TRANSITION_WAVELENGTHS_NP, dtype=DTYPE)
    oscillator_strengths = torch.tensor(OSCILLATOR_STRENGTHS_NP, dtype=DTYPE)

    return dict(
        fluxes=fluxes, lya_1pzs=lya_1pzs, noise_variances=noise_variances,
        z_qsos=z_qsos,
        M=M, log_omega=log_omega, log_c_0=log_c_0, log_tau_0=log_tau_0, log_beta=log_beta,
        transition_wavelengths=transition_wavelengths,
        oscillator_strengths=oscillator_strengths,
        num_forest_lines=num_forest_lines,
        B=B, n_pix=n_pix, k=k,
    )


def _legacy_loop_loss_with_grads(inp):
    """Reproduce the per-spectrum sum from gpy_dla_detection.objective.objective,
    *without* the Y1 prior block. Returns (total_loss, autograd_grads_dict).

    The autograd path is the correct gradient (it backprops through the
    actual nlog_p formula, including per-line log(lyman_1pz_i) terms).
    Use this for v2 parity. The analytical formulas in spectrum_loss return
    an approximation for dlog_beta — see module docstring.
    """
    M = inp["M"].clone().detach().requires_grad_(True)
    log_omega = inp["log_omega"].clone().detach().requires_grad_(True)
    log_c_0 = inp["log_c_0"].clone().detach().requires_grad_(True)
    log_tau_0 = inp["log_tau_0"].clone().detach().requires_grad_(True)
    log_beta = inp["log_beta"].clone().detach().requires_grad_(True)

    omega2 = torch.exp(2 * log_omega)
    c_0 = torch.exp(log_c_0)
    tau_0 = torch.exp(log_tau_0)
    beta = torch.exp(log_beta)

    fluxes = inp["fluxes"]
    lya_1pzs = inp["lya_1pzs"]
    noise_variances = inp["noise_variances"]
    z_qsos = inp["z_qsos"]
    valid = torch.isfinite(fluxes) & torch.isfinite(noise_variances)

    total_loss = torch.zeros((), dtype=DTYPE)

    for b in range(inp["B"]):
        v = valid[b]
        if v.sum() == 0:
            continue
        nlog_p, *_ = spectrum_loss(
            fluxes[b, v], lya_1pzs[b, v], noise_variances[b, v],
            M[v, :], omega2[v], c_0, tau_0, beta,
            inp["num_forest_lines"],
            inp["transition_wavelengths"], inp["oscillator_strengths"],
            (z_qsos[b] + 1.0),
        )
        total_loss = total_loss + nlog_p

    # Autograd through legacy spectrum_loss is the CORRECT gradient.
    grads = torch.autograd.grad(total_loss, [M, log_omega, log_c_0, log_tau_0, log_beta])
    return dict(
        loss=total_loss.detach(),
        dM=grads[0], dlog_omega=grads[1],
        dlog_c_0=grads[2], dlog_tau_0=grads[3], dlog_beta=grads[4],
    )


def _v2(inp, apply_y1_prior: bool):
    """Run vectorized_nll and capture autograd gradients."""
    # Make leaf tensors with grad tracking.
    M = inp["M"].clone().detach().requires_grad_(True)
    log_omega = inp["log_omega"].clone().detach().requires_grad_(True)
    log_c_0 = inp["log_c_0"].clone().detach().requires_grad_(True)
    log_tau_0 = inp["log_tau_0"].clone().detach().requires_grad_(True)
    log_beta = inp["log_beta"].clone().detach().requires_grad_(True)

    loss = vectorized_nll(
        inp["fluxes"], inp["lya_1pzs"], inp["noise_variances"], inp["z_qsos"],
        M, log_omega, log_c_0, log_tau_0, log_beta,
        inp["transition_wavelengths"], inp["oscillator_strengths"],
        num_forest_lines=inp["num_forest_lines"],
        apply_y1_prior=apply_y1_prior,
    )
    grads = torch.autograd.grad(loss, [M, log_omega, log_c_0, log_tau_0, log_beta])
    return dict(
        loss=loss.detach(), dM=grads[0], dlog_omega=grads[1],
        dlog_c_0=grads[2], dlog_tau_0=grads[3], dlog_beta=grads[4],
    )


# ---------------------------------------------------------------------------
# Test 2.1 — likelihood-only parity (no Y1 prior on either side)
# ---------------------------------------------------------------------------
def test_v2_likelihood_only_parity_loss():
    inp = _make_batch(seed=11)
    legacy = _legacy_loop_loss_with_grads(inp)
    v2 = _v2(inp, apply_y1_prior=False)

    diff = (legacy["loss"] - v2["loss"]).abs().item()
    assert diff < 1e-9, (
        f"Likelihood-only loss disagrees: legacy={legacy['loss'].item():.10e}, "
        f"v2={v2['loss'].item():.10e}, |Δ|={diff:.3e}"
    )


@pytest.mark.parametrize("which", ["dM", "dlog_omega", "dlog_c_0", "dlog_tau_0", "dlog_beta"])
def test_v2_likelihood_only_parity_grads(which):
    inp = _make_batch(seed=12)
    legacy = _legacy_loop_loss_with_grads(inp)
    v2 = _v2(inp, apply_y1_prior=False)

    a = legacy[which]
    b = v2[which]
    assert torch.allclose(a, b, rtol=1e-9, atol=1e-12), (
        f"{which}: legacy vs v2 max |Δ| = {(a - b).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Test 2.2 — posterior parity: gradients match (legacy gradient block adds Y1
# prior); legacy LOSS excludes the prior — that's a legacy quirk we don't
# replicate. Test only the gradients here.
# ---------------------------------------------------------------------------
def _legacy_with_prior_grads(inp):
    """Add the analytic Y1 prior gradient block (matches the chain-rule formula
    used in gpy_dla_detection.objective.objective lines 70-71)."""
    base = _legacy_loop_loss_with_grads(inp)
    tau_0 = torch.exp(inp["log_tau_0"])
    beta = torch.exp(inp["log_beta"])

    tau_0_mu, tau_0_sigma = 0.00246, 0.00014
    beta_mu, beta_sigma = 3.62, 0.04

    base["dlog_tau_0"] = base["dlog_tau_0"] + (tau_0 - tau_0_mu) / tau_0_sigma ** 2 * tau_0
    base["dlog_beta"] = base["dlog_beta"] + (beta - beta_mu) / beta_sigma ** 2 * beta
    return base


@pytest.mark.parametrize("which", ["dM", "dlog_omega", "dlog_c_0", "dlog_tau_0", "dlog_beta"])
def test_v2_posterior_parity_grads(which):
    inp = _make_batch(seed=13)
    legacy = _legacy_with_prior_grads(inp)
    v2 = _v2(inp, apply_y1_prior=True)

    a = legacy[which]
    b = v2[which]
    assert torch.allclose(a, b, rtol=1e-9, atol=1e-12), (
        f"{which} (posterior path): legacy vs v2 max |Δ| = {(a - b).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Test 2.3 — full-NaN spectrum is dropped (loss contribution = 0)
# ---------------------------------------------------------------------------
def test_v2_handles_all_nan_spectrum():
    """A spectrum with all-NaN flux must contribute exactly zero to the loss
    (matches the legacy `if valid_mask.sum() == 0: continue` skip)."""
    inp = _make_batch(seed=14, B=4)
    # Force spectrum 1 to be entirely NaN.
    inp["fluxes"][1, :] = float("nan")
    inp["noise_variances"][1, :] = float("nan")

    legacy = _legacy_loop_loss_with_grads(inp)
    v2 = _v2(inp, apply_y1_prior=False)

    assert torch.allclose(legacy["loss"], v2["loss"], rtol=1e-9, atol=1e-12)
    # And v2 grads must not have NaN propagated from the all-NaN row.
    for key in ["dM", "dlog_omega", "dlog_c_0", "dlog_tau_0", "dlog_beta"]:
        assert torch.all(torch.isfinite(v2[key])), f"NaN/inf leaked into v2[{key}]"
