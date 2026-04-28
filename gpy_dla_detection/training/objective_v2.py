"""Vectorized NLL across a batch of spectra (vs the legacy per-spectrum loop).

The legacy ``gpy_dla_detection.objective.objective`` iterates over every
spectrum in a batch in pure Python and manually accumulates analytical
gradients into ``model.<param>.grad``. Layer 1 confirmed those analytical
gradients match ``torch.autograd`` to 1e-9, so we can drop the manual
accumulation: a vectorized forward pass with ``loss.backward()`` is
mathematically identical *and* eliminates the Python-loop bottleneck
that Layer 3 measured at ~25 % of CPU time on a 128-spectrum batch.

This implementation:

  - Pads NaN pixels with finite sentinels and carries a ``(B, n_pix)``
    ``valid_mask``. Invalid pixels contribute 0 to the loss via
    ``d_inv = 0`` there.
  - Uses ``torch.einsum`` and ``torch.bmm`` for matrix products and
    ``torch.linalg.cholesky`` / ``torch.cholesky_solve`` on stacked
    ``(B, k, k)`` tensors.
  - Returns a single scalar ``nll`` (sum over batch). The optional
    Y1 Gaussian prior on ``(τ₀, β)`` is added to ``nll`` so
    ``loss.backward()`` produces the *posterior* gradient — unlike the
    legacy code, which optimizes prior-augmented gradients but reports
    likelihood-only loss.

The legacy ``objective`` and ``spectrum_loss`` are *not* changed.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


# Default DESI Y1 Gaussian prior (Turner+2024) — same values as
# ``gpy_dla_detection/objective.py`` lines 64–67.
DEFAULT_TAU_0_PRIOR_MU = 0.00246
DEFAULT_TAU_0_PRIOR_SIGMA = 0.00014
DEFAULT_BETA_PRIOR_MU = 3.62
DEFAULT_BETA_PRIOR_SIGMA = 0.04


def vectorized_nll(
    fluxes: torch.Tensor,
    lya_1pzs: torch.Tensor,
    noise_variances: torch.Tensor,
    z_qsos: torch.Tensor,
    M: torch.Tensor,
    log_omega: torch.Tensor,
    log_c_0: torch.Tensor,
    log_tau_0: torch.Tensor,
    log_beta: torch.Tensor,
    transition_wavelengths: torch.Tensor,
    oscillator_strengths: torch.Tensor,
    *,
    num_forest_lines: int = 3,
    apply_y1_prior: bool = True,
    tau_0_prior_mu: float = DEFAULT_TAU_0_PRIOR_MU,
    tau_0_prior_sigma: float = DEFAULT_TAU_0_PRIOR_SIGMA,
    beta_prior_mu: float = DEFAULT_BETA_PRIOR_MU,
    beta_prior_sigma: float = DEFAULT_BETA_PRIOR_SIGMA,
) -> torch.Tensor:
    """Compute the (posterior, by default) NLL over a batch of spectra.

    Parameters
    ----------
    fluxes, lya_1pzs, noise_variances : Tensor of shape (B, n_pix)
        Per-spectrum centered flux, (1+z_lya) at each pixel, and pipeline
        noise variance. NaN entries (in either ``fluxes`` or
        ``noise_variances``) are treated as masked.
    z_qsos : Tensor of shape (B,)
        QSO redshifts.
    M : Tensor of shape (n_pix, k)
        Low-rank emission basis.
    log_omega : Tensor of shape (n_pix,)
        Per-pixel "absorption noise" amplitude (``omega^2 = exp(2*log_omega)``).
    log_c_0, log_tau_0, log_beta : 0-dim Tensors
        Scalar GP hyperparameters in log space.
    transition_wavelengths, oscillator_strengths : Tensor (≥ num_forest_lines)
        Lyman-series constants. Same as the existing ``voigt`` module.
    num_forest_lines : int
        Number of Lyman series lines (Lyα + ... + Ly-num_forest_lines).
    apply_y1_prior : bool
        If True, add ``-log p_prior(τ₀, β)`` to the loss. The prior is the
        same DESI Y1 Gaussian used by the legacy ``objective.py``.
    tau_0_prior_mu, tau_0_prior_sigma, beta_prior_mu, beta_prior_sigma : float
        Override the prior parameters if needed.

    Returns
    -------
    Tensor (scalar)
        Sum of per-spectrum NLL over the batch (plus optional Y1 prior).
        Differentiable through PyTorch's autograd graph w.r.t. all five
        learnable parameters (M, log_omega, log_c_0, log_tau_0, log_beta).

    Notes
    -----
    - For invalid (NaN) pixels we sanitize ``fluxes -> 0``, ``lya_1pz -> 1``,
      ``noise_variances -> 1`` and zero the corresponding ``d_inv`` so they
      contribute zero to the loss. ``log|K|`` likewise excludes them.
    - The ``num_forest_lines`` defaults to 3 (Lyα + Lyβ + Lyγ), matching
      the production training configuration.
    """
    if fluxes.dim() != 2:
        raise ValueError(f"fluxes must be (B, n_pix), got shape {tuple(fluxes.shape)}")
    if M.dim() != 2:
        raise ValueError(f"M must be (n_pix, k), got shape {tuple(M.shape)}")
    if fluxes.shape[1] != M.shape[0]:
        raise ValueError(
            f"fluxes.shape[1]={fluxes.shape[1]} must equal M.shape[0]={M.shape[0]}"
        )

    B, n_pix = fluxes.shape
    n_pix_M, k = M.shape
    dtype = fluxes.dtype
    device = fluxes.device

    # Validity mask: pixels with finite flux AND finite noise variance.
    valid_mask = torch.isfinite(fluxes) & torch.isfinite(noise_variances)
    valid_f = valid_mask.to(dtype)

    # Sanitize masked positions so the math doesn't propagate NaN/inf.
    y = torch.where(valid_mask, fluxes, torch.zeros_like(fluxes))
    lya_1pz = torch.where(valid_mask, lya_1pzs, torch.ones_like(lya_1pzs))
    nv = torch.where(valid_mask, noise_variances, torch.ones_like(noise_variances))

    # Hyperparameters from log-space.
    omega2 = torch.exp(2 * log_omega)        # (n_pix,)
    c_0 = torch.exp(log_c_0)                 # ()
    tau_0 = torch.exp(log_tau_0)             # ()
    beta = torch.exp(log_beta)               # ()

    zqso_1pz = (1.0 + z_qsos).to(dtype).unsqueeze(-1)   # (B, 1)

    # Lyman-series mean optical depth at each pixel, per spectrum.
    # Important: we must NOT compute pow(0, β). For masked pixels we leave
    # the underlying ``lya_1pz`` / ``lyman_1pz`` values strictly positive
    # (they already are — they're 1 + z_lya >= 1) and multiply the result
    # of pow(...) by the indicator. Otherwise, multiplying by the indicator
    # *before* pow gives a 0 base, and autograd evaluates the local
    # derivative as 0 * log(0) = NaN, which contaminates dlog_beta.
    indicator_lya = (lya_1pz <= zqso_1pz).to(dtype)             # (B, n_pix)
    tau_optical_depth = tau_0 * lya_1pz.pow(beta) * indicator_lya
    if num_forest_lines > 1:
        tw0 = transition_wavelengths[0]
        os0 = oscillator_strengths[0]
        for i in range(1, num_forest_lines):
            lyman_1pz = tw0 * lya_1pz / transition_wavelengths[i]
            lyman_indicator = (lyman_1pz <= zqso_1pz).to(dtype)
            tau_i = (
                tau_0 * transition_wavelengths[i] * oscillator_strengths[i]
                / (tw0 * os0)
            )
            # Note: lyman_1pz stays strictly positive here (we do NOT multiply
            # it by lyman_indicator before pow). The mask is applied via
            # multiplication AFTER the pow.
            tau_optical_depth = tau_optical_depth + tau_i * lyman_1pz.pow(beta) * lyman_indicator

    lya_absorption = torch.exp(-tau_optical_depth)          # (B, n_pix)
    scaling_factor = 1.0 - lya_absorption + c_0             # (B, n_pix)
    absorption_noise = omega2.unsqueeze(0) * scaling_factor.pow(2)  # (B, n_pix)
    d = nv + absorption_noise                                # (B, n_pix)

    # d_inv with invalid pixels zeroed.
    d_inv = torch.where(valid_mask, 1.0 / d, torch.zeros_like(d))   # (B, n_pix)

    # B_batched[b] = I_k + M.T @ diag(d_inv[b]) @ M
    # Using einsum: (n_pix,k) x (B,n_pix) x (n_pix,k) -> (B,k,k)
    # = sum_i d_inv[b,i] * M[i,j] * M[i,l]
    B_batched = torch.einsum("ij,bi,il->bjl", M, d_inv, M)   # (B, k, k)
    B_batched = B_batched + torch.eye(k, device=device, dtype=dtype).unsqueeze(0)

    # Cholesky and solve, batched.
    L = torch.linalg.cholesky(B_batched)                     # (B, k, k)

    # K_inv_y[b] = d_inv[b] * y[b] - D_inv_M[b] @ B[b]^-1 @ D_inv_M[b].T @ y[b]
    # D_inv_M[b].T @ y[b] = sum_i d_inv[b,i] * M[i,j] * y[b,i] = einsum
    Mty = torch.einsum("ij,bi->bj", M, d_inv * y)            # (B, k)
    z_solve = torch.cholesky_solve(Mty.unsqueeze(-1), L).squeeze(-1)   # (B, k)
    # D_inv_M[b] @ z_solve[b] elementwise: sum_j d_inv[b,i] M[i,j] z[b,j]
    correction = d_inv * torch.einsum("ij,bj->bi", M, z_solve)  # (B, n_pix)
    K_inv_y = d_inv * y - correction                            # (B, n_pix)

    # log|K| per spectrum.
    # log|K[b]| = sum_{i in valid} log(d[b,i]) + 2*sum log(diag(L[b]))
    log_d_safe = torch.where(valid_mask, torch.log(d), torch.zeros_like(d))
    log_det_K = log_d_safe.sum(dim=-1) + 2 * torch.log(
        torch.diagonal(L, dim1=-2, dim2=-1)
    ).sum(dim=-1)                                              # (B,)

    # Quadratic form per spectrum.
    yKy = (y * K_inv_y).sum(dim=-1)                            # (B,)

    # Effective n per spectrum (number of valid pixels).
    n_valid = valid_f.sum(dim=-1)                              # (B,)

    log_2pi = math.log(2.0 * math.pi)
    nll_per_spectrum = 0.5 * (yKy + log_det_K + n_valid * log_2pi)   # (B,)

    nll = nll_per_spectrum.sum()

    if apply_y1_prior:
        # -log p(τ₀) = 0.5 * (τ₀ - μ)² / σ² (up to const). Same for β.
        nll = nll + 0.5 * (tau_0 - tau_0_prior_mu) ** 2 / tau_0_prior_sigma ** 2
        nll = nll + 0.5 * (beta - beta_prior_mu) ** 2 / beta_prior_sigma ** 2

    return nll
