"""
Objective function for the GP-DLA model.

Effective optical depth for DESI Y1 (https://arxiv.org/abs/2405.06743):
 τ(z)=τ0(1+z)^γ t
 τ0=(2.46±0.14)×10−3
 γ=3.62±0.04
"""

import torch
import numpy as np
from .voigt import transition_wavelengths, oscillator_strengths

def objective(model, fluxes, lya_1pzs, noise_variances, num_forest_lines,
              all_transition_wavelengths, all_oscillator_strengths, z_qsos):
    """
    Computes the negative log-likelihood for the entire training dataset.

    Automatically supports multi-GPU through `DataParallel`, assuming model is already wrapped.
    """
    # ✅ Ensure `model.module` is used inside DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # ✅ Extract model for parameter access

    # ✅ Extract learnable parameters correctly
    M = model.M.to(fluxes.device)  # Move to correct device
    omega2 = torch.exp(2 * model.log_omega).to(fluxes.device)
    c_0 = torch.exp(model.log_c_0).to(fluxes.device)
    tau_0 = torch.exp(model.log_tau_0).to(fluxes.device)
    beta = torch.exp(model.log_beta).to(fluxes.device)

    # ✅ Ensure all tensors are on the correct GPU
    device = fluxes.device
    all_transition_wavelengths = all_transition_wavelengths.to(device)
    all_oscillator_strengths = all_oscillator_strengths.to(device)

    print(f"Device: {device}, M: {M.device}, omega2: {omega2.device}, c_0: {c_0.device}, tau_0: {tau_0.device}, beta: {beta.device}")

    # ✅ Vectorized filtering: Get valid indices (NaN removal)
    valid_masks = ~torch.isnan(fluxes)

    # ✅ Compute loss for all quasars in parallel
    losses = torch.stack([
        spectrum_loss(fluxes[i, valid_masks[i]], lya_1pzs[i, valid_masks[i]], noise_variances[i, valid_masks[i]], 
                      M[valid_masks[i], :], omega2[valid_masks[i]], c_0, tau_0, beta, 
                      num_forest_lines, all_transition_wavelengths, all_oscillator_strengths, z_qsos[i])
        for i in range(len(fluxes))  # ✅ Vectorized computation per quasar
    ])

    # ✅ Sum losses across the batch
    loss = losses.sum().to(device)  # ✅ Ensure it’s on GPU

    return loss

def spectrum_loss(y, lya_1pz, noise_variance, M, omega2, c_0, tau_0, beta,
                  num_forest_lines, all_transition_wavelengths, all_oscillator_strengths, zqso_1pz):
    """
    Computes the negative log-likelihood of a single spectrum.
    """
    num_forest_lines = max(2, min(num_forest_lines, len(all_transition_wavelengths)))

    # ✅ Vectorized Optical Depth Computation
    lya_optical_depth = tau_0 * torch.pow(lya_1pz, beta)

    lyman_1pz = (all_transition_wavelengths[0] * lya_1pz[:, None]) / all_transition_wavelengths[1:num_forest_lines]
    indicator = (lyman_1pz <= zqso_1pz[:, None]).float()

    tau = (tau_0 * all_transition_wavelengths[1:num_forest_lines] * all_oscillator_strengths[1:num_forest_lines]) / \
          (all_transition_wavelengths[0] * all_oscillator_strengths[0])

    # ✅ Compute final optical depth efficiently
    lya_optical_depth += torch.sum(tau * torch.pow(lyman_1pz, beta) * indicator, dim=1)

    # ✅ Absorption effects
    lya_absorption = torch.exp(-lya_optical_depth)
    scaling_factor = 1 - lya_absorption + c_0
    absorption_noise = omega2 * torch.square(scaling_factor)

    # ✅ Avoid division errors
    d = noise_variance + absorption_noise + 1e-6
    d_inv = 1 / d

    # ✅ Compute covariance inverse
    D_inv_y = d_inv * y
    D_inv_M = d_inv[:, None] * M

    B = M.T @ D_inv_M
    B.diagonal().add_(1e-6)  # ✅ Stability improvement

    # ✅ Cholesky Decomposition (Try-Except for Robustness)
    try:
        L = torch.linalg.cholesky(B)
    except RuntimeError as e:
        print(f"Cholesky failed: {e}")
        min_eigval = torch.min(torch.linalg.eigvalsh(B))
        print(f"Min eigenvalue of B: {min_eigval}")
        raise

    # ✅ Solve for C Matrix
    C = torch.linalg.solve_triangular(L, D_inv_M.T, upper=False)
    C = torch.linalg.solve_triangular(L.T, C, upper=True)

    # ✅ Compute Final Loss
    K_inv_y = D_inv_y - D_inv_M @ (C @ y)
    log_det_K = torch.sum(torch.log(d)) + 2 * torch.sum(torch.log(torch.diag(L)))

    # ✅ Ensure loss has correct shape
    nlog_p = 0.5 * ((y @ K_inv_y).view(-1) + log_det_K + len(y) * torch.log(torch.tensor(2 * np.pi, dtype=torch.float32)))

    # ✅ Apply Gaussian Prior
    prior_loss = 0.5 * ((tau_0 - 0.00246) ** 2 / 0.14 ** 2 + (beta - 3.62) ** 2 / 0.04 ** 2)
    nlog_p = nlog_p + prior_loss.view(-1)

    return nlog_p.view(-1)  # ✅ Fix shape mismatch issue for DataParallel
