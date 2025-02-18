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

def print_gpu_memory(prefix=""):
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # Convert to MB
    reserved = torch.cuda.memory_reserved(device) / 1024**2  # Convert to MB
    print(f"{prefix} | GPU {device}: Allocated {allocated:.2f} MB, Reserved {reserved:.2f} MB")

def objective(model, fluxes, lya_1pzs, noise_variances, num_forest_lines,
              all_transition_wavelengths, all_oscillator_strengths, z_qsos):
    """
    Computes the negative log-likelihood for the entire training dataset.

    Automatically supports multi-GPU through `DataParallel`, assuming model is already wrapped.
    """
    # # ✅ Ensure `model.module` is used inside DataParallel
    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module  # Extract actual model

    # ✅ Ensure model parameters are already on the correct device
    device = fluxes.device

    # ✅ Move all tensors to device ONCE (Avoid multiple `.to(device)` calls)
    all_transition_wavelengths = all_transition_wavelengths.to(device, non_blocking=True)
    all_oscillator_strengths = all_oscillator_strengths.to(device, non_blocking=True)

    # ✅ Move model parameters to GPU (Avoid repeated `.to(device)` calls)
    M = model.M.to(device, non_blocking=True)
    omega2 = torch.exp(2 * model.log_omega).to(device, non_blocking=True)
    c_0 = torch.exp(model.log_c_0).to(device, non_blocking=True)
    tau_0 = torch.exp(model.log_tau_0).to(device, non_blocking=True)
    beta = torch.exp(model.log_beta).to(device, non_blocking=True)

    print(f"Device: {device}, M: {M.device}, omega2: {omega2.device}, c_0: {c_0.device}, tau_0: {tau_0.device}, beta: {beta.device}")
    print("Fluxes Shapes: ", fluxes.shape)

    # ✅ Vectorized filtering: Get valid indices (NaN removal)
    valid_masks = ~torch.isnan(fluxes)

    # ✅ Batch-processing to avoid looping over each quasar
    batch_losses = torch.zeros(len(fluxes), device=device)
    dM_accum = torch.zeros_like(M, device=device)
    dlog_omega_accum = torch.zeros_like(model.log_omega, device=device)
    dlog_c_0_accum = torch.zeros_like(model.log_c_0, device=device)
    dlog_tau_0_accum = torch.zeros_like(model.log_tau_0, device=device)
    dlog_beta_accum = torch.zeros_like(model.log_beta, device=device)

    for i in range(len(fluxes)):  
        valid_mask = valid_masks[i]

        if valid_mask.sum() == 0:
            continue  # ✅ Skip fully NaN spectra

        nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta = spectrum_loss(
            fluxes[i, valid_mask], lya_1pzs[i, valid_mask], noise_variances[i, valid_mask], 
            M[valid_mask, :], omega2[valid_mask], c_0, tau_0, beta, 
            num_forest_lines, all_transition_wavelengths, all_oscillator_strengths, z_qsos[i]
        )

        batch_losses[i] = nlog_p
        dM_accum[valid_mask, :] += dM
        dlog_omega_accum += dlog_omega
        dlog_c_0_accum += dlog_c_0
        dlog_tau_0_accum += dlog_tau_0
        dlog_beta_accum += dlog_beta

    loss = batch_losses.sum()

    # ✅ Apply gradients manually
    model.M.grad = dM_accum
    model.log_omega.grad = dlog_omega_accum
    model.log_c_0.grad = dlog_c_0_accum
    model.log_tau_0.grad = dlog_tau_0_accum
    model.log_beta.grad = dlog_beta_accum

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

    # ✅ Analytical Gradients

    # Gradient w.r.t. M
    K_inv_M = D_inv_M - D_inv_M @ (C @ M)
    dM = -(K_inv_y[:, None] @ (K_inv_y[None, :] @ M) - K_inv_M)

    # Compute diag(K⁻¹) efficiently
    diag_K_inv = d_inv - torch.sum(C * D_inv_M.T, dim=0)

    # Gradient w.r.t. log ω
    dlog_omega = -(absorption_noise * (K_inv_y**2 - diag_K_inv))

    # Gradient w.r.t. log c₀
    da_c0 = c_0 * omega2 * scaling_factor
    dlog_c_0 = -(K_inv_y @ da_c0 - diag_K_inv @ da_c0)

    # Gradient w.r.t. log τ₀
    da_tau0 = omega2 * scaling_factor * lya_optical_depth * lya_absorption
    dlog_tau_0 = -(K_inv_y @ da_tau0 - diag_K_inv @ da_tau0)

    # Gradient w.r.t. log β
    da_beta = da_tau0 * torch.log(lya_1pz + 1e-6) * indicator
    dlog_beta = -(K_inv_y @ da_beta - diag_K_inv @ da_beta)

    return nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta
