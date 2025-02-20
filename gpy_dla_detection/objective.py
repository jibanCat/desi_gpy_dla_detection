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
    device = fluxes.device

    # ✅ Move model parameters to the correct device
    M = model.M.to(device, non_blocking=True)  # (num_pixels, k)
    omega2 = torch.exp(2 * model.log_omega).to(device, non_blocking=True)  # (num_pixels,)
    c_0 = torch.exp(model.log_c_0).to(device, non_blocking=True)  # (scalar)
    tau_0 = torch.exp(model.log_tau_0).to(device, non_blocking=True)  # (scalar)
    beta = torch.exp(model.log_beta).to(device, non_blocking=True)  # (scalar)

    # ✅ Initialize accumulators (matching MATLAB structure)
    total_loss = torch.tensor(0.0, device=device)
    dM_accum = torch.zeros_like(M, device=device)  # (num_pixels, k)
    dlog_omega_accum = torch.zeros_like(model.log_omega, device=device)  # (num_pixels,)
    dlog_c_0_accum = torch.tensor(0.0, device=device)
    dlog_tau_0_accum = torch.tensor(0.0, device=device)
    dlog_beta_accum = torch.tensor(0.0, device=device)

    # ✅ Loop over each spectrum (vectorized NaN handling)
    valid_masks = ~torch.isnan(fluxes)  # (batch_size, num_pixels)

    for i in range(len(fluxes)):  
        valid_mask = valid_masks[i]

        if valid_mask.sum() == 0:
            continue  # Skip spectra that are completely NaN

        # ✅ Call spectrum_loss on valid pixels only
        nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta = spectrum_loss(
            fluxes[i, valid_mask], lya_1pzs[i, valid_mask], noise_variances[i, valid_mask], 
            M[valid_mask, :], omega2[valid_mask], c_0, tau_0, beta, 
            num_forest_lines, all_transition_wavelengths, all_oscillator_strengths, z_qsos[i]
        )

        # ✅ Accumulate results correctly (matching MATLAB)
        total_loss += nlog_p.detach()  # (scalar)
        dM_accum[valid_mask, :] += dM.detach()
        dlog_omega_accum[valid_mask] += dlog_omega.detach()
        dlog_c_0_accum += dlog_c_0.detach()
        dlog_tau_0_accum += dlog_tau_0.detach()
        dlog_beta_accum += dlog_beta.detach()

    print(f"dlog_omega shape: {dlog_omega.shape}, dlog_omega_accum shape: {dlog_omega_accum.shape}")

    # ✅ Apply accumulated gradients to the model
    model.M.grad = dM_accum
    model.log_omega.grad = dlog_omega_accum
    model.log_c_0.grad = dlog_c_0_accum
    model.log_tau_0.grad = dlog_tau_0_accum
    model.log_beta.grad = dlog_beta_accum

    del dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta
    return total_loss

def spectrum_loss(y, lya_1pz, noise_variance, M, omega2, c_0, tau_0, beta,
                  num_forest_lines, all_transition_wavelengths, all_oscillator_strengths, zqso_1pz):
    """
    Computes the negative log-likelihood and gradients for a single spectrum.
    """
    log_2pi = 1.83787706640934534  # log(2π)

    n, k = M.shape  # (n, k) = (num_pixels, num_latent_components)
    
    # ✅ Compute approximate Lyα optical depth
    lya_optical_depth = tau_0 * torch.pow(lya_1pz, beta)  # (n,)

    # ✅ Apply indicator mask (only consider pixels within quasar redshift)
    indicator = (lya_1pz <= zqso_1pz).float()  # (n,)
    lya_optical_depth *= indicator  # (n,)

    # ✅ Compute Lyman series optical depth using scaling relationships
    for i in range(1, num_forest_lines):
        lyman_1pz = (all_transition_wavelengths[0] * lya_1pz) / all_transition_wavelengths[i]  # (n,)
        lyman_indicator = (lyman_1pz <= zqso_1pz).float()  # (n,)
        lyman_1pz *= lyman_indicator  # (n,)

        tau = (tau_0 * all_transition_wavelengths[i] * all_oscillator_strengths[i]) / \
              (all_transition_wavelengths[0] * all_oscillator_strengths[0])  # (scalar)

        lya_optical_depth += tau * torch.pow(lyman_1pz, beta)  # (n,)

    # ✅ Compute approximate absorption due to Lyα and Lyman series
    lya_absorption = torch.exp(-lya_optical_depth)  # (n,)

    # ✅ Compute "absorption noise" contribution
    scaling_factor = 1 - lya_absorption + c_0  # (n,)
    absorption_noise = omega2 * scaling_factor ** 2  # (n,)

    # ✅ Compute total variance (including instrumental noise)
    d = noise_variance + absorption_noise + 1e-6  # (n,)
    d_inv = 1.0 / d  # (n,)

    # ✅ Compute inverse terms
    D_inv_y = d_inv * y  # (n,)
    D_inv_M = d_inv[:, None] * M  # (n, k)

    # ✅ Compute covariance matrix using Woodbury identity
    B = M.T @ D_inv_M  # (k, k)
    B.diagonal().add_(1.0)  # (k, k) → adding 1 to diagonal for stability

    # ✅ Perform Cholesky decomposition for numerical stability
    L = torch.linalg.cholesky(B)  # (k, k)

    # ✅ Compute inverse of B using Cholesky
    C = torch.cholesky_solve(D_inv_M.T, L) # (k, n)

    # ✅ Compute K⁻¹ y
    C_y = C @ y.unsqueeze(-1)  # (k, n) @ (n, 1) → should result in (k, 1)
    K_inv_y = D_inv_y - torch.matmul(D_inv_M, C_y).view(-1)  # (n,) - ((n, k) @ (k, 1)).squeeze(-1) → (n,)

    # ✅ Compute log determinant of K
    log_det_K = torch.sum(torch.log(d)) + 2 * torch.sum(torch.log(torch.diagonal(L)))  # (scalar)

    # ✅ Compute negative log-likelihood
    nlog_p = 0.5 * (y @ K_inv_y + log_det_K + n * log_2pi)  # (scalar)

    # ✅ Compute gradients analytically
    # Compute inverse covariance terms    
    print(f"D_inv_M shape: {D_inv_M.shape}, C shape: {C.shape}, M shape: {M.shape}")

    tmp = C @ M  # (k, k)
    K_inv_M = D_inv_M - torch.matmul(D_inv_M, tmp)  # Explicit torch.matmul
    del tmp  # Free memory immediately    # K_inv_M = D_inv_M - (D_inv_M @ C @ M)  # (n, k)

    # Gradient wrt M
    dM = -(K_inv_y[:, None] @ (K_inv_y[None, :] @ M) - K_inv_M)  # (n, k)

    # Compute diag K⁻¹ efficiently (without full product)
    diag_K_inv = d_inv - torch.sum(C * D_inv_M.T, dim=0)  # (n,)

    # Gradient wrt log ω
    dlog_omega = -(absorption_noise * (K_inv_y ** 2 - diag_K_inv))  # (n,)

    # Gradient wrt log c₀
    da_c0 = c_0 * omega2 * scaling_factor  # (n,)
    dlog_c_0 = -(K_inv_y @ da_c0 - diag_K_inv @ da_c0)  # (scalar)

    # Gradient wrt log τ₀
    da_tau0 = omega2 * scaling_factor * lya_optical_depth * lya_absorption  # (n,)
    dlog_tau_0 = -(K_inv_y @ da_tau0 - diag_K_inv @ da_tau0)  # (scalar)

    # Gradient wrt log β
    da_beta = da_tau0 * torch.log(lya_1pz + 1e-6) * indicator  # (n,)
    dlog_beta = -(K_inv_y @ da_beta - diag_K_inv @ da_beta)  # (scalar)

    return nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta