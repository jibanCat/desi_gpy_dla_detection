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

    Equivalent to MATLAB's `objective.m`, with automatic gradient tracking via PyTorch autograd.
    """
    # Extract learnable parameters from the model
    M, omega2, c_0, tau_0, beta = model()

    # Initialize total loss
    loss = torch.tensor(0.0, dtype=torch.float32, device=M.device)  # ✅ Ensure on correct device
    device = M.device  # Get device

    # Iterate over all quasars in training set
    for i in range(len(fluxes)):
        valid_idx = ~torch.isnan(fluxes[i])  # Remove NaNs
        valid_idx = valid_idx.to(device)  # ✅ Move mask to correct device

        y = fluxes[i, valid_idx].to(device)  # ✅ Move indexed tensor to same device
        noise_var = noise_variances[i, valid_idx].to(device)
        lya_1pz = lya_1pzs[i, valid_idx].to(device)  # ✅ Fix indexing device mismatch
        zqso_1pz = z_qsos[i].to(device) + 1  # ✅ Move `z_qsos` to device
        M_valid = M[valid_idx, :].to(device)  # ✅ Fix: Correct 2D selection

        # Compute per-spectrum likelihood
        this_loss = spectrum_loss(y, lya_1pz, noise_var, M_valid, omega2[valid_idx].to(device),
                                  c_0, tau_0, beta, num_forest_lines,
                                  all_transition_wavelengths.to(device), all_oscillator_strengths.to(device),
                                  zqso_1pz)

        loss = loss + this_loss  # ✅ No in-place operation

    return loss  # ✅ Return accumulated loss tensor
def spectrum_loss(y, lya_1pz, noise_variance, M, omega2, c_0, tau_0, beta,
                  num_forest_lines, all_transition_wavelengths, all_oscillator_strengths, zqso_1pz):
    """
    Computes the negative log-likelihood of a single spectrum.
    """
    # ✅ Ensure `num_forest_lines` is valid
    num_forest_lines = max(2, min(num_forest_lines, len(all_transition_wavelengths)))

    # Compute the initial optical depth for Lyα
    lya_optical_depth = tau_0 * torch.pow(lya_1pz, beta)

    # ✅ Vectorized Lyman Series Computation
    lyman_1pz = (all_transition_wavelengths[0] * lya_1pz[:, None]) / all_transition_wavelengths[1:num_forest_lines]

    # Prevent division errors
    lyman_1pz = torch.where(all_transition_wavelengths[1:num_forest_lines] == 0, 
                            torch.tensor(1.0, dtype=lya_1pz.dtype, device=lya_1pz.device), 
                            lyman_1pz)

    # Compute the indicator function for valid absorbers
    indicator = (lyman_1pz <= zqso_1pz[:, None]).float()

    # ✅ Ensure `tau` is properly defined
    if num_forest_lines > 1:
        tau = (tau_0 * all_transition_wavelengths[1:num_forest_lines] * all_oscillator_strengths[1:num_forest_lines]) / \
              (all_transition_wavelengths[0] * all_oscillator_strengths[0])
    else:
        tau = torch.tensor([0.0], dtype=lya_1pz.dtype, device=lya_1pz.device)

    # Compute the final optical depth in a single vectorized operation
    lya_optical_depth += torch.sum(tau * torch.pow(lyman_1pz, beta) * indicator, dim=1)    

    # Compute absorption noise
    lya_absorption = torch.exp(-lya_optical_depth)
    scaling_factor = 1 - lya_absorption + c_0
    absorption_noise = omega2 * torch.square(scaling_factor)

    # Prevent division errors
    d = noise_variance + absorption_noise + 1e-6
    d_inv = 1 / d

    # Compute inverse covariance
    D_inv_y = d_inv * y
    D_inv_M = d_inv[:, None] * M

    B = M.T @ D_inv_M
    B.diagonal().add_(1e-6)  # ✅ Better numerical stability

    # ✅ Cholesky decomposition (Robust)
    try:
        L = torch.linalg.cholesky(B)
    except RuntimeError as e:
        print(f"Cholesky failed: {e}")
        min_eigval = torch.min(torch.linalg.eigvalsh(B))
        print(f"Min eigenvalue of B: {min_eigval}")
        raise

    # Compute C matrix
    C = torch.linalg.solve_triangular(L, D_inv_M.T, upper=False)
    C = torch.linalg.solve_triangular(L.T, C, upper=True)

    K_inv_y = D_inv_y - D_inv_M @ (C @ y)
    log_det_K = torch.sum(torch.log(d)) + 2 * torch.sum(torch.log(torch.diag(L)))

    # ✅ Ensure `nlog_p` has correct shape
    nlog_p = 0.5 * ((y @ K_inv_y).view(-1) + log_det_K + len(y) * torch.log(torch.tensor(2 * np.pi, dtype=torch.float32)))

    # ✅ Ensure Gaussian prior is batch-compatible
    prior_loss = 0.5 * ((tau_0 - 0.00246) ** 2 / 0.14 ** 2 + (beta - 3.62) ** 2 / 0.04 ** 2)
    nlog_p = nlog_p + prior_loss.view(-1)

    return nlog_p  # ✅ Ensures correct shape for DataParallel