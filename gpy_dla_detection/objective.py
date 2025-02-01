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
    loss = torch.tensor(0.0, dtype=torch.float32, device=M.device)  # ✅ Fix: No requires_grad=True

    # Iterate over all quasars in training set
    for i in range(len(fluxes)):
        valid_idx = ~torch.isnan(fluxes[i])  # Remove NaNs
        y = fluxes[i, valid_idx]
        noise_var = noise_variances[i, valid_idx]
        lya_1pz = lya_1pzs[i, valid_idx]
        zqso_1pz = z_qsos[i] + 1  # Redshift factor for Lyα absorbers
        M_valid = M[valid_idx, :]  # Fix: Correct 2D selection

        # Compute per-spectrum likelihood
        this_loss = spectrum_loss(y, lya_1pz, noise_var, M_valid, omega2[valid_idx],
                                  c_0, tau_0, beta, num_forest_lines,
                                  all_transition_wavelengths, all_oscillator_strengths, zqso_1pz)

        loss = loss + this_loss  # ✅ Fix: No in-place operation

    return loss  # ✅ Return the accumulated loss tensor

def spectrum_loss(y, lya_1pz, noise_variance, M, omega2, c_0, tau_0, beta,
                  num_forest_lines, all_transition_wavelengths, all_oscillator_strengths, zqso_1pz):
    """
    Computes the negative log-likelihood of a single spectrum.

    PyTorch automatically tracks gradients, so we do NOT need to compute them manually.
    """

    # Compute Lyman absorption effects
    lya_optical_depth = tau_0 * torch.pow(lya_1pz, beta)

    for i in range(1, num_forest_lines):
        lyman_1pz = (all_transition_wavelengths[0] * lya_1pz) / all_transition_wavelengths[i]
        indicator = (lyman_1pz <= zqso_1pz).float()

        tau = (tau_0 * all_transition_wavelengths[i] * all_oscillator_strengths[i]) / \
              (all_transition_wavelengths[0] * all_oscillator_strengths[0])

        lya_optical_depth += tau * torch.pow(lyman_1pz, beta) * indicator

    lya_absorption = torch.exp(-lya_optical_depth)

    # Compute absorption noise
    scaling_factor = 1 - lya_absorption + c_0
    absorption_noise = omega2 * torch.square(scaling_factor)

    # Compute total noise variance
    d = noise_variance + absorption_noise
    d_inv = 1 / d

    # Compute inverse covariance
    D_inv_y = d_inv * y
    D_inv_M = d_inv[:, None] * M

    B = M.T @ D_inv_M
    B.diagonal().add_(1.0)

    # Cholesky decomposition
    L = torch.linalg.cholesky(B)

    # Compute C matrix
    C = torch.linalg.solve_triangular(L, D_inv_M.T, upper=False)
    C = torch.linalg.solve_triangular(L.T, C, upper=True)

    K_inv_y = D_inv_y - D_inv_M @ (C @ y)
    log_det_K = torch.sum(torch.log(d)) + 2 * torch.sum(torch.log(torch.diag(L)))

    # Negative log-likelihood (final loss)
    nlog_p = 0.5 * (y @ K_inv_y + log_det_K + len(y) * torch.log(torch.tensor(2 * np.pi)))

    # # Compute gradients
    # K_inv_M = D_inv_M - D_inv_M @ (C @ M)
    # dM = -(K_inv_y[:, None] * (K_inv_y[None, :] @ M) - K_inv_M)

    # diag_K_inv = d_inv - torch.sum(C.T * D_inv_M, dim=1)

    # dlog_omega = -(absorption_noise * (K_inv_y**2 - diag_K_inv))

    # da = c_0 * omega2 * scaling_factor
    # dlog_c_0 = -torch.sum(K_inv_y * da * K_inv_y) + torch.sum(diag_K_inv * da)

    # da = omega2 * scaling_factor * lya_optical_depth * lya_absorption
    # dlog_tau_0 = -torch.sum(K_inv_y * da * K_inv_y) + torch.sum(diag_K_inv * da)

    # da = da * torch.log(lya_1pz) * beta
    # dlog_beta = -torch.sum(K_inv_y * da * K_inv_y) + torch.sum(diag_K_inv * da)

    return nlog_p  # Ensure it does NOT return gradients