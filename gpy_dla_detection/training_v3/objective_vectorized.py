"""Step B: vectorized batch version of v1 ``spectrum_loss``.

Numeric-equivalent re-expression of the per-spectrum loop in
``gpy_dla_detection.objective.spectrum_loss`` across a batch axis.

Hand-coded analytic gradients only — NO autograd. The math is identical
to v1; only the indexing changes from per-spectrum (variable-length valid
arrays) to batched (padded to a common grid length, with a boolean
valid_mask zeroing out invalid-pixel contributions).

Why padding works:
    - At invalid pixels we set d_inv = 0, log(d) = 0, y = 0.
    - That makes B = M.T @ D_inv_M, K_inv_y, K_inv_M, and diag_K_inv all
      pick up zero contribution from invalid pixels.
    - The per-pixel gradient blocks (dM, dlog_omega) are then identically
      zero at invalid rows, so the batch-sum equals the per-spectrum
      ``dM_accum[valid_i, :] += dM_i`` scatter.
    - The per-spectrum scalars (dlog_c_0, dlog_tau_0, dlog_beta) are
      identical because their summands are zero at invalid pixels.
    - The only piece that needs explicit attention is ``log_det_K``: we
      include only ``log(d)`` at valid pixels, hence the masked sum.

Memory: scales as O(B · N · k) for D_inv_M and (B, k, N) for the C
factor. Caller is responsible for chunking large batches; this function
operates on whatever it's given.

Parity test: ``tests/test_v3_objective_vectorized_parity.py`` confirms
batch-equivalent to per-spectrum to ~1e-10 (float64) on the 6 frozen
2lpt fixtures.
"""
from __future__ import annotations

import torch


def spectrum_loss_batch(
    y, lya_1pz, noise_variance, valid_mask,
    M, omega2, c_0, tau_0, beta,
    num_forest_lines, all_transition_wavelengths, all_oscillator_strengths,
    zqso_1pz,
):
    """Vectorized per-spectrum loss + gradients, summed over the batch axis.

    Equivalent to::

        total = 0
        dM_accum = zeros(N, k); dlog_omega_accum = zeros(N)
        dlog_c_0_accum = dlog_tau_0_accum = dlog_beta_accum = 0
        for b in range(B):
            valid = valid_mask[b]
            nlog, dM, dlogw, dlc0, dlt0, dlb = spectrum_loss(
                y[b, valid], lya_1pz[b, valid], noise_variance[b, valid],
                M[valid, :], omega2[valid], c_0, tau_0, beta,
                num_forest_lines, all_transition_wavelengths,
                all_oscillator_strengths, zqso_1pz[b])
            total += nlog
            dM_accum[valid, :]     += dM
            dlog_omega_accum[valid] += dlogw
            dlog_c_0_accum   += dlc0
            dlog_tau_0_accum += dlt0
            dlog_beta_accum  += dlb

    Parameters
    ----------
    y                  : (B, N)  centered fluxes; invalid pixels masked out internally
    lya_1pz            : (B, N)  per-pixel (1+z_lya); MUST be finite/positive everywhere
                                 (caller fills with sensible values at invalid pixels)
    noise_variance     : (B, N)  per-pixel σ²; invalid pixels masked out internally
    valid_mask         : (B, N)  bool, true for pixels that should contribute
    M                  : (N, k)  shared low-rank GP basis
    omega2             : (N,)    shared exp(2 log_ω)
    c_0, tau_0, beta   : scalars (0-d torch tensors)
    num_forest_lines   : int     same as v1 spectrum_loss
    all_transition_wavelengths, all_oscillator_strengths
                       : (>=num_forest_lines,) 1-d torch tensors
    zqso_1pz           : (B,)    per-spectrum (1+z_qso)

    Returns
    -------
    nlog_p_total       : 0-d tensor, sum_b nlog_p(b)
    dM_accum           : (N, k)
    dlog_omega_accum   : (N,)
    dlog_c_0_accum     : 0-d tensor
    dlog_tau_0_accum   : 0-d tensor
    dlog_beta_accum    : 0-d tensor
    """
    log_2pi = 1.83787706640934534

    B, N = y.shape
    k = M.shape[1]
    dtype = M.dtype

    # ---- Sanitize at invalid pixels so torch ops never see NaN/Inf there.
    # 1.0 chosen to make d = 1 + ω²·(1−A+c₀)² > 0 everywhere; it's irrelevant
    # since masked-out via d_inv.
    nv_safe = torch.where(valid_mask, noise_variance, torch.ones_like(noise_variance))
    y_safe = torch.where(valid_mask, y, torch.zeros_like(y))

    # ---- Lya optical depth, per-pixel
    zqso_1pz_b = zqso_1pz.unsqueeze(-1)               # (B, 1)
    indicator = (lya_1pz <= zqso_1pz_b).to(dtype)     # (B, N)
    lya_optical_depth = tau_0 * torch.pow(lya_1pz, beta) * indicator  # (B, N)

    # Lyman series — additive contributions from Lyβ, Lyγ, …
    for i in range(1, num_forest_lines):
        lyman_1pz = (all_transition_wavelengths[0] * lya_1pz) / all_transition_wavelengths[i]
        lyman_indicator = (lyman_1pz <= zqso_1pz_b).to(dtype)
        lyman_1pz = lyman_1pz * lyman_indicator
        tau_i = (tau_0 * all_transition_wavelengths[i] * all_oscillator_strengths[i]) / \
                (all_transition_wavelengths[0] * all_oscillator_strengths[0])
        lya_optical_depth = lya_optical_depth + tau_i * torch.pow(lyman_1pz, beta)

    lya_absorption = torch.exp(-lya_optical_depth)            # (B, N)
    scaling_factor = 1 - lya_absorption + c_0                 # (B, N)
    absorption_noise = omega2.unsqueeze(0) * scaling_factor ** 2  # (B, N)

    d = nv_safe + absorption_noise                            # (B, N), > 0 everywhere
    d_inv = torch.where(valid_mask, 1.0 / d, torch.zeros_like(d))   # (B, N)
    log_d = torch.where(valid_mask, torch.log(d), torch.zeros_like(d))

    # ---- Woodbury setup
    D_inv_y = d_inv * y_safe                                  # (B, N)
    D_inv_M = d_inv.unsqueeze(-1) * M.unsqueeze(0)            # (B, N, k)

    # B_b = M^T @ D_inv_M_b — broadcasting rule promotes M^T to (1, k, N)
    Bmat = torch.matmul(M.transpose(0, 1), D_inv_M)           # (B, k, k)
    eye_k = torch.eye(k, dtype=dtype, device=M.device).unsqueeze(0)
    Bmat = Bmat + eye_k

    L = torch.linalg.cholesky(Bmat)                           # (B, k, k)

    # Two triangular solves to get C = (L L^T)^{-1} D_inv_M^T
    D_inv_M_T = D_inv_M.transpose(-1, -2)                     # (B, k, N)
    X = torch.linalg.solve_triangular(L, D_inv_M_T, upper=False)
    C = torch.linalg.solve_triangular(L.transpose(-1, -2), X, upper=True)  # (B, k, N)

    # ---- K_inv_y per spectrum
    C_y = torch.matmul(C, y_safe.unsqueeze(-1))               # (B, k, 1)
    K_inv_y = D_inv_y - torch.matmul(D_inv_M, C_y).squeeze(-1)  # (B, N), 0 at invalid

    # ---- nlog_p per spectrum + sum
    log_det_K = log_d.sum(dim=1) + 2 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(dim=1)
    n_valid = valid_mask.to(dtype).sum(dim=1)                 # (B,)
    yK_invy = (y_safe * K_inv_y).sum(dim=1)                   # (B,)
    nlog_p_per = 0.5 * (yK_invy + log_det_K + n_valid * log_2pi)
    nlog_p_total = nlog_p_per.sum()

    # ---- Gradients (analytic, matches v1 spectrum_loss line-for-line)
    tmp = torch.matmul(C, M.unsqueeze(0))                     # (B, k, k)
    K_inv_M = D_inv_M - torch.matmul(D_inv_M, tmp)            # (B, N, k); 0 at invalid rows

    # dM block: -(K_inv_y ⊗ (K_inv_y · M) − K_inv_M)
    K_invy_M = torch.matmul(K_inv_y.unsqueeze(1), M.unsqueeze(0))  # (B, 1, k)
    outer = torch.matmul(K_inv_y.unsqueeze(2), K_invy_M)            # (B, N, k)
    dM_per_spectrum = -(outer - K_inv_M)                            # (B, N, k); 0 at invalid rows
    dM_accum = dM_per_spectrum.sum(dim=0)                           # (N, k)

    # diag(K^{-1}) per pixel per spectrum
    diag_K_inv = d_inv - (C * D_inv_M_T).sum(dim=1)                 # (B, N); 0 at invalid

    # dlog_omega per pixel per spectrum, summed over batch
    dlog_omega_per = -(absorption_noise * (K_inv_y ** 2 - diag_K_inv))  # (B, N); 0 at invalid
    dlog_omega_accum = dlog_omega_per.sum(dim=0)                         # (N,)

    # Scalar gradients (each is a per-spectrum dot product summed over batch)
    da_c0 = c_0 * omega2.unsqueeze(0) * scaling_factor                   # (B, N)
    dlog_c_0_accum = (
        -(K_inv_y ** 2 * da_c0).sum() + (diag_K_inv * da_c0).sum()
    )

    da_tau0 = omega2.unsqueeze(0) * scaling_factor * lya_optical_depth * lya_absorption
    dlog_tau_0_accum = (
        -(K_inv_y ** 2 * da_tau0).sum() + (diag_K_inv * da_tau0).sum()
    )

    da_beta = da_tau0 * torch.log(lya_1pz) * beta * indicator
    dlog_beta_accum = (
        -(K_inv_y ** 2 * da_beta).sum() + (diag_K_inv * da_beta).sum()
    )

    return (nlog_p_total, dM_accum, dlog_omega_accum,
            dlog_c_0_accum, dlog_tau_0_accum, dlog_beta_accum)
