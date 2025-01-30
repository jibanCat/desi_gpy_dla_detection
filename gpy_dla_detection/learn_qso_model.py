import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA


class DataLoader:
    """Handles loading QSO catalog and spectra data, supporting both .npy and .h5 formats."""

    def __init__(self, catalog_path, spectra_path):
        self.catalog = self._load_catalog(catalog_path)
        self.spectra = self._load_spectra(spectra_path)

    def _load_catalog(self, path):
        """Loads the QSO catalog from either .npy or .h5 file."""
        if path.endswith('.npy'):
            return np.load(path, allow_pickle=True).item()
        elif path.endswith('.h5'):
            with h5py.File(path, 'r') as f:
                return {key: f[key][()] for key in f.keys()}
        else:
            raise ValueError("Unsupported file format. Use .npy or .h5")

    def _load_spectra(self, path):
        """Loads preprocessed spectra from either .npy or .h5 file."""
        if path.endswith('.npy'):
            return np.load(path, allow_pickle=True).item()
        elif path.endswith('.h5'):
            with h5py.File(path, 'r') as f:
                return {key: f[key][()] for key in f.keys()}
        else:
            raise ValueError("Unsupported file format. Use .npy or .h5")


class SpectrumProcessor:
    """Processor for interpolating spectra onto a fixed rest-frame wavelength grid."""

    def __init__(self, rest_wavelengths):
        self.rest_wavelengths = rest_wavelengths

    def interpolate_to_restframe(self, wavelengths, fluxes, noise_variance, z_qso):
        """Interpolates observed spectra onto a fixed rest-frame grid."""
        rest_wavelengths = wavelengths / (1 + z_qso)
        interp_flux = interp1d(rest_wavelengths, fluxes, bounds_error=False, fill_value=np.nan)
        interp_noise = interp1d(rest_wavelengths, noise_variance, bounds_error=False, fill_value=np.nan)
        return interp_flux(self.rest_wavelengths), interp_noise(self.rest_wavelengths)


class GaussianProcessModel(nn.Module):
    """Implements the Gaussian Process model for QSO flux modeling."""

    def __init__(self, num_rest_pixels, k, external_pca=None):
        super().__init__()
        self.num_rest_pixels = num_rest_pixels
        self.k = k

        if external_pca is not None:
            assert external_pca.shape == (num_rest_pixels, k), "PCA eigenspectra dimensions do not match"
            self.M = nn.Parameter(torch.tensor(external_pca, dtype=torch.float32))
        else:
            self.M = nn.Parameter(torch.randn(num_rest_pixels, k))

        self.log_omega = nn.Parameter(torch.zeros(num_rest_pixels))
        self.log_c_0 = nn.Parameter(torch.tensor(0.0))
        self.log_tau_0 = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(0.0))

    def forward(self):
        """Returns model parameters in exponential space."""
        omega2 = torch.exp(2 * self.log_omega)
        c_0 = torch.exp(self.log_c_0)
        tau_0 = torch.exp(self.log_tau_0)
        beta = torch.exp(self.log_beta)
        return self.M, omega2, c_0, tau_0, beta


def spectrum_loss(y, lya_1pz, noise_variance, M, omega2, c_0, tau_0, beta):
    """
    Computes the negative log-likelihood for the spectrum.

    The covariance model follows:
        K = MM^T + diag(σ² + (ω (c₀ + a(1 + z)))²),
    where:
        - a(1 + z) = 1 - exp(-τ₀ (1 + z)ᵝ)
    """

    # Compute Lyα optical depth and absorption
    lya_optical_depth = tau_0 * torch.pow(lya_1pz, beta)
    lya_absorption = torch.exp(-lya_optical_depth)

    # Compute additional noise contribution due to absorption
    scaling_factor = 1 - lya_absorption + c_0
    absorption_noise = omega2 * torch.square(scaling_factor)

    # Total noise variance
    d = noise_variance + absorption_noise
    d_inv = 1 / d

    # Compute inverse covariance using Woodbury identity
    D_inv_y = d_inv * y
    D_inv_M = d_inv[:, None] * M

    B = M.T @ D_inv_M
    B.diagonal().add_(1.0)
    L = torch.linalg.cholesky(B)
    C = torch.cholesky_solve(D_inv_M, L)

    K_inv_y = D_inv_y - D_inv_M @ (C @ y)
    log_det_K = torch.sum(torch.log(d)) + 2 * torch.sum(torch.log(torch.diag(L)))

    return 0.5 * (y @ K_inv_y + log_det_K + len(y) * torch.log(torch.tensor(2 * np.pi)))


class Trainer:
    """Trainer for optimizing the GP model using L-BFGS."""

    def __init__(self, gp_model, learning_rate=0.01):
        self.model = gp_model
        self.optimizer = optim.LBFGS(self.model.parameters(), lr=learning_rate)

    def spectrum_loss_with_priors(self, fluxes, noise_variances, lya_1pzs):
        """Computes the total loss including priors on τ₀ and β."""
        M, omega2, c_0, tau_0, beta = self.model()
        loss = 0.0

        for i in range(len(fluxes)):
            valid_idx = ~np.isnan(fluxes[i])
            y = torch.tensor(fluxes[i][valid_idx], dtype=torch.float32)
            noise_var = torch.tensor(noise_variances[i][valid_idx], dtype=torch.float32)
            lya_1pz = torch.tensor(lya_1pzs[i][valid_idx], dtype=torch.float32)

            loss += spectrum_loss(y, lya_1pz, noise_var, M[valid_idx], omega2[valid_idx], c_0, tau_0, beta)

        prior_tau_0 = 0.5 * ((tau_0 - 0.00554) / 0.00064) ** 2
        prior_beta = 0.5 * ((beta - 3.182) / 0.074) ** 2
        return loss + prior_tau_0 + prior_beta

    def train(self, fluxes, noise_variances, lya_1pzs, max_iter=50):
        """Trains the GP model using L-BFGS."""
        def closure():
            self.optimizer.zero_grad()
            loss = self.spectrum_loss_with_priors(fluxes, noise_variances, lya_1pzs)
            loss.backward()
            print(f"Loss: {loss.item()}")
            return loss

        for _ in range(max_iter):
            self.optimizer.step(closure)