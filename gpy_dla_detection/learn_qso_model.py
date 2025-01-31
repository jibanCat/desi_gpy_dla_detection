import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import h5py
import numpy as np

class DataLoader:
    """Loads QSO spectra from HDF5 files, filtering by redshift and SNR."""

    def __init__(self, catalog_file, preloaded_file, z_range=(3.0, 4.25), min_snr=2.0, max_spectra=1000):
        self.catalog_file = catalog_file
        self.preloaded_file = preloaded_file
        self.z_range = z_range
        self.min_snr = min_snr
        self.max_spectra = max_spectra

        self.fluxes = []
        self.wavelengths = []
        self.noise_variances = []
        self.z_qsos = []

    def load_data(self):
        """Loads spectra and applies redshift & SNR filtering."""
        with h5py.File(self.catalog_file, 'r') as f:
            z_qsos = f['z_qsos'][:].flatten()  # Convert to 1D

        with h5py.File(self.preloaded_file, "r") as f:
            total_spectra = len(f["all_flux"][0])
            print(f"Total available spectra: {total_spectra}")

            flux_refs = f["all_flux"][0]
            wavelength_refs = f["all_wavelengths"][0]
            noise_refs = f["all_noise_variance"][0]

            snr_values = []

            # Iterate through all spectra and filter based on conditions
            for i in range(total_spectra):
                flux = np.array(f[flux_refs[i]])
                wavelengths = np.array(f[wavelength_refs[i]])
                noise_variance = np.array(f[noise_refs[i]])
                z_qso = z_qsos[i]

                # Skip empty or zero-only spectra
                if np.all(flux == 0):
                    continue

                # Apply redshift filter
                if not (self.z_range[0] <= z_qso <= self.z_range[1]):
                    continue

                # Compute SNR (median flux / sqrt(noise))
                snr = np.median(flux / np.sqrt(noise_variance))
                if np.isnan(snr) or np.isinf(snr) or snr < self.min_snr:
                    continue  # Skip spectra below SNR threshold

                # Store valid spectra
                self.fluxes.append(flux)
                self.wavelengths.append(wavelengths)
                self.noise_variances.append(noise_variance)
                self.z_qsos.append(z_qso)
                snr_values.append(snr)

        # Select top 1000 highest SNR spectra
        if len(snr_values) > self.max_spectra:
            sorted_indices = np.argsort(snr_values)[::-1]  # Sort descending
            self.fluxes = [self.fluxes[i] for i in sorted_indices[:self.max_spectra]]
            self.wavelengths = [self.wavelengths[i] for i in sorted_indices[:self.max_spectra]]
            self.noise_variances = [self.noise_variances[i] for i in sorted_indices[:self.max_spectra]]
            self.z_qsos = [self.z_qsos[i] for i in sorted_indices[:self.max_spectra]]

        print(f"Loaded {len(self.fluxes)} spectra with SNR > {self.min_snr} in z = [{self.z_range[0]}, {self.z_range[1]}]")

    def get_data(self):
        """Returns the processed spectra."""
        return self.fluxes, self.wavelengths, self.noise_variances, self.z_qsos

class SpectrumProcessor:
    """Preprocesses spectra by normalizing and interpolating onto a fixed grid."""

    def __init__(self, min_lambda=911, max_lambda=1216, num_pixels=200,
                 norm_min_lambda=1425, norm_max_lambda=1475):
        """Initialize wavelength grid and normalization range."""
        self.rest_wavelengths = np.linspace(min_lambda, max_lambda, num_pixels)
        self.norm_min_lambda = norm_min_lambda
        self.norm_max_lambda = norm_max_lambda

    def normalize_spectra(self, wavelengths, fluxes, noise_variances):
        """Normalizes spectra using the median flux in the range [1425Å, 1475Å]."""
        norm_fluxes = []
        norm_noise_variances = []

        for wave, flux, noise_var in zip(wavelengths, fluxes, noise_variances):
            # Select the region for normalization
            norm_mask = (wave >= self.norm_min_lambda) & (wave <= self.norm_max_lambda)
            
            if not np.any(norm_mask):
                continue  # Skip if no valid pixels in normalization range

            median_flux = np.median(flux[norm_mask])
            if median_flux == 0:
                continue  # Avoid division by zero

            norm_fluxes.append(flux / median_flux)
            norm_noise_variances.append(noise_var / (median_flux**2))

        return norm_fluxes, norm_noise_variances

    def interpolate_spectra(self, wavelengths, fluxes, noise_variances):
        """Interpolates spectra onto the fixed rest-frame grid."""
        interpolated_fluxes = []
        interpolated_noise_variances = []

        for wave, flux, noise_var in zip(wavelengths, fluxes, noise_variances):
            flux_interp = np.interp(self.rest_wavelengths, wave, flux, left=np.nan, right=np.nan)
            noise_var_interp = np.interp(self.rest_wavelengths, wave, noise_var, left=np.nan, right=np.nan)

            interpolated_fluxes.append(flux_interp)
            interpolated_noise_variances.append(noise_var_interp)

        return np.array(interpolated_fluxes), np.array(interpolated_noise_variances)

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

    B = M.T @ D_inv_M  # Shape (k, k)
    B.diagonal().add_(1.0)  # I + M^T D^{-1} M

    # Compute Cholesky decomposition
    L = torch.linalg.cholesky(B)  # Lower triangular (k, k)

    # Solve for C using numerically stable triangular solver
    C = torch.linalg.solve_triangular(L, D_inv_M.T, upper=False)
    C = torch.linalg.solve_triangular(L.T, C, upper=True)

    # Compute inverse covariance matrix
    K_inv_y = D_inv_y - D_inv_M @ (C @ y)

    log_det_K = torch.sum(torch.log(d)) + 2 * torch.sum(torch.log(torch.diag(L)))

    return 0.5 * (y @ K_inv_y + log_det_K + len(y) * torch.log(torch.tensor(2 * np.pi)))

class Trainer:
    """Trainer for optimizing the GP model with either L-BFGS or Adam."""

    def __init__(self, gp_model, optimizer_type="adam", learning_rate=0.01):
        """
        Initialize the trainer with the chosen optimizer.
        
        Parameters:
        - gp_model: The Gaussian Process Model.
        - optimizer_type: "adam" or "lbfgs".
        - learning_rate: Learning rate for the optimizer.
        """
        self.model = gp_model
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.loss_history = []

        # Choose optimizer
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif self.optimizer_type == "lbfgs":
            self.optimizer = optim.LBFGS(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer type must be either 'adam' or 'lbfgs'")

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

        # Add Gaussian priors for τ₀ and β
        prior_tau_0 = 0.5 * ((tau_0 - 0.00554) / 0.00064) ** 2
        prior_beta = 0.5 * ((beta - 3.182) / 0.074) ** 2
        total_loss = loss + prior_tau_0 + prior_beta

        return total_loss

    def train(self, fluxes, noise_variances, lya_1pzs, max_epochs=500):
        """Trains the GP model using either Adam or L-BFGS."""
        if self.optimizer_type == "adam":
            # Adam Optimization
            for epoch in range(max_epochs):
                self.optimizer.zero_grad()
                loss = self.spectrum_loss_with_priors(fluxes, noise_variances, lya_1pzs)
                loss.backward()
                self.optimizer.step()
                self.loss_history.append(loss.item())

                if epoch % 50 == 0:
                    print(f"Epoch {epoch}: Loss = {loss.item()}")

        elif self.optimizer_type == "lbfgs":
            # L-BFGS Optimization
            def closure():
                self.optimizer.zero_grad()
                loss = self.spectrum_loss_with_priors(fluxes, noise_variances, lya_1pzs)
                loss.backward()
                self.loss_history.append(loss.item())
                return loss

            for _ in range(max_epochs):
                self.optimizer.step(closure)

            print(f"Final Loss: {self.loss_history[-1]}")