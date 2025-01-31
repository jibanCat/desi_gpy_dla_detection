import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from .effective_optical_depth import effective_optical_depth
from .objective import spectrum_loss, objective
from .voigt import transition_wavelengths, oscillator_strengths

class DataLoader:
    """Loads QSO spectra, applies redshift & SNR filtering, and returns clean data."""
    
    def __init__(self, catalog_file, preloaded_file, z_range=(3.0, 4.25), min_snr=2.0, max_spectra=1000):
        self.catalog_file = catalog_file
        self.preloaded_file = preloaded_file
        self.z_range = z_range
        self.min_snr = min_snr
        self.max_spectra = max_spectra

    def load_data(self):
        """Loads spectra and applies redshift & SNR filtering."""
        with h5py.File(self.catalog_file, 'r') as f:
            z_qsos = f['z_qsos'][:].flatten()

        with h5py.File(self.preloaded_file, "r") as f:
            total_spectra = len(f["all_flux"][0])
            print(f"Total available spectra: {total_spectra}")

            flux_refs = f["all_flux"][0]
            wavelength_refs = f["all_wavelengths"][0]
            noise_refs = f["all_noise_variance"][0]

            snr_values, fluxes, wavelengths, noise_variances, selected_z_qsos = [], [], [], [], []

            for i in range(total_spectra):
                flux = np.array(f[flux_refs[i]])
                wave = np.array(f[wavelength_refs[i]])
                noise = np.array(f[noise_refs[i]])
                z_qso = z_qsos[i]

                # Apply redshift and SNR filtering
                if np.all(flux == 0) or not (self.z_range[0] <= z_qso <= self.z_range[1]):
                    continue  
                
                # Compute SNR and apply threshold
                snr = np.median(flux / np.sqrt(noise))
                if np.isnan(snr) or np.isinf(snr) or snr < self.min_snr:
                    continue  

                fluxes.append(flux)
                wavelengths.append(wave)
                noise_variances.append(noise)
                selected_z_qsos.append(z_qso)
                snr_values.append(snr)

        # Sort spectra by SNR and keep only the top `max_spectra` spectra
        if len(snr_values) > self.max_spectra:
            top_indices = np.argsort(snr_values)[::-1][:self.max_spectra]
            fluxes = [fluxes[i] for i in top_indices]
            wavelengths = [wavelengths[i] for i in top_indices]
            noise_variances = [noise_variances[i] for i in top_indices]
            selected_z_qsos = [selected_z_qsos[i] for i in top_indices]

        return fluxes, wavelengths, noise_variances, selected_z_qsos


class SpectrumProcessor:
    """Preprocesses spectra: normalizes, interpolates, and de-forests them."""

    def __init__(self, min_lambda=911, max_lambda=1216, num_pixels=200,
                 norm_min_lambda=1425, norm_max_lambda=1475):
        self.rest_wavelengths = np.linspace(min_lambda, max_lambda, num_pixels)
        self.norm_min_lambda = norm_min_lambda
        self.norm_max_lambda = norm_max_lambda

    def normalize_spectra(self, wavelengths, fluxes, noise_variances):
        """Normalizes spectra using median flux in [1425Å, 1475Å]."""
        norm_fluxes, norm_noise_variances = [], []

        # Interpolate spectra to the common rest-frame wavelength grid
        for wave, flux, noise in zip(wavelengths, fluxes, noise_variances):
            norm_mask = (wave >= self.norm_min_lambda) & (wave <= self.norm_max_lambda)
            if not np.any(norm_mask):
                continue  

            median_flux = np.median(flux[norm_mask])
            if median_flux == 0:
                continue  

            norm_fluxes.append(flux / median_flux)
            norm_noise_variances.append(noise / (median_flux**2))

        return np.array(norm_fluxes), np.array(norm_noise_variances)

    def de_forest_spectra(self, wavelengths, fluxes, noise_variances, z_qsos, tau_0=0.00554, beta=3.182):
        """Removes effective Lyα forest absorption using `effective_optical_depth()`."""
        de_forest_fluxes, de_forest_noise = [], []

        for wave, flux, noise, z_qso in zip(wavelengths, fluxes, noise_variances, z_qsos):
            optical_depth = effective_optical_depth(wave, beta, tau_0, z_qso, num_forest_lines=10)
            lya_absorption = np.exp(-np.sum(optical_depth, axis=1))

            # Interpolate the effective optical depth to the observed wavelength grid
            de_forest_fluxes.append(flux / lya_absorption)
            de_forest_noise.append(noise / (lya_absorption**2))

        return np.array(de_forest_fluxes), np.array(de_forest_noise)


def compute_pca(centered_fluxes, num_components=10):
    """Computes PCA eigenspectra for GP initialization."""
    pca = PCA(n_components=num_components)
    pca.fit(centered_fluxes)
    return pca.components_.T


class GaussianProcessModel(nn.Module):
    """Gaussian Process model with PCA eigenspectra initialization.
    """

    def __init__(self, num_pixels, k, pca_eigenspectra):
        super().__init__()
        self.num_pixels = num_pixels
        self.k = k

        # Initialize model parameters
        self.M = nn.Parameter(torch.tensor(pca_eigenspectra, dtype=torch.float32))
        self.log_omega = nn.Parameter(torch.zeros(num_pixels))
        self.log_c_0 = nn.Parameter(torch.tensor(0.0))
        self.log_tau_0 = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(0.0))

    def forward(self):
        """Returns model parameters in exponential space."""
        return self.M, torch.exp(2 * self.log_omega), torch.exp(self.log_c_0), torch.exp(self.log_tau_0), torch.exp(self.log_beta)

import torch
import torch.optim as optim

class Trainer:
    """
    Trainer for optimizing the Gaussian Process model with PyTorch autograd support.
    Allows switching between Adam and L-BFGS optimization.
    """

    def __init__(self, gp_model, optimizer_type="adam", learning_rate=0.01):
        """
        Initialize the trainer.

        Parameters:
        - gp_model: GaussianProcessModel instance
        - optimizer_type: "adam" or "lbfgs"
        - learning_rate: Step size for optimizer
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

    def train(self, fluxes, lya_1pzs, noise_variances, z_qsos, num_forest_lines,
              all_transition_wavelengths, all_oscillator_strengths, max_epochs=500):
        """
        Trains the GP model using either Adam or L-BFGS.
        """

        def closure():
            """
            Closure function for L-BFGS optimization.
            """
            self.optimizer.zero_grad()
            loss = objective(self.model, fluxes, lya_1pzs, noise_variances, num_forest_lines,
                             all_transition_wavelengths, all_oscillator_strengths, z_qsos)
            loss.backward()
            self.loss_history.append(loss.item())
            return loss

        if self.optimizer_type == "adam":
            # Adam optimization
            for epoch in range(max_epochs):
                self.optimizer.zero_grad()
                loss = objective(self.model, fluxes, lya_1pzs, noise_variances, num_forest_lines,
                                 all_transition_wavelengths, all_oscillator_strengths, z_qsos)
                loss.backward()
                self.optimizer.step()
                self.loss_history.append(loss.item())

                if epoch % 50 == 0:
                    print(f"Epoch {epoch}: Loss = {loss.item()}")

        elif self.optimizer_type == "lbfgs":
            # L-BFGS optimization (uses closure)
            for _ in range(max_epochs):
                self.optimizer.step(closure)

        print(f"Final Loss: {self.loss_history[-1]}")

if __name__ == "__main__":

    # 🚀 Training Steps
    fluxes, wavelengths, noise_variances, z_qsos = DataLoader("catalog.h5", "preloaded.h5").load_data()
    processor = SpectrumProcessor()
    norm_fluxes, norm_noise = processor.normalize_spectra(wavelengths, fluxes, noise_variances)
    de_forest_fluxes, de_forest_noise = processor.de_forest_spectra(wavelengths, norm_fluxes, norm_noise, z_qsos)

    centered_fluxes = de_forest_fluxes - np.mean(de_forest_fluxes, axis=0)
    pca_eigenspectra = compute_pca(centered_fluxes)

    gp_model = GaussianProcessModel(200, 10, pca_eigenspectra)
    trainer = Trainer(gp_model, optimizer="adam")
    trainer.train(de_forest_fluxes, de_forest_noise)