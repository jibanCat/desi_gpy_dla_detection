import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import torch
import torch.optim as optim
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import interp1d

from .effective_optical_depth import effective_optical_depth
from .objective import spectrum_loss, objective
from .voigt import transition_wavelengths, oscillator_strengths
from tqdm import tqdm  # For progress bar

class QSOLoader:
    """Loads QSO spectra, applies redshift & SNR filtering, and returns clean data efficiently."""

    def __init__(self, catalog_file, preloaded_file, z_range=(3.0, 4.25), min_snr=2.0, max_spectra=1000):
        self.catalog_file = catalog_file
        self.preloaded_file = preloaded_file
        self.z_range = z_range
        self.min_snr = min_snr
        self.max_spectra = max_spectra

    def load_data(self):
        """Loads spectra efficiently with early stopping and a progress bar."""

        # Load quasar redshifts
        with h5py.File(self.catalog_file, 'r') as f:
            z_qsos = f['z_qsos'][:].flatten()

        # Load preprocessed spectral data
        with h5py.File(self.preloaded_file, "r") as f:
            total_spectra = len(f["all_flux"][0])
            print(f"Total available spectra: {total_spectra}")

            flux_refs = f["all_flux"][0]
            wavelength_refs = f["all_wavelengths"][0]
            noise_refs = f["all_noise_variance"][0]

            snr_values, fluxes, wavelengths, noise_variances, selected_z_qsos = [], [], [], [], []

            # Use tqdm progress bar
            for i in tqdm(range(total_spectra), desc="Loading spectra", unit="spec"):

                # Early stopping if max_spectra reached
                if len(fluxes) >= self.max_spectra:
                    break  

                flux = np.array(f[flux_refs[i]])
                wave = np.array(f[wavelength_refs[i]])
                noise = np.array(f[noise_refs[i]])
                z_qso = z_qsos[i]

                # check the shape, if (1, N) then flatten
                if flux.shape[0] == 1:
                    flux = flux.flatten()
                    wave = wave.flatten()
                    noise = noise.flatten()

                # Apply redshift and SNR filtering
                if np.all(flux == 0) or not (self.z_range[0] <= z_qso <= self.z_range[1]):
                    continue  

                # Compute SNR and apply threshold
                snr = np.median(flux / np.sqrt(noise))
                if np.isnan(snr) or np.isinf(snr) or snr < self.min_snr:
                    continue  

                # Store only good spectra
                fluxes.append(flux)
                wavelengths.append(wave)
                noise_variances.append(noise)
                selected_z_qsos.append(z_qso)
                snr_values.append(snr)

        # Sort by SNR and keep only the top `max_spectra`
        if len(snr_values) > self.max_spectra:
            top_indices = np.argsort(snr_values)[::-1][:self.max_spectra]
            fluxes = [fluxes[i] for i in top_indices]
            wavelengths = [wavelengths[i] for i in top_indices]
            noise_variances = [noise_variances[i] for i in top_indices]
            selected_z_qsos = [selected_z_qsos[i] for i in top_indices]

        print(f"Loaded {len(fluxes)} high-SNR spectra.")
        return fluxes, wavelengths, noise_variances, selected_z_qsos

class SpectrumProcessor:
    """Preprocesses spectra: normalizes, interpolates, and de-forests them."""

    def __init__(self, min_lambda=911, max_lambda=1216, num_pixels=200,
                 norm_min_lambda=1425, norm_max_lambda=1475):
        self.rest_wavelengths = np.linspace(min_lambda, max_lambda, num_pixels)
        self.norm_min_lambda = norm_min_lambda
        self.norm_max_lambda = norm_max_lambda

    def normalize_spectra(self, wavelengths, fluxes, noise_variances, z_qsos=None):
        """Normalizes spectra using median flux in [norm_min_lambda, norm_max_lambda]."""
        norm_fluxes, norm_noise_variances = [], []
        all_wave = []

        for i, (wave, flux, noise) in enumerate(zip(wavelengths, fluxes, noise_variances)):
            
            # Apply redshift to wavelengths
            if z_qsos is not None:
                wave = wave / (1 + z_qsos[i])

            norm_mask = (wave >= self.norm_min_lambda) & (wave <= self.norm_max_lambda)
            if not np.any(norm_mask):
                continue  

            median_flux = np.median(flux[norm_mask])
            if median_flux == 0:
                continue  

            norm_fluxes.append(flux / median_flux)
            norm_noise_variances.append(noise / (median_flux**2))
            all_wave.append(wave)

        return norm_fluxes, norm_noise_variances, all_wave

    def interpolate_spectra(self, wavelengths, fluxes, noise_variances):
        """Interpolates fluxes and noise variances onto `rest_wavelengths`."""
        interp_fluxes, interp_noise_variances = [], []
        interp_wave = []

        for wave, flux, noise in zip(wavelengths, fluxes, noise_variances):
            interp_flux = interp1d(wave, flux, bounds_error=False, fill_value=np.nan)
            interp_noise = interp1d(wave, noise, bounds_error=False, fill_value=np.nan)

            interp_fluxes.append(interp_flux(self.rest_wavelengths))
            interp_noise_variances.append(interp_noise(self.rest_wavelengths))

            interp_wave.append(self.rest_wavelengths)

        return np.array(interp_fluxes), np.array(interp_noise_variances), np.array(interp_wave)

    def de_forest_spectra(self, wavelengths, fluxes, noise_variances, z_qsos, tau_0=0.00554, beta=3.182):
        """Removes effective Lyα forest absorption using `effective_optical_depth()`."""
        de_forest_fluxes, de_forest_noise = [], []

        for wave, flux, noise, z_qso in zip(wavelengths, fluxes, noise_variances, z_qsos):
            # Apply redshift to wavelengths
            obs_wave = wave * (1 + z_qso)
            # Compute effective optical depth
            optical_depth = effective_optical_depth(obs_wave, beta, tau_0, z_qso, num_forest_lines=10)
            lya_absorption = np.exp(-np.sum(optical_depth, axis=1))

            # Interpolate the effective optical depth to the observed wavelength grid
            de_forest_fluxes.append(flux / lya_absorption)
            de_forest_noise.append(noise / (lya_absorption**2))

        return np.array(de_forest_fluxes), np.array(de_forest_noise)
    
    def center_fluxes(self, fluxes, noise_variances):
        """Centers fluxes by subtracting the mean."""
        # get the inverse variance average of the fluxes
        ivar = 1 / np.array(noise_variances)
        mean_flux = np.sum(fluxes * ivar, axis=0) / np.sum(ivar, axis=0)
        centered_fluxes = fluxes - mean_flux
        return centered_fluxes, mean_flux

    def fill_nan_with_median(self, fluxes):
        """Fills NaN values in fluxes with the median value."""
        for i in range(len(fluxes)):
            ind = np.isnan(fluxes[i])
            # fill the median of the whole dataset, so this won't affect GP training
            fluxes[i][ind] = np.nanmedian(fluxes)
        return fluxes

def compute_pca(centered_fluxes, num_components=10):
    """Computes PCA eigenspectra for GP initialization."""
    pca = PCA(n_components=num_components)
    pca.fit(centered_fluxes)
    return pca.components_.T

class GaussianProcessModel(nn.Module):
    """Gaussian Process model with PCA eigenspectra initialization."""

    def __init__(self, num_pixels, k, pca_eigenspectra, min_lambda=911, max_lambda=1216):
        super().__init__()
        self.num_pixels = num_pixels
        self.k = k

        # Define a consistent rest-wavelength grid
        self.rest_wavelengths = torch.linspace(min_lambda, max_lambda, num_pixels, dtype=torch.float32)

        # Initialize model parameters
        self.M = nn.Parameter(torch.tensor(pca_eigenspectra, dtype=torch.float32).clone().detach())
        self.log_omega = nn.Parameter(torch.zeros(num_pixels))
        self.log_c_0 = nn.Parameter(torch.tensor(0.0))
        self.log_tau_0 = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(0.0))

    def forward(self):
        """Returns model parameters in exponential space."""
        return self.M, torch.exp(2 * self.log_omega), torch.exp(self.log_c_0), torch.exp(self.log_tau_0), torch.exp(self.log_beta)

class Trainer:
    """
    Trainer for optimizing the Gaussian Process model with PyTorch autograd support.
    Uses Mini-Batch Training.
    """

    def __init__(self, gp_model, optimizer_type="adam", learning_rate=0.01, batch_size=32):
        """
        Initialize the trainer.

        Parameters:
        - gp_model: GaussianProcessModel instance
        - optimizer_type: "adam" or "lbfgs"
        - learning_rate: Step size for optimizer
        - batch_size: Number of samples per mini-batch
        """
        self.model = gp_model
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
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
        Trains the GP model using either Adam or L-BFGS with mini-batches.
        """

        # Create PyTorch DataLoader for mini-batches
        dataset = TensorDataset(fluxes, lya_1pzs, noise_variances, z_qsos)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        def closure():
            """
            Closure function for L-BFGS optimization.
            """
            self.optimizer.zero_grad()
            loss = objective(self.model, fluxes, lya_1pzs, noise_variances, num_forest_lines,
                             all_transition_wavelengths, all_oscillator_strengths, z_qsos)
            loss.backward()
            self.loss_history.append(loss.item())  # 👈 This is where loss should be stored
            return loss

        if self.optimizer_type == "adam":
            for epoch in range(max_epochs):
                total_loss = 0.0

                for batch in dataloader:
                    batch_fluxes, batch_lya_1pzs, batch_noise_variances, batch_z_qsos = batch

                    self.optimizer.zero_grad()
                    loss = objective(self.model, batch_fluxes, batch_lya_1pzs, batch_noise_variances,
                                    num_forest_lines, all_transition_wavelengths, all_oscillator_strengths, batch_z_qsos)

                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    self.loss_history.append(loss.item())  # 👈 Append loss here!

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader)}")
                        
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