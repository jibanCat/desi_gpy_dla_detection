"""
Learning script for the Gaussian Process model with Lyα forest absorption.

 DESI: power-law of the form τ(z)=τ0(1+z)γ to our measurements and find τ0=(2.46±0.14)×10−3 and γ=3.62±0.04
"""
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
import torch.optim.lr_scheduler as lr_scheduler

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
    """Preprocesses spectra: masks noisy pixels, interpolates, normalizes, and de-forests them."""

    def __init__(self, min_lambda=911, max_lambda=1216, num_pixels=200,
                 norm_min_lambda=1425, norm_max_lambda=1475, max_noise_variance=9.0):
        self.rest_wavelengths = np.linspace(min_lambda, max_lambda, num_pixels)
        self.norm_min_lambda = norm_min_lambda
        self.norm_max_lambda = norm_max_lambda
        self.max_noise_variance = max_noise_variance  # Threshold for masking high-noise pixels

    def mask_noisy_pixels(self, fluxes, noise_variances):
        """Masks pixels with noise variance above a threshold."""
        less_noisy_fluxes, less_noisy_variances = [], []
        for flux, noise in zip(fluxes, noise_variances):
            mask = noise > self.max_noise_variance
            flux = np.where(mask, np.nan, flux)  # Use np.where to avoid modifying input in-place
            noise = np.where(mask, np.nan, noise)
            less_noisy_fluxes.append(flux)
            less_noisy_variances.append(noise)
        return less_noisy_fluxes, less_noisy_variances

    def normalize_spectra(self, wavelengths, fluxes, noise_variances, z_qsos=None):
        """Normalizes spectra using median flux in [norm_min_lambda, norm_max_lambda]."""
        norm_fluxes, norm_noise_variances, all_wave = [], [], []

        for i, (wave, flux, noise) in enumerate(zip(wavelengths, fluxes, noise_variances)):
            if z_qsos is not None:
                wave = wave / (1 + z_qsos[i])

            norm_mask = (wave >= self.norm_min_lambda) & (wave <= self.norm_max_lambda)
            if np.sum(norm_mask) < 2:
                continue  

            median_flux = np.nanmedian(flux[norm_mask])
            if median_flux == 0 or np.isnan(median_flux):
                continue  

            norm_fluxes.append(flux / median_flux)
            norm_noise_variances.append(noise / (median_flux**2))
            all_wave.append(wave)

        return norm_fluxes, norm_noise_variances, all_wave

    def interpolate_spectra(self, wavelengths, fluxes, noise_variances):
        """Interpolates fluxes and noise variances onto `rest_wavelengths` while handling NaNs properly."""
        interp_fluxes, interp_noise_variances = [], []
        all_wave = []

        for wave, flux, noise in zip(wavelengths, fluxes, noise_variances):
            valid = np.isfinite(wave) & np.isfinite(flux)

            all_wave.append(self.rest_wavelengths)

            if np.sum(valid) < 2:  # Skip if too few valid points
                interp_fluxes.append(np.full_like(self.rest_wavelengths, np.nan))
                interp_noise_variances.append(np.full_like(self.rest_wavelengths, np.nan))
                continue  

            try:
                flux_interp = interp1d(wave[valid], flux[valid], kind="linear",
                                       bounds_error=False, fill_value=np.nan)
                noise_interp = interp1d(wave[valid], noise[valid], kind="linear",
                                        bounds_error=False, fill_value=np.nan)

                interp_flux = flux_interp(self.rest_wavelengths)
                interp_noise = noise_interp(self.rest_wavelengths)

                interp_flux[~np.isfinite(interp_flux)] = np.nan
                interp_noise[~np.isfinite(interp_noise)] = np.nan

                interp_fluxes.append(interp_flux)
                interp_noise_variances.append(interp_noise)
            except Exception as e:
                print(f"Interpolation failed for spectrum: {e}")
                interp_fluxes.append(np.full_like(self.rest_wavelengths, np.nan))
                interp_noise_variances.append(np.full_like(self.rest_wavelengths, np.nan))

        return np.array(interp_fluxes), np.array(interp_noise_variances), np.array(all_wave)

    def de_forest_spectra(self, wavelengths, fluxes, noise_variances, z_qsos, tau_0=0.00246, beta=3.62):
        """Removes effective Lyα forest absorption using effective_optical_depth()."""
        de_forest_fluxes, de_forest_noise = [], []

        for wave, flux, noise, z_qso in zip(wavelengths, fluxes, noise_variances, z_qsos):
            # Apply redshift to wavelengths
            obs_wave = wave * (1 + z_qso)
            # Compute effective optical depth
            optical_depth = effective_optical_depth(obs_wave, beta, tau_0, z_qso, num_forest_lines=10)
            lya_absorption = np.exp(-np.sum(optical_depth, axis=1))

            # Remove Lyα forest absorption
            de_forest_fluxes.append(flux / lya_absorption)
            de_forest_noise.append(noise / (lya_absorption**2))

        return np.array(de_forest_fluxes), np.array(de_forest_noise)

    def center_fluxes(self, fluxes, noise_variances):
        """Centers fluxes by subtracting the inverse-variance weighted mean."""
        ivar = np.where(noise_variances > 0, 1 / noise_variances, 0)  # Avoid division by zero
        mean_flux = np.nansum(fluxes * ivar, axis=0) / np.nansum(ivar, axis=0)
        mean_flux[np.isnan(mean_flux)] = np.nanmedian(mean_flux)  # Ensure mean_flux has no NaNs

        centered_fluxes = fluxes - mean_flux
        return centered_fluxes, mean_flux

    def fill_nan_with_median(self, fluxes):
        """Fills NaN values in fluxes with the dataset-wide median."""
        for flux in fluxes:
            flux[np.isnan(flux)] = np.nanmedian(flux)
        return fluxes
    
    def remove_nan_spectra(self, fluxes, noise_variances, wavelengths, z_qsos):
        """Removes entire spectra if they contain NaN values in flux, noise variance, or wavelength."""
        cleaned_fluxes, cleaned_noises, cleaned_waves, cleaned_z_qsos = [], [], [], []
        
        for flux, noise, wave, z_qso in zip(fluxes, noise_variances, wavelengths, z_qsos):
            if np.isnan(flux).any() or np.isnan(noise).any() or np.isnan(wave).any():
                continue  # Skip spectra with any NaN values
            cleaned_fluxes.append(flux)
            cleaned_noises.append(noise)
            cleaned_waves.append(wave)
            cleaned_z_qsos.append(z_qso)

        return np.array(cleaned_fluxes), np.array(cleaned_noises), np.array(cleaned_waves), np.array(cleaned_z_qsos)

def compute_pca(centered_fluxes, num_components=10):
    """Computes PCA eigenspectra for GP initialization."""
    pca = PCA(n_components=num_components)
    pca.fit(centered_fluxes)  # Fit PCA without transformation

    # Get top-k PCA components
    coefficients = pca.components_.T  # Shape (num_pixels, k)
    latent = pca.explained_variance_  # Shape (k,)
    return coefficients, latent

class GaussianProcessModel(nn.Module):
    """Gaussian Process model with PCA eigenspectra initialization."""

    def __init__(self, num_pixels, k, centered_rest_fluxes, initial_M=None, min_lambda=911, max_lambda=1216):
        super().__init__()
        self.num_pixels = num_pixels
        self.k = k

        # Define a consistent rest-wavelength grid
        self.rest_wavelengths = torch.linspace(min_lambda, max_lambda, num_pixels, dtype=torch.float32)

        # Initialize model parameters
        # initial_c_0   = 0.1;                          % initial guess for c₀
        # initial_tau_0 = 0.00554;                      % initial guess for τ₀
        # initial_beta  = 3.182;                        % initial guess for β
        if initial_M is None:
            coefficients, latent = compute_pca(centered_rest_fluxes, k)
            # Compute initial M using PCA coefficients and square root of eigenvalues
            initial_M = coefficients[:, :k] * np.sqrt(latent[:k])  # Broadcasting happens automatically
        self.M = nn.Parameter(torch.tensor(initial_M, dtype=torch.float32).clone().detach())
        initial_log_omega = np.log(np.nanstd(centered_rest_fluxes, axis=0))  # Standard deviation per wavelength pixel
        self.log_omega = nn.Parameter(torch.tensor(initial_log_omega, dtype=torch.float32).clone().detach())
        self.log_c_0 = nn.Parameter(torch.tensor(np.log(0.1)))
        self.log_tau_0 = nn.Parameter(torch.tensor(np.log(0.00246)))
        self.log_beta = nn.Parameter(torch.tensor(np.log(3.62)))

    def forward(self):
        """Returns model parameters in exponential space."""
        return self.M, torch.exp(2 * self.log_omega), torch.exp(self.log_c_0), torch.exp(self.log_tau_0), torch.exp(self.log_beta)
    
    def predict_flux(self, observed_wavelengths, observed_fluxes, noise_variances, 
                    new_wavelengths, all_transition_wavelengths, all_oscillator_strengths, z_qso):
        """
        Computes the GP conditional prediction for missing pixels.

        Parameters:
        - observed_wavelengths: (num_pixels) Tensor of observed wavelengths
        - observed_fluxes: (num_pixels) Tensor of observed fluxes
        - noise_variances: (num_pixels) Tensor of noise variances
        - new_wavelengths: (num_new_pixels) Tensor of new wavelengths for prediction
        - all_transition_wavelengths: (31,) Tensor of Lyman series transition wavelengths
        - all_oscillator_strengths: (31,) Tensor of oscillator strengths
        - z_qso: Scalar tensor for quasar redshift

        Returns:
        - pred_flux: (num_new_pixels) Predicted mean flux at `new_wavelengths`
        - pred_var: (num_new_pixels) Predictive variance at `new_wavelengths`
        """

        # Retrieve model parameters (exponential transformation ensures positivity)
        M, omega2, c_0, tau_0, beta = self()

        # ---------------------- Step 1: Compute Lyα Optical Depth ---------------------- #
        lya_1pz = 1 + ((1 + z_qso) * observed_wavelengths - all_transition_wavelengths[0]) / all_transition_wavelengths[0]
        lya_optical_depth = tau_0 * torch.pow(lya_1pz, beta)

        for i in range(1, len(all_transition_wavelengths)):
            lyman_1pz = (all_transition_wavelengths[0] * lya_1pz) / all_transition_wavelengths[i]
            indicator = (lyman_1pz <= (1 + z_qso)).float()

            tau = (tau_0 * all_transition_wavelengths[i] * all_oscillator_strengths[i]) / \
                (all_transition_wavelengths[0] * all_oscillator_strengths[0])

            lya_optical_depth += tau * torch.pow(lyman_1pz, beta) * indicator

        lya_absorption = torch.exp(-lya_optical_depth)

        # ---------------------- Step 2: Compute Total Noise Variance ---------------------- #
        scaling_factor = 1 - lya_absorption + c_0
        absorption_noise = omega2 * torch.square(scaling_factor)

        D = noise_variances + absorption_noise  # Total noise
        D_inv = 1 / D  # Element-wise inverse

        # ---------------------- Step 3: Compute Low-Rank GP Covariance ---------------------- #
        # Split `M` into observed (M2) and missing (M1) wavelengths
        M2 = M[observed_wavelengths]
        M1 = M[new_wavelengths]

        D2_inv = D_inv[observed_wavelengths].unsqueeze(-1)  # Shape: (num_pixels, 1)

        D2_inv_M2 = D2_inv * M2  # Shape: (num_pixels, num_pca_components)

        B2 = M2.T @ D2_inv_M2  # Shape: (num_pca_components, num_pca_components)
        B2.diagonal().add_(1.0)  # Add identity for numerical stability

        # Cholesky decomposition for numerical stability
        L2 = torch.linalg.cholesky(B2)

        # Compute inverse using Woodbury identity
        C2 = torch.cholesky_solve(D2_inv_M2.T, L2)  # Shape: (num_pca_components, num_pixels)

        K22_inv_y2 = D2_inv.squeeze(-1) * observed_fluxes - D2_inv_M2 @ (C2 @ observed_fluxes)

        # ---------------------- Step 4: Compute Conditional Mean Prediction ---------------------- #
        Sigma12 = M1 @ M2.T  # Shape: (num_new_pixels, num_pixels)
        pred_flux = M1 @ (C2 @ observed_fluxes)  # Shape: (num_new_pixels)

        # ---------------------- Step 5: Compute Conditional Variance Prediction ---------------------- #
        K22_inv_Sigma21 = D2_inv * Sigma12.T - D2_inv_M2 @ (C2 @ Sigma12.T)
        pred_var = torch.diag(M1 @ M1.T) - torch.diag(Sigma12 @ K22_inv_Sigma21)

        return pred_flux, pred_var

class Trainer:
    """
    Trainer for optimizing the Gaussian Process model with PyTorch autograd support.
    Uses Mini-Batch Training with optional learning rate scheduling.
    """

    def __init__(self, gp_model, optimizer_type="adam", learning_rate=0.01, batch_size=32,
                 scheduler_type="cosine", scheduler_params=None):
        """
        Initialize the trainer.

        Parameters:
        - gp_model: GaussianProcessModel instance
        - optimizer_type: "adam" or "lbfgs"
        - learning_rate: Step size for optimizer
        - batch_size: Number of samples per mini-batch
        - scheduler_type: Type of learning rate scheduler ("cosine", "step", "reduce_on_plateau", None)
        - scheduler_params: Dictionary of parameters for the chosen scheduler
        """
        self.model = gp_model
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_history = []

        # Lists to save the parameter values
        # self.log_omega_values = []
        self.log_c_0_values = []
        self.log_tau_0_values = []
        self.log_beta_values = []


        # Choose optimizer
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif self.optimizer_type == "lbfgs":
            self.optimizer = optim.LBFGS(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer type must be either 'adam' or 'lbfgs'")

        # Default scheduler parameters if none provided
        if scheduler_params is None:
            scheduler_params = {}

        # Choose learning rate scheduler
        self.scheduler = None
        if scheduler_type == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_params)
        elif scheduler_type == "step":
            self.scheduler = lr_scheduler.StepLR(self.optimizer, **scheduler_params)
        elif scheduler_type == "reduce_on_plateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', **scheduler_params)

    def train(self, fluxes, lya_1pzs, noise_variances, z_qsos, num_forest_lines,
              all_transition_wavelengths, all_oscillator_strengths, max_epochs=500):
        """
        Trains the GP model using either Adam or L-BFGS with mini-batches.
        """

        # Create PyTorch DataLoader for mini-batches
        dataset = TensorDataset(fluxes, lya_1pzs, noise_variances, z_qsos)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        def closure():
            """Closure function for L-BFGS optimization."""
            self.optimizer.zero_grad()
            loss = objective(self.model, fluxes, lya_1pzs, noise_variances, num_forest_lines,
                             all_transition_wavelengths, all_oscillator_strengths, z_qsos)
            loss.backward()
            self.loss_history.append(loss.item())  

            # Save parameter values
            # self.log_omega_values.append(self.model.log_omega.item())
            self.log_c_0_values.append(self.model.log_c_0.item())
            self.log_tau_0_values.append(self.model.log_tau_0.item())
            self.log_beta_values.append(self.model.log_beta.item())

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
                    self.loss_history.append(loss.item())

                    # Save parameter values
                    # self.log_omega_values.append(self.model.log_omega.item())
                    self.log_c_0_values.append(self.model.log_c_0.item())
                    self.log_tau_0_values.append(self.model.log_tau_0.item())
                    self.log_beta_values.append(self.model.log_beta.item())


                # Step the scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(total_loss / len(dataloader))  # ReduceLROnPlateau needs loss as input
                    else:
                        self.scheduler.step()

                # Print progress
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader)}, Learning Rate = {self.optimizer.param_groups[0]['lr']:.6f}")
                    print(f"Epoch {epoch}: log_beta = {self.model.log_beta.item()}, log_tau_0 = {self.model.log_tau_0.item()}")

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