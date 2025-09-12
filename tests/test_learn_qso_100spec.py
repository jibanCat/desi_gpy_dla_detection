import pytest
import numpy as np
import torch
import time
from gpy_dla_detection.learn_qso_model import GaussianProcessModel, Trainer


@pytest.mark.benchmark
def test_gp_model_training_benchmark():
    """
    Test training of the GP model with 100 spectra (100 pixels each)
    in the redshift range z = [2.0, 4.25], and benchmark training time.
    """
    # Set up random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Constants
    num_spectra = 100  # Number of spectra
    num_pixels = 100  # Pixels per spectrum
    redshift_range = (2.0, 4.25)
    k = 10  # Number of PCA components

    # Generate synthetic test data
    rest_wavelengths = np.linspace(1040, 1200, num_pixels)  # Fixed rest-frame wavelengths
    fluxes = np.random.normal(loc=1.0, scale=0.1, size=(num_spectra, num_pixels))  # Simulated fluxes
    noise_variances = np.random.normal(loc=0.01, scale=0.005, size=(num_spectra, num_pixels))  # Noise
    noise_variances = np.abs(noise_variances)  # Ensure positivity

    z_qsos = np.random.uniform(*redshift_range, size=num_spectra)  # Random redshifts in range [2.0, 4.25]

    # Compute synthetic Lyα 1+z scaling factor
    lya_1pzs = 1 + np.tile(z_qsos[:, None], (1, num_pixels))

    # Initialize Gaussian Process Model
    gp_model = GaussianProcessModel(num_pixels, k)

    # Initialize Trainer
    trainer = Trainer(gp_model)

    # Benchmark training time
    start_time = time.time()
    trainer.train(fluxes, noise_variances, lya_1pzs, max_iter=10)  # Limit iterations for speed
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.3f} seconds for {num_spectra} spectra.")

    # Assertions to check if training modified model parameters
    for param in gp_model.parameters():
        assert param.grad is not None, "Gradient should be computed for all parameters."
        assert torch.any(param.grad != 0), "Gradient should not be zero."
    
if __name__ == "__main__":
    pytest.main(["-v", "tests/test_learn_qso_100spec.py"])