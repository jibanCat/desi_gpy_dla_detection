import torch
import numpy as np
import pytest
from gpy_dla_detection.learn_qso_model import GaussianProcessModel
from gpy_dla_detection.objective import objective, spectrum_loss
from gpy_dla_detection.voigt import transition_wavelengths, oscillator_strengths

# Define test parameters
num_pixels = 100
num_spectra = 10
k = 5  # PCA components
num_forest_lines = 10

@pytest.fixture
def generate_test_data():
    """Generate random test data for QSO spectra"""
    torch.manual_seed(42)  # Ensure reproducibility

    # Fake fluxes with Gaussian noise
    fluxes = torch.randn((num_spectra, num_pixels))

    # Fake noise variance (small positive values)
    noise_variances = torch.abs(torch.randn((num_spectra, num_pixels)) * 0.1)

    # Fake Lyman-alpha redshift scaling factors
    lya_1pzs = 1 + torch.linspace(2.0, 4.25, num_spectra).reshape(-1, 1).repeat(1, num_pixels)

    # Fake quasar redshifts
    z_qsos = torch.linspace(2.0, 4.25, num_spectra)

    return fluxes, noise_variances, lya_1pzs, z_qsos

@pytest.fixture
def initialize_gp_model():
    """Initialize GP model with fixed PCA components"""
    torch.manual_seed(42)  # Ensure reproducibility
    pca_eigenspectra = torch.randn(num_pixels, k)  # Generate fake PCA eigenspectra
    return GaussianProcessModel(num_pixels, k, pca_eigenspectra=pca_eigenspectra)

def test_spectrum_loss(generate_test_data, initialize_gp_model):
    """Test spectrum loss function outputs reasonable values"""
    fluxes, noise_variances, lya_1pzs, z_qsos = generate_test_data
    model = initialize_gp_model

    # Get model parameters
    M, omega2, c_0, tau_0, beta = model()

    # Compute loss for first spectrum
    loss = spectrum_loss(fluxes[0], lya_1pzs[0], noise_variances[0], M, omega2, c_0, tau_0, beta, 
                         num_forest_lines, torch.tensor(transition_wavelengths, dtype=torch.float32), 
                         torch.tensor(oscillator_strengths, dtype=torch.float32), z_qsos[0] + 1)

    # Check that loss is a finite number
    assert torch.isfinite(loss), "Spectrum loss should be finite"
    assert loss.item() > 0, "Loss should be positive"

def test_objective_function(generate_test_data, initialize_gp_model):
    """Test that the objective function computes total dataset loss correctly"""
    fluxes, noise_variances, lya_1pzs, z_qsos = generate_test_data
    model = initialize_gp_model

    # Compute full objective loss
    loss = objective(model, fluxes, lya_1pzs, noise_variances, num_forest_lines,
                     torch.tensor(transition_wavelengths, dtype=torch.float32),
                     torch.tensor(oscillator_strengths, dtype=torch.float32), z_qsos)

    # Check that loss is a finite number
    assert torch.isfinite(loss), "Objective loss should be finite"
    assert loss.item() > 0, "Objective loss should be positive"

def test_gradient_computation(generate_test_data, initialize_gp_model):
    """Ensure that gradients are computed correctly via autograd"""
    fluxes, noise_variances, lya_1pzs, z_qsos = generate_test_data
    model = initialize_gp_model

    # Compute loss with autograd tracking
    loss = objective(model, fluxes, lya_1pzs, noise_variances, num_forest_lines,
                     torch.tensor(transition_wavelengths, dtype=torch.float32),
                     torch.tensor(oscillator_strengths, dtype=torch.float32), z_qsos)

    # Backpropagate
    loss.backward()

    # Ensure gradients exist for each parameter
    for param in model.parameters():
        assert param.grad is not None, f"Gradient should be computed for {param}"
        assert torch.isfinite(param.grad).all(), "Gradients should not contain NaNs"

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_gp_loss.py"])