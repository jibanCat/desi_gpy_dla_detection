import pytest
import torch
import numpy as np
from gpy_dla_detection.learn_qso_model import GaussianProcessModel, Trainer
from gpy_dla_detection.objective import objective
from gpy_dla_detection.voigt import transition_wavelengths, oscillator_strengths

@pytest.fixture
def generate_mock_spectra():
    """Generate mock spectra data for testing."""
    num_spectra = 10  # Small batch for testing
    num_pixels = 100  # Reduced for speed
    k = 5  # Number of PCA components

    # Generate random flux values and noise variances
    torch.manual_seed(42)
    fluxes = torch.randn(num_spectra, num_pixels)
    noise_variances = torch.abs(torch.randn(num_spectra, num_pixels) * 0.1)  # Avoid negative noise

    # Simulated Lyman-alpha redshift factors
    lya_1pzs = torch.full((num_spectra, num_pixels), 3.0)  # Constant 1+z for simplicity
    z_qsos = torch.linspace(2.0, 4.5, num_spectra)

    return fluxes, lya_1pzs, noise_variances, z_qsos

@pytest.fixture
def initialize_gp_model(generate_mock_spectra):
    """Initialize GP model with mock PCA components."""
    _, _, _, _ = generate_mock_spectra
    num_pixels = 100
    k = 5
    pca_eigenspectra = torch.randn(num_pixels, k)  # Random PCA basis for testing
    return GaussianProcessModel(num_pixels, k, pca_eigenspectra)

def test_training_loss_decreasing(generate_mock_spectra, initialize_gp_model):
    """Test that the objective function loss decreases during training."""
    fluxes, lya_1pzs, noise_variances, z_qsos = generate_mock_spectra
    model = initialize_gp_model

    num_forest_lines = 10
    all_transition_wavelengths = torch.tensor(transition_wavelengths, dtype=torch.float32)
    all_oscillator_strengths = torch.tensor(oscillator_strengths, dtype=torch.float32)

    # Initialize trainer with a small batch size for testing
    trainer = Trainer(model, optimizer_type="adam", learning_rate=0.01, batch_size=2)
    num_epochs = 10

    # Store initial loss
    initial_loss = objective(model, fluxes, lya_1pzs, noise_variances, num_forest_lines,
                             all_transition_wavelengths, all_oscillator_strengths, z_qsos).item()

    # Train the model
    trainer.train(fluxes, lya_1pzs, noise_variances, z_qsos, num_forest_lines,
                  all_transition_wavelengths, all_oscillator_strengths, max_epochs=num_epochs)

    # Extract loss history
    loss_history = trainer.loss_history

    # Ensure loss is decreasing over time
    assert len(loss_history) > 1, "Training did not iterate over multiple epochs"
    assert loss_history[-1] < initial_loss, "Training loss did not decrease"

    # Print final loss trend (for debugging)
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {loss_history[-1]:.4f}")