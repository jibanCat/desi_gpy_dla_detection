import pytest
import numpy as np
import torch
from gpy_dla_detection.learn_qso_model import GaussianProcessModel, DataLoader, SpectrumProcessor

# Test Gaussian Process Model Initialization
@pytest.mark.parametrize("num_pixels, k", [(200, 10), (300, 20)])
def test_gp_model(num_pixels, k):
    """Test Gaussian Process Model initialization."""
    model = GaussianProcessModel(num_pixels, k)
    assert model.M.shape == (num_pixels, k)
    assert model.log_omega.shape == (num_pixels,)

# Test Loading Data from Numpy
def test_data_loader_npy():
    """Test DataLoader with a dummy numpy file."""
    dummy_catalog = {"z_qsos": np.array([2.1, 3.0, 2.5])}
    np.save("dummy_catalog.npy", dummy_catalog)

    loader = DataLoader("dummy_catalog.npy", "dummy_catalog.npy")
    assert "z_qsos" in loader.catalog
    assert len(loader.catalog["z_qsos"]) == 3

# Test Rest-Frame Interpolation
def test_spectrum_processor():
    """Test that SpectrumProcessor correctly interpolates onto the rest-frame grid."""
    rest_wavelengths = np.linspace(1040, 1200, 200)
    processor = SpectrumProcessor(rest_wavelengths)

    wavelengths = np.array([1050, 1100, 1150, 1200])
    fluxes = np.array([1.0, 0.9, 0.8, 0.7])
    noise_variance = np.array([0.01, 0.02, 0.03, 0.04])
    z_qso = 2.0

    interp_flux, interp_noise = processor.interpolate_to_restframe(wavelengths, fluxes, noise_variance, z_qso)
    assert len(interp_flux) == len(rest_wavelengths)
    assert len(interp_noise) == len(rest_wavelengths)

if __name__ == "__main__":
    pytest.main()