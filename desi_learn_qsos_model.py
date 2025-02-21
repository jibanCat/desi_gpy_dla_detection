import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import os
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import interp1d
import sys
from gpy_dla_detection.learn_qso_model import (
    QSOLoader,
    GPTrainingSetLoader,
    SpectrumProcessor,
    GaussianProcessModel,
    Trainer,
)
from gpy_dla_detection.objective import objective
from gpy_dla_detection.voigt import transition_wavelengths as all_transition_wavelengths
from gpy_dla_detection.voigt import oscillator_strengths as all_oscillator_strengths
from gpy_dla_detection.learn_qso_model import compute_pca


class GPModelTrainer:
    """
    Handles the training process of a Gaussian Process (GP) model using quasar spectra.
    This includes data loading, preprocessing, training, and saving results.
    """

    def __init__(
        self,
        catalog_file,
        preloaded_file,
        z_range,
        min_snr,
        max_spectra,
        min_lambda,
        max_lambda,
        num_pixels,
        min_num_pixels,
        norm_min_lambda,
        norm_max_lambda,
        max_noise_variance,
        num_pca_components,
        learning_rate,
        batch_size,
        num_epochs,
        output_dir,
        sdss_test=False,
    ):
        """
        Initializes the GPModelTrainer with training parameters.
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize data loaders and processors
        if sdss_test:
            catalog_file = "data/dr12q/processed//catalog.mat"
            preloaded_file = "data/dr12q/processed/preloaded_qsos.mat"
            self.qso_loader = QSOLoader(catalog_file, preloaded_file, z_range=z_range, min_snr=min_snr, max_spectra=max_spectra)
        else:
            self.qso_loader = GPTrainingSetLoader(
                catalog_file, preloaded_file, z_range, min_snr, max_spectra
            )

        # Initialize spectrum processor: pre-processes the spectra
        self.spectrum_processor = SpectrumProcessor(
            min_lambda,
            max_lambda,
            num_pixels,
            norm_min_lambda,
            norm_max_lambda,
            max_noise_variance,
            min_num_pixels=min_num_pixels,
        )

        self.num_pca_components = num_pca_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        self.max_noise_variance = max_noise_variance

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir


    def prepare_data(self):
        """Loads and preprocesses the QSO spectra for training."""
        fluxes, all_rest_wavelengths, noise_variances, z_qsos = self.qso_loader.load_data()

        # --- Step 2: Normalize Spectra ---
        # (
        #     norm_fluxes,
        #     norm_noise_variances,
        #     all_rest_wavelengths,
        # ) = self.spectrum_processor.normalize_spectra(
        #     wavelengths, fluxes, noise_variances, z_qsos=z_qsos
        # )

        # --- Step 3: Mask Noisy Pixels BEFORE Interpolation ---
        (
            masked_fluxes,
            masked_noise_variances,
        ) = self.spectrum_processor.mask_noisy_pixels(fluxes, noise_variances)
        
        print("masked_fluxes shape:", len(masked_fluxes))

        # --- Step 4: Interpolate onto Common Rest-Frame Grid ---
        (
            fluxes_interpolated,
            noise_variances_interpolated,
            all_rest_wavelengths,
        ) = self.spectrum_processor.interpolate_spectra(
            all_rest_wavelengths, masked_fluxes, masked_noise_variances
        )
        print("fluxes_interpolated shape:", fluxes_interpolated.shape)

        # # --- Step 5: Fill in NaNs with Median BEFORE Deforesting ---
        # fluxes_interpolated = self.spectrum_processor.fill_nan_with_median(
        #     fluxes_interpolated
        # )
        # print("fluxes_median shape:", fluxes_interpolated.shape)

        # --- Step 6: Deforest the Spectra ---
        (
            deforest_fluxes,
            deforest_noise_variance,
        ) = self.spectrum_processor.de_forest_spectra(
            all_rest_wavelengths,
            fluxes_interpolated,
            noise_variances_interpolated,
            z_qsos=z_qsos,
        )
        print("deforest_fluxes shape:", deforest_fluxes.shape)

        # --- Step 7: Center the Flux for GP Training ---
        centered_fluxes, mu = self.spectrum_processor.center_fluxes(
            deforest_fluxes, deforest_noise_variance
        )
        self.mu = mu
        print("centered_fluxes shape:", centered_fluxes.shape)

        # --- Step 8: Remove NaN Spectra ---
        (
            centered_fluxes,
            deforest_noise_variance,
            all_rest_wavelengths,
            z_qsos,
        ) = self.spectrum_processor.remove_nan_spectra(
            centered_fluxes, deforest_noise_variance, all_rest_wavelengths, z_qsos
        )

        print("remove_nan_spectra shape:", centered_fluxes.shape)

        return (
            torch.tensor(centered_fluxes, dtype=torch.float32,),
            torch.tensor(
                deforest_noise_variance, dtype=torch.float32, 
            ),
            torch.tensor(z_qsos, dtype=torch.float32, ).unsqueeze(-1),
            # wavelength tensor
            torch.tensor(all_rest_wavelengths, dtype=torch.float32, ),
            centered_fluxes,
        )

    def train_model(self, if_use_template=False, initial_M=None):
        """Trains the Gaussian Process model using the prepared spectra."""
        from gpy_dla_detection.voigt import transition_wavelengths as all_transition_wavelengths
        from gpy_dla_detection.voigt import oscillator_strengths as all_oscillator_strengths

        fluxes_tensor, noise_variances_tensor, z_qsos_tensor, all_rest_wavelengths, centered_fluxes = self.prepare_data()

        if if_use_template:
            # TODO : understand if I can use these as init points
            print("Using template model ...")
            temp_model_path = "data/temp_model/QSO-HIZv1.1_RR.npz"
            assert os.path.exists(temp_model_path)
            temp_model = np.load(temp_model_path) # hard-coded path
            temp_pca = temp_model["PCA_COMP"] # normalized PCA components, centered 0
            temp_wave = 10**temp_model["LOGLAM"] # rest-frame wavelength

            # model rest-frame wavelength
            model_wave = all_rest_wavelengths[0].cpu().numpy()

            # interpolate PCA components
            temp_pca_interp = np.zeros((temp_pca.shape[0], len(model_wave)))

            for i in range(temp_pca.shape[0]):
                f = interp1d(temp_wave, temp_pca[i], kind="linear", fill_value="extrapolate")
                temp_pca_interp[i, :] = f(model_wave)

            # Replace the last few PCA components with the template PCA components
            coefficients, latent = compute_pca(fluxes_tensor, self.num_pca_components)
            # Compute initial M using PCA coefficients and square root of eigenvalues
            k = self.num_pca_components
            # Shape of coefficients: (num_pixels, num_pca_components)
            initial_M = coefficients[:, :k] * np.sqrt(latent[:k])  # Broadcasting happens automatically
            initial_M[:, -temp_pca_interp.shape[0]:] = temp_pca_interp.T

        ####### Initialize the GP model #######
        model = GaussianProcessModel(
            fluxes_tensor.shape[1], self.num_pca_components, centered_fluxes, initial_M=initial_M,
            min_lambda=self.min_lambda, max_lambda=self.max_lambda, mu=self.mu, max_noise_variance=self.max_noise_variance,
        ) 

        # Ensure `model.rest_wavelengths` remains on CPU
        model.rest_wavelengths = model.rest_wavelengths.cpu()  # ✅ Explicitly ensure it's on CPU

        # Compute Lyα redshift grid for training
        all_transition_wavelengths = torch.tensor(
            all_transition_wavelengths, dtype=torch.float32, 
        )
        all_oscillator_strengths = torch.tensor(
            all_oscillator_strengths, dtype=torch.float32, 
        )

        lya_wavelength = all_transition_wavelengths[0] * 1e8 

        # Ensure `z_qsos_tensor` is on CPU
        z_qsos_tensor = z_qsos_tensor.cpu()

        lya_1pz = 1 + (((1 + z_qsos_tensor) * model.rest_wavelengths.unsqueeze(0)) - lya_wavelength) / lya_wavelength


        self.model = model

        trainer = Trainer(
            model,
            optimizer_type="lbfgs",
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            scheduler_type="cosine",
            scheduler_params={"T_max": 50, "eta_min": 1e-5},
            output_dir=self.output_dir,
        )


        print("Before calling objective:")
        print(
            "all_transition_wavelengths shape:", all_transition_wavelengths.shape
        )  # Should be (31,)
        print(
            "all_oscillator_strengths shape:", all_oscillator_strengths.shape
        )  # Should also be (31,)

        # # Compute initial loss
        # initial_loss = objective(self.model, fluxes_tensor, lya_1pz, noise_variances_tensor, ...).detach()
        # print("Initial loss:", initial_loss)
        # print("After calling objective:")
        # print("all_transition_wavelengths shape:", all_transition_wavelengths.shape)
        # print("all_oscillator_strengths shape:", all_oscillator_strengths.shape)
        # Train the model
        trainer.train(
            fluxes_tensor,
            lya_1pz,
            noise_variances_tensor,
            z_qsos_tensor,
            3,
            all_transition_wavelengths,
            all_oscillator_strengths,
            max_epochs=self.num_epochs,
        )

        return self.model, trainer.loss_history


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Train a Gaussian Process Model on QSO Spectra"
    )
    parser.add_argument(
        "--catalog_file",
        type=str,
        required=True,
        help="Path to QSO catalog file",
    )
    parser.add_argument(
        "--preloaded_file",
        type=str,
        required=True,
        help="Path to preloaded QSO spectra file",
    )
    parser.add_argument(
        "--z_min", type=float, default=2.1, help="Minimum redshift for QSOs"
    )
    parser.add_argument(
        "--z_max", type=float, default=4.0, help="Maximum redshift for QSOs"
    )
    parser.add_argument(
        "--num_pca_components",
        type=int,
        default=10,
        help="Number of PCA components to use",
    )
    parser.add_argument(
        "--max_spectra",
        type=int,
        default=600000,
        help="Maximum number of spectra to use",
    )
    parser.add_argument(
        "--num_pixels",
        type=int,
        default=4000,
        help="Number of pixels in the spectra",
    )
    parser.add_argument(
        "--min_num_pixels",
        type=int,
        default=200,
        help="Minimum number of pixels in the spectra",
    )
    parser.add_argument(
        "--min_snr",
        type=float,
        default=0.0,
        help="Minimum SNR for QSO spectra",
    )
    parser.add_argument(
        "--min_lambda",
        type=float,
        default=911,
        help="Minimum rest wavelength for spectra",
    )
    parser.add_argument(
        "--max_lambda",
        type=float,
        default=1216,
        help="Maximum rest wavelength for spectra",
    )
    parser.add_argument(
        "--norm_min_lambda",
        type=float,
        default=900,
        help="Minimum rest wavelength for normalization",
    )
    parser.add_argument(
        "--norm_max_lambda",
        type=float,
        default=1200,
        help="Maximum rest wavelength for normalization",
    )
    parser.add_argument(
        "--max_noise_variance",
        type=float,
        default=9.0,
        help="Maximum allowed pixel noise variance",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save outputs"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.005,
        help="Learning rate for optimization",
    )
    parser.add_argument(
        "--batch_size", type=int, default=500, help="Batch size for training"
    )
    args = parser.parse_args()

    trainer = GPModelTrainer(
        catalog_file=args.catalog_file,
        preloaded_file=args.preloaded_file,
        z_range=(args.z_min, args.z_max),
        min_snr=args.min_snr,
        max_spectra=args.max_spectra,
        min_lambda=args.min_lambda,
        max_lambda=args.max_lambda,
        num_pixels=args.num_pixels,
        min_num_pixels=args.min_num_pixels,
        norm_min_lambda=args.norm_min_lambda,
        norm_max_lambda=args.norm_max_lambda,
        max_noise_variance=args.max_noise_variance,
        num_pca_components=args.num_pca_components,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
    )

    trained_model, loss_history = trainer.train_model()
