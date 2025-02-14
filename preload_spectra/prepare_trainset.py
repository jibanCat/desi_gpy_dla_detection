import os
import glob
import h5py
import argparse
import numpy as np
from gpy_dla_detection.learn_qso_model import SpectrumProcessor

class GPTrainingSetPreparer:
    """
    Prepares the training set for a Gaussian Process (GP) model by:
    - Loading preloaded QSO spectra from HDF5 files
    - Masking noisy pixels
    - Interpolating spectra onto a common wavelength grid
    - Saving processed data to an output HDF5 file
    """
    
    def __init__(self, input_dir, output_file, min_lambda, max_lambda, dlambda,
                 norm_min_lambda, norm_max_lambda, max_noise_variance):
        """
        Initializes the training set preparer with the given parameters.
        
        :param input_dir: Directory containing preloaded HDF5 files
        :param output_file: Path for the output HDF5 file
        :param min_lambda: Minimum rest-frame wavelength
        :param max_lambda: Maximum rest-frame wavelength
        :param dlambda: Grid separation in wavelength
        :param norm_min_lambda: Minimum wavelength for flux normalization
        :param norm_max_lambda: Maximum wavelength for flux normalization
        :param max_noise_variance: Maximum allowable noise variance per pixel
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.dlambda = dlambda
        self.num_pixels = int((max_lambda - min_lambda) / dlambda) + 1
        self.norm_min_lambda = norm_min_lambda
        self.norm_max_lambda = norm_max_lambda
        self.max_noise_variance = max_noise_variance

    def load_filelist(self):
        """
        Retrieves a list of preloaded HDF5 files from the input directory.
        
        :return: List of file paths
        """
        filelist = glob.glob(os.path.join(self.input_dir, "preloaded*.h5"))
        print(f"Found {len(filelist)} files")
        return filelist

    def process_files(self, filelist):
        """
        Processes spectra from a list of HDF5 files, performing noise masking
        and interpolation onto a common grid.
        
        :param filelist: List of file paths to process
        :return: Processed data arrays
        """
        all_tids, all_rest_wavelengths, all_fluxes = [], [], []
        all_noise_variance, all_zqso, all_redsnr, all_bluesnr = [], [], [], []

        # Initialize spectrum processor
        spectrum_processor = SpectrumProcessor(
            min_lambda=self.min_lambda,
            max_lambda=self.max_lambda,
            num_pixels=self.num_pixels,
            norm_min_lambda=self.norm_min_lambda,
            norm_max_lambda=self.norm_max_lambda,
            max_noise_variance=self.max_noise_variance,
        )

        for i, this_file in enumerate(filelist):
            print(f"Processing file: {i+1}/{len(filelist)} {this_file}")
            with h5py.File(this_file, "r") as f:
                tidlist = f["tidlist"][:]
                this_rest_wavelengths = f["rest_wavelength_list"][:]
                this_fluxes = f["flux_list"][:]
                this_noise_variance = f["noise_variance_list"][:]
                this_zqso = f["zqsolist"][:]
                this_redsnr = f["redsnrlist"][:]
                this_bluesnr = f["bluesnrlist"][:]

            # Mask noisy pixels
            masked_fluxes, masked_noise_variances = spectrum_processor.mask_noisy_pixels(
                this_fluxes, this_noise_variance
            )

            # Interpolate onto common grid
            fluxes_interpolated, noise_variances_interpolated, this_rest_wavelengths = (
                spectrum_processor.interpolate_spectra(
                    this_rest_wavelengths, masked_fluxes, masked_noise_variances
                )
            )

            # Collect processed data
            all_tids.append(tidlist)
            all_rest_wavelengths.append(this_rest_wavelengths)
            all_fluxes.append(fluxes_interpolated)
            all_noise_variance.append(noise_variances_interpolated)
            all_zqso.append(this_zqso)
            all_redsnr.append(this_redsnr)
            all_bluesnr.append(this_bluesnr)

        return all_tids, all_rest_wavelengths, all_fluxes, all_noise_variance, all_zqso, all_redsnr, all_bluesnr

    def save_data(self, all_tids, all_rest_wavelengths, all_fluxes, all_noise_variance, all_zqso, all_redsnr, all_bluesnr):
        """
        Saves processed data to an output HDF5 file.
        
        :param all_tids: List of QSO IDs
        :param all_rest_wavelengths: List of rest-frame wavelengths
        :param all_fluxes: List of flux values
        :param all_noise_variance: List of noise variance values
        :param all_zqso: List of redshift values
        :param all_redsnr: List of red-side SNR values
        :param all_bluesnr: List of blue-side SNR values
        """
        print(f"Saving the data to: {self.output_file}")
        with h5py.File(self.output_file, "w") as f:
            f.create_dataset("tids", data=np.concatenate(all_tids))
            f.create_dataset("rest_wavelengths", data=np.concatenate(all_rest_wavelengths))
            f.create_dataset("fluxes", data=np.concatenate(all_fluxes))
            f.create_dataset("noise_variance", data=np.concatenate(all_noise_variance))
            f.create_dataset("zqso", data=np.concatenate(all_zqso))
            f.create_dataset("redsnr", data=np.concatenate(all_redsnr))
            f.create_dataset("bluesnr", data=np.concatenate(all_bluesnr))

    def run(self):
        """Executes the complete pipeline of loading, processing, and saving data."""
        filelist = self.load_filelist()
        data = self.process_files(filelist)
        self.save_data(*data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the training set for the GP model.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing preloaded .h5 files.")
    parser.add_argument("--output_file", type=str, default="trainset.h5", help="Output HDF5 file.")
    parser.add_argument("--min_lambda", type=float, default=850.75, help="Minimum rest wavelength (Å).")
    parser.add_argument("--max_lambda", type=float, default=1420.75, help="Maximum rest wavelength (Å).")
    parser.add_argument("--dlambda", type=float, default=0.15, help="Wavelength grid separation (Å).")
    parser.add_argument("--norm_min_lambda", type=float, default=1425, help="Normalization min wavelength (Å).")
    parser.add_argument("--norm_max_lambda", type=float, default=1475, help="Normalization max wavelength (Å).")
    parser.add_argument("--max_noise_variance", type=float, default=9, help="Maximum allowed pixel noise variance.")
    
    args = parser.parse_args()
    preparer = GPTrainingSetPreparer(**vars(args))
    preparer.run()
