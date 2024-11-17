"""
Combine individual processed QSO files into a single HDF5 file.
"""

import os
import numpy as np
import h5py
from preload_qsos import read_catalog


# Default paths and arguments
DEFAULT_CATALOG_PATH = "/global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits"
DEFAULT_PROCESSED_DIR = (
    "/pscratch/sd/j/jibancat/desi-kibo-gpdla-nobal-2_15-7-nozwarn/processed"
)
DEFAULT_OUTPUT_FILE = "/pscratch/sd/j/jibancat/desi-kibo-gpdla-nobal-2_15-7-nozwarn/processed-main-dark.h5"
DEFAULT_SURVEY = "main"
DEFAULT_PROGRAM = "dark"
DEFAULT_RELEASE = "kibo"


def construct_filename(processed_dir, healpix):
    """
    Construct the file path for a given healpix.

    Args:
        processed_dir (str): Directory containing processed files.
        healpix (int): Healpix pixel number.

    Returns:
        str: Full file path for the healpix.
    """
    return os.path.join(
        processed_dir, f"processed-{DEFAULT_SURVEY}-{DEFAULT_PROGRAM}-{healpix}.h5"
    )


def load_healpix_from_catalog(catalog_path):
    """
    Load unique healpix values from the QSO catalog.

    Args:
        catalog_path (str): Path to the QSO catalog.

    Returns:
        np.ndarray: Unique healpix pixel values.
    """
    catalog = read_catalog(catalog_path, balmask=True, bytile=False)
    if "HPXPIXEL" not in catalog.colnames:
        raise ValueError("Catalog does not contain 'HPXPIXEL' column.")
    return np.unique(catalog["HPXPIXEL"])


def combine_processed_files(processed_dir, healpix_list, output_file):
    """
    Combine individual processed HDF5 files into a single file.

    Args:
        processed_dir (str): Directory containing individual processed HDF5 files.
        healpix_list (np.ndarray): List of healpix pixels to combine.
        output_file (str): Path to save the combined HDF5 file.
    """
    combined_results = {}
    processed_files = []

    for healpix in healpix_list:
        filepath = construct_filename(processed_dir, healpix)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}. Skipping...")
            continue

        processed_files.append(filepath)
        print(f"Reading file: {filepath}")

        with h5py.File(filepath, "r") as f:
            for key in f.keys():
                data = f[key][:]
                if key not in combined_results:
                    combined_results[key] = [data]
                else:
                    combined_results[key].append(data)

    if not processed_files:
        print("No processed files were found. Exiting.")
        return

    # Combine arrays for each key
    for key in combined_results.keys():
        print(f"Combining key: {key}")
        try:
            combined_results[key] = np.concatenate(combined_results[key], axis=0)
        except ValueError:
            print(f"Warning: Could not concatenate key '{key}'. Keeping as list.")

    # Save combined results to a single HDF5 file
    print(f"Writing combined results to {output_file}")
    with h5py.File(output_file, "w") as f:
        for key, data in combined_results.items():
            f.create_dataset(key, data=data)
        f.attrs["combined_files"] = len(processed_files)
        f.attrs["healpix_combined"] = list(healpix_list)

    print(f"Combined results saved to {output_file}")


def main():
    # Use default paths and arguments
    catalog_path = DEFAULT_CATALOG_PATH
    processed_dir = DEFAULT_PROCESSED_DIR
    output_file = DEFAULT_OUTPUT_FILE

    print(f"Using catalog: {catalog_path}")
    print(f"Reading processed files from: {processed_dir}")
    print(f"Saving combined results to: {output_file}")

    # Load unique healpix pixels from the catalog
    healpix_list = load_healpix_from_catalog(catalog_path)

    # Combine processed files
    combine_processed_files(
        processed_dir=processed_dir,
        healpix_list=healpix_list,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
