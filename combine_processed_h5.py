import os
import argparse
import numpy as np
import h5py
from astropy.table import Table
from utilities.read_catalogs import read_catalog
from desiutil.log import log

def construct_filename(processed_dir, survey, program, healpix):
    """
    Construct the file path for a given healpix.
    """
    return os.path.join(processed_dir, f"processed-{survey}-{program}-{healpix}.h5")

def load_healpix_from_catalog(catalog_path):
    """
    Load unique healpix values from the QSO catalog.
    """
    catalog = read_catalog(catalog_path, balmask=True, bytile=False)
    if "HPXPIXEL" not in catalog.colnames:
        raise ValueError("Catalog does not contain 'HPXPIXEL' column.")
    return np.unique(catalog["HPXPIXEL"])

def get_healpix_from_folder(processed_dir, survey, program):
    """
    Get healpix values from available files in the processed directory.
    """
    files = [f for f in os.listdir(processed_dir) if f.startswith(f"processed-{survey}-{program}-") and f.endswith(".h5")]
    healpix_list = [int(f.split("-")[-1].split(".")[0]) for f in files]
    return np.unique(healpix_list)

def filter_data_by_target_ids(data_dict, target_ids, selected_target_ids):
    """
    Filter the data dictionary to only include selected target IDs.
    """
    mask = np.isin(target_ids, selected_target_ids)
    log.info(f"Found {np.sum(mask)} matching target IDs.")
    if not np.any(mask):
        return None  # Return None if no matching target IDs
    
    return {key: data[mask] for key, data in data_dict.items()}

def combine_processed_files(processed_dir, healpix_list, output_file, survey, program, target_catalog):
    """
    Combine individual processed HDF5 files into a single file, filtering by target IDs.
    """
    combined_results = {}
    processed_files = []
    
    selected_target_ids = target_catalog["TARGETID"]
    
    for healpix in healpix_list:
        filepath = construct_filename(processed_dir, survey, program, healpix)
        
        if not os.path.exists(filepath):
            log.info(f"File not found: {filepath}. Skipping...")
            continue

        processed_files.append(filepath)
        log.info(f"Reading processed file: {filepath}")

        with h5py.File(filepath, "r") as f:
            target_ids = f["target_ids"][:]
            data_dict = {key: f[key][:] for key in f.keys()}
            filtered_data = filter_data_by_target_ids(data_dict, target_ids, selected_target_ids)
            
            if filtered_data is None:
                log.info(f"No matching target IDs found in {filepath}. Skipping...")
                continue
            
            for key, data in filtered_data.items():
                if key not in combined_results:
                    combined_results[key] = [data]
                else:
                    combined_results[key].append(data)
    
    if not processed_files:
        log.info("No processed files were found. Exiting.")
        return

    # Combine arrays for each key
    for key in combined_results.keys():
        log.info(f"Combining key: {key}")
        try:
            combined_results[key] = np.concatenate(combined_results[key], axis=0)
        except ValueError:
            log.info(f"Warning: Could not concatenate key '{key}'. Keeping as list.")

    # Save combined results to a single HDF5 file
    log.info(f"Writing combined results to {output_file}")
    with h5py.File(output_file, "w") as f:
        for key, data in combined_results.items():
            f.create_dataset(key, data=data)
        f.attrs["combined_files"] = len(processed_files)
    
    log.info(f"Combined results saved to {output_file}")

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Combine processed QSO files into a single HDF5 file, filtering by target IDs.")
    parser.add_argument("--catalog", type=str, default="/global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits",
                        help="Path to the original catalog file.")
    parser.add_argument("--load_catalog", type=str, default="processed_to_load.fits",
                        help="Path to the catalog file containing target IDs to load.")
    parser.add_argument("--processed_dir", type=str, default="/pscratch/sd/j/jibancat/desi-kibo-gpdla-nobal-2_15-7-nozwarn/processed",
                        help="Directory containing processed files.")
    parser.add_argument("--output_file", type=str, default="/pscratch/sd/j/jibancat/desi-kibo-gpdla-nobal-2_15-7-nozwarn/processed-main-dark.h5",
                        help="Path to save the combined HDF5 file.")
    parser.add_argument("--survey", type=str, default="main", help="Survey name (default: main)")
    parser.add_argument("--program", type=str, default="dark", help="Program name (default: dark)")
    parser.add_argument("--mock", action="store_true", help="Enable this flag to process mock data without using a catalog file.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    log.info(f"Reading processed files from: {args.processed_dir}")
    log.info(f"Saving combined results to: {args.output_file}")
    
    # Load the target catalog for filtering
    target_catalog = Table.read(args.load_catalog)
    
    # Load healpix pixels either from catalog or from folder
    if args.mock:
        log.info("Processing mock data. Extracting healpix values from available files.")
        healpix_list = get_healpix_from_folder(args.processed_dir, args.survey, args.program)
    else:
        log.info(f"Using catalog: {args.catalog}")
        healpix_list = load_healpix_from_catalog(args.catalog)
    
    # Combine processed files with filtering
    combine_processed_files(
        processed_dir=args.processed_dir,
        healpix_list=healpix_list,
        output_file=args.output_file,
        survey=args.survey,
        program=args.program,
        target_catalog=target_catalog
    )

if __name__ == "__main__":
    main()
