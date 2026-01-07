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
    files = [
        f for f in os.listdir(processed_dir)
        if f.startswith(f"processed-{survey}-{program}-") and f.endswith(".h5")
    ]
    healpix_list = [int(f.split("-")[-1].split(".")[0]) for f in files]
    return np.unique(healpix_list)


def filter_data_by_target_ids(target_ids, selected_target_ids):
    """
    Filter helper: return a boolean mask for selected TARGETIDs.

    ### CHANGED: previously this function sliced a whole data_dict (which required
    ### loading everything into RAM). Now it only returns the mask.
    """
    mask = np.isin(target_ids, selected_target_ids)
    nmatch = int(np.sum(mask))
    log.info(f"Found {nmatch} matching target IDs.")
    if nmatch == 0:
        return None
    return mask


def _append_dataset(out_f, key, data, written_counts, compression=None):
    """
    Append data to out_f[key] (create resizable dataset on first write).

    ### CHANGED: new helper to avoid accumulating arrays + np.concatenate.
    """
    data = np.asarray(data)

    if key not in out_f:
        # Create dataset lazily with unlimited first dimension
        maxshape = (None,) + data.shape[1:]
        out_f.create_dataset(
            key,
            data=data,
            maxshape=maxshape,
            chunks=True,
            compression=compression,
        )
        written_counts[key] = data.shape[0]
        return

    dset = out_f[key]
    old_n = written_counts[key]
    new_n = old_n + data.shape[0]
    dset.resize((new_n,) + dset.shape[1:])
    dset[old_n:new_n, ...] = data
    written_counts[key] = new_n


def combine_processed_files(processed_dir, healpix_list, output_file, survey, program, target_catalog):
    """
    Combine individual processed HDF5 files into a single file, filtering by target IDs.

    ### CHANGED: stream write to output file instead of keeping everything in RAM and concatenating.
    """
    processed_files = []
    selected_target_ids = np.asarray(target_catalog["TARGETID"])

    log.info(f"Writing combined results to {output_file}")

    # Track how many rows written per dataset key (in case some keys are missing in some files)
    ### CHANGED
    written_counts = {}

    with h5py.File(output_file, "w") as out_f:  # ### CHANGED: open output once
        for healpix in healpix_list:
            filepath = construct_filename(processed_dir, survey, program, healpix)

            if not os.path.exists(filepath):
                log.info(f"File not found: {filepath}. Skipping...")
                continue

            processed_files.append(filepath)
            log.info(f"Reading processed file: {filepath}")

            with h5py.File(filepath, "r") as f:
                if "target_ids" not in f:
                    log.info(f"Missing 'target_ids' in {filepath}. Skipping...")
                    continue

                # We need target_ids to compute mask (this is unavoidable)
                target_ids = f["target_ids"][:]

                mask = filter_data_by_target_ids(target_ids, selected_target_ids)
                if mask is None:
                    log.info(f"No matching target IDs found in {filepath}. Skipping...")
                    continue

                # ### CHANGED: do NOT do data_dict = {key: f[key][:] ...}
                # Instead, slice each dataset on demand and immediately append to output.
                for key in f.keys():
                    d = f[key]

                    # Keep behavior consistent with your original assumption:
                    # all datasets should be row-aligned with target_ids on axis=0.
                    # If a dataset is scalar or has mismatched first dimension, we skip it
                    # rather than silently creating nonsense.
                    if d.shape == ():
                        log.info(f"Skipping scalar dataset '{key}' in {filepath}")
                        continue
                    if d.shape[0] != target_ids.shape[0]:
                        raise ValueError(
                            f"{filepath}: dataset '{key}' has shape {d.shape}, "
                            f"expected first dim {target_ids.shape[0]}"
                        )

                    data = d[mask, ...]  # only read selected rows
                    _append_dataset(out_f, key, data, written_counts, compression=None)

        # preserve your attr
        out_f.attrs["combined_files"] = len(processed_files)

    if not processed_files:
        log.info("No processed files were found. Exiting.")
        return

    log.info(f"Combined results saved to {output_file}")


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Combine processed QSO files into a single HDF5 file, filtering by target IDs."
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default="/global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits",
        help="Path to the original catalog file.",
    )
    parser.add_argument(
        "--load_catalog",
        type=str,
        default="processed_to_load.fits",
        help="Path to the catalog file containing target IDs to load.",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="/pscratch/sd/j/jibancat/desi-kibo-gpdla-nobal-2_15-7-nozwarn/processed",
        help="Directory containing processed files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/pscratch/sd/j/jibancat/desi-kibo-gpdla-nobal-2_15-7-nozwarn/processed-main-dark.h5",
        help="Path to save the combined HDF5 file.",
    )
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
        target_catalog=target_catalog,
    )


if __name__ == "__main__":
    main()