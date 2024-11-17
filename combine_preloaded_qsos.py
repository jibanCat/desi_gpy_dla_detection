"""
Combine selected HDF5 batch files into one file in order.
"""

import os
import h5py
import numpy as np


def combine_h5_files_in_order(
    temp_dir, output_file, batch_index_start, batch_step, file_prefix="temp_batch_"
):
    """
    Combine selected HDF5 files from a directory into a single HDF5 file in order.

    Parameters:
    - temp_dir (str): Directory containing temporary HDF5 files.
    - output_file (str): Path to save the combined HDF5 file.
    - batch_index_start (int): Starting batch index to process.
    - batch_step (int): Step size for batch indices.
    - file_prefix (str): Prefix of the temporary HDF5 files (default: "temp_batch_").
    """
    # Generate the list of batch indices to process
    batch_indices = list(range(batch_index_start, 100, batch_step))

    # Generate file paths based on indices
    temp_files = [
        os.path.join(temp_dir, f"{file_prefix}{batch_index}.h5")
        for batch_index in batch_indices
    ]

    # Filter out files that don't exist
    temp_files = [file for file in temp_files if os.path.exists(file)]
    if not temp_files:
        print(
            f"No valid HDF5 files found in directory: {temp_dir} with prefix: {file_prefix}"
        )
        return

    combined_data = {}
    file_count = 0

    # Read and combine data from each file
    for temp_file in temp_files:
        print(f"Processing file: {temp_file}")
        with h5py.File(temp_file, "r") as f:
            for key in f.keys():
                data = f[key][:]
                if key not in combined_data:
                    combined_data[key] = [data]
                else:
                    combined_data[key].append(data)
        file_count += 1

    # Combine arrays for each key
    for key in combined_data.keys():
        print(f"Combining key: {key}")
        try:
            combined_data[key] = np.concatenate(combined_data[key], axis=0)
        except ValueError:
            print(f"Warning: Could not concatenate key '{key}'. Keeping as list.")

    # Write combined data to a single HDF5 file
    print(f"Writing combined results to {output_file}")
    with h5py.File(output_file, "w") as f:
        for key, data in combined_data.items():
            f.create_dataset(key, data=data)
        f.attrs["combined_files"] = file_count
        f.attrs["source_directory"] = temp_dir
        f.attrs["file_order"] = [os.path.basename(file) for file in temp_files]
        f.attrs["batch_index_start"] = batch_index_start
        f.attrs["batch_step"] = batch_step

    print(f"Combined results saved to {output_file}")


if __name__ == "__main__":
    # Manually set the arguments
    temp_dir = "temp_batches"
    output_file = "temp_batches/preloaded_qsos.h5"
    batch_index_start = 0  # Starting index for batches
    batch_step = 1  # Step size for batch indices

    # Call the function
    combine_h5_files_in_order(temp_dir, output_file, batch_index_start, batch_step)
