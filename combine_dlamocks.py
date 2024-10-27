#!/usr/bin/env python

import os
import argparse
from astropy.table import Table, vstack


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine individual DLA catalog files into one FITS file."
    )
    parser.add_argument(
        "--release",
        type=str,
        required=True,
        help="Release version (e.g., v5.9.5).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory containing individual FITS files and for saving the combined file.",
    )
    parser.add_argument(
        "--initial",
        type=int,
        required=True,
        help="Starting index for file range.",
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="Ending index for file range.",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Step size for file indexing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Define the output filename for the combined table
    combined_filename = os.path.join(args.outdir, f"dlacat-{args.release}-mockcat.fits")

    # Initialize an empty list to store individual tables
    tables = []

    # Generate filenames based on the range and step size, and read each file
    for start in range(args.initial, args.end + 1, args.step):
        end_range = start + args.step
        filename = os.path.join(
            args.outdir, f"dlacat-{args.release}-mockcat-{start}-{end_range}.fits"
        )

        # Check if the file exists before reading
        if os.path.isfile(filename):
            print(f"Reading {filename}...")
            table = Table.read(filename)

            # Rename column 'Z' to 'Z_QSO' if it exists
            if "Z" in table.colnames and "Z_QSO" not in table.colnames:
                table.rename_column("Z", "Z_QSO")

            tables.append(table)
        else:
            print(f"File {filename} not found. Skipping...")

    # Combine all tables into one if any were loaded
    if tables:
        print("Combining tables...")
        combined_table = vstack(tables)

        # Write the combined table to a new FITS file
        print(f"Writing combined table to {combined_filename}...")
        combined_table.write(combined_filename, overwrite=True)

        print("Combination complete.")
    else:
        print("No files were loaded. Check the file paths or parameters.")


if __name__ == "__main__":
    main()
