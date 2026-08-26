#!/usr/bin/env python

import os
import argparse
import re
from astropy.table import Table, vstack


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine individual DLA catalog files into one FITS file."
    )

    parser.add_argument(
        "-r",
        "--release",
        type=str,
        default=None,
        required=True,
        help="DESI redux version (e.g. iron)",
    )

    parser.add_argument(
        "-p",
        "--program",
        type=str,
        default="dark",
        required=False,
        help="observing program, default is dark",
    )

    parser.add_argument(
        "-s",
        "--survey",
        type=str,
        default="main",
        required=False,
        help="survey, default is main",
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
    combined_filename = os.path.join(
        args.outdir, f"dlacat-{args.release}-{args.survey}-{args.program}.fits"
    )

    # Initialize an empty list to store individual tables
    tables = []

    # Regex pattern to match filenames like 'dlacat-kibo-main-dark-hpx-{start}-{end}.fits'
    file_pattern = re.compile(
        rf"dlacat-{args.release}-{args.survey}-{args.program}-hpx-(\d+)-(\d+)\.fits"
    )

    # Generate filenames based on the specified range and step size
    for start in range(args.initial, args.end + 1, args.step):
        end_range = start + args.step
        filename = f"dlacat-{args.release}-{args.survey}-{args.program}-hpx-{start}-{end_range}.fits"
        filepath = os.path.join(args.outdir, filename)

        # Verify if the file exists and matches the pattern
        if os.path.isfile(filepath) and file_pattern.match(filename):
            print(f"Reading {filepath}...")
            table = Table.read(filepath)

            # Rename column 'Z' to 'Z_QSO' if it exists
            if "Z" in table.colnames and "Z_QSO" not in table.colnames:
                table.rename_column("Z", "Z_QSO")

            tables.append(table)
        else:
            print(f"File {filename} not found or does not match pattern. Skipping...")

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
