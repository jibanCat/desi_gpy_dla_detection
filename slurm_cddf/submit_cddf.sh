#!/bin/bash
#SBATCH --job-name=combine_processed_h5     # Job name
#SBATCH --output=logs/cddf_%j.out          # Standard output log (%j expands to job ID)
#SBATCH --error=logs/cddf_%j.err           # Standard error log (%j expands to job ID)
#SBATCH --time=00:30:00                    # Time limit
#SBATCH --nodes=1                          # Use one node
#SBATCH -C cpu                             # Use CPU node
#SBATCH -q debug                           # Use regular queue
#SBATCH -A desi                            # Account name

# Load the required environment
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Ensure the logs directory exists
mkdir -p logs

# Define input files (can be overridden at submission)
PROCESSED_FILE=${PROCESSED_FILE:-"/pscratch/sd/j/jibancat/desi-loa-gpdla-20250222-desi-learned/processed-main-dark.h5"}
SAMPLE_FILE=${SAMPLE_FILE:-"data/dr12q/processed/dla_samples_a03.mat"}
CATALOG_FILE=${CATALOG_FILE:-"data/desi/processed/catalog.fits"}

# Define hyperparameters (can be changed at submission)
SNR=${SNR:--2}  # Default to -2 if unset
SECOND=${SECOND:-1}  # Default to 1 if unset
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"/pscratch/sd/j/jibancat/dla_cddf"}

# Run the Python script with parameters
python desi_cddf.py \
    --processed_file "$PROCESSED_FILE" \
    --sample_file "$SAMPLE_FILE" \
    --catalog_file "$CATALOG_FILE" \
    --snr "$SNR" \
    --second "$SECOND" \
    --output_prefix "$OUTPUT_PREFIX"
