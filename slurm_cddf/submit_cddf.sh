#!/bin/bash
#SBATCH --job-name=cddf                      # Job name
#SBATCH --output=cddf_%j.out                 # Standard output log (%j expands to job ID)
#SBATCH --error=cddf_%j.err                  # Standard error log
#SBATCH --time=00:30:00                      # Time limit
#SBATCH --nodes=1                            # Use one node
#SBATCH -C cpu                               # Use CPU node
#SBATCH -q debug                             # Use debug queue (change if needed)
#SBATCH -A desi                              # Account name

# Load the required environment
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Ensure the logs directory exists (parameterized)
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

# Define input files (can be overridden at submission)
PROCESSED_FILE=${PROCESSED_FILE:-"/pscratch/sd/j/jibancat/desi-loa-gpdla-20250222-desi-learned/processed-main-dark.h5"}
SAMPLE_FILE=${SAMPLE_FILE:-"data/dr12q/processed/dla_samples_a03.mat"}
CATALOG_FILE=${CATALOG_FILE:-"data/desi/processed/catalog.fits"}

# Define hyperparameters (can be changed at submission)
SNR=${SNR:--2}  # Default to -2 if unset
SECOND=${SECOND:-1}  # Default to 1 if unset
OCCAMS_RAZOR=${OCCAMS_RAZOR:-1}  # Default to 1 if unset
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"/pscratch/sd/j/jibancat/dla_cddf"}

# Run the Python script with parameters
python desi_cddf.py \
    --processed_file "$PROCESSED_FILE" \
    --sample_file "$SAMPLE_FILE" \
    --catalog_file "$CATALOG_FILE" \
    --snr "$SNR" \
    --second "$SECOND" \
    --output_prefix "$OUTPUT_PREFIX" \
    --occams_razor "$OCCAMS_RAZOR"
