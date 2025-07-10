#!/bin/bash
#SBATCH --job-name=cddf                      # Job name
#SBATCH --output=logs/cddf_%j.out            # Standard output log (%j expands to job ID)
#SBATCH --error=logs/cddf_%j.err             # Standard error log
#SBATCH --time=01:30:00                      # Time limit
#SBATCH --nodes=1                            # Use one node
#SBATCH -C cpu                               # Use CPU node
#SBATCH -q regular                             # Use debug queue (change if needed)
#SBATCH -A desi                              # Account name

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
OCCAMS_RAZOR=${OCCAMS_RAZOR:-1}  # Default to 1 if unset
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"/pscratch/sd/j/jibancat/dla_cddf"}
HIGH_Z_QSO=${HIGH_Z_QSO:-6}  # Default to 6 if unset
LOW_Z_QSO=${LOW_Z_QSO:-2.1}  # Default to 2.1 if unset

# Optional flags (only included if set)
EXTRA_FLAGS=""
[ "${SUB_DLA:-0}" -eq 1 ] && EXTRA_FLAGS+=" --sub_dla"
[ "${HIGH_NHI_CUT:-0}" -eq 1 ] && EXTRA_FLAGS+=" --high_nhi_cut"
[ "${LOWZCUT:-0}" -eq 1 ] && EXTRA_FLAGS+=" --lowzcut"
[ "${HIGHZCUT:-0}" -eq 1 ] && EXTRA_FLAGS+=" --highzcut"

# extra for z_min_lyb
[ "${Z_MIN_LYB:-0}" -eq 1 ] && EXTRA_FLAGS+=" --z_min_lyb"

# high_nhi_cut_value
HIGH_NHI_CUT_VALUE=${HIGH_NHI_CUT_VALUE:-22.0}

# Bins per z
BINS_PER_Z=${BINS_PER_Z:-6}

# Run the Python script with parameters
python desi_cddf.py \
    --processed_file "$PROCESSED_FILE" \
    --sample_file "$SAMPLE_FILE" \
    --catalog_file "$CATALOG_FILE" \
    --snr "$SNR" \
    --second "$SECOND" \
    --output_prefix "$OUTPUT_PREFIX" \
    --occams_razor "$OCCAMS_RAZOR" \
    --high_z_qso "$HIGH_Z_QSO" \
    --low_z_qso "$LOW_Z_QSO" \
    $EXTRA_FLAGS \
    --min_obs_wavelength_cut \
    --min_obs_wavelength 3700 \
    --high_nhi_cut_value $HIGH_NHI_CUT_VALUE \
    --bins_per_z $BINS_PER_Z
