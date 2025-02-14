#!/bin/bash
#SBATCH -N 1                                # Number of nodes
#SBATCH -C cpu                              # CPU type (regular CPU)
#SBATCH -q regular                          # Queue (regular for longer runs)
#SBATCH --job-name=prepare_trainset         # Job name
#SBATCH --output=logs/prepare_trainset_%j.log  # Standard output log (%j = job ID)
#SBATCH --error=logs/error_train_%j.log        # Standard error log
#SBATCH --mail-user=mfho@umich.edu          # Email for notifications
#SBATCH --mail-type=ALL                     # Notification options
#SBATCH -A desi                             # NERSC account
#SBATCH --time=00:30:00                     # Time limit
#SBATCH --ntasks=32                         # Number of tasks
#SBATCH --export=ALL                        # Export all environment variables

# Exit on error
set -e

# Define variables for flexibility (can be overridden at submission time)
INPUT_DIR=${INPUT_DIR:-"/pscratch/sd/j/jibancat/preload-loa-gpdla-20250202/preloaded"}
OUTPUT_FILE=${OUTPUT_FILE:-"/pscratch/sd/j/jibancat/preload-loa-gpdla-20250202/gp_interp_trainset.h5"}
MIN_LAMBDA=${MIN_LAMBDA:-850.75}
MAX_LAMBDA=${MAX_LAMBDA:-1420.75}
DLAMBDA=${DLAMBDA:-0.15}
NORM_MIN_LAMBDA=${NORM_MIN_LAMBDA:-1425}
NORM_MAX_LAMBDA=${NORM_MAX_LAMBDA:-1475}
MAX_NOISE_VARIANCE=${MAX_NOISE_VARIANCE:-9}

# Load environment
source /global/cfs/cdirs/desi/software/desi_environment.sh main

echo "Starting GP Training Set Preparation..."
echo "Input Directory: $INPUT_DIR"
echo "Output File: $OUTPUT_FILE"
echo "Min Lambda: $MIN_LAMBDA"
echo "Max Lambda: $MAX_LAMBDA"
echo "Delta Lambda: $DLAMBDA"
echo "Normalization Min Lambda: $NORM_MIN_LAMBDA"
echo "Normalization Max Lambda: $NORM_MAX_LAMBDA"
echo "Max Noise Variance: $MAX_NOISE_VARIANCE"

# Run the Python script
python gp_training_prep.py \
    --input_dir "$INPUT_DIR" \
    --output_file "$OUTPUT_FILE" \
    --min_lambda "$MIN_LAMBDA" \
    --max_lambda "$MAX_LAMBDA" \
    --dlambda "$DLAMBDA" \
    --norm_min_lambda "$NORM_MIN_LAMBDA" \
    --norm_max_lambda "$NORM_MAX_LAMBDA" \
    --max_noise_variance "$MAX_NOISE_VARIANCE"

echo "GP Training Set Preparation Completed!"

# sbatch --export=INPUT_DIR="/new/path/to/data",OUTPUT_FILE="new_output.h5" prepare_trainset.slurm