#!/bin/bash

# Set the path to your SLURM script
SLURM_SCRIPT="submit_desi_kibo_error_patch.sh"

# Loop over START_INDEX values from 0 to 40, incrementing by 8
for START_INDEX in $(seq 0 8 40); do
    echo "Submitting job with START_INDEX=$START_INDEX"

    # Submit the SLURM script with the specific START_INDEX
    echo "sbatch --export=ALL,START_INDEX=$START_INDEX "$SLURM_SCRIPT""
    sbatch --export=ALL,START_INDEX=$START_INDEX "$SLURM_SCRIPT"
    
    # Optional: Sleep between submissions to stagger job starts slightly
    sleep 1m
done