#!/bin/bash
#SBATCH --job-name=preload_batches       # Job name
#SBATCH --output=logs/batch_%A_%a.out   # Standard output log
#SBATCH --error=logs/batch_%A_%a.err    # Standard error log
#SBATCH --time=00:30:00                 # Debug queue time limit
#SBATCH --nodes=1                       # Single node
#SBATCH --ntasks=256                    # Total number of tasks
#SBATCH --cpus-per-task=1               # Each task uses 1 CPU
#SBATCH -C cpu                      # CPU type (use 'cpu' for regular CPUs)
#SBATCH -q debug                    # Queue (debug queue for short runs/testing)
#SBATCH -A desi                           # Account name to use on NERSC systems

# Load required modules
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Ensure logs directory exists
mkdir -p logs

# Parameters
BATCH_SIZE=64        # Number of healpix pixels per batch
NUM_BATCHES=256      # Total number of batches to process (update as needed)
OUTPUT_DIR="temp_batches"  # Output directory for temporary files
PYTHON_SCRIPT="preload_qsos.py"

# Calculate the batch index for each task
for TASK_ID in $(seq 0 $((SLURM_NTASKS-1))); do
    (
        LOG_FILE="logs/preload_${TASK_ID}.log"
        ERR_FILE="logs/preload_${TASK_ID}.err"
        
        srun -n 1 --exclusive python $PYTHON_SCRIPT $TASK_ID $BATCH_SIZE > $LOG_FILE 2> $ERR_FILE &
    )
done

# Wait for all background processes to finish
wait

echo "All batch processes completed."