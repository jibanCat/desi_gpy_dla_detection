#!/bin/bash
#SBATCH --job-name=preload_batches       # Job name
#SBATCH --output=logs/batch_%A_%a.out   # Standard output log
#SBATCH --error=logs/batch_%A_%a.err    # Standard error log
#SBATCH --time=05:00:00                 # Debug queue time limit
#SBATCH --nodes=1                       # Single node
#SBATCH --ntasks=64                    # Total number of tasks
#SBATCH --cpus-per-task=4               # Each task uses 2 CPU
#SBATCH -C cpu                          # CPU type
#SBATCH -q regular                        # Debug queue
#SBATCH -A desi                         # Account name

# Load required modules
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Ensure logs directory exists
mkdir -p logs

# Parameters
BATCH_SIZE=258        # Number of healpix pixels per batch
NUM_BATCHES=64      # Total number of batches to process
OUTPUT_DIR="temp_batches"  # Output directory for temporary files
PYTHON_SCRIPT="preload_qsos.py"

# Submit tasks
for TASK_ID in $(seq 0 $((NUM_BATCHES-1))); do
    LOG_FILE="logs/preload_${TASK_ID}.log"
    ERR_FILE="logs/preload_${TASK_ID}.err"

    echo "Submitting task $TASK_ID with batch size $BATCH_SIZE"
    srun -N 1 -n 1 -c 4 python $PYTHON_SCRIPT $TASK_ID $BATCH_SIZE > $LOG_FILE 2> $ERR_FILE &
done

# Wait for all tasks to complete
wait

echo "All batch processes completed."