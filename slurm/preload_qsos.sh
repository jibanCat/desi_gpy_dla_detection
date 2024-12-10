#!/bin/bash
#SBATCH --job-name=preload_batches       # Job name
#SBATCH --output=logs/batch_%A_%a.out   # Standard output log
#SBATCH --error=logs/batch_%A_%a.err    # Standard error log
#SBATCH --time=05:00:00                 # Time limit
#SBATCH --nodes=1                       # Single node
#SBATCH --ntasks=16                     # Total number of tasks
#SBATCH --cpus-per-task=16               # Each task uses 4 CPUs
#SBATCH -C cpu                          # CPU type
#SBATCH -q regular                      # Queue type
#SBATCH -A desi                         # Account name

# Load required modules
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Ensure logs directory exists
mkdir -p logs

# Parameters
BATCH_SIZE=258                           # Number of healpix pixels per batch
NUM_BATCHES=64                           # Total number of batches to process
BATCHES_PER_JOB=16                       # Number of batches per job
START_BATCH=$((SLURM_ARRAY_TASK_ID * BATCHES_PER_JOB)) # Starting batch for this submission
END_BATCH=$((START_BATCH + BATCHES_PER_JOB - 1))       # Ending batch for this submission
OUTPUT_DIR="temp_batches"                # Output directory for temporary files
PYTHON_SCRIPT="preload_qsos.py"

# Double checking the variables by printing them
echo "BATCH_SIZE: $BATCH_SIZE"
echo "NUM_BATCHES: $NUM_BATCHES"
echo "BATCHES_PER_JOB: $BATCHES_PER_JOB"
echo "START_BATCH: $START_BATCH"
echo "END_BATCH: $END_BATCH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PYTHON_SCRIPT: $PYTHON_SCRIPT"


# Submit tasks for this job's range of batches
for TASK_ID in $(seq $START_BATCH $END_BATCH); do
    if [ $TASK_ID -ge $NUM_BATCHES ]; then
        break
    fi

    LOG_FILE="logs/preload_${TASK_ID}.log"
    ERR_FILE="logs/preload_${TASK_ID}.err"

    echo "Submitting task $TASK_ID with batch size $BATCH_SIZE"
    srun -N 1 -n 1 -c 16 python $PYTHON_SCRIPT $TASK_ID $BATCH_SIZE > $LOG_FILE 2> $ERR_FILE &
done

# Wait for all tasks to complete
wait

echo "All batch processes completed."