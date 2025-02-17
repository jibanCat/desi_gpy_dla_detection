#!/bin/bash

#SBATCH -N 1                        # Number of nodes (1 node requested)
#SBATCH -C gpu                      # GPU type
#SBATCH -q regular                  # Queue (regular for longer runs)
#SBATCH --job-name=train_gp         # Job name for identification in the queue
#SBATCH --output=train_gp_%j.log    # Standard output log (%j is replaced by the job ID)
#SBATCH --error=error_train_gp_%j.log   # Standard error log (%j is replaced by the job ID)
#SBATCH --mail-user=mfho@umich.edu  # Your email for notifications
#SBATCH --mail-type=ALL             # Notification options (ALL = begin, end, fail, etc.)
#SBATCH -A desi                     # Account name to use on NERSC systems
#SBATCH --time=48:00:00              # Time limit for the job

# Debugging flags
export CUDA_LAUNCH_BLOCKING=1  # Helps debug CUDA issues
export PYTHONUNBUFFERED=1      # Forces immediate output

# Load the environment
source /global/cfs/cdirs/desi/software/desi_environment.sh main

python -u desi_learn_qsos_model.py \
    --catalog_file "/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/gp_trainset_loa.fits" \
    --preloaded_file "/pscratch/sd/j/jibancat/preload-loa-gpdla-20250202/gp_interp_trainset.h5" \
    --z_min 2.5 \
    --z_max 4.25 \
    --num_pca_components 20 \
    --max_spectra 300000 \
    --num_pixels 3798 \
    --min_num_pixels 400 \
    --min_snr 0.0 \
    --min_lambda 850.90 \
    --max_lambda 1420.60 \
    --norm_min_lambda 900 \
    --norm_max_lambda 1200 \
    --max_noise_variance 9.0 \
    --output_dir "learnlogs/20250216/" \
    --num_epochs 100 \
    --learning_rate 0.1 \
    --batch_size 16384  # Updated batch size

