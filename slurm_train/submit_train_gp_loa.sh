#!/bin/bash

#SBATCH -N 1                        # Number of nodes (1 node requested)
#SBATCH --gres=gpu:4                 # Explicitly request 4 GPUs
#SBATCH -q debug                     # Queue (use 'regular' for longer runs)
#SBATCH --job-name=train_gp          # Job name for identification in the queue
#SBATCH --output=train_gp_%j.log     # Standard output log (%j is replaced by the job ID)
#SBATCH --error=error_train_gp_%j.log  # Standard error log (%j is replaced by the job ID)
#SBATCH --mail-user=mfho@umich.edu   # Your email for notifications
#SBATCH --mail-type=ALL              # Notification options (ALL = begin, end, fail, etc.)
#SBATCH -A desi                      # Account name to use on NERSC systems
#SBATCH --time=0:30:00               # Time limit for the job
#SBATCH --ntasks=1                   # One task to avoid multiple Python instances
#SBATCH --cpus-per-task=8            # Use 8 CPU threads for data loading

# ============================
# ⚡ Performance Optimizations
# ============================
export OMP_NUM_THREADS=8             # Optimize CPU performance
export CUDA_VISIBLE_DEVICES=0,1,2,3   # Ensure all 4 GPUs are visible
export NCCL_DEBUG=INFO                # Debug NCCL communication issues
export NCCL_P2P_DISABLE=1             # Disable P2P if multi-GPU issues occur
export PYTHONUNBUFFERED=1             # Print logs immediately
export CUDA_LAUNCH_BLOCKING=0         # Set to 1 if debugging CUDA errors

# Load the environment
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Run Python training script
srun --gpu-bind=none python -u desi_learn_qsos_model.py \
    --catalog_file "/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/gp_trainset_loa.fits" \
    --preloaded_file "/pscratch/sd/j/jibancat/preload-loa-gpdla-20250202/gp_interp_trainset.h5" \
    --z_min 2.5 \
    --z_max 4.25 \
    --num_pca_components 30 \
    --max_spectra 300000 \
    --num_pixels 3798 \
    --min_num_pixels 400 \
    --min_snr 0.0 \
    --min_lambda 850.90 \
    --max_lambda 1420.60 \
    --norm_min_lambda 900 \
    --norm_max_lambda 1200 \
    --max_noise_variance 9.0 \
    --output_dir "learnlogs/" \
    --num_epochs 5000 \
    --learning_rate 0.1 \
    --batch_size 8972  # Updated batch size