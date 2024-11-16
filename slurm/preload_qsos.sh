#!/bin/bash
#SBATCH -N 1                          # Number of nodes
#SBATCH -C cpu                        # Use CPU nodes
#SBATCH -q debug                      # Use the debug queue
#SBATCH --job-name=preload_qsos       # Job name
#SBATCH --output=preload_qsos_%j.log  # Standard output log (%j is replaced by the job ID)
#SBATCH --error=preload_qsos_%j.err   # Standard error log (%j is replaced by the job ID)
#SBATCH --mail-user=mfho@umich.edu    # Email for notifications
#SBATCH --mail-type=ALL               # Notification options: ALL = begin, end, fail, etc.
#SBATCH -A desi                       # Account name
#SBATCH --time=00:30:00               # Time limit (30 minutes)
#SBATCH --ntasks=256                  # Total tasks (1 per worker)
#SBATCH --cpus-per-task=1             # 1 CPU per task (matches n_workers)

# Load the DESI environment
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Activate your Python environment if needed
# source activate my_python_env

# Path to your Python script
PYTHON_SCRIPT="preload_qsos.py"

# Parameters for the script
CATALOG_PATH="/global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits"
SPECTRA_DIR="/global/cfs/cdirs/desi/spectro/redux/"
OUTPUT_FILE="preloaded_qsos.h5"
SURVEY="main"
PROGRAM="dark"
RELEASE="kibo"

# Run the script
srun -n 256 python $PYTHON_SCRIPT # \
    # --catalog_path $CATALOG_PATH \
    # --spectra_dir $SPECTRA_DIR \
    # --output_file $OUTPUT_FILE \
    # --survey $SURVEY \
    # --program $PROGRAM \
    # --release $RELEASE \
    # --n_workers 256