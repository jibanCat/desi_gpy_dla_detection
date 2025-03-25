#!/bin/bash

#SBATCH -N 1                              # Number of nodes (1 node requested)
#SBATCH -C gpu                            # GPU type
#SBATCH -q regular                        # Queue (regular for longer runs)
#SBATCH --job-name=train_gp               # Job name for identification in the queue
#SBATCH --output=train_gp_%j.log          # Standard output log (%j is replaced by the job ID)
#SBATCH --error=error_train_gp_%j.log     # Standard error log (%j is replaced by the job ID)
#SBATCH --mail-user=mfho@umich.edu        # Your email for notifications
#SBATCH --mail-type=ALL                   # Notification options (ALL = begin, end, fail, etc.)
#SBATCH -A desi                           # Account name to use on NERSC systems
#SBATCH --time=12:00:00                   # Time limit for the job

# ========= Set the TARGETID =========
# You can override this from the command line with:
# sbatch --export=tid=12345678901234567 this_script.sh
tid=${tid:-39627666508219798}  # Default value if not set

# ========= Debugging flags =========
export CUDA_LAUNCH_BLOCKING=1  # Helps debug CUDA issues
export PYTHONUNBUFFERED=1      # Forces immediate output

# ========= Load the environment =========
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# ========= Run Python Script =========
python -u examples/plot_visual_inspect.py \
  --catalog /global/cfs/cdirs/desi/users/martini/bal-catalogs/loa/QSO_cat_loa_main_dark_healpix_v2-altbal.fits \
  --output_dir vi_output \
  --release loa \
  --survey main \
  --program dark \
  --spectra_filename /path/to/spectra-16-724.fits \
  --zbest_filename /path/to/zbest-16-724.fits \
  --learned_file_eBOSS ../data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat \
  --learned_file_desi ../learnlogs/model_epoch_682.h5 \
  --catalog_name ../data/dr12q/processed/catalog.mat \
  --los_catalog ../data/dla_catalogs/dr9q_concordance/processed/los_catalog \
  --dla_catalog ../data/dla_catalogs/dr9q_concordance/processed/dla_catalog \
  --dla_samples_file ../data/dr12q/processed/dla_samples_a03.mat \
  --sub_dla_samples_file ../data/dr12q/processed/subdla_samples.mat \
  --tid_list ${tid}
