#!/bin/bash

#SBATCH -N 1                        # Number of nodes (1 node requested)
#SBATCH -C cpu                      # CPU type (use 'cpu' for regular CPUs)
#SBATCH -q regular                  # Queue (regular for longer runs)
#SBATCH --job-name=dla_detection    # Job name for identification in the queue
#SBATCH --output=gpdla_%j.log       # Standard output log (%j is replaced by the job ID)
#SBATCH --error=error_%j.log        # Standard error log (%j is replaced by the job ID)
#SBATCH --mail-user=mfho@umich.edu  # Your email for notifications
#SBATCH --mail-type=ALL             # Notification options (ALL = begin, end, fail, etc.)
#SBATCH -A desi                     # Account name to use on NERSC systems
#SBATCH --time=04:00:00             # Time limit for the job
#SBATCH --ntasks=32                  # 8 tasks total (each running one instance of the Python script)
#SBATCH --cpus-per-task=8          # Each task uses 8 CPUs

# Ensure the environment is loaded
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Set default values for variables if they are not provided
QSOCAT="${QSOCAT:-/global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi_y3/v5.9.5/mock-0/jura-124/zcat.fits}"
RELEASE="${RELEASE:-v5.9.5}"
PROGRAM="${PROGRAM:-dark}"
SURVEY="${SURVEY:-main}"
MOCKDIR="${MOCKDIR:-/global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi_y3/v5.9.5/mock-0/jura-124/}"
OUTDIR="${OUTDIR:-/pscratch/sd/j/jibancat/desi-mock-gpdla-20241211/}"
BALMASK="${BALMASK:-false}"

LEARNED_FILE="${LEARNED_FILE:-data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat}"
CATALOG_NAME="${CATALOG_NAME:-data/dr12q/processed/catalog.mat}"
LOS_CATALOG="${LOS_CATALOG:-data/dla_catalogs/dr9q_concordance/processed/los_catalog}"
DLA_CATALOG="${DLA_CATALOG:-data/dla_catalogs/dr9q_concordance/processed/dla_catalog}"
DLA_SAMPLES_FILE="${DLA_SAMPLES_FILE:-data/dr12q/processed/dla_samples_a03.mat}"
SUB_DLA_SAMPLES_FILE="${SUB_DLA_SAMPLES_FILE:-data/dr12q/processed/subdla_samples.mat}"
MIN_Z_SEPARATION="${MIN_Z_SEPARATION:-3000.0}"
PREV_TAU_0="${PREV_TAU_0:-0.00554}"
PREV_BETA="${PREV_BETA:-3.182}"
MAX_DLAS="${MAX_DLAS:-3}"
PLOT_FIGURES="${PLOT_FIGURES:-0}"
MAX_WORKERS="${MAX_WORKERS:-8}"               # Reduced for debug
BATCH_SIZE="${BATCH_SIZE:-1250}"               # Smaller batch size for debug
LOADING_MIN_LAMBDA="${LOADING_MIN_LAMBDA:-910}"
LOADING_MAX_LAMBDA="${LOADING_MAX_LAMBDA:-1550}"
NORMALIZATION_MIN_LAMBDA="${NORMALIZATION_MIN_LAMBDA:-1425}"
NORMALIZATION_MAX_LAMBDA="${NORMALIZATION_MAX_LAMBDA:-1475}"
MIN_LAMBDA="${MIN_LAMBDA:-911.75}"
MAX_LAMBDA="${MAX_LAMBDA:-1216.75}"
DLAMBDA="${DLAMBDA:-0.25}"
K="${K:-20}"
MAX_NOISE_VARIANCE="${MAX_NOISE_VARIANCE:-9}"
LEVEL2_START="${LEVEL2_START:-0}"
LEVEL2_END="${LEVEL2_END:-1}"                 # Reduced range for quick debug

FIGURE_DIR="${FIGURE_DIR:-figures/}"

# Start and end range variables for controlling the loop
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-62}"
STEP="${STEP:-2}"

# Loop over the specified range, incrementing by the specified step
for (( i = START_INDEX; i <= END_INDEX; i += STEP )); do
    LEVEL2_START=$((i))
    LEVEL2_END=$((i + 2))
    echo "Running for ${LEVEL2_START} <= LEVEL2 < ${LEVEL2_END}"

    srun -N 1 -n 1 -c 8 --output="mock_run_${LEVEL2_START}-${LEVEL2_END}_%j_%t.log" --error="error_mock_${LEVEL2_START}-${LEVEL2_END}_%j_%t.log" python desi-DLAGP.py \
        --qsocat "$QSOCAT" \
        --release "$RELEASE" \
        --program "$PROGRAM" \
        --survey "$SURVEY" \
        --mocks \
        --mockdir "$MOCKDIR" \
        $(if [ "$BALMASK" == "true" ]; then echo "--balmask"; fi) \
        --outdir "$OUTDIR" \
        --learned_file "$LEARNED_FILE" \
        --catalog_name "$CATALOG_NAME" \
        --los_catalog "$LOS_CATALOG" \
        --dla_catalog "$DLA_CATALOG" \
        --dla_samples_file "$DLA_SAMPLES_FILE" \
        --sub_dla_samples_file "$SUB_DLA_SAMPLES_FILE" \
        --min_z_separation "$MIN_Z_SEPARATION" \
        --prev_tau_0 "$PREV_TAU_0" \
        --prev_beta "$PREV_BETA" \
        --max_dlas "$MAX_DLAS" \
        --plot_figures "$PLOT_FIGURES" \
        --max_workers "$MAX_WORKERS" \
        --batch_size "$BATCH_SIZE" \
        --loading_min_lambda "$LOADING_MIN_LAMBDA" \
        --loading_max_lambda "$LOADING_MAX_LAMBDA" \
        --normalization_min_lambda "$NORMALIZATION_MIN_LAMBDA" \
        --normalization_max_lambda "$NORMALIZATION_MAX_LAMBDA" \
        --min_lambda "$MIN_LAMBDA" \
        --max_lambda "$MAX_LAMBDA" \
        --dlambda "$DLAMBDA" \
        --k "$K" \
        --max_noise_variance "$MAX_NOISE_VARIANCE" \
        --figure_dir "$OUTDIR" \
        --level2_start "$LEVEL2_START" \
        --level2_end "$LEVEL2_END" &
done

# Wait for all background jobs to finish
wait

# Example usage:
# - Run the script with default values:
# sbatch --export=ALL,QSOCAT="/global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi_y3/v5.9.5/mock-0/jura-124/zcat.fits",\
# MOCKDIR="/global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi_y3/v5.9.5/mock-0/jura-124/",\
# OUTDIR="/pscratch/sd/j/jibancat/desi-mock-gpdla/",\
# MAX_DLAS=3,\
# PLOT_FIGURES=0,\
# BATCH_SIZE=313,\
# MAX_WORKERS=32,\
# START_INDEX=0,\
# END_INDEX=14,\
# STEP=2 slurm/submit_desi_mock.sh