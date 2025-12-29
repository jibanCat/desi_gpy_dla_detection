#!/bin/bash

#SBATCH -N 1                        # Number of nodes (1 node requested)
#SBATCH -C cpu                      # CPU type (use 'cpu' for regular CPUs)
#SBATCH -q regular                  # Queue (regular for longer runs)
#SBATCH --job-name=dla_detection    # Job name for identification in the queue
#SBATCH --output=gpdla_loa_%j.log  # Standard output log (%j is replaced by the job ID)
#SBATCH --error=error_loa_%j.log   # Standard error log (%j is replaced by the job ID)
#SBATCH --mail-user=mfho@umich.edu  # Your email for notifications
#SBATCH --mail-type=ALL             # Notification options (ALL = begin, end, fail, etc.)
#SBATCH -A desi                     # Account name to use on NERSC systems
#SBATCH --time=08:00:00             # Time limit for the job
#SBATCH --ntasks=32                 # 32 tasks total (each running one instance of the Python script)
#SBATCH --cpus-per-task=8           # Each task uses 8 CPUs

# Load the environment
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Set default values for variables if they are not provided
QSOCAT="${QSOCAT:-/global/cfs/cdirs/desi/users/martini/bal-catalogs/loa/QSO_cat_loa_main_dark_healpix_v2-altbal.fits}"
RELEASE="${RELEASE:-loa}"
PROGRAM="${PROGRAM:-dark}"
SURVEY="${SURVEY:-main}"
OUTDIR="${OUTDIR:-/pscratch/sd/j/jibancat/desi-loa-gpdla-20241211/}"
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
MAX_WORKERS="${MAX_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1250}"
LOADING_MIN_LAMBDA="${LOADING_MIN_LAMBDA:-910}"
LOADING_MAX_LAMBDA="${LOADING_MAX_LAMBDA:-1550}"
NORMALIZATION_MIN_LAMBDA="${NORMALIZATION_MIN_LAMBDA:-1425}"
NORMALIZATION_MAX_LAMBDA="${NORMALIZATION_MAX_LAMBDA:-1475}"
MIN_LAMBDA="${MIN_LAMBDA:-911.75}"
MAX_LAMBDA="${MAX_LAMBDA:-1216.75}"
DLAMBDA="${DLAMBDA:-0.25}"
K="${K:-20}"
MAX_Z_CUT="${MAX_Z_CUT:-3000.0}" # Maximum redshift cut for the DLA samples
MIN_Z_CUT="${MIN_Z_CUT:-3000.0}" # Minimum redshift cut for the DLA samples
MAX_NOISE_VARIANCE="${MAX_NOISE_VARIANCE:-9}"
# num_forest_lines
NUM_FOREST_LINES="${NUM_FOREST_LINES:-31}"
# num_lines
NUM_LINES="${NUM_LINES:-3}"

# num_dla_samples
NUM_DLA_SAMPLES="${NUM_DLA_SAMPLES:-100000}"
# num_subdla_samples
NUM_SUBDLA_SAMPLES="${NUM_SUBDLA_SAMPLES:-10000}"

# Filter low likelihood samples during model evidence computation
FILTER_LOW_LIKELIHOOD="${FILTER_LOW_LIKELIHOOD:-1}"

# Single absorber model flag
SINGLE_ABSORBER_MODEL="${SINGLE_ABSORBER_MODEL:-0}"

# Define start and end healpix index ranges for 8 tasks, with each task processing 40 healpix pixels
HPX_STEP=52
HPX_START_INDEX="${HPX_START_INDEX:-0}"
HPX_END_INDEX="${HPX_END_INDEX:-1612}"  # 52 healpix pixels * 32 tasks = 1664 healpix pixels

# Loop over each healpix range and start 8 concurrent jobs
for (( i = HPX_START_INDEX; i <= HPX_END_INDEX; i += HPX_STEP )); do
    HPX_START=$i
    HPX_END=$(( i + HPX_STEP ))

    echo "Running for healpix ${HPX_START} <= HPX < ${HPX_END}"

    srun -N 1 -n 1 -c 8 --output="loa_run_${HPX_START}-${HPX_END}_%j_%t.log" --error="error_loa_${HPX_START}-${HPX_END}_%j_%t.log" python desi-DLAGP.py \
        --qsocat "$QSOCAT" \
        --release "$RELEASE" \
        --program "$PROGRAM" \
        --survey "$SURVEY" \
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
        --filter_low_likelihood "$FILTER_LOW_LIKELIHOOD" \
        --single_absorber_model "$SINGLE_ABSORBER_MODEL" \
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
        --num_dla_samples "$NUM_DLA_SAMPLES" \
        --num_subdla_samples "$NUM_SUBDLA_SAMPLES" \
        --max_z_cut "$MAX_Z_CUT" \
        --min_z_cut "$MIN_Z_CUT" \
        --max_noise_variance "$MAX_NOISE_VARIANCE" \
        --num_forest_lines "$NUM_FOREST_LINES" \
        --num_lines "$NUM_LINES" \
        --figure_dir "$OUTDIR" \
        --hpx_start "$HPX_START" \
        --hpx_end "$HPX_END" &
done

# Wait for all background jobs to finish
wait