#!/bin/bash

#SBATCH -N 1                        # Number of nodes (1 node requested)
#SBATCH -C cpu                      # CPU type (use 'cpu' for regular CPUs)
#SBATCH -q regular                  # Queue (regular for longer runs)
#SBATCH --job-name=dla_detection    # Job name for identification in the queue
#SBATCH --output=gpdla_kibo_%j.log  # Standard output log (%j is replaced by the job ID)
#SBATCH --error=error_kibo_%j.log   # Standard error log (%j is replaced by the job ID)
#SBATCH --mail-user=mfho@umich.edu  # Your email for notifications
#SBATCH --mail-type=ALL             # Notification options (ALL = begin, end, fail, etc.)
#SBATCH -A desi                     # Account name to use on NERSC systems
#SBATCH --time=03:00:00             # Time limit for the job
#SBATCH --ntasks=8                  # 8 tasks total (each running one instance of the Python script)
#SBATCH --cpus-per-task=32          # Each task uses 32 CPUs

# Load the environment
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Set default values for variables if they are not provided
QSOCAT="${QSOCAT:-/global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits}"
RELEASE="${RELEASE:-kibo}"
PROGRAM="${PROGRAM:-dark}"
SURVEY="${SURVEY:-main}"
OUTDIR="${OUTDIR:-/pscratch/sd/j/jibancat/desi-kibo-gpdla/}"
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
MAX_WORKERS="${MAX_WORKERS:-32}"
BATCH_SIZE="${BATCH_SIZE:-313}"
LOADING_MIN_LAMBDA="${LOADING_MIN_LAMBDA:-800}"
LOADING_MAX_LAMBDA="${LOADING_MAX_LAMBDA:-1550}"
NORMALIZATION_MIN_LAMBDA="${NORMALIZATION_MIN_LAMBDA:-1425}"
NORMALIZATION_MAX_LAMBDA="${NORMALIZATION_MAX_LAMBDA:-1475}"
MIN_LAMBDA="${MIN_LAMBDA:-850.75}"
MAX_LAMBDA="${MAX_LAMBDA:-1420.75}"
DLAMBDA="${DLAMBDA:-0.25}"
K="${K:-20}"
MAX_NOISE_VARIANCE="${MAX_NOISE_VARIANCE:-9}"

# Path to the error list file
ERROR_LIST_FILE="${ERROR_LIST_FILE:-error_file_list.txt}"

# Check if the error list file exists
if [[ ! -f "$ERROR_LIST_FILE" ]]; then
    echo "Error: Error list file $ERROR_LIST_FILE not found!"
    exit 1
fi

# Read error list into an array
error_list=()
while read -r hpx_start hpx_end; do
    error_list+=("$hpx_start $hpx_end")
done < "$ERROR_LIST_FILE"

# Set starting index for this job (useful for splitting into multiple jobs if needed)
START_INDEX=${START_INDEX:-0}

# Process only a batch of 8 pairs for this SLURM job
for (( j = 0; j < 8 && (START_INDEX + j) < ${#error_list[@]}; j++ )); do
    # Get the hpx_start and hpx_end for each task in this batch
    hpx_pair=(${error_list[START_INDEX + j]})
    HPX_START="${hpx_pair[0]}"
    HPX_END="${hpx_pair[1]}"

    echo "---"
    echo "Current index: $((START_INDEX + j))"
    echo "Running for healpix ${HPX_START} <= HPX < ${HPX_END}"

    echo "srun -N 1 -n 1 -c 32 --output="kibo_run_${HPX_START}-${HPX_END}_%j_%t.log" --error="error_kibo_${HPX_START}-${HPX_END}_%j_%t.log" python desi-DLAGP.py \
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
--figure_dir "figures/healpix_${HPX_START}_${HPX_END}" \
--hpx_start "$HPX_START" \
--hpx_end "$HPX_END" &"

    Run each job as an individual srun command with 32 CPUs
    srun -N 1 -n 1 -c 32 --output="kibo_run_${HPX_START}-${HPX_END}_%j_%t.log" --error="error_kibo_${HPX_START}-${HPX_END}_%j_%t.log" python desi-DLAGP.py \
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
        --figure_dir "figures/healpix_${HPX_START}_${HPX_END}" \
        --hpx_start "$HPX_START" \
        --hpx_end "$HPX_END" &
done

# Wait for all background jobs in this batch to finish
wait