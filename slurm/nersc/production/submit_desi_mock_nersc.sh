#!/bin/bash
# slurm/nersc/production/submit_desi_mock_nersc.sh
#
# NERSC (Perlmutter) mock inference inner script — mirror of
# submit_desi_mock_gl.sh, but using the NERSC-validated parallelism pattern:
# ONE `srun -n NTASKS -c W` multi-task launch (PROCID-dispatched to contiguous
# file chunks), NOT shell-backgrounded `srun &` (which NERSC rejects with
# "step creation disabled, retrying (nodes busy)"). All SCIENCE args are
# byte-identical to the GL inner script; only SBATCH header, env activation,
# and the task-decomposition differ.
#
# Run via launch_nersc.sh (which sets the --export list + --ntasks/--cpus-per-task).
# Re-entrant: the driver branch launches `srun ... "$SELF"` with MOCK_TASK=1,
# re-entering the task branch. Task k processes file indices
# [START_INDEX + k*per, +per) where per = ceil((END_INDEX-START_INDEX)/NTASKS).

#SBATCH -A desi
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J dla_nersc
#SBATCH -o slurm/nersc/production/logs/gpdla_nersc_%j.log
#SBATCH -e slurm/nersc/production/logs/error_nersc_%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1

# BLAS pinned to 1 thread per worker (config-only; inference code untouched).
# Measured on GL: pinning gave W=8 a clean 8.9x speedup vs oversubscription.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- required + default knobs (mirror submit_desi_mock_gl.sh) ----------------
QSOCAT="${QSOCAT:?must be set via --export}"
RELEASE="${RELEASE:-v5.9.5}"
PROGRAM="${PROGRAM:-dark}"
SURVEY="${SURVEY:-main}"
MOCKDIR="${MOCKDIR:?must be set via --export}"
OUTDIR="${OUTDIR:?must be set via --export}"
BALMASK="${BALMASK:-false}"
LEARNED_FILE="${LEARNED_FILE:?must be set via --export}"
CATALOG_NAME="${CATALOG_NAME:?must be set via --export}"
LOS_CATALOG="${LOS_CATALOG:?must be set via --export}"
DLA_CATALOG="${DLA_CATALOG:?must be set via --export}"
DLA_SAMPLES_FILE="${DLA_SAMPLES_FILE:?must be set via --export}"
SUB_DLA_SAMPLES_FILE="${SUB_DLA_SAMPLES_FILE:?must be set via --export}"
MIN_Z_SEPARATION="${MIN_Z_SEPARATION:-3000.0}"
PREV_TAU_0="${PREV_TAU_0:-0.00246}"
PREV_BETA="${PREV_BETA:-3.62}"
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
DLAMBDA="${DLAMBDA:-0.15}"
K="${K:-30}"
MAX_NOISE_VARIANCE="${MAX_NOISE_VARIANCE:-9}"
NUM_FOREST_LINES="${NUM_FOREST_LINES:-3}"
NUM_LINES="${NUM_LINES:-3}"
NUM_DLA_SAMPLES="${NUM_DLA_SAMPLES:-10000}"
NUM_SUBDLA_SAMPLES="${NUM_SUBDLA_SAMPLES:-10000}"
FIGURE_DIR="${FIGURE_DIR:-${OUTDIR}/figures}"
FILTER_LOW_LIKELIHOOD="${FILTER_LOW_LIKELIHOOD:-1}"
SINGLE_ABSORBER_MODEL="${SINGLE_ABSORBER_MODEL:-0}"
ENABLE_TAU_EB="${ENABLE_TAU_EB:-1}"
TAU_EB_OBJECTIVE="${TAU_EB_OBJECTIVE:-null}"
EARLY_STOP_MODE="${EARLY_STOP_MODE:-baseline}"
PAIR_PRIOR_MODE="${PAIR_PRIOR_MODE:-off}"
DLA_BIAS="${DLA_BIAS:-2.0}"
FILTER_N_INITIAL_FLOOR="${FILTER_N_INITIAL_FLOOR:-}"
FILTER_EMPTY_MASK_FALLTHROUGH="${FILTER_EMPTY_MASK_FALLTHROUGH:-0}"

# Window of healpix-file indices for THIS sbatch, split across NTASKS tasks.
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-32}"
NTASKS="${NTASKS:-32}"
SELF="${SELF:?must be set via --export (absolute path to this script)}"

build_and_run () {
    local l2s="$1" l2e="$2"
    python desi-DLAGP.py \
        --qsocat "$QSOCAT" --release "$RELEASE" --program "$PROGRAM" --survey "$SURVEY" \
        --mocks --mockdir "$MOCKDIR" \
        $(if [ "$BALMASK" = "true" ]; then echo "--balmask"; fi) \
        --outdir "$OUTDIR" \
        --learned_file "$LEARNED_FILE" \
        --catalog_name "$CATALOG_NAME" --los_catalog "$LOS_CATALOG" --dla_catalog "$DLA_CATALOG" \
        --dla_samples_file "$DLA_SAMPLES_FILE" --sub_dla_samples_file "$SUB_DLA_SAMPLES_FILE" \
        --min_z_separation "$MIN_Z_SEPARATION" --prev_tau_0 "$PREV_TAU_0" --prev_beta "$PREV_BETA" \
        --max_dlas "$MAX_DLAS" --plot_figures "$PLOT_FIGURES" \
        --filter_low_likelihood "$FILTER_LOW_LIKELIHOOD" --single_absorber_model "$SINGLE_ABSORBER_MODEL" \
        --max_workers "$MAX_WORKERS" --batch_size "$BATCH_SIZE" \
        --loading_min_lambda "$LOADING_MIN_LAMBDA" --loading_max_lambda "$LOADING_MAX_LAMBDA" \
        --normalization_min_lambda "$NORMALIZATION_MIN_LAMBDA" --normalization_max_lambda "$NORMALIZATION_MAX_LAMBDA" \
        --min_lambda "$MIN_LAMBDA" --max_lambda "$MAX_LAMBDA" --dlambda "$DLAMBDA" --k "$K" \
        --num_dla_samples "$NUM_DLA_SAMPLES" --num_subdla_samples "$NUM_SUBDLA_SAMPLES" \
        --max_noise_variance "$MAX_NOISE_VARIANCE" --num_forest_lines "$NUM_FOREST_LINES" --num_lines "$NUM_LINES" \
        --enable_tau_eb "$ENABLE_TAU_EB" --tau_eb_objective "$TAU_EB_OBJECTIVE" --early_stop_mode "$EARLY_STOP_MODE" \
        --pair_prior_mode "$PAIR_PRIOR_MODE" --dla_bias "$DLA_BIAS" \
        $([ -n "$FILTER_N_INITIAL_FLOOR" ] && echo "--filter_n_initial_floor $FILTER_N_INITIAL_FLOOR") \
        $([ "$FILTER_EMPTY_MASK_FALLTHROUGH" = "1" ] && echo "--filter_empty_mask_fallthrough 1") \
        --figure_dir "$FIGURE_DIR" \
        --level2_start "$l2s" --level2_end "$l2e"
}

# ---------------------------------------------------------------------------
# TASK branch: process this task's contiguous file chunk of [START,END).
# ---------------------------------------------------------------------------
if [ "${MOCK_TASK:-0}" = "1" ]; then
    k="${SLURM_PROCID:-0}"
    span=$(( END_INDEX - START_INDEX ))
    per=$(( (span + NTASKS - 1) / NTASKS ))      # ceil
    [ "$per" -lt 1 ] && per=1
    l2s=$(( START_INDEX + k * per ))
    l2e=$(( l2s + per ))
    [ "$l2e" -gt "$END_INDEX" ] && l2e="$END_INDEX"
    if [ "$l2s" -ge "$END_INDEX" ]; then
        echo "[task $k] no work (l2s=$l2s >= END=$END_INDEX)"; exit 0
    fi
    echo "[task $k] level2 ${l2s}..${l2e}"
    build_and_run "$l2s" "$l2e"
    exit 0
fi

# ---------------------------------------------------------------------------
# DRIVER branch: env + ONE srun -n NTASKS (no backgrounding).
# ---------------------------------------------------------------------------
NERSC_ENV_SETUP="${NERSC_ENV_SETUP:-source /global/cfs/cdirs/desi/software/desi_environment.sh main}"
set +u; eval "$NERSC_ENV_SETUP"; set -u

mkdir -p "$OUTDIR" "${OUTDIR}/logs"
echo "[nersc] $(date) job=${SLURM_JOB_ID:-NA} window=${START_INDEX}..${END_INDEX} ntasks=${NTASKS} W=${MAX_WORKERS}"
echo "[nersc] LEARNED_FILE=$LEARNED_FILE"
echo "[nersc] OUTDIR=$OUTDIR"
echo "[nersc] MAX_LAMBDA=$MAX_LAMBDA MAX_DLAS=$MAX_DLAS SINGLE_ABSORBER_MODEL=$SINGLE_ABSORBER_MODEL NUM_DLA_SAMPLES=$NUM_DLA_SAMPLES FILTER=$FILTER_LOW_LIKELIHOOD"

srun -N 1 -n "${NTASKS}" -c "${MAX_WORKERS}" --cpu-bind=cores \
     --output="${OUTDIR}/logs/mock_run_${START_INDEX}-${END_INDEX}_%j_%t.log" \
     --error="${OUTDIR}/logs/error_mock_${START_INDEX}-${END_INDEX}_%j_%t.log" \
     --export=ALL,MOCK_TASK=1 "${SELF}"

echo "[nersc] $(date) window ${START_INDEX}..${END_INDEX} done"
