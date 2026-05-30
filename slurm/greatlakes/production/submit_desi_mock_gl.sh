#!/bin/bash
# slurm/greatlakes/production/submit_desi_mock_gl.sh
#
# GreatLakes analog of slurm/submit_desi_mock.sh. Run via launch_gl.sh
# (or sbatch directly with the same --export list). All scientific args
# are identical to the NERSC inner script; only the SBATCH directives +
# env setup differ.
#
# Differences vs slurm/submit_desi_mock.sh:
#   - SBATCH: cavestru0 account, standard partition, no -C cpu / -q regular.
#   - Env: conda activate gpdla + LD_LIBRARY_PATH for libcerf.
#     No NERSC `desi_environment.sh main` (which doesn't exist on GL).
#   - --output / --error paths live under OUTDIR/logs/ (caller mkdir'd).
#   - Parallelism: GL standard nodes are 36 cores (≥28). We request 2 srun
#     tasks × 16 CPUs = 32 cores per sbatch (fits any ≥32-core node, leaves
#     4 idle for OS overhead). Each srun runs ONE python on ONE level2-slice
#     with MAX_WORKERS=16 inner worker processes, BLAS pinned to 1 thread.
#     This N=2 × W=16 packing was the throughput OPTIMUM in a pinned
#     concurrency sweep (jobs 50635651/50638321): 32.6 spec/min — 3× the old
#     N=8 × W=4 packing and ~100× the single-spectrum-latency pick (N=32 ×
#     W=1 = 0.3 spec/min). Why N=2 wins: two spectra in flight overlap each
#     other's serial (τ-EB / load / resample) and parallel phases, while only
#     2 distinct GP working sets keep memory-bandwidth/cache contention low;
#     N≥4 thrashes the memory subsystem. The level2 loop emits ~31 background
#     srun's per window; 2 at a time execute concurrently, so size
#     OUTER_WINDOW (or -t) so the window fits the wall limit (~67 min per
#     2-file slice at ~3.7 s/spec). Production NERSC pattern is 32 × 8 = 256
#     cores per sbatch; GL recovers via parallelism across multiple sbatch's.

#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J dla_inference_gl
#SBATCH -o slurm/greatlakes/production/logs/gpdla_gl_%j.log
#SBATCH -e slurm/greatlakes/production/logs/error_gl_%j.log

set -eo pipefail
export PYTHONUNBUFFERED=1

# --- Pin BLAS threads to 1 (config-only; dla_gp.py is NOT touched) -----------
# numpy here is OpenBLAS. Without this, each MAX_WORKERS process spawns its own
# BLAS thread pool (~2 threads), so W workers => ~2*W threads thrashing the
# allocated cores. Measured A/B (job 50635346, gl3010, one spectrum, identical
# p=0.9999): W=4 14.6s->4.0s (3.7x), W=8 25.0s->2.8s (8.9x) purely by pinning.
# One BLAS thread per worker => the ProcessPoolExecutor over QMC samples is the
# only parallelism and it scales cleanly. (See memory: gl-blas-oversubscription.)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- conda + libcerf runtime --------------------------------------------------
GL_CONDA_SETUP="${GL_CONDA_SETUP:-/sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh}"
GL_CONDA_ENV="${GL_CONDA_ENV:-gpdla}"
GL_LIBCERF_PATH="${GL_LIBCERF_PATH:-$HOME/.local/usr/local/lib64}"

# shellcheck disable=SC1090
source "$GL_CONDA_SETUP"
conda activate "$GL_CONDA_ENV"
export LD_LIBRARY_PATH="${GL_LIBCERF_PATH}:${LD_LIBRARY_PATH:-}"

# --- defaults (mirror submit_desi_mock.sh; flavour configs override) ---------
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

# τ-EB (production "best baseline" = ON with the null objective; runbook
# §10.1 flags that the NERSC submit scripts silently DROP these, so we
# forward them explicitly here). EARLY_STOP_MODE=baseline is the prod choice.
ENABLE_TAU_EB="${ENABLE_TAU_EB:-1}"
TAU_EB_OBJECTIVE="${TAU_EB_OBJECTIVE:-null}"
EARLY_STOP_MODE="${EARLY_STOP_MODE:-baseline}"

# DLA clustering prior (default off => byte-identical to production).
PAIR_PRIOR_MODE="${PAIR_PRIOR_MODE:-off}"
DLA_BIAS="${DLA_BIAS:-2.0}"

# FILTER=1 truncated-sampler tuning knobs (mirrors run_local.sh:208-215).
# Defaults left empty → desi-DLAGP.py uses its built-in CLI defaults; set in
# the flavour .env to override. (see docs/notes/2026-05-13_filter1_knob_tuning.md)
FILTER_N_INITIAL_FLOOR="${FILTER_N_INITIAL_FLOOR:-}"
FILTER_EMPTY_MASK_FALLTHROUGH="${FILTER_EMPTY_MASK_FALLTHROUGH:-0}"

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-62}"
STEP="${STEP:-2}"

mkdir -p "$OUTDIR" "${OUTDIR}/logs"

# Banner — shows up in the SLURM log header
echo "[gl] $(date) job=$SLURM_JOB_ID  range=${START_INDEX}..${END_INDEX} step=${STEP}"
echo "[gl] LEARNED_FILE=$LEARNED_FILE"
echo "[gl] OUTDIR=$OUTDIR"
echo "[gl] MAX_LAMBDA=$MAX_LAMBDA MAX_DLAS=$MAX_DLAS SINGLE_ABSORBER_MODEL=$SINGLE_ABSORBER_MODEL"
echo "[gl] NUM_DLA_SAMPLES=$NUM_DLA_SAMPLES FILTER=$FILTER_LOW_LIKELIHOOD"

# --- level2 background loop (mirrors NERSC inner script) ---------------------
for (( i = START_INDEX; i <= END_INDEX; i += STEP )); do
    LEVEL2_START=$((i))
    LEVEL2_END=$((i + STEP))
    echo "[gl] level2 ${LEVEL2_START}..${LEVEL2_END}"

    # --exact + --overlap so the backgrounded srun steps PACK onto the
    # allocation (each takes exactly 1 task / SRUN_CPUS cpus) and run
    # concurrently. Without --exact the first srun grabs the whole
    # allocation and the rest serialize (observed in job 50565955:
    # 1/8 steps active, 28/32 cores idle but billed).
    srun --exact --overlap -N 1 -n 1 -c "${SRUN_CPUS:-${MAX_WORKERS}}" \
        --output="${OUTDIR}/logs/mock_run_${LEVEL2_START}-${LEVEL2_END}_%j_%t.log" \
        --error="${OUTDIR}/logs/error_mock_${LEVEL2_START}-${LEVEL2_END}_%j_%t.log" \
        python desi-DLAGP.py \
            --qsocat "$QSOCAT" \
            --release "$RELEASE" \
            --program "$PROGRAM" \
            --survey "$SURVEY" \
            --mocks \
            --mockdir "$MOCKDIR" \
            $(if [ "$BALMASK" = "true" ]; then echo "--balmask"; fi) \
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
            --max_noise_variance "$MAX_NOISE_VARIANCE" \
            --num_forest_lines "$NUM_FOREST_LINES" \
            --num_lines "$NUM_LINES" \
            --enable_tau_eb "$ENABLE_TAU_EB" \
            --tau_eb_objective "$TAU_EB_OBJECTIVE" \
            --early_stop_mode "$EARLY_STOP_MODE" \
            --pair_prior_mode "$PAIR_PRIOR_MODE" \
            --dla_bias "$DLA_BIAS" \
            $([ -n "$FILTER_N_INITIAL_FLOOR" ] && echo "--filter_n_initial_floor $FILTER_N_INITIAL_FLOOR") \
            $([ "$FILTER_EMPTY_MASK_FALLTHROUGH" = "1" ] && echo "--filter_empty_mask_fallthrough 1") \
            --figure_dir "$FIGURE_DIR" \
            --level2_start "$LEVEL2_START" \
            --level2_end "$LEVEL2_END" &
done

wait
echo "[gl] $(date) all level2 slices done"
