#!/bin/bash
# slurm/greatlakes/production/submit_desi_mock_gl_resume.sh
#
# RESUME sibling of submit_desi_mock_gl.sh. Identical SBATCH directives, env
# setup, BLAS pinning, and (critically) the SAME desi-DLAGP.py science args —
# but instead of a contiguous START_INDEX..END_INDEX range it processes an
# EXPLICIT, comma-separated list of speclist POSITIONS (LEVEL2_LIST), each as a
# single-file slice (--level2_start P --level2_end P+1).
#
# WHY a separate script: the first production attempt
# (gl_prod_london0_v1_preclustering_20260522) left ~340 healpix not done
# (absent or truncated), SCATTERED through the 0..1149 position space (many
# isolated singletons interleaved with completed files). The pipeline has no
# skip-if-output-exists, so a contiguous re-run would recompute hundreds of
# already-done files. A list-driven loop recomputes EXACTLY the not-done
# positions (0 waste). submit_desi_mock_gl.sh is left byte-identical (proven
# path); desi-DLAGP.py is untouched (science args identical to the range run).
#
# IMPORTANT: the python arg block below MUST stay in sync with
# submit_desi_mock_gl.sh. Only the loop differs.
#
# Driven by launch_gl_resume.sh, which computes LEVEL2_LIST from the run's
# processed/ dir (a position is "done" iff its h5 opens AND is gzip-compressed)
# and chunks it across sbatch jobs. LEVEL2_LIST is passed via --export.

#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J dla_inference_gl_resume
#SBATCH -o slurm/greatlakes/production/logs/gpdla_gl_resume_%j.log
#SBATCH -e slurm/greatlakes/production/logs/error_gl_resume_%j.log

set -eo pipefail
export PYTHONUNBUFFERED=1

# --- Pin BLAS threads to 1 (config-only; dla_gp.py is NOT touched) -----------
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

# --- defaults (mirror submit_desi_mock_gl.sh; flavour configs override) -------
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

# FILTER=1 truncated-sampler tuning knobs (defaults left empty → CLI defaults).
FILTER_N_INITIAL_FLOOR="${FILTER_N_INITIAL_FLOOR:-}"
FILTER_EMPTY_MASK_FALLTHROUGH="${FILTER_EMPTY_MASK_FALLTHROUGH:-0}"

LEVEL2_LIST="${LEVEL2_LIST:?must be set via --export (comma-separated speclist positions)}"

mkdir -p "$OUTDIR" "${OUTDIR}/logs"

echo "[gl-resume] $(date) job=$SLURM_JOB_ID"
echo "[gl-resume] LEARNED_FILE=$LEARNED_FILE"
echo "[gl-resume] OUTDIR=$OUTDIR"
echo "[gl-resume] positions (${LEVEL2_LIST//:/ } )"
echo "[gl-resume] n_positions=$(awk -F: '{print NF}' <<< "$LEVEL2_LIST")"

# --- list-driven loop: one single-file slice per not-done position -----------
# --exact + --overlap so the backgrounded srun steps PACK onto the allocation
# (N=2 concurrent at SRUN_CPUS each), identical to submit_desi_mock_gl.sh.
# LEVEL2_LIST is ':'-separated (NOT ',' — sbatch --export reserves the comma).
IFS=':' read -ra POSITIONS <<< "$LEVEL2_LIST"
for P in "${POSITIONS[@]}"; do
    [ -z "$P" ] && continue
    LEVEL2_START=$((P))
    LEVEL2_END=$((P + 1))
    echo "[gl-resume] position ${LEVEL2_START} (slice ${LEVEL2_START}..${LEVEL2_END})"

    srun --exact --overlap -N 1 -n 1 -c "${SRUN_CPUS:-${MAX_WORKERS}}" \
        --output="${OUTDIR}/logs/resume_run_${LEVEL2_START}_%j_%t.log" \
        --error="${OUTDIR}/logs/error_resume_${LEVEL2_START}_%j_%t.log" \
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
            $([ -n "$FILTER_N_INITIAL_FLOOR" ] && echo "--filter_n_initial_floor $FILTER_N_INITIAL_FLOOR") \
            $([ "$FILTER_EMPTY_MASK_FALLTHROUGH" = "1" ] && echo "--filter_empty_mask_fallthrough 1") \
            --figure_dir "$FIGURE_DIR" \
            --level2_start "$LEVEL2_START" \
            --level2_end "$LEVEL2_END" &
done

wait
echo "[gl-resume] $(date) all positions done"
