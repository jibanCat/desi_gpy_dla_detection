#!/bin/bash
#
# slurm/resume_missing_slices.sh — sbatch wrapper for re-launching just the
# missing slices of an interrupted run_local.sh run.
#
# Use this when a previous run launched via slurm/run_local.sh on a jupyter
# compute node was killed mid-flight (e.g. session expired) and only some of
# the per-spectra-16 slices completed. Pass the list of MISSING level2 indices
# and this script will re-run only those, each as a backgrounded python.
#
# Submit with:
#   sbatch --export=ALL,\
#     CONFIG_PATH=slurm/configs/london0_y3.env,\
#     OUTDIR=/pscratch/sd/j/jibancat/prod533_5k_20260511/london_v3_loa124_early_stop_A,\
#     MISSING_SLICES="0 2 4 6 7",\
#     EARLY_STOP_MODE=A \
#     slurm/resume_missing_slices.sh
#
# Optional env vars: TAU_EB_OBJECTIVE, anything that run_local.sh's build_cmd
# uses can be overridden via --export.
#
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A desi
#SBATCH --time=02:00:00
#SBATCH --job-name=gpdla_resume
#SBATCH --output=resume_%j.log
#SBATCH --error=resume_%j.err

set -eo pipefail

# ---- locate repo root ------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- mandatory inputs from --export ----------------------------------------
: "${CONFIG_PATH:?CONFIG_PATH env var is required}"
: "${OUTDIR:?OUTDIR env var is required}"
: "${MISSING_SLICES:?MISSING_SLICES env var is required (space-separated level2 indices)}"

# Resolve CONFIG_PATH relative to repo if not absolute
if [ ! -f "$CONFIG_PATH" ]; then
    if [ -f "${REPO_ROOT}/${CONFIG_PATH}" ]; then
        CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
    else
        echo "[resume] config not found: $CONFIG_PATH" >&2
        exit 2
    fi
fi

# Resolve OUTDIR to absolute
OUTDIR="$(readlink -m "$OUTDIR")"

# Validate OUTDIR is inside allowed write area
ALLOWED_PREFIXES=(
    "/pscratch/sd/j/jibancat/"
    "/global/homes/j/jibancat/"
    "/global/cfs/cdirs/desicollab/users/jibancat/"
)
ok=0
for p in "${ALLOWED_PREFIXES[@]}"; do
    case "$OUTDIR/" in "$p"*) ok=1; break ;; esac
done
if [ "$ok" -ne 1 ]; then
    echo "[resume] REFUSING: OUTDIR=$OUTDIR is outside the allowed write roots" >&2
    exit 3
fi

# ---- env setup -------------------------------------------------------------
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main

# ---- source the run config -------------------------------------------------
# shellcheck disable=SC1090
source "$CONFIG_PATH"

# Override OUTDIR after sourcing (config files set their own OUTDIR sometimes)
OUTDIR="$(readlink -m "$OUTDIR")"
mkdir -p "${OUTDIR}/logs"

# ---- echo plan -------------------------------------------------------------
HOST="$(hostname)"
JOB="${SLURM_JOB_ID:-(none)}"
START_UTC="$(date -u +%FT%TZ)"
echo "[resume] job=$JOB host=$HOST start=$START_UTC"
echo "[resume] CONFIG_PATH=$CONFIG_PATH"
echo "[resume] OUTDIR=$OUTDIR"
echo "[resume] MISSING_SLICES=$MISSING_SLICES"
echo "[resume] EARLY_STOP_MODE=${EARLY_STOP_MODE:-(unset)}"
echo "[resume] ENABLE_TAU_EB=${ENABLE_TAU_EB:-(unset)} TAU_EB_OBJECTIVE=${TAU_EB_OBJECTIVE:-(unset)}"
echo "[resume] MAX_DLAS=${MAX_DLAS} SINGLE_ABSORBER_MODEL=${SINGLE_ABSORBER_MODEL} FILTER_LOW_LIKELIHOOD=${FILTER_LOW_LIKELIHOOD}"
echo "[resume] NUM_DLA_SAMPLES=${NUM_DLA_SAMPLES} DLA_SAMPLES_FILE=${DLA_SAMPLES_FILE}"
echo "[resume] node has $(nproc) CPUs"

# Snapshot resume settings beside RUN_SETTINGS.md so the resume is traceable.
{
    echo "# Resume run — $START_UTC"
    echo
    echo "Resumed by sbatch job=$JOB on host=$HOST."
    echo
    echo "## Source config"
    echo "- file: \`$CONFIG_PATH\`"
    echo
    echo "## Missing slices re-run"
    echo "- \`MISSING_SLICES\` = \`$MISSING_SLICES\`"
    echo
    echo "## Key resolved values"
    for v in MODE QSOCAT MOCKDIR LEARNED_FILE \
             DLA_SAMPLES_FILE NUM_DLA_SAMPLES \
             SUB_DLA_SAMPLES_FILE NUM_SUBDLA_SAMPLES \
             MAX_DLAS SINGLE_ABSORBER_MODEL FILTER_LOW_LIKELIHOOD \
             PREV_TAU_0 PREV_BETA DLAMBDA K \
             ENABLE_TAU_EB TAU_EB_OBJECTIVE EARLY_STOP_MODE; do
        echo "- \`$v\` = \`${!v:-(unset)}\`"
    done
} > "${OUTDIR}/RESUME_SETTINGS_${JOB}.md"

# ---- build_cmd factored from run_local.sh (verbatim flag list) -------------
build_cmd() {
    local _idx_start="$1" _idx_end="$2"
    local cmd=(
        python "${REPO_ROOT}/desi-DLAGP.py"
        --qsocat "$QSOCAT"
        --release "$RELEASE"
        --program "$PROGRAM"
        --survey "$SURVEY"
        --outdir "$OUTDIR"
        --learned_file "$LEARNED_FILE"
        --catalog_name "$CATALOG_NAME"
        --los_catalog "$LOS_CATALOG"
        --dla_catalog "$DLA_CATALOG"
        --dla_samples_file "$DLA_SAMPLES_FILE"
        --sub_dla_samples_file "$SUB_DLA_SAMPLES_FILE"
        --min_z_separation "$MIN_Z_SEPARATION"
        --prev_tau_0 "$PREV_TAU_0"
        --prev_beta "$PREV_BETA"
        --max_dlas "$MAX_DLAS"
        --plot_figures "$PLOT_FIGURES"
        --filter_low_likelihood "$FILTER_LOW_LIKELIHOOD"
        --single_absorber_model "$SINGLE_ABSORBER_MODEL"
        --max_workers "$MAX_WORKERS"
        --batch_size "$BATCH_SIZE"
        --loading_min_lambda "$LOADING_MIN_LAMBDA"
        --loading_max_lambda "$LOADING_MAX_LAMBDA"
        --normalization_min_lambda "$NORMALIZATION_MIN_LAMBDA"
        --normalization_max_lambda "$NORMALIZATION_MAX_LAMBDA"
        --min_lambda "$MIN_LAMBDA"
        --max_lambda "$MAX_LAMBDA"
        --dlambda "$DLAMBDA"
        --k "$K"
        --num_dla_samples "$NUM_DLA_SAMPLES"
        --num_subdla_samples "$NUM_SUBDLA_SAMPLES"
        --max_noise_variance "$MAX_NOISE_VARIANCE"
        --num_forest_lines "$NUM_FOREST_LINES"
        --num_lines "$NUM_LINES"
        --figure_dir "$OUTDIR"
    )
    if [ "$MODE" = "loa" ]; then
        cmd+=(--hpx_start "$_idx_start" --hpx_end "$_idx_end")
    else
        cmd+=(--mocks --mockdir "$MOCKDIR" --level2_start "$_idx_start" --level2_end "$_idx_end")
    fi
    if [ "$BALMASK" = "true" ]; then
        cmd+=(--balmask)
    fi
    if [ "${ENABLE_TAU_EB:-0}" = "1" ]; then
        cmd+=(--enable_tau_eb 1)
        if [ -n "${TAU_EB_FACTORS:-}" ]; then
            cmd+=(--tau_eb_factors ${TAU_EB_FACTORS})
        fi
        if [ -n "${TAU_EB_OBJECTIVE:-}" ]; then
            cmd+=(--tau_eb_objective "${TAU_EB_OBJECTIVE}")
        fi
        if [ "${TAU_EB_APPLY_HCD_MASK:-0}" = "1" ]; then
            cmd+=(--tau_eb_apply_hcd_mask 1)
        fi
    fi
    if [ -n "${EARLY_STOP_MODE:-}" ]; then
        cmd+=(--early_stop_mode "${EARLY_STOP_MODE}")
    fi
    printf '%q ' "${cmd[@]}"
}

# ---- launch all missing slices in parallel ---------------------------------
declare -a pids
for idx_start in $MISSING_SLICES; do
    idx_end=$(( idx_start + 1 ))
    log_file="${OUTDIR}/logs/resume_${idx_start}_${idx_end}_${JOB}.log"
    cmd_str=$(build_cmd "$idx_start" "$idx_end")
    echo "[resume] $(date +%H:%M:%S) launching slice ${idx_start}..${idx_end}  log=${log_file##*/}"
    (eval "$cmd_str" > "$log_file" 2>&1) &
    pids+=("$!")
done

echo "[resume] $(date +%H:%M:%S) waiting for ${#pids[@]} python procs ..."
fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[resume] pid $pid exited non-zero" >&2
        fail=1
    fi
done

echo "[resume] $(date +%H:%M:%S) all done. fail=$fail"
exit $fail
