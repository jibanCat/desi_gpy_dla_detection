#!/bin/bash
#
# slurm/resume_local.sh — re-launch the missing slices of an interrupted
# run_local.sh run, directly on the current (allocated) compute node, without
# going through the slurm scheduler. Use this when the queue wait is longer
# than the jupyter session itself.
#
# Logic mirrors slurm/resume_missing_slices.sh exactly; the only difference is
# (a) no #SBATCH header and (b) no scancel-fragile dependency on $SLURM_JOB_ID
# (we tag log files with $$ instead).
#
# Usage:
#   CONFIG_PATH=slurm/configs/london0_y3.env \
#   OUTDIR=/pscratch/sd/j/jibancat/.../london_v3_loa124_early_stop_A \
#   MISSING_SLICES="0 2 4 6 7" EARLY_STOP_MODE=A \
#   bash slurm/resume_local.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${CONFIG_PATH:?CONFIG_PATH env var is required}"
: "${OUTDIR:?OUTDIR env var is required}"
: "${MISSING_SLICES:?MISSING_SLICES env var is required}"

if [ ! -f "$CONFIG_PATH" ]; then
    if [ -f "${REPO_ROOT}/${CONFIG_PATH}" ]; then
        CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
    else
        echo "[resume-local] config not found: $CONFIG_PATH" >&2
        exit 2
    fi
fi
OUTDIR="$(readlink -m "$OUTDIR")"

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
    echo "[resume-local] REFUSING: OUTDIR=$OUTDIR outside allowed write roots" >&2
    exit 3
fi

# DESI env (idempotent)
if [ -z "${DESI_ROOT:-}" ]; then
    source /usr/share/lmod/lmod/init/bash
    export DESI_ROOT=/global/cfs/cdirs/desi
    source /global/common/software/desi/desi_environment.sh main
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"
OUTDIR="$(readlink -m "$OUTDIR")"
mkdir -p "${OUTDIR}/logs"

TAG="$$_$(date +%s)"
HOST="$(hostname)"
START_UTC="$(date -u +%FT%TZ)"
echo "[resume-local] tag=$TAG host=$HOST start=$START_UTC"
echo "[resume-local] CONFIG_PATH=$CONFIG_PATH"
echo "[resume-local] OUTDIR=$OUTDIR"
echo "[resume-local] MISSING_SLICES=$MISSING_SLICES"
echo "[resume-local] EARLY_STOP_MODE=${EARLY_STOP_MODE:-(unset)}  ENABLE_TAU_EB=${ENABLE_TAU_EB:-(unset)}"

{
    echo "# Resume run (local) — $START_UTC"
    echo
    echo "Resumed by inline shell tag=$TAG on host=$HOST (no sbatch — queue wait >3 days)."
    echo
    echo "## Source config"
    echo "- file: \`$CONFIG_PATH\`"
    echo
    echo "## Missing slices"
    echo "- \`MISSING_SLICES\` = \`$MISSING_SLICES\`"
    echo
    echo "## Key resolved values"
    for v in MODE QSOCAT MOCKDIR LEARNED_FILE \
             DLA_SAMPLES_FILE NUM_DLA_SAMPLES \
             SUB_DLA_SAMPLES_FILE NUM_SUBDLA_SAMPLES \
             MAX_DLAS SINGLE_ABSORBER_MODEL FILTER_LOW_LIKELIHOOD \
             FILTER_N_INITIAL_FLOOR FILTER_EMPTY_MASK_FALLTHROUGH \
             PREV_TAU_0 PREV_BETA DLAMBDA K \
             ENABLE_TAU_EB TAU_EB_OBJECTIVE EARLY_STOP_MODE; do
        echo "- \`$v\` = \`${!v:-(unset)}\`"
    done
} > "${OUTDIR}/RESUME_LOCAL_${TAG}.md"

build_cmd() {
    local _idx_start="$1" _idx_end="$2"
    local cmd=(
        python "${REPO_ROOT}/desi-DLAGP.py"
        --qsocat "$QSOCAT" --release "$RELEASE" --program "$PROGRAM" --survey "$SURVEY"
        --outdir "$OUTDIR" --learned_file "$LEARNED_FILE"
        --catalog_name "$CATALOG_NAME" --los_catalog "$LOS_CATALOG" --dla_catalog "$DLA_CATALOG"
        --dla_samples_file "$DLA_SAMPLES_FILE" --sub_dla_samples_file "$SUB_DLA_SAMPLES_FILE"
        --min_z_separation "$MIN_Z_SEPARATION"
        --prev_tau_0 "$PREV_TAU_0" --prev_beta "$PREV_BETA"
        --max_dlas "$MAX_DLAS" --plot_figures "$PLOT_FIGURES"
        --filter_low_likelihood "$FILTER_LOW_LIKELIHOOD"
        --single_absorber_model "$SINGLE_ABSORBER_MODEL"
        --max_workers "$MAX_WORKERS" --batch_size "$BATCH_SIZE"
        --loading_min_lambda "$LOADING_MIN_LAMBDA" --loading_max_lambda "$LOADING_MAX_LAMBDA"
        --normalization_min_lambda "$NORMALIZATION_MIN_LAMBDA" --normalization_max_lambda "$NORMALIZATION_MAX_LAMBDA"
        --min_lambda "$MIN_LAMBDA" --max_lambda "$MAX_LAMBDA"
        --dlambda "$DLAMBDA" --k "$K"
        --num_dla_samples "$NUM_DLA_SAMPLES" --num_subdla_samples "$NUM_SUBDLA_SAMPLES"
        --max_noise_variance "$MAX_NOISE_VARIANCE"
        --num_forest_lines "$NUM_FOREST_LINES" --num_lines "$NUM_LINES"
        --figure_dir "$OUTDIR"
    )
    if [ "$MODE" = "loa" ]; then
        cmd+=(--hpx_start "$_idx_start" --hpx_end "$_idx_end")
    else
        cmd+=(--mocks --mockdir "$MOCKDIR" --level2_start "$_idx_start" --level2_end "$_idx_end")
    fi
    if [ "$BALMASK" = "true" ]; then cmd+=(--balmask); fi
    if [ "${ENABLE_TAU_EB:-0}" = "1" ]; then
        cmd+=(--enable_tau_eb 1)
        if [ -n "${TAU_EB_FACTORS:-}" ]; then cmd+=(--tau_eb_factors ${TAU_EB_FACTORS}); fi
        if [ -n "${TAU_EB_OBJECTIVE:-}" ]; then cmd+=(--tau_eb_objective "${TAU_EB_OBJECTIVE}"); fi
        if [ "${TAU_EB_APPLY_HCD_MASK:-0}" = "1" ]; then cmd+=(--tau_eb_apply_hcd_mask 1); fi
    fi
    if [ -n "${EARLY_STOP_MODE:-}" ]; then cmd+=(--early_stop_mode "${EARLY_STOP_MODE}"); fi
    # FILTER=1 knobs (see docs/notes/2026-05-13_filter1_knob_tuning.md)
    if [ -n "${FILTER_N_INITIAL_FLOOR:-}" ]; then
        cmd+=(--filter_n_initial_floor "${FILTER_N_INITIAL_FLOOR}")
    fi
    if [ "${FILTER_EMPTY_MASK_FALLTHROUGH:-0}" = "1" ]; then
        cmd+=(--filter_empty_mask_fallthrough 1)
    fi
    printf '%q ' "${cmd[@]}"
}

declare -a pids
for idx_start in $MISSING_SLICES; do
    idx_end=$(( idx_start + 1 ))
    log_file="${OUTDIR}/logs/resume_local_${idx_start}_${idx_end}_${TAG}.log"
    cmd_str=$(build_cmd "$idx_start" "$idx_end")
    echo "[resume-local] $(date +%H:%M:%S) launching slice ${idx_start}..${idx_end}  log=${log_file##*/}"
    (eval "$cmd_str" > "$log_file" 2>&1) &
    pids+=("$!")
done

echo "[resume-local] $(date +%H:%M:%S) waiting for ${#pids[@]} python procs ..."
fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[resume-local] pid $pid exited non-zero" >&2
        fail=1
    fi
done

echo "[resume-local] $(date +%H:%M:%S) all done. fail=$fail"
exit $fail
