#!/bin/bash
# slurm/nersc/production/submit_desi_loa_nersc.sh
#
# NERSC (Perlmutter) REAL-LOA inference inner script — the healpix-mode analog of
# submit_desi_mock_nersc.sh. Same NERSC-validated parallelism (ONE `srun -n NTASKS`
# multi-task launch, PROCID-dispatched contiguous HPX chunks; NO backgrounded srun).
# Differs from the mock inner only in: no --mocks/--mockdir; --hpx_start/--hpx_end
# instead of --level2_start/--level2_end; real LOA release/survey/program.
#
# Run via launch_nersc.sh (MODE=loa). Re-entrant: the driver branch launches
# `srun ... "$SELF"` with LOA_TASK=1. Task k processes HPX indices
# [START_INDEX + k*per, +per) where per = ceil((END_INDEX-START_INDEX)/NTASKS).
# Use a LARGE --window (~12-20 hpx/task) for load balancing.

#SBATCH -A desi
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J dla_loa_nersc
#SBATCH -o slurm/nersc/production/logs/gpdla_loa_nersc_%j.log
#SBATCH -e slurm/nersc/production/logs/error_loa_nersc_%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- required + default knobs ------------------------------------------------
QSOCAT="${QSOCAT:?must be set via --export}"
RELEASE="${RELEASE:-loa}"
PROGRAM="${PROGRAM:-dark}"
SURVEY="${SURVEY:-main}"
OUTDIR="${OUTDIR:?must be set via --export}"
BALMASK="${BALMASK:-false}"
LEARNED_FILE="${LEARNED_FILE:?must be set via --export}"
CATALOG_NAME="${CATALOG_NAME:?must be set via --export}"
LOS_CATALOG="${LOS_CATALOG:?must be set via --export}"
DLA_CATALOG="${DLA_CATALOG:?must be set via --export}"
DLA_SAMPLES_FILE="${DLA_SAMPLES_FILE:?must be set via --export}"
SUB_DLA_SAMPLES_FILE="${SUB_DLA_SAMPLES_FILE:?must be set via --export}"
MIN_Z_SEPARATION="${MIN_Z_SEPARATION:-3000.0}"
MAX_Z_CUT="${MAX_Z_CUT:-3000.0}"
MIN_Z_CUT="${MIN_Z_CUT:-3000.0}"
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

START_INDEX="${START_INDEX:-0}"     # HPX window for THIS sbatch (split across NTASKS)
END_INDEX="${END_INDEX:-32}"
NTASKS="${NTASKS:-32}"
SELF="${SELF:?must be set via --export (absolute path to this script)}"
# Option B (opt-in, default off): path to a per-HPX-index spec-count table
# (tools/loa_hpx_spec_counts.py). When set, the driver computes spec-weighted
# CONTIGUOUS task boundaries instead of the equal-COUNT split. Coverage is
# identical either way; only task load balance changes.
LOA_BALANCE_COUNTS="${LOA_BALANCE_COUNTS:-}"

build_and_run () {
    local hs="$1" he="$2"
    python desi-DLAGP.py \
        --qsocat "$QSOCAT" --release "$RELEASE" --program "$PROGRAM" --survey "$SURVEY" \
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
        --max_z_cut "$MAX_Z_CUT" --min_z_cut "$MIN_Z_CUT" \
        --max_noise_variance "$MAX_NOISE_VARIANCE" --num_forest_lines "$NUM_FOREST_LINES" --num_lines "$NUM_LINES" \
        --enable_tau_eb "$ENABLE_TAU_EB" --tau_eb_objective "$TAU_EB_OBJECTIVE" --early_stop_mode "$EARLY_STOP_MODE" \
        --pair_prior_mode "$PAIR_PRIOR_MODE" --dla_bias "$DLA_BIAS" \
        $([ -n "$FILTER_N_INITIAL_FLOOR" ] && echo "--filter_n_initial_floor $FILTER_N_INITIAL_FLOOR") \
        $([ "$FILTER_EMPTY_MASK_FALLTHROUGH" = "1" ] && echo "--filter_empty_mask_fallthrough 1") \
        --figure_dir "$FIGURE_DIR" \
        $([ -n "${PIXEL_COL:-}" ] && echo "--pixel_col $PIXEL_COL") \
        $([ -n "${EXTERNAL_HPX_LIST:-}" ] && echo "--use_external_hpx_list --external_hpx_list $EXTERNAL_HPX_LIST") \
        --hpx_start "$hs" --hpx_end "$he"
}

# TASK branch: this task's CONTIGUOUS HPX index range.
#   balanced (Option B): [b_k, b_{k+1}) read from BALANCE_BOUNDARIES_FILE
#                        (NTASKS+1 boundary lines, written by the driver)
#   default            : equal-COUNT split [START + k*per, +per)
# Both tile [START,END) exactly; the empty-range guard (hs >= he) is shared.
if [ "${LOA_TASK:-0}" = "1" ]; then
    k="${SLURM_PROCID:-0}"
    if [ -n "${BALANCE_BOUNDARIES_FILE:-}" ] && [ -r "${BALANCE_BOUNDARIES_FILE:-}" ]; then
        hs=$(sed -n "$(( k + 1 ))p" "$BALANCE_BOUNDARIES_FILE")
        he=$(sed -n "$(( k + 2 ))p" "$BALANCE_BOUNDARIES_FILE")
        if [ -z "$hs" ] || [ -z "$he" ]; then
            echo "[task $k] no boundary lines in $BALANCE_BOUNDARIES_FILE; exit"; exit 0
        fi
    else
        span=$(( END_INDEX - START_INDEX ))
        per=$(( (span + NTASKS - 1) / NTASKS )); [ "$per" -lt 1 ] && per=1
        hs=$(( START_INDEX + k * per )); he=$(( hs + per ))
        [ "$he" -gt "$END_INDEX" ] && he="$END_INDEX"
    fi
    if [ "$hs" -ge "$he" ]; then echo "[task $k] no work (hs=$hs >= he=$he)"; exit 0; fi
    echo "[task $k] hpx ${hs}..${he}"
    build_and_run "$hs" "$he"; rc=$?   # propagate inner-python exit (don't mask a crash as COMPLETED)
    exit "$rc"
fi

# DRIVER branch: env + ONE srun -n NTASKS.
NERSC_ENV_SETUP="${NERSC_ENV_SETUP:-source /global/cfs/cdirs/desi/software/desi_environment.sh main}"
set +u; eval "$NERSC_ENV_SETUP"; set -u
mkdir -p "$OUTDIR" "${OUTDIR}/logs"
echo "[loa] $(date) job=${SLURM_JOB_ID:-NA} hpx_window=${START_INDEX}..${END_INDEX} ntasks=${NTASKS} W=${MAX_WORKERS}"
echo "[loa] LEARNED_FILE=$LEARNED_FILE  OUTDIR=$OUTDIR"

# Option B (opt-in): compute spec-weighted task boundaries ONCE for this window;
# each task reads its [b_k,b_{k+1}) via SLURM_PROCID. Requested-but-broken is a
# loud failure (never silently fall back to a different split).
if [ -n "$LOA_BALANCE_COUNTS" ]; then
    if [ ! -r "$LOA_BALANCE_COUNTS" ]; then
        echo "[loa] ERROR: LOA_BALANCE_COUNTS=$LOA_BALANCE_COUNTS not readable" >&2; exit 1
    fi
    BALANCE_BOUNDARIES_FILE="${OUTDIR}/logs/boundaries_${START_INDEX}-${END_INDEX}_${SLURM_JOB_ID:-NA}.txt"
    if ! python "${REPO_ROOT:-.}/tools/loa_balance_boundaries.py" \
            --counts "$LOA_BALANCE_COUNTS" --start "$START_INDEX" --end "$END_INDEX" \
            --ntasks "$NTASKS" --out "$BALANCE_BOUNDARIES_FILE" --verify; then
        echo "[loa] ERROR: boundary computation failed (balance requested); aborting" >&2; exit 1
    fi
    export BALANCE_BOUNDARIES_FILE
    echo "[loa] balance=ON (Option B) -> $BALANCE_BOUNDARIES_FILE"
else
    echo "[loa] balance=OFF (equal-count split)"
fi
echo "[loa] MAX_DLAS=$MAX_DLAS SINGLE_ABSORBER_MODEL=$SINGLE_ABSORBER_MODEL NUM_DLA_SAMPLES=$NUM_DLA_SAMPLES FILTER=$FILTER_LOW_LIKELIHOOD"
srun -N 1 -n "${NTASKS}" -c "${MAX_WORKERS}" --cpu-bind=cores \
     --output="${OUTDIR}/logs/loa_run_${START_INDEX}-${END_INDEX}_%j_%t.log" \
     --error="${OUTDIR}/logs/error_loa_${START_INDEX}-${END_INDEX}_%j_%t.log" \
     --export=ALL,LOA_TASK=1 "${SELF}"
echo "[loa] $(date) hpx_window ${START_INDEX}..${END_INDEX} done"
