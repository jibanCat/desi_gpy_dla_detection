#!/usr/bin/env bash
# slurm/launch.sh — generic outer driver for GP-DLA production runs.
#
# Sources a flavour config from slurm/configs/<flavour>.env, validates the
# resolved OUTDIR is inside one of the allowed write areas on NERSC, and
# sbatches the matching inner script (submit_desi_loa.sh / submit_desi_mock.sh)
# with the right index window.
#
# Usage:
#   bash slurm/launch.sh slurm/configs/london0_y3.env
#   bash slurm/launch.sh slurm/configs/loa_y3_lls172.env --start 0 --end 4992
#   bash slurm/launch.sh slurm/configs/saclay0_y3.env --dry-run
#
# Options:
#   --start N      override outer-loop start (default 0)
#   --end N        override outer-loop end (default OUTER_MAX_INDEX)
#   --window N     override the spectra-per-sbatch window (default OUTER_WINDOW
#                  from config; smaller = fewer spectra per sbatch, useful for
#                  debug passes — e.g. --window 4 with --end 0 gives a single
#                  sbatch covering ~4 spectra-16 files for mocks)
#   --dry-run      print the sbatch commands without submitting
#   --no-sleep     skip the inter-sbatch sleep (default sleeps 60 s)
#   --outdir PATH  override OUTDIR from the config
#
# The script must run from anywhere — paths are resolved relative to the repo
# root inferred from the script's own location.

set -euo pipefail

# ---- locate repo root (parent of slurm/) ------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- parse args -------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.env> [--start N] [--end N] [--dry-run] [--no-sleep] [--outdir PATH]" >&2
    exit 2
fi

CONFIG_PATH="$1"; shift
START_OVERRIDE=""
END_OVERRIDE=""
WINDOW_OVERRIDE=""
DRY_RUN=0
NO_SLEEP=0
OUTDIR_OVERRIDE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --start)    START_OVERRIDE="$2"; shift 2 ;;
        --end)      END_OVERRIDE="$2";   shift 2 ;;
        --window)   WINDOW_OVERRIDE="$2"; shift 2 ;;
        --outdir)   OUTDIR_OVERRIDE="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --no-sleep) NO_SLEEP=1; shift ;;
        *) echo "[launch] unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Resolve config relative to repo root if it's not absolute and doesn't exist
if [ ! -f "$CONFIG_PATH" ]; then
    if [ -f "${REPO_ROOT}/${CONFIG_PATH}" ]; then
        CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
    elif [ -f "${REPO_ROOT}/slurm/configs/${CONFIG_PATH}" ]; then
        CONFIG_PATH="${REPO_ROOT}/slurm/configs/${CONFIG_PATH}"
    else
        echo "[launch] config not found: $CONFIG_PATH" >&2
        exit 2
    fi
fi

# ---- source the config ------------------------------------------------------
# shellcheck disable=SC1090
source "$CONFIG_PATH"

# ---- apply overrides --------------------------------------------------------
if [ -n "$OUTDIR_OVERRIDE" ]; then OUTDIR="$OUTDIR_OVERRIDE"; fi
if [ -n "$WINDOW_OVERRIDE" ]; then OUTER_WINDOW="$WINDOW_OVERRIDE"; fi

# ---- validate required vars from the config ---------------------------------
for var in MODE QSOCAT OUTDIR LEARNED_FILE OUTER_MAX_INDEX OUTER_STEP OUTER_WINDOW; do
    if [ -z "${!var:-}" ]; then
        echo "[launch] config $CONFIG_PATH missing required var: $var" >&2
        exit 2
    fi
done
if [ "$MODE" = "mock" ] && [ -z "${MOCKDIR:-}" ]; then
    echo "[launch] mock mode requires MOCKDIR in $CONFIG_PATH" >&2
    exit 2
fi

# ---- enforce allowed write paths --------------------------------------------
# OUTDIR must start with one of: /pscratch/sd/j/jibancat/,
#                                 /global/homes/j/jibancat/,
#                                 /global/cfs/cdirs/desicollab/users/jibancat/
ALLOWED_PREFIXES=(
    "/pscratch/sd/j/jibancat/"
    "/global/homes/j/jibancat/"
    "/global/cfs/cdirs/desicollab/users/jibancat/"
)
abs_outdir="$(readlink -m "$OUTDIR")"
ok=0
for p in "${ALLOWED_PREFIXES[@]}"; do
    case "$abs_outdir/" in "$p"*) ok=1; break ;; esac
done
if [ "$ok" -ne 1 ]; then
    echo "[launch] REFUSING: OUTDIR=$abs_outdir is outside the allowed write roots" >&2
    printf '  Allowed: %s\n' "${ALLOWED_PREFIXES[@]}" >&2
    exit 3
fi
OUTDIR="$abs_outdir"

# ---- input sanity checks ----------------------------------------------------
[ -r "$QSOCAT" ]       || { echo "[launch] QSOCAT not readable: $QSOCAT" >&2; exit 4; }
[ -r "$LEARNED_FILE" ] || { echo "[launch] LEARNED_FILE not readable: $LEARNED_FILE" >&2; exit 4; }
if [ "$MODE" = "mock" ]; then
    [ -d "$MOCKDIR" ] || { echo "[launch] MOCKDIR not readable: $MOCKDIR" >&2; exit 4; }
fi

# ---- pick the inner sbatch script -------------------------------------------
case "$MODE" in
    loa)  INNER="${REPO_ROOT}/slurm/submit_desi_loa.sh"  ;;
    mock) INNER="${REPO_ROOT}/slurm/submit_desi_mock.sh" ;;
    *) echo "[launch] unknown MODE=$MODE (must be loa|mock)" >&2; exit 2 ;;
esac
[ -f "$INNER" ] || { echo "[launch] inner script missing: $INNER" >&2; exit 4; }

# ---- determine loop range ---------------------------------------------------
LOOP_START="${START_OVERRIDE:-0}"
LOOP_END="${END_OVERRIDE:-$OUTER_MAX_INDEX}"

# ---- mkdir OUTDIR + logs/ (skipped on --dry-run) ----------------------------
if [ "$DRY_RUN" -ne 1 ]; then
    mkdir -p "$OUTDIR" "${OUTDIR}/logs"
fi

# ---- echo plan --------------------------------------------------------------
cat <<EOF
[launch] config:          $CONFIG_PATH
[launch] MODE=$MODE
[launch] OUTDIR=$OUTDIR
[launch] QSOCAT=$QSOCAT
[launch] LEARNED_FILE=$LEARNED_FILE
[launch] inner=$INNER
[launch] outer loop: $LOOP_START .. $LOOP_END  (step=$OUTER_STEP, window=$OUTER_WINDOW)
[launch] dry_run=$DRY_RUN  no_sleep=$NO_SLEEP
EOF

# Common export vars handed to every sbatch invocation
COMMON_EXPORT="QSOCAT=${QSOCAT},OUTDIR=${OUTDIR},LEARNED_FILE=${LEARNED_FILE},\
CATALOG_NAME=${CATALOG_NAME},LOS_CATALOG=${LOS_CATALOG},DLA_CATALOG=${DLA_CATALOG},\
DLA_SAMPLES_FILE=${DLA_SAMPLES_FILE},SUB_DLA_SAMPLES_FILE=${SUB_DLA_SAMPLES_FILE},\
PREV_TAU_0=${PREV_TAU_0},PREV_BETA=${PREV_BETA},\
LOADING_MIN_LAMBDA=${LOADING_MIN_LAMBDA},LOADING_MAX_LAMBDA=${LOADING_MAX_LAMBDA},\
NORMALIZATION_MIN_LAMBDA=${NORMALIZATION_MIN_LAMBDA},NORMALIZATION_MAX_LAMBDA=${NORMALIZATION_MAX_LAMBDA},\
MIN_LAMBDA=${MIN_LAMBDA},MAX_LAMBDA=${MAX_LAMBDA},DLAMBDA=${DLAMBDA},K=${K},\
NUM_FOREST_LINES=${NUM_FOREST_LINES},NUM_LINES=${NUM_LINES},\
MIN_Z_SEPARATION=${MIN_Z_SEPARATION},MAX_Z_CUT=${MAX_Z_CUT},MIN_Z_CUT=${MIN_Z_CUT},\
MAX_NOISE_VARIANCE=${MAX_NOISE_VARIANCE},\
MAX_DLAS=${MAX_DLAS},SINGLE_ABSORBER_MODEL=${SINGLE_ABSORBER_MODEL},\
FILTER_LOW_LIKELIHOOD=${FILTER_LOW_LIKELIHOOD},\
NUM_DLA_SAMPLES=${NUM_DLA_SAMPLES},NUM_SUBDLA_SAMPLES=${NUM_SUBDLA_SAMPLES},\
MAX_WORKERS=${MAX_WORKERS},BATCH_SIZE=${BATCH_SIZE},PLOT_FIGURES=${PLOT_FIGURES},\
BALMASK=${BALMASK},RELEASE=${RELEASE},PROGRAM=${PROGRAM},SURVEY=${SURVEY}"

if [ "$MODE" = "mock" ]; then
    COMMON_EXPORT="${COMMON_EXPORT},MOCKDIR=${MOCKDIR}"
fi

# ---- launch loop ------------------------------------------------------------
n_jobs=0
for (( i=LOOP_START; i<=LOOP_END; i+=OUTER_STEP )); do
    chunk_end=$(( i + OUTER_WINDOW ))
    case "$MODE" in
        loa)
            window_export="HPX_START_INDEX=${i},HPX_END_INDEX=${chunk_end}"
            ;;
        mock)
            # submit_desi_mock.sh further sub-loops with STEP=2 (level2 chunk).
            window_export="START_INDEX=${i},END_INDEX=${chunk_end},STEP=2"
            ;;
    esac

    full_export="ALL,${COMMON_EXPORT},${window_export}"

    cmd=(sbatch --chdir="$REPO_ROOT" --export="$full_export" "$INNER")

    echo "[launch] $(date +%H:%M:%S) chunk ${i}..${chunk_end}"
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  %q ' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi

    n_jobs=$(( n_jobs + 1 ))
    if [ "$NO_SLEEP" -ne 1 ] && [ "$DRY_RUN" -ne 1 ] && [ "$i" -lt "$LOOP_END" ]; then
        sleep 60
    fi
done

echo "[launch] done — submitted ${n_jobs} sbatch jobs."
