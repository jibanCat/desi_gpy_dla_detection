#!/usr/bin/env bash
# slurm/nersc/production/launch_nersc.sh
# NERSC (Perlmutter) analog of launch_gl.sh. Sources a flavour config, validates
# OUTDIR is inside an allowed NERSC write area, and submits one
# submit_desi_mock_nersc.sh per window of OUTER_WINDOW healpix files. Each sbatch
# is 1 node; the inner script splits its window across NTASKS srun tasks.
#
# Usage:
#   bash slurm/nersc/production/launch_nersc.sh london0_nersc_v1.env
#   bash slurm/nersc/production/launch_nersc.sh london0_nersc_v1.env --start 0 --end 32 --window 32
#   bash slurm/nersc/production/launch_nersc.sh london0_nersc_v1.env --qos debug --dry-run
#   NUM_SAMPLES=10000 bash slurm/nersc/production/launch_nersc.sh london0_nersc_v1.env --start 0 --end 32 --window 32

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <flavour.env> [--start N] [--end N] [--window N] [--qos Q] [--outdir PATH] [--dry-run] [--no-sleep]" >&2
    exit 2
fi

CONFIG_PATH="$1"; shift
START_OVERRIDE=""; END_OVERRIDE=""; WINDOW_OVERRIDE=""; OUTDIR_OVERRIDE=""; QOS_OVERRIDE=""; TIME_OVERRIDE=""
DRY_RUN=0; NO_SLEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --start)    START_OVERRIDE="$2"; shift 2 ;;
        --end)      END_OVERRIDE="$2";   shift 2 ;;
        --window)   WINDOW_OVERRIDE="$2"; shift 2 ;;
        --qos)      QOS_OVERRIDE="$2";   shift 2 ;;
        --time)     TIME_OVERRIDE="$2";  shift 2 ;;
        --outdir)   OUTDIR_OVERRIDE="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --no-sleep) NO_SLEEP=1; shift ;;
        *) echo "[launch-nersc] unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Resolve config (allow bare name relative to SCRIPT_DIR)
if [ ! -f "$CONFIG_PATH" ]; then
    if [ -f "${SCRIPT_DIR}/${CONFIG_PATH}" ]; then CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_PATH}";
    else echo "[launch-nersc] config not found: $CONFIG_PATH" >&2; exit 2; fi
fi
# shellcheck disable=SC1090
source "$CONFIG_PATH"

[ -n "$OUTDIR_OVERRIDE" ] && OUTDIR="$OUTDIR_OVERRIDE"

# Required vars
for var in MODE QSOCAT OUTDIR LEARNED_FILE OUTER_MAX_INDEX; do
    if [ -z "${!var:-}" ]; then echo "[launch-nersc] config missing required var: $var" >&2; exit 2; fi
done
[ "$MODE" = "mock" ] && [ -z "${MOCKDIR:-}" ] && { echo "[launch-nersc] mock mode requires MOCKDIR" >&2; exit 2; }

# Scheduler knobs (from _base_nersc.env, with CLI override for QOS)
ACCOUNT="${NERSC_SLURM_ACCOUNT:-desi}"
QOS="${QOS_OVERRIDE:-${NERSC_SLURM_QOS:-regular}}"
CONSTRAINT="${NERSC_SLURM_CONSTRAINT:-cpu}"
SLURM_TIME="${TIME_OVERRIDE:-${NERSC_SLURM_TIME:-08:00:00}}"
NTASKS="${NERSC_NTASKS:-32}"
W="${MAX_WORKERS:-8}"
WINDOW="${WINDOW_OVERRIDE:-${NERSC_WINDOW:-$NTASKS}}"   # files per sbatch; default 1/task

# Enforce allowed NERSC write roots (docs/nersc_write_permissions.md)
ALLOWED_PREFIXES=("${NERSC_ALLOWED_OUTPUT_PREFIXES[@]:-/pscratch/sd/j/jibancat/ /global/cfs/cdirs/desicollab/users/jibancat/}")
abs_outdir="$(readlink -m "$OUTDIR")"
ok=0
for p in "${ALLOWED_PREFIXES[@]}"; do case "$abs_outdir/" in "$p"*) ok=1; break ;; esac; done
if [ "$ok" -ne 1 ]; then
    echo "[launch-nersc] REFUSING: OUTDIR=$abs_outdir outside allowed NERSC write roots:" >&2
    printf '  %s\n' "${ALLOWED_PREFIXES[@]}" >&2; exit 3
fi
OUTDIR="$abs_outdir"

# Input sanity
[ -r "$QSOCAT" ]       || { echo "[launch-nersc] QSOCAT not readable: $QSOCAT" >&2; exit 4; }
[ -r "$LEARNED_FILE" ] || { echo "[launch-nersc] LEARNED_FILE not readable: $LEARNED_FILE" >&2; exit 4; }
[ -r "$DLA_SAMPLES_FILE" ] || { echo "[launch-nersc] DLA_SAMPLES_FILE not readable: $DLA_SAMPLES_FILE" >&2; exit 4; }
[ "$MODE" = "mock" ] && [ ! -d "$MOCKDIR" ] && { echo "[launch-nersc] MOCKDIR not a dir: $MOCKDIR" >&2; exit 4; }

case "$MODE" in
    mock) INNER="${SCRIPT_DIR}/submit_desi_mock_nersc.sh" ;;
    loa)  INNER="${SCRIPT_DIR}/submit_desi_loa_nersc.sh" ;;
    *) echo "[launch-nersc] unknown MODE=$MODE" >&2; exit 2 ;;
esac
[ -f "$INNER" ] || { echo "[launch-nersc] inner script missing: $INNER" >&2; exit 4; }

LOOP_START="${START_OVERRIDE:-0}"
LOOP_END="${END_OVERRIDE:-$OUTER_MAX_INDEX}"

# FS mutation only when NOT a dry run (dry-run hygiene)
if [ "$DRY_RUN" -ne 1 ]; then
    mkdir -p "$OUTDIR" "${OUTDIR}/logs" "${SCRIPT_DIR}/logs"
    cp "$CONFIG_PATH" "${OUTDIR}/$(basename "$CONFIG_PATH")"
    {
        echo "# Resolved env for run launched $(date)"
        echo "# config: $CONFIG_PATH"
        echo "CODE_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "CODE_BRANCH=$(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
        echo "CODE_DIRTY=$(git -C "$SCRIPT_DIR" diff --quiet 2>/dev/null && echo clean || echo dirty)"
        echo "PACKING=N${NTASKS}xW${W}"
        for var in MODE QSOCAT MOCKDIR OUTDIR LEARNED_FILE CATALOG_NAME LOS_CATALOG DLA_CATALOG \
                   DLA_SAMPLES_FILE SUB_DLA_SAMPLES_FILE NUM_DLA_SAMPLES NUM_SUBDLA_SAMPLES \
                   MAX_DLAS SINGLE_ABSORBER_MODEL FILTER_LOW_LIKELIHOOD MAX_LAMBDA MIN_LAMBDA DLAMBDA K \
                   NUM_FOREST_LINES NUM_LINES BALMASK PREV_TAU_0 PREV_BETA MAX_NOISE_VARIANCE \
                   MAX_WORKERS BATCH_SIZE ENABLE_TAU_EB TAU_EB_OBJECTIVE EARLY_STOP_MODE PAIR_PRIOR_MODE; do
            echo "$var=${!var:-(unset)}"
        done
    } > "${OUTDIR}/BASELINE.env"
fi

cat <<EOF
[launch-nersc] config:   $CONFIG_PATH
[launch-nersc] MODE=$MODE  OUTDIR=$OUTDIR
[launch-nersc] QSOCAT=$QSOCAT
[launch-nersc] LEARNED_FILE=$LEARNED_FILE
[launch-nersc] inner=$INNER  packing=N${NTASKS}xW${W}  qos=$QOS  window=$WINDOW
[launch-nersc] NUM_DLA_SAMPLES=$NUM_DLA_SAMPLES  loop ${LOOP_START}..${LOOP_END}
[launch-nersc] dry_run=$DRY_RUN
EOF

COMMON_EXPORT="QSOCAT=${QSOCAT},MOCKDIR=${MOCKDIR},OUTDIR=${OUTDIR},LEARNED_FILE=${LEARNED_FILE},\
CATALOG_NAME=${CATALOG_NAME},LOS_CATALOG=${LOS_CATALOG},DLA_CATALOG=${DLA_CATALOG},\
DLA_SAMPLES_FILE=${DLA_SAMPLES_FILE},SUB_DLA_SAMPLES_FILE=${SUB_DLA_SAMPLES_FILE},\
PREV_TAU_0=${PREV_TAU_0},PREV_BETA=${PREV_BETA},\
LOADING_MIN_LAMBDA=${LOADING_MIN_LAMBDA},LOADING_MAX_LAMBDA=${LOADING_MAX_LAMBDA},\
NORMALIZATION_MIN_LAMBDA=${NORMALIZATION_MIN_LAMBDA},NORMALIZATION_MAX_LAMBDA=${NORMALIZATION_MAX_LAMBDA},\
MIN_LAMBDA=${MIN_LAMBDA},MAX_LAMBDA=${MAX_LAMBDA},DLAMBDA=${DLAMBDA},K=${K},\
NUM_FOREST_LINES=${NUM_FOREST_LINES},NUM_LINES=${NUM_LINES},\
MIN_Z_SEPARATION=${MIN_Z_SEPARATION},MAX_Z_CUT=${MAX_Z_CUT:-3000.0},MIN_Z_CUT=${MIN_Z_CUT:-3000.0},MAX_NOISE_VARIANCE=${MAX_NOISE_VARIANCE},\
MAX_DLAS=${MAX_DLAS},SINGLE_ABSORBER_MODEL=${SINGLE_ABSORBER_MODEL},\
FILTER_LOW_LIKELIHOOD=${FILTER_LOW_LIKELIHOOD},\
FILTER_N_INITIAL_FLOOR=${FILTER_N_INITIAL_FLOOR:-},FILTER_EMPTY_MASK_FALLTHROUGH=${FILTER_EMPTY_MASK_FALLTHROUGH:-0},\
NUM_DLA_SAMPLES=${NUM_DLA_SAMPLES},NUM_SUBDLA_SAMPLES=${NUM_SUBDLA_SAMPLES},\
MAX_WORKERS=${W},BATCH_SIZE=${BATCH_SIZE},PLOT_FIGURES=${PLOT_FIGURES:-0},\
BALMASK=${BALMASK},RELEASE=${RELEASE},PROGRAM=${PROGRAM:-dark},SURVEY=${SURVEY:-main},\
ENABLE_TAU_EB=${ENABLE_TAU_EB:-1},TAU_EB_OBJECTIVE=${TAU_EB_OBJECTIVE:-null},EARLY_STOP_MODE=${EARLY_STOP_MODE:-baseline},\
PAIR_PRIOR_MODE=${PAIR_PRIOR_MODE:-off},DLA_BIAS=${DLA_BIAS:-2.0},\
NTASKS=${NTASKS},SELF=${INNER},NERSC_ENV_SETUP=${NERSC_ENV_SETUP:-source /global/cfs/cdirs/desi/software/desi_environment.sh main}"

n_jobs=0
for (( i=LOOP_START; i<LOOP_END; i+=WINDOW )); do
    chunk_end=$(( i + WINDOW )); [ "$chunk_end" -gt "$LOOP_END" ] && chunk_end="$LOOP_END"
    window_export="START_INDEX=${i},END_INDEX=${chunk_end}"
    full_export="ALL,${COMMON_EXPORT},${window_export}"
    cmd=(sbatch --chdir="$REPO_ROOT"
                --account="$ACCOUNT" --qos="$QOS" --constraint="$CONSTRAINT" --time="$SLURM_TIME"
                --nodes=1 --ntasks="$NTASKS" --cpus-per-task="$W"
                --export="$full_export" "$INNER")
    echo "[launch-nersc] $(date +%H:%M:%S) window ${i}..${chunk_end}"
    if [ "$DRY_RUN" -eq 1 ]; then printf '  %q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
    n_jobs=$(( n_jobs + 1 ))
    [ "$NO_SLEEP" -ne 1 ] && [ "$DRY_RUN" -ne 1 ] && [ "$chunk_end" -lt "$LOOP_END" ] && sleep 30
done
echo "[launch-nersc] submitted $n_jobs sbatch job(s)"
