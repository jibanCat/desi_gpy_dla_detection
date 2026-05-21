#!/usr/bin/env bash
# slurm/greatlakes/production/launch_gl.sh
# GreatLakes analog of slurm/launch.sh. Sources a flavour config from
# slurm/greatlakes/production/<flavour>.env, validates OUTDIR sits inside
# an allowed GL write area (/scratch/cavestru_root/cavestru0/mfho/* or
# /nfs/turbo/lsa-cavestru/mfho/*), and submits the matching GL inner
# script with the right index window.
#
# Usage:
#   bash slurm/greatlakes/production/launch_gl.sh london0_gl_v1.env
#   bash slurm/greatlakes/production/launch_gl.sh london0_gl_v1.env --start 0 --end 64
#   bash slurm/greatlakes/production/launch_gl.sh london0_gl_v1.env --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <flavour.env> [--start N] [--end N] [--window N] [--outdir PATH] [--dry-run] [--no-sleep]" >&2
    exit 2
fi

CONFIG_PATH="$1"; shift
START_OVERRIDE=""; END_OVERRIDE=""; WINDOW_OVERRIDE=""; OUTDIR_OVERRIDE=""
DRY_RUN=0; NO_SLEEP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --start)    START_OVERRIDE="$2"; shift 2 ;;
        --end)      END_OVERRIDE="$2";   shift 2 ;;
        --window)   WINDOW_OVERRIDE="$2"; shift 2 ;;
        --outdir)   OUTDIR_OVERRIDE="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --no-sleep) NO_SLEEP=1; shift ;;
        *) echo "[launch-gl] unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Resolve config
if [ ! -f "$CONFIG_PATH" ]; then
    if [ -f "${SCRIPT_DIR}/${CONFIG_PATH}" ]; then
        CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_PATH}"
    else
        echo "[launch-gl] config not found: $CONFIG_PATH" >&2; exit 2
    fi
fi
# shellcheck disable=SC1090
source "$CONFIG_PATH"

# Apply overrides
[ -n "$OUTDIR_OVERRIDE" ]  && OUTDIR="$OUTDIR_OVERRIDE"
[ -n "$WINDOW_OVERRIDE" ]  && OUTER_WINDOW="$WINDOW_OVERRIDE"

# Validate required vars
for var in MODE QSOCAT OUTDIR LEARNED_FILE OUTER_MAX_INDEX OUTER_STEP OUTER_WINDOW; do
    if [ -z "${!var:-}" ]; then
        echo "[launch-gl] config missing required var: $var" >&2; exit 2
    fi
done
[ "$MODE" = "mock" ] && [ -z "${MOCKDIR:-}" ] && {
    echo "[launch-gl] mock mode requires MOCKDIR" >&2; exit 2; }

# Enforce GL allowed-write roots
ALLOWED_PREFIXES=("${GL_ALLOWED_OUTPUT_PREFIXES[@]:-/scratch/cavestru_root/cavestru0/mfho/ /nfs/turbo/lsa-cavestru/mfho/}")
abs_outdir="$(readlink -m "$OUTDIR")"
ok=0
for p in "${ALLOWED_PREFIXES[@]}"; do
    case "$abs_outdir/" in "$p"*) ok=1; break ;; esac
done
if [ "$ok" -ne 1 ]; then
    echo "[launch-gl] REFUSING: OUTDIR=$abs_outdir outside allowed GL write roots:" >&2
    printf '  %s\n' "${ALLOWED_PREFIXES[@]}" >&2
    exit 3
fi
OUTDIR="$abs_outdir"

# Input sanity
[ -r "$QSOCAT" ]       || { echo "[launch-gl] QSOCAT not readable: $QSOCAT" >&2; exit 4; }
[ -r "$LEARNED_FILE" ] || { echo "[launch-gl] LEARNED_FILE not readable: $LEARNED_FILE" >&2; exit 4; }
[ "$MODE" = "mock" ] && [ ! -d "$MOCKDIR" ] && {
    echo "[launch-gl] MOCKDIR not readable: $MOCKDIR" >&2; exit 4; }

# Pick inner script
case "$MODE" in
    mock) INNER="${SCRIPT_DIR}/submit_desi_mock_gl.sh" ;;
    loa)  echo "[launch-gl] LOA inner script not implemented yet" >&2; exit 5 ;;
    *) echo "[launch-gl] unknown MODE=$MODE (must be loa|mock)" >&2; exit 2 ;;
esac
[ -f "$INNER" ] || { echo "[launch-gl] inner script missing: $INNER" >&2; exit 4; }

# Loop range
LOOP_START="${START_OVERRIDE:-0}"
LOOP_END="${END_OVERRIDE:-$OUTER_MAX_INDEX}"

# Run output dir + SLURM log dir
if [ "$DRY_RUN" -ne 1 ]; then
    mkdir -p "$OUTDIR" "${OUTDIR}/logs" "${SCRIPT_DIR}/logs"
    # Pin the config + a BASELINE marker into the run dir
    cp "$CONFIG_PATH" "${OUTDIR}/$(basename "$CONFIG_PATH")"
    {
        echo "# Resolved env for run launched $(date)"
        echo "# config: $CONFIG_PATH"
        for var in MODE QSOCAT MOCKDIR OUTDIR LEARNED_FILE CATALOG_NAME LOS_CATALOG \
                   DLA_CATALOG DLA_SAMPLES_FILE SUB_DLA_SAMPLES_FILE NUM_DLA_SAMPLES \
                   NUM_SUBDLA_SAMPLES MAX_DLAS SINGLE_ABSORBER_MODEL FILTER_LOW_LIKELIHOOD \
                   MAX_LAMBDA MIN_LAMBDA DLAMBDA K NUM_FOREST_LINES NUM_LINES BALMASK \
                   PREV_TAU_0 PREV_BETA MAX_NOISE_VARIANCE MAX_WORKERS BATCH_SIZE \
                   ENABLE_TAU_EB TAU_EB_OBJECTIVE EARLY_STOP_MODE; do
            echo "$var=${!var:-(unset)}"
        done
    } > "${OUTDIR}/BASELINE.env"
fi

cat <<EOF
[launch-gl] config:        $CONFIG_PATH
[launch-gl] MODE=$MODE
[launch-gl] OUTDIR=$OUTDIR
[launch-gl] QSOCAT=$QSOCAT
[launch-gl] LEARNED_FILE=$LEARNED_FILE
[launch-gl] inner=$INNER
[launch-gl] outer loop: ${LOOP_START} .. ${LOOP_END}  (step=${OUTER_STEP}, window=${OUTER_WINDOW})
[launch-gl] dry_run=$DRY_RUN  no_sleep=$NO_SLEEP
EOF

COMMON_EXPORT="QSOCAT=${QSOCAT},MOCKDIR=${MOCKDIR},OUTDIR=${OUTDIR},LEARNED_FILE=${LEARNED_FILE},\
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
BALMASK=${BALMASK},RELEASE=${RELEASE},PROGRAM=${PROGRAM},SURVEY=${SURVEY},\
ENABLE_TAU_EB=${ENABLE_TAU_EB:-1},TAU_EB_OBJECTIVE=${TAU_EB_OBJECTIVE:-null},EARLY_STOP_MODE=${EARLY_STOP_MODE:-baseline},\
GL_CONDA_SETUP=${GL_CONDA_SETUP},GL_CONDA_ENV=${GL_CONDA_ENV},GL_LIBCERF_PATH=${GL_LIBCERF_PATH}"

n_jobs=0
for (( i=LOOP_START; i<=LOOP_END; i+=OUTER_STEP )); do
    chunk_end=$(( i + OUTER_WINDOW ))
    window_export="START_INDEX=${i},END_INDEX=${chunk_end},STEP=2"
    full_export="ALL,${COMMON_EXPORT},${window_export}"

    cmd=(sbatch --chdir="$REPO_ROOT" \
                --account="$GL_SLURM_ACCOUNT" \
                --partition="$GL_SLURM_PARTITION" \
                --time="$GL_SLURM_TIME" \
                --export="$full_export" "$INNER")

    echo "[launch-gl] $(date +%H:%M:%S) chunk ${i}..${chunk_end}"
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  %q ' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi

    n_jobs=$(( n_jobs + 1 ))
    [ "$NO_SLEEP" -ne 1 ] && [ "$DRY_RUN" -ne 1 ] && [ "$i" -lt "$LOOP_END" ] && sleep 60
done
echo "[launch-gl] submitted $n_jobs sbatch job(s)"
