#!/usr/bin/env bash
# slurm/greatlakes/production/launch_gl_resume.sh
#
# RESUME launcher: complete a partially-done mock production run by recomputing
# ONLY the not-done healpix (absent or truncated), with zero recompute of the
# files that are already valid.
#
# It sources the same flavour .env as launch_gl.sh, derives the not-done
# speclist positions from the run's processed/ dir via resume_positions.py
# (done iff the h5 opens with the core datasets; --require-gzip for the stricter
# post-repack marker), chunks them, and submits submit_desi_mock_gl_resume.sh
# (list-driven, 1-file slices) — one sbatch per chunk, N=2 × W=16 packing.
#
# Usage:
#   bash launch_gl_resume.sh london0_gl_v1.env --outdir <RUN>/outputs [--chunk 10]
#   bash launch_gl_resume.sh london0_gl_v1.env --outdir <RUN>/outputs --dry-run
#   bash launch_gl_resume.sh london0_gl_v1.env --outdir <RUN>/outputs --require-gzip
#
# The flavour .env's RUN_NAME default points at a NEW dir; pass --outdir to
# target the EXISTING (incomplete) run's outputs/ so the resume lands beside the
# files already there.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYBIN="${PYBIN:-/home/mfho/.conda/envs/gpdla/bin/python}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <flavour.env> --outdir <RUN>/outputs [--chunk N] [--require-gzip] [--dry-run]" >&2
    exit 2
fi

CONFIG_PATH="$1"; shift
OUTDIR_OVERRIDE=""; CHUNK=10; DRY_RUN=0; REQUIRE_GZIP=0; LIMIT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --outdir)       OUTDIR_OVERRIDE="$2"; shift 2 ;;
        --chunk)        CHUNK="$2"; shift 2 ;;
        --limit)        LIMIT="$2"; shift 2 ;;   # submit at most N sbatch jobs (0 = all); for canary runs
        --require-gzip) REQUIRE_GZIP=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        *) echo "[resume] unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Resolve config
if [ ! -f "$CONFIG_PATH" ]; then
    if [ -f "${SCRIPT_DIR}/${CONFIG_PATH}" ]; then CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_PATH}"
    else echo "[resume] config not found: $CONFIG_PATH" >&2; exit 2; fi
fi
# shellcheck disable=SC1090
source "$CONFIG_PATH"

[ -n "$OUTDIR_OVERRIDE" ] && OUTDIR="$OUTDIR_OVERRIDE"

for var in MODE QSOCAT OUTDIR LEARNED_FILE MOCKDIR; do
    [ -z "${!var:-}" ] && { echo "[resume] config missing required var: $var" >&2; exit 2; }
done
[ "$MODE" = "mock" ] || { echo "[resume] only MODE=mock supported" >&2; exit 2; }

# Enforce GL allowed-write roots (same as launch_gl.sh)
ALLOWED_PREFIXES=("${GL_ALLOWED_OUTPUT_PREFIXES[@]:-/scratch/cavestru_root/cavestru0/mfho/ /nfs/turbo/lsa-cavestru/mfho/}")
abs_outdir="$(readlink -m "$OUTDIR")"
ok=0
for p in "${ALLOWED_PREFIXES[@]}"; do
    case "$abs_outdir/" in "$p"*) ok=1; break ;; esac
done
[ "$ok" -ne 1 ] && { echo "[resume] REFUSING: OUTDIR=$abs_outdir outside allowed GL write roots" >&2; exit 3; }
OUTDIR="$abs_outdir"

PROCDIR="${OUTDIR}/figures/processed"
[ -d "$PROCDIR" ] || { echo "[resume] processed dir not found: $PROCDIR" >&2; exit 4; }
[ -r "$QSOCAT" ]       || { echo "[resume] QSOCAT not readable: $QSOCAT" >&2; exit 4; }
[ -r "$LEARNED_FILE" ] || { echo "[resume] LEARNED_FILE not readable: $LEARNED_FILE" >&2; exit 4; }
[ -d "$MOCKDIR" ]      || { echo "[resume] MOCKDIR not readable: $MOCKDIR" >&2; exit 4; }

INNER="${SCRIPT_DIR}/submit_desi_mock_gl_resume.sh"
[ -f "$INNER" ] || { echo "[resume] inner script missing: $INNER" >&2; exit 4; }

# --- Derive not-done positions ----------------------------------------------
gz_flag=""; [ "$REQUIRE_GZIP" -eq 1 ] && gz_flag="--require-gzip"
mapfile -t POSITIONS < <("$PYBIN" "${SCRIPT_DIR}/resume_positions.py" \
    --mockdir "$MOCKDIR" --procdir "$PROCDIR" --summary $gz_flag)
NP=${#POSITIONS[@]}

echo "[resume] config:   $CONFIG_PATH"
echo "[resume] OUTDIR=$OUTDIR"
echo "[resume] PROCDIR=$PROCDIR"
echo "[resume] not-done positions: $NP   chunk=$CHUNK   require_gzip=$REQUIRE_GZIP"
if [ "$NP" -eq 0 ]; then echo "[resume] nothing to do — run is complete."; exit 0; fi

# --- COMMON_EXPORT (MUST mirror launch_gl.sh; only LEVEL2_LIST differs) -------
FIGURE_DIR="${FIGURE_DIR:-${OUTDIR}/figures}"
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
FIGURE_DIR=${FIGURE_DIR},\
GL_CONDA_SETUP=${GL_CONDA_SETUP},GL_CONDA_ENV=${GL_CONDA_ENV},GL_LIBCERF_PATH=${GL_LIBCERF_PATH}"

[ "$DRY_RUN" -ne 1 ] && mkdir -p "$OUTDIR" "${OUTDIR}/logs" "${SCRIPT_DIR}/logs"

# --- Chunk + submit ----------------------------------------------------------
n_jobs=0
for (( off=0; off<NP; off+=CHUNK )); do
    chunk_arr=("${POSITIONS[@]:off:CHUNK}")
    # Join with ':' NOT ',' — sbatch --export uses comma to separate KEY=VALUE
    # pairs, so a comma-joined list would be silently truncated to its first
    # element (the canary bug: LEVEL2_LIST=13,49 => job saw only 13).
    LEVEL2_LIST=$(IFS=:; echo "${chunk_arr[*]}")
    full_export="ALL,${COMMON_EXPORT},LEVEL2_LIST=${LEVEL2_LIST}"

    cmd=(sbatch --chdir="$REPO_ROOT" \
                --account="${GL_SLURM_ACCOUNT:-cavestru0}" \
                --partition="${GL_SLURM_PARTITION:-standard}" \
                --time="${GL_SLURM_TIME:-08:00:00}" \
                --mem="${GL_SLURM_MEM:-64G}" \
                --export="$full_export" "$INNER")

    echo "[resume] $(date +%H:%M:%S) chunk $((n_jobs+1)): positions ${LEVEL2_LIST}"
    if [ "$DRY_RUN" -eq 1 ]; then printf '  %q ' "${cmd[@]}"; echo
    else "${cmd[@]}"; fi
    n_jobs=$((n_jobs+1))
    if [ "$LIMIT" -gt 0 ] && [ "$n_jobs" -ge "$LIMIT" ]; then
        echo "[resume] --limit $LIMIT reached, stopping (idempotent: re-run to submit the rest)"
        break
    fi
    [ "$DRY_RUN" -ne 1 ] && [ $((off+CHUNK)) -lt "$NP" ] && sleep 30
done
echo "[resume] submitted $n_jobs sbatch job(s) covering $NP positions"
