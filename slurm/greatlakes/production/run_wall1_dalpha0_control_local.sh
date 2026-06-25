#!/usr/bin/env bash
# run_wall1_dalpha0_control_local.sh
#
# WALL-1 closure UNTILTED CONTROL arm (Δα=0.0), run LOCALLY in the background.
#
# WHY: the full-injection ±0.5 arms (run_diagnostics_local.sh steps 3/4) both
# under-recover R0 (≈0.16–0.68) — a COMMON-MODE deficit from the injected
# one-absorber-per-clean-sightline substrate's normalization vs the frozen
# loa-124 molly/R_emp calibration. The reweighting WALL-1 DIVIDED OUT its Δα=0
# baseline; the injection arms had no Δα=0 control. This arm IS that control:
# the matched-substrate Δα=0 baseline R0(0), so the closure becomes
# R0(+0.5)/R0(0) vs R0(−0.5)/R0(0), directly comparable to the reweighting's
# ±4–9 opposite-sign pulls.
#
# This is a config-only single-arm clone of run_diagnostics_local.sh step 3,
# with Δα swapped 0.5 -> 0.0 and the arm dir -> full_dalpha0.0. EVERY inference
# parameter is inherited UNCHANGED via wall1_inject_gl_v1.env (the EXACT
# production config the frozen kernel + molly C/ρ were calibrated on). The GP
# path (dla_gp.py / run_bayes_select.py) is byte-untouched. Only the data
# pointer (--dalpha 0.0, arm dir) + --max_workers + --outdir vary.
#
# Usage (intended to be launched as a nohup background process):
#   bash slurm/greatlakes/production/run_wall1_dalpha0_control_local.sh
#   N_WORKERS=8 bash .../run_wall1_dalpha0_control_local.sh

set -uo pipefail   # NOT -e — log per-step failures but report cleanly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

GL_SCRATCH="/scratch/cavestru_root/cavestru0/mfho"
N_WORKERS="${N_WORKERS:-8}"        # mirror the ±0.5 arms (8 workers, BLAS pinned)
N_INJ="${N_INJ:-2000}"             # task: --n_inj 2000
N_HEALPIX="${N_HEALPIX:-0}"        # 0 = all clean loa-0 healpix (high-z fill)

DALPHA="0.0"
WALL1_ARM_DIR="${GL_SCRATCH}/wall1_inject/full_dalpha0.0"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${GL_SCRATCH}/run_wall1_dalpha0_control_${TS}"
mkdir -p "$LOG_ROOT"
DRIVER_LOG="${LOG_ROOT}/driver.log"
GEN_LOG="${LOG_ROOT}/gen_dalpha0.log"
GP_LOG="${LOG_ROOT}/gp_dalpha0.log"

# --- Environment (conda + libcerf + BLAS pinning), identical to the ±0.5 arms.
GL_CONDA_SETUP="${GL_CONDA_SETUP:-/sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh}"
GL_CONDA_ENV="${GL_CONDA_ENV:-gpdla}"
GL_LIBCERF_PATH="${GL_LIBCERF_PATH:-$HOME/.local/usr/local/lib64}"
# shellcheck disable=SC1090
source "$GL_CONDA_SETUP"
conda activate "$GL_CONDA_ENV"
export LD_LIBRARY_PATH="${GL_LIBCERF_PATH}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }

log "############################################################"
log "# run_wall1_dalpha0_control_local.sh  ts=${TS}"
log "# node=$(hostname)  nproc=$(nproc)  load='$(cut -d' ' -f1-3 /proc/loadavg)'"
log "# N_WORKERS=${N_WORKERS}  N_INJ=${N_INJ}  N_HEALPIX=${N_HEALPIX}  DALPHA=${DALPHA}"
log "# repo=${REPO_ROOT}  commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "# arm=${WALL1_ARM_DIR}"
log "# LOG_ROOT=${LOG_ROOT}"
log "############################################################"

# ---------------------------------------------------------------------------
# STEP A: generate the Δα=0.0 injectable arm tree (if missing).
#   Mirrors run_diagnostics_local.sh::_gen_wall1_arm verbatim (swap dalpha->0.0).
# ---------------------------------------------------------------------------
if [ -d "${WALL1_ARM_DIR}/spectra-16" ] && \
   [ -n "$(find "${WALL1_ARM_DIR}/spectra-16" -name 'spectra-16-*.fits' 2>/dev/null | head -1)" ] && \
   [ -r "${WALL1_ARM_DIR}/pilot_qsocat.fits" ]; then
    log "STEP A (gen): arm tree present — reusing ${WALL1_ARM_DIR}/spectra-16"
else
    log "STEP A (gen): generating arm Δα=${DALPHA} n_inj=${N_INJ} n_healpix=${N_HEALPIX} -> ${WALL1_ARM_DIR}"
    python injection/gen_wall1_inject.py \
        --out "$WALL1_ARM_DIR" --dalpha "$DALPHA" --n_inj "$N_INJ" --n_healpix "$N_HEALPIX" \
        2>&1 | tee -a "$GEN_LOG"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        log "STEP A FAILED rc=${rc} (see ${GEN_LOG}); aborting."
        exit "$rc"
    fi
fi

# ---------------------------------------------------------------------------
# STEP B: production-config GP over the arm (mirrors run_diagnostics_local.sh
#   step_wall1_plus: export WALL1_ARM, source wall1_inject_gl_v1.env, run_gp over
#   [0, n_files)). Inference arg set is IDENTICAL to submit_desi_mock_gl.sh.
# ---------------------------------------------------------------------------
export WALL1_ARM="$WALL1_ARM_DIR"
# shellcheck disable=SC1090
source "${SCRIPT_DIR}/wall1_inject_gl_v1.env"   # sets MOCKDIR/QSOCAT/OUTDIR from WALL1_ARM

N_FILES=$(find "${WALL1_ARM}/spectra-16" -name 'spectra-16-*.fits' 2>/dev/null | wc -l)
log "STEP B (GP): arm has ${N_FILES} healpix files; OUTDIR=${OUTDIR}"
mkdir -p "$OUTDIR" "${OUTDIR}/logs" "${OUTDIR}/figures"

# Optional external TID list (unused for injection arms, kept for parity).
tid_arg=()
if [ -n "${EXTERNAL_TID_LIST:-}" ]; then
    tid_arg=(--external_tid_list "$EXTERNAL_TID_LIST")
fi

t0=$(date +%s)
python desi-DLAGP.py \
    --qsocat "$QSOCAT" \
    --release "$RELEASE" \
    --program "${PROGRAM:-dark}" \
    --survey "${SURVEY:-main}" \
    --mocks \
    --mockdir "$MOCKDIR" \
    $(if [ "${BALMASK:-false}" = "true" ]; then echo "--balmask"; fi) \
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
    --plot_figures 0 \
    --filter_low_likelihood "$FILTER_LOW_LIKELIHOOD" \
    --single_absorber_model "$SINGLE_ABSORBER_MODEL" \
    --max_workers "$N_WORKERS" \
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
    --pair_prior_mode "${PAIR_PRIOR_MODE:-off}" \
    --dla_bias "${DLA_BIAS:-2.0}" \
    $([ -n "${FILTER_N_INITIAL_FLOOR:-}" ] && echo "--filter_n_initial_floor ${FILTER_N_INITIAL_FLOOR}") \
    $([ "${FILTER_EMPTY_MASK_FALLTHROUGH:-0}" = "1" ] && echo "--filter_empty_mask_fallthrough 1") \
    "${tid_arg[@]}" \
    --figure_dir "${OUTDIR}/figures" \
    --level2_start 0 \
    --level2_end "$N_FILES" 2>&1 | tee -a "$GP_LOG"
rc=${PIPESTATUS[0]}
dt=$(( $(date +%s) - t0 ))

n_h5=$(find "$OUTDIR" -name 'processed-*.h5' 2>/dev/null | wc -l)
log "############################################################"
log "# FINAL SUMMARY (ts=${TS})  rc=${rc}  ${dt}s  processed-h5=${n_h5}"
log "#   arm tree: ${WALL1_ARM_DIR}/spectra-16  (${N_FILES} healpix)"
log "#   gp_out:   ${OUTDIR}"
log "#   truth:    ${WALL1_ARM_DIR}/injected_truth_cat.fits"
log "#   gen log:  ${GEN_LOG}"
log "#   gp log:   ${GP_LOG}"
log "# REDUCE (run after this lands):"
log "#   python CDDF_analysis/diagnostics/wall1/wall1_full_injection.py \\"
log "#     --arm ${WALL1_ARM_DIR} --label dalpha0.0 --n-mc 200 \\"
log "#     --out ${GL_SCRATCH}/wall1_inject/reduce_out"
log "# DONE."
log "############################################################"
exit "$rc"
