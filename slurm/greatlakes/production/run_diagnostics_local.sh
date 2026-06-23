#!/usr/bin/env bash
# slurm/greatlakes/production/run_diagnostics_local.sh
#
# LOCAL (non-SLURM) background driver that runs the 4 GP re-inference
# diagnostics SEQUENTIALLY on a single GreatLakes node. Built + smoke-validated
# 2026-06-18 because the SLURM queue was jammed and the PI asked to run the
# diagnostics in the background locally. Re-measured cost is tiny (~5.9 core-s
# /spec): falsifier ~3 core-h + WALL-1 ~13 core-h => ~16 core-h ≈ ~2 wall-h at
# 8 workers.
#
# CRITICAL — production inference faithfulness:
#   All 4 runs use the SAME production config the frozen kernel + molly P/C were
#   calibrated on. Each step SOURCES the exact production .env that submit_desi_
#   mock_gl.sh / launch_gl.sh use, then calls desi-DLAGP.py DIRECTLY with the
#   identical argument set as submit_desi_mock_gl.sh (only --max_workers, the
#   level2 window, and --outdir differ — never an inference parameter). The GP
#   path (dla_gp.py / run_bayes_select.py) is byte-untouched. Production config:
#   MAX_DLAS=4, SINGLE_ABSORBER_MODEL=1, FILTER_LOW_LIKELIHOOD=1,
#   NUM_FOREST_LINES=31, MAX_LAMBDA=1250, DLAMBDA=0.15, K=30, τ₀=0.00246, β=3.62,
#   model 2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5, PW 100k samples.
#
# The 4 diagnostics (run in order; one failing step is logged + does NOT abort
# the rest):
#   1. Falsifier Arm A — τ-EB OFF      (2lpt0_highz_taueb_off.env,  903-TID subset)
#   2. Falsifier Arm B — τ-EB dla obj  (2lpt0_highz_taueb_dla.env,  903-TID subset)
#   3. WALL-1 inject Arm +0.5          (gen_wall1_inject Δα=+0.5 -> production GP)
#   4. WALL-1 inject Arm −0.5          (gen_wall1_inject Δα=−0.5 -> production GP)
#
# Usage (foreground for testing — but intended to be launched as a tracked
# background process by the PI):
#   bash slurm/greatlakes/production/run_diagnostics_local.sh
#   N_WORKERS=8 bash slurm/greatlakes/production/run_diagnostics_local.sh
#   N_INJ=4000 bash slurm/greatlakes/production/run_diagnostics_local.sh
#
# Idempotent-ish: a step is SKIPPED if its expected processed-h5 output already
# exists (so a re-launch resumes at the first incomplete step). Delete the
# step's output dir to force a re-run.
#
# Monitor:
#   tail -f <LOG_ROOT>/run_diagnostics_local_<ts>/step{1,2,3,4}_*.log
#   The driver also writes a SUMMARY line per step + a final summary block.

set -uo pipefail   # NOTE: NOT -e — we trap per-step failures and continue.

# ---------------------------------------------------------------------------
# Paths + knobs
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
FALS_DIR="${SCRIPT_DIR}/tau_eb_falsifier"

GL_SCRATCH="/scratch/cavestru_root/cavestru0/mfho"
N_WORKERS="${N_WORKERS:-8}"        # 8 workers (node load already ~13 of 24)
N_INJ="${N_INJ:-4000}"             # WALL-1 injections per arm
N_HEALPIX="${N_HEALPIX:-0}"        # 0 = all clean healpix (need high-z fill)

# Falsifier full-run level2 window. The 903 high-z host TIDs span sorted level2
# index [0,186) (per 2lpt0_highz_taueb_off.env). The per-file TID intersection
# in dlasearch_mock skips empty-overlap files fast, so this is the host span.
FALS_LEVEL2_START="${FALS_LEVEL2_START:-0}"
FALS_LEVEL2_END="${FALS_LEVEL2_END:-186}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${GL_SCRATCH}/run_diagnostics_local_${TS}"
mkdir -p "$LOG_ROOT"
DRIVER_LOG="${LOG_ROOT}/driver.log"

# ---------------------------------------------------------------------------
# Environment (conda + libcerf + BLAS pinning) — once, inherited by every step.
# ---------------------------------------------------------------------------
GL_CONDA_SETUP="${GL_CONDA_SETUP:-/sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh}"
GL_CONDA_ENV="${GL_CONDA_ENV:-gpdla}"
GL_LIBCERF_PATH="${GL_LIBCERF_PATH:-$HOME/.local/usr/local/lib64}"
# shellcheck disable=SC1090
source "$GL_CONDA_SETUP"
conda activate "$GL_CONDA_ENV"
export LD_LIBRARY_PATH="${GL_LIBCERF_PATH}:${LD_LIBRARY_PATH:-}"
# Pin BLAS to 1 thread per worker (config-only; see submit_desi_mock_gl.sh:44).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }

# ---------------------------------------------------------------------------
# Core: invoke desi-DLAGP.py with the EXACT production arg set (mirrors
# submit_desi_mock_gl.sh:154-201). All inference knobs come from the sourced
# .env. Only OUTDIR / level2 window / max_workers / external_tid_list vary.
#   $1 = OUTDIR
#   $2 = level2_start
#   $3 = level2_end
#   $4 = step log file
# (Assumes the caller has sourced the flavour .env so the knob vars are set.)
# ---------------------------------------------------------------------------
run_gp() {
    local outdir="$1" l2s="$2" l2e="$3" steplog="$4"
    mkdir -p "$outdir" "${outdir}/logs"

    # Optional external TID list (falsifier) — only forwarded if set+nonempty.
    local tid_arg=()
    if [ -n "${EXTERNAL_TID_LIST:-}" ]; then
        tid_arg=(--external_tid_list "$EXTERNAL_TID_LIST")
    fi

    python desi-DLAGP.py \
        --qsocat "$QSOCAT" \
        --release "$RELEASE" \
        --program "${PROGRAM:-dark}" \
        --survey "${SURVEY:-main}" \
        --mocks \
        --mockdir "$MOCKDIR" \
        $(if [ "${BALMASK:-false}" = "true" ]; then echo "--balmask"; fi) \
        --outdir "$outdir" \
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
        --figure_dir "${outdir}/figures" \
        --level2_start "$l2s" \
        --level2_end "$l2e" 2>&1 | tee -a "$steplog"
    return "${PIPESTATUS[0]}"
}

# True if at least one processed-h5 exists under OUTDIR (our "done-ish" marker).
has_output() { [ -n "$(find "$1" -name 'processed-*.h5' 2>/dev/null | head -1)" ]; }

# Run one step in a SUBSHELL so a sourced .env / pipefail / exit can't leak
# across steps. Logs result; never aborts the driver.
declare -A STEP_RESULT
run_step() {
    local name="$1" steplog="$2" outdir="$3"; shift 3
    local fn="$1"   # name of the body function to invoke
    log "=== STEP: ${name} ==="
    log "    log:    ${steplog}"
    log "    outdir: ${outdir}"
    if has_output "$outdir"; then
        log "    SKIP — processed-h5 already present in ${outdir} (delete to force re-run)."
        STEP_RESULT["$name"]="SKIPPED"
        return 0
    fi
    local t0 rc; t0=$(date +%s)
    ( set -o pipefail; "$fn" "$steplog" "$outdir" )
    rc=$?
    local dt=$(( $(date +%s) - t0 ))
    if [ "$rc" -eq 0 ] && has_output "$outdir"; then
        local n; n=$(find "$outdir" -name 'processed-*.h5' 2>/dev/null | wc -l)
        log "    SUMMARY: ${name} OK  rc=0  ${dt}s  processed-h5=${n}"
        STEP_RESULT["$name"]="OK(${n}h5,${dt}s)"
    else
        log "    SUMMARY: ${name} FAILED  rc=${rc}  ${dt}s  (see ${steplog}); continuing."
        STEP_RESULT["$name"]="FAILED(rc=${rc})"
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Step bodies. Each sources its production .env in the subshell, then runs.
# ---------------------------------------------------------------------------

# 1 + 2: falsifier arms (same mechanism; env differs).
FALS_OFF_OUT="${GL_SCRATCH}/gl_taueb_falsifier_2lpt0_highz_OFF_local/outputs/"
FALS_DLA_OUT="${GL_SCRATCH}/gl_taueb_falsifier_2lpt0_highz_DLAOBJ_local/outputs/"

step_falsifier_off() {  # $1=steplog $2=outdir
    source "${FALS_DIR}/2lpt0_highz_taueb_off.env"
    run_gp "$2" "$FALS_LEVEL2_START" "$FALS_LEVEL2_END" "$1"
}
step_falsifier_dla() {
    source "${FALS_DIR}/2lpt0_highz_taueb_dla.env"
    run_gp "$2" "$FALS_LEVEL2_START" "$FALS_LEVEL2_END" "$1"
}

# 3 + 4: WALL-1 injection arms. Generate the arm tree (if missing), then run the
# production GP via wall1_inject_gl_v1.env (WALL1_ARM points at the tree).
WALL1_PLUS_ARM="${GL_SCRATCH}/wall1_inject/full_dalpha+0.5"
WALL1_MINUS_ARM="${GL_SCRATCH}/wall1_inject/full_dalpha-0.5"

# gen_wall1_inject.py writes the arm with a "0.5" / "-0.5" --dalpha; the launch
# script names the dir full_dalpha0.5 / full_dalpha-0.5. We mirror that here but
# the task asked for full_dalpha+0.5; we honor the task's +0.5 spelling for the
# plus arm and -0.5 for the minus arm. (gen_wall1_inject only cares about --out.)

_gen_wall1_arm() {  # $1=arm_dir $2=dalpha $3=steplog
    local arm="$1" dalpha="$2" steplog="$3"
    if [ -d "${arm}/spectra-16" ] && \
       [ -n "$(find "${arm}/spectra-16" -name 'spectra-16-*.fits' 2>/dev/null | head -1)" ] && \
       [ -r "${arm}/pilot_qsocat.fits" ]; then
        echo "[wall1] arm tree present — reusing ${arm}/spectra-16" | tee -a "$steplog"
        return 0
    fi
    echo "[wall1] generating arm Δα=${dalpha} n_inj=${N_INJ} n_healpix=${N_HEALPIX} -> ${arm}" | tee -a "$steplog"
    python injection/gen_wall1_inject.py \
        --out "$arm" --dalpha "$dalpha" --n_inj "$N_INJ" --n_healpix "$N_HEALPIX" 2>&1 | tee -a "$steplog"
    return "${PIPESTATUS[0]}"
}

step_wall1_plus() {   # $1=steplog $2=outdir (== ${arm}/gp_out)
    _gen_wall1_arm "$WALL1_PLUS_ARM" "0.5" "$1" || return $?
    export WALL1_ARM="$WALL1_PLUS_ARM"
    source "${SCRIPT_DIR}/wall1_inject_gl_v1.env"   # sets MOCKDIR/QSOCAT/OUTDIR from WALL1_ARM
    local n_files; n_files=$(find "${WALL1_ARM}/spectra-16" -name 'spectra-16-*.fits' 2>/dev/null | wc -l)
    echo "[wall1+] arm has ${n_files} healpix files; OUTDIR=${OUTDIR}" | tee -a "$1"
    run_gp "$OUTDIR" 0 "$n_files" "$1"
}
step_wall1_minus() {
    _gen_wall1_arm "$WALL1_MINUS_ARM" "-0.5" "$1" || return $?
    export WALL1_ARM="$WALL1_MINUS_ARM"
    source "${SCRIPT_DIR}/wall1_inject_gl_v1.env"
    local n_files; n_files=$(find "${WALL1_ARM}/spectra-16" -name 'spectra-16-*.fits' 2>/dev/null | wc -l)
    echo "[wall1-] arm has ${n_files} healpix files; OUTDIR=${OUTDIR}" | tee -a "$1"
    run_gp "$OUTDIR" 0 "$n_files" "$1"
}

# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------
log "############################################################"
log "# run_diagnostics_local.sh  ts=${TS}"
log "# node=$(hostname)  nproc=$(nproc)  load='$(cut -d' ' -f1-3 /proc/loadavg)'"
log "# N_WORKERS=${N_WORKERS}  N_INJ=${N_INJ}  N_HEALPIX=${N_HEALPIX}"
log "# repo=${REPO_ROOT}  commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "# BLAS pinned to 1 thread/worker; conda=${GL_CONDA_ENV}"
log "# LOG_ROOT=${LOG_ROOT}"
log "############################################################"

run_step "1_falsifier_OFF"    "${LOG_ROOT}/step1_falsifier_off.log"   "$FALS_OFF_OUT"                step_falsifier_off
run_step "2_falsifier_DLAOBJ" "${LOG_ROOT}/step2_falsifier_dlaobj.log" "$FALS_DLA_OUT"               step_falsifier_dla
run_step "3_wall1_inject_+0.5" "${LOG_ROOT}/step3_wall1_plus.log"      "${WALL1_PLUS_ARM}/gp_out"    step_wall1_plus
run_step "4_wall1_inject_-0.5" "${LOG_ROOT}/step4_wall1_minus.log"     "${WALL1_MINUS_ARM}/gp_out"   step_wall1_minus

log "############################################################"
log "# FINAL SUMMARY (ts=${TS})"
for s in "1_falsifier_OFF" "2_falsifier_DLAOBJ" "3_wall1_inject_+0.5" "4_wall1_inject_-0.5"; do
    log "#   ${s}: ${STEP_RESULT[$s]:-NOT_RUN}"
done
log "# outputs:"
log "#   ${FALS_OFF_OUT}"
log "#   ${FALS_DLA_OUT}"
log "#   ${WALL1_PLUS_ARM}/gp_out"
log "#   ${WALL1_MINUS_ARM}/gp_out"
log "# ALL DIAGNOSTICS DONE."
log "############################################################"
