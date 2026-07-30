#!/usr/bin/env bash
# arm2_fitting_window_pilot.sh — ARM 2 of the matched spectral-window study.
#
# Runs the FINDER twice on the SAME small set of spectra, at two values of the
# FITTING window's blue edge (`Parameters.min_lambda`), everything else
# byte-identical to the 2LPT-0 V1 production config. This is the ONLY arm that
# can test the PI's actual mechanism (blue-edge truncation inside the GP fit);
# the ANALYSIS-window arm is a post-hoc selection and cannot.
#
#   MIN_LAMBDA = 911.75  -> production (Lya + Lyb modelled)
#   MIN_LAMBDA = 1025.0  -> the controlled BLUE-END CUT under test
#
# INTERACTIVE + BOUNDED ON PURPOSE. A full fitting-window campaign is NOT
# authorized (project rule: > ~500 CPU-h needs PI sign-off). This script exists
# to MEASURE the per-spectrum cost so the campaign can be COSTED, and to give a
# first directional pointer on large-DLA NHI recovery. Tens of spectra is a
# POINTER, NOT A MEASUREMENT.
#
# Usage:
#   bash slurm/greatlakes/production/arm2_fitting_window_pilot.sh <MIN_LAMBDA> <TAG> <TIDLIST>
set -eo pipefail

MIN_LAMBDA_OVERRIDE="$1"      # 911.75 | 1025.0
TAG="$2"                      # e.g. lam911p75
TIDLIST="$3"                  # text file of TARGETIDs, one per line

REPO="${REPO:-/home/mfho/wt_win}"
ARM2="${ARM2:-/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/window_study/arm2}"

# the 2LPT-0 V1 production knobs, sourced (not retyped) from the committed env
GL_REPO="$REPO"
# shellcheck disable=SC1090
source "${REPO}/slurm/greatlakes/production/2lpt0_gl_v1.env"

# knobs the GL mock submitter defaults rather than the env (matches
# submit_desi_mock_nersc.sh:76 / submit_desi_loa_nersc.sh:72-73 and the
# desi-DLAGP.py argparse defaults) — set explicitly so the two pilot arms are
# provably identical in everything but MIN_LAMBDA
PAIR_PRIOR_MODE="${PAIR_PRIOR_MODE:-off}"
DLA_BIAS="${DLA_BIAS:-2.0}"

# the ONE knob under test
MIN_LAMBDA="$MIN_LAMBDA_OVERRIDE"

OUTDIR="${ARM2}/run_${TAG}"
mkdir -p "${OUTDIR}/logs"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="${HOME}/.local/usr/local/lib64:${LD_LIBRARY_PATH}"

# LEVEL2 slice: the pilot's TARGETIDs all live in ONE spectra-16 file
LEVEL2_START="${LEVEL2_START:-18}"
LEVEL2_END="${LEVEL2_END:-19}"

echo "[arm2] MIN_LAMBDA=${MIN_LAMBDA} MAX_LAMBDA=${MAX_LAMBDA} tag=${TAG}"
echo "[arm2] tids=$(wc -l < "$TIDLIST")  level2=[${LEVEL2_START},${LEVEL2_END})"
echo "[arm2] outdir=${OUTDIR}"

cd "$REPO"
/usr/bin/time -v /home/mfho/.conda/envs/gpdla/bin/python desi-DLAGP.py \
    --qsocat "$QSOCAT" --release "$RELEASE" --program "$PROGRAM" \
    --survey "$SURVEY" --mocks --mockdir "$MOCKDIR" \
    --outdir "$OUTDIR" \
    --learned_file "$LEARNED_FILE" \
    --catalog_name "$CATALOG_NAME" \
    --los_catalog "$LOS_CATALOG" --dla_catalog "$DLA_CATALOG" \
    --dla_samples_file "$DLA_SAMPLES_FILE" \
    --sub_dla_samples_file "$SUB_DLA_SAMPLES_FILE" \
    --min_z_separation "$MIN_Z_SEPARATION" \
    --prev_tau_0 "$PREV_TAU_0" --prev_beta "$PREV_BETA" \
    --max_dlas "$MAX_DLAS" --plot_figures 0 \
    --filter_low_likelihood "$FILTER_LOW_LIKELIHOOD" \
    --single_absorber_model "$SINGLE_ABSORBER_MODEL" \
    --max_workers 1 --batch_size "$BATCH_SIZE" \
    --loading_min_lambda "$LOADING_MIN_LAMBDA" \
    --loading_max_lambda "$LOADING_MAX_LAMBDA" \
    --normalization_min_lambda "$NORMALIZATION_MIN_LAMBDA" \
    --normalization_max_lambda "$NORMALIZATION_MAX_LAMBDA" \
    --min_lambda "$MIN_LAMBDA" \
    --max_lambda "$MAX_LAMBDA" \
    --dlambda "$DLAMBDA" --k "$K" \
    --num_dla_samples "$NUM_DLA_SAMPLES" \
    --num_subdla_samples "$NUM_SUBDLA_SAMPLES" \
    --max_noise_variance "$MAX_NOISE_VARIANCE" \
    --num_forest_lines "$NUM_FOREST_LINES" --num_lines "$NUM_LINES" \
    --enable_tau_eb "$ENABLE_TAU_EB" \
    --tau_eb_objective "$TAU_EB_OBJECTIVE" \
    --early_stop_mode "$EARLY_STOP_MODE" \
    --pair_prior_mode "$PAIR_PRIOR_MODE" \
    --dla_bias "$DLA_BIAS" \
    --external_tid_list "$TIDLIST" \
    --figure_dir "${OUTDIR}/figures" \
    --level2_start "$LEVEL2_START" --level2_end "$LEVEL2_END" \
    2>&1 | tee "${OUTDIR}/logs/pilot_${TAG}.log"

echo "[arm2] done ${TAG}"
