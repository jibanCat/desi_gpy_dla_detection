#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --job-name=e2e_train_loa
#SBATCH --output=slurm/train/e2e_train_loa_%j.log
#SBATCH --error=slurm/train/e2e_train_loa_%j.err
#SBATCH -A desi
#SBATCH --time=24:00:00
#SBATCH --gpus=1

# End-to-end NERSC submit: PRELOAD real LOA data + TRAIN v2 GP, in one job.
#
# ============================================================
# EXPLICIT FILTER PIPELINE (applied INSIDE preload_loa_real.py):
#
#   filter 0: full altbal QSO catalog                  (~2.7M rows)
#   filter 1: z ∈ [Z_MIN, Z_MAX] AND ZWARN==0          (always on)
#   filter 2: BAL anti-join — drop BI_CIV > 0          (only if --exclude-bal)
#   filter 3: HCD anti-join — drop TARGETIDs whose
#             max(MAP_log_nhis) ≥ HCD_MIN_NHI          (only if --hcd-cat)
#   filter 4: random cap to MAX_SPECTRA                (if needed)
#
# The HCD anti-join uses an EXTERNAL catalog (the user's previous
# combined.h5 from the GP-DLA pipeline by default; can be overridden
# with --hcd-cat). It reads the per-spectrum log-NHI MAP across DLA
# slots and excludes TARGETIDs whose ANY slot is above HCD_MIN_NHI.
#
# A TARGETID NOT present in --hcd-cat is KEPT — so the HCD catalog
# determines coverage. For LOA real data, only catalog-detected
# absorbers are filtered; uncatalogued LLS slip through.
# ============================================================
#
# Three named VARIANTs:
#
# ┌──────────────────┬──────────────┬──────────────┬─────────────────────────────────┐
# │ VARIANT          │ HCD_MIN_NHI  │ exclude_bal  │ what stays in the training set  │
# ├──────────────────┼──────────────┼──────────────┼─────────────────────────────────┤
# │ no_dla_no_bal    │ 20.3 (DLAs)  │ YES          │ no DLAs, no BALs;               │
# │ (legacy)         │              │              │ sub-DLAs + LLS are KEPT         │
# ├──────────────────┼──────────────┼──────────────┼─────────────────────────────────┤
# │ no_hcd_with_bal  │ 17.2 (any    │ NO           │ no HCDs (DLA+sub-DLA+LLS);      │
# │ ("BAL model")    │      HCD)    │              │ BALs KEPT — model learns BAL   │
# │                  │              │              │ continuum / can be applied at   │
# │                  │              │              │ inference to BAL targets        │
# ├──────────────────┼──────────────┼──────────────┼─────────────────────────────────┤
# │ no_hcd_no_bal    │ 17.2 (any    │ YES          │ no HCDs, no BALs;               │
# │ (clean baseline) │      HCD)    │              │ cleanest possible LOA sample    │
# └──────────────────┴──────────────┴──────────────┴─────────────────────────────────┘
#
# NHI threshold semantics:
#   logNHI ≥ 20.3 = DLAs only
#   logNHI ≥ 19.0 = DLAs + sub-DLAs
#   logNHI ≥ 17.2 = DLAs + sub-DLAs + LLS  (the conventional "all HCD" cut)
#
# To submit:
#   sbatch --export=ALL,VARIANT=no_dla_no_bal slurm/train/submit_e2e_train_loa_nersc.sh
#   sbatch --export=ALL,VARIANT=no_hcd_with_bal slurm/train/submit_e2e_train_loa_nersc.sh
#   sbatch --export=ALL,VARIANT=no_hcd_no_bal slurm/train/submit_e2e_train_loa_nersc.sh
#
# All paths and thresholds can be overridden:
#   sbatch --export=ALL,VARIANT=no_hcd_no_bal,HCD_MIN_NHI=19.0,HCD_CAT=/your/dla.fits \
#       slurm/train/submit_e2e_train_loa_nersc.sh

# NB: drop `-u` because /global/cfs/cdirs/desi/software/desi_environment.sh
# references DESI_ROOT before defining it.
set -eo pipefail
export PYTHONUNBUFFERED=1

# Load NERSC desi env.
source /global/cfs/cdirs/desi/software/desi_environment.sh main || {
    echo "[submit] ERROR: failed to load NERSC desi environment" >&2; exit 1
}

# --- Tunables ---
VARIANT="${VARIANT:?must be set: no_dla_no_bal | no_hcd_with_bal | no_hcd_no_bal}"

# Default LOA paths on NERSC. Override via --export=ALL,QSOCAT=...
QSOCAT="${QSOCAT:-/global/cfs/cdirs/desi/users/martini/bal-catalogs/loa/QSO_cat_loa_main_dark_healpix_v3-altbal.fits}"
SPECDIR="${SPECDIR:-/global/cfs/cdirs/desi/spectro/redux/loa}"
# HCD catalog: a per-absorber FITS table (one row per detected
# DLA / sub-DLA / LLS) with columns TARGETID, NHI, P_DLA.
# The user keeps these in:
#   /global/cfs/cdirs/desicollab/users/jibancat/DLA/processed_gp_samples/
# The "LLS-mode" run is the natural choice for an "all HCDs" filter
# because it covers logNHI ≥ 17.2.
HCD_CAT="${HCD_CAT:-/global/cfs/cdirs/desicollab/users/jibancat/DLA/processed_gp_samples/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/dlacat-loa-main-dark.fits}"
HCD_TID_COL="${HCD_TID_COL:-TARGETID}"
HCD_NHI_COL="${HCD_NHI_COL:-NHI}"
# Optional P_DLA gate: only exclude TARGETIDs whose absorber has
# P_DLA ≥ this. Default 0 = no P_DLA cut (any absorber in the
# catalog excludes the sightline). Set to e.g. 0.99 to filter only
# confident detections.
HCD_MIN_PDLA="${HCD_MIN_PDLA:-0.0}"

# Self-contained run layout: everything for this job lives under ONE
# folder so you can rsync it to GreatLakes in one shot, e.g.
#
#   rsync -av /pscratch/.../v2_runs/loa_no_hcd_with_bal_<jobid>/  \
#       greatlakes:/nfs/turbo/.../v2_runs/loa_no_hcd_with_bal_<jobid>/
#
# Layout inside RUN_DIR:
#   trainset.h5               — preload output (legacy schema)
#   config.json               — TrainConfig snapshot
#   loss_history.json         — per-epoch loss
#   slurm.log                 — SLURM stdout (copied at end)
#   checkpoint_epoch_NNNN.pt  — full Adam state, every save_every
#   model_epoch_NNNN.h5       — inference-ready models
OUTDIR_BASE="${OUTDIR_BASE:-/pscratch/sd/j/jibancat/desi_gpy_dla_detection}"
RUN_TAG="${RUN_TAG:-loa_${VARIANT}_${SLURM_JOB_ID}}"
RUN_DIR="${RUN_DIR:-${OUTDIR_BASE}/v2_runs/${RUN_TAG}}"
TRAINSET_H5="${TRAINSET_H5:-${RUN_DIR}/trainset.h5}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}}"

# Filter args per VARIANT.
case "$VARIANT" in
    no_dla_no_bal)
        # Exclude only DLAs (logNHI ≥ 20.3) AND BALs (BI_CIV > 0).
        # Sub-DLAs and LLS are kept. Matches the legacy convention.
        HCD_MIN_NHI="${HCD_MIN_NHI:-20.3}"
        EXCLUDE_BAL_FLAG="--exclude-bal"
        VARIANT_DESCRIPTION="DLAs (logNHI ≥ 20.3) + BALs excluded; sub-DLAs/LLS kept"
        ;;
    no_hcd_with_bal)
        # Exclude all HCDs (logNHI ≥ 17.2 = DLA + sub-DLA + LLS). Keep BALs.
        # The trained GP can be applied to BAL spectra at inference time.
        HCD_MIN_NHI="${HCD_MIN_NHI:-17.2}"
        EXCLUDE_BAL_FLAG=""
        VARIANT_DESCRIPTION="all HCDs (logNHI ≥ 17.2) excluded; BALs KEPT"
        ;;
    no_hcd_no_bal)
        # Exclude all HCDs AND all BALs. Cleanest possible LOA training set.
        HCD_MIN_NHI="${HCD_MIN_NHI:-17.2}"
        EXCLUDE_BAL_FLAG="--exclude-bal"
        VARIANT_DESCRIPTION="all HCDs (logNHI ≥ 17.2) + BALs excluded"
        ;;
    *)
        echo "[error] VARIANT must be one of: no_dla_no_bal, no_hcd_with_bal, no_hcd_no_bal" >&2
        exit 2
        ;;
esac

# Other knobs (overridable).
Z_MIN="${Z_MIN:-2.0}"
Z_MAX="${Z_MAX:-4.25}"
MAX_SPECTRA="${MAX_SPECTRA:-300000}"
NUM_EPOCHS="${NUM_EPOCHS:-800}"
BATCH_SIZE="${BATCH_SIZE:-12500}"
LEARNING_RATE="${LEARNING_RATE:-0.005}"
NUM_PCA="${NUM_PCA:-30}"
NUM_FOREST_LINES="${NUM_FOREST_LINES:-3}"
DLAMBDA="${DLAMBDA:-0.15}"

# Pre-flight: required paths.
[ -r "$QSOCAT" ] || { echo "[error] QSOCAT not readable: $QSOCAT" >&2; exit 3; }
[ -d "$SPECDIR" ] || { echo "[error] SPECDIR not a directory: $SPECDIR" >&2; exit 4; }
if [ "$HCD_MIN_NHI" != "" ] && [ -n "$HCD_CAT" ]; then
    [ -r "$HCD_CAT" ] || { echo "[error] HCD_CAT not readable: $HCD_CAT" >&2; exit 5; }
fi

mkdir -p "$RUN_DIR"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

# Pre-flight: imports.
python -c "
import torch
from gpy_dla_detection.training.dataset import load_preprocessed_h5
from gpy_dla_detection.training.objective_v2 import vectorized_nll
from gpy_dla_detection.training.trainer_v2 import train, TrainConfig
from gpy_dla_detection.training.model_v2 import GPModelV2
import desispec.io
print(f'[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'CUDA not available; check NERSC env'
" || { echo "[error] preflight import failed" >&2; exit 6; }

echo "===================================================="
echo "  e2e_train_loa  variant: $VARIANT  job: $SLURM_JOB_ID"
echo "  $VARIANT_DESCRIPTION"
echo "===================================================="
echo "  Inputs:"
echo "    qsocat:         $QSOCAT"
echo "    specdir:        $SPECDIR"
echo "    hcd_cat:        ${HCD_CAT:-(none)}"
echo "    hcd cols:       tid='$HCD_TID_COL'  nhi='$HCD_NHI_COL'"
echo "  Filter pipeline (applied IN ORDER inside preload):"
echo "    1) z in [$Z_MIN, $Z_MAX]  AND  ZWARN==0"
echo "    2) BAL anti-join:  ${EXCLUDE_BAL_FLAG:+ON  (drop BI_CIV > 0)}${EXCLUDE_BAL_FLAG:-OFF (BALs KEPT)}"
if [ -n "${HCD_CAT}" ]; then
    PDLA_DESC=""
    if [ "$(awk "BEGIN{print ($HCD_MIN_PDLA > 0)}")" = "1" ]; then
        PDLA_DESC=" AND P_DLA ≥ $HCD_MIN_PDLA"
    fi
    echo "    3) HCD anti-join:  drop TARGETIDs with absorber NHI ≥ $HCD_MIN_NHI${PDLA_DESC}"
else
    echo "    3) HCD anti-join:  OFF (no --hcd-cat)"
fi
echo "    4) random cap to MAX_SPECTRA=$MAX_SPECTRA"
echo "  Outputs:"
echo "    trainset_h5:    $TRAINSET_H5"
echo "    output_dir:     $OUTPUT_DIR"
echo "  Training:"
echo "    epochs=$NUM_EPOCHS  batch=$BATCH_SIZE  lr=$LEARNING_RATE  k=$NUM_PCA"
echo "===================================================="
echo

# --- Step 1: PRELOAD ---
echo "=== STEP 1: preload ==="
PRELOAD_CMD="python -u preload_spectra/preload_loa_real.py \
    --qsocat \"$QSOCAT\" \
    --specdir \"$SPECDIR\" \
    --output \"$TRAINSET_H5\" \
    --z-min $Z_MIN --z-max $Z_MAX \
    --max-spectra $MAX_SPECTRA \
    --dlambda $DLAMBDA \
    $EXCLUDE_BAL_FLAG"
if [ -n "${HCD_CAT}" ]; then
    PRELOAD_CMD="$PRELOAD_CMD \
        --hcd-cat \"$HCD_CAT\" \
        --hcd-tid-col \"$HCD_TID_COL\" \
        --hcd-nhi-col \"$HCD_NHI_COL\" \
        --hcd-min-nhi $HCD_MIN_NHI \
        --hcd-min-pdla $HCD_MIN_PDLA"
fi

eval "$PRELOAD_CMD"

[ -r "$TRAINSET_H5" ] || { echo "[error] preload did not produce $TRAINSET_H5" >&2; exit 7; }
echo "preload wrote: $TRAINSET_H5 ($(du -h $TRAINSET_H5 | cut -f1))"
echo

# --- Step 2: TRAIN ---
echo "=== STEP 2: train ==="
python -u train_gp.py \
    --preloaded-file "$TRAINSET_H5" \
    --z-min $Z_MIN --z-max $Z_MAX \
    --max-spectra $MAX_SPECTRA \
    --num-pca-components $NUM_PCA \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --num-forest-lines $NUM_FOREST_LINES \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --save-every 10

# Post-flight: confirm training finished without NaN.
LOSS_FILE="$OUTPUT_DIR/loss_history.json"
[ -r "$LOSS_FILE" ] || { echo "[error] loss_history.json not written" >&2; exit 8; }
python -c "
import json, math, sys
with open('$LOSS_FILE') as f:
    h = json.load(f)
assert all(math.isfinite(x) for x in h), 'loss history contains non-finite values'
print(f'[postflight] loss start={h[0]:.4e} end={h[-1]:.4e} ({len(h)} epochs, monotone-ish: {h[-1] < h[0]})')
" || { echo "[error] training produced non-finite loss" >&2; exit 9; }

# Copy the SLURM stdout into the run dir so everything for this run is
# in one place (rsync to GreatLakes in one shot).
cp slurm/train/e2e_train_loa_${SLURM_JOB_ID}.log "$RUN_DIR/slurm.log" 2>/dev/null || true

echo
echo "===================================================="
echo "  e2e_train_loa  $VARIANT  COMPLETE"
echo "  Run folder:  $RUN_DIR"
echo "===================================================="
