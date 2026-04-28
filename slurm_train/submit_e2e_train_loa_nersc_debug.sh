#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --job-name=e2e_loa_dbg
#SBATCH --output=slurm_train/e2e_loa_dbg_%j.log
#SBATCH --error=slurm_train/e2e_loa_dbg_%j.err
#SBATCH -A desi
#SBATCH --time=00:30:00
#SBATCH --gpus=1

# NERSC `-q debug` variant of the e2e LOA submit. Same VARIANT logic
# and pre/post-flight checks as submit_e2e_train_loa_nersc.sh, but:
#   - 30 min walltime (debug queue limit)
#   - reduced NUM_EPOCHS=200 (vs 800 in regular) to fit
#   - debug queue has ~minutes-of-pending vs regular's days
#
# Why 200 epochs is enough for a debug run:
#   On A100 the steady-state per-epoch wall is ~3 s/epoch on 300k spectra
#   (extrapolated from the 5,000-spectrum debug at ~0.05 s/epoch). 200 epochs
#   ≈ 10 min wall. Plus preload (5–15 min) and overhead → fits in 30 min.
#   Cosine schedule's T_max=50 means most convergence happens in the first
#   ~100 epochs anyway; running 200 vs 800 gives a "decent" model whose loss
#   has flattened, sufficient for inference comparison and downstream tests.
#   For a fully-converged production model, switch back to the regular-queue
#   submit (submit_e2e_train_loa_nersc.sh) when its turn comes up.
#
# Submit:
#   sbatch --export=ALL,VARIANT=no_dla_no_bal slurm_train/submit_e2e_train_loa_nersc_debug.sh
#   sbatch --export=ALL,VARIANT=no_hcd_with_bal slurm_train/submit_e2e_train_loa_nersc_debug.sh
#   sbatch --export=ALL,VARIANT=no_hcd_no_bal slurm_train/submit_e2e_train_loa_nersc_debug.sh
#
# All three variants of the regular submit are also valid here; the only
# difference is queue + walltime + default NUM_EPOCHS.

set -eo pipefail
export PYTHONUNBUFFERED=1

source /global/cfs/cdirs/desi/software/desi_environment.sh main || {
    echo "[submit] ERROR: failed to load NERSC desi environment" >&2; exit 1
}

VARIANT="${VARIANT:?must be set: no_dla_no_bal | no_hcd_with_bal | no_hcd_no_bal}"

QSOCAT="${QSOCAT:-/global/cfs/cdirs/desi/users/martini/bal-catalogs/loa/QSO_cat_loa_main_dark_healpix_v3-altbal.fits}"
SPECDIR="${SPECDIR:-/global/cfs/cdirs/desi/spectro/redux/loa}"
HCD_CAT="${HCD_CAT:-/global/cfs/cdirs/desicollab/users/jibancat/DLA/processed_gp_samples/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/dlacat-loa-main-dark.fits}"
HCD_TID_COL="${HCD_TID_COL:-TARGETID}"
HCD_NHI_COL="${HCD_NHI_COL:-NHI}"
HCD_MIN_PDLA="${HCD_MIN_PDLA:-0.0}"

# Self-contained run layout: one folder per run for easy rsync to GreatLakes.
OUTDIR_BASE="${OUTDIR_BASE:-/pscratch/sd/j/jibancat/desi_gpy_dla_detection}"
RUN_TAG="${RUN_TAG:-loa_${VARIANT}_dbg_${SLURM_JOB_ID}}"
RUN_DIR="${RUN_DIR:-${OUTDIR_BASE}/v2_runs/${RUN_TAG}}"
TRAINSET_H5="${TRAINSET_H5:-${RUN_DIR}/trainset.h5}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}}"

case "$VARIANT" in
    no_dla_no_bal)
        HCD_MIN_NHI="${HCD_MIN_NHI:-20.3}"
        EXCLUDE_BAL_FLAG="--exclude-bal"
        VARIANT_DESC="DLAs (logNHI ≥ 20.3) + BALs excluded; sub-DLAs/LLS kept"
        ;;
    no_hcd_with_bal)
        HCD_MIN_NHI="${HCD_MIN_NHI:-17.2}"
        EXCLUDE_BAL_FLAG=""
        VARIANT_DESC="all HCDs (logNHI ≥ 17.2) excluded; BALs KEPT"
        ;;
    no_hcd_no_bal)
        HCD_MIN_NHI="${HCD_MIN_NHI:-17.2}"
        EXCLUDE_BAL_FLAG="--exclude-bal"
        VARIANT_DESC="all HCDs (logNHI ≥ 17.2) + BALs excluded"
        ;;
    *)
        echo "[error] VARIANT must be one of: no_dla_no_bal, no_hcd_with_bal, no_hcd_no_bal" >&2
        exit 2
        ;;
esac

# Debug-scale knobs.
Z_MIN="${Z_MIN:-2.0}"
Z_MAX="${Z_MAX:-4.25}"
MAX_SPECTRA="${MAX_SPECTRA:-300000}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"          # debug-scale: 200 (vs 800 in regular)
BATCH_SIZE="${BATCH_SIZE:-12500}"
LEARNING_RATE="${LEARNING_RATE:-0.005}"
NUM_PCA="${NUM_PCA:-30}"
NUM_FOREST_LINES="${NUM_FOREST_LINES:-3}"
DLAMBDA="${DLAMBDA:-0.15}"

[ -r "$QSOCAT" ] || { echo "[error] QSOCAT: $QSOCAT" >&2; exit 3; }
[ -d "$SPECDIR" ] || { echo "[error] SPECDIR: $SPECDIR" >&2; exit 4; }
[ -r "$HCD_CAT" ] || { echo "[error] HCD_CAT: $HCD_CAT" >&2; exit 5; }

mkdir -p "$RUN_DIR"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

python -c "
import torch
from gpy_dla_detection.training.dataset import load_preprocessed_h5
from gpy_dla_detection.training.objective_v2 import vectorized_nll
from gpy_dla_detection.training.trainer_v2 import train, TrainConfig
from gpy_dla_detection.training.model_v2 import GPModelV2
import desispec.io
print(f'[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'CUDA not available'
" || { echo "[error] preflight import failed" >&2; exit 6; }

echo "===================================================="
echo "  e2e_loa DEBUG variant: $VARIANT  job: $SLURM_JOB_ID"
echo "  $VARIANT_DESC"
echo "  -q debug  walltime 30 min  epochs $NUM_EPOCHS"
echo "===================================================="
echo "  trainset:    $TRAINSET_H5"
echo "  output_dir:  $OUTPUT_DIR"
echo "  filter:      z [$Z_MIN, $Z_MAX]; HCD NHI ≥ $HCD_MIN_NHI; BAL=${EXCLUDE_BAL_FLAG:+ON}${EXCLUDE_BAL_FLAG:-OFF}; P_DLA=$HCD_MIN_PDLA"
echo "  scale:       max_spectra=$MAX_SPECTRA epochs=$NUM_EPOCHS batch=$BATCH_SIZE"
echo "===================================================="
echo

echo "=== STEP 1: preload ==="
PRELOAD_CMD="python -u preload_spectra/preload_loa_real.py \
    --qsocat \"$QSOCAT\" \
    --specdir \"$SPECDIR\" \
    --output \"$TRAINSET_H5\" \
    --z-min $Z_MIN --z-max $Z_MAX \
    --max-spectra $MAX_SPECTRA \
    --dlambda $DLAMBDA \
    $EXCLUDE_BAL_FLAG \
    --hcd-cat \"$HCD_CAT\" \
    --hcd-tid-col \"$HCD_TID_COL\" \
    --hcd-nhi-col \"$HCD_NHI_COL\" \
    --hcd-min-nhi $HCD_MIN_NHI \
    --hcd-min-pdla $HCD_MIN_PDLA"
eval "$PRELOAD_CMD"

[ -r "$TRAINSET_H5" ] || { echo "[error] preload did not produce $TRAINSET_H5" >&2; exit 7; }
echo "preload wrote: $TRAINSET_H5 ($(du -h "$TRAINSET_H5" | cut -f1))"

echo
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
    --save-every 25

# Post-flight: confirm finite loss.
LOSS_FILE="$OUTPUT_DIR/loss_history.json"
[ -r "$LOSS_FILE" ] || { echo "[error] loss_history.json not written" >&2; exit 8; }
python -c "
import json, math
with open('$LOSS_FILE') as f: h = json.load(f)
assert all(math.isfinite(x) for x in h), 'non-finite loss'
print(f'[postflight] loss start={h[0]:.4e} end={h[-1]:.4e} ({len(h)} epochs)')
" || { echo "[error] post-flight loss check failed" >&2; exit 9; }

cp slurm_train/e2e_loa_dbg_${SLURM_JOB_ID}.log "$RUN_DIR/slurm.log" 2>/dev/null || true

echo
echo "=== e2e_loa DEBUG $VARIANT COMPLETE ==="
echo "  Run folder:  $RUN_DIR"
