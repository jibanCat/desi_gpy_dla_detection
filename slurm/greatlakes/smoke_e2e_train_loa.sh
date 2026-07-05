#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p spgpu
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0:45:00
#SBATCH -J e2e_loa_smoke
#SBATCH -o slurm/greatlakes/e2e_loa_smoke_%j.log
#SBATCH -e slurm/greatlakes/e2e_loa_smoke_%j.log

# GreatLakes smoke-test of the e2e LOA training pipeline.
# Same VARIANT logic as slurm/train/submit_e2e_train_loa_nersc.sh but:
#   - paths point at the GreatLakes /nfs/turbo mirror
#   - small MAX_SPECTRA (5,000) and NUM_EPOCHS (5) so the whole job
#     fits in ~45 min wall on spgpu (A40)
#   - intended only to exercise the preload + train code paths before
#     submitting the full NERSC job
#
# Submit one variant:
#   sbatch --export=ALL,VARIANT=no_dla_no_bal slurm/greatlakes/smoke_e2e_train_loa.sh
#   sbatch --export=ALL,VARIANT=no_hcd_with_bal slurm/greatlakes/smoke_e2e_train_loa.sh
#   sbatch --export=ALL,VARIANT=no_hcd_no_bal slurm/greatlakes/smoke_e2e_train_loa.sh

set -eo pipefail
export PYTHONUNBUFFERED=1

VARIANT="${VARIANT:?must be set: no_dla_no_bal | no_hcd_with_bal | no_hcd_no_bal}"

# GreatLakes mirror paths.
QSOCAT="${QSOCAT:-/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v3-altbal.fits}"
SPECDIR="${SPECDIR:-/nfs/turbo/lsa-cavestru/mfho/DESI/loa}"
HCD_CAT="${HCD_CAT:-/nfs/turbo/lsa-cavestru/mfho/DESI/DLA/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/dlacat-loa-main-dark.fits}"
HCD_TID_COL="${HCD_TID_COL:-TARGETID}"
HCD_NHI_COL="${HCD_NHI_COL:-NHI}"
HCD_MIN_PDLA="${HCD_MIN_PDLA:-0.0}"

OUTDIR_BASE="${OUTDIR_BASE:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection}"
TRAINSET_H5="${TRAINSET_H5:-${OUTDIR_BASE}/trainsets_smoke/loa_${VARIANT}_${SLURM_JOB_ID}.h5}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTDIR_BASE}/learnlogs_v2_smoke/loa_${VARIANT}_${SLURM_JOB_ID}}"

# Filter args per VARIANT (same as NERSC).
case "$VARIANT" in
    no_dla_no_bal)
        HCD_MIN_NHI="${HCD_MIN_NHI:-20.3}"
        EXCLUDE_BAL_FLAG="--exclude-bal"
        VARIANT_DESCRIPTION="DLAs (logNHI ≥ 20.3) + BALs excluded"
        ;;
    no_hcd_with_bal)
        HCD_MIN_NHI="${HCD_MIN_NHI:-17.2}"
        EXCLUDE_BAL_FLAG=""
        VARIANT_DESCRIPTION="all HCDs (logNHI ≥ 17.2) excluded; BALs KEPT"
        ;;
    no_hcd_no_bal)
        HCD_MIN_NHI="${HCD_MIN_NHI:-17.2}"
        EXCLUDE_BAL_FLAG="--exclude-bal"
        VARIANT_DESCRIPTION="all HCDs (logNHI ≥ 17.2) + BALs excluded"
        ;;
    *)
        echo "[error] VARIANT must be one of: no_dla_no_bal, no_hcd_with_bal, no_hcd_no_bal" >&2
        exit 2
        ;;
esac

# Smoke-scale knobs.
Z_MIN="${Z_MIN:-2.0}"
Z_MAX="${Z_MAX:-4.25}"
MAX_SPECTRA="${MAX_SPECTRA:-5000}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2500}"
LEARNING_RATE="${LEARNING_RATE:-0.005}"
NUM_PCA="${NUM_PCA:-30}"
NUM_FOREST_LINES="${NUM_FOREST_LINES:-3}"
DLAMBDA="${DLAMBDA:-0.15}"

# Pre-flight: input file existence.
[ -r "$QSOCAT" ] || { echo "[error] QSOCAT not readable: $QSOCAT" >&2; exit 3; }
[ -d "$SPECDIR" ] || { echo "[error] SPECDIR not a directory: $SPECDIR" >&2; exit 4; }
[ -r "$HCD_CAT" ] || { echo "[error] HCD_CAT not readable: $HCD_CAT" >&2; exit 5; }

mkdir -p "$(dirname "$TRAINSET_H5")" "$OUTPUT_DIR"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

# Env.
module load cuda/12.4.0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

# Pre-flight imports.
python -c "
import torch, desispec.io
from gpy_dla_detection.training.dataset import load_preprocessed_h5
from gpy_dla_detection.training.objective_v2 import vectorized_nll
from gpy_dla_detection.training.trainer_v2 import train, TrainConfig
from gpy_dla_detection.training.model_v2 import GPModelV2
print(f'[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'CUDA not available'
" || { echo "[error] preflight import failed" >&2; exit 6; }

echo "===================================================="
echo "  e2e_loa_smoke  variant: $VARIANT  job: $SLURM_JOB_ID"
echo "  $VARIANT_DESCRIPTION"
echo "===================================================="
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  qsocat:    $QSOCAT"
echo "  specdir:   $SPECDIR"
echo "  hcd_cat:   $HCD_CAT  (cols: tid='$HCD_TID_COL' nhi='$HCD_NHI_COL')"
echo "  trainset:  $TRAINSET_H5"
echo "  output:    $OUTPUT_DIR"
echo "  filter:    z in [$Z_MIN, $Z_MAX]; HCD NHI ≥ $HCD_MIN_NHI; "
echo "             BAL exclude=${EXCLUDE_BAL_FLAG:+ON}${EXCLUDE_BAL_FLAG:-OFF}; "
echo "             P_DLA gate=$HCD_MIN_PDLA"
echo "  smoke:     max_spectra=$MAX_SPECTRA epochs=$NUM_EPOCHS batch=$BATCH_SIZE"
echo "===================================================="
echo

# Step 1: preload
echo "=== STEP 1: preload ==="
python -u preload_spectra/preload_loa_real.py \
    --qsocat "$QSOCAT" \
    --specdir "$SPECDIR" \
    --output "$TRAINSET_H5" \
    --z-min $Z_MIN --z-max $Z_MAX \
    --max-spectra $MAX_SPECTRA \
    --dlambda $DLAMBDA \
    --hcd-cat "$HCD_CAT" \
    --hcd-tid-col "$HCD_TID_COL" \
    --hcd-nhi-col "$HCD_NHI_COL" \
    --hcd-min-nhi $HCD_MIN_NHI \
    --hcd-min-pdla $HCD_MIN_PDLA \
    $EXCLUDE_BAL_FLAG

[ -r "$TRAINSET_H5" ] || { echo "[error] preload did not produce $TRAINSET_H5" >&2; exit 7; }
echo "preload wrote: $TRAINSET_H5 ($(du -h "$TRAINSET_H5" | cut -f1))"
echo

# Step 2: train
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
    --save-every 1

# Post-flight: confirm finite loss.
LOSS_FILE="$OUTPUT_DIR/loss_history.json"
[ -r "$LOSS_FILE" ] || { echo "[error] loss_history.json not written" >&2; exit 8; }
python -c "
import json, math
with open('$LOSS_FILE') as f:
    h = json.load(f)
assert all(math.isfinite(x) for x in h), 'non-finite loss'
print(f'[postflight] loss start={h[0]:.4e} end={h[-1]:.4e} ({len(h)} epochs)')
" || { echo "[error] post-flight loss check failed" >&2; exit 9; }

echo
echo "=== SMOKE SUCCESS for VARIANT=$VARIANT ==="
