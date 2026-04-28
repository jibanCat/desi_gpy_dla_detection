#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --job-name=2lpt_train
#SBATCH --output=slurm_train/2lpt_train_%j.log
#SBATCH --error=slurm_train/2lpt_train_%j.err
#SBATCH -A desi
#SBATCH --time=00:30:00
#SBATCH --gpus=1

# NERSC -q debug 2LPT GP training. Mirrors slurm/greatlakes/preload_train_2lpt.sh
# but on NERSC A100 with the desicollab 2LPT mock mirror.
#
# 2LPT on NERSC: /global/cfs/cdirs/desicollab/mocks/lya_forest/develop/
#                lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/{loa-0, loa-124}/
#
# Two variants:
#   loa0               : uncontaminated mock (no DLAs/metals/BALs).
#   loa124_nohcd_nobal : contaminated mock; HCDs (any logNHI in
#                        hcd_truth_cat.fits ≥ 17.0) and BALs (BI_CIV>0
#                        in bal_cat.fits) anti-joined out.
#
# Submit:
#   sbatch --export=ALL,VARIANT=loa0 slurm_train/preload_train_2lpt_nersc.sh
#   sbatch --export=ALL,VARIANT=loa124_nohcd_nobal slurm_train/preload_train_2lpt_nersc.sh
#
# Walltime budget: 30 min on debug queue.
#   Preload: ~5–15 min (depends on max_spectra).
#   Train:   ~5–10 min for 200 epochs at 50k spectra on A100.
#   Total fits comfortably under 30 min.

# NB: drop `-u` because /global/cfs/cdirs/desi/software/desi_environment.sh
# references DESI_ROOT before defining it (commit a521ad8 noted this).
set -eo pipefail
export PYTHONUNBUFFERED=1

source /global/cfs/cdirs/desi/software/desi_environment.sh main || {
    echo "[submit] ERROR: failed to load NERSC desi environment" >&2; exit 1
}

VARIANT="${VARIANT:?must be set: loa0 | loa124_nohcd_nobal}"

# Tunables. Sized for the 30-min debug walltime: at ~57 spectra/s preload
# (FITS I/O bound), 20k spectra ≈ 6 min preload + ~3 min train + overhead
# = ~12 min total, leaving ~18 min margin. Job 52188320 timed out on 50k.
MAX_SPECTRA="${MAX_SPECTRA:-20000}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-12500}"
LEARNING_RATE="${LEARNING_RATE:-0.005}"
NUM_PCA="${NUM_PCA:-30}"
Z_MIN="${Z_MIN:-2.0}"
Z_MAX="${Z_MAX:-4.0}"          # 2LPT z range is ~1.8–3.8 → cap at 4.0

DATA_BASE="/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0"
SCRATCH="${SCRATCH:-/pscratch/sd/j/jibancat/desi_gpy_dla_detection}"

case "$VARIANT" in
    loa0)
        MOCK_DIR="$DATA_BASE/loa-0"
        EXTRA_FLAGS=""
        TAG="loa0"
        ;;
    loa124_nohcd_nobal)
        MOCK_DIR="$DATA_BASE/loa-124"
        EXTRA_FLAGS="--exclude-hcd --exclude-bal"
        TAG="loa124_nohcd_nobal"
        ;;
    *)
        echo "[error] VARIANT must be loa0 or loa124_nohcd_nobal, got: $VARIANT" >&2
        exit 2
        ;;
esac

[ -d "$MOCK_DIR" ] || { echo "[error] MOCK_DIR not found: $MOCK_DIR" >&2; exit 3; }
[ -r "$MOCK_DIR/zcat.fits" ] || { echo "[error] zcat.fits not in $MOCK_DIR" >&2; exit 4; }

# Self-contained run layout: one folder per run for easy rsync to GreatLakes.
# Inside: trainset.h5, model_epoch_*.h5, checkpoint_*.pt, config.json,
# loss_history.json, slurm.log.
RUN_TAG="${RUN_TAG:-2lpt_${TAG}_${SLURM_JOB_ID}}"
RUN_DIR="${RUN_DIR:-${SCRATCH}/v2_runs/${RUN_TAG}}"
TRAINSET_H5="${TRAINSET_H5:-${RUN_DIR}/trainset.h5}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}}"
mkdir -p "$RUN_DIR"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

# Pre-flight: imports.
python -c "
import torch, desispec.io, healpy
from gpy_dla_detection.training.dataset import load_preprocessed_h5
from gpy_dla_detection.training.objective_v2 import vectorized_nll
from gpy_dla_detection.training.trainer_v2 import train, TrainConfig
from gpy_dla_detection.training.model_v2 import GPModelV2
print(f'[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'CUDA not available'
" || { echo "[error] preflight import failed" >&2; exit 5; }

echo "===================================================="
echo "  2LPT v2 training on NERSC -q debug"
echo "  variant: $VARIANT  job: $SLURM_JOB_ID"
echo "===================================================="
echo "  mock_dir:    $MOCK_DIR"
echo "  trainset:    $TRAINSET_H5"
echo "  output_dir:  $OUTPUT_DIR"
echo "  filter:      ${EXTRA_FLAGS:-(none — uncontaminated by construction)}"
echo "  scale:       max_spectra=$MAX_SPECTRA epochs=$NUM_EPOCHS batch=$BATCH_SIZE k=$NUM_PCA"
echo "  z range:     [$Z_MIN, $Z_MAX]"
echo "===================================================="

# Step 1: preload (uses preload_2lpt_simple.py — RA/DEC → healpix because
# 2LPT zcat lacks HPXPIXEL, unlike the LOA altbal catalog).
echo
echo "=== STEP 1: preload ==="
python -u preload_spectra/preload_2lpt_simple.py \
    --mock-dir "$MOCK_DIR" \
    --output "$TRAINSET_H5" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --max-spectra "$MAX_SPECTRA" \
    $EXTRA_FLAGS

[ -r "$TRAINSET_H5" ] || { echo "[error] preload did not produce $TRAINSET_H5" >&2; exit 6; }
echo "preload wrote: $TRAINSET_H5 ($(du -h "$TRAINSET_H5" | cut -f1))"

# Step 2: train.
echo
echo "=== STEP 2: train ==="
python -u train_gp.py \
    --preloaded-file "$TRAINSET_H5" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --max-spectra "$MAX_SPECTRA" \
    --num-pca-components "$NUM_PCA" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --num-forest-lines 3 \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --save-every 25

# Post-flight: confirm finite loss.
LOSS_FILE="$OUTPUT_DIR/loss_history.json"
[ -r "$LOSS_FILE" ] || { echo "[error] loss_history.json not written" >&2; exit 7; }
python -c "
import json, math
with open('$LOSS_FILE') as f: h = json.load(f)
assert all(math.isfinite(x) for x in h), 'non-finite loss'
print(f'[postflight] loss start={h[0]:.4e} end={h[-1]:.4e} ({len(h)} epochs)')
" || { echo "[error] post-flight loss check failed" >&2; exit 8; }

cp slurm_train/2lpt_train_${SLURM_JOB_ID}.log "$RUN_DIR/slurm.log" 2>/dev/null || true

echo
echo "=== 2LPT $VARIANT TRAINING COMPLETE ==="
echo "  Run folder:  $RUN_DIR"
