#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --job-name=train_gp_v2_dbg
#SBATCH --output=slurm_train/train_gp_v2_dbg_%j.log
#SBATCH --error=slurm_train/train_gp_v2_dbg_%j.err
#SBATCH -A desi
#SBATCH --time=00:30:00
#SBATCH --gpus=1

# DEBUG submit for the v2 trainer on NERSC Perlmutter `-q debug`.
# Goals:
#   1) Validate the env loads, the preloaded HDF5 reads, the v2 trainer
#      runs end-to-end on a small subset before queuing a multi-day job.
#   2) Surface any path / module / package mismatch FAST. The debug
#      queue is short (≤ 30 min wall) and high-priority.
#
# Behaviour:
#   - Reads the same preloaded HDF5 as the production submit (override
#     via --export=ALL,PRELOADED_FILE=...).
#   - Caps to 5,000 spectra, runs 5 epochs.
#   - Writes its output to a separate `learnlogs_v2/debug_<jobid>/` dir
#     so production and debug runs don't clash.
#
# To submit:
#   sbatch slurm_train/debug_train_gp_v2_nersc.sh
#
# Override defaults:
#   sbatch --export=ALL,PRELOADED_FILE=/path/to/your.h5,MAX_SPECTRA=2000 \
#       slurm_train/debug_train_gp_v2_nersc.sh

set -euo pipefail

export PYTHONUNBUFFERED=1

# --- Pre-flight: env load ---
source /global/cfs/cdirs/desi/software/desi_environment.sh main || {
    echo "[debug] ERROR: failed to load NERSC desi environment" >&2
    exit 1
}

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

# --- Pre-flight: required input files ---
PRELOADED_FILE="${PRELOADED_FILE:-/pscratch/sd/j/jibancat/preload-loa-gpdla-20250202/gp_interp_trainset.h5}"
CATALOG_FILE="${CATALOG_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/pscratch/sd/j/jibancat/desi_gpy_dla_detection/learnlogs_v2/debug_${SLURM_JOB_ID}}"
MAX_SPECTRA="${MAX_SPECTRA:-5000}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2500}"

if [ ! -r "$PRELOADED_FILE" ]; then
    echo "[debug] ERROR: PRELOADED_FILE not readable: $PRELOADED_FILE" >&2
    echo "        This file is produced by preload_spectra/prepare_trainset.py" >&2
    echo "        (legacy preload pipeline; see docs/training_v2_workflow.md)." >&2
    exit 2
fi
if [ -n "$CATALOG_FILE" ] && [ ! -r "$CATALOG_FILE" ]; then
    echo "[debug] ERROR: CATALOG_FILE not readable: $CATALOG_FILE" >&2
    exit 3
fi

mkdir -p "$OUTPUT_DIR"

# --- Pre-flight: import sanity check (no GPU work yet) ---
python -c "
import sys
import torch
print(f'[preflight] python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.cuda.is_available()}')
from gpy_dla_detection.training.dataset import load_preprocessed_h5
from gpy_dla_detection.training.objective_v2 import vectorized_nll
from gpy_dla_detection.training.trainer_v2 import train, TrainConfig
from gpy_dla_detection.training.model_v2 import GPModelV2
print('[preflight] training/ imports OK')
" || { echo "[debug] ERROR: preflight import failed" >&2; exit 4; }

echo "=== DEBUG train_gp_v2 NERSC submit ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "host: $(hostname)"
echo "preloaded: $PRELOADED_FILE"
[ -n "$CATALOG_FILE" ] && echo "catalog:   $CATALOG_FILE"
echo "output:    $OUTPUT_DIR"
echo "epochs=$NUM_EPOCHS batch_size=$BATCH_SIZE max_spectra=$MAX_SPECTRA"
echo

CMD="python -u train_gp.py \
    --preloaded-file \"$PRELOADED_FILE\" \
    --z-min 2.5 --z-max 4.25 \
    --max-spectra $MAX_SPECTRA \
    --num-pca-components 30 \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate 0.005 \
    --num-forest-lines 3 \
    --output-dir \"$OUTPUT_DIR\" \
    --device cuda \
    --save-every 1"

if [ -n "$CATALOG_FILE" ]; then
    CMD="$CMD --catalog-file \"$CATALOG_FILE\""
fi

eval "$CMD"

# --- Post-flight: confirm output ---
H5_COUNT=$(find "$OUTPUT_DIR" -name "model_epoch_*.h5" | wc -l)
LOSS_FILE="$OUTPUT_DIR/loss_history.json"
if [ "$H5_COUNT" -lt 1 ]; then
    echo "[debug] ERROR: no model_epoch_*.h5 written to $OUTPUT_DIR" >&2
    exit 5
fi
if [ ! -r "$LOSS_FILE" ]; then
    echo "[debug] ERROR: loss_history.json not written" >&2
    exit 6
fi

echo
echo "=== DEBUG SUCCESS ==="
echo "wrote $H5_COUNT model H5 files"
echo "loss history (first/last):"
python -c "
import json
with open('$LOSS_FILE') as f:
    h = json.load(f)
print(f'  epoch 0: {h[0]:.4f}')
print(f'  epoch {len(h)-1}: {h[-1]:.4f}')
print(f'  monotone-ish: {h[-1] < h[0]}')
"

echo
echo "If this debug job succeeded, submit production with:"
echo "  sbatch slurm_train/submit_train_gp_v2_loa_nersc.sh"
