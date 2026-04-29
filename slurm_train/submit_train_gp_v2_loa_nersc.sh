#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --job-name=train_gp_v2
#SBATCH --output=slurm_train/train_gp_v2_%j.log
#SBATCH --error=slurm_train/train_gp_v2_%j.err
#SBATCH --mail-user=mfho@umich.edu
#SBATCH --mail-type=ALL
#SBATCH -A desi
#SBATCH --time=24:00:00
#SBATCH --gpus=1

# Streamlined v2 training submit for NERSC Perlmutter.
# Uses gpy_dla_detection.training (objective_v2 + trainer_v2):
#   - vectorized NLL across the batch (no per-spectrum Python loop)
#   - autograd backward (legacy dlog_beta approximation fixed)
#   - per-epoch checkpointing with resume support
#
# Layer 3 measured 14x CPU speedup at batch_size=32 vs the legacy trainer.
# GPU should be substantially better; run a profile sweep first if you
# care about exact numbers.
#
# Expected behaviour:
#   - byte-stable to legacy at the LOSS level (parity-tested in
#     tests/test_objective_v2_parity.py)
#   - SLIGHTLY DIFFERENT model after training because v2 uses the
#     correct dlog_beta gradient (legacy used an approximation; both
#     converge but to slightly different log_beta values)
#
# Prerequisites (NOT done by this script):
#   - The preloaded HDF5 must exist. It is produced by the existing
#     preload pipeline:
#         preload_spectra/desi-preload.py     (raw spectra → per-hpx HDF5)
#         preload_spectra/prepare_trainset.py (per-hpx → gp_interp_trainset.h5)
#     See docs/training_v2_workflow.md.
#   - Recommended: run the debug submit first
#         sbatch slurm_train/debug_train_gp_v2_nersc.sh
#     to validate paths/env before queuing this multi-hour job.
#
# To submit:
#   sbatch slurm_train/submit_train_gp_v2_loa_nersc.sh
#
# To override defaults at submit time:
#   sbatch --export=ALL,NUM_EPOCHS=400,BATCH_SIZE=25000 \
#       slurm_train/submit_train_gp_v2_loa_nersc.sh

# NB: drop `-u` because /global/cfs/cdirs/desi/software/desi_environment.sh
# references DESI_ROOT before defining it, which trips `set -u`. Same
# workaround used in slurm/debug_pr3_test_plan.sh (commit a521ad8).
set -eo pipefail

export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export PYTHONUNBUFFERED=1

# Load NERSC desi env.
source /global/cfs/cdirs/desi/software/desi_environment.sh main || {
    echo "[submit] ERROR: failed to load NERSC desi environment" >&2; exit 1
}

# Allow caller to override these at submit time.
PRELOADED_FILE="${PRELOADED_FILE:-/pscratch/sd/j/jibancat/preload-loa-gpdla-20250202/gp_interp_trainset.h5}"
CATALOG_FILE="${CATALOG_FILE:-}"  # optional TARGETID filter; leave empty to use all spectra
OUTPUT_DIR="${OUTPUT_DIR:-/pscratch/sd/j/jibancat/desi_gpy_dla_detection/learnlogs_v2/run_${SLURM_JOB_ID}}"
NUM_EPOCHS="${NUM_EPOCHS:-800}"
BATCH_SIZE="${BATCH_SIZE:-12500}"
LEARNING_RATE="${LEARNING_RATE:-0.005}"
MAX_SPECTRA="${MAX_SPECTRA:-300000}"
NUM_PCA="${NUM_PCA:-30}"
NUM_FOREST_LINES="${NUM_FOREST_LINES:-3}"
Z_MIN="${Z_MIN:-2.5}"
Z_MAX="${Z_MAX:-4.25}"

# Pre-flight: required input file must exist.
if [ ! -r "$PRELOADED_FILE" ]; then
    echo "[submit] ERROR: PRELOADED_FILE not readable: $PRELOADED_FILE" >&2
    echo "        Override via --export=ALL,PRELOADED_FILE=/path/to/your.h5" >&2
    exit 2
fi
if [ -n "$CATALOG_FILE" ] && [ ! -r "$CATALOG_FILE" ]; then
    echo "[submit] ERROR: CATALOG_FILE not readable: $CATALOG_FILE" >&2
    exit 3
fi

mkdir -p "$OUTPUT_DIR"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

# Pre-flight: imports work (catches torch / module mismatches before
# burning queue time on a multi-hour job).
python -c "
from gpy_dla_detection.training.dataset import load_preprocessed_h5
from gpy_dla_detection.training.objective_v2 import vectorized_nll
from gpy_dla_detection.training.trainer_v2 import train, TrainConfig
from gpy_dla_detection.training.model_v2 import GPModelV2
import torch
assert torch.cuda.is_available(), 'CUDA not available; check NERSC env'
print('[preflight] training/ imports OK; CUDA available')
" || { echo "[submit] ERROR: preflight import failed" >&2; exit 4; }

echo "=== train_gp_v2 NERSC submit ==="
echo "host: $(hostname)"
echo "repo: $REPO_DIR"
echo "preloaded: $PRELOADED_FILE"
echo "catalog:   $CATALOG_FILE"
echo "output:    $OUTPUT_DIR"
echo "epochs=$NUM_EPOCHS batch_size=$BATCH_SIZE lr=$LEARNING_RATE max_spectra=$MAX_SPECTRA"
echo

python -u train_gp.py \
    --preloaded-file "$PRELOADED_FILE" \
    --catalog-file "$CATALOG_FILE" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --max-spectra "$MAX_SPECTRA" \
    --num-pca-components "$NUM_PCA" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --num-forest-lines "$NUM_FOREST_LINES" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

echo "=== done ==="
