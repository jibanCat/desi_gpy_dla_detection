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
# To submit:
#   sbatch slurm_train/submit_train_gp_v2_loa_nersc.sh
#
# To override defaults at submit time:
#   sbatch --export=ALL,NUM_EPOCHS=400,BATCH_SIZE=25000 \
#       slurm_train/submit_train_gp_v2_loa_nersc.sh

set -euo pipefail

export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export PYTHONUNBUFFERED=1

# Load NERSC desi env.
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Allow caller to override these at submit time.
PRELOADED_FILE="${PRELOADED_FILE:-/pscratch/sd/j/jibancat/preload-loa-gpdla-20250202/gp_interp_trainset.h5}"
CATALOG_FILE="${CATALOG_FILE:-/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/loa/gp_trainset_loa.fits}"
OUTPUT_DIR="${OUTPUT_DIR:-/pscratch/sd/j/jibancat/desi_gpy_dla_detection/learnlogs_v2/run_${SLURM_JOB_ID}}"
NUM_EPOCHS="${NUM_EPOCHS:-800}"
BATCH_SIZE="${BATCH_SIZE:-12500}"
LEARNING_RATE="${LEARNING_RATE:-0.005}"
MAX_SPECTRA="${MAX_SPECTRA:-300000}"
NUM_PCA="${NUM_PCA:-30}"
NUM_FOREST_LINES="${NUM_FOREST_LINES:-3}"
Z_MIN="${Z_MIN:-2.5}"
Z_MAX="${Z_MAX:-4.25}"

mkdir -p "$OUTPUT_DIR"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

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
