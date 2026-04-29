#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p spgpu
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -J train_gp_v2_2lpt
#SBATCH -o slurm/greatlakes/train_gp_v2_%j.log
#SBATCH -e slurm/greatlakes/train_gp_v2_%j.log

# Streamlined v2 training submit for UMich GreatLakes.
# Uses the same train_gp.py / training/ stack as the NERSC submit.
#
# This template targets the 2LPT mock-0 contaminated dataset on
# /nfs/turbo. Override the data paths via --export=ALL,... if you
# want to point at a different set.
#
# To submit:
#   sbatch slurm/greatlakes/train_gp_v2_2lpt.sh
#
# To override at submit time:
#   sbatch --export=ALL,NUM_EPOCHS=200,MAX_SPECTRA=50000 \
#       slurm/greatlakes/train_gp_v2_2lpt.sh

set -euo pipefail

export PYTHONUNBUFFERED=1

# Load CUDA + activate conda env.
module load cuda/12.4.0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

# Defaults — caller can override.
PRELOADED_FILE="${PRELOADED_FILE:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/data/loa/gp_interp_trainset.h5}"
CATALOG_FILE="${CATALOG_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs_v2/run_${SLURM_JOB_ID}}"
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

echo "=== train_gp_v2 GreatLakes submit ==="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "repo: $REPO_DIR"
echo "preloaded: $PRELOADED_FILE"
echo "catalog:   ${CATALOG_FILE:-(none)}"
echo "output:    $OUTPUT_DIR"
echo "epochs=$NUM_EPOCHS batch_size=$BATCH_SIZE lr=$LEARNING_RATE max_spectra=$MAX_SPECTRA"
echo

CMD="python -u train_gp.py \
    --preloaded-file \"$PRELOADED_FILE\" \
    --z-min \"$Z_MIN\" --z-max \"$Z_MAX\" \
    --max-spectra \"$MAX_SPECTRA\" \
    --num-pca-components \"$NUM_PCA\" \
    --num-epochs \"$NUM_EPOCHS\" \
    --batch-size \"$BATCH_SIZE\" \
    --learning-rate \"$LEARNING_RATE\" \
    --num-forest-lines \"$NUM_FOREST_LINES\" \
    --output-dir \"$OUTPUT_DIR\" \
    --device cuda"

if [ -n "$CATALOG_FILE" ]; then
    CMD="$CMD --catalog-file \"$CATALOG_FILE\""
fi

eval "$CMD"

echo "=== done ==="
