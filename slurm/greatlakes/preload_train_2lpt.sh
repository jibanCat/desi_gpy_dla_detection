#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p spgpu
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J train_2lpt
#SBATCH -o slurm/greatlakes/train_2lpt_%j.log
#SBATCH -e slurm/greatlakes/train_2lpt_%j.log

# Production-scale 2LPT GP training on GreatLakes A40.
# Pipeline:
#   1) preload_2lpt_simple.py — read zcat + spectra → gp_interp_trainset.h5
#   2) train_gp.py            — train v2 on the produced HDF5
#
# VARIANT controls which 2LPT subset:
#   - "loa0"             : uncontaminated (loa-0); no HCDs/BALs by construction.
#   - "loa124_nohcd_nobal": loa-124 with HCDs (any logNHI ≥ 17) and BALs (BI_CIV > 0)
#                          filtered out via truth catalogs.
#
# Submit with:
#   sbatch --export=ALL,VARIANT=loa0 slurm/greatlakes/preload_train_2lpt.sh
#   sbatch --export=ALL,VARIANT=loa124_nohcd_nobal slurm/greatlakes/preload_train_2lpt.sh

set -eo pipefail
export PYTHONUNBUFFERED=1

VARIANT="${VARIANT:-loa0}"
MAX_SPECTRA="${MAX_SPECTRA:-50000}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-12500}"
LEARNING_RATE="${LEARNING_RATE:-0.005}"
NUM_PCA="${NUM_PCA:-30}"
Z_MIN="${Z_MIN:-2.5}"
Z_MAX="${Z_MAX:-4.0}"          # 2LPT z range is ~1.8–3.8

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

DATA_BASE="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0"
SCRATCH="${SCRATCH:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection}"

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

TRAINSET_H5="${SCRATCH}/trainset_2lpt_${TAG}_${SLURM_JOB_ID}.h5"
OUTPUT_DIR="${SCRATCH}/learnlogs_v2/2lpt_${TAG}_${SLURM_JOB_ID}"
mkdir -p "$(dirname "$TRAINSET_H5")" "$OUTPUT_DIR"

# Load env
module load cuda/12.4.0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

echo "=== train_2lpt $VARIANT on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "mock_dir:    $MOCK_DIR"
echo "trainset:    $TRAINSET_H5"
echo "output:      $OUTPUT_DIR"
echo "max_spectra: $MAX_SPECTRA"
echo "epochs=$NUM_EPOCHS batch_size=$BATCH_SIZE lr=$LEARNING_RATE k=$NUM_PCA"
echo

# Pre-flight: imports
python -c "
from gpy_dla_detection.training.dataset import load_preprocessed_h5
from gpy_dla_detection.training.objective_v2 import vectorized_nll
from gpy_dla_detection.training.trainer_v2 import train, TrainConfig
from gpy_dla_detection.training.model_v2 import GPModelV2
import desispec.io
import healpy
import torch
print(f'[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()}')
" || { echo "[error] preflight import failed" >&2; exit 4; }

# Step 1: preload
echo "=== step 1: preload ==="
python -u preload_spectra/preload_2lpt_simple.py \
    --mock-dir "$MOCK_DIR" \
    --output "$TRAINSET_H5" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --max-spectra "$MAX_SPECTRA" \
    $EXTRA_FLAGS

if [ ! -r "$TRAINSET_H5" ]; then
    echo "[error] preload did not produce $TRAINSET_H5" >&2; exit 5
fi

# Step 2: train
echo
echo "=== step 2: train ==="
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
    --save-every 10

echo "=== done ==="
