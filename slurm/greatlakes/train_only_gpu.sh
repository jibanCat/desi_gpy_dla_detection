#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p gpu_mig40
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -c 8
# 300k spectra × 3801 px × float32 = 4.6 GB per array, ×3 main arrays plus
# de-forest/centering intermediates → loader peaks ~32-40 GB. 32G OOMs every
# time. 64G has comfortable headroom.
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J train_v2
#SBATCH -o slurm/greatlakes/train_v2_%j.log
#SBATCH -e slurm/greatlakes/train_v2_%j.log

# TRAIN-ONLY on GreatLakes A100 MIG (CC 8.0).
# Assumes a preload job has already produced
#   ${OUTDIR_BASE}/v2_runs/${RUN_TAG}/trainset.h5
#
# Outputs land in the SAME RUN_DIR (alongside trainset.h5).
#
# Submit:
#   sbatch --export=ALL,RUN_TAG=2lpt_loa0_<jobid> slurm/greatlakes/train_only_gpu.sh
#
# To switch to A40 partition (longer queue but more memory):
#   sbatch --export=ALL,RUN_TAG=...  --partition=spgpu  slurm/greatlakes/train_only_gpu.sh

set -eo pipefail
export PYTHONUNBUFFERED=1

RUN_TAG="${RUN_TAG:?must be set to the RUN_TAG of the preload job, e.g. 2lpt_loa0_52234567}"

OUTDIR_BASE="${OUTDIR_BASE:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection}"
RUN_DIR="${RUN_DIR:-${OUTDIR_BASE}/v2_runs/${RUN_TAG}}"
# Allow re-using a trainset from a different RUN_DIR (e.g. the original
# pre-fix trainset for a "from-scratch" retrain into a fresh output dir).
TRAINSET_H5="${TRAINSET_H5:-${RUN_DIR}/trainset.h5}"

[ -r "$TRAINSET_H5" ] || {
    echo "[error] trainset.h5 not found: $TRAINSET_H5" >&2
    echo "        Run the preload job first" >&2
    exit 2
}

NUM_EPOCHS="${NUM_EPOCHS:-800}"          # full converge by default; v2 trainer auto-resumes
BATCH_SIZE="${BATCH_SIZE:-12500}"
LEARNING_RATE="${LEARNING_RATE:-0.005}"
NUM_PCA="${NUM_PCA:-30}"
NUM_FOREST_LINES="${NUM_FOREST_LINES:-3}"

if [ -r "$RUN_DIR/dataset_metadata.json" ]; then
    Z_MIN_DEFAULT=$(python -c "import json; print(json.load(open('$RUN_DIR/dataset_metadata.json'))['z_min'])")
    Z_MAX_DEFAULT=$(python -c "import json; print(json.load(open('$RUN_DIR/dataset_metadata.json'))['z_max'])")
else
    Z_MIN_DEFAULT=2.0
    Z_MAX_DEFAULT=4.25
fi
Z_MIN="${Z_MIN:-$Z_MIN_DEFAULT}"
Z_MAX="${Z_MAX:-$Z_MAX_DEFAULT}"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

module load cuda/12.4.0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

python -c "
import torch
from gpy_dla_detection.training.dataset import load_preprocessed_h5
from gpy_dla_detection.training.objective_v2 import vectorized_nll
from gpy_dla_detection.training.trainer_v2 import train, TrainConfig
from gpy_dla_detection.training.model_v2 import GPModelV2
print(f'[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()}')
assert torch.cuda.is_available()
" || { echo "[error] preflight import failed" >&2; exit 5; }

echo "===================================================="
echo "  TRAIN ONLY (GreatLakes)  RUN_TAG=$RUN_TAG  job: $SLURM_JOB_ID"
echo "===================================================="
echo "  GPU:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  trainset:    $TRAINSET_H5 ($(du -h $TRAINSET_H5 | cut -f1))"
echo "  output_dir:  $RUN_DIR"
echo "  z range:     [$Z_MIN, $Z_MAX]"
echo "  scale:       epochs=$NUM_EPOCHS batch=$BATCH_SIZE lr=$LEARNING_RATE k=$NUM_PCA"
echo "===================================================="

python -u train_gp.py \
    --preloaded-file "$TRAINSET_H5" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --num-pca-components "$NUM_PCA" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --num-forest-lines "$NUM_FOREST_LINES" \
    --output-dir "$RUN_DIR" \
    --device cuda \
    --save-every 25 \
    ${EXTRA_TRAIN_FLAGS:-}

LOSS_FILE="$RUN_DIR/loss_history.json"
[ -r "$LOSS_FILE" ] || { echo "[error] loss_history.json not written" >&2; exit 7; }
python -c "
import json, math
with open('$LOSS_FILE') as f: h = json.load(f)
assert all(math.isfinite(x) for x in h), 'non-finite loss'
print(f'[postflight] loss start={h[0]:.4e} end={h[-1]:.4e} ({len(h)} epochs)')
" || { echo "[error] post-flight loss check failed" >&2; exit 8; }

cp "slurm/greatlakes/train_v2_${SLURM_JOB_ID}.log" "$RUN_DIR/train.slurm.log" 2>/dev/null || true

echo
echo "===================================================="
echo "  TRAIN COMPLETE (GreatLakes)  $RUN_TAG"
echo "  RUN_DIR:    $RUN_DIR"
echo "===================================================="
