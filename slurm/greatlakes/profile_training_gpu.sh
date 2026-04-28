#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0:30:00
#SBATCH -J gpdla-train-profile
#SBATCH -o slurm/greatlakes/profile_train_%j.log
#SBATCH -e slurm/greatlakes/profile_train_%j.log
set -euo pipefail

# Layer 3 profile of GP training on a single GreatLakes GPU.
# Compares CPU baseline (~36 s/epoch on 128 spectra) against GPU.
#
# Submit with:
#   sbatch slurm/greatlakes/profile_training_gpu.sh
#
# Override defaults:
#   sbatch --export=ALL,NUM_SPECTRA=512,EPOCHS=5 slurm/greatlakes/profile_training_gpu.sh

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

# GreatLakes module setup. Adjust if your conda env is different.
module load cuda/12.4.0 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

NUM_SPECTRA="${NUM_SPECTRA:-1024}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
N_PIX="${N_PIX:-600}"
K="${K:-30}"
LR="${LR:-0.005}"

echo "=== GPU train profile on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo

python tests/profile/profile_training.py \
    --device cuda \
    --num-spectra "$NUM_SPECTRA" \
    --n-pix "$N_PIX" \
    --k "$K" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --tag "gpu_n${NUM_SPECTRA}_k${K}_npix${N_PIX}_bs${BATCH_SIZE}"

echo "=== done ==="
