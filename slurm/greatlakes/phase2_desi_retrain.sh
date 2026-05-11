#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p spgpu
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -J phase2_desi_retrain
#SBATCH -o slurm/greatlakes/phase2_desi_retrain_%j.log
#SBATCH -e slurm/greatlakes/phase2_desi_retrain_%j.log

# Step C production retrain: 1500 iter on a v2 preload (DESI 2lpt or LOA).
# Uses tests/phase2_train_desi.py — corrected trainer (PCA init +
# hand-coded gradient via training_v3) on GPU.
#
# Walltime budget 8h based on extrapolation from trainer_v2 (3.2 s/iter
# on A100 80GB for 118k×3801, 1500 iter = 1h20m). Step C scales to
# 300k×5662, expected ~12 s/iter → ~5h for 1500 iter; 8h budget gives
# headroom. Trainer also writes a checkpoint every 25 iter and supports
# --resume PATH for chained jobs if any single run hits the wall.
#
# This script is parameterized — submit with PRELOAD + OUT_DIR + RUN_NAME:
#
#   # 2lpt loa-0 (DLAs included, no HCD/BAL mask)
#   sbatch --export=ALL,RUN_NAME=2lpt_loa0_wide,\
# PRELOAD=/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_wide_v2_1778186324/trainset.h5 \
#       slurm/greatlakes/phase2_desi_retrain.sh
#
#   # 2lpt loa-124 (HCD+BAL masked)
#   sbatch --export=ALL,RUN_NAME=2lpt_loa124_nohcd_nobal_wide,\
# PRELOAD=/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa124_nohcd_nobal_wide_v2_1778186324/trainset.h5 \
#       slurm/greatlakes/phase2_desi_retrain.sh
#
#   # Resume from a checkpoint (after a walltime kill)
#   sbatch --export=ALL,RUN_NAME=2lpt_loa0_wide,\
# PRELOAD=...,RESUME=/scratch/.../phase2_desi_checkpoint_iter0250.pt \
#       slurm/greatlakes/phase2_desi_retrain.sh

set -eo pipefail
export PYTHONUNBUFFERED=1

module load cuda/12.4.0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

# Required:
PRELOAD="${PRELOAD:?must set PRELOAD=/path/to/trainset.h5}"
RUN_NAME="${RUN_NAME:?must set RUN_NAME=<dataset_label>}"

# Optional:
N_ITERS="${N_ITERS:-1500}"
LR="${LR:-0.005}"
CHUNK_SIZE="${CHUNK_SIZE:-12500}"
MAX_SPECTRA="${MAX_SPECTRA:-}"   # empty = use all
RESUME="${RESUME:-}"

OUT_DIR="${OUT_DIR:-docs/notes/2026-05-11_desi_phase2_${RUN_NAME}}"
CKPT_DIR="${CKPT_DIR:-/scratch/cavestru_root/cavestru0/mfho/phase2_desi/${RUN_NAME}/checkpoints}"

mkdir -p "$OUT_DIR" "$CKPT_DIR"

echo "=== phase2_desi_retrain ==="
echo "  run_name     : $RUN_NAME"
echo "  preload      : $PRELOAD"
echo "  out_dir      : $OUT_DIR"
echo "  checkpoints  : $CKPT_DIR"
echo "  n_iters      : $N_ITERS"
echo "  lr           : $LR"
echo "  chunk_size   : $CHUNK_SIZE"
echo "  max_spectra  : ${MAX_SPECTRA:-all}"
echo "  resume       : ${RESUME:-(none)}"
echo "  job_id       : ${SLURM_JOB_ID:-(local)}"
echo "  node         : $(hostname)"
echo "  gpu          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
echo

EXTRA_ARGS=""
if [ -n "$MAX_SPECTRA" ]; then EXTRA_ARGS="$EXTRA_ARGS --max-spectra $MAX_SPECTRA"; fi
if [ -n "$RESUME" ]; then EXTRA_ARGS="$EXTRA_ARGS --resume $RESUME"; fi

python -u tests/phase2_train_desi.py \
    --preload "$PRELOAD" \
    --n-iters "$N_ITERS" \
    --lr "$LR" \
    --device cuda \
    --chunk-size "$CHUNK_SIZE" \
    --checkpoint-dir "$CKPT_DIR" \
    --checkpoint-every 25 \
    --max-walltime-sec 27000 \
    --out-dir "$OUT_DIR" \
    $EXTRA_ARGS

echo
echo "=== retrain complete; outputs in $OUT_DIR ==="
ls -la "$OUT_DIR"
