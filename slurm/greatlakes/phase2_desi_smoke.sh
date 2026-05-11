#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p gpu_mig40
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -J phase2_desi_smoke
#SBATCH -o slurm/greatlakes/phase2_desi_smoke_%j.log
#SBATCH -e slurm/greatlakes/phase2_desi_smoke_%j.log
# gpu_mig40 = A100 MIG 40GB slice — same partition as the v2 corrected
# retrain jobs (49243842-49268620). Shorter queue than spgpu (28 vs 286
# pending at last check). cavestru0 allocation has GPU access (verified
# from past jobs' AllocTRES).

# Step C smoke: 5k spectra × 50 iter on 2lpt loa-0 wide v2 preload.
# Verifies tests/phase2_train_desi.py works end-to-end on GPU before
# committing to a full 1500-iter production run.
#
# Expected wall: ~5-15 min depending on actual A100 throughput.
#
# Submit:
#   sbatch slurm/greatlakes/phase2_desi_smoke.sh

set -eo pipefail
export PYTHONUNBUFFERED=1

module load cuda/12.4.0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

PRELOAD="${PRELOAD:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_wide_v2_1778186324/trainset.h5}"
OUT_DIR="${OUT_DIR:-docs/notes/2026-05-11_desi_smoke}"
CKPT_DIR="${CKPT_DIR:-/scratch/cavestru_root/cavestru0/mfho/phase2_desi_smoke_${SLURM_JOB_ID:-test}/checkpoints}"
N_ITERS="${N_ITERS:-50}"
MAX_SPECTRA="${MAX_SPECTRA:-5000}"

mkdir -p "$OUT_DIR" "$CKPT_DIR"

echo "=== phase2_desi_smoke ==="
echo "  preload      : $PRELOAD"
echo "  out_dir      : $OUT_DIR"
echo "  checkpoints  : $CKPT_DIR"
echo "  n_iters      : $N_ITERS"
echo "  max_spectra  : $MAX_SPECTRA"
echo "  job_id       : ${SLURM_JOB_ID:-(local)}"
echo "  node         : $(hostname)"
echo "  gpu          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
echo

python -u tests/phase2_train_desi.py \
    --preload "$PRELOAD" \
    --max-spectra "$MAX_SPECTRA" \
    --n-iters "$N_ITERS" \
    --device cuda \
    --chunk-size 5000 \
    --checkpoint-dir "$CKPT_DIR" \
    --checkpoint-every 25 \
    --out-dir "$OUT_DIR"

echo
echo "=== smoke complete; outputs in $OUT_DIR ==="
ls -la "$OUT_DIR"
