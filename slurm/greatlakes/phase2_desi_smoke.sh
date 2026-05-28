#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p spgpu
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -J phase2_desi_smoke
#SBATCH -o slurm/greatlakes/phase2_desi_smoke_%j.log
#SBATCH -e slurm/greatlakes/phase2_desi_smoke_%j.log
# spgpu = larger node pool than gpu_mig40 with faster turnover; per
# 2026-05-11 attempt, gpu_mig40 projected START_TIME 24h+ for our 1h
# job. spgpu has more nodes and the same training_v3 path runs there.

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
    --chunk-size 2000 \
    --checkpoint-dir "$CKPT_DIR" \
    --checkpoint-every 25 \
    --out-dir "$OUT_DIR"
# chunk_size 2000 chosen to fit in ~10 GB GPU memory headroom.
# Per-chunk (B=2000, N=5662, k=30) intermediates ~1.4 GB each;
# spectrum_loss_batch holds ~5 of them simultaneously. First smoke
# OOM'd at chunk=5000 → 3.4 GB each ⇒ ~17 GB peak ⇒ exceeded the
# 44 GiB GPU after upfront-data-load overhead. The per-chunk
# CPU→GPU transfer fix in phase2_train_desi.py also reduces peak.

echo
echo "=== smoke complete; outputs in $OUT_DIR ==="
ls -la "$OUT_DIR"
