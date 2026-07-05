#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --job-name=train_v2
#SBATCH --output=slurm/train/train_v2_%j.log
#SBATCH --error=slurm/train/train_v2_%j.err
#SBATCH -A desi
#SBATCH --time=12:00:00
#SBATCH --gpus=1

# TRAIN-ONLY: assumes a preload job has already produced
#   ${OUTDIR_BASE}/v2_runs/${RUN_TAG}/trainset.h5
# (e.g. via slurm/train/preload_loa_only_nersc.sh or any other producer).
#
# Defaults to `-q regular` with 12-hour walltime — NERSC GPU queue can
# be long, so we'd rather sit through one wait and run to convergence
# than risk a too-short walltime cutting off training mid-run. At ~7s/
# epoch on A100 with 300k spectra, NUM_EPOCHS=1500 finishes in ~3 h
# with plenty of headroom for slow startup, GPU contention, etc. The
# trainer auto-resumes from the latest checkpoint if more epochs are
# wanted later.
#
# For a fast smoke test override on the command line — `-q debug` is
# hard-capped at 30 min:
#
#   sbatch -q debug --time=00:30:00 --export=ALL,RUN_TAG=...,NUM_EPOCHS=5 \
#       slurm/train/train_only_nersc.sh
#
# Outputs land in the SAME RUN_DIR (alongside trainset.h5):
#   model_epoch_NNNN.h5
#   checkpoint_epoch_NNNN.pt
#   config.json
#   loss_history.json
#   train.slurm.log
#
# Submit (production):
#   sbatch --export=ALL,RUN_TAG=loa_no_dla_no_bal_<jobid> slurm/train/train_only_nersc.sh
#
# Tunables overridable via --export=ALL,KEY=val,...
#   NUM_EPOCHS=1500 (production default — historical Y3 used model_epoch_920;
#                   1500 leaves headroom to converge fully without re-queueing.
#                   Trainer auto-resumes from the latest checkpoint if even
#                   more epochs are wanted later).
#   BATCH_SIZE, LEARNING_RATE, NUM_PCA, etc.

set -eo pipefail
export PYTHONUNBUFFERED=1

source /global/cfs/cdirs/desi/software/desi_environment.sh main || {
    echo "[error] failed to load desi env" >&2; exit 1
}

RUN_TAG="${RUN_TAG:?must be set to the RUN_TAG of the preload job (e.g. loa_no_dla_no_bal_52234567)}"

OUTDIR_BASE="${OUTDIR_BASE:-/pscratch/sd/j/jibancat/desi_gpy_dla_detection}"
RUN_DIR="${RUN_DIR:-${OUTDIR_BASE}/v2_runs/${RUN_TAG}}"
TRAINSET_H5="${RUN_DIR}/trainset.h5"

[ -r "$TRAINSET_H5" ] || {
    echo "[error] trainset.h5 not found: $TRAINSET_H5" >&2
    echo "        Did you run the preload job first? Check that"
    echo "        --export=ALL,RUN_TAG matches the RUN_TAG of the preload job." >&2
    exit 2
}

NUM_EPOCHS="${NUM_EPOCHS:-1500}"
BATCH_SIZE="${BATCH_SIZE:-12500}"
LEARNING_RATE="${LEARNING_RATE:-0.005}"
NUM_PCA="${NUM_PCA:-30}"
NUM_FOREST_LINES="${NUM_FOREST_LINES:-3}"

# z-range carries over from preload via the metadata; if not set, fall
# back to the legacy LOA range. Read from dataset_metadata.json if present.
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

python -c "
import torch
from gpy_dla_detection.training.dataset import load_preprocessed_h5
from gpy_dla_detection.training.objective_v2 import vectorized_nll
from gpy_dla_detection.training.trainer_v2 import train, TrainConfig
from gpy_dla_detection.training.model_v2 import GPModelV2
print(f'[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'CUDA not available'
" || { echo "[error] preflight import failed" >&2; exit 6; }

echo "===================================================="
echo "  TRAIN ONLY  RUN_TAG=$RUN_TAG  job: $SLURM_JOB_ID"
echo "  queue:       $SLURM_JOB_QOS  walltime: $SLURM_JOB_TIME_LIMIT"
echo "===================================================="
echo "  trainset:    $TRAINSET_H5  ($(du -h $TRAINSET_H5 | cut -f1))"
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
    --save-every 25

LOSS_FILE="$RUN_DIR/loss_history.json"
[ -r "$LOSS_FILE" ] || { echo "[error] loss_history.json not written" >&2; exit 8; }
python -c "
import json, math
with open('$LOSS_FILE') as f: h = json.load(f)
assert all(math.isfinite(x) for x in h), 'non-finite loss'
print(f'[postflight] loss start={h[0]:.4e} end={h[-1]:.4e} ({len(h)} epochs)')
" || { echo "[error] post-flight loss check failed" >&2; exit 9; }

cp "slurm/train/train_v2_${SLURM_JOB_ID}.log" "$RUN_DIR/train.slurm.log" 2>/dev/null || true

echo
echo "===================================================="
echo "  TRAIN COMPLETE  $RUN_TAG"
echo "  RUN_DIR:    $RUN_DIR"
echo "  Models:     $RUN_DIR/model_epoch_NNNN.h5"
echo
echo "  Move via Globus from:"
echo "    /pscratch/sd/j/jibancat/desi_gpy_dla_detection/v2_runs/"
echo "  to GreatLakes:"
echo "    /nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/"
echo "===================================================="
