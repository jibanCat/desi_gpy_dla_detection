#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J phase2_dr16_retrain
#SBATCH -o slurm/greatlakes/phase2_dr16_%j.log
#SBATCH -e slurm/greatlakes/phase2_dr16_%j.log

# A.5 Phase 2 production retrain — full DR16 train_ind set.
#
# Trains v1 spectrum_loss (Adam + BOSS DR12Q priors) on the full 89k
# train_ind QSOs from /home/mfho/MATLAB/.../preloaded_qsos.mat.
# Compares trained M to MATLAB's converged final M.
#
# Outputs (under docs/notes/2026-05-08_matlab_dr16_validation/):
#   phase2_result.npz         — trained M, μ, log_ω, log_c_0/τ_0/β + history
#   phase2_corr_compare.png   — 4-panel ours_init / ours_trained / matlab_init / matlab_final
#   phase2_endpoint_table.md  — c_0/τ_0/β scalars table
#
# The data cache (~5 GB at 89k spectra) lives at
# tests/fixtures/dr16_phase2_cache/data_cache_n89408.npz.
#
# Submit:
#   sbatch [--export=ALL,N_ITERS=200] slurm/greatlakes/phase2_dr16_retrain.sh

set -eo pipefail
export PYTHONUNBUFFERED=1
# Thread cap matches the documented finding (2026-05-07): the 89k-spectrum
# Python loop suffers BLAS thread oversubscription with default threading.
# Single-thread + parallel small-matrix ops gives ~10× speedup vs default.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Tunables
N_ITERS="${N_ITERS:-200}"      # 200 Adam iter; loss should converge at this scale
N_SPECTRA="${N_SPECTRA:-89408}"  # full DR16 train_ind set
LR="${LR:-0.01}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
# Save and exit if training elapsed > this (seconds). 23h = 82800s, leaves
# ~1h for clean shutdown / final artifact write before SLURM walltime kill.
MAX_WALLTIME_SEC="${MAX_WALLTIME_SEC:-82800}"
RESUME="${RESUME:-}"  # path to .pt checkpoint to resume from, or empty

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

echo "===================================================="
echo "  Phase 2 DR16 retrain  job: $SLURM_JOB_ID"
echo "  GreatLakes -p standard  c=8  mem=64G  t=24h"
echo "  spectra: $N_SPECTRA   iters: $N_ITERS   lr: $LR"
echo "  thread cap: OMP=$OMP_NUM_THREADS"
echo "  checkpoint_every: $CHECKPOINT_EVERY"
echo "  max_walltime_sec: $MAX_WALLTIME_SEC"
echo "  resume:  ${RESUME:-<from scratch>}"
echo "===================================================="

RESUME_ARG=()
if [ -n "$RESUME" ]; then
    RESUME_ARG=(--resume "$RESUME")
fi

python -u tests/phase2_train_dr16.py \
    --n-spectra "$N_SPECTRA" \
    --n-iters "$N_ITERS" \
    --lr "$LR" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --max-walltime-sec "$MAX_WALLTIME_SEC" \
    "${RESUME_ARG[@]}"

echo
echo "===================================================="
echo "  RUN COMPLETE  job: $SLURM_JOB_ID"
echo "  endpoints: docs/notes/2026-05-08_matlab_dr16_validation/phase2_endpoint_table.md"
echo "  trained .npz: docs/notes/2026-05-08_matlab_dr16_validation/phase2_result.npz"
echo "  corr plot: docs/notes/2026-05-08_matlab_dr16_validation/phase2_corr_compare.png"
echo "===================================================="
