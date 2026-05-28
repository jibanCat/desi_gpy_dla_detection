#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J phase2_vec_smoke
#SBATCH -o slurm/greatlakes/phase2_vec_smoke_%j.log
#SBATCH -e slurm/greatlakes/phase2_vec_smoke_%j.log

# Vectorized smoke: 5k spectra × 50 Adam iter on the same Phase-1 sizing as
# commit 0918ea7. Lets us cross-check the trained M and per-iter wall-time of
# the vectorized path against the existing per-spectrum 5k×50 baseline at
# docs/notes/2026-05-08_matlab_dr16_validation/phase2_*.
#
# Output goes to docs/notes/2026-05-08_matlab_dr16_validation_vec_smoke/
# (DOES NOT clobber the Phase-1 baseline).
#
# Submit:
#   sbatch slurm/greatlakes/phase2_dr16_vectorized_smoke.sh

set -eo pipefail
export PYTHONUNBUFFERED=1

# Vectorized path benefits from BLAS parallelism. 4 threads matches the -c 4
# allocation; OMP+MKL+OPENBLAS all set so torch's matmul + Cholesky go
# multi-threaded.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

OUT_DIR="$REPO_DIR/docs/notes/2026-05-08_matlab_dr16_validation_vec_smoke"
mkdir -p "$OUT_DIR"

echo "===================================================="
echo "  Phase 2 VECTORIZED smoke  job: $SLURM_JOB_ID"
echo "  GreatLakes -p standard  c=4  mem=32G  t=2h"
echo "  spectra: 5000   iters: 50   lr: 0.01"
echo "  thread cap: OMP=$OMP_NUM_THREADS"
echo "  vectorized: 1   chunk_size: 1000"
echo "  out-dir: $OUT_DIR"
echo "===================================================="

python -u tests/phase2_train_dr16.py \
    --n-spectra 5000 \
    --n-iters 50 \
    --lr 0.01 \
    --checkpoint-every 0 \
    --vectorized 1 \
    --chunk-size 1000 \
    --out-dir "$OUT_DIR"

echo
echo "===================================================="
echo "  RUN COMPLETE  job: $SLURM_JOB_ID"
echo "  endpoints: $OUT_DIR/phase2_endpoint_table.md"
echo "  trained:   $OUT_DIR/phase2_result.npz"
echo "  corr:      $OUT_DIR/phase2_corr_compare.png"
echo "===================================================="
