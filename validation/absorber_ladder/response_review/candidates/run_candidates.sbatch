#!/bin/bash
#SBATCH --job-name=respcand
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/mfho/lowz_clean_work_2026-09-12/absorber_ladder/logs/%x_%j.log
# Low-DOF response-representation candidates (sealed opening rule
# RESPONSE_FAMILY_OPENING_RULE_PREDECLARATION.md, sha256 84ce1de6...).
# CALIBRATION SIDE ONLY: no HBI run, no real data, no mock dN/dX closure number.
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla-hbi
export HDF5_USE_FILE_LOCKING=FALSE JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export PYTHONPATH=/home/mfho/wt_abs_diag_2026-09
cd /home/mfho/wt_abs_diag_2026-09
OUT=${OUT:-/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/response_review/candidates}
mkdir -p "$OUT"
echo "code=$(git rev-parse HEAD) out=$OUT start=$(date)"
/usr/bin/time -v python3 validation/absorber_ladder/response_review/candidates/run_candidates.py \
    --out-dir "$OUT" ${EXTRA_ARGS:-}
echo "done rc=$? end=$(date)"
