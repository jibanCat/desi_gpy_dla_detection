#!/bin/bash
#SBATCH --job-name=ebaserefit
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/mfho/lowz_clean_work_2026-09-12/absorber_ladder/logs/%x_%j.log
# DIAGNOSTIC: section 6A reweighting invariance for E with the base REFITTED.
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla-hbi
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/mfho/wt_abs_diag_2026-09 JAX_PLATFORMS=cpu
cd /home/mfho/wt_abs_diag_2026-09
D=validation/absorber_ladder/response_review/candidates/base_refit_check
echo "code=$(git rev-parse HEAD) start=$(date)"
python3 $D/run_base_refit_check.py --out $D/E_base_refit_reweighting_check.json
echo "rc=$? ; running tests"
python3 $D/test_base_refit_check.py
echo "done end=$(date)"
