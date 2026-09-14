#!/bin/bash
#SBATCH --job-name=realc1_ident
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/real_c1/identity/ident_%j.log
#
# THE IDENTITY TEST at PRODUCTION chain length: run_real_c1.py (under
# --allow-mock-for-identity-test) vs the FROZEN run_ladder.py on the 2LPT-0 v3
# MOCK pack, same seed, same frozen fixed objects, same NUTS configuration.
# MOCK ONLY — no real pack, no real likelihood, no real value.
set -eo pipefail
for v in $(env | grep -o '^SLURM_[A-Z_]*' | grep -vE 'SLURM_ARRAY_TASK_ID|SLURM_JOB_ID'); do unset $v; done
export SBATCH_CONSTRAINT=""
source ~/.bashrc; conda activate gpdla-hbi
set -u
W=/home/mfho/wt_abs_diag_2026-09
cd $W
export PYTHONPATH=$W HDF5_USE_FILE_LOCKING=FALSE JAX_PLATFORMS=cpu \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/real_c1/identity
mkdir -p $OUT
echo "code=$(git -C $W rev-parse HEAD) host=$(hostname) start=$(date)"

python validation/real_c1/identity_test.py \
  --out-dir $OUT --seed 20260811 \
  --chains 4 --warmup 1500 --samples 1000 -J 8 -j 3

echo "end=$(date)"
