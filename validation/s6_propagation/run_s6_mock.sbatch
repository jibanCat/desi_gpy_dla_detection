#!/bin/bash
#SBATCH --job-name=s6mock
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/s6_propagation/runs/logs/s6mock_%j.log
#
# S6 MOCK CONTROL (sealed predeclaration §3, "mock control, optional"): ONE
# 2LPT-0 run at the completeness draw with the largest |Delta m|, to show the
# closure moves by the same amount.  No selection consequence.
#
# Convention = the F2 mock member B-phi2lpt-C1nsadd-M1CUTj1 (2LPT-0, seed
# 20260811): v3 support pack + census + ops, M1CUT with --lam-imputations 1
# --lam-imputation 0, truth-pinned sub-floor --fix P, Mg_B_2lpt0, a0 = 1/K.
# The ONLY difference from that frozen run is --c-fixed-file.
# Env: DRAW (the draw index).
set -eo pipefail
for v in $(env | grep -o '^SLURM_[A-Z_]*' | grep -vE 'SLURM_ARRAY_TASK_ID|SLURM_JOB_ID'); do unset $v; done
export SBATCH_CONSTRAINT=""
source ~/.bashrc; conda activate gpdla-hbi
set -u
W=/home/mfho/wt_abs_diag_2026-09
cd $W
export PYTHONPATH=$W HDF5_USE_FILE_LOCKING=FALSE JAX_PLATFORMS=cpu \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

L=/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13
V3=$L/support_v3; CAND=$L/response_review/candidates
S=$L/s6_propagation; OUT=$S/runs; mkdir -p $OUT/logs
CFILE=$S/tables/C_S6_draw${DRAW}.npz
[ -f "$CFILE" ] || { echo "BLOCKED: $CFILE missing"; exit 8; }

echo "s6 mock control draw=$DRAW code=$(git -C $W rev-parse HEAD) host=$(hostname) start=$(date)"

python validation/fp_ladder/run_ladder.py \
  --pack $V3/scanpack_2lpt0_b300_v3.npz \
  --ladder M1CUT \
  --seed 20260811 --chains 4 --warmup 1500 --samples 1000 --target-accept 0.95 \
  --census $V3/fp_census_2lpt0_v3.npz --ops $V3/empirical_ops_2lpt0_v3.npz \
  --require-support --stage S6_MOCK_CONTROL \
  --mg-fixed-file $CAND/Mg_B_2lpt0.npz \
  --c-fixed-file $CFILE \
  --fix P --lam-imputations 1 --lam-imputation 0 \
  --out $OUT/RUN_S6_MOCK_2lpt0_draw${DRAW}.json

echo "done mock draw=$DRAW rc=$? end=$(date)"
