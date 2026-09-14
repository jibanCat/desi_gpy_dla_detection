#!/bin/bash
#SBATCH --job-name=finalladder
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/mfho/lowz_clean_work_2026-09-12/final_campaign/logs/%x_%A_%a.log
# Final HBI ladder (PI 2026-09-13d §13; sealed FINAL_LADDER_PREDECLARATION). One task = one (variant, family, seed[, a0]).
# Env: VARIANT (E|B), PHI (2lpt|smooth|family), FAM, SEED, LADDER (ORACLE|M1CUT), A0 (optional), TAG
set -eo pipefail
for v in $(env | grep -o '^SLURM_[A-Z_]*' | grep -vE 'SLURM_ARRAY_TASK_ID|SLURM_JOB_ID'); do unset $v; done
export SBATCH_CONSTRAINT=""
source ~/.bashrc; conda activate gpdla-hbi
set -u
W=/home/mfho/wt_abs_diag_2026-09; cd $W; export PYTHONPATH=$W HDF5_USE_FILE_LOCKING=FALSE JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
L=/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13; V3=$L/support_v3; CAND=$L/response_review/candidates
FAMS=(2lpt0 london0 saclay0); FAM=${FAMS[$SLURM_ARRAY_TASK_ID]}
MG=$CAND/Mg_${VARIANT}_${FAM}.npz
case "$PHI" in
  2lpt)   MGARG="--mg-fixed-file $MG";;
  smooth) MGARG="--mg-fixed-file $MG --mg-fixed-key Mg_phi_smooth";;
  family) MGARG="--mg-fixed-file $MG --mg-fixed-key Mg_phi_family";;
esac
A0ARG=""; [ -n "${A0:-}" ] && A0ARG="--fp-a0 $A0"
LAMARG=""; [ "$LADDER" = "M1CUT" ] && LAMARG="--lam-imputations ${LAMJ:-1} --lam-imputation ${LAMj:-0}"
OUT=$L/final/runs/$TAG; mkdir -p $OUT
python validation/fp_ladder/run_ladder.py --pack $V3/scanpack_${FAM}_b300_v3.npz --ladder $LADDER --seed $SEED --chains 4 --warmup 1500 --samples 1000 --target-accept 0.95 \
  --census $V3/fp_census_${FAM}_v3.npz --ops $V3/empirical_ops_${FAM}_v3.npz --require-support --stage FINAL_$TAG \
  $MGARG --c-fixed-file $L/completeness/C_C1nsadd_${FAM}.npz --fix P $LAMARG $A0ARG \
  --out $OUT/RUN_${TAG}_${FAM}_s${SEED}${A0:+_a0$A0}${LAMJ:+_J${LAMJ}j${LAMj}}.json
echo "DONE $TAG $FAM $SEED rc=$?"
