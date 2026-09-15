#!/bin/bash
#SBATCH --job-name=s6prop
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/s6_propagation/runs/logs/s6_%A_%a.log
#
# S6 COMPLETENESS CALIBRATION-UNCERTAINTY PROPAGATION — the ONLY additional
# science computation authorised by PI ruling 2026-09-14b §9, executed under the
# sealed predeclaration S6_COMPLETENESS_COVARIANCE_PROPAGATION_PREDECLARATION.md
# (e36cac98).
#
# 9 tasks:
#   i = 0..7  the FROZEN model run with the completeness table rebuilt at
#             beta_bootstrap[i] (indices 0..7 of the released 200-realisation
#             parity bootstrap, STORED ORDER, sealed a priori)
#   i = 8     the SAME configuration at the frozen beta-hat (the frozen table
#             C_C1nsadd_real.npz), i.e. the reference point of the linear fit.
#             It is run because the record real runs are J = 8 and the closest
#             imputation (j = 3) has Lambda 1.66 % from the J = 1 median, i.e.
#             > the 1 % tolerance in the task spec; the predeclaration's
#             "Lambda at the J = 1 median" is therefore realised explicitly.
#
# EVERY other argument is identical to the record real runs
# (validation/real_c1/run_real_c1.sbatch): M1CUT, seed 20260811, 4 x (1500 +
# 1000), target_accept 0.95, --require-support, Mg_B_real, the transported
# sub-floor mu_extra, a0 = 1/K (--fp-a0 deliberately omitted), --quiet-values.
# Nothing frozen is changed; beta is never updated with real data.
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
I=$L/real_c1_inputs
S=$L/s6_propagation
OUT=$S/runs
mkdir -p $OUT/logs

RUNNER=$W/validation/real_c1/run_real_c1.py
[ -f "$RUNNER" ] || { echo "BLOCKED: $RUNNER does not exist. Refusing."; exit 3; }
IDENT=$L/real_c1/identity/IDENTITY_VERDICT.json
[ -f "$IDENT" ] && grep -q '"PASS": true' "$IDENT" || { echo "BLOCKED: $IDENT missing or not PASS. Refusing."; exit 6; }

PACK=$I/C1_pack.npz
PACK_SHA=6865781e37a7463de72c7289ea873cbde315b0438dbb25491f9afb256768848a
echo "$PACK_SHA  $PACK" | sha256sum -c - || { echo "BLOCKED: real pack sha mismatch"; exit 4; }
sha256sum -c $I/SHA256SUMS || { echo "BLOCKED: prepared inputs failed their SHA256SUMS"; exit 5; }
( cd $S && sha256sum -c SHA256SUMS ) || { echo "BLOCKED: S6 draw tables failed their SHA256SUMS"; exit 7; }

i=$SLURM_ARRAY_TASK_ID
SEED=20260811
if [ "$i" -lt 8 ]; then
  CFILE=$S/tables/C_S6_draw${i}.npz
  TAG=draw${i}
else
  CFILE=$I/C_C1nsadd_real.npz          # the FROZEN beta-hat table, unchanged
  TAG=betahat
fi
[ -f "$CFILE" ] || { echo "BLOCKED: $CFILE missing"; exit 8; }

echo "s6 task=$i tag=$TAG c_fixed=$CFILE seed=$SEED code=$(git -C $W rev-parse HEAD) host=$(hostname) start=$(date)"

/usr/bin/time -v python $RUNNER \
  --pack $PACK \
  --ladder M1CUT \
  --seed $SEED --chains 4 --warmup 1500 --samples 1000 --target-accept 0.95 \
  --require-support \
  --mg-fixed-file $I/Mg_B_real.npz --mg-fixed-key Mg \
  --c-fixed-file  $CFILE \
  --extra-fixed-file $I/mu_extra_P6bcal_real.npz \
  --lam-imputations 1 --lam-imputation 0 \
  --quiet-values \
  --stage S6_COMPLETENESS_PROPAGATION \
  --out $OUT/RUN_S6_${TAG}.json

echo "done task=$i tag=$TAG rc=$? end=$(date)"
