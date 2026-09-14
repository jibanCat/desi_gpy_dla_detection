#!/bin/bash
#SBATCH --job-name=realc1
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/real_c1/runs/logs/realc1_%A_%a.log
#
# ============================== DO NOT SUBMIT ==============================
# BLIND REAL C1 PRODUCTION RUN — the frozen Paper-1 low-z model of record
# (PI ruling 2026-09-14: ADOPT B + phi_2LPT + C1nsadd + M1CUT).
#
# §17 of the ruling: "Blind real C1 = GO **after J = 8 certification passes**."
# This file is PREPARED, NOT SUBMITTED. Two preconditions must be signed off
# before anyone runs `sbatch` on it:
#   (1) the J = 8 mock sampler-health certification has PASSED (ruling §15);
#   (2) `validation/real_c1/run_real_c1.py` EXISTS. It does not yet. The
#       frozen runner `validation/fp_ladder/run_ladder.py` is MOCK-ONLY: it
#       refuses any pack whose `truth_counts` is the all-zero real sentinel
#       (its "MOCK GATE"), and its read-out is built entirely from
#       `forward_selftest.truth_f` / `perz_recovery`, which do not exist on
#       real data. See governance/final_campaign_2026-09-13/
#       REAL_C1_READINESS.md §6 for the required delta.
# ===========================================================================
#
# Layout: 16 tasks = 8 Lambda imputations x 2 seeds. One task = one
# (seed, imputation j) pair; the science posterior is the equal-weight pool
# over the J = 8 imputations at each seed (M1CUT cut-feedback, ruling §14).
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
I=$L/real_c1_inputs                 # the prepared blind inputs
OUT=$L/real_c1/runs
mkdir -p $OUT/logs

RUNNER=$W/validation/real_c1/run_real_c1.py
[ -f "$RUNNER" ] || { echo "BLOCKED: $RUNNER does not exist (see REAL_C1_READINESS.md §6). Refusing."; exit 3; }

# --- the pack of record, staged read-only beside its support sidecar -------
PACK=$I/C1_pack.npz
PACK_SHA=6865781e37a7463de72c7289ea873cbde315b0438dbb25491f9afb256768848a
echo "$PACK_SHA  $PACK" | sha256sum -c - || { echo "BLOCKED: real pack sha mismatch"; exit 4; }
sha256sum -c $I/SHA256SUMS || { echo "BLOCKED: prepared inputs failed their SHA256SUMS"; exit 5; }

SEEDS=(20260811 20260812)
J=8
i=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((i / J))]}
j=$((i % J))

echo "seed=$SEED imputation=$j/$J code=$(git -C $W rev-parse HEAD) host=$(hostname) start=$(date)"

/usr/bin/time -v python $RUNNER \
  --pack $PACK \
  --ladder M1CUT \
  --seed $SEED --chains 4 --warmup 1500 --samples 1000 --target-accept 0.95 \
  --require-support \
  --mg-fixed-file $I/Mg_B_real.npz --mg-fixed-key Mg \
  --c-fixed-file  $I/C_C1nsadd_real.npz \
  --extra-fixed-file $I/mu_extra_P6bcal_real.npz \
  --lam-imputations $J --lam-imputation $j \
  --stage REAL_C1_BLIND \
  --out $OUT/RUN_REALC1_s${SEED}_j${j}.json

echo "done seed=$SEED j=$j rc=$? end=$(date)"
#
# NOTE ON --fp-a0: OMITTED deliberately. The record value is the Perks
# pseudo-count a0 = 1/K (ruling §9); `run_ladder.py --fp-a0` defaults to None,
# which `fp_ladder.perks_log_share` resolves to exactly 1/K. Passing a value
# here would be the a0 SENSITIVITY battery, not the run of record.
