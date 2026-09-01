#!/bin/bash
#SBATCH --job-name=hbiv_chain
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
# 2026-09-02 HBI identifiability campaign (science-lane VALIDATION runs; never a production
# artifact). One cc_real_posterior arm per array task, configuration = the frozen CP-3
# production battery (run_cp3_real_battery.sbatch / run_cp3_deep.sbatch of record) plus
# the single stated intervention in $EXTRA, plus the additive draw saves.
#   env: RUN_ID OUT SEEDS_FILE WARMUP DEEP(0/1) EXTRA REPO PACK
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla-hbi
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
mapfile -t SEEDS < "$SEEDS_FILE"
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
TAG=$([ "${DEEP:-0}" = "1" ] && echo "deep_" || echo "")
B=$OUT/REAL_ln_${TAG}s${S}
echo "run=$RUN_ID seed=$S deep=${DEEP:-0} warmup=$WARMUP extra='${EXTRA:-}' code=$(git rev-parse HEAD) dirty=$(git status --porcelain -uno | wc -l) host=$(hostname) start=$(date -Is)"
python3 -m CDDF_analysis.hbi_mcmc.cc_real_posterior \
  --pack "$PACK" \
  --samples 500 --warmup "$WARMUP" --chains 2 --target-accept 0.95 \
  --fp-mode informative_ln --seed "$S" ${EXTRA:-} \
  --save-nuisance-draws ${B}_nuisance.npz --save-all-sites ${B}_allsites.npz \
  --out ${B}.json > ${B}.stdout 2>&1
RC=$?
echo "done run=$RUN_ID seed=$S deep=${DEEP:-0} rc=$RC end=$(date -Is)"
exit $RC
