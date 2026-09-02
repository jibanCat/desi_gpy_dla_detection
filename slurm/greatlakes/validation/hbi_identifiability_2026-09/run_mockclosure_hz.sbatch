#!/bin/bash
#SBATCH --job-name=hz_mockclosure
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
# High-z HBI extension trial — OPERATOR CLOSURE on the 2LPT random arm (mock-only runner; predeclaration §4): env PACK OUT REPO SEED
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh; conda activate gpdla-hbi
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"; mkdir -p "$OUT"
echo "mockclosure pack=$PACK seed=$SEED code=$(git rev-parse HEAD) dirty=$(git status --porcelain -uno | wc -l) start=$(date -Is)"
python3 -m CDDF_analysis.hbi_mcmc.cc_posterior_validation --pack "$PACK" --allow-nonstandard-grid --samples 500 --warmup 1500 --chains 2 --target-accept 0.95 \
  --fp-mode informative_ln --seed "$SEED" --save-fdraws --out "$OUT/mockclosure_s${SEED}.json" > "$OUT/mockclosure_s${SEED}.stdout" 2>&1
RC=$?; echo "done rc=$RC end=$(date -Is)"; exit $RC
