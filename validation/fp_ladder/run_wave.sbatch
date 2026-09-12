#!/bin/bash
#SBATCH --job-name=fpladder
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/mfho/lowz_clean_work_2026-09-12/fp_ladder/logs/%x_%A_%a.log
# FP-model ladder, MOCK-ONLY (sealed predeclaration 3112022a...). Env: LADDER (M0..M5), WAVE (dir name).
# array index -> (family, seed): 0-5 = 2lpt0/london0/saclay0 x seeds 20260811/20260812
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla-hbi
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/mfho/wt_fp_ladder_2026-09
S=/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12
cd /home/mfho/wt_fp_ladder_2026-09
FAMS=(2lpt0 2lpt0 london0 london0 saclay0 saclay0)
SEEDS=(20260811 20260812 20260811 20260812 20260811 20260812)
i=$SLURM_ARRAY_TASK_ID; F=${FAMS[$i]}; SD=${SEEDS[$i]}
mkdir -p $S/$WAVE
echo "ladder=$LADDER fam=$F seed=$SD chains=4 warmup=1500 samples=1000 code=$(git rev-parse HEAD) pack=$(sha256sum $S/packs/scanpack_${F}_b300.npz | cut -c1-16) host=$(hostname) start=$(date)"
/usr/bin/time -v python3 validation/fp_ladder/run_ladder.py \
  --pack $S/packs/scanpack_${F}_b300.npz --ladder $LADDER --seed $SD \
  --chains 4 --warmup 1500 --samples 1000 --target-accept 0.95 \
  --census $S/census/fp_census_${F}.npz --stage $WAVE \
  --out $S/$WAVE/RUN_${LADDER}_${F}_s${SD}.json
echo "done ladder=$LADDER fam=$F seed=$SD rc=$? end=$(date)"
