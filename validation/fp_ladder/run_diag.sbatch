#!/bin/bash
#SBATCH --job-name=absdiag
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/mfho/lowz_clean_work_2026-09-12/absorber_diag/logs/%x_%A_%a.log
# Absorber-side truth-replacement DIAGNOSTICS (sealed DIAG_PREDECLARATION 2f585d1a). Env: CONFIG (OP, OC, OCz, OM, OCM, OE), SEED.
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla-hbi
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/mfho/wt_abs_diag_2026-09
S=/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12
D=/scratch/cavestru_root/cavestru0/mfho/absorber_diag_2026-09-13
cd /home/mfho/wt_abs_diag_2026-09
FAMS=(2lpt0 london0 saclay0); F=${FAMS[$SLURM_ARRAY_TASK_ID]}
case $CONFIG in
  OP) FIX="P";; OC) FIX="P,C";; OCz) FIX="P,Cz";; OM) FIX="P,M";; OCM) FIX="P,C,M";; OE) FIX="P,E";; *) echo "bad CONFIG"; exit 2;;
esac
mkdir -p $D/$CONFIG
OPSARG=""; OPSSHA="none"
if [ "$CONFIG" != "OP" ]; then OPSARG="--ops $D/ops/empirical_ops_${F}.npz"; OPSSHA=$(sha256sum $D/ops/empirical_ops_${F}.npz | cut -c1-16); fi
echo "config=$CONFIG fix=$FIX fam=$F seed=$SEED code=$(git rev-parse HEAD) ops=$OPSSHA census=$(sha256sum $S/census/fp_census_${F}.npz | cut -c1-16) start=$(date)"
/usr/bin/time -v python3 validation/fp_ladder/run_ladder.py \
  --pack $S/packs/scanpack_${F}_b300.npz --ladder ORACLE --seed $SEED \
  --chains 4 --warmup 1500 --samples 1000 --target-accept 0.95 \
  --census $S/census/fp_census_${F}.npz $OPSARG --fix $FIX --stage DIAG_$CONFIG \
  --out $D/$CONFIG/RUN_${CONFIG}_${F}_s${SEED}.json
echo "done config=$CONFIG fam=$F seed=$SEED rc=$? end=$(date)"
