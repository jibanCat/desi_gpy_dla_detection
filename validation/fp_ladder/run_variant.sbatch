#!/bin/bash
#SBATCH --job-name=absladder
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/mfho/lowz_clean_work_2026-09-12/absorber_ladder/logs/%x_%A_%a.log
# First absorber-side ladder (sealed ABSORBER_LADDER_PREDECLARATION). Env: VARIANT (label), SEED, FIX (e.g. "P"),
# LADDER (ORACLE|M1CUT), MGFILE_TPL / CFILE_TPL / EXTRAFILE_TPL (paths with <fam> placeholder, or empty), LAMJ, LAMI.
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla-hbi
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/mfho/wt_abs_diag_2026-09
L=/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13
cd /home/mfho/wt_abs_diag_2026-09
FAMS=(2lpt0 london0 saclay0); F=${FAMS[$SLURM_ARRAY_TASK_ID]}
PACK=$L/support/scanpack_${F}_b300_A0.npz; CENSUS=$L/support/fp_census_${F}_A0v2.npz; OPS=$L/support/empirical_ops_${F}_A0.npz
ARGS=""
[ -n "${MGFILE_TPL:-}" ] && ARGS="$ARGS --mg-fixed-file ${MGFILE_TPL//<fam>/$F}"
[ -n "${CFILE_TPL:-}" ] && ARGS="$ARGS --c-fixed-file ${CFILE_TPL//<fam>/$F}"
[ -n "${EXTRAFILE_TPL:-}" ] && ARGS="$ARGS --extra-fixed-file ${EXTRAFILE_TPL//<fam>/$F}"
[ -n "${FIX:-}" ] && ARGS="$ARGS --fix $FIX"
[ "${LADDER:-ORACLE}" = "M1CUT" ] && ARGS="$ARGS --lam-imputations ${LAMJ:-1} --lam-imputation ${LAMI:-0}"
mkdir -p $L/runs/$VARIANT
echo "variant=$VARIANT fam=$F seed=$SEED ladder=${LADDER:-ORACLE} args=[$ARGS] code=$(git rev-parse HEAD) pack=$(sha256sum $PACK | cut -c1-16) census=$(sha256sum $CENSUS | cut -c1-16) start=$(date)"
/usr/bin/time -v python3 validation/fp_ladder/run_ladder.py \
  --pack $PACK --ladder ${LADDER:-ORACLE} --seed $SEED --chains 4 --warmup 1500 --samples 1000 --target-accept 0.95 \
  --census $CENSUS --ops $OPS --require-support --stage LADDER_$VARIANT $ARGS \
  --out $L/runs/$VARIANT/RUN_${VARIANT}_${F}_s${SEED}$( [ "${LADDER:-ORACLE}" = "M1CUT" ] && echo "_j${LAMI:-0}" ).json
echo "done variant=$VARIANT fam=$F seed=$SEED rc=$? end=$(date)"
