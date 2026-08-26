#!/bin/bash
#SBATCH --job-name=cp2_gate
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/cp2_validation/cp2_gate_%A_%a.log
# PAPER-1 PRODUCT 2 OF RECORD (CP-2 per-z gate v2, production packs; seed 20260811 = the Battery-2/3 seed).
# Committed copy 2026-08-26 of the archived sbatch; unchanged. Env gpdla-hbi. ~11 min per family.
# CP-2 (PI ruling 2026-08-21 item 11): mock posterior validation on the REGENERATED
# production scan packs, one run per family at a Battery-2/3 seed (20260811) so the
# record must BIT-REPRODUCE the DIAGPACK run if (and only if) the arrays are identical.
# Same configuration as Batteries 2-4. Mock-only; nothing adopted by this job.
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla-hbi
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/mfho/wt_forward_2026_08
NEW=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821
OUT=$NEW/cp2_validation
FAMS=(2lpt0 london0 saclay0); fam=${FAMS[$SLURM_ARRAY_TASK_ID]}; S=20260811
echo "family=$fam seed=$S code=$(git rev-parse --short HEAD) start=$(date)"
python3 -m CDDF_analysis.hbi_mcmc.cc_posterior_validation \
  --pack $NEW/scanpack_${fam}_b300.npz --samples 500 --warmup 1500 --chains 2 \
  --fp-mode informative_ln --target-accept 0.95 --seed $S --save-fdraws \
  --out $OUT/cp2_ln_w1500_${fam}_s${S}.json > $OUT/cp2_ln_w1500_${fam}_s${S}.stdout 2>&1
echo "done $fam rc=$? end=$(date)"
