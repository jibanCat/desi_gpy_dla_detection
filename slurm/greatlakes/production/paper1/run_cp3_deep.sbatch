#!/bin/bash
#SBATCH --job-name=cp3_deep
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/cp3_real/logs/cp3_deep_%A_%a.log
# PAPER-1 PRODUCT 3 OF RECORD (CP-3 deep-adaptation reruns, warmup 3000). Committed copy 2026-08-26; unchanged.
# The seeds of record are 20260821 20260823 20260826 20260828 (selection_stage1.txt); the archived
# deep_seeds.txt was truncated to 0 bytes by the stage-2 collector -- the committed collector keeps
# deep_seeds_of_record.txt. Ran at code ea4c7bb.
# CP-3 predeclared deep-adaptation rerun: same seed, --warmup 3000, everything else identical.
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla-hbi
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/mfho/wt_forward_2026_08
D=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821
# CP3_OUT: output dir override for clean reproductions (PI ruling 6, 2026-08-26); default = the run dir of record.
R=${CP3_OUT:-$D/cp3_real}; mkdir -p $R
mapfile -t SEEDS < $R/deep_seeds.txt
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "deep seed=$S code=$(git rev-parse --short HEAD) start=$(date)"
python3 -m CDDF_analysis.hbi_mcmc.cc_real_posterior \
  --pack $D/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.npz \
  --samples 500 --warmup 3000 --chains 2 --target-accept 0.95 \
  --fp-mode informative_ln --seed $S \
  --out $R/REAL_ln_deep_s${S}.json > $R/REAL_ln_deep_s${S}.stdout 2>&1
echo "done deep $S rc=$? end=$(date)"
