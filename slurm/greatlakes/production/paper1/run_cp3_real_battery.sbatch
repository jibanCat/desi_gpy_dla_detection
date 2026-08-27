#!/bin/bash
#SBATCH --job-name=cp3_real
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/cp3_real/logs/cp3_real_%A_%a.log
# PAPER-1 PRODUCT 3 OF RECORD (CP-3 real battery, 8 seeds 20260821..28). Committed copy 2026-08-26; unchanged.
# Runs at code b59e0b5 (frozen artifact); seeds/warmup/chains now stamped into each run JSON (run_config).
# CP-3 (PI ruling 2026-08-21 #17; predeclaration notes 7c74dd7): corrected candidate-grade
# real posterior — ckpt-10.10 configuration and seeds UNCHANGED. Outputs stay in scratch.
set -uo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla-hbi
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/mfho/wt_forward_2026_08
D=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821
# CP3_OUT: output dir override for clean reproductions (PI ruling 6, 2026-08-26); default = the run dir of record.
R=${CP3_OUT:-$D/cp3_real}; mkdir -p $R
SEEDS=(20260821 20260822 20260823 20260824 20260825 20260826 20260827 20260828)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "seed=$S code=$(git rev-parse --short HEAD) start=$(date)"
python3 -m CDDF_analysis.hbi_mcmc.cc_real_posterior \
  --pack $D/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.npz \
  --samples 500 --warmup 1500 --chains 2 --target-accept 0.95 \
  --fp-mode informative_ln --seed $S \
  --out $R/REAL_ln_s${S}.json > $R/REAL_ln_s${S}.stdout 2>&1
echo "done $S rc=$? end=$(date)"
