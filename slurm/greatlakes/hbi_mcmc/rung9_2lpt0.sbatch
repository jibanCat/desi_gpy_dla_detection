#!/bin/bash
#SBATCH --job-name=modelA_rung9_2lpt0
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs/rung9_2lpt0_%j.log

# BLAS oversubscription guard (project standing rule)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

cd /home/mfho/hbi_mcmc_wt
conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc.run_rung9 \
  --pack /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs/modelA_pack_2lpt0.npz \
  --out  /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs/rung9_2lpt0.json \
  --warmup 1000 --samples 1000 --chains 4 --seed 0 \
  --allow-low-farr "on-mock self-calibration: ratio 1.67 structural (calibration set == the mock under fit); finite-calibration variance sampled via psi_C"
