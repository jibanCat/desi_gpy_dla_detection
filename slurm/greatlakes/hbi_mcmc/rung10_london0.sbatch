#!/bin/bash
#SBATCH --job-name=modelA_rung10_london0
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs/rung10_london0_%j.log

# BLAS oversubscription guard (project standing rule)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

cd /home/mfho/hbi_mcmc_wt
conda run --no-capture-output -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc.run_rung9 \
  --pack /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs/modelA_pack_london0.npz \
  --out  /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs/rung10_london0.json \
  --warmup 1500 --samples 1000 --chains 4 --seed 0 --target-accept 0.95 \
  --allow-low-farr "on-mock self-calibration: ratio 1.67 structural (calibration set == the mock under fit); finite-calibration variance sampled via psi_C"
