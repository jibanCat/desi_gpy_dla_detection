#!/bin/bash
#SBATCH --job-name=modelA_rung9v3_2lpt0
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs/rung9v3_2lpt0_%j.log

# ==========================================================================
# rung 9 v3 — PREPARED, NOT SUBMITTED (2026-07-28).
#
# GATE (do NOT submit until BOTH are true):
#   [ ] the pack is BASIS-PADDED (finding D1). rung9_2lpt0_v2 failed closure
#       because the true-N basis stopped at the reporting floor 19.5 while the
#       observed n-hat bins just above it are fed overwhelmingly by TRUE
#       systems BELOW it. modelA_pack_2lpt0_v11.npz fixes only D2. The pad
#       needs a RE-EXTRACTION emitting, on a true-N grid down to ~18.5:
#         - ntrue_edges  (schema v1.1 now permits the downward pad)
#         - truth_counts / truth_counts_bks on the padded grid
#             (extract_pack.build_truth_counts: drop the >=19.5 floor on the
#              TRUTH histogram only; the detection window is unchanged)
#         - molly completeness on the padded range, or an explicit statement
#           that the committed constant-extrapolation-of-cell-0 convention
#           (cddf_catalog_hbi.py:6117 basis_pad_floor block) is adopted
#       and it must PASS forward_selftest before any sampler time is spent.
#   [ ] CDDF_analysis/hbi_mcmc/forward_selftest.py on that pack closes the
#       per-n-hat ratio table to within Poisson over [19.7, 22.4).
#
# CHANGES vs rung9v2_2lpt0.sbatch:
#   * --pack   -> the v1.1 (resp_N_fit_range-carrying) pack; the fold now
#                 FAIL-CLOSES on a pack without it (finding D2).
#   * --out    -> rung9_2lpt0_v3.json (v2 is RETAINED, never overwritten).
#   * sampler config is otherwise IDENTICAL to v2 (1500/1000/4, ta 0.95):
#     v2's sampler was HEALTHY (r_hat 1.00978, ESS_bulk 554, 2 divergences);
#     the failure was the forward model, so the sampler must not be re-tuned
#     in the same step or the two changes cannot be separated.
#   * --time 36:00:00 (v2's first attempt outran a 16h limit).
# ==========================================================================

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

PACKDIR=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs
PACK=$PACKDIR/modelA_pack_2lpt0_v11.npz     # <- replace with the PADDED pack
OUT=$PACKDIR/rung9_2lpt0_v3.json

cd /home/mfho/hbi_mcmc_wt

# fail-closed pre-flight: the forward model must close BEFORE the sampler runs
conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc.forward_selftest \
  --pack "$PACK" --out "${OUT%.json}_selftest.json" || exit 1

conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc.run_rung9 \
  --pack "$PACK" \
  --out  "$OUT" \
  --warmup 1500 --samples 1000 --chains 4 --seed 0 --target-accept 0.95 \
  --allow-low-farr "on-mock self-calibration: ratio 1.67 structural (calibration set == the mock under fit); finite-calibration variance sampled via psi_C"
