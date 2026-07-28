#!/bin/bash
#SBATCH --job-name=modelA_posterior
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs/posterior_%x_%j.log

# ==========================================================================
# PRODUCTION SCALE for the paper-facing posterior estimator
# (CDDF_analysis/hbi_mcmc/run_posterior.py — DLA + sub-DLA, one posterior).
#
#   *** PREPARED AND VALIDATED, NOT SUBMITTED (2026-07-28). ***
#
# ---------------------------------------------------------------------------
# WHY THIS IS SAFE TO LEAVE LYING AROUND
# ---------------------------------------------------------------------------
# run_posterior FAIL-CLOSES on the forward-model truth-fold gate BEFORE the
# sampler is constructed, and it exits NONZERO when the gate fails.  (This is
# the difference from slurm/greatlakes/hbi_mcmc/rung9v3_2lpt0.sbatch, whose
# advertised "fail-closed pre-flight" calls forward_selftest, which has no
# closure assertion and exits 0 on the failing pack.)  As of today EVERY real
# mock pack FAILS the gate:
#
#   modelA_pack_2lpt0_v11.npz   total mu/obs 0.7312  z=+93.3
#                               lowest n-hat bin    0.1655  z=+216.4
#                               chi2/dof over 29 reported bins = 2216
#
# so submitting this against $PACK below aborts in ~30 s having spent no
# sampler time.  That is the intended behaviour, not a bug.
#
# ---------------------------------------------------------------------------
# THE GATE (do NOT submit until BOTH are true)
# ---------------------------------------------------------------------------
#   [ ] A BASIS-PADDED pack exists (finding D1).  The true-N basis must extend
#       to ~18.5 with truth_counts/truth_counts_bks rebuilt on that grid (the
#       DETECTION window must not move) and a stated completeness convention in
#       the pad.  $PACK below deliberately names a file that DOES NOT EXIST so
#       this script cannot silently run on the unpadded pack.
#   [ ] run_posterior's gate PASSES on it (it is run automatically; you will
#       see "[gate] forward-model closure: PASS" in the log).
#
# Until then the ONLY admissible production run is the SYNTHETIC one below
# (MODE=synthetic), where closure is provable and is proved on every run.
#
# ---------------------------------------------------------------------------
# COMPUTE ESTIMATE  (empirical anchor: SLURM 53363263, the rung-9 v2 run,
# 1500/1000/4 vectorized on the 29 x 15 x 8 grid = 16.15 h wall on 8 CPUs
# = 129 CPU-h, MaxRSS 1.76 GB)
# ---------------------------------------------------------------------------
#   MODE=synthetic (29 x 15 x 8, unpadded)      ~16 h wall   ~130 CPU-h
#   MODE=mock      (padded to ~18.5, B: 29->39) ~22 h wall   ~180 CPU-h
#       the fold's dominant cost is the (B x C) kernel contraction, so the
#       pad scales it ~B_new/B_old = 1.34, plus ~10 extra eps_N and
#       10 x (Kf-1) = 140 extra eps_z coordinates in the NUTS geometry.
#   Worst case at the 36 h wall limit:          288 CPU-h.
#   Three mocks (2lpt0 / london0 / saclay0):    ~540-860 CPU-h  -> that is
#       ABOVE the ~500 CPU-h PI sign-off threshold and must be estimated and
#       signed off SEPARATELY. This script runs ONE pack.
#
# Memory: 32G is 18x the measured 1.76 GB peak; the pad raises it ~35%.
# ==========================================================================

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

PY=/home/mfho/.conda/envs/gpdla-hbi/bin/python
PACKDIR=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs
WT=/home/mfho/hbi_mcmc_wt

# "synthetic" = the production-geometry synthetic pack (closure provable).
# "mock"      = a real mock pack (BLOCKED until the padded pack exists).
MODE=${MODE:-synthetic}

# NOTE: intentionally a NON-EXISTENT path. Replace only once the padded pack
# has been extracted AND run_posterior's gate passes on it interactively.
PACK=$PACKDIR/modelA_pack_2lpt0_v12_padded.npz

WARMUP=1500
SAMPLES=1000
CHAINS=4
SEED=0
TARGET_ACCEPT=0.95   # identical to rung-9 v2, whose SAMPLER was healthy
                     # (r_hat 1.00978, ESS_bulk 554, 2 divergences / 4000).
                     # The forward model is what changed; do not re-tune the
                     # sampler in the same step or the two cannot be separated.

cd "$WT"

if [ "$MODE" = "synthetic" ]; then
  OUT=$PACKDIR/posterior_synthetic_prod_seed${SEED}.json
  SRC="--synthetic-smoke --synthetic-full-grid --synthetic-seed 0"
elif [ "$MODE" = "mock" ]; then
  OUT=$PACKDIR/posterior_2lpt0_prod.json
  SRC="--pack $PACK"
  if [ ! -f "$PACK" ]; then
    echo "REFUSING: $PACK does not exist. The basis-padded pack (finding D1)" >&2
    echo "has not been extracted; see the GATE block at the top of this file." >&2
    exit 2
  fi
else
  echo "unknown MODE=$MODE (expected 'synthetic' or 'mock')" >&2; exit 2
fi

echo "[sbatch] MODE=$MODE OUT=$OUT"
$PY -m CDDF_analysis.hbi_mcmc.run_posterior \
    $SRC \
    --out "$OUT" \
    --warmup $WARMUP --samples $SAMPLES --chains $CHAINS --seed $SEED \
    --target-accept $TARGET_ACCEPT \
    --resp-clamp both --fp-mode joint \
    --map-diagnostic \
    --allow-low-farr "on-mock SELF-calibration: the molly calibration set IS \
the mock under fit, so the Farr 4x headroom is structurally unattainable; the \
finite-calibration variance is instead SAMPLED via psi_C"

echo "[sbatch] done: $OUT"
