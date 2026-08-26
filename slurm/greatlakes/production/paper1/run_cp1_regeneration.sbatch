#!/bin/bash
#SBATCH --job-name=cp1_gfix
#SBATCH --account=cavestru0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/logs/cp1_gfix_%j.log
# ---------------------------------------------------------------------------
# PAPER-1 PRODUCT 1 OF RECORD: certified corrected-g pack construction (CP-1).
# Committed copy (Paper-1 code review 2026-08-26) of the sbatch archived beside the
# artifact and in the notes repo. EXECUTION HISTORY OF THE FROZEN PRODUCT: job 58432225
# ran steps 1-2 at code ebf3787 and stopped at the step-3 recipe regression; job 58432277
# resumed with START_STEP=3 at code 0babe21 (documentation-only diff). The mock packs
# therefore carry ebf3787, the real pack 0babe21. This copy differs from the archived
# script in two ways for FUTURE runs: (i) the contract-guard step fails CLOSED (the
# archived `|| echo GUARD FAIL` masked a G_A verdict that is now reported as
# NOT_APPLICABLE on real packs by contract_guards_check); (ii) START_STEP and the code
# commit are echoed into a run manifest. Environments: gpdla for extraction/scan
# packs/certification, gpdla-hbi for the v1.2 stamp (see docs/PAPER1_REPRODUCTION.md).
# ---------------------------------------------------------------------------
# CP-1 (PI rulings 2026-08-21 items 1-2, 4, 9-11): regenerate EVERY pack through the
# committed extractor with the corrected g(N,z) support; certify. No inference.
set -euo pipefail
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/mfho/wt_forward_2026_08
echo "code=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) start=$(date)"
BASE=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata
V1NEW=$BASE/adopted_packs_gfix_v1_20260821
V2NEW=$BASE/adopted_packs_v2p2_20260821
NEW=$BASE/real_pack_v2_20260821
REGR=$NEW/recipe_regression_on_v2p1
OLDV2=$BASE/adopted_packs_v2p1_20260817
OLDREAL=$BASE/real_pack_v1
DIAG=$OLDREAL/diag_20260820_gsupport
TAG=bw0p2_pad19p0_molly172
START_STEP=${START_STEP:-1}
echo "RUN_MANIFEST start_step=$START_STEP code=$(git rev-parse --short HEAD) date=$(date -Is)" | tee -a "$NEW/run_manifest.txt"
step() { echo "=== STEP $1 $(date)"; }
skip() { [ "${1%% *}" -lt "$START_STEP" ]; }

if ! skip 1; then
step "1 extract (gpdla, jax-free): corrected frozen calibration + 3 mock packs"
conda activate gpdla
python CDDF_analysis/hbi_mcmc/extract_pack.py --mocks 2lpt0 london0 saclay0 \
  --out-dir $V1NEW --basis-pad-floor 19.0 --completeness-below-floor molly172 \
  --basis-width 0.2 --tag _${TAG} > $V1NEW/extract.stdout 2>&1
tail -3 $V1NEW/extract.stdout
fi
if ! skip 2; then
step "2 stamp v1.2 (gpdla-hbi)"
conda activate gpdla-hbi
python -m CDDF_analysis.hbi_mcmc.upgrade_packs_v2 --srcdir $V1NEW --outdir $V2NEW
fi

step "3 scan-pack recipe PROOF (gpdla — byte-identity is numpy-version sensitive at the ULP level): old v2p1 inputs must reproduce the existing b300 scan packs byte-for-byte"
conda activate gpdla
python CDDF_analysis/hbi_mcmc/build_scan_packs.py --src-dir $OLDV2 --out-dir $REGR --regress-against $OLDREAL

step "4 scan packs from the corrected v2p2 packs"
python CDDF_analysis/hbi_mcmc/build_scan_packs.py --src-dir $V2NEW --out-dir $NEW

step "5 real-pack machinery certification vs the NEW 2LPT-0 reference"
python CDDF_analysis/hbi_mcmc/extract_pack_real.py --cert-2lpt0 --out-dir $NEW \
  --ref-pack $V2NEW/modelA_pack_2lpt0_${TAG}_v2.npz

step "6 real pack extraction (corrected frozen calibration; data plane unchanged)"
python CDDF_analysis/hbi_mcmc/extract_pack_real.py --real --out-dir $NEW \
  --ref-pack $V2NEW/modelA_pack_2lpt0_${TAG}_v2.npz

step "7 real pack v1.2 stamp (gpdla-hbi, as the 2026-08-17 stamp)"
conda activate gpdla-hbi
python CDDF_analysis/hbi_mcmc/extract_pack_real.py --stamp-v12 --out-dir $NEW

step "8 contract guards on every regenerated pack"
for p in $V2NEW/modelA_pack_2lpt0_${TAG}_v2.npz $V2NEW/modelA_pack_london0_${TAG}_v2.npz $V2NEW/modelA_pack_saclay0_${TAG}_v2.npz $NEW/modelA_pack_REAL_loa50k_c3300_${TAG}_v2.npz; do
  python -m CDDF_analysis.hbi_mcmc.contract_guards_check --pack $p --json ${p%.npz}.contract_guards.json || { echo "GUARD FAIL $p"; exit 1; }
done

step "9 certification record (CP-1 a-d; d pending CP-2)"
conda activate gpdla
python CDDF_analysis/hbi_mcmc/certify_g_support.py \
  --new $V2NEW/modelA_pack_2lpt0_${TAG}_v2.npz $V2NEW/modelA_pack_london0_${TAG}_v2.npz $V2NEW/modelA_pack_saclay0_${TAG}_v2.npz \
        $NEW/scanpack_2lpt0_b300.npz $NEW/scanpack_london0_b300.npz $NEW/scanpack_saclay0_b300.npz \
        $NEW/modelA_pack_REAL_loa50k_c3300_${TAG}_v2.npz \
  --old $OLDV2/modelA_pack_2lpt0_${TAG}_v2.npz $OLDV2/modelA_pack_london0_${TAG}_v2.npz $OLDV2/modelA_pack_saclay0_${TAG}_v2.npz \
        $OLDREAL/scanpack_2lpt0_b300.npz $OLDREAL/scanpack_london0_b300.npz $OLDREAL/scanpack_saclay0_b300.npz \
        $OLDREAL/modelA_pack_REAL_loa50k_c3300_${TAG}_v2.npz \
  --diag $DIAG/scanpack_2lpt0_b300_DIAGPACK_gcons.npz $DIAG/scanpack_london0_b300_DIAGPACK_gcons.npz $DIAG/scanpack_saclay0_b300_DIAGPACK_gcons.npz \
  --calib-index 0 --floor 19.0 --tol-rows 2 --out $NEW/CERT_G_SUPPORT_CP1.json
echo "CP1_DONE end=$(date)"
