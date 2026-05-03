#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -J 2lpt_preload
#SBATCH -o slurm/greatlakes/2lpt_preload_%j.log
#SBATCH -e slurm/greatlakes/2lpt_preload_%j.log

# PRELOAD-ONLY: 2LPT mock → trainset.h5 + README + metadata.
# Runs on a GreatLakes CPU compute node (no GPU needed for FITS I/O).
# `-p standard` queue is the default and uncrowded.
#
# Output layout (everything in ONE folder, ready for Globus):
#   ${OUTDIR_BASE}/v2_runs/${RUN_TAG}/
#     trainset.h5
#     README.md
#     dataset_metadata.json
#     preload.slurm.log
#     (later, training adds: model_epoch_*.h5, checkpoint_*.pt,
#                            config.json, loss_history.json,
#                            train.slurm.log)
#
# After this completes:
#   sbatch --export=ALL,RUN_TAG=<same tag> slurm/greatlakes/train_only_gpu.sh
#
# VARIANTs:
#   loa0                  2LPT loa-0 uncontaminated (no DLAs/metals/BALs by mock)
#   loa124_nohcd_nobal    2LPT loa-124 contaminated; HCDs (logNHI ≥ 17) and
#                         BALs (BI_CIV > 0) anti-joined out via truth catalogs
#   loa124_nohcd_with_bal 2LPT loa-124; HCDs anti-joined out (truth-based);
#                         BALs KEPT (intended as BAL-GP training base — at
#                         training time, subset to BAL-positive TIDs via
#                         --catalog-file from mock-0/loa-124/bal_cat.fits)

set -eo pipefail
export PYTHONUNBUFFERED=1

VARIANT="${VARIANT:?must be set: loa0 | loa124_nohcd_nobal | loa124_nohcd_with_bal}"

MAX_SPECTRA="${MAX_SPECTRA:-300000}"  # full mock; preload caps to whatever's in zcat after filters
Z_MIN="${Z_MIN:-2.0}"
Z_MAX="${Z_MAX:-4.0}"   # 2LPT z range is ~1.8–3.8

DATA_BASE="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0"
OUTDIR_BASE="${OUTDIR_BASE:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection}"

case "$VARIANT" in
    loa0)
        MOCK_DIR="$DATA_BASE/loa-0"
        EXTRA_FLAGS=""
        ;;
    loa124_nohcd_nobal)
        MOCK_DIR="$DATA_BASE/loa-124"
        EXTRA_FLAGS="--exclude-hcd --exclude-bal"
        ;;
    loa124_nohcd_with_bal)
        MOCK_DIR="$DATA_BASE/loa-124"
        EXTRA_FLAGS="--exclude-hcd"
        ;;
    *)
        echo "[error] VARIANT must be loa0 | loa124_nohcd_nobal | loa124_nohcd_with_bal" >&2
        exit 2
        ;;
esac

[ -d "$MOCK_DIR" ] || { echo "[error] MOCK_DIR not found: $MOCK_DIR" >&2; exit 3; }
[ -r "$MOCK_DIR/zcat.fits" ] || { echo "[error] zcat.fits not in $MOCK_DIR" >&2; exit 4; }

RUN_TAG="${RUN_TAG:-2lpt_${VARIANT}_${SLURM_JOB_ID}}"
RUN_DIR="${RUN_DIR:-${OUTDIR_BASE}/v2_runs/${RUN_TAG}}"
TRAINSET_H5="${RUN_DIR}/trainset.h5"
mkdir -p "$RUN_DIR"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

echo "===================================================="
echo "  2LPT PRELOAD ONLY  variant: $VARIANT  job: $SLURM_JOB_ID"
echo "  GreatLakes -p standard -c 8  walltime 4 h"
echo "===================================================="
echo "  mock_dir:    $MOCK_DIR"
echo "  RUN_DIR:     $RUN_DIR"
echo "  filter:      ${EXTRA_FLAGS:-(none — uncontaminated)}"
echo "  scale:       max_spectra=$MAX_SPECTRA z [$Z_MIN, $Z_MAX]"
echo "===================================================="

python -u preload_spectra/preload_2lpt_simple.py \
    --mock-dir "$MOCK_DIR" \
    --output "$TRAINSET_H5" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --max-spectra "$MAX_SPECTRA" \
    $EXTRA_FLAGS

[ -r "$TRAINSET_H5" ] || { echo "[error] trainset.h5 not produced" >&2; exit 6; }
echo "preload wrote: $TRAINSET_H5 ($(du -h "$TRAINSET_H5" | cut -f1))"

cp "slurm/greatlakes/2lpt_preload_${SLURM_JOB_ID}.log" "$RUN_DIR/preload.slurm.log" 2>/dev/null || true

echo
echo "===================================================="
echo "  PRELOAD COMPLETE  $RUN_TAG"
echo "  RUN_DIR:   $RUN_DIR"
echo
echo "  NEXT STEP: train on this dataset"
echo "  sbatch --export=ALL,RUN_TAG=$RUN_TAG slurm/greatlakes/train_only_gpu.sh"
echo "===================================================="
