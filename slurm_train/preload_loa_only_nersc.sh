#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --job-name=loa_preload
#SBATCH --output=slurm_train/loa_preload_%j.log
#SBATCH --error=slurm_train/loa_preload_%j.err
#SBATCH -A desi
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Walltime history: original 6h was insufficient for the bal_only
# variant — jobs 52338154 (2026-05-01) and 52367927 (2026-05-02) BOTH
# hit TIME LIMIT after processing only ~7h of work, leaving an empty
# output directory because the writer doesn't flush partial trainsets.
# Root cause: the BAL-fiber selection happens AFTER the per-coadd read,
# so the I/O budget is identical to a full-LOA preload (~16k healpixes
# × ~3-5s/coadd ≈ 18-20h). Bumped to 24h (NERSC -q regular max for
# ≤16 nodes). The other variants (no_dla_no_bal, no_hcd_with_bal,
# no_hcd_no_bal) drop more fibers per coadd so they completed in 6h —
# but to keep one walltime for all variants, 24h is safe across the
# board.

# PRELOAD-ONLY: real LOA → trainset.h5 + README + metadata.
# Runs on CPU (no GPU needed for FITS I/O), regular queue
# (CPU regular ≪ GPU regular wait time).
#
# Output layout (everything in ONE folder, ready for Globus):
#   ${OUTDIR_BASE}/v2_runs/${RUN_TAG}/
#     trainset.h5                  ← this job
#     README.md                    ← this job
#     dataset_metadata.json        ← this job
#     preload.slurm.log            ← this job (copied at end)
#     (later, training adds:       model_epoch_*.h5, checkpoint_*.pt,
#                                  config.json, loss_history.json,
#                                  train.slurm.log)
#
# After this job completes, train with:
#   sbatch --export=ALL,RUN_TAG=<same tag> slurm_train/train_only_nersc.sh
#
# VARIANTs (same as the chained submit):
#   no_dla_no_bal      DLAs (NHI ≥ 20.3) + BALs excluded; sub-DLAs/LLS kept
#   no_hcd_with_bal    All HCDs (NHI ≥ 17.2) excluded; BALs KEPT
#   no_hcd_no_bal      All HCDs + BALs excluded
#   bal_only           ONLY BAL spectra (BI_CIV>0); all HCDs excluded
#                       For training a BAL-only GP that can be Bayesian-
#                       model-selected against the non-BAL GP.
#
# Submit:
#   sbatch --export=ALL,VARIANT=no_dla_no_bal     slurm_train/preload_loa_only_nersc.sh
#   sbatch --export=ALL,VARIANT=no_hcd_with_bal   slurm_train/preload_loa_only_nersc.sh
#   sbatch --export=ALL,VARIANT=no_hcd_no_bal     slurm_train/preload_loa_only_nersc.sh
#   sbatch --export=ALL,VARIANT=bal_only          slurm_train/preload_loa_only_nersc.sh

set -eo pipefail
export PYTHONUNBUFFERED=1

source /global/cfs/cdirs/desi/software/desi_environment.sh main || {
    echo "[error] failed to load desi env" >&2; exit 1
}

VARIANT="${VARIANT:?must be set: no_dla_no_bal | no_hcd_with_bal | no_hcd_no_bal}"

# Inputs.
QSOCAT="${QSOCAT:-/global/cfs/cdirs/desi/users/martini/bal-catalogs/loa/QSO_cat_loa_main_dark_healpix_v3-altbal.fits}"
SPECDIR="${SPECDIR:-/global/cfs/cdirs/desi/spectro/redux/loa}"
HCD_CAT="${HCD_CAT:-/global/cfs/cdirs/desicollab/users/jibancat/DLA/processed_gp_samples/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/dlacat-loa-main-dark.fits}"
HCD_TID_COL="${HCD_TID_COL:-TARGETID}"
HCD_NHI_COL="${HCD_NHI_COL:-NHI}"
HCD_MIN_PDLA="${HCD_MIN_PDLA:-0.0}"

OUTDIR_BASE="${OUTDIR_BASE:-/pscratch/sd/j/jibancat/desi_gpy_dla_detection}"
RUN_TAG="${RUN_TAG:-loa_${VARIANT}_${SLURM_JOB_ID}}"
RUN_DIR="${RUN_DIR:-${OUTDIR_BASE}/v2_runs/${RUN_TAG}}"
TRAINSET_H5="${RUN_DIR}/trainset.h5"

case "$VARIANT" in
    no_dla_no_bal)
        HCD_MIN_NHI="${HCD_MIN_NHI:-20.3}"
        EXCLUDE_BAL_FLAG="--exclude-bal"
        ;;
    no_hcd_with_bal)
        HCD_MIN_NHI="${HCD_MIN_NHI:-17.2}"
        EXCLUDE_BAL_FLAG=""
        ;;
    no_hcd_no_bal)
        HCD_MIN_NHI="${HCD_MIN_NHI:-17.2}"
        EXCLUDE_BAL_FLAG="--exclude-bal"
        ;;
    bal_only)
        HCD_MIN_NHI="${HCD_MIN_NHI:-17.2}"
        EXCLUDE_BAL_FLAG="--bal-only"
        ;;
    *)
        echo "[error] VARIANT must be no_dla_no_bal | no_hcd_with_bal | no_hcd_no_bal | bal_only" >&2
        exit 2
        ;;
esac

Z_MIN="${Z_MIN:-2.0}"
Z_MAX="${Z_MAX:-4.25}"
MAX_SPECTRA="${MAX_SPECTRA:-300000}"
DLAMBDA="${DLAMBDA:-0.15}"

[ -r "$QSOCAT" ] || { echo "[error] QSOCAT: $QSOCAT" >&2; exit 3; }
[ -d "$SPECDIR" ] || { echo "[error] SPECDIR: $SPECDIR" >&2; exit 4; }
[ -r "$HCD_CAT" ] || { echo "[error] HCD_CAT: $HCD_CAT" >&2; exit 5; }
mkdir -p "$RUN_DIR"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

echo "===================================================="
echo "  PRELOAD ONLY  variant: $VARIANT  job: $SLURM_JOB_ID"
echo "  -q regular -C cpu  walltime 6 h"
echo "===================================================="
echo "  qsocat:    $QSOCAT"
echo "  specdir:   $SPECDIR"
echo "  hcd_cat:   $HCD_CAT"
echo "  filter:    z [$Z_MIN, $Z_MAX]; HCD NHI ≥ $HCD_MIN_NHI; "
echo "             BAL=${EXCLUDE_BAL_FLAG:+ON}${EXCLUDE_BAL_FLAG:-OFF}; P_DLA=$HCD_MIN_PDLA"
echo "  max_spec:  $MAX_SPECTRA"
echo "  RUN_DIR:   $RUN_DIR"
echo "===================================================="
echo

python -u preload_spectra/preload_loa_real.py \
    --qsocat "$QSOCAT" \
    --specdir "$SPECDIR" \
    --output "$TRAINSET_H5" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --max-spectra "$MAX_SPECTRA" \
    --dlambda "$DLAMBDA" \
    $EXCLUDE_BAL_FLAG \
    --hcd-cat "$HCD_CAT" \
    --hcd-tid-col "$HCD_TID_COL" \
    --hcd-nhi-col "$HCD_NHI_COL" \
    --hcd-min-nhi "$HCD_MIN_NHI" \
    --hcd-min-pdla "$HCD_MIN_PDLA"

[ -r "$TRAINSET_H5" ] || { echo "[error] trainset.h5 not produced" >&2; exit 7; }
echo "preload wrote: $TRAINSET_H5 ($(du -h "$TRAINSET_H5" | cut -f1))"

cp "slurm_train/loa_preload_${SLURM_JOB_ID}.log" "$RUN_DIR/preload.slurm.log" 2>/dev/null || true

echo
echo "===================================================="
echo "  PRELOAD COMPLETE  $RUN_TAG"
echo "  RUN_DIR:   $RUN_DIR"
echo
echo "  NEXT STEP: train on this dataset"
echo "  sbatch --export=ALL,RUN_TAG=$RUN_TAG slurm_train/train_only_nersc.sh"
echo "===================================================="
