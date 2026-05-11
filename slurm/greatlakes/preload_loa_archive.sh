#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J preload_loa_archive
#SBATCH -o slurm/greatlakes/preload_loa_archive_%j.log
#SBATCH -e slurm/greatlakes/preload_loa_archive_%j.log

# Run preload_spectra/preload_from_loa_archive.py to convert the 75 GB
# LoaArchive (loa_full_z2_noR_v2.h5, 928k QSOs at observed [3600, 9824])
# to a v2 trainset.h5 on the wider rest grid [850.75, 1700, 0.15] —
# matches the 2lpt wide_v2 grid that phase2_train_desi.py is set up for.
#
# Two configs, parameterized via RUN_NAME:
#   loa_no_dla_no_bal_wide   (exclude HCDs with NHI≥20.3 + BAL via BI_CIV>0)
#   loa_no_hcd_with_bal_wide (exclude HCDs with NHI≥17.2, keep BAL)
#
# Submit:
#   sbatch --export=ALL,RUN_NAME=loa_no_dla_no_bal_wide \
#       slurm/greatlakes/preload_loa_archive.sh
#   sbatch --export=ALL,RUN_NAME=loa_no_hcd_with_bal_wide \
#       slurm/greatlakes/preload_loa_archive.sh
#
# Estimated wall ~2-4 h depending on filter cuts.
# Output → /scratch/cavestru_root/cavestru0/mfho/loa_wide_v2/$RUN_NAME/trainset.h5
#
# Once both trainset.h5 files exist, submit phase2_desi_retrain.sh on each.

set -eo pipefail
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

ARCHIVE="${ARCHIVE:-/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5}"
QSOCAT="${QSOCAT:-/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v3-altbal.fits}"
RUN_NAME="${RUN_NAME:?must set RUN_NAME (loa_no_dla_no_bal_wide or loa_no_hcd_with_bal_wide)}"

OUT_DIR="${OUT_DIR:-/scratch/cavestru_root/cavestru0/mfho/loa_wide_v2/${RUN_NAME}}"
mkdir -p "$OUT_DIR"

# Filter cuts depend on RUN_NAME. Match the existing legacy trainset attrs:
#   loa_no_dla_no_bal_52198069: hcd_min_nhi=20.3, exclude_bal=True
#   loa_no_hcd_with_bal_52198070: hcd_min_nhi=17.2, exclude_bal=False
case "$RUN_NAME" in
  loa_no_dla_no_bal_wide)
    HCD_MIN_NHI=20.3
    EXCLUDE_BAL_FLAG="--bal-cat $QSOCAT --bal-col BI_CIV --bal-min 0.0"
    ;;
  loa_no_hcd_with_bal_wide)
    HCD_MIN_NHI=17.2
    EXCLUDE_BAL_FLAG=""
    ;;
  *)
    echo "ERROR: unknown RUN_NAME=$RUN_NAME"
    exit 1
    ;;
esac

# HCD catalog: use the nhi172 catalog (includes both DLAs and sub-DLAs)
# and apply the NHI threshold via --hcd-min-nhi inside the adapter
# (matches preload_loa_real.py convention).
HCD_CAT=/scratch/cavestru_root/cavestru0/mfho/gl_outputs/DLA/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/dlacat-loa-main-dark.fits

echo "=== preload_loa_archive ==="
echo "  run_name     : $RUN_NAME"
echo "  archive      : $ARCHIVE"
echo "  qsocat       : $QSOCAT"
echo "  hcd_cat      : $HCD_CAT"
echo "  hcd_min_nhi  : $HCD_MIN_NHI"
echo "  bal_filter   : ${EXCLUDE_BAL_FLAG:-(keep BALs)}"
echo "  out_dir      : $OUT_DIR"
echo "  job_id       : ${SLURM_JOB_ID:-(local)}"
echo "  node         : $(hostname)"
echo

python -u preload_spectra/preload_from_loa_archive.py \
    --archive "$ARCHIVE" \
    --out "$OUT_DIR/trainset.h5" \
    --z-min 2.15 --z-max 4.25 \
    --rest-min 850.75 --rest-max 1700.0 --rest-dlambda 0.15 \
    --hcd-cat "$HCD_CAT" --hcd-tid-col TARGETID \
    --hcd-nhi-col NHI --hcd-min-nhi "$HCD_MIN_NHI" \
    $EXCLUDE_BAL_FLAG

echo
echo "=== adapter complete ==="
ls -la "$OUT_DIR"
