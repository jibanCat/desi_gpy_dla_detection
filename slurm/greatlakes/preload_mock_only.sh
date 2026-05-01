#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -J mock_preload
#SBATCH -o slurm/greatlakes/mock_preload_%j.log
#SBATCH -e slurm/greatlakes/mock_preload_%j.log

# Generic mock preload — points at any DESI Y3 mock directory and
# applies HCD + BAL anti-joins via preload_2lpt_simple.py.
#
# Currently works for:
#   - 2lpt (loa-0, loa-124)
#   - saclay (mock-0/juraLy8-124)
# Does NOT work for london mock-0 — london uses ``dla_cat.fits`` not
# ``hcd_truth_cat.fits``; preload_2lpt_simple.py reads only the latter.
# The next-session task is to extend the simple preloader to also accept
# ``dla_cat.fits`` for the HCD filter.
#
# Submit:
#   sbatch --export=ALL,MOCK_NAME=saclay_mock0_nohcd_nobal,\
#                  MOCK_DIR=/nfs/turbo/.../mocks/saclay/.../mock-0/juraLy8-124 \
#       slurm/greatlakes/preload_mock_only.sh

set -eo pipefail
export PYTHONUNBUFFERED=1

MOCK_NAME="${MOCK_NAME:?must be set, e.g. saclay_mock0_nohcd_nobal}"
MOCK_DIR="${MOCK_DIR:?must be set to absolute mock dir path}"

OUTDIR_BASE="${OUTDIR_BASE:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection}"
RUN_DIR="${RUN_DIR:-${OUTDIR_BASE}/v2_runs/${MOCK_NAME}}"
TRAINSET_H5="${RUN_DIR}/trainset.h5"

mkdir -p "$RUN_DIR"

[ -d "$MOCK_DIR" ] || { echo "[error] MOCK_DIR not found: $MOCK_DIR" >&2; exit 2; }

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla
export LD_LIBRARY_PATH="$HOME/.local/usr/local/lib64:${LD_LIBRARY_PATH:-}"

echo "===================================================="
echo "  MOCK PRELOAD  $MOCK_NAME  job: $SLURM_JOB_ID"
echo "  source:    $MOCK_DIR"
echo "  output:    $TRAINSET_H5"
echo "===================================================="

# Default = drop HCDs (NHI ≥ 17.2 in hcd_truth_cat.fits) and BALs (BI_CIV>0).
# Override these via env vars if needed.
EXCLUDE_HCD_FLAG="${EXCLUDE_HCD_FLAG:---exclude-hcd}"
EXCLUDE_BAL_FLAG="${EXCLUDE_BAL_FLAG:---exclude-bal}"
HCD_MIN_NHI="${HCD_MIN_NHI:-17.2}"
Z_MIN="${Z_MIN:-2.0}"
Z_MAX="${Z_MAX:-4.0}"
MAX_SPECTRA="${MAX_SPECTRA:-300000}"

python -u preload_spectra/preload_2lpt_simple.py \
    --mock-dir "$MOCK_DIR" \
    --output "$TRAINSET_H5" \
    --max-spectra "$MAX_SPECTRA" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --hcd-min-nhi "$HCD_MIN_NHI" \
    $EXCLUDE_HCD_FLAG $EXCLUDE_BAL_FLAG

[ -r "$TRAINSET_H5" ] || { echo "[error] trainset.h5 not produced" >&2; exit 7; }
echo "preload wrote: $TRAINSET_H5 ($(du -h $TRAINSET_H5 | cut -f1))"

cp "slurm/greatlakes/mock_preload_${SLURM_JOB_ID}.log" "$RUN_DIR/preload.slurm.log" 2>/dev/null || true

echo
echo "===================================================="
echo "  PRELOAD COMPLETE  $MOCK_NAME"
echo "  RUN_DIR:   $RUN_DIR"
echo
echo "  NEXT STEP: train"
echo "  sbatch --export=ALL,RUN_TAG=$MOCK_NAME,RUN_DIR=$RUN_DIR,\\"
echo "                 TRAINSET_H5=$TRAINSET_H5,NUM_EPOCHS=1500 \\"
echo "      slurm/greatlakes/train_only_gpu.sh"
echo "===================================================="
