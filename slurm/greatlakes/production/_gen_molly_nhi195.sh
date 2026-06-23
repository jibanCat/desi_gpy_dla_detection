#!/usr/bin/env bash
# Reproduce the 2LPT-0 V1 molly P/C matrices at a chosen truth-NHI floor.
# config-only: re-bins the existing catalog/truth/snr_cat. ZERO GP re-inference.
set -eo pipefail
source ~/.bashrc
conda activate gpdla
export LD_LIBRARY_PATH="${HOME}/.local/usr/local/lib64:${LD_LIBRARY_PATH}"
cd /home/mfho/desi_gpy_dla_detection

CAT="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/"
MOCKDIR="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124"
TRUTH="${MOCKDIR}/hcd_truth_cat.fits"
BAL="${MOCKDIR}/bal_cat.fits"

FLOOR="$1"          # e.g. 19.5
NHIBINS="$2"        # e.g. 19.5,20,20.3,20.5,21,21.5,22,inf
OUT="$3"            # output dir
TITLE="$4"

python examples/molly_faithful_pc_plots.py \
    --catalog-dir "${CAT}" \
    --truth       "${TRUTH}" \
    --bal-cat     "${BAL}" --no-bal \
    --mockdir     "${MOCKDIR}" \
    --truth-nhi-min "${FLOOR}" \
    --nhi-min     "${FLOOR}" \
    --snr-min     2.0 \
    --gp-conf     0.99 \
    --nhi-bins    "${NHIBINS}" \
    --out "${OUT}" \
    --title "${TITLE}"
