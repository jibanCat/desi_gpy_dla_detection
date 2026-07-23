#!/bin/bash
# z-resolved purity/completeness: P/C vs z_DLA, split by z_QSO, across mocks.
#
# Spec: notes repo `2026-07-22_zresolved_pc_design.md`.
# Drives examples/molly_faithful_pc_plots.py once per mock; each invocation
# does both forest windows and both N_HI floors in one pass.
#
# Usage:
#   bash examples/run_zresolved_pc.sh [MOCK ...]      # default: all four
#
#   MOCK   2lpt0 | 2lpt1 | london0 | saclay0
#
# Env overrides:
#   OUT_ROOT   output root      (default /scratch/.../mfho/zresolved_pc)
#   PYTHON     interpreter      (default python; needs fitsio + matplotlib)
#   ZDLA_BINS / ZQSO_BINS / ZNHI_FLOORS   binning (defaults = the spec's)
#
# Outputs, per mock x window:
#   molly_pc_z_matrix.tsv        one row per (floor, z_DLA bin, z_QSO bin)
#   fig_pc_vs_zdla_nhi{203,200}.png
# plus every pre-existing molly product, unchanged.
#
# Data class: MOCK-ONLY (truth known). No real-LoA. Publishable.
#
# NOTE ON CATALOG PATHS. All four point at the packaged catalogs under
# `DESI/gpdla_catalogs/` on turbo, not at the purgeable production run dirs on
# /scratch. For 2lpt0 and saclay0 the two are the same file (byte size and row
# count verified identical, 2026-07-23); turbo is the durable copy.
#
# NOTE ON LONDON-0. Its mockdir has no `snr_cat.fits` (only `zcat.fits`), so
# the SNR lookup falls back to the processed-h5 path and drops truth rows with
# no h5 record (3605/172206 = 2.1%). That is a data limitation of the London
# mock, not a runner choice; the other three use the full-snr_cat molly recipe.
# Record it alongside any London-0 completeness number.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"
PY="${PYTHON:-python}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT_ROOT="${OUT_ROOT:-/scratch/cavestru_root/cavestru0/mfho/zresolved_pc}"
CATROOT="/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs"
MOCKROOT="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks"

# Binning fixed by the spec: z_DLA edges match Track-C / decompose_r0_zstructure
# so this product is directly comparable to the z-tilt work; the last z_QSO edge
# is the existing --z-qso-max default.
ZDLA_BINS="${ZDLA_BINS:-2.0,2.5,3.0,3.5}"
ZQSO_BINS="${ZQSO_BINS:-2.0,2.5,3.0,4.25}"
ZNHI_FLOORS="${ZNHI_FLOORS:-20.3,20.0}"

# The truth filename differs by mock family: the 2LPT/Saclay QQ runs ship
# `hcd_truth_cat.fits`, London ships `dla_cat.fits`. Not interchangeable --
# pass the one that exists for that mock.
mock_cfg() {   # -> CAT MOCKDIR TRUTH TITLE
    case "$1" in
        2lpt0)
            CAT="${CATROOT}/2lpt0_loa124_v1/"
            MOCKDIR="${MOCKROOT}/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124"
            TRUTH="hcd_truth_cat.fits"
            TITLE="2LPT-0 loa-124 V1" ;;
        2lpt1)
            CAT="${CATROOT}/2lpt1_loa124_v1/"
            MOCKDIR="${MOCKROOT}/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-1/loa-124"
            TRUTH="hcd_truth_cat.fits"
            TITLE="2LPT-1 loa-124 V1" ;;
        london0)
            CAT="${CATROOT}/london0_jura124_v1/"
            MOCKDIR="${MOCKROOT}/london/qq_desi_y3/v5.9.5/mock-0/jura-124"
            TRUTH="dla_cat.fits"
            TITLE="London-0 jura-124 V1" ;;
        saclay0)
            CAT="${CATROOT}/saclay0_juraLy8124_v1/"
            MOCKDIR="${MOCKROOT}/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124"
            TRUTH="hcd_truth_cat.fits"
            TITLE="Saclay-0 juraLy8-124 V1" ;;
        *)  echo "unknown mock: $1  (2lpt0|2lpt1|london0|saclay0)" >&2; exit 2 ;;
    esac
    if [ ! -f "${MOCKDIR}/${TRUTH}" ]; then
        echo "missing truth catalog: ${MOCKDIR}/${TRUTH}" >&2; exit 2
    fi
}

MOCKS=("$@")
if [ ${#MOCKS[@]} -eq 0 ]; then MOCKS=(2lpt0 2lpt1 london0 saclay0); fi

for m in "${MOCKS[@]}"; do
    mock_cfg "$m"
    echo "=============================================================="
    echo "=== ${TITLE}  ->  ${OUT_ROOT}/${m}"
    echo "=============================================================="
    mkdir -p "${OUT_ROOT}/${m}"
    # Cuts below are the molly recipe and match every other P/C product in this
    # project: snr>2, P_DLA>0.99, DLAFLAG==0, z_qso in [2.0,4.25], BAL excluded.
    # --nhi-min 20.0 sets the run floor; --z-nhi-floors re-cuts to 20.3 and 20.0
    # inside the z reduction, so both floors come from one pass.
    "$PY" examples/molly_faithful_pc_plots.py \
        --catalog-dir "${CAT}" \
        --truth       "${MOCKDIR}/${TRUTH}" \
        --bal-cat     "${MOCKDIR}/bal_cat.fits" --no-bal \
        --mockdir     "${MOCKDIR}" \
        --truth-nhi-min 20.0 --nhi-min 20.0 \
        --snr-min 2.0 --gp-conf 0.99 --lyb-veto \
        --zdla-bins "${ZDLA_BINS}" \
        --zqso-bins "${ZQSO_BINS}" \
        --z-nhi-floors "${ZNHI_FLOORS}" \
        --out "${OUT_ROOT}/${m}" \
        --title "${TITLE}" 2>&1 | tee "${OUT_ROOT}/run_${m}.log"
done

echo
echo "done. git stamp for provenance:"
git -C "$REPO_DIR" rev-parse HEAD
