#!/bin/bash
# slurm/greatlakes/production/crossmock_loa0_pointonly.sbatch
#
# Track-C task #40 — cross-mock loa0-FP cross-check, POINT-ONLY rerun.
# =====================================================================================
# WHY THIS RUN EXISTS
#   crossmock_loa0.sbatch (job 52628514) crashed in ALL THREE held-out legs at
#   make_v3x_refit_fn (cddf_catalog_hbi.py:6952) with NotImplementedError: the per-z MC
#   BAND hard-codes the purity-mixture (1−ρ) FP and refuses fp_estimator='loa0' (a loa0
#   band must come from loa0_full_posterior_mc, which the cross-mock driver does not yet
#   wire). We do NOT need the band — only the POINT R0 (integrated dN/dX & Ω at ≥20.0 /
#   ≥20.3 plus per-z MAP/truth), which the loa0 point estimator (v3x_refit, untilted)
#   computes correctly and BEFORE the crash.
#
#   This rerun passes --point-only to each driver: run_tf_variant returns right after the
#   integrated_point + per-z MAP/truth R0 (band fields = NaN), skipping the band entirely.
#   Reports/JSON land in the SAME tf_{2lpt1,london0,saclay}_loa0/ dirs as the crashed run.
#
#   Everything else is IDENTICAL to crossmock_loa0.sbatch (frozen 2LPT-0 forward kernel +
#   g(N,z) + molly C/ρ ratio; fp_estimator=loa0 for the held-out POINT; loa0 product
#   vol-scaled by each leg's own n_sl_prod). The GP inference path (gpy_dla_detection/) is
#   byte-FROZEN; the estimator (cddf_catalog_hbi.py) is NOT edited; the purity_mixture
#   path is byte-identical (--point-only defaults OFF for pm; here it is set ON for loa0).
#
# SUBMIT WITH:
#     sbatch slurm/greatlakes/production/crossmock_loa0_pointonly.sbatch
# =====================================================================================

#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -J crossmock_loa0_po
#SBATCH -o slurm/greatlakes/production/logs/crossmock_loa0_po_%j.log
#SBATCH -e slurm/greatlakes/production/logs/crossmock_loa0_po_%j.err

set -eo pipefail
export PYTHONUNBUFFERED=1

# --- Pin BLAS threads to 1 (config-only; no inference-path code touched). ------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# --- conda env (no NERSC desi_environment.sh on GreatLakes) -------------------------
source ~/.bashrc
conda activate gpdla
export LD_LIBRARY_PATH="${HOME}/.local/usr/local/lib64:${LD_LIBRARY_PATH}"

REPO="/home/mfho/desi_gpy_dla_detection"
cd "${REPO}"
mkdir -p slurm/greatlakes/production/logs

# =====================================================================================
# FROZEN 2LPT-0 calibration artifacts (identical to the band run)
# =====================================================================================
FORWARD="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/stage0/forward_response_2lpt0.npz"
MOLLY="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/figures_molly_nhi195/lya_only/molly_matrix.tsv"
CAL_CAT="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/"
CAL_TRUTH="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"
CAL_BAL="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"

LOA0_PRODUCT="/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/outputs/loa0_fp_product_lyaonly1025.npz"

ZBINS="2.0,2.5,3.0,3.5"
REPORT_LIMITS="20.0,20.3"
N_MC=200              # ignored in --point-only (no band) but kept for arg parity
SEED=0
VARIANT="A"          # loa0 cross-check: fully-frozen variant only

OUTBASE="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/crossmock_alpha"
OUT_2LPT1="${OUTBASE}/tf_2lpt1_loa0"
OUT_LONDON0="${OUTBASE}/tf_london0_loa0"
OUT_SACLAY="${OUTBASE}/tf_saclay_loa0"
mkdir -p "${OUT_2LPT1}" "${OUT_LONDON0}" "${OUT_SACLAY}"

echo "============================================================================"
echo "[$(date)] crossmock_loa0 POINT-ONLY START (job ${SLURM_JOB_ID:-local})"
echo "  fp_estimator=loa0  --point-only  variant=${VARIANT}"
echo "  zbins=${ZBINS} limits=${REPORT_LIMITS}"
echo "============================================================================"

for f in "${FORWARD}" "${MOLLY}" "${CAL_TRUTH}" "${CAL_BAL}" "${LOA0_PRODUCT}"; do
    if [[ ! -e "${f}" ]]; then echo "[FATAL] missing: ${f}" >&2; exit 2; fi
done
if [[ ! -e "${CAL_CAT}" ]]; then echo "[FATAL] cal catalog missing: ${CAL_CAT}" >&2; exit 2; fi

# =====================================================================================
# LEG 1 — held-out 2lpt-1 (SAME recipe)
# =====================================================================================
C1_CAT="/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/2lpt1_loa124_v1"
C1_MOCKDIR="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-1/loa-124"
C1_TRUTH="${C1_MOCKDIR}/hcd_truth_cat.fits"
C1_BAL="${C1_MOCKDIR}/bal_cat.fits"

echo "[$(date)] LEG 1 (2lpt-1, loa0, point-only) ..."
python CDDF_analysis/hbi/track_c_tf_2lpt1.py \
    --variant "${VARIANT}" --fp-estimator loa0 --loa0-product "${LOA0_PRODUCT}" \
    --point-only \
    --catalog-dir "${CAL_CAT}" --truth "${CAL_TRUTH}" --bal-cat "${CAL_BAL}" \
    --molly-tsv "${MOLLY}" --forward-model "${FORWARD}" \
    --heldout-cat "${C1_CAT}" --heldout-truth "${C1_TRUTH}" --heldout-bal "${C1_BAL}" \
    --heldout-mockdir "${C1_MOCKDIR}" \
    --zbins "${ZBINS}" --report-limits "${REPORT_LIMITS}" \
    --n-mc "${N_MC}" --seed "${SEED}" \
    --out "${OUT_2LPT1}" --report-out "${OUT_2LPT1}/report_pointonly.md" \
    || echo "[leg1] FAILED (continuing to remaining legs)" >&2
echo "[$(date)] LEG 1 DONE -> ${OUT_2LPT1}"

# =====================================================================================
# LEG 2 — held-out london-0 (DIFFERENT recipe)
# =====================================================================================
L0_CAT="/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/london0_jura124_v1"
L0_MOCKDIR="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_london0/mockdir"
L0_TRUTH="${L0_MOCKDIR}/dla_cat.fits"
L0_BAL="${L0_MOCKDIR}/bal_cat.fits"

echo "[$(date)] LEG 2 (london-0, loa0, point-only) ..."
python CDDF_analysis/hbi/track_c_tf_london0.py \
    --variant "${VARIANT}" --fp-estimator loa0 --loa0-product "${LOA0_PRODUCT}" \
    --point-only \
    --catalog-dir "${CAL_CAT}" --truth "${CAL_TRUTH}" --bal-cat "${CAL_BAL}" \
    --molly-tsv "${MOLLY}" --forward-model "${FORWARD}" \
    --heldout-cat "${L0_CAT}" --heldout-truth "${L0_TRUTH}" --heldout-bal "${L0_BAL}" \
    --heldout-mockdir "${L0_MOCKDIR}" \
    --zbins "${ZBINS}" --report-limits "${REPORT_LIMITS}" \
    --n-mc "${N_MC}" --seed "${SEED}" \
    --out "${OUT_LONDON0}" --report-out "${OUT_LONDON0}/report_pointonly.md" \
    || echo "[leg2] FAILED (continuing to remaining legs)" >&2
echo "[$(date)] LEG 2 DONE -> ${OUT_LONDON0}"

# =====================================================================================
# LEG 3 — held-out saclay-0 (DIFFERENT recipe)
# =====================================================================================
S0_CAT="/gpfs/accounts/cavestru_root/cavestru0/mfho/gl_prod_saclay0_v1_20260630/combined_catalog"
S0_MOCKDIR="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124"
S0_TRUTH="${S0_MOCKDIR}/hcd_truth_cat.fits"
S0_BAL="${S0_MOCKDIR}/bal_cat.fits"

echo "[$(date)] LEG 3 (saclay-0, loa0, point-only) ..."
python CDDF_analysis/hbi/track_c_tf_saclay.py \
    --variant "${VARIANT}" --fp-estimator loa0 --loa0-product "${LOA0_PRODUCT}" \
    --point-only \
    --catalog-dir "${CAL_CAT}" --truth "${CAL_TRUTH}" --bal-cat "${CAL_BAL}" \
    --molly-tsv "${MOLLY}" --forward-model "${FORWARD}" \
    --heldout-cat "${S0_CAT}" --heldout-truth "${S0_TRUTH}" --heldout-bal "${S0_BAL}" \
    --heldout-mockdir "${S0_MOCKDIR}" \
    --zbins "${ZBINS}" --report-limits "${REPORT_LIMITS}" \
    --n-mc "${N_MC}" --seed "${SEED}" \
    --out "${OUT_SACLAY}" --report-out "${OUT_SACLAY}/report_pointonly.md" \
    || echo "[leg3] FAILED (continuing)" >&2
echo "[$(date)] LEG 3 DONE -> ${OUT_SACLAY}"

echo "============================================================================"
echo "[$(date)] crossmock_loa0 POINT-ONLY DONE"
echo "  leg1 (2lpt-1)  -> ${OUT_2LPT1}/report_pointonly.md + track_c_tf_2lpt1.json"
echo "  leg2 (london0) -> ${OUT_LONDON0}/report_pointonly.md + track_c_tf_london0.json"
echo "  leg3 (saclay0) -> ${OUT_SACLAY}/report_pointonly.md + track_c_tf_saclay.json"
echo "============================================================================"
