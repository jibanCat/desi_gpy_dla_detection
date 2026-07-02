#!/bin/bash
# slurm/greatlakes/production/crossmock_loa0.sbatch
#
# Track-C task #40 — cross-mock validation, loa0-FP CROSS-CHECK of the alpha(z)~=1 legs.
# =====================================================================================
# WHY THIS RUN EXISTS
#   The headline cross-mock legs (crossmock_alpha_tf.sbatch + crossmock_saclay.sbatch)
#   estimate the held-out POINT with fp_estimator='purity_mixture'. A referee finding is
#   that the residual ~13% Omega deficit is largely a purity_mixture FP artifact that the
#   directly-measured loa-0 forest-FP estimator ('loa0') fixes. This run re-runs ALL THREE
#   held-out legs IDENTICALLY but with --fp-estimator loa0, into SEPARATE output dirs, so
#   the recovered dN/dX / Omega can be compared one-to-one against the purity_mixture
#   results already in .../crossmock_alpha/tf_{2lpt1,london0,saclay}/.
#
#   ONLY the reduce-side FP estimator for the held-out POINT changes. The FROZEN 2LPT-0
#   calibration (forward-response kernel + z-resolved completeness g(N,z) + molly C/rho
#   RATIO matrix) is byte-identical to the purity_mixture legs and is built with
#   fp_estimator='purity_mixture' (it is FP-INDEPENDENT — it only extracts g(N,z) + molly
#   counts). The GP inference path (gpy_dla_detection/) is byte-FROZEN; the estimator
#   logic (cddf_catalog_hbi.py) is NOT edited. Only the three driver argparse wirings
#   gained --fp-estimator / --loa0-product (default purity_mixture = unchanged).
#
#   loa0 needs the production sightline count for its vol-scale extrapolation; each driver
#   sets cfg.n_sl_prod from build_pathlength BEFORE make_fp_model, so the held-out leg's
#   own n_sl flows into the loa0 forward FP intensity (verified ordering).
#
#   Three legs (same as the purity_mixture run):
#     LEG 1  2lpt-1   (track_c_tf_2lpt1.py)   — SAME recipe (lyacolore_2lpt v2.8.5).
#     LEG 2  london-0 (track_c_tf_london0.py) — DIFFERENT recipe (CoLoRe/london v5.9.5).
#     LEG 3  saclay-0 (track_c_tf_saclay.py)  — DIFFERENT recipe (Saclay v4.7.5).
#   VARIANT A ONLY (fully frozen: 2LPT-0 molly counts + g + kernel) — the loa0 cross-check
#   only needs the real-data-applicable fully-frozen variant.
#
# DISCIPLINE: analysis / reduce-only. NO GP inference. All held-out data is mock-only.
#
# SUBMIT WITH:
#     sbatch slurm/greatlakes/production/crossmock_loa0.sbatch
# =====================================================================================

#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -J crossmock_loa0
#SBATCH -o slurm/greatlakes/production/logs/crossmock_loa0_%j.log
#SBATCH -e slurm/greatlakes/production/logs/crossmock_loa0_%j.err

set -eo pipefail
export PYTHONUNBUFFERED=1

# --- Pin BLAS threads to 1 (config-only; no inference-path code touched). ------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Turbo/GPFS NFS: disable HDF5 file locking so processed-h5 / snr_cat.h5 reads never block.
export HDF5_USE_FILE_LOCKING=FALSE

# --- conda env (no NERSC desi_environment.sh on GreatLakes) -------------------------
source ~/.bashrc
conda activate gpdla
export LD_LIBRARY_PATH="${HOME}/.local/usr/local/lib64:${LD_LIBRARY_PATH}"

REPO="/home/mfho/desi_gpy_dla_detection"
cd "${REPO}"
mkdir -p slurm/greatlakes/production/logs

# =====================================================================================
# FROZEN 2LPT-0 calibration artifacts (identical to the purity_mixture legs)
# =====================================================================================
FORWARD="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/stage0/forward_response_2lpt0.npz"
MOLLY="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/figures_molly_nhi195/lya_only/molly_matrix.tsv"
CAL_CAT="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/"
CAL_TRUTH="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"
CAL_BAL="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"

# --- the directly-measured loa-0 forest-FP product (vol-scaled per held-out leg) -----
LOA0_PRODUCT="/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/outputs/loa0_fp_product_lyaonly1025.npz"

# --- run knobs (match the headline per-z band recipe) --------------------------------
ZBINS="2.0,2.5,3.0,3.5"
REPORT_LIMITS="20.0,20.3"
N_MC=200
SEED=0
VARIANT="A"                 # loa0 cross-check: fully-frozen variant only

# --- NEW output base (separate from the purity_mixture results) ----------------------
OUTBASE="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/crossmock_alpha"
OUT_2LPT1="${OUTBASE}/tf_2lpt1_loa0"
OUT_LONDON0="${OUTBASE}/tf_london0_loa0"
OUT_SACLAY="${OUTBASE}/tf_saclay_loa0"
mkdir -p "${OUT_2LPT1}" "${OUT_LONDON0}" "${OUT_SACLAY}"

echo "============================================================================"
echo "[$(date)] crossmock_loa0 START (job ${SLURM_JOB_ID:-local})"
echo "  REPO          : ${REPO}"
echo "  frozen kernel : ${FORWARD}"
echo "  frozen molly  : ${MOLLY}"
echo "  loa0 product  : ${LOA0_PRODUCT}"
echo "  n_mc=${N_MC} seed=${SEED} variant=${VARIANT} fp_estimator=loa0"
echo "  zbins=${ZBINS} limits=${REPORT_LIMITS}"
echo "============================================================================"

# Sanity: the FROZEN calibration artifacts + loa0 product must exist.
for f in "${FORWARD}" "${MOLLY}" "${CAL_TRUTH}" "${CAL_BAL}" "${LOA0_PRODUCT}"; do
    if [[ ! -e "${f}" ]]; then
        echo "[FATAL] required artifact missing: ${f}" >&2
        exit 2
    fi
done
if [[ ! -e "${CAL_CAT}" ]]; then
    echo "[FATAL] 2LPT-0 calibration catalog missing: ${CAL_CAT}" >&2
    exit 2
fi

# =====================================================================================
# LEG 1 — held-out 2lpt-1 (SAME recipe; catalog DIRECTORY, loader globs dlacat-*.fits).
# =====================================================================================
C1_CAT="/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/2lpt1_loa124_v1"
C1_MOCKDIR="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-1/loa-124"
C1_TRUTH="${C1_MOCKDIR}/hcd_truth_cat.fits"
C1_BAL="${C1_MOCKDIR}/bal_cat.fits"

echo "[$(date)] LEG 1 (2lpt-1, loa0) ..."
python CDDF_analysis/hbi/track_c_tf_2lpt1.py \
    --variant "${VARIANT}" \
    --fp-estimator loa0 \
    --loa0-product "${LOA0_PRODUCT}" \
    --catalog-dir "${CAL_CAT}" \
    --truth "${CAL_TRUTH}" \
    --bal-cat "${CAL_BAL}" \
    --molly-tsv "${MOLLY}" \
    --forward-model "${FORWARD}" \
    --heldout-cat "${C1_CAT}" \
    --heldout-truth "${C1_TRUTH}" \
    --heldout-bal "${C1_BAL}" \
    --heldout-mockdir "${C1_MOCKDIR}" \
    --zbins "${ZBINS}" \
    --report-limits "${REPORT_LIMITS}" \
    --n-mc "${N_MC}" \
    --seed "${SEED}" \
    --out "${OUT_2LPT1}" \
    --report-out "${OUT_2LPT1}/report.md" \
    || echo "[leg1] FAILED (continuing to remaining legs)" >&2
echo "[$(date)] LEG 1 DONE -> ${OUT_2LPT1}"

# =====================================================================================
# LEG 2 — held-out london-0 (DIFFERENT recipe). Catalog DIRECTORY + pre-staged mockdir.
# =====================================================================================
L0_CAT="/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/london0_jura124_v1"
L0_MOCKDIR="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_london0/mockdir"
L0_TRUTH="${L0_MOCKDIR}/dla_cat.fits"
L0_BAL="${L0_MOCKDIR}/bal_cat.fits"

echo "[$(date)] LEG 2 (london-0, loa0) ..."
python CDDF_analysis/hbi/track_c_tf_london0.py \
    --variant "${VARIANT}" \
    --fp-estimator loa0 \
    --loa0-product "${LOA0_PRODUCT}" \
    --catalog-dir "${CAL_CAT}" \
    --truth "${CAL_TRUTH}" \
    --bal-cat "${CAL_BAL}" \
    --molly-tsv "${MOLLY}" \
    --forward-model "${FORWARD}" \
    --heldout-cat "${L0_CAT}" \
    --heldout-truth "${L0_TRUTH}" \
    --heldout-bal "${L0_BAL}" \
    --heldout-mockdir "${L0_MOCKDIR}" \
    --zbins "${ZBINS}" \
    --report-limits "${REPORT_LIMITS}" \
    --n-mc "${N_MC}" \
    --seed "${SEED}" \
    --out "${OUT_LONDON0}" \
    --report-out "${OUT_LONDON0}/report.md" \
    || echo "[leg2] FAILED (continuing to remaining legs)" >&2
echo "[$(date)] LEG 2 DONE -> ${OUT_LONDON0}"

# =====================================================================================
# LEG 3 — held-out saclay-0 (DIFFERENT recipe). Native juraLy8-124 mockdir, nothing staged.
# =====================================================================================
S0_CAT="/gpfs/accounts/cavestru_root/cavestru0/mfho/gl_prod_saclay0_v1_20260630/combined_catalog"
S0_MOCKDIR="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124"
S0_TRUTH="${S0_MOCKDIR}/hcd_truth_cat.fits"
S0_BAL="${S0_MOCKDIR}/bal_cat.fits"

echo "[$(date)] LEG 3 (saclay-0, loa0) ..."
python CDDF_analysis/hbi/track_c_tf_saclay.py \
    --variant "${VARIANT}" \
    --fp-estimator loa0 \
    --loa0-product "${LOA0_PRODUCT}" \
    --catalog-dir "${CAL_CAT}" \
    --truth "${CAL_TRUTH}" \
    --bal-cat "${CAL_BAL}" \
    --molly-tsv "${MOLLY}" \
    --forward-model "${FORWARD}" \
    --heldout-cat "${S0_CAT}" \
    --heldout-truth "${S0_TRUTH}" \
    --heldout-bal "${S0_BAL}" \
    --heldout-mockdir "${S0_MOCKDIR}" \
    --zbins "${ZBINS}" \
    --report-limits "${REPORT_LIMITS}" \
    --n-mc "${N_MC}" \
    --seed "${SEED}" \
    --out "${OUT_SACLAY}" \
    --report-out "${OUT_SACLAY}/report.md" \
    || echo "[leg3] FAILED (continuing)" >&2
echo "[$(date)] LEG 3 DONE -> ${OUT_SACLAY}"

echo "============================================================================"
echo "[$(date)] crossmock_loa0 DONE"
echo "  leg1 (2lpt-1)  -> ${OUT_2LPT1}"
echo "  leg2 (london0) -> ${OUT_LONDON0}"
echo "  leg3 (saclay0) -> ${OUT_SACLAY}"
echo "  Compare loa0 Omega/dN/dX against the purity_mixture results in the sibling"
echo "  tf_{2lpt1,london0,saclay}/ dirs (read integrated_R0 / coverage from the JSONs)."
echo "============================================================================"
