#!/bin/bash
# slurm/greatlakes/production/crossmock_saclay.sbatch
#
# Track-C task #40 — the NON-CIRCULAR cross-mock alpha(z) ~= 1 validation, LEG 3 (Saclay).
# =====================================================================================
# WHY THIS RUN EXISTS
#   On-mock, alpha = 1/R0 is a TAUTOLOGY: dividing the 2LPT-0 recovery by 2LPT-0's own
#   truth is residual-zero by construction. The genuine, non-circular proof is to take
#   the recipe CALIBRATED ONLY on 2LPT-0 (the FROZEN forward-response kernel + the FROZEN
#   z-resolved completeness g(N,z) + the FROZEN molly C/rho ratio matrix) and apply it,
#   WITH NO REFIT of alpha and NO REFIT of the kernel, to a HELD-OUT mock catalog. We then
#   check R0(z) = HBI(recovered) / held-out-truth ~= 1 per z-bin. R0(z) ~= 1 on data the
#   estimator never saw = a real generalization claim.
#
#   This is the SACLAY leg = the 3rd recipe in the cross-mock generalization:
#     LEG 1  2lpt-1   (track_c_tf_2lpt1.py)   — SAME recipe (lyacolore_2lpt v2.8.5),
#                                               different realization (gentlest transfer).
#     LEG 2  london-0 (track_c_tf_london0.py) — DIFFERENT recipe (CoLoRe/london v5.9.5);
#                                               PASSED (dN/dX R0 ~ 0.98-0.99).
#     LEG 3  saclay-0 (track_c_tf_saclay.py)  — a SECOND DIFFERENT recipe (Saclay v4.7.5);
#                                               this script. Strongest cross-recipe set.
#   (track_c_tf_loa.py is REAL LOA — DELIBERATELY EXCLUDED here. Mock-only, shareable.)
#
#   Saclay's INPUT FORMAT matches the 2lpt leg (hcd_truth_cat.fits truth + a NATIVE
#   snr_cat.fits in the mockdir — nothing is staged), but the RECIPE is different from
#   2lpt, so the driver uses the london-0 band logic (mm_resample separation +
#   mean-flux rescale diagnostic) with saclay paths/labels.
#
# WHAT IS FROZEN (the crux — nothing below is re-fit on the held-out mock):
#   * forward-response kernel   -> forward_response_2lpt0.npz  (FROZEN file)
#   * z-resolved completeness   -> built ONCE on 2LPT-0 truth-match, stashed on
#                                  cfg._cnz_resolved (NOT rebuilt on the held-out mock)
#   * molly C/rho RATIO matrix  -> the 2LPT-0 lya_only-nhi195 matrix (FROZEN file)
#   The driver rebuilds ONLY the 2LPT-0 calibration ingredients (cheap, ~couple min),
#   extracts the frozen pieces, then scores the held-out saclay-0 catalog against ITS OWN
#   truth.
#   Variant A = fully frozen (2LPT-0 molly counts too; real-data-applicable).
#   Variant B = kernel+g frozen, molly COUNT denominators rebuilt on the held-out mock
#               (A/B localizes kernel-transfer vs completeness-count-transfer).
#
# MEAN-FLUX RESCALE / RESIDUAL CIRCULARITY (Track-C task #40 note):
#   The driver's fit_meanflux_rescale() is a DIAGNOSTIC ONLY — it fits a scalar s to the
#   held-out truth and REPORTS what s would do, but it does NOT modify the recovered
#   dN/dX. The headline per-z R0 and integrated R0 emitted in the verdict tables / JSON
#   are ALREADY the PRE-rescale (non-circular) values. Read the verdict-table R0 /
#   integrated_R0 as the result; treat meanflux_rescale.s in the JSON as a secondary
#   diagnostic.
#
# DISCIPLINE: analysis / reduce-only. NO GP inference is run here — gpy_dla_detection/
#   is byte-FROZEN, dla_gp.py / run_bayes_select.py / dlasearch.py are untouched, and the
#   estimator logic (cddf_catalog_hbi.py) is not edited. All held-out mock data is
#   shareable (mocks only); real-LOA paths are never read.
#
# IDEMPOTENT / SAFE TO RE-RUN: writes into its own OUT dir (figure + JSON + report.md are
#   overwritten, never appended).
#
# SUBMIT WITH:
#     sbatch slurm/greatlakes/production/crossmock_saclay.sbatch
# =====================================================================================

#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -J crossmock_saclay
#SBATCH -o slurm/greatlakes/production/logs/crossmock_saclay_%j.log
#SBATCH -e slurm/greatlakes/production/logs/crossmock_saclay_%j.err

set -eo pipefail
export PYTHONUNBUFFERED=1

# --- Pin BLAS threads to 1 (config-only; no inference-path code touched). The per-z MC
#     band is serial numpy; pinning keeps a single thread from oversubscribing the node
#     and matches the validated phase3d_postkernel setup. -------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Turbo/GPFS NFS: disable HDF5 file locking so processed-h5 / snr_cat.h5 reads never block.
export HDF5_USE_FILE_LOCKING=FALSE

# --- conda env (no NERSC desi_environment.sh on GreatLakes) -------------------------
source ~/.bashrc
conda activate gpdla
# libcerf for the Voigt C extension (not used in this reduce-only run; keep env consistent)
export LD_LIBRARY_PATH="${HOME}/.local/usr/local/lib64:${LD_LIBRARY_PATH}"

REPO="/home/mfho/desi_gpy_dla_detection"
cd "${REPO}"
mkdir -p slurm/greatlakes/production/logs

# =====================================================================================
# FROZEN 2LPT-0 calibration artifacts (verified to exist; never re-fit on a held-out mock)
# =====================================================================================
FORWARD="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/stage0/forward_response_2lpt0.npz"
# The lya_only-nhi195 molly C/rho RATIO matrix the broaden012 kernel was calibrated
# against (AB.DEF_LYAONLY_MOLLY; VERIFIED to reproduce the frozen baseline_R0). Passing it
# explicitly pins provenance and avoids the driver's hard-fail-on-missing fallback path.
MOLLY="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/figures_molly_nhi195/lya_only/molly_matrix.tsv"
# 2LPT-0 CALIBRATION catalog/truth/bal (AB.DEF_CAT / DEF_TRUTH / DEF_BAL) — the mock the
# frozen g(N,z) + molly counts are built from.
CAL_CAT="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/"
CAL_TRUTH="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"
CAL_BAL="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"

# --- run knobs (match the validated headline per-z band recipe) ----------------------
ZBINS="2.0,2.5,3.0,3.5"      # per-z R0(z) bins (the alpha(z)~=1 check is per z-bin)
REPORT_LIMITS="20.0,20.3"    # DLA-tier report limits; >=20.3 is the headline
N_MC=200                     # MC draws for the band (headline rigor; matches london-0 leg)
SEED=0
VARIANT="both"              # A (fully frozen) + B (counts rebuilt) -> kernel/completeness A/B

# =====================================================================================
# HELD-OUT saclay-0 (DIFFERENT recipe; the strongest transfer test set). READY.
#   Held-out catalog = the saclay-0 combined_catalog DIRECTORY (driver globs dlacat-*.fits
#   inside; the dir holds exactly one dlacat-v4.7.5-mockcat.fits + BASELINE.env + README).
#   Held-out mockdir = saclay's NATIVE juraLy8-124 dir holding the truth
#   (hcd_truth_cat.fits), zcat.fits/seed_zcat.fits and a real native snr_cat.fits — NOTHING
#   staged (unlike london-0). Truth is saclay's hcd_truth_cat.fits (cols
#   NHI/Z/TARGETID/DLAID/SNR), the SAME schema as the 2lpt calibration truth.
# =====================================================================================
S0_CAT="/gpfs/accounts/cavestru_root/cavestru0/mfho/gl_prod_saclay0_v1_20260630/combined_catalog"
S0_MOCKDIR="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124"
S0_TRUTH="${S0_MOCKDIR}/hcd_truth_cat.fits"
S0_BAL="${S0_MOCKDIR}/bal_cat.fits"

# --- output (writable scratch; created if absent) ------------------------------------
OUT="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/crossmock_alpha/tf_saclay"
mkdir -p "${OUT}"

echo "============================================================================"
echo "[$(date)] crossmock_saclay START (job ${SLURM_JOB_ID:-local})"
echo "  REPO          : ${REPO}"
echo "  frozen kernel : ${FORWARD}"
echo "  frozen molly  : ${MOLLY}"
echo "  held-out cat  : ${S0_CAT}"
echo "  held-out truth: ${S0_TRUTH}"
echo "  held-out mock : ${S0_MOCKDIR}"
echo "  OUT           : ${OUT}"
echo "  n_mc=${N_MC} seed=${SEED} variant=${VARIANT} zbins=${ZBINS} limits=${REPORT_LIMITS}"
echo "============================================================================"

# Sanity: the FROZEN calibration artifacts must exist (else the leg is not meaningful).
for f in "${FORWARD}" "${MOLLY}" "${CAL_TRUTH}" "${CAL_BAL}"; do
    if [[ ! -e "${f}" ]]; then
        echo "[FATAL] frozen 2LPT-0 calibration artifact missing: ${f}" >&2
        exit 2
    fi
done
if [[ ! -e "${CAL_CAT}" ]]; then
    echo "[FATAL] 2LPT-0 calibration catalog missing: ${CAL_CAT}" >&2
    exit 2
fi

# Sanity: the held-out saclay-0 truth + QSO/snr catalog must exist (R0 needs truth).
s0_ok=1
[[ -e "${S0_CAT}" ]]                  || { echo "[saclay] missing held-out catalog dir: ${S0_CAT}"; s0_ok=0; }
[[ -f "${S0_TRUTH}" ]]                || { echo "[saclay] missing saclay-0 TRUTH hcd_truth_cat.fits: ${S0_TRUTH}"; s0_ok=0; }
[[ -f "${S0_MOCKDIR}/snr_cat.fits" ]] || { echo "[saclay] missing saclay-0 snr_cat.fits: ${S0_MOCKDIR}/snr_cat.fits"; s0_ok=0; }
if [[ ! -f "${S0_MOCKDIR}/zcat.fits" && ! -f "${S0_MOCKDIR}/seed_zcat.fits" ]]; then
    echo "[saclay] missing saclay-0 zcat.fits / seed_zcat.fits in ${S0_MOCKDIR}"; s0_ok=0
fi
if [[ "${s0_ok}" -ne 1 ]]; then
    echo "[FATAL] held-out saclay-0 data not staged — cannot score R0." >&2
    exit 3
fi

echo "[$(date)] SACLAY LEG — frozen 2LPT-0 recipe on held-out saclay-0 (DIFFERENT recipe, no refit)"
python CDDF_analysis/hbi/track_c_tf_saclay.py \
    --variant "${VARIANT}" \
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
    --out "${OUT}" \
    --report-out "${OUT}/report.md"
echo "[$(date)] SACLAY LEG DONE -> ${OUT}"

echo "============================================================================"
echo "[$(date)] crossmock_saclay DONE"
echo "  Read R0(z) / integrated R0 from ${OUT}/report.md + ${OUT}/track_c_tf_saclay.json"
echo "  Headline = PRE-rescale R0 (verdict tables / integrated_R0); meanflux_rescale.s is"
echo "  a secondary diagnostic only."
echo "============================================================================"
