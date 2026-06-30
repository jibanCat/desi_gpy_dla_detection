#!/bin/bash
# slurm/greatlakes/production/crossmock_alpha_tf.sbatch
#
# Track-C task #40 — the NON-CIRCULAR cross-mock alpha(z) ~= 1 validation.
# =====================================================================================
# WHY THIS RUN EXISTS
#   On-mock, alpha = 1/R0 is a TAUTOLOGY: dividing the 2LPT-0 recovery by 2LPT-0's own
#   truth is residual-zero by construction. The genuine, non-circular proof is to take
#   the recipe CALIBRATED ONLY on 2LPT-0 (the FROZEN forward-response kernel + the FROZEN
#   z-resolved completeness g(N,z) + the FROZEN molly C/rho ratio matrix) and apply it,
#   WITH NO REFIT of alpha and NO REFIT of the kernel, to HELD-OUT mock catalogs. We then
#   check R0(z) = HBI(recovered) / held-out-truth ~= 1 per z-bin. R0(z) ~= 1 on data the
#   estimator never saw = a real generalization claim.
#
#   Two legs (escalating difficulty):
#     LEG 1  2lpt-1  (track_c_tf_2lpt1.py)  — SAME recipe (lyacolore_2lpt v2.8.5),
#                                             different realization (gentlest transfer).
#     LEG 2  london-0 (track_c_tf_london0.py) — DIFFERENT recipe (CoLoRe/london v5.9.5);
#                                             the STRONGEST transfer test.
#   (track_c_tf_loa.py is REAL LOA — DELIBERATELY EXCLUDED here. Mock-only, shareable.)
#
# WHAT IS FROZEN (the crux — nothing below is re-fit on the held-out mock):
#   * forward-response kernel   -> forward_response_2lpt0.npz  (FROZEN file)
#   * z-resolved completeness   -> built ONCE on 2LPT-0 truth-match, stashed on
#                                  cfg._cnz_resolved (NOT rebuilt on the held-out mock)
#   * molly C/rho RATIO matrix  -> the 2LPT-0 lya_only-nhi195 matrix (FROZEN file)
#   Each driver rebuilds ONLY the 2LPT-0 calibration ingredients (cheap, ~couple min),
#   extracts the frozen pieces, then scores the held-out catalog against ITS OWN truth.
#   Variant A = fully frozen (2LPT-0 molly counts too; real-data-applicable).
#   Variant B = kernel+g frozen, molly COUNT denominators rebuilt on the held-out mock
#               (A/B localizes kernel-transfer vs completeness-count-transfer).
#
# MEAN-FLUX RESCALE / RESIDUAL CIRCULARITY (Track-C task #40 note):
#   NEITHER driver applies a mean-flux rescale to the recovery. The 2lpt-1 driver has
#   none at all. The london-0 driver's fit_meanflux_rescale() is a DIAGNOSTIC ONLY — it
#   fits a scalar s to the held-out truth and REPORTS what s would do, but it does NOT
#   modify the recovered dN/dX. The headline per-z R0 and integrated R0 emitted in the
#   verdict tables / JSON are ALREADY the PRE-rescale (non-circular) values. So there is
#   no flag to disable — and none is needed: read the verdict-table R0 / integrated_R0
#   as the result, and treat meanflux_rescale.s in the JSON as a secondary diagnostic.
#
# DISCIPLINE: analysis / reduce-only. NO GP inference is run here — gpy_dla_detection/
#   is byte-FROZEN, dla_gp.py / run_bayes_select.py / dlasearch.py are untouched, and the
#   estimator logic (cddf_catalog_hbi.py) is not edited. All held-out mock data is
#   shareable (mocks only); real-LOA paths are never read.
#
# IDEMPOTENT / SAFE TO RE-RUN: each leg writes into its own OUT subdir (figure + JSON +
#   report.md are overwritten, never appended). The 2lpt-1 leg is GATED on its truth +
#   QSO/snr catalog being present; if they are not yet staged the leg is SKIPPED with a
#   loud BLOCKER message and the london-0 leg still completes. Once the 3-file rsync lands
#   (see the BLOCKER block below) re-running this script UNCHANGED picks up leg 1
#   automatically.
#
# SUBMIT WITH:
#     sbatch slurm/greatlakes/production/crossmock_alpha_tf.sbatch
# =====================================================================================

#SBATCH -A yueyingn0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -J crossmock_alpha
#SBATCH -o slurm/greatlakes/production/logs/crossmock_alpha_%j.log
#SBATCH -e slurm/greatlakes/production/logs/crossmock_alpha_%j.err

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
N_MC=200                     # MC draws for the band (headline rigor; prior leg ran 120)
SEED=0
VARIANT="both"              # A (fully frozen) + B (counts rebuilt) -> kernel/completeness A/B

# --- output base (writable scratch; created if absent) -------------------------------
OUTBASE="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/crossmock_alpha"
OUT_2LPT1="${OUTBASE}/tf_2lpt1"
OUT_LONDON0="${OUTBASE}/tf_london0"
mkdir -p "${OUT_2LPT1}" "${OUT_LONDON0}"

echo "============================================================================"
echo "[$(date)] crossmock_alpha_tf START (job ${SLURM_JOB_ID:-local})"
echo "  REPO          : ${REPO}"
echo "  frozen kernel : ${FORWARD}"
echo "  frozen molly  : ${MOLLY}"
echo "  n_mc=${N_MC} seed=${SEED} variant=${VARIANT} zbins=${ZBINS} limits=${REPORT_LIMITS}"
echo "============================================================================"

# Sanity: the FROZEN calibration artifacts must exist (else neither leg is meaningful).
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

# =====================================================================================
# LEG 1 — held-out 2lpt-1 (SAME recipe, different realization).
#   GATED: the mock-1 mockdir (truth + zcat/seed_zcat + snr_cat) is currently NOT staged
#   on GreatLakes (only mock-0 exists under .../lyacolore_2lpt/.../v2.8.5/). R0 needs the
#   mock-1 TRUTH (denominator) and the mock-1 zcat/snr (full pathlength X_tot over ALL
#   sightlines, not just detected ones). If absent we SKIP this leg loudly; london-0 still
#   runs. When the rsync lands, this leg runs automatically with no edit.
# =====================================================================================
# NOTE: pass the catalog DIRECTORY (not the .fits file) — the loader globs dlacat-*.fits
# inside it, exactly like the london-0 leg (L0_CAT). The dir holds one dlacat-v2.8.5-mockcat.fits.
C1_CAT="/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/2lpt1_loa124_v1"
C1_MOCKDIR="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-1/loa-124"
C1_TRUTH="${C1_MOCKDIR}/hcd_truth_cat.fits"
C1_BAL="${C1_MOCKDIR}/bal_cat.fits"

c1_ok=1
[[ -e "${C1_CAT}" ]]   || { echo "[leg1] missing held-out catalog: ${C1_CAT}"; c1_ok=0; }
[[ -f "${C1_TRUTH}" ]] || { echo "[leg1] missing 2lpt-1 TRUTH (R0 denominator): ${C1_TRUTH}"; c1_ok=0; }
if [[ ! -f "${C1_MOCKDIR}/snr_cat.fits" ]]; then
    echo "[leg1] missing 2lpt-1 snr_cat.fits in ${C1_MOCKDIR}"; c1_ok=0
fi
if [[ ! -f "${C1_MOCKDIR}/zcat.fits" && ! -f "${C1_MOCKDIR}/seed_zcat.fits" ]]; then
    echo "[leg1] missing 2lpt-1 zcat.fits / seed_zcat.fits in ${C1_MOCKDIR}"; c1_ok=0
fi

if [[ "${c1_ok}" -eq 1 ]]; then
    echo "[$(date)] LEG 1 — frozen 2LPT-0 recipe on held-out 2lpt-1 (no refit)"
    python CDDF_analysis/hbi/track_c_tf_2lpt1.py \
        --variant "${VARIANT}" \
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
        --report-out "${OUT_2LPT1}/report.md"
    echo "[$(date)] LEG 1 DONE -> ${OUT_2LPT1}"
else
    echo "============================================================================" >&2
    echo "[BLOCKER] LEG 1 (2lpt-1) SKIPPED — held-out mock-1 catalog tree not staged."  >&2
    echo "  The 2lpt-1 dlacat is present, but R0 = recovered / 2lpt-1-truth needs the"   >&2
    echo "  mock-1 truth + QSO/snr catalogs, which are NOT on GreatLakes (only mock-0"    >&2
    echo "  exists under .../lyacolore_2lpt/qq_desi_y3/v2.8.5/). Stage these 3-4 files"   >&2
    echo "  into ${C1_MOCKDIR}/ (mirror the mock-0 layout), e.g. rsync from NERSC"        >&2
    echo "  /global/cfs/projectdirs/desi/mocks/lya_forest/develop/lyacolore_2lpt/"        >&2
    echo "  qq_desi_y3/v2.8.5/mock-1/loa-124/ :"                                          >&2
    echo "      hcd_truth_cat.fits   (R0 denominator / truth)"                            >&2
    echo "      zcat.fits  (or seed_zcat.fits)   (pathlength X_tot)"                      >&2
    echo "      snr_cat.fits   (truth-match SNR)"                                         >&2
    echo "      bal_cat.fits   (optional; no_bal=True so unused, stage for parity)"       >&2
    echo "  Then re-run this script UNCHANGED — leg 1 will execute automatically."        >&2
    echo "============================================================================" >&2
fi

# =====================================================================================
# LEG 2 — held-out london-0 (DIFFERENT recipe; the STRONGEST transfer test). READY.
#   Held-out catalog = the london0 catalog DIRECTORY (driver globs dlacat-*.fits inside).
#   Held-out mockdir = the pre-staged scratch dir holding london truth (dla_cat.fits),
#   zcat.fits (symlinks to the read-only london mock) + a real snr_cat.fits built from the
#   catalog's processed-h5 (london has no native snr_cat). Truth is london's dla_cat.fits
#   (cols NHI/Z_DLA/TARGETID/DLAID), NOT hcd_truth_cat.fits.
# =====================================================================================
L0_CAT="/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/london0_jura124_v1"
L0_MOCKDIR="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_london0/mockdir"
L0_TRUTH="${L0_MOCKDIR}/dla_cat.fits"
L0_BAL="${L0_MOCKDIR}/bal_cat.fits"

l0_ok=1
[[ -e "${L0_CAT}" ]]                       || { echo "[leg2] missing held-out london-0 catalog dir: ${L0_CAT}"; l0_ok=0; }
[[ -e "${L0_TRUTH}" ]]                      || { echo "[leg2] missing london-0 TRUTH dla_cat.fits: ${L0_TRUTH}"; l0_ok=0; }
[[ -f "${L0_MOCKDIR}/snr_cat.fits" ]]      || { echo "[leg2] missing london-0 snr_cat.fits: ${L0_MOCKDIR}/snr_cat.fits"; l0_ok=0; }
[[ -e "${L0_MOCKDIR}/zcat.fits" ]]         || { echo "[leg2] missing london-0 zcat.fits: ${L0_MOCKDIR}/zcat.fits"; l0_ok=0; }

if [[ "${l0_ok}" -eq 1 ]]; then
    echo "[$(date)] LEG 2 — frozen 2LPT-0 recipe on held-out london-0 (DIFFERENT recipe, no refit)"
    python CDDF_analysis/hbi/track_c_tf_london0.py \
        --variant "${VARIANT}" \
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
        --report-out "${OUT_LONDON0}/report.md"
    echo "[$(date)] LEG 2 DONE -> ${OUT_LONDON0}"
else
    echo "[BLOCKER] LEG 2 (london-0) SKIPPED — held-out london-0 data not staged (unexpected)." >&2
fi

echo "============================================================================"
echo "[$(date)] crossmock_alpha_tf DONE"
echo "  leg1 (2lpt-1) : $([[ ${c1_ok} -eq 1 ]] && echo 'ran -> '${OUT_2LPT1} || echo 'SKIPPED (blocked, see above)')"
echo "  leg2 (london0): $([[ ${l0_ok} -eq 1 ]] && echo 'ran -> '${OUT_LONDON0} || echo 'SKIPPED')"
echo "  Read R0(z) / integrated R0 from report.md + track_c_tf_*.json in each OUT dir."
echo "  Headline = PRE-rescale R0 (verdict tables / integrated_R0); meanflux_rescale.s is"
echo "  a secondary diagnostic only (london-0)."
echo "============================================================================"
