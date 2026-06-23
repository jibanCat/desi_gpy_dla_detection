#!/bin/bash
# slurm/greatlakes/production/phase3d_postkernel.sbatch
#
# Phase-3d calibrated 2-D posterior-kernel pipeline (the 1/pi rate-form HBI fix).
# Self-contained: runs CDDF_analysis/run_phase3d_postkernel.py in three stages on a
# single GreatLakes standard node. The human submits this AFTER the login-node build
# + S2 gate are green:
#
#     sbatch slurm/greatlakes/production/phase3d_postkernel.sbatch
#
#   stage1  full-survey build_posterior_kernel over all 1150 processed-h5 (file-
#           parallel, -n 16) -> cache <out>/posterior_kernel_2lpt0.npz (float32).
#   stage2  v3 bspbody MAP fit at the pre-chosen lambda WITH the cached kernel.
#   stage3  WALL-1 both tilts (+/-dalpha), v3 estimator + kernel.
#   s3_*    the S3 falsifiers (settle kernel-vs-starvation; WALL1_GATE_FROZEN.md §B):
#             s3_prior_null      = bare-pi (likelihood removed) through kernel + WALL-1
#             s3_dense_synthetic = WALL-1 on a dense known-f_b/known-scatter injection
#
# HARD PRECONDITION: the FROZEN gate doc WALL1_GATE_FROZEN.md MUST exist in OUT before
# this batch runs (the committed PASS/KILL/beta-discipline criteria; frozen+hashed
# before the single out-of-sample re-run). The script aborts if it is missing.
#
# DISCIPLINE: analysis-side only — NEVER touches dla_gp.py / run_bayes_select.py /
# dlasearch.py / any inference. No SLURM job submission from the login-node workflow;
# this script is the ONE batch job. All reads point at /scratch + /nfs/turbo (read).

#SBATCH -A cavestru1
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -J phase3d_postkernel
#SBATCH -o slurm/greatlakes/production/logs/phase3d_postkernel_%j.log
#SBATCH -e slurm/greatlakes/production/logs/phase3d_postkernel_%j.err

set -eo pipefail
export PYTHONUNBUFFERED=1

# --- Pin BLAS threads to 1 (config-only; no inference-path code touched). The kernel
#     build is file-parallel via joblib processes; each numpy op should stay single-
#     threaded so the 16 file workers do not oversubscribe the 16 cores. -------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- conda env (no NERSC desi_environment.sh on GreatLakes) -----------------------
source ~/.bashrc
conda activate gpdla
# libcerf for the Voigt C extension (not used here, but keep the env consistent)
export LD_LIBRARY_PATH="${HOME}/.local/usr/local/lib64:${LD_LIBRARY_PATH}"

REPO="/home/mfho/desi_gpy_dla_detection"
cd "${REPO}"
mkdir -p slurm/greatlakes/production/logs

# --- all paths on /scratch (read+write) and /nfs/turbo (read-only truth) ----------
OUT="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phase3d_postkernel_out"
CAT="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/"
TRUTH="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"
BAL="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"
MOLLY="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/figures_molly/molly_matrix.tsv"
PROCESSED_GLOB="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/outputs/figures/processed/processed-spectra-16-*.h5"
PWSAMPLES="/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_225_100000.mat"

mkdir -p "${OUT}"

# --- HARD PRECONDITION: the FROZEN WALL-1 gate doc must exist BEFORE we run --------
GATE_DOC="${OUT}/WALL1_GATE_FROZEN.md"
if [[ ! -f "${GATE_DOC}" ]]; then
    echo "[FATAL] FROZEN gate doc missing: ${GATE_DOC}" >&2
    echo "        The committed PASS/KILL/beta-discipline criteria must be frozen" >&2
    echo "        (and beta hashed) BEFORE the single out-of-sample WALL-1 re-run." >&2
    exit 2
fi
echo "[gate] FROZEN criteria: ${GATE_DOC}"
sha256sum "${GATE_DOC}" || md5sum "${GATE_DOC}" || true

# --- pre-chosen fit knobs (the WALL-1 bspbody operating point) --------------------
FIT_FLOOR=19.5          # bspbody detection-row floor (v3_logN_fit_floor)
LAMBDA_BSPBODY=30.0     # 2nd-diff curvature penalty (the swept minimal-DOF rung)
DALPHA=0.5              # WALL-1 tilt magnitude (gate runs +dalpha and -dalpha)
HOST_TRUTH_FLOOR=19.0   # tilt-host truth floor (CS-review F1)
ZBINS="2.0,2.5,3.0,3.5"
# bayesian-review MAJOR fix (item 3): the FROZEN gate doc criterion (b)/(c)/(d) require
# the integrated dN/dX & Ω closure on report limits {>=20.0, >=20.3, >=20.6}, and the
# v3 untilted-R0 precondition iterates the same report_limits. Passing only "20.0,20.3"
# meant the >=20.6 headline tier (and its R0~1 precondition) was NEVER evaluated, so a
# 'PASS' would not satisfy the committed contract. Include 20.6.
REPORT_LIMITS="20.0,20.3,20.6"
NJOBS=16

echo "[$(date)] phase3d_postkernel start (job ${SLURM_JOB_ID:-local}, ${NJOBS} workers)"

python CDDF_analysis/run_phase3d_postkernel.py \
    --stage all \
    --out "${OUT}" \
    --catalog-dir "${CAT}" \
    --truth "${TRUTH}" \
    --bal-cat "${BAL}" \
    --molly-tsv "${MOLLY}" \
    --processed-glob "${PROCESSED_GLOB}" \
    --pw-samples "${PWSAMPLES}" \
    --zbins "${ZBINS}" \
    --report-limits "${REPORT_LIMITS}" \
    --fit-floor "${FIT_FLOOR}" \
    --lambda-bspbody "${LAMBDA_BSPBODY}" \
    --dalpha "${DALPHA}" \
    --host-truth-floor "${HOST_TRUTH_FLOOR}" \
    --n-mc 200 \
    --seed 0 \
    --n-jobs "${NJOBS}"

# --- S3 falsifiers (settle kernel-vs-starvation; reuse the stage-1 kernel cache) ---
echo "[$(date)] phase3d_postkernel S3 falsifiers (prior-only null + dense-synthetic)"
python CDDF_analysis/run_phase3d_postkernel.py \
    --stage s3_all \
    --out "${OUT}" \
    --catalog-dir "${CAT}" \
    --truth "${TRUTH}" \
    --bal-cat "${BAL}" \
    --molly-tsv "${MOLLY}" \
    --processed-glob "${PROCESSED_GLOB}" \
    --pw-samples "${PWSAMPLES}" \
    --zbins "${ZBINS}" \
    --report-limits "${REPORT_LIMITS}" \
    --fit-floor "${FIT_FLOOR}" \
    --lambda-bspbody "${LAMBDA_BSPBODY}" \
    --dalpha "${DALPHA}" \
    --host-truth-floor "${HOST_TRUTH_FLOOR}" \
    --n-mc 200 \
    --seed 0 \
    --n-jobs "${NJOBS}"

echo "[$(date)] phase3d_postkernel DONE -> ${OUT}"
