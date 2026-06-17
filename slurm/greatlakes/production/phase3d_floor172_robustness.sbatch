#!/bin/bash
# slurm/greatlakes/production/phase3d_floor172_robustness.sbatch
#
# Finding-5 FLOOR ROBUSTNESS CHECK (fit floor 17.2). Self-contained job that
# REBUILDS the calibrated phase-3d posterior kernel at FIT FLOOR 17.2 (so the
# keep_in_base detection cut + the molly C/rho extend down through the LLS/
# sub-DLA band 17.2-19.5), BROADENS it in-place to sigma=0.12 dex (the calibrated
# config), then runs the v3 bspbody MAP fit (stage 2) and the WALL-1 tilt closure
# (stage 3). The headline test: do the UNTILTED R0(dN/dX>=20.3) and R0(Omega>=20.3)
# at floor 17.2 AGREE with the floor-19.5 calibrated values (1.090 / 1.029)?
# Disagreement => including the FP-dominated low-N data pulls the DLA slope (a
# problem); agreement => the parametric family is robust to the fit floor.
#
# The kernel kappa rows are op_base-ordered (S2N>2 & P_DLA>0.99 & DLAFLAG==0, NO
# predicted-NHI floor) so the kappa MATRIX is mathematically independent of the
# fit floor; the floor only changes (a) keep_in_base (which rows enter the fit),
# (b) build_A_ib's active-column floor (lowered to 17.2 here), and (c) the molly
# matrix C/rho (the floor-17.2 lya_only matrix supplies the LLS-band cells). The
# rebuild is still done from scratch so the run carries its OWN floor-17.2
# provenance and the in-build molly consistency gate (regenerate_molly_counts vs
# the TSV, hard <=5e-3) is enforced against the floor-17.2 catalog cut.
#
# DISCIPLINE: analysis-side only — NEVER touches dla_gp.py / run_bayes_select.py /
# inference. Identical paths/knobs/env to phase3d_postkernel_staged.sbatch; the
# ONLY differences are FIT_FLOOR=17.2, the floor-17.2 lya_only MOLLY, LAM_RF_MIN=
# 1025.0, and the chained build->broaden->stages2/3 sequence in one job.

#SBATCH -A cavestru1
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=150G
#SBATCH -t 08:00:00
#SBATCH -J phase3d_floor172
#SBATCH -o slurm/greatlakes/production/logs/phase3d_floor172_%j.log
#SBATCH -e slurm/greatlakes/production/logs/phase3d_floor172_%j.err

set -eo pipefail
export PYTHONUNBUFFERED=1

# BLAS threads pinned to 1 (file-parallel joblib processes must not oversubscribe).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/.bashrc
conda activate gpdla
export LD_LIBRARY_PATH="${HOME}/.local/usr/local/lib64:${LD_LIBRARY_PATH}"

REPO="/home/mfho/desi_gpy_dla_detection"
cd "${REPO}"
mkdir -p slurm/greatlakes/production/logs

# --- floor-17.2 robustness configuration (the ONLY deltas from the calibrated run) ---
OUT="${OUT_OVERRIDE:-/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phase3d_experiments/floor172_lyaonly1025_broaden012}"
# floor-17.2 lya_only molly (the G0 LLS-coverage matrix; NHI bins extend to 17.2)
MOLLY="${MOLLY_OVERRIDE:-/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/figures_molly_nhi172/molly_matrix.tsv}"
FIT_FLOOR="${FIT_FLOOR:-17.2}"
LAM_RF_MIN="${LAM_RF_MIN:-1025.0}"   # Lya-only forest (pair with the lya_only molly)
BROADEN_SIGMA="${BROADEN_SIGMA:-0.12}"

# --- shared paths/knobs (identical to the calibrated config) ---
CAT="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/"
TRUTH="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"
BAL="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"
PROCESSED_GLOB="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/outputs/figures/processed/processed-spectra-16-*.h5"
PWSAMPLES="/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_225_100000.mat"

LAMBDA_BSPBODY="${LAMBDA_BSPBODY:-30.0}"
DALPHA="${DALPHA:-0.5}"
MC_NRESTART="${MC_NRESTART:-2}"
EDGE_SLOPE_LAM="${EDGE_SLOPE_LAM:-40.0}"
GL_NODES="${GL_NODES:-1}"
FIT_CEIL="${FIT_CEIL:-99.0}"
FAMILY="${FAMILY:-bspbody}"
ZBINS="${ZBINS:-2.0,2.5,3.0,3.5}"
HOST_TRUTH_FLOOR=19.0
REPORT_LIMITS="20.0,20.3,20.6"
NJOBS=16

mkdir -p "${OUT}"

# HARD PRECONDITION: the FROZEN WALL-1 gate doc must exist BEFORE any stage.
GATE_DOC="${OUT}/WALL1_GATE_FROZEN.md"
if [[ ! -f "${GATE_DOC}" ]]; then
    echo "[FATAL] FROZEN gate doc missing: ${GATE_DOC}" >&2
    exit 2
fi
echo "[gate] FROZEN criteria: ${GATE_DOC}"
sha256sum "${GATE_DOC}" || md5sum "${GATE_DOC}" || true

KERNEL_NPZ="${OUT}/posterior_kernel_2lpt0.npz"

run_stage () {
    local ST="$1"
    echo "[$(date)] phase3d_floor172 STAGE=${ST} start (job ${SLURM_JOB_ID:-local}, ${NJOBS} workers)"
    python CDDF_analysis/run_phase3d_postkernel.py \
        --stage "${ST}" \
        --out "${OUT}" \
        --catalog-dir "${CAT}" \
        --truth "${TRUTH}" \
        --bal-cat "${BAL}" \
        --molly-tsv "${MOLLY}" \
        --processed-glob "${PROCESSED_GLOB}" \
        --pw-samples "${PWSAMPLES}" \
        --zbins "${ZBINS}" \
        --report-limits "${REPORT_LIMITS}" \
        --family "${FAMILY}" \
        --fit-floor "${FIT_FLOOR}" \
        --fit-ceil "${FIT_CEIL}" \
        --lambda-bspbody "${LAMBDA_BSPBODY}" \
        --mc-n-restart "${MC_NRESTART}" \
        --lam-rf-min "${LAM_RF_MIN}" \
        --edge-slope-lam "${EDGE_SLOPE_LAM}" \
        --gl-nodes "${GL_NODES}" \
        --dalpha "${DALPHA}" \
        --host-truth-floor "${HOST_TRUTH_FLOOR}" \
        --n-mc 200 \
        --seed 0 \
        --n-jobs "${NJOBS}"
    echo "[$(date)] phase3d_floor172 STAGE=${ST} DONE -> ${OUT}"
}

# --- STAGE 1: build the NARROW floor-17.2 kernel over all 1150 processed-h5 ---
run_stage 1

# --- BROADEN the rebuilt kernel in-place to sigma=0.12 dex (calibrated config) ---
# NOTE: broaden_kernel.py writes via np.savez, which APPENDS '.npz' if the out path
# does not already end in '.npz'. Use a '_broad.npz' temp name (ends in .npz so no
# auto-suffix) and then mv it onto the cache, so the broadened kernel replaces the
# narrow one in-place at KERNEL_NPZ.
echo "[$(date)] broaden_kernel sigma=${BROADEN_SIGMA} (in-place) -> ${KERNEL_NPZ}"
BROAD_TMP="${OUT}/posterior_kernel_2lpt0_broad.npz"
python CDDF_analysis/broaden_kernel.py "${BROADEN_SIGMA}" "${KERNEL_NPZ}" "${BROAD_TMP}"
mv -f "${BROAD_TMP}" "${KERNEL_NPZ}"
echo "[$(date)] broaden DONE -> ${KERNEL_NPZ}"

# --- STAGE 2 (v3 point fit) + STAGE 3 (WALL-1 tilt closure) on the broadened kernel ---
run_stage 2
run_stage 3

echo "[$(date)] phase3d_floor172 ALL STAGES DONE -> ${OUT}"
