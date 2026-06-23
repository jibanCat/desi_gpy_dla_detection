#!/bin/bash
# slurm/greatlakes/production/remp_kernel.sbatch
#
# EMPIRICAL TRUTH-MATCH RESPONSE kernel (R_emp) — the principled width fix.
# Builds R_emp(x_true|x_hat,SNR) from the 2LPT-0 truth-match, attaches it as
# cfg._posterior_kernel_2d (SAME consumption path as the SIR posterior kernel), and
# runs the v3 bspbody fit (PIT cov68 + dN/dX + Omega) + WALL-1 tilt closure.
#
# DISCIPLINE: analysis-side only — NEVER touches dla_gp.py / run_bayes_select.py /
# inference. R_emp uses ONLY load_and_cut_catalog truth-matches (no processed-h5
# reads), so it is far lighter than the SIR-kernel stage1.

#SBATCH -A cavestru1
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -J remp_kernel
#SBATCH -o slurm/greatlakes/production/logs/remp_kernel_%j.log
#SBATCH -e slurm/greatlakes/production/logs/remp_kernel_%j.err

set -eo pipefail
export PYTHONUNBUFFERED=1

# BLAS threads pinned to 1 (joblib processes must not oversubscribe).
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

OUT="${OUT_OVERRIDE:-/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phase3d_experiments/r_emp}"
CAT="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/"
TRUTH="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"
BAL="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"
MOLLY="${MOLLY_OVERRIDE:-/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/figures_molly/molly_matrix.tsv}"

mkdir -p "${OUT}"

# Bring the FROZEN WALL-1 gate doc into the R_emp out dir (stage3 reads the committed
# PASS/KILL criteria; the doc is the SAME one used for the SIR-kernel run, hashed).
GATE_SRC="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phase3d_postkernel_out/WALL1_GATE_FROZEN.md"
GATE_DOC="${OUT}/WALL1_GATE_FROZEN.md"
if [[ ! -f "${GATE_DOC}" ]]; then
    if [[ -f "${GATE_SRC}" ]]; then
        cp "${GATE_SRC}" "${GATE_DOC}"
        echo "[gate] copied FROZEN gate doc -> ${GATE_DOC}"
    else
        echo "[FATAL] FROZEN gate doc missing: ${GATE_SRC}" >&2
        exit 2
    fi
fi
echo "[gate] FROZEN criteria: ${GATE_DOC}"
sha256sum "${GATE_DOC}" || md5sum "${GATE_DOC}" || true

# defaults match the SIR-kernel run for a like-for-like comparison
FIT_FLOOR="${FIT_FLOOR:-19.5}"
LAMBDA_BSPBODY="${LAMBDA_BSPBODY:-30.0}"
LAM_RF_MIN="${LAM_RF_MIN:-911.0}"   # 911=Lyα+Lyβ (default); 1025.0=Lyα-only forest
DALPHA="${DALPHA:-0.5}"
HOST_TRUTH_FLOOR=19.0
ZBINS="2.0,2.5,3.0,3.5"
REPORT_LIMITS="20.0,20.3,20.6"
SMOOTH_BINS="${SMOOTH_BINS:-1.0}"
N_FLOOR="${N_FLOOR:-20}"
HOST_COL="${HOST_COL:-NHI_TILT_HOST}"
NJOBS=16

STAGES="${STAGES:-build 2 3}"

for ST in ${STAGES}; do
echo "[$(date)] remp_kernel STAGE=${ST} start (job ${SLURM_JOB_ID:-local}, ${NJOBS} workers)"

python CDDF_analysis/run_remp_kernel.py \
    --stage "${ST}" \
    --out "${OUT}" \
    --catalog-dir "${CAT}" \
    --truth "${TRUTH}" \
    --bal-cat "${BAL}" \
    --molly-tsv "${MOLLY}" \
    --zbins "${ZBINS}" \
    --report-limits "${REPORT_LIMITS}" \
    --fit-floor "${FIT_FLOOR}" \
    --lambda-bspbody "${LAMBDA_BSPBODY}" \
    --lam-rf-min "${LAM_RF_MIN}" \
    --dalpha "${DALPHA}" \
    --host-truth-floor "${HOST_TRUTH_FLOOR}" \
    --smooth-bins "${SMOOTH_BINS}" \
    --n-floor "${N_FLOOR}" \
    --host-col "${HOST_COL}" \
    --n-mc 200 \
    --seed 0 \
    --n-jobs "${NJOBS}"

echo "[$(date)] remp_kernel STAGE=${ST} DONE -> ${OUT}"
done
