#!/bin/bash
# slurm/greatlakes/production/phase3d_postkernel_staged.sbatch
#
# Stage-parametrized variant of phase3d_postkernel.sbatch: runs ONE --stage of
# CDDF_analysis/run_phase3d_postkernel.py per job, so each stage gets a gating
# 4-lens review before the next is launched. Stage 1 (kernel rebuild over 1150
# processed-h5) is the heavy step; stages 2/3/s3 reuse the cached kernel and are fast.
#
#   STAGE=1            sbatch ... phase3d_postkernel_staged.sbatch   (default)
#   STAGE=2  / 3       sbatch --export=ALL,STAGE=2 ...
#   STAGES="2 3"       sbatch --export=ALL,STAGES="2 3" ...   (run several in order)
#   STAGE=s3_all       sbatch --export=ALL,STAGE=s3_all ...
#
# DISCIPLINE: analysis-side only — NEVER touches dla_gp.py / run_bayes_select.py /
# inference. Identical paths/knobs/env to phase3d_postkernel.sbatch.

#SBATCH -A cavestru1
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 16
# stage 1 (kernel rebuild) reads base_sample_inds (nq*3*S int32 ~0.8GB/file) + winning
# sll columns across 16 workers -> 48G OOM'd (job 51779167). Standard nodes have 180G;
# 150G gives ~9GB/worker. Light stages 2/3/s3 over-request harmlessly on a -N1 node.
#SBATCH --mem=150G
#SBATCH -t 08:00:00
#SBATCH -J phase3d_staged
#SBATCH -o slurm/greatlakes/production/logs/phase3d_staged_%j.log
#SBATCH -e slurm/greatlakes/production/logs/phase3d_staged_%j.err

set -eo pipefail
export PYTHONUNBUFFERED=1
STAGES="${STAGES:-${STAGE:-1}}"   # space-separated list of --stage values, run in order

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

OUT="${OUT_OVERRIDE:-/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phase3d_postkernel_out}"
CAT="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/"
TRUTH="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"
BAL="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"
MOLLY="${MOLLY_OVERRIDE:-/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/figures_molly/molly_matrix.tsv}"
PROCESSED_GLOB="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/outputs/figures/processed/processed-spectra-16-*.h5"
PWSAMPLES="/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_172_225_100000.mat"

mkdir -p "${OUT}"

# HARD PRECONDITION: the FROZEN WALL-1 gate doc must exist BEFORE any stage (the
# committed PASS/KILL/beta-discipline criteria, frozen+hashed before the out-of-sample
# re-run). Required for stage 3; checked for every stage so the freeze can never be
# (re)written after seeing rebuilt-kernel results.
GATE_DOC="${OUT}/WALL1_GATE_FROZEN.md"
if [[ ! -f "${GATE_DOC}" ]]; then
    echo "[FATAL] FROZEN gate doc missing: ${GATE_DOC}" >&2
    exit 2
fi
echo "[gate] FROZEN criteria: ${GATE_DOC}"
sha256sum "${GATE_DOC}" || md5sum "${GATE_DOC}" || true

FIT_FLOOR="${FIT_FLOOR:-19.5}"
LAMBDA_BSPBODY="${LAMBDA_BSPBODY:-30.0}"
DALPHA="${DALPHA:-0.5}"
HOST_TRUTH_FLOOR=19.0
ZBINS="2.0,2.5,3.0,3.5"
REPORT_LIMITS="20.0,20.3,20.6"
NJOBS=16

for ST in ${STAGES}; do
echo "[$(date)] phase3d_postkernel_staged STAGE=${ST} start (job ${SLURM_JOB_ID:-local}, ${NJOBS} workers)"

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
    --fit-floor "${FIT_FLOOR}" \
    --lambda-bspbody "${LAMBDA_BSPBODY}" \
    --dalpha "${DALPHA}" \
    --host-truth-floor "${HOST_TRUTH_FLOOR}" \
    --n-mc 200 \
    --seed 0 \
    --n-jobs "${NJOBS}"

echo "[$(date)] phase3d_postkernel_staged STAGE=${ST} DONE -> ${OUT}"
done
