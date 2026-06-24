#!/bin/bash
# slurm/greatlakes/production/phase3d_postkernel_staged.sbatch
#
# Stage-parametrized variant of phase3d_postkernel.sbatch: runs ONE --stage of
# CDDF_analysis/hbi/run_phase3d_postkernel.py per job, so each stage gets a gating
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
MC_NRESTART="${MC_NRESTART:-2}"   # WALL-1 per-draw MC-refit multistart; raise to 8 to
                                  # test the gate-MC-convergence hypothesis (point uses 8)
LAM_RF_MIN="${LAM_RF_MIN:-911.0}" # rest-frame blue edge: 911.0=Lyman limit (full Lyα+Lyβ,
                                  # default); 1025.0=Lyα-only forest (pair with the
                                  # lya_only molly matrix via MOLLY_OVERRIDE)
EDGE_SLOPE_LAM="${EDGE_SLOPE_LAM:-40.0}"  # low-N edge-slope prior anchor strength; 0.0
                                  # neutralizes the fixed -1.4 slope memory (WALL-1 probe)
GL_NODES="${GL_NODES:-1}"         # within-bin density quadrature: 1=midpoint (default),
                                  # 3=Gauss-Legendre bin-mean (slope-bias structural fix)
FIT_CEIL="${FIT_CEIL:-99.0}"      # fit ceiling (99=none); set 21.0 = throw-away-high-N
                                  # (fit only low-N, extrapolate power-law above)
FAMILY="${FAMILY:-bspbody}"       # v3 CDDF family: bspbody (default) | plaw | plawcut | bplcut
ZBINS="${ZBINS:-2.0,2.5,3.0,3.5}" # reporting/fit z-bin edges (override e.g. 2.15,2.5,3.0,3.5)
HOST_TRUTH_FLOOR=19.0
REPORT_LIMITS="20.0,20.3,20.6"
NJOBS=16

for ST in ${STAGES}; do
echo "[$(date)] phase3d_postkernel_staged STAGE=${ST} start (job ${SLURM_JOB_ID:-local}, ${NJOBS} workers)"

python CDDF_analysis/hbi/run_phase3d_postkernel.py \
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

echo "[$(date)] phase3d_postkernel_staged STAGE=${ST} DONE -> ${OUT}"
done
