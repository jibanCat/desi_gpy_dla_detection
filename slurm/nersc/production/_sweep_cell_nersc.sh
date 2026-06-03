#!/bin/bash
# slurm/nersc/production/_sweep_cell_nersc.sh
#
# ONE parallelism-sweep cell = ONE sbatch job (submitted by
# parallelism_sweep_nersc.sh). Runs N concurrent desi-DLAGP.py tasks on one
# Perlmutter node via a SINGLE `srun -n N` (NOT shell-backgrounded `srun &`,
# which fails on NERSC with "step creation disabled, retrying (nodes busy)").
# Each task picks its own level2 slice + outdir from $SLURM_PROCID.
#
# Re-entrant: the cell branch (sbatch) launches `srun ... "$CELL_SCRIPT"` with
# SWEEP_TASK=1, which re-enters this same file in the task branch.
#
# Driven entirely by --export vars (set by the driver):
#   SWEEP_OUT   shared output root for the whole sweep
#   CELL_TAG    e.g. latency_W16 or concurrency_N16_W16
#   NTASKS      concurrent tasks on the node (1 for latency cells)
#   W           MAX_WORKERS / cpus-per-task for each task
#   TIMEBOX     seconds to let each task crunch (the `timeout`)
#   CELL_SCRIPT absolute path to THIS file (so srun re-execs the right script)

#SBATCH -A desi
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J nersc_sweep_cell
#SBATCH -o slurm/nersc/production/logs/sweep_cell_%j.log
#SBATCH -e slurm/nersc/production/logs/sweep_cell_%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1

# BLAS pinned to 1 thread per worker (config-only; inference code untouched).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO_ROOT="/pscratch/sd/j/jibancat/desi_gpy_dla_detection"
MOCKDIR="/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124"
MODEL="/global/cfs/cdirs/desicollab/users/jibancat/DLA/learned/phase2_desi/2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5"
cd "$REPO_ROOT"

# QMC sample count + grid files. Default = PW100k (the parallelism-sweep recipe);
# the sample-count cost sweep overrides NUM_SAMPLES to {10000,30000,50000}.
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
DLA_SAMPLES_FILE="${DLA_SAMPLES_FILE:-${REPO_ROOT}/data/dr12q/processed/pw_samples_a3_172_225_${NUM_SAMPLES}.mat}"
SUBDLA_SAMPLES_FILE="${SUBDLA_SAMPLES_FILE:-${REPO_ROOT}/data/dr12q/processed/subdla_samples_a03_191_200_${NUM_SAMPLES}.mat}"

# V1 production recipe — byte-identical science knobs to london0_nersc_v1.env.
# Run at the real recipe so cell throughput doubles as a Perlmutter cost figure.
common_args=(
    --qsocat "$MOCKDIR/zcat.fits" --release v5.9.5 --program dark --survey main
    --mocks --mockdir "$MOCKDIR"
    --learned_file "$MODEL"
    --catalog_name "$REPO_ROOT/data/dr12q/processed/catalog.mat"
    --los_catalog "$REPO_ROOT/data/dla_catalogs/dr9q_concordance/processed/los_catalog"
    --dla_catalog "$REPO_ROOT/data/dla_catalogs/dr9q_concordance/processed/dla_catalog"
    --dla_samples_file "$DLA_SAMPLES_FILE"
    --sub_dla_samples_file "$SUBDLA_SAMPLES_FILE"
    --min_z_separation 3000.0 --prev_tau_0 0.00246 --prev_beta 3.62
    --max_dlas "${MAX_DLAS_CELL:-4}" --plot_figures 0 --filter_low_likelihood "${FILTER_CELL:-1}" --single_absorber_model 1
    --batch_size 1250 --loading_min_lambda 910 --loading_max_lambda 1550
    --normalization_min_lambda 1425 --normalization_max_lambda 1475
    --min_lambda 911.75 --max_lambda 1250 --dlambda 0.15 --k 30
    --num_dla_samples "$NUM_SAMPLES" --num_subdla_samples "$NUM_SAMPLES"
    --max_noise_variance 9 --num_forest_lines 31 --num_lines 3
    --enable_tau_eb 1 --tau_eb_objective null --early_stop_mode baseline
)

# ---------------------------------------------------------------------------
# TASK branch: one srun task. Pick a unique slice + outdir from SLURM_PROCID.
# ---------------------------------------------------------------------------
if [ "${SWEEP_TASK:-0}" = "1" ]; then
    k="${SLURM_PROCID:-0}"
    l2s=$(( 2 * k )); l2e=$(( 2 * k + 2 ))
    od="${SWEEP_OUT}/${CELL_TAG}/srun_${k}"; mkdir -p "$od/figures"
    timeout "${TIMEBOX}" python desi-DLAGP.py "${common_args[@]}" \
        --max_workers "$W" --outdir "$od" --figure_dir "$od/figures" \
        --level2_start "$l2s" --level2_end "$l2e" \
        > "$od/run.log" 2>&1
    exit 0
fi

# ---------------------------------------------------------------------------
# CELL branch: the sbatch job. Set up env, launch N tasks with ONE srun.
# ---------------------------------------------------------------------------
# desi_environment.sh references unbound vars (DESI_ROOT) — relax -u around it.
set +u
source /global/cfs/cdirs/desi/software/desi_environment.sh main
set -u

mkdir -p "${SWEEP_OUT}/${CELL_TAG}"
echo "[cell] $(date) job=${SLURM_JOB_ID:-NA} node=$(hostname) cell=${CELL_TAG} N=${NTASKS} W=${W} timebox=${TIMEBOX}s"

# Single srun, N tasks, W cpus each — no shell backgrounding.
srun -N 1 -n "${NTASKS}" -c "${W}" --cpu-bind=cores \
     --export=ALL,SWEEP_TASK=1 "${CELL_SCRIPT}"

total=$(cat "${SWEEP_OUT}/${CELL_TAG}"/srun_*/run.log 2>/dev/null | grep -c 'time spent' || echo 0)
echo "[cell] $(date) cell=${CELL_TAG} aggregate ~${total} spectra in ${TIMEBOX}s across ${NTASKS} task(s)"
