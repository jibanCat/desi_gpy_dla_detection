#!/bin/bash
# slurm/greatlakes/production/parallelism_sweep_gl.sh
#
# One-time calibration: find the best MAX_WORKERS / srun-packing on a GL
# standard node (36 cores) for desi-DLAGP.py mock inference.
#
# Two phases, all on ONE exclusive node:
#   A) Latency sweep — sequential, one srun at a time, MAX_WORKERS ∈
#      {1,2,4,8,16}. Measures per-spectrum compute time vs worker count
#      (isolated, no contention). Same level2 slice each cell so the
#      QSOs are identical → apples-to-apples.
#   B) Concurrency check — launch N concurrent srun's at a candidate
#      width and measure aggregate throughput, to catch memory-bandwidth
#      contention that the isolated latency sweep can't see.
#
# Per-spectrum cost is read from the "time spent: XmYs" log lines (these
# exclude the one-time model+sample load), so short timeboxes are fine.
#
# Submit:
#   sbatch slurm/greatlakes/production/parallelism_sweep_gl.sh
# Then analyse the logs under $SWEEP_OUT (see the companion analysis the
# driver runs afterward).

# NB: 32 single-cpu tasks (not --exclusive). --exclusive whole-node
# requests sat in PD with "Reserved for maintenance" (overlap with
# class/retirement node holds); an explicit 32-core request schedules
# immediately, matches the production sizing, and lets srun --exact
# carve out variable-width steps (up to -c16) from the 32 cpus.
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J gl_par_sweep
#SBATCH -o slurm/greatlakes/production/logs/par_sweep_%j.log
#SBATCH -e slurm/greatlakes/production/logs/par_sweep_%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1

# --- env ----------------------------------------------------------------------
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla
export LD_LIBRARY_PATH="$HOME/.local/usr/local/lib64:${LD_LIBRARY_PATH:-}"

NCORES=$(nproc)
echo "[sweep] node=$(hostname) cores=$NCORES job=$SLURM_JOB_ID start=$(date)"

# --- paths (V1 candidate config — same as london0_gl_v1.env) -----------------
SWEEP_OUT="/scratch/cavestru_root/cavestru0/mfho/gl_parallelism_sweep_$(date +%Y%m%d)"
mkdir -p "$SWEEP_OUT"

# Catalogs + samples on cavestru scratch (Turbo-independent). Mock spectra
# still on Turbo (MOCKDIR) — the sweep needs Turbo back to read them.
DATA_ROOT=/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection
MOCKDIR=/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/london/qq_desi_y3/v5.9.5/mock-0/jura-124
MODEL=/scratch/cavestru_root/cavestru0/mfho/phase2_desi/2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5

# Common desi-DLAGP args (V1 candidate recipe)
common_args=(
    --qsocat "$MOCKDIR/zcat.fits" --release v5.9.5 --program dark --survey main
    --mocks --mockdir "$MOCKDIR"
    --learned_file "$MODEL"
    --catalog_name "$DATA_ROOT/data/dr12q/processed/catalog.mat"
    --los_catalog "$DATA_ROOT/data/dla_catalogs/dr9q_concordance/processed/los_catalog"
    --dla_catalog "$DATA_ROOT/data/dla_catalogs/dr9q_concordance/processed/dla_catalog"
    --dla_samples_file "$DATA_ROOT/data/dr12q/processed/pw_samples_a3_172_225_100000.mat"
    --sub_dla_samples_file "$DATA_ROOT/data/dr12q/processed/subdla_samples_a03_191_200_100000.mat"
    --min_z_separation 3000.0 --prev_tau_0 0.00246 --prev_beta 3.62
    --max_dlas 3 --plot_figures 0 --filter_low_likelihood 1 --single_absorber_model 1
    --batch_size 1250 --loading_min_lambda 910 --loading_max_lambda 1550
    --normalization_min_lambda 1425 --normalization_max_lambda 1475
    --min_lambda 911.75 --max_lambda 1250 --dlambda 0.15 --k 30
    --num_dla_samples 100000 --num_subdla_samples 100000
    --max_noise_variance 9 --num_forest_lines 3 --num_lines 3
    --enable_tau_eb 1 --tau_eb_objective null --early_stop_mode baseline
)
# NB: τ-EB ON (null objective) matches the production best baseline — and it
# adds a per-spectrum τ-grid search, so measuring throughput WITHOUT it would
# understate production per-spectrum cost. Keep it on for representative timing.

# --- Phase A: latency sweep (sequential, isolated) ---------------------------
echo "[sweep] === Phase A: latency vs MAX_WORKERS (level2 0..2, ${PHASE_A_SECS:-300}s each) ==="
for W in 1 2 4 8 16; do
    od="$SWEEP_OUT/latency_W${W}"; mkdir -p "$od/logs"
    echo "[sweep] $(date +%H:%M:%S) latency W=$W (cpus=$W)"
    timeout "${PHASE_A_SECS:-300}" srun --exact --overlap -N1 -n1 -c"$W" \
        python desi-DLAGP.py "${common_args[@]}" \
            --max_workers "$W" --outdir "$od" --figure_dir "$od/figures" \
            --level2_start 0 --level2_end 2 \
        > "$od/run.log" 2>&1
    n=$(grep -c 'time spent' "$od/run.log" 2>/dev/null || echo 0)
    echo "[sweep]   W=$W processed ~$n spectra in ${PHASE_A_SECS:-300}s"
done

# --- Phase B: concurrency check (candidate packings) -------------------------
# N concurrent srun's at width W, N*W ≈ NCORES. Aggregate throughput reveals
# contention the isolated latency sweep can't.
echo "[sweep] === Phase B: concurrency aggregate throughput (${PHASE_B_SECS:-420}s each) ==="
run_concurrency () {
    local N="$1" W="$2"
    local tag="concurrency_N${N}_W${W}"
    local base="$SWEEP_OUT/$tag"; mkdir -p "$base"
    echo "[sweep] $(date +%H:%M:%S) concurrency N=$N W=$W (=$((N*W)) cores)"
    local k
    for (( k=0; k<N; k++ )); do
        local l2s=$(( 2*k )) l2e=$(( 2*k + 2 ))
        local od="$base/srun_${k}"; mkdir -p "$od"
        timeout "${PHASE_B_SECS:-420}" srun --exact --overlap -N1 -n1 -c"$W" \
            python desi-DLAGP.py "${common_args[@]}" \
                --max_workers "$W" --outdir "$od" --figure_dir "$od/figures" \
                --level2_start "$l2s" --level2_end "$l2e" \
            > "$od/run.log" 2>&1 &
    done
    wait
    local total
    total=$(cat "$base"/srun_*/run.log 2>/dev/null | grep -c 'time spent')
    echo "[sweep]   N=$N W=$W aggregate ~$total spectra in ${PHASE_B_SECS:-420}s across $N srun's"
}
# Both fit the 32-core allocation (8*4=32, 4*8=32).
run_concurrency 8 4
run_concurrency 4 8

echo "[sweep] done $(date). results under $SWEEP_OUT"
