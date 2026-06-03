#!/usr/bin/env bash
# slurm/nersc/production/parallelism_sweep_nersc.sh
#
# NERSC (Perlmutter) parallelism calibration DRIVER. Submits ONE sbatch job per
# sweep cell to the `debug` QOS (fast turnaround), each running a single
# `srun -n N` on one node — NO shell-backgrounded srun's (that pattern fails on
# NERSC with "step creation disabled, retrying (nodes busy)"). Run on the login
# node:
#   bash slurm/nersc/production/parallelism_sweep_nersc.sh            # submit
#   bash slurm/nersc/production/parallelism_sweep_nersc.sh --dry-run  # print only
#
# WHY: GL found N=2 × W=16 the throughput optimum on a 36-core Intel node; the
# original NERSC scripts assumed N=32 × W=8. A Perlmutter CPU node is 128
# physical / 256 logical AMD Milan cores with much higher memory bandwidth, so
# the optimal packing differs and must be measured on the real hardware. Cells
# run the V1 recipe (PW100k, nfl31, single-absorber, tau-EB null), so aggregate
# throughput also yields a real Perlmutter per-spectrum cost.
#
# After the jobs finish:
#   python slurm/nersc/production/analyze_sweep.py "$SWEEP_OUT" 256

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CELL_SCRIPT="${SCRIPT_DIR}/_sweep_cell_nersc.sh"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# Timeboxes (seconds) — fit inside the debug 30-min wall incl. ~1-2 min model load.
PHASE_A_SECS="${PHASE_A_SECS:-900}"    # latency cells (1 task)
PHASE_B_SECS="${PHASE_B_SECS:-1000}"   # concurrency cells (N tasks)

# Shared output root for the whole sweep (timestamped). NOT created under --dry-run.
SWEEP_OUT="${SWEEP_OUT:-/pscratch/sd/j/jibancat/nersc_parallelism_sweep_$(date +%Y%m%d_%H%M)}"

# Cell list: "TAG NTASKS W TIMEBOX"
#   Phase A latency (1 task, isolated per-spec cost). W=1,2 skipped: at PW100k
#   nfl31 they don't complete enough spectra inside the debug wall; the packing
#   decision rests on Phase B aggregate throughput anyway.
#   Phase B concurrency: 4 packings that each fill the 256-logical-core node.
CELLS=(
    "latency_W8     1  8  ${PHASE_A_SECS}"
    "latency_W16    1 16  ${PHASE_A_SECS}"
    "latency_W32    1 32  ${PHASE_A_SECS}"
    "concurrency_N16_W16 16 16 ${PHASE_B_SECS}"
    "concurrency_N32_W8  32  8 ${PHASE_B_SECS}"
    "concurrency_N8_W32   8 32 ${PHASE_B_SECS}"
    "concurrency_N64_W4  64  4 ${PHASE_B_SECS}"
)

echo "[sweep-driver] REPO_ROOT=$REPO_ROOT"
echo "[sweep-driver] SWEEP_OUT=$SWEEP_OUT"
echo "[sweep-driver] cell_script=$CELL_SCRIPT  dry_run=$DRY_RUN"
[ -f "$CELL_SCRIPT" ] || { echo "[sweep-driver] missing cell script: $CELL_SCRIPT" >&2; exit 2; }

if [ "$DRY_RUN" -ne 1 ]; then
    mkdir -p "$SWEEP_OUT" "${SCRIPT_DIR}/logs"
fi

for spec in "${CELLS[@]}"; do
    read -r TAG NTASKS W TIMEBOX <<< "$spec"
    exp="ALL,SWEEP_OUT=${SWEEP_OUT},CELL_TAG=${TAG},NTASKS=${NTASKS},W=${W},TIMEBOX=${TIMEBOX},CELL_SCRIPT=${CELL_SCRIPT}"
    cmd=(sbatch --chdir="$REPO_ROOT"
                --account=desi --qos=debug --constraint=cpu
                --nodes=1 --ntasks="$NTASKS" --cpus-per-task="$W"
                --time=00:30:00
                --job-name="sweep_${TAG}"
                --export="$exp" "$CELL_SCRIPT")
    echo "[sweep-driver] cell ${TAG}  (N=${NTASKS} W=${W} timebox=${TIMEBOX}s)"
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  %q ' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi
done

echo "[sweep-driver] submitted ${#CELLS[@]} cell(s) to the debug QOS"
echo "[sweep-driver] when done: python slurm/nersc/production/analyze_sweep.py $SWEEP_OUT 256"
