#!/usr/bin/env bash
# slurm/nersc/production/samplecost_sweep_nersc.sh
#
# Node-hours-vs-NUM_SAMPLES sweep. At the parallelism-sweep WINNER packing
# (N=32 × W=8 on Perlmutter), submit one debug cell per NUM_DLA_SAMPLES ∈
# {10000, 30000, 50000, 100000}. Aggregate spectra/min/node at each → real
# node-hours per dataset at each sample count, so we can pick the cheapest
# sample count that still meets the P/C target (paired with the calibration
# slices). All science knobs are byte-identical to V1 except NUM_SAMPLES.
#
#   bash slurm/nersc/production/samplecost_sweep_nersc.sh            # submit
#   bash slurm/nersc/production/samplecost_sweep_nersc.sh --dry-run
#
# Reuses _sweep_cell_nersc.sh (NUM_SAMPLES + grid files driven by --export).
# Grids must exist: pw_samples_a3_172_225_{N}.mat + subdla_samples_a03_191_200_{N}.mat
# (made by tools/make_subsampled_grids.py).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CELL_SCRIPT="${SCRIPT_DIR}/_sweep_cell_nersc.sh"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

NTASKS=32          # winning packing
W=8
TIMEBOX="${TIMEBOX:-1000}"
SWEEP_OUT="${SWEEP_OUT:-/pscratch/sd/j/jibancat/nersc_samplecost_sweep_$(date +%Y%m%d_%H%M)}"

echo "[samplecost] SWEEP_OUT=$SWEEP_OUT  packing N=${NTASKS} W=${W}  dry_run=$DRY_RUN"
[ -f "$CELL_SCRIPT" ] || { echo "[samplecost] missing cell script: $CELL_SCRIPT" >&2; exit 2; }

if [ "$DRY_RUN" -ne 1 ]; then
    mkdir -p "$SWEEP_OUT" "${SCRIPT_DIR}/logs"
fi

for N in 10000 30000 50000 100000; do
    # pre-flight: the grids must exist (skip the cell otherwise, don't fail silently)
    dla="${REPO_ROOT}/data/dr12q/processed/pw_samples_a3_172_225_${N}.mat"
    sub="${REPO_ROOT}/data/dr12q/processed/subdla_samples_a03_191_200_${N}.mat"
    if [ ! -r "$dla" ] || [ ! -r "$sub" ]; then
        echo "[samplecost] SKIP S=$N — missing grid(s): $dla / $sub" >&2
        continue
    fi
    TAG="samplecost_S${N}"
    exp="ALL,SWEEP_OUT=${SWEEP_OUT},CELL_TAG=${TAG},NTASKS=${NTASKS},W=${W},TIMEBOX=${TIMEBOX},NUM_SAMPLES=${N},CELL_SCRIPT=${CELL_SCRIPT}"
    cmd=(sbatch --chdir="$REPO_ROOT"
                --account=desi --qos=debug --constraint=cpu
                --nodes=1 --ntasks="$NTASKS" --cpus-per-task="$W"
                --time=00:30:00 --job-name="sweep_${TAG}"
                --export="$exp" "$CELL_SCRIPT")
    echo "[samplecost] cell ${TAG}  (NUM_SAMPLES=$N)"
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  %q ' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi
done

echo "[samplecost] done submitting. analyse with:"
echo "  for d in $SWEEP_OUT/samplecost_S*; do n=\$(cat \$d/srun_*/run.log 2>/dev/null | grep -c 'time spent'); echo \"\$(basename \$d): \$n spectra / ${TIMEBOX}s\"; done"
