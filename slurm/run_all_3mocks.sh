#!/usr/bin/env bash
# slurm/run_all_3mocks.sh — sequentially run the 3 mock-0 multi-DLA sanity
# tests directly on this node (no sbatch). Each takes ~30 min on a 256-CPU
# Perlmutter node with --parallel-files 32 --max-workers 8 (production-matching
# parallelism).
#
# Usage:
#   bash slurm/run_all_3mocks.sh <tag>
#     where <tag> becomes /pscratch/sd/j/jibancat/<tag>/{london0,saclay0,2lpt0}_y3/

set -eo pipefail
TAG="${1:?usage: $0 <tag-suffix>}"
BASE="/pscratch/sd/j/jibancat/${TAG}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /global/cfs/cdirs/desi/software/desi_environment.sh main >/dev/null 2>&1

for flavour in london0_y3 saclay0_y3 2lpt0_y3; do
    OUT="${BASE}/${flavour}/"
    echo "============================================================"
    echo "  [$flavour] start $(date +%H:%M:%S)   →  $OUT"
    echo "============================================================"
    time bash slurm/run_local.sh "slurm/configs/${flavour}.env" \
        --outdir "$OUT" \
        --window 32 --end 0 --parallel-files 32 --max-workers 8
    echo "  [$flavour] inference done $(date +%H:%M:%S)"

    short="${flavour%_y3}"  # london0 → london0, etc.
    bash slurm/run_pc_for_run.sh "$OUT" "$short" || echo "  [warn] pc step failed for $flavour"
done

echo "============================================================"
echo "  ALL 3 MOCKS DONE $(date +%H:%M:%S)"
echo "  outputs under $BASE/"
ls -d "${BASE}/"*/  2>/dev/null
echo "============================================================"
