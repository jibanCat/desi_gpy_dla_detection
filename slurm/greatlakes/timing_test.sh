#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 0:20:00
#SBATCH -J gpdla-timing
#SBATCH -o slurm/greatlakes/timing_%j.log
#SBATCH -e slurm/greatlakes/timing_%j.log
set -euo pipefail
# Defaults below are the GreatLakes paths that this harness was developed
# against. Override via --export=ALL,REPO_DIR=...,DATA_ROOT=...,SPEC=...,ZCAT=...
# when submitting from a different machine. PYTHON defaults to whatever's
# on PATH after the env is set up.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$HOME/.local/usr/local/lib64}"
REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

PY="${PYTHON:-${PY:-python}}"
DATA_ROOT="${DATA_ROOT:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection}"
SPEC="${SPEC:-/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits}"
ZCAT="${ZCAT:-/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits}"

run() {
  local preset=$1
  echo "=== ${preset} on compute node $(hostname) ==="
  /usr/bin/time -f 'wall=%es' $PY examples/smoke_one_spectrum.py \
    --specfile $SPEC --zcat $ZCAT --target-id 120046865 \
    --preset $preset \
    --data-root $DATA_ROOT \
    --dla-samples-file     $DATA_ROOT/data/dr12q/processed/dla_samples_a03.mat \
    --sub-dla-samples-file $DATA_ROOT/data/dr12q/processed/subdla_samples.mat \
    --single-absorber-model 0 --max-dlas 4 --filter-low-likelihood 1 \
    --num-dla-samples 10000 --num-subdla-samples 10000 \
    --max-workers 8 --batch-size 1250 \
    --output out/smoke/${preset}_compute_120046865.h5 2>&1 | grep -E 'inference took|wall=|MAP|p\(>=1' | tail -6
  echo
}

run eboss
run y3
echo "=== done ==="
