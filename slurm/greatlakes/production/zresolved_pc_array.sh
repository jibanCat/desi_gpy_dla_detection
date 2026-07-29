#!/bin/bash
# z-resolved P/C across the four mocks, one array task per mock.
#
# The reduction itself is a handful of boolean masks; runtime is dominated by
# the catalog load + truth matching (~15-20 min/mock), so this is ~1.5 CPU-h
# total. It goes to SLURM rather than the login node because that node has only
# 4 CPUs and habitually sits at load ~20.
#
# Usage:
#   sbatch slurm/greatlakes/production/zresolved_pc_array.sh
#
# Binning and output root come from the environment, so re-binning needs no
# edit here -- both are forwarded to examples/run_zresolved_pc.sh:
#
#   export ZDLA_BINS=2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8
#   export OUT_ROOT=/scratch/cavestru_root/cavestru0/mfho/zresolved_pc_dz0.2
#   sbatch slurm/greatlakes/production/zresolved_pc_array.sh
#
# EXPORT THEM, do not pass them via `--export=ALL,ZDLA_BINS=...`. The --export
# list is itself comma-separated, so a comma-separated VALUE is truncated at its
# first comma: ZDLA_BINS=2.0,2.2,... silently arrives as "2.0". sbatch defaults
# to --export=ALL, so a plain `export` in the submitting shell is all it needs.
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH -J zres_pc
#SBATCH --array=0-3
#SBATCH -o slurm/greatlakes/production/logs/zres_pc_%A_%a.log
#SBATCH -e slurm/greatlakes/production/logs/zres_pc_%A_%a.log
set -eo pipefail

MOCKS=(2lpt0 2lpt1 london0 saclay0)
MOCK="${MOCKS[$SLURM_ARRAY_TASK_ID]}"

# NOT `set -u` here: /etc/bashrc dereferences BASHRCSOURCED unset, and conda's
# shell hook trips on unbound vars too, so -u kills the job in ~2 seconds.
source ~/.bashrc
conda activate gpdla
export LD_LIBRARY_PATH="${HOME}/.local/usr/local/lib64:${LD_LIBRARY_PATH:-}"
# Pin BLAS to 1 thread: this workload is IO- and mask-bound, and oversubscribing
# BLAS on a shared node cost 3.7-8.9x on this cluster before.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHON="${PYTHON:-python}"

cd "${SLURM_SUBMIT_DIR:-/home/mfho/desi_gpy_dla_detection}"

# Fail fast. The bin edges are not consumed until AFTER the catalog load and
# truth match, so a malformed value costs ~6 min per task before it surfaces.
for v in ZDLA_BINS ZQSO_BINS; do
    val="${!v:-}"
    if [ -n "$val" ] && [ "${val#*,}" = "$val" ]; then
        echo "FATAL: $v='$val' has no comma -- need >=2 edges. If you passed it" >&2
        echo "       via --export=ALL,$v=..., the comma-separated value was" >&2
        echo "       truncated at its first comma; export it instead." >&2
        exit 2
    fi
done

echo "=== task ${SLURM_ARRAY_TASK_ID} -> ${MOCK}"
echo "=== ZDLA_BINS=${ZDLA_BINS:-<default>}  ZQSO_BINS=${ZQSO_BINS:-<default>}"
echo "=== OUT_ROOT=${OUT_ROOT:-<default>}"
echo "=== code $(git rev-parse HEAD)"

bash examples/run_zresolved_pc.sh "${MOCK}"
