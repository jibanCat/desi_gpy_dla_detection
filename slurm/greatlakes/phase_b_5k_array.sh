#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -J tau_eb_phase_b
#SBATCH --array=0-15
#SBATCH -o slurm/greatlakes/phase_b_%A_%a.log
#SBATCH -e slurm/greatlakes/phase_b_%A_%a.log
#
# Phase B: full production bayes (BASELINE + ENABLED τ-EB) on 5000
# random 2LPT spectra, distributed across 16 SLURM array tasks.
#
# Array slicing: 5000 / 16 = 312.5; tasks 0-14 do 313 spectra,
# task 15 does 320. Per task wall ~3-7 hours at 16 CPUs.
#
# Inputs (env-overridable):
#   TARGETS_TSV   /path/to/random_2lpt_5k_z2.tsv  (skip-header is automatic)
#   OUT_BASE      /path/where/results_go/  (per-task chunk_<id>.tsv lands here)
#   N_TOTAL       5000
#   N_CHUNKS      16  (matches --array=0-15)
#
# Submit:
#   sbatch slurm/greatlakes/phase_b_5k_array.sh
#
# Aggregate after all tasks complete:
#   head -1 $OUT_BASE/chunk_0.tsv > $OUT_BASE/phase_b_5k.tsv
#   for i in $(seq 0 15); do tail -n +2 $OUT_BASE/chunk_$i.tsv; done >> $OUT_BASE/phase_b_5k.tsv

set -eo pipefail
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$HOME/.local/usr/local/lib64:${LD_LIBRARY_PATH:-}"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

TARGETS_TSV="${TARGETS_TSV:-/tmp/random_2lpt_5k_z2.tsv}"
OUT_BASE="${OUT_BASE:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/phase_b_${SLURM_ARRAY_JOB_ID}}"
N_TOTAL="${N_TOTAL:-5000}"
N_CHUNKS="${N_CHUNKS:-16}"

mkdir -p "$OUT_BASE"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PER_CHUNK=$(( N_TOTAL / N_CHUNKS ))
START=$(( TASK_ID * PER_CHUNK ))
if [ "$TASK_ID" -eq $(( N_CHUNKS - 1 )) ]; then
    END="$N_TOTAL"
else
    END=$(( START + PER_CHUNK ))
fi

CHUNK_OUT="$OUT_BASE/chunk_${TASK_ID}.tsv"

echo "===================================================="
echo "  PHASE B  array_id=$SLURM_ARRAY_JOB_ID  task=$TASK_ID  cpus=$(nproc)"
echo "  targets:  $TARGETS_TSV"
echo "  range:    [$START, $END)  (=${PER_CHUNK} spectra)"
echo "  out:      $CHUNK_OUT"
echo "===================================================="

python -u examples/run_tau_eb_phase_b.py \
    --targets-tsv "$TARGETS_TSV" \
    --start "$START" --end "$END" \
    --out "$CHUNK_OUT" \
    --max-workers 16 --num-dla-samples 10000

echo
echo "  task $TASK_ID DONE → $CHUNK_OUT"
