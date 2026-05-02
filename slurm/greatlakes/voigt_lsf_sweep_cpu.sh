#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -J voigt_sweep_cpu
#SBATCH -o slurm/greatlakes/voigt_sweep_cpu_%j.log
#SBATCH -e slurm/greatlakes/voigt_sweep_cpu_%j.log

# Voigt LSF + num_lines hypothesis-test sweep on GreatLakes CPU partition.
# The inference path (Voigt v2 wofz + numpy GP + ProcessPoolExecutor) is
# CPU-bound — the GPU partition was wasting GPU time. -c 16 gives 8 inner
# workers + 8 cores headroom for the OS / parent / IO.
#
# 3 mocks × 3 NHI regimes × 4 configs × N_PER_BIN targets. With production-
# matched settings (max_workers=8, batch_size=1250) ~10-30 sec per inference.

set -eo pipefail
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$HOME/.local/usr/local/lib64:${LD_LIBRARY_PATH:-}"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

N_PER_BIN="${N_PER_BIN:-5}"
SNR_MIN="${SNR_MIN:-2.0}"
CONFIGS="${CONFIGS:-A,B,C,D}"
SEED="${SEED:-42}"

OUT_BASE="${OUT_BASE:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/voigt_sweep_${SLURM_JOB_ID}}"
TARGETS_TSV="${TARGETS_TSV:-$OUT_BASE/targets.tsv}"
RUNS_DIR="$OUT_BASE/runs"
REPORT_DIR="$OUT_BASE/report"
mkdir -p "$OUT_BASE" "$RUNS_DIR" "$REPORT_DIR"

python -c "
from gpy_dla_detection.voigt_v2 import voigt_absorption
from gpy_dla_detection.voigt_v2_inject import inject
import desispec.io
print('[preflight] imports OK')
" || { echo "[error] preflight import failed" >&2; exit 1; }

echo "===================================================="
echo "  Voigt LSF + num_lines sweep (CPU partition)"
echo "===================================================="
echo "  job: $SLURM_JOB_ID  configs: $CONFIGS  n_per_bin: $N_PER_BIN"
echo "  out: $OUT_BASE  targets: $TARGETS_TSV"
echo "  cpus: $(nproc)"
echo "===================================================="

# Step 1: pick targets — skipped if a TARGETS_TSV is already present
# (so we can re-use a known-good target list across jobs).
if [ -r "$TARGETS_TSV" ]; then
    echo
    echo "=== STEP 1: re-using existing TARGETS_TSV ==="
    echo "  $TARGETS_TSV ($(wc -l < $TARGETS_TSV) lines)"
else
    echo
    echo "=== STEP 1: pick targets ==="
    python -u examples/pick_voigt_sweep_targets.py \
        --out "$TARGETS_TSV" \
        --n-per-bin "$N_PER_BIN" \
        --snr-min "$SNR_MIN" \
        --seed "$SEED"
fi

[ -r "$TARGETS_TSV" ] || { echo "[error] no targets picked" >&2; exit 2; }

# Step 2: run sweep
echo
echo "=== STEP 2: run inference sweep ==="
python -u examples/voigt_lsf_sweep.py \
    --picked-targets "$TARGETS_TSV" \
    --out-dir "$RUNS_DIR" \
    --configs "$CONFIGS"

[ -r "$RUNS_DIR/master.csv" ] || { echo "[error] master.csv not produced" >&2; exit 3; }

# Step 3: analyze
echo
echo "=== STEP 3: analyze + report ==="
python -u examples/analyze_voigt_sweep.py \
    --master "$RUNS_DIR/master.csv" \
    --out-dir "$REPORT_DIR"

DATE_TAG=$(date +%Y-%m-%d)
DOC_DIR="$REPO_DIR/docs/notes/${DATE_TAG}_voigt_lsf_sweep"
mkdir -p "$DOC_DIR"
cp -r "$REPORT_DIR"/* "$DOC_DIR/"

echo
echo "=== VOIGT SWEEP COMPLETE ==="
echo "  base:   $OUT_BASE"
echo "  csv:    $RUNS_DIR/master.csv"
echo "  report: $DOC_DIR/report.md"
