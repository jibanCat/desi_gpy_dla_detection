#!/bin/bash
#SBATCH -A cavestru0
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -J hcd_mask_n54
#SBATCH -o slurm/greatlakes/hcd_mask_n54_%j.log
#SBATCH -e slurm/greatlakes/hcd_mask_n54_%j.log

# Scale-out test of the HCD-masked τ-EB recipe on 54 targets (6 per
# mock × NHI-regime cell). 3 mocks × 3 regimes × 6 = 54.
#
# Driver: examples/check_tau_eb_robust_mask.py per target. CPU-bound;
# runs serial at ~5 min/target → ~4-5 h total. Output captured to
# /nfs/turbo/lsa-cavestru/.../voigt_sweep_hcd_mask_<jobid>/results.log
# and parsed into a summary CSV.
#
# Submit:
#   sbatch slurm/greatlakes/hcd_mask_scale_out.sh

set -eo pipefail
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$HOME/.local/usr/local/lib64:${LD_LIBRARY_PATH:-}"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

TARGETS_TSV="${TARGETS_TSV:-/tmp/voigt_scale_out/targets_n54.tsv}"
OUT_BASE="${OUT_BASE:-/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/hcd_mask_scale_${SLURM_JOB_ID}}"
mkdir -p "$OUT_BASE"
RESULTS_LOG="$OUT_BASE/results.log"
SUMMARY_CSV="$OUT_BASE/summary.csv"

echo "===================================================="
echo "  HCD-MASK SCALE-OUT  job: $SLURM_JOB_ID  cpus: $(nproc)"
echo "  targets: $TARGETS_TSV ($(wc -l < $TARGETS_TSV) lines)"
echo "  out:     $OUT_BASE"
echo "===================================================="

# Header for summary CSV
echo "mock,target_id,z_qso,truth_z,truth_log_nhi,nhi_regime,prod_map,prod_bias,eb_naive_tau,eb_naive_map,eb_naive_bias,eb_mask_tau,eb_mask_map,eb_mask_bias" > "$SUMMARY_CSV"

# Iterate: tail -n +2 to skip header
total=$(($(wc -l < "$TARGETS_TSV") - 1))
i=0
tail -n +2 "$TARGETS_TSV" | while IFS=$'\t' read -r mock tid z_qso truth_z truth_n regime spec zcat; do
    i=$((i + 1))
    echo "" | tee -a "$RESULTS_LOG"
    echo "[$i/$total] mock=$mock tid=$tid z_qso=$z_qso truth_z=$truth_z truth_NHI=$truth_n regime=$regime" | tee -a "$RESULTS_LOG"
    echo "==================================================================" | tee -a "$RESULTS_LOG"

    OUT=$(timeout 900 python -u examples/check_tau_eb_robust_mask.py \
        --target-id "$tid" \
        --spec "$spec" \
        --zcat "$zcat" \
        --truth-z "$truth_z" --truth-log-nhi "$truth_n" \
        --tau-factors 0.5 0.75 1.0 1.25 1.5 2.0 \
        --mask-threshold-sigma 1.5 \
        2>&1 | grep -vE "^INFO|^ERROR|^DESI|^WARNING") || {
        echo "  [error] target $tid timed out or crashed" | tee -a "$RESULTS_LOG"
        echo "$mock,$tid,$z_qso,$truth_z,$truth_n,$regime,nan,nan,nan,nan,nan,nan,nan,nan" >> "$SUMMARY_CSV"
        continue
    }
    echo "$OUT" | tee -a "$RESULTS_LOG"

    # Parse the summary lines from the output
    # (1) production τ=1.0   ...   MAP_NHI    bias
    # (2) EB naive            ...   MAP_NHI    bias
    # (2) EB + HCD-mask       ...   MAP_NHI    bias
    PROD_LINE=$(echo "$OUT" | awk '/\(1\) production/ {print; exit}')
    EB_NAIVE_LINE=$(echo "$OUT" | awk '/\(2\) EB naive/ {print; exit}')
    EB_MASK_LINE=$(echo "$OUT" | awk '/\(2\) EB \+ HCD-mask/ {print; exit}')

    # Each line: "(N) label              tau_best   MAP NHI   bias   logL_truth"
    # Need columns 3, 4 (or 5, 6) — parse carefully
    parse() {
        # echo "$1" → returns "tau_best MAP_NHI bias"
        echo "$1" | awk '{n=NF; print $(n-3), $(n-2), $(n-1)}'
    }
    PROD=$(parse "$PROD_LINE")
    EB_NAIVE=$(parse "$EB_NAIVE_LINE")
    EB_MASK=$(parse "$EB_MASK_LINE")

    PROD_TAU=$(echo $PROD | cut -d' ' -f1)
    PROD_MAP=$(echo $PROD | cut -d' ' -f2)
    PROD_BIAS=$(echo $PROD | cut -d' ' -f3)
    EB_NAIVE_TAU=$(echo $EB_NAIVE | cut -d' ' -f1)
    EB_NAIVE_MAP=$(echo $EB_NAIVE | cut -d' ' -f2)
    EB_NAIVE_BIAS=$(echo $EB_NAIVE | cut -d' ' -f3)
    EB_MASK_TAU=$(echo $EB_MASK | cut -d' ' -f1)
    EB_MASK_MAP=$(echo $EB_MASK | cut -d' ' -f2)
    EB_MASK_BIAS=$(echo $EB_MASK | cut -d' ' -f3)

    echo "$mock,$tid,$z_qso,$truth_z,$truth_n,$regime,$PROD_MAP,$PROD_BIAS,$EB_NAIVE_TAU,$EB_NAIVE_MAP,$EB_NAIVE_BIAS,$EB_MASK_TAU,$EB_MASK_MAP,$EB_MASK_BIAS" >> "$SUMMARY_CSV"
done

echo "" | tee -a "$RESULTS_LOG"
echo "ALL_DONE  rows in summary.csv: $(wc -l < $SUMMARY_CSV)" | tee -a "$RESULTS_LOG"
echo "" | tee -a "$RESULTS_LOG"
echo "===================================================="
echo "  HCD-MASK SCALE-OUT COMPLETE"
echo "  log:     $RESULTS_LOG"
echo "  summary: $SUMMARY_CSV"
echo "===================================================="
