#!/bin/bash
# Larger-n unbiasedness test for the HCD-masked τ-EB recipe (H5).
#
# Runs ``examples/check_tau_eb_robust_mask.py`` IN PARALLEL across N_JOBS
# CPU cores on the local node — designed for the GreatLakes 16-core
# interactive sessions, not SLURM-batch.  No --time / --account; just bash.
#
# Inputs:
#   TARGETS_TSV  — TSV with header (mock target_id z_qso truth_z_dla
#                  truth_log_nhi nhi_regime spec_path zcat_path)
#                  ; defaults to /tmp/targets_dla_n90.tsv
#   N_JOBS       — parallel python invocations (default 6: each numpy
#                  thread gets ~2 CPUs on a 16-core node)
#   OUT_BASE     — directory for results.log + summary.csv
#                  (default /tmp/hcd_mask_unbiased)
#
# Usage:
#   bash slurm/greatlakes/hcd_mask_unbiased_test.sh
#   N_JOBS=8 TARGETS_TSV=/tmp/foo.tsv \
#       bash slurm/greatlakes/hcd_mask_unbiased_test.sh
set -eo pipefail
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$HOME/.local/usr/local/lib64:${LD_LIBRARY_PATH:-}"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla

TARGETS_TSV="${TARGETS_TSV:-/tmp/targets_dla_n90.tsv}"
OUT_BASE="${OUT_BASE:-/tmp/hcd_mask_unbiased_$(date +%s)}"
N_JOBS="${N_JOBS:-6}"
mkdir -p "$OUT_BASE"
RESULTS_LOG="$OUT_BASE/results.log"
SUMMARY_CSV="$OUT_BASE/summary.csv"

echo "===================================================="
echo "  HCD-MASK UNBIASEDNESS TEST  cpus=$(nproc)  jobs=$N_JOBS"
echo "  targets:  $TARGETS_TSV ($(($(wc -l < "$TARGETS_TSV") - 1)) rows)"
echo "  out:      $OUT_BASE"
echo "===================================================="

echo "mock,target_id,z_qso,truth_z,truth_log_nhi,nhi_regime,prod_map,prod_bias,eb_naive_tau,eb_naive_map,eb_naive_bias,eb_mask_tau,eb_mask_map,eb_mask_bias,wall_s" > "$SUMMARY_CSV"

run_one() {
    local entry="$1"
    local out_base="$2"
    IFS=$'\t' read -r mock tid z_qso truth_z truth_n regime spec zcat <<< "$entry"
    local logf="$out_base/per_target/$mock.$tid.log"
    mkdir -p "$out_base/per_target"

    local t0
    t0=$(date +%s.%N)
    local tau_factors_arg
    tau_factors_arg="${TAU_FACTORS:-0.5 0.75 1.0 1.25 1.5 2.0 3.0 4.0}"
    if ! timeout 1500 python -u examples/check_tau_eb_robust_mask.py \
            --target-id "$tid" \
            --spec "$spec" \
            --zcat "$zcat" \
            --truth-z "$truth_z" --truth-log-nhi "$truth_n" \
            --tau-factors $tau_factors_arg \
            --mask-threshold-sigma 1.5 \
            > "$logf" 2>&1; then
        echo "$mock,$tid,$z_qso,$truth_z,$truth_n,$regime,nan,nan,nan,nan,nan,nan,nan,nan,nan" >> "$out_base/summary.csv"
        echo "  [error] target $tid timed out or crashed"
        return
    fi
    local t1
    t1=$(date +%s.%N)
    local dt
    dt=$(awk -v a="$t1" -v b="$t0" 'BEGIN{print a-b}')

    # Parse the three summary lines from the log:
    #   "(1) production τ=1.0          tau_best  MAP NHI  bias  logL_truth"
    #   "(2) EB naive                  ..."
    #   "(2) EB + HCD-mask (1.5σ)      ..."
    local prod_line eb_naive_line eb_mask_line
    prod_line=$(awk '/\(1\) production/ {print; exit}' "$logf")
    eb_naive_line=$(awk '/\(2\) EB naive/ {print; exit}' "$logf")
    eb_mask_line=$(awk '/\(2\) EB \+ HCD-mask/ {print; exit}' "$logf")

    parse() {
        echo "$1" | awk '{n=NF; print $(n-3), $(n-2), $(n-1)}'
    }
    local prod eb_naive eb_mask
    prod=$(parse "$prod_line")
    eb_naive=$(parse "$eb_naive_line")
    eb_mask=$(parse "$eb_mask_line")
    local prod_tau prod_map prod_bias eb_naive_tau eb_naive_map eb_naive_bias eb_mask_tau eb_mask_map eb_mask_bias
    read prod_tau prod_map prod_bias <<< "$prod"
    read eb_naive_tau eb_naive_map eb_naive_bias <<< "$eb_naive"
    read eb_mask_tau eb_mask_map eb_mask_bias <<< "$eb_mask"

    {
        flock -x 9
        echo "$mock,$tid,$z_qso,$truth_z,$truth_n,$regime,$prod_map,$prod_bias,$eb_naive_tau,$eb_naive_map,$eb_naive_bias,$eb_mask_tau,$eb_mask_map,$eb_mask_bias,$dt" >> "$out_base/summary.csv"
    } 9>"$out_base/.csv.lock"
    echo "  [ok] $mock $tid  prod=$prod_bias  eb_mask=$eb_mask_bias  ${dt}s"
}
export -f run_one

# Drive in parallel via xargs.
tail -n +2 "$TARGETS_TSV" | \
    xargs -d "\n" -I '{}' -P "$N_JOBS" \
    bash -c 'run_one "$1" "$2"' _ '{}' "$OUT_BASE" 2>&1 | \
    tee "$RESULTS_LOG"

echo
echo "ALL_DONE  rows in summary.csv: $(wc -l < "$SUMMARY_CSV")"
echo "Summary: $SUMMARY_CSV"
