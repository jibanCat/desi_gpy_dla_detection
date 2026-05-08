#!/bin/bash
# Overnight self-driving chain for the wider-grid A.3+A.4 verification.
#
# Waits for the current A.3 short retrain (lane=both) to land v3.5.npz,
# then runs:
#   - tests/compare_a3_results.py            → comparison.png + SUMMARY.md
#   - tests/plot_a3_corr_matrices.py         → corr_grid_large.png +
#                                               corr_delta_grid.png
#   - tests/a4_inference.py --lanes v1 v3.5  → canonical TID inference
#
# Then commits the result artifacts + pushes.
#
# MATLAB lane on the wider grid is intentionally NOT redone; the
# MATLAB-vs-Python optimizer-comparison story already established at
# the narrow grid (commit b090ace). The wider-grid A.3 only needs
# v1 vs v3.5 to confirm the dlog_β term-B benign-ness still holds.

set -e
cd "$(git rev-parse --show-toplevel)"

PY=/home/mfho/.conda/envs/gpdla/bin/python

# 1) wait for v3.5 to land (after current run started at 19:49)
echo "[$(date)] waiting for v3.5.npz to land (timestamp > 1778281000)..."
until [ -f tests/fixtures/2lpt_frozen/short_retrain/v3.5.npz ] && \
      [ $(date +%s -r tests/fixtures/2lpt_frozen/short_retrain/v3.5.npz 2>/dev/null) -gt 1778281000 ]; do
  sleep 30
done
echo "[$(date)] v3.5.npz landed."

# 2) comparison plots + summary
echo "[$(date)] generating comparison + corr plots..."
$PY tests/compare_a3_results.py
$PY tests/plot_a3_corr_matrices.py

# 3) A.4 inference (v1 + v3.5; no matlab on wider grid)
echo "[$(date)] running A.4 inference..."
$PY tests/a4_inference.py --lanes v1 v3.5

# 4) commit + push
echo "[$(date)] committing artifacts..."
git add tests/fixtures/2lpt_frozen/ tests/short_retrain_2lpt.py tests/plot_a3_corr_matrices.py
git diff --cached --stat | tail -20

git commit -m "$(cat <<'EOF'
A.3 + A.4 retest on wider [850.75, 1700, 0.15] fixture (5662 px)

Re-ran A.3 (v1 + v3.5 only, both Adam) and A.4 against the new wider
2LPT preload (jobs 49628373 + 49628374, ~50 min wall each, 9.5 GB and
8.9 GB trainset.h5 outputs). Drove by:
  - new fixture builder: build_2lpt_frozen_test_fixture.py points at
    2lpt_loa0_wide_v2_1778186324; uses v1's effective_optical_depth
    with num_forest_lines=31 (MATLAB DR16 convention).
  - new 6 frozen TIDs picked from inside the wider trainset with
    real SNR_FOREST/REDSIDE diversity.
  - thread cap baked in: short_retrain_2lpt.py now sets
    OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS=1 at module
    load. Per-iter time dropped from ~145s under default thread
    storm to ~14s with the cap (10× speedup; 22 min total for 50
    iter × 2 lanes).
  - plot_a3_corr_matrices.py now tolerant to missing matlab lane.

Step A re-validation on wider fixture:
  A.1 Jacobian:       max rel_err 4.22e-5  ✓ (dlog_β APPROX 2.05e-2)
  A.2 v1 ≡ MATLAB:    max rel_err 5.30e-11 ✓
  A.2 v3.5 vs MATLAB: dlog_β diverges 0.70-2.05% as predicted
  A.3 + A.4 endpoints documented in:
    tests/fixtures/2lpt_frozen/short_retrain/SUMMARY.md
    tests/fixtures/2lpt_frozen/short_retrain/canonical_tid_summary.md

The old narrow-grid endpoints (commits b090ace + 1f7b024) are now
superseded for the v3 production rebuild; kept in git history for
diff reference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git push

echo "[$(date)] DONE — see PR at https://github.com/jibanCat/desi_gpy_dla_detection/pull/6"
