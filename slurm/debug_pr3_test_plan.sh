#!/bin/bash

#SBATCH -N 1                                  # Single CPU node
#SBATCH -C cpu                                # Perlmutter CPU partition
#SBATCH -q debug                              # Debug queue (≤30 min, full node)
#SBATCH --job-name=gpdla_pr3_tests
#SBATCH --output=gpdla_pr3_tests_%j.log
#SBATCH --error=gpdla_pr3_tests_%j.log
#SBATCH --mail-user=mfho@umich.edu
#SBATCH --mail-type=ALL
#SBATCH -A desi
#SBATCH --time=00:30:00
#SBATCH --ntasks=1                            # One bash driver, parallelism via &
#SBATCH --cpus-per-task=256                   # Full Perlmutter CPU node

# ---------------------------------------------------------------------------
# Test plan for PR #3 — GreatLakes setup + Voigt v2 + post-processing helpers
# ---------------------------------------------------------------------------
# Runs the 5-step verification described in the PR body. Steps 1–3 are
# serial (each <30 s); steps 4 and 5 are launched as background processes
# so they share the 256-cpu node.
#
# Submit with:
#     sbatch slurm/debug_pr3_test_plan.sh
#
# Reports land in $REPORTS_DIR (default: $SCRATCH/gpdla_pr3_tests_<jobid>/).
# Pipeline stdout is in the slurm log gpdla_pr3_tests_<jobid>.log.
# ---------------------------------------------------------------------------

# NOTE: do NOT use `set -u`/`set -e` at script scope — the DESI environment
# script (`desi_environment.sh`) references unset variables (`DESI_ROOT`, ...)
# during sourcing, which would abort us under `-u`. We use `pipefail` only,
# and check exit codes explicitly via PIPESTATUS / $? per step.
set -o pipefail

# ===== Environment =========================================================
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# ===== Paths ===============================================================
REPO_DIR="${REPO_DIR:-/pscratch/sd/j/jibancat/desi_gpy_dla_detection}"
DLA_RUN="${DLA_RUN:-/pscratch/sd/j/jibancat/desi-mock-gpdla-20250912-y3-learned-epoch920-filter}"
LLS_RUN="${LLS_RUN:-/pscratch/sd/j/jibancat/desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172}"
# NB: London is at /mocks/lya_forest/london/...  (no "develop/" prefix);
# Saclay is the one under /mocks/lya_forest/develop/saclay/... — don't
# confuse them. CLAUDE.md §10 has the canonical path table.
TRUTH="${TRUTH:-/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124}"

REPORTS_DIR="${REPORTS_DIR:-$SCRATCH/gpdla_pr3_tests_${SLURM_JOB_ID:-manual}}"
mkdir -p "$REPORTS_DIR"

cd "$REPO_DIR" || { echo "[fatal] could not cd $REPO_DIR"; exit 2; }

# ===== Pre-flight path checks ============================================
# Steps 4 and 5 need the multi-DLA dir, the LLS dir, the truth dir, and
# three specific files inside the truth dir. Fail early with a clear
# message rather than getting an opaque CFITSIO traceback later.
missing=0
for path in "$DLA_RUN" "$LLS_RUN" "$TRUTH" \
            "$TRUTH/dla_cat.fits" \
            "$TRUTH/dla_cat_mask_20.30.fits" \
            "$TRUTH/zcat.fits" \
            "$TRUTH/bal_cat.fits"; do
    if [[ ! -e "$path" ]]; then
        echo "[fatal] required path missing: $path"
        missing=1
    fi
done
if [[ "$missing" -eq 1 ]]; then
    echo
    echo "Override paths via --export, e.g.:"
    echo "  sbatch --export=ALL,TRUTH=/path/to/jura-124,DLA_RUN=/path/to/multi-dla,LLS_RUN=/path/to/lls slurm/debug_pr3_test_plan.sh"
    exit 2
fi

echo "============================================================"
echo " PR #3 test plan"
echo " host:           $(hostname)"
echo " job:            ${SLURM_JOB_ID:-manual}"
echo " repo:           $REPO_DIR"
echo " branch:         $(git rev-parse --abbrev-ref HEAD)"
echo " commit:         $(git rev-parse --short HEAD)"
echo " reports dir:    $REPORTS_DIR"
echo "============================================================"

# Track per-step pass/fail
declare -A STEP_RESULT

# ===== Step 1 — Test suite =================================================
echo
echo "===== Step 1: pytest (focused 93-test suite) ====="
t0=$SECONDS
python -m pytest \
    tests/test_voigt_v2_parity.py \
    tests/test_lyb_veto.py \
    tests/test_smoke_target_contamination.py \
    tests/test_cddf_mock.py \
    tests/test_cddf_calibration.py \
    tests/test_generate_samples.py \
    -v --tb=short 2>&1 | tee "$REPORTS_DIR/step1_pytest.log"
STEP_RESULT[1]=${PIPESTATUS[0]}
echo "[step1] exit=${STEP_RESULT[1]}  elapsed=$((SECONDS-t0))s"

# ===== Step 2 — Voigt v2 parity ============================================
echo
echo "===== Step 2: Voigt v2 ↔ C-extension parity ====="
t0=$SECONDS
python - 2>&1 <<'PY' | tee "$REPORTS_DIR/step2_voigt_parity.log"
from gpy_dla_detection.voigt_fast import VoigtProfile
from gpy_dla_detection.voigt_v2 import voigt_absorption
import numpy as np

wave = np.arange(3500, 5000, 0.25, dtype=np.float64)
v1 = VoigtProfile().compute_voigt_profile(wave, nhi=10**21.0, z_dla=2.5, num_lines=31)
v2 = voigt_absorption(wave, 21.0, 2.5, num_lines=31, kernel="boss-log-r2000")

assert v1.shape == v2.shape, f"shape mismatch: {v1.shape} vs {v2.shape}"
diff = float(np.max(np.abs(v1 - v2)))
print(f"max abs diff = {diff:.2e}")
assert diff < 1e-9, f"parity broken (diff={diff:.2e} > 1e-9)"
print("PASS")
PY
STEP_RESULT[2]=${PIPESTATUS[0]}
echo "[step2] exit=${STEP_RESULT[2]}  elapsed=$((SECONDS-t0))s"

# ===== Step 3 — Lyβ veto synthetic =========================================
echo
echo "===== Step 3: Lyβ veto on a synthetic catalog ====="
t0=$SECONDS
python - 2>&1 <<'PY' | tee "$REPORTS_DIR/step3_lybeta.log"
from astropy.table import Table
from gpy_dla_detection.postprocess.lyb_veto import flag_lybeta, lybeta_apparent_z

parent_z = 2.7
child_z = lybeta_apparent_z(parent_z)
print(f"parent z={parent_z:.4f}, child apparent z={child_z:.4f}")

cat = Table(
    rows=[(1, parent_z, 21.30, 0.99), (1, child_z, 20.32, 0.40)],
    names=["TARGETID", "Z_DLA", "LOG_NHI", "MODEL_P"],
)
out = flag_lybeta(cat)
print(out)
flags = list(out["LYBETA_FLAG"])
assert flags == [False, True], f"unexpected flags: {flags}"
print("PASS")
PY
STEP_RESULT[3]=${PIPESTATUS[0]}
echo "[step3] exit=${STEP_RESULT[3]}  elapsed=$((SECONDS-t0))s"

# ===== Steps 4 & 5 — heavy IO, run in parallel =============================
echo
echo "===== Step 4: production-catalog analyzer (background) ====="
echo "===== Step 5: P_DLA cut scan          (background)         ====="
t0=$SECONDS

# Step 4: Lyβ + LLS xref at the historical operating point
(
    set -o pipefail
    python examples/analyze_production_catalog.py \
        --catalog-dir "$DLA_RUN" \
        --truth   "$TRUTH/dla_cat_mask_20.30.fits" \
        --zcat    "$TRUTH/zcat.fits" \
        --lls-dir "$LLS_RUN" \
        --bal-cat "$TRUTH/bal_cat.fits" \
        --no-bal --p-dla-cut 0.99 \
        --out     "$REPORTS_DIR/step4_postproc_p99_no_bal.md" \
        2>&1 | tee "$REPORTS_DIR/step4_postproc.log"
    exit ${PIPESTATUS[0]}
) &
PID4=$!

# Step 5: P_DLA scan over {0.5, 0.9, 0.99, 0.999}
(
    set -o pipefail
    python examples/scan_pdla_cuts.py \
        --catalog-dir "$DLA_RUN" \
        --truth-dla   "$TRUTH/dla_cat_mask_20.30.fits" \
        --truth-full  "$TRUTH/dla_cat.fits" \
        --bal-cat     "$TRUTH/bal_cat.fits" \
        --no-bal \
        --p-cuts 0.5,0.9,0.99,0.999 \
        --out "$REPORTS_DIR/step5_pdla_scan.md" \
        2>&1 | tee "$REPORTS_DIR/step5_scan.log"
    exit ${PIPESTATUS[0]}
) &
PID5=$!

# Wait for both, capture per-step exit codes
wait "$PID4"; STEP_RESULT[4]=$?
wait "$PID5"; STEP_RESULT[5]=$?
echo "[step4] exit=${STEP_RESULT[4]}"
echo "[step5] exit=${STEP_RESULT[5]}  elapsed (parallel)=$((SECONDS-t0))s"

# ===== Final summary =======================================================
echo
echo "============================================================"
echo " RESULT SUMMARY"
echo "============================================================"
PASS=0; FAIL=0
for i in 1 2 3 4 5; do
    if [[ "${STEP_RESULT[$i]}" -eq 0 ]]; then
        echo "  step $i:  PASS"
        PASS=$((PASS+1))
    else
        echo "  step $i:  FAIL  (exit ${STEP_RESULT[$i]})"
        FAIL=$((FAIL+1))
    fi
done
echo "------------------------------------------------------------"
echo "  ${PASS} passed, ${FAIL} failed"
echo "  reports: $REPORTS_DIR"
echo "============================================================"

# Print the two analyzer reports inline so they land in the slurm log.
echo
echo "===== step 4 report (analyze_production_catalog) ====="
cat "$REPORTS_DIR/step4_postproc_p99_no_bal.md" 2>/dev/null \
    || echo "(report missing — see $REPORTS_DIR/step4_postproc.log for failure detail)"
echo
echo "===== step 5 report (scan_pdla_cuts) ====="
cat "$REPORTS_DIR/step5_pdla_scan.md" 2>/dev/null \
    || echo "(report missing — see $REPORTS_DIR/step5_scan.log for failure detail)"

exit $FAIL
