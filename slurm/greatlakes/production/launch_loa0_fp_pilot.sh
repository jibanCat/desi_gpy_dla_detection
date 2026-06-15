#!/usr/bin/env bash
# launch_loa0_fp_pilot.sh — PILOT for the loa-0 forest-FP run (CDDF catalog-HBI).
#
# Submits ONE sbatch covering 4 healpix files (indices 0..4, inner STEP=2 ->
# two level2 slices, run 2-at-a-time by the N=2 packing) on the loa-0 mock,
# using the byte-identical production inference path (loa0_fp_gl_v1.env ->
# 2lpt0_gl_v1.env). This is the MANDATORY first step before scaling: the
# ~16x tau-EB cost uncertainty + the HCD-free forest (more candidate peaks may
# survive FILTER, since there are no real absorbers to "win") means per-spectrum
# cost on loa-0 could differ from loa-124. The pilot measures it for real.
#
# Cost (from the loa-124 production log, ~51 CPU-h/healpix, ~167 CPU-s/spec):
#   ~4 healpix x 51 CPU-h ~= 205 CPU-h, wall ~6.4 h (one sbatch, under the 12h -t).
#
# After it finishes:
#   1. Inspect outputs:    ls $OUTDIR/processed-*.h5
#   2. Check the per-spectrum timing in the mock_run logs vs the loa-124 ~167 CPU-s.
#   3. Sanity-check the FP rate: detections here are ALL forest false positives.
#   4. Re-estimate the full-subsample cost from the measured pilot cost, then
#      launch the full subsample (launch_loa0_fp_full.sh) if cost is acceptable.
#
# Usage:
#   bash slurm/greatlakes/production/launch_loa0_fp_pilot.sh            # submit
#   bash slurm/greatlakes/production/launch_loa0_fp_pilot.sh --dry-run  # preview
#
# DO NOT submit during cluster maintenance / without the PI go-ahead.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pilot: 4 healpix files. --start 0 --end 0 => exactly one sbatch; --window 4 =>
# END_INDEX=START_INDEX+4 (two STEP=2 level2 slices).
exec bash "${SCRIPT_DIR}/launch_gl.sh" loa0_fp_gl_v1.env \
    --start 0 --end 0 --window 4 --no-sleep "$@"
