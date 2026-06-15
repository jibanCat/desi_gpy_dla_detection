#!/usr/bin/env bash
# launch_loa0_fp_full.sh — FULL subsample for the loa-0 forest-FP run.
#
# RUN THE PILOT FIRST (launch_loa0_fp_pilot.sh) AND CONFIRM THE COST before this.
#
# Submits a contiguous ~230-healpix subsample (indices 0..230) of the loa-0
# mock on the byte-identical production inference path (loa0_fp_gl_v1.env). The
# inherited OUTER_STEP=12 / OUTER_WINDOW=10 tiling from the production env means
# launch_gl.sh emits one sbatch per 10-index chunk (= 5 level2 slices each),
# stepping by 12 — i.e. ceil(230/12) ~= 20 sbatch jobs, each ~12h -t / 128G /
# N=2 x W=16, exactly as production. A 60s sleep between submits (production
# default) avoids a scheduler/disk-write storm.
#
# Why ~230 healpix:
#   - The catalog-HBI forest-FP rate needs enough quasars across (N̂, SNR, z)
#     bins to constrain the <~20 sub-DLA/LLS tiers. 230/1150 ~= 20% of the mock
#     ~= 240k quasars (pre-cut) — comparable to the 5k validation slices scaled
#     up, with FP-rate bin counts dominated by the SNR>2, z in [2,3.5] bulk.
#   - Cost (from loa-124 prod, ~51 CPU-h/healpix): ~230 x 51 ~= 11,800 CPU-h.
#     This sits at the upper end of the planned ~4.5-9k band because the prod
#     per-spec cost (maxdla4 + tau-EB null = ~167 CPU-s) is heavy; the PILOT
#     re-measures it on loa-0 and you can shrink/grow the range here before
#     launching. For a leaner ~7.7k CPU-h, use 150 healpix (--end 150 below).
#
# To change the subsample size, edit FULL_END (number of healpix files):
FULL_END="${FULL_END:-230}"
#
# Usage:
#   bash slurm/greatlakes/production/launch_loa0_fp_full.sh             # submit ~230 hpx
#   FULL_END=150 bash slurm/greatlakes/production/launch_loa0_fp_full.sh  # leaner 150 hpx
#   bash slurm/greatlakes/production/launch_loa0_fp_full.sh --dry-run   # preview all sbatch
#
# DO NOT submit during cluster maintenance / without the PI go-ahead.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --start 0 --end FULL_END: launch_gl.sh tiles [0..FULL_END] in OUTER_STEP=12
# chunks (window=10), one sbatch per chunk, identical to the production launch.
exec bash "${SCRIPT_DIR}/launch_gl.sh" loa0_fp_gl_v1.env \
    --start 0 --end "$FULL_END" "$@"
