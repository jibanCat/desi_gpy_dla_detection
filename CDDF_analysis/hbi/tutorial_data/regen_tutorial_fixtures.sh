#!/usr/bin/env bash
# Refresh the small mock-only tutorial fixtures by COPYING from the scratch
# validation cache. This is not a recompute -- the provenance is the 2LPT-0
# injection validation run (hbi_validation_2lpt0). Mock data only; no real-LOA.
set -euo pipefail
SRC=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/hbi_validation_2lpt0/figures
DST="$(cd "$(dirname "$0")" && pwd)"
for f in compare_synthesis.json compare_R0_table.md; do
    cp "$SRC/$f" "$DST/$f"
    echo "refreshed $DST/$f"
done
