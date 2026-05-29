#!/usr/bin/env bash
# slurm/run_pc_for_run.sh — combine a run's HDF5 output + compute
# purity/completeness against the truth catalog for that mock.
#
# Usage:
#   bash slurm/run_pc_for_run.sh <outdir> <flavour>
#     where <flavour> is one of: london0  saclay0  2lpt0
#
# Writes:
#   <outdir>/combined.h5
#   <outdir>/purity_completeness.md
#   <outdir>/purity_completeness_raw.md  (without BAL exclusion, for comparison)

set -eo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <outdir> {london0|saclay0|2lpt0}" >&2; exit 2
fi

OUTDIR="$1"
FLAVOUR="$2"

case "$FLAVOUR" in
    london0)
        MOCKDIR="/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124"
        TRUTH="${MOCKDIR}/dla_cat.fits"          # cols: NHI, Z_DLA, TARGETID, DLAID
        ;;
    saclay0)
        MOCKDIR="/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124"
        TRUTH="${MOCKDIR}/hcd_truth_cat.fits"    # cols: NHI, Z, TARGETID, DLAID, SNR
        ;;
    2lpt0)
        MOCKDIR="/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124"
        TRUTH="${MOCKDIR}/hcd_truth_cat.fits"
        ;;
    *) echo "unknown flavour: $FLAVOUR" >&2; exit 2 ;;
esac

ZCAT="${MOCKDIR}/zcat.fits"
BAL="${MOCKDIR}/bal_cat.fits"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /global/cfs/cdirs/desi/software/desi_environment.sh main >/dev/null 2>&1

# Note: we skip combine_processed_h5.py for mocks because the mock processed-file
# naming pattern (processed-spectra-16-N.h5) doesn't match its survey/program
# filename template. analyze_production_catalog.py globs dlacat-*.fits directly,
# so a separate combined.h5 isn't needed for purity/completeness.

echo "=== catalog files ==="
ls "$OUTDIR"/dlacat-*.fits 2>/dev/null | wc -l | awk '{print "  dlacat files: " $1}'

echo
echo "=== purity/completeness — BAL-excluded (matches molly notebook convention) ==="
python3 examples/analyze_production_catalog.py \
    --catalog-dir "$OUTDIR" \
    --truth "$TRUTH" \
    --zcat "$ZCAT" \
    --bal-cat "$BAL" --no-bal \
    --truth-nhi-min 20.3 \
    --p-dla-cut 0.5 \
    --out "$OUTDIR/purity_completeness.md" 2>&1 | tail -25

echo
echo "=== purity/completeness — RAW (no BAL exclusion) ==="
python3 examples/analyze_production_catalog.py \
    --catalog-dir "$OUTDIR" \
    --truth "$TRUTH" \
    --zcat "$ZCAT" \
    --truth-nhi-min 20.3 \
    --p-dla-cut 0.5 \
    --out "$OUTDIR/purity_completeness_raw.md" 2>&1 | tail -10
echo
echo "=== done. saved: ==="
# This script no longer combines per-healpix h5 → combined.h5 (earlier steps skip it),
# so don't list combined.h5 here. With `set -e`, a missing combined.h5 would fail
# the whole job after a successful P/C run.
ls -lh "$OUTDIR"/purity_completeness*.md 2>&1
