#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --job-name=loa_compress
#SBATCH --output=slurm/loa_compress_%j.log
#SBATCH --error=slurm/loa_compress_%j.err
#SBATCH -A desi
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Compress raw LOA healpix coadds → single HDF5 (LoaArchive) for fast
# inference IO. Real-data fast path via desispec.io.read_spectra +
# coadd_cameras (no resampling needed for LOA).
#
# Output: ${OUTPUT} containing per-QSO flux/ivar/mask/wave/fwhm and
# (with --with-resolution) the full 11-band R matrix. ~110 GB for
# full LOA z>=2 with R; ~10 GB without.
#
# Smoke-test mode: pass HEALPIX_LIST=10351,10452,10531 to do a
# 285-QSO / ~30 s build before launching the full job.
#
# Submit (smoke):
#   sbatch --export=ALL,HEALPIX_LIST=10351,10452,10531,WITH_RES=1 \
#          slurm/compress_loa_archive_nersc.sh
#
# Submit (full LOA z>=2 with R):
#   sbatch --export=ALL,WITH_RES=1 slurm/compress_loa_archive_nersc.sh
#
# Submit (full LOA z>=2 without R, ~10 GB out):
#   sbatch slurm/compress_loa_archive_nersc.sh

set -eo pipefail
export PYTHONUNBUFFERED=1

source /global/cfs/cdirs/desi/software/desi_environment.sh main || {
    echo "[error] failed to load desi env" >&2; exit 1
}

# Inputs.
QSOCAT="${QSOCAT:-/global/cfs/cdirs/desi/science/lya/y3/loa/catalogs/QSO_cat_loa_main_dark_healpix_v2-altbal-20241115.fits}"
LOA_ROOT="${LOA_ROOT:-/global/cfs/cdirs/desi/spectro/redux/loa}"

# Output: default to a self-describing name under the user's pscratch.
OUTDIR_BASE="${OUTDIR_BASE:-/pscratch/sd/j/jibancat/desi_gpy_dla_detection}"
TAG="${TAG:-loa_archive_${SLURM_JOB_ID}}"
OUTPUT="${OUTPUT:-${OUTDIR_BASE}/loa_archives/${TAG}.h5}"

# Filters.
Z_MIN="${Z_MIN:-2.0}"
Z_MAX="${Z_MAX:-5.0}"
SPECTYPE="${SPECTYPE:-QSO}"
HEALPIX_LIST="${HEALPIX_LIST:-}"      # empty = all
MAX_SPECTRA="${MAX_SPECTRA:-}"        # empty = no cap
LIMIT_FIBERS="${LIMIT_FIBERS:-}"      # smoke-mode per-coadd cap

# Toggles.
WITH_RES="${WITH_RES:-0}"             # 1 → store full 11-band R (~3× size)

[ -r "$QSOCAT" ]   || { echo "[error] QSOCAT: $QSOCAT" >&2; exit 3; }
[ -d "$LOA_ROOT" ] || { echo "[error] LOA_ROOT: $LOA_ROOT" >&2; exit 4; }
mkdir -p "$(dirname "$OUTPUT")"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

EXTRA_FLAGS=()
[ "$WITH_RES" = "1" ] && EXTRA_FLAGS+=(--with-resolution)
[ -n "$HEALPIX_LIST" ] && EXTRA_FLAGS+=(--healpix-list "$HEALPIX_LIST")
[ -n "$MAX_SPECTRA" ]  && EXTRA_FLAGS+=(--max-spectra "$MAX_SPECTRA")
[ -n "$LIMIT_FIBERS" ] && EXTRA_FLAGS+=(--limit-fibers-per-coadd "$LIMIT_FIBERS")

echo "===================================================="
echo "  LOA ARCHIVE COMPRESSION  job: $SLURM_JOB_ID"
echo "  -q regular -C cpu  walltime 8 h"
echo "===================================================="
echo "  qsocat:        $QSOCAT"
echo "  loa_root:      $LOA_ROOT"
echo "  output:        $OUTPUT"
echo "  z range:       [$Z_MIN, $Z_MAX]"
echo "  spectype:      $SPECTYPE"
echo "  healpix_list:  ${HEALPIX_LIST:-<all>}"
echo "  max_spectra:   ${MAX_SPECTRA:-<all>}"
echo "  with_res:      $WITH_RES"
echo "===================================================="
echo

python -u preload_spectra/compress_loa_archive.py \
    --qso-catalog "$QSOCAT" \
    --loa-root "$LOA_ROOT" \
    --output "$OUTPUT" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --spectype "$SPECTYPE" \
    "${EXTRA_FLAGS[@]}"

[ -r "$OUTPUT" ] || { echo "[error] archive not produced: $OUTPUT" >&2; exit 7; }

SIZE=$(du -h "$OUTPUT" | cut -f1)
echo
echo "===================================================="
echo "  ARCHIVE COMPLETE"
echo "  output:        $OUTPUT  ($SIZE)"
echo
echo "  Quick verify (random TID round-trip):"
echo "  python -c \"from gpy_dla_detection.loa_archive import LoaArchive; "
echo "             ar=LoaArchive('$OUTPUT'); ar.open(); "
echo "             print(f'n_qsos={ar.n_qsos} n_pix={ar.wavelength.shape[0]} R={ar.has_resolution}'); "
echo "             ar.close()\""
echo "===================================================="
