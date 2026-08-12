#!/bin/bash
#SBATCH --job-name=loa_hz_archive
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=loa_hz_archive_%j.log
#
# High-z loaArchive job (PI ruling 2026-08-12 §21–§25). ARCHIVE ONLY —
# no GP production, no candidate selection, no injection, no calibration.
#
# Builds ONE LoaArchive HDF5 covering (a) the never-searched
# z_qso in (4.25, 7.0] production population and (b, optional) the
# z_DLA >= 4.0 diagnostic-host sightlines, from the native CFS healpix
# coadds. The production finder does NOT consume this archive (it reads
# raw CFS FITS); the archive serves GL diagnostics/injection prep.
#
# Required environment (export before sbatch, or edit here):
#   GIT_SHA_EXPECTED   the approved forward SHA (job refuses to run otherwise)
#   MANIFEST_SHA_EXPECTED  optional: canonical target-list sha256 from the
#                          GL-side generation; if set, mismatch aborts.
#   DLACAT             optional: path to dlacat-loa-cddf-main-dark-v1.fits
#                      (adds diagnostic_host targets; omit → production set only)
# Submit: sbatch slurm/nersc/production/archive_loa_highz.sbatch
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/pscratch/sd/j/jibancat/desi_gpy_dla_detection}"
QSOCAT="${QSOCAT:-/global/cfs/cdirs/desi/science/lya/y3/loa/catalogs/QSO_cat_loa_main_dark_healpix_v2-altbal-20241115.fits}"
SOURCE_ROOT="${SOURCE_ROOT:-/global/cfs/cdirs/desi/spectro/redux/loa/healpix/main/dark}"
DEST_DIR="${DEST_DIR:-/pscratch/sd/j/jibancat/loa_hz_archive_v1}"
ARCHIVE_OUT="${DEST_DIR}/loa_hz_archive_v1.h5"
MANIFEST="${DEST_DIR}/hz_manifest.csv"

echo "== frozen Git-state verification (PI §18) =="
cd "$REPO_ROOT"
git remote -v
git rev-parse HEAD
git status --porcelain | head -5
[ -z "$(git status --porcelain)" ] || { echo "FATAL: dirty tree"; exit 1; }
[ -n "${GIT_SHA_EXPECTED:-}" ] || { echo "FATAL: GIT_SHA_EXPECTED unset"; exit 1; }
[ "$(git rev-parse HEAD)" = "$GIT_SHA_EXPECTED" ] || {
    echo "FATAL: HEAD != GIT_SHA_EXPECTED ($GIT_SHA_EXPECTED)"; exit 1; }

echo "== environment =="
${NERSC_ENV_SETUP:-source /global/cfs/cdirs/desi/software/desi_environment.sh main}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$DEST_DIR"

echo "== 1/3 manifest (deterministic regeneration at NERSC; PI §22) =="
python tools/hz_manifest.py \
    --qsocat "$QSOCAT" \
    $([ -n "${DLACAT:-}" ] && echo "--dlacat $DLACAT") \
    --source-root "$SOURCE_ROOT" --check-source \
    --dest "$ARCHIVE_OUT" --out "$MANIFEST"
GEN_SHA=$(python -c "import json;print(json.load(open('$MANIFEST.summary.json'))['target_list_sha256'])")
echo "manifest target_list_sha256 = $GEN_SHA"
if [ -n "${MANIFEST_SHA_EXPECTED:-}" ] && [ "$GEN_SHA" != "$MANIFEST_SHA_EXPECTED" ]; then
    echo "FATAL: manifest sha mismatch vs GL-side generation"; exit 1
fi
N_MISS=$(python -c "import json;print(json.load(open('$MANIFEST.summary.json'))['n_source_missing'])")
[ "$N_MISS" = "0" ] || { echo "FATAL: $N_MISS source coadds missing on CFS"; exit 1; }

echo "== 2/3 build archive =="
python tools/build_loa_archive_subset.py \
    --manifest "$MANIFEST" --source-root "$SOURCE_ROOT" --out "$ARCHIVE_OUT"

echo "== 3/3 verify (fail-loud; PI §24) =="
python tools/verify_loa_archive.py --manifest "$MANIFEST" --archive "$ARCHIVE_OUT"

sha256sum "$ARCHIVE_OUT" | tee "$ARCHIVE_OUT.sha256"
echo '{"job":"loa_hz_archive","status":"COMPLETE"}'
