#!/usr/bin/env bash
# package_catalog.sh — bundle a finished GP-DLA run into a shareable,
# reproducible absorber catalog. This is the standard POST-RUN routine for
# distributing a dlacat to collaborators; it codifies the steps that were done
# by hand for the 2LPT-0 V1 catalog (2026-05-27).
#
# Steps:
#   1. combine the per-slice dlacat-*.fits     (examples/combine_dlacat.py; globs
#      every slice incl. resume 1-file slices, gap-checked — NOT the old rigid
#      combine_dlamocks.py grid-walk that silently dropped off-grid slices)
#   2. add flag columns + fold DLAFLAG bits     (tools/postprocess/add_dla_flags.py:
#      LYBETA_FLAG/BAL_FLAG -> DLAFLAG bits 3/4, plus informational flag columns)
#   3. stamp provenance + git commit + EXTNAME into the FITS header
#   4. copy the run's BASELINE.env (resolved config + CODE_COMMIT) alongside
#   5. write README.md (column dictionary, flag semantics, recommended cut,
#      reproducibility)
#   6. optionally copy the bundle to a persistent share dir (--share-to, e.g. Turbo)
#
# The combined FITS is written OUTSIDE the per-slice dir so a later glob-based
# load (load_catalog_dir) cannot double-count it.
#
# Usage:
#   package_catalog.sh --rundir <RUN_DIR> --release v2.8.5 \
#       --bal-cat /path/bal_cat.fits [--expect-positions 1150] \
#       [--out <bundle dir>]            # default <rundir>/combined_catalog
#       [--share-to <persistent dir>]   # e.g. /nfs/turbo/.../gpdla_catalogs/<name>
#       [--lyb-veto-dz 0.005] [--source-label "free text for the FITS SRCRUN card"]
#
# <RUN_DIR> is the run root; per-slice dlacat-*.fits + BASELINE.env are expected
# under <RUN_DIR>/outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

# Steps below call `python` (fitsio/numpy) — ensure the DESI env is active so we don't
# fall back to system python2. desi_environment.sh is not set -u clean, so relax -u.
set +u; source /global/cfs/cdirs/desi/software/desi_environment.sh main >/dev/null 2>&1 || true; set -u

RUNDIR="" RELEASE="" BAL_CAT="" EXPECT="" OUTDIR="" SHARE_TO="" LYBDZ="0.005" SRCLABEL=""
PURITY="" COMPL=""
# --data-kind selects the FITS naming + README text. Default mock for back-compat
# (byte-identical to the historical packaging). real names by SURVEY/PROGRAM, which
# must be passed explicitly (no defaults) so a real run on a non-main/dark program
# can't be silently mislabeled dlacat-<release>-main-dark.fits.
DATA_KIND="mock" SURVEY="" PROGRAM=""
while [ $# -gt 0 ]; do
    case "$1" in
        --rundir)           RUNDIR="$2"; shift 2;;
        --release)          RELEASE="$2"; shift 2;;
        --bal-cat)          BAL_CAT="$2"; shift 2;;
        --expect-positions) EXPECT="$2"; shift 2;;
        --out)              OUTDIR="$2"; shift 2;;
        --share-to)         SHARE_TO="$2"; shift 2;;
        --lyb-veto-dz)      LYBDZ="$2"; shift 2;;
        --source-label)     SRCLABEL="$2"; shift 2;;
        --purity)           PURITY="$2"; shift 2;;   # optional, for the README P/C table
        --completeness)     COMPL="$2"; shift 2;;
        --data-kind)        DATA_KIND="$2"; shift 2;; # mock (default) | real
        --survey)           SURVEY="$2"; shift 2;;    # real only: e.g. main
        --program)          PROGRAM="$2"; shift 2;;   # real only: e.g. dark
        --code-commit)      CODE_COMMIT_OVERRIDE="$2"; shift 2;;  # backfill old runs
        *) echo "[package] unknown arg: $1" >&2; exit 2;;
    esac
done
CODE_COMMIT_OVERRIDE="${CODE_COMMIT_OVERRIDE:-}"
[ -n "$RUNDIR" ] && [ -n "$RELEASE" ] && [ -n "$BAL_CAT" ] || {
    echo "[package] --rundir, --release and --bal-cat are required" >&2; exit 2; }
case "$DATA_KIND" in
    mock|real) ;;
    *) echo "[package] --data-kind must be mock or real (got: $DATA_KIND)" >&2; exit 2;;
esac
# real-data bundles are named by survey/program; require them explicitly so a
# non-main/dark real run can't be mislabeled. mock ignores survey/program.
if [ "$DATA_KIND" = "real" ] && { [ -z "$SURVEY" ] || [ -z "$PROGRAM" ]; }; then
    echo "[package] --data-kind real requires --survey and --program" >&2; exit 2
fi

PROCDIR="$RUNDIR/outputs"
[ -d "$PROCDIR" ] || { echo "[package] no $PROCDIR" >&2; exit 2; }
OUTDIR="${OUTDIR:-$RUNDIR/combined_catalog}"
# Naming: mock keeps the historical dlacat-<release>-mockcat.fits; real names the
# bundle by SURVEY/PROGRAM (no truth, so "mockcat" would be misleading).
if [ "$DATA_KIND" = "real" ]; then
    FITS="$OUTDIR/dlacat-${RELEASE}-${SURVEY}-${PROGRAM}.fits"
else
    FITS="$OUTDIR/dlacat-${RELEASE}-mockcat.fits"
fi
FITS_NAME="$(basename "$FITS")"
mkdir -p "$OUTDIR"
[ -n "$SRCLABEL" ] || SRCLABEL="$(basename "$RUNDIR")"

# Code commit, in priority order: explicit --code-commit (backfilling old runs)
# > the run's recorded CODE_COMMIT in BASELINE.env (what actually ran)
# > current repo HEAD (last-resort fallback; only right if HEAD == what ran).
RUN_COMMIT="$CODE_COMMIT_OVERRIDE"
[ -n "$RUN_COMMIT" ] || RUN_COMMIT="$(grep -E '^CODE_COMMIT=' "$PROCDIR/BASELINE.env" 2>/dev/null | cut -d= -f2 || true)"
[ -n "$RUN_COMMIT" ] || RUN_COMMIT="$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"

echo "[package] 1/5 combine per-slice dlacat -> $FITS"
python "$REPO/examples/combine_dlacat.py" --procdir "$PROCDIR" --out "$FITS" \
    ${EXPECT:+--expect-positions "$EXPECT" --fail-on-gap}

echo "[package] 2/5 add flag columns (lyb dz=$LYBDZ, BAL from $BAL_CAT)"
python "$REPO/tools/postprocess/add_dla_flags.py" --catalog-dir "$OUTDIR" \
    --bal-cat "$BAL_CAT" --lyb-veto-dz "$LYBDZ" --no-bf-band

echo "[package] 3/5 stamp provenance + commit into FITS header"
python - "$FITS" "$RUN_COMMIT" "$SRCLABEL" "$LYBDZ" <<'PY'
import sys, datetime, numpy as np, fitsio
fits, commit, srclabel, lybdz = sys.argv[1:5]
d = fitsio.read(fits, ext=1)
hdr = fitsio.read_header(fits, ext=1)   # preserve combine_dlacat's coverage cards (NSLICES/NPOSCOV/...)
# Clip probabilities to [0,1] — raw inference can overshoot 1 by ~1e-13 (float round-off).
for col in ('P_DLA', 'P_NULL'):
    if col in d.dtype.names:
        d[col] = np.clip(d[col], 0.0, 1.0)
with fitsio.FITS(fits, 'rw', clobber=True) as h:
    h.write(d, header=hdr, extname='DLACAT')
    hdu = h[1]
    hdu.write_key('NROWS', len(d))
    hdu.write_key('NUNQTID', int(np.unique(d['TARGETID']).size))
    hdu.write_key('CODECMT', commit, comment='git commit that ran the inference')
    hdu.write_key('COMBTOOL', 'examples/combine_dlacat.py')
    hdu.write_key('FLAGTOOL', 'tools/postprocess/add_dla_flags.py')
    hdu.write_key('LYBDZ', float(lybdz), comment='lyb-veto dz_match')
    hdu.write_key('SRCRUN', srclabel)
    hdu.write_key('PDLACLIP', 'T', comment='P_DLA/P_NULL clipped to [0,1]')
    hdu.write_key('PKGDATE', datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'))
print(f"  rows={len(d)} commit={commit} (P_DLA/P_NULL clipped to [0,1])")
PY

echo "[package] 4/5 copy BASELINE.env"
if [ -f "$PROCDIR/BASELINE.env" ]; then
    cp "$PROCDIR/BASELINE.env" "$OUTDIR/BASELINE.env"
else
    echo "[package]   (warning: no BASELINE.env in $PROCDIR — reproducibility config missing)" >&2
fi

echo "[package] 5/5 write README.md"
# For real data, forward --data-kind real + the real FITS basename so the README
# uses the real filename and real-data (no-truth) caveats. Mock omits both to keep
# the historical invocation (and README output) byte-identical.
README_KIND_ARGS=()
if [ "$DATA_KIND" = "real" ]; then
    README_KIND_ARGS=(--data-kind real --fits-name "$FITS_NAME")
fi
python "$SCRIPT_DIR/_write_catalog_readme.py" \
    --fits "$FITS" --release "$RELEASE" --source-label "$SRCLABEL" \
    --commit "$RUN_COMMIT" --lyb-veto-dz "$LYBDZ" --out "$OUTDIR/README.md" \
    "${README_KIND_ARGS[@]}" \
    ${PURITY:+--purity "$PURITY"} ${COMPL:+--compl "$COMPL"}

if [ -n "$SHARE_TO" ]; then
    echo "[package] copying bundle -> $SHARE_TO"
    mkdir -p "$SHARE_TO"
    cp -p "$FITS" "$OUTDIR/README.md" "$SHARE_TO/"
    [ -f "$OUTDIR/BASELINE.env" ] && cp -p "$OUTDIR/BASELINE.env" "$SHARE_TO/"
fi

echo "[package] DONE. Bundle in $OUTDIR"
ls -la "$OUTDIR"
