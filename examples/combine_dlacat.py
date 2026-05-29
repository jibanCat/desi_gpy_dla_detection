#!/usr/bin/env python
"""combine_dlacat.py — combine per-slice DLA catalogs into one shareable FITS.

Why this exists (vs the older ``combine_dlamocks.py``)
------------------------------------------------------
``combine_dlamocks.py`` walks a *rigid uniform grid* — ``for start in
range(initial, end+1, step)`` and reads ``dlacat-{release}-mockcat-{start}-
{start+step}.fits``. If a run's slices are NOT uniformly ``step``-spaced (e.g.
a resume produced 1-file slices, so the directory mixes ``...-12-14.fits`` and
``...-13-14.fits``), the rigid walk **silently skips** the off-grid files and
logs their positions as "missing" — so the combined catalog is quietly
incomplete. That is exactly the failure mode that made a 161-healpix gap in a
London resume run invisible (h5 present, dlacat absent, P/C completeness
crushed).

This combiner instead **globs every** ``dlacat-*-{mockcat,hpx}-<int>-<int>.fits``
in the directory — the same set ``examples/{molly,gp_native}_pc_plots.py``
``load_catalog_dir`` reads — so the combined file is row-for-row equivalent to
what the P/C eval consumes (combined ≡ eval input by construction). It then:

  * skips empty / truncated per-slice files gracefully (reporting them),
  * reports **position coverage** parsed from the filenames and warns loudly on
    gaps (the missing-slice lesson), with ``--fail-on-gap`` to make gaps fatal,
  * preserves ``EXTNAME=DLACAT`` and writes provenance header cards
    (n slices, n rows, unique TARGETIDs, coverage, source dir, UTC date) so a
    collaborator can audit the file's origin from its header alone.

It does NOT modify any row values — pure concatenation (astropy ``vstack``).

Usage
-----
    python examples/combine_dlacat.py \
        --procdir /path/to/run/outputs \
        --out     /path/to/run/combined_catalog/dlacat-v2.8.5-mockcat.fits

    # make a coverage gap fatal (recommended before sharing a "complete" cat):
    python examples/combine_dlacat.py --procdir ... --out ... \
        --expect-positions 1150 --fail-on-gap

The combined file is written OUTSIDE the per-slice dir by default intent — keep
``--out`` in a separate folder so ``load_catalog_dir`` (which globs
``dlacat-*.fits``) does not later double-count it alongside the per-slice files.
"""
import argparse
import datetime as _dt
import glob
import os
import re
import sys

import numpy as np
from astropy.table import Table, vstack

# Per-slice filename position range: dlacat-<release>-mockcat-<s>-<e>.fits  (mocks)
#                                    dlacat-<release>-<survey>-<program>-hpx-<s>-<e>.fits (real)
_RANGE_RE = re.compile(r"-(?:mockcat|hpx)-(\d+)-(\d+)\.fits$")


def per_slice_files(procdir, pattern):
    """All per-slice dlacat files (those with a -<int>-<int> position range).

    Excludes any already-combined file like ``dlacat-<release>-mockcat.fits``
    (no trailing range) so re-running over a dir that contains a previous
    combine output does not fold it back in.
    """
    files = sorted(glob.glob(os.path.join(procdir, pattern)))
    return [f for f in files if _RANGE_RE.search(os.path.basename(f))]


def parse_positions(files):
    """Union of integer positions covered by the slice filename ranges."""
    covered = set()
    for f in files:
        m = _RANGE_RE.search(os.path.basename(f))
        if m:
            covered.update(range(int(m.group(1)), int(m.group(2))))
    return covered


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procdir", required=True,
                    help="Directory holding the per-slice dlacat-*.fits files.")
    ap.add_argument("--out", required=True,
                    help="Output combined FITS path (keep it OUTSIDE --procdir).")
    ap.add_argument("--pattern", default="dlacat-*.fits",
                    help="Glob for per-slice files (default dlacat-*.fits; files "
                         "without a -<int>-<int> range, e.g. a prior combined "
                         "file, are excluded automatically).")
    ap.add_argument("--expect-positions", type=int, default=None,
                    help="Expected number of positions [0,N) for a gap check "
                         "(e.g. 1150 for a London/2LPT loa-124 mock).")
    ap.add_argument("--fail-on-gap", action="store_true",
                    help="Exit non-zero if --expect-positions reveals missing "
                         "positions (use before sharing a 'complete' catalog).")
    args = ap.parse_args()

    files = per_slice_files(args.procdir, args.pattern)
    if not files:
        sys.exit(f"[error] no per-slice dlacat files in {args.procdir} "
                 f"matching {args.pattern}")

    tables, skipped = [], []
    for f in files:
        try:
            tables.append(Table.read(f, hdu=1))
        except Exception as e:  # empty extension / truncated mid-write
            skipped.append((os.path.basename(f), str(e)))
    if not tables:
        sys.exit("[error] every per-slice file failed to read")

    combined = vstack(tables, metadata_conflicts="silent")

    # --- coverage report (parsed from filenames) ----------------------------
    covered = parse_positions(files)
    pos_lo, pos_hi = (min(covered), max(covered)) if covered else (0, 0)
    n_unique = int(np.unique(np.asarray(combined["TARGETID"])).size)
    missing = []
    if args.expect_positions is not None:
        missing = sorted(set(range(args.expect_positions)) - covered)

    print(f"[combine] {len(files)} per-slice files "
          f"({len(skipped)} skipped empty/bad)")
    print(f"[combine] {len(combined):,} rows, {n_unique:,} unique TARGETIDs")
    print(f"[combine] positions covered: {len(covered)} "
          f"(range {pos_lo}..{pos_hi})")
    if skipped:
        print("[combine] skipped: " + ", ".join(n for n, _ in skipped[:10])
              + (" ..." if len(skipped) > 10 else ""))
    if args.expect_positions is not None:
        if missing:
            print(f"[combine] ⚠ MISSING {len(missing)}/{args.expect_positions} "
                  f"positions: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        else:
            print(f"[combine] ✓ all {args.expect_positions} positions covered")

    # --- provenance header (auditable from the file alone) ------------------
    combined.meta["EXTNAME"] = "DLACAT"
    combined.meta["NSLICES"] = len(files)
    combined.meta["NSKIPPED"] = len(skipped)
    combined.meta["NROWS"] = len(combined)
    combined.meta["NUNQTID"] = n_unique
    combined.meta["NPOSCOV"] = len(covered)
    combined.meta["POSLO"] = pos_lo
    combined.meta["POSHI"] = pos_hi
    if args.expect_positions is not None:
        combined.meta["NPOSEXP"] = args.expect_positions
        combined.meta["NPOSMIS"] = len(missing)
    combined.meta["COMBDATE"] = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    combined.meta["COMBTOOL"] = "examples/combine_dlacat.py"
    # source dir as a COMMENT (path may exceed the 8-char keyword/68-char value)
    combined.meta["COMMENT"] = f"combined from {args.procdir}"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    combined.write(args.out, overwrite=True)
    print(f"[combine] wrote {args.out}")

    if missing and args.fail_on_gap:
        sys.exit(f"[error] {len(missing)} positions missing and --fail-on-gap set")


if __name__ == "__main__":
    main()
