#!/usr/bin/env python
"""Spec-weighted contiguous HPX-index boundaries for balanced LOA tasks (Option B).

Splits an HPX-index window ``[start, end)`` into ``ntasks`` CONTIGUOUS sub-ranges
of approximately equal cumulative spectra, using a per-healpix spec-count table.
Task ``k`` then processes ``[b_k, b_{k+1})``.

Why contiguous (not round-robin / greedy): each task keeps a single contiguous
``--hpx_start/--hpx_end`` range, so ``desi-DLAGP.py``'s healpix handling is
UNCHANGED -- only the *boundaries* between tasks move (a dense task gets fewer
healpix, a sparse task more), equalising work.

Correctness is independent of count accuracy: the boundaries always tile
``[start, end)`` exactly (b_0 = start, b_ntasks = end, monotone non-decreasing),
so every index is processed exactly once regardless of the counts. The counts
only affect *balance quality*. A stale/misaligned counts table therefore cannot
drop or double-process a healpix -- it can only make the split less even, which
degrades gracefully to the equal-index split.

Counts file: one line per HPX index in the SAME order as
``np.unique(catalog["HPXPIXEL"])`` (ascending healpix id over the z-masked
catalog -- the index space that ``--hpx_start/--hpx_end`` slice into). Format
``"<healpix_id> <count>"`` or just ``"<count>"``; the last whitespace field is
the count. Generate with ``tools/loa_hpx_spec_counts.py``.
"""
import argparse
import sys


def read_counts(path):
    counts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            counts.append(int(float(line.split()[-1])))  # last col = count
    return counts


def compute_boundaries(counts, start, end, ntasks):
    """Return ntasks+1 monotone indices b with b[0]=start, b[ntasks]=end."""
    if not (0 <= start <= end <= len(counts)):
        raise ValueError(f"window [{start},{end}) out of range for "
                         f"{len(counts)} counts")
    if ntasks < 1:
        raise ValueError("ntasks must be >= 1")

    window = counts[start:end]
    n = len(window)
    b = [start] * (ntasks + 1)
    b[ntasks] = end

    total = sum(window)
    # Degenerate cases -> deterministic equal-index split (still tiles exactly).
    if total == 0 or n <= ntasks:
        for k in range(1, ntasks):
            b[k] = start + (n * k) // ntasks
    else:
        prefix = [0] * (n + 1)            # prefix[i] = specs in window[:i]
        for i, c in enumerate(window):
            prefix[i + 1] = prefix[i] + c
        target = total / ntasks
        j = 0
        for k in range(1, ntasks):
            thresh = k * target
            while j < n and prefix[j] < thresh:
                j += 1
            b[k] = start + j

    # Enforce monotonicity and endpoints (defensive; clamps degenerate output).
    b[0] = start
    for k in range(1, ntasks + 1):
        if b[k] < b[k - 1]:
            b[k] = b[k - 1]
        if b[k] > end:
            b[k] = end
    b[ntasks] = end
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", required=True, help="per-HPX-index spec-count table")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--ntasks", type=int, required=True)
    ap.add_argument("--out", default=None,
                    help="write ntasks+1 boundary lines here; else stdout")
    ap.add_argument("--verify", action="store_true",
                    help="print balance stats + assert exact tiling to stderr")
    a = ap.parse_args()

    counts = read_counts(a.counts)
    b = compute_boundaries(counts, a.start, a.end, a.ntasks)

    # Hard invariants: exact tiling of [start, end), no gaps / overlaps.
    assert b[0] == a.start and b[-1] == a.end, f"endpoints {b[0]},{b[-1]}"
    for k in range(len(b) - 1):
        assert b[k] <= b[k + 1], f"non-monotone at {k}: {b}"

    if a.verify:
        loads = [sum(counts[b[k]:b[k + 1]]) for k in range(a.ntasks)]
        nz = [l for l in loads if l > 0]
        mean = sum(loads) / a.ntasks if a.ntasks else 0
        print(f"[verify] window [{a.start},{a.end}) ntasks={a.ntasks} "
              f"nonempty={len(nz)} total_specs={sum(loads)} "
              f"max={max(loads)} min_nonzero={min(nz) if nz else 0} "
              f"mean={mean:.0f} max/mean={(max(loads)/mean if mean else 0):.3f}",
              file=sys.stderr)

    text = "\n".join(str(x) for x in b) + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
