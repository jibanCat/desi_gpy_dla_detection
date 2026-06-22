#!/usr/bin/env python3
"""Make reduced-N QMC sample grids by subsampling the existing 100k grids.

Why subsampling is faithful: the production grids are scrambled-Halton
low-discrepancy sequences (generate_samples.py: Halton(seed=42).random(N)).
The first N points of the 100k sequence ARE the N-point sequence that the
generator would produce at that N, so a prefix is a valid QMC grid at N.
The prior-normalization scalars (alpha, Z_dla, Z_lls, fit/uniform bounds,
extrapolate_min_log_nhi) are N-independent and copied verbatim. Any stored
``num_dla_samples`` scalar is overwritten to N so SubDLASamplesMAT's
``assert num_dla_samples == file`` passes.

Per-sample arrays are detected as those whose first axis == the source N.
Light I/O only (MB-scale, seconds) — safe on a login node.

Usage:
    python tools/make_subsampled_grids.py SRC.mat N [N ...]
"""
import sys
import h5py
import numpy as np


def subsample(src_path: str, n: int) -> str:
    with h5py.File(src_path, "r") as h:
        src_n = None
        data = {}
        for k in h.keys():
            data[k] = h[k][()]
        # infer source sample count from the largest first-axis
        src_n = max(v.shape[0] for v in data.values() if getattr(v, "ndim", 0) >= 1)
    if n > src_n:
        raise ValueError(f"requested N={n} exceeds source N={src_n}")

    out_path = src_path.replace(f"_{src_n}.mat", f"_{n}.mat")
    if out_path == src_path:
        raise ValueError(f"could not derive output name from {src_path} (expected _{src_n}.mat suffix)")

    with h5py.File(out_path, "w") as f:
        for k, v in data.items():
            arr = np.asarray(v)
            if arr.ndim >= 1 and arr.shape[0] == src_n:
                arr = arr[:n]                       # per-sample array → take first N
            if k == "num_dla_samples":
                arr = np.array([[float(n)]])        # keep assert happy
            f.create_dataset(k, data=arr)
    # verify
    with h5py.File(out_path, "r") as f:
        keys = sorted(f.keys())
        first = max(f[k].shape[0] for k in f.keys() if f[k].ndim >= 1)
        ndla = float(np.ravel(f["num_dla_samples"][()])[0]) if "num_dla_samples" in f else None
    print(f"  wrote {out_path}  (rows={first}, num_dla_samples={ndla}, keys={keys})")
    return out_path


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    src = sys.argv[1]
    for n in (int(x) for x in sys.argv[2:]):
        subsample(src, n)


if __name__ == "__main__":
    main()
