#!/usr/bin/env python
"""broaden_kernel.py — broaden the cached posterior kernel along the logN axis.

The cached per-object kernel `kappa` (n_op, 52, 15) is correctly CENTERED but
TOO NARROW in logN (PIT cov68=0.46, multi-DLA logN-std ~0.046 dex vs a realistic
0.1-0.2 dex). This convolves each op-row along the logN axis (axis=1, 0.1 dex
bins) with a Gaussian of width sigma_dex, leaves the z axis (axis=2) untouched,
and renormalizes each op-row to sum=1 over (logN, z). All-zero rows (no support)
stay zero.

Usage:
    python broaden_kernel.py SIGMA_DEX IN_NPZ OUT_NPZ
"""
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d

DLOGN = 0.1  # logN bin width (dex)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    sigma_dex = float(sys.argv[1])
    in_npz = sys.argv[2]
    out_npz = sys.argv[3]

    sigma_bins = sigma_dex / DLOGN
    print(f"[broaden] sigma_dex={sigma_dex} -> sigma_bins={sigma_bins:.4f} "
          f"(logN axis=1, dlogN={DLOGN})")

    # allow_pickle: dlaid_op is an object (string) array in the cached kernel.
    d = np.load(in_npz, allow_pickle=True)
    kappa = d["kappa"].astype(np.float64)  # (n_op, 52, 15)
    n_op, n_logN, n_z = kappa.shape
    print(f"[broaden] loaded kappa shape={kappa.shape} dtype(in)={d['kappa'].dtype}")

    # row support BEFORE broadening (so all-zero rows stay zero)
    row_sum_in = kappa.reshape(n_op, -1).sum(axis=1)
    has_support = row_sum_in > 0

    # convolve along logN axis only; z axis untouched. mode='constant' (zero pad)
    # means probability can leak off the logN grid edges; we renormalize after.
    broad = gaussian_filter1d(kappa, sigma=sigma_bins, axis=1, mode="constant")

    # renormalize each op-row to sum=1 over (logN, z); zero-support rows stay zero
    row_sum_out = broad.reshape(n_op, -1).sum(axis=1)
    safe = row_sum_out.copy()
    safe[~has_support] = 1.0          # avoid div-by-zero; numerator is ~0 anyway
    safe[safe == 0] = 1.0
    broad = broad / safe[:, None, None]
    broad[~has_support] = 0.0          # force no-support rows exactly zero

    broad = broad.astype(np.float32)

    # diagnostics: per-row logN-std (marginalize z, weight by logN bin centers)
    logN_centers = np.arange(n_logN, dtype=np.float64) * DLOGN  # relative grid (dex)

    def row_logN_std(K):
        marg = K.sum(axis=2)                       # (n_op, n_logN)
        s = marg.sum(axis=1)
        good = s > 0
        m = marg[good] / s[good, None]
        mean = (m * logN_centers).sum(axis=1)
        var = (m * (logN_centers - mean[:, None]) ** 2).sum(axis=1)
        return np.sqrt(np.clip(var, 0, None))

    std_in = row_logN_std(d["kappa"].astype(np.float64))
    std_out = row_logN_std(broad.astype(np.float64))
    print(f"[broaden] n_op={n_op}  with_support={has_support.sum()}")
    print(f"[broaden] median logN-std  in={np.median(std_in):.4f} dex  "
          f"out={np.median(std_out):.4f} dex  (delta={np.median(std_out)-np.median(std_in):+.4f})")
    print(f"[broaden] mean   logN-std  in={np.mean(std_in):.4f} dex  "
          f"out={np.mean(std_out):.4f} dex")

    # row-sum sanity on supported rows
    rs = broad.reshape(n_op, -1).sum(axis=1)[has_support]
    print(f"[broaden] post-norm supported row-sums: "
          f"min={rs.min():.6f} max={rs.max():.6f} median={np.median(rs):.6f}")

    # Pass through EVERY original key unchanged except kappa. The cached kernel
    # carries metadata (n_Nbins, n_zf, logN_lo/hi, z_edges_fine, n_no_support,
    # n_unmatched, norm) in addition to slot_op/tid_op/dlaid_op/ess_*; the stage
    # 2/3 reader needs the grid metadata, so preserve all of it.
    out = {"kappa": broad}
    for k in d.files:
        if k == "kappa":
            continue
        out[k] = d[k]
    np.savez(out_npz, **out)
    print(f"[broaden] wrote {out_npz}  keys={sorted(out.keys())}")


if __name__ == "__main__":
    main()
