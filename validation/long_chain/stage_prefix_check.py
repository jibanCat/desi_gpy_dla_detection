"""VALIDATION-ONLY: the predeclared internal consistency check (PREDECLARATION sec.2).

Warmup is held constant across stages, so a longer run of the same seed continues the SAME
RNG stream: the first N_short retained draws of each chain in the longer stage must be
bitwise identical to the shorter stage's draws.  Usage:

    stage_prefix_check.py <short_run.json> <long_run.json> <out.json>
"""
import json
import sys

import numpy as np


def main():
    short, long_, out = sys.argv[1:4]
    js, jl = json.load(open(short)), json.load(open(long_))
    a = np.asarray(np.load(short[:-5] + "_fdraws.npz")["f"])
    b = np.asarray(np.load(long_[:-5] + "_fdraws.npz")["f"])
    ms, ml = int(js["chains"]), int(jl["chains"])
    ns, nl = int(js["samples"]), int(jl["samples"])
    A = a.reshape(ms, ns, *a.shape[1:])
    B = b.reshape(ml, nl, *b.shape[1:])[:, :ns]
    ident = bool(A.shape == B.shape and np.array_equal(A, B))
    rec = {"short": short, "long": long_, "short_samples": ns, "long_samples": nl,
           "chains": ms, "prefix_bitwise_identical": ident,
           "max_abs_diff": float(np.max(np.abs(A - B))) if A.shape == B.shape else None}
    # the same check on the nuisance block
    za, zb = np.load(short[:-5] + "_bychain.npz"), np.load(long_[:-5] + "_bychain.npz")
    rec["sites"] = {}
    for k in ("t", "sigma_N", "sigma_z", "psi_c", "fp_shape_v", "eps_z", "eps_N",
              "theta_level", "theta_slope", "fp_lam_total", "potential_energy"):
        if k not in za.files:
            continue
        x, y = np.asarray(za[k]), np.asarray(zb[k])[:, :ns]
        rec["sites"][k] = bool(x.shape == y.shape and np.array_equal(x, y))
    rec["ALL_IDENTICAL"] = bool(ident and all(rec["sites"].values()))
    json.dump(rec, open(out, "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
