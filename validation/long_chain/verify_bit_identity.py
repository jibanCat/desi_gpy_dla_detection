"""VALIDATION-ONLY: does the long-chain wrapper reproduce a PRODUCTION-config run bit-for-bit?

PI ruling 2026-09-12 sec.3: "any wrapper used only to retain additional diagnostics must first
reproduce a production-config run bit-for-bit".  Usage:
    verify_bit_identity.py <wrapper_fdraws.npz> <reference_fdraws.npz> <out.json>
"""
import hashlib
import json
import sys

import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    mine, ref, out = sys.argv[1:4]
    A, B = np.load(mine), np.load(ref)
    rec = {"wrapper_fdraws": mine, "wrapper_fdraws_sha256": sha(mine),
           "reference_fdraws": ref, "reference_fdraws_sha256": sha(ref), "keys": {}}
    ok = True
    for k in ("f", "ntrue_edges", "zf_edges"):
        a, b = np.asarray(A[k]), np.asarray(B[k])
        same_shape = a.shape == b.shape
        ident = bool(same_shape and np.array_equal(a, b))
        rec["keys"][k] = {"shape_wrapper": list(a.shape), "shape_reference": list(b.shape),
                          "np_array_equal": ident,
                          "max_abs_diff": float(np.max(np.abs(a - b))) if same_shape else None}
        ok = ok and ident
    rec["BIT_IDENTICAL"] = bool(ok)
    json.dump(rec, open(out, "w"), indent=1)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
