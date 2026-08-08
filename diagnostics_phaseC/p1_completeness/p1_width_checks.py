#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PI §2.3 width checks — is the natural-kernel width excess a
catalogued-overlap artifact?

FROZEN INTERPRETATION RULE (stated before this script's first run):
  * If the natural-vs-injected width excess DISAPPEARS (ratio ≤ ~1.05)
    in the isolated and shell-zero subsets, catalogued blends are a
    plausible major contributor and the kernel's width treatment must
    carry a blend-composition width term.
  * If the excess PERSISTS in isolated / shell-zero / no-catalogued-
    neighbor-30k subsets, the catalogued-shell classes are ruled out as
    a SUFFICIENT explanation. This does NOT exclude sub-threshold
    neighbors, unresolved multi-component structure, non-catalogued
    overlap, quickquasars imprint complexity, or truth-side profile
    variation — and we will NOT write that "all overlap is excluded".

Subsets (natural side, kernel-event population = the estimand's
numerator events, live support):
  all      — every kernel pair (the K parent population);
  iso5k    — no catalogued ≥17.2 truth neighbor within 5,000 km/s
             (the frozen Tier-2 isolation mirror);
  shell0   — iso5k AND no neighbor in the 5–10k km/s shell;
  nonb30k  — no catalogued neighbor within 30,000 km/s.
Injected reference: the frozen Tier-2 injected pairs (t2_pairing).
Bins: the frozen Tier-2 bins. Deterministic, catalog-level, no holdout.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "injection"))

from t2_pairing import truth_neighbors, injected_pairs, NB      # noqa: E402
from build_p1_natpair_ck import extract_kernel_events, C_KMS    # noqa: E402


def robust_sigma(v):
    if len(v) < 5:
        return np.nan
    q16, q84 = np.percentile(v, [15.865, 84.135])
    return float(0.5 * (q84 - q16))


def main():
    t0 = time.time()
    by = truth_neighbors()
    ev, _ = extract_kernel_events()
    kin = ev["IN_KERNEL"] & (ev["S2N"] > 2.0)
    N, Z, DX, TID = (ev["N"][kin], ev["Z"][kin], ev["DX"][kin],
                     ev["TID"][kin])

    d5 = np.zeros(len(N), bool)      # any neighbor < 5k
    dsh = np.zeros(len(N), bool)     # any neighbor in 5-10k shell
    d30 = np.zeros(len(N), bool)     # any neighbor < 30k
    for k in range(len(N)):
        for z, _n in by.get(int(TID[k]), []):
            dv = abs(z - Z[k]) / (1 + Z[k]) * C_KMS
            if dv < 1e-9:
                continue
            if dv < 5000.0:
                d5[k] = True
            elif dv < 10000.0:
                dsh[k] = True
            if dv < 30000.0:
                d30[k] = True
    subsets = {
        "all": np.ones(len(N), bool),
        "iso5k": ~d5,
        "shell0": (~d5) & (~dsh),
        "nonb30k": ~d30,
    }
    inj = injected_pairs(by)

    out = {"schema": "p1_width_checks/v1", "date": time.strftime("%Y-%m-%d"),
           "rows": []}
    print(f"kernel live pairs {len(N)}; injected {len(inj)}")
    for lo, hi in NB:
        vi = np.asarray([r["dx"] for r in inj if lo <= r["N"] < hi])
        wi = robust_sigma(vi)
        row = {"N": [lo, hi], "injected": {"n": int(len(vi)), "robust": wi}}
        msg = f"[{lo},{hi}) inj {wi:.3f} |"
        for name, m in subsets.items():
            v = DX[m & (N >= lo) & (N < hi)]
            w = robust_sigma(v)
            row[name] = {"n": int(len(v)), "robust": w,
                         "ratio_vs_injected": (w / wi if np.isfinite(w)
                                               and np.isfinite(wi) else None)}
            msg += f" {name} {w:.3f} (x{w/wi:.2f}, n={len(v)})"
        out["rows"].append(row)
        print(msg)
    out["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(_HERE, "p1_width_checks.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
