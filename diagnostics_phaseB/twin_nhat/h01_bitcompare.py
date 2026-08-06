#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-B bounded diagnosis (exploratory) — H1: stale/inconsistent artifacts.

PRE-STATED PREDICTED SIGNATURE (frozen spec section 7, item 1): if stale or
inconsistent artifact definitions are causal, the fresh Phase-B packs differ at
the bit level from the Phase-A-era window-study packs in one or more
CALIBRATION blocks (completeness, response, g, FP, dX, counts, truth), and the
residual changes when the artifacts are rebuilt.  EXPECTED under the null:
identical except the new schema field fp_eta_c (absent in the Phase-A pack).

Bit-compares every shared npz key of
  fresh : phaseB_packs/modelA_pack_<mock>_winlya_only_pad19p0_molly172_bw0p2.npz
  old   : window_study/packs/modelA_pack_<mock>_winlya_only_pad19p0_molly172_bw0p2.npz
for the twin (2lpt0) and, since they are on disk anyway, london0/saclay0.
"""
import json
import sys

import numpy as np

FRESH = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/"
         "modelA_pack_{m}_winlya_only_pad19p0_molly172_bw0p2.npz")
OLD = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/window_study/"
       "packs/modelA_pack_{m}_winlya_only_pad19p0_molly172_bw0p2.npz")

out = {"hypothesis": "H1 stale/inconsistent artifacts",
       "predicted_signature": "bit-level diff in calibration blocks; residual "
                              "changes with rebuilt artifacts",
       "label": "exploratory (prespecified discriminant, spec s7.1)",
       "mocks": {}}

for m in ("2lpt0", "london0", "saclay0"):
    fa = np.load(FRESH.format(m=m), allow_pickle=False)
    fb = np.load(OLD.format(m=m), allow_pickle=False)
    keys_fresh = set(fa.files)
    keys_old = set(fb.files)
    rec = {"keys_only_in_fresh": sorted(keys_fresh - keys_old),
           "keys_only_in_old": sorted(keys_old - keys_fresh),
           "diff_keys": [], "identical_keys": []}
    for k in sorted(keys_fresh & keys_old):
        a, b = fa[k], fb[k]
        same = (a.shape == b.shape and a.dtype == b.dtype
                and np.array_equal(a, b))
        if same:
            rec["identical_keys"].append(k)
        else:
            d = {"key": k, "shape_fresh": list(a.shape),
                 "shape_old": list(b.shape),
                 "dtype_fresh": str(a.dtype), "dtype_old": str(b.dtype)}
            if a.shape == b.shape and np.issubdtype(a.dtype, np.number):
                delta = np.abs(np.asarray(a, float) - np.asarray(b, float))
                d["max_abs_diff"] = float(delta.max())
                d["n_diff"] = int((delta > 0).sum())
            rec["diff_keys"].append(d)
    rec["n_shared"] = len(keys_fresh & keys_old)
    rec["bit_identical_on_shared"] = (len(rec["diff_keys"]) == 0)
    out["mocks"][m] = rec

out["verdict"] = ("shared blocks bit-identical for all three mocks; only "
                  "additions are the new schema fields"
                  if all(v["bit_identical_on_shared"]
                         for v in out["mocks"].values())
                  else "DIFFERENCES FOUND — see diff_keys")
print(json.dumps(out, indent=1))
with open(sys.path[0] + "/h01_bitcompare.json", "w") as fh:
    json.dump(out, fh, indent=1)
