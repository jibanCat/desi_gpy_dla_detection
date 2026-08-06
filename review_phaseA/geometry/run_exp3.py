"""REVIEW-ONLY (Phase A) — experiment 3: reference-point stability (2lpt0).

Grid: (T_A, T_B) x pad_slope x fp_shape; for every variant, pad<->FP and
pad<->[window u FP] angle structure, under BOTH FP column selections
(all floored columns = probe convention, and data-supported only).
"""
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (get_pack, build_truth, theta_from_f, all_jacobians,  # noqa: E402
                    whitened_blocks, angle_summary, informative)
from CDDF_analysis.hbi_mcmc.forward import build_consts                  # noqa: E402

TATB = [(24000.0, 1086.7), (24000.0, 14768.0), (5000.0, 14768.0),
        (24000.0, 5000.0)]
SLOPES = [("fitted", None), ("m1p5", -1.5), ("flat0", 0.0)]
SHAPES = ["n0", "flat"]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "out_exp3.json")


def main():
    pack = get_pack("2lpt0")
    consts = build_consts(pack)
    n0 = np.asarray(pack.fp_counts, float).ravel()
    res = {"grid": {"TATB": TATB, "slopes": [s for s, _ in SLOPES],
                    "shapes": SHAPES}, "configs": []}
    mins = {"pad_FP_all": [], "pad_FP_datasup": [], "pad_WF_all": [],
            "pad_WF_datasup": []}
    for (ta, tb), (stag, sval), shape in itertools.product(
            TATB, SLOPES, SHAPES):
        t0 = time.time()
        f, lam = build_truth(pack, consts, ta, tb, pad_slope=sval,
                             fp_shape=shape)
        theta = theta_from_f(f)
        mu, J = all_jacobians(pack, consts, theta, lam,
                              blocks=("theta", "lam_raw"))
        D, w, lv = whitened_blocks(pack, consts, mu, J)
        keep_fp = informative(D["loglam"])
        keep_fp_ds = keep_fp & (n0 > 0)
        P = D["pad"][:, informative(D["pad"])]
        W = D["window"][:, informative(D["window"])]
        o = {"T_A": ta, "T_B": tb, "pad_slope": stag, "fp_shape": shape}
        for sel, keep in [("all", keep_fp), ("datasup", keep_fp_ds)]:
            F = D["loglam"][:, keep]
            a1 = angle_summary(P, F)
            a2 = angle_summary(P, np.hstack([W, F]))
            o[f"ang_pad_vs_FP_{sel}"] = a1
            o[f"ang_pad_vs_window+FP_{sel}"] = a2
            mins[f"pad_FP_{sel}"].append(a1["min_deg"])
            mins[f"pad_WF_{sel}"].append(a2["min_deg"])
        res["configs"].append(o)
        print("TA %6.0f TB %7.1f slope %-6s shape %-4s | padFP all %7.4f "
              "ds %7.4f | pad(W+F) all %7.4f ds %7.4f  (%.1fs)" % (
                  ta, tb, stag, shape,
                  o["ang_pad_vs_FP_all"]["min_deg"],
                  o["ang_pad_vs_FP_datasup"]["min_deg"],
                  o["ang_pad_vs_window+FP_all"]["min_deg"],
                  o["ang_pad_vs_window+FP_datasup"]["min_deg"],
                  time.time() - t0), flush=True)
    res["min_angle_ranges"] = {
        k: {"min": float(np.min(v)), "q25": float(np.percentile(v, 25)),
            "q50": float(np.percentile(v, 50)),
            "q75": float(np.percentile(v, 75)), "max": float(np.max(v))}
        for k, v in mins.items()}
    print("ranges:", json.dumps(res["min_angle_ranges"], indent=1),
          flush=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("EXP3 DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
