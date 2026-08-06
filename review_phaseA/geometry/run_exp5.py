"""REVIEW-ONLY (Phase A) — experiment 5: the RATIFIED 0.2-dex basis (2lpt0).

Repeat experiment 1's angle structure on coarsen_basis(p, 0.2,
pad_floor=19.0) — the analysis basis PI decision 3 ratified (0.1 dex is
plotting-only).  Pad drops from 5 x 15 = 75 columns to 2 x 15 = 30.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_pack                                   # noqa: E402
from run_exp1 import run_config, FP_LEVELS, T_A               # noqa: E402
from CDDF_analysis.hbi_mcmc.forward import build_consts       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "out_exp5.json")


def main():
    pack = get_pack("2lpt0", basis_width=0.2)
    consts = build_consts(pack)
    res = {"basis_width": 0.2, "T_A": T_A, "fp_levels": FP_LEVELS,
           "B": pack.n_b, "npad": pack.n_pad_bins}
    for tag, T_B in FP_LEVELS.items():
        t0 = time.time()
        o = run_config(pack, consts, T_B)
        res[tag] = o
        print(f"== 0.2dex 2lpt0 {tag} ({time.time()-t0:.1f}s) "
              f"n_informative={o['n_informative']}", flush=True)
        for k in ("pad_vs_FP", "pad_vs_window", "pad_vs_window+FP",
                  "pad_vs_ALL", "FP_vs_window+pad"):
            a = o["ang_" + k]
            print("   %-18s dim(%3d,%3d) min %8.4f deg  <0.1:%d <1:%d <5:%d"
                  " | q25/50/75 %.1f/%.1f/%.1f" % (
                      k, a["n_x"], a["n_y"], a["min_deg"], a["n_lt0p1"],
                      a["n_lt1"], a["n_lt5"], a["q25_deg"], a["q50_deg"],
                      a["q75_deg"]), flush=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("EXP5 DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
