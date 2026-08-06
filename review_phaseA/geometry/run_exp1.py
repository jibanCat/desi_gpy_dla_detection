"""REVIEW-ONLY (Phase A) — experiment 1: reproduce-then-separate.

At the archived probe's reference point (T_A = 24000 folded pad counts) and at
BOTH FP reference levels (folded FP total 1086.7 = the pre-repair amplitude,
and 14768 = the repaired loa-0 anchor level fp_w * n0_total), compute the
principal-angle structure between the pad, window, FP and nuisance column
spaces of the autodiff Jacobian of the production fold.

Pairs reported: pad<->FP, pad<->window, pad<->[window u FP],
pad<->[everything], FP<->[window u pad].  The point of the split: the
scientific A-vs-B question is pad<->FP; "16 of 75 within 1 degree" was quoted
off pad<->[window u FP u nuisances], which conflates it with pad<->window
(basis truncation).

Includes the one-off finite-difference validation of the autodiff Jacobian.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (MOCKS, get_pack, build_truth, theta_from_f,        # noqa: E402
                    all_jacobians, fd_check, whitened_blocks,
                    angle_summary, informative, live_mask)
from CDDF_analysis.hbi_mcmc.forward import build_consts                # noqa: E402

T_A = 24000.0
FP_LEVELS = {"TB_1086p7_prerepair": 1086.7, "TB_14768_anchor": 14768.0}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "out_exp1.json")


def run_config(pack, consts, T_B, *, do_fd=False):
    f, lam = build_truth(pack, consts, T_A, T_B)
    theta = theta_from_f(f)
    mu, J = all_jacobians(pack, consts, theta, lam)
    o = {}
    if do_fd:
        o["fd_max_rel_err"] = fd_check(pack, consts, theta, lam, J, mu)
    live = live_mask(pack, consts)
    o["mu_total_live"] = float(mu[live].sum())
    o["mu_fp_total_live"] = float((J["lam_raw"] @ np.asarray(lam).ravel())
                                  .reshape(mu.shape)[live].sum())
    D, w, lv = whitened_blocks(pack, consts, mu, J)
    keep = {k: informative(D[k]) for k in ("pad", "window", "loglam", "psi_c",
                                           "psi_k", "log_t")}
    P = D["pad"][:, keep["pad"]]
    W = D["window"][:, keep["window"]]
    F = D["loglam"][:, keep["loglam"]]
    C_ = D["psi_c"][:, keep["psi_c"]]
    K_ = D["psi_k"][:, keep["psi_k"]]
    T_ = D["log_t"][:, keep["log_t"]]
    o["n_informative"] = {k: int(v.sum()) for k, v in keep.items()}
    o["n_live"] = int(lv.sum())
    pairs = {
        "pad_vs_FP": (P, F),
        "pad_vs_window": (P, W),
        "pad_vs_window+FP": (P, np.hstack([W, F])),
        "pad_vs_ALL": (P, np.hstack([W, F, C_, K_, T_])),
        "FP_vs_window+pad": (F, np.hstack([W, P])),
        "FP_vs_ALL": (F, np.hstack([P, W, C_, K_, T_])),
    }
    for tag, (X, Y) in pairs.items():
        o["ang_" + tag] = angle_summary(X, Y)
    return o


def main():
    res = {"T_A": T_A, "fp_levels": FP_LEVELS}
    first = True
    for mock in MOCKS:
        pack = get_pack(mock)
        consts = build_consts(pack)
        res[mock] = {}
        for tag, T_B in FP_LEVELS.items():
            t0 = time.time()
            o = run_config(pack, consts, T_B, do_fd=first)
            first = False
            res[mock][tag] = o
            print(f"== {mock} {tag}  ({time.time()-t0:.1f}s)  "
                  f"n_informative={o['n_informative']}", flush=True)
            if "fd_max_rel_err" in o:
                print("   FD check max rel err:", o["fd_max_rel_err"],
                      flush=True)
            for k in ("pad_vs_FP", "pad_vs_window", "pad_vs_window+FP",
                      "pad_vs_ALL", "FP_vs_window+pad"):
                a = o["ang_" + k]
                print("   %-18s dim(%3d,%3d) min %8.4f deg  "
                      "<0.1:%d <1:%d <5:%d <10:%d | q25/50/75 %.1f/%.1f/%.1f"
                      % (k, a["n_x"], a["n_y"], a["min_deg"], a["n_lt0p1"],
                         a["n_lt1"], a["n_lt5"], a["n_lt10"], a["q25_deg"],
                         a["q50_deg"], a["q75_deg"]), flush=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("EXP1 DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
