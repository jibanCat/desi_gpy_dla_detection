"""REVIEW-ONLY (Phase A) — experiment 2: column-selection artifacts (2lpt0).

Question: are the smallest angles (the 0.0176 deg pad<->[window u FP]
direction and the ~0 deg pad<->ALL direction) artifacts of columns without
data support?

Variants:
  all      : every nonzero-norm column (the archived probe's choice; the FP
             block owes 149 of its 174 columns to the lam floor placed in
             cells with ZERO loa-0 counts, and psi_c includes molly cells
             whose only data footprint is the pad itself).
  datasup  : FP columns only where fp_counts > 0 (25 cells); psi_c columns
             only where the column would be nonzero WITHOUT the pad (i.e. the
             molly cell has a window-truth footprint in the survey data).
  fp_only / psic_only : the two restrictions applied separately.

Also attributes the smallest pad<->Y principal directions: which Y-blocks the
collapsing direction needs (incremental-span attribution + least-squares
block energy).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (get_pack, build_truth, theta_from_f, all_jacobians,  # noqa: E402
                    whitened_blocks, angle_summary, informative,
                    principal_angles, orthonormal_basis, live_mask,
                    THETA_DEAD)
from CDDF_analysis.hbi_mcmc.forward import build_consts                  # noqa: E402

T_A, T_B = 24000.0, 1086.7
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "out_exp2.json")


def psi_c_window_support(pack, consts, f, lam):
    """Norm of each psi_c column when the pad rows of f are REMOVED."""
    f_win = f.copy()
    f_win[:pack.n_pad_bins] = 0.0
    theta_win = theta_from_f(f_win)
    _, Jw = all_jacobians(pack, consts, theta_win, lam,
                          blocks=("psi_c",))
    lv = live_mask(pack, consts).ravel()
    return np.linalg.norm(Jw["psi_c"][lv], axis=0)


def ls_block_energy(x, blocks):
    """Least-squares attribution of x over concatenated blocks (energy per
    block of Y_b @ a_b; NOT an orthogonal decomposition, diagnostic only)."""
    names = [n for n, _ in blocks]
    Y = np.hstack([b for _, b in blocks])
    a, *_ = np.linalg.lstsq(Y, x, rcond=None)
    out, i = {}, 0
    for n, b in blocks:
        out[n] = float(np.linalg.norm(b @ a[i:i + b.shape[1]]) ** 2)
        i += b.shape[1]
    resid = float(np.linalg.norm(Y @ a - x) ** 2)
    tot = sum(out.values())
    return {"block_energy": {n: out[n] / max(tot, 1e-300) for n in names},
            "residual_over_x": resid / max(float(x @ x), 1e-300)}


def main():
    pack = get_pack("2lpt0")
    consts = build_consts(pack)
    f, lam = build_truth(pack, consts, T_A, T_B)          # probe-style floor
    theta = theta_from_f(f)
    mu, J = all_jacobians(pack, consts, theta, lam)
    D, w, lv = whitened_blocks(pack, consts, mu, J)

    keep_all = {k: informative(D[k]) for k in
                ("pad", "window", "loglam", "psi_c", "psi_k", "log_t")}
    n0 = np.asarray(pack.fp_counts, float).ravel()
    fp_datasup = keep_all["loglam"] & (n0 > 0)
    psic_win = psi_c_window_support(pack, consts, f, lam)
    psic_datasup = keep_all["psi_c"] & (psic_win >
                                        1e-10 * max(psic_win.max(), 1e-300))

    res = {"T_A": T_A, "T_B": T_B,
           "n_fp_all": int(keep_all["loglam"].sum()),
           "n_fp_datasup": int(fp_datasup.sum()),
           "n_psic_all": int(keep_all["psi_c"].sum()),
           "n_psic_datasup": int(psic_datasup.sum()),
           "molly_nhi_edges": np.asarray(pack.molly_nhi_edges, float).tolist(),
           "psic_dropped_cells_sm": [
               [int(i // consts.n_molly), int(i % consts.n_molly)]
               for i in np.flatnonzero(keep_all["psi_c"] & ~psic_datasup)]}

    P = D["pad"][:, keep_all["pad"]]
    W = D["window"][:, keep_all["window"]]
    K_ = D["psi_k"][:, keep_all["psi_k"]]
    T_ = D["log_t"][:, keep_all["log_t"]]

    variants = {
        "all":       (keep_all["loglam"], keep_all["psi_c"]),
        "datasup":   (fp_datasup, psic_datasup),
        "fp_only":   (fp_datasup, keep_all["psi_c"]),
        "psic_only": (keep_all["loglam"], psic_datasup),
    }
    for tag, (fp_keep, pc_keep) in variants.items():
        F = D["loglam"][:, fp_keep]
        C_ = D["psi_c"][:, pc_keep]
        o = {"n_fp": int(fp_keep.sum()), "n_psic": int(pc_keep.sum())}
        for name, X, Y in [("pad_vs_FP", P, F),
                           ("pad_vs_window+FP", P, np.hstack([W, F])),
                           ("pad_vs_ALL", P, np.hstack([W, F, C_, K_, T_])),
                           ("FP_vs_ALL", F, np.hstack([P, W, C_, K_, T_]))]:
            o["ang_" + name] = angle_summary(X, Y)
        res["variant_" + tag] = o
        print("== variant %-9s FP %3d psi_c %2d | padFP %.4f | pad(W+F) %.4f"
              " | padALL %.3e | FPALL %.4f deg" % (
                  tag, o["n_fp"], o["n_psic"],
                  o["ang_pad_vs_FP"]["min_deg"],
                  o["ang_pad_vs_window+FP"]["min_deg"],
                  o["ang_pad_vs_ALL"]["min_deg"],
                  o["ang_FP_vs_ALL"]["min_deg"]), flush=True)

    # ---- attribution of the collapsing directions (all-columns variant) ----
    F = D["loglam"][:, keep_all["loglam"]]
    C_ = D["psi_c"][:, keep_all["psi_c"]]
    Qp = orthonormal_basis(P)
    attr = {}
    for tag, Yblocks in [
            ("window", [("window", W)]),
            ("window+FP", [("window", W), ("FP", F)]),
            ("ALL", [("window", W), ("FP", F), ("psi_c", C_),
                     ("psi_k", K_), ("log_t", T_)])]:
        Y = np.hstack([b for _, b in Yblocks])
        Qy = orthonormal_basis(Y)
        u, s, vt = np.linalg.svd(Qp.T @ Qy)
        x1 = Qp @ u[:, 0]          # the pad-side direction closest to span(Y)
        a = {"min_deg": float(np.degrees(np.arccos(min(s[0], 1.0))))}
        a.update(ls_block_energy(x1, Yblocks))
        attr["closest_to_" + tag] = a
        print("   closest pad dir to %-9s: %.4g deg, LS block energy %s"
              % (tag, a["min_deg"],
                 {k: round(v, 3) for k, v in a["block_energy"].items()}),
              flush=True)
    res["attribution"] = attr

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("EXP2 DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
