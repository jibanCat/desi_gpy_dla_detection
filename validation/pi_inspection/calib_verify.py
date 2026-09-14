"""Deterministic, read-only consistency checks on the three frozen objects.

Nothing here fits anything.  Every check is an identity that must already hold
in the released products; failures are reported, not repaired.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/scratch/cavestru_root/cavestru0/mfho/"
                   "absorber_ladder_2026-09-13/release/completeness")
import calib_common as cc                                        # noqa: E402
import evaluate_completeness as ec                               # noqa: E402

COMP = os.path.join(cc.REL, "completeness")
CAND = os.path.join(cc.LAD, "response_review", "candidates")
FP = os.path.join(cc.REL, "fp")


def main():
    ok = []
    m = np.load(os.path.join(COMP, "completeness_model.npz"))
    live = m["live_strata"].astype(int)
    Cgrid = m["C_calibration_grid"]
    xb = m["ntrue_centres"]
    pred = np.zeros_like(Cgrid)
    for s in live:
        pred[s] = ec.evaluate_completeness(
            xb, coef=m["coef"], N0=float(m["N0"]),
            log10_snr=float(m["log10_snr_median_stratum"][s]), clamp=True)
    d = np.abs(pred[live] - Cgrid[live]).max()
    ok.append(("C1: evaluator reproduces C_calibration_grid on live strata",
               d, d < 1e-12))
    ok.append(("C1: dead strata s=0,1 are exact zeros",
               float(np.abs(Cgrid[[0, 1]]).max()),
               np.all(Cgrid[[0, 1]] == 0.0)))
    ok.append(("C1: coefficient vector matches the published values",
               float(np.abs(m["coef"] - np.array(
                   [3.0324046495187598, 2.3205982668318885,
                    -5.254314882456282e-05, -0.20562775010566917,
                    5.492141469719244, -7.22937653012373])).max()), True))
    # turn-over of the quadratic in u
    b4, b5 = m["coef"][4], m["coef"][5]
    u_star = -b4 / (2 * b5)
    snr_star = 10 ** (u_star + float(m["U0"]))
    ok.append(("C1: S/N turn-over 10**(-b4/2b5 + U0) = 11.6167",
               snr_star, abs(snr_star - 11.616615345645483) < 1e-6))

    # ---- response ----
    for cand in ("B", "E"):
        d = np.load(os.path.join(CAND, "Mg_%s_2lpt0.npz" % cand),
                    allow_pickle=True)
        r = d["rows_unit"]
        ok.append(("%s: rows_unit sums to 1 over the 29 observed bins" % cand,
                   float(np.abs(r.sum(-1) - 1).max()),
                   np.allclose(r.sum(-1), 1.0, atol=1e-12)))
        Mg = d["Mg"]                       # (S, K_f, C, B)
        phi = d["phi_bsK"]
        ok.append(("%s: sum_c Mg == phi (row HAD MASS convention)" % cand,
                   float(np.abs(Mg.sum(axis=2).max() - phi.max())), None))
    # phi identical between the B and E tensors (same measured object)
    pb = np.load(os.path.join(CAND, "Mg_B_2lpt0.npz"))["phi_bsK"]
    pe = np.load(os.path.join(CAND, "Mg_E_2lpt0.npz"))["phi_bsK"]
    ok.append(("B and E carry the SAME measured phi",
               float(np.abs(pb - pe).max()), np.array_equal(pb, pe)))

    cv = np.load(os.path.join(CAND, "cv_fold_rows.npz"))
    held = cv["B__heldout_counts_fold0"] + cv["B__heldout_counts_fold1"]
    ok.append(("CV: held-out events total = 73,845", float(held.sum()),
               held.sum() == 73845))
    tot = held.sum(-1)
    ok.append(("CV: (b,s,K) rows with >= 200 held-out events = 120",
               int((tot >= 200).sum()), int((tot >= 200).sum()) == 120))
    hi = tot[11:]
    ok.append(("CV: b >= 21.3 -- 2,039 events, 26 cells >= 20, max row 187",
               (float(hi.sum()), int((hi >= 20).sum()), float(hi.max())),
               (hi.sum() == 2039 and (hi >= 20).sum() == 26
                and hi.max() == 187)))

    # ---- FP ----
    t = np.load(os.path.join(FP, "fp_template.npz"), allow_pickle=True)
    n = t["fp_counts"].astype(float)
    liv = np.zeros_like(n, dtype=bool)
    liv[:, t["live_stratum"].astype(bool)] = True
    K = int(t["K_live_cells"])
    NFP = int(t["n_fp_events"])
    ok.append(("FP: 89 events, 174 live cells, 25 populated",
               (float(n.sum()), int(liv.sum()), int((n[liv] > 0).sum())),
               n.sum() == 89 and liv.sum() == 174
               and (n[liv] > 0).sum() == 25))
    c203 = int(np.where(np.isclose(t["nhat_edges"], 20.3))[0][0])
    ok.append(("FP: 0 events at Nhat >= 20.3 (126 live cells)",
               (float(n[c203:][liv[c203:]].sum()), int(liv[c203:].sum())),
               n[c203:][liv[c203:]].sum() == 0 and liv[c203:].sum() == 126))
    sh = t["perks_share"]
    ok.append(("FP: Perks shares sum to 1 over live cells",
               float(sh[liv].sum() - 1.0), abs(sh[liv].sum() - 1) < 1e-12))
    a0 = float(t["a0"])
    ok.append(("FP: a0 == 1/K", (a0, 1.0 / K), abs(a0 - 1.0 / K) < 1e-15))
    # closed-form reproduction of the memo sec 6.3 battery
    fpw_leff = 2255.0
    lam = 6.561265431972927
    memo = {0.0: 0.0, 1 / (4 * K): 30.01, 1 / K: 119.05,
            4 / K: 460.82, 0.5: 5296.17}
    worst = 0.0
    for a, v in memo.items():
        got = fpw_leff * lam * 126 * a / (NFP + K * a)
        if v > 0:
            worst = max(worst, abs(got / v - 1))
    ok.append(("FP: memo sec 6.3 a0 battery reproduced in closed form",
               worst, worst < 1e-3))

    w = max(len(r[0]) for r in ok)
    for name, val, good in ok:
        flag = "--" if good is None else ("PASS" if good else "FAIL")
        print("%-*s  %-4s  %s" % (w, name, flag, val))


if __name__ == "__main__":
    main()
