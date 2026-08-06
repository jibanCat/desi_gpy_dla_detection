#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-B bounded diagnosis (exploratory) — H2: N-hat bin-edge/normalization
mismatch.

PRE-STATED PREDICTED SIGNATURE (spec s7.2), stated BEFORE the test: a true
edge/offset mismatch (observed N-hat histogram binned against shifted edges,
or a mis-normalized bin width) produces (i) ALTERNATING per-bin residual signs
(sawtooth) whose phase tracks the OBSERVED 0.1-dex binning, and (ii) a strong
parity/sign change of the 3-group residual when the aggregation edges are
shifted by one bin.  A smooth shape misfit produces neither: group residuals
move smoothly and keep their sign pattern under the shift.

DISAMBIGUATION RECORDED UP FRONT: the latent basis is 0.2 dex while the
observed grid is 0.1 dex, so a DIFFERENT sawtooth is possible — one whose
period is TWO observed bins and whose phase is locked to the 0.2-dex BASIS
edges (piecewise-constant f on a wide basis folded through a narrow kernel).
That is basis discretization (finding-D1 family), not an observed-edge
mismatch.  The two are separated here by the phase test: observed-edge
sawtooth alternates every bin (lag-1 sign correlation ~ -1); basis ripple
alternates every OTHER bin with phase locked to basis edges (within-pair vs
across-pair structure).  Group-edge shift snapping: the spec's half-bin
(0.05 dex) shift cannot land on a bin edge, so it is snapped to the adjacent
one-bin (0.1 dex) shifts in both directions, as prespecified.
"""
import json

import numpy as np

OUT = "/home/mfho/wt_repair_phaseB/diagnostics_phaseB/twin_nhat"
WIN = (19.7, 21.6)
SHIFTS = {"minus_0p1": -0.1, "nominal": 0.0, "plus_0p1": +0.1}
BASE_EDGES = np.array([19.7, 20.3, 21.0, 21.6])

res = {"hypothesis": "H2 bin-edge/normalization mismatch",
       "label": "exploratory (prespecified discriminant, spec s7.2)",
       "mocks": {}}

for m in ("2lpt0", "london0", "saclay0"):
    d = np.load(f"{OUT}/base_{m}_both.npz")
    ne = d["nhat_edges"]
    win = d["win_mask"].astype(bool)
    z = d["z_c"]
    zc = z[win]
    lo = ne[:-1][win]
    # (i) lag-1 sign correlation of the 19 window per-bin z's
    s = np.sign(zc)
    lag1 = float(np.mean(s[:-1] * s[1:]))
    # (ii) basis-phase decomposition: phase 0 = lower half of its 0.2-dex
    # basis bin, phase 1 = upper half (basis edges from ntrue_edges)
    te = d["ntrue_edges"]
    phase = np.array([int(np.argmin(np.abs(te - l)) == -1) for l in lo])
    # a bin's basis parent: largest ntrue edge <= lo + eps
    parent = np.searchsorted(te, lo + 1e-9) - 1
    phase = np.array([0 if abs(lo[i] - te[parent[i]]) < 1e-6 else 1
                      for i in range(len(lo))])
    within_pair_sign = []
    for p in np.unique(parent):
        ii = np.where(parent == p)[0]
        if len(ii) == 2:
            within_pair_sign.append(float(np.sign(zc[ii[0]] * zc[ii[1]])))
    chi2_phase0 = float(np.sum(zc[phase == 0] ** 2))
    chi2_phase1 = float(np.sum(zc[phase == 1] ** 2))
    # (iii) rebin the residual onto the 0.2-dex BASIS bins (pairs): if the
    # per-bin chi2 collapses, the per-bin failure is basis-width ripple
    mu_c, obs_c = d["mu_c"], d["obs_c"]
    pair_z, pair_lo = [], []
    for p in np.unique(parent):
        ii = np.where(parent == p)[0]
        if len(ii) != 2:
            continue                      # the [21.5,21.6) half-bin leftover
        idx = np.where(win)[0][ii]
        mu2, ob2 = mu_c[idx].sum(), obs_c[idx].sum()
        pair_z.append(float((ob2 - mu2) / np.sqrt(max(mu2, 1e-12))))
        pair_lo.append(float(lo[ii[0]]))
    pair_z = np.array(pair_z)
    # (iv) one-bin group-edge shifts
    shift_tab = {}
    idx_all = np.arange(len(ne) - 1)
    for name, sh in SHIFTS.items():
        ed = BASE_EDGES + sh
        gres, gmu = [], []
        for glo, ghi in zip(ed[:-1], ed[1:]):
            msk = (ne[:-1] >= glo - 1e-9) & (ne[1:] <= ghi + 1e-9)
            gres.append(float((obs_c[msk] - mu_c[msk]).sum()))
            gmu.append(float(mu_c[msk].sum()))
        var = np.array(gmu)
        # descriptive z with survey-only variance (the cal part is G1-only and
        # is reported separately in H9); sign pattern is what H2 reads
        shift_tab[name] = {"group_edges": ed.tolist(),
                           "residual": gres,
                           "z_survey_only": (np.array(gres)
                                             / np.sqrt(var)).tolist()}
    res["mocks"][m] = {
        "per_bin_lo": lo.tolist(), "per_bin_z": zc.tolist(),
        "lag1_sign_correlation": lag1,
        "n_sign_changes_of_18": int(np.sum(s[:-1] * s[1:] < 0)),
        "basis_phase_of_bins": phase.tolist(),
        "within_basis_pair_sign_products": within_pair_sign,
        "chi2_lower_half_bins": chi2_phase0,
        "chi2_upper_half_bins": chi2_phase1,
        "chi2_dof_0p1dex": float(np.sum(zc ** 2) / len(zc)),
        "pair_rebin_lo": pair_lo,
        "pair_rebin_z": pair_z.tolist(),
        "chi2_dof_0p2dex_pairs": float(np.sum(pair_z ** 2) / len(pair_z)),
        "group_edge_shift": shift_tab,
    }

with open(f"{OUT}/h02_edges_morphology.json", "w") as fh:
    json.dump(res, fh, indent=1)

for m, r in res["mocks"].items():
    print(m)
    print("  z (0.1 dex):", " ".join("%+.1f" % v for v in r["per_bin_z"]))
    print("  lag1 sign corr = %+.2f, sign changes %d/18"
          % (r["lag1_sign_correlation"], r["n_sign_changes_of_18"]))
    print("  chi2 lower-half %.1f vs upper-half %.1f"
          % (r["chi2_lower_half_bins"], r["chi2_upper_half_bins"]))
    print("  z (0.2 dex pairs):", " ".join("%+.1f" % v
                                           for v in r["pair_rebin_z"]))
    print("  chi2/dof: 0.1dex %.2f -> 0.2dex pairs %.2f"
          % (r["chi2_dof_0p1dex"], r["chi2_dof_0p2dex_pairs"]))
    for name, t in r["group_edge_shift"].items():
        print("  shift %-9s residual %s z %s"
              % (name, ["%.0f" % v for v in t["residual"]],
                 ["%+.2f" % v for v in t["z_survey_only"]]))
