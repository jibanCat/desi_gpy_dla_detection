#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-B bounded diagnosis (exploratory) — base folds.

Runs forward_selftest.selftest on the three fresh packs under
resp_clamp="both" (adopted) and "hi" (committed diagnostic bracket, H6), and
saves the per-bin / per-(bin,stratum) aggregates every later discriminant
reads.  NO new model freedom: the fold is called at the truth-equivalent
point exactly as the closure table calls it.

Sanity gate: the "both" twin run must reproduce the closure table's window
by_nhat chi2/dof (22.094...) and 3-group residual (-1760.8, +130.5, +450.2)
before anything downstream is trusted.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/mfho/wt_repair_phaseB")

from CDDF_analysis.hbi_mcmc.pack import load_pack                    # noqa: E402
from CDDF_analysis.hbi_mcmc import forward_selftest as FS            # noqa: E402
from CDDF_analysis.hbi_mcmc.gate_covariance import (                 # noqa: E402
    group_aggregator, PRIMARY_GROUP_EDGES)

PACK = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/"
        "modelA_pack_{m}_winlya_only_pad19p0_molly172_bw0p2.npz")
OUT = "/home/mfho/wt_repair_phaseB/diagnostics_phaseB/twin_nhat"
WIN = (19.7, 21.6)

summary = {"label": "exploratory (bounded diagnostic pass)",
           "window": list(WIN), "mocks": {}}

for m in ("2lpt0", "london0", "saclay0"):
    pk = load_pack(PACK.format(m=m))
    ne = np.asarray(pk.nhat_edges, float)
    win_mask = (ne[:-1] >= WIN[0] - 1e-9) & (ne[1:] <= WIN[1] + 1e-9)  # (C,)
    A = group_aggregator(pk, PRIMARY_GROUP_EDGES)                       # (3, C)
    live = (np.asarray(pk.dX, float) > 0)[None, :, :]                   # (1,Kf,S)
    n0 = np.asarray(pk.fp_counts, float)                                # (C, S)
    eta_c = np.asarray(pk.fp_eta_c, float)
    w = float(pk.fp_w_sightline_ratio)
    E = np.asarray(pk.fp_E_alloc, float)                                # (Kf, S)

    rec = {"nhat_edges": ne.tolist(), "win_mask": win_mask.tolist(),
           "clamp": {}}
    for clamp in ("both", "hi"):
        st = FS.selftest(pk, resp_clamp=clamp)
        mu = np.where(live, st["mu"], 0.0)          # (C, Kf, S)
        mu_fp = np.where(live, st["mu_fp"], 0.0)
        obs = np.where(live, st["counts"], 0.0)
        mu_c = mu.sum(axis=(1, 2))
        mu_fp_c = mu_fp.sum(axis=(1, 2))
        obs_c = obs.sum(axis=(1, 2))
        # per (c, s) sums over k — for the SNR-stratum table (H10) and the
        # delta-method calibration variance (H9 / fit metric)
        mu_cs = mu.sum(axis=1)                      # (C, S)
        mu_fp_cs = mu_fp.sum(axis=1)
        obs_cs = obs.sum(axis=1)
        # delta method: mu_fp[c,k,s] = w*(1-eta_c)*n0[c,s]*E[k,s]  (log_t=0)
        # d mu_c/d n0[c,s] = w*(1-eta_c[c]) * sum_k(live E[k,s]); var = sum s
        Elive_s = np.where(live[0], E, 0.0).sum(axis=0)      # (S,)
        dmu_dn0 = w * (1.0 - eta_c)[:, None] * Elive_s[None, :]
        var_cal_c = np.where(n0 > 0, dmu_dn0 ** 2 * n0, 0.0).sum(axis=1)
        resid_c = obs_c - mu_c
        d_grp = (A @ obs_c) - (A @ mu_c)
        z_c = resid_c / np.sqrt(np.maximum(mu_c, 1e-12))
        win_idx = np.where(win_mask)[0]
        chi2_dof_win = float(np.sum(z_c[win_idx] ** 2) / len(win_idx))
        np.savez(f"{OUT}/base_{m}_{clamp}.npz",
                 nhat_edges=ne, win_mask=win_mask,
                 mu_c=mu_c, mu_fp_c=mu_fp_c, obs_c=obs_c,
                 mu_cs=mu_cs, mu_fp_cs=mu_fp_cs, obs_cs=obs_cs,
                 var_cal_c=var_cal_c, resid_c=resid_c, z_c=z_c,
                 d_grp=d_grp, A=A, snr_edges=np.asarray(pk.snr_edges, float),
                 ntrue_edges=np.asarray(pk.ntrue_edges, float))
        rec["clamp"][clamp] = {
            "chi2_dof_window": chi2_dof_win,
            "group_residual": d_grp.tolist(),
            "window_total_mu": float(mu_c[win_idx].sum()),
            "window_total_obs": float(obs_c[win_idx].sum()),
            "full_total_mu": float(mu_c.sum()),
            "full_total_obs": float(obs_c.sum()),
        }
    summary["mocks"][m] = rec

# sanity gate against the closure table
ct = json.load(open("/home/mfho/wt_repair_phaseB/CDDF_analysis/hbi_mcmc/"
                    "closure_table_phaseB.json"))
row0 = ct["rows"][0]
want_chi2 = row0["conditional"]["window"]["chi2_dof"]
want_resid = row0["predictive"]["residual"]
got = summary["mocks"]["2lpt0"]["clamp"]["both"]
ok_chi2 = abs(got["chi2_dof_window"] - want_chi2) < 1e-6
ok_res = np.allclose(got["group_residual"], want_resid, rtol=0, atol=1e-6)
summary["sanity"] = {
    "closure_chi2_dof": want_chi2, "recomputed": got["chi2_dof_window"],
    "closure_group_residual": want_resid,
    "recomputed_group_residual": got["group_residual"],
    "reproduced": bool(ok_chi2 and ok_res)}
if not (ok_chi2 and ok_res):
    print("SANITY FAILURE — downstream results must not be trusted",
          file=sys.stderr)

with open(f"{OUT}/h00_base_folds.json", "w") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary["sanity"], indent=1))
for m in summary["mocks"]:
    for cl in ("both", "hi"):
        r = summary["mocks"][m]["clamp"][cl]
        print(m, cl, "chi2/dof(win) = %.3f" % r["chi2_dof_window"],
              "d_grp =", ["%.1f" % v for v in r["group_residual"]])
