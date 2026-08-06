#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage-2A/2B step: predict group changes, then rerun the UNCHANGED closure.

ORDER ENFORCED (rulings §10 items 9 and 14): the PREDICTION pass runs
and is WRITTEN before any closure statistic is computed; the closure
pass re-runs the frozen Phase-B Layer-A/B/C machinery verbatim
(`gate_covariance.estimate_covariance` / `predictive_gate`,
`forward_selftest.ratio_tables`) with ONLY the calibration artifact
swapped for the ADOPTED Stage-2A response.

Artifact substitution note (equivalence documented): `extract_pack`
derives the pack's response block EXCLUSIVELY from the forward envelope
(mu/sig/skew coefficient surfaces + `resp_fit_range_from_forward_npz`);
substituting those fields on the FROZEN Phase-B packs via
`dataclasses.replace` is therefore bit-equivalent to re-extraction for
the response block, and every other pack block is response-independent
by construction. The adopted envelope's `phaseC_status` must be ADOPTED
— a quarantined artifact is REFUSED (rulings §3.1).

Usage:
  python diagnostics_phaseC/stage2A/predict_and_close.py --phase predict
  python diagnostics_phaseC/stage2A/predict_and_close.py --phase close
`--phase close` REFUSES to run unless the prediction JSON already
exists (the order guard).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

from CDDF_analysis.hbi_mcmc.pack import load_pack                    # noqa: E402
from CDDF_analysis.hbi_mcmc import forward_selftest as FS            # noqa: E402
from CDDF_analysis.hbi_mcmc import gate_covariance as GC             # noqa: E402

PACKS = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/"
         "modelA_pack_{m}_winlya_only_pad19p0_molly172_bw0p2.npz")
MOCKS = ("2lpt0", "london0", "saclay0")
NEW_ENV = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/"
           "stage0/forward_response_2lpt0_phaseC.npz")
PRED_JSON = os.path.join(_HERE, "stage2A_prediction.json")
CLOSE_JSON = os.path.join(_HERE, "stage2A_closure.json")


def _adopted_env():
    env = np.load(NEW_ENV, allow_pickle=True)
    status = str(env["phaseC_status"])
    if status != "ADOPTED":
        raise SystemExit(
            f"REFUSED: envelope status is {status!r}, not ADOPTED — a "
            "quarantined/stopped artifact never enters closure (rulings "
            "§3.1).")
    return env


def _swap_response(pk, env):
    return dataclasses.replace(
        pk,
        resp_mu_coef=np.asarray(env["mu_coef"], float),
        resp_sig_coef=np.asarray(env["sig_coef"], float),
        resp_skew_coef=np.asarray(env["skew_coef"], float),
        resp_N_fit_range=np.asarray(env["phaseC_fit_range"], float),
    )


def phase_predict():
    env = _adopted_env()
    out = {"schema": "phaseC_stage2A_prediction/v1",
           "date": time.strftime("%Y-%m-%d"),
           "code_commit": subprocess.check_output(
               ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip(),
           "note": ("PREDICTED group-mu changes from the adopted response, "
                    "computed and committed BEFORE any closure statistic "
                    "(rulings §10 step 9). No observed count enters this "
                    "file beyond the frozen calibration inputs."),
           "mocks": {}}
    for m in MOCKS:
        pk = load_pack(PACKS.format(m=m))
        A = GC.group_aggregator(pk, GC.PRIMARY_GROUP_EDGES)
        live = (np.asarray(pk.dX, float) > 0)[None, :, :]
        st0 = FS.selftest(pk, resp_clamp="both")
        pk2 = _swap_response(pk, env)
        st1 = FS.selftest(pk2, resp_clamp="both")
        mu0 = np.where(live, st0["mu"], 0).sum(axis=(1, 2))
        mu1 = np.where(live, st1["mu"], 0).sum(axis=(1, 2))
        d = (A @ mu1) - (A @ mu0)
        out["mocks"][m] = {
            "predicted_group_mu_old": (A @ mu0).tolist(),
            "predicted_group_mu_new": (A @ mu1).tolist(),
            "predicted_group_change_G1_G2_G3": d.tolist(),
            "predicted_per_bin_mu_change": (mu1 - mu0).tolist(),
        }
        print(f"[{m}] predicted dG = {np.round(d, 1).tolist()}")
    with open(PRED_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", PRED_JSON)


def phase_close():
    if not os.path.exists(PRED_JSON):
        raise SystemExit("REFUSED: prediction JSON absent — run "
                         "--phase predict first (order guard, §10).")
    env = _adopted_env()
    out = {"schema": "phaseC_stage2A_closure/v1",
           "date": time.strftime("%Y-%m-%d"),
           "code_commit": subprocess.check_output(
               ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip(),
           "statistics": "UNCHANGED frozen Phase-B machinery "
                         "(spec ef67a6d); only the calibration artifact "
                         "differs (adopted Stage-2A response)",
           "mocks": {}}
    pred = json.load(open(PRED_JSON))
    for m in MOCKS:
        pk2 = _swap_response(load_pack(PACKS.format(m=m)), env)
        res = GC.predictive_gate(pk2)                  # frozen B recipe
        st = FS.selftest(pk2, resp_clamp="both")
        tab = FS.ratio_tables(st, pk2)                 # frozen A recipe
        out["mocks"][m] = {
            "layerA_window_chi2_dof": tab["total"]["chi2_dof"],
            "layerB": res.report(),
            "predicted_change": pred["mocks"][m][
                "predicted_group_change_G1_G2_G3"],
            "observed_residual": [float(v) for v in res.residual],
        }
        print(f"[{m}] chi2/dof={tab['total']['chi2_dof']:.2f}  "
              f"T={res.T_obs:.2f} p={res.p_value}  "
              f"resid={np.round(res.residual, 1).tolist()}")
    with open(CLOSE_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", CLOSE_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("predict", "close"), required=True)
    a = ap.parse_args()
    (phase_predict if a.phase == "predict" else phase_close)()


if __name__ == "__main__":
    main()
