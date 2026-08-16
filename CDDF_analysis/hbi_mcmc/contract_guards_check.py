#!/usr/bin/env python
"""contract_guards_check.py — standalone audit of the ratified joint C/K/FP
contract (PI rulings 2026-08-17: contract ratified in principle; guards
implemented after the count-conservation confirmation).

Checks, per pack:
  G-CC (count conservation): an admissible kernel representation must either
        (a) be the deployed frozen kernel, or (b) be folded RENORMALIZED to
        unit in-grid mass with the deployed phi_ref — a candidate folded
        naively whose in-grid fraction phi differs from the deployed phi by
        more than --phi-tol anywhere with exposure is REFUSED (it silently
        moves counting probability that belongs to C).
  G-A  (partition): the fold's arms must account for the observed counts:
        |mu_total/obs_total - 1| <= --level-tol at the truth point, with the
        TP/FP split reported. (On mocks this is the closure level; on any
        future real pack the check runs against the calibrated level band.)
  G-C  (atomicity): the pack must carry a single TP-convention identifier
        shared by its C/K/FP inputs. Current packs carry none — reported as
        MISSING (schema extension pending PI-authorized pack rebuild).

Usage:
  python -m CDDF_analysis.hbi_mcmc.contract_guards_check --pack PACK.npz
      [--cand-npz VARIANTS.npz --cand-prefix name] [--phi-tol 0.02]
      [--level-tol 0.06] [--json OUT]
Exit code 0 = all hard checks pass; 1 = a hard check failed.
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np

from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc.forward_selftest import truth_f
from CDDF_analysis.hbi_mcmc.count_conserving_fold import (
    cc_fold_cmarginal, phi_from_surfaces, surface_masses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--cand-npz", default=None,
                    help="npz with <prefix>__mu/sig/skew/rng candidate "
                         "surfaces to audit against G-CC")
    ap.add_argument("--cand-prefix", default=None)
    ap.add_argument("--phi-tol", type=float, default=0.02)
    ap.add_argument("--level-tol", type=float, default=0.06)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    pk = load_pack(a.pack)
    ne = np.asarray(pk.nhat_edges, float)
    theta = np.log(np.clip(np.asarray(truth_f(pk), float), 1e-300, None))
    lam = np.asarray(pk.fp_counts, float) / float(pk.fp_ell_eff)
    obs = float(np.asarray(pk.counts, float).sum())
    report = {"pack": a.pack}
    fail = False

    # G-A partition at the truth point
    mu, parts = cc_fold_cmarginal(pk, theta, lam)
    level = float(mu.sum() / obs)
    fp_share = float(parts["fp"].sum() / mu.sum())
    ga_ok = abs(level - 1.0) <= a.level_tol
    report["G_A_partition"] = dict(
        level_mu_over_obs=round(level, 4), fp_share=round(fp_share, 4),
        tol=a.level_tol, status="PASS" if ga_ok else "FAIL")
    fail |= not ga_ok

    # G-CC count conservation for a candidate representation
    if a.cand_npz:
        vz = np.load(a.cand_npz)
        p = a.cand_prefix
        cand = dict(mu_coef=vz[f"{p}__mu"], sig_coef=vz[f"{p}__sig"],
                    skew_coef=vz[f"{p}__skew"],
                    fit_rng=np.asarray(vz[f"{p}__rng"], float))
        phi_ref = phi_from_surfaces(pk)
        _, phi_c = surface_masses(pk, cand["mu_coef"], cand["sig_coef"],
                                  cand["skew_coef"], cand["fit_rng"], ne)
        # exposure-weighted: only truth bins with any completeness-weighted f
        f = np.exp(theta).sum(axis=1)
        wgt = f / f.sum()
        dphi = np.abs(phi_c - phi_ref)
        dphi_max = float(dphi.max())
        dphi_wmean = float((dphi * wgt[None, None, :]).sum() / 9.0)
        gcc_naive_ok = dphi_max <= a.phi_tol
        report["G_CC_count_conservation"] = dict(
            candidate=p, max_abs_dphi=round(dphi_max, 4),
            exposure_weighted_mean_dphi=round(dphi_wmean, 5),
            tol=a.phi_tol,
            naive_fold_admissible=bool(gcc_naive_ok),
            rule=("naive fold REFUSED (renormalize with the deployed "
                  "phi_ref, or re-derive C atomically)"
                  if not gcc_naive_ok else "naive fold within tolerance"))
        # the renormalized fold must restore the deployed level exactly
        mu_cc, _ = cc_fold_cmarginal(pk, theta, lam, renormalize=True,
                                     phi_ref=phi_ref, **cand)
        lvl_cc = float(mu_cc.sum() / obs)
        cc_ok = abs(lvl_cc - level) <= 1e-6
        report["G_CC_renormalized_level_identity"] = dict(
            level_cc=round(lvl_cc, 6), level_deployed=round(level, 6),
            status="PASS" if cc_ok else "FAIL")
        fail |= not cc_ok

    # G-C atomicity: TP-convention identifier
    tp_id = getattr(pk, "tp_convention_id", None)
    report["G_C_atomic_tp_id"] = dict(
        value=tp_id, status="MISSING" if tp_id is None else "PRESENT",
        note=("packs predate the ratified contract; the identifier lands "
              "with the next PI-authorized pack rebuild — MISSING is "
              "reported, not failed, until then"))

    print(json.dumps(report, indent=1))
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
