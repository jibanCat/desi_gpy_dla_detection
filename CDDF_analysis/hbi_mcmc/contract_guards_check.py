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


def ga_partition(pk, level, fp_share, level_tol):
    """G-A verdict. On a REAL pack the truth block is the all-zero sentinel, so the
    truth-point partition is undefined by construction: the level then measures the
    FP-only fold (~0.16) and used to be recorded as a spurious FAIL that the CP-1
    sbatch swallowed with `|| echo` (Paper-1 code review 2026-08-26). The real-data
    substitute, G_A_real_mode, is evaluated by cc_real_posterior on the predictive
    level of the fitted model; here the verdict is NOT_APPLICABLE, not FAIL."""
    truth = np.asarray(getattr(pk, "truth_counts", np.zeros(1)), float)
    if truth.sum() <= 0:
        return dict(level_mu_over_obs=round(level, 4), fp_share=round(fp_share, 4), tol=level_tol,
                    status="NOT_APPLICABLE_REAL_PACK",
                    note="truth_counts is the all-zero real-data sentinel; the truth-point partition "
                         "is undefined; G_A_real_mode in cc_real_posterior is the guard of record")
    ga_ok = abs(level - 1.0) <= level_tol
    return dict(level_mu_over_obs=round(level, 4), fp_share=round(fp_share, 4), tol=level_tol,
                status="PASS" if ga_ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-nonstandard-grid", action="store_true", help="VALIDATION-ONLY (high-z HBI extension trial)")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--cand-npz", default=None,
                    help="npz with <prefix>__mu/sig/skew/rng candidate "
                         "surfaces to audit against G-CC")
    ap.add_argument("--cand-prefix", default=None)
    ap.add_argument("--phi-tol", type=float, default=0.02)
    ap.add_argument("--level-tol", type=float, default=0.06)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    pk = load_pack(a.pack, allow_nonstandard_grid=a.allow_nonstandard_grid)
    ne = np.asarray(pk.nhat_edges, float)
    theta = np.log(np.clip(np.asarray(truth_f(pk), float), 1e-300, None))
    lam = np.asarray(pk.fp_counts, float) / float(pk.fp_ell_eff)
    obs = float(np.asarray(pk.counts, float).sum())
    report = {"pack": a.pack}
    fail = False

    # G-A partition at the truth point. For an adopted_masses_override pack (2026-09-03 HZ2, default-off) the DEPLOYED
    # kernel is the override, not the legacy resp_* surfaces the extractor still carries, so the partition is evaluated with
    # the fail-closed adopted fold.
    if getattr(pk, "adopted_masses_override", None) is not None:
        from CDDF_analysis.hbi_mcmc.count_conserving_fold import cc_fold_adopted as _cfa
        mu, parts = _cfa(pk, theta, lam)
    else:
        mu, parts = cc_fold_cmarginal(pk, theta, lam)
    level = float(mu.sum() / obs)
    fp_share = float(parts["fp"].sum() / mu.sum())
    report["G_A_partition"] = ga_partition(pk, level, fp_share, a.level_tol)
    fail |= report["G_A_partition"]["status"] == "FAIL"

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

    # G-C atomicity: TP-convention identifier + adopted stamps (v1.2 packs)
    tp_id = getattr(pk, "tp_convention_id", None)
    if tp_id is None:
        report["G_C_atomic_tp_id"] = dict(
            value=None, status="MISSING",
            note=("pack predates the ratified contract; the identifier "
                  "lands with the PI-authorized pack rebuild — MISSING is "
                  "reported, not failed, for legacy packs"))
    else:
        # v1.2 pack: the stamp group already passed the loader's all-or-none
        # validation; here verify the stored count-conservation reference
        # and exercise the fail-closed adopted fold end to end.
        from CDDF_analysis.hbi_mcmc.count_conserving_fold import \
            cc_fold_adopted
        phi_stored = np.asarray(pk.adopted_phi_ref, float)
        override = getattr(pk, "adopted_masses_override", None)
        # 2026-09-03 HZ2 (default-off extension): for an adopted_masses_override pack the deployed kernel IS the
        # override, whose column sums are the in-grid fractions (see count_conserving_fold.cc_fold_adopted).
        phi_deployed = (np.asarray(override, float).sum(axis=2) if override is not None
                        else phi_from_surfaces(pk))
        dphi_ref = float(np.max(np.abs(phi_stored - phi_deployed)))
        gc_ok = dphi_ref <= 1e-9
        mu_a, parts_a = cc_fold_adopted(pk, theta, lam)
        lvl_a = float(mu_a.sum() / obs)
        report["G_C_atomic_tp_id"] = dict(
            value=tp_id, contract=pk.contract_id,
            adopted_version=pk.adopted_resp_version,
            adopted_representation=("masses_override" if override is not None else "surfaces"),
            carrier_draws=int(np.asarray(pk.adopted_carrier_mu).shape[0]),
            stored_phi_ref_max_dev=dphi_ref,
            adopted_cc_level=round(lvl_a, 4),
            adopted_level_identity=("PASS" if abs(lvl_a - level) <= 1e-6
                                    else "FAIL"),
            status="PASS" if (gc_ok and abs(lvl_a - level) <= 1e-6)
                   else "FAIL")
        if override is not None:
            # the legacy surfaces are NOT the deployed kernel here; the level identity against them is undefined, and
            # `level` above is already the adopted fold (so the identity is trivially exact) — say so explicitly.
            report["G_C_atomic_tp_id"]["adopted_level_identity"] = "NOT_APPLICABLE_OVERRIDE (deployed kernel = masses override; G_A_partition uses it)"
            report["G_C_atomic_tp_id"]["status"] = "PASS" if gc_ok else "FAIL"
        fail |= report["G_C_atomic_tp_id"]["status"] == "FAIL"

    print(json.dumps(report, indent=1))
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
