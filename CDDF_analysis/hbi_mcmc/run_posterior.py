# -*- coding: utf-8 -*-
"""run_posterior.py — THE paper-facing posterior estimator (DLA + sub-DLA, one model).

What this is
------------
One joint posterior over f(N, z) and every nuisance (completeness psi_C,
response coefficients psi_K, transfer factors t, the loa-0 FP block), sampled
with NUTS, from which BOTH tiers are read off the SAME draws:

    sub-DLA  window [19.5, 20.3)
    DLA      >= 20.0 and >= 20.3

They are one inference, not two: the two tiers share the completeness surface,
the FP model, the response kernel and the pathlength, so any tier ratio or
difference is formed PER DRAW.  Nothing here combines independently
marginalized intervals and nothing here recenters a band on a plug-in point.

    estimand = POSTERIOR_MEDIAN_CI
        point = posterior MEDIAN of the draws
        band  = 16/84 and 2.5/97.5 percentiles of THE SAME draws

A plug-in MAP may be computed with ``--map-diagnostic``.  It is stamped
estimand=PLUGIN_MAP, carries NO band, and is a diagnostic of the mode-vs-median
gap only.

THE FORWARD-MODEL GATE (fail-closed)
------------------------------------
Sampler convergence is NOT sufficient.  Before any sampler time is spent this
runner folds the pack's OWN truth through the pack's OWN kernel
(``forward_selftest``) and REFUSES to run unless the fold reproduces the
observed counts to the stated tolerance.  As of 2026-07-28 every REAL mock pack
FAILS this gate (2lpt0 v1.1: total mu/obs 0.7312, z=+93.3; the lowest n-hat bin
0.1655 at z=+216) because of finding D1 (the true-N basis is truncated at the
reporting floor, so the pack's truth cannot arithmetically reproduce its own
counts).  Until a basis-padded pack is re-extracted, this estimator is
PAPER-FACING ONLY ON SYNTHETIC PACKS, where closure is provable and is proved
by the same gate on every run.

``--allow-open-forward-model REASON`` runs anyway; the artifact is then stamped
``paper_facing=False`` and ``forward_model_closes=False``, permanently.

Scales
------
  --smoke        500 warmup / 500 samples / 4 chains  (minutes, 8 cores)
  (default)     1000 / 1000 / 4                        (the production scale;
                                                        see the sbatch)

Usage
-----
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m CDDF_analysis.hbi_mcmc.run_posterior \
        --synthetic-smoke --out out.json --smoke

    ... --pack /scratch/.../modelA_pack_2lpt0_v11.npz --out out.json

MOCK / SYNTHETIC ONLY.  Refuses any pack naming the real survey.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np

from CDDF_analysis.hbi_mcmc import ratification as RAT

__all__ = ["forward_closure_gate", "GATE", "stamp_metadata", "main",
           "PROVISIONAL_GATE_TOLERANCES"]

# --- the forward-model closure gate -------------------------------------------
# Tolerances are on Poisson z-scores of the pure truth-fold (no sampling).
# They are deliberately loose: this gate is not a goodness-of-fit test, it is a
# tripwire against a forward model that is broken by ORDERS of magnitude.
#
# EVERY tolerance is a named constant here so it can be ratified as a number
# rather than discovered inside the gate body.
GATE = {
    "z_total_max": 5.0,      # |z| on the total predicted-vs-observed counts
    "z_bin_max": 5.0,        # max |z| over reported n-hat bins with obs > 0
    "chi2_dof_max": 3.0,     # chi2/dof over those bins
    # --- the z- and SNR-marginal arms (added 2026-07-29) --------------------
    # ``ratio_tables`` has ALWAYS computed ``by_z`` and ``by_snr``; the gate
    # consumed only ``total`` and ``by_nhat`` and DISCARDED them.  A forward
    # model can close in total and in the N-marginal while carrying a large
    # z-marginal tilt -- which is precisely the standing ZTILT defect -- and
    # such a pack sailed through.  Two arms per marginal, because they fail
    # in different regimes:
    #   * the |z| arm catches a swing that is large relative to Poisson noise
    #     (high-count packs);
    #   * the RATIO-SPAN arm catches a swing that is large in PHYSICAL terms
    #     but small in z because the marginal is count-starved.  A ~22%
    #     max-to-min spread in mu/obs across z is a systematic no sampler can
    #     repair, whatever its z-score.
    "z_zbin_max": 5.0,           # max |z| over fine-z bins with obs > 0
    "z_snrbin_max": 5.0,         # max |z| over SNR strata with obs > 0
    # 🔴 UNRATIFIED as of 2026-07-29 (decision 8) -- REPORTED, DOES NOT GATE.
    "ratio_span_by_z_max": 0.10,     # max(mu/obs) - min(mu/obs) across z
    "ratio_span_by_snr_max": 0.15,   # ... across SNR strata (fewer, noisier)
}

# 🔴 UNRATIFIED GATE TOLERANCES -- decision 8 (PI, 2026-07-29).
#
# The two RATIO-SPAN numbers were chosen by eye by the author when the by_z /
# by_snr arms were added on 2026-07-29.  The PI DECLINED to ratify them and
# required that the statistics be defined and calibrated prospectively.
#
# WHAT THAT NOW MEANS IN CODE (changed 2026-07-29, this branch):
#   the two span statistics are still COMPUTED and REPORTED on every run, and
#   are stamped into every artifact, but they NO LONGER CONTRIBUTE TO
#   PASS/FAIL.  Exceeding the proposed number produces an entry in
#   ``advisories``, never in ``failures``.  Previously they were armed and
#   could refuse work on an uncalibrated number.
#
# The rationale for report-but-do-not-gate (rather than deleting the arms) is
# in ``ratification.py``: the reported values ARE the calibration data.  The
# prospective calibration -- exact definitions, null distribution, sampling
# procedure, false-alarm rate -- is
# ``docs/ratio_span_calibration_spec.md``.
#
# THE SINGLE SOURCE OF TRUTH IS ``ratification.py``.  The names below are
# derived from it, not maintained in parallel.
PROVISIONAL_GATE_TOLERANCES = RAT.unratified_names()

PROVISIONAL_GATE_TOLERANCES_NOTE = RAT.UNRATIFIED_NOTE

_REAL_TOKENS = ("main_dark", "loa_main_dark", "matterhorn", "dr3")


def forward_closure_gate(pack, *, resp_clamp="both", gate=None):
    """Fold the pack's own truth through its own kernel; PASS/FAIL + evidence.

    Returns a dict; ``["pass"]`` is the fail-closed verdict.  Requires no
    sampling (seconds), so it is cheap enough to run on every invocation and
    is run BEFORE the sampler is constructed.
    """
    from CDDF_analysis.hbi_mcmc import forward_selftest as FS

    gate = dict(GATE if gate is None else gate)
    if getattr(pack, "truth_counts", None) is None:
        return {"pass": False, "reason": "pack carries no truth_counts; "
                "the forward model cannot be self-tested", "gate": gate}
    res = FS.selftest(pack, resp_clamp=resp_clamp)
    tab = FS.ratio_tables(res, pack)

    floor = float(np.asarray(pack.nhat_edges, float)[0])
    rows = [b for b in tab["by_nhat"]
            if b["obs"] > 0 and b["lo"] >= floor - 1e-9]
    z = np.array([b["z"] for b in rows], float)
    chi2_dof = float((z ** 2).sum() / max(len(z), 1))
    z_bin_max = float(np.abs(z).max()) if len(z) else float("nan")
    z_total = float(abs(tab["total"]["z"]))

    fails = []
    # ``advisories`` are computed, reported, and DO NOT contribute to
    # ``pass``.  They exist because decision 8 declined to ratify the two
    # ratio-span thresholds: the statistic is worth accumulating, the number
    # is not worth refusing work on.  See ratification.py.
    advisories = []
    if not (z_total <= gate["z_total_max"]):
        fails.append(f"|z_total|={z_total:.2f} > {gate['z_total_max']}")
    if not (z_bin_max <= gate["z_bin_max"]):
        fails.append(f"max|z_bin|={z_bin_max:.2f} > {gate['z_bin_max']}")
    if not (chi2_dof <= gate["chi2_dof_max"]):
        fails.append(f"chi2/dof={chi2_dof:.2f} > {gate['chi2_dof_max']}")

    # --- the z- and SNR-marginal arms -------------------------------------
    # ``ratio_tables`` already computed these; the gate used to throw them
    # away, so a forward model with a large z-marginal tilt but a closing
    # total and N-marginal passed.  No new compute.
    marg = {}
    for key, zkey, spankey in (("by_z", "z_zbin_max", "ratio_span_by_z_max"),
                               ("by_snr", "z_snrbin_max",
                                "ratio_span_by_snr_max")):
        mrows = [r for r in (tab.get(key) or []) if r.get("obs", 0) > 0]
        zs = np.array([r["z"] for r in mrows], float)
        zmax = float(np.abs(zs).max()) if len(zs) else float("nan")
        # ONE definition of the span, in FS.ratio_span, with the exact
        # mathematical statement in its docstring (decision 8 asked for it).
        sp = FS.ratio_span(mrows)
        span = sp["span"]
        marg[key] = dict(rows=mrows, zmax=zmax, span=span, span_detail=sp)
        if len(zs) and not (zmax <= gate[zkey]):
            fails.append(f"max|z| in {key} = {zmax:.2f} > {gate[zkey]}")
        if not (span <= gate[spankey]):
            msg = (f"ratio span in {key} = {span:.4f} "
                   f"(mu/obs {sp['lo']:.4f}..{sp['hi']:.4f}, "
                   f"n_rows_used={sp['n_rows_used']}) "
                   f"> {spankey} = {gate[spankey]}")
            if RAT.is_ratified(spankey):
                fails.append(msg)
            else:
                # UNRATIFIED (decision 8): report, do not gate.
                advisories.append(
                    msg + " [UNRATIFIED tolerance -- ADVISORY ONLY, this does "
                          "NOT block the run; the threshold has no calibrated "
                          "false-alarm rate. See "
                          "docs/ratio_span_calibration_spec.md]")

    worst = sorted(rows, key=lambda b: -abs(b["z"]))[:5]
    return {
        "by_z": marg["by_z"]["rows"],
        "by_snr": marg["by_snr"]["rows"],
        "z_zbin_max": marg["by_z"]["zmax"],
        "z_snrbin_max": marg["by_snr"]["zmax"],
        "ratio_span_by_z_detail": marg["by_z"]["span_detail"],
        "ratio_span_by_snr_detail": marg["by_snr"]["span_detail"],
        "ratio_span_definition": (
            "max_r(mu_r/obs_r) - min_r(mu_r/obs_r) over the arm's rows with "
            "obs_r > 0; 0 (VACUOUS) when fewer than 2 such rows. Exact "
            "statement: CDDF_analysis.hbi_mcmc.forward_selftest.ratio_span. "
            "UNRATIFIED statistic -- reported, does not gate."),
        "ratio_span_by_z": marg["by_z"]["span"],
        "ratio_span_by_snr": marg["by_snr"]["span"],
        "pass": not fails,
        "failures": fails,
        # reported, never gating -- see ratification.py
        "advisories": advisories,
        "gate": gate,
        # which of the numbers above a deciding authority has NOT ratified
        "gate_tolerances_provisional": list(PROVISIONAL_GATE_TOLERANCES),
        "gate_tolerances_provisional_note": PROVISIONAL_GATE_TOLERANCES_NOTE,
        "gate_tolerances_unratified": list(RAT.unratified_names()),
        "gate_tolerances_ratified": list(RAT.ratified_names()),
        "unratified_effect": RAT.UNRATIFIED_EFFECT,
        "ratification": RAT.ratification_stamp(),
        "total_mu": float(tab["total"]["mu"]),
        "total_obs": float(tab["total"]["obs"]),
        "total_ratio": float(tab["total"]["ratio"]),
        "z_total": z_total,
        "z_bin_max": z_bin_max,
        "chi2_dof": chi2_dof,
        "n_bins": int(len(z)),
        "worst_bins": [{"lo": b["lo"], "hi": b["hi"], "ratio": b["ratio"],
                        "z": b["z"]} for b in worst],
        "resp_clamp": resp_clamp,
    }


# --- provenance ----------------------------------------------------------------

def _git_sha_full():
    """40-char HEAD SHA + a dirty flag on THIS module's dependency set.

    Captured at PROCESS START by ``main`` (never at write time): a multi-hour
    run whose repo advances mid-flight would otherwise stamp a commit it never
    executed -- exactly how the 2026-07-11 broken-kernel ablation ended up
    stamped with the commit containing the kernel fix it predates.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    deps = [os.path.join(here, f) for f in
            ("run_posterior.py", "model_a.py", "forward.py", "pack.py",
             "forward_selftest.py", "diagnostics.py")]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--"] + deps,
            cwd=here, text=True).strip())
        return sha, dirty
    except Exception:
        return "unknown", True


def stamp_metadata(*, code_commit, code_dirty, cfg, args, gate_report,
                   estimand, paper_facing, pack_provenance=None,
                   bypasses=None):
    """The artifact metadata block. Every field the 2026-07-28 contract requires.

    ``bypasses`` -- every gate-bypass flag actually in force for this run
    (``allow_low_farr``, ``allow_open_forward_model``, ...).  Until 2026-07-29
    ``--allow-low-farr`` appeared in NEITHER ``paper_facing`` NOR the stamp, so
    a run with the Farr headroom gate switched off was indistinguishable in the
    artifact from one that passed it.  A bypass is now RECORDED and FORCES
    ``paper_facing=False``: a number obtained by switching a gate off cannot be
    certified by that gate.
    """
    bypasses = dict(bypasses or {})
    paper_facing = bool(paper_facing) and not bypasses
    return {
        "bypasses": bypasses,
        "estimand": estimand,
        "resp_kind": "forward",   # Model A packs carry NO kappa object by schema
        "kernel_note": ("the measured forward response (skew-normal moment "
                        "surfaces); the GP-posterior 'kappa' object is not "
                        "representable in a Model A pack"),
        "code_commit": code_commit,
        "code_dirty": bool(code_dirty),
        "routine": "CDDF_analysis/hbi_mcmc/run_posterior.py",
        "rederive": args["rederive"],
        "n_chains": int(cfg.num_chains),
        "n_warmup": int(cfg.num_warmup),
        "n_samples": int(cfg.num_samples),
        "seed": int(cfg.seed),
        "target_accept": float(cfg.target_accept),
        "resp_clamp": cfg.resp_clamp,
        "fp_mode": cfg.fp_mode,
        "band_recenter": False,       # structurally impossible here
        "marginal_combined": False,   # tiers share draws; ratios are per-draw
        "paper_facing": bool(paper_facing),
        "forward_model_closes": bool(gate_report.get("pass")),
        "forward_gate": gate_report,
        # hoisted to the TOP of the stamp, not left buried in forward_gate: a
        # reader of the artifact alone must see which gate numbers are not
        # ratified without opening the nested report.
        "gate_tolerances_provisional": list(
            gate_report.get("gate_tolerances_provisional")
            or PROVISIONAL_GATE_TOLERANCES),
        "gate_tolerances_provisional_note": (
            gate_report.get("gate_tolerances_provisional_note")
            or PROVISIONAL_GATE_TOLERANCES_NOTE),
        # the FULL ratification state, at the top of the stamp: which criteria
        # were authorised to refuse this run, by whom, and on what date.
        "ratification": RAT.ratification_stamp(),
        "gate_advisories": list(gate_report.get("advisories") or []),
        "date": time.strftime("%Y-%m-%d"),
        "pack_provenance": pack_provenance,
        "scope": "MOCK / SYNTHETIC ONLY",
    }


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


# --- runner ---------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pack", help="path to a stamped Model A pack (.npz)")
    src.add_argument("--synthetic-smoke", action="store_true",
                     help="generate the small synthetic pack (known truth, "
                          "closure provable) instead of loading one")
    ap.add_argument("--synthetic-seed", type=int, default=0)
    ap.add_argument("--synthetic-full-grid", action="store_true",
                    help="synthetic pack on the REAL 29x15x8 grid (production "
                         "geometry, still fully synthetic)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--resp-clamp", default="both", choices=("both", "hi", "off"))
    ap.add_argument("--fp-mode", default="joint", choices=("joint", "off"))
    ap.add_argument("--smoke", action="store_true",
                    help="500/500/4 -- minutes on 8 cores, for wiring + tests")
    ap.add_argument("--map-diagnostic", action="store_true",
                    help="also compute the plug-in MAP (labelled DIAGNOSTIC)")
    ap.add_argument("--allow-low-farr", metavar="REASON", default=None)
    ap.add_argument("--allow-open-forward-model", metavar="REASON", default=None,
                    help="run even though the truth-fold gate FAILS. The "
                         "artifact is stamped paper_facing=False forever.")
    a = ap.parse_args(argv)

    # ---- capture provenance at PROCESS START, before anything can move
    code_commit, code_dirty = _git_sha_full()
    t_start = time.time()

    # lazy: importing jax costs seconds, and the guards below must fire first
    from CDDF_analysis.hbi_mcmc.pack import (load_pack, synthetic_pack,
                                             small_test_grid)
    from CDDF_analysis.hbi_mcmc import model_a as MA

    # ---- real-data guard
    if a.pack:
        low = a.pack.lower()
        for tok in _REAL_TOKENS:
            assert tok not in low, f"REAL-DATA guard: pack path names {tok!r}"
        pack = load_pack(a.pack)
        prov = pack.provenance or {}
        blob = json.dumps(prov).lower()
        for tok in _REAL_TOKENS:
            assert tok not in blob, f"REAL-DATA guard (provenance): {tok!r}"
        pack_name = os.path.basename(a.pack)
    else:
        kw = {} if a.synthetic_full_grid else small_test_grid()
        pack = synthetic_pack(seed=a.synthetic_seed, **kw)
        prov = pack.provenance
        pack_name = (f"synthetic(seed={a.synthetic_seed},"
                     f"{'full' if a.synthetic_full_grid else 'small'}_grid)")

    # ---- THE FORWARD-MODEL GATE, before any sampler time
    gate = forward_closure_gate(pack, resp_clamp=a.resp_clamp)
    print(f"[gate] forward-model closure: "
          f"{'PASS' if gate['pass'] else 'FAIL'}  "
          f"total mu/obs={gate['total_ratio']:.4f} z={gate['z_total']:+.1f}  "
          f"max|z_bin|={gate['z_bin_max']:.1f}  chi2/dof={gate['chi2_dof']:.2f}")
    for b in gate["worst_bins"]:
        print(f"        [{b['lo']:.1f},{b['hi']:.1f})  ratio={b['ratio']:.4f}  "
              f"z={b['z']:+.1f}")
    for adv in gate.get("advisories") or []:
        print(f"[gate] ADVISORY (does NOT block): {adv}")
    if not gate["pass"]:
        msg = ("FORWARD-MODEL CLOSURE GATE FAILED: " + "; ".join(gate["failures"])
               + ".\nThe posterior would be a faithful sample of a WRONG model. "
                 "Sampler convergence cannot repair this. "
                 "Re-run with --allow-open-forward-model REASON to proceed "
                 "with a permanently non-paper-facing artifact.")
        if a.allow_open_forward_model is None:
            raise SystemExit("[gate] " + msg)
        print("[gate] OVERRIDDEN: " + a.allow_open_forward_model)

    if a.smoke:
        a.warmup, a.samples, a.chains = 500, 500, 4

    cfg = MA.ModelAConfig(
        num_warmup=a.warmup, num_samples=a.samples, num_chains=a.chains,
        seed=a.seed, target_accept=a.target_accept, resp_clamp=a.resp_clamp,
        fp_mode=a.fp_mode, enforce_farr_gate=(a.allow_low_farr is None))

    mcmc, red = MA.run_model_a(pack, cfg)
    summ = MA.posterior_summary(red, pack)
    wall = time.time() - t_start

    bypasses = {}
    if a.allow_low_farr is not None:
        bypasses["allow_low_farr"] = a.allow_low_farr
    if a.allow_open_forward_model is not None:
        bypasses["allow_open_forward_model"] = a.allow_open_forward_model
    paper_facing = (bool(gate["pass"])
                    and bool((red.get("diagnostics") or {}).get("policy_pass"))
                    and not bypasses)
    if bypasses:
        for k, v in sorted(bypasses.items()):
            print(f"[gate] BYPASS IN FORCE: {k} = {v!r}\n"
                  f"       -> this artifact is stamped paper_facing=False, "
                  f"permanently.")

    rederive = (
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "/home/mfho/.conda/envs/gpdla-hbi/bin/python -m "
        "CDDF_analysis.hbi_mcmc.run_posterior "
        + (f"--pack {a.pack} " if a.pack else
           f"--synthetic-smoke --synthetic-seed {a.synthetic_seed} "
           + ("--synthetic-full-grid " if a.synthetic_full_grid else ""))
        + f"--out <out> --warmup {a.warmup} --samples {a.samples} "
          f"--chains {a.chains} --seed {a.seed} "
          f"--resp-clamp {a.resp_clamp} --fp-mode {a.fp_mode}"
        + (f" --allow-low-farr '{a.allow_low_farr}'" if a.allow_low_farr else "")
        + (f" --allow-open-forward-model '{a.allow_open_forward_model}'"
           if a.allow_open_forward_model else ""))

    out = {
        "pack": pack_name,
        "posterior": summ,
        "diagnostics": red.get("diagnostics"),
        "farr_ratio": red.get("farr_ratio"),
        "wallclock_s": wall,
        "metadata": stamp_metadata(
            code_commit=code_commit, code_dirty=code_dirty, cfg=cfg,
            args={"rederive": rederive}, gate_report=gate,
            estimand=MA.ESTIMAND_POSTERIOR, paper_facing=paper_facing,
            pack_provenance=prov, bypasses=bypasses),
    }
    # truth closure when the pack carries a known truth (synthetic packs do)
    if getattr(pack, "truth", None) is not None:
        out["truth_closure"] = _truth_closure(pack, summ)

    if a.map_diagnostic:
        out["plugin_map_DIAGNOSTIC"] = MA.plugin_map_diagnostic(
            pack, cfg, seed=a.seed)
        out["plugin_map_DIAGNOSTIC"]["note"] = (
            "DIAGNOSTIC ONLY. Never the reported point; never to be paired "
            "with the credible intervals above.")

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=_jsonable)

    d = red.get("diagnostics") or {}
    print(f"[posterior] wrote {a.out}  wall={wall:.0f}s  "
          f"r_hat_max={d.get('r_hat_max'):.5f} "
          f"ess_bulk_min={d.get('ess_bulk_min'):.0f} "
          f"ess_tail_min={d.get('ess_tail_min'):.0f} "
          f"div={d.get('n_divergent')} policy_pass={d.get('policy_pass')} "
          f"paper_facing={paper_facing}")
    for tier, blk in summ["tiers"].items():
        q = blk["dndx_allz"]
        print(f"    {tier:<18s} dN/dX  q50={q['point_q50']:.6g}  "
              f"[{q['q16']:.6g}, {q['q84']:.6g}]  (68% CI)")
    return out


def _truth_closure(pack, summ):
    """Synthetic-truth closure: is the KNOWN truth inside the credible interval?

    This is the coverage evidence the PI decision requires, at n=1 per run; the
    coverage TEST over many synthetic realizations is tests/
    test_posterior_estimator.py::test_synthetic_coverage_smoke.
    """
    from CDDF_analysis.hbi_mcmc.model_a import TIERS
    f_true = np.asarray(pack.truth["f_true"], float)      # (B, Kf)
    ntrue = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    dN = np.diff(ntrue)
    dX_k = np.asarray(pack.dX, float).sum(axis=1)
    rep = Nc >= float(np.asarray(pack.nhat_edges, float)[0]) - 1e-9
    out = {}
    for tier, (lo, hi) in TIERS.items():
        if tier not in summ["tiers"]:
            continue
        sel = (Nc >= lo - 1e-9) & (Nc < hi - 1e-9) & rep
        dndx_k = (f_true[sel, :] * dN[sel, None]).sum(axis=0)
        om_k = (f_true[sel, :] * (10.0 ** (Nc[sel] - 21.0))[:, None]
                * dN[sel, None]).sum(axis=0)
        row = {}
        for name, tk in (("dndx_allz", dndx_k), ("omega_allz", om_k)):
            t = float((tk * dX_k).sum() / dX_k.sum())
            b = summ["tiers"][tier][name]
            row[name] = {
                "truth": t,
                "point_over_truth": float(b["point_q50"] / t) if t else None,
                "inside_68": bool(b["q16"] <= t <= b["q84"]),
                "inside_95": bool(b["q025"] <= t <= b["q975"]),
            }
        out[tier] = row
    return out


if __name__ == "__main__":
    main()
