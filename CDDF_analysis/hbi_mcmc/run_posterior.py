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
#
# 🔴 RATIFICATION STATUS PER NUMBER -- decision 8 ratified EXACTLY THREE things
# and NONE of them is a |z| number.  ``|z| <= 5`` was called MALFORMED AS
# STATED and sent back for restatement; the restated form has not been
# ratified.  The four |z| arms below therefore GATE WITHOUT RATIFIED AUTHORITY
# (status RESTATED_NOT_RATIFIED in ratification.py) and that is stamped into
# every artifact as ``gate_tolerances_unratified_but_gating``.  Do NOT read
# "it is in GATE" as "somebody authorised it".
GATE = {
    # RESTATED_NOT_RATIFIED (inherited, f23961e 2026-07-28)
    "z_total_max": 5.0,      # |z| on the total predicted-vs-observed counts
    "z_bin_max": 5.0,        # max |z| over reported n-hat bins with obs > 0
    # RATIFIED by the PI (decision 8) -- the only ratified number in this dict
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
    # 🔴 RESTATED_NOT_RATIFIED.  These two were added in the SAME HUNK as the
    # two span numbers below, which the PI declined the same day (0e7fa0b,
    # 2026-07-29 10:21).  They ARE armed; nobody ratified them.
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
# 🔴 HISTORICAL CORRECTION, kept because it is the sentence that caused the
# defect.  This comment used to end:
#     "The z-score arms (z_total_max, z_bin_max, chi2_dof_max, z_zbin_max,
#      z_snrbin_max) are NOT in this set: they are conventional 5-sigma /
#      chi2-per-dof thresholds and pre-date this change."
# and the NOTE used to end "Every other tolerance in GATE is a conventional
# z-score/chi2 threshold and pre-dates this change."  BOTH WERE FALSE, and that
# sentence is how four |z| arms came to be written into a committed artifact as
# PI-ratified:
#   * z_zbin_max and z_snrbin_max were added BY THAT CHANGE (0e7fa0b,
#     2026-07-29 10:21), on four consecutive added lines of the SAME HUNK that
#     added the two ratio_span numbers the PI declined the same day.  They
#     pre-date nothing.  MEASURED, and re-measured independently twice since:
#       git log --format=%H -Sz_zbin_max -- CDDF_analysis/hbi_mcmc/run_posterior.py | tail -1
#   * z_total_max and z_bin_max DO pre-date it (f23961e, 2026-07-28), but
#     pre-dating a decision is not being ratified by it.
#
# "PROVISIONAL" here means ONLY "declined by the PI on 2026-07-29".  It is NOT
# the complement of "ratified": FOUR MORE tolerances are unratified.
#
# THE SINGLE SOURCE OF TRUTH IS ``ratification.py``.  The names below are
# derived from it, not maintained in parallel -- a literal list of names typed
# next to the thing it describes is exactly how the fabricated claim survived.
PROVISIONAL_GATE_TOLERANCES = RAT.unratified_names()

PROVISIONAL_GATE_TOLERANCES_NOTE = RAT.UNRATIFIED_NOTE


def ratified_gate_tolerances():
    """The ratified items that are actually GATE TOLERANCES.

    🔴 NOT the same set as ``RAT.ratified_names()``. Decision 8 ratified THREE
    things and only ONE of them is a number in ``GATE``: ``chi2_dof_max``. The
    other two -- ``fail_closed_framework`` and ``matched_configuration_sbc`` --
    are ratified COMMITMENTS, not thresholds, and a field named
    ``gate_tolerances_ratified`` that listed them would overstate how much of
    the gate carries ratified authority, which is the family of error this whole
    correction exists to undo. Intersecting with ``GATE`` keeps the two
    vocabularies from being conflated again.
    """
    return [n for n in RAT.ratified_names() if n in GATE]

_REAL_TOKENS = ("main_dark", "loa_main_dark", "matterhorn", "dr3")


def forward_closure_gate(pack, *, resp_clamp="both", gate=None):
    """Fold the pack's own truth through its own kernel; PASS/FAIL + evidence.

    Returns a dict; ``["pass"]`` is the fail-closed verdict.  Requires no
    sampling (seconds), so it is cheap enough to run on every invocation and
    is run BEFORE the sampler is constructed.
    """
    from CDDF_analysis.hbi_mcmc import forward_selftest as FS
    from CDDF_analysis.hbi_mcmc import reporting as REP

    gate = dict(GATE if gate is None else gate)
    if getattr(pack, "truth_counts", None) is None:
        return {"pass": False, "reason": "pack carries no truth_counts; "
                "the forward model cannot be self-tested", "gate": gate}
    res = FS.selftest(pack, resp_clamp=resp_clamp)
    tab = FS.ratio_tables(res, pack)

    # ONE closure metric.  chi2/dof, max|z_bin| and the bin count come from
    # ``reporting.window_closure_metrics`` called UNRESTRICTED (no lo/hi), which
    # is by construction the same arithmetic this function used to inline.
    #
    # The removed ``b["lo"] >= nhat_edges[0] - 1e-9`` clause was PROVABLY INERT,
    # not merely inert in practice: ``ratio_tables`` builds one ``by_nhat`` row
    # per OBSERVED bin with ``lo = pack.nhat_edges[c]`` for c in range(C), so the
    # row set IS ``nhat_edges[:-1]`` and no row can ever sit below
    # ``nhat_edges[0]``.  (Measured on all 18 committed mock packs +
    # ``synthetic_pack``: rows dropped = 0.)  It read as a reporting-window
    # restriction and was not one; a real window restriction is the lo/hi
    # argument of ``window_closure_metrics`` and MOVES the number (measured:
    # lo=19.7 on the synthetic pack gives 0.8066535048313024 over 8 bins vs
    # 0.6479548471525799 over 10).
    m = REP.window_closure_metrics(tab["by_nhat"])
    rows = [b for b in tab["by_nhat"] if b["obs"] > 0]
    chi2_dof = m["chi2_dof"]
    z_bin_max = m["z_bin_max"]
    n_gate_bins = m["n_bins_in_z_set"]
    z_total = float(abs(tab["total"]["z"]))

    # 🔴 A refusal must name the AUTHORITY of the number it refused on.  Six of
    # the seven tolerances in GATE are not ratified; before 2026-08-05 only the
    # two DECLINED span numbers were labelled, so a reader of a refusal
    # reasonably concluded the |z| arms were authorised.  Read from the ONE
    # table, never typed here.
    # MERGE NOTE: this helper came from the adopted-basis stream reading
    # ``reporting.GATE_AUTHORITY``. That was a COMPLETE PARALLEL authority table
    # -- its own PI_RATIFIED_ITEMS, its own guard, its own stamp -- and two
    # tables sharing one vocabulary is exactly how the claim drifts back out of
    # sync. Repointed at the single source, ``ratification.py``. ``RAT.record``
    # fails closed: an unknown name returns status "UNKNOWN" with
    # contributes_to_pass_fail=False rather than raising or defaulting to
    # ratified.
    def _tag(name):
        st = RAT.record(name).get("status")
        if st == "RATIFIED":
            return ""
        if st == "UNRATIFIED":
            return " [UNRATIFIED tolerance -- the PI was asked and DECLINED]"
        if st == "RESTATED_NOT_RATIFIED":
            return (" [NOT RATIFIED -- this arm gates; no deciding authority "
                    "ratified it]")
        return " [NO GATE-AUTHORITY RECORD -- treat as unratified]"

    fails = []
    # ``advisories`` are computed, reported, and DO NOT contribute to
    # ``pass``.  They exist because decision 8 declined to ratify the two
    # ratio-span thresholds: the statistic is worth accumulating, the number
    # is not worth refusing work on.  See ratification.py.
    advisories = []
    if not (z_total <= gate["z_total_max"]):
        fails.append(f"|z_total|={z_total:.2f} > {gate['z_total_max']}"
                     + _tag("z_total_max"))
    if not (z_bin_max <= gate["z_bin_max"]):
        fails.append(f"max|z_bin|={z_bin_max:.2f} > {gate['z_bin_max']}"
                     + _tag("z_bin_max"))
    if not (chi2_dof <= gate["chi2_dof_max"]):
        fails.append(f"chi2/dof={chi2_dof:.2f} > {gate['chi2_dof_max']}"
                     + _tag("chi2_dof_max"))

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
            fails.append(f"max|z| in {key} = {zmax:.2f} > {gate[zkey]}"
                         + _tag(zkey))
        if not (span <= gate[spankey]):
            msg = (f"ratio span in {key} = {span:.4f} "
                   f"(mu/obs {sp['lo']:.4f}..{sp['hi']:.4f}, "
                   f"n_rows_used={sp['n_rows_used']}) "
                   f"> {spankey} = {gate[spankey]}")
            # ``gates()``, not ``is_ratified()``: authority and gating are
            # SEPARATE facts (see ratification.py, "THE THREE STATES").  The
            # four |z| arms gate WITHOUT being ratified, and conflating the two
            # is exactly the defect that fabricated a PI authority for them.
            if RAT.gates(spankey):
                # MERGE NOTE: _tag() comes from the adopted-basis stream -- every
                # refusal names the authority of the number it refused on. Kept
                # on this branch too, so a span arm that DOES gate cannot refuse
                # work anonymously.
                fails.append(msg + _tag(spankey))
            else:
                # UNRATIFIED (decision 8, and the PI direction of 2026-08-05):
                # report, do not gate.
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
        # 🔴 which of the numbers above a PI DECLINED.  This is NOT the
        # complement of "ratified" -- see the four |z| arms below.
        "gate_tolerances_provisional": list(PROVISIONAL_GATE_TOLERANCES),
        "gate_tolerances_provisional_note": PROVISIONAL_GATE_TOLERANCES_NOTE,
        "gate_tolerances_unratified": list(RAT.unratified_names()),
        "gate_tolerances_ratified":
            [n for n in RAT.ratified_names() if n in GATE],
        # MERGE NOTE: the adopted-basis stream emitted this same picture from
        # reporting.GATE_AUTHORITY. Both streams built a table with the same
        # vocabulary; two tables is how the claim drifts back out of sync, so
        # every key here is derived from the ONE source, ratification.py.
        # ``restated_not_ratified`` is the adopted-basis stream's key name, kept
        # because its tests and artifacts read it.
        "gate_tolerances_restated_not_ratified":
            list(RAT.restated_not_ratified_names()),
        # 🔴 the numbers that REFUSE WORK with no ratified authority.  A reader
        # of the JSON alone must be able to see this without opening the source.
        "gate_tolerances_unratified_but_gating":
            list(RAT.unratified_but_gating_names()),
        "gate_tolerances_unratified_but_gating_note":
            RAT.UNRATIFIED_BUT_GATING_NOTE,
        "unratified_effect": RAT.UNRATIFIED_EFFECT,
        "ratification": RAT.ratification_stamp(),
        "total_mu": float(tab["total"]["mu"]),
        "total_obs": float(tab["total"]["obs"]),
        "total_ratio": float(tab["total"]["ratio"]),
        "z_total": z_total,
        "z_bin_max": z_bin_max,
        "chi2_dof": chi2_dof,
        "n_bins": int(n_gate_bins),
        # the ONE routine the three numbers above came from (A1, 2026-08-05)
        "closure_metric_routine":
            "CDDF_analysis/hbi_mcmc/reporting.py:window_closure_metrics "
            "(called UNRESTRICTED — full observed grid)",
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
        # ratified without opening the nested report.  🔴 the PROVISIONAL list
        # alone is NOT that picture -- it names only the two the PI DECLINED,
        # and reading it as "everything else is ratified" is precisely the
        # error that put z_total_max and z_bin_max into a committed artifact
        # under `gate_tolerances_ratified`.  All four lists are read from
        # reporting.GATE_AUTHORITY.
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
        # MERGE NOTE: the adopted-basis stream hoisted these three lists into
        # every stamp so a reader of the JSON alone sees the authority picture
        # without opening the source. Kept, derived from the ONE source.
        "gate_tolerances_ratified":
            [n for n in RAT.ratified_names() if n in GATE],
        "gate_tolerances_restated_not_ratified":
            list(RAT.restated_not_ratified_names()),
        "gate_tolerances_unratified_but_gating":
            list(RAT.unratified_but_gating_names()),
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
        # SAME weights as the posterior reduction (PI decision 3: the latent
        # basis may be coarser than the window, so the truth must be integrated
        # with the identical overlap weights or the closure compares two
        # different estimands). Identical to the old centre-selection on any
        # 0.1-dex pack.
        from CDDF_analysis.hbi_mcmc import reporting as RP
        w = np.where(rep, RP.window_overlap_weights(ntrue, lo, hi), 0.0)
        dndx_k = (f_true * w[:, None]).sum(axis=0)
        om_k = (f_true * (10.0 ** (Nc - 21.0))[:, None] * w[:, None]).sum(axis=0)
        row = {}
        for name, tk in (("dndx_allz", dndx_k), ("omega_allz", om_k)):
            t = float((tk * dX_k).sum() / dX_k.sum())
            b = summ["tiers"][tier].get(name)
            if b is None:      # Omega REFUSED outside [19.7, 21.6] (decision 1)
                row[name] = {"truth": t, "REFUSED": summ["tiers"][tier].get(
                    "omega_REFUSED", {}).get("reason")}
                continue
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
