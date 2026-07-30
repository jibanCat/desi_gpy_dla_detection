# -*- coding: utf-8 -*-
"""ratification.py -- THE single source of truth for which validation gates a
deciding authority has actually ratified, and what an UNRATIFIED tolerance is
allowed to do.

WHY THIS MODULE EXISTS
----------------------
Tolerances inside a production fail-closed gate have twice been introduced by
whoever was writing the arm at the time, with a number chosen by eye, and have
then silently acquired the authority of a ratified criterion because they lived
in the same dict as the conventional ones.  On 2026-07-29 the PI ratified three
things and explicitly DECLINED to ratify two others, asking that they be
"defined and calibrated prospectively".

So the ratification state is data, in one place, and every gate and every
artifact stamp reads it from here.  A grep for a tolerance name lands on its
ratification record, not on archaeology.

THE TWO STATES
--------------
``RATIFIED``    the criterion is load-bearing: it CONTRIBUTES TO PASS/FAIL and
                refusing work on it is authorised.
``UNRATIFIED``  the statistic is COMPUTED and REPORTED on every run, is stamped
                into the artifact, and DOES NOT contribute to pass/fail.

The unratified policy is deliberate and is stated here rather than in the gate
body.  The three candidate behaviours were:

  (a) keep the invented threshold armed  -- rejected: it refuses work on a
      number with no null distribution and no false-alarm rate.  That is
      precisely what the PI declined.
  (b) delete the arm                     -- rejected: the statistic is the
      calibration data.  Deleting it means the prospective calibration can
      never be run, because nothing accumulates the numbers.
  (c) compute, report, do not gate       -- ADOPTED.  Every run contributes a
      sample of the statistic under whatever configuration it ran; the
      artifact records the value and records, explicitly, that no threshold
      was applied.  When the null distribution has been sampled (see
      ``docs/ratio_span_calibration_spec.md``) a threshold can be set from a
      stated false-alarm rate and the entry moves to ``RATIFIED`` here, in one
      edit, with no change to the arm.

An UNRATIFIED entry is NOT a weaker gate.  It is an explicitly absent gate,
and the artifact says so, so that no downstream reader can mistake "the span
arm did not fire" for "the span arm passed".
"""
from __future__ import annotations

__all__ = [
    "RATIFIED", "UNRATIFIED", "RATIFICATION_DATE", "RATIFYING_AUTHORITY",
    "is_ratified", "record", "unratified_names", "ratified_names",
    "ratification_stamp", "UNRATIFIED_EFFECT",
]

RATIFICATION_DATE = "2026-07-29"
RATIFYING_AUTHORITY = "PI (project decision 8, 2026-07-29)"

#: what an unratified tolerance is permitted to do
UNRATIFIED_EFFECT = "REPORT_ONLY_DOES_NOT_GATE"


def _r(statement, *, applies_to, date=RATIFICATION_DATE,
       authority=RATIFYING_AUTHORITY, note=""):
    return {"status": "RATIFIED", "statement": statement,
            "applies_to": list(applies_to), "date": date,
            "authority": authority, "note": note,
            "contributes_to_pass_fail": True}


def _u(statement, *, applies_to, note, spec):
    return {"status": "UNRATIFIED", "statement": statement,
            "applies_to": list(applies_to),
            "date_proposed": "2026-07-29",
            "declined_by": RATIFYING_AUTHORITY,
            "effect": UNRATIFIED_EFFECT,
            "contributes_to_pass_fail": False,
            "note": note, "calibration_spec": spec}


# ---------------------------------------------------------------------------
# RATIFIED -- decision 8, 2026-07-29
# ---------------------------------------------------------------------------
RATIFIED = {
    "fail_closed_framework": _r(
        "The evidence gate is FAIL-CLOSED: a missing required block is a "
        "FAILURE, a block that reports no checks is a FAILURE, a malformed "
        "block/check/incomplete field is a FAILURE, a gate bypass in force is "
        "a FAILURE for paper-facing purposes, and `required` may only ever "
        "GROW. There is no code path by which not running a check produces a "
        "stamp.",
        applies_to=("CDDF_analysis.hbi_mcmc.evidence.gate",
                    "CDDF_analysis.hbi_mcmc.evidence.assemble_evidence",
                    "CDDF_analysis.hbi_mcmc.run_posterior.forward_closure_gate")),

    "matched_configuration_sbc": _r(
        "Simulation-based calibration may certify ONLY the configuration it "
        "actually ran. An SBC whose grid, prior, FP mode, response clamp or "
        "reported-quantity set differs from the run it is attached to does "
        "NOT certify that run, and an artifact carrying an unmatched (or "
        "unspecified) SBC is NOT STAMPABLE.",
        applies_to=("CDDF_analysis.hbi_mcmc.sbc.sbc_block",
                    "CDDF_analysis.hbi_mcmc.sbc.configuration_match",
                    "CDDF_analysis.hbi_mcmc.evidence.gate"),
        note="The pre-existing SBC runs a reduced grid, a NARROWED prior and "
             "the FP block OFF (reductions R1/R2/R3/R4 in sbc.py). Under this "
             "ratification such an SBC is still reportable and still "
             "diagnostic, but it can no longer certify a production run. "
             "Making a matched SBC CHEAP is out of scope; making an UNMATCHED "
             "one REFUSE is what is implemented."),

    "chi2_dof_max": _r(
        "Forward-model closure requires chi2/dof <= 3 over the reported n-hat "
        "bins with obs > 0, with chi2 = sum of squared Poisson score residuals "
        "(see forward_selftest.poisson_z) and dof = the number of such bins "
        "(the truth fold estimates no parameters).",
        applies_to=("CDDF_analysis.hbi_mcmc.run_posterior.GATE['chi2_dof_max']",
                    "CDDF_analysis.hbi_mcmc.forward_selftest.ratio_tables")),
}

# The conventional 5-sigma z-score arms pre-date decision 8 and were carried
# forward with it; they are recorded as ratified so that no tolerance in GATE
# is unaccounted for.  Their EXACT definition is decision 8 item 3 and lives in
# ``forward_selftest.poisson_z``.
for _name, _what in (
        ("z_total_max", "the total predicted-vs-observed count"),
        ("z_bin_max", "the reported n-hat bins with obs > 0"),
        ("z_zbin_max", "the fine-z marginal bins with obs > 0"),
        ("z_snrbin_max", "the SNR-stratum marginals with obs > 0")):
    RATIFIED[_name] = _r(
        f"|z| <= 5 on {_what}, where z is the Poisson score residual defined "
        f"EXACTLY in CDDF_analysis.hbi_mcmc.forward_selftest.poisson_z. This "
        f"is a tripwire against an order-of-magnitude-wrong forward model, "
        f"NOT a goodness-of-fit test; it is NOT scale-free (see that "
        f"docstring, 'WHY 5 IS NOT SCALE-FREE').",
        applies_to=(f"CDDF_analysis.hbi_mcmc.run_posterior.GATE['{_name}']",
                    "CDDF_analysis.hbi_mcmc.forward_selftest.poisson_z"))
del _name, _what


# ---------------------------------------------------------------------------
# UNRATIFIED -- decision 8 explicitly declined these two
# ---------------------------------------------------------------------------
_SPEC = "docs/ratio_span_calibration_spec.md"

UNRATIFIED = {
    "ratio_span_by_z_max": _u(
        "PROPOSED (NOT ratified): ratio_span_by_z <= 0.10.",
        applies_to=("CDDF_analysis.hbi_mcmc.run_posterior.GATE"
                    "['ratio_span_by_z_max']",),
        note="0.10 was chosen by eye by the author of the by_z arm on "
             "2026-07-29 as 'a swing a sampler cannot repair'. It is not "
             "measured, not calibrated against any reference forward model, "
             "and carries no false-alarm rate. The PI declined to ratify it "
             "and required prospective calibration.",
        spec=_SPEC),
    "ratio_span_by_snr_max": _u(
        "PROPOSED (NOT ratified): ratio_span_by_snr <= 0.15.",
        applies_to=("CDDF_analysis.hbi_mcmc.run_posterior.GATE"
                    "['ratio_span_by_snr_max']",),
        note="Same provenance as ratio_span_by_z_max; the wider value only "
             "reflects that the SNR marginal has fewer, noisier strata, which "
             "is an argument about the NULL DISTRIBUTION and is exactly what "
             "the prospective calibration is for.",
        spec=_SPEC),
}

UNRATIFIED_NOTE = (
    "UNRATIFIED TOLERANCES: {names}. Their statistics ARE computed and ARE "
    "reported on every run; they DO NOT contribute to pass/fail "
    "(effect={effect}). A reported value that exceeds the proposed number is "
    "an ADVISORY, never a refusal. Do NOT read 'the span arm did not fire' as "
    "'the span arm passed': no threshold was applied. Prospective calibration "
    "spec: {spec}."
).format(names=", ".join(sorted(UNRATIFIED)), effect=UNRATIFIED_EFFECT,
         spec=_SPEC)


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------

def ratified_names():
    return tuple(sorted(RATIFIED))


def unratified_names():
    return tuple(sorted(UNRATIFIED))


def is_ratified(name):
    """True only if ``name`` has an explicit RATIFIED record.

    Unknown names are NOT ratified.  This is the fail-closed direction: a
    tolerance somebody adds tomorrow without a record here does not inherit
    authority from its neighbours in ``GATE``.
    """
    return name in RATIFIED


def record(name):
    """The ratification record for ``name``, or an explicit UNKNOWN record."""
    if name in RATIFIED:
        return dict(RATIFIED[name])
    if name in UNRATIFIED:
        return dict(UNRATIFIED[name])
    return {"status": "UNKNOWN", "contributes_to_pass_fail": False,
            "note": f"{name!r} has no ratification record; treated as "
                    f"UNRATIFIED (report-only)."}


def ratification_stamp():
    """The block every artifact carries, verbatim.

    Small on purpose: a reader of the JSON alone must be able to see which
    criteria were authorised to refuse work, without opening the source.
    """
    return {
        "schema": "gate_ratification/v1",
        "ratification_date": RATIFICATION_DATE,
        "authority": RATIFYING_AUTHORITY,
        "ratified": {k: {"status": "RATIFIED",
                         "statement": v["statement"],
                         "date": v["date"], "authority": v["authority"]}
                     for k, v in sorted(RATIFIED.items())},
        "unratified": {k: {"status": "UNRATIFIED",
                           "statement": v["statement"],
                           "effect": v["effect"],
                           "declined_by": v["declined_by"],
                           "calibration_spec": v["calibration_spec"]}
                       for k, v in sorted(UNRATIFIED.items())},
        "unratified_effect": UNRATIFIED_EFFECT,
        "unratified_note": UNRATIFIED_NOTE,
    }
