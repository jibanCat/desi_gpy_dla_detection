# -*- coding: utf-8 -*-
"""ratification.py -- THE single source of truth for which validation gates a
deciding authority has actually ratified, what an UNRATIFIED tolerance is
allowed to do, and -- the part this module got wrong once already -- which
numbers REFUSE WORK WITHOUT ANY RATIFIED AUTHORITY AT ALL.

WHY THIS MODULE EXISTS
----------------------
Tolerances inside a production fail-closed gate have twice been introduced by
whoever was writing the arm at the time, with a number chosen by eye, and have
then silently acquired the authority of a ratified criterion because they lived
in the same dict as the conventional ones.  On 2026-07-29 the PI ratified three
things and explicitly DECLINED to ratify two others, asking that they be
"defined and calibrated prospectively".

🔴 THIS MODULE THEN COMMITTED THE SAME OFFENCE IT WAS WRITTEN TO PREVENT, and
that is corrected here.  Its first version (``88f2ecb``) recorded all four
``|z| <= 5`` arms as ``status="RATIFIED"``, ``date="2026-07-29"``,
``authority="PI (project decision 8, 2026-07-29)"``,
``contributes_to_pass_fail=True``, on the stated grounds that "the conventional
5-sigma z-score arms pre-date decision 8 and were carried forward with it".
Both halves of that were false:

  * decision 8, verbatim, said: "Also restate the malformed |z| <= 5 criterion
    with its exact mathematical definition."  Calling a criterion MALFORMED AS
    STATED and sending it back for restatement is the OPPOSITE of ratifying it.
    The restatement was written (``forward_selftest.poisson_z``); the RESTATED
    form has not been put to the PI and has not been ratified.
  * "pre-date" was false for two of the four.  ``git log -S z_zbin_max --
    CDDF_analysis/hbi_mcmc/run_posterior.py`` reports ``0e7fa0b`` (2026-07-29
    10:21) as their introducing commit, and the diff shows ``z_zbin_max``,
    ``z_snrbin_max``, ``ratio_span_by_z_max`` and ``ratio_span_by_snr_max``
    added as FOUR CONSECUTIVE LINES OF ONE HUNK.  Two of those four the PI
    declined the same day; the other two were stamped RATIFIED-BY-PI.  They
    pre-date nothing.

So the ratification state is data, in one place, every field is checkable
against the repository, and a fabricated authority claim now RAISES at import
(``enforce_authority_allow_list``).  A grep for a tolerance name lands on its
ratification record, not on archaeology.

THE THREE STATES  (status is about AUTHORITY; gating is a SEPARATE field)
------------------------------------------------------------------------
The first version's fatal simplification was to treat "ratified" and "gates"
as one bit.  They are orthogonal, and conflating them is what let a gating
number acquire an authority it did not have:

``RATIFIED``               a deciding authority (the PI, decision 8) authorised
                           this criterion to refuse work.  EXACTLY THREE items
                           qualify -- see ``PI_RATIFIED_ITEMS``.
``UNRATIFIED``             the PI was asked and DECLINED.  The statistic is
                           COMPUTED and REPORTED on every run, is stamped into
                           the artifact, and DOES NOT contribute to pass/fail.
``RESTATED_NOT_RATIFIED``  the criterion DOES contribute to pass/fail but NO
                           deciding authority has ratified it.  It is inherited
                           from the pre-decision-8 gate (or, for two of the
                           four, was introduced alongside the declined pair).
                           This state is UNCOMFORTABLE ON PURPOSE: it is a
                           standing PI decision, recorded rather than resolved
                           by whoever is writing the arm.

``contributes_to_pass_fail`` is therefore reported per entry and is what
``gates()`` reads.  ``is_ratified()`` answers ONLY the authority question.

WHY THE FOUR |z| ARMS ARE LEFT ARMED
------------------------------------
Disarming them would be a unilateral decision of exactly the kind this module
exists to stop, and it would remove the only remaining guard against the
standing z-marginal tilt defect (the span arms having been disarmed under
decision 8).  Recording that they gate on unratified authority is honest;
silently disarming them would not be, and neither would silently calling them
ratified.  Both the choice and its cost are in ``OPEN_PI_DECISIONS``.

THE UNRATIFIED (DECLINED) POLICY
--------------------------------
The three candidate behaviours for a DECLINED tolerance were:

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
arm did not fire" for "the span arm passed".  🔴 The COST of (c) is not
rhetorical and is stated in ``OPEN_PI_DECISIONS['span_arms_disarmed']``: see
``docs/ratio_span_calibration_spec.md`` §4.1 for the measured comparison of
what the disarmed span arm and the still-armed ``z_zbin_max`` each detect at
production geometry.
"""
from __future__ import annotations

__all__ = [
    "RATIFIED", "UNRATIFIED", "RESTATED_NOT_RATIFIED",
    "RATIFICATION_DATE", "RATIFYING_AUTHORITY", "PI_AUTHORITY",
    "PI_RATIFIED_ITEMS", "FabricatedAuthorityError",
    "is_ratified", "gates", "record", "all_records",
    "unratified_names", "ratified_names", "restated_not_ratified_names",
    "unratified_but_gating_names", "ratification_stamp", "UNRATIFIED_EFFECT",
    "audit_authority_claims", "enforce_authority_allow_list",
    "OPEN_PI_DECISIONS",
]

RATIFICATION_DATE = "2026-07-29"
#: the ONE authority string a genuinely PI-ratified entry may carry
PI_AUTHORITY = "PI (project decision 8, 2026-07-29)"
#: back-compatible alias (the artifact stamps quote it)
RATIFYING_AUTHORITY = PI_AUTHORITY

#: 🔴 THE ALLOW-LIST.  Decision 8, verbatim: "Ratify the fail-closed
#: framework, matched-configuration SBC and chi2/dof <= 3 closure
#: requirement."  Three items.  No entry outside this tuple may claim ``PI``
#: as its authority; ``enforce_authority_allow_list`` raises if one does, and
#: it runs at import.  Growing this tuple requires a new PI decision quoted
#: verbatim next to it.
PI_RATIFIED_ITEMS = ("fail_closed_framework", "matched_configuration_sbc",
                     "chi2_dof_max")

#: what an unratified (DECLINED) tolerance is permitted to do
UNRATIFIED_EFFECT = "REPORT_ONLY_DOES_NOT_GATE"

#: what an unratified-but-GATING tolerance is, in one line
UNRATIFIED_BUT_GATING_EFFECT = "GATES_WITHOUT_RATIFIED_AUTHORITY"


class FabricatedAuthorityError(RuntimeError):
    """Raised when a ratification record claims an authority it does not have."""


def _r(statement, *, applies_to, date=RATIFICATION_DATE,
       authority=PI_AUTHORITY, note=""):
    return {"status": "RATIFIED", "statement": statement,
            "applies_to": list(applies_to), "date": date,
            "authority": authority, "note": note,
            "contributes_to_pass_fail": True}


def _u(statement, *, applies_to, note, spec):
    return {"status": "UNRATIFIED", "statement": statement,
            "applies_to": list(applies_to),
            "date_proposed": "2026-07-29",
            "declined_by": PI_AUTHORITY,
            "authority": "NONE -- the PI was asked and DECLINED",
            "effect": UNRATIFIED_EFFECT,
            "contributes_to_pass_fail": False,
            "note": note, "calibration_spec": spec}


def _s(statement, *, applies_to, introduced_by, introduced_date,
       predates_decision_8, introduced_same_hunk_as=(), note=""):
    """A criterion that GATES but that no deciding authority has ratified."""
    return {
        "status": "RESTATED_NOT_RATIFIED",
        "statement": statement,
        "applies_to": list(applies_to),
        "contributes_to_pass_fail": True,
        "effect": UNRATIFIED_BUT_GATING_EFFECT,
        # 🔴 NOT the PI.  Whoever wrote the arm.
        "authority": ("the author of the arm (gate stream), inherited by the "
                      "production gate; NO deciding authority has ratified it"),
        "pi_disposition": (
            "decision 8, item 3, verbatim: \"Also restate the malformed "
            "|z| <= 5 criterion with its exact mathematical definition.\" The "
            "PI called the criterion MALFORMED AS STATED and sent it back for "
            "restatement. That is NOT a ratification. The restatement has "
            "been written; the restated form has NOT been put to the PI. "
            "STATUS: RESTATED, NOT RATIFIED."),
        "restatement_lives_in": "CDDF_analysis.hbi_mcmc.forward_selftest.poisson_z",
        "introduced_by": introduced_by,
        "introduced_date": introduced_date,
        "predates_decision_8": bool(predates_decision_8),
        "introduced_same_hunk_as": list(introduced_same_hunk_as),
        "provenance_check": (
            "git log --format=%H -S<name> -- "
            "CDDF_analysis/hbi_mcmc/run_posterior.py | tail -1"),
        "note": note,
        "open_pi_decision": "z_arms_gate_unratified",
    }


# ---------------------------------------------------------------------------
# RATIFIED -- decision 8, 2026-07-29.  EXACTLY THREE.
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
             "CONSTRUCTIBILITY: a matched SBC for a PADDED-basis pack "
             "(n_pad_bins > 0, decisions 3/4) was IMPOSSIBLE until "
             "2026-07-29 -- `synthetic_pack` had no `ntrue_edges` parameter "
             "and hardcoded `ntrue_edges = nhat_edges.copy()`, while "
             "`grid.ntrue_edges` is a MATCH_KEY. It is constructible now "
             "(`synthetic_pack(..., ntrue_edges=...)`, threaded through "
             "`matched_sbc_kwargs`); what remains is COST, ~1600 CPU-h, which "
             "is a PI sign-off, not a capability gap."),

    "chi2_dof_max": _r(
        "Forward-model closure requires chi2/dof <= 3 over the reported n-hat "
        "bins with obs > 0, with chi2 = sum of squared Poisson score residuals "
        "(see forward_selftest.poisson_z) and dof = the number of such bins "
        "(the truth fold estimates no parameters).",
        applies_to=("CDDF_analysis.hbi_mcmc.run_posterior.GATE['chi2_dof_max']",
                    "CDDF_analysis.hbi_mcmc.forward_selftest.ratio_tables")),
}


# ---------------------------------------------------------------------------
# RESTATED_NOT_RATIFIED -- the four |z| <= 5 arms.  THEY GATE.  NOBODY
# RATIFIED THEM.  Both halves of that sentence are recorded.
# ---------------------------------------------------------------------------
_Z_STATEMENT = (
    "|z| <= 5 on {what}, where z is the Poisson score residual defined "
    "EXACTLY in CDDF_analysis.hbi_mcmc.forward_selftest.poisson_z. This is a "
    "tripwire against an order-of-magnitude-wrong forward model, NOT a "
    "goodness-of-fit test; it is NOT scale-free (see that docstring, 'WHY 5 "
    "IS NOT SCALE-FREE'). RESTATED per decision 8 item 3; NOT RATIFIED.")

#: full 40-char SHAs, verified by ``git log -S`` (see ``provenance_check``)
_SHA_PRE = "f23961ec1e2cf47748a5a1b660205966a8d793f0"        # 2026-07-28 16:55
_SHA_SAMEHUNK = "0e7fa0bd62d1f177126737fa32d1963e558b18d2"   # 2026-07-29 10:21

_DECLINED_PAIR = ("ratio_span_by_z_max", "ratio_span_by_snr_max")

RESTATED_NOT_RATIFIED = {}
for _name, _what, _sha, _date, _pre, _hunk, _n in (
        ("z_total_max", "the total predicted-vs-observed count",
         _SHA_PRE, "2026-07-28", True, (),
         "Inherited from the pre-decision-8 gate. Pre-dating decision 8 is "
         "NOT ratification and this record does not imply it."),
        ("z_bin_max", "the reported n-hat bins with obs > 0",
         _SHA_PRE, "2026-07-28", True, (),
         "Inherited from the pre-decision-8 gate. Pre-dating decision 8 is "
         "NOT ratification and this record does not imply it."),
        ("z_zbin_max", "the fine-z marginal bins with obs > 0",
         _SHA_SAMEHUNK, "2026-07-29", False, _DECLINED_PAIR,
         "🔴 Introduced 2026-07-29 10:21, by the same author, in the SAME "
         "HUNK as the two ratio_span numbers the PI declined the same day "
         "(four consecutive added lines of one hunk of 0e7fa0b). It PRE-DATES "
         "NOTHING. An earlier version of this file claimed it was a "
         "conventional arm pre-dating decision 8 and stamped it "
         "authority=PI; that claim was fabricated."),
        ("z_snrbin_max", "the SNR-stratum marginals with obs > 0",
         _SHA_SAMEHUNK, "2026-07-29", False, _DECLINED_PAIR,
         "🔴 Same hunk, same commit, same day as the declined pair -- see "
         "z_zbin_max. Additionally: on a single-SNR-stratum grid this arm has "
         "never been able to fire (one row), so its apparent 'passes' are "
         "vacuous; see docs/ratio_span_calibration_spec.md §1.1 item 4."),
):
    RESTATED_NOT_RATIFIED[_name] = _s(
        _Z_STATEMENT.format(what=_what),
        applies_to=(f"CDDF_analysis.hbi_mcmc.run_posterior.GATE['{_name}']",
                    "CDDF_analysis.hbi_mcmc.forward_selftest.poisson_z"),
        introduced_by=_sha, introduced_date=_date,
        predates_decision_8=_pre, introduced_same_hunk_as=_hunk, note=_n)
del _name, _what, _sha, _date, _pre, _hunk, _n


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
             "and required prospective calibration. 🔴 ITS MEASURED "
             "FALSE-ALARM RATE IS GRID-DEPENDENT AND MUST NEVER BE QUOTED "
             "WITHOUT ITS GRID: 0.3434 on the 5x4x2 calibration pack (4 z "
             "rows), but 0.0893 on a 17x15x8 and 0.0819 on the 29x15x8 "
             "production geometry (15 z rows) -- all three at n_draws=20000, "
             "seed=1, which is the only reason those digits are quotable. "
             "The AUTHORITATIVE copy of every one of these numbers is "
             "CDDF_analysis/hbi_mcmc/ratio_span_null_calibration.json; the "
             "prose here is a pointer and drifts if it is trusted. See "
             "docs/ratio_span_calibration_spec.md §4 and §4.1.",
        spec=_SPEC),
    "ratio_span_by_snr_max": _u(
        "PROPOSED (NOT ratified): ratio_span_by_snr <= 0.15.",
        applies_to=("CDDF_analysis.hbi_mcmc.run_posterior.GATE"
                    "['ratio_span_by_snr_max']",),
        note="Same provenance as ratio_span_by_z_max; the wider value only "
             "reflects that the SNR marginal has fewer, noisier strata, which "
             "is an argument about the NULL DISTRIBUTION and is exactly what "
             "the prospective calibration is for. Measured false-alarm rate "
             "0.0003 on the 5x4x2 pack and 0.0000 on both production "
             "geometries: on every grid measured so far this arm is inert, "
             "which is the mismatch with its by_z partner that the PI's "
             "refusal was right about.",
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

UNRATIFIED_BUT_GATING_NOTE = (
    "UNRATIFIED BUT GATING: {names}. These numbers DO contribute to pass/fail "
    "-- they call fails.append in forward_closure_gate -- and no deciding "
    "authority has ratified them. Two of them (z_zbin_max, z_snrbin_max) were "
    "introduced on 2026-07-29 in the SAME HUNK as the two tolerances the PI "
    "declined that day; the other two are inherited from the pre-decision-8 "
    "gate, which is not the same thing as ratified. They are left ARMED "
    "deliberately (disarming would be the same unilateral act in the other "
    "direction, and would leave the z-marginal tilt defect unguarded once the "
    "span arms were disarmed). This is a STANDING PI DECISION, recorded here "
    "rather than resolved: see OPEN_PI_DECISIONS."
).format(names=", ".join(sorted(RESTATED_NOT_RATIFIED)))


# ---------------------------------------------------------------------------
# OPEN PI DECISIONS -- what this stream refused to resolve for itself
# ---------------------------------------------------------------------------
OPEN_PI_DECISIONS = {
    "z_arms_gate_unratified": {
        "question": (
            "Four |z| <= 5 arms refuse work with no ratified authority. "
            "RATIFY the restated criterion (the restatement is in "
            "forward_selftest.poisson_z), DISARM the two 2026-07-29 arms "
            "(z_zbin_max, z_snrbin_max) pending calibration, or leave as is "
            "and carry it as a stated limitation?"),
        "what_the_code_does_meanwhile": "all four remain ARMED",
        "why_not_resolved_here": (
            "choosing would be the same unilateral act that produced the "
            "defect; and disarming z_zbin_max at the same time as the span "
            "arms would leave the standing z-marginal tilt defect with no "
            "guard at all"),
    },
    "span_arms_disarmed": {
        "question": (
            "The two ratio_span arms were ARMED before this stream (a9fe97b) "
            "and are now report-only. That follows the PI's instruction that "
            "an unratified tolerance must not gate, but it is the only guard "
            "that fires on a PHYSICALLY large, statistically quiet z-tilt. "
            "Accept report-only, or arm a calibrated pack-specific threshold "
            "(spec §3/§6 option A)?"),
        "what_the_code_does_meanwhile": (
            "both span arms REPORT ONLY; z_zbin_max stays armed"),
        "measured_tradeoff": (
            "the 0.10 threshold's 34% false-alarm rate was measured on the "
            "5x4x2 calibration pack (4 z rows) ONLY. At production geometry "
            "it is 0.0893 (17x15x8) / 0.0819 (29x15x8). AND THE COST OF "
            "DISARMING IS NOW MEASURED, not asserted: exposed to the same "
            "injected peak-to-peak z-tilt d on the 17x15x8 grid, the span arm "
            "reaches 90% detection at d = 0.098 and the still-armed "
            "z_zbin_max only at d = 0.197. Disarming the span arms therefore "
            "roughly DOUBLES the smallest z-tilt that any armed gate catches; "
            "z_zbin_max does NOT cover what the span arm covered. A "
            "threshold calibrated on the production null instead (0.1292, "
            "measured false-alarm rate 0.0073) reaches 90% at d = 0.142, "
            "i.e. option A recovers most of the lost power at a defensible "
            "false-alarm rate -- which is what makes option A a measured "
            "option rather than a suggestion. Curves and full table: "
            "docs/ratio_span_calibration_spec.md §4.1 and the `power` block "
            "of ratio_span_null_calibration.json (n_draws=20000/4000, "
            "seed=1). THIS DOES NOT DECIDE THE QUESTION: the null is a lower "
            "bound on the true null width (spec §2.1), so every false-alarm "
            "rate here is optimistic, and all of it is synthetic."),
    },
}

OPEN_PI_DECISIONS_NOTE = (
    "PI DECISION REQUIRED, {n} open: {names}. This artifact does not resolve "
    "them and does not claim they are resolved. Details: "
    "CDDF_analysis.hbi_mcmc.ratification.OPEN_PI_DECISIONS."
).format(n=len(OPEN_PI_DECISIONS), names=", ".join(sorted(OPEN_PI_DECISIONS)))


# ---------------------------------------------------------------------------
# THE AUTHORITY GUARD
# ---------------------------------------------------------------------------

def all_records():
    """Every ratification record, keyed by tolerance/criterion name.

    The three dicts are disjoint as shipped (pinned by a test).  Precedence is
    fixed anyway, RATIFIED LAST and therefore WINNING, so that a name promoted
    to RATIFIED cannot be shadowed by a stale weaker entry -- ``record()`` in
    schema v1 checked RATIFIED first and that behaviour is preserved.
    """
    out = {}
    out.update(UNRATIFIED)
    out.update(RESTATED_NOT_RATIFIED)
    out.update(RATIFIED)
    return out


def audit_authority_claims(records=None):
    """Names that claim ``PI`` authority without being on ``PI_RATIFIED_ITEMS``.

    Returns a list of human-readable violation strings; ``[]`` means clean.
    An entry may only name the PI in ``authority`` if it is allow-listed; the
    ``declined_by`` field is exempt (declining is not authorising), as is the
    literal "NONE -- the PI was asked and DECLINED" non-claim.
    """
    recs = all_records() if records is None else records
    bad = []
    for name, rec in sorted(recs.items()):
        if not isinstance(rec, dict):
            bad.append(f"{name}: ratification record is not a dict")
            continue
        auth = str(rec.get("authority") or "")
        claims_pi = ("PI" in auth) and ("DECLINED" not in auth.upper())
        if claims_pi and name not in PI_RATIFIED_ITEMS:
            bad.append(
                f"{name}: claims authority={auth!r} but is NOT on "
                f"PI_RATIFIED_ITEMS={PI_RATIFIED_ITEMS}. Decision 8 ratified "
                f"exactly three things; everything else must record its real "
                f"provenance.")
        if rec.get("status") == "RATIFIED" and name not in PI_RATIFIED_ITEMS:
            bad.append(
                f"{name}: status=RATIFIED but is not on PI_RATIFIED_ITEMS")
    return bad


def enforce_authority_allow_list(records=None):
    """Raise ``FabricatedAuthorityError`` on any off-allow-list PI claim.

    Called at import, so the module cannot be loaded in the state the
    2026-07-29 defect left it in.
    """
    bad = audit_authority_claims(records)
    if bad:
        raise FabricatedAuthorityError(
            "fabricated ratification authority:\n  " + "\n  ".join(bad))
    return True


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------

def ratified_names():
    """The names a deciding authority ratified.  EXACTLY three."""
    return tuple(sorted(RATIFIED))


def unratified_names():
    """The names the PI was asked about and DECLINED (report-only)."""
    return tuple(sorted(UNRATIFIED))


def restated_not_ratified_names():
    """The names that GATE with no ratified authority."""
    return tuple(sorted(RESTATED_NOT_RATIFIED))


def unratified_but_gating_names():
    """Every name that contributes to pass/fail without being ratified.

    Derived from the records, not maintained in parallel, so a future entry
    cannot escape it.
    """
    return tuple(sorted(
        n for n, r in all_records().items()
        if r.get("contributes_to_pass_fail") and n not in RATIFIED))


def is_ratified(name):
    """True only if ``name`` has an explicit RATIFIED record.

    This answers the AUTHORITY question and nothing else.  It is NOT "does
    this gate" -- use ``gates()`` for that.  Unknown names are NOT ratified:
    a tolerance somebody adds tomorrow without a record here does not inherit
    authority from its neighbours in ``GATE``.
    """
    return name in RATIFIED


def gates(name):
    """Does ``name`` contribute to pass/fail?

    Reads ``contributes_to_pass_fail`` off the record.  An UNKNOWN name
    returns False: a number with no record must not refuse work, which is the
    direction decision 8 ordered for uncalibrated tolerances.  (Note the
    asymmetry, stated so it is not a surprise: for a required evidence BLOCK,
    absence FAILS CLOSED; for a TOLERANCE, absence must not refuse.)
    """
    return bool(record(name).get("contributes_to_pass_fail", False))


def record(name):
    """The ratification record for ``name``, or an explicit UNKNOWN record."""
    recs = all_records()
    if name in recs:
        return dict(recs[name])
    return {"status": "UNKNOWN", "contributes_to_pass_fail": False,
            "authority": "NONE -- no ratification record",
            "note": f"{name!r} has no ratification record; treated as "
                    f"UNRATIFIED (report-only)."}


def ratification_stamp():
    """The block every artifact carries, verbatim.

    Small on purpose: a reader of the JSON alone must be able to see which
    criteria were authorised to refuse work, WHICH REFUSE WORK WITHOUT BEING
    AUTHORISED, and which are report-only -- without opening the source.
    """
    return {
        "schema": "gate_ratification/v2",
        "ratification_date": RATIFICATION_DATE,
        "authority": PI_AUTHORITY,
        # 🔴 v1 put this key at the top of the stamp with NOTHING saying what it
        # covered, and a reader (including the author of v1) took it to
        # authorise the whole block.  That is the mechanism by which the |z|
        # arms acquired PI authority.  The scope is now stated IN the stamp, so
        # the JSON alone cannot be misread the same way.
        "authority_scope": (
            "The `authority` field above applies to `pi_ratified_items` AND TO "
            "NOTHING ELSE IN THIS STAMP. Entries under `restated_not_ratified` "
            "and `unratified` are NOT covered by it: each carries its own "
            "`authority` field naming who actually set it, and for all of them "
            "that is not the PI. Read those fields, not this one."),
        "pi_ratified_items": list(PI_RATIFIED_ITEMS),
        "ratified": {k: {"status": "RATIFIED",
                         "statement": v["statement"],
                         "date": v["date"], "authority": v["authority"]}
                     for k, v in sorted(RATIFIED.items())},
        "restated_not_ratified": {
            k: {"status": v["status"],
                "statement": v["statement"],
                "authority": v["authority"],
                "pi_disposition": v["pi_disposition"],
                "contributes_to_pass_fail": v["contributes_to_pass_fail"],
                "introduced_by": v["introduced_by"],
                "introduced_date": v["introduced_date"],
                "predates_decision_8": v["predates_decision_8"],
                "introduced_same_hunk_as": v["introduced_same_hunk_as"]}
            for k, v in sorted(RESTATED_NOT_RATIFIED.items())},
        "unratified": {k: {"status": "UNRATIFIED",
                           "statement": v["statement"],
                           "effect": v["effect"],
                           "declined_by": v["declined_by"],
                           "calibration_spec": v["calibration_spec"]}
                       for k, v in sorted(UNRATIFIED.items())},
        "unratified_effect": UNRATIFIED_EFFECT,
        "unratified_note": UNRATIFIED_NOTE,
        "unratified_but_gating": list(unratified_but_gating_names()),
        "unratified_but_gating_effect": UNRATIFIED_BUT_GATING_EFFECT,
        "unratified_but_gating_note": UNRATIFIED_BUT_GATING_NOTE,
        "open_pi_decisions": {k: dict(v)
                              for k, v in sorted(OPEN_PI_DECISIONS.items())},
        "open_pi_decisions_note": OPEN_PI_DECISIONS_NOTE,
        "authority_allow_list_clean": (audit_authority_claims() == []),
        "correction_note": (
            "schema v2 CORRECTS v1 (88f2ecb), which recorded the four "
            "|z| <= 5 arms as RATIFIED by the PI on grounds that were false: "
            "decision 8 called |z| <= 5 MALFORMED AS STATED and asked for a "
            "restatement, and two of the four arms were introduced the same "
            "day, in the same hunk, as the two tolerances the PI declined. "
            "Any artifact carrying gate_ratification/v1 overstates its "
            "authority."),
    }


# 🔴 fail at IMPORT, not at review time.
enforce_authority_allow_list()
