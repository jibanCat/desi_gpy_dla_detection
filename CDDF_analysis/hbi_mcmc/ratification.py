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
rhetorical and is stated in ``pi_decision('span_arms_disarmed')``: see
``docs/ratio_span_calibration_spec.md`` §4.1 for the measured comparison of
what the disarmed span arm and the still-armed ``z_zbin_max`` each detect at
production geometry.  (That entry was OPEN when this paragraph was written
and is RESOLVED as of 2026-08-05; ``pi_decision`` reads both views, which is
why the pointer here is to it and not to ``OPEN_PI_DECISIONS``.)

WHY THE IMPORT-TIME GUARD WAS NOT ENOUGH  (2026-08-05)
------------------------------------------------------
``enforce_authority_allow_list`` polices ONE dict in ONE module on ONE branch.
It was measured to be too narrow, not suspected of it: at the time this
paragraph was written, TWO fabricated-authority sites were live on sibling
branches that this module does not exist on, and BOTH used field names the
import-time guard has never looked at.

  * ``adopted_config.py`` writes ``gate_tolerances_ratified=["z_total_max",
    "z_bin_max", "chi2_dof_max"]``, and the same list appears in
    ``adopted_config_closure.json`` at ``/verdict/gate_tolerances_ratified``;
    two of those three names are NOT RATIFIED.
  * ``window_study.py`` writes ``ratified_arms={"abs_z_total_max": 5.0,
    "z_bin_max": 5.0, "chi2_dof_max": 3.0}``, repeats it as
    ``metadata.gate.ratified_arms`` in ``spectral_window_study.json``, and
    states it in prose as "THE THREE RATIFIED ARMS (PI decision 8) are ...";
    two of those three names are NOT RATIFIED.

Neither is a ``record``, so ``audit_authority_claims`` cannot see either; and
``ratification.py`` was not on those branches, so nothing ran at all.  The
guard is therefore widened along three axes at once:

  1. it scans ANY key that ASSERTS ratification (``/ratifi/`` minus the
     negated and subject-naming forms -- see ``classify_key``), not the two
     field names it happened to know;
  2. it scans ARTIFACTS (JSON) and CODE (Python, by AST) and PROSE, because
     both live sites appear in all three forms;
  3. it is RUNNABLE OVER A TREE, not only at import of this one module:
     ``python -m CDDF_analysis.hbi_mcmc.ratification --check <paths>``
     exits non-zero on a fabricated claim, so a merge can run it over branches
     that do not contain this file.

It FAILS CLOSED.  An unreadable path, an unparseable file, a ratification
claim whose shape is not recognised, a claim whose subject cannot be resolved,
and a scan that inspected ZERO files are all FAILURES, not passes.

PI DIRECTION OF 2026-08-05 (verbatim, recorded in ``PI_DIRECTIONS``)
--------------------------------------------------------------------
"The only currently ratified numerical closure gate is chi2/dof <= 3.  Any
z-based or span-based threshold must be: precisely defined; calibrated under
production geometry; tested for false-alarm behavior; proposed prospectively
at a PI checkpoint."  And: "Keep span-by-z and span-by-SNR active as advisory
diagnostics, not ratified hard gates."  And: "Do not write artifacts or code
claiming PI ratification where none exists."

That RESOLVES ``pi_decision('span_arms_disarmed')`` (option "accept
report-only", now with the word "advisory" from the PI) and does NOT resolve
``OPEN_PI_DECISIONS['z_arms_gate_unratified']``: the PI RELABELLED the |z|
arms (they must be defined, calibrated and proposed prospectively before they
can be ratified) and did not disarm them.  They still gate.  See
``RESOLVED_PI_DECISIONS``.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import sys

__all__ = [
    "RATIFIED", "UNRATIFIED", "RESTATED_NOT_RATIFIED",
    "RATIFICATION_DATE", "RATIFYING_AUTHORITY", "PI_AUTHORITY",
    "PI_RATIFIED_ITEMS", "FabricatedAuthorityError",
    "is_ratified", "gates", "record", "all_records",
    "unratified_names", "ratified_names", "restated_not_ratified_names",
    "unratified_but_gating_names", "ratification_stamp", "UNRATIFIED_EFFECT",
    "audit_authority_claims", "enforce_authority_allow_list",
    "OPEN_PI_DECISIONS", "RESOLVED_PI_DECISIONS", "PI_DECISIONS",
    "PI_DIRECTIONS", "pi_decision", "PROSPECTIVE_THRESHOLD_CONDITIONS",
    "REQUIRED_STAMP_KEYS", "IncompleteStampError", "SELF_SCAN_MIN_CLAIMS",
    # the widened, tree-runnable guard
    "SCAN_SCHEMA", "SCAN_RULES", "SCAN_SUFFIXES", "classify_key",
    "claim_is_name_bearing", "format_violation",
    "scan_data", "scan_json_text", "scan_python_source", "scan_markdown",
    "scan_file", "scan_paths", "ScanResult", "enforce_no_fabricated_claims",
    "enforce_no_fabricated_claims_data", "main",
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
         "🔴 Introduced 2026-07-29 10:21 in the SAME HUNK as the two "
         "ratio_span numbers the PI declined the same day (four consecutive "
         "added lines, 0e7fa0b, run_posterior.py). It PRE-DATES NOTHING. An "
         "earlier version of this file claimed it was a conventional arm "
         "pre-dating decision 8 and stamped it authority=PI; that claim was "
         "fabricated. 🔴 SECOND CORRECTION (2026-07-30): this note, and the "
         "message of the retracting commit, also said 'by the same author'. "
         "MEASURED, git: 0e7fa0b's author is 'panel5' and 88f2ecb's -- the "
         "commit that stamped the false RATIFIED record -- is 'jibanmich'. "
         "They are NOT the same git author. The claim was not checked before "
         "it was written, which is the same failure as the original "
         "fabrication in miniature, so it is withdrawn rather than reworded. "
         "The SAME-HUNK fact is what matters and it is the one that is "
         "verifiable: `git show 0e7fa0b -- "
         "CDDF_analysis/hbi_mcmc/run_posterior.py`. (The retracting commit "
         "message cannot be amended -- no history rewrite on this branch -- so "
         "this record is the correction of record.)"),
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

#: MERGE NOTE (adopted-basis x gate, 2026-08-05). The last three sentences come
#: from the adopted-basis stream, which found that the DANGEROUS reading of this
#: note is not "the span arm passed" but the COMPLEMENT INFERENCE: a reader sees
#: a short list of unratified numbers and concludes everything else in GATE is
#: ratified. That inference is exactly how four |z| arms ended up in a committed
#: artifact as PI-ratified. Forbidding it in the note itself is cheap and the
#: sentence is pinned by a test, so keep it here rather than in a second table.
UNRATIFIED_NOTE = (
    "UNRATIFIED TOLERANCES: {names}. Their statistics ARE computed and ARE "
    "reported on every run; they DO NOT contribute to pass/fail "
    "(effect={effect}). A reported value that exceeds the proposed number is "
    "an ADVISORY, never a refusal. Do NOT read 'the span arm did not fire' as "
    "'the span arm passed': no threshold was applied. Prospective calibration "
    "spec: {spec}. "
    "\U0001F534 DO NOT READ THIS LIST AS 'everything else is ratified'. "
    "Exactly ONE tolerance in GATE is ratified (chi2_dof_max, chi2/dof <= 3, "
    "PI decision 8). The four |z| arms ({restated}) are RESTATED_NOT_RATIFIED: "
    "they refuse work and no deciding authority ratified them -- decision 8 "
    "called |z| <= 5 MALFORMED and sent it back for restatement."
).format(names=", ".join(sorted(UNRATIFIED)), effect=UNRATIFIED_EFFECT,
         spec=_SPEC, restated=", ".join(sorted(RESTATED_NOT_RATIFIED)))

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
# PI DIRECTIONS -- quoted, dated, sourced.  NOT a ratification list.
# ---------------------------------------------------------------------------
#: 🔴 A direction is not a ratification.  ``PI_RATIFIED_ITEMS`` is unchanged by
#: everything below and the 2026-08-05 direction CONFIRMS it: of the numerical
#: closure gates, exactly ``chi2_dof_max`` is ratified.
PI_DIRECTIONS = {
    "2026-08-05": {
        "date": "2026-08-05",
        "source": "PI direction, quoted verbatim in the task brief that "
                  "produced this commit",
        "quotes": [
            "The only currently ratified numerical closure gate is "
            "chi2/dof <= 3. Any z-based or span-based threshold must be: "
            "precisely defined; calibrated under production geometry; tested "
            "for false-alarm behavior; proposed prospectively at a PI "
            "checkpoint.",
            "Keep span-by-z and span-by-SNR active as advisory diagnostics, "
            "not ratified hard gates.",
            "Do not write artifacts or code claiming PI ratification where "
            "none exists.",
        ],
        "effect_on_PI_RATIFIED_ITEMS": "NONE -- it confirms the allow-list",
        "resolves": ["span_arms_disarmed"],
        "does_not_resolve": ["z_arms_gate_unratified"],
    },
}

#: the FOUR conditions the PI attached to any future z-based or span-based
#: threshold.  A proposal that does not meet all four is not proposable.
PROSPECTIVE_THRESHOLD_CONDITIONS = (
    "precisely defined",
    "calibrated under production geometry",
    "tested for false-alarm behavior",
    "proposed prospectively at a PI checkpoint",
)


# ---------------------------------------------------------------------------
# PI DECISIONS -- the master list.  ``OPEN_PI_DECISIONS`` and
# ``RESOLVED_PI_DECISIONS`` are DERIVED VIEWS of it, so an entry cannot be in
# neither (or in both) and the "n open" count cannot drift from the data.
#
# 🔴 A decision leaves this list only by being ANSWERED BY A DECIDING
# AUTHORITY, with the answer, the date and the source recorded.  It is never
# deleted, so an artifact that pointed at it while it was open still resolves
# -- use ``pi_decision(name)``, which reads BOTH views and reports the status.
# ---------------------------------------------------------------------------
PI_DECISIONS = {
    "z_arms_gate_unratified": {
        "status": "OPEN",
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
        # 🔴 the 2026-08-05 direction touched this entry WITHOUT closing it.
        "pi_direction_2026_08_05": (
            "RELABELLED, NOT RESOLVED, and the distinction is the whole point. "
            "The PI said \"The only currently ratified numerical closure gate "
            "is chi2/dof <= 3\" and that any z-based threshold must be "
            "\"precisely defined; calibrated under production geometry; tested "
            "for false-alarm behavior; proposed prospectively at a PI "
            "checkpoint\". That CONFIRMS the four |z| arms are unratified and "
            "states what would be needed to ratify them. It does NOT say to "
            "disarm them and it does not say to keep them armed, so the "
            "question -- ratify / disarm / carry as a stated limitation -- is "
            "still open and all four are still ARMED. Reading a relabelling "
            "as a disarm order would be the same unilateral act in the other "
            "direction. Contrast span_arms_disarmed, where the PI DID say "
            "what the arms should do (\"keep ... active as advisory "
            "diagnostics, not ratified hard gates\")."),
    },
    "span_arms_disarmed": {
        "status": "RESOLVED",
        "question": (
            "The two ratio_span arms were ARMED before this stream (a9fe97b) "
            "and are now report-only. That follows the PI's instruction that "
            "an unratified tolerance must not gate, but it is the only guard "
            "that fires on a PHYSICALLY large, statistically quiet z-tilt. "
            "Accept report-only, or arm a calibrated pack-specific threshold "
            "(spec §3/§6 option A)?"),
        "resolution": (
            "ACCEPT REPORT-ONLY. The reported statistic stays ACTIVE as an "
            "ADVISORY DIAGNOSTIC and gates nothing. The PI's words, verbatim: "
            "\"Keep span-by-z and span-by-SNR active as advisory diagnostics, "
            "not ratified hard gates.\" BOTH halves bind: deleting the arms "
            "would disobey \"active\", arming them would disobey \"not "
            "ratified hard gates\". The code already does exactly this "
            "(contributes_to_pass_fail=False, effect=" + UNRATIFIED_EFFECT
            + "), so no tolerance and no code path changes; what changes is "
            "that the choice is no longer this stream's to make."),
        "resolved_date": "2026-08-05",
        "resolved_by": "PI direction 2026-08-05 -- see PI_DIRECTIONS['2026-08-05']",
        "option_a_status": (
            "CLOSED. Arming a null-calibrated span threshold (spec §3/§6 "
            "option A, 0.1292 at a measured false-alarm rate of 0.0073) is "
            "not available unless it is proposed PROSPECTIVELY at a PI "
            "checkpoint meeting all four of PROSPECTIVE_THRESHOLD_CONDITIONS. "
            "The measurement stands; the option does not."),
        "status_of_the_tolerances": (
            "UNCHANGED: ratio_span_by_z_max and ratio_span_by_snr_max remain "
            "UNRATIFIED, report-only, and are stamped into every artifact."),
        "what_the_code_does": (
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
            "seed=1). THE RESOLUTION DOES NOT MAKE THIS COST GO AWAY: it "
            "becomes a STATED LIMITATION rather than an open option, and the "
            "null is a lower bound on the true null width (spec §2.1), so "
            "every false-alarm rate here is optimistic and all of it is "
            "synthetic."),
        "superseded_pointers": (
            "artifacts and prose generated BEFORE 2026-08-05 point at "
            "\"ratification.OPEN_PI_DECISIONS['span_arms_disarmed']\" -- e.g. "
            "the committed CDDF_analysis/hbi_mcmc/ratio_span_null_calibration"
            ".json verdict. Those are correct as DATED EVIDENCE of the state "
            "at their generation date. The entry has not been deleted: "
            "pi_decision('span_arms_disarmed') resolves it from either view "
            "and reports status=RESOLVED."),
    },
}

#: DERIVED.  What still needs a deciding authority.
OPEN_PI_DECISIONS = {k: v for k, v in PI_DECISIONS.items()
                     if v["status"] == "OPEN"}
#: DERIVED.  Closed out, each with its answer, date and source.
RESOLVED_PI_DECISIONS = {k: v for k, v in PI_DECISIONS.items()
                         if v["status"] == "RESOLVED"}

OPEN_PI_DECISIONS_NOTE = (
    "PI DECISION REQUIRED, {n} open: {names}. This artifact does not resolve "
    "them and does not claim they are resolved. Details: "
    "CDDF_analysis.hbi_mcmc.ratification.OPEN_PI_DECISIONS. "
    "{nr} previously-open decision(s) have since been answered and are in "
    "RESOLVED_PI_DECISIONS: {rnames}."
).format(n=len(OPEN_PI_DECISIONS), names=", ".join(sorted(OPEN_PI_DECISIONS)),
         nr=len(RESOLVED_PI_DECISIONS),
         rnames=", ".join(sorted(RESOLVED_PI_DECISIONS)) or "none")


def pi_decision(name):
    """A PI decision by name, from EITHER view, with its status.

    The redirect for stale pointers: an artifact written while a decision was
    open names it as ``OPEN_PI_DECISIONS[name]``, and that string is still in
    the artifact after the decision closes.  This resolves it either way.
    ``KeyError`` for an unknown name -- a decision that was never recorded
    must not silently look resolved.
    """
    return dict(PI_DECISIONS[name])


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
    default = records is None
    recs = all_records() if default else records
    bad = []
    if not isinstance(recs, dict):
        return [f"ratification records are {type(recs).__name__}, not a "
                f"mapping; nothing could be audited"]
    if not recs:
        return ["ratification records are EMPTY; an audit that inspected no "
                "record is not a pass"]
    if default:
        # 🔴 FAIL-OPEN HOLE, ROUND 4.  Before this, emptying ``RATIFIED``
        # made the audit return [] and the module import CLEANLY: the guard
        # scanned the records that were there and there were none.  A guard
        # that passes because the thing it guards was deleted is the worst
        # shape in this project's catalogue.  The allow-list is now checked in
        # BOTH directions -- nothing off it may claim PI authority, and
        # everything ON it must actually have a RATIFIED record.
        for name in PI_RATIFIED_ITEMS:
            rec = recs.get(name)
            if not isinstance(rec, dict) or rec.get("status") != "RATIFIED":
                bad.append(
                    f"{name}: is on PI_RATIFIED_ITEMS but has no RATIFIED "
                    f"record. The allow-list and the records must agree; a "
                    f"missing record cannot be read as 'nothing to check'.")
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


#: 🔴 FAIL-OPEN HOLE, ROUND 4.  The worst pattern in this project's catalogue
#: is "delete an entire artifact block and every test stays green".  Of the
#: stamp's blocks only four were pinned by any test; the rest could be dropped
#: silently, and a reader of the JSON would simply not see that the |z| arms
#: gate without authority.  ``ratification_stamp`` now REFUSES to build an
#: incomplete stamp, and `required` may only ever GROW (the ratified
#: fail-closed framework says so in as many words).
REQUIRED_STAMP_KEYS = (
    "schema", "ratification_date", "authority", "authority_scope",
    "pi_ratified_items", "ratified", "restated_not_ratified", "unratified",
    "unratified_effect", "unratified_note", "unratified_but_gating",
    "unratified_but_gating_effect", "unratified_but_gating_note",
    "open_pi_decisions", "open_pi_decisions_note", "resolved_pi_decisions",
    "pi_directions", "prospective_threshold_conditions",
    "authority_allow_list_clean", "self_scan", "correction_note",
)


#: 🔴 THE POWER CHECK for the import-time self-scan (see the bottom of this
#: module).  "Zero violations" is not evidence unless something was inspected.
SELF_SCAN_MIN_CLAIMS = 2
_SELF_SCAN_COUNTER = {"claims": 0}


class IncompleteStampError(RuntimeError):
    """Raised when a ratification stamp is missing a block it must carry."""


def ratification_stamp():
    """The block every artifact carries, verbatim.

    Small on purpose: a reader of the JSON alone must be able to see which
    criteria were authorised to refuse work, WHICH REFUSE WORK WITHOUT BEING
    AUTHORISED, and which are report-only -- without opening the source.

    FAIL-CLOSED: raises :class:`IncompleteStampError` rather than returning a
    stamp that is missing any of ``REQUIRED_STAMP_KEYS``.
    """
    stamp = _build_stamp()
    missing = [k for k in REQUIRED_STAMP_KEYS if k not in stamp]
    if missing:
        raise IncompleteStampError(
            "the ratification stamp is missing required block(s): "
            + ", ".join(missing)
            + ". A stamp that silently drops a block lets an artifact stop "
              "reporting which criteria refuse work without authority.")
    return stamp


def _build_stamp():
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
        # 🔴 a decision that closed is NOT deleted: an artifact written while
        # it was open points at OPEN_PI_DECISIONS[name], and that pointer must
        # still land somewhere.  ``pi_decision(name)`` reads both views.
        "resolved_pi_decisions": {
            k: dict(v) for k, v in sorted(RESOLVED_PI_DECISIONS.items())},
        "pi_directions": {k: dict(v)
                          for k, v in sorted(PI_DIRECTIONS.items())},
        "prospective_threshold_conditions": list(
            PROSPECTIVE_THRESHOLD_CONDITIONS),
        "authority_allow_list_clean": (audit_authority_claims() == []),
        # 🔴 A POWER CHECK, IN THE ARTIFACT.  "clean" is worthless without
        # "of how many": a stamp whose ratification blocks were deleted also
        # scans clean.  This says what the guard actually inspected, so a
        # reader of the JSON alone can tell a real pass from a vacuous one.
        "self_scan": {
            "schema": SCAN_SCHEMA,
            "rules": sorted(SCAN_RULES),
            "n_claims_inspected": _SELF_SCAN_COUNTER["claims"],
            "min_claims_required": SELF_SCAN_MIN_CLAIMS,
            "note": "the import-time scan of this stamp. A zero here would "
                    "mean the stamp carries no checkable ratification claim "
                    "at all, and the module refuses to import in that state. "
                    "Tree-wide equivalent: python -m "
                    "CDDF_analysis.hbi_mcmc.ratification --check <paths>",
        },
        "correction_note": (
            "schema v2 CORRECTS v1 (88f2ecb), which recorded the four "
            "|z| <= 5 arms as RATIFIED by the PI on grounds that were false: "
            "decision 8 called |z| <= 5 MALFORMED AS STATED and asked for a "
            "restatement, and two of the four arms were introduced the same "
            "day, in the same hunk, as the two tolerances the PI declined. "
            "Any artifact carrying gate_ratification/v1 overstates its "
            "authority."),
    }


# ===========================================================================
# THE WIDENED GUARD -- a scanner for ratification CLAIMS, in any shape,
# anywhere in a tree, runnable from a merge hook.
#
# The import-time allow-list above sees ONE dict, in ONE module, on ONE
# branch, under TWO field names (``status`` and ``authority``).  Every one of
# those four narrownesses was measured to be a real miss -- see the module
# docstring, "WHY THE IMPORT-TIME GUARD WAS NOT ENOUGH".
#
# WHAT COUNTS AS A CLAIM.  Four independent rules, each of which fires on its
# own, so no single reshuffle of a record evades all four:
#
#   R1_NAME_CLAIM      a key that ASSERTS ratification (``classify_key`` ->
#                      "CLAIM") holding NAMES: a list of strings, or the keys
#                      of a mapping.  Every name must be on
#                      ``PI_RATIFIED_ITEMS``.  This is the rule that catches
#                      ``gate_tolerances_ratified=[...]`` and
#                      ``ratified_arms={...}``, neither of which is a record
#                      and neither of which the import-time guard can see.
#   R2_STATUS_CLAIM    any mapping with a ``*status`` field whose value is the
#                      bare token RATIFIED.  Its SUBJECT (its key in the
#                      enclosing mapping, or its own ``name``/``key``/
#                      ``tolerance`` field) must be allow-listed.
#   R3_PI_AUTHORITY    any mapping with an ``*authority*`` field naming the PI
#                      affirmatively.  Same subject rule.  Generalises
#                      ``audit_authority_claims`` to arbitrary nesting and to
#                      JSON.  ONE escape, and it is the v1 lesson made
#                      executable: a stamp may carry a top-level PI authority
#                      IF it carries an ``authority_scope`` that names
#                      ``pi_ratified_items`` and says NOTHING ELSE.
#   R4_PROSE_CLAIM     a sentence containing an unqualified ``RATIFIED`` and a
#                      criterion name that is not allow-listed.  Catches the
#                      form that has escaped every structural guard so far
#                      (three times after the retraction).
#
# AND THE FAIL-CLOSED RULES, which exist because a guard that returns "clean"
# for something it could not read is worse than no guard:
#
#   R5_UNRECOGNISED    a CLAIM key whose value is not a shape this scanner
#                      knows how to check (a bare number, a mixed list, a
#                      mapping with computed keys), or a status/authority
#                      claim whose subject cannot be resolved.
#   R6_UNPARSEABLE     a path that does not exist, cannot be read, or does not
#                      parse as the JSON/Python it is named as.
#   R7_UNDERIVED       a CLAIM key in CODE whose value is computed rather than
#                      literal AND whose expression does not reference this
#                      module.  ``list(RAT.ratified_names())`` is fine --
#                      that IS the single source of truth.  An expression that
#                      builds a ratified-name list from anywhere else is not
#                      checkable and is refused.
#   R8_NOTHING_SCANNED a scan that inspected zero files.  A green check that
#                      looked at nothing is the fail-open shape this project
#                      has been burned by; it is an error, not a pass.
# ===========================================================================

SCAN_SCHEMA = "ratification_scan/v1"

#: file kinds the tree scanner understands.  ``.md`` is prose-only (R4).
SCAN_SUFFIXES = (".py", ".json", ".md")

SCAN_RULES = {
    "R1_NAME_CLAIM": "a ratification-asserting key names something that is "
                     "not on PI_RATIFIED_ITEMS",
    "R2_STATUS_CLAIM": "a record's status is RATIFIED but its subject is not "
                       "on PI_RATIFIED_ITEMS",
    # 🔴 phrased WITHOUT a bare "PI" token on purpose: R3 fires on any
    # authority field that affirms it, and the first run of this scanner
    # flagged this very glossary entry.  That is the guard working, and the
    # fix is to reword the glossary, NOT to widen _AUTHORITY_META_KEYS.
    "R3_PI_AUTHORITY": "a record names the deciding authority as having "
                       "ratified a subject that is not on PI_RATIFIED_ITEMS",
    "R4_PROSE_CLAIM": "a sentence calls a non-allow-listed criterion RATIFIED",
    "R5_UNRECOGNISED": "a ratification claim in a shape this scanner cannot "
                       "check, or whose subject cannot be resolved "
                       "(FAIL-CLOSED)",
    "R6_UNPARSEABLE": "a path that cannot be read or parsed (FAIL-CLOSED)",
    "R7_UNDERIVED": "a computed ratified-name list not derived from "
                    "CDDF_analysis.hbi_mcmc.ratification (FAIL-CLOSED)",
    "R8_NOTHING_SCANNED": "the scan inspected zero files (FAIL-CLOSED)",
}

_NONWORD_RE = re.compile(r"[^0-9a-z]+")

#: key tokens that ASSERT ratification, as opposed to ``ratification`` /
#: ``ratifying``, which merely NAME the subject of a record.
_CLAIM_TOKENS = ("ratified", "ratifies", "ratify")
#: token prefixes that negate the assertion outright (``unratified``, ...)
_NEGATED_PREFIXES = ("unratif", "nonratif", "notratif", "deratif", "disratif")
#: a word immediately before a claim token that negates it
_NEGATING_PREDECESSORS = ("not", "non", "never", "un", "no", "yet", "without",
                          "pending", "declined", "unratified")

#: an ``authority``-bearing key whose LAST token is one of these is a
#: META-field ABOUT an authority claim (a note, a scope, a correction) rather
#: than the claim itself.  A suffix rule rather than a name list, so a new
#: ``*_note`` does not need registering -- and it is exactly what makes R3
#: leave a RETRACTION NOTE alone while still firing on the ``authority`` field
#: it retracts.  🔴 THE COST, stated rather than hidden: an authority claim
#: written as prose inside a ``*_note`` is R4's job, not R3's, and R4 keys on
#: an UPPERCASE ``RATIFIED``.  A lowercase prose claim in a note escapes both.
_AUTHORITY_META_SUFFIXES = (
    "note", "notes", "scope", "question", "detail", "details", "correction",
    "explanation", "disposition", "history", "rationale", "clean", "check",
    "policy", "glossary", "meaning", "meanings",
)

# ---------------------------------------------------------------------------
# 🔴 ASSERTION vs REFERENCE.  MEASURED FALSE-ALARM DEFECT, 2026-08-05.
#
# The first version of R1 asked only "does this key contain the participle",
# and on the CORRECTED sibling tips that produced false alarms on keys that
# merely REFER to the ratified arm instead of claiming something is ratified:
#
#   verdict_rests_on_the_ratified_arm_alone   a sensitivity block whose
#                                             sub-keys are what/full_grid/
#                                             window/answer
#   min_factor_over_ratified_chi2_gate        a float: a margin OVER the gate
#   n_closing_pi_ratified_arm_only            a count of configurations
#
# All three are the honest disclosure this module exists to REQUIRE.  A guard
# that flags them pressures people to delete correct disclosures, which is the
# same defect inverted.  Patching the names one at a time does not converge --
# the siblings keep adding more -- so the discriminator is GRAMMATICAL:
#
#   ASSERTION  the participle is the HEAD of the phrase.  Either it is last
#              (``gate_tolerances_ratified``), or everything after it ends in
#              a COLLECTION noun naming what is claimed (``ratified_arms``,
#              ``pi_ratified_items``, ``ratified_z_arms``).
#   REFERENCE  a PREPOSITION or DETERMINER governs it (``rests_ON_THE_ratified
#              _arm``, ``factor_OVER_ratified_chi2_gate``), or the phrase
#              continues past the participle into something that is not a
#              collection (``..._ratified_arm_ALONE``, ``..._arm_ONLY``).
#              English puts the head last; a modifier under a preposition is
#              pointing AT the ratified thing, not asserting one.
# ---------------------------------------------------------------------------

#: a token that makes what follows a REFERENCE rather than an assertion
_REFERENCE_GOVERNORS = (
    # prepositions
    "on", "over", "under", "by", "for", "with", "from", "to", "of", "in",
    "at", "above", "below", "against", "per", "than", "versus", "vs",
    "beyond", "within", "without", "across", "between", "into", "onto",
    # determiners / relatives (a key spelling out a sentence)
    "the", "a", "an", "that", "which", "whose", "its",
    # verbs that make the phrase a statement ABOUT the ratified thing
    "rests", "rest", "depends", "relies", "uses", "reads", "counts",
    "closing", "failing", "passing", "measured", "compared",
)

#: nouns that name a COLLECTION of the things claimed ratified.  A key may
#: continue past the participle only into one of these.
_COLLECTION_NOUNS = (
    "items", "item", "arms", "arm", "names", "name", "list", "lists", "set",
    "sets", "tolerances", "tolerance", "gates", "gate", "criteria",
    "criterion", "thresholds", "threshold", "entries", "keys", "ids",
    "things", "requirements", "requirement",
)

#: a CLAIM key whose first token COUNTS does not hold a set of names.
_COUNT_PREFIXES = ("n", "num", "count", "total", "len", "min", "max", "mean",
                   "median", "sum", "frac", "fraction", "pct", "percent",
                   "ratio", "factor")

#: ``status``, ``gate_status``, ``authority_state`` -- all the same field.
_STATUS_KEY_RE = re.compile(r"(?:^|_)(?:status|state)$")
_PI_TOKEN_RE = re.compile(r"\bPI\b")

#: fields that name, on the record itself, WHAT the record is about
_SUBJECT_FIELDS = ("name", "key", "tolerance", "criterion", "canonical_name",
                   "arm", "item", "gate", "subject")

#: fields whose presence makes a mapping a GLOSSARY ENTRY -- a definition of a
#: term -- rather than a record about a criterion.  ``{"state": "RATIFIED",
#: "meaning": "A deciding authority ratified this IN WRITING..."}`` is the
#: state VOCABULARY, and asserts nothing about any tolerance.
_DEFINITION_FIELDS = ("meaning", "definition", "description", "doc",
                      "docstring", "explanation", "means")

#: fields that make a mapping a RECORD ABOUT SOMETHING (a value it governs)
_VALUE_FIELDS = ("value", "threshold", "number", "tolerance_value", "max",
                 "limit", "gates", "contributes_to_pass_fail")

#: an identifier that LOOKS like a gate tolerance.  Deliberately loose on the
#: stem and strict on the suffix, so ``abs_z_total_max`` -- a local spelling
#: that appears in NO ratification record -- is still recognised as a name.
_TOLERANCE_NAME_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+_max\b")

#: prose forms that are the CORRECT way to mention an unratified name next to
#: the word.  A sentence carrying one of these is not making a claim.
#: 🔴 KEPT NARROW ON PURPOSE, and this is the guard's stated weakness: every
#: form here is a way for a real claim to be skipped, so a looser list is a
#: wider bypass.  The five original forms are the ones the 2026-07-30 prose
#: guard shipped with; the rest were each added only after a real, checked
#: sentence in this tree needed them.
_PROSE_QUALIFIER_RE = re.compile(
    r"NOT[ _-]RATIFIED|not ratified|UNRATIFIED|unratified|"
    r"RESTATED_NOT_RATIFIED|nobody ratified|no deciding authority|"
    r"never ratified|not been ratified|declined to ratif|"
    r"not yet ratified|NOT A RATIFICATION|not a ratification")

#: an expression that is DERIVED from this module is checkable at run time by
#: the import-time guard, so a static scanner must not refuse it.
_DERIVED_RE = re.compile(
    r"ratification|\bRAT\b|\b_RAT\b|ratified_names|PI_RATIFIED_ITEMS|"
    r"RATIFIED|ratification_stamp")


def _norm_key(key):
    """``"Gate Tolerances Ratified"`` -> ``"gate_tolerances_ratified"``."""
    return _NONWORD_RE.sub("_", str(key).lower()).strip("_")


def classify_key(key):
    """``"CLAIM"`` | ``"NEGATED"`` | ``"SUBJECT"`` | ``None``.

    The discriminator is grammatical and it is the reason the widened guard
    needs no hand-maintained field-name list: the past participle
    (``ratified``) ASSERTS, the noun (``ratification``) merely NAMES.

        >>> classify_key("gate_tolerances_ratified")
        'CLAIM'
        >>> classify_key("ratified_arms")
        'CLAIM'
        >>> classify_key("gate_tolerances_not_ratified")
        'NEGATED'
        >>> classify_key("unratified_but_gating")
        'NEGATED'
        >>> classify_key("ratification_date")
        'SUBJECT'
        >>> classify_key("chi2_dof_max") is None
        True
    """
    toks = [t for t in _norm_key(key).split("_") if t]
    for t in toks:
        if t.startswith(_NEGATED_PREFIXES):
            return "NEGATED"
    for i, t in enumerate(toks):
        if t in _CLAIM_TOKENS:
            if i and toks[i - 1] in _NEGATING_PREDECESSORS:
                return "NEGATED"
            return "CLAIM"
    for t in toks:
        if t.startswith("ratif"):     # ratification, ratifying, ratifiable
            return "SUBJECT"
    return None


def claim_is_name_bearing(key):
    """Is the participle the HEAD of this key -- an ASSERTION, not a REFERENCE?

    See "ASSERTION vs REFERENCE" above.  ``True`` means the key claims that
    the things it holds ARE ratified, and R1/R5/R7 apply.  ``False`` means it
    points at the ratified thing while saying something else about it, and
    only the prose rule applies.

        >>> claim_is_name_bearing("gate_tolerances_ratified")
        True
        >>> claim_is_name_bearing("ratified_arms")
        True
        >>> claim_is_name_bearing("pi_ratified_items")
        True
        >>> claim_is_name_bearing("ratified_z_arms")
        True
        >>> claim_is_name_bearing("verdict_rests_on_the_ratified_arm_alone")
        False
        >>> claim_is_name_bearing("min_factor_over_ratified_chi2_gate")
        False
        >>> claim_is_name_bearing("n_closing_pi_ratified_arm_only")
        False
        >>> claim_is_name_bearing("n_failing_ratified_chi2_arm_alone")
        False
    """
    toks = [t for t in _norm_key(key).split("_") if t]
    if not toks:
        return False
    if toks[0] in _COUNT_PREFIXES:
        return False                    # it counts or measures; it does not name
    idx = next((i for i, t in enumerate(toks) if t in _CLAIM_TOKENS), None)
    if idx is None:
        return False
    if any(t in _REFERENCE_GOVERNORS for t in toks[:idx]):
        return False                    # a preposition/determiner governs it
    tail = toks[idx + 1:]
    if not tail:
        return True                     # the participle is the head
    return tail[-1] in _COLLECTION_NOUNS


def _v(rule, source, path, subject, detail):
    return {"rule": rule, "source": str(source), "path": str(path),
            "subject": (None if subject is None else str(subject)),
            "detail": str(detail)}


def format_violation(v):
    subj = f" [{v['subject']}]" if v.get("subject") else ""
    return f"{v['source']}:{v['path']}{subj} {v['rule']}: {v['detail']}"


def _affirms_pi(text):
    """Does this authority string name the PI as the AUTHORISER?

    Declining is not authorising, and neither is an explicit non-claim.
    """
    if not (_PI_TOKEN_RE.search(text)
            or "principal investigator" in text.lower()):
        return False
    upper = text.upper()
    if "DECLIN" in upper:
        return False
    if upper.lstrip().startswith("NONE"):
        return False
    return True


def _scope_limits_authority(mapping):
    """The ONE escape from R3, and it is v1's defect written as a condition.

    v1 put ``authority: "PI (...)"`` at the top of the stamp with nothing
    saying what it covered, and that is how four |z| arms acquired PI
    authority.  A stamp may carry a top-level PI authority only if it also
    states, IN THE ARTIFACT, that the authority covers ``pi_ratified_items``
    and nothing else.
    """
    scope = mapping.get("authority_scope")
    if not isinstance(scope, str):
        return False
    return "pi_ratified_items" in scope and "NOTHING ELSE" in scope.upper()


def _named_entities(text):
    """Criterion names a string NAMES, on identifier boundaries.

    The same extractor R4 uses, so "what does this string talk about" has one
    definition.  ``"chi2_dof_max <= 3.0 (PI decision 8)"`` -> ``{chi2_dof_max}``.
    """
    known = set(RESTATED_NOT_RATIFIED) | set(UNRATIFIED) | set(RATIFIED)
    found = {n for n in known
             if re.search(r"(?<![0-9A-Za-z_])" + re.escape(n)
                          + r"(?![0-9A-Za-z_])", text)}
    return found | set(_TOLERANCE_NAME_RE.findall(text))


#: a subject must be a NAME.  ``z_criterion`` carries a field literally called
#: ``criterion`` whose value is a PARAGRAPH describing the criterion; taking
#: that as the subject is how a correct block got attributed to prose.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _as_subject_name(value):
    """``value`` if it is an identifier-shaped NAME, else ``None``."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v or len(v) > 64 or not _IDENTIFIER_RE.match(v):
        return None
    return v


def _is_criterion_name(name):
    """Does ``name`` denote a CRITERION, as opposed to a structural container?

    ``z_bin_max`` does; ``z_criterion``, ``full_grid`` and
    ``authority_sensitivity`` do not.  R3 needs one: an authority field whose
    subject is a container is not making a per-criterion authority claim.
    """
    if not isinstance(name, str) or not name:
        return False
    if name in all_records() or name in PI_RATIFIED_ITEMS:
        return True
    return bool(_TOLERANCE_NAME_RE.fullmatch(name))


def _is_ratification_stamp(mapping):
    """Does this mapping publish a ratified-name list of its own?

    Such a mapping IS a record about criteria collectively, so a bare
    top-level PI authority on it must be scoped -- that is the v1 defect, and
    it keeps firing even though the stamp's own subject is not a criterion.
    """
    return any(classify_key(k) == "CLAIM" and claim_is_name_bearing(k)
               for k in mapping)


def _subject_of(mapping, owner, field=None):
    """WHAT a status/authority field is about.

    🔴 MEASURED FALSE-ALARM DEFECT, 2026-08-05.  This used to fall straight
    through to ``owner`` -- the enclosing key -- and on the corrected sibling
    tips that attributed correct records to the wrong subject:

        {"gate_max_arm": "chi2_dof_max",
         "gate_max_authority_state": "RATIFIED"}    subject was
                                                    "best_by_reporting_chi2_dof"
        {"pi_authority_gating_arm": "chi2_dof_max <= 3.0 (PI decision 8)"}
                                                    subject was
                                                    "authority_sensitivity"

    In both the real subject is ``chi2_dof_max``, which IS allow-listed, so
    both were false alarms.  Resolution order, most specific first, and none
    of it keys on a particular field name:

      1. a subject field ON THE RECORD (``name``/``arm``/``canonical_name``);
      2. the STEM SIBLING: strip the status/authority suffix off the field's
         own key and look for a sibling sharing that stem whose value names an
         entity (``gate_max_authority_state`` -> ``gate_max_arm``);
      3. the entity the FIELD'S OWN VALUE names, when it names exactly one;
      4. the enclosing key.
    """
    for f in _SUBJECT_FIELDS:
        name = _as_subject_name(mapping.get(f))
        if name is not None:
            return name
    if field is not None:
        nfield = _norm_key(field)
        stem = re.sub(r"(?:^|_)(?:authority_state|authority_status|"
                      r"authority|status|state)$", "", nfield)
        if stem and stem != nfield:
            for k, v in mapping.items():
                nk = _norm_key(k)
                if nk == nfield or not nk.startswith(stem + "_"):
                    continue
                name = _as_subject_name(v)
                if name is not None and _norm_key(name) != "ratified":
                    return name
        named = _named_entities(mapping.get(field) or "")
        if len(named) == 1:
            return next(iter(named))
    return owner


def _is_glossary_entry(mapping):
    """Is this mapping a DEFINITION OF A TERM rather than a record?

    ``{"state": "RATIFIED", "meaning": "A deciding authority ratified this IN
    WRITING. Only the items in PI_RATIFIED_ITEMS may carry this state."}`` is
    the state VOCABULARY.  It asserts nothing about any criterion, and a guard
    that flags it is telling people not to publish their own legend.

    Recognised by SHAPE, not by name: it defines a term and governs nothing --
    no subject field and no value field.
    """
    if not any(f in mapping for f in _DEFINITION_FIELDS):
        return False
    if any(f in mapping for f in _SUBJECT_FIELDS):
        return False
    return not any(f in mapping for f in _VALUE_FIELDS)


def _scan_prose(text, *, source, path):
    """R4.  Sentence-scoped, so the modules can (and must) DISCUSS these names.

    A sentence claims ratification if it carries an unqualified ``RATIFIED``
    and names a criterion that is not allow-listed.  Names come from two
    places, and the second is what makes this catch a spelling no record uses:
    the known unratified names, AND anything matching ``*_max``.
    """
    if "RATIFIED" not in text:
        return []
    known = set(RESTATED_NOT_RATIFIED) | set(UNRATIFIED)
    flat = re.sub(r"\s+", " ", text)
    out = []
    for sent in re.split(r"(?<=[.!?])\s", flat):
        if "RATIFIED" not in sent:
            continue
        if _PROSE_QUALIFIER_RE.search(sent):
            continue
        # identifier-boundary, not substring: ``abs_z_total_max`` must report
        # itself and NOT also its suffix ``z_total_max``, or one claim reads
        # as two and the violation count stops meaning anything.
        names = {n for n in known
                 if re.search(r"(?<![0-9A-Za-z_])" + re.escape(n)
                              + r"(?![0-9A-Za-z_])", sent)}
        names |= set(_TOLERANCE_NAME_RE.findall(sent))
        for name in sorted(names - set(PI_RATIFIED_ITEMS)):
            out.append(_v(
                "R4_PROSE_CLAIM", source, path, name,
                f"calls {name!r} RATIFIED without qualification: "
                f"{sent[:220]!r}"))
    return out


def _check_claim_names(names, *, source, path):
    out = []
    for name in names:
        if name not in PI_RATIFIED_ITEMS:
            out.append(_v(
                "R1_NAME_CLAIM", source, path, name,
                f"asserts that {name!r} is ratified; PI_RATIFIED_ITEMS is "
                f"{PI_RATIFIED_ITEMS} and nothing else may be claimed. "
                f"Decision 8 ratified exactly three things."))
    return out


def _check_claim_key(key, value, *, source, path, owner):
    """R1 / R5 for one ``key: value`` pair whose key CLAIMS ratification."""
    if value is None or value is False:
        return []
    if value is True:
        subject = _subject_of({}, owner)
        if subject is None:
            return [_v("R5_UNRECOGNISED", source, path, None,
                       f"{key!r} is True but the scanner cannot resolve WHAT "
                       f"is being called ratified")]
        return _check_claim_names([subject], source=source, path=path)
    if isinstance(value, str):
        if _norm_key(key) == _norm_key(value):
            # a self-naming ENUM constant (``RATIFIED = "RATIFIED"``), which
            # DEFINES the state token rather than applying it to anything.
            return []
        if _norm_key(value) in ("ratified", "pi_ratified", "ratified_by_pi",
                               "ratified_by_the_pi"):
            subject = _subject_of({}, owner)
            if subject is None:
                return [_v("R5_UNRECOGNISED", source, path, None,
                           f"{key!r}={value!r} asserts ratification with no "
                           f"resolvable subject")]
            return _check_claim_names([subject], source=source, path=path)
        return []                       # prose; the string walker handles it
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if not seq:
            return []
        if not all(isinstance(e, str) for e in seq):
            return [_v("R5_UNRECOGNISED", source, path, None,
                       f"{key!r} claims ratification but holds a "
                       f"non-string element; the scanner cannot tell what is "
                       f"being claimed")]
        return _check_claim_names(seq, source=source, path=path)
    if isinstance(value, dict):
        names = [k for k in value if isinstance(k, str)]
        if len(names) != len(value):
            return [_v("R5_UNRECOGNISED", source, path, None,
                       f"{key!r} claims ratification but has non-string keys")]
        return _check_claim_names(names, source=source, path=path)
    return [_v("R5_UNRECOGNISED", source, path, None,
               f"{key!r} claims ratification and holds "
               f"{type(value).__name__}, which is not a shape this scanner "
               f"can check")]


def _check_mapping(mapping, *, source, path, owner):
    """R2 and R3 for one mapping."""
    out = []
    glossary = _is_glossary_entry(mapping)
    for key, val in mapping.items():
        nkey = _norm_key(key)
        if (_STATUS_KEY_RE.search(nkey) and isinstance(val, str)
                and _norm_key(val) == "ratified" and not glossary):
            subject = _subject_of(mapping, owner, field=key)
            if subject is None:
                out.append(_v(
                    "R5_UNRECOGNISED", source, f"{path}/{key}", None,
                    "status=RATIFIED with no resolvable subject "
                    "(no enclosing key and no name/key/tolerance field)"))
            elif subject not in PI_RATIFIED_ITEMS:
                out.append(_v(
                    "R2_STATUS_CLAIM", source, f"{path}/{key}", subject,
                    f"records status=RATIFIED for {subject!r}, which is "
                    f"not on PI_RATIFIED_ITEMS={PI_RATIFIED_ITEMS}"))
        ntoks = nkey.split("_")
        if ("authority" in ntoks and ntoks[-1] not in _AUTHORITY_META_SUFFIXES
                and isinstance(val, str) and _affirms_pi(val)):
            if _scope_limits_authority(mapping):
                continue
            # 🔴 WHAT DOES THE STRING ITSELF SAY IT IS ABOUT?  A per-tolerance
            # authority disclosure names its subjects INSIDE the value --
            # "chi2_dof_max = 3.0 is RATIFIED (PI decision 8); z_total_max and
            # z_bin_max are RESTATED_NOT_RATIFIED" -- and that is the honest
            # record this module exists to require.  If every entity the
            # string names is allow-listed, the structural claim is correct;
            # whether the PROSE then says something false is R4's job, on this
            # same string, with its qualifier vocabulary.
            named = _named_entities(val)
            if named and named <= set(PI_RATIFIED_ITEMS):
                continue
            subject = _subject_of(mapping, owner, field=key)
            if subject in PI_RATIFIED_ITEMS:
                continue
            # 🔴 R3 NEEDS A CRITERION.  If the subject is a structural
            # container (`z_criterion`, `full_grid`, `authority_sensitivity`),
            # the field is not claiming that a particular criterion was
            # ratified, and treating it as one flags RETRACTION NOTES -- the
            # honest disclosure this module exists to require.  The prose is
            # then R4's business, on this same string.
            # THE COST, stated: an authority claim written as prose that names
            # NO criterion escapes R3, and escapes R4 unless it carries an
            # unqualified uppercase RATIFIED next to a name.  Pinned by
            # test_the_R3_subject_rule_states_the_gap_it_leaves.
            if not (_is_criterion_name(subject)
                    or _is_ratification_stamp(mapping)):
                continue
            out.append(_v(
                "R3_PI_AUTHORITY", source, f"{path}/{key}", subject,
                f"names the PI as authority ({val[:90]!r}) for subject "
                f"{subject!r}, which is not on "
                f"PI_RATIFIED_ITEMS={PI_RATIFIED_ITEMS}"
                + ("" if subject is not None else
                   "; and the subject could not be resolved, so the claim "
                   "cannot be checked at all")))
    return out


def scan_data(obj, *, source="<data>", path="", owner=None, _counter=None):
    """Every rule, over a parsed JSON-like structure.  Returns violations.

    ``_counter`` (a one-key dict) accumulates how many ratification CLAIMS
    were inspected, so a clean result can be told apart from a vacuous one --
    a containment/coverage check that cannot fail is not a check.
    """
    out = []
    if isinstance(obj, dict):
        out += _check_mapping(obj, source=source, path=path or "/",
                              owner=owner)
        for key, val in obj.items():
            kpath = f"{path}/{key}"
            if classify_key(key) == "CLAIM" and claim_is_name_bearing(key):
                if _counter is not None:
                    _counter["claims"] = _counter.get("claims", 0) + 1
                out += _check_claim_key(key, val, source=source, path=kpath,
                                        owner=owner)
            out += scan_data(val, source=source, path=kpath, owner=str(key),
                             _counter=_counter)
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            out += scan_data(val, source=source, path=f"{path}[{i}]",
                             owner=owner, _counter=_counter)
    elif isinstance(obj, str):
        out += _scan_prose(obj, source=source, path=path or "/")
    return out


def scan_json_text(text, *, source="<json>", _counter=None):
    try:
        data = json.loads(text)
    except Exception as exc:                                # noqa: BLE001
        return [_v("R6_UNPARSEABLE", source, "/", None,
                   f"does not parse as JSON: {exc}")]
    return scan_data(data, source=source, _counter=_counter)


# --- the code side ---------------------------------------------------------

_UNRESOLVED = object()


def _const(node):
    """The literal value of an AST node, or ``_UNRESOLVED``."""
    if isinstance(node, ast.Constant):
        return node.value
    return _UNRESOLVED


def _target_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        sub = _const(node.slice)
        if isinstance(sub, str):
            return sub
    return None


def _literal_names(node):
    """Names claimed by an AST value node, or ``None`` if it is not literal.

    Deliberately tolerant of PARTLY-computed containers: ``{"a": _r(...)}``
    still tells us the NAMES, which is what R1 checks.
    """
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):               # {**other}
            return _UNRESOLVED
        names = [_const(k) for k in node.keys]
        if any(not isinstance(n, str) for n in names):
            return _UNRESOLVED
        return names
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        vals = [_const(e) for e in node.elts]
        if any(not isinstance(v, str) for v in vals):
            return _UNRESOLVED
        return vals
    if isinstance(node, ast.Constant):
        return node.value
    return _UNRESOLVED


def _is_factory_dict(node, resolver=_const):
    """A dict LITERAL that is a template: at least one value is computed.

    ``_r(...)`` in this module returns ``{"status": "RATIFIED", "statement":
    statement, ...}``.  Its subject is supplied by the CALLER, and every
    caller is a dict entry whose key R1 already checks.  A dict whose values
    all RESOLVE to constants is not a template and gets no such exemption --
    including through a module-level string constant, so ``{"status":
    RATIFIED}`` cannot buy the exemption a genuine template gets.
    """
    if not isinstance(node, ast.Dict):
        return False
    return any(resolver(v) is _UNRESOLVED for v in node.values)


def scan_python_source(text, *, source="<py>", _counter=None):
    """R1/R2/R3/R4/R5/R7 over Python source, by AST.

    Claims in code are keyword arguments, assignment targets and string dict
    keys.  Both live fabricated-authority sites are of the first two kinds:
    ``gate_tolerances_ratified=[...]`` is a keyword argument and
    ``ratified_arms={...}`` is a keyword argument.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [_v("R6_UNPARSEABLE", source, "/", None,
                   f"does not parse as Python: {exc}")]

    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node

    # 🔴 FAIL-OPEN HOLE, ROUND 4.  R2/R3 read only CONSTANT field values, so a
    # module that writes ``authority_state=RATIFIED`` with a module-level
    # ``RATIFIED = "RATIFIED"`` constant -- which is exactly how the spectral
    # -window module spells it -- was invisible to both.  Module-level string
    # constants are resolved so the Name and the literal are checked alike.
    strconst = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value,
                                                       ast.Constant) \
                and isinstance(node.value.value, str):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    strconst[tgt.id] = node.value.value

    def resolve(node):
        val = _const(node)
        if val is not _UNRESOLVED:
            return val
        if isinstance(node, ast.Name) and node.id in strconst:
            return strconst[node.id]
        return _UNRESOLVED

    out = []

    def claim(key, value_node, where):
        if classify_key(key) != "CLAIM" or not claim_is_name_bearing(key):
            return
        if isinstance(value_node, ast.Constant) and isinstance(
                value_node.value, str) and _norm_key(key) == _norm_key(
                    value_node.value):
            return                          # enum constant, see above
        if _counter is not None:
            _counter["claims"] = _counter.get("claims", 0) + 1
        names = _literal_names(value_node)
        if names is _UNRESOLVED:
            seg = ast.get_source_segment(text, value_node) or ""
            if _DERIVED_RE.search(seg):
                return                       # derived from this module: OK
            out.append(_v(
                "R7_UNDERIVED", source, where, None,
                f"{key!r} claims ratification from a computed expression "
                f"that does not reference "
                f"CDDF_analysis.hbi_mcmc.ratification: {seg[:120]!r}"))
            return
        if isinstance(names, list):
            out.extend(_check_claim_names(names, source=source, path=where))
            return
        out.extend(_check_claim_key(key, names, source=source, path=where,
                                    owner=None))

    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.keyword) and node.arg:
            claim(node.arg, node.value, f"line {line}:{node.arg}")
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                nm = _target_name(tgt)
                if nm:
                    claim(nm, node.value, f"line {line}:{nm}")
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            nm = _target_name(node.target)
            if nm:
                claim(nm, node.value, f"line {line}:{nm}")
        elif isinstance(node, ast.Dict):
            for knode, vnode in zip(node.keys, node.values):
                kv = _const(knode) if knode is not None else _UNRESOLVED
                if isinstance(kv, str):
                    claim(kv, vnode, f"line {line}:{kv}")
            # R2/R3 need the mapping's own fields plus its SUBJECT
            flat = {}
            for knode, vnode in zip(node.keys, node.values):
                kv = _const(knode) if knode is not None else _UNRESOLVED
                vv = resolve(vnode)
                if isinstance(kv, str) and vv is not _UNRESOLVED:
                    flat[kv] = vv
            if not flat:
                continue
            owner = None
            par = parent.get(id(node))
            if isinstance(par, ast.Dict):
                for knode, vnode in zip(par.keys, par.values):
                    if vnode is node and knode is not None:
                        kv = _const(knode)
                        if isinstance(kv, str):
                            owner = kv
            elif isinstance(par, ast.keyword) and par.arg:
                owner = par.arg
            elif isinstance(par, ast.Assign):
                for tgt in par.targets:
                    owner = owner or _target_name(tgt)
            if owner is None and _is_factory_dict(node, resolve):
                continue                     # a template; callers carry names
            out.extend(_check_mapping(flat, source=source,
                                      path=f"line {line}", owner=owner))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.extend(_scan_prose(node.value, source=source,
                                   path=f"line {line}"))

    # dedupe: ast.walk reaches nested dicts through several roots
    seen, uniq = set(), []
    for v in out:
        k = (v["rule"], v["path"], v["subject"], v["detail"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(v)
    return uniq


def scan_markdown(text, *, source="<md>"):
    return _scan_prose(text, source=source, path="/")


def scan_file(path, *, _counter=None):
    """Dispatch on suffix.  An unreadable file is a FAILURE, not a skip."""
    path = str(path)
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as fh:
            text = fh.read()
    except Exception as exc:                                # noqa: BLE001
        return [_v("R6_UNPARSEABLE", path, "/", None,
                   f"cannot be read: {exc}")]
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        return scan_json_text(text, source=path, _counter=_counter)
    if suffix == ".py":
        return scan_python_source(text, source=path, _counter=_counter)
    if suffix == ".md":
        return scan_markdown(text, source=path)
    return [_v("R6_UNPARSEABLE", path, "/", None,
               f"suffix {suffix!r} is not one this scanner knows how to "
               f"check; SCAN_SUFFIXES={SCAN_SUFFIXES}")]


class ScanResult(object):
    """What a scan measured.  ``ok`` is the ONLY thing a caller should trust.

    ``n_files`` and ``n_claims`` are reported so that a clean result can be
    distinguished from a vacuous one: a scan of an empty tree, or of a tree
    with no ratification claims in it, is not evidence that the guard works.
    """

    __slots__ = ("violations", "files", "n_claims", "roots", "excluded")

    def __init__(self, violations, files, n_claims, roots, excluded):
        self.violations = list(violations)
        self.files = list(files)
        self.n_claims = int(n_claims)
        self.roots = list(roots)
        self.excluded = list(excluded)

    @property
    def n_files(self):
        return len(self.files)

    @property
    def ok(self):
        return not self.violations

    def as_dict(self):
        return {"schema": SCAN_SCHEMA, "roots": self.roots,
                "excluded": self.excluded, "n_files": self.n_files,
                "n_claims_inspected": self.n_claims,
                "n_violations": len(self.violations),
                "ok": self.ok, "violations": self.violations,
                "rules": dict(SCAN_RULES)}

    def report(self):
        lines = [f"RATIFICATION SCAN  ({SCAN_SCHEMA})",
                 f"  roots      : {', '.join(self.roots) or '(none)'}",
                 f"  excluded   : {', '.join(self.excluded) or '(none)'}",
                 f"  files      : {self.n_files}",
                 f"  claims     : {self.n_claims} ratification assertion(s) "
                 f"inspected",
                 f"  violations : {len(self.violations)}"]
        for v in self.violations:
            lines.append("    " + format_violation(v))
        lines.append("OK" if self.ok else "FABRICATED RATIFICATION AUTHORITY")
        return "\n".join(lines)


def _iter_files(roots, excluded):
    """Every scannable file under ``roots``.  A missing root is a FAILURE."""
    files, errors = [], []
    for root in roots:
        if not os.path.exists(root):
            errors.append(_v("R6_UNPARSEABLE", root, "/", None,
                             "path does not exist"))
            continue
        if os.path.isfile(root):
            files.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in sorted(dirnames)
                           if d not in (".git", "__pycache__", ".pytest_cache",
                                        ".mypy_cache", ".ipynb_checkpoints",
                                        "node_modules", ".eggs")]
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() in SCAN_SUFFIXES:
                    files.append(os.path.join(dirpath, fn))
    keep = []
    for f in files:
        if any(fnmatch.fnmatch(f, pat) or fnmatch.fnmatch(os.path.basename(f),
                                                          pat)
               for pat in excluded):
            continue
        keep.append(f)
    return keep, errors


def scan_paths(roots, *, exclude=()):
    """Scan files and directories.  Returns a :class:`ScanResult`.

    FAIL-CLOSED in four places, each separately tested: a path that does not
    exist, a file that cannot be read, a file that does not parse, and a scan
    that inspected zero files.
    """
    roots = [str(r) for r in roots]
    exclude = [str(p) for p in exclude]
    files, violations = _iter_files(roots, exclude)
    counter = {"claims": 0}
    for path in files:
        violations.extend(scan_file(path, _counter=counter))
    if not files:
        violations.append(_v(
            "R8_NOTHING_SCANNED", ", ".join(roots) or "(no roots)", "/", None,
            "the scan inspected ZERO files. A guard that looked at nothing "
            "is not a passing guard."))
    return ScanResult(violations, files, counter["claims"], roots, exclude)


def enforce_no_fabricated_claims(roots, *, exclude=()):
    """``scan_paths`` that RAISES.  For use in tests and merge hooks."""
    res = scan_paths(roots, exclude=exclude)
    if not res.ok:
        raise FabricatedAuthorityError(res.report())
    return res


def enforce_no_fabricated_claims_data(obj, *, source="<data>"):
    """``scan_data`` that RAISES.  The in-process form, for a routine that is
    about to WRITE an artifact: refuse to emit it rather than emit a
    fabricated claim and rely on someone scanning the tree later."""
    bad = scan_data(obj, source=source)
    if bad:
        raise FabricatedAuthorityError(
            "fabricated ratification authority:\n  "
            + "\n  ".join(format_violation(v) for v in bad))
    return True


def main(argv=None):
    """``python -m CDDF_analysis.hbi_mcmc.ratification --check <paths>``.

    Exit 0 clean, 1 on any violation, 2 on usage error.  Runnable over a whole
    tree -- including branches on which this module does not exist -- because
    the two live fabricated-authority sites were on exactly such branches.
    """
    ap = argparse.ArgumentParser(
        prog="python -m CDDF_analysis.hbi_mcmc.ratification",
        description="Refuse any code or artifact that claims a ratification "
                    "no deciding authority granted.")
    ap.add_argument("--check", nargs="+", metavar="PATH", required=True,
                    help="files or directories to scan (.py, .json, .md)")
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                    help="skip paths matching GLOB. ECHOED in the report: an "
                         "exclusion that is not visible is a bypass.")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output")
    ap.add_argument("--stamp", action="store_true",
                    help="print the ratification stamp and exit")
    args = ap.parse_args(argv)

    if args.stamp:
        print(json.dumps(ratification_stamp(), indent=2, sort_keys=True))
        return 0

    res = scan_paths(args.check, exclude=args.exclude)
    if args.json:
        print(json.dumps(res.as_dict(), indent=2, sort_keys=True))
    else:
        print(res.report())
    return 0 if res.ok else 1


# ---------------------------------------------------------------------------
# 🔴 fail at IMPORT, not at review time.
# ---------------------------------------------------------------------------
enforce_authority_allow_list()

#: 🔴 and the WIDENED rules, on this module's own data, also at import: the
#: allow-list guard cannot see a claim written in any shape other than a
#: record, and this module's own stamp is a JSON block like any other.
_SELF_SCAN = scan_data(ratification_stamp(), source=__name__ + ".stamp",
                       _counter=_SELF_SCAN_COUNTER)
if _SELF_SCAN:
    raise FabricatedAuthorityError(
        "fabricated ratification authority in this module's own stamp:\n  "
        + "\n  ".join(format_violation(v) for v in _SELF_SCAN))
# 🔴 THE POWER CHECK, at import.  A stamp whose ratification blocks had been
# DELETED would also scan clean -- the fail-open shape ("deleting an entire
# artifact block leaves everything green") this project has hit repeatedly.
if _SELF_SCAN_COUNTER["claims"] < SELF_SCAN_MIN_CLAIMS:
    raise FabricatedAuthorityError(
        f"the import-time self-scan inspected only "
        f"{_SELF_SCAN_COUNTER['claims']} ratification claim(s) in this "
        f"module's own stamp, fewer than the {SELF_SCAN_MIN_CLAIMS} it must "
        f"carry ('pi_ratified_items' and 'ratified'). A clean scan of a stamp "
        f"with its ratification blocks removed is not a pass.")

if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())
