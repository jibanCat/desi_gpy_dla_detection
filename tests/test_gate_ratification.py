"""test_gate_ratification.py -- decision 8 (PI, 2026-07-29).

Three things were RATIFIED (the fail-closed framework, matched-configuration
SBC, chi2/dof <= 3) and two were explicitly DECLINED
(``ratio_span_by_z_max = 0.10``, ``ratio_span_by_snr_max = 0.15``).  This file
pins all five, plus the exact restatement of the ``|z| <= 5`` criterion.

WHAT EACH GROUP HAS TO PROVE, and why a weaker test would be worthless:

  1. the ratification RECORD discriminates.  A record that called everything
     ratified, or everything unratified, would satisfy a naive test.  So every
     test here checks BOTH directions.
  2. an UNRATIFIED tolerance is COMPUTED and REPORTED but does NOT gate.  The
     control is the mirror image: the same arm on a RATIFIED tolerance must
     still refuse.  Without that control the tests would pass on an arm that
     had simply been deleted.
  3. an UNMATCHED SBC must REFUSE to certify, and a MATCHED one must certify --
     otherwise the check is just a hardcoded False.
  4. the ``|z|`` definition is pinned NUMERICALLY, not by reading the
     docstring, and the docstring's own claims (sign, denominator, empty-bin
     convention, non-scale-freeness) are pinned as arithmetic facts.

SYNTHETIC PACKS ONLY.
"""
import copy
import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from CDDF_analysis.hbi_mcmc import evidence as EV           # noqa: E402
from CDDF_analysis.hbi_mcmc import forward_selftest as FS   # noqa: E402
from CDDF_analysis.hbi_mcmc import model_a as MA            # noqa: E402
from CDDF_analysis.hbi_mcmc import ratification as RAT      # noqa: E402
from CDDF_analysis.hbi_mcmc import run_posterior as RP      # noqa: E402
from CDDF_analysis.hbi_mcmc import sbc as SBC               # noqa: E402
from CDDF_analysis.hbi_mcmc.pack import synthetic_pack      # noqa: E402

_SPAN_TOLERANCES = ("ratio_span_by_z_max", "ratio_span_by_snr_max")
_Z_ARMS = ("z_total_max", "z_bin_max", "z_zbin_max", "z_snrbin_max")

#: EXACTLY the three things decision 8 ratified, verbatim from the decision:
#: "Ratify the fail-closed framework, matched-configuration SBC and chi2/dof
#: <= 3 closure requirement."  Nothing else may claim PI authority.
_PI_RATIFIED = ("chi2_dof_max", "fail_closed_framework",
                "matched_configuration_sbc")

#: the introduction commit of each |z| arm, as `git log -S<name>` reports it.
#: ``z_zbin_max`` / ``z_snrbin_max`` were added on 2026-07-29 in the SAME HUNK
#: as the two span numbers the PI declined -- they pre-date nothing.
_Z_ARM_INTRODUCED_BY = {
    "z_total_max": "f23961ec1e2cf47748a5a1b660205966a8d793f0",
    "z_bin_max": "f23961ec1e2cf47748a5a1b660205966a8d793f0",
    "z_zbin_max": "0e7fa0bd62d1f177126737fa32d1963e558b18d2",
    "z_snrbin_max": "0e7fa0bd62d1f177126737fa32d1963e558b18d2",
}


@pytest.fixture(scope="module")
def spack():
    """A pack with >=2 fine-z bins and >=2 SNR strata, so BOTH span arms are
    non-vacuous (on a 1-stratum grid ``ratio_span_by_snr`` is identically 0)."""
    return synthetic_pack(
        0, nhat_edges=np.round(np.arange(19.9, 20.4 + 1e-9, 0.1), 10),
        zf_edges=np.round(np.arange(2.0, 2.4 + 1e-9, 0.1), 10),
        zc_edges=np.array([2.0, 2.2, 2.4]),
        snr_edges=np.array([0.0, 3.0, np.inf]), n_molly_cells=3,
        fp_frac=0.15, t_true=np.array([0.2, -0.15]))


@pytest.fixture(scope="module")
def padded_spack():
    """The geometry decisions 3 and 4 ACTUALLY adopted: a true-N basis padded
    two bins BELOW the reporting floor (schema v1.1, ``n_pad_bins = 2``).

    Every matched-SBC test that ran only on an UNPADDED pack passed vacuously:
    with ``ntrue_edges == nhat_edges`` an omission of ``ntrue_edges`` from the
    match kwargs is invisible."""
    return synthetic_pack(
        0, nhat_edges=np.round(np.arange(19.9, 20.4 + 1e-9, 0.1), 10),
        ntrue_edges=np.round(np.arange(19.7, 20.4 + 1e-9, 0.1), 10),
        zf_edges=np.round(np.arange(2.0, 2.4 + 1e-9, 0.1), 10),
        zc_edges=np.array([2.0, 2.2, 2.4]),
        snr_edges=np.array([0.0, 3.0, np.inf]), n_molly_cells=3,
        fp_frac=0.15, t_true=np.array([0.2, -0.15]))


# ==========================================================================
# 1. THE RATIFICATION RECORD
# ==========================================================================

def test_the_three_ratified_criteria_are_recorded_with_date_and_authority():
    for key in _PI_RATIFIED:
        assert RAT.is_ratified(key), key
        rec = RAT.record(key)
        assert rec["status"] == "RATIFIED"
        assert rec["contributes_to_pass_fail"] is True
        assert rec["date"] == "2026-07-29"
        assert "PI" in rec["authority"]
        assert rec["statement"].strip()
        assert rec["applies_to"], f"{key} names no code it governs"


def test_EXACTLY_three_things_are_ratified_and_nothing_else_claims_PI():
    """🔴 THE DEFECT THIS PINS.  ``ratification.py`` used to record the four
    ``|z| <= 5`` arms as RATIFIED, dated 2026-07-29, ``authority="PI (project
    decision 8, 2026-07-29)"``, on the stated grounds that they "pre-date
    decision 8".  Decision 8 ratified three things and called ``|z| <= 5``
    MALFORMED AS STATED, asking for a restatement -- the opposite of ratifying
    it.  So the ratified set is EXACTLY three, and PI authority may be claimed
    by exactly those three."""
    assert set(RAT.ratified_names()) == set(_PI_RATIFIED), RAT.ratified_names()
    assert set(RAT.PI_RATIFIED_ITEMS) == set(_PI_RATIFIED)
    for key in _Z_ARMS:
        assert RAT.is_ratified(key) is False, (
            f"{key} claims ratification the PI never granted")


def test_no_record_anywhere_may_claim_PI_authority_off_the_allow_list():
    """The guard, not just the data.  ``audit_authority_claims`` must return
    the empty list for the shipped record AND must actually catch a violation
    -- otherwise it is a hardcoded pass."""
    assert RAT.audit_authority_claims() == []
    bad = dict(RAT.all_records())
    bad["z_bin_max"] = dict(bad["z_bin_max"],
                            authority="PI (project decision 8, 2026-07-29)")
    v = RAT.audit_authority_claims(bad)
    assert v and any("z_bin_max" in x for x in v), v
    # ... and the same for a brand-new entry somebody adds tomorrow
    bad2 = dict(RAT.all_records())
    bad2["invented_tomorrow_max"] = {"status": "RATIFIED", "authority": "PI",
                                     "contributes_to_pass_fail": True}
    assert any("invented_tomorrow_max" in x
               for x in RAT.audit_authority_claims(bad2))


def test_the_import_time_guard_refuses_a_fabricated_PI_authority_claim():
    """The allow-list must be ENFORCED, not merely reportable: a record that
    claims PI authority off the allow-list must raise, so the module cannot be
    imported in the state the defect left it in."""
    with pytest.raises(RAT.FabricatedAuthorityError) as ei:
        RAT.enforce_authority_allow_list(
            {"z_zbin_max": {"status": "RATIFIED", "authority": RAT.PI_AUTHORITY,
                            "contributes_to_pass_fail": True}})
    assert "z_zbin_max" in str(ei.value)
    RAT.enforce_authority_allow_list(RAT.all_records())     # no raise


# --------------------------------------------------------------------------
# 1b. 🔴 THE SAME FABRICATION, IN PROSE, IN THREE PLACES THE ALLOW-LIST
#     CANNOT SEE.
#
# ``enforce_authority_allow_list`` polices the RECORDS.  It cannot police an
# English sentence in a docstring or a stamp string, and after the retraction
# three of those still asserted the fabricated authority:
#   * the stamp's own top-level ``authority`` key, which v1 left unscoped -- the
#     exact mechanism by which the |z| arms acquired PI authority;
#   * ``forward_selftest.poisson_z``: "The criterion is nevertheless kept,
#     RATIFIED";
#   * ``d1_ladder``'s ``closes_criteria_note``: "All three are RATIFIED", of
#     z_total_max + z_bin_max + chi2_dof_max, of which exactly one is.
# --------------------------------------------------------------------------

def test_the_stamps_top_level_authority_states_its_own_SCOPE():
    """v1's mechanism, closed.  A bare ``authority: "PI (...)"`` at the top of
    the stamp reads as authorising the whole block, which is how four |z| arms
    became PI-ratified.  The stamp must scope it in the JSON itself, because a
    reader of the artifact cannot open ``ratification.py``."""
    st = RAT.ratification_stamp()
    assert st["authority"] == RAT.PI_AUTHORITY
    scope = st["authority_scope"]
    assert "pi_ratified_items" in scope
    assert "NOTHING ELSE" in scope.upper()
    # it must name the sections it does NOT cover, or it is not a scope
    for k in ("restated_not_ratified", "unratified"):
        assert k in scope, k


def test_no_docstring_or_stamp_string_calls_an_UNRATIFIED_criterion_RATIFIED():
    """PROSE-DRIFT GUARD.  For every name whose status is not RATIFIED, no
    module under ``hbi_mcmc`` may put that name and a bare ``RATIFIED`` in the
    same sentence.  Deliberately sentence-scoped: the modules must and do
    discuss these names next to the word (``RESTATED_NOT_RATIFIED``, "NOT
    RATIFIED", "is not ratified"), so the guard keys on the BARE claim."""
    import re
    from pathlib import Path
    mods = sorted((Path(RAT.__file__).parent).glob("*.py"))
    assert len(mods) > 5, mods
    unratified = set(RAT.RESTATED_NOT_RATIFIED) | set(RAT.UNRATIFIED)
    assert unratified                      # not vacuous
    # a sentence: text between terminators, with newlines flattened
    n_checked = 0
    for m in mods:
        if m.name == "ratification.py":
            continue                       # the record itself; tested above
        txt = re.sub(r"\s+", " ", m.read_text())
        for sent in re.split(r"(?<=[.!?])\s", txt):
            if "RATIFIED" not in sent:
                continue
            # the qualified forms are the CORRECT way to mention these names
            if re.search(r"NOT[ _]RATIFIED|not ratified|UNRATIFIED|"
                         r"RESTATED_NOT_RATIFIED|nobody ratified|"
                         r"no deciding authority", sent):
                continue
            n_checked += 1
            for name in unratified:
                assert name not in sent, (
                    f"{m.name}: calls {name} RATIFIED without qualification:\n"
                    f"{sent[:300]}")
    assert n_checked > 0, ("the guard inspected no unqualified RATIFIED "
                           "sentence at all -- it would pass vacuously")


def test_every_provenance_CLAIM_in_the_z_arm_records_checks_out_against_git():
    """A retraction whose own evidence is unverified is not a retraction.
    Each ``RESTATED_NOT_RATIFIED`` record asserts a 40-char introducing SHA, a
    date and a same-hunk relationship; all three are checked against the
    repository here rather than trusted.

    🔴 This is how the "by the same author" claim was caught: it was in the
    retracting commit's message AND in the record, and it is FALSE -- 0e7fa0b's
    git author is `panel5`, 88f2ecb's is `jibanmich`. No record may make an
    author claim, because none of them was checked."""
    import subprocess
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])

    def git(*a):
        return subprocess.check_output(["git", *a], cwd=root, text=True).strip()

    for key, rec in RAT.RESTATED_NOT_RATIFIED.items():
        sha = rec["introduced_by"]
        assert len(sha) == 40, (key, sha)
        # the SHA must exist and its date must be the one claimed
        assert git("show", "-s", "--format=%ad", "--date=short",
                   sha) == rec["introduced_date"], key
        # ... and no record may assert an author, since that was the one claim
        # nobody verified
        assert "same author" not in rec["note"] or "MEASURED" in rec["note"], key

    # the SAME-HUNK claim, verified as text: the four names on added lines of
    # ONE commit's diff of run_posterior.py
    sha = RAT.RESTATED_NOT_RATIFIED["z_zbin_max"]["introduced_by"]
    diff = git("show", sha, "--", "CDDF_analysis/hbi_mcmc/run_posterior.py")
    added = [l for l in diff.splitlines() if l.startswith("+")]
    for name in ("z_zbin_max", "z_snrbin_max",
                 "ratio_span_by_z_max", "ratio_span_by_snr_max"):
        assert any(f'"{name}"' in l for l in added), (name, sha)
    # and it is the EARLIEST such commit, i.e. the arm really does not pre-date
    hist = git("log", "--format=%H", "-Sz_zbin_max", "--",
               "CDDF_analysis/hbi_mcmc/run_posterior.py").split()
    assert hist[-1] == sha, (hist, sha)

    # the AUTHOR FACT itself, so the correction cannot silently drift back
    a_intro = git("show", "-s", "--format=%an", sha)
    a_stamp = git("show", "-s", "--format=%an",
                  "88f2ecb43eff2a8f2baa5df2535988c577a2ff3e")
    assert a_intro != a_stamp, (a_intro, a_stamp)
    note = RAT.RESTATED_NOT_RATIFIED["z_zbin_max"]["note"]
    assert a_intro in note and a_stamp in note, (a_intro, a_stamp)
    assert "NOT the same git author" in note


def test_the_RESTATEMENT_ITSELF_does_not_claim_to_have_been_ratified():
    """The sentence-scoped guard above cannot catch this one, and that is why it
    is here: ``poisson_z``'s docstring said "The criterion is nevertheless kept,
    RATIFIED" without naming any arm, so no name-based scan sees it.  Yet this
    docstring is what all four ``RESTATED_NOT_RATIFIED`` records point to as
    their statement -- it IS the restatement decision 8 asked for, and a
    restatement that calls itself ratified re-fabricates the authority the
    retraction removed."""
    doc = FS.poisson_z.__doc__
    # every |z| arm delegates its definition to this docstring ...
    for key in RAT.RESTATED_NOT_RATIFIED:
        assert "poisson_z" in RAT.record(key)["statement"], key
    # ... so this docstring may not call it ratified
    assert "kept, RATIFIED" not in doc
    assert "RATIFIED, with its purpose" not in doc
    # and it must state the true status positively, not merely omit the claim
    assert "RESTATED_NOT_RATIFIED" in doc
    assert "NOT RATIFIED" in doc
    assert "z_arms_gate_unratified" in doc


def test_d1_ladders_closes_note_is_DERIVED_from_the_record_not_asserted():
    """The hardcoded predecessor said "All three are RATIFIED" of three criteria
    of which exactly ONE is.  Derived, it cannot drift: change the record and
    the note changes with it."""
    from CDDF_analysis.hbi_mcmc import d1_ladder as D1
    note = D1._closes_criteria_note()
    assert "All three are RATIFIED" not in note
    # exactly one of the three is PI-ratified, and the note says which
    rat = [k for k in D1.CLOSES_CRITERIA if RAT.is_ratified(k)]
    assert rat == ["chi2_dof_max"], rat
    assert "PI-RATIFIED: chi2_dof_max." in note
    # ... and it names the two that gate without authority, with their status
    for k in ("z_total_max", "z_bin_max"):
        assert k in note.split("NOT RATIFIED BY ANY DECIDING AUTHORITY")[1], k
    assert "RESTATED_NOT_RATIFIED" in note
    assert "z_arms_gate_unratified" in note
    # DERIVATION, not coincidence: flip the record and the note must follow
    import unittest.mock as _m
    with _m.patch.object(RAT, "PI_RATIFIED_ITEMS",
                         RAT.PI_RATIFIED_ITEMS + ("z_total_max",)):
        with _m.patch.dict(RAT.RATIFIED,
                           {"z_total_max": dict(RAT.RATIFIED["chi2_dof_max"])}):
            note2 = D1._closes_criteria_note()
    assert "PI-RATIFIED: z_total_max, chi2_dof_max." in note2, note2
    assert note2 != note


def test_the_z_arms_are_RESTATED_NOT_RATIFIED_with_honest_provenance():
    """Decision 8 item 3, verbatim: "restate the malformed |z| <= 5 criterion
    with its exact mathematical definition".  The restatement was delivered
    (``forward_selftest.poisson_z``); the RESTATED form has not been ratified.
    Each arm must say so, must NOT name the PI as its authority, and must name
    the commit that actually introduced it."""
    for key in _Z_ARMS:
        rec = RAT.record(key)
        assert rec["status"] == "RESTATED_NOT_RATIFIED", (key, rec["status"])
        assert "PI" not in rec["authority"], (key, rec["authority"])
        assert "MALFORMED" in rec["pi_disposition"].upper(), key
        assert "NOT RATIFIED" in rec["pi_disposition"].upper(), key
        assert rec["restatement_lives_in"].endswith("poisson_z")
        assert rec["introduced_by"] == _Z_ARM_INTRODUCED_BY[key], key
        assert len(rec["introduced_by"]) == 40


def test_the_two_new_z_arms_do_not_predate_decision_8_and_the_record_says_so():
    """The false premise, pinned.  ``z_zbin_max`` and ``z_snrbin_max`` were
    added in the SAME HUNK as the declined ``ratio_span_*`` pair, by the same
    author, on the same day.  A record that calls them "conventional arms that
    pre-date decision 8" is false on the git history."""
    for key in ("z_zbin_max", "z_snrbin_max"):
        rec = RAT.record(key)
        assert rec["predates_decision_8"] is False, key
        assert rec["introduced_same_hunk_as"] == list(_SPAN_TOLERANCES), key
        assert rec["introduced_date"] == "2026-07-29", key
    for key in ("z_total_max", "z_bin_max"):
        rec = RAT.record(key)
        assert rec["predates_decision_8"] is True, key
        assert rec["introduced_same_hunk_as"] == [], key
        assert rec["introduced_date"] == "2026-07-28", key
        # pre-dating is NOT ratification, and the record must not imply it
        assert rec["status"] == "RESTATED_NOT_RATIFIED"


def test_the_recorded_introduction_commits_are_the_ones_git_reports():
    """Provenance verified against the repository, not asserted.  ``git log
    -S<name>`` lists newest-first; its LAST entry is the commit that
    introduced the string."""
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for key in _Z_ARMS:
        out = subprocess.run(
            ["git", "log", "--format=%H", f"-S{key}", "--",
             "CDDF_analysis/hbi_mcmc/run_posterior.py"],
            cwd=str(root), capture_output=True, text=True, check=True).stdout
        shas = [s for s in out.split() if s]
        assert shas, key
        assert shas[-1] == RAT.record(key)["introduced_by"], (
            key, shas[-1], RAT.record(key)["introduced_by"])


def test_the_four_z_arms_GATE_and_the_record_admits_it_without_claiming_authority():
    """The honest record must state the UNCOMFORTABLE fact rather than resolve
    it: these four numbers DO refuse work (``fails.append`` in
    ``forward_closure_gate``) and no deciding authority has ratified them.  A
    record that reported ``contributes_to_pass_fail=False`` here would be as
    false as the fabricated-PI one, in the other direction."""
    for key in _Z_ARMS:
        rec = RAT.record(key)
        assert rec["contributes_to_pass_fail"] is True, key
        assert RAT.gates(key) is True, key
        assert RAT.is_ratified(key) is False, key
    assert set(RAT.unratified_but_gating_names()) == set(_Z_ARMS)
    st = RAT.ratification_stamp()
    assert set(st["unratified_but_gating"]) == set(_Z_ARMS)
    assert "no deciding authority" in st["unratified_but_gating_note"].lower()
    assert "PI DECISION REQUIRED" in st["open_pi_decisions_note"].upper()


def test_the_two_declined_tolerances_are_unratified_and_report_only():
    for key in _SPAN_TOLERANCES:
        assert RAT.is_ratified(key) is False, key
        rec = RAT.record(key)
        assert rec["status"] == "UNRATIFIED"
        assert rec["contributes_to_pass_fail"] is False
        assert rec["effect"] == "REPORT_ONLY_DOES_NOT_GATE"
        assert "declined" in rec["declined_by"].lower() or "PI" in rec["declined_by"]
        assert rec["calibration_spec"].endswith(
            "docs/ratio_span_calibration_spec.md")


def test_the_record_discriminates_and_is_fail_closed_on_unknown_names():
    """A record that said RATIFIED (or UNRATIFIED) for everything would pass a
    one-sided test.  Both sets must be non-empty, disjoint, and an unknown name
    must inherit NOTHING from its neighbours in GATE."""
    r, u = set(RAT.ratified_names()), set(RAT.unratified_names())
    s = set(RAT.restated_not_ratified_names())
    assert r and u and s
    assert not (r & u) and not (r & s) and not (u & s)
    assert u == set(_SPAN_TOLERANCES)
    assert s == set(_Z_ARMS)
    assert r == set(_PI_RATIFIED)
    assert "chi2_dof_max" in r
    rec = RAT.record("some_tolerance_added_tomorrow")
    assert rec["status"] == "UNKNOWN"
    assert rec["contributes_to_pass_fail"] is False
    assert RAT.is_ratified("some_tolerance_added_tomorrow") is False


def test_every_tolerance_in_GATE_has_a_ratification_record():
    """The point of the module: no number in a production fail-closed gate may
    be unaccounted for."""
    for key in RP.GATE:
        assert RAT.record(key)["status"] in (
            "RATIFIED", "UNRATIFIED", "RESTATED_NOT_RATIFIED"), key
        # and every gating number says, explicitly, who authorised it
        rec = RAT.record(key)
        if rec["contributes_to_pass_fail"]:
            assert rec["authority"].strip(), key


def test_the_calibration_spec_exists_and_states_the_unratified_status():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "docs" / \
        "ratio_span_calibration_spec.md"
    txt = p.read_text()
    assert "UNRATIFIED" in txt
    for key in _SPAN_TOLERANCES:
        assert key in txt, key
    # it must actually contain a null-distribution definition and a
    # false-alarm-rate procedure, not just a promise to write one
    assert "Poisson" in txt and "false-alarm" in txt
    assert "ratio_span_null" in txt


# ==========================================================================
# 2. UNRATIFIED => COMPUTED, REPORTED, DOES NOT GATE
# ==========================================================================

def _flat_tab(*, ratio_by_z=(1.0, 1.0), ratio_by_snr=(1.0, 1.0), z=0.0):
    """A ratio table with clean total/N-marginal and a controllable span."""
    return {
        "total": {"mu": 1000.0, "obs": 1000.0, "ratio": 1.0, "z": 0.0,
                  "chi2_dof": 0.0, "n_gate_bins": 2},
        "by_nhat": [{"lo": 19.9 + 0.1 * i, "hi": 20.0 + 0.1 * i, "mu": 500.0,
                     "obs": 500.0, "ratio": 1.0, "z": 0.0} for i in range(2)],
        "by_z": [{"lo": 2.0 + 0.1 * i, "hi": 2.1 + 0.1 * i, "mu": 500.0 * r,
                  "obs": 500.0, "ratio": r, "z": (z if i == 0 else -z)}
                 for i, r in enumerate(ratio_by_z)],
        "by_snr": [{"s": i, "mu": 500.0 * r, "obs": 500.0, "ratio": r,
                    "z": (z if i == 0 else -z)}
                   for i, r in enumerate(ratio_by_snr)],
    }


@pytest.fixture
def fake_fold(monkeypatch):
    def _install(tab):
        monkeypatch.setattr(FS, "selftest", lambda *a, **k: {"mu": None})
        monkeypatch.setattr(FS, "ratio_tables", lambda *a, **k: tab)
    return _install


def test_an_unratified_span_exceedance_is_an_ADVISORY_not_a_failure(
        spack, fake_fold):
    """The behavioural core of decision 8.  A span of 0.44 is >4x the proposed
    0.10; before this change it FAILED the run."""
    fake_fold(_flat_tab(ratio_by_z=(1.22, 0.78)))
    g = RP.forward_closure_gate(spack)
    assert g["ratio_span_by_z"] == pytest.approx(0.44)      # COMPUTED
    assert g["pass"] is True, g["failures"]                 # DOES NOT GATE
    assert not any("ratio_span" in f for f in g["failures"]), g["failures"]
    adv = [a for a in g["advisories"] if "ratio_span_by_z" in a]
    assert adv, g["advisories"]                             # REPORTED
    assert "UNRATIFIED" in adv[0].upper()
    assert "does not block" in adv[0].lower()
    assert "docs/ratio_span_calibration_spec.md" in adv[0]


def test_a_RATIFIED_span_tolerance_WOULD_still_gate(spack, fake_fold,
                                                    monkeypatch):
    """THE CONTROL.  Without this, the test above would also pass on an arm
    that had simply been deleted.  Ratify the tolerance and the identical
    exceedance must become a hard failure."""
    monkeypatch.setitem(
        RAT.RATIFIED, "ratio_span_by_z_max",
        {"status": "RATIFIED", "statement": "test", "applies_to": [],
         "date": "2026-07-29", "authority": "test", "note": "",
         "contributes_to_pass_fail": True})
    fake_fold(_flat_tab(ratio_by_z=(1.22, 0.78)))
    g = RP.forward_closure_gate(spack)
    assert g["pass"] is False, g
    assert any("ratio_span_by_z" in f for f in g["failures"]), g["failures"]
    assert not any("ratio_span_by_z" in a for a in g["advisories"])


def test_the_z_marginal_arms_still_gate(spack, fake_fold):
    """Decision 8 moved only the two SPAN numbers.  ``z_zbin_max`` is
    UNRATIFIED-BUT-GATING (see the provenance tests above): it still refuses a
    30-sigma z-marginal residual, and nothing in this stream may silently
    disarm it."""
    fake_fold(_flat_tab(z=30.0))
    g = RP.forward_closure_gate(spack)
    assert g["pass"] is False, g
    assert any("by_z" in f and "max|z|" in f for f in g["failures"]), \
        g["failures"]


def test_the_gate_report_and_the_stamp_carry_the_ratification_state(spack):
    g = RP.forward_closure_gate(spack)
    assert set(g["gate_tolerances_unratified"]) == set(_SPAN_TOLERANCES)
    assert set(g["gate_tolerances_ratified"]) == set(_PI_RATIFIED)
    assert set(g["gate_tolerances_unratified_but_gating"]) == set(_Z_ARMS)
    assert g["unratified_effect"] == "REPORT_ONLY_DOES_NOT_GATE"
    assert g["ratification"]["ratification_date"] == "2026-07-29"
    md = RP.stamp_metadata(
        code_commit="0" * 40, code_dirty=False,
        cfg=MA.ModelAConfig(num_warmup=1, num_samples=1, num_chains=1),
        args={"rederive": "x"}, gate_report=g,
        estimand="POSTERIOR_MEDIAN_CI", paper_facing=False)
    blob = json.dumps(md, default=RP._jsonable)
    assert md["ratification"]["authority"] == RAT.RATIFYING_AUTHORITY
    assert set(md["ratification"]["unratified"]) == set(_SPAN_TOLERANCES)
    assert "REPORT_ONLY_DOES_NOT_GATE" in blob
    for key in _SPAN_TOLERANCES:
        assert key in blob, key


def test_the_evidence_verdict_and_artifact_carry_the_ratification_state():
    blocks = {b: {"checks": {"x": True}, "incomplete": []}
              for b in EV.REQUIRED_BLOCKS}
    blocks["coverage_sbc"]["checks"]["sbc_configuration_matches_run"] = True
    g = EV.gate(blocks)
    assert g["ratification"]["ratification_date"] == "2026-07-29"
    assert set(g["ratification"]["unratified"]) == set(_SPAN_TOLERANCES)
    ev = EV.assemble_evidence(blocks)
    assert ev["ratification"]["authority"] == RAT.RATIFYING_AUTHORITY


def test_the_d1_ladder_stamp_names_which_tolerances_decided_closes():
    """The 60-config ladder artifact dumps the WHOLE of GATE into
    ``metadata.gate_tolerances``, which implies every number in it was
    load-bearing.  Only three were.  The stamp must distinguish them, or a
    reader concludes the ladder was gated on an uncalibrated tolerance."""
    from CDDF_analysis.hbi_mcmc import d1_ladder as D
    src = __import__("inspect").getsource(D)
    # the criteria are asserted as a VALUE, not as a source line.  The earlier
    # form grepped for the literal ``closes_criteria=["z_total_max", ...]`` and
    # so broke the moment the list was named ``CLOSES_CRITERIA`` -- a test that
    # pins formatting instead of behaviour.  The value is the claim.
    assert list(D.CLOSES_CRITERIA) == ["z_total_max", "z_bin_max",
                                       "chi2_dof_max"]
    assert "closes_criteria=list(CLOSES_CRITERIA)" in src, (
        "the ladder stamp does not name its closes criteria")
    assert "gate_tolerances_unratified=list(_RAT().unratified_names())" in src
    assert "ratification=_RAT().ratification_stamp()" in src
    # and the criteria it names are exactly the ones the code ANDs together
    closes_src = src[src.index("closes=bool("):]
    closes_src = closes_src[:closes_src.index(")," + "\n")]
    for k in ("z_total_max", "z_bin_max", "chi2_dof_max"):
        assert k in closes_src, k
    for k in _SPAN_TOLERANCES:
        assert k not in closes_src, (
            f"{k} is UNRATIFIED and must not decide the ladder's `closes`")
    # the ratified/unratified split it stamps must be the real one
    assert set(D._RAT().unratified_names()) == set(_SPAN_TOLERANCES)


# ==========================================================================
# 3. THE SPAN STATISTIC AND ITS PROSPECTIVE CALIBRATION
# ==========================================================================

def test_ratio_span_is_a_max_minus_min_over_obs_positive_rows():
    rows = [{"obs": 10.0, "ratio": 1.30}, {"obs": 10.0, "ratio": 0.90},
            {"obs": 10.0, "ratio": 1.00},
            {"obs": 0.0, "ratio": 99.0},          # dropped: obs == 0
            {"obs": 5.0, "ratio": float("nan")}]  # dropped: not finite
    sp = FS.ratio_span(rows)
    assert sp["span"] == pytest.approx(0.40)
    assert (sp["lo"], sp["hi"]) == (0.90, 1.30)
    assert sp["n_rows_used"] == 3 and sp["vacuous"] is False
    # a RANGE, not a dispersion: adding an interior row must not change it
    sp2 = FS.ratio_span(rows + [{"obs": 10.0, "ratio": 1.10}])
    assert sp2["span"] == pytest.approx(0.40)


def test_ratio_span_is_vacuously_zero_below_two_usable_rows():
    """The 1-stratum-grid hole, pinned: the by_snr arm CANNOT fire there, and a
    vacuous 0 must be labelled as such."""
    for rows in ([], [{"obs": 10.0, "ratio": 1.7}],
                 [{"obs": 10.0, "ratio": 1.7}, {"obs": 0.0, "ratio": 0.1}]):
        sp = FS.ratio_span(rows)
        assert sp["span"] == 0.0
        assert sp["vacuous"] is True


def test_the_gate_uses_the_named_ratio_span_definition(spack, monkeypatch):
    """ONE definition.  If the gate recomputed the span inline, this stub
    would not be able to change its answer."""
    fake = {"span": 7.5, "lo": 1.0, "hi": 8.5, "n_rows_used": 2,
            "vacuous": False}
    monkeypatch.setattr(FS, "ratio_span", lambda rows: dict(fake))
    g = RP.forward_closure_gate(spack)
    assert g["ratio_span_by_z"] == 7.5 and g["ratio_span_by_snr"] == 7.5
    assert g["ratio_span_by_z_detail"] == fake


def test_ratio_span_null_measures_the_false_alarm_rate_of_the_proposed_numbers(
        spack):
    """The prospective calibration, and the empirical justification for the
    PI's refusal: under a null in which the forward model is EXACTLY right,
    ``ratio_span_by_z_max = 0.10`` refuses a large fraction of runs while
    ``ratio_span_by_snr_max = 0.15`` never fires.  A matched pair of
    tolerances cannot have false-alarm rates orders of magnitude apart.

    🔴 ON THIS PACK (5 x 4 x 2, FOUR fine-z rows).  The magnitude does NOT
    transfer to a 15-row production arm -- see the geometry tests below."""
    nul = FS.ratio_span_null(spack, n_draws=4000, seed=1)
    az, asn = nul["arms"]["by_z"], nul["arms"]["by_snr"]
    assert az["n_rows"] == 4 and asn["n_rows"] == 2
    # the proposed by_z threshold sits BELOW the null 95th percentile
    assert az["quantiles"]["0.95"] > RP.GATE["ratio_span_by_z_max"], az
    # ... while the proposed by_snr threshold sits ABOVE the null 99th
    assert asn["quantiles"]["0.99"] < RP.GATE["ratio_span_by_snr_max"], asn
    # monotone quantiles and a stated omission list
    qs = [az["quantiles"][k] for k in ("0.5", "0.9", "0.95", "0.99", "0.999")]
    assert qs == sorted(qs)
    assert "LOWER bound" in nul["null_note"]
    assert "ANTI-conservative" in nul["null_note"]


def test_ratio_span_null_is_reproducible_and_seed_sensitive(spack):
    a = FS.ratio_span_null(spack, n_draws=500, seed=3)["arms"]["by_z"]
    b = FS.ratio_span_null(spack, n_draws=500, seed=3)["arms"]["by_z"]
    c = FS.ratio_span_null(spack, n_draws=500, seed=4)["arms"]["by_z"]
    assert a["quantiles"] == b["quantiles"]
    assert a["quantiles"] != c["quantiles"]


def test_the_calibration_report_is_a_committed_stamped_routine():
    """Project rule: a number that is quoted anywhere comes from a committed
    routine with a full 40-char SHA, never a scratch script."""
    rep = FS.ratio_span_null_report(n_draws=1000, seed=1)
    md = rep["metadata"]
    assert len(md["code_commit"]) == 40 or md["code_commit"] == "unknown"
    assert md["paper_facing"] is False
    assert "SYNTHETIC ONLY" in md["scope"]
    assert md["ratification"]["ratification_date"] == "2026-07-29"
    assert "--ratio-span-null" in md["rederive"]
    for arm, key in (("by_z", "ratio_span_by_z_max"),
                     ("by_snr", "ratio_span_by_snr_max")):
        e = rep["null"]["arms"][arm]
        assert e["proposed_threshold_name"] == key
        assert e["proposed_threshold"] == RP.GATE[key]
        assert e["ratification_status"] == "UNRATIFIED"
        assert 0.0 <= e["measured_false_alarm_rate"] <= 1.0
    assert "NO THRESHOLD IS PROPOSED" in rep["verdict"]


def test_the_committed_calibration_artifact_agrees_with_the_spec_table():
    """Guards against doc/artifact drift -- the exact failure mode that made
    an earlier stream's quoted numbers unreproducible.  Every headline number
    in the spec's §4 table must be present in the committed artifact."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    art = json.loads(
        (root / "CDDF_analysis" / "hbi_mcmc"
         / "ratio_span_null_calibration.json").read_text())
    txt = (root / "docs" / "ratio_span_calibration_spec.md").read_text()
    assert len(art["metadata"]["code_commit"]) == 40
    for arm in ("by_z", "by_snr"):
        e = art["null"]["arms"][arm]
        assert f"{e['measured_false_alarm_rate']:.4f}" in txt, (arm, e)
        for q in ("0.5", "0.95", "0.99"):
            assert f"{e['quantiles'][q]:.4f}" in txt, (arm, q)
    assert f"{art['pack']['total_mu']:.2f}" in txt
    assert str(int(art["pack"]["total_obs"])) in txt
    # ... and the PRODUCTION-GEOMETRY table too (defect 2)
    for gname, g in art["geometries"].items():
        assert gname in txt, gname
        for arm in ("by_z", "by_snr"):
            far = g["arms"][arm]["measured_false_alarm_rate"]
            assert f"{far:.4f}" in txt, (gname, arm, far)


# ==========================================================================
# 3b. 🔴 DEFECT 2 -- THE 34% DOES NOT TRANSFER BETWEEN GRIDS
#
# ``ratio_span_by_z_max = 0.10 refuses 34% of perfectly correct forward
# models'' was measured on a 5x4x2 synthetic pack whose by_z arm has FOUR rows,
# and was then quoted unqualified in the spec intro, in the artifact verdict,
# in two commit messages and in a report.  The spec's own §1.1 item 1 says a
# range statistic's null grows with the row count and "a 15-bin fine-z arm and
# a 4-bin one do not share a threshold".
# ==========================================================================

_GEOM_N_DRAWS = 2000


@pytest.fixture(scope="module")
def geom_report():
    """One (measured ~3 min) calibration report, shared by the geometry tests."""
    return FS.ratio_span_null_report(n_draws=_GEOM_N_DRAWS, seed=1)


def test_the_false_alarm_rate_is_measured_PER_GEOMETRY_and_they_do_not_agree(
        geom_report):
    """The defect, as arithmetic: the same threshold, the same statistic, the
    same null -- and a false-alarm rate that differs by a factor ~4 between the
    calibration pack and production.  A single unqualified number is therefore
    not a property of the tolerance."""
    g = geom_report["geometries"]
    assert set(g) >= {"calib_5x4x2", "prod_17x15x8", "prod_29x15x8"}
    assert g["calib_5x4x2"]["arms"]["by_z"]["n_rows"] == 4
    assert g["prod_17x15x8"]["arms"]["by_z"]["n_rows"] == 15
    assert g["prod_29x15x8"]["arms"]["by_z"]["n_rows"] == 15
    far = {k: v["arms"]["by_z"]["measured_false_alarm_rate"]
           for k, v in g.items()}
    # the 5x4x2 number is the ~0.34 that was quoted unqualified ...
    assert 0.28 < far["calib_5x4x2"] < 0.42, far
    # ... and production is several times smaller, on BOTH production grids
    for k in ("prod_17x15x8", "prod_29x15x8"):
        assert 0.03 < far[k] < 0.15, (k, far[k])
        assert far[k] < 0.4 * far["calib_5x4x2"], far
    # the by_snr arm is inert at production scale -- the pair-mismatch that
    # justified the PI's refusal SURVIVES the geometry change
    for k in ("prod_17x15x8", "prod_29x15x8"):
        assert g[k]["arms"]["by_snr"]["measured_false_alarm_rate"] == 0.0, k


def test_the_artifact_CORRECTS_the_unqualified_34_percent_and_names_the_grids(
        geom_report):
    txt = geom_report["geometry_correction"] + " " + geom_report["verdict"]
    assert "CORRECTION" in txt
    assert "0.3434" in txt and "5x4x2" in txt
    for gname in ("prod_17x15x8", "prod_29x15x8"):
        assert gname in txt, gname
    assert "must name its grid" in txt
    assert geom_report["schema"] == "ratio_span_null_calibration/v2"


def test_no_document_quotes_a_span_false_alarm_rate_without_its_grid():
    """DOC-DRIFT GUARD with teeth.  Every place that quotes the 34% / 0.3434 /
    'a third' must name the geometry it was measured on, in the same paragraph.
    This is the exact failure mode of defect 2: a correct number, quoted where
    it does not apply."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    targets = [root / "docs" / "ratio_span_calibration_spec.md",
               root / "CDDF_analysis" / "hbi_mcmc" / "ratification.py",
               root / "CDDF_analysis" / "hbi_mcmc" / "forward_selftest.py"]
    pat = re.compile(r"0\.3434|34%|34 %|third of perfectly|a third of")
    # forms that genuinely NAME the geometry.  Deliberately narrow: "on the
    # synthetic pack" is not a geometry, and must not satisfy this guard.
    grid_pat = re.compile(r"5\s*[x×]\s*4\s*[x×]\s*2|4 z rows|"
                          r"[Ff]our fine-z|FOUR fine-z|4 rows|FOUR rows|"
                          r"[Ff]our[- ]row|4-row|calib_5x4x2")
    n_quotes = 0
    for p in targets:
        txt = p.read_text()
        # paragraphs, so "in the same breath" is testable
        for para in re.split(r"\n\s*\n", txt):
            if pat.search(para):
                n_quotes += 1
                assert grid_pat.search(para), (
                    f"{p.name}: quotes the span false-alarm rate without its "
                    f"geometry:\n{para[:400]}")
    assert n_quotes >= 2, ("the guard found nothing to check -- it would pass "
                           "vacuously if the quotes were merely deleted")


def test_the_power_curve_compares_BOTH_guards_on_the_SAME_injected_tilt(
        geom_report):
    """🔴 THE MEASUREMENT THE DISARM DECISION TURNS ON, and which did not exist.

    Disarming the span arms is only harmless if the still-armed ``z_zbin_max``
    catches what they would have caught.  Exposed to the same injected
    peak-to-peak z-tilt at PRODUCTION geometry, it does not: the span arm is
    the more sensitive of the two, so disarming it DOES cost real detection
    power.  That is a PI tradeoff, and it must be measured rather than
    asserted in either direction."""
    for gname in ("prod_17x15x8", "prod_29x15x8"):
        p = geom_report["power"][gname]
        assert p["n_rows_by_z"] == 15
        assert p["z_threshold"] == 5.0
        # the d = 0 row IS the false-alarm rate, so both live on one curve
        assert p["curve"][0]["tilt_peak_to_peak"] == 0.0
        # NOT ``== 0.0``: this is a Monte-Carlo rate and exact equality was a
        # coincidence of this fixture's n_draws.  MEASURED: 0.0 at n_draws =
        # 2000 (the fixture) but 0.00025 at 4000 (the committed artifact's
        # power block), so the strict form would have failed the moment the
        # artifact was regenerated at production draw counts.  The claim being
        # made is "inert at the null", and 2e-3 states it without pinning an
        # accident; the discrimination against the span arm's ~0.085 below is
        # a factor >40 either way, so no power is given up.
        assert p["false_alarm_z_arm"] <= 2e-3, p["false_alarm_z_arm"]
        assert 0.03 < p["false_alarm_span_arm"] < 0.15
        assert p["false_alarm_span_arm"] > 20 * max(p["false_alarm_z_arm"],
                                                    1e-4)
        # power rises with the injected tilt for BOTH arms (not a flat curve)
        assert p["curve"][-1]["p_z_arm_fires"] > 0.9
        assert p["curve"][-1]["p_span_arm_fires"] > 0.9
        # the span arm detects a SMALLER tilt than the z arm, at both levels
        assert p["span_arm_d90"] < p["z_arm_d90"], (gname, p)
        assert p["span_arm_d50"] < p["z_arm_d50"], (gname, p)
        # and a CALIBRATED span threshold (FAR ~ 0.005) is still the more
        # sensitive guard -- which is what makes spec option A actionable
        c = p["calibrated"]
        assert c["false_alarm_span_arm"] <= 0.02, c
        assert c["span_threshold"] > RP.GATE["ratio_span_by_z_max"]
        assert c["span_arm_d90"] < p["z_arm_d90"], (gname, c)


def test_ratio_span_power_null_row_reproduces_the_null_false_alarm_rate(spack):
    """The two routines must agree where they overlap, or the power curve is
    measuring something else."""
    pw = FS.ratio_span_power(spack, tilts=(0.0,), n_draws=3000, seed=5)
    nul = FS.ratio_span_null(spack, n_draws=3000, seed=5)
    thr = RP.GATE["ratio_span_by_z_max"]
    q = nul["arms"]["by_z"]["quantiles"]
    # the null's own quantiles bracket the measured false-alarm rate: if
    # FAR ~ 0.35 then the threshold sits between q50 and q95
    assert q["0.5"] < thr < q["0.95"]
    assert 0.25 < pw["false_alarm_span_arm"] < 0.45, pw


def test_the_disarm_decision_is_recorded_as_an_OPEN_PI_TRADEOFF(geom_report):
    """My instruction was that an unratified tolerance must not gate -- and
    that if disarming removes the only guard against the standing z-marginal
    defect, that must be said plainly and handed back, not resolved here."""
    d = RAT.OPEN_PI_DECISIONS["span_arms_disarmed"]
    assert "PI" in RAT.OPEN_PI_DECISIONS_NOTE
    assert "5x4x2" in d["measured_tradeoff"]
    assert "0.0894" in d["measured_tradeoff"] or "0.08" in d["measured_tradeoff"]
    assert d["what_the_code_does_meanwhile"]
    # the artifact must point at the tradeoff rather than settle it
    assert "PI TRADEOFF" in geom_report["verdict"]
    assert "not resolved here" in geom_report["verdict"]
    assert "OPEN_PI_DECISIONS" in geom_report["verdict"]


# ==========================================================================
# 4. THE EXACT |z| <= 5 DEFINITION  (decision 8, item 3)
# ==========================================================================

def test_poisson_z_is_the_poisson_score_residual_pinned_numerically():
    """z = (obs - mu) / sqrt(mu).  Every clause pinned as arithmetic:
    the numerator's SIGN, and that the denominator is the PREDICTED mean --
    not the observed count, and not a pooled or fitted variance."""
    assert FS.poisson_z(100.0, 110.0) == pytest.approx(1.0)   # (110-100)/10
    assert FS.poisson_z(100.0, 90.0) == pytest.approx(-1.0)
    assert FS.poisson_z(100.0, 100.0) == 0.0
    # SIGN: z > 0 <=> the model UNDER-predicts <=> ratio mu/obs < 1
    assert FS.poisson_z(90.0, 100.0) > 0 and 90.0 / 100.0 < 1
    # denominator is sqrt(mu), NOT sqrt(obs): the two differ, and it is mu
    assert FS.poisson_z(400.0, 100.0) == pytest.approx((100 - 400) / 20.0)
    assert FS.poisson_z(400.0, 100.0) != pytest.approx((100 - 400) / 10.0)
    # vectorised, elementwise, no pooling
    got = FS.poisson_z([100.0, 400.0], [110.0, 380.0])
    assert np.allclose(got, [1.0, -1.0])


def test_poisson_z_empty_and_zero_prediction_conventions():
    """mu == 0 with obs > 0 must produce a HUGE finite z (so the gate FAILS
    rather than raising), from the documented 1e-12 variance floor; the
    all-zero cell is exactly 0 and is dropped by the gate as obs == 0."""
    z = FS.poisson_z(0.0, 7.0)
    assert np.isfinite(z) and z == pytest.approx(7.0 / 1e-6)
    assert FS.poisson_z(0.0, 0.0) == 0.0
    assert FS.poisson_z(100.0, 0.0) == pytest.approx(-10.0)


def test_the_z_definition_is_not_scale_free_which_the_docstring_asserts():
    """The docstring's central claim -- a fixed |z| <= 5 tightens without limit
    as counts grow -- as arithmetic.  A 10% under-prediction is invisible at
    mu = 100 and a 5-sigma refusal at mu = 10000."""
    for mu, expect_fires in ((100.0, False), (10000.0, True)):
        obs = mu / 0.9                       # mu = 0.9 * obs  (10% low)
        z = abs(FS.poisson_z(mu, obs))
        assert bool(z > 5.0) is expect_fires, (mu, z)
    # the crossing scale is mu ~ 25/d^2 = 2500 for d = 0.1
    assert abs(FS.poisson_z(2500.0, 2500.0 / 0.9)) == pytest.approx(
        5.0, rel=0.15)


def test_poisson_z_docstring_states_every_clause_the_PI_asked_for():
    """Decision 8 item 3 asked for the definition to live in the code as the
    docstring of the function that computes it.  A missing clause here is the
    definition being incomplete, which is the defect being fixed."""
    d = FS.poisson_z.__doc__
    assert d
    for clause in ("(obs - mu) / sqrt(max(mu, 1e-12))",
                   "WHICH RATIO / WHICH ESTIMATOR", "SIGN", "OVER WHAT ROWS",
                   "WHAT IS IN THE DENOMINATOR", "NO nuisance uncertainty",
                   "EMPTY AND ZERO-PREDICTION BINS", "CHI2/DOF",
                   "WHY 5 IS NOT SCALE-FREE", "NOT multiplicity-corrected"):
        assert clause in d, f"poisson_z docstring is missing: {clause!r}"


def test_the_docstrings_worked_example_matches_the_committed_artifact():
    """The ``|z|`` docstring justifies keeping a fixed threshold of 5 by
    pointing at an observed order-of-magnitude failure.  If that example does
    not reproduce from a committed artifact it is rhetoric, and this project
    has been burned by exactly that.  (The earlier draft also mislabelled the
    pack as 'v1.1' when the artifact records n_pad_bins=0.)"""
    from pathlib import Path
    art = json.loads(
        (Path(__file__).resolve().parents[1] / "CDDF_analysis" / "hbi_mcmc"
         / "rung9_forward_selftest.json").read_text())
    e = art["mocks"]["2lpt0"]
    assert e["n_pad_bins"] == 0, "the docstring's 'UNPADDED' claim"
    tot = e["clamp_both"]["total"]
    worst = max(e["clamp_both"]["by_nhat"], key=lambda b: abs(b["z"]))
    assert tot["z"] == pytest.approx(93.3, abs=0.05)
    assert worst["z"] == pytest.approx(216.4, abs=0.05)
    assert worst["lo"] == pytest.approx(19.5)
    d = FS.poisson_z.__doc__
    assert "+93.3" in d and "+216.4" in d and "n_pad_bins=0" in d


def test_ratio_tables_and_the_gate_use_poisson_z_and_nothing_else(
        spack, monkeypatch):
    """If ratio_tables recomputed z inline, negating the named function could
    not flip the table."""
    monkeypatch.setattr(FS, "poisson_z",
                        lambda mu, obs: -np.asarray(
                            (np.asarray(obs, float) - np.asarray(mu, float))
                            / np.sqrt(np.maximum(np.asarray(mu, float), 1e-12))))
    res = FS.selftest(spack, resp_clamp="both")
    tab = FS.ratio_tables(res, spack)
    monkeypatch.undo()
    res2 = FS.selftest(spack, resp_clamp="both")
    tab2 = FS.ratio_tables(res2, spack)
    assert tab["total"]["z"] == pytest.approx(-tab2["total"]["z"])
    assert tab["total"]["z"] != 0.0


def test_chi2_dof_is_sum_z_squared_over_kept_bins_with_no_parameter_penalty(
        spack):
    """The RATIFIED chi2/dof <= 3: dof is the NUMBER OF KEPT BINS, not
    n_bins - n_params (the truth fold estimates nothing)."""
    assert RP.GATE["chi2_dof_max"] == 3.0
    res = FS.selftest(spack, resp_clamp="both")
    tab = FS.ratio_tables(res, spack)
    floor = float(np.asarray(spack.nhat_edges, float)[0])
    kept = [b for b in tab["by_nhat"]
            if b["obs"] > 0 and b["lo"] >= floor - 1e-9]
    z = np.array([b["z"] for b in kept], float)
    expect = float((z ** 2).sum() / len(z))
    assert tab["total"]["chi2_dof"] == pytest.approx(expect)
    assert tab["total"]["n_gate_bins"] == len(kept)
    g = RP.forward_closure_gate(spack)
    assert g["chi2_dof"] == pytest.approx(expect)


def test_chi2_dof_above_3_refuses_the_run(spack, fake_fold):
    tab = _flat_tab()
    tab["total"]["chi2_dof"] = 99.0
    for b in tab["by_nhat"]:              # chi2/dof is recomputed from by_nhat
        b["z"] = 4.0                      # 4^2 = 16 > 3, |z| = 4 < 5
    fake_fold(tab)
    g = RP.forward_closure_gate(spack)
    assert g["pass"] is False
    assert any("chi2/dof" in f for f in g["failures"]), g["failures"]
    assert not any("max|z_bin|" in f for f in g["failures"]), (
        "the chi2 arm must be able to fire ALONE -- otherwise chi2/dof <= 3 "
        "adds nothing to |z| <= 5")


# ==========================================================================
# 5. MATCHED-CONFIGURATION SBC (RATIFIED)
# ==========================================================================

def _run_cfg(spack, **over):
    cfg = MA.ModelAConfig(**over)
    return SBC.run_configuration(spack, cfg)


def test_a_configuration_matches_itself(spack):
    rc = _run_cfg(spack)
    m = SBC.configuration_match(rc, rc)
    assert m["matched"] is True and m["mismatches"] == []
    assert set(m["keys_compared"]) == set(SBC.MATCH_KEYS)


@pytest.mark.parametrize("over,key", [
    (dict(num_chains=2), "sampler.num_chains"),
    (dict(num_warmup=7), "sampler.num_warmup"),
    (dict(max_tree_depth=8), "sampler.max_tree_depth"),
    (dict(level_scale=0.6), "prior.level_scale"),
    (dict(sigma_N_scale=0.15), "prior.sigma_N_scale"),
    (dict(fp_mode="off"), "prior.fp_mode"),
    (dict(resp_clamp="hi"), "response.resp_clamp"),
])
def test_any_single_configuration_difference_refuses_to_certify(
        spack, over, key):
    """Coordinate-by-coordinate omission sensitivity: change ONE thing and the
    SBC no longer certifies the run."""
    m = SBC.configuration_match(_run_cfg(spack), _run_cfg(spack, **over))
    assert m["matched"] is False
    assert key in [x["key"] for x in m["mismatches"]], m["mismatches"]


def test_a_different_grid_refuses_to_certify(spack):
    other = synthetic_pack(
        0, nhat_edges=np.round(np.arange(19.9, 20.4 + 1e-9, 0.1), 10),
        zf_edges=np.round(np.arange(2.0, 2.2 + 1e-9, 0.1), 10),
        zc_edges=np.array([2.0, 2.2]), snr_edges=np.array([0.0, np.inf]),
        n_molly_cells=2, fp_frac=0.0)
    cfg = MA.ModelAConfig()
    m = SBC.configuration_match(SBC.run_configuration(other, cfg),
                                SBC.run_configuration(spack, cfg))
    assert m["matched"] is False
    keys = [x["key"] for x in m["mismatches"]]
    assert "grid.zf_edges" in keys and "grid.snr_edges" in keys
    assert "reported.quantities" in keys, (
        "a different coarse-z count changes the REPORTED functional set, "
        "which is the whole reason grid mismatch matters")


def test_an_absent_configuration_on_either_side_refuses_to_certify(spack):
    """FAIL CLOSED.  'We did not record what the SBC ran' must not certify,
    and neither must 'there is no run to compare against'."""
    rc = _run_cfg(spack)
    for a, b in ((None, rc), (rc, None), (None, None), ("not a dict", rc)):
        m = SBC.configuration_match(a, b)
        assert m["matched"] is False, (a is None, b is None)
        assert m["reasons"]


def test_an_absent_MATCH_KEY_is_a_mismatch_not_a_pass(spack):
    rc = _run_cfg(spack)
    stripped = copy.deepcopy(rc)
    del stripped["prior"]["fp_mode"]
    m = SBC.configuration_match(stripped, rc)
    assert m["matched"] is False
    assert "prior.fp_mode" in [x["key"] for x in m["mismatches"]]


def test_the_MATCH_KEYS_cover_every_documented_reduction():
    """R1 grid, R2 sampler, R3 prior + FP mode, R4 response, and the reported
    functionals.  A MATCH_KEYS that forgot one would let that reduction
    certify silently."""
    heads = {k.split(".")[0] for k in SBC.MATCH_KEYS}
    assert heads == {"grid", "prior", "sampler", "response", "reported"}
    assert "prior.fp_mode" in SBC.MATCH_KEYS
    assert "grid.ntrue_edges" in SBC.MATCH_KEYS      # the latent basis
    assert "reported.quantities" in SBC.MATCH_KEYS


def test_the_committed_reduced_SBC_constants_do_NOT_match_production(spack):
    """The KNOWN DEFECT, pinned as a fact rather than a footnote: the SBC that
    ships in this module (reduced grid, narrowed prior, FP block OFF, 1 chain)
    cannot certify a production ModelAConfig run."""
    sp = synthetic_pack(0, **SBC.SBC_GRID, fp_frac=0.0)
    sbc_cfg = SBC._configuration(
        sp, prior=SBC.SBC_PRIOR, sampler=SBC.SBC_SAMPLER, resp_clamp="both",
        reported_names=SBC._reported_names(sp))
    m = SBC.configuration_match(sbc_cfg, _run_cfg(spack))
    assert m["matched"] is False
    keys = [x["key"] for x in m["mismatches"]]
    for expect in ("prior.fp_mode", "prior.level_scale",
                   "sampler.num_chains", "grid.snr_edges"):
        assert expect in keys, (expect, keys)


@pytest.mark.parametrize("fixture_name", ["spack", "padded_spack"])
def test_matched_sbc_kwargs_reproduces_the_run_configuration(
        request, fixture_name):
    """The escape hatch must actually work: the kwargs it hands back, fed
    through the same configuration builder, MATCH.  Otherwise the ratified
    requirement would be unsatisfiable in principle.

    🔴 RUN ON A PADDED FIXTURE TOO.  The unpadded case passed VACUOUSLY: with
    ``ntrue_edges == nhat_edges`` the omission of ``ntrue_edges`` from
    ``matched_sbc_kwargs['grid']`` is invisible, because ``synthetic_pack``
    used to hardcode ``ntrue_edges = nhat_edges.copy()`` and so reproduced it
    by accident.  ``grid.ntrue_edges`` IS a MATCH_KEY, decisions 3 and 4
    adopted a PADDED basis, and on a padded pack the old code could not
    construct a matched SBC at any price."""
    pack = request.getfixturevalue(fixture_name)
    cfg = MA.ModelAConfig()
    kw = SBC.matched_sbc_kwargs(pack, cfg)
    sp = synthetic_pack(0, **kw["grid"], fp_frac=0.0)
    sampler = dict(kw["sampler"])
    sampler.pop("n_ranks")
    sbc_cfg = SBC._configuration(sp, prior=kw["prior"], sampler=sampler,
                                 resp_clamp=kw["resp_clamp"],
                                 reported_names=SBC._reported_names(sp))
    m = SBC.configuration_match(sbc_cfg, SBC.run_configuration(pack, cfg))
    assert m["matched"] is True, m["mismatches"]


def test_matched_sbc_kwargs_carries_the_LATENT_basis_not_just_the_observed_one(
        padded_spack):
    """The precise omission: ``grid.ntrue_edges`` is in ``MATCH_KEYS`` and was
    absent from the kwargs, so the kwargs described a DIFFERENT latent
    parameter vector from the run they claimed to match."""
    kw = SBC.matched_sbc_kwargs(padded_spack, MA.ModelAConfig())
    assert "ntrue_edges" in kw["grid"], (
        "matched_sbc_kwargs omits the latent basis, which is a MATCH_KEY")
    np.testing.assert_allclose(kw["grid"]["ntrue_edges"],
                               padded_spack.ntrue_edges)
    # and the omission is only detectable when the two axes DIFFER
    assert padded_spack.n_pad_bins == 2
    assert len(padded_spack.ntrue_edges) != len(padded_spack.nhat_edges)


def test_synthetic_pack_can_BUILD_a_padded_basis_at_all(padded_spack):
    """The capability the ratified requirement needed and did not have.
    ``synthetic_pack`` had no ``ntrue_edges`` parameter and hardcoded
    ``ntrue_edges = nhat_edges.copy()`` (pack.py:694), so NO padded pack could
    be generated -- every padded pack in the suite was a hand-built
    ``SimpleNamespace`` or a ``dataclasses.replace`` diagnostic that dropped
    ``truth_counts`` and skipped validation."""
    from CDDF_analysis.hbi_mcmc.pack import validate_pack
    p = padded_spack
    assert p.n_pad_bins == 2
    assert p.n_b == p.n_c + 2
    np.testing.assert_allclose(p.ntrue_edges[2:], p.nhat_edges)
    validate_pack(p, allow_nonstandard_grid=True)          # schema-conformant
    assert p.truth_counts is not None                      # a REAL pack
    assert p.truth.get("f_true") is not None
    # the fold works on it: a padded basis is not merely constructible but usable
    g = RP.forward_closure_gate(p)
    assert np.isfinite(g["total_mu"]) and g["total_mu"] > 0


def test_synthetic_pack_refuses_a_basis_pad_the_schema_forbids():
    """Pad DOWN only, same step, exact tail subset.  The new parameter must not
    become a way to build an invalid pack."""
    from CDDF_analysis.hbi_mcmc.pack import PackSchemaError
    nhat = np.round(np.arange(19.9, 20.4 + 1e-9, 0.1), 10)
    common = dict(zf_edges=np.round(np.arange(2.0, 2.2 + 1e-9, 0.1), 10),
                  zc_edges=np.array([2.0, 2.2]),
                  snr_edges=np.array([0.0, np.inf]), n_molly_cells=3,
                  fp_frac=0.0)
    for bad, why in (
            (np.round(np.arange(19.9, 20.6 + 1e-9, 0.1), 10), "pads UP"),
            (np.round(np.arange(20.0, 20.4 + 1e-9, 0.1), 10), "shrinks"),
            (np.round(np.arange(19.7, 20.4 + 1e-9, 0.2), 10), "wrong step"),
    ):
        with pytest.raises((PackSchemaError, ValueError)):
            synthetic_pack(0, nhat_edges=nhat, ntrue_edges=bad, **common)


def test_a_REAL_sbc_run_on_a_PADDED_grid_records_the_padded_basis_and_matches():
    """END-TO-END CONSTRUCTIBILITY of the ratified requirement, on the geometry
    decisions 3 and 4 actually adopted.  ``sbc_run`` builds its template pack
    with ``synthetic_pack(pack_seed, **grid, fp_frac=0.0)``, so an
    ``ntrue_edges`` that ``synthetic_pack`` cannot accept makes a padded
    matched SBC impossible at ANY cost -- which is a capability gap, not the
    ~1600 CPU-h the docstring blamed.  Absurdly small sampler; measured ~30 s.
    """
    grid = dict(nhat_edges=np.round(np.arange(19.9, 20.4 + 1e-9, 0.1), 10),
                ntrue_edges=np.round(np.arange(19.7, 20.4 + 1e-9, 0.1), 10),
                zf_edges=np.round(np.arange(2.0, 2.2 + 1e-9, 0.1), 10),
                zc_edges=np.array([2.0, 2.2]),
                snr_edges=np.array([0.0, np.inf]), n_molly_cells=3)
    samp = dict(num_warmup=8, num_samples=8, num_chains=1, max_tree_depth=4,
                target_accept=0.8, n_ranks=4)
    ranks, meta = SBC.sbc_run(2, seed=0, grid=grid, sampler=samp)
    cfg = SBC.sbc_configuration(meta)
    assert cfg is not None
    np.testing.assert_allclose(cfg["grid"]["ntrue_edges"], grid["ntrue_edges"])
    assert len(cfg["grid"]["ntrue_edges"]) == len(cfg["grid"]["nhat_edges"]) + 2
    # ... and it MATCHES a run configuration on the same padded pack, given the
    # same prior/sampler: the requirement is satisfiable, only expensive.
    padded = synthetic_pack(0, **grid, fp_frac=0.0)
    run_cfg = SBC._configuration(
        padded, prior=SBC.SBC_PRIOR,
        sampler={k: v for k, v in samp.items() if k != "n_ranks"},
        resp_clamp="both", reported_names=SBC._reported_names(padded))
    assert SBC.configuration_match(cfg, run_cfg)["matched"] is True, \
        SBC.configuration_match(cfg, run_cfg)["mismatches"]


def test_fp_nuisance_prior_scales_are_inert_when_the_FP_BLOCK_IS_OFF(spack):
    """Not a loophole: with fp_mode='off' the FP hyper-scales parameterise
    nothing, so comparing them would manufacture a mismatch that is not a
    difference.  fp_mode ITSELF is always compared (test above)."""
    a = _run_cfg(spack, fp_mode="off", fp_shape_sd=3.0)
    b = _run_cfg(spack, fp_mode="off", fp_shape_sd=99.0)
    assert SBC.configuration_match(a, b)["matched"] is True
    c = _run_cfg(spack, fp_mode="joint", fp_shape_sd=3.0)
    d = _run_cfg(spack, fp_mode="joint", fp_shape_sd=99.0)
    assert SBC.configuration_match(c, d)["matched"] is False


# --- the gate-side half: an unmatched SBC must not be STAMPABLE ------------

def _blocks(**sbc_checks):
    b = {name: {"checks": {f"{name}_ok": True}, "incomplete": []}
         for name in EV.REQUIRED_BLOCKS}
    b["coverage_sbc"]["checks"].update(sbc_checks)
    return b


def test_an_unmatched_sbc_makes_the_artifact_not_stampable():
    g = EV.gate(_blocks(sbc_configuration_matches_run=False))
    assert g["stampable"] is False and g["paper_facing"] is False
    assert any("sbc_configuration_matches_run" in r for r in g["reasons"])


def test_a_matched_sbc_stamps_which_proves_the_check_discriminates():
    g = EV.gate(_blocks(sbc_configuration_matches_run=True))
    assert g["stampable"] is True, g["reasons"]


def test_an_ABSENT_match_check_is_synthesised_False_no_passing_by_silence():
    """The structural half.  A coverage_sbc block that simply never mentions
    its configuration -- an old block, a hand-written one, one from a module
    predating the ratification -- must NOT stamp."""
    g = EV.gate(_blocks())
    assert g["stampable"] is False
    assert g["checks"]["coverage_sbc.sbc_configuration_matches_run"] is False
    assert any("ABSENT" in r and "sbc_configuration_matches_run" in r
               for r in g["reasons"]), g["reasons"]


def test_required_checks_are_published_in_the_verdict_and_may_only_grow():
    assert EV.REQUIRED_CHECKS["coverage_sbc"] == (
        "sbc_configuration_matches_run",)
    g = EV.gate(_blocks(sbc_configuration_matches_run=True))
    assert g["required_checks"]["coverage_sbc"] == [
        "sbc_configuration_matches_run"]


def test_sbc_block_attaches_the_match_verdict_without_running_the_sampler(
        monkeypatch, spack):
    """``sbc_block`` must derive the check from the CONFIGURATION, so stub the
    expensive run and vary only the configuration."""
    rc = SBC.run_configuration(spack, MA.ModelAConfig())
    meta = {"n_ranks_L": 4, "n_sims_used": 30, "configuration": rc}
    monkeypatch.setattr(
        SBC, "sbc_run",
        lambda n, **kw: ({"q": [0, 1, 2, 3, 0, 1, 2, 3] * 5}, dict(meta)))
    ok = SBC.sbc_block(40, run_config=rc)
    assert ok["checks"]["sbc_configuration_matches_run"] is True
    assert ok["configuration_match"]["matched"] is True

    bad = SBC.sbc_block(40, run_config=SBC.run_configuration(
        spack, MA.ModelAConfig(num_chains=1)))
    assert bad["checks"]["sbc_configuration_matches_run"] is False
    assert "sampler.num_chains" in [
        x["key"] for x in bad["configuration_match"]["mismatches"]]

    # no run_config at all -> refuses (this is the --mode sbc path)
    none = SBC.sbc_block(40)
    assert none["checks"]["sbc_configuration_matches_run"] is False
    assert none["configuration_match"]["reasons"]
    assert "RATIFIED" in none["configuration_match_note"]


def test_sbc_block_cannot_claim_a_match_when_it_produced_no_replicas(
        monkeypatch, spack):
    """The degenerate early-return path had to be closed too, or a broken SBC
    would return a block with no match check at all."""
    rc = SBC.run_configuration(spack, MA.ModelAConfig())
    monkeypatch.setattr(SBC, "sbc_run",
                        lambda n, **kw: ({}, {"configuration": rc}))
    blk = SBC.sbc_block(40, run_config=rc)
    assert "sbc_configuration_matches_run" in blk["checks"]
    assert blk["checks"]["sbc_uniform_ok"] is False
    assert blk["incomplete"] == ["sbc_produced_no_usable_replicas"]


def test_a_REAL_sbc_run_records_the_configuration_it_actually_ran():
    """The one link the stubbed tests above cannot cover: ``sbc_run`` itself
    must WRITE the configuration into its meta.  A mutation that renames that
    field survives every other test in this file, so this test runs a genuine
    (absurdly small) SBC -- measured ~27 s -- and reads the record back.

    It also pins the DEFECT end-to-end: what the shipped SBC records does not
    match a production ModelAConfig, so it cannot certify one.
    """
    samp = dict(num_warmup=8, num_samples=8, num_chains=1, max_tree_depth=4,
                target_accept=0.8, n_ranks=4)
    ranks, meta = SBC.sbc_run(2, seed=0, sampler=samp)
    cfg = SBC.sbc_configuration(meta)
    assert cfg is not None, "sbc_run did not record what it ran"
    # the SAMPLER it actually used, not the module default
    assert cfg["sampler"] == {k: v for k, v in samp.items() if k != "n_ranks"}
    assert cfg["sampler"]["num_warmup"] == 8            # not SBC_SAMPLER's 150
    # the NARROWED prior with the FP block OFF (reduction R3), as run
    assert cfg["prior"]["fp_mode"] == "off"
    assert cfg["prior"]["level_scale"] == SBC.SBC_PRIOR["level_scale"]
    # the REALIZED grid, read off the pack
    assert cfg["grid"]["nhat_edges"][0] == pytest.approx(19.9)
    assert cfg["grid"]["snr_edges"] == [0.0, float("inf")]   # 1 stratum
    # the functionals the ranks were actually computed on
    assert cfg["reported"]["quantities"] == sorted(ranks)
    assert SBC.configuration_match(cfg, cfg)["matched"] is True
    # ... and it does NOT certify a production run
    prod = SBC.run_configuration(
        __import__("CDDF_analysis.hbi_mcmc.pack", fromlist=["x"])
        .synthetic_pack(0, **SBC.SBC_GRID, fp_frac=0.0), MA.ModelAConfig())
    m = SBC.configuration_match(cfg, prod)
    assert m["matched"] is False
    assert "prior.fp_mode" in [x["key"] for x in m["mismatches"]]


def test_an_sbc_meta_without_a_configuration_certifies_nothing(spack):
    """Fail-closed on the SBC's own silence."""
    assert SBC.sbc_configuration({"meta": {"n_ranks_L": 5}}) is None
    assert SBC.sbc_configuration(None) is None
    m = SBC.configuration_match(
        SBC.sbc_configuration({"meta": {}}),
        SBC.run_configuration(spack, MA.ModelAConfig()))
    assert m["matched"] is False
    assert any("NO configuration" in r for r in m["reasons"])
