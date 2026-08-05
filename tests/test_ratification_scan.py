"""test_ratification_scan.py -- the WIDENED, tree-runnable ratification guard.

``tests/test_gate_ratification.py`` pins the ratification RECORD.  This file
pins the GUARD, and specifically the four ways the guard was measured to be
too narrow on 2026-08-05:

  1. it read two field names (``status``, ``authority``) and neither of the
     two live fabricated-authority sites used either of them;
  2. it read ONE dict in ONE module, not artifacts and not code elsewhere;
  3. it ran only at import of that module, and the module does not exist on
     the branches the two live sites were on;
  4. it could return "clean" for something it had not read.

THE DELIVERABLE IS THE BEFORE/AFTER, not the assertion that it is wider.
``test_the_OLD_guard_PASSES_both_live_sites`` loads the pre-widening module
out of git by SHA and shows it returning ``[]`` for both live-site shapes,
next to the new guard rejecting them.  A guard that merely looks wider is not
evidence.

EVERY FIXTURE IS SYNTHETIC OR IS THE PUBLIC SHAPE OF A COMMITTED SITE.  No
survey data, no real-survey values.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")

from CDDF_analysis.hbi_mcmc import ratification as RAT      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: the commit whose ``ratification.py`` is the guard AS IT WAS before the
#: widening -- the "old guard" half of the demonstration.  Immutable, so this
#: reference cannot rot.  Verified to exist by the test that uses it.
_OLD_GUARD_SHA = "c2a8943"

#: 🔴 LIVE SITE 1, verbatim shape.  ``adopted_config.py`` on
#: ``wip/adopt-basis-pad-window`` and ``/verdict/gate_tolerances_ratified`` in
#: ``adopted_config_closure.json``.  Two of these three names are NOT
#: RATIFIED.
_SITE1_NAMES = ["z_total_max", "z_bin_max", "chi2_dof_max"]
#: 🔴 LIVE SITE 2, verbatim shape.  ``window_study.py`` on
#: ``wip/spectral-window`` and ``metadata.gate.ratified_arms`` in
#: ``spectral_window_study.json``.  Two of these three names are NOT RATIFIED.
_SITE2_ARMS = {"abs_z_total_max": 5.0, "z_bin_max": 5.0, "chi2_dof_max": 3.0}

#: what each live site SHOULD have caught, by name
_SITE1_BAD = {"z_total_max", "z_bin_max"}
_SITE2_BAD = {"abs_z_total_max", "z_bin_max"}


# ==========================================================================
# helpers
# ==========================================================================

def _load_old_guard(tmp_path):
    """The pre-widening ``ratification.py``, out of git, as a live module."""
    import importlib.util
    src = subprocess.check_output(
        ["git", "show", f"{_OLD_GUARD_SHA}:CDDF_analysis/hbi_mcmc/"
         f"ratification.py"], cwd=str(ROOT), text=True)
    # it is the OLD file, so it must NOT already have the widened API
    assert "def scan_paths" not in src, (
        f"{_OLD_GUARD_SHA} already contains the widened guard; the "
        f"before/after demonstration would be circular")
    path = tmp_path / "ratification_old.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("_rat_old", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _exec_mutant(tmp_path, name, *replacements):
    """Exec a MUTATED copy of ``ratification.py``; return the exception or None.

    Proves an import-time guard is actually WIRED, not merely callable.
    """
    import importlib.util
    src = (ROOT / "CDDF_analysis" / "hbi_mcmc" / "ratification.py").read_text()
    for old, new in replacements:
        assert old in src, f"mutation anchor not found: {old!r}"
        src = src.replace(old, new, 1)
    path = tmp_path / f"{name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                                # noqa: BLE001
        return exc
    return None


def _subjects(violations, rule=None):
    return {v["subject"] for v in violations
            if rule is None or v["rule"] == rule}


def _rules(violations):
    return {v["rule"] for v in violations}


# ==========================================================================
# 1. 🔴 THE DELIVERABLE: OLD PASSES, NEW REJECTS -- both live-site shapes
# ==========================================================================

def test_the_OLD_guard_PASSES_both_live_sites_and_the_NEW_guard_REJECTS_them(
        tmp_path):
    """🔴 THE BEFORE/AFTER, on the same input, through each guard's own API.

    The old guard's only input is a mapping of ratification RECORDS.  Handed
    the two live sites in exactly that shape it returns ``[]`` and
    ``enforce_authority_allow_list`` does not raise -- because it reads only
    ``status`` and ``authority``, and neither live site has either field.  The
    claim lives in a field it never looks at.
    """
    old = _load_old_guard(tmp_path)

    site1 = {"adopted_config_verdict": {
        "question": "does the forward model close in the reporting window?",
        "answer": "NO",
        "gate_tolerances": {"z_total_max": 5.0, "z_bin_max": 5.0,
                            "chi2_dof_max": 3.0},
        "gate_tolerances_ratified": list(_SITE1_NAMES),
        "gate_tolerances_not_ratified": ["ratio_span_by_z_max",
                                         "ratio_span_by_snr_max"]}}
    site2 = {"window_study_gate": {
        "z_bin_max": "max over B+(W) of |z_c|",
        "ratified_arms": dict(_SITE2_ARMS),
        "not_ratified": {"ratio_span_by_z_max": 0.10,
                         "ratio_span_by_snr_max": 0.15}}}

    # --- BEFORE: the old guard sees nothing wrong with either -------------
    assert old.audit_authority_claims(site1) == []
    assert old.audit_authority_claims(site2) == []
    assert old.enforce_authority_allow_list(site1) is True
    assert old.enforce_authority_allow_list(site2) is True
    # ... and it has no way to be pointed at a FILE or a TREE at all
    for attr in ("scan_paths", "scan_file", "scan_data", "main"):
        assert not hasattr(old, attr), attr

    # --- AFTER: the widened guard rejects both, by name -------------------
    v1 = RAT.scan_data(site1, source="site1")
    v2 = RAT.scan_data(site2, source="site2")
    assert _rules(v1) == {"R1_NAME_CLAIM"}, v1
    assert _rules(v2) == {"R1_NAME_CLAIM"}, v2
    assert _subjects(v1) == _SITE1_BAD, v1
    assert _subjects(v2) == _SITE2_BAD, v2
    # the ratified one is NOT flagged -- the guard discriminates
    assert "chi2_dof_max" not in _subjects(v1) | _subjects(v2)
    with pytest.raises(RAT.FabricatedAuthorityError):
        RAT.enforce_no_fabricated_claims_data(site1)


def test_both_live_sites_are_rejected_AS_FILES_in_code_AND_artifact_form(
        tmp_path):
    """Both sites exist as CODE and as JSON, so both readers must fire.  The
    fixtures reproduce the committed shapes exactly."""
    (tmp_path / "adopted_config.py").write_text(textwrap.dedent(f"""\
        def verdict_block(rows):
            return dict(
                answer="NO",
                gate_tolerances_ratified={_SITE1_NAMES!r},
                gate_tolerances_not_ratified=["ratio_span_by_z_max"],
                n_configurations=len(rows))
        """))
    (tmp_path / "adopted_config_closure.json").write_text(json.dumps(
        {"verdict": {"answer": "NO",
                     "gate_tolerances_ratified": list(_SITE1_NAMES),
                     "gate_tolerances_not_ratified": ["ratio_span_by_z_max"]}}))
    (tmp_path / "window_study.py").write_text(textwrap.dedent(f"""\
        def restated_gate_criteria():
            return dict(
                z_bin_max="max over B+(W) of |z_c|",
                ratified_arms={_SITE2_ARMS!r},
                not_ratified={{"ratio_span_by_z_max": 0.10}})
        """))
    (tmp_path / "spectral_window_study.json").write_text(json.dumps(
        {"metadata": {"gate": {"ratified_arms": dict(_SITE2_ARMS),
                               "not_ratified": {"ratio_span_by_z_max": 0.1}}}}))

    res = RAT.scan_paths([str(tmp_path)])
    assert res.ok is False
    assert res.n_files == 4, res.files
    assert res.n_claims == 4, res.n_claims          # one claim per file
    assert _rules(res.violations) == {"R1_NAME_CLAIM"}
    by_file = {}
    for v in res.violations:
        by_file.setdefault(os.path.basename(v["source"]), set()).add(
            v["subject"])
    assert by_file == {"adopted_config.py": _SITE1_BAD,
                       "adopted_config_closure.json": _SITE1_BAD,
                       "window_study.py": _SITE2_BAD,
                       "spectral_window_study.json": _SITE2_BAD}, by_file


def test_the_live_site_PROSE_is_rejected_too(tmp_path):
    """The third form the same fabrication took: an English sentence.  The
    2026-07-30 retraction had to be re-issued because the claim survived in
    prose in three places, so a structural-only guard is not enough."""
    (tmp_path / "window_study.py").write_text(textwrap.dedent('''\
        def restated_gate_criteria():
            """Metric definitions for the window study.

            THE THREE RATIFIED ARMS (PI decision 8) are

                |z_total(W)| <= 5   z_bin_max(W) <= 5   chi2_dof(W) <= 3

            and they decide closure.
            """
            return {}
        '''))
    res = RAT.scan_paths([str(tmp_path)])
    assert res.ok is False
    assert _rules(res.violations) == {"R4_PROSE_CLAIM"}
    assert _subjects(res.violations) == {"z_bin_max"}, res.violations


def test_the_CORRECTED_forms_of_both_live_sites_PASS(tmp_path):
    """🔴 THE CONTROL, without which every test above is worthless: a guard
    that rejected everything would pass all of them.  The corrected shape of
    each live site -- exactly one name claimed, the rest recorded as not
    ratified -- must scan CLEAN, and the scan must still have INSPECTED the
    claims rather than skipping the files."""
    (tmp_path / "adopted_config.py").write_text(textwrap.dedent("""\
        def verdict_block():
            return dict(
                gate_tolerances_ratified=["chi2_dof_max"],
                gate_tolerances_not_ratified=[
                    "z_total_max", "z_bin_max",
                    "ratio_span_by_z_max", "ratio_span_by_snr_max"])
        """))
    (tmp_path / "window_study.py").write_text(textwrap.dedent("""\
        def restated_gate_criteria():
            return dict(
                ratified_arms={"chi2_dof_max": 3.0},
                restated_not_ratified_arms={"abs_z_total_max": 5.0,
                                            "z_bin_max": 5.0},
                not_ratified={"ratio_span_by_z_max": 0.10})
        """))
    (tmp_path / "closure.json").write_text(json.dumps(
        {"verdict": {"gate_tolerances_ratified": ["chi2_dof_max"],
                     "gate_tolerances_not_ratified": ["z_total_max",
                                                      "z_bin_max"]}}))
    res = RAT.scan_paths([str(tmp_path)])
    assert res.ok is True, res.report()
    assert res.n_files == 3
    assert res.n_claims == 3, (
        "the corrected fixtures scanned clean but NO claim was inspected -- "
        "that is a vacuous pass, not a passing guard")


# ==========================================================================
# 2. RUNNABLE ACROSS BRANCHES: the CLI
# ==========================================================================

def _cli(*args):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-m", "CDDF_analysis.hbi_mcmc.ratification", *args],
        cwd=str(ROOT), capture_output=True, text=True, env=env)


def test_the_cli_exits_NONZERO_on_a_fabricated_claim_and_ZERO_on_a_clean_tree(
        tmp_path):
    """The PI asked for a check that runs "across all branches rather than
    one".  A merge can only invoke something with an exit code."""
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "a.json").write_text(json.dumps(
        {"verdict": {"gate_tolerances_ratified": list(_SITE1_NAMES)}}))
    out = _cli("--check", str(bad))
    assert out.returncode == 1, out.stdout + out.stderr
    assert "R1_NAME_CLAIM" in out.stdout
    assert "z_total_max" in out.stdout

    good = tmp_path / "good"
    good.mkdir()
    (good / "a.json").write_text(json.dumps(
        {"verdict": {"gate_tolerances_ratified": ["chi2_dof_max"]}}))
    out = _cli("--check", str(good))
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.rstrip().endswith("OK")


def test_the_cli_REPORTS_WHAT_IT_INSPECTED_so_a_clean_run_is_not_vacuous(
        tmp_path):
    """Containment-style pass counts are monotone in how little you look at.
    The report must state the files AND the claims, and ``--json`` must carry
    both, or "0 violations" is unfalsifiable."""
    (tmp_path / "a.json").write_text(json.dumps(
        {"pi_ratified_items": ["chi2_dof_max"]}))
    out = _cli("--check", str(tmp_path), "--json")
    assert out.returncode == 0, out.stdout + out.stderr
    rep = json.loads(out.stdout)
    assert rep["schema"] == RAT.SCAN_SCHEMA
    assert rep["ok"] is True
    assert rep["n_files"] == 1
    assert rep["n_claims_inspected"] == 1
    assert rep["n_violations"] == 0
    assert set(rep["rules"]) == set(RAT.SCAN_RULES)


def test_an_EXCLUSION_is_echoed_in_the_report_because_a_hidden_one_is_a_bypass(
        tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(
        {"verdict": {"gate_tolerances_ratified": list(_SITE1_NAMES)}}))
    (tmp_path / "b.json").write_text(json.dumps(
        {"verdict": {"gate_tolerances_ratified": ["chi2_dof_max"]}}))
    res = RAT.scan_paths([str(tmp_path)], exclude=["a.json"])
    assert res.ok is True
    assert res.excluded == ["a.json"]
    assert "a.json" in res.report()          # visible, not silent
    assert res.n_files == 1
    # and excluding EVERYTHING is a failure, not a clean bill of health
    res = RAT.scan_paths([str(tmp_path)], exclude=["*"])
    assert res.ok is False
    assert _rules(res.violations) == {"R8_NOTHING_SCANNED"}


def test_the_PRODUCTION_TREE_is_clean_under_the_widened_guard():
    """The guard has to be true of this branch before it is imposed on any
    other.  ``CDDF_analysis`` and ``docs`` scan clean -- and the claim count
    proves the scan was not vacuous."""
    res = RAT.scan_paths([str(ROOT / "CDDF_analysis"), str(ROOT / "docs")])
    assert res.ok is True, res.report()
    assert res.n_files > 100, res.n_files
    assert res.n_claims >= 5, (
        f"only {res.n_claims} ratification claim(s) inspected across "
        f"{res.n_files} files -- a clean scan that inspected nothing proves "
        f"nothing")


def test_the_scanner_has_NO_SILENT_EXEMPTION_for_its_own_test_fixtures():
    """🔴 A FAIL-OPEN SHAPE REFUSED.  ``tests/`` holds deliberately fabricated
    records (this file, and ``test_gate_ratification.py``'s negative
    fixtures).  The tempting fix is to exempt ``tests/`` inside the scanner;
    that would be a bypass nobody could see.  Instead the scanner flags them
    like anything else, and the recommended merge invocation simply does not
    point at ``tests/`` -- which is a visible choice at the call site."""
    res = RAT.scan_paths([str(ROOT / "tests" / "test_gate_ratification.py")])
    assert res.ok is False, (
        "the scanner found nothing in a file that deliberately contains "
        "fabricated ratification records -- it has an exemption it should "
        "not have")
    assert "R2_STATUS_CLAIM" in _rules(res.violations), res.violations
    assert {"z_zbin_max", "invented_tomorrow_max"} <= _subjects(res.violations)


# ==========================================================================
# 3. FAIL CLOSED -- an unread or unreadable thing is a FAILURE
# ==========================================================================

def test_no_LIVE_document_points_at_a_RESOLVED_decision_as_though_it_were_OPEN():
    """🔴 DOC-DRIFT GUARD, and the reason this one exists: closing
    ``span_arms_disarmed`` turned every ``OPEN_PI_DECISIONS['span_arms_
    disarmed']`` in the tree into a pointer at a key that is no longer there.
    Three of them were live prose and one was a dated artifact.

    The rule is paragraph-scoped, like the false-alarm-rate guard: a file may
    quote the old pointer, but must say IN THE SAME PARAGRAPH that it is
    resolved, superseded or dated evidence.  A committed ARTIFACT is exempt --
    it records the state at its generation date and must not be rewritten --
    and the exemption is narrow enough to be listed."""
    import re
    resolved = sorted(RAT.RESOLVED_PI_DECISIONS)
    assert resolved, "no resolved decision: this guard would pass vacuously"
    targets = [p for p in
               list((ROOT / "CDDF_analysis" / "hbi_mcmc").glob("*.py"))
               + list((ROOT / "docs").glob("*.md"))]
    assert len(targets) > 5, targets
    qualifier = re.compile(
        r"RESOLVED|resolved|superseded|SUPERSEDED|dated evidence|"
        r"ANSWERED|answered|before that date|pi_decision")
    n_quotes = 0
    for path in targets:
        text = path.read_text()
        for para in re.split(r"\n\s*\n", text):
            for name in resolved:
                if f"OPEN_PI_DECISIONS['{name}']" not in para and \
                        f'OPEN_PI_DECISIONS["{name}"]' not in para:
                    continue
                n_quotes += 1
                assert qualifier.search(para), (
                    f"{path.name}: points at the RESOLVED decision {name!r} "
                    f"through OPEN_PI_DECISIONS with nothing saying it "
                    f"closed:\n{para[:400]}")
    assert n_quotes >= 2, (
        "the guard found nothing to check -- it would pass vacuously if the "
        "pointers were merely deleted")


def test_a_path_that_does_not_exist_is_a_FAILURE():
    res = RAT.scan_paths(["/no/such/path/anywhere"])
    assert res.ok is False
    assert "R6_UNPARSEABLE" in _rules(res.violations)


def test_an_UNPARSEABLE_json_or_python_is_a_FAILURE(tmp_path):
    (tmp_path / "broken.json").write_text('{"verdict": [1, 2,')
    (tmp_path / "broken.py").write_text("def f(:\n    pass\n")
    res = RAT.scan_paths([str(tmp_path)])
    assert res.ok is False
    assert _rules(res.violations) == {"R6_UNPARSEABLE"}
    assert len(res.violations) == 2, res.violations
    # a file kind the scanner does not understand is also a failure, not a skip
    other = tmp_path / "notes.rst"
    other.write_text("ratified_arms = whatever")
    assert _rules(RAT.scan_file(str(other))) == {"R6_UNPARSEABLE"}


def test_a_file_that_cannot_be_READ_is_a_FAILURE_not_a_skip(tmp_path):
    """🔴 A SURVIVING MUTANT, FOUND BY MUTATION TESTING AND CLOSED.  Deleting
    the read-failure branch of ``scan_file`` left every other test green: the
    PARSE failures were covered and the READ failure was not, so a file the
    scanner could not open would have counted as a clean file.

    Two ways a real file refuses to be read, both of which must FAIL: bytes
    that are not UTF-8, and a mode the process cannot open."""
    raw = tmp_path / "latin1.json"
    raw.write_bytes(b'{"note": "\xff\xfe not utf-8"}')
    assert _rules(RAT.scan_file(str(raw))) == {"R6_UNPARSEABLE"}
    assert "cannot be read" in RAT.scan_file(str(raw))[0]["detail"]
    # and it propagates through the tree scan rather than being dropped
    res = RAT.scan_paths([str(tmp_path)])
    assert res.ok is False
    assert _rules(res.violations) == {"R6_UNPARSEABLE"}

    locked = tmp_path / "locked.py"
    locked.write_text("gate_tolerances_ratified = ['chi2_dof_max']\n")
    os.chmod(str(locked), 0o000)
    try:
        if os.access(str(locked), os.R_OK):
            pytest.skip("running as a user that can read mode-000 files")
        assert _rules(RAT.scan_file(str(locked))) == {"R6_UNPARSEABLE"}
    finally:
        os.chmod(str(locked), 0o600)


def test_a_scan_that_INSPECTED_ZERO_FILES_is_a_FAILURE(tmp_path):
    """The fail-open shape this project keeps meeting: a green check that
    looked at nothing."""
    empty = tmp_path / "empty"
    empty.mkdir()
    res = RAT.scan_paths([str(empty)])
    assert res.ok is False
    assert _rules(res.violations) == {"R8_NOTHING_SCANNED"}
    assert res.n_files == 0


def test_an_UNRECOGNISED_claim_shape_is_a_FAILURE_not_a_pass():
    """R5.  A ratification key holding something the scanner cannot read as a
    set of names must not be waved through."""
    for value in (3, 3.5, ["z_bin_max", 5.0], {1: "x"}):
        v = RAT.scan_data({"gate_tolerances_ratified": value}, source="s")
        assert "R5_UNRECOGNISED" in _rules(v), (value, v)
    # a status claim whose SUBJECT cannot be resolved is also a failure
    v = RAT.scan_data({"status": "RATIFIED"}, source="s")
    assert "R5_UNRECOGNISED" in _rules(v), v
    # ... and the same claim WITH a subject is checked, not skipped
    v = RAT.scan_data({"z_bin_max": {"status": "RATIFIED"}}, source="s")
    assert _rules(v) == {"R2_STATUS_CLAIM"} and _subjects(v) == {"z_bin_max"}


def test_a_COMPUTED_ratified_list_must_be_DERIVED_from_this_module(tmp_path):
    """R7.  A static scanner cannot evaluate ``gate_tolerances_ratified=f(x)``.
    It refuses it -- UNLESS the expression reads this module, which is the
    single source of truth and is itself guarded at import.  Both directions
    are checked, because a rule that refused everything would be useless and
    one that accepted everything would be the defect."""
    (tmp_path / "derived.py").write_text(textwrap.dedent("""\
        from CDDF_analysis.hbi_mcmc import ratification as RAT

        def report():
            return {"gate_tolerances_ratified": list(RAT.ratified_names())}
        """))
    (tmp_path / "underived.py").write_text(textwrap.dedent("""\
        def report(gate):
            return {"gate_tolerances_ratified": sorted(gate)}
        """))
    res = RAT.scan_paths([str(tmp_path / "derived.py")])
    assert res.ok is True, res.report()
    assert res.n_claims == 1
    res = RAT.scan_paths([str(tmp_path / "underived.py")])
    assert res.ok is False
    assert _rules(res.violations) == {"R7_UNDERIVED"}


# ==========================================================================
# 4. THE IMPORT-TIME GUARDS ARE WIRED, AND HAVE TEETH
# ==========================================================================

def test_the_module_REFUSES_TO_IMPORT_with_a_fabricated_record(tmp_path):
    """Not "the function raises when called" -- that is already tested.  This
    execs a MUTATED COPY of the module and requires the import itself to
    fail, which is the only thing that stops a fabricated state shipping."""
    exc = _exec_mutant(
        tmp_path, "mut_fabricated",
        ("# 🔴 fail at IMPORT, not at review time.",
         'RESTATED_NOT_RATIFIED["z_bin_max"]["status"] = "RATIFIED"\n'
         'RESTATED_NOT_RATIFIED["z_bin_max"]["authority"] = PI_AUTHORITY\n'))
    assert exc is not None, "the mutated module imported cleanly"
    assert type(exc).__name__ == "FabricatedAuthorityError", exc
    assert "z_bin_max" in str(exc)


def test_EMPTYING_the_ratified_records_no_longer_passes_the_audit(tmp_path):
    """🔴 FAIL-OPEN HOLE FOUND AND CLOSED (round 4).  Before this, clearing
    ``RATIFIED`` left ``audit_authority_claims`` with nothing to complain
    about and the module imported clean: the guard passed because the thing it
    guards had been deleted.  That is the "delete the block and everything
    stays green" pattern, in the guard itself."""
    exc = _exec_mutant(
        tmp_path, "mut_emptied",
        ("# 🔴 fail at IMPORT, not at review time.", "RATIFIED.clear()\n"))
    assert exc is not None, "a module with NOTHING ratified imported cleanly"
    assert type(exc).__name__ == "FabricatedAuthorityError", exc
    assert "no RATIFIED record" in str(exc)
    # and the audit says so directly, in both directions
    assert RAT.audit_authority_claims() == []
    assert RAT.audit_authority_claims({}) != []
    assert RAT.audit_authority_claims([]) != []          # not even a mapping


def test_the_IMPORT_TIME_SELF_SCAN_has_a_measured_POWER_CHECK(tmp_path):
    """"Zero violations" over a stamp whose ratification blocks were removed
    is not a pass.  The self-scan therefore asserts how much it inspected, and
    THAT assertion is wired at import too."""
    assert RAT.SELF_SCAN_MIN_CLAIMS >= 2
    counter = {"claims": 0}
    RAT.scan_data(RAT.ratification_stamp(), source="t", _counter=counter)
    assert counter["claims"] >= RAT.SELF_SCAN_MIN_CLAIMS
    # a stamp with the claim-bearing blocks stripped scans CLEAN and must
    # therefore be refused on the count, not on the violations
    stripped = {k: v for k, v in RAT.ratification_stamp().items()
                if k not in ("pi_ratified_items", "ratified")}
    c2 = {"claims": 0}
    assert RAT.scan_data(stripped, source="t", _counter=c2) == []
    assert c2["claims"] < RAT.SELF_SCAN_MIN_CLAIMS
    # ... and the import-time count check is live
    exc = _exec_mutant(tmp_path, "mut_power",
                       ("SELF_SCAN_MIN_CLAIMS = 2", "SELF_SCAN_MIN_CLAIMS = 99"))
    assert exc is not None
    assert type(exc).__name__ == "FabricatedAuthorityError", exc
    assert "self-scan inspected only" in str(exc)


def test_a_STAMP_MISSING_A_BLOCK_is_refused_and_required_may_only_GROW():
    """🔴 FAIL-OPEN HOLE FOUND AND CLOSED (round 4).  Of the stamp's blocks
    only four were pinned by any test; the rest could be dropped and every
    test stayed green, while a reader of the artifact silently stopped being
    told which criteria refuse work without authority."""
    st = RAT.ratification_stamp()
    for key in RAT.REQUIRED_STAMP_KEYS:
        assert key in st, key
    # the blocks that carry the UNCOMFORTABLE facts are required by name
    for key in ("restated_not_ratified", "unratified_but_gating",
                "open_pi_decisions", "authority_scope"):
        assert key in RAT.REQUIRED_STAMP_KEYS, key
    # and the check has teeth: build a stamp minus one block
    import unittest.mock as mock
    for drop in ("ratified", "restated_not_ratified", "unratified_but_gating",
                 "authority_scope", "open_pi_decisions", "self_scan"):
        full = RAT.ratification_stamp()
        full.pop(drop)
        with mock.patch.object(RAT, "_build_stamp", lambda f=full: dict(f)):
            with pytest.raises(RAT.IncompleteStampError) as ei:
                RAT.ratification_stamp()
        assert drop in str(ei.value), drop


def test_the_stamps_top_level_PI_AUTHORITY_is_legal_ONLY_because_it_is_SCOPED():
    """v1's exact defect, as an executable condition.  A bare ``authority:
    "PI (...)"`` at the top of a stamp reads as authorising the whole block --
    that is how four |z| arms became PI-ratified.  R3 allows it only when the
    stamp says, in the artifact, that it covers ``pi_ratified_items`` and
    nothing else."""
    st = RAT.ratification_stamp()
    assert RAT.scan_data(st, source="stamp") == []
    # remove the scope and the SAME stamp is refused
    unscoped = dict(st)
    unscoped.pop("authority_scope")
    v = RAT.scan_data(unscoped, source="stamp")
    assert "R3_PI_AUTHORITY" in _rules(v), v
    # weaken the scope to something that does not limit anything: also refused
    weak = dict(st, authority_scope="This stamp records the ratification "
                                    "state of the gate.")
    assert "R3_PI_AUTHORITY" in _rules(RAT.scan_data(weak, source="stamp"))


# ==========================================================================
# 5. THE CLASSIFIER, AND WHERE THE GUARD DELIBERATELY STOPS
# ==========================================================================

@pytest.mark.parametrize("key,kind", [
    ("gate_tolerances_ratified", "CLAIM"),
    ("ratified_arms", "CLAIM"),
    ("pi_ratified_items", "CLAIM"),
    ("RATIFIED", "CLAIM"),
    ("Gate Tolerances Ratified", "CLAIM"),
    ("gate-tolerances-ratified", "CLAIM"),
    ("gate_tolerances_not_ratified", "NEGATED"),
    ("not_ratified", "NEGATED"),
    ("unratified", "NEGATED"),
    ("unratified_but_gating", "NEGATED"),
    ("restated_not_ratified", "NEGATED"),
    ("z_arms_gate_unratified", "NEGATED"),
    ("ratification", "SUBJECT"),
    ("ratification_date", "SUBJECT"),
    ("ratification_status", "SUBJECT"),
    ("ratifying_authority", "SUBJECT"),
    ("ratifies", "CLAIM"),
    ("chi2_dof_max", None),
    ("authority", None),
])
def test_classify_key_discriminates(key, kind):
    """The grammatical discriminator is what makes the guard need no
    hand-maintained field-name list: the past participle ASSERTS, the noun
    NAMES.  Both live sites used a field name nobody had listed."""
    assert RAT.classify_key(key) == kind, key


def test_the_NAME_BEARING_narrowing_is_stated_and_its_boundary_is_MEASURED():
    """🔴 A DELIBERATE NARROWING, recorded rather than hidden.  A CLAIM key
    that COUNTS (``n_closing_pi_ratified_arm_only``) or names a RESTRICTED
    subset (``closing_configurations_pi_ratified_arm_only``) does not hold
    ratified names, and flagging it would be wrong.  The cost is that those
    key shapes are not checked -- so the boundary is pinned in both
    directions here, and the SAME payload under a name-bearing key is still
    caught."""
    payload = ["z_total_max", "z_bin_max"]
    assert RAT.claim_is_name_bearing("gate_tolerances_ratified") is True
    assert RAT.claim_is_name_bearing("ratified_arms") is True
    assert RAT.claim_is_name_bearing("n_closing_pi_ratified_arm_only") is False
    assert RAT.claim_is_name_bearing(
        "closing_configurations_pi_ratified_arm_only") is False
    skipped = RAT.scan_data({"n_closing_pi_ratified_arm_only": 4,
                             "closing_configurations_pi_ratified_arm_only":
                                 payload}, source="s")
    assert skipped == [], skipped
    caught = RAT.scan_data({"gate_tolerances_ratified": payload}, source="s")
    assert _subjects(caught) == set(payload), caught


def test_R3_leaves_a_RETRACTION_NOTE_alone_but_fires_on_the_field_it_retracts():
    """A note explaining that an authority claim was fabricated must not
    itself be read as an authority claim, or a retraction can never be
    written.  The discriminator is a suffix rule on the KEY, and the field it
    describes is still checked."""
    note = ("CORRECTED 2026-08-05. An earlier revision returned these "
            "thresholds under a key that named the PI as their authority.")
    clean = RAT.scan_data({"gate": {"authority_correction_note": note,
                                    "z_bin_max": 5.0}}, source="s")
    assert clean == [], clean
    dirty = RAT.scan_data({"z_bin_max": {"authority": RAT.PI_AUTHORITY}},
                          source="s")
    assert _rules(dirty) == {"R3_PI_AUTHORITY"}, dirty
    # DECLINING is not authorising, and an explicit non-claim is not a claim
    assert RAT.scan_data(
        {"ratio_span_by_z_max": {
            "authority": "NONE -- the PI was asked and DECLINED"}},
        source="s") == []


def test_a_MODULE_LEVEL_status_constant_is_resolved_not_stepped_over(tmp_path):
    """🔴 FAIL-OPEN HOLE FOUND AND CLOSED (round 4).  R2/R3 read constant
    field values, so ``{"authority_state": RATIFIED}`` with a module-level
    ``RATIFIED = "RATIFIED"`` -- which is how one sibling module spells it --
    slipped past both.  Module-level string constants are now resolved."""
    (tmp_path / "m.py").write_text(textwrap.dedent("""\
        RATIFIED = "RATIFIED"
        RESTATED_NOT_RATIFIED = "RESTATED_NOT_RATIFIED"

        GATE_ARMS = {
            "z_bin_max": {"value": 5.0, "authority_state": RATIFIED},
            "chi2_dof_max": {"value": 3.0, "authority_state": RATIFIED},
        }
        """))
    res = RAT.scan_paths([str(tmp_path / "m.py")])
    assert res.ok is False
    assert _rules(res.violations) == {"R2_STATUS_CLAIM"}
    assert _subjects(res.violations) == {"z_bin_max"}, res.violations
    # the honest spelling of the same file passes
    (tmp_path / "ok.py").write_text(textwrap.dedent("""\
        RATIFIED = "RATIFIED"
        RESTATED_NOT_RATIFIED = "RESTATED_NOT_RATIFIED"

        GATE_ARMS = {
            "z_bin_max": {"value": 5.0,
                          "authority_state": RESTATED_NOT_RATIFIED},
            "chi2_dof_max": {"value": 3.0, "authority_state": RATIFIED},
        }
        """))
    assert RAT.scan_paths([str(tmp_path / "ok.py")]).ok is True


def test_the_PROSE_rule_keys_on_the_BARE_claim_and_its_bypass_is_stated():
    """R4 is sentence-scoped because the modules must be able to DISCUSS these
    names next to the word.  That is also its weakness, and the weakness is
    pinned here so it cannot quietly widen: every qualifier is a way for a
    real claim to be skipped."""
    bad = "z_bin_max is RATIFIED and decides closure."
    assert _subjects(RAT.scan_data({"note": bad}, source="s")) == {"z_bin_max"}
    for good in ("z_bin_max gates but is NOT RATIFIED.",
                 "z_bin_max is UNRATIFIED and gates anyway.",
                 "z_bin_max: RATIFIED is not a status no deciding authority "
                 "granted.",
                 "chi2_dof_max is RATIFIED."):
        assert RAT.scan_data({"note": good}, source="s") == [], good
    # a name that appears in NO ratification record is still caught, by shape
    assert _subjects(RAT.scan_data(
        {"note": "abs_z_total_max is RATIFIED."},
        source="s")) == {"abs_z_total_max"}
