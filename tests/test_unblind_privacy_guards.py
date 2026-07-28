"""The privacy guards must never return a vacuous PASS, and must catch a real-data
leak written with ANY separator.

Two defects, both verified against the real tree on 2026-07-28:

1. FALSE PASS.  ``privacy.assert_no_outputs`` iterated ``nb.get("cells", [])``.
   Handed a ``.json`` artifact it got ``[]``, both loops were vacuous, and it
   returned 0 -- a clean PASS on a file it never inspected.  It was cited as
   privacy clearance for JSON artifacts.  A guard that cannot inspect its input
   must ERROR.

2. SEPARATOR-BLIND TOKEN TEST.  The one committed real-data token test scanned for
   ``main_dark`` (UNDERSCORE) while DESI filenames and the artifacts write
   ``main-dark`` (HYPHEN): ``dlacat-loa-main-dark-v1.fits``,
   ``coadd-main-dark-705.fits``, ``processed-main-dark-*.h5``.  MEASURED: the
   committed forward artifact contains the string "loa main-dark" and the old token
   set matched NONE of it -- the test passed vacuously on the very file it guards.

The deliberately-real-looking fixture below MUST fail; if a future "simplification"
makes it pass, the guard is worthless.
"""

import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.unblind import privacy as PV  # noqa: E402


def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc) if not isinstance(doc, str) else doc)
    return str(p)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------
def _nb(cells):
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}


CLEAN_NB = _nb([{"cell_type": "code", "source": ["x = 1\n"], "outputs": [],
                 "execution_count": None, "metadata": {}}])
EXECUTED_NB = _nb([{"cell_type": "code", "source": ["print(dndx)\n"], "execution_count": 3,
                    "metadata": {},
                    "outputs": [{"output_type": "stream", "name": "stdout",
                                 "text": ["dN/dX = 0.04512\n"]}]}])

# A MOCK artifact: mock TARGETIDs, mock paths, and a prose disclaimer that NAMES the
# real dataset in order to deny using it.  This is the shape of the real committed
# subdla_mock_validation_forward.json and it must PASS.
MOCK_ARTIFACT = {
    "metadata": {
        "code_commit": "a" * 40,
        "mock": "2LPT-0 (loa-124); values are MOCK recovery ratios, not real-LOA. "
                "No real-LOA (loa main-dark) data was read.",
        "inputs": {"catalog_dir": "/scratch/.../gl_prod_2lpt0_v1_20260526/"},
    },
    "targetids": [1000123, 88123456],
    "integrated": {"loa0": {"r0_dndx_195_203": 0.849}},
}

# A REAL artifact wearing a mock costume: it claims to be 2LPT, but it carries a real
# DESI TARGETID (O(1e16)), a real catalog PATH (hyphenated -- the exact form the old
# token test missed), and real result values.  MUST FAIL.
REAL_LOOKING_ARTIFACT = {
    "metadata": {
        "code_commit": "b" * 40,
        "mock": "2lpt0",                       # the lie
        "inputs": {
            "dlacat": "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
                      "loa_main_dark_v1/dlacat-loa-main-dark-v1.fits",
            "vac": "/nfs/turbo/.../QSO_cat_loa_main_dark_healpix_v3-altbal.fits",
        },
    },
    "targetid": 39627939081946569,
    "measurement": {"20.3": {"dndx": {"integrated": {"MAP": 0.0451}}}},
}


# ---------------------------------------------------------------------------
# 1. the false pass
# ---------------------------------------------------------------------------
def test_assert_no_outputs_refuses_a_json_artifact(tmp_path):
    """THE BUG: this used to return 0 on a .json and was cited as clearance."""
    path = _write(tmp_path, "artifact.json", MOCK_ARTIFACT)
    with pytest.raises(PV.PrivacyError) as exc:
        PV.assert_no_outputs(path)
    assert "not a .ipynb" in str(exc.value)


def test_scan_notebook_outputs_refuses_a_json_artifact(tmp_path):
    with pytest.raises(PV.PrivacyError):
        PV.scan_notebook_outputs(_write(tmp_path, "artifact.json", MOCK_ARTIFACT))


def test_ipynb_extension_but_not_a_notebook_is_an_error(tmp_path):
    """An artifact RENAMED to .ipynb must ERROR: 'cells' is REQUIRED, never defaulted.
    (The original bug was `nb.get("cells", [])` -- the default turned 'no cells' into
    'no offending cells'.)"""
    with pytest.raises(PV.PrivacyError, match="no 'cells' list"):
        PV.assert_no_outputs(_write(tmp_path, "fake.ipynb", MOCK_ARTIFACT))
    with pytest.raises(PV.PrivacyError, match="not notebook cells"):
        PV.assert_no_outputs(_write(tmp_path, "fake2.ipynb", {"cells": [{"x": 1}]}))


def test_minimal_notebook_without_nbformat_key_still_works(tmp_path):
    """UNBLIND_00's own soft-tripwire demo builds a synthetic notebook with no
    'nbformat' key; the guard must read it rather than break the notebook."""
    synthetic = {"cells": [
        {"cell_type": "code", "outputs": [
            {"output_type": "stream", "name": "stdout", "text": "SENTINEL 424242.4242"}]},
        {"cell_type": "markdown", "source": "no outputs here"}]}
    path = _write(tmp_path, "synthetic.ipynb", synthetic)
    assert len(PV.scan_notebook_outputs(path)) == 1
    with pytest.raises(RuntimeError):
        PV.assert_no_outputs(path)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(PV.PrivacyError):
        PV.assert_no_outputs(str(tmp_path / "nope.ipynb"))


def test_notebook_guard_still_works_on_real_notebooks(tmp_path):
    assert PV.assert_no_outputs(_write(tmp_path, "clean.ipynb", CLEAN_NB)) == 0
    assert PV.scan_notebook_outputs(_write(tmp_path, "clean.ipynb", CLEAN_NB)) == []
    with pytest.raises(PV.PrivacyError, match="still carry outputs"):
        PV.assert_no_outputs(_write(tmp_path, "dirty.ipynb", EXECUTED_NB))
    hits = PV.scan_notebook_outputs(_write(tmp_path, "dirty.ipynb", EXECUTED_NB))
    assert len(hits) == 1 and hits[0].n_numbers >= 1


def test_privacy_error_is_a_runtime_error():
    """Back-compat: existing callers catch RuntimeError."""
    assert issubclass(PV.PrivacyError, RuntimeError)


# ---------------------------------------------------------------------------
# 2. the separator-blind token test
# ---------------------------------------------------------------------------
OLD_BROKEN_TOKENS = ("main_dark", "loa_main", "processed-main-dark")

SEPARATOR_VARIANTS = [
    "/x/loa_main_dark_v1/dlacat-loa-main-dark-v1.fits",   # the real committed form
    "/x/LOA-MAIN-DARK/coadd-Main-Dark-705.fits",
    "/x/healpix/main/dark/107/10978/coadd-main-dark-10978.fits",
    "/x/processed-main-dark-0.h5",
    "/x/QSO_cat_loa_main_dark_healpix_v3-altbal.fits",
]


@pytest.mark.parametrize("path_str", SEPARATOR_VARIANTS)
def test_real_path_caught_regardless_of_separator_or_case(path_str):
    hits = PV.scan_json_artifact({"inputs": {"file": path_str}})
    assert any(h.kind == "path_token" for h in hits), f"missed real path {path_str!r}"


def test_the_original_miss_is_pinned():
    """The hyphenated real catalog path that the committed token set did NOT match."""
    leaked = "/x/gpdla_catalogs/loa-main-dark-v1/dlacat-loa-main-dark-v1.fits"
    blob = json.dumps({"inputs": {"dlacat": leaked}}).lower()
    assert not any(t in blob for t in OLD_BROKEN_TOKENS), (
        "test premise: the OLD token set misses this hyphenated real path")
    assert any(h.kind == "path_token" for h in PV.scan_json_artifact(
        {"inputs": {"dlacat": leaked}})), "the fixed scanner must catch it"


# ---------------------------------------------------------------------------
# 3. the JSON-artifact guard
# ---------------------------------------------------------------------------
def test_real_looking_artifact_must_fail(tmp_path):
    """The deliberately-real-looking fixture. If this ever passes, the guard is dead."""
    path = _write(tmp_path, "leak.json", REAL_LOOKING_ARTIFACT)
    with pytest.raises(PV.PrivacyError) as exc:
        PV.assert_json_artifact_mock_only(path)
    msg = str(exc.value)
    kinds = {h.kind for h in PV.scan_json_artifact(REAL_LOOKING_ARTIFACT)}
    # all three independent tells must fire, so removing any one still fails the file
    assert {"targetid", "path_token", "value_cooccurrence"} <= kinds, kinds
    assert "39627939081946569" in msg


def test_targetid_magnitude_is_decisive():
    """mock O(1e3-1e8) passes; real DESI O(1e16) fails."""
    assert PV.assert_json_artifact_mock_only({"targetid": 88123456}) == 0
    with pytest.raises(PV.PrivacyError):
        PV.assert_json_artifact_mock_only({"targetid": 39627939081946569})
    # an integral float must not launder it
    with pytest.raises(PV.PrivacyError):
        PV.assert_json_artifact_mock_only({"targetid": 3.9627939081946568e16})
    # booleans are ints in python -- must not be mistaken for TARGETIDs
    assert PV.assert_json_artifact_mock_only({"flag": True}) == 0


def test_value_cooccurrence_needs_a_hard_tell():
    """Science keys alone are not a leak; science keys + a real path are."""
    assert PV.assert_json_artifact_mock_only(
        {"measurement": {"dndx": 0.045}, "mock": "2lpt0"}) == 0
    with pytest.raises(PV.PrivacyError):
        PV.assert_json_artifact_mock_only(
            {"measurement": {"dndx": 0.045},
             "inputs": {"f": "/x/dlacat-loa-main-dark-v1.fits"}})


def test_mock_artifact_with_a_prose_disclaimer_passes(tmp_path):
    """A mock artifact that NAMES the real dataset in order to DENY using it is clean.
    Failing it would train people to delete the disclaimer. Reported, not fatal."""
    path = _write(tmp_path, "mock.json", MOCK_ARTIFACT)
    assert PV.assert_json_artifact_mock_only(path) == 0
    prose = [h for h in PV.scan_json_artifact(path) if h.kind == "prose_token"]
    assert prose, "the disclaimer must still be REPORTED as a soft hit"
    with pytest.raises(PV.PrivacyError):
        PV.assert_json_artifact_mock_only(path, strict_prose=True)


def test_json_guard_refuses_a_notebook(tmp_path):
    with pytest.raises(PV.PrivacyError, match="JSON-ARTIFACT guard"):
        PV.assert_json_artifact_mock_only(_write(tmp_path, "n.ipynb", CLEAN_NB))


def test_allow_tokens_cannot_whitelist_a_targetid():
    with pytest.raises(PV.PrivacyError):
        PV.assert_json_artifact_mock_only({"targetid": 39627939081946569},
                                          allow_tokens=("main-dark",))


# ---------------------------------------------------------------------------
# 4. every COMMITTED artifact in this worktree is privacy-clean
# ---------------------------------------------------------------------------
def test_all_committed_artifacts_are_privacy_clean():
    import subprocess
    rels = subprocess.run(["git", "ls-files", "*.json"], cwd=_REPO,
                          capture_output=True, text=True).stdout.split()
    offenders = []
    for rel in rels:
        try:
            PV.assert_json_artifact_mock_only(os.path.join(_REPO, rel))
        except PV.PrivacyError as exc:
            offenders.append(f"{rel}: {exc}")
    assert not offenders, "\n".join(offenders)
