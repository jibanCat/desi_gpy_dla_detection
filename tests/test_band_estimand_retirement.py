"""test_band_estimand_retirement.py -- the PI band-estimand retirement (2026-07-28).

The PI decision: a paper-facing uncertainty band must be a CREDIBLE INTERVAL of a
faithful joint posterior whose MEDIAN is the reported point.  Everything else --
plug-in MAP + MC cloud, recentered cloud, independently-combined marginals -- is
DIAGNOSTIC ONLY and must say so on the artifact.

These tests are the anti-regression net.  They fail if:

  * ``recenter_band_on_point`` / ``recenter_differential_band_quantiles`` are called
    without the explicit ``diagnostic_only=True`` opt-in;
  * ``resolve_band_recenter`` lets ``band_recenter=True`` through without
    ``allow_diagnostic_recenter=True``;
  * ANY headline / real-data driver source sets ``band_recenter`` True (or defaults its
    CLI flag to True) -- a pure SOURCE scan, so it catches a re-flip even in a driver
    that is too expensive to execute in a test;
  * an artifact stamped ``paper_facing=True`` carries a recentered band.

Runs anywhere (no scratch, no catalogs, no GP).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

import CDDF_analysis.hbi.cddf_catalog_hbi as H
from CDDF_analysis.unblind import estimand as EST

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# 1. the primitive is gated
# =============================================================================
def test_recenter_band_on_point_refuses_without_diagnostic_optin():
    """The retired primitive raises unless the caller declares diagnostic intent."""
    s = np.linspace(0.0, 1.0, 101)
    with pytest.raises(EST.RecenteredBandRetired):
        H.recenter_band_on_point(s, 5.0)
    # explicit opt-in still works (diagnostic overlays must remain reproducible)
    out = H.recenter_band_on_point(s, 5.0, diagnostic_only=True)
    assert np.isclose(np.median(out), 5.0)
    assert np.isclose(np.std(out), np.std(s), rtol=1e-12)


def test_recenter_differential_band_quantiles_refuses_without_diagnostic_optin():
    stats = dict(q025=np.array([1.0]), q16=np.array([2.0]), q50=np.array([3.0]),
                 q84=np.array([4.0]), q975=np.array([5.0]))
    with pytest.raises(EST.RecenteredBandRetired):
        H.recenter_differential_band_quantiles(stats, np.array([10.0]))
    out = H.recenter_differential_band_quantiles(stats, np.array([10.0]),
                                                 diagnostic_only=True)
    assert out["q50"][0] == pytest.approx(10.0)


# =============================================================================
# 2. the config choke point
# =============================================================================
class _Cfg:
    def __init__(self, band_recenter=False, allow=False):
        self.band_recenter = band_recenter
        self.allow_diagnostic_recenter = allow


def test_resolve_band_recenter_default_off():
    assert H.resolve_band_recenter(_Cfg()) is False


def test_resolve_band_recenter_requires_explicit_diagnostic_optin():
    """band_recenter=True ALONE must raise -- a driver cannot re-enable the retired
    band by flipping one switch."""
    with pytest.raises(EST.RecenteredBandRetired):
        H.resolve_band_recenter(_Cfg(band_recenter=True), where="unit-test")


def test_resolve_band_recenter_allows_declared_diagnostic():
    assert H.resolve_band_recenter(_Cfg(band_recenter=True, allow=True)) is True


def test_hbiconfig_defaults_retire_recentering():
    cfg = H.HBIConfig(catalog_dir="x", truth_path="y", bal_cat_path="z",
                      molly_tsv="m", out_dir="o")
    assert cfg.band_recenter is False
    assert cfg.allow_diagnostic_recenter is False


# =============================================================================
# 3. NO headline driver may set band_recenter=True  (SOURCE scan)
# =============================================================================
#: drivers that produce a headline / paper-facing measurement artifact.
HEADLINE_DRIVERS = (
    "CDDF_analysis/hbi/track_c_tf_loa.py",
    "CDDF_analysis/hbi/track_c_perz_band.py",
    "CDDF_analysis/hbi/track_c_td_band.py",
    "CDDF_analysis/hbi/track_c_tf_2lpt1.py",
    "CDDF_analysis/hbi/track_c_tf_london0.py",
    "CDDF_analysis/hbi/track_c_tf_saclay.py",
    "CDDF_analysis/diagnostics/subdla/run_subdla_headline_full.py",
    "CDDF_analysis/diagnostics/subdla/subdla_loa0_validation.py",
    "CDDF_analysis/diagnostics/bal_metal_fp/arbiter/apply_broadtrough_veto_headline.py",
    "CDDF_analysis/diagnostics/track_c/track_c_czresolve_point_ab.py",
    "CDDF_analysis/diagnostics/track_c/track_c_ztilt_guard.py",
)

#: any assignment that turns recentering ON. Comments are stripped before matching so a
#: prose mention of ``band_recenter=True`` in a docstring/comment does not trip it.
_ON_PATTERNS = (
    re.compile(r"\bband_recenter\s*=\s*True\b"),
    re.compile(r"\ballow_diagnostic_recenter\s*=\s*True\b"),
)
def _add_argument_calls(code: str):
    """Yield the body of every ``add_argument(...)`` call, delimited by PAREN BALANCE.

    A regex cannot do this safely: the help strings contain ')' and ';', and the calls
    span lines. Balance-scanning is exact enough for a source guard.
    """
    for m in re.finditer(r"add_argument\(", code):
        i = m.end()
        depth = 1
        while i < len(code) and depth:
            if code[i] == "(":
                depth += 1
            elif code[i] == ")":
                depth -= 1
            i += 1
        yield code[m.end():i - 1]


def _band_recenter_cli_defaults_true(code: str) -> bool:
    """True if the add_argument call declaring --band-recenter carries default=True."""
    for body in _add_argument_calls(code):
        if "--band-recenter" not in body:
            continue
        if re.search(r"default\s*=\s*True", body):
            return True
    return False


def _strip_comments_and_strings(src: str) -> str:
    """Drop full-line comments, trailing comments and triple-quoted blocks so the scan
    sees CODE only."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    out = []
    for line in src.splitlines():
        stripped = line.split("#", 1)[0]
        out.append(stripped)
    return "\n".join(out)


@pytest.mark.parametrize("rel", HEADLINE_DRIVERS)
def test_no_headline_driver_enables_band_recenter(rel):
    """FAILS if a headline driver sets band_recenter=True (or defaults its CLI flag to
    True, or hard-codes the diagnostic opt-in). Recentering is retired for paper-facing
    output (PI, 2026-07-28)."""
    path = os.path.join(_REPO, rel)
    assert os.path.exists(path), f"headline driver missing: {rel}"
    with open(path) as fh:
        raw = fh.read()
    code = _strip_comments_and_strings(raw)
    for pat in _ON_PATTERNS:
        m = pat.search(code)
        assert m is None, (
            f"{rel}: headline driver turns band recentering ON -> {m.group(0)!r}. "
            "Recentering is DIAGNOSTIC-ONLY (PI, 2026-07-28); a headline/real-data "
            "driver must leave band_recenter=False.")
    assert not _band_recenter_cli_defaults_true(raw), (
        f"{rel}: the --band-recenter CLI flag defaults to True. Recentering is "
        "DIAGNOSTIC-ONLY (PI, 2026-07-28); the default must be False.")


def test_headline_driver_list_is_not_silently_empty():
    """Guard the guard: if the driver list is emptied, the parametrized test above
    would trivially pass."""
    assert len(HEADLINE_DRIVERS) >= 10


def test_the_source_guard_actually_catches_a_regression():
    """Guard the guard #2: prove the scanner fires on the exact re-flip it exists to
    stop -- a --band-recenter default flipped back to True, in a call whose help string
    contains the ')' and ';' characters that defeated a naive regex, and next to a
    NEIGHBOURING flag that legitimately defaults to True."""
    bad = (
        '    p.add_argument("--band-recenter", dest="band_recenter",\n'
        '                   action="store_true", default=True,\n'
        '                   help="DIAGNOSTIC ONLY (retired); requires the opt-in.")\n'
        '    p.add_argument("--omega-slope-extrap", action="store_true", default=True)\n'
    )
    good = bad.replace("default=True,\n", "default=False,\n")
    assert _band_recenter_cli_defaults_true(bad) is True
    assert _band_recenter_cli_defaults_true(good) is False
    # and the assignment scanner fires on a hard-coded cfg override
    src = _strip_comments_and_strings("        band_recenter=True, omega_slope_extrap=True,\n")
    assert any(p.search(src) for p in _ON_PATTERNS)


# =============================================================================
# 4. an artifact stamped paper_facing=True may not carry a recentered band
# =============================================================================
def _artifact(map_val, q50, *, paper_facing, estimand=None, jensen=None):
    band = dict(MAP=map_val, q16=map_val * 0.9, q50=q50, q84=map_val * 1.1,
                q025=map_val * 0.8, q975=map_val * 1.2, std=map_val * 0.1)
    if jensen is not None:
        band["jensen_shift"] = jensen
    md = dict(paper_facing=paper_facing)
    if estimand is not None:
        md["estimand"] = estimand
    return dict(metadata=md, measurement={"20.3": {"dndx": {"integrated": band}}})


def test_paper_facing_artifact_with_recentered_band_is_detected():
    """FAILS the artifact: a band whose q50 sits EXACTLY on the MAP is the rigid-shift
    signature, so it cannot be stamped paper_facing=True."""
    art = _artifact(1.0, 1.0, paper_facing=True, estimand=EST.POSTERIOR_MEDIAN_CI)
    res = EST.classify_estimand(art, name="fixture")
    assert res["estimand"] == EST.DIAGNOSTIC_RECENTERED, res
    assert res["paper_facing"] is False
    # and the loud guard refuses the artifact's own (stale) stamp
    art["metadata"]["estimand"] = res["estimand"]
    with pytest.raises(EST.RecenteredBandRetired):
        EST.assert_paper_facing(art["metadata"], where="fixture")


def test_recorded_jensen_shift_marks_the_artifact_recentered():
    art = _artifact(1.0, 0.93, paper_facing=False, jensen=-0.055)
    res = EST.classify_estimand(art, name="fixture")
    assert res["estimand"] == EST.DIAGNOSTIC_RECENTERED
    assert res["paper_facing"] is False


def test_unrecentered_mc_band_classifies_as_plugin_map_mc():
    art = _artifact(1.0, 0.93, paper_facing=False)
    res = EST.classify_estimand(art, name="fixture")
    assert res["estimand"] == EST.PLUGIN_MAP_MC
    assert res["paper_facing"] is False


def test_metadata_band_recenter_flag_alone_marks_it_recentered():
    art = _artifact(1.0, 0.93, paper_facing=False)
    art["metadata"]["band_recenter"] = True
    assert EST.classify_estimand(art)["estimand"] == EST.DIAGNOSTIC_RECENTERED


# =============================================================================
# 5. the stamping vocabulary
# =============================================================================
def test_stamp_marks_recentered_artifacts_diagnostic_and_not_paper_facing():
    md = {}
    EST.stamp_band_estimand(md, band_recenter=True)
    assert md["estimand"] == EST.DIAGNOSTIC_RECENTERED
    assert md["paper_facing"] is False
    assert "retired_reason" in md


def test_stamp_marks_plugin_map_mc_not_paper_facing():
    md = {}
    EST.stamp_band_estimand(md, band_recenter=False, posterior_sampled=False)
    assert md["estimand"] == EST.PLUGIN_MAP_MC
    assert md["paper_facing"] is False


def test_stamp_marks_posterior_median_ci_paper_facing():
    md = {}
    EST.stamp_band_estimand(md, band_recenter=False, posterior_sampled=True)
    assert md["estimand"] == EST.POSTERIOR_MEDIAN_CI
    assert md["paper_facing"] is True
    assert "retired_reason" not in md


def test_stamp_only_vetoes_never_reauthorizes():
    """paper_facing is a CONJUNCTION of gates: a prior False (e.g. the resp_kind kernel
    gate) must survive an otherwise-admissible band stamp."""
    md = {"paper_facing": False, "resp_kind": "kappa"}
    EST.stamp_band_estimand(md, band_recenter=False, posterior_sampled=True)
    assert md["estimand"] == EST.POSTERIOR_MEDIAN_CI
    assert md["paper_facing"] is False


def test_marginal_combined_is_its_own_class_and_not_paper_facing():
    md = {}
    EST.stamp_band_estimand(md, band_recenter=False, marginal_combined=True)
    assert md["estimand"] == EST.MARGINAL_COMBINED
    assert md["paper_facing"] is False


def test_vocabulary_is_closed():
    for name in (EST.POSTERIOR_MEDIAN_CI, EST.PLUGIN_MAP_MC, EST.DIAGNOSTIC_RECENTERED,
                 EST.MARGINAL_COMBINED, EST.POINT_ONLY, EST.UNKNOWN):
        assert name in EST.ESTIMAND_VOCABULARY
    assert EST.PAPER_FACING_ESTIMANDS == frozenset({EST.POSTERIOR_MEDIAN_CI})


# =============================================================================
# 6. the classifier round-trips through a file (retire-marking never deletes)
# =============================================================================
def test_mark_retired_writes_metadata_and_keeps_the_science_values(tmp_path):
    art = _artifact(1.0, 1.0, paper_facing=True)
    p = tmp_path / "a.json"
    p.write_text(json.dumps(art))
    before = json.loads(p.read_text())["measurement"]
    res = EST.mark_retired(str(p), dry_run=False)
    after = json.loads(p.read_text())
    assert res["estimand"] == EST.DIAGNOSTIC_RECENTERED
    assert after["metadata"]["paper_facing"] is False
    assert after["metadata"]["retired_reason"].startswith("RETIRED 2026-07-28")
    assert after["measurement"] == before          # values untouched, nothing deleted


def test_mark_retired_dry_run_does_not_write(tmp_path):
    art = _artifact(1.0, 1.0, paper_facing=True)
    p = tmp_path / "a.json"
    p.write_text(json.dumps(art))
    raw = p.read_text()
    res = EST.mark_retired(str(p), dry_run=True)
    assert res["written"] is False
    assert p.read_text() == raw


# =============================================================================
# 7. the measured estimand gap is recorded where the artifact carries it
# =============================================================================
_SUBDLA_MOCK = os.path.join(_REPO, "CDDF_analysis", "hbi", "subdla_mock_headline.json")


@pytest.mark.skipif(not os.path.exists(_SUBDLA_MOCK),
                    reason="sub-DLA mock headline artifact not present")
def test_subdla_mock_records_a_multi_sigma_point_vs_median_gap():
    """The sub-DLA mock headline records raw_median + jensen_shift explicitly. The gap
    must be MANY band half-widths -- the evidence that the point and the band are
    different estimands (so the band is not a credible interval for the point)."""
    with open(_SUBDLA_MOCK) as fh:
        d = json.load(fh)
    wb = d.get("window_band_195_203")
    if wb is None:
        pytest.skip("artifact has no window band")
    for kind in ("dndx", "omega"):
        rec = wb[kind]
        hw = 0.5 * (rec["q84"] - rec["q16"])
        gap = rec["raw_median"] - rec["MAP"]
        assert hw > 0
        assert abs(gap) / hw > 5.0, (
            f"{kind}: |raw_median-MAP|/halfwidth68 = {abs(gap)/hw:.2f}")
        assert gap > 0, f"{kind}: expected the sub-DLA MC median ABOVE the MAP"


# =============================================================================
# 8. the classifier must FAIL CLOSED and must not crash on the second schema
# =============================================================================
#: a real posterior artifact keys its point as ``point_q50`` beside q16/q84 -- there is
#: no ``MAP``/``point`` key anywhere in it.
def _posterior_artifact(*, paper_facing, estimand=EST.POSTERIOR_MEDIAN_CI, n=3):
    tiers = {}
    for i in range(n):
        tiers[f"tier{i}"] = dict(mean=1.0, sd=0.1, n_draws=2000, point_q50=1.0,
                                 q025=0.8, q16=0.9, q84=1.1, q975=1.2)
    md = dict(paper_facing=paper_facing)
    if estimand is not None:
        md["estimand"] = estimand
    return dict(metadata=md, posterior={"tiers": tiers})


def test_classifier_sees_bands_keyed_by_point_q50():
    """REGRESSION: _POINT_KEYS lacked point_q50/q50, so a posterior artifact with dozens
    of band records looked like a POINT_ONLY artifact and every structural check was
    silently skipped."""
    art = _posterior_artifact(paper_facing=True, n=7)
    res = EST.classify_estimand(art, name="fixture")
    assert res["n_band_records"] == 7, res


def test_producer_paper_facing_false_is_never_overturned():
    """FAIL CLOSED: the artifact's own producer stamped paper_facing=False. No amount of
    inference may re-authorize it -- a gate may veto, never re-authorize."""
    art = _posterior_artifact(paper_facing=False)
    res = EST.classify_estimand(art, name="fixture")
    assert res["estimand"] == EST.POSTERIOR_MEDIAN_CI
    assert res["paper_facing"] is False, res
    assert res["producer_paper_facing"] is False


def test_producer_paper_facing_false_vetoes_a_band_free_artifact_too():
    """The band-free (POINT_ONLY-shaped) return path is the one that actually shipped
    the bug: it returned early and never looked at metadata['paper_facing']."""
    art = dict(metadata=dict(estimand=EST.POSTERIOR_MEDIAN_CI, paper_facing=False),
               summary=dict(n=3))
    res = EST.classify_estimand(art, name="fixture")
    assert res["n_band_records"] == 0
    assert res["paper_facing"] is False, res


def test_producer_paper_facing_true_cannot_launder_a_recentered_band():
    """The veto is one-directional: a producer's True does NOT re-authorize."""
    art = _artifact(1.0, 1.0, paper_facing=True, estimand=EST.POSTERIOR_MEDIAN_CI)
    res = EST.classify_estimand(art, name="fixture")
    assert res["estimand"] == EST.DIAGNOSTIC_RECENTERED
    assert res["paper_facing"] is False


# --- the two incompatible metadata['estimand'] schemas ----------------------
_DICT_STAMP = {
    "quantity": "R0 = recovered / truth, INTEGRATED (z-marginalised)",
    "point": "PLUG-IN MAP of the catalog-HBI v3 estimator. NOT a posterior median.",
    "interval": "NONE. --point-only: no MC band was drawn.",
}


def test_classify_estimand_does_not_crash_on_a_dict_stamp():
    """REGRESSION: `stamped in ESTIMAND_VOCABULARY` raised TypeError (unhashable dict)
    on every artifact using the descriptive schema."""
    art = dict(metadata=dict(estimand=dict(_DICT_STAMP)), summary=dict(n=1))
    res = EST.classify_estimand(art, name="fixture")
    assert res["estimand"] in EST.ESTIMAND_VOCABULARY
    assert res["paper_facing"] is False
    assert res["estimand_detail"] == _DICT_STAMP


def test_dict_stamp_carrying_a_vocabulary_word_is_read_from_it():
    stamp = dict(_DICT_STAMP, **{"class": EST.PLUGIN_MAP_MC})
    art = dict(metadata=dict(estimand=stamp), summary=dict(n=1))
    res = EST.classify_estimand(art, name="fixture")
    assert res["estimand"] == EST.PLUGIN_MAP_MC
    assert res["estimand_detail"]["quantity"] == _DICT_STAMP["quantity"]


def test_assert_paper_facing_does_not_crash_on_a_dict_stamp():
    md = dict(estimand=dict(_DICT_STAMP), paper_facing=True)
    with pytest.raises(EST.RecenteredBandRetired):
        EST.assert_paper_facing(md, where="fixture")


def test_normalize_estimand_stamp_accepts_both_schemas():
    assert EST.normalize_estimand_stamp(EST.PLUGIN_MAP_MC) == (EST.PLUGIN_MAP_MC, None)
    label, detail = EST.normalize_estimand_stamp(dict(_DICT_STAMP))
    assert label is None and detail == _DICT_STAMP
    label, detail = EST.normalize_estimand_stamp("some free text")
    assert label is None and detail == {"legacy_estimand_text": "some free text"}


def test_normalize_estimand_metadata_emits_one_schema():
    """ACCEPT BOTH ON READ, EMIT ONE ON WRITE: after normalisation metadata['estimand']
    is always a vocabulary STRING and the descriptive dict lives under
    metadata['estimand_detail'] -- nothing is discarded."""
    md = dict(estimand=dict(_DICT_STAMP), paper_facing=False)
    out = EST.normalize_estimand_metadata(md, label=EST.POINT_ONLY)
    assert out is md                                   # in place
    assert md["estimand"] == EST.POINT_ONLY
    assert md[EST.ESTIMAND_DETAIL_KEY] == _DICT_STAMP
    # idempotent
    again = dict(md)
    EST.normalize_estimand_metadata(md)
    assert md == again


def test_stamp_band_estimand_migrates_a_pre_existing_dict_stamp():
    md = dict(estimand=dict(_DICT_STAMP))
    EST.stamp_band_estimand(md, band_recenter=False, posterior_sampled=False)
    assert md["estimand"] == EST.PLUGIN_MAP_MC
    assert md[EST.ESTIMAND_DETAIL_KEY] == _DICT_STAMP


# --- the real artifacts that exercised both defects -------------------------
_DICT_SCHEMA_ARTIFACTS = (
    os.path.join(_REPO, "CDDF_analysis", "hbi", "crossmock_transfer_artifact.json"),
    "/home/mfho/hbi_mcmc_wt/CDDF_analysis/hbi/calccddf_vs_hbi.json",
)
_FAIL_OPEN_ARTIFACT = (
    "/home/mfho/hbi_mcmc_wt/CDDF_analysis/hbi_mcmc/posterior_synthetic_smoke.json")


@pytest.mark.parametrize("path", _DICT_SCHEMA_ARTIFACTS)
def test_real_dict_schema_artifacts_classify_without_crashing(path):
    if not os.path.exists(path):
        pytest.skip(f"artifact absent: {path}")
    res = EST.classify_estimand(path)
    assert res["estimand"] in EST.ESTIMAND_VOCABULARY
    assert res["paper_facing"] is False


@pytest.mark.skipif(not os.path.exists(_FAIL_OPEN_ARTIFACT),
                    reason="second worktree artifact absent")
def test_real_posterior_smoke_artifact_is_not_certified_paper_facing():
    """The exact fail-open case: a synthetic SELF-calibration smoke whose producer
    stamped paper_facing=False, which the classifier certified True."""
    with open(_FAIL_OPEN_ARTIFACT) as fh:
        doc = json.load(fh)
    assert doc["metadata"]["paper_facing"] is False, "fixture assumption changed"
    res = EST.classify_estimand(doc, name=_FAIL_OPEN_ARTIFACT)
    assert res["n_band_records"] > 0, "point_q50 band records still invisible"
    assert res["paper_facing"] is False, res


# =============================================================================
# 9. the package surface: every imported estimand name is EXPORTED
# =============================================================================
def test_unblind_package_exports_every_name_it_imports_from_estimand():
    """The uncommitted __init__ diff imported ``estimand`` + 12 of its names but listed
    NONE of them in __all__, so ``from CDDF_analysis.unblind import *`` (and every
    tooling surface that trusts __all__) could not see the estimand vocabulary."""
    import CDDF_analysis.unblind as U

    assert "estimand" in U.__all__
    for name in ("classify_estimand", "band_estimand", "stamp_band_estimand",
                 "assert_paper_facing", "is_paper_facing",
                 "normalize_estimand_stamp", "normalize_estimand_metadata",
                 "RecenteredBandRetired", "ESTIMAND_VOCABULARY",
                 "ESTIMAND_DETAIL_KEY", "POSTERIOR_MEDIAN_CI", "PLUGIN_MAP_MC",
                 "DIAGNOSTIC_RECENTERED", "MARGINAL_COMBINED", "POINT_ONLY",
                 "UNKNOWN"):
        assert name in U.__all__, f"{name} imported but missing from __all__"
        assert getattr(U, name) is getattr(EST, name)


def test_unblind_all_is_fully_resolvable():
    """Guard the guard: every __all__ entry must actually exist on the package."""
    import CDDF_analysis.unblind as U
    missing = [n for n in U.__all__ if not hasattr(U, n)]
    assert not missing, missing
    assert len(set(U.__all__)) == len(U.__all__), "duplicate entries in __all__"
