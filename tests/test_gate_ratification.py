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


# ==========================================================================
# 1. THE RATIFICATION RECORD
# ==========================================================================

def test_the_three_ratified_criteria_are_recorded_with_date_and_authority():
    for key in ("fail_closed_framework", "matched_configuration_sbc",
                "chi2_dof_max"):
        assert RAT.is_ratified(key), key
        rec = RAT.record(key)
        assert rec["status"] == "RATIFIED"
        assert rec["contributes_to_pass_fail"] is True
        assert rec["date"] == "2026-07-29"
        assert "PI" in rec["authority"]
        assert rec["statement"].strip()
        assert rec["applies_to"], f"{key} names no code it governs"


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
    assert r and u and not (r & u)
    assert u == set(_SPAN_TOLERANCES)
    for key in _Z_ARMS + ("chi2_dof_max",):
        assert key in r, key
    rec = RAT.record("some_tolerance_added_tomorrow")
    assert rec["status"] == "UNKNOWN"
    assert rec["contributes_to_pass_fail"] is False
    assert RAT.is_ratified("some_tolerance_added_tomorrow") is False


def test_every_tolerance_in_GATE_has_a_ratification_record():
    """The point of the module: no number in a production fail-closed gate may
    be unaccounted for."""
    for key in RP.GATE:
        assert RAT.record(key)["status"] in ("RATIFIED", "UNRATIFIED"), key


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


def test_the_ratified_z_marginal_arms_still_gate(spack, fake_fold):
    """Decision 8 moved only the two SPAN numbers.  ``z_zbin_max`` is ratified
    and a 30-sigma z-marginal residual must still refuse."""
    fake_fold(_flat_tab(z=30.0))
    g = RP.forward_closure_gate(spack)
    assert g["pass"] is False, g
    assert any("by_z" in f and "max|z|" in f for f in g["failures"]), \
        g["failures"]


def test_the_gate_report_and_the_stamp_carry_the_ratification_state(spack):
    g = RP.forward_closure_gate(spack)
    assert set(g["gate_tolerances_unratified"]) == set(_SPAN_TOLERANCES)
    assert set(g["gate_tolerances_ratified"]) >= set(_Z_ARMS + ("chi2_dof_max",))
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
    tolerances cannot have false-alarm rates orders of magnitude apart."""
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


def test_matched_sbc_kwargs_reproduces_the_run_configuration(spack):
    """The escape hatch must actually work: the kwargs it hands back, fed
    through the same configuration builder, MATCH.  Otherwise the ratified
    requirement would be unsatisfiable in principle."""
    cfg = MA.ModelAConfig()
    kw = SBC.matched_sbc_kwargs(spack, cfg)
    sp = synthetic_pack(0, **kw["grid"], fp_frac=0.0)
    sampler = dict(kw["sampler"])
    sampler.pop("n_ranks")
    sbc_cfg = SBC._configuration(sp, prior=kw["prior"], sampler=sampler,
                                 resp_clamp=kw["resp_clamp"],
                                 reported_names=SBC._reported_names(sp))
    m = SBC.configuration_match(sbc_cfg, SBC.run_configuration(spack, cfg))
    assert m["matched"] is True, m["mismatches"]


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
