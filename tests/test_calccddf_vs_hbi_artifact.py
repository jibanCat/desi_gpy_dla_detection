# -*- coding: utf-8 -*-
"""Contract tests for the feed-forward (FF-B) aggregation artifact.

Covers:
  * the closure inputs are the full-scale 2026-07-11 runs (file counts, 0 skips,
    single-absorber slot);
  * 2LPT-0 is stamped as the ON-MOCK CALIBRATION / RECOVERY FLOOR in the
    ARTIFACT (not only in a comment), and London-0 / Saclay-0 as transfers;
  * NO Omega anywhere in the aggregate (B16);
  * the FF interval is present AND is honestly labelled as a sampling interval
    on a plug-in estimator, not a credible interval;
  * the HBI reference is the FORWARD artifact and the RETIRED kappa artifact is
    rejected fail-closed;
  * calc_cddf's retired multi-DLA increment path raises.
"""
import os
import json
import subprocess

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HBI = os.path.join(REPO, "CDDF_analysis", "hbi")

CLOSURES = {
    "2lpt0": (os.path.join(HBI, "calccddf_2lpt0_closure.json"), 1150),
    "london0": (os.path.join(HBI, "calccddf_london0_closure.json"), 1149),
    "saclay0": (os.path.join(HBI, "calccddf_saclay0_closure.json"), 1127),
}
AGGREGATE = os.path.join(HBI, "calccddf_vs_hbi.json")


def _skip_if_missing(path):
    if not os.path.exists(path):
        pytest.skip("missing artifact: {}".format(path))


# --------------------------------------------------------------------------- #
# 1. the closure inputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mock,expected", [(m, n) for m, (_, n) in CLOSURES.items()])
def test_closure_ran_at_full_scale(mock, expected):
    path, _ = CLOSURES[mock]
    _skip_if_missing(path)
    d = json.load(open(path))
    assert d["mock"] == mock
    assert d["n_files"] == expected
    assert d["n_files_total"] == expected
    assert d["n_files_skipped"] == 0
    assert d["checkpoint"] is False
    # slot-0 only: the multi-DLA increment path is retired/broken
    assert d["second"] == 0
    assert d["z_range"] == [2.0, 3.5]
    assert d["snr_min"] == 2.0
    assert d["dX_total"] > 0


@pytest.mark.parametrize("mock", list(CLOSURES))
def test_closure_carries_a_provenance_block(mock):
    path, _ = CLOSURES[mock]
    _skip_if_missing(path)
    prov = json.load(open(path)).get("provenance")
    assert prov, "closure artifact has no provenance block"
    assert prov["routine"] == "CDDF_analysis/hbi/calccddf_vs_hbi.py"
    assert prov["date"] == "2026-07-11"
    assert prov["code_commit"]


# --------------------------------------------------------------------------- #
# 2. the HBI reference must be FORWARD, never the retired kappa artifact
# --------------------------------------------------------------------------- #
def test_hbi_reference_is_forward_and_not_retired():
    from CDDF_analysis.hbi.calccddf_vs_hbi_artifact import load_hbi_forward
    doc, prov = load_hbi_forward()
    assert doc["metadata"]["resp_kind"] == "forward"
    assert "retired" not in doc["metadata"]
    assert prov["resp_kind"] == "forward"
    assert len(prov["branch_commit"]) == 40
    assert len(prov["blob_sha"]) == 40


def test_retired_kappa_artifact_is_rejected(monkeypatch):
    """Fail-closed: pointing the loader at the RETIRED artifact must raise."""
    import CDDF_analysis.hbi.calccddf_vs_hbi_artifact as A
    monkeypatch.setattr(A, "HBI_FORWARD_PATH",
                        "CDDF_analysis/hbi/subdla_mock_validation.json")
    with pytest.raises(RuntimeError) as e:
        A.load_hbi_forward()
    msg = str(e.value)
    assert "resp_kind" in msg or "retired" in msg


def test_figure_module_has_no_hardcoded_hbi_numbers():
    """The per-bin HBI table must equal the forward artifact, bit for bit."""
    from CDDF_analysis.hbi.calccddf_vs_hbi_artifact import load_hbi_forward
    import CDDF_analysis.hbi.calccddf_vs_hbi_fig as fig
    doc, _ = load_hbi_forward()
    for row in doc["per_bin"]["loa0"]:
        tru, est = fig.HBI_BAND_2LPT0[round(row["blo"], 3)]
        assert est == row["dndx_est"]      # exact: it is the same object
        assert tru == row["dndx_tru"]
    assert fig.HBI_BAND_2LPT0_RESP_KIND == "forward"
    # no un-stamped transfer legs may be quoted
    assert set(fig.HBI_CUM) == {"2lpt0"}
    # Omega must not travel with the figure tables (B16)
    assert "omega" not in fig.HBI_CUM["2lpt0"]


# --------------------------------------------------------------------------- #
# 3. the aggregate contract
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def agg():
    _skip_if_missing(AGGREGATE)
    return json.load(open(AGGREGATE))


def test_aggregate_labels_2lpt0_as_calibration_floor(agg):
    m = agg["mocks"]
    assert m["2lpt0"]["held_out"] is False
    assert "CALIBRATION" in m["2lpt0"]["role"].upper()
    assert "FLOOR" in m["2lpt0"]["role"].upper()
    for t in ("london0", "saclay0"):
        assert m[t]["held_out"] is True
        assert "TRANSFER" in m[t]["role"].upper()
    roles = agg["metadata"]["leg_roles"]
    assert roles["calibration_floor"] == ["2lpt0"]
    assert sorted(roles["held_out_transfer"]) == ["london0", "saclay0"]


def test_aggregate_contains_no_omega(agg):
    """B16: every Omega from the leaky truth f(N) is biased -> excluded."""
    blob = json.dumps(agg)
    # metadata may DISCUSS omega; the data payload may not contain it.
    payload = json.dumps({"mocks": agg["mocks"], "hbi_forward": agg["hbi_forward"]})
    assert "omega" not in payload.lower()
    assert agg["metadata"]["omega_excluded"]["excluded"] is True
    assert "B16" in agg["metadata"]["omega_excluded"]["reason"]
    assert "omega_excluded" in blob


def test_aggregate_stamps_provenance(agg):
    md = agg["metadata"]
    assert len(md["code_commit"]) == 40, md["code_commit"]
    assert md["routine"] == "CDDF_analysis/hbi/calccddf_vs_hbi_artifact.py"
    assert "calccddf_vs_hbi_artifact.py" in md["rederive"]
    assert md["input_files"], "no input file list"
    for rec in md["input_files"]:
        assert len(rec["sha256"]) == 64
        assert rec["git_tracked"] is True
        assert rec["n_files_skipped"] == 0


def test_aggregate_declares_the_ff_estimand(agg):
    est = agg["metadata"]["estimand"]
    ff = est["ff"].lower()
    assert "plug-in" in ff
    assert "posterior-weighted" in ff
    assert "alpha" in ff
    assert "DIFFERENT ESTIMANDS" in est["matching_rule"]


def test_ff_uncertainty_is_present_and_honestly_labelled(agg):
    u = agg["metadata"]["uncertainty"]
    assert u["is_credible_interval"] is False
    assert "SAMPLING INTERVAL ON A PLUG-IN ESTIMATOR" in u["kind"]
    assert "NOT a posterior credible interval" in u["honest_label"]
    assert "NOT a posterior credible interval" in u["figure_caption"]
    for m, d in agg["mocks"].items():
        assert d["fN"]["ci_is_credible_interval"] is False
        for lim, (lo, hi) in d["dndx"]["calccddf_68"].items():
            pt = d["dndx"]["calccddf"][lim]
            assert lo < pt < hi, (m, lim, lo, pt, hi)
            lo95, hi95 = d["dndx"]["calccddf_95"][lim]
            assert lo95 <= lo and hi <= hi95


def test_ff_interval_is_conservative_poisson_scale(agg):
    """The Poisson-limit half-width must track sqrt(N) on the count scale."""
    for m, d in agg["mocks"].items():
        dX = d["dX_total"]
        for lim in ("20.3", "20.0", "19.5"):
            n = d["dndx"]["calccddf_counts"][lim]
            lo, hi = d["dndx"]["calccddf_68"][lim]
            half = 0.5 * (hi - lo) * dX
            assert 0.8 * np.sqrt(n) < half < 1.3 * np.sqrt(n), (m, lim, half, np.sqrt(n))


def test_aggregate_dndx_matches_the_closure_inputs(agg):
    for m, (path, _) in CLOSURES.items():
        _skip_if_missing(path)
        d = json.load(open(path))
        ref = d["cumulative"]["R0_calccddf"]["dndx"]
        got = agg["mocks"][m]["dndx"]["R0_calccddf"]
        for k in ("20.3", "20.0", "19.5", "band_195_203"):
            assert got[k] == pytest.approx(ref[k], rel=1e-12)


def test_aggregate_warns_about_the_z_tilt(agg):
    w = agg["metadata"]["z_resolved_warning"]
    assert "z-tilt" in w
    assert "MARGINALISED" in w.upper()


def test_aggregate_does_not_quote_unstamped_transfer_legs(agg):
    """crossmock_transfer_loa0.json is untracked + '-dirty' => not quotable."""
    assert set(agg["hbi_forward"]) == {"2lpt0"}
    assert "NOT QUOTABLE" in agg["metadata"]["hbi_coverage"]


def test_aggregate_is_mock_only(agg):
    blob = json.dumps(agg)
    assert "main_dark" not in blob
    assert "main-dark" not in blob.replace("loa main-dark", "")


# --------------------------------------------------------------------------- #
# 4. calc_cddf multi-DLA retirement guard
# --------------------------------------------------------------------------- #
def test_multi_dla_increment_path_raises():
    from CDDF_analysis import calc_cddf
    cat = calc_cddf.DLACatalogue.__new__(calc_cddf.DLACatalogue)
    cat.sub_dla = False
    cat._log_norm_like = lambda spec, second=False: np.zeros(4)
    cat._p_dla = lambda second=False: np.zeros(4)
    assert calc_cddf.ALLOW_BROKEN_MULTI_DLA is False
    with pytest.raises(RuntimeError) as e:
        cat._get_prob_dla_this_bin(0, np.array([0, 1]), second=1)
    msg = str(e.value)
    assert "RETIRED" in msg
    assert "b00e6e4" in msg
    assert "SINGLE_ABSORBER_MODEL=1" in msg


def test_single_absorber_path_is_untouched_by_the_guard():
    from CDDF_analysis import calc_cddf
    cat = calc_cddf.DLACatalogue.__new__(calc_cddf.DLACatalogue)
    cat.sub_dla = False
    cat._log_norm_like = lambda spec, second=False: np.log(np.full(4, 0.25))
    cat._p_dla = lambda second=False: np.full(4, 0.5)
    out = cat._get_prob_dla_this_bin(0, np.array([0, 1]), second=False)
    np.testing.assert_allclose(out, [0.125, 0.125])


def test_module_status_is_written_down():
    """The retire-or-not question must be settled IN THE MODULE, in writing."""
    from CDDF_analysis import calc_cddf
    doc = calc_cddf.__doc__
    assert "MODULE STATUS" in doc
    assert "PARTIAL RETIREMENT" in doc
    assert "SINGLE_ABSORBER_MODEL=1" in doc
    assert "b00e6e4" in doc


# --------------------------------------------------------------------------- #
# 5. the aggregation entry point actually runs
# --------------------------------------------------------------------------- #
def test_aggregator_reruns_and_reproduces(tmp_path):
    for path, _ in CLOSURES.values():
        _skip_if_missing(path)
    _skip_if_missing(AGGREGATE)
    out = tmp_path / "rerun.json"
    cmd = [os.sys.executable,
           os.path.join(HBI, "calccddf_vs_hbi_artifact.py"), "--out", str(out), "--in"]
    cmd += ["{}={}".format(m, p) for m, (p, _) in CLOSURES.items()]
    subprocess.check_call(cmd, cwd=REPO)
    a = json.load(open(AGGREGATE))
    b = json.load(open(out))
    assert a["mocks"] == b["mocks"]
    assert a["hbi_forward"] == b["hbi_forward"]
