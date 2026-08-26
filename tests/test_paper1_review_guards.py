"""Guards added or made testable in the Paper-1 code review (2026-08-26):
contract_guards_check.ga_partition (G-A not applicable on a real pack), the
superseded-directory refusal and pack-hash expectation of cc_pool_posterior,
bitrepro_check.compare / fail-closed main, and perz_gate.gate_one (criteria incl.
the B3 exemption and the B5 in-95 rescue)."""
import json
import types
import numpy as np
import pytest

from CDDF_analysis.hbi_mcmc import contract_guards_check as G
from CDDF_analysis.hbi_mcmc import bitrepro_check as B
from CDDF_analysis.hbi_mcmc import perz_gate as P
from CDDF_analysis.hbi_mcmc import cc_pool_posterior as C


# ---------------------------------------------------------------- G-A on a real pack
def test_ga_partition_is_not_applicable_when_truth_is_the_zero_sentinel():
    pk = types.SimpleNamespace(truth_counts=np.zeros((4, 3)))
    r = G.ga_partition(pk, level=0.16, fp_share=0.9, level_tol=0.06)
    assert r["status"] == "NOT_APPLICABLE_REAL_PACK" and r["level_mu_over_obs"] == 0.16


def test_ga_partition_pass_and_fail_on_a_mock_pack():
    pk = types.SimpleNamespace(truth_counts=np.ones((4, 3)))
    assert G.ga_partition(pk, 1.03, 0.1, 0.06)["status"] == "PASS"
    assert G.ga_partition(pk, 0.80, 0.1, 0.06)["status"] == "FAIL"


# ---------------------------------------------------------------- pooling refusals
def _fake_run(path, pack="/x/pack.npz"):
    json.dump({"pack": pack, "diagnostics": {"estimand_mixing": {}, "divergences": 0},
               "guards": {"G_A_real_mode": {"status": "PASS"}}}, open(path, "w"))


def test_pool_refuses_a_superseded_directory(tmp_path):
    d = tmp_path / "real_pack_v1"; d.mkdir()
    _fake_run(d / "REAL_ln_s20260822.json")
    (d / "SUPERSEDED_CANDIDATE_STATUS.json").write_text("{}")
    with pytest.raises(SystemExit, match="superseded-status sidecar"):
        C.main(["--runs", str(d / "REAL_ln_s20260822.json"), "--out", str(tmp_path / "o.json")])


def test_pool_refuses_the_wrong_pack_hash(tmp_path, monkeypatch):
    d = tmp_path / "cp3"; d.mkdir()
    pack = tmp_path / "pack.npz"; pack.write_bytes(b"not-a-pack")
    _fake_run(d / "REAL_ln_s20260822.json", pack=str(pack))
    # select_runs must see a converged run so that the refusal tested is the HASH one
    monkeypatch.setattr(C, "select_runs", lambda recs, rule: {"included": recs, "excluded": [], "needs_deep_rerun": []})
    with pytest.raises(SystemExit, match="expect-pack-sha256"):
        C.main(["--runs", str(d / "REAL_ln_s20260822.json"), "--out", str(tmp_path / "o.json"),
                "--expect-pack-sha256", "0" * 64])


# ---------------------------------------------------------------- bit-reproduction check
def _val(bias0, bias3, perz=(0.1, 0.2), div=1):
    return {"thresholds": {"ge20.0": {"median_bias_pct": bias0}, "ge20.3": {"median_bias_pct": bias3}},
            "perz_recovery": {"estimand": {t: {"paper1_bins": [{"bin": "B1", "median_bias_pct": perz[0]},
                                                                 {"bin": "B2", "median_bias_pct": perz[1]}]}
                                           for t in ("ge20.0", "ge20.3")}},
            "divergences": div}


def test_bitrepro_compare_identical_and_different():
    r = B.compare(_val(0.15, 1.97), _val(0.15, 1.97))
    assert r["thresholds_identical"] and r["perz_identical"] and r["max_abs_diff_allz_pct"] == 0.0
    r = B.compare(_val(0.15, 1.97), _val(0.15, 2.10, perz=(0.1, 0.3)))
    assert not r["thresholds_identical"] and not r["perz_identical"] and abs(r["max_abs_diff_allz_pct"] - 0.13) < 1e-12


def test_bitrepro_main_fails_closed_on_a_missing_reference(tmp_path):
    new = tmp_path / "new"; new.mkdir(); ref = tmp_path / "ref"; ref.mkdir()
    json.dump(_val(0.1, 1.0), open(new / "cp2_ln_w1500_2lpt0_s20260811.json", "w"))
    with pytest.raises(SystemExit, match="missing"):
        B.main(["--new-dir", str(new), "--ref-dir", str(ref), "--families", "2lpt0"])


# ---------------------------------------------------------------- per-z gate v2
def _gate_input(rhat=1.01, div=0, a0=0.1, a3=1.5, b0=None, b3=None, in95=None):
    b0 = b0 or {"B1": 0.5, "B2": -0.5, "B3": 1.0, "B4": -1.0, "B5": 0.7}
    b3 = b3 or {"B1": -0.8, "B2": 1.4, "B3": 6.7, "B4": 0.5, "B5": 0.7}
    in95 = in95 or {k: True for k in b0}
    mk = lambda bb: [{"bin": k, "median_bias_pct": v, "truth_in_95": in95[k], "available": True} for k, v in bb.items()]
    return {"pack": "/x/scanpack_2lpt0_b300.npz", "divergences": div, "n_draws": 1000,
            "diagnostics": {"estimand_mixing": {"dndx_dla_20p0_allz": {"split_rhat": rhat}, "dndx_dla_20p3_allz": {"split_rhat": rhat}},
                            "fp_mode": "informative_ln", "fp_s_empty": None, "fp_total_scale": 1.0, "t_scale": 1.0},
            "thresholds": {"ge20.0": {"median_bias_pct": a0}, "ge20.3": {"median_bias_pct": a3}},
            "perz_recovery": {"estimand": {"ge20.0": {"paper1_bins": mk(b0)}, "ge20.3": {"paper1_bins": mk(b3)}}}}


def test_gate_passes_the_cp2_like_record_with_b3_exempt():
    r = P.gate_one(_gate_input())
    assert r["status"] == "PASS" and r["fails"] == [] and r["named_residual_B3_ge20p3"] == 6.7


def test_gate_fails_on_rhat_and_on_a_b1_excursion():
    assert any("rhat" in f for f in P.gate_one(_gate_input(rhat=1.2))["fails"])
    b3 = {"B1": 4.5, "B2": 1.4, "B3": 6.7, "B4": 0.5, "B5": 0.7}
    assert any(f.startswith("B1 ge20.3") for f in P.gate_one(_gate_input(b3=b3))["fails"])


def test_gate_b5_rescued_only_by_truth_in_95():
    b3 = {"B1": -0.8, "B2": 1.4, "B3": 6.7, "B4": 0.5, "B5": 5.0}
    assert P.gate_one(_gate_input(b3=b3))["status"] == "PASS"                       # in95 True by default
    in95 = {"B1": True, "B2": True, "B3": True, "B4": True, "B5": False}
    assert P.gate_one(_gate_input(b3=b3, in95=in95))["status"] == "FAIL"
