"""Tests for the S6 completeness calibration-uncertainty propagation.

Nothing here reads real data: the end-to-end test builds SYNTHETIC run JSONs
with a known exact-linear response, so the fitting, the sigma formula and the
privacy of stdout are all checked without a single real value.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_S6 = os.path.join(_REPO, "validation", "s6_propagation")
L = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13"
COV = os.path.join(L, "completeness", "C1nsadd_covariance_2lpt0.npz")
TAB = os.path.join(L, "completeness", "C_C1nsadd_2lpt0.npz")
DRAWS = os.path.join(L, "s6_propagation", "tables")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_S6, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("build_s6_draws")
P = _load("propagate_s6")

needs_scratch = pytest.mark.skipif(
    not (os.path.exists(COV) and os.path.exists(TAB)),
    reason="the sealed calibration products are not staged on this host")
needs_draws = pytest.mark.skipif(
    not os.path.exists(os.path.join(DRAWS, "C_S6_draw7.npz")),
    reason="the S6 draw tables have not been built on this host")


# ------------------------------------------------------------------ gate (1)
@needs_scratch
def test_released_evaluator_reproduces_the_frozen_table():
    """The RELEASED evaluator at beta-hat rebuilds C_C1nsadd_2lpt0 to <= 1e-12."""
    gate = B.verify_frozen(L, "2lpt0")
    assert gate["max_abs_dev_live"] <= 1e-12
    assert gate["max_abs_dev_all"] <= 1e-12
    assert gate["beta_identical_to_frozen_table"]
    # and it is a real check: a perturbed beta must NOT reproduce the table
    ec = B.load_evaluator(L)
    cv = np.load(COV, allow_pickle=True)
    ct = np.load(TAB, allow_pickle=True)
    bad = np.asarray(cv["beta"], float).copy()
    bad[1] += 0.01
    C_bad = B.build_table(bad, cv["x_b"], cv["u_live"], cv["live_idx"], 8, ec)
    assert np.abs(C_bad - np.asarray(ct["C_fixed"], float)).max() > 1e-4


# --------------------------------------------------- the sealed draw identity
@needs_draws
def test_draws_are_bootstrap_indices_0_to_7_in_stored_order():
    cv = np.load(COV, allow_pickle=True)
    bb = np.asarray(cv["beta_bootstrap"], float)
    ec = B.load_evaluator(L)
    for i in range(8):
        z = np.load(os.path.join(DRAWS, f"C_S6_draw{i}.npz"), allow_pickle=True)
        beta_i = np.asarray(z["beta"], float)
        assert np.array_equal(beta_i, bb[i]), f"draw {i} is not beta_bootstrap[{i}]"
        prov = json.loads(str(z["provenance"]))
        assert prov["draw_index"] == i
        assert prov["fitted_to_real_data"] is False
        assert prov["beta_updated_with_real_data"] is False
        # the stored table IS the evaluator's table at that beta
        C = B.build_table(beta_i, cv["x_b"], cv["u_live"], cv["live_idx"], 8, ec)
        assert np.array_equal(C, np.asarray(z["C_fixed"], float))
    # the eight are distinct and none of them is beta-hat
    betas = np.array([np.load(os.path.join(DRAWS, f"C_S6_draw{i}.npz"))["beta"]
                      for i in range(8)])
    assert len({tuple(b) for b in betas}) == 8
    assert not any(np.array_equal(b, np.asarray(cv["beta"], float)) for b in betas)


@needs_draws
def test_draw_tables_are_well_formed_and_clamped():
    ct = np.load(TAB, allow_pickle=True)
    live = np.asarray(ct["live_strata_mask"], bool)
    ec = B.load_evaluator(L)
    cv = np.load(COV, allow_pickle=True)
    for i in range(8):
        z = np.load(os.path.join(DRAWS, f"C_S6_draw{i}.npz"), allow_pickle=True)
        C = np.asarray(z["C_fixed"], float)
        assert C.shape == (8, 16)
        assert np.all(np.isfinite(C)) and C.min() >= 0.0 and C.max() <= 1.0
        assert np.array_equal(C[~live], np.zeros_like(C[~live]))   # dead strata
        assert np.all(C[live] > 0.0)
        assert np.array_equal(np.asarray(z["C_fixed_sd"]), np.zeros_like(C))
        assert np.array_equal(np.asarray(z["live_strata_mask"], bool), live)
        for k in ("ntrue_edges", "snr_edges", "kz_to_K", "b_to_cell"):
            assert np.array_equal(np.asarray(z[k]), np.asarray(ct[k]))
        # the runner's 2-D branch must be taken: no (B,Kf,S) key may exist
        assert "C_fixed_bks" not in z.files and "C_fixed_bKs" not in z.files
        # the production clamp is ACTIVE: an S/N far above the top calibrated
        # stratum gives exactly the top live stratum's completeness
        beta_i = np.asarray(z["beta"], float)
        top = ec.evaluate_completeness(20.3, snr=1e4, coef=beta_i, N0=20.0,
                                       clamp=True)
        cap = ec.evaluate_completeness(20.3, log10_snr=ec.LOG10_SNR_CLAMP_HI,
                                       coef=beta_i, N0=20.0, clamp=True)
        assert float(top) == pytest.approx(float(cap), abs=0.0, rel=1e-15)
        uncl = ec.evaluate_completeness(20.3, snr=1e4, coef=beta_i, N0=20.0,
                                        clamp=False)
        assert abs(float(uncl) - float(cap)) > 1e-6     # the clamp really bites
    assert np.array_equal(np.asarray(cv["live_idx"], int), np.where(live)[0])


# -------------------------------------------------------- the linear response
def test_linear_fit_recovers_J_on_an_exact_linear_case():
    rng = np.random.default_rng(20260914)
    dbeta = rng.normal(size=(9, 6)) * 0.05
    J_true = np.array([0.3, -1.1, 0.02, 0.7, -0.4, 0.05])
    a_true = 0.0612
    m = a_true + dbeta @ J_true
    fit = P.fit_linear_response(dbeta, m)
    assert fit["a"] == pytest.approx(a_true, abs=1e-12)
    assert np.allclose(fit["J"], J_true, rtol=0, atol=1e-12)
    assert fit["resid_rms"] < 1e-12
    # a curved response leaves a residual the 25 % rule can see
    m2 = m + 3.0 * (dbeta[:, 1] ** 2)
    fit2 = P.fit_linear_response(dbeta, m2)
    assert fit2["resid_rms"] > 1e-4
    with pytest.raises(ValueError):
        P.fit_linear_response(dbeta, m[:3])


def test_sigma_formula():
    J = np.array([1.0, -2.0, 0.0, 0.5, 0.0, 0.0])
    Sig = np.diag([4.0, 1.0, 9.0, 16.0, 1.0, 1.0])
    assert P.sigma_from_J(J, Sig) == pytest.approx(
        np.sqrt(1.0 * 4.0 + 4.0 * 1.0 + 0.25 * 16.0), rel=0, abs=1e-12)
    # a correlated covariance is handled by the full quadratic form
    S2 = np.eye(6)
    S2[0, 1] = S2[1, 0] = 0.5
    assert P.sigma_from_J(J, S2) == pytest.approx(
        np.sqrt(float(J @ S2 @ J)), abs=1e-12)
    assert P.sigma_from_J(np.zeros(6), Sig) == 0.0
    with pytest.raises(ValueError):
        P.sigma_from_J(J, np.eye(3))


def test_direct_spread_and_nonlinearity_rule_constant():
    assert P.NONLINEARITY_RULE == 0.25
    ds = P.direct_spread([1.0, 2.0, 3.0, 4.0])
    assert ds["range"] == pytest.approx(3.0)
    assert ds["sd"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))
    assert ds["n"] == 4


# -------------------------------------------------- end-to-end, no real values
def _fake_estimands(vals):
    """A run-JSON estimands block whose every median is ``vals`` (scaled)."""
    def q3(v):
        return dict(post_p16_50_84=[v * 0.99, v, v * 1.01],
                    post_p2p5_97p5=[v * 0.98, v * 1.02])

    def q5(v):
        return dict(bin="X", z=[2.0, 2.2], available=True, dX=1.0,
                    post_p2p5_16_50_84_97p5=[v * 0.98, v * 0.99, v, v * 1.01,
                                             v * 1.02])
    pz = {}
    for thr in ("ge20.0", "ge20.3"):
        bins = []
        for n, b in enumerate(P.PAPER1_BINS):
            e = q5(vals * (1.0 + 0.1 * n))
            e["bin"] = b
            bins.append(e)
        pz[thr] = dict(native_cells=[], coarse_blocks=[], paper1_bins=bins,
                       allz={})
    rb = []
    for n in range(10):
        e = q5(vals * (1.0 + 0.01 * n))
        e["bin"] = [19.7 + 0.2 * n, 19.9 + 0.2 * n]
        rb.append(e)
    return dict(estimand="POSTERIOR_MEDIAN_CI",
                thresholds_allz={"ge20.0": q3(vals * 1.5), "ge20.3": q3(vals)},
                reporting_bins_0p2dex=rb,
                perz_posterior=dict(z_cells=[], dX_k=[], estimand=pz),
                omega_20p3_21p6_allz=q3(vals * 0.01))


@needs_draws
def test_end_to_end_recovers_J_and_prints_no_estimand_value(tmp_path, capsys):
    cv = np.load(COV, allow_pickle=True)
    beta_hat = np.asarray(cv["beta"], float)
    bb = np.asarray(cv["beta_bootstrap"], float)
    J_true = np.array([0.011, -0.023, 0.005, 0.002, -0.017, 0.003])
    base = 0.0612345678
    runs = tmp_path / "runs"
    runs.mkdir()
    for i in list(range(8)) + [None]:
        beta = beta_hat if i is None else bb[i]
        v = base + float((beta - beta_hat) @ J_true)
        tag = "betahat" if i is None else f"draw{i}"
        cpath = (os.path.join(L, "real_c1_inputs", "C_C1nsadd_real.npz")
                 if i is None else os.path.join(DRAWS, f"C_S6_draw{i}.npz"))
        j = dict(estimands=_fake_estimands(v),
                 fixed_files=dict(c=cpath + "@deadbeef"),
                 lam_cut=dict(J=1, j=0, lam_fixed=7.33, fp_a0=None),
                 run_config=dict(seed=20260811, code_commit="0" * 40),
                 sampler_health=dict(divergences=0))
        json.dump(j, open(runs / f"RUN_S6_{tag}.json", "w"))
    pooled = tmp_path / "POOLED.json"
    json.dump(dict(pools=dict(all=_fake_estimands(base))), open(pooled, "w"))
    oj, om = tmp_path / "S6.json", tmp_path / "S6.md"
    P.main(["--runs-dir", str(runs), "--tables-dir", DRAWS,
            "--pooled", str(pooled), "--out-json", str(oj), "--out-md", str(om)])
    out = json.load(open(oj))
    r = out["estimands"]["dndx_ge20p3_allz"]
    assert np.allclose(r["fit"]["J"], J_true, rtol=0, atol=1e-9)
    assert r["fit"]["resid_rms"] < 1e-12
    assert r["nonlinearity"]["triggered"] is False
    expect = float(np.sqrt(J_true @ np.asarray(cv["cov_bootstrap"], float) @ J_true))
    assert r["sigma_S6_bootstrap"] == pytest.approx(expect, rel=1e-10)
    assert r["quoted_S6"]["absolute"] == pytest.approx(expect, rel=1e-10)
    assert len(out["estimands"]) == 23        # 2 thresholds + Omega + 10 z + 10 N
    assert len(out["draws"]) == 8
    # PRIVACY: stdout carries hw68 ratios only — no estimand value, no median
    cap = capsys.readouterr().out
    for nm, rr in out["estimands"].items():
        for val in (rr["record_median"], rr["quoted_S6"]["absolute"],
                    rr["betahat_run_median"]):
            assert f"{val:.6g}" not in cap
            assert f"{val:.4g}" not in cap
    assert "hw68" in cap
    # ... but the private product does carry them
    assert "record_median" in open(oj).read()
    assert "S6 per estimand" in open(om).read()
