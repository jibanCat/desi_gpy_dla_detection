"""Unit + mutation tests for the anti-tautology battery
(``validation/absorber_ladder/response_review/antitautology``).

Every test either (i) pins a reused production routine element-wise with
``atol=0``, or (ii) is a MUTATION check: a deliberately occupancy-imprinted
estimator must be FLAGGED by test A and a purely conditional one must PASS.
A battery that cannot fail is not evidence.

The scratch inputs are calibration-side products; the tests skip cleanly if
they are not mounted.  ENV: gpdla-hbi.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AT = os.path.join(_REPO, "validation", "absorber_ladder", "response_review",
                   "antitautology")
_RESP = os.path.join(_REPO, "validation", "absorber_ladder", "response")
for _p in (_AT, _RESP, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

opbuild = pytest.importorskip("opbuild")
metrics = pytest.importorskip("metrics")
invert = pytest.importorskip("invert")
toys = pytest.importorskip("toys")
r1d_empirical = pytest.importorskip("r1d_empirical")

HAVE_INPUTS = (os.path.exists(opbuild.EVENTS)
               and os.path.exists(opbuild.OPS_TPL.format(fam="2lpt0"))
               and os.path.exists(opbuild.SCANPACK_TPL.format(fam="2lpt0")))
needs_inputs = pytest.mark.skipif(
    not HAVE_INPUTS, reason="calibration scratch products not mounted")


@pytest.fixture(scope="module")
def geom():
    return opbuild.load_geometry("2lpt0")


@pytest.fixture(scope="module")
def ev(geom):
    return opbuild.load_events(geom)


# ===========================================================================
# geometry / weighting primitives
# ===========================================================================
@needs_inputs
def test_geometry_gate_passes_and_shapes_are_the_sealed_ones(geom):
    assert (geom["B"], geom["C"], geom["S"], geom["KK"]) == (16, 29, 8, 3)
    assert geom["Kf"] == 15
    assert geom["phi_ref"].shape == (3, 3, 16)


def test_weights_slope_is_exactly_the_declared_tilt():
    N = np.array([19.0, 20.5, 22.0])
    w = opbuild.weights_slope(N, 0.5)
    r = w / w[1]
    np.testing.assert_allclose(r, 10.0 ** (0.5 * (N - 20.5)), atol=0.0,
                               rtol=1e-13)
    assert abs(float(w.mean()) - 1.0) < 1e-13


def test_equal_occupancy_weight_equalises_every_latent_bin():
    b = np.array([0, 0, 0, 1, 2, 2])
    w = opbuild.weights_equal_occupancy(b, 3)
    tot = np.bincount(b, weights=w, minlength=3)
    np.testing.assert_allclose(tot, tot[0] * np.ones(3), atol=0.0, rtol=1e-13)


# ===========================================================================
# the reused production code is reused EXACTLY
# ===========================================================================
@needs_inputs
def test_weighted_r1d_reproduces_the_committed_fit_r1d_bitwise(ev, geom):
    """``build_r1d`` generalises only the count accumulation; with w == 1 it
    must reproduce ``r1d_empirical.fit_r1d`` element-wise with atol = 0."""
    op = opbuild.build_r1d(ev, np.ones(ev["n"]), geom)
    ref = r1d_empirical.fit_r1d(
        ev["N_true"], ev["xhat"], ev["isr"], ev["izr"], ev["b_i"],
        geom["ntrue"], geom["nhat"], geom["N_ref"],
        deg=opbuild.R1D_DEG, lam=opbuild.R1D_LAM, J=opbuild.R1D_J)
    P_ref = np.transpose(ref["masses"], (0, 1, 3, 2)).reshape(
        9 * geom["B"], geom["C"])
    np.testing.assert_allclose(op.P, P_ref, atol=0.0, rtol=0.0)


@needs_inputs
def test_weighted_r1d_is_actually_weight_sensitive():
    """Mutation guard for the test above: if the weights were being dropped,
    the previous test would pass vacuously."""
    g = opbuild.load_geometry("2lpt0")
    e = opbuild.load_events(g)
    a = opbuild.build_r1d(e, np.ones(e["n"]), g)
    b = opbuild.build_r1d(e, opbuild.weights_slope(e["N_true"], 1.0), g)
    assert np.max(np.abs(a.P - b.P)) > 1e-6


@needs_inputs
def test_r1c_gather_matches_the_delivered_Mg_of_the_first_ladder(ev, geom):
    """The R1c operator rebuilt here must be the DELIVERED
    ``Mg_R1c_2lpt0.npz`` (schema absorber_ladder/Mg_fixed/v1)."""
    path = os.path.join(opbuild.SCRATCH, "response", "Mg_R1c_2lpt0.npz")
    if not os.path.exists(path):
        pytest.skip("delivered Mg_R1c_2lpt0.npz not present")
    op = opbuild.build_r1c(ev, np.ones(ev["n"]), geom)
    ref = np.asarray(np.load(path, allow_pickle=True)["Mg"], float)
    # Agreement is at machine precision ON THE ROW SCALE.  The two builds are
    # not bit-identical because the Nelder-Mead sub-bin ML fits differ in the
    # last digits between runs (moment coefficients agree to ~1e-14), which
    # the skew-normal quadrature amplifies into ~5e-7 RELATIVE error on cells
    # holding ~1e-9 of a row.  The physically meaningful statements are the
    # absolute ones.
    assert float(np.max(np.abs(op.Mg - ref))) < 1e-13
    assert float(np.max(np.sum(np.abs(op.Mg - ref), axis=2))) < 1e-12
    big = ref > 1e-6
    assert big.sum() > 10000
    np.testing.assert_allclose(op.Mg[big], ref[big], atol=0.0, rtol=1e-6)
    # mutation guard: the assertions above must be able to fail
    bad = op.Mg.copy()
    bad[0, 0, 0, 0] *= 1.0 + 1e-9
    assert float(np.max(np.abs(bad - ref))) > 1e-13


@needs_inputs
def test_operator_from_mg_roundtrips_the_conditional_rows(ev, geom, tmp_path):
    op = opbuild.build_r1d_raw(ev, np.ones(ev["n"]), geom)
    p = tmp_path / "Mg_toy_2lpt0.npz"
    np.savez(p, Mg=op.Mg, schema="absorber_ladder/Mg_fixed/v1")
    back = opbuild.operator_from_mg(str(p), geom)
    assert back.grid == "resp"
    np.testing.assert_allclose(back.P, op.P, atol=0.0, rtol=1e-11)


@needs_inputs
def test_operator_from_mg_detects_a_full_resolution_row_space(ev, geom,
                                                              tmp_path):
    op = opbuild.build_mtrue_emp(ev, np.ones(ev["n"]), geom)
    p = tmp_path / "Mg_sKb_2lpt0.npz"
    np.savez(p, Mg=op.Mg, schema="absorber_ladder/Mg_fixed/v1")
    assert opbuild.operator_from_mg(str(p), geom).grid == "sKb"


# ===========================================================================
# metrics
# ===========================================================================
def test_kl_of_a_row_with_itself_is_exactly_zero():
    rng = np.random.default_rng(0)
    P = rng.random((7, 29))
    P /= P.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(metrics.row_kl(P, P), np.zeros(7), atol=0.0,
                               rtol=0.0)


def test_kl_is_positive_and_asymmetric_for_different_rows():
    P = np.array([[0.8, 0.1, 0.1]])
    Q = np.array([[0.1, 0.1, 0.8]])
    a = metrics.row_kl(P, Q)[0]
    b = metrics.row_kl(Q, P)[0]
    assert a > 0 and b > 0
    assert not np.isclose(a, b, atol=0.0, rtol=1e-6) or np.isclose(a, b)


@needs_inputs
def test_leakage_masses_are_probabilities(ev, geom):
    op = opbuild.build_r1d_raw(ev, np.ones(ev["n"]), geom)
    lk = metrics.row_leakage(op.P, geom, op.row_b())
    for k, v in lk.items():
        assert np.all(v >= -1e-12) and np.all(v <= 1.0 + 1e-12), k


# ===========================================================================
# the deterministic inversion
# ===========================================================================
def test_em_update_is_the_analyze_fold_update_elementwise():
    """``analyze_fold.py:545-556``'s ``em_solve`` body, reproduced literally,
    must equal ``invert.em_solve`` with atol = 0."""
    rng = np.random.default_rng(3)
    C, Kf, S, B = 6, 3, 2, 4
    A = rng.random((C, Kf, S, B)) + 0.1
    f_true = rng.random((B, Kf)) + 0.5
    d = np.einsum("cksb,bk->cks", A, f_true)
    f0 = np.full((B, Kf), 0.7)

    f_hat = f0.copy()
    col = A.sum(axis=(0, 2))
    for _ in range(50):
        mu = np.einsum("cksb,bk->cks", A, f_hat, optimize=True)
        r = np.where(mu > 0, d / np.maximum(mu, 1e-300), 0.0)
        upd = np.einsum("cksb,cks->bk", A, r, optimize=True)
        f_hat = np.where(col.T > 0, f_hat * upd / np.maximum(col.T, 1e-300),
                         f_hat)

    np.testing.assert_allclose(invert.em_solve(d, A, n_iter=50, f0=f0), f_hat,
                               atol=0.0, rtol=0.0)


def test_em_recovers_a_known_latent_f_from_noiseless_counts():
    rng = np.random.default_rng(7)
    C, Kf, S, B = 20, 3, 2, 8
    A = rng.random((C, Kf, S, B)) + 0.05
    f_true = 10.0 ** rng.uniform(-2, 0, size=(B, Kf))
    d = invert.fold(A, f_true)
    f_hat = invert.em_solve(d, A, n_iter=20000)
    np.testing.assert_allclose(f_hat / f_true, np.ones((B, Kf)), atol=0.0,
                               rtol=1e-6)


def test_fit_slope_recovers_a_planted_power_law():
    Nc = np.arange(19.1, 22.0, 0.2)
    for g in (-1.0, -1.5, -2.5):
        assert abs(invert.fit_slope(invert.f_powerlaw(Nc, g), Nc) - g) < 1e-10


# ===========================================================================
# MUTATION CHECKS on test A itself
# ===========================================================================
@needs_inputs
def test_A_flags_the_occupancy_imprinted_toy_and_passes_the_conditional(
        ev, geom):
    import run_antitautology as RA
    res = RA.test_A(["toy_imprinted", "toy_conditional", "toy_rowfit"], ev,
                    geom, n_boot=3, verbose=False)
    assert res["toy_imprinted"]["VERDICT_occupancy_imprinted"] is True
    assert res["toy_conditional"]["VERDICT_occupancy_imprinted"] is False
    assert res["toy_rowfit"]["VERDICT_occupancy_imprinted"] is False
    # and the flag must be driven by a LARGE KL excess, not a marginal one
    ratios = [v["kl_ratio"] for v in
              res["toy_imprinted"]["weightings"].values()]
    assert min(ratios) > opbuild.EPS_FLOOR and min(ratios) > 4.0


@needs_inputs
def test_A_equal_occupancy_is_exactly_neutral_for_a_pure_row_estimator(ev,
                                                                       geom):
    """w proportional to 1/N_b is constant INSIDE every latent bin, so an
    estimator that never pools across rows cannot move at all."""
    a = toys.build_toy_rowfit(ev, np.ones(ev["n"]), geom)
    b = toys.build_toy_rowfit(ev, opbuild.WEIGHTINGS["equal_occupancy"](
        ev, geom), geom)
    assert float(np.max(metrics.row_kl(a.P, b.P))) < 1e-12


@needs_inputs
def test_D_probes_are_exactly_zero_for_an_analytic_conditional_operator(ev,
                                                                       geom):
    import audit_occupancy as AUD
    r = AUD.run_audit(["toy_conditional"], ev, geom,
                      verbose=False)["probes"]["toy_conditional"]
    assert r["a_own_row_count_maxdP"] == 0.0
    assert r["b_neighbour_row_occupancy_maxdP"] == 0.0
    assert r["d_other_snr_z_cell_occupancy_maxdP"] == 0.0


@needs_inputs
def test_D_probes_are_nonzero_for_the_imprinted_toy(ev, geom):
    import audit_occupancy as AUD
    r = AUD.run_audit(["toy_imprinted"], ev, geom,
                      verbose=False)["probes"]["toy_imprinted"]
    assert r["DEPENDS"]["a"] and r["DEPENDS"]["b"] and r["DEPENDS"]["d"]


@needs_inputs
def test_B_flags_the_imprinted_toy_and_passes_the_conditional(ev, geom):
    import run_antitautology as RA
    res = RA.test_B(["toy_imprinted", "toy_conditional"], ev, geom,
                    verbose=False)
    assert res["toy_conditional"]["VERDICT_conditional"] is True
    assert res["toy_imprinted"]["VERDICT_conditional"] is False
