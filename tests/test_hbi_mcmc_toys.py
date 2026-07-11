"""test_hbi_mcmc_toys.py — validation-ladder rungs 1-3 for CDDF_analysis.hbi_mcmc.

Pure-synthetic, fixed seeds, small sampler settings (400/400, 2 chains) so the
whole file stays well under the ~3 minute budget in the gpdla-hbi env:

    conda run -n gpdla-hbi python -m pytest tests/test_hbi_mcmc_toys.py -v

Rung 1: NUTS vs exact Poisson-Gamma conjugate posterior (mean within 3*MCSE).
Rung 2: binned-Poisson f_b recovery (direct + response variant B) + a pure-
        function test that the jnp forward fold equals a numpy reference.
Rung 3: prior-QMC recycling integral vs brute-force trapezoid quadrature
        (rel. agreement <= 1%) + Lambda recovery within 2 sigma + per-object
        recycling ESS diagnostics.
"""
import numpy as np
import pytest

from CDDF_analysis.hbi_mcmc import __version__
from CDDF_analysis.hbi_mcmc.diagnostics import recycling_ess
from CDDF_analysis.hbi_mcmc.validation import rung1_conjugate as r1
from CDDF_analysis.hbi_mcmc.validation import rung2_binned_poisson as r2
from CDDF_analysis.hbi_mcmc.validation import rung3_recycling as r3

SEED = 0
NUM_WARMUP = 400
NUM_SAMPLES = 400
NUM_CHAINS = 2


# --- shared runs (module-scoped so each sampler runs once) ---------------------

@pytest.fixture(scope="module")
def rung1_result():
    return r1.run_rung1(seed=SEED, num_warmup=NUM_WARMUP,
                        num_samples=NUM_SAMPLES, num_chains=NUM_CHAINS)


@pytest.fixture(scope="module", params=["direct", "response"])
def rung2_result(request):
    return r2.run_rung2(variant=request.param, seed=SEED, num_warmup=NUM_WARMUP,
                        num_samples=NUM_SAMPLES, num_chains=NUM_CHAINS)


@pytest.fixture(scope="module")
def rung3_result():
    return r3.run_rung3(seed=SEED, num_warmup=NUM_WARMUP,
                        num_samples=NUM_SAMPLES, num_chains=NUM_CHAINS)


# --- package smoke -------------------------------------------------------------

def test_package_version_and_x64():
    import jax
    assert __version__ == "0.0.1"
    assert jax.config.jax_enable_x64
    assert jax.numpy.zeros(1).dtype == np.float64


def test_recycling_ess_units():
    S = 100
    assert recycling_ess(np.ones(S)) == pytest.approx(S)
    one_hot = np.zeros(S)
    one_hot[3] = 5.0
    assert recycling_ess(one_hot) == pytest.approx(1.0)
    # 2-D: per-row
    ess = recycling_ess(np.stack([np.ones(S), one_hot]))
    assert ess.shape == (2,)
    assert ess[0] == pytest.approx(S)
    assert ess[1] == pytest.approx(1.0)


# --- RUNG 1: conjugate ----------------------------------------------------------

def test_rung1_nuts_matches_analytic(rung1_result):
    res = rung1_result
    assert res["r_hat_max"] < 1.01
    assert res["n_divergent"] == 0
    assert res["ess_bulk_min"] > 100
    assert res["ess_tail_min"] > 100
    # posterior mean within 3 * Monte-Carlo SE of the exact conjugate mean
    assert abs(res["nuts_mean"] - res["analytic_mean"]) < 3.0 * res["mcse_mean"]
    # posterior sd within 10% of the exact conjugate sd (sd MCSE ~ sd/sqrt(2*ESS))
    assert abs(res["nuts_sd"] - res["analytic_sd"]) / res["analytic_sd"] < 0.10


# --- RUNG 2: binned Poisson ------------------------------------------------------

def test_rung2_forward_fold_matches_numpy_reference():
    """Pure-function check: the jnp fold at fixed f equals a plain numpy loop."""
    f = r2.make_f_true()
    R = r2.make_response_matrix()
    dX, dN = r2.DELTA_X, r2.DN_B
    # numpy reference
    lam_ref = dX * f * dN
    mu_ref = np.array([np.sum(R[c, :] * lam_ref) for c in range(len(f))])
    # jnp fold (the exact code path used inside the model)
    import jax.numpy as jnp
    np.testing.assert_allclose(np.asarray(r2.forward_fold(jnp.asarray(f), dX, dN)),
                               lam_ref, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(r2.forward_fold(jnp.asarray(f), dX, dN, R=R)),
        mu_ref, rtol=1e-12)
    # rows of R are stochastic by construction
    np.testing.assert_allclose(R.sum(axis=1), 1.0, rtol=1e-12)


def test_rung2_recovery(rung2_result):
    res = rung2_result
    assert res["r_hat_max"] < 1.05
    assert res["n_divergent"] == 0
    # loose, non-flaky 68% coverage bound on the fixed seed (spec: >= ~0.6)
    assert res["coverage68"] >= 0.60, (
        f"variant={res['variant']} coverage68={res['coverage68']}")
    # integrated total recovered without bias (well within 4 posterior sd)
    assert abs(res["total_mean"] - res["total_true"]) < 4.0 * res["total_sd"], (
        f"variant={res['variant']} total {res['total_mean']} vs "
        f"{res['total_true']} +- {res['total_sd']}")


# --- RUNG 3: recycling core -------------------------------------------------------

def test_rung3_qmc_matches_quadrature():
    """Recycled I_i(Lambda) vs brute-force trapezoid: <= 1% relative, 5 sl x 3 Lambda."""
    cmp_res = r3.compare_qmc_vs_quadrature(seed=SEED)
    assert cmp_res["rel_err"].shape == (5, 3)
    assert np.all(np.isfinite(cmp_res["rel_err"]))
    assert cmp_res["max_rel_err"] <= 0.01, cmp_res["rel_err"]


def test_rung3_lambda_recovery_and_ess(rung3_result):
    res = rung3_result
    assert res["r_hat_max"] < 1.02
    assert res["n_divergent"] == 0
    assert res["ess_bulk_min"] > 100
    # Lambda recovered within 2 sigma of truth
    assert res["lam_z"] <= 2.0, (
        f"lam {res['lam_mean']} +- {res['lam_sd']} vs true {res['lam_true']}")
    # per-object recycling ESS: finite everywhere, healthy on detected sightlines
    assert np.isfinite(res["ess_recycle_min_all"])
    assert res["ess_recycle_min_all"] > 1.0
    assert res["ess_recycle_min_detected"] > 20.0
    assert res["ess_recycle_median_detected"] > 50.0
