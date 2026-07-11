"""test_modelA_forward.py — Q3 Model A: pack schema validation + forward-fold oracle.

Fast, sampler-free tests:
  * pack round-trip (real grid + small test grid), default-grid enforcement,
    precise refusal on every schema-rule violation class.
  * skew-normal CDF (jnp, Gauss-Legendre Owen's T) vs scipy exact:
    zero skew and |skew| <= 1.4, tolerance 1e-4 on bin masses (measured ~1e-13;
    the brief's Azzalini fallback was NOT needed — the GL rule is exact to
    quadrature accuracy and fully differentiable).
  * fold_mu (pure jnp) vs fold_mu_reference (independent numpy/scipy oracle)
    at multiple random parameter points: rtol 1e-10 (+ tiny scaled atol for
    far-tail cells), small grid and one full-grid point; jit-compilability.

Run: conda run -n gpdla-hbi python -m pytest tests/test_modelA_forward.py -v
"""
import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from CDDF_analysis.hbi_mcmc import pack as packmod  # noqa: E402  (enables x64)
from CDDF_analysis.hbi_mcmc import forward as fwd  # noqa: E402
from CDDF_analysis.hbi_mcmc.pack import (  # noqa: E402
    ModelAPack, PackSchemaError, load_pack, save_pack, small_test_grid,
    synthetic_pack)

SEED = 0


@pytest.fixture(scope="module")
def small_pack():
    return synthetic_pack(SEED, **small_test_grid())


# --- pack: round trip + validation ------------------------------------------------

def test_small_pack_roundtrip(small_pack, tmp_path):
    p = tmp_path / "modelA_pack_test.npz"
    save_pack(small_pack, p, allow_nonstandard_grid=True)
    assert (tmp_path / "modelA_pack_test.provenance.json").exists()
    # small grid REFUSED by default (real grid enforced) ...
    with pytest.raises(PackSchemaError, match="REAL grid"):
        load_pack(p)
    # ... allowed via the explicit override
    pk = load_pack(p, allow_nonstandard_grid=True)
    assert np.array_equal(pk.counts, small_pack.counts)
    assert np.array_equal(pk.fp_counts, small_pack.fp_counts)
    assert pk.provenance is not None and pk.provenance.get("synthetic") in (True, "True")


def test_full_grid_pack_conforms(tmp_path):
    pk = synthetic_pack(1)  # default = the REAL grid
    assert pk.counts.shape == (29, 15, 8)
    assert pk.dX.shape == (15, 8)
    assert pk.fp_counts.shape == (29, 8)
    assert pk.fp_E_alloc.shape == (15, 8)
    assert pk.t_sigma.shape == (3,)
    p = tmp_path / "modelA_pack_full.npz"
    save_pack(pk, p)
    pk2 = load_pack(p)  # real grid loads WITHOUT the override
    assert np.array_equal(pk2.counts, pk.counts)


@pytest.mark.parametrize("mutation, match", [
    (dict(counts=lambda pk: pk.counts[:, :, :-1]), "counts.*shape"),
    (dict(counts=lambda pk: pk.counts.astype(np.float64)), "int64"),
    (dict(counts=lambda pk: np.abs(pk.counts) * -1 - 1), "non-negative"),
    (dict(dX=lambda pk: pk.dX * 0.0), "zero-pathlength"),  # dX==0 legal only with zero counts
    (dict(dX=lambda pk: pk.dX * -1.0), "non-negative"),
    (dict(dX=lambda pk: np.where(np.arange(pk.dX.size).reshape(pk.dX.shape) == 0,
                                 np.nan, pk.dX)), "finite"),
    (dict(ntrue_edges=lambda pk: pk.ntrue_edges + 0.05), "ntrue_edges"),
    (dict(zc_edges=lambda pk: pk.zc_edges + 0.05), "zc_edges"),
    (dict(kz_to_K=lambda pk: pk.kz_to_K[::-1].copy()), "kz_to_K"),
    (dict(fp_E_alloc=lambda pk: pk.fp_E_alloc * 2.0), "sum_k"),
    (dict(molly_n_det=lambda pk: pk.molly_n_det + pk.molly_n_tot), "n_det > n_tot"),
    (dict(t_sigma=lambda pk: pk.t_sigma * 0.0), "positive"),
    (dict(snr_edges=lambda pk: np.linspace(0, 7, len(pk.snr_edges))), "inf"),
    (dict(resp_sig_floor=lambda pk: -1.0), "resp_sig_floor"),
])
def test_validation_refuses_each_break(small_pack, mutation, match):
    (field, fn), = mutation.items()
    broken = dataclasses.replace(small_pack, **{field: fn(small_pack)})
    with pytest.raises(PackSchemaError, match=match):
        packmod.validate_pack(broken, allow_nonstandard_grid=True)


def test_loader_refuses_missing_key(small_pack, tmp_path):
    p = tmp_path / "modelA_pack_missing.npz"
    save_pack(small_pack, p, allow_nonstandard_grid=True)
    with np.load(p) as z:
        data = {k: z[k] for k in z.files if k != "t_sigma"}
    p2 = tmp_path / "modelA_pack_missing2.npz"
    np.savez(p2, **data)
    with pytest.raises(PackSchemaError, match="missing required schema keys.*t_sigma"):
        load_pack(p2, allow_nonstandard_grid=True)


def test_loader_refuses_unknown_key(small_pack, tmp_path):
    p = tmp_path / "modelA_pack_extra.npz"
    save_pack(small_pack, p, allow_nonstandard_grid=True)
    with np.load(p) as z:
        data = {k: z[k] for k in z.files}
    data["kappa_matrix"] = np.eye(3)  # e.g. a smuggled kappa object
    p2 = tmp_path / "modelA_pack_extra2.npz"
    np.savez(p2, **data)
    with pytest.raises(PackSchemaError, match="unknown keys.*kappa_matrix"):
        load_pack(p2, allow_nonstandard_grid=True)


# --- skew-normal CDF accuracy -------------------------------------------------------

def test_skewnorm_cdf_vs_scipy_bin_masses():
    """GL-quadrature Owen's T path vs scipy exact: bin masses to <= 1e-4 (brief),
    measured ~1e-13. Covers zero skew and |skew| <= 1.4."""
    from scipy.stats import skewnorm
    edges = np.arange(19.5, 22.4 + 1e-9, 0.1)
    for alpha in (0.0, 0.7, -0.7, 1.4, -1.4):
        for xi, om in ((20.0, 0.15), (21.3, 0.35), (19.6, 0.08)):
            F_j = np.asarray(fwd.skewnorm_cdf_jnp(jnp.asarray(edges), xi, om, alpha))
            m_j = np.diff(F_j)
            m_s = np.diff(skewnorm.cdf(edges, alpha, loc=xi, scale=om))
            assert np.max(np.abs(m_j - m_s)) < 1e-4, (alpha, xi, om)
            assert np.max(np.abs(m_j - m_s)) < 1e-10  # actual accuracy


def test_owens_t_symmetries():
    h = np.array([0.0, 0.5, 2.0, -1.3])
    a = np.array([0.3, 1.0, 1.4, 0.8])
    T = np.asarray(fwd.owens_t_jnp(h, a))
    T_neg_a = np.asarray(fwd.owens_t_jnp(h, -a))
    T_neg_h = np.asarray(fwd.owens_t_jnp(-h, a))
    assert np.allclose(T_neg_a, -T, atol=1e-14)   # odd in a
    assert np.allclose(T_neg_h, T, atol=1e-14)    # even in h
    # T(0, a) = arctan(a) / 2pi
    T0 = np.asarray(fwd.owens_t_jnp(0.0, a))
    assert np.allclose(T0, np.arctan(a) / (2 * np.pi), atol=1e-13)


# --- fold vs independent oracle -------------------------------------------------------

def _random_params(pk, rng, scale=1.0):
    tr = pk.truth
    th = tr["theta_true"] + 0.3 * scale * rng.standard_normal(tr["theta_true"].shape)
    pc = 0.2 * scale * rng.standard_normal((pk.n_s, pk.n_molly))
    pkd = 0.05 * scale * rng.standard_normal((2,) + pk.resp_mu_coef.shape[:2])
    lt = 0.3 * scale * rng.standard_normal(pk.n_kk)
    lam = np.abs(rng.standard_normal((pk.n_c, pk.n_s))) * 3.0
    return th, pc, pkd, lt, lam


def _assert_fold_agrees(pk, params):
    consts = fwd.build_consts(pk)
    th, pc, pkd, lt, lam = params
    mu_j = np.asarray(fwd.fold_mu(jnp.asarray(th), jnp.asarray(pc),
                                  jnp.asarray(pkd), jnp.asarray(lt),
                                  jnp.asarray(lam), consts))
    mu_r = fwd.fold_mu_reference(th, pc, pkd, lt, lam, pk)
    assert mu_j.shape == (pk.n_c, pk.n_k, pk.n_s)
    assert np.all(np.isfinite(mu_j)) and np.all(mu_r >= 0.0)
    # rtol 1e-10 with a tiny scale-relative atol for far-tail cells
    np.testing.assert_allclose(mu_j, mu_r, rtol=1e-10, atol=1e-12 * mu_r.max())


def test_fold_vs_reference_small_grid(small_pack):
    rng = np.random.default_rng(7)
    for i in range(5):  # multiple random parameter points
        _assert_fold_agrees(small_pack, _random_params(small_pack, rng))


def test_fold_vs_reference_zero_fp_and_truth_point(small_pack):
    tr = small_pack.truth
    zeros_lam = np.zeros((small_pack.n_c, small_pack.n_s))
    _assert_fold_agrees(small_pack, (tr["theta_true"], tr["psi_c_true"],
                                     np.zeros((2,) + small_pack.resp_mu_coef.shape[:2]),
                                     tr["t_true"], zeros_lam))
    # and: the generator's mu_true is reproduced by the jnp fold at the truth
    consts = fwd.build_consts(small_pack)
    mu_j = np.asarray(fwd.fold_mu(
        jnp.asarray(tr["theta_true"]), jnp.asarray(tr["psi_c_true"]),
        jnp.zeros((2,) + small_pack.resp_mu_coef.shape[:2]),
        jnp.asarray(tr["t_true"]), jnp.asarray(tr["lam_fp_true"]), consts))
    np.testing.assert_allclose(mu_j, tr["mu_true"], rtol=1e-10,
                               atol=1e-12 * tr["mu_true"].max())


def test_fold_vs_reference_full_grid():
    pk = synthetic_pack(3)  # real 29 x 15 x 8 grid
    rng = np.random.default_rng(11)
    _assert_fold_agrees(pk, _random_params(pk, rng))


def test_fold_jits_and_differentiates(small_pack):
    consts = fwd.build_consts(small_pack)
    tr = small_pack.truth
    pkd0 = jnp.zeros((2,) + small_pack.resp_mu_coef.shape[:2])
    lam0 = jnp.asarray(np.abs(tr["lam_fp_true"]) + 0.1)

    def total_mu(theta):
        return fwd.fold_mu(theta, jnp.asarray(tr["psi_c_true"]), pkd0,
                           jnp.asarray(tr["t_true"]), lam0, consts).sum()

    jitted = jax.jit(total_mu)
    v = float(jitted(jnp.asarray(tr["theta_true"])))
    assert np.isfinite(v) and v > 0
    g = np.asarray(jax.grad(total_mu)(jnp.asarray(tr["theta_true"])))
    assert g.shape == tr["theta_true"].shape and np.all(np.isfinite(g))
    assert np.all(g >= 0)  # d(total mu)/d(log f) >= 0 everywhere


def test_kernel_columns_are_probability_masses(small_pack):
    consts = fwd.build_consts(small_pack)
    K = np.asarray(fwd.build_K(jnp.zeros((2,) + small_pack.resp_mu_coef.shape[:2]),
                               consts))
    assert K.shape == (small_pack.n_s, small_pack.n_kk,
                       small_pack.n_c, small_pack.n_b)
    assert np.all(K >= 0.0) and np.all(K <= 1.0)
    colsum = K.sum(axis=2)  # mass of each true bin landing anywhere observed
    assert np.all(colsum <= 1.0 + 1e-12)
    assert np.all(colsum > 0.05)  # nothing pathologically lost on this grid
