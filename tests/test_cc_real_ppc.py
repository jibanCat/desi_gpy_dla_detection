"""cc_real_ppc — posterior-predictive check of a model_cc real-data run
(PI ruling 2026-08-26, B3 checkpoint item 2). The fold used for mu MUST be
model_cc's own count-conserving fold, pinned here against the model's
`counts` site rate; the PPC statistics are the committed evidence.ppc_block
ones, reached through an additive `mu_draws` argument."""
import dataclasses
import importlib.util as _ilu
import os
import types

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = _ilu.spec_from_file_location(
    "cc_real_ppc_mod", os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "cc_real_ppc.py"))
PPC = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(PPC)

from CDDF_analysis.hbi_mcmc import evidence as EV                    # noqa: E402
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import model_cc  # noqa: E402
from CDDF_analysis.hbi_mcmc.pack import synthetic_pack               # noqa: E402


def _synthetic_consts(seed=0, B=6, Kf=4, C=5, S=3, M=2, KK=2):
    rng = np.random.default_rng(seed)
    import jax.numpy as jnp
    c = types.SimpleNamespace(
        n_b=B, n_k=Kf, n_c=C, n_s=S,
        sigma_hat=jnp.asarray(rng.uniform(0.2, 0.5, (S, M))),
        eta_hat=jnp.asarray(rng.normal(0.0, 1.0, (S, M))),
        t_sigma=jnp.asarray(rng.uniform(0.1, 0.2, KK)),
        b_to_cell=np.asarray(rng.integers(0, M, B)),
        g_bk=jnp.asarray(rng.uniform(0.5, 1.5, (B, Kf))),
        dN_b=jnp.asarray(np.full(B, 0.2)),
        dX=jnp.asarray(rng.uniform(100.0, 300.0, (Kf, S))),
        fp_w=3.0, fp_ell_eff=7.0,
        fp_eta_c=jnp.asarray(rng.uniform(0.0, 0.3, C)),
        kz_to_K=np.asarray(rng.integers(0, KK, Kf)),
        fp_E=jnp.asarray(rng.uniform(0.5, 1.5, (Kf, S))),
    )
    Mg = jnp.asarray(rng.uniform(0.0, 1.0, (S, Kf, C, B)) * 1e-3)
    return c, Mg


def _params(c, rng):
    B, Kf, C, S, KK = c.n_b, c.n_k, c.n_c, c.n_s, int(np.asarray(c.t_sigma).size)
    M = int(np.asarray(c.sigma_hat).shape[1])
    return dict(sigma_N=0.3, sigma_z=0.2, theta_level=-49.5, theta_slope=-0.3,
                eps_N=rng.normal(size=max(B - 2, 0)),
                eps_z=rng.normal(size=(B, max(Kf - 1, 0))),
                psi_c=rng.normal(size=(S, M)) * 0.3,
                fp_lam_total=2.0, fp_shape_v=rng.normal(size=C * S),
                t=rng.normal(size=KK) * 0.1)


def test_cc_fold_mu_is_model_cc_counts_rate():
    """The PPC fold must be THE model's fold, not a re-derivation."""
    import jax.numpy as jnp
    import numpyro
    c, Mg = _synthetic_consts()
    rng = np.random.default_rng(1)
    fpc = rng.poisson(4.0, (c.n_c, c.n_s)).astype(float)
    p = _params(c, rng)
    m = numpyro.handlers.seed(numpyro.handlers.substitute(model_cc, data=p), rng_seed=0)
    tr = numpyro.handlers.trace(m).get_trace(
        c, Mg, counts=jnp.zeros((c.n_c, c.n_k, c.n_s)), fp_counts=jnp.asarray(fpc),
        fp_mode="informative_ln")
    fn = tr["counts"]["fn"]
    rate = np.asarray(getattr(fn, "rate", None) if hasattr(fn, "rate") else fn.base_dist.rate)
    mu = np.asarray(PPC.cc_fold_mu(c, Mg, tr["theta_pop"]["value"], jnp.asarray(p["psi_c"]),
                                   jnp.asarray(p["t"]), tr["lam_fp"]["value"]))
    assert mu.shape == (c.n_c, c.n_k, c.n_s)
    assert np.allclose(mu, rate, rtol=1e-6, atol=0.0)
    assert mu.min() > 0.0                      # a vacuous all-zero rate would pass allclose


def test_mu_draws_cc_subsamples_like_evidence_and_is_deterministic():
    c, Mg = _synthetic_consts()
    rng = np.random.default_rng(2)
    n = 40
    flat = dict(theta_pop=np.linspace(-50, -49, c.n_b)[None, :, None]
                + rng.normal(0, 0.01, (n, c.n_b, c.n_k)),
                psi_c=rng.normal(0, 0.3, (n, c.n_s, 2)),
                t=rng.normal(0, 0.1, (n, 2)),
                lam_fp=rng.uniform(0.1, 0.5, (n, c.n_c, c.n_s)))
    mu1, idx1 = PPC.mu_draws_cc(c, Mg, flat, n_max=10, seed=0)
    mu2, idx2 = PPC.mu_draws_cc(c, Mg, flat, n_max=10, seed=0)
    assert mu1.shape == (10, c.n_c, c.n_k, c.n_s) and np.array_equal(idx1, idx2)
    assert np.array_equal(mu1, mu2)
    # same convention as evidence._mu_draws: sorted choice without replacement from rng(seed)
    exp = np.sort(np.random.default_rng(0).choice(n, size=10, replace=False))
    assert np.array_equal(idx1, exp)
    mu_all, idx_all = PPC.mu_draws_cc(c, Mg, flat, n_max=None, seed=0)
    assert mu_all.shape[0] == n and np.array_equal(mu_all[idx1], mu1)


def _pack_and_mu(seed=0):
    pk = synthetic_pack(seed, nhat_edges=np.round(np.arange(19.9, 20.5 + 1e-9, 0.1), 10),
                        zf_edges=np.round(np.arange(2.0, 2.3 + 1e-9, 0.1), 10),
                        zc_edges=np.array([2.0, 2.3]), snr_edges=np.array([0.0, 3.0, np.inf]),
                        n_molly_cells=2, fp_frac=0.15)
    rng = np.random.default_rng(seed + 100)
    base = np.asarray(pk.counts, float)
    mu = np.clip(base[None] * rng.uniform(0.97, 1.03, (60,) + base.shape), 0.5, None)
    mu = np.where((np.asarray(pk.dX) > 0)[None, None], mu, 0.0)
    return pk, mu


def test_ppc_block_accepts_precomputed_mu_and_needs_no_nuisance_draws():
    pk, mu = _pack_and_mu()
    run = {"samples_by_chain": None, "f_by_chain": None}
    blk = EV.ppc_block(run, pk, None, n_rep_draws=60, seed=0,
                       mu_draws=(mu, np.arange(mu.shape[0])))
    assert blk["incomplete"] == []
    assert blk["n_posterior_draws_used"] == 60
    assert blk["checks"]["ppc_cells_ok"] is True
    assert EV.PPC_OMNIBUS_MIN <= blk["omnibus_chi2_discrepancy"]["posterior_predictive_p"]


def test_ppc_block_with_precomputed_mu_still_detects_unreproducible_counts():
    pk, mu = _pack_and_mu()
    bad = dataclasses.replace(pk, counts=(np.asarray(pk.counts) * 3).astype(np.int64))
    blk = EV.ppc_block({"samples_by_chain": None}, bad, None, n_rep_draws=60, seed=0,
                       mu_draws=(mu, np.arange(mu.shape[0])))
    assert blk["checks"]["ppc_cells_ok"] is False and blk["n_cells_failed"] > 0
    assert blk["checks"]["ppc_omnibus_ok"] is False


def test_ppc_block_default_path_unchanged_when_mu_draws_absent():
    pk, _ = _pack_and_mu()
    blk = EV.ppc_block({"samples_by_chain": None, "f_by_chain": None}, pk, None)
    assert blk["incomplete"] == ["ppc_needs_latent_posterior_draws"]


def test_report_grain_ppc_is_calibrated_for_a_correct_model_and_flags_a_tilt():
    pk, mu = _pack_and_mu(seed=3)
    rng = np.random.default_rng(7)
    obs = rng.poisson(mu.mean(axis=0)).astype(float)
    nhat = np.asarray(pk.nhat_edges, float)
    edges = np.round(np.arange(nhat[0], nhat[-1] + 1e-9, 2 * (nhat[1] - nhat[0])), 3)
    zf = np.asarray(pk.zf_edges, float)
    ok = PPC.report_grain_ppc(mu, obs, np.asarray(pk.dX), nhat, zf, report_edges=edges,
                              z_blocks=((zf[0], zf[-1]),), seed=0)
    allz = ok["blocks"][0]
    assert len(allz["bins"]) == len(edges) - 1
    assert sum(b["obs"] for b in allz["bins"]) == pytest.approx(obs.sum())
    assert 0.02 <= allz["T_over_n"]["posterior_predictive_p"] <= 0.98
    assert allz["n_bins_failed"] == 0
    # a 25 % low->high tilt across the reporting bins is a shape failure
    tilt = np.linspace(0.75, 1.25, mu.shape[1])[None, :, None, None]
    bad = PPC.report_grain_ppc(mu * tilt, obs, np.asarray(pk.dX), nhat, zf,
                               report_edges=edges, z_blocks=((zf[0], zf[-1]),), seed=0)
    assert bad["blocks"][0]["T_over_n"]["posterior_predictive_p"] < 0.01
    assert bad["blocks"][0]["T_over_n"]["T_obs_over_n"] > allz["T_over_n"]["T_obs_over_n"]


def test_nuisance_payload_keeps_chain_structure_and_only_the_fold_sites():
    rng = np.random.default_rng(0)
    sam_g = dict(theta_pop=rng.normal(size=(2, 5, 6, 3)), psi_c=rng.normal(size=(2, 5, 2, 2)),
                 t=rng.normal(size=(2, 5, 2)), lam_fp=rng.uniform(size=(2, 5, 6, 2)),
                 f=rng.uniform(size=(2, 5, 6, 3)), sigma_N=rng.uniform(size=(2, 5)),
                 eps_z=rng.normal(size=(2, 5, 6, 2)))
    pay = PPC.nuisance_payload(sam_g)
    assert set(pay) == {"theta_pop", "psi_c", "t", "lam_fp", "f"}
    assert pay["theta_pop"].shape == (2, 5, 6, 3)
    flat = PPC.flatten_by_chain(pay)
    assert flat["t"].shape == (10, 2) and np.array_equal(flat["t"][5], sam_g["t"][1, 0])
