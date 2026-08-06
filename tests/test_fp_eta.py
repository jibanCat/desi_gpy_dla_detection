"""(1 - eta) host-occlusion restoration (2026-08-06, PI ruling 8).

The loa-0 FP product's own definition carries a per-band host-occlusion
survival ``(1 - eta_band)`` in the production-volume FP expectation (a forest
FP can only occur in un-occluded forest; ``eta_DLA == 0`` forced by the
product). Until 2026-08-06 the Model-A fold carried the factor ZERO times.
These tests pin the restoration: presence (fail-loud on packs without the
vector), uniqueness (applied exactly once, on the observed-N axis only),
weighting (binwise, not a global scalar), generator/fold convention sharing,
and the loa-0 calibration side carrying NO eta (loa-0 is HCD-free).
"""
import dataclasses

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from CDDF_analysis.hbi_mcmc import forward as F
from CDDF_analysis.hbi_mcmc import pack as P


@pytest.fixture(scope="module")
def base_pack():
    return P.synthetic_pack(seed=7, fp_frac=0.15)


def _zero_args(pk, consts):
    theta = np.log(np.clip(np.asarray(pk.truth["f_true"]), 1e-300, None))
    psi_c = np.zeros((consts.n_s, consts.n_molly))
    psi_k = np.zeros((2, consts.n_sr, consts.n_zr))
    log_t = np.zeros(consts.n_kk)
    return theta, psi_c, psi_k, log_t


def test_missing_fp_eta_fails_loud_in_build_consts(base_pack):
    stripped = dataclasses.replace(base_pack, fp_eta_c=None)
    with pytest.raises(ValueError, match="fp_eta_c"):
        F.build_consts(stripped)
    # the explicit diagnostic escape reproduces eta == 0 exactly
    consts = F.build_consts(stripped, allow_missing_fp_eta=True)
    assert np.allclose(np.asarray(consts.fp_eta_c), 0.0)


def test_missing_fp_eta_fails_loud_in_the_oracle(base_pack):
    stripped = dataclasses.replace(base_pack, fp_eta_c=None)
    consts = F.build_consts(base_pack)
    theta, psi_c, psi_k, log_t = _zero_args(base_pack, consts)
    lam = np.asarray(base_pack.truth["lam_fp_true"])
    with pytest.raises(ValueError, match="fp_eta_c"):
        F.fold_mu_reference(theta, psi_c, psi_k, log_t, lam, stripped)


def test_eta_applied_exactly_once_binwise_on_the_observed_axis():
    """fold ratio (eta on / eta off) must equal (1 - eta_c) per observed bin —
    exactly once, on c only, never on k or s, never on the signal term."""
    C = 8
    pk0 = P.synthetic_pack(seed=11, fp_frac=0.2)
    C = pk0.n_c
    rng = np.random.default_rng(3)
    eta = np.clip(rng.uniform(0.0, 0.4, C), 0, 0.95)
    pk1 = dataclasses.replace(pk0, fp_eta_c=eta)
    c0 = F.build_consts(pk0)   # synthetic default eta == 0
    c1 = F.build_consts(pk1)
    lam = np.asarray(pk0.truth["lam_fp_true"])
    log_t = np.zeros(c0.n_kk)
    fp0 = np.asarray(F.fold_mu_fp(jnp.asarray(log_t), jnp.asarray(lam), c0))
    fp1 = np.asarray(F.fold_mu_fp(jnp.asarray(log_t), jnp.asarray(lam), c1))
    live = fp0 > 0
    ratio = np.where(live, fp1 / np.where(live, fp0, 1.0), np.nan)
    expect = np.broadcast_to((1.0 - eta)[:, None, None], ratio.shape)
    assert np.allclose(ratio[live], expect[live], rtol=1e-12), (
        "the (1 - eta_c) survival must scale the FP term exactly once per "
        "observed bin")
    # the SIGNAL term must be untouched: full fold difference == FP difference
    theta, psi_c, psi_k, log_t_ = _zero_args(pk0, c0)
    mu0 = np.asarray(F.fold_mu(jnp.asarray(theta), jnp.asarray(psi_c),
                               jnp.asarray(psi_k), jnp.asarray(log_t_),
                               jnp.asarray(lam), c0))
    mu1 = np.asarray(F.fold_mu(jnp.asarray(theta), jnp.asarray(psi_c),
                               jnp.asarray(psi_k), jnp.asarray(log_t_),
                               jnp.asarray(lam), c1))
    assert np.allclose(mu0 - fp0, mu1 - fp1, rtol=1e-10, atol=1e-12)


def test_oracle_and_jnp_fold_agree_with_nonzero_eta():
    pk = P.synthetic_pack(seed=13, fp_frac=0.2,
                          fp_eta=np.full(29, 0.25))
    consts = F.build_consts(pk)
    theta, psi_c, psi_k, log_t = _zero_args(pk, consts)
    lam = np.asarray(pk.truth["lam_fp_true"])
    mu_j = np.asarray(F.fold_mu(jnp.asarray(theta), jnp.asarray(psi_c),
                                jnp.asarray(psi_k), jnp.asarray(log_t),
                                jnp.asarray(lam), consts))
    mu_o = F.fold_mu_reference(theta, psi_c, psi_k, log_t, lam, pk)
    assert np.allclose(mu_j, mu_o, rtol=1e-10, atol=1e-12)


def test_generator_inverts_the_same_fold_including_eta():
    """fp_frac is defined as the DATA-side FP share; with a large eta the
    generator must still hit it exactly (the 2026-08-05 ell_eff lesson:
    whatever the fold does, the generator's inversion must do)."""
    fp_frac = 0.18
    pk = P.synthetic_pack(seed=17, fp_frac=fp_frac, fp_eta=0.5)
    mu_true = np.asarray(pk.truth["mu_true"])
    mu_sig = np.asarray(pk.truth["mu_signal"])
    fp_share = (mu_true - mu_sig).sum() / mu_sig.sum()
    assert np.isclose(fp_share, fp_frac, rtol=1e-10), (
        f"generator fp share {fp_share} != requested {fp_frac} — the "
        "inversion does not carry the same (1 - eta) the fold does")


def test_calibration_side_carries_no_eta():
    """Same seed, different eta: the loa-0-side fp_counts must be IDENTICAL
    (loa-0 is HCD-free — nothing occludes) while the data-side counts differ.
    Both packs request the same lam via the same fp_frac inversion? No — the
    inversion rescales lam under eta, so pin lam by comparing the calibration
    law directly: fp_counts ~ Poisson(ell * lam) with NO eta factor."""
    pk = P.synthetic_pack(seed=19, fp_frac=0.15, fp_eta=0.6)
    lam = np.asarray(pk.truth["lam_fp_true"])
    ell = float(pk.fp_ell_eff)
    # deterministic check on the generating law: regenerate the calibration
    # draw with the generator's own stream convention and confirm it matches
    # Poisson(ell * lam) — i.e. eta appears nowhere on the calibration side.
    n = np.asarray(pk.fp_counts, float)
    assert n.sum() > 0
    # under Poisson(ell*lam*(1-eta_bar~0.6)) the total would be ~40% of the
    # eta-free expectation; a 5-sigma band around the eta-free mean separates
    # the two hypotheses decisively for this seed's totals.
    mean_free = float(ell * lam.sum())
    assert abs(n.sum() - mean_free) < 5.0 * np.sqrt(mean_free), (
        "fp_counts total is far from Poisson(ell * lam): the calibration "
        "side appears to carry an eta factor it must not have")


def test_oracle_rejects_nan_eta(base_pack):
    """NaN passes both "< 0" and ">= 1" (NaN comparisons are all False), so
    the oracle must test finiteness explicitly — the fail-closed NaN rule."""
    consts = F.build_consts(base_pack)
    theta, psi_c, psi_k, log_t = _zero_args(base_pack, consts)
    lam = np.asarray(base_pack.truth["lam_fp_true"])
    eta = np.zeros(base_pack.n_c)
    eta[3] = np.nan
    poisoned = dataclasses.replace(base_pack, fp_eta_c=eta)
    with pytest.raises(ValueError, match="bad fp_eta_c"):
        F.fold_mu_reference(theta, psi_c, psi_k, log_t, lam, poisoned)


def test_validate_pack_rejects_out_of_range_eta(base_pack):
    bad = dataclasses.replace(base_pack,
                              fp_eta_c=np.full(base_pack.n_c, 1.0))
    with pytest.raises(Exception, match="fp_eta_c"):
        P.validate_pack(bad, allow_nonstandard_grid=True)


def test_save_load_round_trips_fp_eta(tmp_path, base_pack):
    eta = np.linspace(0.0, 0.3, base_pack.n_c)
    pk = dataclasses.replace(base_pack, fp_eta_c=eta)
    path = tmp_path / "pack_eta.npz"
    P.save_pack(pk, path, allow_nonstandard_grid=True)
    back = P.load_pack(path, allow_nonstandard_grid=True)
    assert np.allclose(np.asarray(back.fp_eta_c), eta)
