"""count_conserving_fold: the count-conservation reference guard (G-CC) and the
renormalised fold — the guard that had no direct test (Paper-1 code review, F15)."""
import types
import numpy as np
import pytest

from CDDF_analysis.hbi_mcmc import count_conserving_fold as CC
from CDDF_analysis.hbi_mcmc.pack import synthetic_pack


def _stamped(pack, phi_ref=None):
    """A v1.2-style stamped view of a synthetic pack: the adopted surfaces are the
    deployed ones, the stored reference is their own in-grid mass (the production
    identity), unless a perturbed phi_ref is supplied."""
    ns = types.SimpleNamespace(**{k: getattr(pack, k) for k in dir(pack) if not k.startswith("_")})
    ns.tp_convention_id = "tp_natpair_tilthost_op/v1"
    ns.contract_id = "ckfp_lown_contract/v1"
    ns.adopted_resp_version = "adopted_response/v1.1"
    ns.adopted_resp_mu_coef = np.asarray(pack.resp_mu_coef)
    ns.adopted_resp_sig_coef = np.asarray(pack.resp_sig_coef)
    ns.adopted_resp_skew_coef = np.asarray(pack.resp_skew_coef)
    ns.adopted_resp_fit_range = np.asarray(pack.resp_N_fit_range, float)
    ns.adopted_phi_ref = CC.phi_from_surfaces(pack) if phi_ref is None else phi_ref
    return ns


@pytest.fixture(scope="module")
def pk():
    return synthetic_pack(seed=3)


def _theta_lam(pack):
    B = len(np.asarray(pack.ntrue_edges)) - 1
    K = len(np.asarray(pack.zf_edges)) - 1
    S = len(np.asarray(pack.snr_edges)) - 1
    theta = np.full((B, K), np.log(1e-21))
    lam = np.zeros((len(np.asarray(pack.nhat_edges)) - 1, S))
    return theta, lam


def test_unstamped_pack_is_refused(pk):
    theta, lam = _theta_lam(pk)
    with pytest.raises(ValueError, match="adopted-contract stamp group"):
        CC.cc_fold_adopted(pk, theta, lam)


def test_stored_reference_identity_passes_and_a_2e9_deviation_is_refused(pk):
    theta, lam = _theta_lam(pk)
    good = _stamped(pk)
    mu, parts = CC.cc_fold_adopted(good, theta, lam)          # tolerance 1e-9 (default)
    assert np.all(np.isfinite(mu)) and mu.shape[0] == len(np.asarray(pk.nhat_edges)) - 1
    bad = _stamped(pk, phi_ref=CC.phi_from_surfaces(pk) + 2e-9)
    with pytest.raises(ValueError, match="count-conservation reference is corrupt"):
        CC.cc_fold_adopted(bad, theta, lam)
    ok = _stamped(pk, phi_ref=CC.phi_from_surfaces(pk) + 5e-10)
    CC.cc_fold_adopted(ok, theta, lam)                          # inside the tolerance


def test_renormalised_kernel_conserves_the_reference_in_grid_mass(pk):
    """K~ = K/phi * phi_ref: the in-grid mass of every (sr, zr, b) column equals phi_ref."""
    ne = np.asarray(pk.nhat_edges, float)
    masses, phi = CC.surface_masses(pk, pk.resp_mu_coef, pk.resp_sig_coef, pk.resp_skew_coef,
                                    np.asarray(pk.resp_N_fit_range, float), ne)
    phi_ref = 0.5 * phi                                         # any stored reference
    ktilde = masses / np.maximum(phi, 1e-12)[:, :, None, :] * phi_ref[:, :, None, :]
    np.testing.assert_allclose(ktilde.sum(axis=2), phi_ref, rtol=1e-12, atol=0.0)
    # and with the deployed reference the renormalised fold IS the deployed fold
    theta, lam = _theta_lam(pk)
    mu_dep, _ = CC.cc_fold_cmarginal(pk, theta, lam)
    mu_cc, _ = CC.cc_fold_cmarginal(pk, theta, lam, renormalize=True, phi_ref=phi)
    np.testing.assert_allclose(mu_cc, mu_dep, rtol=1e-10, atol=0.0)
