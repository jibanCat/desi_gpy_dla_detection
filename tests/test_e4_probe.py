"""test_e4_probe.py — E4 conditioning probe: the operator recovery must be EXACT.

The whole E4 diagnostic rests on one claim: the matrix ``A`` recovered by
probing the committed fold with one-hot basis vectors IS the fold, so every
singular value reported is a property of production and not of a reimplemented
kernel.  These tests pin that claim, plus the algebraic identities the
basis-width sweep relies on.

Run: conda run -n gpdla-hbi python -m pytest tests/test_e4_probe.py -v
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from CDDF_analysis.hbi_mcmc import pack as packmod  # noqa: E402 (enables x64)
from CDDF_analysis.hbi_mcmc import e4_probe as e4  # noqa: E402
from CDDF_analysis.hbi_mcmc.forward import build_consts, fold_mu  # noqa: E402
from CDDF_analysis.hbi_mcmc.pack import (  # noqa: E402
    small_test_grid, synthetic_pack)

SEED = 0


@pytest.fixture(scope="module")
def small_pack():
    return synthetic_pack(SEED, **small_test_grid())


@pytest.fixture(scope="module")
def small_A(small_pack):
    return e4.build_fold_operator(small_pack)


# --- operator recovery -------------------------------------------------------

def test_off_probe_folds_to_exact_zero(small_pack):
    """exp(_THETA_OFF) must underflow to EXACTLY 0.0 — no one-hot leakage."""
    assert np.exp(e4._THETA_OFF) == 0.0
    consts = build_consts(small_pack)
    mu = np.asarray(fold_mu(
        jnp.full((consts.n_b, consts.n_k), e4._THETA_OFF),
        jnp.zeros((consts.n_s, consts.n_molly)),
        jnp.zeros((2, consts.n_sr, consts.n_zr)),
        jnp.zeros(consts.n_kk), jnp.zeros((consts.n_c, consts.n_s)), consts))
    assert np.all(mu == 0.0)


def test_operator_shape(small_pack, small_A):
    assert small_A.shape == (small_pack.n_c, small_pack.n_k,
                             small_pack.n_s, small_pack.n_b)
    assert np.all(np.isfinite(small_A))
    assert np.all(small_A >= 0.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_operator_reproduces_the_committed_fold(small_pack, small_A, seed):
    """A @ f == fold_mu(log f) to float64 round-off, at random f."""
    err = e4.check_linearity(small_pack, small_A, seed=seed)
    assert err < 1e-12, f"probe/fold relative mismatch {err:.3e}"


def test_operator_is_the_fold_at_the_pack_truth(small_pack, small_A):
    """The same identity at the pack's OWN truth f (not a random point)."""
    consts = build_consts(small_pack)
    f = e4.truth_f(small_pack, consts)
    if not np.any(f > 0):
        pytest.skip("synthetic pack carries no positive truth_counts")
    mu_probe = np.einsum("cksb,bk->cks", small_A, f)
    theta = np.where(f > 0, np.log(np.where(f > 0, f, 1.0)), e4._THETA_OFF)
    mu_fold = np.asarray(fold_mu(
        jnp.asarray(theta), jnp.zeros((consts.n_s, consts.n_molly)),
        jnp.zeros((2, consts.n_sr, consts.n_zr)), jnp.zeros(consts.n_kk),
        jnp.zeros((consts.n_c, consts.n_s)), consts))
    m = mu_fold > 0
    assert np.max(np.abs(mu_probe[m] - mu_fold[m]) / mu_fold[m]) < 1e-12


def test_fold_is_block_diagonal_in_fine_z(small_pack):
    """Perturbing f at ONE fine-z bin must not move mu at any other."""
    consts = build_consts(small_pack)
    B, Kf = consts.n_b, consts.n_k
    base = np.full((B, Kf), -20.0)
    args = (jnp.zeros((consts.n_s, consts.n_molly)),
            jnp.zeros((2, consts.n_sr, consts.n_zr)),
            jnp.zeros(consts.n_kk), jnp.zeros((consts.n_c, consts.n_s)), consts)
    mu0 = np.asarray(fold_mu(jnp.asarray(base), *args))
    bumped = base.copy()
    bumped[:, 0] += 1.0
    mu1 = np.asarray(fold_mu(jnp.asarray(bumped), *args))
    other = np.delete(np.arange(Kf), 0)
    assert np.array_equal(mu0[:, other, :], mu1[:, other, :])


# --- design matrix / live rows ----------------------------------------------

def test_operator_matrix_rows_are_live_strata(small_pack, small_A):
    k = 0
    M = e4.operator_matrix(small_A, small_pack, k)
    n_live = int((np.asarray(small_pack.dX, float)[k] > 0).sum())
    assert M.shape == (small_pack.n_c * n_live, small_pack.n_b)


def test_dead_strata_columns_are_zero(small_pack, small_A):
    dX = np.asarray(small_pack.dX, float)
    dead = np.argwhere(dX == 0.0)
    for k, s in dead[:5]:
        assert np.all(small_A[:, k, s, :] == 0.0)


# --- basis merging algebra ---------------------------------------------------

@pytest.mark.parametrize("g", [1, 2, 3, 4])
def test_basis_groups_partition(g):
    B = 29
    groups = e4.basis_groups(B, g)
    flat = [i for gr in groups for i in gr]
    assert flat == list(range(B))
    assert all(len(gr) == g for gr in groups[:-1])
    assert len(groups[-1]) >= g


def test_merge_is_exact_for_a_locally_constant_f(small_pack, small_A):
    """Column-summing == assuming f constant in the group: check exactly."""
    k = 0
    M = e4.operator_matrix(small_A, small_pack, k)
    groups = e4.basis_groups(M.shape[1], 2)
    Mg = e4.merge_basis_columns(M, groups)
    rng = np.random.default_rng(0)
    fg = rng.uniform(0.5, 2.0, size=len(groups))
    f_fine = np.empty(M.shape[1])
    for j, gr in enumerate(groups):
        f_fine[list(gr)] = fg[j]
    assert np.allclose(Mg @ fg, M @ f_fine, rtol=0, atol=1e-12 * np.abs(M @ f_fine).max())


def test_merged_truth_conserves_the_dN_weighted_integral(small_pack):
    consts = build_consts(small_pack)
    dN = np.asarray(consts.dN_b)
    f = e4.truth_f(small_pack, consts)[:, 0]
    groups = e4.basis_groups(len(f), 3)
    fg = e4.merged_truth(f, dN, groups)
    tot_fine = float(np.sum(f * dN))
    tot_coarse = sum(float(fg[j] * np.sum(dN[list(gr)])) for j, gr in enumerate(groups))
    assert tot_coarse == pytest.approx(tot_fine, rel=1e-12)


# --- spectra / inversion helpers --------------------------------------------

def test_spectrum_on_a_known_matrix():
    M = np.diag([4.0, 2.0, 1e-9])
    sp = e4.spectrum(M)
    assert sp.cond == pytest.approx(4.0 / 1e-9, rel=1e-9)
    assert sp.rank_thresholds["1e-06"] == 2
    assert sp.rank_thresholds["1e-12"] == 3


def test_d2_matrix_annihilates_a_straight_line():
    D = e4.d2_matrix(6)
    x = 3.0 - 0.7 * np.arange(6)
    assert np.allclose(D @ x, 0.0, atol=1e-12)
    assert D.shape == (4, 6)


def test_nnls_recovers_a_well_conditioned_system():
    rng = np.random.default_rng(0)
    M = np.abs(rng.normal(size=(20, 5))) + 0.5
    x = np.array([1.0, 2.0, 0.0, 4.0, 0.5])
    xh = e4.nnls_invert(M, M @ x)
    assert np.allclose(xh, x, atol=1e-8)


def test_map_rw2_recovers_a_smooth_truth_on_a_well_posed_system():
    rng = np.random.default_rng(1)
    M = np.abs(rng.normal(size=(40, 6))) + 1.0
    f = np.exp(np.linspace(6.0, 4.0, 6))          # exactly log-linear -> D2 f = 0
    mu = M @ f
    fh, res = e4.map_invert_rw2(M, mu, sigma_N=0.5)
    assert np.max(np.abs(fh / f - 1.0)) < 1e-3, (fh / f, res.message)


def test_ratio_profile_masks_empty_truth():
    r = e4.ratio_profile(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    assert r[0] == pytest.approx(1.0)
    assert np.isnan(r[1])


# --- driver-level diagnostics (run_e4_conditioning) --------------------------

from CDDF_analysis.hbi_mcmc import run_e4_conditioning as e4run  # noqa: E402


def test_amplification_is_one_for_an_identity_kernel():
    """No deconvolution == no inflation: amp must be exactly 1 per bin."""
    f = np.array([3.0, 7.0, 11.0])
    M = np.eye(3) * np.array([100.0, 50.0, 25.0])   # diagonal fold
    amp, sd, n_b = e4run.amplification(M, f)
    assert np.allclose(amp, 1.0, rtol=1e-10)
    assert np.allclose(n_b, f * np.diag(M))
    # and the sd is the plain counting error on log f
    assert np.allclose(sd, 1.0 / np.sqrt(n_b), rtol=1e-10)


def test_amplification_grows_when_columns_are_made_degenerate():
    f = np.array([5.0, 5.0])
    sharp = np.array([[100.0, 0.0], [0.0, 100.0]])
    blunt = np.array([[100.0, 99.0], [99.0, 100.0]])
    a_sharp, _, _ = e4run.amplification(sharp, f)
    a_blunt, _, _ = e4run.amplification(blunt, f)
    assert np.max(a_blunt) > 10.0 * np.max(a_sharp)


def test_rw2_precision_scales_as_inverse_variance():
    P1 = e4run.rw2_precision(6, 1.0)
    P2 = e4run.rw2_precision(6, 0.5)
    assert np.allclose(P2, 4.0 * P1)
    # a straight line in theta is in the RW2 null space
    x = 2.0 - 0.3 * np.arange(6)
    assert np.allclose(P1 @ x, 0.0, atol=1e-12)


def test_prior_regularisation_monotonically_reduces_amplification():
    rng = np.random.default_rng(3)
    M = np.abs(rng.normal(size=(30, 8))) + 0.05
    M = M + 0.9 * M[:, [0]]                     # deliberately near-degenerate
    f = np.exp(np.linspace(2.0, 0.0, 8))
    prev = np.inf
    for s in (1.0, 0.5, 0.25, 0.1):
        amp, _, _ = e4run.amplification(M, f, prior_prec=e4run.rw2_precision(8, s))
        cur = float(np.nanmax(amp))
        assert cur < prev
        prev = cur


def test_systematic_misfit_test_is_noiseless_and_zero_at_eps_zero():
    rng = np.random.default_rng(4)
    M = np.abs(rng.normal(size=(30, 5))) + 1.0
    f = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    r = e4run.systematic_misfit_test(M, f, eps=0.0)
    assert r["data_perturbation_rel_l2"] == pytest.approx(0.0)
    assert r["max_abs_ratio_minus_1"] < 1e-8


def test_prior_vs_data_precision_counts_directions():
    f = np.array([1.0, 1.0, 1.0, 1.0])
    M = np.eye(4) * 1e6                          # enormous data precision
    d = e4run.prior_vs_data_precision(M, f, sigma_N=0.5)
    assert d["n_prior_dominated"] == 0
    d2 = e4run.prior_vs_data_precision(M * 1e-12, f, sigma_N=0.01)
    assert d2["n_prior_dominated"] == d2["n_active_bins"]


def test_provenance_block_stamps_a_full_40_char_sha():
    md = e4run.provenance_block(["--out", "x.json"])
    assert len(md["code_commit"]) == 40
    assert int(md["code_commit"], 16) >= 0
    assert md["paper_facing"] is False
