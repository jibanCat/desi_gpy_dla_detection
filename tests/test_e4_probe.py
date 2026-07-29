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


# --- the tilt SHAPE is load-bearing (referee defect 2) -----------------------
#
# The section-3 "mechanism" result (a smooth misfit returns as oscillatory
# ringing) depends ENTIRELY on the tilt being a monotone RAMP.  The previous
# guard only asserted eps == 0 gives zero perturbation, which is true for ANY
# tilt function: replacing the ramp with a flat rescale destroyed the finding
# and the suite still passed 27/27.  These tests pin the SHAPE.

def _ringing_system():
    """A smoothing (near-Gaussian) fold: ill-conditioned like the real one."""
    Nc = np.linspace(0.0, 5.8, 30)
    M = np.exp(-0.5 * ((Nc[:, None] - Nc[None, :]) / 0.3) ** 2)
    f = 100.0 * np.exp(-0.9 * Nc)
    return M, f


def test_tilt_vector_shapes_are_what_they_say():
    ramp = e4run.tilt_vector(11, 0.05, "ramp")
    flat = e4run.tilt_vector(11, 0.05, "flat")
    assert ramp[0] == pytest.approx(0.95) and ramp[-1] == pytest.approx(1.05)
    assert np.all(np.diff(ramp) > 0)                      # strictly monotone
    assert np.allclose(flat, 1.05) and np.ptp(flat) == 0.0  # no shape at all
    # same L2 amplitude is NOT the point; the shape is
    with pytest.raises(ValueError):
        e4run.tilt_vector(11, 0.05, "sawtooth")


def test_systematic_misfit_ramp_rings_but_a_flat_rescale_does_not():
    """THE power test: a monotone ramp must ring, a flat rescale must not.

    NNLS is positively homogeneous, so a flat rescale maps EXACTLY to
    f_hat = (1 + eps) f: gain 1, zero sign changes.  A monotone ramp of the
    same L2 size comes back amplified and sign-changing.  If the ramp were
    ever replaced by a flat rescale, the first three assertions all fail.
    """
    M, f = _ringing_system()
    ramp = e4run.systematic_misfit_test(M, f, eps=0.05)
    flat = e4run.systematic_misfit_test(M, f, eps=0.05, shape="flat")

    assert ramp["tilt_shape"] == "ramp"
    # (1) the ramp AMPLIFIES: the f error is larger than the data error
    assert ramp["gain"] > 1.5, ramp["gain"]
    # (2) the ramp error OSCILLATES
    assert ramp["n_sign_changes_of_error"] >= 4, ramp["n_sign_changes_of_error"]
    # (3) the two perturbations are the same ORDER of size (the ramp's rms is
    # ~eps/sqrt(3) by construction), so what differs between them is the SHAPE
    assert 0.5 < (ramp["data_perturbation_rel_l2"]
                  / flat["data_perturbation_rel_l2"]) < 2.0
    # (4) the flat control does NOT ring: exactly gain 1, zero sign changes
    assert flat["gain"] == pytest.approx(1.0, rel=1e-4), flat["gain"]
    assert flat["n_sign_changes_of_error"] == 0
    assert flat["max_abs_ratio_minus_1"] == pytest.approx(0.05, rel=1e-4)
    # and the two differ by a wide margin, so the test cannot pass by accident
    assert ramp["gain"] > 1.4 * flat["gain"]


def test_systematic_misfit_shape_matters_on_the_production_operator(
        small_pack, small_A):
    """The same contrast, on the operator probed out of the committed fold."""
    consts = build_consts(small_pack)
    f = e4.truth_f(small_pack, consts)
    ks = [k for k in range(small_pack.n_k) if np.any(f[:, k] > 0)]
    if not ks:
        pytest.skip("synthetic pack carries no positive truth_counts")
    k = ks[0]
    M = e4.operator_matrix(small_A, small_pack, k)
    flat = e4run.systematic_misfit_test(M, f[:, k], eps=0.05, shape="flat")
    assert flat["gain"] == pytest.approx(1.0, rel=1e-5)
    assert flat["n_sign_changes_of_error"] == 0
    ramp = e4run.systematic_misfit_test(M, f[:, k], eps=0.05)
    assert ramp["gain"] > flat["gain"]
    assert ramp["n_sign_changes_of_error"] > flat["n_sign_changes_of_error"]


# --- the conditioning gain has TWO sources, not one (referee defect 1) -------

def test_decompose_stack_gain_attributes_a_known_split():
    """Constructed factors where the answer is known by construction.

    Three cases, each with the SAME baseline: (a) only the per-stratum
    response differs, (b) only the per-stratum weights differ, (c) both.  The
    decomposition must credit the gain to the input that actually varies and
    must credit EXACTLY 1.0 to the one that does not.
    """
    B, C, n = 6, 8, 3
    x, y = np.linspace(0, 5, C), np.linspace(0, 5, B)

    def kern(sig):
        return np.exp(-0.5 * ((x[:, None] - y[None, :]) / sig) ** 2)

    # per-stratum responses of DIFFERENT width (the real fold's 3 SNR response
    # cells), reference = the widest/worst-conditioned one
    Ks_var = [kern(s) for s in (1.2, 0.6, 0.3)]
    # per-stratum weights emphasising DIFFERENT parts of the basis (the real
    # fold's completeness step x per-stratum dX)
    ws_var = [np.exp(-0.5 * ((y - c) / 1.5) ** 2) + 0.05 for c in (0.0, 2.5, 5.0)]
    w0, K0 = np.ones(B), Ks_var[0]

    only_K = e4run.decompose_stack_gain(Ks_var, [w0] * n, ref=0)
    only_w = e4run.decompose_stack_gain([K0] * n, ws_var, ref=0)
    both = e4run.decompose_stack_gain(Ks_var, ws_var, ref=0)

    # a common input contributes EXACTLY nothing
    assert only_K["gain_from_w"] == pytest.approx(1.0, rel=1e-9)
    assert only_w["gain_from_K"] == pytest.approx(1.0, rel=1e-9)
    # the varying input carries the whole gain in its own case
    assert only_K["gain_from_K"] == pytest.approx(only_K["gain_total"], rel=1e-9)
    assert only_w["gain_from_w"] == pytest.approx(only_w["gain_total"], rel=1e-9)
    # both varying: BOTH gains are real and > 2 — the referee's point, that two
    # frozen inputs carry the load and neither alone explains the improvement
    assert both["gain_from_K"] > 2.0, both
    assert both["gain_from_w"] > 2.0, both
    # neither alone accounts for the total
    assert both["gain_from_K"] < both["gain_total"] * both["gain_from_w"]
    # stacking identical blocks adds no conditioning: baseline == one block
    assert both["cond_baseline"] == pytest.approx(
        np.linalg.cond(Ks_var[0] * np.mean(np.stack(ws_var), axis=0)[None, :]),
        rel=1e-8)
    assert unused_input_is_ignored(e4run.decompose_stack_gain, Ks_var, ws_var)


def unused_input_is_ignored(fn, Ks, ws):
    """Reference index must change the counterfactual, not be silently dropped."""
    a = fn(Ks, ws, ref=0)
    b = fn(Ks, ws, ref=len(Ks) - 1)
    return a["cond_baseline"] != b["cond_baseline"]


def test_decompose_stack_gain_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        e4run.decompose_stack_gain([np.eye(3)], [np.ones(3), np.ones(3)], ref=0)


def test_conditioning_decomposition_factorisation_holds_on_the_probed_operator(
        small_pack, small_A):
    """A[:, k, s, :] == K_s diag(w_s) — the identity the attribution rests on."""
    from CDDF_analysis.hbi_mcmc.forward import build_K
    consts = build_consts(small_pack)
    K = np.asarray(build_K(jnp.zeros((2, consts.n_sr, consts.n_zr)), consts))
    dX = np.asarray(small_pack.dX, float)
    ks = [k for k in range(small_pack.n_k) if (dX[k] > 0).sum() >= 2]
    if not ks:
        pytest.skip("synthetic pack has no fine-z bin with >= 2 live strata")
    d = e4run.conditioning_decomposition(small_A, small_pack, consts, K, ks[0])
    assert d is not None
    assert d["factorisation_max_rel_error"] < 1e-10
    assert len(d["k_reference_sensitivity"]) == len(d["live_strata"])
    for r in d["k_reference_sensitivity"]:
        assert r["cond_actual"] == pytest.approx(
            np.linalg.cond(e4.operator_matrix(small_A, small_pack, ks[0])),
            rel=1e-8)


def test_conditioning_decomposition_refuses_a_broken_factorisation(
        small_pack, small_A):
    """MUTATION GUARD: corrupt one block and the attribution must REFUSE."""
    from CDDF_analysis.hbi_mcmc.forward import build_K
    consts = build_consts(small_pack)
    K = np.asarray(build_K(jnp.zeros((2, consts.n_sr, consts.n_zr)), consts))
    dX = np.asarray(small_pack.dX, float)
    ks = [k for k in range(small_pack.n_k) if (dX[k] > 0).sum() >= 2]
    if not ks:
        pytest.skip("synthetic pack has no fine-z bin with >= 2 live strata")
    k = ks[0]
    bad = np.array(small_A, copy=True)
    s = int(np.argmax(dX[k] > 0))
    bad[0, k, s, :] *= 1.5                      # break ONE row of ONE block
    with pytest.raises(RuntimeError, match="factorisation"):
        e4run.conditioning_decomposition(bad, small_pack, consts, K, k)


def test_summary_caveat_names_both_frozen_inputs_and_quotes_measured_gains():
    """The stamped caveat must be BUILT from the measured decomposition.

    Referee defect 1: the shipped caveat said the gain was bought ENTIRELY by
    the per-SNR response.  It is not; the completeness/dX weights buy an
    independent factor.  This pins that the caveat (a) names both inputs and
    (b) carries numbers taken from the artifact, not typed in.
    """
    def mock_block(gk, gw, gt):
        return dict(
            response_kernel_spectra=[dict(cond=2.77e10)],
            per_z_stacked=[dict(cond=300.0)],
            grid=dict(n_b=29),
            conditioning_decomposition=dict(
                gain_from_K_median_over_z=gk,
                gain_from_w_median_over_z=gw,
                gain_total_median_over_z=gt,
                per_z=[dict(
                    k=e4run.DETAIL_K,
                    gain_from_K_min=gk, gain_from_w_min=gw,
                    canonical=dict(
                        gain_from_K=gk, gain_from_w=gw, gain_total=gt,
                        cond_baseline=3.45e10, cond_per_stratum_K_only=469.0,
                        cond_per_stratum_w_only=3827.0, cond_actual=344.0),
                    k_reference_sensitivity=[
                        dict(gain_from_K=gk, gain_from_w=gw)])]),
            self_inversion=dict(
                exact_summary=dict(max_over_z=1e-10),
                poisson_summary=dict(
                    dynamic_range_of_median_unpinned_median_over_z=9.0,
                    dynamic_range_of_median_unpinned_max_over_z=200.0,
                    frac_pinned_zero_median=0.25),
                systematic_misfit=dict(
                    n_sign_changes_min_over_z=1, n_sign_changes_max_over_z=9,
                    n_sign_changes_median_over_z=5.0,
                    tilt_rel_l2_min_over_z=0.0267, tilt_rel_l2_max_over_z=0.0288,
                    max_abs_ratio_minus_1_max_over_z=0.55,
                    gain_median_over_z=3.0, gain_max_over_z=19.0,
                    flat_control_gain_max_over_z=1.0,
                    flat_control_sign_changes_max_over_z=0)),
            detail_z_bin=dict(null_directions=[
                dict(n_sign_changes=25, node_spacing_dex=0.11)]),
            basis_width_sweep=[
                dict(cond_median=176.0, amp_max_median_over_z=46.0,
                     representation_rel_error_median=0.0,
                     rms_log_ratio_noise_only_median=0.87, n_basis_bins=29),
                dict(cond_median=6.4, amp_max_median_over_z=4.15,
                     representation_rel_error_median=0.00808,
                     rms_log_ratio_noise_only_median=0.56, n_basis_bins=14),
                dict(cond_median=3.28, amp_max_median_over_z=2.21,
                     representation_rel_error_median=0.0154,
                     rms_log_ratio_noise_only_median=0.33, n_basis_bins=9),
                dict(cond_median=2.14, amp_max_median_over_z=1.65,
                     representation_rel_error_median=0.0261,
                     rms_log_ratio_noise_only_median=0.25, n_basis_bins=7)],
            prior_vs_data_precision=[
                dict(sigma_N=0.5, n_prior_dominated=18, n_active_bins=29),
                dict(sigma_N=0.1, n_prior_dominated=22, n_active_bins=29)],
            rw2_regularisation=dict(
                sigma_N_for_amp_le_2_median=0.16,
                per_z=[{"amp_max_at_sigma_N_0.5": 5.0}]),
        )

    art = dict(mocks=dict(m1=mock_block(7.4e7, 9.0e6, 1.0e8)))
    cav = _summary_caveat(art, ["m1"])
    assert "ENTIRELY" not in cav
    for word in ("completeness", "response", "dX"):
        assert word in cav, word
    assert "7.4e+07" in cav and "9e+06" in cav, cav

    # POWER: a different measured split must change the stamped text
    art2 = dict(mocks=dict(m1=mock_block(1.0e3, 5.0e2, 5.0e5)))
    assert _summary_caveat(art2, ["m1"]) != cav
    assert "7.4e+07" not in _summary_caveat(art2, ["m1"])


def _summary_caveat(art, mocks):
    return e4run._summary(art, mocks)["qualification"]["caveat"]


# --- e4_probe.truth_f vs forward_selftest.truth_f (referee minor) ------------

def test_truth_f_divergence_from_forward_selftest_is_exactly_g_bk(small_pack):
    """The two truth_f are NOT the same quantity: they differ by exactly g_bk.

    ``pack.synthetic_pack`` ships g_grid == 1, where the two agree identically
    and no test could see the divergence.  This overrides g_grid to a non-unit
    z-shape ON PURPOSE, so the identity has something to bite on.
    """
    import dataclasses
    from CDDF_analysis.hbi_mcmc import forward_selftest as fst

    g = np.asarray(small_pack.g_grid, float)
    assert np.allclose(g, 1.0), "fixture assumption: synthetic g_grid is unity"
    # with g == 1 the two are identical — i.e. the naive test has NO power
    assert np.allclose(e4.truth_f(small_pack), fst.truth_f(small_pack))

    shape = 0.5 + np.linspace(0.0, 1.5, g.shape[1])[None, :] * np.ones_like(g)
    pk = dataclasses.replace(small_pack, g_grid=g * shape)
    consts = build_consts(pk)
    a = e4.truth_f(pk, consts)
    b = fst.truth_f(pk)
    g_bk = np.asarray(consts.g_bk)
    assert np.allclose(a * g_bk, b, rtol=1e-12, atol=0.0)
    # ... and they genuinely DIFFER, so the identity above is not vacuous
    occ = b > 0
    assert occ.any(), "fixture carries no positive truth"
    assert not np.allclose(a[occ], b[occ], rtol=1e-6)


def test_provenance_block_stamps_a_full_40_char_sha():
    md = e4run.provenance_block(["--out", "x.json"])
    assert len(md["code_commit"]) == 40
    assert int(md["code_commit"], 16) >= 0
    assert md["paper_facing"] is False
