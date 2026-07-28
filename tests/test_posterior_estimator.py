# -*- coding: utf-8 -*-
"""Tests for the paper-facing posterior estimator (model_a reductions +
run_posterior gate/stamp).

Three families the 2026-07-28 contract names explicitly:
  * PRIOR-PREDICTIVE SANITY      -- the prior is proper, finite, and weak.
  * THE REDUCTION f(N,z) -> dN/dX, Omega -- hand-computed, not self-referential.
  * OMISSION SENSITIVITY        -- deleting a term MUST change the answer.
    (A reduction that is insensitive to a term it claims to apply is not
    applying it; several of these fail on plausible one-line "simplifications".)

Everything here is synthetic or hand-built. No survey data of any kind.
"""
from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
numpyro = pytest.importorskip("numpyro")
import jax  # noqa: E402

from CDDF_analysis.hbi_mcmc import model_a as MA  # noqa: E402
from CDDF_analysis.hbi_mcmc import run_posterior as RP  # noqa: E402
from CDDF_analysis.hbi_mcmc.pack import (  # noqa: E402
    synthetic_pack, small_test_grid)
from CDDF_analysis.hbi_mcmc.forward import build_consts, fold_mu  # noqa: E402


# --------------------------------------------------------------------------
# a minimal hand-built "pack" for the pure-reduction tests: every grid quantity
# is chosen so dN/dX and Omega can be computed on paper.
# --------------------------------------------------------------------------
def _toy_pack(*, pad=0):
    """4 reported N-bins [19.5,19.9) in 0.1 dex ... extended if pad>0 downward.

    dX[k, s] chosen so the two fine-z bins carry pathlength 3 and 1.
    """
    nhat = np.round(np.arange(19.5, 19.9 + 1e-9, 0.1), 10)     # 4 bins
    ntrue = np.round(np.arange(19.5 - 0.1 * pad, 19.9 + 1e-9, 0.1), 10)
    return types.SimpleNamespace(
        nhat_edges=nhat, ntrue_edges=ntrue,
        zf_edges=np.array([2.0, 2.1, 2.2]),
        zc_edges=np.array([2.0, 2.2]),
        kz_to_K=np.array([0, 0]), n_kk=1,
        dX=np.array([[3.0], [1.0]]),                             # (Kf=2, S=1)
    )


@pytest.fixture(scope="module")
def spack():
    return synthetic_pack(seed=0, **small_test_grid())


# ==========================================================================
# 1. PRIOR-PREDICTIVE SANITY
# ==========================================================================

def _prior_predictive(pack, n=64, seed=1):
    consts = build_consts(pack)
    from functools import partial
    model = partial(MA.model_a, fp_mode="joint")
    pred = numpyro.infer.Predictive(model, num_samples=n)
    return pred(jax.random.PRNGKey(seed), consts, None, None), consts


def test_prior_predictive_is_finite_positive_and_correctly_shaped(spack):
    s, consts = _prior_predictive(spack)
    f = np.asarray(s["f"])
    assert f.shape == (64, spack.n_b, spack.n_k)
    assert np.all(np.isfinite(f)), "prior-predictive f has non-finite draws"
    assert np.all(f > 0), "f = exp(theta) must be strictly positive"
    for site in ("sigma_N", "sigma_z"):
        v = np.asarray(s[site])
        assert np.all(v >= 0) and np.all(np.isfinite(v)), site
    lam = np.asarray(s["lam_fp"])
    assert lam.shape == (64, consts.n_c, consts.n_s)
    assert np.all(lam >= 0) and np.all(np.isfinite(lam))
    # the FP shape is a simplex per draw (softmax of a zero-sum logit)
    tot = np.asarray(s["fp_lam_total"])
    assert np.allclose(lam.reshape(64, -1).sum(axis=1), tot, rtol=1e-5)


def test_prior_predictive_is_weakly_informative_not_a_hidden_answer(spack):
    """The prior must not pin the reported quantity. If the prior-predictive
    dN/dX is concentrated, the 'posterior' band is really a prior band."""
    s, _ = _prior_predictive(spack, n=256, seed=3)
    red = MA.reduce_f_posterior(np.asarray(s["f"]), spack)
    d = np.asarray(red["dndx_dla_20p3_allz"])
    d = d[np.isfinite(d) & (d > 0)]
    assert d.size > 100
    spread_dex = np.log10(np.percentile(d, 97.5) / np.percentile(d, 2.5))
    assert spread_dex > 2.0, (
        f"prior-predictive dN/dX(>=20.3) spans only {spread_dex:.2f} dex -- "
        "the prior is doing the work, not the data")


def test_prior_predictive_smoothness_scale_actually_smooths(spack):
    """sigma_N -> 0 must drive theta toward the linear anchor (no curvature)."""
    consts = build_consts(spack)
    from numpyro.handlers import seed as _seed, substitute as _sub, trace as _trace
    B, Kf = consts.n_b, consts.n_k
    base = dict(sigma_z=jnp.asarray(0.0), theta_level=jnp.asarray(0.0),
                theta_slope=jnp.asarray(1.0),
                eps_N=jnp.ones(max(B - 2, 0)), eps_z=jnp.zeros((B, Kf - 1)))
    out = {}
    for sN in (0.0, 0.5):
        vals = dict(base, sigma_N=jnp.asarray(sN))
        tr = _trace(_sub(_seed(MA.model_a, jax.random.PRNGKey(0)), vals)
                    ).get_trace(consts, None, None)
        out[sN] = np.asarray(tr["theta_pop"]["value"])[:, 0]
    curv0 = np.diff(out[0.0], n=2)
    curv1 = np.diff(out[0.5], n=2)
    assert np.allclose(curv0, 0.0, atol=1e-9), "sigma_N=0 must give zero curvature"
    assert np.abs(curv1).max() > 1e-3, "sigma_N>0 must admit curvature"


# ==========================================================================
# 2. THE REDUCTION f(N, z) -> dN/dX and Omega
# ==========================================================================

def test_reduction_matches_hand_computed_dndx_and_omega():
    """f constant = 1 over 4 bins of width 0.1 dex, centres 19.55..19.85."""
    p = _toy_pack()
    f = np.ones((1, 4, 2))
    red = MA.reduce_f_posterior(f, p)
    Nc = np.array([19.55, 19.65, 19.75, 19.85])

    # sub-DLA window [19.5, 20.3) takes all 4 bins
    want_dndx = 4 * 0.1
    assert red["dndx_subdla_195_203"].shape == (1, 2)
    np.testing.assert_allclose(red["dndx_subdla_195_203"], want_dndx, rtol=1e-12)
    want_om = float((10.0 ** (Nc - 21.0) * 0.1).sum())
    np.testing.assert_allclose(red["omega_subdla_195_203"], want_om, rtol=1e-12)
    # the >=20.3 tier is empty on this grid and must be absent, not zero-filled
    assert "dndx_dla_20p3_allz" not in red
    # pathlength-weighted all-z of a z-flat field is that same value
    np.testing.assert_allclose(red["dndx_subdla_195_203_allz"], want_dndx,
                               rtol=1e-12)


def test_reduction_allz_is_pathlength_weighted_not_a_plain_mean():
    p = _toy_pack()                       # dX per z-bin = 3 and 1
    f = np.zeros((1, 4, 2))
    f[0, :, 0] = 1.0                      # all the signal in the dX=3 bin
    red = MA.reduce_f_posterior(f, p)
    per_k = red["dndx_subdla_195_203"][0]
    np.testing.assert_allclose(per_k, [0.4, 0.0], rtol=1e-12)
    # pathlength-weighted: (0.4*3 + 0*1)/4 = 0.3 ; a plain mean would give 0.2
    np.testing.assert_allclose(red["dndx_subdla_195_203_allz"], 0.3, rtol=1e-12)


def test_reduction_tiers_partition_the_support_exactly(spack):
    """[19.5,20.3) + [20.3,inf) == [19.5,inf), draw by draw. This is what makes
    a per-draw tier ratio meaningful."""
    rng = np.random.default_rng(0)
    f = np.exp(rng.normal(size=(37, spack.n_b, spack.n_k)))
    red = MA.reduce_f_posterior(f, spack)
    for q in ("dndx", "omega"):
        lhs = red[f"{q}_subdla_195_203"] + red[f"{q}_dla_20p3"]
        np.testing.assert_allclose(lhs, red[f"{q}_all_195_up"], rtol=1e-12)


def test_reduction_excludes_the_basis_pad_from_every_reported_number():
    """A padded pack: arbitrarily large f in the UNREPORTED sub-floor bins must
    not move any reported reduction. (Guards the schema-v1.1 pad convention.)"""
    p0, p1 = _toy_pack(pad=0), _toy_pack(pad=3)
    f0 = np.ones((1, 4, 2))
    f1 = np.ones((1, 7, 2))
    f1[0, :3, :] = 1e6                      # absurd values in the pad
    r0, r1 = MA.reduce_f_posterior(f0, p0), MA.reduce_f_posterior(f1, p1)
    assert r1["n_pad_bins"] == 3 and r0["n_pad_bins"] == 0
    for key in ("integrated_total", "dndx_subdla_195_203_allz",
                "omega_subdla_195_203_allz", "dndx_all_195_up_allz"):
        np.testing.assert_allclose(r1[key], r0[key], rtol=1e-12,
                                   err_msg=f"{key} leaked the basis pad")
    # and the differential CDDF must NOT report the pad bins
    assert np.all(np.isnan(r1["cddf_masked"][:, :3, :]))
    assert bool(r1["reported_mask"][:3].sum() == 0)


def test_reduction_excludes_pad_bins_that_fall_INSIDE_a_tier_window():
    """The hard case the previous test cannot reach.

    When the pad sits entirely BELOW every tier's lower edge, the window test
    alone already excludes it and the `& reported` mask is a no-op -- so that
    configuration cannot detect its removal (verified by mutation).  Here the
    reporting floor is 19.7 while the true-N basis is padded down to 19.5, so
    the pad bins land INSIDE the sub-DLA window [19.5, 20.3) and only the
    `reported` mask keeps them out of a reported number.
    """
    p = types.SimpleNamespace(
        nhat_edges=np.round(np.arange(19.7, 20.0 + 1e-9, 0.1), 10),   # floor 19.7
        ntrue_edges=np.round(np.arange(19.5, 20.0 + 1e-9, 0.1), 10),  # pad to 19.5
        zf_edges=np.array([2.0, 2.1]), zc_edges=np.array([2.0, 2.1]),
        kz_to_K=np.array([0]), n_kk=1, dX=np.array([[1.0]]))
    f = np.ones((1, 5, 1))
    f[0, :2, 0] = 1e6                    # the two pad bins, 19.5 and 19.6
    red = MA.reduce_f_posterior(f, p)
    assert red["n_pad_bins"] == 2
    assert red["n_bins_subdla_195_203"] == 3, (
        "the sub-DLA window must contain the 3 REPORTED bins only")
    # 3 reported bins x width 0.1 x f=1
    np.testing.assert_allclose(red["dndx_subdla_195_203_allz"], 0.3, rtol=1e-12)
    np.testing.assert_allclose(red["integrated_total"], 0.3, rtol=1e-12)


def test_point_is_literally_the_median_of_the_same_draws(spack):
    """point == q50 of the SAME array the band comes from. No shift exists."""
    rng = np.random.default_rng(7)
    f = np.exp(rng.normal(size=(211, spack.n_b, spack.n_k)))
    red = MA.reduce_f_posterior(f, spack)
    s = MA.posterior_summary(red, spack)
    assert s["estimand"] == "POSTERIOR_MEDIAN_CI"
    for tier, blk in s["tiers"].items():
        d = np.asarray(red[f"dndx_{tier}_allz"])
        assert blk["dndx_allz"]["point_q50"] == float(np.percentile(d, 50))
        assert blk["dndx_allz"]["q16"] == float(np.percentile(d, 16))
        assert blk["dndx_allz"]["q84"] == float(np.percentile(d, 84))
        assert blk["dndx_allz"]["q16"] <= blk["dndx_allz"]["point_q50"] \
            <= blk["dndx_allz"]["q84"]


def test_tier_ratio_is_formed_per_draw_not_from_the_marginals(spack):
    """The per-draw ratio median differs from the ratio of tier medians when the
    tiers are correlated -- which is exactly why it must be formed per draw."""
    rng = np.random.default_rng(11)
    # a common level shared by both tiers => strongly correlated tiers
    lvl = rng.normal(scale=0.6, size=(400, 1, 1))
    f = np.exp(lvl + 0.05 * rng.normal(size=(400, spack.n_b, spack.n_k)))
    red = MA.reduce_f_posterior(f, spack)
    s = MA.posterior_summary(red, spack)
    per_draw = s["subdla_over_dla_dndx_perdraw"]
    ratio_of_medians = (s["tiers"]["subdla_195_203"]["dndx_allz"]["point_q50"]
                        / s["tiers"]["dla_20p3"]["dndx_allz"]["point_q50"])
    # correlated tiers: the per-draw ratio is far TIGHTER than either tier
    rel_ratio = (per_draw["q84"] - per_draw["q16"]) / per_draw["point_q50"]
    tier = s["tiers"]["subdla_195_203"]["dndx_allz"]
    rel_tier = (tier["q84"] - tier["q16"]) / tier["point_q50"]
    assert rel_ratio < 0.2 * rel_tier, (
        "the per-draw ratio is not exploiting the tier correlation "
        f"({rel_ratio:.4f} vs {rel_tier:.4f}) -- are the tiers really one "
        "posterior?")
    assert np.isfinite(ratio_of_medians)


# ==========================================================================
# 3. OMISSION SENSITIVITY -- delete a term, the answer MUST move
# ==========================================================================

def test_omission_bin_width_jacobian(spack):
    """Drop dN from the dN/dX integral -> the answer must change. (dN=0.1 here,
    so omitting it inflates dN/dX by exactly 10x.)"""
    rng = np.random.default_rng(0)
    f = np.exp(rng.normal(size=(5, spack.n_b, spack.n_k)))
    red = MA.reduce_f_posterior(f, spack)
    ntrue = np.asarray(spack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    sel = (Nc >= 19.5 - 1e-9) & (Nc < 20.3 - 1e-9)
    no_jac = f[:, sel, :].sum(axis=1)                 # dN omitted
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(red["dndx_subdla_195_203"], no_jac, rtol=1e-6)
    np.testing.assert_allclose(red["dndx_subdla_195_203"], 0.1 * no_jac,
                               rtol=1e-12)


def test_omission_column_density_weight_in_omega(spack):
    """Drop the 10^(N-21) weight -> Omega collapses onto dN/dX. It must not."""
    rng = np.random.default_rng(1)
    f = np.exp(rng.normal(size=(5, spack.n_b, spack.n_k)))
    red = MA.reduce_f_posterior(f, spack)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(red["omega_subdla_195_203"],
                                   red["dndx_subdla_195_203"], rtol=1e-3)


def test_omission_false_positive_term_changes_the_forward_prediction(spack):
    """lam_fp -> 0 must change mu. If it does not, the FP model is inert and the
    41%-FP-subtraction sub-DLA tier is silently unsubtracted."""
    consts = build_consts(spack)
    theta = jnp.log(jnp.asarray(spack.truth["f_true"]))
    psi_c = jnp.asarray(spack.truth["psi_c_true"])
    zk = jnp.zeros((2, consts.n_sr, consts.n_zr))
    lt = jnp.zeros(consts.n_kk)
    lam = jnp.asarray(spack.truth["lam_fp_true"])
    mu_on = np.asarray(fold_mu(theta, psi_c, zk, lt, lam, consts))
    mu_off = np.asarray(fold_mu(theta, psi_c, zk, lt, jnp.zeros_like(lam), consts))
    assert mu_on.sum() > mu_off.sum() * 1.02, (
        f"dropping the FP term moved the total by only "
        f"{mu_on.sum()/mu_off.sum() - 1:.4%}")


def test_omission_completeness_changes_the_forward_prediction(spack):
    """psi_c -> large positive drives C -> 1 (no selection). mu must rise."""
    consts = build_consts(spack)
    theta = jnp.log(jnp.asarray(spack.truth["f_true"]))
    zk = jnp.zeros((2, consts.n_sr, consts.n_zr))
    lt = jnp.zeros(consts.n_kk)
    lam = jnp.zeros((consts.n_c, consts.n_s))
    mu = np.asarray(fold_mu(jnp.asarray(theta),
                            jnp.asarray(spack.truth["psi_c_true"]),
                            zk, lt, lam, consts))
    mu_nosel = np.asarray(fold_mu(
        jnp.asarray(theta), jnp.full((consts.n_s, consts.n_molly), 25.0),
        zk, lt, lam, consts))
    assert mu_nosel.sum() > 1.1 * mu.sum(), (
        "removing the completeness term did not change mu -- the molly surface "
        "is not entering the fold")


def test_omission_response_kernel_smearing_changes_the_forward_prediction():
    """A pack whose response is DIAGONAL must fold differently from the skewed
    one. If not, the kernel is being applied as identity."""
    kw = small_test_grid()
    a = synthetic_pack(seed=0, response_mode="skewed", **kw)
    b = synthetic_pack(seed=0, response_mode="diagonal", **kw)
    ca, cb = build_consts(a), build_consts(b)
    th = jnp.log(jnp.asarray(a.truth["f_true"]))
    lam_a = jnp.zeros((ca.n_c, ca.n_s))
    args = (jnp.zeros((2, ca.n_sr, ca.n_zr)), jnp.zeros(ca.n_kk))
    mu_a = np.asarray(fold_mu(th, jnp.asarray(a.truth["psi_c_true"]),
                              *args, lam_a, ca))
    mu_b = np.asarray(fold_mu(th, jnp.asarray(a.truth["psi_c_true"]),
                              *args, lam_a, cb))
    # per-nhat-bin shape must differ: the skewed kernel migrates mass upward
    sa = mu_a.sum(axis=(1, 2)) / mu_a.sum()
    sb = mu_b.sum(axis=(1, 2)) / mu_b.sum()
    assert np.abs(sa - sb).max() > 1e-3, (
        "diagonal and skewed response kernels fold identically -- the response "
        "is not being applied")


def test_omission_fp_block_changes_the_model_log_density(spack):
    """fp_mode='off' must change the joint log density at a fixed parameter
    point (the loa-0 likelihood term is then gone)."""
    from functools import partial
    from numpyro.infer.util import log_density
    consts = build_consts(spack)
    B, Kf = consts.n_b, consts.n_k
    vals = dict(sigma_N=jnp.asarray(0.2), sigma_z=jnp.asarray(0.2),
                theta_level=jnp.asarray(-1.0), theta_slope=jnp.asarray(-0.2),
                eps_N=jnp.zeros(max(B - 2, 0)), eps_z=jnp.zeros((B, Kf - 1)),
                psi_c=jnp.zeros((consts.n_s, consts.n_molly)),
                psi_k_delta=jnp.zeros((2, consts.n_sr, consts.n_zr)),
                t=jnp.zeros(consts.n_kk))
    args = (consts, jnp.asarray(spack.counts))
    ld_off, _ = log_density(partial(MA.model_a, fp_mode="off"), args,
                            {"fp_counts": None}, vals)
    vals_on = dict(vals, fp_lam_total=jnp.asarray(10.0),
                   fp_shape_v=jnp.zeros(consts.n_c * consts.n_s))
    ld_on, _ = log_density(partial(MA.model_a, fp_mode="joint"), args,
                           {"fp_counts": jnp.asarray(spack.fp_counts)}, vals_on)
    assert np.isfinite(float(ld_off)) and np.isfinite(float(ld_on))
    assert abs(float(ld_on) - float(ld_off)) > 1.0, (
        "removing the FP block did not change the log density")


# ==========================================================================
# 4. THE FORWARD-MODEL GATE + THE STAMP
# ==========================================================================

def test_forward_gate_passes_on_a_synthetic_pack(spack):
    g = RP.forward_closure_gate(spack)
    assert g["pass"], g
    assert g["chi2_dof"] < RP.GATE["chi2_dof_max"]
    assert 0.9 < g["total_ratio"] < 1.1


def test_forward_gate_fails_when_the_truth_support_is_truncated(spack):
    """Reproduce finding D1 in miniature: delete the bottom two true-N rows of
    the truth histogram and the gate must refuse."""
    tc = np.array(spack.truth_counts, dtype=np.int64)
    tc[:2, :] = 0
    broken = dataclasses.replace(spack, truth_counts=tc)
    g = RP.forward_closure_gate(broken)
    assert not g["pass"], "gate accepted a truncated-truth pack"
    assert g["total_ratio"] < 1.0
    assert any("z_bin" in f or "chi2" in f or "z_total" in f
               for f in g["failures"]), g["failures"]


def test_forward_gate_fails_with_no_truth(spack):
    g = RP.forward_closure_gate(dataclasses.replace(spack, truth_counts=None))
    assert not g["pass"] and "truth_counts" in g["reason"]


def test_stamp_carries_every_required_provenance_field(spack):
    cfg = MA.ModelAConfig(num_warmup=3, num_samples=3, num_chains=2, seed=5)
    md = RP.stamp_metadata(
        code_commit="a" * 40, code_dirty=False, cfg=cfg,
        args={"rederive": "python -m ... --out x"},
        gate_report=RP.forward_closure_gate(spack),
        estimand=MA.ESTIMAND_POSTERIOR, paper_facing=True)
    for k in ("estimand", "resp_kind", "code_commit", "routine", "rederive",
              "n_chains", "n_warmup", "n_samples", "seed",
              "forward_model_closes", "paper_facing"):
        assert k in md, f"stamp is missing {k}"
    assert md["estimand"] == "POSTERIOR_MEDIAN_CI"
    assert md["resp_kind"] == "forward", "kappa must be unreachable here"
    assert len(md["code_commit"]) == 40
    assert md["n_chains"] == 2 and md["n_warmup"] == 3 and md["seed"] == 5
    assert md["band_recenter"] is False and md["marginal_combined"] is False
    json.dumps(md, default=RP._jsonable)      # must be serializable


def test_stamped_commit_is_a_real_40_char_sha():
    sha, _dirty = RP._git_sha_full()
    assert sha == "unknown" or (len(sha) == 40 and
                                all(c in "0123456789abcdef" for c in sha))


def test_real_data_tokens_are_refused():
    with pytest.raises((AssertionError, SystemExit)):
        RP.main(["--pack", "/x/modelA_pack_loa_main_dark_v1.npz", "--out", "/x"])


def test_runner_refuses_to_sample_when_the_forward_model_is_open(tmp_path,
                                                                 monkeypatch):
    """The fail-closed gate must abort BEFORE the sampler is built."""
    import CDDF_analysis.hbi_mcmc.model_a as _MA

    def _boom(*a, **k):
        raise AssertionError("sampler was constructed despite a FAILED gate")

    monkeypatch.setattr(_MA, "run_model_a", _boom)
    monkeypatch.setattr(
        RP, "forward_closure_gate",
        lambda *a, **k: {"pass": False, "failures": ["synthetic failure"],
                         "total_ratio": 0.5, "z_total": 99.0, "z_bin_max": 99.0,
                         "chi2_dof": 999.0, "worst_bins": []})
    with pytest.raises(SystemExit) as e:
        RP.main(["--synthetic-smoke", "--out", str(tmp_path / "o.json")])
    assert "CLOSURE GATE FAILED" in str(e.value)


# ==========================================================================
# 5. THE PLUG-IN MAP IS A DIAGNOSTIC
# ==========================================================================

@pytest.mark.slow
def test_plugin_map_is_labelled_and_carries_no_band(spack):
    cfg = MA.ModelAConfig(enforce_farr_gate=False)
    m = MA.plugin_map_diagnostic(spack, cfg, num_steps=200)
    assert m["estimand"] == "PLUGIN_MAP"
    assert m["band"] is None, "a plug-in MAP must never carry a band"
    for tier, blk in m["tiers"].items():
        assert set(blk) == {"dndx_allz", "omega_allz"}
        assert np.isfinite(blk["dndx_allz"]) and blk["dndx_allz"] > 0
