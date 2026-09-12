"""tests/test_fp_ladder.py — the ladder model must be model_cc except for the FP
block. Runs against the 2LPT-0 pack of record (skipped if absent)."""
import os
import numpy as np
import pytest

PACK = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
        "real_pack_v2_20260821/scanpack_2lpt0_b300.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(PACK), reason="pack of record absent")


@pytest.fixture(scope="module")
def setup():
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
    pk = load_pack(PACK)
    consts, Mg = build_cc_tensors(pk)
    return pk, consts, Mg, jnp.asarray(np.asarray(pk.counts, float)), jnp.asarray(np.asarray(pk.fp_counts, float))


def _trace(model, seed, *args, **kw):
    import jax
    from numpyro.handlers import seed as hseed, trace
    return trace(hseed(model, jax.random.PRNGKey(seed))).get_trace(*args, **kw)


def test_live_mask_and_dead_cells(setup):
    from CDDF_analysis.hbi_mcmc.fp_ladder import live_mask
    pk, consts, *_ = setup
    live = live_mask(consts)
    assert live.tolist() == [False, False, True, True, True, True, True, True]
    assert np.asarray(pk.fp_counts)[:, ~live].sum() == 0
    assert np.asarray(pk.counts)[:, :, ~live].sum() == 0


@pytest.mark.parametrize("ladder,nsites", [("M0", 1), ("M1", 2), ("M2", 4), ("M3", 5), ("M4", 7), ("M5", 9)])
def test_fp_sites_and_zero_off_live(setup, ladder, nsites):
    from CDDF_analysis.hbi_mcmc.fp_ladder import model_cc_ladder, FP_SITES, live_mask
    pk, consts, Mg, counts, fpc = setup
    tr = _trace(model_cc_ladder, 1, consts, Mg, counts=counts, fp_counts=fpc, ladder=ladder)
    sampled = [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]
    fp_sampled = [k for k in sampled if k.startswith("fp_") or k == "t"]
    assert sorted(fp_sampled) == sorted(FP_SITES[ladder]), (fp_sampled, FP_SITES[ladder])
    assert len(fp_sampled) == nsites
    lam = np.asarray(tr["lam_fp"]["value"])
    live = live_mask(consts)
    assert lam.shape == (consts.n_c, consts.n_s)
    assert np.all(lam[:, ~live] == 0.0)
    assert np.all(lam[:, live] > 0.0)
    assert "fp_counts" in tr and tr["fp_counts"]["is_observed"]
    assert "fp_shape_v" not in tr and "fp_lam_total" in tr
    # observed sites only counts + fp_counts
    obs = [k for k, v in tr.items() if v["type"] == "sample" and v.get("is_observed")]
    assert sorted(obs) == ["counts", "fp_counts"]


def test_calibration_term_masked_to_live_cells(setup):
    """log-prob of fp_counts must equal the live-cell Poisson sum exactly."""
    import jax.numpy as jnp
    from scipy.stats import poisson
    from CDDF_analysis.hbi_mcmc.fp_ladder import model_cc_ladder, live_mask
    pk, consts, Mg, counts, fpc = setup
    tr = _trace(model_cc_ladder, 3, consts, Mg, counts=counts, fp_counts=fpc, ladder="M2")
    site = tr["fp_counts"]
    lp = float(site["fn"].log_prob(site["value"]).sum()) if not site.get("mask", None) is not None else None
    lam = np.asarray(tr["lam_fp"]["value"]); live = live_mask(consts)
    mu = float(consts.fp_ell_eff) * lam[:, live]
    expected = poisson.logpmf(np.asarray(pk.fp_counts)[:, live], mu).sum()
    # numpyro applies the mask in the joint density; recompute the masked sum here
    masked = np.asarray(site["fn"].log_prob(site["value"]))[:, live].sum()
    assert np.isclose(masked, expected, rtol=1e-10)
    # numpyro's mask handler wraps the distribution: its log_prob is already zero off-live
    assert type(site["fn"]).__name__ == "MaskedDistribution"
    lp_full = np.asarray(site["fn"].log_prob(site["value"]))
    assert np.all(lp_full[:, ~live] == 0.0)
    assert np.isclose(lp_full.sum(), expected, rtol=1e-10)


def test_fold_identity_with_model_cc(setup):
    """At equal (theta, psi_c, t, lam_fp) the ladder's mu equals model_cc's mu."""
    import jax, jax.numpy as jnp
    from numpyro.handlers import seed as hseed, trace, substitute
    from CDDF_analysis.hbi_mcmc.fp_ladder import model_cc_ladder
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import model_cc
    pk, consts, Mg, counts, fpc = setup
    tr = _trace(model_cc_ladder, 5, consts, Mg, counts=counts, fp_counts=fpc, ladder="M2")
    def _rate(site):
        fn = site["fn"]
        return np.asarray(getattr(fn, "base_dist", fn).rate)
    mu_ladder = _rate(tr["counts"])
    # feed model_cc (informative_ln) the same population/completeness draws and force
    # its lam_fp / t to the ladder's values via the deterministic sites' inputs
    vals = {k: tr[k]["value"] for k in ("sigma_N", "sigma_z", "theta_level", "theta_slope",
                                        "eps_N", "eps_z", "psi_c")}
    lam = tr["lam_fp"]["value"]; t = tr["t"]["value"]
    # informative_ln parameterises lam_fp = lam_total * softmax(v): invert on live cells
    live = np.asarray(consts.dX, float).sum(0) > 0
    lam_np = np.asarray(lam)
    tot = lam_np.sum()
    v = np.log(np.clip(lam_np / tot, 1e-300, None)).reshape(-1)
    v[np.isclose(lam_np.reshape(-1), 0.0)] = -700.0   # dead cells: negligible share
    vals.update({"fp_lam_total": jnp.asarray(tot), "fp_shape_v": jnp.asarray(v), "t": t})
    tr2 = trace(substitute(hseed(model_cc, jax.random.PRNGKey(0)), data=vals)).get_trace(
        consts, Mg, counts=counts, fp_counts=fpc, fp_mode="informative_ln")
    mu_cc = _rate(tr2["counts"])
    assert np.allclose(mu_ladder, mu_cc, rtol=1e-9, atol=0.0), np.max(np.abs(mu_ladder / mu_cc - 1))


def test_M0_is_template_times_scalar(setup):
    from CDDF_analysis.hbi_mcmc.fp_ladder import model_cc_ladder, perks_log_share, live_mask
    pk, consts, Mg, counts, fpc = setup
    tr = _trace(model_cc_ladder, 7, consts, Mg, counts=counts, fp_counts=fpc, ladder="M0")
    live = live_mask(consts)
    m = perks_log_share(np.asarray(pk.fp_counts), live)
    lam = np.asarray(tr["lam_fp"]["value"]); l0 = float(tr["fp_l0"]["value"])
    assert np.allclose(np.log(lam[:, live]), l0 + m[:, live], atol=1e-6)
    assert np.all(np.asarray(tr["t"]["value"]) == 0.0)


def test_prior_moments_count(setup):
    from CDDF_analysis.hbi_mcmc.fp_ladder import fp_prior_moments, NOMINAL_FP_DOF
    pk, consts, *_ = setup
    for lad in ("M0", "M1", "M2", "M3", "M4", "M5"):
        names, m, s = fp_prior_moments(consts, pk.fp_counts, lad)
        assert len(names) == NOMINAL_FP_DOF[lad], (lad, len(names))


def test_oracle_pins_mu_fp_exactly(setup):
    """ORACLE: mu = tp + mu_fp_fixed with no FP parameters, no calibration site."""
    from CDDF_analysis.hbi_mcmc.fp_ladder import model_cc_ladder
    pk, consts, Mg, counts, fpc = setup
    rng = np.random.default_rng(0)
    mu_fixed = rng.poisson(2.0, size=np.asarray(pk.counts).shape).astype(float)
    mu_fixed[:, :, :2] = 0.0
    tr = _trace(model_cc_ladder, 11, consts, Mg, counts=counts, fp_counts=fpc, ladder="ORACLE", mu_fp_fixed=mu_fixed)
    sampled = [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]
    assert not any(k.startswith("fp_") for k in sampled) and "t" not in sampled
    assert "fp_counts" not in tr
    fn = tr["counts"]["fn"]; mu = np.asarray(getattr(fn, "base_dist", fn).rate)
    # tp alone from a second trace with mu_fp_fixed = 0 at the same seed
    tr0 = _trace(model_cc_ladder, 11, consts, Mg, counts=counts, fp_counts=fpc, ladder="ORACLE", mu_fp_fixed=np.zeros_like(mu_fixed))
    fn0 = tr0["counts"]["fn"]; tp = np.asarray(getattr(fn0, "base_dist", fn0).rate)
    # same seed => identical population draws; mu must equal tp + mu_fixed to float32 precision
    # (prior draws of f can be astronomically large, so compare relatively, and absolutely
    # only where the TP term is small enough for the FP term to be resolvable)
    assert np.allclose(mu, tp + mu_fixed, rtol=1e-6, atol=1e-6)
    small = tp < 1e3
    assert small.sum() > 100
    assert np.allclose((mu - tp)[small], mu_fixed[small], rtol=1e-4, atol=1e-3)
