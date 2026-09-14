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


def test_fixed_component_hooks(setup):
    """Diagnostic hooks: P adds a fixed term; C 2D/3D, M and E paths run and change tp as expected."""
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.fp_ladder import model_cc_ladder
    pk, consts, Mg, counts, fpc = setup
    C, Kf, S, B = consts.n_c, consts.n_k, consts.n_s, consts.n_b
    z3 = np.zeros((C, Kf, S))
    def rate(**kw):
        tr = _trace(model_cc_ladder, 21, consts, Mg, counts=counts, fp_counts=fpc, ladder="ORACLE", mu_fp_fixed=z3, **kw)
        fn = tr["counts"]["fn"]; return np.asarray(getattr(fn, "base_dist", fn).rate)
    base = rate()
    extra = np.full((C, Kf, S), 3.0); extra[:, :, :2] = 0
    assert np.allclose(rate(mu_extra_fixed=extra) - base, extra, rtol=1e-5, atol=1e-3)
    # E = 0 kills the TP term entirely
    assert np.allclose(rate(E_fixed=np.zeros((C, Kf, S, B))), 0.0, atol=1e-9)
    # Mg_fixed identical to Mg reproduces base; scaled by 2 doubles tp
    assert np.allclose(rate(Mg_fixed=np.asarray(Mg)), base, rtol=1e-6)
    assert np.allclose(rate(Mg_fixed=2 * np.asarray(Mg)), 2 * base, rtol=1e-6)
    # C_fixed 2D equal to the calibrated C at psi_c=0 gives a finite fold; 3D ones broadcast
    C2 = np.asarray(1 / (1 + np.exp(-np.asarray(consts.eta_hat))))[:, np.asarray(consts.b_to_cell)]
    r2 = rate(C_fixed=C2); assert np.all(np.isfinite(r2)) and r2.shape == base.shape
    C3 = np.broadcast_to(C2.T[:, None, :], (B, Kf, S)).copy()
    r3 = rate(C_fixed=C3); assert np.all(np.isfinite(r3)) and r3.shape == base.shape


def test_m1cut_lambda_not_a_site_and_imputations(setup):
    from CDDF_analysis.hbi_mcmc.fp_ladder import model_cc_ladder, lambda_calibration_posterior_quantiles, live_mask
    pk, consts, Mg, counts, fpc = setup
    qs = lambda_calibration_posterior_quantiles(pk.fp_counts, consts.fp_ell_eff, 8)
    naive = float(np.asarray(pk.fp_counts).sum() / consts.fp_ell_eff)
    assert qs.shape == (8,) and np.all(np.diff(qs) > 0) and 0.7 * naive < qs[3] < 1.3 * naive
    med = lambda_calibration_posterior_quantiles(pk.fp_counts, consts.fp_ell_eff, 1)[0]
    tr = _trace(model_cc_ladder, 31, consts, Mg, counts=counts, fp_counts=fpc, ladder="M1CUT", lam_fixed=med)
    sampled = [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]
    assert "fp_lam_total" not in sampled and "fp_l0" not in sampled and "t" in sampled
    assert "fp_counts" not in tr                       # no calibration likelihood term (cut)
    lam = np.asarray(tr["lam_fp"]["value"]); live = live_mask(consts)
    assert np.isclose(lam.sum(), med, rtol=1e-5) and np.all(lam[:, ~live] == 0)


def test_perks_a0_battery_shares(setup):
    """PI 2026-09-13d §12: a0 = None reproduces the record template (1/K); a0 = 0 gives zero share to
    empty cells and renormalises; larger a0 moves mass into empty cells monotonically; shares sum to 1."""
    from CDDF_analysis.hbi_mcmc.fp_ladder import perks_log_share, live_mask
    pk, consts, Mg, counts, fpc = setup
    live = live_mask(consts); K = int(live.sum()) * fpc.shape[0]
    m_rec = perks_log_share(fpc, live); m_K = perks_log_share(fpc, live, a0=1.0 / K)
    assert np.allclose(m_rec[:, live], m_K[:, live], rtol=0, atol=1e-12)
    m0 = perks_log_share(fpc, live, a0=0.0)
    sh0 = np.exp(m0[:, live]); assert np.isclose(sh0.sum(), 1.0, rtol=0, atol=1e-12)
    empty = np.asarray(fpc, float)[:, live] == 0
    assert empty.any() and np.all(sh0[empty] == 0.0)
    prev = 0.0
    for a0 in (0.25 / K, 1.0 / K, 4.0 / K, 0.5):
        sh = np.exp(perks_log_share(fpc, live, a0=a0)[:, live]); assert np.isclose(sh.sum(), 1.0, rtol=0, atol=1e-12)
        mass_empty = sh[empty].sum(); assert mass_empty > prev; prev = mass_empty


def test_mg_phi_family_reconstruction_matches_rows_times_phi():
    """The runner's Mg_phi_family diagnostic path: rows_unit x phi_family reproduces the stored Mg when phi_family == phi_bsK."""
    import json, glob
    f = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/response_review/candidates/Mg_E_2lpt0.npz"
    if not os.path.exists(f):
        pytest.skip("candidate products not present")
    d = np.load(f, allow_pickle=True); kz = np.asarray(json.loads(str(d["provenance"]))["kz_to_K"], int)
    M = np.einsum("bsKc,bsK->sKcb", np.asarray(d["rows_unit"], float), np.asarray(d["phi_bsK_family_measured"], float))[:, kz]
    assert np.allclose(M, np.asarray(d["Mg"], float), rtol=0, atol=1e-12)   # 2LPT-0: family phi == 2LPT phi
