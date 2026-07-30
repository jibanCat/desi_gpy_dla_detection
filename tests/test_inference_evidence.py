"""test_inference_evidence.py -- tests for the evidence harness itself.

The harness decides whether a posterior is citable, so the tests that matter
most are the ones that prove it cannot be fooled:

  * OMISSION SENSITIVITY (the headline requirement): removing any required
    evidence block, or any single check inside a block, or marking a block
    incomplete, must flip ``stampable`` to False.  A harness that stamps a run
    because a check was never run is worse than no harness.
  * A TRUTH IDENTITY: folding the pack's OWN truth f through the reported-
    quantity reducer must reproduce the block-3 truth values exactly.  This is
    what catches a mis-specified truth counterpart (it caught one: the plain-
    sum-over-z ``integrated_total`` had been given a pathlength-weighted-mean
    truth, a factor ~n_z error that looked like a 16-sigma closure failure).
  * PPC DISCRIMINATION: the posterior predictive block must PASS a self-
    consistent pack and FAIL data the fitted forward model cannot reproduce.
    That is the rung-9 failure mode, and it is the only block that sees it.

All packs are SYNTHETIC.  Sampler settings are tiny; the assertions test
structure, signs and discrimination, not sampler beauty.

Run: conda run -n gpdla-hbi python -m pytest tests/test_inference_evidence.py -q

TEST-COUNT PROVENANCE (2026-07-29 correction)
---------------------------------------------
Earlier reporting on this work quoted test counts with no stated selection,
and three of those numbers do not reproduce.  Numbers are only meaningful with
the selection that produced them, so the measured ones are recorded here:

  "~66 tests, every one failing before its fix"  -> WRONG in "every one".
      Selection: every test id present in tests/test_inference_evidence.py +
      test_modelA_forward_selftest.py + test_posterior_estimator.py at 29a22e3
      but ABSENT at 7bcaa5a^, run with the eight touched source files reverted
      to 7bcaa5a^.
      Measured: 77 before, 147 after, 70 ADDED; 66 red / 4 green on revert.
      The 4 green are controls and one measurement artefact, not omissions:
        test_forward_gate_still_passes_a_clean_table          (negative control)
        test_no_prepared_sbatch_bypasses_the_forward_closure_gate (already true)
        test_require_closure_exits_nonzero_on_the_v11_pack    (pack fails either way)
        test_committed_rung9_selftest_artifact_carries_a_resolvable_full_sha
              (reads the artifact JSON, which the revert set did not revert)
      An independent referee measured 65 red / 5 green with a slightly
      different revert set; both readings agree that "every one" is false.

  "22 passed"   -> does NOT reproduce.  No selection at 29a22e3 gives 22.
                   Dropped rather than reinterpreted.
  "253 passed"  -> did NOT reproduce at the time it was written.  At 29a22e3
                   this file held 96 tests; the three gate files together held
                   147; modelA* + evidence + posterior held 241.  253 is the
                   count of THIS FILE at a8e81fa and later, which is not what
                   the original claim referred to.

  The one count that DOES reproduce: d95668a's "50 new cases ... all failing
  before" -- 46 -> 96 collected, and those exact 50 ids give 50 failed with
  evidence.py at d95668a^ and 50 passed at d95668a.

  Current (HEAD): this file 253 passed; test_posterior_estimator.py 39;
  test_modelA_forward_selftest.py 24; the three together 316 passed.

  2026-07-29, gate-ratification branch (measured, `--collect-only | grep -c ::`
  in env gpdla-hbi): this file 253; test_posterior_estimator.py 39;
  test_modelA_forward_selftest.py 27; test_gate_ratification.py 45 (new);
  all four together 364 passed in 116 s.
"""
import copy

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from CDDF_analysis.hbi_mcmc import evidence as EV  # noqa: E402
from CDDF_analysis.hbi_mcmc import sbc as SBC  # noqa: E402
from CDDF_analysis.hbi_mcmc import forward_selftest as FS  # noqa: E402
from CDDF_analysis.hbi_mcmc.forward import build_consts  # noqa: E402
from CDDF_analysis.hbi_mcmc.model_a import (  # noqa: E402
    ModelAConfig, run_model_a)
from CDDF_analysis.hbi_mcmc.pack import synthetic_pack  # noqa: E402

# the SBC-scale grid: 5 N bins straddling BOTH thresholds, 2 fine z, 1 coarse
GRID = dict(
    nhat_edges=np.round(np.arange(19.9, 20.4 + 1e-9, 0.1), 10),
    zf_edges=np.round(np.arange(2.0, 2.2 + 1e-9, 0.1), 10),
    zc_edges=np.array([2.0, 2.2]),
    snr_edges=np.array([0.0, np.inf]),
    n_molly_cells=2,
)
# a slightly bigger grid with 2 coarse-z bins, for the z-tilt block
ZGRID = dict(
    nhat_edges=np.round(np.arange(19.9, 20.4 + 1e-9, 0.1), 10),
    zf_edges=np.round(np.arange(2.0, 2.4 + 1e-9, 0.1), 10),
    zc_edges=np.array([2.0, 2.2, 2.4]),
    snr_edges=np.array([0.0, np.inf]),
    n_molly_cells=2,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pack():
    return synthetic_pack(0, **GRID, fp_frac=0.15)


@pytest.fixture(scope="module")
def zpack():
    return synthetic_pack(1, **ZGRID, fp_frac=0.15, t_true=np.array([0.1, -0.1]))


@pytest.fixture(scope="module")
def fit(pack):
    cfg = ModelAConfig(num_warmup=150, num_samples=150, num_chains=2,
                       max_tree_depth=8, seed=0)
    mcmc, red = run_model_a(pack, cfg)
    run = EV.posterior_run_from_mcmc(mcmc, pack,
                                     max_tree_depth=cfg.max_tree_depth)
    return run, red, build_consts(pack, resp_clamp=cfg.resp_clamp)


def _passing_blocks():
    """A synthetic evidence set in which EVERY check passes.  The omission
    tests all start here, so the baseline must genuinely stamp."""
    mk = lambda **ck: {"checks": dict(ck), "incomplete": []}   # noqa: E731
    return {
        "convergence": mk(reported_r_hat_ok=True, reported_ess_bulk_ok=True,
                          reported_ess_tail_ok=True, latent_r_hat_ok=True,
                          latent_ess_bulk_ok=True, latent_ess_tail_ok=True,
                          divergences_ok=True, treedepth_ok=True,
                          ebfmi_ok=True),
        "ppc": mk(ppc_cells_ok=True, ppc_omnibus_ok=True),
        "closure": mk(closure_cover95_ok=True,
                      closure_cover68_not_pathological=True),
        # ``sbc_configuration_matches_run`` is a STRUCTURALLY required check
        # (evidence.REQUIRED_CHECKS) since the matched-configuration-SBC
        # ratification of 2026-07-29: a coverage_sbc block that omits it is
        # not stampable, so the passing baseline has to assert it.
        "coverage_sbc": mk(sbc_uniform_ok=True, sbc_enough_replicas=True,
                           sbc_configuration_matches_run=True),
        "ztilt": mk(ztilt_has_a_defensible_product=True,
                    ztilt_z_resolved_ok=True),
    }


# --------------------------------------------------------------------------
# 1. the gate: fail-closed + OMISSION SENSITIVITY
# --------------------------------------------------------------------------

def test_baseline_all_checks_pass_is_stampable():
    g = EV.gate(_passing_blocks())
    assert g["stampable"] is True
    assert g["paper_facing"] is True
    assert g["estimand"] == "POSTERIOR_MEDIAN_CI"
    assert g["n_failed"] == 0 and g["n_checks"] == 18   # +1: sbc_configuration_matches_run
    assert g["reasons"] == []


@pytest.mark.parametrize("block", EV.REQUIRED_BLOCKS)
def test_omitting_any_required_block_refuses_the_stamp(block):
    """OMISSION SENSITIVITY: a check that was never run is a FAILURE."""
    b = _passing_blocks()
    del b[block]
    g = EV.gate(b)
    assert g["stampable"] is False
    assert block in g["missing_blocks"]
    assert any(block in r for r in g["reasons"])


@pytest.mark.parametrize("block", EV.REQUIRED_BLOCKS)
def test_a_required_block_present_but_empty_refuses_the_stamp(block):
    """Deleting a block's CHECKS (rather than the block) must not stamp either."""
    b = _passing_blocks()
    b[block] = {"checks": {}, "incomplete": []}
    g = EV.gate(b)
    assert g["stampable"] is False
    assert g["checks"].get(f"{block}.__present__") is False


def test_every_single_check_is_load_bearing():
    """Flipping ANY one check to False must refuse the stamp."""
    base = _passing_blocks()
    keys = [(bn, ck) for bn, blk in base.items() for ck in blk["checks"]]
    assert len(keys) == 18   # +1: sbc_configuration_matches_run (2026-07-29)
    for bn, ck in keys:
        b = copy.deepcopy(base)
        b[bn]["checks"][ck] = False
        g = EV.gate(b)
        assert g["stampable"] is False, f"{bn}.{ck} was not load-bearing"
        assert f"failed check: {bn}.{ck}" in g["reasons"]


@pytest.mark.parametrize("block", EV.REQUIRED_BLOCKS)
def test_incomplete_evidence_alone_refuses_the_stamp(block):
    """All checks green but the block could not compute something -> refuse."""
    b = _passing_blocks()
    b[block]["incomplete"] = ["something_unrecoverable"]
    g = EV.gate(b)
    assert g["stampable"] is False
    assert block in g["incomplete"]


def test_a_block_set_to_none_is_missing_not_passing():
    b = _passing_blocks()
    b["ppc"] = None
    g = EV.gate(b)
    assert g["stampable"] is False and "ppc" in g["missing_blocks"]


def test_assemble_stamps_provenance_and_scope():
    ev = EV.assemble_evidence(_passing_blocks())
    assert ev["schema"] == "inference_evidence/v1"
    assert ev["provenance"]["module"] == "CDDF_analysis/hbi_mcmc/evidence.py"
    assert ev["provenance"]["code_commit"]
    assert ev["scope"] == "MOCK ONLY"


# --------------------------------------------------------------------------
# 2. reported quantities + the TRUTH IDENTITY
# --------------------------------------------------------------------------

def test_reported_quantities_are_chain_resolved_and_named(pack):
    f = np.abs(np.random.default_rng(0).normal(
        1.0, 0.1, size=(3, 7, pack.n_b, pack.n_k)))
    rep = EV.reported_quantities(f, pack)
    assert all(v.shape == (3, 7) for v in rep.values())
    for thr in ("20p0", "20p3"):
        for stat in ("dndx", "omega"):
            assert f"{stat}_{thr}_integrated" in rep
            for K in range(pack.n_kk):
                assert f"{stat}_{thr}_z{K}" in rep
    assert "integrated_total" in rep


def test_chain_reshape_preserves_draw_identity(pack):
    """(chains, draws) must map back to the same flat draws in order."""
    rng = np.random.default_rng(3)
    f = np.abs(rng.normal(1.0, 0.2, size=(2, 5, pack.n_b, pack.n_k)))
    a = EV.reported_quantities(f, pack)["dndx_20p3_integrated"]
    b = EV.reported_quantities(f.reshape(1, 10, pack.n_b, pack.n_k),
                               pack)["dndx_20p3_integrated"]
    assert np.allclose(a.reshape(-1), b.reshape(-1))


def test_truth_counterpart_is_the_reducer_applied_to_the_truth(pack):
    """THE identity: reported_quantities(truth f) == the block-3 truth values.

    Any truth counterpart that is not literally the reducer applied to the
    pack's own truth f is a bug; this is the test that caught the
    integrated_total mis-specification.
    """
    f_true = FS.truth_f(pack)
    got = EV.reported_quantities(f_true[None, None], pack)
    want, _ = EV._truth_reported(pack)
    assert set(got) == set(want)
    for k in want:
        assert float(np.asarray(got[k]).reshape(-1)[0]) == pytest.approx(
            want[k], rel=1e-12), k


def test_truth_requires_a_mock_pack(pack):
    import dataclasses
    p = dataclasses.replace(pack, truth_counts=None)
    with pytest.raises(ValueError):
        EV._truth_reported(p)


# --------------------------------------------------------------------------
# 3. convergence block
# --------------------------------------------------------------------------

def test_model_a_retains_the_fields_the_harness_needs():
    """ANTI-REGRESSION: tree-depth saturation and E-BFMI are required evidence
    and are UNRECOVERABLE after the run.  Shrinking EXTRA_FIELDS back to
    ("diverging",) silently makes every future run un-stampable."""
    from CDDF_analysis.hbi_mcmc import model_a as MA
    assert set(MA.EXTRA_FIELDS) >= {"diverging", "num_steps", "energy"}


def test_convergence_reports_raw_numbers_for_every_reported_quantity(fit, pack):
    run, _, _ = fit
    blk = EV.convergence_block(run, pack)
    rep = EV.reported_quantities(run["f_by_chain"], pack)
    assert set(blk["reported"]) == set(rep)
    for v in blk["reported"].values():
        assert np.isfinite(v["r_hat"]) and v["ess_bulk"] > 0
    # raw numbers are present even when the gate fails
    assert np.isfinite(blk["summary"]["reported_r_hat_max"])
    assert blk["treedepth"]["max_tree_depth"] == 8
    assert 0.0 <= blk["treedepth"]["frac_saturated"] <= 1.0
    assert np.isfinite(blk["ebfmi"]["min"])
    assert blk["divergences"]["n_divergent"] >= 0
    assert set(blk["checks"]) == {
        "reported_r_hat_ok", "reported_ess_bulk_ok", "reported_ess_tail_ok",
        "latent_r_hat_ok", "latent_ess_bulk_ok", "latent_ess_tail_ok",
        "divergences_ok", "treedepth_ok", "ebfmi_ok"}


def test_convergence_gate_is_fail_closed_on_a_bad_rhat(fit, pack):
    run, _, _ = fit
    blk = EV.convergence_block(run, pack, policy=dict(
        r_hat_max=1.0, ess_bulk_min=1e12, ess_tail_min=1e12, n_divergent=0))
    assert blk["checks"]["reported_r_hat_ok"] is False
    assert blk["checks"]["reported_ess_bulk_ok"] is False


def test_artifact_mode_cannot_supply_the_missing_evidence(fit, pack):
    """A saved reductions-only artifact is structurally un-stampable: the chain
    axis is an assumption and num_steps / energy were never written."""
    run, red, _ = fit
    f = np.asarray(run["f_by_chain"])
    result = {"reductions": {"f": f.reshape((-1,) + f.shape[2:]).tolist()},
              "sampler": {"chains": int(f.shape[0])}, "diagnostics": {}}
    arun = EV.posterior_run_from_artifact(result, pack)
    assert arun["chain_axis_assumed_contiguous"] is True
    assert np.allclose(arun["f_by_chain"], f)
    blk = EV.convergence_block(arun, pack)
    for tag in ("latent_sites_absent", "divergence_flags_absent",
                "treedepth_needs_num_steps_extra_field",
                "ebfmi_needs_energy_extra_field"):
        assert tag in blk["incomplete"]
    assert blk["checks"]["treedepth_ok"] is False
    assert blk["checks"]["ebfmi_ok"] is False
    assert EV.gate({**_passing_blocks(), "convergence": blk})["stampable"] \
        is False


def test_divergence_location_finds_an_injected_cluster():
    ramp = np.tile(np.linspace(0.0, 1.0, 100), (2, 1))
    lat = {"a": ramp, "b": np.random.default_rng(0).normal(size=(2, 100))}
    div = np.zeros((2, 100), bool)
    div[:, -10:] = True                      # divergences at the top of "a"
    loc = EV._divergence_location(EV._flatten_sites(lat), div)
    assert loc["n_divergent"] == 20
    assert loc["top_coords"][0]["coord"] == "a"
    assert loc["top_coords"][0]["z_shift"] > 1.0
    assert loc["localized"] is True
    assert loc["top_coords"][0]["mean_quantile_of_divergent"] > 0.9


def test_no_divergences_reports_cleanly():
    lat = {"a": np.random.default_rng(0).normal(size=(2, 50))}
    loc = EV._divergence_location(EV._flatten_sites(lat), np.zeros((2, 50), bool))
    assert loc == {"n_divergent": 0, "localized": False, "top_coords": []}


# --------------------------------------------------------------------------
# 4. posterior predictive checks -- the rung-9 discriminator
# --------------------------------------------------------------------------

def test_ppc_passes_a_self_consistent_pack(fit, pack):
    run, _, consts = fit
    blk = EV.ppc_block(run, pack, consts, n_rep_draws=150, seed=0)
    assert blk["n_cells"] > 0
    assert blk["checks"]["ppc_cells_ok"] is True
    p = blk["omnibus_chi2_discrepancy"]["posterior_predictive_p"]
    assert EV.PPC_OMNIBUS_MIN <= p <= 1 - EV.PPC_OMNIBUS_MIN
    assert blk["checks"]["ppc_omnibus_ok"] is True


def test_ppc_fails_counts_the_fitted_model_cannot_reproduce(fit, pack):
    """The rung-9 signature: observed counts the forward model cannot reach.

    Scaling the observed counts by 3 leaves the FITTED posterior where it was,
    so the replicated counts fall far short -- exactly the 0.165x mu/obs
    situation, and the PPC must say so while R-hat stays green.
    """
    import dataclasses
    run, _, consts = fit
    bad = dataclasses.replace(
        pack, counts=(np.asarray(pack.counts) * 3).astype(np.int64))
    blk = EV.ppc_block(run, bad, consts, n_rep_draws=150, seed=0)
    assert blk["n_cells_failed"] > 0
    assert blk["checks"]["ppc_cells_ok"] is False
    worst = blk["failed_cells"][0]
    assert worst["ratio_mu_over_obs"] < 0.9
    assert worst["p_two_sided"] < EV.PPC_PVAL_MIN
    assert blk["checks"]["ppc_omnibus_ok"] is False


def test_ppc_refuses_when_the_nuisance_draws_are_absent(pack):
    run = {"samples_by_chain": None, "f_by_chain": np.ones((1, 1, pack.n_b,
                                                            pack.n_k))}
    blk = EV.ppc_block(run, pack, None)
    assert blk["checks"] == {"ppc_cells_ok": False, "ppc_omnibus_ok": False}
    assert blk["incomplete"] == ["ppc_needs_latent_posterior_draws"]


def test_mid_p_is_uniform_for_a_correct_model():
    rng = np.random.default_rng(0)
    lam = 40.0
    rep = rng.poisson(lam, size=(4000, 500)).astype(float)
    obs = rng.poisson(lam, size=500).astype(float)
    p = EV._mid_p(rep, obs[None])
    assert 0.45 < p.mean() < 0.55
    assert 0.24 < p.std() < 0.32          # Uniform(0,1) has sd 0.289


# --------------------------------------------------------------------------
# 5. closure
# --------------------------------------------------------------------------

def test_closure_recovers_the_truth_when_the_posterior_is_the_truth(pack):
    """Degenerate posterior sitting exactly on the truth: ratio == 1 exactly."""
    f_true = FS.truth_f(pack)
    rng = np.random.default_rng(11)
    # a hair of spread: an EXACTLY degenerate posterior makes the 2.5/97.5
    # quantiles equal the truth to within float summation order, which is a
    # coin flip, not a coverage statement.
    draws = f_true[None, None] * (
        1.0 + 1e-4 * rng.normal(size=(2, 40, pack.n_b, pack.n_k)))
    blk = EV.closure_block({"f_by_chain": draws}, pack)
    for r in blk["rows"]:
        assert r["ratio_median"] == pytest.approx(1.0, rel=1e-3)
    assert blk["coverage95"] == 1.0
    assert blk["checks"]["closure_cover95_ok"] is True


def test_closure_fails_a_biased_posterior(pack):
    f_true = FS.truth_f(pack)
    rng = np.random.default_rng(0)
    draws = 0.5 * f_true[None, None] * (
        1.0 + 0.01 * rng.normal(size=(2, 40, pack.n_b, pack.n_k)))
    blk = EV.closure_block({"f_by_chain": draws}, pack)
    assert blk["coverage95"] == 0.0
    assert blk["checks"]["closure_cover95_ok"] is False
    assert all(r["ratio_median"] < 0.6 for r in blk["rows"])
    assert blk["worst_z"] > 5


# --------------------------------------------------------------------------
# 6. z-tilt
# --------------------------------------------------------------------------

def test_ztilt_clean_run_is_z_resolved_defensible(zpack):
    f_true = FS.truth_f(zpack)
    rng = np.random.default_rng(0)
    draws = f_true[None, None] * (
        1.0 + 0.02 * rng.normal(size=(2, 60, zpack.n_b, zpack.n_k)))
    blk = EV.ztilt_block({"f_by_chain": draws}, zpack, forward_fold=False)
    assert blk["n_z_in95"] == blk["n_z_bins"] == zpack.n_kk
    assert blk["z_resolved_defensible"] is True
    assert blk["integrated_only_defensible"] is False
    assert blk["checks"]["ztilt_has_a_defensible_product"] is True
    assert abs(blk["R0_span"]) < 0.1


def test_ztilt_detects_an_injected_tilt_and_demotes_to_integrated(zpack):
    """An f that is tilted in z but right on average: per-z intervals miss the
    truth, the integrated one still holds it -> the block must say that the
    INTEGRATED product is the only defensible one."""
    f_true = FS.truth_f(zpack)
    kz = np.asarray(zpack.kz_to_K)
    ramp = np.where(kz == 0, 0.80, 1.0)
    ramp = np.where(kz == zpack.n_kk - 1, 1.22, ramp)
    rng = np.random.default_rng(0)
    draws = (f_true * ramp[None, :])[None, None] * (
        1.0 + 0.01 * rng.normal(size=(2, 60, zpack.n_b, zpack.n_k)))
    blk = EV.ztilt_block({"f_by_chain": draws}, zpack, forward_fold=False)
    assert blk["R0_span"] > 0.3
    assert blk["R0_slope_per_unit_z"] > 0
    assert blk["n_z_in95"] < blk["n_z_bins"]
    assert blk["z_resolved_defensible"] is False
    assert blk["checks"]["ztilt_z_resolved_ok"] is False
    assert blk["tilt_over_statistical_width"] > 1.0
    # the reference defect this block exists to measure stays in the artifact
    assert blk["reference_defect"]["R0_by_z"] == [0.908, 1.052, 1.189]


def test_ztilt_forward_fold_is_a_sampling_free_measurement(zpack):
    blk = EV.ztilt_block(
        {"f_by_chain": FS.truth_f(zpack)[None, None]}, zpack, forward_fold=True)
    ff = blk["forward_fold_ztilt"]
    assert len(ff["mu_over_obs_by_fine_z"]) == zpack.n_k
    assert ff["span"] is not None and ff["total_ratio"] > 0
    assert ff["resp_clamp"] == "both"


# --------------------------------------------------------------------------
# 7. SBC helpers
# --------------------------------------------------------------------------

def test_uniformity_test_accepts_uniform_ranks():
    rng = np.random.default_rng(0)
    ranks = rng.integers(0, 51, size=400)
    t = SBC.uniformity_test(ranks, 50, n_bins=10)
    assert t["p_value"] > 0.01
    assert t["shape"] == "uniform"
    assert t["hist"]["n"] == 400


def test_uniformity_test_flags_overconfident_U_shaped_ranks():
    ranks = np.concatenate([np.zeros(100, int), np.full(100, 50)])
    t = SBC.uniformity_test(ranks, 50, n_bins=10)
    assert t["p_value"] < 1e-6
    assert t["shape"] == "U_shaped_intervals_too_narrow"


def test_uniformity_test_flags_overdispersed_central_ranks():
    ranks = np.full(200, 25)
    t = SBC.uniformity_test(ranks, 50, n_bins=10)
    assert t["p_value"] < 1e-6
    assert t["shape"] == "central_hump_intervals_too_wide"


def test_sbc_reductions_are_declared_not_hidden():
    """The task requires saying EXACTLY what was reduced."""
    assert set(SBC.SBC_PRIOR) >= {"level_scale", "sigma_N_scale", "fp_mode"}
    assert SBC.SBC_PRIOR["fp_mode"] == "off"
    edges = np.asarray(SBC.SBC_GRID["nhat_edges"], float)
    # both reporting thresholds must still live on the reduced grid, or the
    # SBC would be calibrating different functionals than the paper quotes
    assert edges[0] < 20.0 < edges[-1] and edges[0] < 20.3 < edges[-1]


def test_sbc_block_refuses_when_no_replica_is_usable(monkeypatch):
    monkeypatch.setattr(SBC, "sbc_run",
                        lambda *a, **k: ({}, {"n_sims_used": 0}))
    blk = SBC.sbc_block(4)
    assert blk["checks"]["sbc_uniform_ok"] is False
    assert blk["incomplete"] == ["sbc_produced_no_usable_replicas"]


# --------------------------------------------------------------------------
# 8. the entry point's guards
# --------------------------------------------------------------------------

def test_entry_point_refuses_real_survey_inputs():
    from CDDF_analysis.hbi_mcmc import run_evidence as RE
    for tok in ("loa_main_dark_v1", "matterhorn", "DR3"):
        with pytest.raises(SystemExit):
            RE._refuse_real({"pack": f"pack_{tok}.npz"}, "test")
    RE._refuse_real({"pack": "modelA_pack_2lpt0.npz"}, "test")   # no raise


def test_skipping_sbc_cannot_stamp():
    b = _passing_blocks()
    del b["coverage_sbc"]
    assert EV.assemble_evidence(b)["gate"]["stampable"] is False


def test_integrated_only_verdict_is_reported_separately():
    """The project's real situation: per-z coverage fails, the z-marginalised
    product still holds.  That must get its OWN name, and must NOT silently
    satisfy the full stamp."""
    b = _passing_blocks()
    b["ztilt"]["checks"]["ztilt_z_resolved_ok"] = False
    g = EV.gate(b)
    assert g["stampable"] is False
    assert g["stampable_integrated_only"] is True
    # any OTHER failure kills the weaker verdict too
    b["ppc"]["checks"]["ppc_cells_ok"] = False
    g2 = EV.gate(b)
    assert g2["stampable"] is False and g2["stampable_integrated_only"] is False
    # and a fully-passing run is not labelled "integrated only"
    assert EV.gate(_passing_blocks())["stampable_integrated_only"] is False


# ==========================================================================
# 9. THE FAIL-OPEN HOLES (2026-07-29 gate audit)
#
# Each test below reproduces a hole through which an artifact could be
# stamped ``stampable=True, paper_facing=True`` without the evidence that
# stamp asserts.  These are CODE-PATH defects, demonstrated by calling the
# gate; no claim is made that a badly-stamped file was found on disk.
# ==========================================================================

def _sbc_only_blocks():
    return {"coverage_sbc": {"checks": {"sbc_uniform_ok": True,
                                        "sbc_enough_replicas": True},
                             "incomplete": []}}


def test_a_narrowed_required_list_cannot_shrink_the_gate():
    """HOLE 1. ``run_evidence --mode sbc`` called

        assemble_evidence(blocks, required=("coverage_sbc",))

    and ``gate`` counts only blocks named in ``required`` as missing, so the
    four absent blocks raised no objection: the call RETURNED stampable=True,
    paper_facing=True, n_checks=2, blocks=['coverage_sbc'].  (No artifact
    written by that path was located on disk during the 2026-07-29 audit --
    the claim here is about the code path's return value, which this test
    exercises directly.)  An SBC-only run must NEVER be stampable."""
    g = EV.gate(_sbc_only_blocks(), required=("coverage_sbc",))
    assert g["stampable"] is False
    assert g["paper_facing"] is False
    for b in ("convergence", "ppc", "closure", "ztilt"):
        assert b in g["missing_blocks"], (b, g["missing_blocks"])


def test_a_narrowed_required_list_cannot_shrink_the_assembled_artifact():
    ev = EV.assemble_evidence(_sbc_only_blocks(), required=("coverage_sbc",))
    assert ev["gate"]["stampable"] is False
    assert ev["gate"]["paper_facing"] is False
    assert set(ev["gate"]["required_blocks"]) >= set(EV.REQUIRED_BLOCKS)


@pytest.mark.parametrize("junk", [[], "", 0, False, 1.0, "ok", ["ppc"], ()])
@pytest.mark.parametrize("block", EV.REQUIRED_BLOCKS)
def test_a_required_block_that_is_not_a_dict_refuses_the_stamp(block, junk):
    """HOLE 2. ``gate`` skipped any non-dict block (``if not isinstance(blk,
    dict): continue``) and only ``None``/``{}`` were caught as missing, so
    ``blocks['ppc'] = []`` gave stampable=True, missing=[].  The existing
    omission tests use exactly the None/{} pair, so they passed vacuously."""
    b = _passing_blocks()
    b[block] = junk
    g = EV.gate(b)
    assert g["stampable"] is False, f"{block}={junk!r} was accepted"
    assert g["paper_facing"] is False
    assert block in g["missing_blocks"] + g["invalid_blocks"]
    assert any(block in r for r in g["reasons"])


@pytest.mark.parametrize("junk", [[], "", 0, False])
def test_a_non_required_block_that_is_not_a_dict_also_refuses(junk):
    """A malformed EXTRA block is a corrupt artifact, not a passing one."""
    b = _passing_blocks()
    b["extra_block"] = junk
    g = EV.gate(b)
    assert g["stampable"] is False
    assert "extra_block" in g["invalid_blocks"]


def test_a_bypass_flag_is_recorded_and_forces_paper_facing_false():
    """HOLE 3. Any gate-bypass flag (--allow-low-farr, --allow-open-forward-
    model) must appear IN the artifact and must make it non-paper-facing."""
    g = EV.gate(_passing_blocks(),
                bypasses={"allow_low_farr": "on-mock self-calibration"})
    assert g["bypasses"] == {"allow_low_farr": "on-mock self-calibration"}
    assert g["paper_facing"] is False
    assert g["stampable"] is False
    assert any("bypass" in r for r in g["reasons"])
    # no bypass -> the field is present and empty, never absent
    assert EV.gate(_passing_blocks())["bypasses"] == {}


def test_assemble_records_bypasses_in_provenance_too():
    ev = EV.assemble_evidence(_passing_blocks(),
                              bypasses={"allow_open_forward_model": "why"})
    assert ev["gate"]["bypasses"]
    assert ev["provenance"]["bypasses"] == {"allow_open_forward_model": "why"}


def test_sbc_mode_end_to_end_is_not_stampable(tmp_path, monkeypatch):
    """The whole hole-1 path, through the CLI that produced the bad artifact."""
    from CDDF_analysis.hbi_mcmc import run_evidence as RE
    from CDDF_analysis.hbi_mcmc import sbc as _sbc
    monkeypatch.setattr(_sbc, "sbc_block", lambda *a, **k: {
        "checks": {"sbc_uniform_ok": True, "sbc_enough_replicas": True},
        "incomplete": []})
    out = tmp_path / "ev_sbc.json"
    ev = RE.main(["--mode", "sbc", "--sbc-sims", "2", "--out", str(out)])
    assert ev["gate"]["stampable"] is False
    assert ev["gate"]["paper_facing"] is False
    assert set(ev["gate"]["missing_blocks"]) == {
        "convergence", "ppc", "closure", "ztilt"}
    import json as _json
    on_disk = _json.loads(out.read_text())
    assert on_disk["gate"]["stampable"] is False
    assert on_disk["gate"]["paper_facing"] is False


def test_run_evidence_forwards_its_bypass_flag_into_the_artifact(tmp_path,
                                                                 monkeypatch):
    from CDDF_analysis.hbi_mcmc import run_evidence as RE
    from CDDF_analysis.hbi_mcmc import sbc as _sbc
    monkeypatch.setattr(_sbc, "sbc_block", lambda *a, **k: {
        "checks": {"sbc_uniform_ok": True}, "incomplete": []})
    out = tmp_path / "ev_sbc2.json"
    ev = RE.main(["--mode", "sbc", "--sbc-sims", "2", "--out", str(out),
                  "--allow-low-farr", "documented on-mock reason"])
    assert ev["gate"]["bypasses"]["allow_low_farr"] == "documented on-mock reason"
    assert ev["gate"]["paper_facing"] is False


# ==========================================================================
# 10. HOLE 4 -- the non-dict hardening stopped one level too shallow
#
# Section 9 hardened the BLOCK level but left the CHECK VALUE level coercing
# with ``bool(v)``, which is the same fail-open class one level down: a check
# value of ``'no'`` or ``[0]`` is TRUTHY in Python while meaning "not ok" to
# a human, so it stamped.  ``incomplete`` had the mirror hole: a non-list
# was silently dropped by ``list(blk.get('incomplete') or [])``, and a
# non-dict ``checks`` crashed with AttributeError instead of failing closed.
# ==========================================================================

# every one of these is a value a human would NOT call "this check passed"
_NON_BOOL_CHECK_VALUES = ["no", [0], 1.0, "ok", ["ppc"], (), None, 0, 1,
                          "True", "False", {}, {"ok": True}]


@pytest.mark.parametrize("val", _NON_BOOL_CHECK_VALUES)
@pytest.mark.parametrize("block", EV.REQUIRED_BLOCKS)
def test_a_check_value_that_is_not_a_bool_refuses_the_stamp(block, val):
    """A check must be a genuine bool.  ``'no'`` and ``[0]`` are the worst
    case: truthy to Python, "not ok" to a human."""
    b = _passing_blocks()
    b[block] = {"checks": {"some_check_ok": val}, "incomplete": []}
    g = EV.gate(b)
    assert g["stampable"] is False, f"{block}.some_check_ok={val!r} stamped"
    assert g["paper_facing"] is False
    assert g["checks"][f"{block}.some_check_ok"] is False
    assert any("not a bool" in r for r in g["reasons"]), g["reasons"]


@pytest.mark.parametrize("val", _NON_BOOL_CHECK_VALUES)
def test_a_non_bool_check_in_a_NON_required_block_also_refuses(val):
    b = _passing_blocks()
    b["extra"] = {"checks": {"whatever_ok": val}, "incomplete": []}
    g = EV.gate(b)
    assert g["stampable"] is False, f"extra.whatever_ok={val!r} stamped"
    assert g["paper_facing"] is False


@pytest.mark.parametrize("good", [True, np.bool_(True)])
def test_genuine_bools_including_numpy_still_pass(good):
    """The hardening must not break the real callers: model_a/ppc/sbc all
    build checks with ``bool(...)`` or numpy comparisons."""
    b = _passing_blocks()
    b["ppc"] = {"checks": {"ppc_ok": good}, "incomplete": []}
    g = EV.gate(b)
    assert g["stampable"] is True, g["reasons"]
    assert g["checks"]["ppc.ppc_ok"] is True


@pytest.mark.parametrize("bad", [False, np.bool_(False)])
def test_genuine_false_still_fails_as_a_failed_check_not_a_malformed_one(bad):
    b = _passing_blocks()
    b["ppc"] = {"checks": {"ppc_ok": bad}, "incomplete": []}
    g = EV.gate(b)
    assert g["stampable"] is False
    assert any("failed check: ppc.ppc_ok" in r for r in g["reasons"]), g["reasons"]
    assert not any("not a bool" in r for r in g["reasons"]), g["reasons"]


@pytest.mark.parametrize("junk", [0, 1, 1.0, True, "ppc", "", object()])
@pytest.mark.parametrize("block", EV.REQUIRED_BLOCKS)
def test_a_non_sequence_incomplete_refuses_the_stamp(block, junk):
    """``list(blk.get('incomplete') or [])`` silently dropped a non-list.
    ``incomplete=0`` and ``incomplete='ppc'`` both used to stamp (the latter
    would have exploded into per-character entries had it been truthy)."""
    b = _passing_blocks()
    b[block] = {"checks": {"x_ok": True}, "incomplete": junk}
    g = EV.gate(b)
    assert g["stampable"] is False, f"{block}.incomplete={junk!r} stamped"
    assert g["paper_facing"] is False
    assert any("incomplete" in r and "sequence" in r for r in g["reasons"]), \
        g["reasons"]


@pytest.mark.parametrize("ok", [[], (), ["a"], ("a", "b")])
def test_genuine_sequences_for_incomplete_are_honoured(ok):
    b = _passing_blocks()
    b["ppc"] = {"checks": {"ppc_ok": True}, "incomplete": ok}
    g = EV.gate(b)
    assert g["stampable"] is (not ok), g["reasons"]
    if ok:
        assert g["incomplete"]["ppc"] == list(ok)


@pytest.mark.parametrize("junk", ["ok", [], 0, 1.0, ["ppc_ok"], (), True])
@pytest.mark.parametrize("block", EV.REQUIRED_BLOCKS)
def test_a_non_dict_checks_mapping_fails_closed_and_does_not_raise(block, junk):
    """``blocks['ppc'] = {'checks': 'ok'}`` used to raise AttributeError out
    of the gate.  A gate that crashes is not a gate that fails closed."""
    b = _passing_blocks()
    b[block] = {"checks": junk, "incomplete": []}
    g = EV.gate(b)          # must NOT raise
    assert g["stampable"] is False, f"{block}.checks={junk!r} stamped"
    assert g["paper_facing"] is False
    assert any(block in r for r in g["reasons"]), g["reasons"]


def test_a_non_bool_check_survives_assembly_into_the_artifact():
    ev = EV.assemble_evidence({**_passing_blocks(),
                               "ppc": {"checks": {"ppc_pval_ok": "no"},
                                       "incomplete": []}})
    assert ev["gate"]["stampable"] is False
    assert ev["gate"]["paper_facing"] is False
