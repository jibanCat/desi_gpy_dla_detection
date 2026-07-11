"""test_modelA_rungs.py — Q3 Model A validation-ladder rungs 4-8 (spec section 4).

All packs are SYNTHETIC (pack.synthetic_pack; no survey data anywhere) on a
schema-consistent reduced grid (8 N-bins, 5 fine-z, 2 coarse-z, 2 SNR strata)
so the whole file stays near the ~5 minute budget with small sampler settings
(2 chains, 200/200 or less; vectorized chains; block-dense anchor mass).
Assertions are calibrated to fixed seeds and kept loose enough to be
non-flaky; they test SIGNS, COVERAGE and MONOTONICITY, not sampler beauty.

Rung map (one sampler fit is reused wherever the spec allows):
  R4 completeness: fit_main — known C from molly counts, sampled psi_c;
     68% coverage of f_true >= 55% of (b,k) bins; integrated total within 2sig.
  R5 finite calibration: fit_main (molly x1) vs fit_lowcal (molly x1/16, Farr
     gate deliberately disabled — and a separate assert that the gate FIRES
     when enabled): posterior sd of the integrated dN/dX total must GROW.
  R6 migration ablation: data generated with the SKEWED kernel on a steep
     slope; fit with a diagonal kernel => biased HIGH at high N (sign test);
     fit with the correct kernel => unbiased.
  R7 FP + exposure: joint t_K recovery within 2 sigma on fit_main;
     zero-FP pack => t posterior ~ prior and f unchanged vs the fp-off model;
     grid-refinement invariance of the single-Jeffreys FP total prior
     (trace-level: the prior on the TOTAL is the same Gamma(1/2, eps)
     regardless of the number of FP cells).
  R8 prior-edge stress: truth mass slammed against the top N edge => the
     summarize() DIAGNOSTIC must report it (flags fire); we assert the
     REPORTING, not that sampling is pretty.

Run: conda run -n gpdla-hbi python -m pytest tests/test_modelA_rungs.py -v
"""
import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from numpyro import handlers  # noqa: E402
from functools import partial  # noqa: E402

from CDDF_analysis.hbi_mcmc import forward as fwd  # noqa: E402
from CDDF_analysis.hbi_mcmc import model_a as ma  # noqa: E402
from CDDF_analysis.hbi_mcmc.model_a import ModelAConfig, run_model_a  # noqa: E402
from CDDF_analysis.hbi_mcmc.pack import small_test_grid, synthetic_pack  # noqa: E402

# reduced rung grid: 8 N-bins x 5 fine-z (2 coarse) x 2 strata
GRID = dict(
    nhat_edges=np.round(np.arange(19.5, 20.3 + 1e-9, 0.1), 10),
    zf_edges=np.round(np.arange(2.0, 2.5 + 1e-9, 0.1), 10),
    zc_edges=np.array([2.0, 2.2, 2.5]),
    snr_edges=np.array([0., 3., np.inf]),
    n_molly_cells=3,
)
T_TRUE = np.array([0.3, -0.25])


def _cfg(**kw):
    base = dict(num_warmup=200, num_samples=200, num_chains=2,
                target_accept=0.9, seed=0)
    base.update(kw)
    return ModelAConfig(**base)


# --- shared fits (module-scoped: each sampler runs once) ---------------------------

@pytest.fixture(scope="module")
def pack_main():
    # FP on (25%), nonzero transfer truth: serves R4 + R5(x1 arm) + R7a
    return synthetic_pack(0, **GRID, fp_frac=0.25, t_true=T_TRUE)


@pytest.fixture(scope="module")
def fit_main(pack_main):
    return run_model_a(pack_main, _cfg())


@pytest.fixture(scope="module")
def fit_lowcal():
    pk = synthetic_pack(0, **GRID, fp_frac=0.25, t_true=T_TRUE, molly_scale=1 / 16)
    # deliberate finite-calibration stress: the Farr gate is disabled HERE ONLY
    cfg = _cfg(num_warmup=150, num_samples=150, enforce_farr_gate=False)
    return pk, run_model_a(pk, cfg)


@pytest.fixture(scope="module")
def packs_r6():
    # steep slope, zero FP; SAME counts (generated with the skewed kernel)
    pk_skew = synthetic_pack(4, **GRID, f_slope=3.5, f_curv=0.0, fp_frac=0.0)
    pk_diag = synthetic_pack(4, **GRID, f_slope=3.5, f_curv=0.0, fp_frac=0.0,
                             response_mode="diagonal")
    pk_diag = dataclasses.replace(pk_diag, counts=pk_skew.counts)
    return pk_skew, pk_diag


@pytest.fixture(scope="module")
def fit_r6_diag(packs_r6):
    _, pk_diag = packs_r6
    return run_model_a(pk_diag, _cfg(num_warmup=150, num_samples=150,
                                     fp_mode="off", seed=1))


@pytest.fixture(scope="module")
def fit_r6_correct(packs_r6):
    pk_skew, _ = packs_r6
    return run_model_a(pk_skew, _cfg(fp_mode="off", seed=1))


@pytest.fixture(scope="module")
def fit_zerofp_joint(packs_r6):
    pk_skew, _ = packs_r6  # fp_frac=0: fp_counts identically zero
    assert pk_skew.fp_counts.sum() == 0
    return run_model_a(pk_skew, _cfg(num_warmup=150, num_samples=150,
                                     fp_mode="joint", seed=2))


@pytest.fixture(scope="module")
def fit_edge():
    pk = synthetic_pack(8, **GRID, edge_push=3.0, fp_frac=0.15,
                        molly_n_per_cell=40000)
    return pk, run_model_a(pk, _cfg(num_warmup=100, num_samples=100, seed=3))


# --- R4: completeness / coverage ----------------------------------------------------

def test_r4_coverage_and_totals(pack_main, fit_main):
    _, red = fit_main
    ft = pack_main.truth["f_true"]
    f = red["f"]
    q16, q84 = np.percentile(f, [15.865, 84.135], axis=0)
    coverage = float(np.mean((ft >= q16) & (ft <= q84)))
    assert coverage >= 0.55, f"R4 68% coverage {coverage:.2f} < 0.55"
    dN = np.diff(pack_main.ntrue_edges)
    total_true = float((ft * dN[:, None]).sum())
    tt = red["integrated_total"]
    z = (tt.mean() - total_true) / tt.std(ddof=1)
    assert abs(z) <= 2.0, f"R4 integrated total off by {z:.2f} sigma"


def test_r4_farr_gate_reported(fit_main):
    _, red = fit_main
    assert red["farr_ratio"] >= 4.0  # this pack passes the build-time gate


# --- R5: finite calibration ----------------------------------------------------------

def test_r5_posterior_width_grows_with_shrunk_calibration(fit_main, fit_lowcal):
    _, red1 = fit_main
    _pk16, (_mcmc16, red16) = fit_lowcal
    sd1 = float(red1["integrated_total"].std(ddof=1))
    sd16 = float(red16["integrated_total"].std(ddof=1))
    assert sd16 > sd1, (
        f"R5: posterior sd must grow when calibration counts shrink x1/16 "
        f"(sd x1={sd1:.4f}, sd x1/16={sd16:.4f})")


def test_r5_farr_gate_fires_on_shrunk_calibration():
    pk = synthetic_pack(0, **GRID, fp_frac=0.25, t_true=T_TRUE, molly_scale=1 / 16)
    with pytest.raises(RuntimeError, match="Farr N_eff gate FAILED"):
        run_model_a(pk, _cfg())  # gate ON by default


# --- R6: migration ablation -----------------------------------------------------------

def _highN_bias(pack, red):
    ft = pack.truth["f_true"]
    Nc = pack.truth["Nc"]
    hi = Nc >= Nc[0] + 0.5
    f = red["f"]
    fm, fs = f.mean(axis=0), f.std(axis=0, ddof=1)
    z = (fm - ft) / fs
    return float(np.mean(fm[hi, :] > ft[hi, :])), float(np.median(z[hi, :]))


def test_r6_diagonal_kernel_biased_high(packs_r6, fit_r6_diag):
    pk_skew, pk_diag = packs_r6
    _, red = fit_r6_diag
    frac_hi, medz_hi = _highN_bias(pk_skew, red)  # truth = the generating pack's
    assert frac_hi >= 0.85, f"R6 diag: frac(post mean > truth) at high N {frac_hi:.2f}"
    assert medz_hi >= 1.5, f"R6 diag: median high-N bias {medz_hi:.2f} sigma (want >> 0)"


def test_r6_correct_kernel_unbiased(packs_r6, fit_r6_correct):
    pk_skew, _ = packs_r6
    _, red = fit_r6_correct
    frac_hi, medz_hi = _highN_bias(pk_skew, red)
    assert 0.15 <= frac_hi <= 0.85, f"R6 correct: sign test frac {frac_hi:.2f}"
    assert abs(medz_hi) <= 1.0, f"R6 correct: median high-N bias {medz_hi:.2f} sigma"


# --- R7: FP + exposure ------------------------------------------------------------------

def test_r7_t_recovery_within_2sigma(pack_main, fit_main):
    _, red = fit_main
    z = (red["t_mean"] - pack_main.truth["t_true"]) / red["t_sd"]
    assert np.all(np.abs(z) <= 2.0), f"R7 t_K recovery z = {np.round(z, 2)}"


def test_r7_zero_fp_t_posterior_matches_prior(packs_r6, fit_zerofp_joint):
    pk_skew, _ = packs_r6
    _, red = fit_zerofp_joint
    t_sig = np.asarray(pk_skew.t_sigma)
    # KS-ish loose: location near 0, width near the prior width
    assert np.all(np.abs(red["t_mean"]) <= 0.75 * t_sig), red["t_mean"]
    ratio = red["t_sd"] / t_sig
    assert np.all((ratio >= 0.5) & (ratio <= 1.5)), ratio
    # FIX-3c guard: no phantom FP mass materializes from empty loa-0 cells
    assert red["fp_lam_total_mean"] * float(pk_skew.fp_w_sightline_ratio) \
        <= 0.01 * float(pk_skew.counts.sum())


def test_r7_zero_fp_f_unchanged_vs_no_fp_model(fit_r6_correct, fit_zerofp_joint):
    _, red_off = fit_r6_correct
    _, red_joint = fit_zerofp_joint
    a, b = red_off["integrated_total"], red_joint["integrated_total"]
    dz = abs(a.mean() - b.mean()) / np.hypot(a.std(ddof=1), b.std(ddof=1))
    assert dz <= 2.0, f"R7 zero-FP: f shifted by {dz:.2f} combined sigma vs fp-off"
    assert abs(a.mean() - b.mean()) / a.mean() <= 0.1


def test_r7_fp_prior_total_grid_refinement_invariant(pack_main):
    """The single-Jeffreys FP TOTAL prior must not depend on the number of FP
    cells (the FIX-3c rule: per-cell Jeffreys mass would grow with the grid)."""
    pk_fine = synthetic_pack(0, **small_test_grid())  # 10 x 3 = 30 FP cells
    dists = []
    for pk in (pack_main, pk_fine):                   # 8 x 2 = 16 vs 30 cells
        consts = fwd.build_consts(pk)
        model = partial(ma.model_a, fp_mode="joint", fp_eps_rate=1e-6)
        tr = handlers.trace(handlers.seed(model, jax.random.PRNGKey(0))).get_trace(
            consts, jnp.asarray(pk.counts), jnp.asarray(pk.fp_counts))
        site = tr["fp_lam_total"]
        dists.append(site["fn"])
        # the deterministic per-cell field must sum exactly to the total
        lam_fp = np.asarray(tr["lam_fp"]["value"])
        assert lam_fp.shape == (pk.n_c, pk.n_s)
        assert np.isclose(lam_fp.sum(), float(site["value"]), rtol=1e-12)
    for fn in dists:
        assert float(fn.concentration) == 0.5   # the single Jeffreys 1/2, total
        assert float(fn.rate) == 1e-6
    # identical prior distribution on the TOTAL regardless of cell count
    assert float(dists[0].concentration) == float(dists[1].concentration)
    assert float(dists[0].rate) == float(dists[1].rate)


# --- R8: prior-edge stress ----------------------------------------------------------------

def test_r8_edge_stress_diagnostics_fire(fit_edge):
    pk, (mcmc, red) = fit_edge
    d = red["diagnostics"]
    for key in ("flag_r_hat", "flag_ess_bulk", "flag_ess_tail", "flag_divergent"):
        assert key in d
    assert d["flags_fired"], "R8: edge-stress run reported NO convergence flags"
    assert d["policy_pass"] is False
    # the stress construction actually loaded the top edge: the edge bin must
    # tower over the mid-grid bins (a falling power law would sit far below)
    ft = pk.truth["f_true"]
    assert ft[-1, :].min() > 3.0 * ft[2:-2, :].max()


def test_r8_differential_mask_carried(fit_edge):
    pk, (_, red) = fit_edge
    assert red["n_mask_bins"] == 2  # [19.5, 19.7) masked in differential reporting
    assert np.all(np.isnan(red["cddf_masked"][:, :2, :]))
    assert np.all(np.isfinite(red["cddf_masked"][:, 2:, :]))
