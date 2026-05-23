"""Tests for the gated DLA clustering-prior hook in ``dla_gp.py``.

The #1 guarantee: with ``pair_prior_mode="off"`` (the default) the inference
path is byte-identical to current behaviour. These tests cover:
  - the ``_logmeanexp_nan`` module helper,
  - the constructor signatures/defaults/validation on ``DLAGP``/``DLAGPMAT``,
  - the ``_clustering_log_factor`` per-MODEL evidence factor in isolation (the
    round-2 referee guarantee: Δ_k = log E_post[ρ] − log E_unif[ρ], with the
    per-sample column and the SIR resampler NEVER touched), without needing a
    full GP model,
  - an OPTIONAL end-to-end null-invariance/parity test on a saved fixture
    (skipped if the fixture is absent so the suite stays runnable).
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from gpy_dla_detection.dla_clustering import DLAClusteringPrior


# --------------------------------------------------------------------------- #
# 1. _logmeanexp_nan helper
# --------------------------------------------------------------------------- #
def test_logmeanexp_nan_helper():
    from gpy_dla_detection.dla_gp import _logmeanexp_nan

    x = np.array([0.0, np.log(3.0), np.nan])
    # mean of exp over the non-nan entries = (1 + 3) / 2 = 2 -> log 2
    assert _logmeanexp_nan(x) == pytest.approx(np.log(2.0))


def test_logmeanexp_nan_all_nan_returns_zero():
    from gpy_dla_detection.dla_gp import _logmeanexp_nan

    assert _logmeanexp_nan(np.array([np.nan, np.nan])) == 0.0


# --------------------------------------------------------------------------- #
# 2. Constructor signatures / defaults / validation
# --------------------------------------------------------------------------- #
def test_dlagp_signature_defaults():
    from gpy_dla_detection.dla_gp import DLAGP

    sig = inspect.signature(DLAGP.__init__)
    assert sig.parameters["pair_prior_mode"].default == "off"
    assert sig.parameters["dla_bias"].default == 2.0


def test_dlagpmat_signature_defaults():
    from gpy_dla_detection.dla_gp import DLAGPMAT

    sig = inspect.signature(DLAGPMAT.__init__)
    assert sig.parameters["pair_prior_mode"].default == "off"
    assert sig.parameters["dla_bias"].default == 2.0


def test_pair_prior_mode_validation():
    from gpy_dla_detection.dla_gp import DLAGP

    # Use __new__ + a hand-rolled call into the validation branch via a tiny
    # subclass-free path: construct enough state by calling the validation
    # logic directly through DLAGP.__init__ is heavy (needs full GP args).
    # Instead exercise the documented invariant: an invalid mode must raise.
    # We replicate the exact guard the constructor uses.
    with pytest.raises(ValueError):
        DLAGP._validate_pair_prior_mode("bogus")


# --------------------------------------------------------------------------- #
# 3. _clustering_log_factor — the per-MODEL evidence-factor guarantee
# --------------------------------------------------------------------------- #
def _stub(pair_prior_mode="off", b_dla=2.0):
    """A minimal stand-in carrying just the attributes the helper reads.

    We bind the real (unbound) ``DLAGP._clustering_log_factor`` to this stub so
    we exercise the production method without building a full GP model.
    """
    from gpy_dla_detection.dla_gp import DLAGP

    obj = DLAGP.__new__(DLAGP)
    obj.pair_prior_mode = pair_prior_mode
    obj.dla_bias = float(b_dla)
    obj.pair_prior = (
        DLAClusteringPrior(b_dla=b_dla) if pair_prior_mode == "clustering" else None
    )
    return obj


def _synthetic_inputs():
    """Build a synthetic (sample_probabilities, all_z_dlas, ind) triple.

    Sample columns 0..4. all_z_dlas is (k=2, N=5). Sample 0 is a CLOSE pair
    (small Δz), sample 1 is a FAR pair, samples 2,3 are intermediate, sample 4
    is masked by ``ind`` (too close -> already NaN'd in the per-sample column).
    By default the CLOSE pair carries the HIGH probability (posterior prefers
    close pairs -> Δ should be positive)."""
    # bare exp(slk - max) probabilities; sample 1 is NaN in the k=2 column,
    # sample 4 is min_z_separation-masked (NaN by ind).
    sample_probabilities = np.array([1.0, np.nan, 0.10, 0.05, 0.50])
    all_z_dlas = np.array(
        [
            [2.500, 2.500, 2.500, 2.500, 2.500],
            [2.502, 2.900, 2.520, 2.560, 2.5005],  # close, far, mid, mid, very-close
        ]
    )
    ind = np.array([False, False, False, False, True])  # sample 4 too close
    return sample_probabilities, all_z_dlas, ind


def _hand_delta(obj, num_dlas, all_z_dlas, sample_probabilities, ind, valid_mask=None):
    """Recompute Δ_k = log E_post[ρ] − log E_unif[ρ] by hand."""
    rho = np.exp(obj.pair_prior.log_rho(all_z_dlas))
    p = np.array(sample_probabilities, dtype=float)
    sel = np.isfinite(p) & np.isfinite(rho) & (~ind)
    if valid_mask is not None:
        sel = sel & valid_mask
    E_post = np.sum(p[sel] * rho[sel]) / np.sum(p[sel])
    z_min, z_max = float(np.nanmin(all_z_dlas)), float(np.nanmax(all_z_dlas))
    E_unif = obj.pair_prior.prior_mean_rho(num_dlas + 1, z_min, z_max)
    return float(np.log(E_post) - np.log(E_unif))


def test_factor_off_is_zero():
    """OFF path: Δ = 0.0 exactly (no clustering correction)."""
    obj = _stub("off")
    p, all_z_dlas, ind = _synthetic_inputs()
    assert obj._clustering_log_factor(1, all_z_dlas, p, ind) == 0.0


def test_factor_num_dlas_0_is_zero():
    """num_dlas=0 (1-DLA model, k=1, no pairs): Δ = 0.0 exactly."""
    obj = _stub("clustering")
    p, all_z_dlas, ind = _synthetic_inputs()
    assert obj._clustering_log_factor(0, all_z_dlas, p, ind) == 0.0


def test_factor_num_dlas_1_matches_hand_computation():
    """num_dlas=1: Δ == log(Σpρ/Σp) − log(prior_mean_rho(2, zmin, zmax)).

    Also asserts NaN-probability and ind-masked samples are EXCLUDED from the
    posterior average (samples 1 and 4 must not contribute)."""
    obj = _stub("clustering")
    p, all_z_dlas, ind = _synthetic_inputs()

    expected = _hand_delta(obj, 1, all_z_dlas, p, ind)
    got = obj._clustering_log_factor(1, all_z_dlas, p, ind)
    assert got == pytest.approx(expected)

    # Independent confirmation that masked/NaN samples are excluded: zeroing
    # sample 1's (already-NaN) prob and sample 4's prob must not change Δ.
    p2 = p.copy()
    p2[1] = np.nan  # already NaN
    p2[4] = 999.0   # ind-masked -> excluded regardless of value
    got2 = obj._clustering_log_factor(1, all_z_dlas, p2, ind)
    assert got2 == pytest.approx(expected)


def test_factor_close_pairs_high_prob_gives_larger_delta():
    """Δ is LARGER (more positive) when the high-probability samples are the
    CLOSE pairs than when they are the FAR pairs. The posterior-prefers-close
    case is the one that boosts the k≥2 evidence; the far case can drive Δ
    below the prior mean (and even negative)."""
    obj = _stub("clustering")
    _, all_z_dlas, ind = _synthetic_inputs()

    # Case A: close pair (sample 0) carries the high probability.
    p_close = np.array([1.0, 0.02, 0.05, 0.05, np.nan])
    delta_close = obj._clustering_log_factor(1, all_z_dlas, p_close, ind)

    # Case B: far pair (sample 1) carries the high probability.
    p_far = np.array([0.02, 1.0, 0.05, 0.05, np.nan])
    delta_far = obj._clustering_log_factor(1, all_z_dlas, p_far, ind)

    assert delta_close > delta_far


def test_factor_far_pair_posterior_can_be_negative():
    """If the posterior overwhelmingly prefers the far pair, E_post[ρ] falls
    below E_unif[ρ] and Δ goes negative (the evidence is correctly penalized)."""
    obj = _stub("clustering")
    _, all_z_dlas, ind = _synthetic_inputs()
    # Far pair (sample 1, Δz large -> ρ≈1) dominates; close pairs near-zero prob.
    p_far = np.array([1e-6, 1.0, 1e-6, 1e-6, np.nan])
    delta = obj._clustering_log_factor(1, all_z_dlas, p_far, ind)
    assert delta < 0.0


def test_factor_respects_valid_mask():
    """When a FILTER=1 region-A ``valid_mask`` is supplied, only the masked
    samples enter E_post (the posterior ≈ region A)."""
    obj = _stub("clustering")
    p, all_z_dlas, ind = _synthetic_inputs()
    valid_mask = np.array([True, False, True, False, False])

    expected = _hand_delta(obj, 1, all_z_dlas, p, ind, valid_mask=valid_mask)
    got = obj._clustering_log_factor(1, all_z_dlas, p, ind, valid_mask=valid_mask)
    assert got == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 4. OPTIONAL end-to-end null-invariance / parity on a saved fixture (RC-3)
# --------------------------------------------------------------------------- #
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "london0_single_dla.npz"
_REPO = Path(__file__).resolve().parent.parent

# Production-model + sample/prior catalog paths (readable from the worktree).
_MODEL = (
    "/scratch/cavestru_root/cavestru0/mfho/phase2_desi/"
    "2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5"
)
_DR = "/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection"
_DLA_SAMPLES = f"{_DR}/data/dr12q/processed/pw_samples_a3_172_225_100000.mat"
_CATALOG = f"{_DR}/data/dr12q/processed/catalog.mat"
_LOS = f"{_DR}/data/dla_catalogs/dr9q_concordance/processed/los_catalog"
_DLA_CAT = f"{_DR}/data/dla_catalogs/dr9q_concordance/processed/dla_catalog"

_HAVE_E2E = all(os.path.exists(p) for p in (_FIXTURE, _MODEL, _DLA_SAMPLES,
                                            _CATALOG, _LOS, _DLA_CAT))


def _build_dla_gp(pair_prior_mode):
    """Build a production DLAGPMAT on the fixture spectrum (mode off|clustering).

    Mirrors the model construction in run_bayes_select.DLAHolder.process_qso
    (params/preset matched to the smoke runner's y3 preset, num_dla_samples
    kept small for a fast test)."""
    import sys

    sys.path.insert(0, str(_REPO))
    from gpy_dla_detection.set_parameters import Parameters
    from gpy_dla_detection.model_priors import PriorCatalog
    from gpy_dla_detection.dla_samples import DLASamplesMAT
    from gpy_dla_detection.dla_gp import DLAGPMAT

    npz = np.load(_FIXTURE, allow_pickle=True)
    obs_wave = npz["rest_wavelengths"]  # fixture stores the OBSERVED grid
    flux = npz["flux"]
    noise_variance = npz["noise_variance"]
    pixel_mask = npz["pixel_mask"]
    z_qso = float(npz["z_qso"])

    params = Parameters(
        loading_min_lambda=910.0, loading_max_lambda=1550.0,
        normalization_min_lambda=1425.0, normalization_max_lambda=1475.0,
        min_lambda=911.75, max_lambda=1216.75, dlambda=0.15, k=30,
        max_noise_variance=9.0, num_lines=3, max_z_cut=3000.0,
        min_z_cut=3000.0, num_forest_lines=3,
        # MUST match the offset_samples count in the .mat file (100000) — the
        # estimator sizes its arrays from num_dla_samples while sample_z_dlas
        # uses the full file.
        num_dla_samples=100000,
    )
    prior = PriorCatalog(params, _CATALOG, _LOS, _DLA_CAT)
    dla_samples = DLASamplesMAT(params, prior, _DLA_SAMPLES)

    dla_gp = DLAGPMAT(
        params, prior, dla_samples,
        min_z_separation=3000.0, learned_file=_MODEL, broadening=True,
        prev_tau_0=0.00246, prev_beta=3.62,
        pair_prior_mode=pair_prior_mode, dla_bias=2.0,
    )
    # set_data expects REST-frame wavelengths (mirrors process_qso, which calls
    # params.emitted_wavelengths(observed, z_qso) before set_data).
    rest_wave = params.emitted_wavelengths(obs_wave, z_qso)
    dla_gp.set_data(rest_wave, flux, noise_variance, pixel_mask, z_qso,
                    build_model=True)
    return dla_gp


def _run_evidences(pair_prior_mode, seed, max_dlas=2, return_gp=False):
    np.random.seed(seed)
    dla_gp = _build_dla_gp(pair_prior_mode)
    ev = dla_gp.parallel_log_model_evidences(
        max_dlas, max_workers=4, batch_size=25000, filter_low_likelihood=False
    )
    if return_gp:
        return ev, dla_gp
    return ev


@pytest.mark.skipif(not _HAVE_E2E, reason=f"needs fixture+model+catalogs: {_FIXTURE}")
def test_end_to_end_off_is_deterministic():
    """OFF path is deterministic with a fixed seed (sanity for the comparisons)."""
    pytest.importorskip("fitsio")
    a = _run_evidences("off", seed=7)
    b = _run_evidences("off", seed=7)
    assert np.allclose(a, b, equal_nan=True)


@pytest.mark.skipif(not _HAVE_E2E, reason=f"needs fixture+model+catalogs: {_FIXTURE}")
def test_end_to_end_z1_unchanged_clustering_vs_off():
    """1-DLA evidence (index 0) is identical ON vs OFF: Δ_k=0 for num_dlas=0
    (k=1, no pairs). The clustering factor is a no-op for the 1-DLA model, so
    this must hold EXACTLY. With the evidence-only mechanism the per-sample
    column and the resampler are never touched, so the seed-driven k>=2 SIR
    path is identical ON vs OFF and cannot perturb index [0]."""
    pytest.importorskip("fitsio")
    off = _run_evidences("off", seed=7)
    clu = _run_evidences("clustering", seed=7)
    assert off[0] == pytest.approx(clu[0], abs=1e-9)


@pytest.mark.skipif(not _HAVE_E2E, reason=f"needs fixture+model+catalogs: {_FIXTURE}")
def test_end_to_end_per_sample_column_and_resampler_byte_identical():
    """The strongest invariant of the evidence-only rework: the per-sample
    ``sample_log_likelihoods`` array (which feeds the SIR resampler) is
    BYTE-IDENTICAL between clustering=ON and OFF, because the factor is applied
    ONLY to the finalized per-MODEL evidence and never mutates the column. This
    is what guarantees the resample weights W are byte-identical to production."""
    pytest.importorskip("fitsio")
    _, gp_off = _run_evidences("off", seed=7, return_gp=True)
    _, gp_clu = _run_evidences("clustering", seed=7, return_gp=True)
    assert np.array_equal(
        gp_off.sample_log_likelihoods, gp_clu.sample_log_likelihoods, equal_nan=True
    )
    assert np.array_equal(gp_off.base_sample_inds, gp_clu.base_sample_inds)
