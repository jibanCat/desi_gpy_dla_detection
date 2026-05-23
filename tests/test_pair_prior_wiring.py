"""Tests for the gated DLA clustering-prior hook in ``dla_gp.py``.

The #1 guarantee: with ``pair_prior_mode="off"`` (the default) the inference
path is byte-identical to current behaviour. These tests cover:
  - the ``_logmeanexp_nan`` module helper,
  - the constructor signatures/defaults/validation on ``DLAGP``/``DLAGPMAT``,
  - the ``_apply_clustering_prior`` injection helper in isolation (the core
    RC-1/RC-2/RC-3 guarantees), without needing a full GP model,
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
# 3. _apply_clustering_prior — the core injection guarantee
# --------------------------------------------------------------------------- #
def _stub(pair_prior_mode="off", b_dla=2.0):
    """A minimal stand-in carrying just the attributes the helper reads.

    We bind the real (unbound) ``DLAGP._apply_clustering_prior`` to this stub
    so we exercise the production method without building a full GP model.
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
    """Build a synthetic (slk, all_z_dlas, ind) triple.

    Sample columns 0..4. all_z_dlas is (k=2, N=5). Pair 0 is a CLOSE pair
    (small Δz), pair 1 is a FAR pair, pairs 2,3 are intermediate, pair 4 is
    masked by ``ind`` (too close -> already NaN'd in slk)."""
    slk = np.array(
        [
            [-10.0, -5.0, -3.0],
            [-11.0, -6.0, -3.5],
            [-12.0, -7.0, -4.0],
            [-13.0, -8.0, -4.5],
            [-14.0, np.nan, -5.0],  # this sample NaN in the k=2 column
        ]
    )
    all_z_dlas = np.array(
        [
            [2.500, 2.500, 2.500, 2.500, 2.500],
            [2.502, 2.900, 2.520, 2.560, 2.5005],  # close, far, mid, mid, very-close
        ]
    )
    ind = np.array([False, False, False, False, True])  # sample 4 too close
    return slk, all_z_dlas, ind


def test_apply_off_is_no_op():
    """OFF path: the column is byte-identical (incl. NaN positions)."""
    obj = _stub("off")
    slk, all_z_dlas, ind = _synthetic_inputs()
    before = slk[:, 1].copy()
    obj._apply_clustering_prior(slk, num_dlas=1, all_z_dlas=all_z_dlas, ind=ind)
    assert np.array_equal(slk[:, 1], before, equal_nan=True)


def test_apply_clustering_num_dlas_0_is_no_op():
    """num_dlas=0 (1-DLA model, Z_1=1, no pairs): column untouched."""
    obj = _stub("clustering")
    slk, all_z_dlas, ind = _synthetic_inputs()
    before = slk[:, 0].copy()
    obj._apply_clustering_prior(slk, num_dlas=0, all_z_dlas=all_z_dlas, ind=ind)
    assert np.array_equal(slk[:, 0], before, equal_nan=True)


def test_apply_clustering_num_dlas_1_matches_hand_computation():
    """num_dlas=1: column == original + log_rho - logmeanexp(valid log_rho);
    NaN entries stay NaN; masked (ind) entries stay NaN."""
    from gpy_dla_detection.dla_gp import _logmeanexp_nan

    obj = _stub("clustering")
    slk, all_z_dlas, ind = _synthetic_inputs()
    original = slk[:, 1].copy()

    # hand-compute the expected weight
    log_rho = obj.pair_prior.log_rho(all_z_dlas)
    log_rho_exp = log_rho.copy()
    log_rho_exp[ind] = np.nan
    log_rho_exp[np.isnan(original)] = np.nan
    log_Zk = _logmeanexp_nan(log_rho_exp)
    expected = original + log_rho_exp - log_Zk  # NaN where masked / NaN

    obj._apply_clustering_prior(slk, num_dlas=1, all_z_dlas=all_z_dlas, ind=ind)

    # valid (finite) entries must match the hand computation
    valid = np.isfinite(expected)
    assert np.allclose(slk[valid, 1], expected[valid])
    # NaN entries (sample 1's original NaN, and the ind-masked sample 4) stay NaN
    assert np.isnan(slk[~valid, 1]).all()


def test_apply_clustering_close_pair_gets_larger_bump_self_normalized():
    """The prior favours close pairs (larger post-injection bump) while it is
    self-normalized (RC-1): the per-sample weights ρ_i/Z_k average to 1 over
    the valid samples (equivalently, the log-shift mean is <=0 by Jensen, but
    the LINEAR weight mean is exactly 1 — that is the Z_k normalization)."""
    obj = _stub("clustering")
    slk, all_z_dlas, ind = _synthetic_inputs()
    original = slk[:, 1].copy()

    obj._apply_clustering_prior(slk, num_dlas=1, all_z_dlas=all_z_dlas, ind=ind)
    shift = slk[:, 1] - original  # = log_rho - log_Zk; NaN where masked / NaN

    # sample 0 is the close pair; sample 1 is the far pair -> close gets the
    # larger bump (the prior upweights the physically more probable close pair)
    assert shift[0] > shift[1]

    # self-normalization: the linear weights ρ_i/Z_k average to exactly 1 over
    # the valid samples. This is what keeps the SIR resample (which runs on the
    # bare likelihood) unbiased and Z_k a proper per-model normalizer.
    valid = np.isfinite(shift)
    assert np.mean(np.exp(shift[valid])) == pytest.approx(1.0, abs=1e-12)


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


def _run_evidences(pair_prior_mode, seed, max_dlas=2):
    np.random.seed(seed)
    dla_gp = _build_dla_gp(pair_prior_mode)
    return dla_gp.parallel_log_model_evidences(
        max_dlas, max_workers=4, batch_size=25000, filter_low_likelihood=False
    )


@pytest.mark.skipif(not _HAVE_E2E, reason=f"needs fixture+model+catalogs: {_FIXTURE}")
def test_end_to_end_off_is_deterministic():
    """OFF path is deterministic with a fixed seed (sanity for the comparisons)."""
    pytest.importorskip("fitsio")
    a = _run_evidences("off", seed=7)
    b = _run_evidences("off", seed=7)
    assert np.allclose(a, b, equal_nan=True)


@pytest.mark.skipif(not _HAVE_E2E, reason=f"needs fixture+model+catalogs: {_FIXTURE}")
def test_end_to_end_z1_unchanged_clustering_vs_off():
    """1-DLA evidence (index 0) is identical ON vs OFF: Z_1=1, no pairs at k=1
    (RC-3). The clustering hook is a no-op for num_dlas=0, so this must hold
    EXACTLY (the seed only affects k>=2 SIR resampling, which doesn't touch [0])."""
    pytest.importorskip("fitsio")
    off = _run_evidences("off", seed=7)
    clu = _run_evidences("clustering", seed=7)
    assert off[0] == pytest.approx(clu[0], abs=1e-9)
