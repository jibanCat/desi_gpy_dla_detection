"""Regression tests for the convention-agnostic per-sample normalization in
``calc_cddf.DLACatalogue`` (the PR#7 log-sum-exp vs pre-PR#7 log-mean-exp bug).

The estimator normalizes per-sample posterior weights so ``sum_j exp(log_norm_like_j)
== 1`` per spectrum.  Pre-PR#7, ``log_likelihoods_dla`` was stored as log-MEAN-exp;
PR#7's ``+log(N)`` evidence fix changed it to log-SUM-exp.  The old normalization
``sample_ll - log_likelihoods_dla - log(N)`` only summed to 1 under the MEAN
convention; on a SUM-convention (post-PR#7) run it summed to ``1/N`` -> every
sample dropped below ``p_thresh_sample`` -> ``f(N)=0`` (silent, because the
guarding assert is stripped under ``-O``).

The fix makes the normalization SELF-NORMALIZING (``sample_ll - logsumexp(sample_ll)``):
byte-identical on MEAN-convention data, correct on SUM-convention data.
"""
import os
import sys

import numpy as np
import pytest
from scipy.special import logsumexp

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "fixtures", "cddf"))

from build_synthetic_cddf_fixture import build_synthetic_cddf  # noqa: E402
from CDDF_analysis.calc_cddf import DLACatalogue  # noqa: E402


def _cat(tmp_path, convention):
    fx = build_synthetic_cddf(
        str(tmp_path), n_spec=4, n_samples=256,
        p_dla=(1.0, 1.0, 1.0, 0.0),
        peak_logN=(20.5, 21.0, 21.5, None),
        peak_z=(2.6, 2.8, 3.0, None),
        convention=convention,
    )
    cat = DLACatalogue(
        processed_file=fx["processed_file"], sample_file=fx["sample_file"],
        catalog_file=fx["catalog_file"], sub_dla=0, snr=-2,
        high_nhi_cut_value=21.9,
    )
    return cat, fx


class TestNormalizationSumsToOne:
    @pytest.mark.parametrize("convention", ["mean", "sum"])
    def test_log_norm_like_sums_to_one(self, tmp_path, convention):
        # The whole point: regardless of the stored-marginal convention, the
        # per-sample posterior weights must sum to 1 per active spectrum.
        cat, fx = _cat(tmp_path, convention)
        for spec in cat.log_norm_like_cache:
            s = float(np.sum(np.exp(cat.log_norm_like_cache[spec])))
            assert 0.95 < s < 1.05, f"convention={convention} spec={spec} sum={s}"

    @pytest.mark.parametrize("convention", ["mean", "sum"])
    def test_cache_equals_softmax(self, tmp_path, convention):
        # Self-normalizing form: cache == sample_ll - logsumexp(sample_ll),
        # independent of the stored marginal.
        cat, fx = _cat(tmp_path, convention)
        sh = cat.filehandle["sample_log_likelihoods_dla"]
        for spec in cat.log_norm_like_cache:
            ll = sh[spec, :, 0]
            expected = ll - logsumexp(ll)
            np.testing.assert_allclose(cat.log_norm_like_cache[spec], expected, atol=1e-9)


class TestCddfNonzeroOnSumConvention:
    def test_cddf_nonzero_under_sum_convention(self, tmp_path):
        # The end-to-end symptom: f(N) must NOT be all-zero on a SUM-convention
        # (post-PR#7) run.  Pre-fix this is identically 0.
        cat, fx = _cat(tmp_path, "sum")
        lN, f, f68, f95, xe = cat.column_density_function(
            z_min=2.2, z_max=3.3, lnhi_nbins=8, lnhi_min=20.3, lnhi_max=21.9
        )
        assert np.nansum(f) > 0.0, "f(N) is identically zero on a SUM-convention run"


class TestConventionEquivalence:
    def test_mean_and_sum_give_identical_cache(self, tmp_path):
        # The two conventions carry the SAME information; the self-normalizing
        # estimator must produce an IDENTICAL normalized cache from both.
        (tmp_path / "m").mkdir()
        (tmp_path / "s").mkdir()
        cat_mean, _ = _cat(tmp_path / "m", "mean")
        cat_sum, _ = _cat(tmp_path / "s", "sum")
        for spec in cat_mean.log_norm_like_cache:
            np.testing.assert_allclose(
                cat_mean.log_norm_like_cache[spec],
                cat_sum.log_norm_like_cache[spec],
                atol=1e-9,
            )
