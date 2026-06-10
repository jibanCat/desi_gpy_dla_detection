"""Regression test for ``calc_cddf.get_poisson_binomial_pdf`` under numpy 2.x.

The per-bin probability container ``pp`` is a LIST of per-sightline arrays of
(generally) different lengths.  The function concatenates them internally
(``np.concatenate(pp)``).  Its empty-check used ``np.size(pp)``, which under
numpy 2.x raises ``ValueError`` (inhomogeneous ragged sequence) the moment a bin
has >= 2 arrays of different length — exactly the real-data case (synthetic
fixtures with <= 1 array per bin masked it).  The fix uses ``len(pp)``.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))

from CDDF_analysis.calc_cddf import get_poisson_binomial_pdf  # noqa: E402


class TestRaggedProbsList:
    def test_multiple_different_length_arrays(self):
        # A bin fed by 3 sightlines contributing 2, 1, 3 large-p samples.
        pp = [
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([0.3], dtype=np.float64),
            np.array([0.15, 0.25, 0.05], dtype=np.float64),
        ]
        pdf = get_poisson_binomial_pdf(pp)
        # PB pdf over 6 Bernoulli trials -> length 7, sums to 1.
        assert pdf.shape[0] == 7
        assert np.isclose(np.sum(pdf), 1.0, atol=1e-9)
        assert np.all(pdf >= -1e-12)

    def test_equals_concatenated_single_array(self):
        pp = [
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([0.3], dtype=np.float64),
            np.array([0.15, 0.25, 0.05], dtype=np.float64),
        ]
        flat = [np.concatenate(pp)]  # one array, same probabilities
        np.testing.assert_allclose(
            get_poisson_binomial_pdf(pp), get_poisson_binomial_pdf(flat), atol=1e-12
        )

    def test_empty_list_returns_unit_mass(self):
        assert np.allclose(get_poisson_binomial_pdf([]), np.ones(1))

    def test_list_of_empty_arrays(self):
        pp = [np.array([], dtype=np.float64), np.array([], dtype=np.float64)]
        assert np.allclose(get_poisson_binomial_pdf(pp), np.ones(1))
