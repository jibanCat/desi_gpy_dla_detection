import numpy as np
from validation.fp_ladder.j8_certify import rank_rhat_ess


def test_rank_rhat_near_one_for_iid_chains():
    rng = np.random.default_rng(0); x = rng.normal(size=(4, 1000))
    rhat, bulk, tail = rank_rhat_ess(x)
    assert abs(rhat - 1) < 0.02 and bulk > 2500 and tail > 800


def test_rank_rhat_detects_separated_chains():
    rng = np.random.default_rng(1); x = rng.normal(size=(4, 1000)); x[0] += 5.0
    rhat, bulk, tail = rank_rhat_ess(x)
    assert rhat > 1.5 and bulk < 200


def test_ess_drops_for_autocorrelated_chains():
    rng = np.random.default_rng(2); n = 2000; x = np.zeros((4, n))
    for c in range(4):
        e = rng.normal(size=n)
        for i in range(1, n): x[c, i] = 0.95 * x[c, i - 1] + e[i]
    rhat, bulk, tail = rank_rhat_ess(x)
    assert bulk < 0.2 * 4 * n
