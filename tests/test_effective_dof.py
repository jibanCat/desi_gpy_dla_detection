import numpy as np
from validation.fp_ladder.effective_dof import whiten, p_eff_block


def test_prior_draws_give_zero_effective_dof():
    rng = np.random.default_rng(0)
    n = 20000
    for name, x in (("eps_z", rng.normal(size=(n, 5, 3))),
                    ("psi_c", rng.normal(size=(n, 4)) * np.array([0.1, 0.5, 1.0, 2.0])),
                    ("sigma_N", np.abs(rng.normal(size=(n,)) * 0.5)),
                    ("theta_level", rng.normal(size=(n,)) * 4.0)):
        u = whiten(name, x, sigma_hat=np.array([0.1, 0.5, 1.0, 2.0]))
        pe, m, n2, n4 = p_eff_block(u)
        assert abs(pe) < 0.05 * m + 0.05, (name, pe)
        assert n2 == 0 and n4 == 0


def test_pinned_posterior_gives_full_effective_dof():
    rng = np.random.default_rng(1)
    u = whiten("eps_z", rng.normal(size=(5000, 6)) * 0.01)
    pe, m, n2, n4 = p_eff_block(u)
    assert m == 6 and abs(pe - 6.0) < 0.01 and n2 == 6 and n4 == 6


def test_halfnormal_whitening_is_monotone_and_pit_exact():
    x = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
    u = whiten("sigma_z", x)
    assert np.all(np.diff(u) > 0)
    # HalfNormal(0.5) median is 0.5*sqrt(2)*erfinv(0.5) = 0.3372 -> u = 0
    assert abs(whiten("sigma_z", np.array([0.5 * np.sqrt(2) * 0.4769362762044699]))[0]) < 1e-6
