"""h2_cgap_inference: the committed producer of the C_gap bracket-uniform posterior."""
import json
import numpy as np
import pytest
from CDDF_analysis.hbi import h2_cgap_inference as H


def _canon(k_lo, n_lo, k_hi, n_hi):
    return {"detection_strata": [{"stratum": H.LO_STRATUM, "k": k_lo, "n": n_lo, "detection_C": k_lo / n_lo},
                                 {"stratum": H.HI_STRATUM, "k": k_hi, "n": n_hi, "detection_C": k_hi / n_hi},
                                 {"stratum": "cell:B_t0", "k": None, "n": None, "detection_C": None}]}


def _record(k_lo, n_lo, k_hi, n_hi, grid, dndx, post=(0.4, 0.5, 0.6)):
    return {"h2_inputs": {"C_lo": f"Beta({k_lo + 0.5},{n_lo - k_lo + 0.5}) ...", "C_hi": f"Beta({k_hi + 0.5},{n_hi - k_hi + 0.5}) ..."},
            "response_map": {"C_grid": grid, "dndx_ge20_3": dndx},
            "posterior": {"C_gap_p16_50_84": list(post), "dndx_ge20_3_p16_50_84": [0.11, 0.108, 0.105]}}


def test_bracket_uniform_between_sharp_endpoints():
    # with enormous counts the endpoints are sharp, and C_gap is uniform between them
    c = H.cgap_draws(300000, 1000000, 600000, 1000000, n=100000, seed=1)
    assert abs(c.mean() - 0.45) < 2e-3 and abs(c.min() - 0.30) < 2e-3 and abs(c.max() - 0.60) < 2e-3
    assert abs(np.percentile(c, 50) - 0.45) < 3e-3


def test_jeffreys_posterior_mean():
    c = H.cgap_draws(63, 175, 49, 77, n=400000, seed=2)
    m_lo, m_hi = 63.5 / 176.0, 49.5 / 78.0
    assert abs(c.mean() - 0.5 * (m_lo + m_hi)) < 2e-3


def test_infer_reads_strata_and_interpolates_map():
    canon = _canon(63, 175, 49, 77)
    rec = _record(63, 175, 49, 77, [0.3, 0.5, 0.9], [0.12, 0.10, 0.08])
    r = H.infer(canon, rec, n=50000, seed=3)
    assert r["C_lo"] == {"stratum": H.LO_STRATUM, "k": 63, "n": 175} and r["C_hi"]["n"] == 77
    q = r["C_gap_p16_50_84"]
    assert 0.38 < q[0] < 0.43 and 0.47 < q[1] < 0.52 and 0.57 < q[2] < 0.62
    d = r["dndx_ge20_3_p16_50_84"]
    assert d[0] < d[1] < d[2] and 0.095 < d[1] < 0.105        # monotone decreasing map at C ~ 0.5


def test_refuses_counts_that_disagree_with_the_record():
    canon = _canon(64, 175, 49, 77)      # one count off
    rec = _record(63, 175, 49, 77, [0.3, 0.9], [0.12, 0.08])
    with pytest.raises(AssertionError):
        H.infer(canon, rec, n=1000, seed=4)
