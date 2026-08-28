"""Synthetic guards for the 2026-08-28 remaining-request generators (R-033 / R-034). No real-data values."""
import math

import numpy as np

from CDDF_analysis.hbi_mcmc import pc_fixed_denominator as PC
from CDDF_analysis.hbi_mcmc import pyigm_default_cddf as PY


def test_bin_average_of_a_power_law_matches_the_analytic_integral():
    lo = np.array([20.3, 21.5]); hi = np.array([20.5, 21.7])
    s = -2.0
    f = PY.bin_average_per_N(lambda g: s * (g - 20.0) - 21.0, lo, hi)     # log10 f = -21 - 2 (logN - 20)
    for a, b, v in zip(lo, hi, f):
        exact = 10.0 ** -21 * 1e40 * (1.0 / 10.0 ** a - 1.0 / 10.0 ** b) / (10.0 ** b - 10.0 ** a)
        assert math.isclose(v, exact, rel_tol=1e-5)


def test_fixed_denominators_are_exact_on_a_hand_built_sample():
    # truth: 4 absorbers in [20.3,20.5) and 2 in [19.7,19.9), all at SNR 4 (response cell 1)
    truth_nhi = np.array([20.4, 20.35, 20.45, 20.4, 19.8, 19.75]); truth_snr = np.array([4.0] * 6)   # all in response cell 1
    # detections: three TPs of the 20.4 group (one reported below 20.3 -> class miss), one TP of the 19.8 group reported at 19.6
    # (found for 'any', not for 'reported'), one FP at 20.4 without host, one FP at 19.8 with a sub-floor host
    nhat = np.array([20.45, 20.5, 20.2, 19.6, 20.4, 19.8]); ntrue = np.array([20.4, 20.35, 20.45, 19.8, np.nan, np.nan])
    snr = np.array([4.0] * 6); is_tp = np.array([1, 1, 1, 1, 0, 0], bool)
    host = np.array([20.4, 20.35, 20.45, 19.8, np.nan, 19.2])
    t = PC.tabulate(nhat, ntrue, snr, is_tp, host, truth_nhi, truth_snr, PC.RESP_SNR_EDGES)
    i_dla = list(PC.NEDGES).index(20.3); i_sub = list(PC.NEDGES).index(19.7)
    assert t["n_true"][i_dla, 1] == 4 and t["n_found_any"][i_dla, 1] == 3 and math.isclose(t["C_abs_any"][i_dla, 1], 0.75)
    assert t["n_true"][i_sub, 1] == 2 and t["n_found_any"][i_sub, 1] == 1 and t["n_found_reported"][i_sub, 1] == 0
    assert math.isclose(t["C_cls"][i_dla, 1], 2.0 / 3.0)                # one of three TP DLAs reported below 20.3
    assert t["n_det"][i_dla, 1] == 3 and t["n_tp_by_nhat"][i_dla, 1] == 2 and math.isclose(t["P_abs"][i_dla, 1], 2.0 / 3.0)
    assert t["n_subfloor_host"][i_sub, 1] == 1 and math.isclose(t["P_abs_incl_subfloor_host"][i_sub, 1], 1.0)
    conf = t["confusion_tp"][1]
    assert conf.sum() == 4 and conf[0, 0] == 2 and conf[0, 1] == 1 and conf[1, 1] == 1
