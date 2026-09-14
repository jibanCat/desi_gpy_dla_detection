#!/usr/bin/env python
"""evaluate_completeness.py -- STANDALONE evaluator for the DESI GP-DLA
Paper-1 fitted completeness (detection-probability) surface C(N_HI, S/N).

SHIPPED IN THE ZENODO RELEASE.  It imports numpy and nothing else: no project
code, no configuration, no data files other than the released
``completeness_model.npz`` (or the six coefficients typed in by hand).

Model (variant ``C1nsadd``, 6 coefficients, additive in the logit)
------------------------------------------------------------------
    x     = log10 N_HI  -  N0                      (N0 = 20.0)
    u_raw = log10(S/N)  -  U0                      (U0 = 0.6852307073614691)
    logit C = b0 + b1 x + b2 x^2 + b3 x^3 + b4 u + b5 u^2
    C = 1 / (1 + exp(-logit C))

so the N dependence is a cubic and the S/N dependence a quadratic, with NO
interaction: the shape in N is common to every S/N stratum and S/N only shifts
the logit.  ``u`` is a log10 S/N offset from the pivot U0, which is itself the
mean of the calibrated strata log-medians.

PRODUCTION CLAMP RULE (predeclared; PI ruling 2026-09-13d §6)
------------------------------------------------------------
The surface is calibrated only where truth systems exist, i.e. between the
stratum S/N medians 2.4383 (lowest live stratum) and 10.6831 (highest live
stratum).  Extrapolating the quadratic above the top calibrated support is not
licensed, so production evaluation uses

    log10(S/N)  ->  min( log10(S/N), log10(10.6830584) = 1.0286956025701406 )

i.e. every sightline better than the top calibrated stratum is given that
stratum's completeness.  The clamp is INERT below the cap.  The lower end is
NOT clamped: the production support already applies an S/N >= 2 cut, and the
lowest calibrated stratum median is 2.4383; evaluations below that are flagged
``in_domain = False`` and are an extrapolation the user must own.

Usage
-----
    import numpy as np, evaluate_completeness as ec
    m = np.load("completeness_model.npz")
    C = ec.evaluate_completeness(20.3, 4.0, m["coef"], float(m["N0"]))
    C, ok = ec.evaluate_completeness(logN, snr, m["coef"], float(m["N0"]),
                                     return_domain=True)
"""
from __future__ import annotations

import numpy as np

# --- frozen model constants (also carried in completeness_model.npz) -------
# NOTE the S/N covariate is the stratum MEDIAN OF log10(S/N) (not log10 of the
# median S/N); the two differ in the 4th decimal.  The clamp and the pivot are
# therefore defined in log10 S/N, and the linear-S/N values below are derived.
N0_DEFAULT = 20.0
U0_DEFAULT = 0.6852307073614691          # log10 S/N pivot (mean of live strata)
LOG10_SNR_CLAMP_HI = 1.0286956025701406  # top calibrated stratum log10-median
LOG10_SNR_DOMAIN_LO = 0.3870901951901383  # lowest calibrated stratum log10-med
SNR_CLAMP_HI = 10.0 ** LOG10_SNR_CLAMP_HI
SNR_DOMAIN_LO = 10.0 ** LOG10_SNR_DOMAIN_LO
LOGN_DOMAIN = (19.0, 22.4)               # calibrated truth-N support

__all__ = ["evaluate_completeness", "completeness_logit", "expit"]


def expit(x):
    """Overflow-safe logistic."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def completeness_logit(logN, snr=None, coef=None, N0=N0_DEFAULT, U0=U0_DEFAULT,
                       clamp=True, log10_snr_clamp_hi=LOG10_SNR_CLAMP_HI,
                       log10_snr=None):
    """The linear predictor ``b0 + b1 x + b2 x^2 + b3 x^3 + b4 u + b5 u^2``.

    Give EITHER ``snr`` (linear) or ``log10_snr``.  Passing ``log10_snr``
    reproduces the calibration table exactly, because the fitted covariate is
    the stratum median of log10(S/N).
    """
    coef = np.asarray(coef, dtype=float).ravel()
    if coef.size != 6:
        raise ValueError("C1nsadd takes exactly 6 coefficients, got %d"
                         % coef.size)
    logN = np.asarray(logN, dtype=float)
    if (snr is None) == (log10_snr is None):
        raise ValueError("give exactly one of snr= or log10_snr=")
    if log10_snr is None:
        snr = np.asarray(snr, dtype=float)
        if np.any(snr <= 0):
            raise ValueError("S/N must be positive (the model is in log10 S/N)")
        lsnr = np.log10(snr)
    else:
        lsnr = np.asarray(log10_snr, dtype=float)
    x = logN - float(N0)
    if clamp:
        lsnr = np.minimum(lsnr, float(log10_snr_clamp_hi))
    u = lsnr - float(U0)
    return (coef[0] + coef[1] * x + coef[2] * x ** 2 + coef[3] * x ** 3
            + coef[4] * u + coef[5] * u ** 2)


def evaluate_completeness(logN, snr=None, coef=None, N0=N0_DEFAULT, clamp=True,
                          U0=U0_DEFAULT,
                          log10_snr_clamp_hi=LOG10_SNR_CLAMP_HI,
                          log10_snr=None, return_domain=False):
    """Detection probability C(N_HI, S/N); see the module docstring.

    Parameters
    ----------
    logN, snr : array_like -- broadcast together; ``logN`` is log10 N_HI in
        cm^-2 and ``snr`` the linear (NOT log) sightline S/N.
    coef : (6,) array -- ``[b0, b1, b2, b3, b4, b5]``.
    N0 : float -- the N pivot (20.0 for the released fit).
    clamp : bool -- apply the predeclared production clamp at the top of the
        calibrated S/N support.  Inert below the cap.
    return_domain : bool -- also return a boolean "inside the calibrated
        domain" mask (before clamping).
    """
    eta = completeness_logit(logN, snr=snr, coef=coef, N0=N0, U0=U0,
                             clamp=clamp,
                             log10_snr_clamp_hi=log10_snr_clamp_hi,
                             log10_snr=log10_snr)
    C = expit(eta)
    if not return_domain:
        return C
    logN_a = np.asarray(logN, dtype=float)
    lsnr_a = (np.log10(np.asarray(snr, dtype=float)) if log10_snr is None
              else np.asarray(log10_snr, dtype=float))
    ok = ((logN_a >= LOGN_DOMAIN[0]) & (logN_a <= LOGN_DOMAIN[1])
          & (lsnr_a >= LOG10_SNR_DOMAIN_LO)
          & ((lsnr_a <= LOG10_SNR_CLAMP_HI) | bool(clamp)))
    return C, np.asarray(ok, dtype=bool)


if __name__ == "__main__":                                   # pragma: no cover
    import sys
    model = np.load(sys.argv[1] if len(sys.argv) > 1
                    else "completeness_model.npz")
    co = model["coef"]
    print("# logN  S/N  C")
    for n in (19.7, 20.0, 20.3, 20.6, 21.0):
        for s in (2.5, 3.5, 5.0, 7.0, 10.0, 25.0):
            print("%6.2f %6.1f %10.6f"
                  % (n, s, float(evaluate_completeness(n, s, co,
                                                       float(model["N0"])))))
