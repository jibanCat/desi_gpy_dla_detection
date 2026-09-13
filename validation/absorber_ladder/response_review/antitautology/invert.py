#!/usr/bin/env python
"""invert.py — the deterministic forward fold and its unpenalised Poisson-MLE
inversion (anti-tautology test C).

The EM (multiplicative / Richardson-Lucy) update is the one the absorber-side
forensics already use:
``validation/absorber_diag/analyze_fold.py:535-560`` — the fold is linear in f
and does not mix fine-z cells, so for each z cell the latent f is the solution
of a non-negative Poisson MLE obtainable by

    f <- f * [ A^T (d / (A f)) ] / [ A^T 1 ]

carrying NO population prior.  ``analyze_fold``'s ``em_solve`` is a closure
over that program's own pack-derived arrays (it needs jax + the production
pack loader), so it cannot be imported; the update is reproduced here
verbatim in form and the equality of the two is asserted by
``tests/test_antitautology.py::test_em_matches_analyze_fold_update``.

Forward model (identical to ``analyze_fold.build_A``, with psi_c = 0):

    A[c, k, s, b] = Mg[s, k, c, b] * C[b, K(k), s] * dX[k, s] * dN[b]
    mu[c, k, s]   = sum_b A[c, k, s, b] * f[b, k]

VALIDATION-ONLY.  ENV: gpdla-hbi.
"""
from __future__ import annotations

import numpy as np


def build_A(Mg, C_bKs, dX, dN, kz2K):
    """(C, Kf, S, B) forward operator."""
    Mg = np.asarray(Mg, float)                  # (S, Kf, C, B)
    C_bKs = np.asarray(C_bKs, float)            # (B, KK, S)
    dX = np.asarray(dX, float)                  # (Kf, S)
    dN = np.asarray(dN, float)                  # (B,)
    kz2K = np.asarray(kz2K, int)
    Cbks = C_bKs[:, kz2K, :]                    # (B, Kf, S)
    A = np.einsum("skcb,bks,ks->cksb", Mg, Cbks, dX, optimize=True)
    return A * dN[None, None, None, :]


def fold(A, f_bk):
    """(C, Kf, S) expected counts of a latent f (B, Kf)."""
    return np.einsum("cksb,bk->cks", A, np.asarray(f_bk, float),
                     optimize=True)


def em_solve(data, A, n_iter=20000, f0=None, tol=0.0):
    """Unpenalised non-negative Poisson MLE by EM.

    ``f0`` defaults to a FLAT start (deliberately uninformative: starting at
    the truth, as the forensics do, would hide a conditioning failure)."""
    A = np.asarray(A, float)
    d = np.asarray(data, float)
    B = A.shape[3]
    Kf = A.shape[1]
    col = A.sum(axis=(0, 2))                                # (Kf, B)
    f = np.ones((B, Kf)) if f0 is None else np.array(f0, float)
    f = np.where(col.T > 0, np.maximum(f, 1e-30), 0.0)
    for _ in range(int(n_iter)):
        mu = np.einsum("cksb,bk->cks", A, f, optimize=True)
        r = np.where(mu > 0, d / np.maximum(mu, 1e-300), 0.0)
        upd = np.einsum("cksb,cks->bk", A, r, optimize=True)
        fn = np.where(col.T > 0, f * upd / np.maximum(col.T, 1e-300), f)
        if tol > 0 and np.max(np.abs(fn - f)) <= tol * np.max(np.abs(f) + 1e-30):
            f = fn
            break
        f = fn
    return f


# ===========================================================================
# synthetic f(N) shapes (sealed §6C)
# ===========================================================================
def f_powerlaw(Nc, gamma, N_piv=20.5):
    return 10.0 ** (float(gamma) * (np.asarray(Nc, float) - N_piv))


def f_broken(Nc, g_lo=-1.0, g_hi=-2.5, N_break=21.0, N_piv=20.5):
    Nc = np.asarray(Nc, float)
    lo = g_lo * (Nc - N_piv)
    hi = g_lo * (N_break - N_piv) + g_hi * (Nc - N_break)
    return 10.0 ** np.where(Nc <= N_break, lo, hi)


def f_bump(Nc, gamma=-1.5, N_bump=21.0, amp=1.0, wid=0.15, N_piv=20.5):
    Nc = np.asarray(Nc, float)
    base = 10.0 ** (gamma * (Nc - N_piv))
    return base * (1.0 + amp * np.exp(-0.5 * ((Nc - N_bump) / wid) ** 2))


def synthetic_shapes(Nc):
    s = {f"powerlaw_g{g:g}": f_powerlaw(Nc, g)
         for g in (-1.0, -1.5, -2.0, -2.5)}
    s["broken_-1.0/-2.5@21.0"] = f_broken(Nc)
    s["bump@21.0"] = f_bump(Nc)
    return s


def native_truth_f(ops_path, dN, dX):
    """The calibration mock's OWN latent f(N, k) from the A0 truth census —
    the population a tautological operator would pull towards."""
    z = np.load(ops_path, allow_pickle=True)
    tc = np.asarray(z["truth_counts_bks"], float)           # (B, Kf, S)
    num = tc.sum(axis=2)                                    # (B, Kf)
    den = np.asarray(dN, float)[:, None] * np.asarray(dX, float).sum(
        axis=1)[None, :]
    return np.divide(num, np.where(den > 0, den, 1.0))


def fit_slope(f_b, Nc, lo=20.0, hi=21.5, w=None, keep=None, min_bins=4):
    """Weighted LS slope of log10 f against N over [lo, hi] — the number the
    'pull toward the calibration slope' question is asked of.

    ``keep`` excludes bins the unpenalised MLE has COLLAPSED (driven to ~0).
    With a mismatched operator the unregularised Poisson MLE can zero a sparse
    high-N column outright; log10 of such a bin is meaningless and would
    dominate the fit, so it is dropped and counted instead of being fitted.
    Returns nan when fewer than ``min_bins`` bins survive.
    """
    Nc = np.asarray(Nc, float)
    f_b = np.asarray(f_b, float)
    m = (Nc >= lo) & (Nc <= hi) & (f_b > 0)
    if keep is not None:
        m &= np.asarray(keep, bool)
    if m.sum() < min_bins:
        return float("nan")
    x = Nc[m]
    y = np.log10(f_b[m])
    ww = np.ones(m.sum()) if w is None else np.asarray(w, float)[m]
    X = np.vstack([np.ones_like(x), x]).T
    W = np.diag(ww)
    beta = np.linalg.lstsq(np.sqrt(W) @ X, np.sqrt(ww) * y, rcond=None)[0]
    return float(beta[1])
