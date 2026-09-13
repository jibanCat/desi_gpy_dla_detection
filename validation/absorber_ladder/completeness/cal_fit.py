"""cal_fit.py — pure binomial-GLM / cross-validation helpers for the FIXED
completeness calibration objects (VALIDATION-ONLY).

Nothing here touches ``CDDF_analysis/``, no sampler, no jax.  Every routine is
plain numpy so the unit tests (``tests/test_completeness_variants.py``) run in
either environment in seconds.

Conventions (deliberately the pack's own, so C0 reproduces the frozen surface
BIT-FOR-BIT):

* the completeness point surface is a **detection probability**
  ``C = n_det / n_tot`` on truth systems, with the Jeffreys-consistent logit

      eta_hat = log((n_det + 1/2) / (n_tot - n_det + 1/2))

  which is ``CDDF_analysis/hbi_mcmc/forward.eta_hat_sigma_hat`` verbatim;
* binomial GLMs are fitted by IRLS (Newton on the canonical logit link) with a
  tiny ridge on the NON-intercept columns only, purely for conditioning;
* model choice is by **held-out** binomial log-loss on sightline halves; the
  comparison statistic is the PAIRED per-trial difference, whose standard error
  is computed from the held-out Bernoulli variance (``logloss_diff_se``).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "expit", "logit", "eta_hat_jeffreys", "poly_design", "design_per_stratum",
    "design_2d_tensor", "design_2d_additive", "design_additive_z",
    "irls_binomial", "predict_eta",
    "binom_logloss", "binom_deviance", "logloss_diff_se", "cv_two_fold",
    "parity_halves", "weighted_ratio_residual",
]


# --------------------------------------------------------------------------
# link functions
# --------------------------------------------------------------------------
def expit(x):
    """Numerically safe logistic; never returns exactly 0 or 1 for finite x."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit(p):
    p = np.asarray(p, float)
    return np.log(p) - np.log1p(-p)


def eta_hat_jeffreys(n_det, n_tot):
    """``forward.eta_hat_sigma_hat`` verbatim: (eta_hat, sigma_hat).

    eta_hat = log((d + 1/2)/(t - d + 1/2)); sig = sqrt(1/(d+1/2) + 1/(t-d+1/2)).
    """
    d = np.asarray(n_det, float)
    t = np.asarray(n_tot, float)
    eta = np.log((d + 0.5) / (t - d + 0.5))
    sig = np.sqrt(1.0 / (d + 0.5) + 1.0 / (t - d + 0.5))
    return eta, sig


# --------------------------------------------------------------------------
# design matrices
# --------------------------------------------------------------------------
def poly_design(x, deg):
    """(n, deg+1) raw-power design ``[1, x, x^2, ...]``."""
    x = np.asarray(x, float).ravel()
    return np.stack([x ** j for j in range(int(deg) + 1)], axis=1)


def design_per_stratum(x_b, n_s, deg):
    """Block-diagonal design for ONE independent polynomial per stratum.

    Rows are ordered ``(s, b)`` C-style, i.e. row = s * B + b.  Returns
    ``(n_s * B, n_s * (deg + 1))``.  Coefficient block ``s`` occupies columns
    ``s*(deg+1) : (s+1)*(deg+1)``.
    """
    x_b = np.asarray(x_b, float).ravel()
    B = x_b.size
    P = poly_design(x_b, deg)                       # (B, deg+1)
    k = P.shape[1]
    X = np.zeros((n_s * B, n_s * k))
    for s in range(int(n_s)):
        X[s * B:(s + 1) * B, s * k:(s + 1) * k] = P
    return X


def design_2d_tensor(x_b, u_s, deg_n, deg_u):
    """Tensor-product design ``sum_{j<=deg_n, i<=deg_u} beta_ji x^j u^i``.

    Rows ordered ``(s, b)``.  ``(deg_n+1)*(deg_u+1)`` columns.  This is the
    compact 2-D smooth: the intercept and every N-slope vary smoothly with the
    continuous S/N covariate ``u``.
    """
    x_b = np.asarray(x_b, float).ravel()
    u_s = np.asarray(u_s, float).ravel()
    B, S = x_b.size, u_s.size
    cols = []
    for j in range(int(deg_n) + 1):
        for i in range(int(deg_u) + 1):
            cols.append(np.outer(u_s ** i, x_b ** j).ravel())
    return np.stack(cols, axis=1)


def design_2d_additive(x_b, u_s, deg_n, deg_u):
    """Additive-in-logit design ``a(x) + m(u)`` (one shared intercept).

    Columns: ``[1, x, ..., x^deg_n, u, ..., u^deg_u]`` →
    ``deg_n + deg_u + 1`` coefficients.  No interaction: the N-shape is common
    to every stratum and S/N only shifts the logit.
    """
    x_b = np.asarray(x_b, float).ravel()
    u_s = np.asarray(u_s, float).ravel()
    B, S = x_b.size, u_s.size
    XX = np.tile(x_b, S)
    UU = np.repeat(u_s, B)
    cols = [np.ones(B * S)]
    cols += [XX ** j for j in range(1, int(deg_n) + 1)]
    cols += [UU ** i for i in range(1, int(deg_u) + 1)]
    return np.stack(cols, axis=1)


def design_additive_z(x_b, u_s, n_K, deg_n, deg_u):
    """Additive design on the (s, K, b) layout with coarse-z logit OFFSETS.

    Rows ordered ``(s, K, b)`` C-style.  Columns
    ``[1, x..x^deg_n, u..u^deg_u, 1{K=1}, ..., 1{K=n_K-1}]`` →
    ``1 + deg_n + deg_u + (n_K - 1)`` coefficients (K=0 is the reference).

    This is the smallest member of the additive family that can carry a z
    trend; it is used as an EXPLORATORY diagnostic (the coarse-z residual the
    z-pooled members cannot represent), not as an adopted variant.
    """
    x_b = np.asarray(x_b, float).ravel()
    u_s = np.asarray(u_s, float).ravel()
    B, S, K = x_b.size, u_s.size, int(n_K)
    s_i = np.repeat(np.arange(S), K * B)
    k_i = np.tile(np.repeat(np.arange(K), B), S)
    b_i = np.tile(np.arange(B), S * K)
    xx, uu = x_b[b_i], u_s[s_i]
    cols = [np.ones(xx.size)]
    cols += [xx ** j for j in range(1, int(deg_n) + 1)]
    cols += [uu ** i for i in range(1, int(deg_u) + 1)]
    cols += [(k_i == kk).astype(float) for kk in range(1, K)]
    return np.stack(cols, axis=1)


# --------------------------------------------------------------------------
# binomial IRLS
# --------------------------------------------------------------------------
def irls_binomial(X, n_det, n_tot, ridge=1e-6, max_iter=200, tol=1e-11,
                  penalise_intercept=False, beta0=None):
    """Newton/IRLS fit of ``n_det ~ Binomial(n_tot, sigmoid(X beta))``.

    ``ridge`` adds ``ridge * I`` to the Hessian on the non-intercept columns
    (column 0 is treated as the intercept unless ``penalise_intercept``); it is
    a CONDITIONING term only — the report states its size and the coefficient
    change it causes.

    Returns dict(beta, eta, p, iters, converged, deviance, cov, hessian_rcond).
    """
    X = np.asarray(X, float)
    d = np.asarray(n_det, float).ravel()
    t = np.asarray(n_tot, float).ravel()
    if X.shape[0] != d.size or d.size != t.size:
        raise ValueError("irls_binomial: X rows must match n_det/n_tot length")
    if np.any(d < 0) or np.any(t < 0) or np.any(d > t):
        raise ValueError("irls_binomial: need 0 <= n_det <= n_tot")
    keep = t > 0
    Xk, dk, tk = X[keep], d[keep], t[keep]
    P = X.shape[1]
    pen = np.full(P, float(ridge))
    if not penalise_intercept and P > 0:
        pen[0] = 0.0
    R = np.diag(pen)
    beta = np.zeros(P) if beta0 is None else np.asarray(beta0, float).copy()
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        eta = Xk @ beta
        p = expit(eta)
        w = tk * p * (1.0 - p)
        w = np.maximum(w, 1e-12)
        grad = Xk.T @ (dk - tk * p) - R @ beta
        H = (Xk.T * w) @ Xk + R
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:                       # pragma: no cover
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        # damped Newton: halve until the penalised log-likelihood improves
        ll0 = _pen_ll(Xk, dk, tk, beta, pen)
        lam = 1.0
        for _ in range(40):
            cand = beta + lam * step
            if _pen_ll(Xk, dk, tk, cand, pen) >= ll0 - 1e-14:
                break
            lam *= 0.5
        beta = beta + lam * step
        if np.max(np.abs(lam * step)) < tol:
            converged = True
            break
    eta_all = X @ beta
    p_all = expit(eta_all)
    etak = Xk @ beta
    pk = expit(etak)
    wk = np.maximum(tk * pk * (1.0 - pk), 1e-12)
    H = (Xk.T * wk) @ Xk + R
    try:
        cov = np.linalg.inv(H)
        rc = float(np.linalg.cond(H))
    except np.linalg.LinAlgError:                           # pragma: no cover
        cov, rc = np.full((P, P), np.nan), np.inf
    return dict(beta=beta, eta=eta_all, p=p_all, iters=int(it),
                converged=bool(converged),
                deviance=float(binom_deviance(p_all, d, t)),
                cov=cov, hessian_cond=rc, n_coef=int(P),
                n_cells_used=int(keep.sum()), ridge=float(ridge))


def _pen_ll(X, d, t, beta, pen):
    eta = X @ beta
    # log-likelihood, computed stably from the logit
    ll = np.sum(d * eta - t * np.logaddexp(0.0, eta))
    return ll - 0.5 * np.sum(pen * beta ** 2)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
_EPS = 1e-12


def binom_logloss(p, n_det, n_tot, total=True):
    """Negative binomial log-likelihood WITHOUT the combinatorial constant.

    ``-(d log p + (t-d) log(1-p))``; this is the held-out predictive score, in
    nats, and is directly comparable between variants because the dropped
    constant depends only on the data.
    """
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    d = np.asarray(n_det, float)
    t = np.asarray(n_tot, float)
    ll = d * np.log(p) + (t - d) * np.log1p(-p)
    return float(-ll.sum()) if total else -ll


def binom_deviance(p, n_det, n_tot):
    """``2 * (loglik_saturated - loglik_model)``, cells with t=0 skipped."""
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    d = np.asarray(n_det, float)
    t = np.asarray(n_tot, float)
    ok = t > 0
    q = np.clip(np.where(ok, d / np.where(ok, t, 1.0), 0.5), _EPS, 1.0 - _EPS)
    sat = d * np.log(q) + (t - d) * np.log1p(-q)
    mod = d * np.log(p) + (t - d) * np.log1p(-p)
    return float(2.0 * np.sum((sat - mod)[ok]))


def logloss_diff_se(p_a, p_b, n_det, n_tot):
    """SE of the PAIRED held-out log-loss difference ``L(p_b) - L(p_a)``.

    Per held-out trial the difference is a Bernoulli variable taking
    ``-log(p_b/p_a)`` with probability q and ``-log((1-p_b)/(1-p_a))`` with
    probability ``1-q``; q is estimated by the held-out cell rate.  Trials are
    independent across cells, so the SE is the root of the summed variances.
    """
    p_a = np.clip(np.asarray(p_a, float), _EPS, 1.0 - _EPS)
    p_b = np.clip(np.asarray(p_b, float), _EPS, 1.0 - _EPS)
    d = np.asarray(n_det, float)
    t = np.asarray(n_tot, float)
    ok = t > 0
    q = np.where(ok, d / np.where(ok, t, 1.0), 0.0)
    delta = (np.log(p_b) - np.log(p_a)) - (np.log1p(-p_b) - np.log1p(-p_a))
    var = np.where(ok, t * q * (1.0 - q) * delta ** 2, 0.0)
    return float(np.sqrt(var.sum()))


def parity_halves(tid):
    """Sightline halves by TARGETID parity: (even_mask, odd_mask)."""
    t = np.asarray(tid, np.int64)
    even = (t % 2) == 0
    return even, ~even


def cv_two_fold(fit_fn, predict_fn, det_A, tot_A, det_B, tot_B, mask=None):
    """Two-fold sightline-half CV.

    ``fit_fn(det, tot) -> state`` and ``predict_fn(state) -> p`` (same cell
    layout as the count arrays).  Fold 1 fits on half A and scores on half B;
    fold 2 fits on B, scores on A.  ``mask`` restricts the SCORED cells (the
    fits always use every cell they are given).

    Returns dict with per-fold and pooled held-out log-loss, the predictions
    (for paired SE computations) and the fitted states.
    """
    det_A = np.asarray(det_A, float)
    tot_A = np.asarray(tot_A, float)
    det_B = np.asarray(det_B, float)
    tot_B = np.asarray(tot_B, float)
    m = np.ones(det_A.shape, bool) if mask is None else np.asarray(mask, bool)
    st_A = fit_fn(det_A, tot_A)
    st_B = fit_fn(det_B, tot_B)
    p_on_B = np.asarray(predict_fn(st_A), float)   # fitted on A, scored on B
    p_on_A = np.asarray(predict_fn(st_B), float)
    L_B = binom_logloss(p_on_B[m], det_B[m], tot_B[m])
    L_A = binom_logloss(p_on_A[m], det_A[m], tot_A[m])
    n_trials = float(tot_A[m].sum() + tot_B[m].sum())
    return dict(fold_fitA_scoreB=L_B, fold_fitB_scoreA=L_A,
                logloss=L_A + L_B,
                logloss_per_trial=(L_A + L_B) / max(n_trials, 1.0),
                n_heldout_trials=n_trials,
                p_on_A=p_on_A, p_on_B=p_on_B, state_A=st_A, state_B=st_B)


def weighted_ratio_residual(pred, obs_det, obs_tot, axis=None):
    """Truth-weighted ``pred/obs - 1`` with obs = det/tot, reduced over ``axis``.

    The weight is ``obs_tot`` (the number of truth trials), so the reduction is
    the ratio of EXPECTED to OBSERVED detections — the quantity the fold sees.
    """
    pred = np.asarray(pred, float)
    d = np.asarray(obs_det, float)
    t = np.asarray(obs_tot, float)
    num = (pred * t).sum(axis=axis)
    den = d.sum(axis=axis)
    out = np.full(np.shape(num), np.nan)
    ok = np.asarray(den) > 0
    np.divide(num, den, out=out, where=ok)
    return np.where(ok, out - 1.0, np.nan)
