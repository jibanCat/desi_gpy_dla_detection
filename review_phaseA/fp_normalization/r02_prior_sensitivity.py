# REVIEW-ONLY (Phase A) — does not alter production behavior.
"""r02 — Is the repair CHOICE (keep lam an intensity, carry ell in the fold)
posterior-distinct from the alternative (rescale lam to a count, drop ell)?

The two options:
  A (repair):  n0 ~ Poisson(ell * lam),  fold uses  w * ell * lam * exp(t) * E
  B (rescale): n0 ~ Poisson(lam'),       fold uses  w * lam' * exp(t) * E
with lam' = ell * lam.  The likelihoods are IDENTICAL functions of mu_FP; the
only question is whether the priors break the equivalence.

EXACT ARGUMENT
--------------
The FP block priors are: fp_lam_total ~ Gamma(1/2, eps), eps = 1e-6, and
pi = softmax(v), v ~ ZeroSumNormal(3.0), with lam[c,s] = total * pi[c,s].
Option B is the pushforward of option A under total' = ell * total (the shape
pi is untouched — ell is a scalar, softmax shares are scale-free, and the
ZeroSumNormal prior on v never sees the total).  So the two posteriors on
mu_FP agree iff the prior on the total is invariant under scaling by ell.

Gamma(1/2, eps) has density  p(x) ∝ x^{-1/2} e^{-eps x}.  Under x' = ell*x the
pushforward is Gamma(1/2, eps/ell) ∝ x'^{-1/2} e^{-(eps/ell) x'}.  The
power-law part x^{-1/2} transforms into itself times a CONSTANT (ell^{-1/2}
from the Jacobian) — constants cancel in the posterior — so the ONLY
non-invariance is the exponential cutoff: e^{-eps x} vs e^{-(eps/ell) x}.
Its effect on the posterior of the total is a tilt of relative size
~ eps * E[total] (option A: eps*total ~ 1e-6 * 6.5 ~ 7e-6; option B:
1e-6 * 89 ~ 9e-5) — i.e. both are >4 orders of magnitude below the 10.6%
Poisson width of the 89-count calibration.  Conjugate closed form (shape
profiled out): total | n0 ~ Gamma(n0 + 1/2, ell + eps) under A and
~ Gamma(n0 + 1/2, 1 + eps) under B, so the folded totals are
  A: mu_FP = w*ell*total   ~ Gamma(n0+1/2, (ell+eps)/(w*ell))
  B: mu_FP = w*total'      ~ Gamma(n0+1/2, (1+eps)/w)
whose rates differ by the factor (ell+eps)/(ell*(1+eps)) = 1 + eps/ell - eps
+ O(eps^2): a relative mean shift of |eps*(1/ell - 1)| ≈ 9.3e-7.  NUTS
geometry is also unchanged: numpyro samples the total through a log
transform, and the two options differ by a constant shift log(ell) in that
coordinate.  CONCLUSION: posterior-identical to ~1e-6 relative; the choice is
genuinely immaterial.  (This also explains how the defect survived: on the
loa-0 SOURCE side ell is pure reparameterization; it is only the data-side
fold that breaks the symmetry, and only the fold was wrong.)

This script verifies the conjugate statement numerically (quadrature, no MCMC)
including the exact eps-induced shift, and demonstrates that the shape prior
term is scale-invariant by construction.
"""
import json
import os

import numpy as np
from scipy import integrate

ELL = 13.589891949531905     # 2LPT-0 pack fp_ell_eff (r01)
W = 165.93215077605322       # 2LPT-0 pack fp_w
N0 = 89.0                    # loa-0 FP counts on the pack support (r01)
EPS = 1e-6                   # fp_eps_rate (model_a.py ModelAConfig)

out = {}


def posterior_moments_mu(option):
    """Exact posterior mean/sd of the folded FP total mu = c_fold * total with
    total | n0 ~ Gamma(n0 + 1/2, rate), by quadrature on the unnormalized
    density (NOT the closed form, so the closed form is checked too)."""
    if option == "A":
        rate, c_fold = ELL + EPS, W * ELL
    else:
        rate, c_fold = 1.0 + EPS, W
    a = N0 + 0.5

    def unnorm(x):
        return x ** (a - 1.0) * np.exp(-rate * x)

    # integration window: +-12 sd around the Gamma mean (a/rate)
    m = a / rate
    sd = np.sqrt(a) / rate
    lo, hi = max(m - 12 * sd, 0.0), m + 12 * sd
    z0, _ = integrate.quad(unnorm, lo, hi, limit=200)
    m1, _ = integrate.quad(lambda x: x * unnorm(x), lo, hi, limit=200)
    m2, _ = integrate.quad(lambda x: x * x * unnorm(x), lo, hi, limit=200)
    mean = m1 / z0
    var = m2 / z0 - mean ** 2
    return c_fold * mean, c_fold * np.sqrt(var)


mA, sA = posterior_moments_mu("A")
mB, sB = posterior_moments_mu("B")
out["mu_fp_posterior"] = dict(
    option_A=dict(mean=mA, sd=sA),
    option_B=dict(mean=mB, sd=sB),
    rel_mean_diff=abs(mA - mB) / mA,
    rel_sd_diff=abs(sA - sB) / sA,
    predicted_rel_shift=float(EPS * abs(1.0 / ELL - 1.0)),
)

# closed-form check: mean = (n0+1/2)*c/rate
out["closed_form"] = dict(
    A=float(W * ELL * (N0 + 0.5) / (ELL + EPS)),
    B=float(W * (N0 + 0.5) / (1.0 + EPS)),
)

# shape-prior scale invariance: the ZeroSumNormal density on v is independent
# of the total by construction (lam = total * softmax(v)); rescaling the total
# never enters the v coordinate. Demonstrate: the multinomial split of the
# likelihood, prod_cs Poisson(ell*tot*pi_cs) = Poisson(ell*tot) *
# Multinomial(n0; pi), factorizes so pi's posterior is total-free.
rng = np.random.default_rng(0)
v = rng.normal(size=8)
v -= v.mean()
pi = np.exp(v) / np.exp(v).sum()
counts = rng.multinomial(int(N0), pi)


def loglik_shape(vv, total, ell):
    p = np.exp(vv - vv.max())
    p = p / p.sum()
    lam = total * p
    return float((counts * np.log(ell * lam) - ell * lam).sum())


# the v-gradient of the log-likelihood must be IDENTICAL for (total, ell) and
# (ell*total, 1) — i.e. the shape coordinate cannot distinguish the options.
def grad_v(vv, total, ell, h=1e-6):
    g = np.zeros_like(vv)
    for i in range(len(vv)):
        e = np.zeros_like(vv)
        e[i] = h
        g[i] = (loglik_shape(vv + e, total, ell)
                - loglik_shape(vv - e, total, ell)) / (2 * h)
    return g


gA = grad_v(v, 6.5, ELL)
gB = grad_v(v, 6.5 * ELL, 1.0)
out["shape_grad_max_abs_diff"] = float(np.max(np.abs(gA - gB)))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "r02_out.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
