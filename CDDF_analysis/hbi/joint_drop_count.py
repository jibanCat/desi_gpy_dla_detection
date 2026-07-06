"""joint_drop_count.py — the JOINT Lyman-limit-drop + FP-corrected-counting HBI (task #31, Phase A).

ONE parametric CDDF f(N,X;theta) inferred from TWO likelihood terms that share theta:

  counting  n_i ~ Poisson( C_i * INT_i f dN dX + F_i )     # trustworthy only at N>=19.5 (shape)
  drop      tau_hat_j ~ Normal( tau_model_j(theta), sig_j ) # purity-immune (LLS normalization)

The marginal on l(X)[17.2,19.5) is the deliverable band. This is an ADDITIVE CONSUMER of the FROZEN
estimator CDDF_analysis/hbi/cddf_catalog_hbi.py — it imports its parametric f-evaluator and the
pure-over-theta counting neg-log-posterior and adds the drop term. NOTHING in the frozen file changes.

Design + review: notes/2026-07-05_hbi_joint_drop_design.md + 2026-07-05_joint_hbi_review.md.
Key review constraints honoured here:
  * broken f(N) with the self-shielding knee -> use family 'bplcut' (or 'bspbody'), which extends the
    CDDF BELOW the counting floor so the drop opacity is explicit (not a degenerate single slope);
  * ONE shared cosmology Omega_m=cfg.Omega_m (0.279) for both dX (counts) and the drop integral —
    NOT the 0.30 hardcoded in examples/lyc_mfp_inversion.py;
  * the drop is one integral dominated by sub-LLS below 17.2 -> a marginalized Rudie/PW14 sub-LLS
    shape prior is REQUIRED for the band to be data- not prior-driven (wired via cfg's family priors);
  * the drop enters as the CURVE tau_eff,LL(z912), not one scalar.

The drop MEASUREMENT (tau_hat_j(z912), sig_j) is NOT in the frozen estimator; it is supplied from the
LyC D-series composite (examples/lyc_inject_closure.py + lyc_mfp_inversion.py) on the (mirror) mock.
"""
from __future__ import annotations
import numpy as np

from CDDF_analysis.hbi.cddf_catalog_hbi import (
    v3x_f_of_N,
    v3x_grad_f_wrt_theta,
    v3x_neg_log_posterior,
    v3x_param_bounds,
    v3x_default_theta0,
    v3x_reduce,
)
from CDDF_analysis.cddf_mock import path_length_int

SIGMA_912 = 6.35e-18   # cm^2 (Verner+1996) — SAME as CDDF_analysis.lyc + the mirror injection
LN10 = np.log(10.0)


# ---------------------------------------------------------------------------
# Drop model: tau_eff,LL(z912) from f(N,z;theta) — reuses the counting f-evaluator
# ---------------------------------------------------------------------------
def drop_tau_model(
    theta,
    family: str,
    cfg,
    z912_arr,
    z_qso: float,
    sigma912: float = SIGMA_912,
    beta: float = 3.0,          # sigma(nu) ~ nu^-beta: 3.0 for the mock injection; 2.75 Worseck-effective (real IGM)
    logN_grid=None,
    logN_lo: float = 15.0,      # integral floor: 17.2 for the HCD-only mock; ~15 (+ sub-LLS prior) for real IGM
    n_zprime: int = 80,
):
    """Model effective Lyman-continuum optical depth at each observed z912, for a QSO at z_qso.

    An absorber at z' (z912 <= z' <= z_qso) absorbs the z912-limit photon (obs lambda = 912(1+z912))
    with a redshifted cross-section sigma912 * ((1+z912)/(1+z'))^beta (the photon is above that
    absorber's own limit for z'>z912). f is per dN per dX, so the path measure is dX = path_length_int
    (Omega_m=cfg.Omega_m):

        tau(z912) = INT_{z912}^{z_qso} (dX/dz') dz'  INT dlogN [N ln10 f(N,z';theta)]
                    * ( 1 - exp( -N sigma912 ((1+z912)/(1+z'))^beta ) )
    """
    om = float(getattr(cfg, "Omega_m", 0.279))
    z912_arr = np.atleast_1d(np.asarray(z912_arr, float))
    if logN_grid is None:
        # cover the drop kernel (peaks ~17.2) up through the DLA tail. logN_lo=17.2 matches the
        # HCD-only mock's injected support; lower (+ a sub-LLS prior) for the real IGM's diffuse sub-LLS.
        logN_grid = np.linspace(float(logN_lo), 22.5, 160)
    logN_grid = np.asarray(logN_grid, float)
    N = 10.0 ** logN_grid                                   # (nN,)
    tau = np.zeros_like(z912_arr)
    for i, z912 in enumerate(z912_arr):
        if z912 >= z_qso:
            continue
        zp = np.linspace(float(z912), float(z_qso), n_zprime)   # (nzp,)
        dXdz = path_length_int(zp, om)                          # (1+zp)^2/E(zp), (nzp,)
        fN = v3x_f_of_N(logN_grid[:, None], zp[None, :], theta, family, cfg)   # (nN, nzp), per dN per dX
        sig = sigma912 * ((1.0 + z912) / (1.0 + zp)) ** beta                    # (nzp,)
        opac = 1.0 - np.exp(-(N[:, None] * sig[None, :]))                        # (nN, nzp)
        # INT dlogN (N ln10) f opac  ->  (nzp,)   [manual trapezoid; np.trapz removed in newer numpy]
        integrand = fN * (N[:, None] * LN10) * opac                             # (nN, nzp)
        integ_N = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(logN_grid)[:, None], axis=0)
        g = integ_N * dXdz                                                      # (nzp,)
        tau[i] = float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(zp)))
    return tau


# ---------------------------------------------------------------------------
# Joint objective: counting neg-log-posterior (count+prior) + drop neg-log-likelihood
# ---------------------------------------------------------------------------
class DropData:
    """The measured drop curve: tau_hat at each z912, per-z sigma, and the QSO redshift(s)."""
    def __init__(self, z912, tau_hat, sigma, z_qso, sigma912=SIGMA_912, beta=3.0, logN_lo=15.0):
        self.z912 = np.asarray(z912, float)
        self.tau_hat = np.asarray(tau_hat, float)
        self.sigma = np.asarray(sigma, float)
        self.z_qso = float(z_qso)                 # representative / effective z_qso for the composite
        self.sigma912 = float(sigma912)
        self.beta = float(beta)
        self.logN_lo = float(logN_lo)             # 17.2 for the HCD-only mock; ~15 (+sub-LLS prior) real


def drop_neg_loglike(theta, family, cfg, drop: DropData):
    tau_m = drop_tau_model(theta, family, cfg, drop.z912, drop.z_qso,
                           sigma912=drop.sigma912, beta=drop.beta, logN_lo=drop.logN_lo)
    r = (tau_m - drop.tau_hat) / drop.sigma
    return 0.5 * float(np.sum(r * r))


def drop_tau_and_grad(theta, family, cfg, z912_arr, z_qso, sigma912=SIGMA_912, beta=3.0,
                      logN_grid=None, n_zprime=60):
    """Return (tau_model (nz912,), dtau/dtheta (n_theta, nz912)) — the ANALYTIC theta-gradient of the
    drop model, reusing v3x_grad_f_wrt_theta so the drop rides on the same f-gradient as the counts."""
    om = float(getattr(cfg, "Omega_m", 0.279))
    z912_arr = np.atleast_1d(np.asarray(z912_arr, float))
    theta = np.asarray(theta, float)
    if logN_grid is None:
        logN_grid = np.linspace(15.0, 22.5, 120)
    logN_grid = np.asarray(logN_grid, float)
    N = 10.0 ** logN_grid
    dlogN = np.diff(logN_grid)
    nth = theta.size
    tau = np.zeros(z912_arr.size)
    grad = np.zeros((nth, z912_arr.size))
    for i, z912 in enumerate(z912_arr):
        if z912 >= z_qso:
            continue
        zp = np.linspace(float(z912), float(z_qso), n_zprime)
        dXdz = path_length_int(zp, om); dzp = np.diff(zp)
        fN = v3x_f_of_N(logN_grid[:, None], zp[None, :], theta, family, cfg)           # (nN,nzp)
        gN = np.asarray(v3x_grad_f_wrt_theta(logN_grid[:, None], zp[None, :], theta, family, cfg))  # (nth,nN,nzp)
        sig = sigma912 * ((1.0 + z912) / (1.0 + zp)) ** beta
        w = (N[:, None] * LN10) * (1.0 - np.exp(-(N[:, None] * sig[None, :])))          # (nN,nzp)
        # tau
        iN = np.sum(0.5 * ((fN * w)[1:] + (fN * w)[:-1]) * dlogN[:, None], axis=0)      # (nzp,)
        g = iN * dXdz
        tau[i] = float(np.sum(0.5 * (g[1:] + g[:-1]) * dzp))
        # dtau/dtheta (vectorized over theta)
        gw = gN * w[None, :, :]                                                          # (nth,nN,nzp)
        iNk = np.sum(0.5 * (gw[:, 1:] + gw[:, :-1]) * dlogN[None, :, None], axis=1)      # (nth,nzp)
        gk = iNk * dXdz[None, :]
        grad[:, i] = np.sum(0.5 * (gk[:, 1:] + gk[:, :-1]) * dzp[None, :], axis=1)
    return tau, grad


def joint_neg_logP(theta, fwd, family, cfg, drop: DropData | None, count_weight_key="op_weights"):
    """Joint counting+drop objective. fwd = the v3x forward dict (A_full, M_full, lam_fp, mu_fp, fine,
    cat_op). Returns a SCALAR (numerical gradient in the driver; theta is low-dim)."""
    obj_w = None
    cat_op = fwd.get("cat_op") if isinstance(fwd, dict) else None
    if cat_op is not None and count_weight_key in cat_op:
        obj_w = cat_op[count_weight_key]
    nc = v3x_neg_log_posterior(
        theta, fwd["A_full"], fwd["M_full"], fwd["lam_fp"], fwd["mu_fp"], fwd["fine"],
        family, cfg, obj_weights=obj_w, with_grad=False,
    )
    nc = float(nc)
    if drop is not None:
        nc = nc + drop_neg_loglike(theta, family, cfg, drop)
    return nc


# ---------------------------------------------------------------------------
# Driver: multistart L-BFGS-B (mirrors v3x_fit_map, joint objective, numerical gradient)
# ---------------------------------------------------------------------------
class SubLLSPrior:
    """Marginalized sub-LLS shape prior (review F1 fix). Anchors log10 f(N;theta) at a few N<17.2
    points to literature/truth values (Rudie+2013 on real data; mock truth for validation), so the
    single drop integral no longer has to fix BOTH the sub-LLS and the LLS band — it pins the LLS."""
    def __init__(self, logN_anchors, log10f_target, sigma_dex):
        self.logN = np.asarray(logN_anchors, float)
        self.log10f = np.asarray(log10f_target, float)
        self.sigma = np.asarray(sigma_dex, float)

    def neg_loglike(self, theta, family, cfg):
        f = np.array([max(v3x_f_of_N(np.array([x]), 3.0, theta, family, cfg)[0], 1e-300)
                      for x in self.logN])
        r = (np.log10(f) - self.log10f) / self.sigma
        return 0.5 * float(np.sum(r * r))


def fit_joint(fwd, family, cfg, drop: DropData | None, sub_lls: "SubLLSPrior | None" = None,
              n_restart: int = 8, seed: int = 0):
    """Maximize the joint posterior over theta. drop=None reproduces the frozen counting-only fit
    (a sanity check that the consumer reduces to the frozen estimator)."""
    from scipy.optimize import minimize
    bounds = v3x_param_bounds(family, cfg)
    theta0 = np.asarray(v3x_default_theta0(family, cfg), float)
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] if b[0] is not None else theta0[k] - 3 for k, b in enumerate(bounds)])
    hi = np.array([b[1] if b[1] is not None else theta0[k] + 3 for k, b in enumerate(bounds)])

    cat_op = fwd.get("cat_op") if isinstance(fwd, dict) else None
    obj_w = cat_op.get("op_weights") if isinstance(cat_op, dict) else None

    def _drop_nd(th):
        tau = drop_tau_model(th, family, cfg, drop.z912, drop.z_qso,
                             sigma912=drop.sigma912, beta=drop.beta, logN_lo=drop.logN_lo)
        r = (tau - drop.tau_hat) / drop.sigma
        return 0.5 * float(np.sum(r * r))

    def obj(th):
        nc, gc = v3x_neg_log_posterior(
            th, fwd["A_full"], fwd["M_full"], fwd["lam_fp"], fwd["mu_fp"], fwd["fine"],
            family, cfg, obj_weights=obj_w, with_grad=True)
        val = float(nc); grad = np.asarray(gc, float).copy()
        h = 1e-4
        if drop is not None:
            nd0 = _drop_nd(th)                       # analytic counting grad + FD drop grad (drop is cheap)
            val += nd0
            for k in range(th.size):
                thp = th.copy(); thp[k] += h
                grad[k] += (_drop_nd(thp) - nd0) / h
        if sub_lls is not None:
            ns0 = sub_lls.neg_loglike(th, family, cfg)
            val += ns0
            for k in range(th.size):
                thp = th.copy(); thp[k] += h
                grad[k] += (sub_lls.neg_loglike(thp, family, cfg) - ns0) / h
        return val, grad

    best = None
    starts = [theta0] + [np.clip(theta0 + rng.normal(scale=0.3, size=theta0.size), lo, hi)
                         for _ in range(max(0, n_restart - 1))]
    for th0 in starts:
        res = minimize(obj, th0, method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": 400, "ftol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    return {"theta_map": best.x, "neg_logP": float(best.fun), "success": bool(best.success),
            "family": family}


def reduce_theta(theta, fwd, family, cfg):
    """Turn a fitted theta into dN/dX, Omega, l(X) bands via the FROZEN reducer."""
    return v3x_reduce(cfg, theta, fwd["fine"], family, fwd.get("M_meta"))
