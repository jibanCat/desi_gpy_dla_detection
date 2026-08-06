"""forward.py — the differentiable Model A forward fold (pure jnp) + numpy oracle.

The expected-count fold (spec section 2, per calibration context):

    mu[c,k,s] = dX[k,s] * sum_b K[c<-b](psi_k_delta; s, kz_to_K[k])
                          * C[b->cell,s](psi_c) * g[b,k]
                          * exp(theta_pop[b,k]) * dN_b
              + w * ell_eff * (1 - eta_c[c]) * exp(t[kz_to_K[k]]) * lam_fp[c,s] * E[k,s]

The ``ell_eff`` factor in the FP term is NOT decoration (repaired 2026-08-05),
and ``(1 - eta_c)`` is the host-occlusion survival of the product's own
definition (restored 2026-08-06, PI ruling 8): a forest FP can only occur in
un-occluded forest, so the production-volume expectation carries
``(1 - eta_band)`` per observed band (eta_DLA == 0 forced by the product;
eta_subdla = 0.00576). The loa-0 calibration side carries NO eta — loa-0 is
HCD-free, nothing occludes.
``lam_fp`` is defined by the loa-0 calibration likelihood

    fp_counts[c,s] ~ Poisson(fp_ell_eff * lam_fp[c,s])        (model_a.py)

so ``lam_fp`` is an intensity PER UNIT of the loa-0 exposure ``fp_ell_eff``,
not a count.  The production-volume FP expectation the loa-0 product defines
(build_loa0_fp_product.py:34-39) is

    mu_FP = (N_prod / N_sl_loa0) * N_FP_loa0 * (1 - eta_bar)
          = fp_w * fp_ell_eff * (1 - eta_bar) * lam_fp
            (since lam_fp = N_FP_loa0/ell_eff; the survival factor does NOT
            cancel into the identity below — it multiplies the fold only)

and ``fp_w * fp_ell_eff == N_sl_loa0`` exactly, because the extractor builds
``fp_w = N_prod/N_sl_loa0`` and ``fp_ell_eff = N_sl_loa0^2/N_prod``.  Until
2026-08-05 the fold omitted ``fp_ell_eff`` here and in ``fold_mu_reference``,
under-normalising the whole FP term by exactly that factor (MEASURED on the
adopted 2LPT-0 pack: 1086.6871844096897 folded against 14767.961419068737
required, ratio 13.589891949531907 == fp_ell_eff).  ``fp_ell_eff`` is inert in
the loa-0 SOURCE route -- Gamma(a, 1/ell)*ell is Gamma(a, 1) -- which is why
it survived; it does not cancel here, where it is a live Poisson exposure.

Two implementations of the SAME expression:

* ``fold_mu``            — pure jnp, fully vectorized (NO python loops over data
                           cells), differentiable, called inside the jitted
                           NumPyro model on every draw.
* ``fold_mu_reference``  — plain numpy, written INDEPENDENTLY (explicit loops
                           over data cells, scipy special functions, its own
                           digitize/eta-hat/polynomial code; no helpers shared
                           with the jnp path). The in-module oracle: tests
                           require agreement at rtol 1e-10 at random parameter
                           points.

Response kernel K (fail-closed forward object; NO kappa anywhere):
per response cell (s_resp, z_resp) the pack carries polynomial coefficient
surfaces (LOWEST order first, covariate u = N_true - resp_N_ref, the SAME
reference the coefficients were FIT at — carried in the pack; REQUIRED, no
midpoint fallback). The surfaces are MOMENTS, exactly the committed
``ForwardResponseModel`` semantics (znz_kernel.py; conventions pinned by the
2026-07-11 legacy characterization, findings F1-F4 in
tests/test_modelA_vs_legacy.py):

    mean(b)  = N_b + poly(resp_mu_coef, u_b)          (up-bias E[x-hat - N])
    sd(b)    = clip(poly(resp_sig_coef, u_b), resp_sig_floor)     (width, dex)
    skew(b)  = clip(poly(resp_skew_coef, u_b), +-0.995*SKEW_MAX)
               * (1 - clip((N_b - ramp_center)/ramp_width, 0, 1))
               (moment skewness; ramped to ZERO above the prior-ceiling
                collapse, full skew below it — znz_kernel.py:1362 semantics)

and the skew-normal (xi, omega, alpha) are the MOMENT-MATCHED parameters
(the closed-form inverse of the skew-normal moment relations, replicating
``_moment_to_skewnormal_vec``), NOT the moments themselves.

``psi_k_delta`` (2, SR, ZR) adds to the LEADING (order-0) terms of the mu and
sig coefficient surfaces per response cell (the fit-cov-diag perturbations).
NOTE (clip boundary): d mu / d psi_sig is ZERO wherever the sd polynomial
sits at/below resp_sig_floor across a cell (the prior then owns that
direction). On the real 2LPT-0 pack the point surface has min sd ~ 0.11 dex
>> floor 1e-3, so the clip is inactive at the point; it can engage ~1
prior-sd below the point in the narrowest cells (documented, benign for NUTS).

K[c<-b] is the ANALYTIC mass of the skew-normal density of N-hat in observed
bin c:  K = F(hi_c) - F(lo_c) with the exact skew-normal CDF

    F(x) = Phi(z) - 2 * OwensT(z, alpha),   z = (x - xi) / sigma .

Owen's T has no elementary closed form; the jnp path evaluates it with a
FIXED 64-node Gauss-Legendre rule on its defining integral

    T(h, a) = (1/2pi) * int_0^a exp(-h^2 (1+t^2)/2) / (1+t^2) dt ,

which is exact to well below 1e-12 absolute for the |alpha| <= ~6 range used
here (the integrand is analytic; the rule converges geometrically). This is
NOT the Azzalini approximation — accuracy is asserted in the tests against
scipy (exact owens_t) at zero skew and |skew| <= 1.4 to << 1e-4 on bin masses.
The numpy oracle uses ``scipy.special.owens_t`` (exact) — an independent path.

Completeness: C_cell = logistic(eta_hat + psi_c) on the molly (s, m) cells,
with the Jeffreys-consistent point surface

    eta_hat = log((n_det + 1/2) / (n_tot - n_det + 1/2)) ,

gathered to true-N bins via b -> molly-cell digitization.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp

from CDDF_analysis.hbi_mcmc.pack import ModelAPack

__all__ = ["ModelAConsts", "build_consts", "fold_mu", "fold_mu_fp",
           "fold_mu_reference", "owens_t_jnp", "skewnorm_cdf_jnp",
           "moment_to_skewnormal_jnp", "eta_hat_sigma_hat"]

_SQRT2 = float(np.sqrt(2.0))
# fixed Gauss-Legendre rule for Owen's T (module-level constants; static in jit)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_GL_X = jnp.asarray(_GL_X)
_GL_W = jnp.asarray(_GL_W)

# Attainable skew-normal moment-skewness ceiling.  DUPLICATED, deliberately:
# the same expression is `znz_kernel._SN_SKEW_MAX`, and it is re-typed here
# rather than imported because this module MUST stay importable without the
# heavy `CDDF_analysis.hbi` chain (verified: importing this module leaves both
# `CDDF_analysis.hbi` and `CDDF_analysis.hbi.znz_kernel` out of sys.modules,
# and `znz_kernel` pulls a much larger surface).
#
# 🔴 The pin is a TEST, not this comment.  Until 2026-08-05 the only thing
# asserting the two agreed was a parenthetical "(== znz_kernel._SN_SKEW_MAX)"
# -- prose cannot fail.  `tests/test_modelA_forward.py`
# ::test_SN_SKEW_MAX_equals_the_znz_kernel_constant asserts bit-for-bit
# equality and is what actually holds the two copies together.
_SN_SKEW_MAX = 0.5 * (4.0 - np.pi) * (np.sqrt(2.0 / np.pi) ** 3) / \
    (1.0 - 2.0 / np.pi) ** 1.5


# --- shared small pieces (jnp path only) ---------------------------------------

def owens_t_jnp(h, a):
    """Owen's T(h, a) by 64-node Gauss-Legendre on the defining integral (jnp).

    Broadcasts over h and a. Odd in a, even in h (both inherited from the
    integral form directly). Accurate to < 1e-12 abs for |a| <= ~6.
    """
    h = jnp.asarray(h)
    a = jnp.asarray(a)
    h, a = jnp.broadcast_arrays(h, a)
    t = 0.5 * (_GL_X + 1.0)                      # (Q,) nodes on [0, 1]
    ta = a[..., None] * t                        # (..., Q) nodes on [0, a]
    one_pt2 = 1.0 + ta * ta
    integ = jnp.exp(-0.5 * (h[..., None] ** 2) * one_pt2) / one_pt2
    return (a / (4.0 * jnp.pi)) * jnp.sum(_GL_W * integ, axis=-1)


def skewnorm_cdf_jnp(x, xi, omega, alpha):
    """Exact skew-normal CDF Phi(z) - 2 T(z, alpha), z = (x - xi)/omega (jnp)."""
    z = (x - xi) / omega
    return 0.5 * (1.0 + jax.scipy.special.erf(z / _SQRT2)) - 2.0 * owens_t_jnp(z, alpha)


def moment_to_skewnormal_jnp(mean, sd, skew):
    """(mean, sd, moment-skewness) -> skew-normal (xi, omega, alpha), jnp.

    Replicates znz_kernel._moment_to_skewnormal_vec (the committed legacy map;
    fix F4 of the 2026-07-11 characterization): closed-form inverse of the
    skew-normal moment relations, |skew| clamped to the attainable ceiling,
    exact Gaussian branch at |skew| < 1e-9.

    Gradient-safe: the fractional power is evaluated on a symm-masked operand
    so the |skew|^(2/3) cusp at 0 never produces NaN cotangents through the
    ``where`` (no sampled parameter currently flows through skew — psi_k_delta
    perturbs only the mu/sig order-0 coefficients — but keep it safe anyway).
    """
    mean = jnp.asarray(mean)
    sd = jnp.clip(jnp.asarray(sd), 1e-9, None)
    s = jnp.clip(jnp.asarray(skew), -0.995 * _SN_SKEW_MAX, 0.995 * _SN_SKEW_MAX)
    b = np.sqrt(2.0 / np.pi)
    c = 0.5 * (4.0 - np.pi)
    sym = jnp.abs(s) < 1e-9
    s_safe = jnp.where(sym, 1.0, jnp.abs(s))          # cusp-safe operand
    r = (s_safe / c) ** (2.0 / 3.0)
    g = r / (1.0 + r)                                 # g = (b*delta)^2 in (0,1)
    bdelta = jnp.sqrt(g)
    delta = jnp.clip(jnp.sign(s) * bdelta / b, -0.999, 0.999)
    delta = jnp.where(sym, 0.0, delta)
    alpha = delta / jnp.sqrt(jnp.clip(1.0 - delta * delta, 1e-12, None))
    omega = sd / jnp.sqrt(jnp.clip(1.0 - (b * delta) ** 2, 1e-12, None))
    xi = mean - omega * b * delta
    alpha = jnp.where(sym, 0.0, alpha)
    omega = jnp.where(sym, sd, omega)
    xi = jnp.where(sym, mean, xi)
    return xi, omega, alpha


def eta_hat_sigma_hat(molly_n_det, molly_n_tot):
    """Jeffreys-consistent completeness point surface + width (numpy).

    eta_hat  = log((n_det + 1/2)/(n_tot - n_det + 1/2))
    sig_hat  = sqrt(1/(n_det + 1/2) + 1/(n_tot - n_det + 1/2))
    """
    d = np.asarray(molly_n_det, float)
    t = np.asarray(molly_n_tot, float)
    eta = np.log((d + 0.5) / (t - d + 0.5))
    sig = np.sqrt(1.0 / (d + 0.5) + 1.0 / (t - d + 0.5))
    return eta, sig


# --- static constants for the fold ------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ModelAConsts:
    """Static (non-sampled) inputs of the fold, precomputed once from a pack."""

    # grids
    nhat_edges: jnp.ndarray      # (C+1,)
    dN_b: jnp.ndarray            # (B,)
    Nc_b: jnp.ndarray            # (B,) true-N bin centers
    u_pow: jnp.ndarray           # (B, D) covariate powers u^0..u^{D-1} (UNCLAMPED)
    # (SR, ZR, B, D) covariate powers with the covariate CLIPPED into each
    # response cell's CALIBRATED range (finding D2). This is what build_K uses.
    u_pow_resp: jnp.ndarray
    kz_to_K: np.ndarray          # (Kf,) static int indices
    s_to_sresp: np.ndarray       # (S,)  static int indices
    K_to_zresp: np.ndarray       # (KK,) static int indices
    b_to_cell: np.ndarray        # (B,)  static int indices into molly cells
    # data-plane constants
    dX: jnp.ndarray              # (Kf, S)
    g_bk: jnp.ndarray            # (B, Kf)
    # completeness
    eta_hat: jnp.ndarray         # (S, M)
    sigma_hat: jnp.ndarray       # (S, M)
    # response
    resp_mu_coef: jnp.ndarray    # (SR, ZR, D)
    resp_sig_coef: jnp.ndarray   # (SR, ZR, D)
    resp_skew_coef: jnp.ndarray  # (SR, ZR, D)
    resp_sig_floor: float
    resp_skew_ramp: jnp.ndarray  # (2,)
    fitcov_sd: jnp.ndarray       # (2, SR, ZR) prior SDs for psi_k_delta
    # FP
    fp_w: float
    fp_ell_eff: float
    fp_E: jnp.ndarray            # (Kf, S)
    fp_eta_c: jnp.ndarray        # (C,) host-occlusion fraction per observed bin
    t_sigma: jnp.ndarray         # (KK,)
    # dims
    n_c: int
    n_b: int
    n_k: int
    n_kk: int
    n_s: int
    n_molly: int
    n_sr: int
    n_zr: int
    n_deg: int
    resp_clamp: str = "both"     # "both" | "hi" | "off" (stamped; see build_consts)


_DEFAULT_FITCOV_DIAG = (0.02 ** 2, 0.10 ** 2)  # documented fallback (mu0, sig0 vars)


_CLAMP_MODES = ("both", "hi", "off")


def build_consts(pack: ModelAPack, *, resp_clamp: str = "both",
                 allow_unclamped_response: bool = False,
                 allow_missing_fp_eta: bool = False) -> ModelAConsts:
    """Precompute the static fold inputs (index maps, powers, Jeffreys eta-hat).

    FAIL-CLOSED on the FP host-occlusion vector (restoration 2026-08-06, PI
    ruling 8): the pack MUST carry ``fp_eta_c`` — the per-observed-bin
    host-occlusion fraction of the loa-0 FP product's own definition
    (``build_loa0_fp_product.py``: a forest FP can only occur in un-occluded
    forest, so the transported production expectation carries ``(1 − η_band)``
    per band, with ``η_DLA ≡ 0`` forced). Until 2026-08-06 the fold carried
    this factor ZERO times while the product definition it implements carries
    it once (+0.58% bias on the sub-DLA-band FP term).
    ``allow_missing_fp_eta=True`` admits a pack extracted before 2026-08-06
    and sets η ≡ 0 — it exists so diagnostics can reproduce the pre-restoration
    numbers, not so production can skip the factor.

    FAIL-CLOSED on the response covariate reference (fix F1/F1b): the pack
    MUST carry ``resp_N_ref`` — the reference N the coefficient polynomials
    were fit at. There is no silent midpoint fallback (evaluating the polys at
    a shifted covariate was the 2026-07-11 F1 kernel defect).

    FAIL-CLOSED on the response covariate RANGE (finding D2, 2026-07-28): the
    pack MUST carry ``resp_N_fit_range`` (SR, ZR, 2) — the min/max true-N
    anchor each cell's MOMENT polynomials were actually fit at. The committed
    ``ForwardResponseModel._eval_surface`` evaluates a degree-2 polynomial at
    ANY N with no range guard; on the frozen 2LPT-0 response the anchors span
    ~19.35–21.22 while the fold reaches N = 22.35, and the quadratic's positive
    curvature turns the mean up-bias from the MEASURED +0.001 dex at the top
    anchor into +0.30 dex (cell 0,0) / +0.78 dex (cell 0,2) — which is the
    entire 1.5–3.5x high-N excess of the rung-9 forward-model failure, and
    drives 97% of the top true-N bin's kernel mass off the top of the observed
    grid. ``resp_clamp`` selects how the covariate is guarded:

      "both" (default) : clip into [N_lo, N_hi] — the defensible general rule,
                         no extrapolation on either side.
      "hi"             : clip only ABOVE N_hi — the side where the extrapolation
                         is EMPIRICALLY refuted (measured bias ~0 at the top
                         anchor). Retained for the systematic bracket.
      "off"            : the pre-fix behaviour. DIAGNOSTIC ONLY; reproduces the
                         defect.

    ``allow_unclamped_response=True`` admits a pack with no range (legacy packs
    extracted before 2026-07-28) and forces ``resp_clamp="off"``; it exists so
    the self-test can reproduce the pre-fix numbers, not so production can skip
    the guard.
    """
    if resp_clamp not in _CLAMP_MODES:
        raise ValueError(f"resp_clamp must be one of {_CLAMP_MODES}, "
                         f"got {resp_clamp!r}")
    if pack.resp_N_ref is None:      # F1 guard first (more specific message)
        raise ValueError(
            "build_consts: pack.resp_N_ref is None — the response covariate "
            "reference (the N_ref the resp_*_coef polynomials were FIT at) is "
            "REQUIRED. Re-extract the pack (extract_pack emits it) or emit it "
            "from the generator; there is NO midpoint fallback (finding F1).")
    if pack.resp_N_fit_range is None:
        if not allow_unclamped_response:
            raise ValueError(
                "build_consts: pack.resp_N_fit_range is None — the CALIBRATED "
                "covariate range of the response moment polynomials is "
                "REQUIRED (finding D2, 2026-07-28). Without it the degree-2 "
                "moment surfaces are extrapolated ~1.2 dex past their top "
                "anchor and manufacture a 1.5-3.5x high-N excess. Re-extract "
                "the pack (emit emp_N_anchors min/max per response cell as "
                "resp_N_fit_range), or pass allow_unclamped_response=True to "
                "REPRODUCE the pre-fix behaviour in a diagnostic.")
        resp_clamp = "off"
    if getattr(pack, "fp_eta_c", None) is None:
        if not allow_missing_fp_eta:
            raise ValueError(
                "build_consts: pack.fp_eta_c is None — the per-observed-bin "
                "host-occlusion fraction of the loa-0 FP product definition "
                "(mu_FP ∝ (1 − η_band)) is REQUIRED (restoration 2026-08-06, "
                "PI ruling 8). Re-extract the pack (extract_pack emits it "
                "from the product's band_eta_per_nbin), or pass "
                "allow_missing_fp_eta=True to REPRODUCE the pre-restoration "
                "behaviour (η ≡ 0) in a diagnostic.")
        fp_eta_c = np.zeros(pack.n_c, float)
    else:
        fp_eta_c = np.asarray(pack.fp_eta_c, float)
        if fp_eta_c.shape != (pack.n_c,):
            raise ValueError(
                f"build_consts: fp_eta_c has shape {fp_eta_c.shape}, "
                f"expected ({pack.n_c},)")
        if np.any(~np.isfinite(fp_eta_c)) or np.any(fp_eta_c < 0) \
                or np.any(fp_eta_c >= 1):
            raise ValueError(
                "build_consts: fp_eta_c must be finite with 0 <= eta < 1")
    ntrue = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    dN = np.diff(ntrue)
    if pack.resp_N_ref is None:
        raise ValueError(
            "build_consts: pack.resp_N_ref is None — the response covariate "
            "reference (the N_ref the resp_*_coef polynomials were FIT at) is "
            "REQUIRED. Re-extract the pack (extract_pack emits it) or emit it "
            "from the generator; there is NO midpoint fallback (finding F1).")
    n_ref = float(pack.resp_N_ref)
    D = pack.resp_mu_coef.shape[-1]
    u = Nc - n_ref
    u_pow = u[:, None] ** np.arange(D)[None, :]

    # per-response-cell CLAMPED covariate powers (finding D2)
    SR_, ZR_ = pack.resp_mu_coef.shape[:2]
    if pack.resp_N_fit_range is not None:
        rr = np.asarray(pack.resp_N_fit_range, float)          # (SR, ZR, 2)
    else:                                                       # resp_clamp=="off"
        rr = np.broadcast_to(np.array([-np.inf, np.inf]), (SR_, ZR_, 2))
    if resp_clamp == "both":
        Nc_cl = np.clip(Nc[None, None, :], rr[..., 0][..., None],
                        rr[..., 1][..., None])
    elif resp_clamp == "hi":
        Nc_cl = np.minimum(Nc[None, None, :], rr[..., 1][..., None])
    else:                                                       # "off"
        Nc_cl = np.broadcast_to(Nc[None, None, :], (SR_, ZR_, len(Nc)))
    u_pow_resp = (Nc_cl - n_ref)[..., None] ** np.arange(D)[None, None, None, :]

    zf = np.asarray(pack.zf_edges, float)
    zc = np.asarray(pack.zc_edges, float)
    snr = np.asarray(pack.snr_edges, float)
    rse = np.asarray(pack.resp_snr_edges, float)
    rze = np.asarray(pack.resp_z_edges, float)

    kz_to_K = np.asarray(pack.kz_to_K, dtype=np.int64)
    s_to_sresp = (np.digitize(snr[:-1] + 1e-9, rse) - 1).astype(np.int64)
    # F5 guard: strata BELOW the response SNR range digitize to -1, and a
    # negative index in a gather silently WRAPS to the highest response cell.
    # Such strata are legal only when structurally empty (dX == 0; the op-mask
    # SNR<=2 strata) — assert that, then clamp them to cell 0 (the legacy
    # ForwardResponseModel._i_snr clip convention). Their K columns are dead
    # weight multiplied by dX == 0 in the fold.
    oob = (s_to_sresp < 0) | (s_to_sresp >= pack.resp_mu_coef.shape[0])
    if np.any(oob):
        dx_oob = np.asarray(pack.dX, float)[:, oob]
        if np.any(dx_oob > 0):
            raise ValueError(
                "build_consts: SNR strata outside the response-cell range "
                f"carry exposure (strata {np.where(oob)[0].tolist()} with "
                "dX > 0) — the response model does not cover them (F5).")
        s_to_sresp = np.clip(s_to_sresp, 0, pack.resp_mu_coef.shape[0] - 1)
    zc_centers = 0.5 * (zc[:-1] + zc[1:])
    K_to_zresp = (np.digitize(zc_centers, rze) - 1).astype(np.int64)
    if np.any(K_to_zresp < 0) or np.any(K_to_zresp >= pack.resp_mu_coef.shape[1]):
        raise ValueError(
            "build_consts: some coarse-z bin does not map into a response "
            "cell — negative-index gathers are refused (F5 guard).")
    me = np.asarray(pack.molly_nhi_edges, float)
    b_to_cell = np.clip(np.digitize(Nc, me) - 1, 0, len(me) - 2).astype(np.int64)

    eta_hat, sigma_hat = eta_hat_sigma_hat(pack.molly_n_det, pack.molly_n_tot)

    if pack.resp_fitcov_diag is not None:
        fitcov_sd = np.sqrt(np.asarray(pack.resp_fitcov_diag, float))
    else:  # documented default when the (extension) key is absent
        SR, ZR = pack.resp_mu_coef.shape[:2]
        fitcov_sd = np.sqrt(np.stack([
            np.full((SR, ZR), _DEFAULT_FITCOV_DIAG[0]),
            np.full((SR, ZR), _DEFAULT_FITCOV_DIAG[1])]))

    return ModelAConsts(
        nhat_edges=jnp.asarray(pack.nhat_edges, float),
        dN_b=jnp.asarray(dN),
        Nc_b=jnp.asarray(Nc),
        u_pow=jnp.asarray(u_pow),
        u_pow_resp=jnp.asarray(u_pow_resp),
        kz_to_K=kz_to_K,
        s_to_sresp=s_to_sresp,
        K_to_zresp=K_to_zresp,
        b_to_cell=b_to_cell,
        dX=jnp.asarray(pack.dX, float),
        g_bk=jnp.asarray(np.asarray(pack.g_grid, float)[b_to_cell, :]),
        eta_hat=jnp.asarray(eta_hat),
        sigma_hat=jnp.asarray(sigma_hat),
        resp_mu_coef=jnp.asarray(pack.resp_mu_coef, float),
        resp_sig_coef=jnp.asarray(pack.resp_sig_coef, float),
        resp_skew_coef=jnp.asarray(pack.resp_skew_coef, float),
        resp_sig_floor=float(pack.resp_sig_floor),
        resp_skew_ramp=jnp.asarray(pack.resp_skew_ramp, float),
        fitcov_sd=jnp.asarray(fitcov_sd),
        fp_w=float(pack.fp_w_sightline_ratio),
        fp_ell_eff=float(pack.fp_ell_eff),
        fp_E=jnp.asarray(pack.fp_E_alloc, float),
        fp_eta_c=jnp.asarray(fp_eta_c, float),
        t_sigma=jnp.asarray(pack.t_sigma, float),
        n_c=pack.n_c, n_b=pack.n_b, n_k=pack.n_k, n_kk=pack.n_kk,
        n_s=pack.n_s, n_molly=pack.n_molly,
        n_sr=pack.resp_mu_coef.shape[0], n_zr=pack.resp_mu_coef.shape[1],
        n_deg=D, resp_clamp=resp_clamp,
    )


# --- the jnp fold ------------------------------------------------------------------

def build_K(psi_k_delta, consts: ModelAConsts):
    """Response kernel K[s, K, c, b]: skew-normal bin masses, fully vectorized.

    psi_k_delta (2, SR, ZR) perturbs the order-0 (leading) mu/sig coef terms.

    Committed ``ForwardResponseModel`` semantics (fixes F2-F4, 2026-07-11):
    the coefficient surfaces are MOMENTS — mean = N + mu-poly,
    sd = clip(sig-poly, floor), moment-skewness = clamp(skew-poly) ramped to
    ZERO over [ramp_center, ramp_center + ramp_width] going UP — and (xi,
    omega, alpha) come from the moment-match, not direct substitution.
    """
    mu_coef = consts.resp_mu_coef.at[..., 0].add(psi_k_delta[0])
    sig_coef = consts.resp_sig_coef.at[..., 0].add(psi_k_delta[1])
    # gather response cells to (S, KK, D)
    mu_sk = mu_coef[consts.s_to_sresp][:, consts.K_to_zresp]
    sig_sk = sig_coef[consts.s_to_sresp][:, consts.K_to_zresp]
    skw_sk = consts.resp_skew_coef[consts.s_to_sresp][:, consts.K_to_zresp]
    # covariate powers CLAMPED to each response cell's calibrated range (D2),
    # gathered onto the same (S, KK) plane as the coefficients: (S, KK, B, D)
    u_sk = consts.u_pow_resp[consts.s_to_sresp][:, consts.K_to_zresp]
    # polynomial MOMENT surfaces over b: (S, KK, B).  NOTE the RAMP and the bin
    # centre stay on the UNCLAMPED Nc_b — the clamp guards the fitted
    # polynomials' covariate, not the physical N of the bin.
    mean = consts.Nc_b[None, None, :] + jnp.einsum("skd,skbd->skb", mu_sk, u_sk)
    sd = jnp.clip(jnp.einsum("skd,skbd->skb", sig_sk, u_sk),
                  consts.resp_sig_floor, None)                       # F2
    ramp = jnp.clip((consts.Nc_b - consts.resp_skew_ramp[0])
                    / consts.resp_skew_ramp[1], 0.0, 1.0)            # F3
    skew = jnp.clip(jnp.einsum("skd,skbd->skb", skw_sk, u_sk),
                    -0.995 * _SN_SKEW_MAX, 0.995 * _SN_SKEW_MAX) \
        * (1.0 - ramp)[None, None, :]
    xi, omega, alpha = moment_to_skewnormal_jnp(mean, sd, skew)      # F4
    # CDF at every observed-bin edge: (S, KK, C+1, B)
    F = skewnorm_cdf_jnp(consts.nhat_edges[None, None, :, None],
                         xi[:, :, None, :], omega[:, :, None, :],
                         alpha[:, :, None, :])
    K = jnp.clip(F[:, :, 1:, :] - F[:, :, :-1, :], 0.0, 1.0)
    return K  # (S, KK, C, B)


def fold_mu_fp(log_t, lam_fp, consts: ModelAConsts):
    """THE false-positive term of the fold. One definition, one call site each.

        mu_FP[c,k,s] = fp_w * fp_ell_eff * (1 - fp_eta_c[c])
                       * exp(t[kz_to_K[k]]) * lam_fp[c,s] * fp_E[k,s]

    Extracted 2026-08-05 (behaviour-preserving; ``fold_mu`` calls it and the
    expression is byte-for-byte the one it used to inline).  It exists because
    the term was RE-TYPED in ``forward_selftest.selftest`` to form the
    ``mu_sig = mu - mu_fp`` split, and the copy had already drifted: it dropped
    the ``exp(log_t)`` factor -- harmless there only because that caller always
    passes ``log_t = 0``, and silently wrong the moment the fold changes.  The
    2026-08-05 fp_ell_eff repair had to be applied at BOTH copies by hand,
    which is the argument for there being one.

    The numpy oracle ``fold_mu_reference`` deliberately does NOT call this: it
    is an independent re-implementation of the whole expression and sharing a
    helper would defeat its purpose.

    Parameters
    ----------
    log_t   : (KK,)   per-coarse-z log transfer factors
    lam_fp  : (C, S)  FP intensity per unit loa-0 exposure
    consts  : ModelAConsts

    Returns
    -------
    mu_fp : (C, Kf, S)
    """
    exp_t_k = jnp.exp(jnp.asarray(log_t))[consts.kz_to_K]  # (Kf,)
    # fp_w * fp_ell_eff == N_sl_loa0 exactly; see the module docstring for why
    # fp_ell_eff must be here (lam_fp is per unit loa-0 exposure, not a count).
    # (1 - fp_eta_c): host-occlusion survival per observed bin — the product's
    # own definition (a forest FP can only occur in un-occluded forest); the
    # loa-0 calibration side (Poisson(ell_eff * lam)) correctly does NOT carry
    # it, because loa-0 has no HCDs to occlude. Restored 2026-08-06.
    return consts.fp_w * consts.fp_ell_eff \
        * (1.0 - consts.fp_eta_c)[:, None, None] \
        * exp_t_k[None, :, None] \
        * jnp.asarray(lam_fp)[:, None, :] * consts.fp_E[None, :, :]


def fold_mu(theta_pop, psi_c, psi_k_delta, log_t, lam_fp, consts: ModelAConsts):
    """The Model A forward fold, pure jnp, no python loops over data cells.

    Parameters
    ----------
    theta_pop   : (B, Kf)  log f on the fine grid
    psi_c       : (S, M)   completeness logit offsets (eta = eta_hat + psi_c)
    psi_k_delta : (2, SR, ZR) order-0 mu/sig response-coef perturbations
    log_t       : (KK,)    per-coarse-z log transfer factors
    lam_fp      : (C, S)   FP intensity (sampled; pass zeros for fp-off)
    consts      : ModelAConsts

    Returns
    -------
    mu : (C, Kf, S) expected counts.

    Note: lam_fp is an explicit argument (it is a sampled site per spec
    section 2 — the fold signature in the Q3 task listing omitted it only
    because it groups with the FP block).
    """
    K = build_K(psi_k_delta, consts)                       # (S, KK, C, B)
    K_full = K[:, consts.kz_to_K]                          # (S, Kf, C, B) static gather
    C_cells = jax.nn.sigmoid(consts.eta_hat + psi_c)       # (S, M)
    C_bs = C_cells[:, consts.b_to_cell]                    # (S, B) static gather
    f = jnp.exp(theta_pop)                                 # (B, Kf)
    contrib = C_bs.T[:, None, :] * consts.g_bk[:, :, None] * f[:, :, None] \
        * consts.dN_b[:, None, None]                       # (B, Kf, S)
    mu_sig = jnp.einsum("skcb,bks->cks", K_full, contrib) * consts.dX[None, :, :]
    return mu_sig + fold_mu_fp(log_t, lam_fp, consts)      # (C, Kf, S)


# --- the INDEPENDENT numpy oracle ---------------------------------------------------
# Written as a separate code path on purpose: explicit loops over data cells,
# scipy special functions, its own digitize/eta-hat/poly evaluation. Do NOT
# refactor to share helpers with the jnp fold above — the whole point is that
# the two implementations can only agree if the expression itself is right.

def fold_mu_reference(theta_pop, psi_c, psi_k_delta, log_t, lam_fp,
                      pack: ModelAPack, resp_clamp="both",
                      allow_unclamped_response=False,
                      allow_missing_fp_eta=False):
    """Plain-numpy oracle of the SAME fold expression, computed cell by cell."""
    from scipy.special import ndtr, owens_t, expit

    theta_pop = np.asarray(theta_pop, float)
    psi_c = np.asarray(psi_c, float)
    psi_k_delta = np.asarray(psi_k_delta, float)
    log_t = np.asarray(log_t, float)
    lam_fp = np.asarray(lam_fp, float)

    nhat = np.asarray(pack.nhat_edges, float)
    ntrue = np.asarray(pack.ntrue_edges, float)
    zf = np.asarray(pack.zf_edges, float)
    zc = np.asarray(pack.zc_edges, float)
    snr = np.asarray(pack.snr_edges, float)
    C_n = len(nhat) - 1
    B_n = len(ntrue) - 1
    K_n = len(zf) - 1
    S_n = len(snr) - 1

    centers_b = np.array([0.5 * (ntrue[i] + ntrue[i + 1]) for i in range(B_n)])
    dN = np.array([ntrue[i + 1] - ntrue[i] for i in range(B_n)])
    if pack.resp_N_ref is None:
        raise ValueError(
            "fold_mu_reference: pack.resp_N_ref is None — the response "
            "covariate reference is REQUIRED (no midpoint fallback; F1).")
    n_ref = float(pack.resp_N_ref)
    # D2 covariate-range guard (own logic, independent of build_consts).
    # NOTE the F1 (resp_N_ref) guard above fires FIRST by design.
    if resp_clamp not in _CLAMP_MODES:
        raise ValueError(f"resp_clamp must be one of {_CLAMP_MODES}")
    if pack.resp_N_fit_range is None:
        if not allow_unclamped_response:
            raise ValueError(
                "fold_mu_reference: pack.resp_N_fit_range is None — the "
                "calibrated covariate range is REQUIRED (finding D2).")
        resp_clamp = "off"
        fit_rng = None
    else:
        fit_rng = np.asarray(pack.resp_N_fit_range, float)

    def _clamp_N(N, sr, zr):
        if resp_clamp == "off" or fit_rng is None:
            return float(N)
        lo, hi = float(fit_rng[sr, zr, 0]), float(fit_rng[sr, zr, 1])
        if resp_clamp == "hi":
            return float(min(N, hi))
        return float(min(max(N, lo), hi))

    # index maps (independent digitize logic)
    zf_centers = np.array([0.5 * (zf[i] + zf[i + 1]) for i in range(K_n)])
    kz2K = np.searchsorted(zc, zf_centers, side="right") - 1
    kz2K = np.minimum(kz2K, len(zc) - 2)
    rse = np.asarray(pack.resp_snr_edges, float)
    s2sr = np.searchsorted(rse, snr[:-1] + 1e-9, side="right") - 1
    # F5 guard (own logic): sub-range strata are legal only with dX == 0;
    # clamp them to cell 0 instead of letting a -1 index wrap.
    oob = (s2sr < 0) | (s2sr >= np.asarray(pack.resp_mu_coef).shape[0])
    if np.any(oob):
        if np.any(np.asarray(pack.dX, float)[:, oob] > 0):
            raise ValueError(
                "fold_mu_reference: SNR strata outside the response range "
                "carry exposure (F5).")
        s2sr = np.clip(s2sr, 0, np.asarray(pack.resp_mu_coef).shape[0] - 1)
    rze = np.asarray(pack.resp_z_edges, float)
    zc_centers = np.array([0.5 * (zc[i] + zc[i + 1]) for i in range(len(zc) - 1)])
    K2zr = np.searchsorted(rze, zc_centers, side="right") - 1
    me = np.asarray(pack.molly_nhi_edges, float)
    b2cell = np.searchsorted(me, centers_b, side="right") - 1
    b2cell = np.clip(b2cell, 0, len(me) - 2)

    # completeness surface (own Jeffreys-logit code)
    nd = np.asarray(pack.molly_n_det, float)
    nt = np.asarray(pack.molly_n_tot, float)
    eta_hat_ref = np.log(nd + 0.5) - np.log(nt - nd + 0.5)
    C_cells = expit(eta_hat_ref + psi_c)                    # (S, M)

    g = np.asarray(pack.g_grid, float)                      # (M, Kf)
    dX = np.asarray(pack.dX, float)
    E = np.asarray(pack.fp_E_alloc, float)
    w = float(pack.fp_w_sightline_ratio)
    # the loa-0 exposure lam_fp is defined per (see the module docstring); the
    # FP term is under-normalised by exactly this factor without it
    ell = float(pack.fp_ell_eff)
    # host-occlusion survival per observed bin (own logic, independent of
    # build_consts): the product's mu_FP definition carries (1 - eta_band).
    if getattr(pack, "fp_eta_c", None) is None:
        if not allow_missing_fp_eta:
            raise ValueError(
                "fold_mu_reference: pack.fp_eta_c is None — REQUIRED "
                "(restoration 2026-08-06); pass allow_missing_fp_eta=True to "
                "reproduce the pre-restoration behaviour in a diagnostic.")
        eta_c = np.zeros(C_n, float)
    else:
        eta_c = np.asarray(pack.fp_eta_c, float)
        # NB: NaN passes both "< 0" and ">= 1" (every NaN comparison is
        # False), so finiteness must be tested explicitly (fail-closed).
        if eta_c.shape != (C_n,) or np.any(~np.isfinite(eta_c)) \
                or np.any(eta_c < 0) or np.any(eta_c >= 1):
            raise ValueError("fold_mu_reference: bad fp_eta_c")
    sig_floor = float(pack.resp_sig_floor)
    ramp_c, ramp_w = [float(v) for v in np.asarray(pack.resp_skew_ramp, float)]

    def _poly(coefs, u):
        return sum(coefs[d] * u ** d for d in range(len(coefs)))

    def _sn_cdf(x, xi, om, al):
        z = (x - xi) / om
        return ndtr(z) - 2.0 * owens_t(z, al)

    # scalar moment -> skew-normal (xi, omega, alpha) map: own implementation
    # of the committed closed-form inverse (mirrors the SCALAR
    # znz_kernel._moment_to_skewnormal; independent of the jnp vec path).
    skew_max = 0.5 * (4.0 - np.pi) * (np.sqrt(2.0 / np.pi) ** 3) / \
        (1.0 - 2.0 / np.pi) ** 1.5

    def _m2sn(mean, sd, skew):
        bb = np.sqrt(2.0 / np.pi)
        s_ = float(np.clip(skew, -0.995 * skew_max, 0.995 * skew_max))
        sd = float(max(sd, 1e-9))
        if abs(s_) < 1e-9:
            return float(mean), sd, 0.0
        cc = 0.5 * (4.0 - np.pi)
        r = (abs(s_) / cc) ** (2.0 / 3.0)
        gg = r / (1.0 + r)
        delta = float(np.clip(np.sign(s_) * np.sqrt(gg) / bb, -0.999, 0.999))
        al = delta / np.sqrt(max(1.0 - delta * delta, 1e-12))
        om = sd / np.sqrt(max(1.0 - (bb * delta) ** 2, 1e-12))
        return float(mean) - om * bb * delta, float(om), float(al)

    f = np.exp(theta_pop)                                   # (B, Kf)
    mu = np.zeros((C_n, K_n, S_n))
    for s in range(S_n):
        sr = int(s2sr[s])
        for k in range(K_n):
            Kc = int(kz2K[k])
            zr = int(K2zr[Kc])
            mu_coefs = np.asarray(pack.resp_mu_coef, float)[sr, zr].copy()
            sig_coefs = np.asarray(pack.resp_sig_coef, float)[sr, zr].copy()
            skw_coefs = np.asarray(pack.resp_skew_coef, float)[sr, zr]
            mu_coefs[0] = mu_coefs[0] + psi_k_delta[0, sr, zr]
            sig_coefs[0] = sig_coefs[0] + psi_k_delta[1, sr, zr]
            for c in range(C_n):
                acc = 0.0
                for b in range(B_n):
                    # D2: the fitted polynomials are evaluated at the CLAMPED
                    # covariate; the bin's physical N (and the skew ramp) are not.
                    u = _clamp_N(centers_b[b], sr, zr) - n_ref
                    # committed ForwardResponseModel MOMENT semantics (F2-F4):
                    mean = centers_b[b] + _poly(mu_coefs, u)
                    sd = max(_poly(sig_coefs, u), sig_floor)
                    ramp = min(max((centers_b[b] - ramp_c) / ramp_w, 0.0), 1.0)
                    skw = float(np.clip(_poly(skw_coefs, u),
                                        -0.995 * skew_max, 0.995 * skew_max))
                    skw = skw * (1.0 - ramp)
                    xi, om, al = _m2sn(mean, sd, skw)
                    mass = _sn_cdf(nhat[c + 1], xi, om, al) - _sn_cdf(nhat[c], xi, om, al)
                    mass = min(max(mass, 0.0), 1.0)
                    acc += (mass * C_cells[s, b2cell[b]] * g[b2cell[b], k]
                            * f[b, k] * dN[b])
                mu[c, k, s] = dX[k, s] * acc \
                    + w * ell * (1.0 - eta_c[c]) * np.exp(log_t[Kc]) \
                    * lam_fp[c, s] * E[k, s]
    return mu
