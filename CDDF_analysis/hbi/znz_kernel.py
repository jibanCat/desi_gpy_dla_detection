"""znz_kernel.py — Track-C Stage-0b: measure + fit the (xhat, z) bias/scatter model
and the (N, z) completeness model from a truth-matched mock catalog.

Used by Stage-1 to build the 2-D posterior kernel that replaces the frozen
broaden012 kernel with a properly prior-edge-corrected version.

Interfaces
----------
ZNZModel   : b(xhat, z) bias + sigma(xhat, z) scatter polynomial model
CNZModel   : g(j_nhi_cell, kz) completeness model (smooth monotone)

Functions
---------
measure_znz_response   measure (xhat, z, dx) from a truth-matched cat_cut
fit_znz_model          2-D polynomial fit -> ZNZModel
measure_c_nz           count-ratio completeness grid from cat_cut + truth_cut
fit_c_nz_model         smooth + normalize -> CNZModel
save_znz / load_znz    NPZ serialization for both dataclasses
build_cache            CLI entrypoint to reproducibly build the stage-0 NPZ cache

Note on b(xhat, z):
    b fits the MEAN of the dx = xhat - xtrue distribution (right-skewed due to
    the prior-edge pile-up at log N_HI ~ 20.3).  b RISES with both xhat and z —
    larger x̂ sits closer to the prior edge (more up-migration) and higher z has
    denser forest (more blending pushes absorbers toward the edge).
    Do NOT interpret b(20.5) > b(21.0) — the measured direction is the opposite:
    b increases monotonically with xhat and with z.

Note on g(j_nhi_cell, kz):
    g lives on the molly nhi_edges grid whose top edge is +inf.  Stage-1 must
    map g onto the fine-N axis and must NOT index the +inf top cell for any
    finite N value.  g is smaller than the (N,z) kernel shift but non-negligible:
    it must be carried, not dropped.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.polynomial.polynomial import polyvander2d
from scipy.ndimage import gaussian_filter1d

# numpy 2.0 renamed np.trapz -> np.trapezoid; alias preserves both (numpy 1.22+ and 2.x).
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ZNZModel:
    """Polynomial model for conditional bias b(xhat, z) and scatter sigma(xhat, z).

    The polynomial is centred at (xhat_ref, z_ref) for numerical stability:
        b(xhat, z)     = polyvander2d(xhat - xhat_ref, z - z_ref, [deg_xhat, deg_z]) @ b_coef
        sigma(xhat, z) = clip(polyvander2d(...) @ sig_coef, 1e-4, inf)

    Attributes
    ----------
    b_coef : shape ((deg_xhat+1)*(deg_z+1),)
        Flat coefficient array for the bias surface.
    sig_coef : shape ((deg_xhat+1)*(deg_z+1),)
        Flat coefficient array for the scatter surface.
    xhat_ref : float
        Reference xhat (median of training set).
    z_ref : float
        Reference z (median of training set).
    b_ref : float
        b(xhat_ref, z_ref) evaluated at the reference point.
    sig_ref : float
        sigma(xhat_ref, z_ref).
    z_covariate : str
        Column used as z; "z_dla" (Phase 1).
    deg_xhat : int
        Polynomial degree in xhat dimension (stored for robust _design recovery).
    deg_z : int
        Polynomial degree in z dimension (stored for robust _design recovery).
    b_med_coef : np.ndarray or None
        Flat coefficient array for the conditional MEDIAN of dx (the q=0.5 quantile
        surface, IRLS-fit). OPTIONAL — Stage III (response-FORM marginalization). When
        present together with ``b_mix`` < 1, ``b_eff`` returns the form-mixed shift
        ``(1-q)·b_med + q·b_mean``. When None (DEFAULT), ``b_eff`` == ``b`` (the MEAN
        surface) so the model is byte-identical to the frozen Stage-1 behaviour. The
        median is ~+0.035 dex LESS up-correction than the mean (the right-skew of dx;
        2026-06-19_track_c_bref_noncircular.md) — the response-form ambiguity spanned by
        ``b_mix`` ∈ [0,1] is the genuine Track-C response uncertainty Stage III folds
        into the band.
    b_mix : float
        Response-FORM mixing parameter q ∈ [0,1] (DEFAULT 1.0 = pure MEAN shift =
        byte-identical frozen behaviour). ``b_eff = (1-q)·b_med + q·b_mean`` — q=1 →
        mean (full correction), q=0 → median (the skew-robust bulk target). Stage III
        draws q per resample from a truth-match-justified prior; the frozen point uses
        q=1.
    """
    b_coef: np.ndarray
    sig_coef: np.ndarray
    xhat_ref: float
    z_ref: float
    b_ref: float
    sig_ref: float
    z_covariate: str
    deg_xhat: int = 1
    deg_z: int = 2
    b_med_coef: Optional[np.ndarray] = None
    b_mix: float = 1.0
    corr_strength: float = 1.0   # response-STRENGTH α ∈ [0,1] (Stage III). 1 = FULL
    #   correction (DEFAULT, byte-identical); 0 = OFF (no re-center, no width-scale =
    #   the broaden012 kernel un-corrected). apply_znz_correction interpolates the SHIFT
    #   (b_eff−b_ref) and the WIDTH-scale toward identity by α. The truth-match shows the
    #   OFF↔corrected span (R0≈1.11 OFF, ≈0.79 corrected) BRACKETS truth (R0=1), so α is
    #   the response-form axis that, marginalized, covers the truth — track_c_bref note.
    skew_coef: Optional[np.ndarray] = None   # 2-D poly coeffs for γ(x̂,z) skew surface
    #   (Track-C T1). None (DEFAULT) ⇒ no skew warp ⇒ byte-identical to Stage-I/III.
    #   Fitted by _skew_fit_2d (T2) from the truth-match dx third-moment conditional.
    #   The surface carries the right-skew that the affine (location+scale) transform
    #   cannot express: γ(x̂,z) rises from +0.34 at 19.6 to +2.10 at ≥21 (science spec).
    skew_strength: float = 0.0   # γ multiplier gating the warp magnitude (Track-C T1).
    #   0.0 (DEFAULT) ⇒ _skew_warp is the identity ⇒ byte-identical. 1.0 = full fitted
    #   skew. Stage III will draw this per-resample (T4); the frozen point uses 0.0 until
    #   T2 fits skew_coef and the user sets skew_strength=1.0 explicitly.

    def _design(self, xhat: np.ndarray, z: np.ndarray) -> np.ndarray:
        xhat = np.asarray(xhat, float).ravel()
        z = np.asarray(z, float).ravel()
        # Use stored degrees — robust for any (deg_xhat, deg_z) combination.
        # The old sqrt(len(b_coef))-1 formula only worked for perfect squares.
        return polyvander2d(xhat - self.xhat_ref, z - self.z_ref,
                            [self.deg_xhat, self.deg_z])

    def b(self, xhat: np.ndarray, z: np.ndarray) -> np.ndarray:
        """E[xhat - xtrue | xhat, z] at given (xhat, z) points (the MEAN surface).

        b is the mean of a right-skewed dx distribution driven by the prior-edge
        pile-up at log N_HI ~ 20.3.  b RISES with xhat (closer to prior edge →
        more up-migration) and RISES with z (denser forest → more blending).
        """
        return self._design(xhat, z) @ self.b_coef

    def b_median(self, xhat: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Conditional MEDIAN of dx (q=0.5 surface). Falls back to the MEAN surface
        when no median surface was fit (so b_eff == b in that case)."""
        if self.b_med_coef is None:
            return self.b(xhat, z)
        return self._design(xhat, z) @ np.asarray(self.b_med_coef, float)

    def b_eff(self, xhat: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The EFFECTIVE per-object shift used by apply_znz_correction. Mixes the MEAN
        and MEDIAN surfaces by the response-form parameter ``b_mix`` (q):
            b_eff = (1-q)·b_median + q·b_mean.
        q=1 (DEFAULT) or b_med_coef=None → pure MEAN = byte-identical frozen behaviour.
        """
        q = float(self.b_mix)
        if self.b_med_coef is None or q >= 1.0 - 1e-12:
            return self.b(xhat, z)
        if q <= 1e-12:
            return self.b_median(xhat, z)
        return q * self.b(xhat, z) + (1.0 - q) * self.b_median(xhat, z)

    def b_eff_ref(self) -> float:
        """b_eff at the reference point (xhat_ref, z_ref) — the form-mixed b_ref the
        transform subtracts as the reference shift. With b_mix=1 / no median surface
        this is exactly ``b_ref`` (byte-identical)."""
        return float(self.b_eff(np.array([self.xhat_ref]),
                                np.array([self.z_ref]))[0])

    def sigma(self, xhat: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Conditional scatter (> 0) at given (xhat, z) points."""
        return np.clip(self._design(xhat, z) @ self.sig_coef, 1e-4, None)


@dataclass
class CNZModel:
    """Smooth (N, z) completeness model: g(N-cell index j, z-bin index kz).

    g_grid[j, kz] is normalised so that g(j, z_ref_col) = 1 at the reference
    z column (closest to median z).  Values are in (0, ~2].

    Attributes
    ----------
    g_grid : shape (n_nhi_cell, n_zf)
        Smoothed completeness grid, normalised at z_ref.
    nhi_edges : shape (n_nhi_cell + 1,)
        NHI cell edges (from MollyMatrix).
    z_edges_fine : shape (n_zf + 1,)
        Fine z-bin edges (from _fine_z_grid).
    """
    g_grid: np.ndarray
    nhi_edges: np.ndarray
    z_edges_fine: np.ndarray

    def g(self, j_nhi_cell: int, kz: int) -> float:
        """Completeness at NHI cell j and fine z-bin kz. Returns float in (0, ~2]."""
        j = int(np.clip(j_nhi_cell, 0, self.g_grid.shape[0] - 1))
        k = int(np.clip(kz, 0, self.g_grid.shape[1] - 1))
        return float(self.g_grid[j, k])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deg_from_coef(coef: np.ndarray, deg_xhat: int) -> int:
    """Recover deg_z from flat coef length and known deg_xhat.

    Used as a cross-check when loading old NPZ files that pre-date the stored-
    degree fields.  ZNZModel now stores deg_xhat/deg_z directly; _design uses
    them, not this function.
    """
    n_total = len(coef)
    return n_total // (deg_xhat + 1) - 1


def _poly_fit_2d(x: np.ndarray, z: np.ndarray, y: np.ndarray,
                 x_ref: float, z_ref: float,
                 deg_x: int, deg_z: int,
                 weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted least-squares 2-D polynomial fit of y ~ poly(x-x_ref, z-z_ref).

    ``weights`` (>=0, per-row) — when None (DEFAULT) this is the plain unweighted
    lstsq (byte-identical to the original). When given, solves the WLS normal
    equations (used by the Stage-III bootstrap re-fit, where ``weights`` is the
    per-detection TID multiplicity from the SHARED resample).

    Returns flat coefficient array of shape ((deg_x+1)*(deg_z+1),).
    """
    V = polyvander2d(x - x_ref, z - z_ref, [deg_x, deg_z])
    if weights is None:
        coef, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        return coef
    w = np.asarray(weights, float)
    sw = np.sqrt(np.clip(w, 0.0, None))
    coef, _, _, _ = np.linalg.lstsq(V * sw[:, None], y * sw, rcond=None)
    return coef


def _quantile_fit_2d(x: np.ndarray, z: np.ndarray, y: np.ndarray,
                     x_ref: float, z_ref: float, deg_x: int, deg_z: int,
                     q: float = 0.5, weights: Optional[np.ndarray] = None,
                     n_iter: int = 30, eps: float = 1e-4) -> np.ndarray:
    """IRLS quantile (default MEDIAN, q=0.5) 2-D polynomial fit of y ~ poly(...).

    Minimises the (weighted) pinball/check loss by iteratively-reweighted LS with the
    quantile-asymmetric weights w_i = q or (1-q) over |resid| (Huber-smoothed by
    ``eps``). ``weights`` multiplies in the per-row resample multiplicity (Stage III).
    Returns flat coefficient array — the conditional-q surface (q=0.5 ⇒ MEDIAN of dx).
    """
    V = polyvander2d(x - x_ref, z - z_ref, [deg_x, deg_z])
    base_w = (np.ones(len(y)) if weights is None
              else np.asarray(weights, float))
    # warm-start from the (weighted) mean LS fit
    coef = _poly_fit_2d(x, z, y, x_ref, z_ref, deg_x, deg_z, weights=weights)
    for _ in range(n_iter):
        r = y - V @ coef
        # check-loss IRLS weight: quantile asymmetry / smoothed |r|
        asym = np.where(r >= 0.0, q, 1.0 - q)
        wr = base_w * asym / np.maximum(np.abs(r), eps)
        swr = np.sqrt(np.clip(wr, 0.0, None))
        coef_new, _, _, _ = np.linalg.lstsq(V * swr[:, None], (y * swr), rcond=None)
        if np.max(np.abs(coef_new - coef)) < 1e-9:
            coef = coef_new
            break
        coef = coef_new
    return coef


# ---------------------------------------------------------------------------
# Track-C T2: γ ↔ skewness inversion + 2-D skew surface fit (reduce-only)
# ---------------------------------------------------------------------------

# Sane skew-parameter clamp.  The SAS-of-Gaussian skewness map (below) SATURATES
# at ~1.44 as γ→∞ (a pure-Gaussian pre-warp column cannot be pushed past that
# standardized third moment), while the measured truth-match skewness reaches
# +2.10 in the strong-DLA tier.  Cells whose target exceeds the achievable ceiling
# are clamped to ±_SKEW_GAMMA_CLAMP; the residual tail skew is then supplied by the
# (already right-skewed) broaden012 base column the warp acts on in production.
_SKEW_GAMMA_CLAMP = 4.0


def _sas_skewness_of_gamma(gamma: np.ndarray) -> np.ndarray:
    """Standardized third moment (skewness) of ``Y = sinh(arcsinh(Z) + γ)``, Z~N(0,1).

    This is EXACTLY the skewness ``_skew_warp`` induces on a symmetric Gaussian column
    of any conditional width ω (skewness is scale-invariant, so ω drops out — verified
    numerically to <0.5% against a 2e7-sample MC).  Computed in closed form from the
    sinh-moments via the Jones & Pewsey (2009) SAS moment identity

        E[sinh(arcsinh Z + γ)^n]  is a finite combination of  E[sinh^k(arcsinh Z)]·…

    but it is simpler and equally exact to use the moment-generating identity
    ``E[exp(t·arcsinh Z)] = exp(t²/2)·…`` — we instead evaluate the three raw moments
    of ``Y`` from the SAS sinh-moment recursion P_q (Jones & Pewsey eq. 2.3):

        m_k(γ) = E[ sinh(arcsinh Z + γ)^k ]

    via the binomial expansion of ``sinh(a+γ) = sinh a cosh γ + cosh a sinh γ`` and the
    Gaussian moments of ``sinh^p(arcsinh Z) = ((Z + √(Z²+1)) ... )`` — closed-form but
    fiddly.  For robustness and self-evidence we evaluate the three needed raw moments by
    GAUSS–HERMITE quadrature (exact for these smooth integrands to machine precision with
    a modest node count), which keeps the function deterministic and dependency-light.

    Parameters
    ----------
    gamma : float or array
        Skew shape parameter (γ of ``_skew_warp``).

    Returns
    -------
    skewness array (same shape as gamma); 0 at γ=0, → +1.44 as γ→+∞ (and the mirror
    for γ<0), strictly monotone in γ (so the inverse map is single-valued).
    """
    g = np.atleast_1d(np.asarray(gamma, float))
    # Gauss–Hermite nodes/weights for ∫ f(z) e^{-z²} dz; convert to E_{N(0,1)}[h(Z)].
    # 64 nodes is exact to ~1e-12 for these smooth (sinh∘arcsinh) integrands.
    nodes, wts = np.polynomial.hermite_e.hermegauss(64)   # weight e^{-z²/2}, ∫w=√(2π)
    wts = wts / np.sqrt(2.0 * np.pi)                       # normalise to a probability
    aZ = np.arcsinh(nodes)                                 # arcsinh(Z) at the nodes
    out = np.empty_like(g)
    for ii, gi in enumerate(g):
        Y = np.sinh(aZ + gi)                              # warped node values
        m1 = float(np.sum(wts * Y))
        d = Y - m1
        m2 = float(np.sum(wts * d * d))
        m3 = float(np.sum(wts * d * d * d))
        out[ii] = 0.0 if m2 <= 0 else m3 / m2 ** 1.5
    return out.reshape(np.asarray(gamma).shape) if np.ndim(gamma) else float(out[0])


def _gamma_from_skewness(skew_target: np.ndarray,
                         clamp: float = _SKEW_GAMMA_CLAMP) -> np.ndarray:
    """Invert the monotone SAS map ``γ → skewness`` to recover ``γ(skew_target)``.

    Builds a dense γ-grid on ``[−clamp, +clamp]``, evaluates ``_sas_skewness_of_gamma``
    (monotone increasing), and interpolates the inverse.  Targets beyond the achievable
    skewness ceiling (≈±1.44 for a Gaussian pre-warp column) are clamped to ±``clamp``
    (their residual tail skew is supplied by the already-skewed broaden012 base column
    in production — see ``_SKEW_GAMMA_CLAMP``).  Sign is preserved: a POSITIVE
    (right-tail) skewness target → POSITIVE γ (the Ω-restoring direction of ``_skew_warp``).
    """
    st = np.asarray(skew_target, float)
    gg = np.linspace(-clamp, clamp, 2001)
    sk = _sas_skewness_of_gamma(gg)                       # monotone increasing in gg
    # np.interp needs increasing xp; sk is increasing in gg by construction.
    gamma = np.interp(st, sk, gg, left=-clamp, right=clamp)
    return np.clip(gamma, -clamp, clamp)


def _conditional_skewness_cells(xhat: np.ndarray, z: np.ndarray, dx: np.ndarray,
                                weights: Optional[np.ndarray] = None,
                                n_xhat: int = 6, n_z: int = 3,
                                min_count: int = 50):
    """Bin (x̂, z) into a coarse grid and measure the (weighted) conditional skewness of
    ``dx`` per cell — the MEASURED third standardized moment that the skew surface fits.

    Returns ``(xc, zc, sc, wc)`` — the cell-center x̂, cell-center z, measured skewness,
    and an effective per-cell weight (Σ row weights) for cells with ≥ ``min_count`` rows.
    Empty / under-populated cells are dropped (they carry no skew information).  Bin edges
    are QUANTILES of the data so each cell is comparably populated (robust to the
    non-uniform x̂/z marginals of the truth-match population).
    """
    xhat = np.asarray(xhat, float); z = np.asarray(z, float); dx = np.asarray(dx, float)
    w = (np.ones(len(dx)) if weights is None else np.asarray(weights, float))
    # quantile edges (robust, equal-occupancy cells); fall back to linspace if degenerate
    def _edges(a, nb):
        qs = np.quantile(a, np.linspace(0.0, 1.0, nb + 1))
        qs = np.unique(qs)
        if len(qs) < 2:
            qs = np.array([a.min(), a.max() + 1e-9])
        qs[-1] = np.nextafter(qs[-1], np.inf)            # include the max in the top cell
        return qs
    xe = _edges(xhat, n_xhat); ze = _edges(z, n_z)
    xc_l, zc_l, sc_l, wc_l = [], [], [], []
    for ix in range(len(xe) - 1):
        mx = (xhat >= xe[ix]) & (xhat < xe[ix + 1])
        for iz in range(len(ze) - 1):
            m = mx & (z >= ze[iz]) & (z < ze[iz + 1])
            if int(np.count_nonzero(m)) < min_count:
                continue
            wi = w[m]; di = dx[m]
            sw = wi.sum()
            if sw <= 0:
                continue
            m1 = float(np.sum(wi * di) / sw)
            dd = di - m1
            m2 = float(np.sum(wi * dd * dd) / sw)
            if m2 <= 0:
                continue
            m3 = float(np.sum(wi * dd * dd * dd) / sw)
            sc_l.append(m3 / m2 ** 1.5)
            xc_l.append(float(np.sum(wi * xhat[m]) / sw))   # weighted cell-center x̂
            zc_l.append(float(np.sum(wi * z[m]) / sw))      # weighted cell-center z
            wc_l.append(float(sw))
    return (np.asarray(xc_l), np.asarray(zc_l),
            np.asarray(sc_l), np.asarray(wc_l))


def _skew_fit_2d(xhat: np.ndarray, dx: np.ndarray, z: np.ndarray,
                 x_ref: float, z_ref: float, deg_x: int, deg_z: int,
                 weights: Optional[np.ndarray] = None,
                 n_xhat_cells: int = 6, n_z_cells: int = 3,
                 min_count: int = 50,
                 xhat_floor: float = 19.5,
                 clamp: float = _SKEW_GAMMA_CLAMP) -> np.ndarray:
    """Fit the smooth 2-D skew surface ``γ(x̂, z)`` from the truth-match ``dx`` 3rd moment.

    REDUCE-ONLY, NON-CIRCULAR: this reads ONLY the truth-match conditional
    ``dx = x̂ − x_true`` (its per-(x̂,z)-cell standardized skewness) — it NEVER sees
    dN/dX, f(N,z), or Ω (no such argument exists; the signature is ``(xhat, dx, z, …)``).
    The fit target is the MEASURED conditional skewness; the dN/dX/Ω histograms are a
    strictly-downstream CHECK computed after the surface is frozen (the α=1/R0 tautology
    is structurally impossible here — there is no reduction edge into this function).

    Procedure (moment-match, skewness-space fit — robust against the SAS ceiling):
      1. Bin (x̂, z) into a coarse equal-occupancy grid; measure the weighted conditional
         skewness ``s(x̂,z)`` of ``dx`` per cell (``_conditional_skewness_cells``).  The
         measured skewness is the smooth, sign-stable, monotone-in-(x̂,z) quantity the data
         actually traces (the +0.34→+2.10 ramp); it is what we regularize.
      2. Fit a smooth 2-D poly to the per-cell skewness (occupancy-weighted) — denoising
         the sparse high-N cells (a single under-populated cell's sample 3rd moment is very
         noisy; the +7+ outliers are not real).  Evaluate the SMOOTH skewness back on the
         cells.
      3. Invert the MONOTONE SAS map ``γ → skewness-of-warped-column`` numerically
         (``_gamma_from_skewness``) on the SMOOTH per-cell skewness → ``γ_cell``.  The map
         is ω-independent (skewness is scale-invariant) so ONE inversion serves all widths;
         it SATURATES at ≈±1.44, so smooth skewness beyond the ceiling maps to the clamp
         γ=±``clamp`` (the residual tail skew is supplied by the already-skewed broaden012
         base column — see ``_SKEW_GAMMA_CLAMP``).  Sign matches T1: positive (right-tail)
         skewness → positive γ (the Ω-restoring direction).
      4. Weighted-least-squares fit the final 2-D poly ``skew_coef`` to that smooth,
         sign-correct ``γ_cell(x̂,z)``, centred at (x_ref, z_ref) for stability.

    Fitting in SKEWNESS-space first (step 2) — rather than poly-fitting the per-cell
    inverted γ directly — is what keeps the surface sign-correct and well-behaved when the
    measured skewness mostly exceeds the SAS ceiling (the real 2LPT-0 case): a direct γ
    poly over clamped/saturated targets extrapolates to the WRONG sign at low x̂ (all cells
    are right-skewed → γ must stay ≥0), whereas the smooth-skewness route stays monotone
    and non-negative.

    ``xhat_floor`` restricts the cell grid to the SCIENCE range x̂ ≥ floor (default 19.5,
    the sub-DLA fit floor).  Below the floor the truth-match dx flips to LEFT-skew (the
    sub-floor edge population the host_truth_floor=19.0 op-set admits at x̂≈19.3) — a
    different physics from the uniformly right-skewed DLA/sub-DLA response the warp acts
    on; letting those cells into a deg-1-in-x̂ fit drags the surface to the wrong sign at
    the low end.  This is a reduce-only restriction on the conditional (still NON-CIRCULAR
    — it reads only x̂/dx, never dN/dX); it matches the note's measured ramp window
    ([19.5,19.7)→[21,23)).

    Returns the flat ``skew_coef`` array of shape ``((deg_x+1)*(deg_z+1),)`` — feeds
    ``_skew_warp`` via ``apply_znz_correction``.  Falls back to an all-zero (γ≡0 = no
    warp = byte-identical) surface if no cell has enough rows to measure a skewness.
    """
    n_coef = (deg_x + 1) * (deg_z + 1)
    # restrict to the science range x̂ ≥ floor (sub-floor cells are left-skewed; excluding
    # them keeps the deg-1 surface sign-correct — see docstring; reduce-only / non-circular)
    xhat = np.asarray(xhat, float); dx = np.asarray(dx, float); z = np.asarray(z, float)
    keep = xhat >= float(xhat_floor)
    w_in = None if weights is None else np.asarray(weights, float)[keep]
    xc, zc, sc, wc = _conditional_skewness_cells(
        xhat[keep], z[keep], dx[keep], weights=w_in,
        n_xhat=n_xhat_cells, n_z=n_z_cells, min_count=min_count)
    if len(sc) < n_coef:
        # not enough resolved cells to constrain the surface → no skew (identity warp)
        return np.zeros(n_coef, dtype=float)
    # robustness guard: cap grossly-outlying per-cell sample skewness (sparse high-N cells
    # can throw |s|>5 from noise) before the smoothing fit so one cell can't tilt it.
    sc_capped = np.clip(sc, -3.0, 3.0)
    # step 2: smooth the skewness surface (denoise), evaluate back on the cells
    sk_coef = _poly_fit_2d(xc, zc, sc_capped, x_ref, z_ref, deg_x, deg_z, weights=wc)
    V_cells = polyvander2d(xc - x_ref, zc - z_ref, [deg_x, deg_z])
    sk_smooth = V_cells @ sk_coef
    # step 3+4: invert the smooth skewness to γ (sign-correct, clamped) and fit the γ poly
    gamma_cell = _gamma_from_skewness(sk_smooth, clamp=clamp)
    coef = _poly_fit_2d(xc, zc, gamma_cell, x_ref, z_ref, deg_x, deg_z, weights=wc)
    return coef


# ---------------------------------------------------------------------------
# Track-C T1: monotone, mass-conserving skew warp
# ---------------------------------------------------------------------------

def _skew_warp(centers: np.ndarray, mu: float, gamma: float,
               omega: float) -> np.ndarray:
    """Monotone, mass-conserving, **pivot-preserving** warp of a column's bin carriers
    that introduces right-skew ``gamma`` about the column center ``mu`` WITHOUT moving
    the median the affine relocate placed at ``mu``.

    Design: **pivot-corrected sinh-arcsinh (Jones & Pewsey 2009)** of the offset
    ``u = centers − mu`` scaled by the CONDITIONAL WIDTH ``ω`` (the response σ for
    this (x̂, z) object — NOT the carrier-grid std).  The map is

        f(u) = (sinh(arcsinh(u/ω) + γ) − sinh(γ)) · ω

    which is:
      - **pivot-preserving**: ``f(0) = (sinh(0 + γ) − sinh(γ)) · ω = 0`` for ANY γ.
        The pre-skew column is symmetric about ``mu`` (the affine relocate centres it
        there), so its median sits at u=0, which maps to f(0)=0 → the warped column's
        **median stays at ``mu`` = m_tgt**.  This is the count-fixing property: dN/dX
        is set by the median, the skew must be ORTHOGONAL to it.  The mass-weighted
        MEAN is ALLOWED to drift up for γ>0 — that drift IS the skew restoring the Ω
        upper tail.  No mean re-centering is applied (that would undo the Ω restoration);
        pivot correction alone preserves the median for this monotone map.
      - strictly monotone for any ``γ`` (sinh and arcsinh are strictly increasing; an
        added constant ``−sinh(−γ)·ω`` does not affect ordering);
      - smooth and invertible;
      - identity at ``γ=0``: ``sinh(arcsinh(u/ω)) = u/ω`` and ``sinh(0)=0`` →
        ``f(u) = u`` (exact);
      - scale-normalised by the conditional width ``ω`` so the shape parameter ``γ``
        is dimensionless and acts on ``u/ω`` (the natural response units).

    Sign convention: with the pivot held fixed (f(0)=0), the map
    ``sinh(arcsinh(u/ω) + γ)`` for γ>0 stretches the right-side (positive u) offsets and
    compresses the left-side offsets, growing the RIGHT tail while the bulk stays at the
    pivot → POSITIVE (right-tail) skewness and a POSITIVE mean drift (the Ω-restoring
    direction).  γ<0 → left tail / mean drifts down.  (Note: this is the OPPOSITE sign to
    the pre-pivot-correction code, where the un-subtracted bulk translation reversed the
    apparent skew direction; with the pivot fixed the genuine shape skew sign is +γ.)

    Mass conservation: this function warps only the CARRIER positions (bin centers)
    of an existing column mass vector.  The caller is responsible for rebinning the
    mass onto the new carriers using ``_mass_conserving_rebin``, which deposits each
    unit of mass into the destination bin containing the new carrier position —
    conserving Σ exactly (up to floating-point rounding) regardless of the warp.

    Parameters
    ----------
    centers : (n,) float array
        Current bin carrier positions (the ``mids`` after affine relocate + scale).
    mu : float
        Pivot for the warp (the bias-corrected target mean ``m_tgt`` of this column).
        The warp leaves the location of ``mu`` invariant (f(0) = 0), so the column's
        median stays at ``mu``.
    gamma : float
        Skew parameter.  γ>0 → right-skewed (positive skewness, mean drifts UP);
        γ<0 → left-skewed (mean drifts DOWN); γ=0 → identity (bit-for-bit unchanged).
    omega : float
        Conditional width of the column (the response σ for this (x̂, z) object, ~0.19–0.21
        dex) — γ acts on ``u/ω``.  Must be > 0; a non-positive ω is treated as degenerate
        (no warp) so the call is a safe no-op.

    Returns
    -------
    (n,) float array — warped carrier positions, strictly monotone when γ≠0, with the
    pivot ``mu`` (hence the column median) left invariant.
    """
    centers = np.asarray(centers, float)
    if gamma == 0.0:
        # FAST GATE: strict γ=0 ⇒ bit-for-bit identity (no arithmetic applied).
        return centers
    omega = float(omega)
    if not (omega > 0.0):
        # Degenerate conditional width — no sensible warp; safe no-op.
        return centers
    u = centers - mu
    u_norm = u / omega
    # Pivot-corrected sinh-arcsinh on the CONDITIONAL width ω.  Subtracting sinh(γ)
    # forces f(0)=0 so the pivot mu (and hence the median of the symmetric pre-skew
    # column) is left invariant; only the SHAPE (skew / mean drift) changes.
    # Sign: +γ so γ>0 → positive (right-tail) skewness + positive mean drift.
    warped_norm = np.sinh(np.arcsinh(u_norm) + gamma) - np.sinh(gamma)
    return mu + warped_norm * omega


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------

def measure_znz_response(cat_cut, good_mask, cfg, mm, fine_grid,
                         z_covariate: str = "z_dla",
                         host_col: str = "NHI_TILT_HOST") -> dict:
    """Measure per-detection (xhat, z, dx) arrays from a truth-matched catalog.

    Parameters
    ----------
    cat_cut : astropy Table
        Output of load_and_cut_catalog — carries NHI, Z_DLA, S2N_RED, P_DLA,
        NHI_TRUE / NHI_TILT_HOST.
    good_mask : np.ndarray[bool]
        Per-row good-geometry mask (already on cat_cut).
    cfg : HBIConfig
        Pipeline config (snr_min, p_dla_min).
    mm : MollyMatrix
        Molly matrix (not used here directly; reserved for future SNR cell logic).
    fine_grid : tuple
        (logN_lo, logN_hi, N_b, dN_b) from build_fine_grid(cfg).
    z_covariate : str
        Which z to use; "z_dla" maps to "Z_DLA" column.
    host_col : str
        Column name carrying the true NHI of the matched host absorber.

    Returns
    -------
    dict with keys: "xhat", "z", "dx", "z_covariate"
        All arrays are float64 of the same length (TPs only).
    """
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask

    # predicted NHI (xhat) and matched true NHI (xtrue)
    xhat_pred = np.asarray(cat_cut["NHI"], float)[op]
    true_col = host_col if host_col in cat_cut.colnames else "NHI_TRUE"
    xtrue = np.asarray(cat_cut[true_col], float)[op]

    # z covariate
    z_col_map = {"z_dla": "Z_DLA", "z_qso": "Z_QSO"}
    z_col = z_col_map.get(z_covariate, z_covariate.upper())
    z_all = np.asarray(cat_cut[z_col], float)[op]

    # TPs only: finite true NHI
    tp = np.isfinite(xtrue)
    xhat_tp = xhat_pred[tp]
    xtrue_tp = xtrue[tp]
    z_tp = z_all[tp]
    dx = xhat_tp - xtrue_tp

    return {"xhat": xhat_tp, "z": z_tp, "dx": dx, "z_covariate": z_covariate}


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_znz_model(meas: dict, deg_z: int = 2, deg_xhat: int = 1,
                  fit_median: bool = False, b_mix: float = 1.0,
                  weights: Optional[np.ndarray] = None,
                  xhat_ref: Optional[float] = None,
                  z_ref: Optional[float] = None,
                  fit_skew: bool = False,
                  skew_strength: float = 0.0,
                  skew_xhat_floor: float = 19.5) -> ZNZModel:
    """Fit a 2-D polynomial model for bias b(xhat, z) and scatter sigma(xhat, z).

    Parameters
    ----------
    meas : dict
        Output of measure_znz_response (or hand-constructed test dict) with keys:
        "xhat", "z", "dx", "z_covariate".
    deg_z : int
        Polynomial degree in z (default 2).
    deg_xhat : int
        Polynomial degree in xhat (default 1).
    fit_median : bool
        Also fit the conditional MEDIAN-of-dx surface (Stage III response-FORM
        marginalization). DEFAULT False → ``b_med_coef`` stays None and the model is
        byte-identical to the original (MEAN-only) fit.
    b_mix : float
        Initial response-FORM mix q ∈ [0,1] stored on the model (1.0 = pure MEAN =
        frozen behaviour). Only meaningful when ``fit_median`` is True.
    fit_skew : bool
        Also fit the 2-D skew surface ``γ(x̂,z)`` (``skew_coef``) from the truth-match
        ``dx`` third-moment conditional (Track-C T2, REDUCE-ONLY / NON-CIRCULAR).
        DEFAULT False → ``skew_coef`` stays None → byte-identical to the frozen behaviour.
    skew_strength : float
        Initial γ multiplier stored on the model (0.0 = skew OFF = byte-identical).
        Only meaningful when ``fit_skew`` is True; the frozen point sets it explicitly.
    weights : np.ndarray or None
        Per-row weights (the Stage-III bootstrap multiplicity). None = unweighted
        (byte-identical default).
    xhat_ref, z_ref : float or None
        Fix the polynomial reference point (so a bootstrap re-fit shares the SAME
        centering as the point model — required for the surfaces to be comparable
        across resamples). None = median of THIS sample (the original behaviour).

    Returns
    -------
    ZNZModel
    """
    xhat = np.asarray(meas["xhat"], float)
    z = np.asarray(meas["z"], float)
    dx = np.asarray(meas["dx"], float)
    z_covariate = meas.get("z_covariate", "z_dla")

    if xhat_ref is None:
        xhat_ref = float(np.median(xhat))
    else:
        xhat_ref = float(xhat_ref)
    if z_ref is None:
        z_ref = float(np.median(z))
    else:
        z_ref = float(z_ref)

    # --- fit bias surface b(xhat, z) (MEAN) ---
    b_coef = _poly_fit_2d(xhat, z, dx, xhat_ref, z_ref, deg_xhat, deg_z,
                          weights=weights)

    # --- fit scatter surface: |dx - b_pred| ---
    V = polyvander2d(xhat - xhat_ref, z - z_ref, [deg_xhat, deg_z])
    b_pred = V @ b_coef
    abs_resid = np.abs(dx - b_pred)
    sig_coef = _poly_fit_2d(xhat, z, abs_resid, xhat_ref, z_ref, deg_xhat, deg_z,
                            weights=weights)

    # --- optional MEDIAN-of-dx surface (Stage III response-form axis) ---
    b_med_coef = None
    if fit_median:
        b_med_coef = _quantile_fit_2d(xhat, z, dx, xhat_ref, z_ref, deg_xhat, deg_z,
                                      q=0.5, weights=weights)

    # --- optional SKEW surface γ(x̂,z) (Track-C T2; reduce-only, non-circular) ---
    skew_coef = None
    if fit_skew:
        skew_coef = _skew_fit_2d(xhat, dx, z, xhat_ref, z_ref, deg_xhat, deg_z,
                                 weights=weights, xhat_floor=skew_xhat_floor)

    # evaluate at reference point
    V_ref = polyvander2d(np.array([0.0]), np.array([0.0]), [deg_xhat, deg_z])
    b_ref = float((V_ref @ b_coef)[0])
    sig_ref = float(np.clip((V_ref @ sig_coef)[0], 1e-4, None))

    return ZNZModel(
        b_coef=b_coef, sig_coef=sig_coef,
        xhat_ref=xhat_ref, z_ref=z_ref,
        b_ref=b_ref, sig_ref=sig_ref,
        z_covariate=z_covariate,
        deg_xhat=deg_xhat, deg_z=deg_z,
        b_med_coef=b_med_coef, b_mix=float(b_mix),
        skew_coef=skew_coef, skew_strength=float(skew_strength),
    )


# ---------------------------------------------------------------------------
# Stage III: response (θ_K) RESAMPLE — re-fit the kernel correction per MC draw
# ---------------------------------------------------------------------------

@dataclass
class ResponseFitResample:
    """Resamplable representation of the θ_K (response) fit population.

    Holds the per-TP-detection response arrays ``(xhat, z, dx)`` that ``fit_znz_model``
    regresses, plus ``tid_idx`` — each detection's index into a unique-TID basis
    (``uniq_tids``) so a length-``n_uniq`` per-TID multiplicity (the SAME shared
    bootstrap that re-derives C/ρ/g in Stage II) re-weights the response fit. Re-fitting
    θ_K from this resample with the shared multiplicity makes the response a
    marginalized nuisance JOINTLY correlated with (C, ρ, g) — the load-bearing Stage III
    coverage lever (the truth-band gap is the FROZEN response).

    The point model's reference (``xhat_ref``/``z_ref``) and degrees are stored so every
    resample shares the SAME polynomial centering (the surfaces are then comparable
    across draws and the unit-weight resample reproduces the point model).
    """
    xhat: np.ndarray
    z: np.ndarray
    dx: np.ndarray
    tid_idx: np.ndarray          # detection -> unique-TID basis index
    n_uniq: int
    z_covariate: str
    xhat_ref: float
    z_ref: float
    deg_xhat: int
    deg_z: int


def build_response_fit_resample(meas: dict, det_tids: np.ndarray,
                                uniq_tids: np.ndarray, znz_point: ZNZModel
                                ) -> ResponseFitResample:
    """Build the response-fit resample table aligned to the SHARED unique-TID basis.

    Parameters
    ----------
    meas : dict
        ``measure_znz_response`` output (the TP-detection xhat/z/dx response population).
    det_tids : np.ndarray
        TARGETID of EACH row in ``meas`` (the SAME op + TP cut), length == len(meas[xhat]).
    uniq_tids : np.ndarray
        The shared unique-TID basis (``TruthMatchResample.uniq_tids``) — so the SAME
        per-TID multiplicity re-weights C/ρ/g AND the response fit (joint correlation).
    znz_point : ZNZModel
        The frozen-point model — supplies the reference (xhat_ref/z_ref) + degrees so
        every resample uses the SAME centering (and unit-weight ⇒ the point surfaces).

    Detections whose TID is not in ``uniq_tids`` (none in practice — the response
    population ⊆ the purity population) are mapped to the nearest basis slot and given
    zero effective leverage by construction (their multiplicity comes from a real TID).
    """
    xhat = np.asarray(meas["xhat"], float)
    z = np.asarray(meas["z"], float)
    dx = np.asarray(meas["dx"], float)
    det_tids = np.asarray(det_tids, np.int64)
    if not (len(xhat) == len(z) == len(dx) == len(det_tids)):
        raise ValueError("build_response_fit_resample: meas arrays and det_tids must "
                         f"be equal length; got {len(xhat)}/{len(z)}/{len(dx)}/"
                         f"{len(det_tids)}")
    uniq_tids = np.asarray(uniq_tids, np.int64)
    pos = np.searchsorted(uniq_tids, det_tids)
    pos = np.clip(pos, 0, len(uniq_tids) - 1)
    # membership guard: any response-TID missing from the shared basis is a logic error
    # (the response population is the TP subset of the purity population). Keep only the
    # matched rows (drop unmatched so a stray TID cannot mis-weight the fit).
    matched = uniq_tids[pos] == det_tids
    if not np.all(matched):
        xhat = xhat[matched]; z = z[matched]; dx = dx[matched]; pos = pos[matched]
    return ResponseFitResample(
        xhat=xhat, z=z, dx=dx, tid_idx=pos, n_uniq=len(uniq_tids),
        z_covariate=meas.get("z_covariate", "z_dla"),
        xhat_ref=float(znz_point.xhat_ref), z_ref=float(znz_point.z_ref),
        deg_xhat=int(znz_point.deg_xhat), deg_z=int(znz_point.deg_z))


def refit_znz_from_resample(rfr: ResponseFitResample, boot_mult: np.ndarray,
                            b_mix: float = 1.0,
                            corr_strength: float = 1.0) -> ZNZModel:
    """Re-fit the response model θ_K on a bootstrap resample of the response population.

    ``boot_mult`` (length ``rfr.n_uniq``) is the per-TID multiplicity from the SHARED
    resample (so θ_K is correlated with C/ρ/g — Stage II's ``boot_mult``). ``b_mix`` is
    the response-FORM mix q ∈ [0,1] (1.0 = pure MEAN; 0.0 = conditional MEDIAN — the
    skew-justified axis). ``corr_strength`` is the response-STRENGTH α ∈ [0,1] (1.0 =
    FULL correction; 0.0 = OFF/un-corrected) — the OFF↔corrected axis that BRACKETS truth
    (2026-06-19_track_c_bref_noncircular.md). Returns a per-draw ``ZNZModel`` carrying
    BOTH the mean and median surfaces and the drawn (b_mix, corr_strength); the b/σ
    PARAMETER scatter (re-fit on the resampled rows) AND the FORM ambiguity (q, α) all
    vary per draw.

    The polynomial reference is fixed from the point model so surfaces are comparable
    across draws; at ``boot_mult == 1``, ``b_mix == 1`` AND ``corr_strength == 1`` the
    MEAN surface reproduces the frozen point model's ``b`` (the unit-weight invariance
    Stage III rests on).
    """
    w = np.asarray(boot_mult, float)[rfr.tid_idx]
    meas = {"xhat": rfr.xhat, "z": rfr.z, "dx": rfr.dx, "z_covariate": rfr.z_covariate}
    m = fit_znz_model(meas, deg_z=rfr.deg_z, deg_xhat=rfr.deg_xhat,
                      fit_median=True, b_mix=float(b_mix), weights=w,
                      xhat_ref=rfr.xhat_ref, z_ref=rfr.z_ref)
    m.corr_strength = float(corr_strength)
    return m


# ---------------------------------------------------------------------------
# Completeness model
# ---------------------------------------------------------------------------

def measure_c_nz(cat_cut, truth_cut, cfg, mm, z_edges_fine: np.ndarray,
                 good_mask: Optional[np.ndarray] = None) -> dict:
    """Measure empirical completeness grid g_raw[j_nhi, kz] = n_rec / n_true.

    Parameters
    ----------
    cat_cut : astropy Table
        GP catalog (truth-matched; carries NHI_TRUE, Z_DLA, S2N_RED, P_DLA).
    truth_cut : astropy Table
        Truth absorber catalog (carries NHI and Z_DLA / Z_DLA_NO_RSD / Z).
    cfg : HBIConfig
        Pipeline config (snr_min, p_dla_min, zbins).
    mm : MollyMatrix
        Molly matrix — provides nhi_edges.
    z_edges_fine : np.ndarray
        Fine z-bin edges from _fine_z_grid(cfg).
    good_mask : np.ndarray[bool] or None
        Per-row good-geometry mask (same as passed to measure_znz_response).
        Must be included to make the op-set IDENTICAL to the b-measurement;
        if None, a permissive all-True mask is used (backward-compat only).

    Returns
    -------
    dict with keys: "g_raw", "n_true", "n_rec", "nhi_edges", "z_edges_fine"

    Note on g:
        g lives on the molly nhi_edges grid whose top edge is +inf.  Stage-1
        must map g onto the fine-N axis and must NOT index the +inf top cell for
        any finite N value.  g is smaller than the (N,z) kernel shift but
        non-negligible — it must be carried, not dropped.
    """
    nhi_edges = mm.nhi_edges
    n_nhi = len(nhi_edges) - 1
    n_zf = len(z_edges_fine) - 1

    # --- truth side: count true absorbers per (nhi-cell, z-bin) ---
    t_nhi = np.asarray(truth_cut["NHI"], float)
    z_col = next((c for c in ("Z_DLA", "Z_DLA_NO_RSD", "Z") if c in truth_cut.colnames), None)
    t_z = np.asarray(truth_cut[z_col], float) if z_col else np.zeros(len(truth_cut))

    j_true = np.searchsorted(nhi_edges, t_nhi, side="right") - 1
    k_true = np.searchsorted(z_edges_fine, t_z, side="right") - 1
    j_true = np.clip(j_true, 0, n_nhi - 1)
    k_true = np.clip(k_true, 0, n_zf - 1)

    # only count truth in the z range of the fine grid
    in_zrange = (t_z >= z_edges_fine[0]) & (t_z < z_edges_fine[-1])
    n_true = np.zeros((n_nhi, n_zf), dtype=float)
    for ii in range(len(t_nhi)):
        if in_zrange[ii]:
            n_true[j_true[ii], k_true[ii]] += 1.0

    # --- detected side: recovered TPs among the operating set ---
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    # good_mask must match the b-measurement's op-set exactly (same as measure_znz_response).
    # If not provided, fall back to all-True (backward-compat only — prefer passing it).
    if good_mask is None:
        good_mask = np.ones(len(cat_cut), dtype=bool)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask

    # true NHI of matched TPs
    nhi_true_all = np.asarray(cat_cut["NHI_TRUE"], float)
    z_dla_col = next((c for c in ("Z_DLA", "Z_QSO") if c in cat_cut.colnames), None)
    z_cat = np.asarray(cat_cut[z_dla_col], float) if z_dla_col else np.zeros(len(cat_cut))

    tp_op = op & np.isfinite(nhi_true_all)

    j_rec = np.searchsorted(nhi_edges, nhi_true_all[tp_op], side="right") - 1
    k_rec = np.searchsorted(z_edges_fine, z_cat[tp_op], side="right") - 1
    j_rec = np.clip(j_rec, 0, n_nhi - 1)
    k_rec = np.clip(k_rec, 0, n_zf - 1)

    in_zrange_cat = (z_cat[tp_op] >= z_edges_fine[0]) & (z_cat[tp_op] < z_edges_fine[-1])
    n_rec = np.zeros((n_nhi, n_zf), dtype=float)
    for ii in range(int(np.sum(tp_op))):
        if in_zrange_cat[ii]:
            n_rec[j_rec[ii], k_rec[ii]] += 1.0

    # --- raw completeness ratio ---
    with np.errstate(invalid="ignore", divide="ignore"):
        g_raw = np.where(n_true > 0, n_rec / n_true, np.nan)

    return {
        "g_raw": g_raw,
        "n_true": n_true,
        "n_rec": n_rec,
        "nhi_edges": nhi_edges,
        "z_edges_fine": z_edges_fine,
    }


def fit_c_nz_model(meas_c: dict, smooth: float = 1.0) -> CNZModel:
    """Smooth and normalise the raw completeness grid to produce CNZModel.

    Parameters
    ----------
    meas_c : dict
        Output of measure_c_nz.
    smooth : float
        Gaussian smoothing sigma in z-bin pixels (applied along z axis).

    Returns
    -------
    CNZModel
    """
    g_raw = np.asarray(meas_c["g_raw"], float)
    nhi_edges = np.asarray(meas_c["nhi_edges"], float)
    z_edges_fine = np.asarray(meas_c["z_edges_fine"], float)

    n_nhi, n_zf = g_raw.shape

    # fill NaN cells with row median (or 1.0 if entire row is NaN)
    g_filled = g_raw.copy()
    for j in range(n_nhi):
        row = g_raw[j]
        valid = row[np.isfinite(row)]
        fill = float(np.median(valid)) if len(valid) > 0 else 1.0
        g_filled[j, ~np.isfinite(row)] = fill

    # smooth along z axis
    if smooth > 0:
        g_smooth = gaussian_filter1d(g_filled, sigma=smooth, axis=1,
                                     mode="nearest")
    else:
        g_smooth = g_filled.copy()

    # normalise each row so that g at the reference z column = 1
    # reference column = index of z closest to the median of z_edges midpoints
    z_mids = 0.5 * (z_edges_fine[:-1] + z_edges_fine[1:])
    z_ref = float(np.median(z_mids))
    kz_ref = int(np.argmin(np.abs(z_mids - z_ref)))

    norms = g_smooth[:, kz_ref].copy()
    norms[norms <= 0] = 1.0  # guard against zero
    g_norm = g_smooth / norms[:, np.newaxis]

    # safety clip
    g_norm = np.clip(g_norm, 0.01, 10.0)

    return CNZModel(g_grid=g_norm, nhi_edges=nhi_edges, z_edges_fine=z_edges_fine)


# ---------------------------------------------------------------------------
# Stage-1 application: transform the cached posterior kernel in place
# ---------------------------------------------------------------------------

def _mass_conserving_rebin(src_mass: np.ndarray, src_centers: np.ndarray,
                           edges: np.ndarray) -> np.ndarray:
    """Redistribute a per-bin mass vector whose carriers sit at NEW centers
    ``src_centers`` back onto the histogram defined by ``edges`` (len = n+1),
    conserving total mass.

    Each unit of mass at ``src_centers[j]`` is deposited into the destination bin
    containing it (a clip-into-grid nearest-bin / piecewise-constant rebin). This is
    the mass-conserving linear rebin in the degenerate limit where each source bin is
    treated as a point mass at its (transformed) center — exact for the delta-kernel
    test and a faithful 1st-order rebin for smooth kernels at the fine grid spacing.

    Mass whose transformed center falls outside [edges[0], edges[-1]] is clipped into
    the boundary bin (so Σ is preserved); for the production kernel the fine grid spans
    the full prior so this is a no-op edge guard.

    Parameters
    ----------
    src_mass : (n,) per-source-bin mass.
    src_centers : (n,) transformed center of each source bin.
    edges : (n+1,) destination bin edges (the fine logN grid edges).

    Returns
    -------
    (n,) destination mass vector (Σ == Σ src_mass up to fp).
    """
    n = len(src_mass)
    out = np.zeros(n, dtype=np.float64)
    # destination bin index for each transformed center (clip into grid)
    dest = np.searchsorted(edges, src_centers, side="right") - 1
    dest = np.clip(dest, 0, n - 1)
    np.add.at(out, dest, src_mass)
    return out


def apply_znz_correction(kappa, cat_op, z_edges_fine, logN_lo, logN_hi,
                         znz: "ZNZModel") -> np.ndarray:
    """Transform a cached posterior kernel ``kappa[i, jN, kz]`` IN N-RESPONSE per
    object i and z-bin kz using the conditional (x̂, z) bias/scatter model ``znz``.

    For each object i (covariate z = ``cat_op['zhat'][i]``, x̂ = ``cat_op['xhat'][i]``):
      * target mean   m_tgt = x̂_i − (b(x̂_i, z_i) − b_ref)   (the bias-corrected mean)
      * width scale   s     = sig_ref / σ(x̂_i, z_i)
    Each fine-N bin's mass in column kz is moved so that its center maps as
        mid'_j = m_tgt + s · (mid_j − μ_col)
    where μ_col is the CURRENT mass-weighted mean of that (i, kz) column. This relocates
    the column to the bias-corrected mean m_tgt and width-scales the deviations about the
    column's own current mean by s (so s=1 ⇒ no spread change). The result is re-binned
    onto the fine logN grid (mass-conserving) and renormalized so Σ_jN is preserved per
    (i, kz). The z-axis (kz) is left untouched (the kernel already carries the
    z-distribution; this is an N-only correction). When b(x̂,z)=b_ref, σ=sig_ref AND the
    column is already centred at x̂, the transform is the identity.

    Empty (i, kz) columns (Σ=0) pass through unchanged. The returned array has the SAME
    shape and dtype as ``kappa``.

    NOTE (Phase 1, default-OFF gate): this is only CALLED when ``cfg.kernel_znz_model``
    is set; with the knob None v3x_build_forward never invokes it, so the estimator is
    byte-identical to the broaden012 headline.
    """
    kappa = np.asarray(kappa)
    out_dtype = kappa.dtype
    n_obs, n_nbins, n_zf = kappa.shape
    assert n_nbins == len(logN_lo) == len(logN_hi), (
        f"kappa N-axis {n_nbins} != logN grid {len(logN_lo)}")
    mids = 0.5 * (np.asarray(logN_lo, float) + np.asarray(logN_hi, float))
    edges = np.concatenate([np.asarray(logN_lo, float),
                            [float(logN_hi[-1])]])
    xhat = np.asarray(cat_op["xhat"], float)
    zhat = np.asarray(cat_op["zhat"], float)
    assert len(xhat) == n_obs and len(zhat) == n_obs, (
        f"cat_op xhat/zhat length {len(xhat)}/{len(zhat)} != kappa rows {n_obs}; "
        "the kernel must be row-aligned with the floored op set (cat_op)")

    # per-object bias + scatter (vectorized over objects)
    # Stage III: b_eff mixes the MEAN/MEDIAN surfaces by znz.b_mix (response FORM).
    # With b_mix=1 / no median surface, b_eff == b and b_eff_ref == b_ref EXACTLY, so
    # the default path is byte-identical to the frozen Stage-1 transform.
    if getattr(znz, "b_med_coef", None) is None or float(getattr(znz, "b_mix", 1.0)) >= 1.0 - 1e-12:
        b_i = np.asarray(znz.b(xhat, zhat), float).ravel()
        b_ref = float(znz.b_ref)
    else:
        b_i = np.asarray(znz.b_eff(xhat, zhat), float).ravel()
        b_ref = float(znz.b_eff_ref())
    sig_i = np.asarray(znz.sigma(xhat, zhat), float).ravel()
    sig_ref = float(znz.sig_ref)
    m_tgt = xhat - (b_i - b_ref)                          # bias-corrected target mean
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(sig_i > 0, sig_ref / sig_i, 1.0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    # Stage III response-STRENGTH α (corr_strength): interpolate the WHOLE map toward the
    # IDENTITY (mass left at its own column center = the un-corrected broaden012 kernel).
    # α=1 (DEFAULT) ⇒ the full transform (byte-identical); α=0 ⇒ new_centers == mids ⇒
    # mass UNCHANGED == broaden012-OFF. The OFF↔corrected span is the response-form axis
    # that BRACKETS truth (track_c_bref note: R0≈1.11 OFF ↔ 0.79 corrected). Applied at
    # the new_centers level INSIDE the per-object loop below.
    alpha = float(getattr(znz, "corr_strength", 1.0))

    # Track-C T1: skew warp gate.  skew_coef=None OR skew_strength==0 ⇒ _skew_warp is
    # the identity on every column ⇒ byte-identical to the pre-T1 function.
    _skew_coef = getattr(znz, "skew_coef", None)
    _skew_strength = float(getattr(znz, "skew_strength", 0.0))
    _skew_active = (_skew_coef is not None) and (_skew_strength != 0.0)

    out = np.zeros_like(kappa, dtype=np.float64)
    kf = kappa.astype(np.float64)
    for i in range(n_obs):
        si = float(scale[i]); mt = float(m_tgt[i])
        # Conditional width ω_i for the skew warp: the response σ(x̂_i, z_i) already in
        # the model (NOT the carrier-grid std).  γ acts on (center − m_tgt)/ω_i.
        omega_i = float(sig_i[i])
        # Per-object skew γ_i: eval the 2-D surface at (x̂_i, z_i) scaled by
        # skew_strength. Done once per object (not per kz) since (x̂,z) is the
        # per-detection covariate.
        if _skew_active:
            _xi = float(xhat[i]); _zi = float(zhat[i])
            _V_i = polyvander2d(
                np.array([_xi - znz.xhat_ref]),
                np.array([_zi - znz.z_ref]),
                [znz.deg_xhat, znz.deg_z])
            gamma_i = float((_V_i @ np.asarray(_skew_coef, float))[0]) * _skew_strength
        else:
            gamma_i = 0.0
        for kz in range(n_zf):
            col = kf[i, :, kz]
            tot = col.sum()
            if tot <= 0.0:
                continue                                  # empty column: leave as zeros
            mu_col = float((col * mids).sum() / tot)      # current mass-weighted mean
            # relocate to the bias-corrected mean, width-scale about the current mean
            new_centers = mt + si * (mids - mu_col)
            # Track-C T1: skew warp inserted BETWEEN affine relocate and α-strength
            # interp.  With gamma_i==0 (skew_coef=None or skew_strength==0) the warp
            # is the identity and this call returns new_centers unchanged bit-for-bit.
            if gamma_i != 0.0:
                new_centers = _skew_warp(new_centers, mu=mt, gamma=gamma_i,
                                         omega=omega_i)
            # Stage III α: interpolate toward the identity (mids) so α=0 leaves the column
            # at its own broaden012 center (OFF). α=1 (default) is the full transform.
            if alpha < 1.0 - 1e-12:
                new_centers = (1.0 - alpha) * mids + alpha * new_centers
            rebinned = _mass_conserving_rebin(col, new_centers, edges)
            s = rebinned.sum()
            if s > 0:
                rebinned *= (tot / s)                     # renormalize per (i, kz)
            out[i, :, kz] = rebinned
    return out.astype(out_dtype)


# ===========================================================================
# Track-C T-A: the FORWARD response  p(x̂ | N_true, SNR, z_QSO)
# ===========================================================================
#
# This is a NEW object, DISTINCT from the ZNZModel posterior-warp above.  The
# ZNZModel re-shapes the GP POSTERIOR column p(x_true | x̂); the forward response
# is the LIKELIHOOD / instrument response p(x̂ | N_true) measured by binning the
# truth-match on the TRUE value N_true (NOT on x̂).  Binning on the truth removes
# the Eddington population-mixing (a property of f, not of the instrument), so the
# fit is NON-CIRCULAR (it never sees dN/dX, f, or Ω — only x̂, N_true, SNR, z_QSO).
#
# Measured shape (2026-06-20 eddington-verification, .superpowers/sdd/
# track_c_eddington_verify.md):
#   - RIGHT-skewed: skew(x̂ | N_true) ≈ +0.8…+1.1 across the DLA tier (mid-tier +0.93),
#     COLLAPSING to ~0 above N_true ≈ 21.0 (the 22.5 prior ceiling crowds the up-tail).
#   - width WIDENS at low SNR: std(dx) 0.11 (hi-SNR) → 0.20 (lo-SNR), a ~1.8× swing.
#   - up-bias mean(dx) RISES with z_QSO: +0.036 → +0.074.
# The forward response carries its measured right-skew, SNR-dependent width and z up-bias;
# the parametric per-bin family is the SKEW-NORMAL (its attainable skew ~|0.995| comfortably
# brackets the measured +0.9, UNLIKE the −1.9 posterior the ZNZModel must express).


# Skew-normal attainable skewness ceiling (|skew| → ~0.9953 as a→∞).  Used to clamp the
# moment-match so a noisy high-N cell can never demand an impossible shape parameter.
_SN_SKEW_MAX = 0.5 * (4.0 - np.pi) * (np.sqrt(2.0 / np.pi) ** 3) / \
    (1.0 - 2.0 / np.pi) ** 1.5


def _sn_median0(a: np.ndarray) -> np.ndarray:
    """Standardized skew-normal median m0(a) of SN(loc=0, scale=1, shape=a).

    The median has no elementary closed form; evaluated via the cached scipy
    quantile and a small monotone table + linear interp (mirrors the toy's
    ``_sn_median0``).  Used to convert a median-anchored response into the
    skew-normal location parameter.
    """
    from scipy import stats as _stats
    a = np.asarray(a, float)
    s = np.sign(a)
    aa = np.abs(a)
    if _SN_MED0_GRID[0] is None:           # lazily build the table once
        ag = np.concatenate([[0.0], np.geomspace(0.05, 80.0, 220)])
        mg = np.array([_stats.skewnorm.ppf(0.5, av) for av in ag])
        _SN_MED0_GRID[0] = (ag, mg)
    ag, mg = _SN_MED0_GRID[0]
    return s * np.interp(aa, ag, mg)


_SN_MED0_GRID = [None]   # lazily-populated (a_grid, median_grid) cache


def _moment_to_skewnormal(mean: float, sd: float, skew: float):
    """Map a response (mean, sd, skewness) → skew-normal (xi, omega, a).

    Closed-form inverse of the skew-normal moment relations (reduce-only):
        delta = a/sqrt(1+a^2),  b = sqrt(2/pi)
        skewness = (4-pi)/2 · (b·delta)^3 / (1 − b²delta²)^{3/2}
        omega = sd / sqrt(1 − b²delta²)
        xi    = mean − omega·b·delta
    |skew| is clamped to ~0.995·_SN_SKEW_MAX (the attainable ceiling).  skew≈0 →
    Gaussian (a=0).  Mirrors the certified toy's ``_moment_to_skewnormal``.
    """
    b = np.sqrt(2.0 / np.pi)
    s = float(np.clip(skew, -0.995 * _SN_SKEW_MAX, 0.995 * _SN_SKEW_MAX))
    sd = float(max(sd, 1e-9))
    if abs(s) < 1e-9:
        return float(mean), sd, 0.0
    c = 0.5 * (4.0 - np.pi)
    r = (abs(s) / c) ** (2.0 / 3.0)
    g = r / (1.0 + r)                      # g = (b·delta)^2 ∈ (0,1)
    bdelta = np.sqrt(g)
    delta = float(np.clip(np.sign(s) * bdelta / b, -0.999, 0.999))
    a = delta / np.sqrt(max(1.0 - delta * delta, 1e-12))
    omega = sd / np.sqrt(max(1.0 - (b * delta) ** 2, 1e-12))
    xi = float(mean) - omega * b * delta
    return float(xi), float(omega), float(a)


def _moment_to_skewnormal_vec(mean, sd, skew):
    """VECTORIZED ``_moment_to_skewnormal`` — map per-element (mean, sd, skewness) arrays
    → (xi, omega, a) skew-normal-parameter arrays.

    Identical math to the scalar ``_moment_to_skewnormal``, element-wise (CS minor: the
    per-N loop in ``response_skewnormal`` is too slow at per-sightline × N scale). Matches
    the scalar function to 1e-12 (asserted by the test suite). The ``abs(skew)<1e-9`` Gaussian
    branch is handled by a mask (delta=0 → a=0, omega=sd, xi=mean) so noise-free symmetric
    cells are exact.
    """
    mean = np.asarray(mean, float)
    sd = np.clip(np.asarray(sd, float), 1e-9, None)
    s = np.clip(np.asarray(skew, float), -0.995 * _SN_SKEW_MAX, 0.995 * _SN_SKEW_MAX)
    b = np.sqrt(2.0 / np.pi)
    c = 0.5 * (4.0 - np.pi)
    sym = np.abs(s) < 1e-9                          # Gaussian (a=0) cells
    r = (np.abs(s) / c) ** (2.0 / 3.0)
    g = r / (1.0 + r)                               # g = (b·delta)^2 ∈ (0,1)
    bdelta = np.sqrt(g)
    delta = np.clip(np.sign(s) * bdelta / b, -0.999, 0.999)
    delta = np.where(sym, 0.0, delta)               # symmetric → delta 0
    a = delta / np.sqrt(np.clip(1.0 - delta * delta, 1e-12, None))
    omega = sd / np.sqrt(np.clip(1.0 - (b * delta) ** 2, 1e-12, None))
    xi = mean - omega * b * delta
    # exact Gaussian-branch values where symmetric (mirrors the scalar early-return)
    a = np.where(sym, 0.0, a)
    omega = np.where(sym, sd, omega)
    xi = np.where(sym, mean, xi)
    return xi, omega, a


@dataclass
class ForwardResponseModel:
    """Forward response  p(x̂ | N_true, SNR, z_QSO)  as a per-cell skew-normal whose
    moments are SMOOTH surfaces in N_true, resolved discretely in (SNR, z_QSO).

    The response at a true column N (given an object's SNR-bin ``i_snr`` and z_QSO-bin
    ``i_z``) is the skew-normal with
        mean      = N + μ_b(N, i_snr, i_z)          (μ_b = E[x̂ − N_true], the up-bias)
        sd        = σ(N, i_snr, i_z)                (widens at low SNR)
        skewness  = γ(N, i_snr, i_z)                (right-skew, collapses above N≈21)
    μ_b, σ, γ are each a degree-``deg_N`` polynomial in (N − N_ref) PER (SNR-bin, z-bin)
    cell — i.e. smooth in N_true, piecewise across the ≥3 SNR × 2–3 z_QSO bins (the
    measured ~1.8× width swing axis and the z up-bias axis).

    NON-CIRCULAR by construction: every coefficient is fit from the truth-match
    conditional (x̂, N_true, SNR, z_QSO) only — there is NO dN/dX / f / Ω input anywhere.

    Attributes
    ----------
    mu_coef, sig_coef, skew_coef : (n_snr, n_z, deg_N+1) float arrays
        Per-cell polynomial coefficients for the up-bias, width and skewness surfaces.
    snr_edges : (n_snr+1,) float
        SNR bin edges (the response width axis; ≥3 bins → ≥4 edges).
    z_edges : (n_z+1,) float
        z_QSO bin edges (the up-bias axis; 2–3 bins).
    N_ref : float
        Polynomial reference in N_true (centering, numerical stability).
    deg_N : int
        Polynomial degree in N_true for all three surfaces.
    N_skew_collapse : float
        N_true above which the fitted skew is ramped toward 0 (the prior-ceiling
        collapse — no spurious right-skew extrapolated past N≈21).
    sig_floor : float
        Lower clip on σ (keeps the skew-normal well-defined; default 1e-3 dex).
    z_covariate : str
        Which redshift the (SNR, z) cell axis was binned on: "zqso" (DEFAULT, Z_QSO) or
        "zdla" (Z_DLA, the absorber/detection redshift). A self-describing label so the
        deconvolution conditions each detection's kernel column on the SAME z the model
        was fit with. Default "zqso" keeps the legacy NPZ / build byte-identical.
    emp : EmpiricalForwardDensity or None
        OPTIONAL smoothed-empirical per-cell forward residual density (the A/B
        ``resp_family="empirical"`` path, T-BC FIX).  When present, ``response_density_empirical``
        evaluates p(x̂|N,SNR,z) from a per-(SNR,z) smoothed histogram of the truth-match
        residual r = x̂ − N_true, resolved/interpolated in N_true — so it carries the TRUE
        high-N response SHAPE (the narrowing + skew collapse at N≈21) that the parametric
        skew-normal OVERSHOOTS.  ``None`` ⇒ only the parametric skew-normal density is available
        (backward-compatible; older NPZs load with ``emp=None``).
    """
    mu_coef: np.ndarray
    sig_coef: np.ndarray
    skew_coef: np.ndarray
    snr_edges: np.ndarray
    z_edges: np.ndarray
    N_ref: float
    deg_N: int = 2
    N_skew_collapse: float = 21.0
    sig_floor: float = 1e-3
    emp: Optional["EmpiricalForwardDensity"] = None
    z_covariate: str = "zqso"   # which redshift the (SNR, z) cell axis is binned on:
    #   "zqso" (DEFAULT — Z_QSO, byte-identical to the frozen forward_response_2lpt0.npz) or
    #   "zdla" (Z_DLA, the absorber/detection redshift — Track-C (b) z_DLA-binned response).
    #   This is a LABEL the deconvolution (_build_A_ib_forward) reads to pick the matching
    #   per-detection z covariate; the model's z_edges/coefficients carry the binning itself.
    #   Stage-0 found z_DLA is the causal mean-flux axis (z_QSO redundant, r=0.92); the
    #   z_DLA model conditions the kernel column on the SAME z the per-z reduction resolves.

    # --- cell lookup ---------------------------------------------------------
    def _i_snr(self, snr):
        return np.clip(np.searchsorted(self.snr_edges, np.asarray(snr, float),
                                       side="right") - 1, 0, len(self.snr_edges) - 2)

    def _i_z(self, zqso):
        return np.clip(np.searchsorted(self.z_edges, np.asarray(zqso, float),
                                       side="right") - 1, 0, len(self.z_edges) - 2)

    def _eval_surface(self, coef_grid, N, i_snr, i_z):
        """Evaluate a per-cell N-polynomial surface at (N, i_snr, i_z) (vectorized).

        BROADCAST CONTRACT (CS minor): ``N``, ``i_snr`` and ``i_z`` are each ``ravel()``-ed
        to 1-D and MUST be the SAME length (one (N, SNR-cell, z-cell) triple per element) —
        this is NOT an outer product. The caller (``mu_b``/``sigma``/``skew`` via
        ``_i_snr``/``_i_z``) guarantees this: ``snr``/``zqso`` are first mapped to integer
        cell indices of the same shape as ``N``. Returns a 1-D array of length len(N).
        """
        N = np.asarray(N, float).ravel()
        i_snr = np.asarray(i_snr, int).ravel()
        i_z = np.asarray(i_z, int).ravel()
        dN = N - self.N_ref
        # Vandermonde in N (deg_N+1 columns), pick the per-row cell coefficients.
        V = np.vander(dN, self.deg_N + 1, increasing=True)     # (n, deg_N+1)
        cf = coef_grid[i_snr, i_z, :]                          # (n, deg_N+1)
        return np.einsum("nd,nd->n", V, cf)

    # --- public surfaces -----------------------------------------------------
    def mu_b(self, N, snr, zqso):
        """Up-bias surface μ_b(N,SNR,z) = E[x̂ − N_true]."""
        return self._eval_surface(self.mu_coef, N, self._i_snr(snr), self._i_z(zqso))

    def sigma(self, N, snr, zqso):
        """Width surface σ(N,SNR,z) (>0)."""
        s = self._eval_surface(self.sig_coef, N, self._i_snr(snr), self._i_z(zqso))
        return np.clip(s, self.sig_floor, None)

    def skew(self, N, snr, zqso):
        """Skewness surface γ(N,SNR,z), with the high-N (prior-ceiling) collapse applied.

        Above ``N_skew_collapse`` the skew is linearly ramped to 0 over a 0.5-dex window
        so no spurious right-skew is extrapolated into the saturated high-N regime.

        NOTE (C1 kink at N = N_skew_collapse, default 21.0): the ``np.clip((N−21)/0.5,0,1)``
        ramp is CONTINUOUS but only C0 — its derivative is discontinuous at N=21.0 and at
        N=21.5 (the ramp endpoints). The kernel-build segment integration is over ΔN_seg
        regions and the response is sampled at segment midpoints, so this kink is harmless
        for the forward A-build (no derivative of γ is taken); it only means the recovered
        f(N) has a (tiny) slope feature at 21.0 where the right-skew correction switches off.
        Documented so a future tail-shape audit near 21.0 is not mistaken for a bug.
        """
        N = np.asarray(N, float).ravel()
        g = self._eval_surface(self.skew_coef, N, self._i_snr(snr), self._i_z(zqso))
        g = np.clip(g, -0.995 * _SN_SKEW_MAX, 0.995 * _SN_SKEW_MAX)
        # prior-ceiling collapse: ramp skew → 0 across [N_collapse, N_collapse+0.5]
        ramp = np.clip((N - self.N_skew_collapse) / 0.5, 0.0, 1.0)
        return g * (1.0 - ramp)

    def response_skewnormal(self, N, snr, zqso):
        """Return (xi, omega, a) skew-normal params of p(x̂ | N, SNR, z_QSO) per element.

        The response is centered so its MEAN is N + μ_b (the absolute x̂ location); the
        caller adds nothing — ``xi`` is already the absolute skew-normal location.

        VECTORIZED (CS minor): the (mean, sd, skew) → (xi, omega, a) moment-match runs over
        the whole N array at once via ``_moment_to_skewnormal_vec`` (the per-N scipy/loop was
        too slow at per-sightline × N scale). Matches the scalar loop to 1e-12.
        """
        N = np.asarray(N, float).ravel()
        mean = N + self.mu_b(N, snr, zqso)
        sd = self.sigma(N, snr, zqso)
        sk = self.skew(N, snr, zqso)
        return _moment_to_skewnormal_vec(mean, sd, sk)

    def response_density(self, xhat, N, snr, zqso):
        """Forward-LIKELIHOOD density p(x̂ | N_true, SNR, z_QSO) evaluated AT the observed x̂.

        This is the object the deconvolution kernel A is built from (T-BC): for a detection
        with observed ``xhat`` (a scalar or per-element array), SNR ``snr`` and z_QSO
        ``zqso``, evaluate the skew-normal DENSITY in x̂ as a function of the TRUE N. The
        result is a density in x̂ (∫ over x̂ = 1 per (N,SNR,z)) — it is NOT normalized over N
        (Σ_N ≠ 1); that asymmetry vs the renormalized posterior kappa is the whole reason
        the forward kernel removes the high-N over-recovery. Mirrors the certified toy's
        column build (``build_empirical_fwd_kernel`` deposits the same p(x̂|N) mass).

        ``xhat``, ``N``, ``snr``, ``zqso`` broadcast element-wise (all ravel to 1-D). Returns
        the per-element skew-normal pdf value at ``xhat``.
        """
        from scipy.stats import skewnorm as _skewnorm
        xhat = np.asarray(xhat, float).ravel()
        xi, om, a = self.response_skewnormal(N, snr, zqso)
        return _skewnorm.pdf(xhat, a, loc=xi, scale=om)

    def response_density_empirical(self, xhat, N, snr, zqso):
        """SMOOTHED-EMPIRICAL forward density p(x̂ | N_true, SNR, z_QSO) from ``self.emp``.

        The GENUINE T-BC empirical density: for each detection, evaluate the per-(SNR,z)
        smoothed residual density of r = x̂ − N_true, interpolated in N_true, at the
        residual r = x̂ − N.  This carries the TRUE high-N response SHAPE — the narrowing +
        skew collapse at N≈21 — that the parametric skew-normal moment-fit OVERSHOOTS (the
        Ω lever).  The density is per-unit-x̂ (dr/dx̂ = 1) and integrates to 1 over x̂ at
        fixed N (the normalization convention), so the deconvolution kernel A is the SAME
        marked-Poisson object as the parametric path — only the column SHAPE differs.

        Requires ``self.emp`` (an ``EmpiricalForwardDensity``); raises if ``None`` (the
        loader leaves it None for legacy NPZs, and the build wires it explicitly).

        ``xhat``, ``N``, ``snr``, ``zqso`` broadcast element-wise (ravel to a common length).
        """
        if self.emp is None:
            raise ValueError(
                "response_density_empirical requires the model's empirical density "
                "(ForwardResponseModel.emp); rebuild with build_empirical_forward_density "
                "(or use resp_family='skewnorm').")
        return self.emp.density(xhat, N, snr, zqso)


# =====================================================================
# Smoothed-empirical per-cell forward density (the T-BC Ω lever)
# =====================================================================

@dataclass
class EmpiricalForwardDensity:
    """SMOOTHED-EMPIRICAL forward response p(x̂ | N_true, SNR, z_QSO), stored as a per-cell
    grid of residual (r = x̂ − N_true) densities resolved in N_true.

    This is the GENUINE non-parametric forward density (the T-BC fix, the toy's
    ``build_empirical_fwd_kernel`` analog): for each (SNR-bin, z-bin) cell it carries a
    smoothed/normalized histogram of the truth-match residual r on a SHARED fixed residual
    grid ``r_grid``, evaluated at several N_true anchors.  ``density(x̂,N,snr,z)`` looks up
    the cell, interpolates the residual density LINEARLY in N_true between anchors and
    LINEARLY in r on ``r_grid``, and returns p(x̂|N) = ρ(r = x̂ − N).

    Because the residual density is built per N_true anchor directly from the truth-match
    (NOT moment-matched to a parametric family), it carries the TRUE high-N shape — the
    width narrowing and the right-skew COLLAPSE at N≈21 that the skew-normal over-spreads.

    NON-CIRCULAR: every cell is a histogram of (x̂ − N_true) binned on the TRUE value
    N_true (× SNR × z_QSO) — there is NO dN/dX / f / Ω input.  Reproducible: the build is a
    deterministic histogram + fixed-σ Gaussian smoothing (no RNG).

    Normalization: each per-anchor residual density integrates to 1 over r on ``r_grid``
    (trapezoid), so ∫ p(x̂|N) dx̂ = 1 (the forward-likelihood convention).  It is NEVER
    renormalized over N (Σ_N ≠ 1) — that asymmetry vs the renormalized posterior kappa is
    the whole reason the forward kernel removes the high-N over-recovery.

    Attributes
    ----------
    rho : (n_snr, n_z, n_anchor, n_r) float
        Per-cell, per-N-anchor smoothed residual density on ``r_grid`` (each ∫dr = 1).
    N_anchors : (n_snr, n_z, n_anchor) float
        The N_true anchor centers per cell (ascending; the interpolation knots in N).
    r_grid : (n_r,) float
        Shared residual axis r = x̂ − N_true (uniform; the density is evaluated/interp'd here).
    snr_edges, z_edges : (n_snr+1,), (n_z+1,) float
        Cell edges (SHARED with the parent ForwardResponseModel; same _i_snr/_i_z mapping).
    """
    rho: np.ndarray
    N_anchors: np.ndarray
    r_grid: np.ndarray
    snr_edges: np.ndarray
    z_edges: np.ndarray

    def _i_snr(self, snr):
        return np.clip(np.searchsorted(self.snr_edges, np.asarray(snr, float),
                                       side="right") - 1, 0, len(self.snr_edges) - 2)

    def _i_z(self, zqso):
        return np.clip(np.searchsorted(self.z_edges, np.asarray(zqso, float),
                                       side="right") - 1, 0, len(self.z_edges) - 2)

    def density(self, xhat, N, snr, zqso):
        """p(x̂|N,SNR,z) = ρ_cell(r = x̂ − N), interpolated linearly in N_true and in r.

        ``xhat``, ``N``, ``snr``, ``zqso`` broadcast element-wise (ravel to a common length).
        Out-of-grid residuals (|r| beyond ``r_grid``) evaluate to 0 (the smoothed-empirical
        density has compact support — its tail mass is captured inside the residual window).
        """
        xhat = np.asarray(xhat, float).ravel()
        N = np.asarray(N, float).ravel()
        i_snr = np.asarray(self._i_snr(snr), int).ravel()
        i_z = np.asarray(self._i_z(zqso), int).ravel()
        n = max(xhat.size, N.size, i_snr.size, i_z.size)
        xhat = np.broadcast_to(xhat, (n,))
        N = np.broadcast_to(N, (n,))
        i_snr = np.broadcast_to(i_snr, (n,))
        i_z = np.broadcast_to(i_z, (n,))
        r = xhat - N
        r_grid = self.r_grid
        r0 = float(r_grid[0]); dr = float(r_grid[1] - r_grid[0]); n_r = len(r_grid)
        out = np.zeros(n)
        # group by (i_snr, i_z) cell so the per-anchor interpolation is vectorized per cell
        cells = np.unique(np.stack([i_snr, i_z], axis=1), axis=0)
        for a, b in cells:
            sel = (i_snr == a) & (i_z == b)
            if not sel.any():
                continue
            Ncen = self.N_anchors[a, b]                # (n_anchor,) ascending
            rho_ab = self.rho[a, b]                    # (n_anchor, n_r)
            Nsel = N[sel]; rsel = r[sel]
            # --- linear interp in N_true between anchors (clamped to the anchor range) ---
            ki = np.clip(np.searchsorted(Ncen, Nsel, side="right") - 1, 0,
                         len(Ncen) - 2)
            N0 = Ncen[ki]; N1 = Ncen[ki + 1]
            wN = np.clip((Nsel - N0) / np.where(N1 > N0, N1 - N0, 1.0), 0.0, 1.0)
            # --- linear interp in r on the uniform residual grid ---
            fr = (rsel - r0) / dr
            j0 = np.floor(fr).astype(int)
            inb = (j0 >= 0) & (j0 < n_r - 1)
            wr = np.zeros(rsel.shape); jj = np.zeros(rsel.shape, int)
            wr[inb] = (fr[inb] - j0[inb]); jj[inb] = j0[inb]
            # bilinear over (anchor, r): blend the two bracketing anchor rows then the two r cols
            d_lo = rho_ab[ki, jj] * (1.0 - wr) + rho_ab[ki, jj + 1] * wr
            d_hi = rho_ab[ki + 1, jj] * (1.0 - wr) + rho_ab[ki + 1, jj + 1] * wr
            dens = d_lo * (1.0 - wN) + d_hi * wN
            dens = np.where(inb, dens, 0.0)
            out[sel] = np.clip(dens, 0.0, None)
        return out


def build_empirical_forward_density(meas: dict,
                                    snr_edges=(2.0, 3.5, 6.5, np.inf),
                                    z_edges=(0.0, 2.56, 2.96, np.inf),
                                    n_N_cells: int = 7,
                                    min_count: int = 60,
                                    r_lo: float = -1.5,
                                    r_hi: float = 1.5,
                                    n_r: int = 121,
                                    smooth_dr: float = 0.05,
                                    weights=None) -> "EmpiricalForwardDensity":
    """Build the smoothed-empirical forward residual density from the truth-match (the
    GENUINE T-BC empirical kernel; the toy's ``build_empirical_fwd_kernel`` analog).

    For each (SNR-bin × z-bin) cell, slice N_true into ``n_N_cells`` quantile sub-bins, and
    in each sub-bin form a SMOOTHED, NORMALIZED histogram of the residual r = x̂ − N_true on
    the shared grid ``r_grid`` = linspace(r_lo, r_hi, n_r).  Smoothing is a fixed-width
    Gaussian (σ = ``smooth_dr`` dex) — a reproducible KDE — and each per-anchor density is
    renormalized to ∫dr = 1.  The N_true anchor of a sub-bin is its mean N_true (ascending).

    Cells with too few rows fall back to the SNR-marginalized residual for that z; still-empty
    cells inherit the nearest populated cell (preserves the high-N tail shape, like the toy).
    Sub-bins below ``min_count`` fall back to the cell-pooled residual density.

    REDUCE-ONLY, NON-CIRCULAR: reads ``meas`` (N_true, snr, zqso, dx) only — no dN/dX / f / Ω.
    DETERMINISTIC (no RNG): a histogram + fixed-σ smoothing → reproducible/STAMP-able.

    Parameters mirror ``fit_forward_response``'s binning (``snr_edges``/``z_edges``/
    ``n_N_cells``/``min_count``) so the empirical and parametric paths share cell geometry.

    ``weights`` (Track-C T-D, REDUCE-ONLY): an optional per-object resample multiplicity.
    When given, the histograms become WEIGHTED (np.histogram weights) and the cell/sub-bin
    occupancy uses the EFFECTIVE (weighted) count — the SAME shared boot_mult that re-derives
    C/ρ/g, so the empirical kernel's calibration uncertainty is jointly correlated. Unit
    weights reproduce the unweighted density bit-for-bit. Still NON-CIRCULAR (multiplicity).
    """
    N_true = np.asarray(meas["N_true"], float)
    snr = np.asarray(meas["snr"], float)
    zqso = np.asarray(meas["zqso"], float)
    dx = np.asarray(meas["dx"], float)
    W = (np.ones(len(N_true)) if weights is None
         else np.asarray(weights, float))
    snr_edges = np.asarray(snr_edges, float)
    z_edges = np.asarray(z_edges, float)
    n_snr = len(snr_edges) - 1
    n_z = len(z_edges) - 1
    r_grid = np.linspace(r_lo, r_hi, int(n_r))
    dr = float(r_grid[1] - r_grid[0])
    sigma_bins = max(smooth_dr / dr, 1e-6)

    i_snr = np.clip(np.searchsorted(snr_edges, snr, "right") - 1, 0, n_snr - 1)
    i_z = np.clip(np.searchsorted(z_edges, zqso, "right") - 1, 0, n_z - 1)
    fin = np.isfinite(zqso) & np.isfinite(N_true) & np.isfinite(dx)

    def _smoothed_density(dvals, wvals=None):
        """Normalized, Gaussian-smoothed residual density on r_grid from raw residuals.
        ``wvals`` (per-residual weight) makes the histogram the resampled one; None ⇒ unit."""
        if dvals.size == 0:
            return None
        edges = np.concatenate([[r_grid[0] - 0.5 * dr],
                                0.5 * (r_grid[:-1] + r_grid[1:]),
                                [r_grid[-1] + 0.5 * dr]])
        hist, _ = np.histogram(dvals, bins=edges, weights=wvals)
        h = gaussian_filter1d(hist.astype(float), sigma_bins, mode="constant")
        area = _trapz(h, r_grid)
        if area <= 0:
            return None
        return h / area

    # pooled-per-cell density (the sub-bin fallback) + the cell N-anchor grid
    rho = np.zeros((n_snr, n_z, int(n_N_cells), int(n_r)))
    N_anchors = np.zeros((n_snr, n_z, int(n_N_cells)))
    # SNR-marginalized (per-z) pooled density, used when a whole SNR cell is too sparse
    z_pool_dens = {}
    for b in range(n_z):
        sel = fin & (i_z == b)
        z_pool_dens[b] = (_smoothed_density(dx[sel], W[sel]) if sel.any() else None)
    global_pool = _smoothed_density(dx[fin], W[fin])

    cell_built = np.zeros((n_snr, n_z), bool)
    for a in range(n_snr):
        for b in range(n_z):
            sel = fin & (i_snr == a) & (i_z == b)
            Ns = N_true[sel]; ds = dx[sel]; ws = W[sel]
            n_eff = float(np.sum(ws))
            cell_pool = (_smoothed_density(ds, ws) if n_eff >= min_count
                         else (z_pool_dens[b] if z_pool_dens[b] is not None
                               else global_pool))
            if n_eff >= min_count and Ns.size:
                qs = np.unique(np.quantile(Ns, np.linspace(0, 1, int(n_N_cells) + 1)))
                qs[-1] = np.nextafter(qs[-1], np.inf) if len(qs) >= 2 else qs[-1]
            else:
                qs = np.array([])
            anchors = []
            dens_rows = []
            for k in range(int(n_N_cells)):
                if len(qs) >= k + 2:
                    m = (Ns >= qs[k]) & (Ns < qs[k + 1])
                    if float(np.sum(ws[m])) >= min_count:
                        d_k = _smoothed_density(ds[m], ws[m])
                        if d_k is not None:
                            anchors.append(float(np.sum(ws[m] * Ns[m]) / np.sum(ws[m])))
                            dens_rows.append(d_k)
                            continue
                # sparse sub-bin → pooled cell density at a spread-out anchor (filled below)
            if anchors:
                # fill remaining anchor slots by extending the populated anchor range with
                # the cell-pooled density (preserves the cell's bulk where N sub-bins are
                # sparse, while the populated anchors carry the per-N shape — incl. high-N).
                cell_built[a, b] = True
                anchors = np.asarray(anchors, float)
                dens_rows = np.asarray(dens_rows, float)
                # pad to n_N_cells anchors so the stored array is regular: append anchors
                # just above the top populated one with the pooled density (flat extrapolation)
                while len(anchors) < int(n_N_cells):
                    anchors = np.append(anchors, anchors[-1] + 0.1)
                    dens_rows = np.vstack([dens_rows,
                                           cell_pool if cell_pool is not None
                                           else dens_rows[-1]])
                rho[a, b] = dens_rows
                N_anchors[a, b] = anchors
            else:
                # whole cell too sparse: a single pooled density spread over the anchor grid
                pool = (cell_pool if cell_pool is not None else
                        (global_pool if global_pool is not None
                         else np.full(int(n_r), 1.0 / (r_grid[-1] - r_grid[0]))))
                N_anchors[a, b] = np.linspace(19.5, 22.0, int(n_N_cells))
                rho[a, b] = np.tile(pool, (int(n_N_cells), 1))

    # still-empty cells (never built and no pool) inherit the nearest BUILT cell
    if cell_built.any() and not cell_built.all():
        built_idx = np.argwhere(cell_built)
        for a in range(n_snr):
            for b in range(n_z):
                if cell_built[a, b]:
                    continue
                d = np.abs(built_idx[:, 0] - a) + np.abs(built_idx[:, 1] - b)
                an, bn = built_idx[int(np.argmin(d))]
                rho[a, b] = rho[an, bn]
                N_anchors[a, b] = N_anchors[an, bn]

    return EmpiricalForwardDensity(
        rho=rho, N_anchors=N_anchors, r_grid=r_grid,
        snr_edges=snr_edges, z_edges=z_edges,
    )


def measure_forward_response(cat_cut, good_mask, cfg,
                             host_col: str = "NHI_TILT_HOST",
                             xhat_floor: float = 19.5,
                             z_covariate: str = "zqso") -> dict:
    """Measure the per-detection forward-response arrays (N_true, SNR, z, dx).

    Reduce-only.  Replicates the EXACT op-set of measure_znz_response / the eddington
    verification (lya-only, S2N_RED > snr_min, P_DLA > p_dla_min, good_mask), keeps TPs
    (finite matched true NHI) with x̂ ≥ ``xhat_floor``, and returns the conditioning
    variables for the FORWARD fit binned on the TRUE value.

    NON-CIRCULAR: reads ONLY x̂ (``NHI``), the matched true host NHI (``host_col``),
    ``S2N_RED`` and the redshift column (``Z_QSO`` or ``Z_DLA``) — never a reduced
    population statistic.

    Parameters
    ----------
    z_covariate : str
        Which redshift the forward (SNR, z) cell axis conditions on:
          "zqso" (DEFAULT) → Z_QSO, byte-identical to the frozen forward_response_2lpt0.npz;
          "zdla"           → Z_DLA, the absorber/detection redshift (Track-C (b), the causal
                             mean-flux axis the per-z dN/dX(z) reduction is resolved in).
        Stored back in the returned dict's ``z_covariate`` key so ``fit_forward_response``
        stamps the matching label onto the ForwardResponseModel.

    Returns
    -------
    dict with keys
        "N_true"      : matched true host log NHI (the binning value)
        "snr"         : S2N_RED of the detection
        "zqso"        : the chosen redshift covariate of the detection (Z_QSO or Z_DLA per
                        ``z_covariate``).  The dict key stays ``"zqso"`` (the internal field
                        name the fit/empirical builders consume) regardless of which column
                        it carries — ``z_covariate`` records WHICH redshift it actually is.
        "dx"          : x̂ − N_true   (the response residual; right-skewed up-bias)
        "xhat"        : the predicted log NHI (kept for cross-checks)
        "z_covariate" : the redshift the ``"zqso"`` array actually carries ("zqso"/"zdla")
    """
    z_covariate = str(z_covariate).lower()
    z_col_map = {"zqso": "Z_QSO", "z_qso": "Z_QSO", "zdla": "Z_DLA", "z_dla": "Z_DLA"}
    z_col = z_col_map.get(z_covariate, "Z_QSO")
    # normalize the stored label to the canonical short form
    z_covariate = "zdla" if z_col == "Z_DLA" else "zqso"

    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask

    xhat = np.asarray(cat_cut["NHI"], float)[op]
    true_col = host_col if host_col in cat_cut.colnames else "NHI_TRUE"
    xtrue = np.asarray(cat_cut[true_col], float)[op]
    snr = s2n[op]
    zqso = (np.asarray(cat_cut[z_col], float)[op]
            if z_col in cat_cut.colnames else np.full(int(op.sum()), np.nan))

    tp = np.isfinite(xtrue)
    xhat, xtrue, snr, zqso = xhat[tp], xtrue[tp], snr[tp], zqso[tp]
    keep = xhat >= float(xhat_floor)
    xhat, xtrue, snr, zqso = xhat[keep], xtrue[keep], snr[keep], zqso[keep]

    return {"N_true": xtrue, "snr": snr, "zqso": zqso,
            "dx": xhat - xtrue, "xhat": xhat, "z_covariate": z_covariate}


def _empirical_forward_cells(N_true, snr, zqso, dx,
                             snr_edges, z_edges,
                             n_N_cells: int = 7, min_count: int = 60,
                             weights=None):
    """Measure the SMOOTHED-EMPIRICAL per-bin forward response in (N_true × SNR × z_QSO).

    For every (SNR-bin, z-bin) cell, slice N_true into ``n_N_cells`` quantile sub-bins and
    measure the per-sub-bin mean(dx), std(dx) and skewness(dx) — the MEASURED forward
    response the parametric fit must reproduce.  Returns a structured dict of per-cell
    arrays used both to fit the surfaces and as the cross-check ("smoothed-empirical").

    Returns a dict: for each (i_snr, i_z) key a list of (N_center, mean, sd, skew, count)
    sub-bin tuples (sub-bins with < ``min_count`` rows dropped).

    ``weights`` (Track-C T-D, REDUCE-ONLY): an optional per-object resample multiplicity.
    When given, the cell/sub-bin membership counts and the per-sub-bin moments become
    WEIGHTED (the bootstrap-resampled statistic; the SAME shared boot_mult that re-derives
    C/ρ/g so the kernel-calibration uncertainty is jointly correlated). Unit weights
    reproduce the unweighted result bit-for-bit (each row contributes 1×). It is a
    multiplicity, NOT a reduced statistic — non-circularity is preserved.
    """
    N_true = np.asarray(N_true, float); snr = np.asarray(snr, float)
    zqso = np.asarray(zqso, float); dx = np.asarray(dx, float)
    w = (np.ones(len(N_true)) if weights is None
         else np.asarray(weights, float))
    i_snr = np.clip(np.searchsorted(snr_edges, snr, "right") - 1, 0, len(snr_edges) - 2)
    i_z = np.clip(np.searchsorted(z_edges, zqso, "right") - 1, 0, len(z_edges) - 2)
    n_snr = len(snr_edges) - 1
    n_z = len(z_edges) - 1
    out = {}
    for a in range(n_snr):
        for b in range(n_z):
            sel = (i_snr == a) & (i_z == b) & np.isfinite(zqso)
            # cell occupancy is the EFFECTIVE (weighted) count when resampling
            if float(np.sum(w[sel])) < min_count:
                out[(a, b)] = []
                continue
            Ns = N_true[sel]; ds = dx[sel]; ws = w[sel]
            qs = np.unique(np.quantile(Ns, np.linspace(0, 1, n_N_cells + 1)))
            if len(qs) < 2:
                out[(a, b)] = []
                continue
            qs[-1] = np.nextafter(qs[-1], np.inf)
            cells = []
            for k in range(len(qs) - 1):
                m = (Ns >= qs[k]) & (Ns < qs[k + 1])
                wm = ws[m]
                c = float(np.sum(wm))
                if c < min_count:
                    continue
                di = ds[m]
                m1 = float(np.sum(wm * di) / c)
                dd = di - m1
                m2 = float(np.sum(wm * dd * dd) / c)
                if m2 <= 0:
                    continue
                m3 = float(np.sum(wm * dd * dd * dd) / c)
                cells.append((float(np.sum(wm * Ns[m]) / c), m1, np.sqrt(m2),
                              m3 / m2 ** 1.5, c))
            out[(a, b)] = cells
    return out


def fit_forward_response(meas: dict,
                         snr_edges=(2.0, 3.5, 6.5, np.inf),
                         z_edges=(0.0, 2.56, 2.96, np.inf),
                         deg_N: int = 2,
                         n_N_cells: int = 7,
                         min_count: int = 60,
                         N_skew_collapse: float = 21.0,
                         N_ref: Optional[float] = None,
                         build_empirical: bool = True,
                         weights=None) -> ForwardResponseModel:
    """Fit the FORWARD response p(x̂ | N_true, SNR, z_QSO) — a per-cell skew-normal whose
    (up-bias, width, skew) moments are SMOOTH polynomials in N_true, resolved across
    ≥3 SNR × 2–3 z_QSO bins.

    REDUCE-ONLY, NON-CIRCULAR (asserted by the test suite): ``meas`` carries ONLY the
    truth-match conditional (N_true, snr, zqso, dx); there is NO dN/dX / f / Ω argument.
    The dN/dX/Ω reductions are strictly-downstream CHECKS computed after this model is
    frozen — the α=1/R0 tautology is structurally impossible because no reduced statistic
    enters the fit.

    The ``"zqso"`` array of ``meas`` is the chosen redshift covariate (Z_QSO by default, or
    Z_DLA when ``measure_forward_response(..., z_covariate="zdla")`` was used).  The label
    ``meas["z_covariate"]`` (default "zqso") is stamped onto the returned model's
    ``z_covariate`` so the deconvolution conditions each detection on the SAME redshift.

    Procedure
    ---------
    1. Slice the truth-match into (SNR × z_QSO × N_true) sub-bins; measure the per-sub-bin
       mean(dx), std(dx), skewness(dx) — the smoothed-empirical forward response.
    2. Per (SNR, z) cell, weighted-least-squares fit a degree-``deg_N`` polynomial in
       N_true to each of mean / std / skew over the populated N sub-bins.  Cells with too
       few populated sub-bins fall back to a CONSTANT (the cell mean / pooled value).
    3. The high-N (≳21) skew collapse is applied at EVALUATION time
       (``ForwardResponseModel.skew`` ramps γ→0 above ``N_skew_collapse``) so the
       fit never extrapolates a spurious right-skew into the prior-ceiling regime.

    Parameters
    ----------
    meas : dict
        Output of ``measure_forward_response`` (keys N_true, snr, zqso, dx).
    snr_edges : sequence
        SNR bin edges (≥4 → ≥3 bins).  Default tertile-like (2,3.5,6.5,∞) matching the
        measured ~1.8× width swing (hi/mid/lo SNR).
    z_edges : sequence
        z_QSO bin edges (default (0,2.56,2.96,∞) → 3 bins matching the measured z up-bias
        tertiles +0.036/+0.050/+0.074).
    deg_N, n_N_cells, min_count : int
        Polynomial degree in N, N sub-bins per cell, per-sub-bin minimum count.
    N_skew_collapse : float
        N above which skew is ramped to 0 (carried onto the model).
    N_ref : float or None
        Polynomial reference in N (median N_true if None).
    build_empirical : bool
        If True (default) ALSO build the smoothed-empirical per-cell residual density
        (``EmpiricalForwardDensity``) and attach it as ``ForwardResponseModel.emp`` — the
        genuine T-BC ``resp_family="empirical"`` path (carries the true high-N shape the
        parametric skew-normal overshoots).  The parametric (skewnorm) surfaces are built
        regardless; this only adds the optional empirical density.  Set False for a
        parametric-only model (smaller NPZ; ``emp=None``).
    weights : array or None
        Per-object resample multiplicity (Track-C T-D). When given, the per-cell/sub-bin
        moments AND the empirical density become WEIGHTED — the SAME shared boot_mult that
        re-derives C/ρ/g per MC draw, so the forward kernel's calibration uncertainty enters
        the marginalized band JOINTLY correlated with the other nuisances. REDUCE-ONLY (a
        multiplicity, not a reduced statistic): unit weights reproduce the point fit
        bit-for-bit; non-circularity is preserved.

    Returns
    -------
    ForwardResponseModel
    """
    N_true = np.asarray(meas["N_true"], float)
    snr = np.asarray(meas["snr"], float)
    zqso = np.asarray(meas["zqso"], float)
    dx = np.asarray(meas["dx"], float)
    W = (np.ones(len(N_true)) if weights is None
         else np.asarray(weights, float))
    snr_edges = np.asarray(snr_edges, float)
    z_edges = np.asarray(z_edges, float)
    n_snr = len(snr_edges) - 1
    n_z = len(z_edges) - 1
    if N_ref is None:
        N_ref = float(np.median(N_true))
    else:
        N_ref = float(N_ref)

    cells = _empirical_forward_cells(N_true, snr, zqso, dx, snr_edges, z_edges,
                                     n_N_cells=n_N_cells, min_count=min_count,
                                     weights=W)

    n_coef = deg_N + 1
    mu_coef = np.zeros((n_snr, n_z, n_coef))
    sig_coef = np.zeros((n_snr, n_z, n_coef))
    skew_coef = np.zeros((n_snr, n_z, n_coef))

    # global pooled fallbacks (used when a cell has too few populated sub-bins) — WEIGHTED
    # so the unit-weight path is byte-identical and a resample shifts the fallback too.
    Wsum = float(np.sum(W))
    pooled_mu = float(np.sum(W * dx) / Wsum) if Wsum > 0 else 0.0
    pdd = dx - pooled_mu
    pooled_var = float(np.sum(W * pdd * pdd) / Wsum) if Wsum > 0 else 0.0
    pooled_sd = float(np.sqrt(pooled_var)) if pooled_var > 0 else 0.1
    pooled_sk = 0.0
    if Wsum > 2 and pooled_sd > 0:
        pooled_sk = float(np.sum(W * pdd ** 3) / Wsum / pooled_sd ** 3)

    def _fit_poly(Nc, yc, wc, fallback):
        """Weighted poly-in-N fit; degrade to a lower degree / constant when too sparse."""
        Nc = np.asarray(Nc, float); yc = np.asarray(yc, float)
        wc = np.asarray(wc, float)
        c = np.zeros(n_coef)
        npts = len(Nc)
        if npts == 0:
            c[0] = fallback
            return c
        if npts == 1:
            c[0] = float(yc[0])
            return c
        use_deg = min(deg_N, npts - 1)
        V = np.vander(Nc - N_ref, use_deg + 1, increasing=True)
        sw = np.sqrt(np.clip(wc, 0.0, None))
        coef, _, _, _ = np.linalg.lstsq(V * sw[:, None], yc * sw, rcond=None)
        c[:use_deg + 1] = coef
        return c

    for a in range(n_snr):
        for b in range(n_z):
            cl = cells.get((a, b), [])
            if len(cl) == 0:
                mu_coef[a, b, 0] = pooled_mu
                sig_coef[a, b, 0] = pooled_sd
                skew_coef[a, b, 0] = pooled_sk
                continue
            Nc = np.array([c[0] for c in cl])
            mn = np.array([c[1] for c in cl])
            sd = np.array([c[2] for c in cl])
            sk = np.array([c[3] for c in cl])
            wt = np.array([c[4] for c in cl], float)
            mu_coef[a, b] = _fit_poly(Nc, mn, wt, pooled_mu)
            sig_coef[a, b] = _fit_poly(Nc, sd, wt, pooled_sd)
            # clip per-cell skewness to the SN ceiling before the smooth fit (a single
            # under-populated sub-bin can throw |skew|>2 from sample noise).
            sk_cap = np.clip(sk, -0.995 * _SN_SKEW_MAX, 0.995 * _SN_SKEW_MAX)
            skew_coef[a, b] = _fit_poly(Nc, sk_cap, wt, pooled_sk)

    emp = None
    if build_empirical:
        emp = build_empirical_forward_density(
            meas, snr_edges=snr_edges, z_edges=z_edges,
            n_N_cells=n_N_cells, min_count=min_count, weights=W)

    # carry the z-covariate label measure_forward_response stamped (default "zqso" ⇒
    # byte-identical to the legacy build / the frozen forward_response_2lpt0.npz).
    z_cov = str(meas.get("z_covariate", "zqso")).lower()
    z_cov = "zdla" if z_cov in ("zdla", "z_dla") else "zqso"

    return ForwardResponseModel(
        mu_coef=mu_coef, sig_coef=sig_coef, skew_coef=skew_coef,
        snr_edges=snr_edges, z_edges=z_edges, N_ref=N_ref, deg_N=int(deg_N),
        N_skew_collapse=float(N_skew_collapse), emp=emp, z_covariate=z_cov,
    )


# ---------------------------------------------------------------------------
# Track-C T-D: resample-aware FORWARD-response refit (kernel-calibration uncertainty
# carry).  Mirrors ResponseFitResample / build_response_fit_resample /
# refit_znz_from_resample, but for the ForwardResponseModel (p(x̂|N_true,SNR,z_QSO)).
# Per MC draw the SAME shared boot_mult that re-derives C/ρ/g also re-fits the forward
# response, so the empirical-kernel jitter enters the inner covariance (the toy's flagged
# over-confidence fix at high count).
# ---------------------------------------------------------------------------
@dataclass
class ForwardResponseFitResample:
    """Resamplable representation of the FORWARD-response fit population.

    Holds the per-TP-detection forward-response arrays ``(N_true, snr, zqso, dx)`` that
    ``fit_forward_response`` regresses, plus ``tid_idx`` — each detection's index into a
    unique-TID basis (``uniq_tids``) so a length-``n_uniq`` per-TID multiplicity (the SAME
    shared bootstrap that re-derives C/ρ/g in Stage II) re-weights the forward fit. The
    point-model binning (snr/z edges, deg_N, N_ref, N_skew_collapse, n_N_cells, min_count,
    build_empirical) is stored so every resample shares the SAME cell geometry/centering and
    the unit-weight resample reproduces the point model.
    """
    N_true: np.ndarray
    snr: np.ndarray
    zqso: np.ndarray
    dx: np.ndarray
    tid_idx: np.ndarray          # detection -> unique-TID basis index
    n_uniq: int
    snr_edges: np.ndarray
    z_edges: np.ndarray
    deg_N: int
    N_ref: float
    N_skew_collapse: float
    n_N_cells: int
    min_count: int
    build_empirical: bool
    z_covariate: str = "zqso"    # the redshift the ``zqso`` array carries (point-model label)


def build_forward_response_fit_resample(
        meas: dict, det_tids: np.ndarray, uniq_tids: np.ndarray,
        frm_point: ForwardResponseModel,
        n_N_cells: int = 7, min_count: int = 60,
        build_empirical: bool = True) -> ForwardResponseFitResample:
    """Build the forward-response-fit resample table aligned to the SHARED unique-TID basis.

    Parameters mirror ``build_response_fit_resample``:
      meas       : ``measure_forward_response`` output (N_true/snr/zqso/dx) — the TP-detection
                   forward population, ALL rows carrying a matched true N_true.
      det_tids   : TARGETID of each row in ``meas`` (same op + TP cut), len == len(meas[dx]).
      uniq_tids  : the shared unique-TID basis (``TruthMatchResample.uniq_tids``) so the SAME
                   per-TID multiplicity re-weights C/ρ/g AND the forward fit.
      frm_point  : the frozen-point ForwardResponseModel — supplies the binning
                   (snr/z edges, deg_N, N_ref, collapse) so a unit-weight resample reproduces
                   the point surfaces.
      n_N_cells, min_count, build_empirical : the point-fit knobs (must match the point fit).

    Detections whose TID is not in ``uniq_tids`` (none in practice — the forward population ⊆
    the purity population) are dropped (matched-only), exactly as build_response_fit_resample.
    """
    N_true = np.asarray(meas["N_true"], float)
    snr = np.asarray(meas["snr"], float)
    zqso = np.asarray(meas["zqso"], float)
    dx = np.asarray(meas["dx"], float)
    det_tids = np.asarray(det_tids, np.int64)
    if not (len(N_true) == len(snr) == len(zqso) == len(dx) == len(det_tids)):
        raise ValueError("build_forward_response_fit_resample: meas arrays and det_tids "
                         f"must be equal length; got {len(N_true)}/{len(snr)}/{len(zqso)}/"
                         f"{len(dx)}/{len(det_tids)}")
    uniq_tids = np.asarray(uniq_tids, np.int64)
    pos = np.searchsorted(uniq_tids, det_tids)
    pos = np.clip(pos, 0, len(uniq_tids) - 1)
    matched = uniq_tids[pos] == det_tids
    if not np.all(matched):
        N_true = N_true[matched]; snr = snr[matched]
        zqso = zqso[matched]; dx = dx[matched]; pos = pos[matched]
    return ForwardResponseFitResample(
        N_true=N_true, snr=snr, zqso=zqso, dx=dx, tid_idx=pos,
        n_uniq=len(uniq_tids),
        snr_edges=np.asarray(frm_point.snr_edges, float),
        z_edges=np.asarray(frm_point.z_edges, float),
        deg_N=int(frm_point.deg_N), N_ref=float(frm_point.N_ref),
        N_skew_collapse=float(frm_point.N_skew_collapse),
        n_N_cells=int(n_N_cells), min_count=int(min_count),
        build_empirical=bool(build_empirical),
        z_covariate=str(getattr(frm_point, "z_covariate", "zqso")))


def refit_forward_response_from_resample(
        rfr: ForwardResponseFitResample, boot_mult: np.ndarray) -> ForwardResponseModel:
    """Re-fit the FORWARD response on a bootstrap resample of the forward population.

    ``boot_mult`` (length ``rfr.n_uniq``) is the per-TID multiplicity from the SHARED
    resample (so the forward kernel is correlated with C/ρ/g — Stage II's ``boot_mult``).
    Expands the per-TID multiplicity to per-detection weights via ``rfr.tid_idx`` (the
    refit_znz_from_resample pattern), re-fits ``fit_forward_response`` with those weights,
    reusing the point-model binning/centering. At ``boot_mult == 1`` the surfaces reproduce
    the frozen point model (the unit-weight invariance the marginalized band rests on).

    REDUCE-ONLY / NON-CIRCULAR: ``boot_mult`` is a multiplicity; no dN/dX/f/Ω enters.
    """
    w = np.asarray(boot_mult, float)[rfr.tid_idx]
    meas = {"N_true": rfr.N_true, "snr": rfr.snr, "zqso": rfr.zqso, "dx": rfr.dx,
            "xhat": rfr.N_true + rfr.dx,
            "z_covariate": str(getattr(rfr, "z_covariate", "zqso"))}
    return fit_forward_response(
        meas, snr_edges=rfr.snr_edges, z_edges=rfr.z_edges,
        deg_N=rfr.deg_N, n_N_cells=rfr.n_N_cells, min_count=rfr.min_count,
        N_skew_collapse=rfr.N_skew_collapse, N_ref=rfr.N_ref,
        build_empirical=rfr.build_empirical, weights=w)


def save_forward_response(path: str, frm: ForwardResponseModel) -> None:
    """Save a ForwardResponseModel to its OWN NPZ envelope (distinct from save_znz).

    Backward-compat: ``load_forward_response`` restores all fields; the envelope is
    self-describing (carries deg_N / collapse / sig_floor) so a future loader needs no
    external metadata.
    """
    payload = dict(
        _fwd_response_kind=np.array("skewnormal_per_cell"),
        mu_coef=frm.mu_coef,
        sig_coef=frm.sig_coef,
        skew_coef=frm.skew_coef,
        snr_edges=frm.snr_edges,
        z_edges=frm.z_edges,
        N_ref=np.array(frm.N_ref),
        deg_N=np.array(frm.deg_N),
        N_skew_collapse=np.array(frm.N_skew_collapse),
        sig_floor=np.array(frm.sig_floor),
        z_covariate=np.array(getattr(frm, "z_covariate", "zqso")),
    )
    # OPTIONAL empirical density block (T-BC resp_family='empirical'). Written ONLY when
    # present so a parametric-only model stays byte-compatible with the legacy envelope;
    # the loader keys off ``emp_rho`` (absent ⇒ emp=None, backward-compatible).
    if frm.emp is not None:
        payload.update(
            emp_rho=frm.emp.rho,
            emp_N_anchors=frm.emp.N_anchors,
            emp_r_grid=frm.emp.r_grid,
            emp_snr_edges=frm.emp.snr_edges,
            emp_z_edges=frm.emp.z_edges,
        )
    np.savez(path, **payload)


def load_forward_response(path: str) -> ForwardResponseModel:
    """Load a ForwardResponseModel saved by ``save_forward_response``.

    Backward-compatible: legacy NPZs without the empirical block load with ``emp=None``
    (only the parametric skew-normal density is available); newer NPZs restore the
    ``EmpiricalForwardDensity`` for the ``resp_family='empirical'`` path.
    """
    d = np.load(path, allow_pickle=True)
    emp = None
    if "emp_rho" in d:
        emp = EmpiricalForwardDensity(
            rho=d["emp_rho"],
            N_anchors=d["emp_N_anchors"],
            r_grid=d["emp_r_grid"],
            snr_edges=d["emp_snr_edges"],
            z_edges=d["emp_z_edges"],
        )
    return ForwardResponseModel(
        mu_coef=d["mu_coef"],
        sig_coef=d["sig_coef"],
        skew_coef=d["skew_coef"],
        snr_edges=d["snr_edges"],
        z_edges=d["z_edges"],
        N_ref=float(d["N_ref"]),
        deg_N=int(d["deg_N"]),
        N_skew_collapse=float(d["N_skew_collapse"]) if "N_skew_collapse" in d else 21.0,
        sig_floor=float(d["sig_floor"]) if "sig_floor" in d else 1e-3,
        emp=emp,
        # legacy NPZs (no z_covariate key) load as "zqso" → byte-identical behaviour.
        z_covariate=(str(d["z_covariate"]) if "z_covariate" in d else "zqso"),
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_znz(path: str, znz: ZNZModel, cnz: CNZModel) -> None:
    """Save both models to a single NPZ file.

    Keys: b_coef, sig_coef, xhat_ref, z_ref, b_ref, sig_ref, z_covariate,
          deg_xhat, deg_z, [b_med_coef, b_mix,] [skew_coef, skew_strength,]
          g_grid, nhi_edges, z_edges_fine

    The optional Stage-III (``b_med_coef``/``b_mix``) and Track-C (``skew_coef``/
    ``skew_strength``) blocks are written only when present on ``znz``; loaders that
    pre-date them ignore the extra keys, and ``load_znz`` restores the byte-identical
    default (None / 1.0 / 0.0) when they are absent.
    """
    extra = {}
    if getattr(znz, "b_med_coef", None) is not None:
        extra["b_med_coef"] = np.asarray(znz.b_med_coef)
    extra["b_mix"] = np.array(float(getattr(znz, "b_mix", 1.0)))
    if getattr(znz, "skew_coef", None) is not None:
        extra["skew_coef"] = np.asarray(znz.skew_coef)
    extra["skew_strength"] = np.array(float(getattr(znz, "skew_strength", 0.0)))
    np.savez(
        path,
        b_coef=znz.b_coef,
        sig_coef=znz.sig_coef,
        xhat_ref=np.array(znz.xhat_ref),
        z_ref=np.array(znz.z_ref),
        b_ref=np.array(znz.b_ref),
        sig_ref=np.array(znz.sig_ref),
        z_covariate=np.array(znz.z_covariate),
        deg_xhat=np.array(znz.deg_xhat),
        deg_z=np.array(znz.deg_z),
        g_grid=cnz.g_grid,
        nhi_edges=cnz.nhi_edges,
        z_edges_fine=cnz.z_edges_fine,
        **extra,
    )


def load_znz(path: str) -> tuple:
    """Load (ZNZModel, CNZModel) from a NPZ file written by save_znz.

    Returns
    -------
    (ZNZModel, CNZModel)

    Backward-compatible: if deg_xhat/deg_z are absent (old NPZ), they are
    recovered from the coef length using _deg_from_coef with a default deg_xhat=1.
    """
    d = np.load(path, allow_pickle=True)
    b_coef = d["b_coef"]
    # Recover degrees: prefer stored fields; fall back to _deg_from_coef for old files.
    if "deg_xhat" in d:
        deg_xhat = int(d["deg_xhat"])
        deg_z = int(d["deg_z"])
    else:
        deg_xhat = 1  # production default
        deg_z = _deg_from_coef(b_coef, deg_xhat)
    # Stage III optional surfaces (absent in old caches → None / 1.0 = frozen MEAN).
    b_med_coef = d["b_med_coef"] if "b_med_coef" in d else None
    if b_med_coef is not None and np.asarray(b_med_coef).size == 0:
        b_med_coef = None
    b_mix = float(d["b_mix"]) if "b_mix" in d else 1.0
    # Track-C optional skew surface (absent in old caches → None / 0.0 = no warp =
    # byte-identical default; the load-bearing backward-compat gate).
    skew_coef = d["skew_coef"] if "skew_coef" in d else None
    if skew_coef is not None and np.asarray(skew_coef).size == 0:
        skew_coef = None
    skew_strength = float(d["skew_strength"]) if "skew_strength" in d else 0.0
    znz = ZNZModel(
        b_coef=b_coef,
        sig_coef=d["sig_coef"],
        xhat_ref=float(d["xhat_ref"]),
        z_ref=float(d["z_ref"]),
        b_ref=float(d["b_ref"]),
        sig_ref=float(d["sig_ref"]),
        z_covariate=str(d["z_covariate"]),
        deg_xhat=deg_xhat,
        deg_z=deg_z,
        b_med_coef=b_med_coef,
        b_mix=b_mix,
        skew_coef=skew_coef,
        skew_strength=skew_strength,
    )
    cnz = CNZModel(
        g_grid=d["g_grid"],
        nhi_edges=d["nhi_edges"],
        z_edges_fine=d["z_edges_fine"],
    )
    return znz, cnz


# ---------------------------------------------------------------------------
# build_cache — reproducible Stage-0 NPZ entrypoint
# ---------------------------------------------------------------------------

def build_cache(argv=None):
    """CLI entrypoint: build (or rebuild) the stage-0 znz NPZ cache deterministically.

    Op-set used here is IDENTICAL to the b-measurement in measure_znz_response:
      (S2N_RED > snr_min) & (P_DLA > p_dla_min) & good_mask
    with NHI_TILT_HOST as the host-truth column (host_truth_floor=19.0).

    The cache is written to --out.  The exact N + b_ref + b(20.5, [2.25,2.75,3.25])
    are printed for verification.

    Usage
    -----
    python -m CDDF_analysis.znz_kernel build-cache \\
        --catalog-dir /scratch/.../gl_prod_2lpt0_v1_20260526/combined_catalog/ \\
        --truth       /nfs/.../hcd_truth_cat.fits \\
        --bal-cat     /nfs/.../bal_cat.fits \\
        --molly-tsv   /scratch/.../figures_molly_nhi195/lya_only/molly_matrix.tsv \\
        --out         /scratch/.../track_c/stage0/znz_2lpt0.npz

    All defaults match the documented WALL-1 calibrated configuration used by
    ab_loa0_fp_baseline.py (figures_molly_nhi195, host_truth_floor=19.0,
    NHI_TILT_HOST, snr_min=2.0, p_dla_min=0.99).
    """
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

    # Import here to avoid hard dependency at module-import time
    from CDDF_analysis.hbi.ab_loa0_fp_baseline import (
        build_ingredients, DEF_CAT, DEF_TRUTH, DEF_BAL,
        DEF_KERNEL, DEF_LOA0_PRODUCT,
    )
    from CDDF_analysis.hbi.cddf_catalog_hbi import build_fine_grid

    p = argparse.ArgumentParser(
        description="Build stage-0 znz NPZ cache (reproducible, documented op-set).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--catalog-dir", default=DEF_CAT)
    p.add_argument("--truth", default=DEF_TRUTH)
    p.add_argument("--bal-cat", default=DEF_BAL)
    p.add_argument("--molly-tsv", default=None,
                   help="Lyα-only nhi195 molly matrix (auto-resolved if not given)")
    p.add_argument("--kernel", default=DEF_KERNEL)
    p.add_argument("--loa0-product", default=DEF_LOA0_PRODUCT)
    p.add_argument("--out",
                   default=("/scratch/cavestru_root/cavestru0/mfho/"
                            "cddf_o3_realdata/track_c/stage0/znz_2lpt0.npz"))
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3,20.6")
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--fit-ceil", type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes", type=int, default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0,
                   help="host_truth_floor for load_and_cut_catalog (default 19.0)")
    p.add_argument("--deg-xhat", type=int, default=1)
    p.add_argument("--deg-z", type=int, default=2)
    p.add_argument("--z-fine-step", type=float, default=0.1)
    args = p.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print("[build_cache] loading ingredients (same op-set as ab_loa0_fp_baseline)...")
    ing = build_ingredients(args, fp_estimator="purity_mixture")
    cfg = ing["cfg"]
    cat_cut = ing["cat_cut"]
    truth_cut = ing["truth_cut"]
    good_mask = ing["good_mask"]
    mm = ing["mm"]
    fine_grid = build_fine_grid(cfg)

    print("[build_cache] measuring b(xhat, z) ...")
    meas = measure_znz_response(
        cat_cut, good_mask, cfg, mm, fine_grid,
        z_covariate="z_dla", host_col="NHI_TILT_HOST")

    N_tp = len(meas["xhat"])
    print(f"[build_cache] N (truth-matched TPs in op-set) = {N_tp:,}")

    znz = fit_znz_model(meas, deg_z=args.deg_z, deg_xhat=args.deg_xhat)
    print(f"[build_cache] b_ref = {znz.b_ref:.4f} at "
          f"(xhat_ref={znz.xhat_ref:.3f}, z_ref={znz.z_ref:.3f})")
    for z_eval in [2.25, 2.75, 3.25]:
        bval = float(znz.b(np.array([20.5]), np.array([z_eval]))[0])
        print(f"[build_cache] b(20.5, z={z_eval}) = {bval:.4f}")

    print("[build_cache] measuring g(N,z) completeness ...")
    zbins = np.asarray(cfg.zbins, float)
    z_lo, z_hi = float(zbins[0]), float(zbins[-1])
    z_edges_fine = np.arange(z_lo, z_hi + args.z_fine_step * 0.5, args.z_fine_step)
    meas_c = measure_c_nz(cat_cut, truth_cut, cfg, mm, z_edges_fine,
                           good_mask=good_mask)
    cnz = fit_c_nz_model(meas_c)

    print(f"[build_cache] saving -> {args.out}")
    save_znz(args.out, znz, cnz)

    # verify round-trip
    znz2, cnz2 = load_znz(args.out)
    assert np.allclose(znz2.b_coef, znz.b_coef), "round-trip b_coef mismatch"
    assert float(znz2.b(np.array([20.5]), np.array([2.75]))[0]) == \
           float(znz.b(np.array([20.5]), np.array([2.75]))[0]), "round-trip b() mismatch"
    print("[build_cache] round-trip verified OK.")

    print("\n[build_cache] STAMP:")
    print(f"  N           = {N_tp:,}")
    print(f"  b_ref       = {znz.b_ref:.4f}  (at xhat_ref={znz.xhat_ref:.4f}, z_ref={znz.z_ref:.4f})")
    for z_eval in [2.25, 2.75, 3.25]:
        bval = float(znz.b(np.array([20.5]), np.array([z_eval]))[0])
        print(f"  b(20.5,{z_eval}) = {bval:.4f}")
    print(f"  deg_xhat    = {znz.deg_xhat},  deg_z = {znz.deg_z}")
    print(f"  host_col    = NHI_TILT_HOST,  host_truth_floor = {args.host_truth_floor}")
    print(f"  op-cut      = (S2N_RED>{cfg.snr_min}) & (P_DLA>{cfg.p_dla_min}) & good_mask")
    print(f"  molly       = {cfg.molly_tsv}")
    return znz, cnz


# ---------------------------------------------------------------------------
# build_forward_cache — reproducible Track-C T-A forward-response NPZ entrypoint
# ---------------------------------------------------------------------------

def build_forward_cache(argv=None):
    """CLI entrypoint: build (or rebuild) the Track-C T-A FORWARD-response NPZ cache.

    Reduce-only, reproducible (deterministic — the fit reads only the frozen truth-match,
    no RNG).  Uses the IDENTICAL op-set / loader as ``build_cache`` (ab_loa0_fp_baseline
    ingredients), then fits ``fit_forward_response`` on the truth-match conditional
    (N_true, SNR, z_QSO, dx) — NON-CIRCULAR (no dN/dX / f / Ω input).

    Prints a STAMP of the recovered surfaces at a few (N, SNR, z) cells for verification
    against the measured forward response (mid-tier skew ≈+0.9, width 0.11→0.20 across
    SNR, up-bias rising with z_QSO).

    Usage
    -----
    python -m CDDF_analysis.znz_kernel build-forward-cache \\
        --out /scratch/.../track_c/stage0/forward_response_2lpt0.npz
    """
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from CDDF_analysis.hbi.ab_loa0_fp_baseline import (
        build_ingredients, DEF_CAT, DEF_TRUTH, DEF_BAL, DEF_KERNEL, DEF_LOA0_PRODUCT,
    )

    p = argparse.ArgumentParser(
        description="Build Track-C T-A forward-response NPZ cache (reproducible, "
                    "non-circular: reads x̂/N_true/SNR/z_QSO only).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--catalog-dir", default=DEF_CAT)
    p.add_argument("--truth", default=DEF_TRUTH)
    p.add_argument("--bal-cat", default=DEF_BAL)
    p.add_argument("--molly-tsv", default=None)
    p.add_argument("--kernel", default=DEF_KERNEL)
    p.add_argument("--loa0-product", default=DEF_LOA0_PRODUCT)
    p.add_argument("--out",
                   default=("/scratch/cavestru_root/cavestru0/mfho/"
                            "cddf_o3_realdata/track_c/stage0/forward_response_2lpt0.npz"))
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3,20.6")
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--fit-ceil", type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes", type=int, default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    p.add_argument("--deg-N", type=int, default=2)
    p.add_argument("--snr-edges", default="2.0,3.5,6.5,inf",
                   help="SNR bin edges (≥4 → ≥3 bins); the response WIDTH axis")
    p.add_argument("--z-edges", default="0.0,2.56,2.96,inf",
                   help="redshift bin edges (default 3 tertile-like bins; the UP-BIAS axis)")
    p.add_argument("--z-covariate", default="zqso", choices=["zqso", "zdla"],
                   help="which redshift the (SNR, z) cell axis conditions on: 'zqso' "
                        "(DEFAULT — Z_QSO, byte-identical to the frozen NPZ) or 'zdla' "
                        "(Z_DLA, the absorber/detection redshift — Track-C (b))")
    p.add_argument("--n-N-cells", type=int, default=7)
    p.add_argument("--min-count", type=int, default=60)
    p.add_argument("--N-skew-collapse", type=float, default=21.0)
    p.add_argument("--xhat-floor", type=float, default=19.5)
    args = p.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def _parse_edges(s):
        return [np.inf if e.strip().lower() in ("inf", "+inf") else float(e)
                for e in s.split(",")]
    snr_edges = _parse_edges(args.snr_edges)
    z_edges = _parse_edges(args.z_edges)

    print("[build_forward_cache] loading ingredients (same op-set as ab_loa0_fp_baseline)...")
    ing = build_ingredients(args, fp_estimator="purity_mixture")
    cfg = ing["cfg"]
    cat_cut = ing["cat_cut"]
    good_mask = ing["good_mask"]

    print(f"[build_forward_cache] measuring forward response (N_true, SNR, "
          f"z={args.z_covariate}, dx)...")
    meas = measure_forward_response(cat_cut, good_mask, cfg,
                                    host_col="NHI_TILT_HOST",
                                    xhat_floor=args.xhat_floor,
                                    z_covariate=args.z_covariate)
    N_tp = len(meas["N_true"])
    print(f"[build_forward_cache] N (truth-matched TPs, x̂>={args.xhat_floor}) = {N_tp:,}")

    frm = fit_forward_response(
        meas, snr_edges=snr_edges, z_edges=z_edges, deg_N=args.deg_N,
        n_N_cells=args.n_N_cells, min_count=args.min_count,
        N_skew_collapse=args.N_skew_collapse)

    print(f"[build_forward_cache] saving -> {args.out}")
    save_forward_response(args.out, frm)
    frm2 = load_forward_response(args.out)
    assert np.array_equal(frm2.mu_coef, frm.mu_coef), "round-trip mu_coef mismatch"
    assert np.array_equal(frm2.skew_coef, frm.skew_coef), "round-trip skew_coef mismatch"
    print("[build_forward_cache] round-trip verified OK.")

    # --- STAMP: recovered surfaces vs the measured forward response ---
    print("\n[build_forward_cache] STAMP (recovered forward-response surfaces):")
    print(f"  N            = {N_tp:,}")
    print(f"  snr_edges    = {snr_edges}")
    print(f"  z_edges      = {z_edges}")
    print(f"  N_ref        = {frm.N_ref:.4f},  deg_N = {frm.deg_N},  "
          f"N_skew_collapse = {frm.N_skew_collapse}")
    print(f"  z_covariate  = {frm.z_covariate}  "
          f"({'Z_DLA (absorber redshift)' if frm.z_covariate == 'zdla' else 'Z_QSO'})")
    snr_probe = [2.5, 5.0, 20.0]      # hi-deficit (lo-SNR), mid, hi-SNR
    z_probe = [2.25, 2.75, 3.25]
    print("  --- mid-tier (N_true=20.4): width should swing 0.11→0.20 across SNR ---")
    for s in snr_probe:
        sd = float(frm.sigma(np.array([20.4]), np.array([s]), np.array([2.75]))[0])
        sk = float(frm.skew(np.array([20.4]), np.array([s]), np.array([2.75]))[0])
        print(f"    SNR={s:>5}: sigma={sd:.3f}  skew={sk:+.3f}")
    z_lab = "z_DLA" if frm.z_covariate == "zdla" else "z_QSO"
    print(f"  --- up-bias mean(dx) vs {z_lab} (N_true=20.4, SNR=5) ---")
    for z in z_probe:
        mu = float(frm.mu_b(np.array([20.4]), np.array([5.0]), np.array([z]))[0])
        print(f"    {z_lab}={z}: mu_b={mu:+.4f}")
    print("  --- high-N skew collapse (SNR=5, z=2.75) ---")
    for Nv in [20.4, 21.0, 21.5]:
        sk = float(frm.skew(np.array([Nv]), np.array([5.0]), np.array([2.75]))[0])
        print(f"    N_true={Nv}: skew={sk:+.3f}")
    print(f"  host_col     = NHI_TILT_HOST,  host_truth_floor = {args.host_truth_floor}")
    print(f"  op-cut       = (S2N_RED>{cfg.snr_min}) & (P_DLA>{cfg.p_dla_min}) & good_mask")
    print(f"  NON-CIRCULAR = fit read x̂/N_true/SNR/{z_lab} only (no dN/dX/f/Ω)")
    return frm


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "build-cache":
        _sys.argv.pop(1)
        build_cache()
    elif len(_sys.argv) > 1 and _sys.argv[1] == "build-forward-cache":
        _sys.argv.pop(1)
        build_forward_cache()
    else:
        print("Usage: python -m CDDF_analysis.znz_kernel build-cache [options]")
        print("       python -m CDDF_analysis.znz_kernel build-forward-cache [options]")
        print("       python znz_kernel.py build-cache [options]")
        _sys.exit(1)
