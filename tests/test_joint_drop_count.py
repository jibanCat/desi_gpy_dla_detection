"""tests/test_joint_drop_count.py — unit tests for CDDF_analysis/hbi/joint_drop_count.py.

The joint Lyman-limit-DROP + FP-corrected-COUNTING estimator produces BOTH LLS
headlines (lambda_mfp / tau_eff,LL and the shape-marginalized ell(X)[17.2,19.5) band)
and had ZERO unit tests after TWO correctness bugs were found in it BY HAND
(commit 9349836, 2026-07-06 two-team cross-check).  This file closes that gap.

Layout
  1. REGRESSION — bug 1: ``_clamp_drop_grid`` clamped only the LOWER knot edge, so the
     drop integral ran to hi_default=22.5 > top knot 22.4 where the cubic-B-spline design
     matrix is all-zeros -> log f = 0 -> f = O(1), weighted by N ln10 ~ 1e22 in the
     opacity integral.  ``TestBug1TopKnotOverflow`` reproduces the PRE-FIX clamp verbatim
     (monkeypatched, the repo copy is never touched) and shows it blows tau up by ~1e20.
  2. REGRESSION — bug 2: ``joint_laplace`` read cfg.v3_lambda_spline, which ``fit_joint``
     RESTORES in a ``finally``, so the Laplace Hessian was the curvature of a DIFFERENT
     (weaker-regularized) objective.  ``TestBug2LaplaceLamSplineMismatch`` pins the
     explicit ``lam_spline`` argument, its exact analytic effect on the Hessian
     (H(lam2) - H(lam1) == (lam2-lam1) * D2^T D2), the cfg restore, and the wiring that
     passes the fit's ESCALATED lam through ``lls_shape_marginalized_band``.
  3. Numerical invariants of the drop channel: monotonicity in amplitude / cross-section /
     path length, the optically-thin LINEARITY and the optically-thick SUM-RULE ceiling
     (the drop normalisation), the analytic theta-gradient, and the logN_lo floor.
  4. The p-spline sub-floor regularisation (SubFloorRidge + the lam_spline curvature
     lever) staying FINITE and positive-definite.
  5. ell(X) band SEMANTICS — the band is a pooled percentile whose single-shape width is a
     STATISTICAL (within-shape Laplace) width only.  No headline VALUE is asserted
     anywhere: the LLS tier is out of Paper 1 and ell(X) is prior-limited.
  6. OMISSION SENSITIVITY — tests that FAIL if a term is silently dropped from the joint
     likelihood (the alpha_F FP latent, the drop term, the LLS anchor, the ridge, the
     counting objective weights).

Synthetic fixtures only; no catalog / FITS / GP inference, no real-DESI values.

Run:  /home/mfho/.conda/envs/gpdla/bin/python -m pytest tests/test_joint_drop_count.py -v
"""
import inspect

import numpy as np
import pytest

from CDDF_analysis.hbi import cddf_catalog_hbi as H
from CDDF_analysis.hbi import joint_drop_count as J
from CDDF_analysis.cddf_mock import path_length_int


# =========================================================================== #
# Fixtures — small synthetic forward model, no I/O
# =========================================================================== #
def _cfg(**kw):
    """A minimal HBIConfig with dummy paths (never loaded) and a COARSE fit grid so the
    joint objective is fast.  The knot span / floor knobs are the ones the drop clamp and
    the sub-floor ridge read."""
    d = dict(
        catalog_dir="/dev/null", truth_path="/dev/null", bal_cat_path="/dev/null",
        molly_tsv="/dev/null", out_dir="/tmp",
        logN_lo=17.2, logN_hi=22.5, dlogN=0.2, drop_top_bin_above=22.4,
        zbins=(2.0, 2.5, 3.0, 3.5), report_logN_limits=(20.0, 20.3),
        H0=70.0, Omega_m=0.279,
        v2_z_fit_lo=2.0, v2_z_fit_hi=3.5, v2_z_fit_step=0.5,
        v3_n_spline_knots=7, v3_lambda_spline=1.0, v3_logN_fit_floor=19.5,
    )
    d.update(kw)
    return H.HBIConfig(**d)


def _fine(cfg):
    logN_lo, logN_hi, N_b, dN_b = H.build_fine_grid(cfg)
    return (logN_lo, logN_hi, N_b, dN_b, H._fine_z_grid(cfg))


def _fwd(cfg, seed=3, n_obs=40, mu_fp=20.0, with_weights=False):
    """Synthetic v3x forward dict: a sparse deconvolution matrix A, a normalizer M, a
    per-object FP intensity lam_fp and its integral mu_fp, plus the M_meta pathlength the
    reducer needs.  Values are arbitrary but FIXED (seeded) — every test that uses this
    asserts a RELATION between two evaluations, never an absolute headline."""
    fine = _fine(cfg)
    n_flat = len(fine[0]) * (len(fine[4]) - 1)
    rng = np.random.default_rng(seed)
    dense = np.abs(rng.normal(0, 1, (n_obs, n_flat))) * (rng.random((n_obs, n_flat)) < 0.15)
    fwd = dict(
        A_full=H._sp.csr_matrix(dense),
        M_full=np.abs(rng.normal(1.0, 0.3, n_flat)) * 1e21,
        lam_fp=np.abs(rng.normal(0.2, 0.05, n_obs)),
        mu_fp=float(mu_fp),
        fine=fine,
        M_meta=dict(PX=np.abs(rng.normal(1e3, 1e2, (2, len(fine[4]) - 1)))),
        cat_op=None,
    )
    if with_weights:
        fwd["cat_op"] = dict(op_weights=rng.uniform(0.5, 1.5, n_obs))
    return fwd


def _theta_pspline(cfg, shift=0.0):
    """Interior pspline theta (bounds are [-30,-10] on the coeffs, [-3,5] on gz)."""
    return np.asarray(H.v3x_default_theta0("pspline", cfg), float) + shift


def _drop(z912=(2.6, 2.9, 3.2), tau_hat=(1.0, 0.8, 0.5), sigma=(0.1, 0.1, 0.1),
          z_qso=3.5, **kw):
    return J.DropData(z912=list(z912), tau_hat=list(tau_hat), sigma=list(sigma),
                      z_qso=z_qso, **kw)


def _ref_tau(theta, family, cfg, z912, z_qso, sigma912=J.SIGMA_912, beta=3.0,
             lo=17.2, hi=22.4, n_logN=1200, n_z=600, opac_one=False):
    """INDEPENDENT re-implementation of the drop integral (written from the docstring
    formula, not from the code path), on a much finer grid:

        tau(z912) = INT_{z912}^{zq} (dX/dz') INT dlogN [N ln10 f(N,z')]
                    * (1 - exp(-N sigma912 ((1+z912)/(1+z'))^beta)) dz'

    ``opac_one=True`` replaces the (1-exp) factor by 1 — the optically-THICK SUM RULE
    (the total number of absorbers on the path), which is the drop channel's
    normalisation ceiling."""
    x = np.linspace(lo, hi, n_logN)
    N = 10.0 ** x
    zp = np.linspace(float(z912), float(z_qso), n_z)
    dXdz = path_length_int(zp, float(getattr(cfg, "Omega_m", 0.279)))
    fN = H.v3x_f_of_N(x[:, None], zp[None, :], theta, family, cfg)
    if opac_one:
        opac = np.ones_like(fN)
    else:
        sig = sigma912 * ((1.0 + z912) / (1.0 + zp)) ** beta
        opac = 1.0 - np.exp(-(N[:, None] * sig[None, :]))
    integrand = fN * (N[:, None] * np.log(10.0)) * opac
    inner = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(x)[:, None], axis=0)
    g = inner * dXdz
    return float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(zp)))


# --------------------------------------------------------------------------- #
# The PRE-FIX (buggy) _clamp_drop_grid, copied verbatim from the parent of
# 9349836.  Used ONLY via monkeypatch, to prove the regression assertions below
# actually discriminate.  The repo's copy is never modified.
# --------------------------------------------------------------------------- #
def _prefix_clamp_drop_grid(logN_grid, logN_lo, hi_default, family, cfg, n=160):
    span = J._v3x_knot_span(family, cfg)
    if logN_grid is None:
        lo_eff = float(logN_lo)
        if span is not None and lo_eff < span[0]:
            lo_eff = span[0]
        return np.linspace(lo_eff, hi_default, n)          # <-- top edge NOT clamped
    g = np.asarray(logN_grid, float)
    if span is not None:
        g = g[g >= span[0] - 1e-9]                          # <-- supra-knot rows kept
    return g


# =========================================================================== #
# 1. REGRESSION — bug 1: top-knot drop overflow
# =========================================================================== #
class TestBug1TopKnotOverflow:
    def test_knot_span_is_reported_for_spline_families_only(self):
        cfg = _cfg()
        assert J._v3x_knot_span("pspline", cfg) == pytest.approx((17.2, 22.4))
        assert J._v3x_knot_span("bspbody", cfg) == pytest.approx((19.2, 22.4))
        # analytic families are valid at all N -> no span, no clamp
        for fam in ("plaw", "plawcut", "bplcut"):
            assert J._v3x_knot_span(fam, cfg) is None

    def test_clamp_drop_grid_clamps_BOTH_edges(self):
        """BUG 1.  The built grid must stop at the TOP knot (22.4), not at the
        hi_default 22.5 the caller passes."""
        cfg = _cfg()
        g = J._clamp_drop_grid(None, 15.0, 22.5, "pspline", cfg, n=64)
        assert g.min() == pytest.approx(17.2)      # lower clamp (was already there)
        assert g.max() == pytest.approx(22.4)      # upper clamp (THE FIX)
        assert g.max() < 22.5 - 1e-9

    def test_clamp_drop_grid_drops_supra_knot_rows_of_a_passed_grid(self):
        """BUG 1, second branch: a caller-supplied grid must have its ABOVE-knot rows
        dropped too (pre-fix only the below-knot rows were dropped)."""
        cfg = _cfg()
        passed = np.linspace(15.0, 23.0, 81)
        g = J._clamp_drop_grid(passed, 15.0, 22.5, "pspline", cfg)
        assert g.size > 0
        assert g.min() >= 17.2 - 1e-9
        assert g.max() <= 22.4 + 1e-9
        assert (passed > 22.4 + 1e-9).any(), "fixture must contain supra-knot rows"

    def test_drop_tau_model_integrates_only_inside_the_knot_span(self):
        """BUG 1 end-to-end: drop_tau_model's tau must equal the SAME integral taken
        explicitly over [knot_lo, knot_hi] — i.e. the [22.4,22.5] sliver contributes
        nothing, because it is not integrated at all."""
        cfg = _cfg()
        th = _theta_pspline(cfg)
        tau = J.drop_tau_model(th, "pspline", cfg, [2.8], 3.5, logN_lo=15.0)[0]
        ref_in = _ref_tau(th, "pspline", cfg, 2.8, 3.5, lo=17.2, hi=22.4)
        assert tau == pytest.approx(ref_in, rel=5e-3)
        # ... and the same integral run to 22.5 (the pre-fix upper limit) is catastrophic
        ref_over = _ref_tau(th, "pspline", cfg, 2.8, 3.5, lo=17.2, hi=22.5)
        assert ref_over / ref_in > 1e10

    def test_prefix_clamp_reproduces_the_overflow(self, monkeypatch):
        """Proof that the assertions above DISCRIMINATE: with the pre-fix clamp
        monkeypatched in, tau explodes by ~20 orders of magnitude."""
        cfg = _cfg()
        th = _theta_pspline(cfg)
        tau_now = J.drop_tau_model(th, "pspline", cfg, [2.8], 3.5, logN_lo=15.0)[0]
        monkeypatch.setattr(J, "_clamp_drop_grid", _prefix_clamp_drop_grid)
        tau_prefix = J.drop_tau_model(th, "pspline", cfg, [2.8], 3.5, logN_lo=15.0)[0]
        assert np.isfinite(tau_now) and tau_now < 1e3
        assert tau_prefix / tau_now > 1e10

    def test_both_drop_entry_points_route_through_the_fixed_clamp(self):
        """The clamp fix must cover the ANALYTIC-GRADIENT entry point too, not just the
        value.  drop_tau_and_grad cannot be exercised at runtime (see TestDropGradient —
        it is dead, broken code), so pin the call site by source."""
        for fn in (J.drop_tau_model, J.drop_tau_and_grad):
            src = inspect.getsource(fn)
            assert "_clamp_drop_grid(logN_grid, logN_lo, 22.5" in src, fn.__name__

    def test_analytic_families_are_not_clamped(self):
        """The fix is family-specific: plaw/bplcut have no knots and MUST keep the full
        [logN_lo, hi_default] support (clamping them would silently truncate real
        sub-LLS opacity)."""
        cfg = _cfg()
        g = J._clamp_drop_grid(None, 15.0, 22.5, "plaw", cfg, n=32)
        assert g.min() == pytest.approx(15.0)
        assert g.max() == pytest.approx(22.5)
        passed = np.linspace(14.0, 23.0, 19)
        np.testing.assert_allclose(J._clamp_drop_grid(passed, 15.0, 22.5, "plaw", cfg),
                                   passed)


# =========================================================================== #
# 2. REGRESSION — bug 2: joint_laplace / fit_joint lam_spline mismatch
# =========================================================================== #
def _padded_D2TD2(n_theta):
    """lam's exact Hessian contribution: the pspline curvature prior is
    -0.5*lam*||D2 c||^2 on the COEFF block only (gz is untouched)."""
    D2 = H._pspline_D2(n_theta - 1)
    P = np.zeros((n_theta, n_theta))
    P[:-1, :-1] = D2.T @ D2
    return P


class TestBug2LaplaceLamSplineMismatch:
    def test_joint_laplace_exposes_lam_spline(self):
        """BUG 2.  Pre-fix joint_laplace had NO lam_spline parameter, so the caller
        could not make the Hessian match the objective the MAP was fit under."""
        sig = inspect.signature(J.joint_laplace)
        assert "lam_spline" in sig.parameters
        assert sig.parameters["lam_spline"].default is None

    def test_hessian_shifts_by_exactly_dlam_times_D2TD2(self):
        """The lam_spline argument must reach the Hessian, with the EXACT analytic
        magnitude.  Pre-fix both calls silently used cfg.v3_lambda_spline and this
        difference was ZERO."""
        cfg = _cfg(v3_lambda_spline=1.0)
        fwd = _fwd(cfg)
        th = _theta_pspline(cfg)
        h1 = J.joint_laplace(th, fwd, "pspline", cfg, None, n_draw=4, lam_spline=1.0)
        h2 = J.joint_laplace(th, fwd, "pspline", cfg, None, n_draw=4, lam_spline=60.0)
        dH = h2["hess"] - h1["hess"]
        expect = (60.0 - 1.0) * _padded_D2TD2(th.size)
        scale = np.max(np.abs(expect))
        np.testing.assert_allclose(dH, expect, atol=1e-6 * scale, rtol=2e-4)
        assert scale > 0

    def test_omitting_lam_spline_uses_the_weaker_cfg_value(self):
        """The pre-fix path: cfg.v3_lambda_spline is what fit_joint RESTORED (1.0),
        NOT the lam the MAP was fit under.  So lam_spline=None must reproduce the
        cfg-lam Hessian and DIFFER from the fit's lam — that difference is the bug."""
        cfg = _cfg(v3_lambda_spline=1.0)
        fwd = _fwd(cfg)
        th = _theta_pspline(cfg)
        h_none = J.joint_laplace(th, fwd, "pspline", cfg, None, n_draw=4, lam_spline=None)
        h_cfg = J.joint_laplace(th, fwd, "pspline", cfg, None, n_draw=4, lam_spline=1.0)
        h_fit = J.joint_laplace(th, fwd, "pspline", cfg, None, n_draw=4, lam_spline=60.0)
        np.testing.assert_allclose(h_none["hess"], h_cfg["hess"], rtol=1e-10, atol=1e-12)
        assert not np.allclose(h_none["hess"], h_fit["hess"])
        # and the weaker lam is the SINGULAR one the fix was about
        assert h_none["cond"] > h_fit["cond"]

    def test_fit_joint_restores_cfg_lambda_spline(self):
        """The `finally` restore in fit_joint is what CREATED bug 2; pin it so the
        contract joint_laplace compensates for cannot silently change."""
        cfg = _cfg(v3_lambda_spline=1.0)
        fwd = _fwd(cfg)
        J.fit_joint(fwd, "pspline", cfg, None, lam_spline=60.0, n_restart=1)
        assert cfg.v3_lambda_spline == 1.0

    def test_joint_laplace_restores_cfg_lambda_spline_even_on_error(self, monkeypatch):
        cfg = _cfg(v3_lambda_spline=1.0)
        fwd = _fwd(cfg)
        th = _theta_pspline(cfg)
        J.joint_laplace(th, fwd, "pspline", cfg, None, n_draw=4, lam_spline=60.0)
        assert cfg.v3_lambda_spline == 1.0

        def _boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(J, "_joint_grad", _boom)
        with pytest.raises(RuntimeError):
            J.joint_laplace(th, fwd, "pspline", cfg, None, n_draw=4, lam_spline=60.0)
        assert cfg.v3_lambda_spline == 1.0

    def test_band_passes_the_ESCALATED_lam_to_joint_laplace(self, monkeypatch):
        """BUG 2, wiring half: lls_shape_marginalized_band escalates lam_spline when the
        MAP ell breaches the ceiling, and must hand the Laplace the ESCALATED value.
        Pre-fix it handed nothing at all."""
        cfg = _cfg()
        assert float(getattr(cfg, "v3_lls_ell_ceiling", 50.0)) == 50.0  # documented default
        fwd = _fwd(cfg)
        seen = {}
        calls = {"reduce": 0}

        def fake_fit_joint(fwd_, fam, cfg_, drop_, sub_lls=None, lam_spline=None,
                           seed=0, **kw):
            seen.setdefault("fit_lams", []).append(lam_spline)
            return dict(theta_map=np.array([1.0, 2.0]))

        def fake_reduce(cfg_, theta, fine, fam, meta):
            calls["reduce"] += 1
            # first MAP reduction blows the ceiling -> forces ONE escalation
            if calls["reduce"] == 1:
                return {"ell_lls_extrap": 1e6}
            return {"ell_lls_extrap": float(np.asarray(theta)[0])}

        def fake_laplace(theta_map, fwd_, fam, cfg_, drop_, sub_lls=None,
                         n_draw=500, rng=None, lam_spline=None, **kw):
            seen["lap_lam"] = lam_spline
            return dict(draws=np.tile(np.array([0.3, 0.0]), (n_draw, 1)), cond=1e3)

        monkeypatch.setattr(J, "fit_joint", fake_fit_joint)
        monkeypatch.setattr(J, "v3x_reduce", fake_reduce)
        monkeypatch.setattr(J, "joint_laplace", fake_laplace)
        out = J.lls_shape_marginalized_band(fwd, cfg, _drop(), [object()],
                                            lam_spline=60.0, n_lap=8)
        assert seen["fit_lams"] == [60.0, 120.0]          # one escalation
        assert seen["lap_lam"] == 120.0                    # the ESCALATED lam, not 60
        assert out["per_shape"][0]["lam_spline"] == 120.0


# =========================================================================== #
# 3. Drop-channel numerical invariants
# =========================================================================== #
class TestDropChannelInvariants:
    def test_tau_is_zero_at_and_above_the_qso_redshift(self):
        cfg = _cfg()
        th = _theta_pspline(cfg)
        tau = J.drop_tau_model(th, "pspline", cfg, [3.5, 3.6, 3.0], 3.5)
        assert tau[0] == 0.0 and tau[1] == 0.0
        assert tau[2] > 0.0

    def test_tau_decreases_with_z912_shorter_path(self):
        cfg = _cfg()
        th = _theta_pspline(cfg)
        z912 = np.array([2.2, 2.6, 3.0, 3.4])
        tau = J.drop_tau_model(th, "pspline", cfg, z912, 3.5)
        assert np.all(np.diff(tau) < 0), tau

    def test_tau_monotone_increasing_in_cddf_amplitude(self):
        """More absorbers -> more opacity, at EVERY z912.  A sign error anywhere in the
        opacity kernel breaks this."""
        cfg = _cfg()
        base = _theta_pspline(cfg)
        z912 = np.array([2.4, 2.8, 3.2])
        prev = None
        for d in (-0.5, 0.0, 0.5, 1.0):
            th = base.copy()
            th[:-1] += d                       # shift log10 f by d dex everywhere
            tau = J.drop_tau_model(th, "pspline", cfg, z912, 3.5)
            if prev is not None:
                assert np.all(tau > prev), (d, tau, prev)
            prev = tau

    def test_tau_monotone_increasing_in_sigma912(self):
        cfg = _cfg()
        th = _theta_pspline(cfg)
        taus = [J.drop_tau_model(th, "pspline", cfg, [2.8], 3.5, sigma912=s)[0]
                for s in (1e-19, 6.35e-18, 1e-16)]
        assert taus[0] < taus[1] < taus[2]

    def test_beta_redshifts_the_cross_section_downwards(self):
        """sigma(nu) ~ nu^-beta with beta>0 SHRINKS the cross-section seen by absorbers
        at z' > z912, so tau(beta=3) < tau(beta=0).  A sign flip on beta inverts this."""
        cfg = _cfg()
        th = _theta_pspline(cfg)
        t3 = J.drop_tau_model(th, "pspline", cfg, [2.6], 3.5, beta=3.0)[0]
        t0 = J.drop_tau_model(th, "pspline", cfg, [2.6], 3.5, beta=0.0)[0]
        tm = J.drop_tau_model(th, "pspline", cfg, [2.6], 3.5, beta=-3.0)[0]
        assert t3 < t0 < tm

    def test_drop_normalisation_matches_independent_quadrature(self):
        """NORMALISATION of the drop channel, on the analytic plaw family (no knots, so
        the whole [logN_lo, 22.5] support is exercised): the module's tau must match an
        INDEPENDENTLY written double integral.  Catches a missing N ln10 measure, a
        missing dX/dz path measure, or a wrong ln10."""
        cfg = _cfg()
        th = np.asarray(H.v3x_default_theta0("plaw", cfg), float)
        for z912 in (2.4, 3.0):
            got = J.drop_tau_model(th, "plaw", cfg, [z912], 3.6, logN_lo=17.2)[0]
            ref = _ref_tau(th, "plaw", cfg, z912, 3.6, lo=17.2, hi=22.5)
            assert got == pytest.approx(ref, rel=2e-3), (z912, got, ref)

    def test_drop_normalisation_is_wrong_if_the_N_ln10_measure_is_dropped(self):
        """Omission sensitivity on the drop measure: f is per dN, the grid is per dlogN,
        so the N ln10 Jacobian is load-bearing.  Without it the integral is off by many
        orders of magnitude, i.e. the previous test cannot pass by accident."""
        cfg = _cfg()
        th = np.asarray(H.v3x_default_theta0("plaw", cfg), float)
        x = np.linspace(17.2, 22.5, 1200)
        zp = np.linspace(2.4, 3.6, 600)
        fN = H.v3x_f_of_N(x[:, None], zp[None, :], th, "plaw", cfg)
        sig = J.SIGMA_912 * ((1.0 + 2.4) / (1.0 + zp)) ** 3.0
        opac = 1.0 - np.exp(-((10.0 ** x)[:, None] * sig[None, :]))
        no_jac = fN * opac                              # <- N ln10 dropped
        inner = np.sum(0.5 * (no_jac[1:] + no_jac[:-1]) * np.diff(x)[:, None], axis=0)
        g = inner * path_length_int(zp, cfg.Omega_m)
        tau_no_jac = float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(zp)))
        got = J.drop_tau_model(th, "plaw", cfg, [2.4], 3.6, logN_lo=17.2)[0]
        assert got / max(tau_no_jac, 1e-300) > 1e10

    def test_optically_thin_limit_is_linear_in_sigma912(self):
        """tau -> sigma912 * INT N f dN dX as sigma912 -> 0.  A 10x smaller cross-section
        must give a 10x smaller tau: the drop channel's LINEAR normalisation."""
        cfg = _cfg()
        th = np.asarray(H.v3x_default_theta0("plaw", cfg), float)
        t1 = J.drop_tau_model(th, "plaw", cfg, [2.8], 3.5, sigma912=1e-26, logN_lo=17.2)[0]
        t2 = J.drop_tau_model(th, "plaw", cfg, [2.8], 3.5, sigma912=1e-27, logN_lo=17.2)[0]
        assert t1 / t2 == pytest.approx(10.0, rel=1e-3)

    def test_optically_thick_limit_saturates_at_the_absorber_count_sum_rule(self):
        """tau -> INT dX INT f dN (the NUMBER of absorbers on the path) as
        sigma912 -> inf, and stays strictly BELOW it for any finite sigma912.  This is
        the drop channel's upper normalisation ceiling."""
        cfg = _cfg()
        th = np.asarray(H.v3x_default_theta0("plaw", cfg), float)
        ceiling = _ref_tau(th, "plaw", cfg, 2.8, 3.5, lo=17.2, hi=22.5, opac_one=True)
        thick = J.drop_tau_model(th, "plaw", cfg, [2.8], 3.5, sigma912=1e10, logN_lo=17.2)[0]
        real = J.drop_tau_model(th, "plaw", cfg, [2.8], 3.5, logN_lo=17.2)[0]
        assert thick == pytest.approx(ceiling, rel=2e-3)
        assert 0.0 < real < ceiling

    def test_drop_neg_loglike_is_exactly_half_chi2(self):
        cfg = _cfg()
        th = _theta_pspline(cfg)
        d = _drop(logN_lo=17.2)
        tau = J.drop_tau_model(th, "pspline", cfg, d.z912, d.z_qso,
                               sigma912=d.sigma912, beta=d.beta, logN_lo=d.logN_lo)
        r = (tau - d.tau_hat) / d.sigma
        assert J.drop_neg_loglike(th, "pspline", cfg, d) == pytest.approx(
            0.5 * float(np.sum(r * r)), rel=1e-12)

    def test_dropdata_carries_its_own_sigma912_beta_and_floor(self, monkeypatch):
        """Argument threading: drop_neg_loglike must forward EVERY DropData knob to the
        model.  A dropped kwarg would silently fall back to the module defaults."""
        cfg = _cfg()
        seen = {}

        def spy(theta, family, cfg_, z912_arr, z_qso, **kw):
            seen.update(kw)
            return np.zeros(np.size(z912_arr))

        monkeypatch.setattr(J, "drop_tau_model", spy)
        d = _drop(sigma912=1.23e-18, beta=2.75, logN_lo=16.0)
        J.drop_neg_loglike(_theta_pspline(cfg), "pspline", cfg, d)
        assert seen == dict(sigma912=1.23e-18, beta=2.75, logN_lo=16.0)

    def test_dropdata_defaults_match_the_mock_convention(self):
        d = J.DropData(z912=[3.0], tau_hat=[1.0], sigma=[0.1], z_qso=3.5)
        assert d.sigma912 == J.SIGMA_912 == 6.35e-18
        assert d.beta == 3.0            # the quickquasars mock injection index
        assert d.logN_lo == 15.0        # real-IGM floor; clamped up for spline families


# =========================================================================== #
#    logN_lo floor behaviour
# =========================================================================== #
class TestLogNFloor:
    def test_spline_floor_is_clamped_to_the_knot_floor(self):
        """Below the lowest knot the B-spline basis is all-zeros -> f = O(1).  So for a
        spline family logN_lo BELOW 17.2 must give EXACTLY the logN_lo=17.2 answer."""
        cfg = _cfg()
        th = _theta_pspline(cfg)
        t_low = J.drop_tau_model(th, "pspline", cfg, [2.8], 3.5, logN_lo=13.0)[0]
        t_knot = J.drop_tau_model(th, "pspline", cfg, [2.8], 3.5, logN_lo=17.2)[0]
        assert t_low == pytest.approx(t_knot, rel=1e-12)

    def test_spline_floor_above_the_knot_floor_is_honoured(self):
        """The clamp is a FLOOR, not an override: a floor ABOVE the knot floor must
        genuinely truncate the sub-LLS opacity (that is how the HCD-only mock is run)."""
        cfg = _cfg()
        th = _theta_pspline(cfg)
        t_172 = J.drop_tau_model(th, "pspline", cfg, [2.8], 3.5, logN_lo=17.2)[0]
        t_190 = J.drop_tau_model(th, "pspline", cfg, [2.8], 3.5, logN_lo=19.0)[0]
        assert t_190 < t_172

    def test_analytic_family_floor_is_not_clamped(self):
        """plaw has no knots: lowering the floor must ADD real sub-LLS opacity.  This is
        what makes the clamp a spline-artifact guard and not a physics cut."""
        cfg = _cfg()
        th = np.asarray(H.v3x_default_theta0("plaw", cfg), float)
        t15 = J.drop_tau_model(th, "plaw", cfg, [2.8], 3.5, logN_lo=15.0)[0]
        t172 = J.drop_tau_model(th, "plaw", cfg, [2.8], 3.5, logN_lo=17.2)[0]
        assert t15 > t172

    def test_tau_is_finite_at_the_floor_for_every_family(self):
        cfg = _cfg()
        for fam in ("plaw", "bplcut", "pspline", "bspbody"):
            th = np.asarray(H.v3x_default_theta0(fam, cfg), float)
            tau = J.drop_tau_model(th, fam, cfg, [2.4, 2.8, 3.2], 3.5, logN_lo=15.0)
            assert np.all(np.isfinite(tau)), fam
            assert np.all(tau >= 0.0), fam


# =========================================================================== #
#    analytic theta-gradient of the drop
# =========================================================================== #
class TestDropGradient:
    """``drop_tau_and_grad`` is DEAD, BROKEN code — a THIRD defect, found by these tests.

    * ZERO callers in the repo (``fit_joint`` and ``_joint_grad`` both use central
      differences on ``_extra``), so no headline depends on it and nothing that ran is
      invalidated.
    * It raises ``ValueError`` on EVERY call, for EVERY family, at the default
      n_zprime=80.  It passes ``x = logN_grid[:, None]`` of shape (nN, 1) and
      ``z = zp[None, :]`` of shape (1, nzp) into ``v3x_grad_f_wrt_theta``, whose very
      first statement is ``np.broadcast_to(zlog, x.shape)`` — (1, nzp) does not
      broadcast to (nN, 1).  ``v3x_f_of_N`` tolerates the same call because it only ever
      ADDS ``zterm`` (ordinary numpy broadcasting), which is why ``drop_tau_model``
      works and its gradient twin does not.
    * The fix is one line — pass FULLY broadcast 2-D x and z, as ``_v3x_fine_density``
      already does.  It is NOT applied here: this agent's remit is tests, and the LLS
      tier is out of Paper 1.

    The two real tests below are ``xfail(strict=True)``: they XFAIL today and turn into
    a hard FAILURE the moment the defect is fixed, which forces the markers off and the
    assertions on.  They are NOT skips.
    """

    def test_grad_and_value_share_the_same_z_grid_default(self):
        """Hygiene fix: drop_tau_and_grad's n_zprime default was 60 while
        drop_tau_model's was 80, so the analytic gradient integrated a DIFFERENT
        function from the value it differentiates."""
        assert (inspect.signature(J.drop_tau_and_grad).parameters["n_zprime"].default
                == inspect.signature(J.drop_tau_model).parameters["n_zprime"].default
                == 80)

    @pytest.mark.parametrize("family", ["plaw", "bplcut", "pspline", "bspbody"])
    def test_drop_tau_and_grad_is_unreachable_today(self, family):
        """Characterisation of the live defect (see the class docstring)."""
        cfg = _cfg()
        th = np.asarray(H.v3x_default_theta0(family, cfg), float)
        with pytest.raises(ValueError, match="broadcast"):
            J.drop_tau_and_grad(th, family, cfg, [2.8], 3.5, logN_lo=17.2)

    def test_diagnosis_fully_broadcast_x_and_z_are_evaluable(self):
        """Proof of the diagnosis: the same f-gradient the function wants IS well
        defined once x and z are broadcast to a common 2-D shape."""
        cfg = _cfg()
        th = _theta_pspline(cfg)
        x = np.linspace(17.2, 22.4, 30)
        zp = np.linspace(2.8, 3.5, 12)
        X = np.broadcast_to(x[:, None], (x.size, zp.size))
        Z = np.broadcast_to(zp[None, :], (x.size, zp.size))
        g = np.asarray(H.v3x_grad_f_wrt_theta(X, Z, th, "pspline", cfg))
        assert g.shape == (th.size, x.size, zp.size)
        assert np.all(np.isfinite(g))

    @pytest.mark.xfail(raises=ValueError, strict=True,
                       reason="drop_tau_and_grad broadcast defect — see class docstring")
    def test_tau_from_grad_path_matches_tau_from_model(self):
        cfg = _cfg()
        th = _theta_pspline(cfg)
        z = np.array([2.4, 2.8, 3.2])
        tau_g, _ = J.drop_tau_and_grad(th, "pspline", cfg, z, 3.5, logN_lo=15.0)
        tau_m = J.drop_tau_model(th, "pspline", cfg, z, 3.5, logN_lo=15.0)
        np.testing.assert_allclose(tau_g, tau_m, rtol=5e-3)

    @pytest.mark.xfail(raises=ValueError, strict=True,
                       reason="drop_tau_and_grad broadcast defect — see class docstring")
    @pytest.mark.parametrize("family", ["plaw", "pspline"])
    def test_analytic_grad_matches_central_differences(self, family):
        cfg = _cfg()
        th = np.asarray(H.v3x_default_theta0(family, cfg), float)
        z = np.array([2.6, 3.0])
        _, grad = J.drop_tau_and_grad(th, family, cfg, z, 3.5, logN_lo=17.2)
        for k in range(th.size):
            h = 1e-5 * max(abs(th[k]), 1.0)
            tp = th.copy(); tp[k] += h
            tm = th.copy(); tm[k] -= h
            fd = (J.drop_tau_and_grad(tp, family, cfg, z, 3.5, logN_lo=17.2)[0]
                  - J.drop_tau_and_grad(tm, family, cfg, z, 3.5, logN_lo=17.2)[0]) / (2 * h)
            np.testing.assert_allclose(grad[k], fd, rtol=1e-4,
                                       atol=1e-8 * max(np.max(np.abs(fd)), 1e-12))


# =========================================================================== #
# 4. p-spline sub-floor regularisation stays FINITE
# =========================================================================== #
class TestSubFloorRegularisation:
    def test_ridge_rejects_non_spline_families(self):
        cfg = _cfg()
        for fam in ("plaw", "plawcut", "bplcut"):
            with pytest.raises(ValueError):
                J.SubFloorRidge(fam, cfg)

    def test_ridge_selects_only_coeffs_whose_center_is_below_the_fit_floor(self):
        cfg = _cfg(v3_logN_fit_floor=19.5)
        r = J.SubFloorRidge("pspline", cfg, lam=10.0)
        assert r.sub.size > 0
        assert np.all(r.centers[r.sub] < 19.5)
        assert r.centers[r.ib] >= 19.5
        assert r.ib == int(np.min(np.where(r.centers >= 19.5 - 1e-9)[0]))

    def test_ridge_vanishes_on_its_own_reference_power_law(self):
        """Zero penalty when the sub-floor coeffs lie exactly on the LLS reference line
        anchored at the first in-body coeff — the ridge removes only the DEGENERATE
        direction, it does not impose a level."""
        cfg = _cfg()
        r = J.SubFloorRidge("pspline", cfg, lam=10.0, slope_lls=-1.5)
        c = np.full(r.centers.size, -21.0)
        c[:] = c[r.ib] + r.slope * (r.centers - r.centers[r.ib])
        theta = np.append(c, 1.5)
        assert r.neg_loglike(theta) == pytest.approx(0.0, abs=1e-20)
        # perturb ONE sub-floor coeff -> strictly positive, exactly 0.5*lam*delta^2
        theta2 = theta.copy(); theta2[r.sub[0]] += 0.3
        assert r.neg_loglike(theta2) == pytest.approx(0.5 * 10.0 * 0.09, rel=1e-12)

    def test_ridge_is_translation_invariant(self):
        """The reference TRACKS the fitted body level, so a global shift of all coeffs
        must not be penalised (else the ridge would fight the drop normalisation)."""
        cfg = _cfg()
        r = J.SubFloorRidge("pspline", cfg, lam=10.0)
        theta = np.append(np.linspace(-19.0, -24.0, r.centers.size), 1.5)
        shifted = theta.copy(); shifted[:-1] += 2.5
        assert r.neg_loglike(theta) == pytest.approx(r.neg_loglike(shifted), rel=1e-12)

    def test_ridge_stays_finite_at_the_parameter_bounds(self):
        """The blow-up the ridge exists to prevent drives coeffs to the bound
        (f -> 1e-30).  The penalty itself must remain FINITE there."""
        cfg = _cfg()
        r = J.SubFloorRidge("pspline", cfg, lam=10.0)
        lo = np.array([b[0] for b in H.v3x_param_bounds("pspline", cfg)])
        val = r.neg_loglike(lo)
        assert np.isfinite(val) and val > 0.0

    def test_ridge_is_a_noop_when_lam_is_zero(self):
        cfg = _cfg()
        r = J.SubFloorRidge("pspline", cfg, lam=0.0)
        theta = np.append(np.linspace(-19.0, -26.0, r.centers.size), 1.5)
        assert r.neg_loglike(theta) == 0.0

    def test_ridge_is_off_by_default_and_opt_in_via_cfg(self):
        """Documented default: cfg.v3_sub_floor_ridge_lam is 0 -> no ridge."""
        cfg = _cfg()
        fwd = _fwd(cfg)
        assert float(getattr(cfg, "v3_sub_floor_ridge_lam", 0.0)) == 0.0
        off = J.fit_joint(fwd, "pspline", cfg, None, n_restart=1)
        assert off["sub_floor_ridge"] is False
        cfg.v3_sub_floor_ridge_lam = 5.0
        on = J.fit_joint(fwd, "pspline", cfg, None, n_restart=1)
        assert on["sub_floor_ridge"] is True

    def test_curvature_penalty_stiffens_the_hessian_monotonically(self):
        """lam_spline is the PRIMARY sub-floor lever: raising it can only ADD curvature
        (H(lam2)-H(lam1) = (lam2-lam1) D2^T D2 >= 0), so the minimum eigenvalue is
        non-decreasing and the Hessian becomes positive-definite / well-conditioned."""
        cfg = _cfg(v3_lambda_spline=1.0)
        fwd = _fwd(cfg)
        th = _theta_pspline(cfg)
        mins, conds = [], []
        for lam in (1.0, 10.0, 60.0, 200.0):
            lap = J.joint_laplace(th, fwd, "pspline", cfg, None, n_draw=4, lam_spline=lam)
            assert np.all(np.isfinite(lap["hess"]))
            mins.append(float(np.min(np.linalg.eigvalsh(lap["hess"]))))
            conds.append(lap["cond"])
        assert np.all(np.diff(mins) > -1e-6), mins
        assert conds[-1] < conds[0]
        assert np.isfinite(conds[-1])

    def test_laplace_draws_are_finite_and_inside_the_bounds(self):
        cfg = _cfg()
        fwd = _fwd(cfg)
        th = _theta_pspline(cfg)
        lap = J.joint_laplace(th, fwd, "pspline", cfg, _drop(logN_lo=17.2),
                              n_draw=64, rng=np.random.default_rng(1), lam_spline=60.0)
        assert np.all(np.isfinite(lap["draws"]))
        assert np.all(np.isfinite(lap["cov"]))
        for j, (lo, hi) in enumerate(H.v3x_param_bounds("pspline", cfg)):
            assert lap["draws"][:, j].min() >= lo - 1e-12
            assert lap["draws"][:, j].max() <= hi + 1e-12


# =========================================================================== #
#    SubLLSPrior (the marginalized LLS shape anchor)
# =========================================================================== #
class TestSubLLSPrior:
    def test_prior_vanishes_at_its_own_targets_evaluated_at_the_cfg_pivot(self):
        cfg = _cfg()
        th = _theta_pspline(cfg)
        anchors = np.array([17.5, 18.5, 19.0])
        tgt = np.log10(H.v3x_f_of_N(anchors, cfg.v3_z_pivot, th, "pspline", cfg))
        sp = J.SubLLSPrior(anchors, tgt, [0.1, 0.1, 0.1])
        assert sp.z_eval is None
        assert sp.neg_loglike(th, "pspline", cfg) == pytest.approx(0.0, abs=1e-16)

    def test_z_eval_default_is_the_cfg_pivot_not_a_hardcoded_three(self):
        """The anchor targets are quoted at cfg.v3_z_pivot; evaluating at z=3.0 instead
        changes the amplitude by the gz factor, so the default must not drift."""
        cfg = _cfg()
        th = _theta_pspline(cfg)
        anchors = np.array([17.5, 18.5])
        tgt = np.log10(H.v3x_f_of_N(anchors, cfg.v3_z_pivot, th, "pspline", cfg))
        assert J.SubLLSPrior(anchors, tgt, [0.1, 0.1]).neg_loglike(
            th, "pspline", cfg) == pytest.approx(0.0, abs=1e-16)
        assert J.SubLLSPrior(anchors, tgt, [0.1, 0.1], z_eval=3.0).neg_loglike(
            th, "pspline", cfg) > 1e-3

    def test_prior_is_half_chi2_in_dex(self):
        cfg = _cfg()
        th = _theta_pspline(cfg)
        anchors = np.array([17.5, 19.0])
        tgt = np.log10(H.v3x_f_of_N(anchors, cfg.v3_z_pivot, th, "pspline", cfg))
        sp = J.SubLLSPrior(anchors, tgt + np.array([0.2, -0.4]), [0.1, 0.2])
        assert sp.neg_loglike(th, "pspline", cfg) == pytest.approx(
            0.5 * ((0.2 / 0.1) ** 2 + (0.4 / 0.2) ** 2), rel=1e-6)

    def test_prior_is_finite_when_f_underflows_to_zero(self):
        """The 1e-300 floor: a theta at the lower bound makes f underflow; log10(0)
        would be -inf and poison the optimiser."""
        cfg = _cfg()
        lo = np.array([b[0] for b in H.v3x_param_bounds("pspline", cfg)])
        sp = J.SubLLSPrior([17.5, 18.5], [-19.0, -18.0], [0.1, 0.1])
        val = sp.neg_loglike(lo, "pspline", cfg)
        assert np.isfinite(val) and val > 0.0


# =========================================================================== #
# 5. ell(X) band SEMANTICS — statistical width only; no headline value asserted
# =========================================================================== #
def _fake_band_env(monkeypatch, map_ells, draw_ells, ceiling_first=False):
    """Wire lls_shape_marginalized_band to deterministic fakes so the POOLING arithmetic
    (which is what the band's semantics rest on) is tested exactly."""
    state = {"shape": -1, "first": True}

    def fake_fit_joint(fwd_, fam, cfg_, drop_, sub_lls=None, lam_spline=None, seed=0, **kw):
        state["shape"] = seed
        return dict(theta_map=np.array([float(seed), 0.0]))

    def fake_reduce(cfg_, theta, fine, fam, meta):
        t = np.asarray(theta, float)
        if t[1] == 0.0:                       # the MAP theta
            if ceiling_first and state["first"]:
                state["first"] = False
                return {"ell_lls_extrap": 1e9}
            return {"ell_lls_extrap": map_ells[int(t[0])]}
        return {"ell_lls_extrap": float(t[1])}   # a draw carries its ell in slot 1

    def fake_laplace(theta_map, fwd_, fam, cfg_, drop_, sub_lls=None, n_draw=500,
                     rng=None, lam_spline=None, **kw):
        i = int(np.asarray(theta_map, float)[0])
        d = np.asarray(draw_ells[i], float)
        return dict(draws=np.column_stack([np.full(d.size, float(i)), d]), cond=1e3)

    monkeypatch.setattr(J, "fit_joint", fake_fit_joint)
    monkeypatch.setattr(J, "v3x_reduce", fake_reduce)
    monkeypatch.setattr(J, "joint_laplace", fake_laplace)


class TestEllBandSemantics:
    def test_band_key_default_is_a_real_reduce_output_labelled_extrapolation(self):
        """The default band_key must exist in the reducer's output — and its NAME
        (ell_lls_extrap) is the estimator's own label that this is an extrapolation,
        not an LLS measurement."""
        cfg = _cfg()
        fwd = _fwd(cfg)
        red = J.reduce_theta(_theta_pspline(cfg), fwd, "pspline", cfg)
        key = inspect.signature(J.lls_shape_marginalized_band).parameters["band_key"].default
        assert key == "ell_lls_extrap"
        assert key in red
        assert np.isfinite(red[key])

    def test_single_shape_band_is_the_within_shape_statistical_width_only(self, monkeypatch):
        """DOCUMENTED SEMANTICS: with ONE shape prior the band is nothing but the
        within-shape Laplace (statistical) spread — map_spread is exactly 0, i.e. the
        band carries NO LLS shape systematic.  A single-shape width must never be
        quoted as the total error."""
        rng = np.random.default_rng(0)
        draws = {0: rng.normal(0.2, 0.02, 400)}
        _fake_band_env(monkeypatch, {0: 0.2}, draws)
        out = J.lls_shape_marginalized_band(_fwd(_cfg()), _cfg(), _drop(), [object()],
                                            n_lap=400)
        assert out["n_shapes"] == 1
        assert out["map_spread"] == 0.0
        np.testing.assert_allclose(out["band"], np.percentile(draws[0], (16, 50, 84)))

    def test_band_is_the_pooled_percentile_across_shapes(self, monkeypatch):
        """The between-shape spread enters ONLY by pooling the per-shape draws with
        equal weight; the returned band must be exactly that pooled percentile."""
        rng = np.random.default_rng(1)
        draws = {0: rng.normal(0.15, 0.01, 300), 1: rng.normal(0.30, 0.01, 300),
                 2: rng.normal(0.22, 0.01, 300)}
        _fake_band_env(monkeypatch, {0: 0.15, 1: 0.30, 2: 0.22}, draws)
        out = J.lls_shape_marginalized_band(_fwd(_cfg()), _cfg(), _drop(),
                                            [object()] * 3, n_lap=300)
        pooled = np.concatenate([draws[0], draws[1], draws[2]])
        np.testing.assert_allclose(out["band"], np.percentile(pooled, (16, 50, 84)))
        assert out["map_spread"] == pytest.approx(0.30 - 0.15)
        assert out["n_shapes"] == 3
        # pooling must WIDEN: the shape systematic is not in any single-shape band
        w_pool = out["band"][2] - out["band"][0]
        for ps in out["per_shape"]:
            assert w_pool > ps["q"][2] - ps["q"][0]

    def test_band_excludes_draws_above_the_physical_ceiling(self, monkeypatch):
        """The sub-floor blow-up sends ell to 1e6-1e9; those draws must be dropped, not
        percentile-ed into the band."""
        good = np.linspace(0.1, 0.4, 50)
        blown = np.concatenate([good, np.full(10, 1e8)])
        _fake_band_env(monkeypatch, {0: 0.2}, {0: blown})
        out = J.lls_shape_marginalized_band(_fwd(_cfg()), _cfg(), _drop(), [object()],
                                            n_lap=blown.size)
        np.testing.assert_allclose(out["band"], np.percentile(good, (16, 50, 84)))
        assert out["band"][2] < 50.0

    def test_band_escalates_curvature_when_the_map_breaches_the_ceiling(self, monkeypatch):
        _fake_band_env(monkeypatch, {0: 0.2}, {0: np.linspace(0.1, 0.3, 40)},
                       ceiling_first=True)
        out = J.lls_shape_marginalized_band(_fwd(_cfg()), _cfg(), _drop(), [object()],
                                            lam_spline=60.0, n_lap=40)
        assert out["per_shape"][0]["lam_spline"] == 120.0
        assert out["per_shape"][0]["map_ell"] == pytest.approx(0.2)


# =========================================================================== #
# 6. OMISSION SENSITIVITY — a silently dropped term must break a test
# =========================================================================== #
class TestOmissionSensitivity:
    def test_joint_objective_is_exactly_counting_plus_drop(self):
        """If the drop term were dropped, joint_neg_logP would collapse onto the
        counting-only value.  Pin the exact additive decomposition."""
        cfg = _cfg()
        fwd = _fwd(cfg)
        th = _theta_pspline(cfg)
        d = _drop(logN_lo=17.2)
        only_count = J.joint_neg_logP(th, fwd, "pspline", cfg, None)
        joint = J.joint_neg_logP(th, fwd, "pspline", cfg, d)
        drop_only = J.drop_neg_loglike(th, "pspline", cfg, d)
        assert drop_only > 0.0
        assert joint == pytest.approx(only_count + drop_only, rel=1e-12)
        assert abs(joint - only_count) > 1e-6      # the drop MOVES the objective

    def test_counting_objective_weights_are_not_ignored(self):
        """cat_op['op_weights'] reweights the counting log-likelihood; if the lookup is
        silently skipped the weighted and unweighted objectives coincide."""
        cfg = _cfg()
        fwd_w = _fwd(cfg, with_weights=True)
        fwd_u = dict(fwd_w); fwd_u["cat_op"] = None
        th = _theta_pspline(cfg)
        assert (J.joint_neg_logP(th, fwd_w, "pspline", cfg, None)
                != pytest.approx(J.joint_neg_logP(th, fwd_u, "pspline", cfg, None),
                                 rel=1e-9))
        # a bogus key must fall back to unweighted (guards the key name plumbing)
        assert J.joint_neg_logP(th, fwd_w, "pspline", cfg, None,
                                count_weight_key="nope") == pytest.approx(
            J.joint_neg_logP(th, fwd_u, "pspline", cfg, None), rel=1e-12)

    def test_alpha_fp_latent_changes_the_fit_and_the_objective(self):
        """HARD REQUIREMENT — remove the FP alpha_F latent and the result must change.

        With alpha_fp=True the FP intensity (lam_fp, mu_fp) is SCALED by alpha_F and a
        log-normal prior on log alpha_F is added.  We assert
          (a) alpha_F moves off 1 and the joint optimum strictly improves;
          (b) theta_map itself changes (the latent is not cosmetic);
          (c) the returned neg_logP equals counting(alpha*lam_fp, alpha*mu_fp) + the
              log-normal prior EXACTLY, and does NOT equal the same expression with the
              alpha scaling omitted — the mutant that drops the FP contribution.
        """
        cfg = _cfg()
        fwd = _fwd(cfg, mu_fp=20.0)
        off = J.fit_joint(fwd, "pspline", cfg, None, alpha_fp=False, n_restart=1)
        on = J.fit_joint(fwd, "pspline", cfg, None, alpha_fp=True,
                         alpha_fp_sigma=0.5, n_restart=1)
        assert off["alpha_fp"] == 1.0
        assert abs(on["alpha_fp"] - 1.0) > 1e-3
        assert on["neg_logP"] < off["neg_logP"]
        assert not np.allclose(on["theta_map"], off["theta_map"], rtol=1e-6)

        aF = on["alpha_fp"]
        with_alpha = float(H.v3x_neg_log_posterior(
            on["theta_map"], fwd["A_full"], fwd["M_full"], aF * fwd["lam_fp"],
            aF * fwd["mu_fp"], fwd["fine"], "pspline", cfg, with_grad=False))
        prior = 0.5 * (np.log(aF) / 0.5) ** 2
        assert on["neg_logP"] == pytest.approx(with_alpha + prior, rel=1e-8)

        without_alpha = float(H.v3x_neg_log_posterior(
            on["theta_map"], fwd["A_full"], fwd["M_full"], fwd["lam_fp"],
            fwd["mu_fp"], fwd["fine"], "pspline", cfg, with_grad=False))
        assert abs(without_alpha - with_alpha) > 1e-3, "alpha_F must scale the FP terms"

    def test_joint_gradient_drops_no_extra_term(self):
        """_joint_grad feeds the Laplace Hessian.  With no extra terms it must be the
        BARE analytic counting gradient; each of drop / sub_lls / sub_floor_ridge must
        shift it by a non-zero amount, and the shifts must add up (each term is present
        exactly once)."""
        cfg = _cfg()
        fwd = _fwd(cfg)
        th = _theta_pspline(cfg)
        d = _drop(logN_lo=17.2)
        anchors = np.array([17.5, 19.0])
        tgt = np.log10(H.v3x_f_of_N(anchors, cfg.v3_z_pivot, th, "pspline", cfg)) + 0.5
        sp = J.SubLLSPrior(anchors, tgt, [0.1, 0.1])
        ridge = J.SubFloorRidge("pspline", cfg, lam=10.0)

        base_analytic = H.v3x_neg_log_posterior(
            th, fwd["A_full"], fwd["M_full"], fwd["lam_fp"], fwd["mu_fp"], fwd["fine"],
            "pspline", cfg, obj_weights=None, with_grad=True)[1]
        g_none = J._joint_grad(th, fwd, "pspline", cfg, None, None, None, None)
        np.testing.assert_allclose(g_none, base_analytic, rtol=1e-12, atol=0.0)

        g_drop = J._joint_grad(th, fwd, "pspline", cfg, d, None, None, None)
        g_anch = J._joint_grad(th, fwd, "pspline", cfg, None, sp, None, None)
        g_ridge = J._joint_grad(th, fwd, "pspline", cfg, None, None, ridge, None)
        g_all = J._joint_grad(th, fwd, "pspline", cfg, d, sp, ridge, None)
        for name, g in (("drop", g_drop), ("sub_lls", g_anch), ("ridge", g_ridge)):
            assert np.max(np.abs(g - g_none)) > 1e-6, f"{name} term silently dropped"
        shift_sum = (g_drop - g_none) + (g_anch - g_none) + (g_ridge - g_none)
        np.testing.assert_allclose(g_all - g_none, shift_sum,
                                   rtol=2e-4, atol=1e-6 * np.max(np.abs(shift_sum)))

    def test_lls_anchor_moves_the_map(self):
        """A TIGHT sub-LLS anchor 1 dex away from the counting solution must move the
        fitted theta and shrink its own residual — otherwise the anchor is inert."""
        cfg = _cfg()
        fwd = _fwd(cfg)
        base = J.fit_joint(fwd, "pspline", cfg, None, lam_spline=60.0, n_restart=1)
        anchors = np.array([17.5, 18.5])
        tgt = np.log10(H.v3x_f_of_N(anchors, cfg.v3_z_pivot, base["theta_map"],
                                    "pspline", cfg)) + 1.0
        sp = J.SubLLSPrior(anchors, tgt, [0.05, 0.05])
        anchored = J.fit_joint(fwd, "pspline", cfg, None, sub_lls=sp,
                               lam_spline=60.0, n_restart=1)
        assert not np.allclose(anchored["theta_map"], base["theta_map"], rtol=1e-6)
        assert (sp.neg_loglike(anchored["theta_map"], "pspline", cfg)
                < sp.neg_loglike(base["theta_map"], "pspline", cfg))

    def test_drop_term_moves_the_map(self):
        """The whole point of the joint estimator: the drop must change the inference
        relative to counting-only.  If ``_extra`` silently skipped the drop, the two
        MAPs would be IDENTICAL and the first assertion would fail."""
        cfg = _cfg()
        fwd = _fwd(cfg)
        th_count = J.fit_joint(fwd, "pspline", cfg, None,
                               lam_spline=60.0, n_restart=1)["theta_map"]
        # an ACHIEVABLE drop: tau_hat generated from a theta 0.5 dex above the
        # counting-only MAP, so the joint optimum genuinely exists
        th_target = th_count.copy(); th_target[:-1] += 0.5
        tau_hat = J.drop_tau_model(th_target, "pspline", cfg, [2.6, 2.9, 3.2], 3.5,
                                   logN_lo=17.2)
        d = _drop(tau_hat=tau_hat, sigma=np.maximum(0.05 * tau_hat, 1e-6), logN_lo=17.2)
        th_joint = J.fit_joint(fwd, "pspline", cfg, d,
                               lam_spline=60.0, n_restart=1)["theta_map"]
        assert not np.allclose(th_joint, th_count, rtol=1e-6)
        # the drop residual it is fitted against must actually shrink
        assert (J.drop_neg_loglike(th_joint, "pspline", cfg, d)
                < 0.01 * J.drop_neg_loglike(th_count, "pspline", cfg, d))
        # ... and the joint MAP is a better point of the JOINT objective
        try:
            cfg.v3_lambda_spline = 60.0
            j_joint = J.joint_neg_logP(th_joint, fwd, "pspline", cfg, d)
            j_count = J.joint_neg_logP(th_count, fwd, "pspline", cfg, d)
        finally:
            cfg.v3_lambda_spline = 1.0
        assert j_joint < j_count


# =========================================================================== #
#    driver plumbing
# =========================================================================== #
class TestFitJointPlumbing:
    def test_fit_joint_reports_family_and_success_and_theta_shape(self):
        cfg = _cfg()
        fwd = _fwd(cfg)
        res = J.fit_joint(fwd, "pspline", cfg, None, n_restart=2, seed=0)
        assert res["family"] == "pspline"
        assert res["theta_map"].size == len(H.v3x_param_names("pspline", cfg))
        assert isinstance(res["success"], bool)
        assert np.isfinite(res["neg_logP"])

    def test_alpha_fp_theta_map_excludes_the_latent(self):
        """theta_map is the FAMILY part only; log alpha_F must not leak into it."""
        cfg = _cfg()
        fwd = _fwd(cfg)
        n_p = len(H.v3x_param_names("pspline", cfg))
        res = J.fit_joint(fwd, "pspline", cfg, None, alpha_fp=True, n_restart=1)
        assert res["theta_map"].size == n_p
        assert 0.13 < res["alpha_fp"] < 7.5        # exp of the (-2,2) bound

    def test_reduce_theta_delegates_to_the_frozen_reducer(self):
        cfg = _cfg()
        fwd = _fwd(cfg)
        th = _theta_pspline(cfg)
        got = J.reduce_theta(th, fwd, "pspline", cfg)
        want = H.v3x_reduce(cfg, th, fwd["fine"], "pspline", fwd["M_meta"])
        assert set(got) == set(want)
        np.testing.assert_allclose(got["f_b"], want["f_b"], rtol=0, atol=0)
        assert got["ell_lls_extrap"] == want["ell_lls_extrap"]
