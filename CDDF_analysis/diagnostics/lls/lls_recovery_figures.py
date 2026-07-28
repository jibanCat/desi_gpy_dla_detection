"""lls_recovery_figures.py — measurement-vs-truth figure set for the LLS joint drop+count
estimator, on the 2LPT-0 mock. Re-derivable + git-stamped (companion to
`joint_drop_count_validation.py`, whose stamped numbers this reproduces as panel 1/2).

Each panel is a UNIT TEST of one reporting decision, plotted against the injected mock truth:

  P1  drop headline        measured tau_hat(z912) [direct-sum over HCD truth] vs the MAP drop model,
                           + lambda_mfp est-vs-truth.  -> is the headline recovered?
  P2  ell(X) recovery      MAP + shape-marginalized band vs truth, for BOTH [17.2,19.5) and
                           [17.5,19.5).  -> does the band cover truth? does the tau>=2 band do better?
  P3  sub-LLS bias curve   ell_MAP vs phi = tau(<17.2)/tau_total, with the literature phi band
                           shaded.  -> how far does ignoring sub-LLS push ell off truth?  [DECISIVE]
  P4  FP sensitivity       ell MAP + band vs the FP scale alpha_F.  -> is the missing FP width
                           bigger than the reported band?
  P5  shape families       band from 3 single-PL SLOPES (committed) vs 3 genuine FAMILIES
                           (single-PL / PW14 spline / broken-PL).  -> does the label 'shape-
                           marginalized' buy width?
  P6  band choice          opacity weight w(N)=1-exp(-N sigma912) vs logN, with tau=1 (17.20) and
                           tau=2 (17.50); kappa/ell over each band.  -> is [17.2,19.5) the right
                           reporting band?
  P7  drop normalization   narrow-z_qso-bin vs wide-sample tau_hat/tau_model vs z912 — the 2026-07-07
                           bug, kept visible as a regression guard.

MOCK values only (public-OK). Real-LOA is PI-unblinding-gated.

.. warning::

   **B16 (z-leaky truth) contaminates BOTH ell(X) truths this routine plots and stamps.**
   Documented 2026-07-28; see `joint_drop_count_validation.py`'s module docstring for the
   full trace. `true_172`/`true_175` (:202-203) integrate `tr["f_truth"]` (:194), which
   `cddf_catalog_hbi.py::truth_reductions` builds WITHOUT the `t_zidx >= 0` mask it applies
   to `dndx_total` (:2140-2142 vs :2153-2154). Measured on 2LPT-0:

     * `truth.ell_172_195` 0.2628520 -> 0.2487742 ; `P2_ell.r0_172` 0.8176435 -> **0.8639**
     * `truth.ell_175_195` 0.2118017 -> 0.2003832 ; `P2_ell.r0_175` 0.4802301 -> **0.5076**

   Both corrected truths equal, to 1 ULP, the z-masked `dndx_tru_{172,175}_195` in the
   committed `CDDF_analysis/hbi/lls_mock_validation.json` -- an independent leg computed by
   a different function in a different routine. Guarded by
   `tests/test_b16_ell_contamination.py`.

   `ell_frac_below_17p5_truth` (:449) is a RATIO of two leaky truths, so the leak nearly
   cancels: 0.194217 -> 0.194515. The "0.194 truth vs 0.522 model" structural-limit
   statement is unaffected. The P1 lambda_mfp truth leg is clean (direct sum over the HCD
   truth catalogue); its estimate leg inherits a +0.026 dex shape-prior anchor shift worth
   +0.12% on `r0_lambda_mfp`.

   `CDDF_analysis/hbi/lls_recovery_figures.json` on disk is UNTRACKED and carries the
   pre-correction values.

Re-derive:
    python CDDF_analysis/diagnostics/lls/lls_recovery_figures.py --force
    python CDDF_analysis/diagnostics/lls/lls_recovery_figures.py --force --quick   # small n_lap
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from astropy.table import Table  # noqa: E402

from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB  # noqa: E402
from CDDF_analysis.hbi.cddf_catalog_hbi import v3x_f_of_N, v3x_reduce  # noqa: E402
import CDDF_analysis.hbi.joint_drop_count as J  # noqa: E402
from CDDF_analysis.lyc import opacity as LYC  # noqa: E402
from CDDF_analysis.diagnostics.lls.joint_drop_count_validation import (  # noqa: E402
    _build_forward, _physical_drop, _git_commit, _shape_priors,
    Z912, BETA, SHAPE_SLOPES)
from gpy_dla_detection.generate_samples import f_pw14  # noqa: E402

LN10 = np.log(10.0)
OUT_DIR = os.path.join(_REPO, "CDDF_analysis", "hbi", "figures", "lls_recovery")
OUT_JSON = os.path.join(_REPO, "CDDF_analysis", "hbi", "lls_recovery_figures.json")

# Sub-LLS injection (P3): a diffuse power law f_sub(N) ~ N^-BETA_SUB over [15, 17.2).
BETA_SUB = 1.5                       # Rudie+2013-like optically-thin forest slope
PHI_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)   # tau(<17.2) / tau_total; literature ~0.4-0.6
ALPHA_F_GRID = (0.5, 0.75, 1.0, 1.25, 1.5)  # FP intensity scale (loa0 <-> purity bracket proxy)
ANCHORS = (17.5, 18.5, 19.2)
ANCHOR_SIG = (0.15, 0.15, 0.10)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ell_integral(theta, cfg, a, b, z0=3.0, n=200):
    """ell(X) = INT_a^b f(N,z0) dN = INT f * N ln10 dlogN, from theta. The pspline f is separable
    in (N,z), so the RATIO of two such integrals is z0-independent — asserted in run()."""
    g = np.linspace(a, b, n)
    N = 10.0 ** g
    f = v3x_f_of_N(g[:, None], np.array([[z0]]), theta, "pspline", cfg)[:, 0]
    y = f * N * LN10
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(g)))


def _tau_sub_unit(z912_arr, z_qso, om, sigma912, beta, n_zprime=80):
    """tau(z912) from a UNIT-amplitude diffuse sub-LLS f_sub(N) = N^-BETA_SUB on [15,17.2).
    Linear in the amplitude, so a single evaluation calibrates any phi."""
    g = np.linspace(15.0, 17.2, 80)
    N = 10.0 ** g
    f_unit = N ** (-BETA_SUB)
    tau = np.zeros(len(z912_arr))
    for i, z912 in enumerate(z912_arr):
        if z912 >= z_qso:
            continue
        zp = np.linspace(float(z912), float(z_qso), n_zprime)
        dXdz = J.path_length_int(zp, om)
        sig = sigma912 * ((1.0 + z912) / (1.0 + zp)) ** beta
        opac = 1.0 - np.exp(-(N[:, None] * sig[None, :]))
        integrand = f_unit[:, None] * (N[:, None] * LN10) * opac
        integ_N = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(g)[:, None], axis=0)
        gz = integ_N * dXdz
        tau[i] = float(np.sum(0.5 * (gz[1:] + gz[:-1]) * np.diff(zp)))
    return tau


def _physical_drop_wide():
    """The WRONG (pre-2026-07-07) normalization: divide by ALL z_qso>=3 sightlines. Kept as the
    regression guard for P7 — only ~30% of them reach z912=3.35, so tau_hat is diluted z-dependently."""
    hcd = Table.read(AB.DEF_TRUTH)
    zc = Table.read(os.path.join(os.path.dirname(AB.DEF_TRUTH), "zcat.fits"))
    zq = {int(t): float(z) for t, z in zip(zc["TARGETID"], zc["Z"])}
    wide = {int(t) for t in zc["TARGETID"] if zq.get(int(t), 0.0) >= 3.0}
    n_sl = len(wide)
    keep = np.array([int(t) in wide for t in hcd["TARGETID"]])
    aN = (10.0 ** np.asarray(hcd["NHI"], float))[keep]
    az = np.asarray(hcd["Z"], float)[keep]
    zqa = np.array([zq[int(t)] for t in hcd["TARGETID"][keep]])
    sg = LYC.SIGMA_912
    tau = np.array([
        float(np.sum(1.0 - np.exp(-(aN[m] * sg * ((1 + zz) / (1 + az[m])) ** BETA)))) / n_sl
        for zz in Z912 for m in [(az > zz) & (az < zqa)]])
    return tau, n_sl


def _family_priors(f_truth, lo, hi):
    """Three GENUINELY different shape families (vs the committed 3 single-PL slopes).
      F0 single-PL   : slope -1.5, anchored to the mock truth f(19.2)   [= the committed canonical]
      F1 PW14 spline : log10 f_pw14(logN) at its OWN absolute level (PchipInterpolator, 1310.0052)
      F2 broken-PL   : slope -2.0 below 18.0, -1.3 above; anchored to PW14 at 19.2
    NOTE f_pw14 takes log10 N, NOT linear N (passing 10**logN clamps to the last node).
    """
    mid = 0.5 * (lo + hi)
    f192 = np.log10(max(f_truth[np.argmin(np.abs(mid - 19.2))], 1e-300))
    a = np.array(ANCHORS)

    single = [f192 + (-1.5) * (x - 19.2) for x in a]
    pw14 = [float(np.log10(f_pw14(np.array([x]))[0])) for x in a]
    pw192 = float(np.log10(f_pw14(np.array([19.2]))[0]))
    broken = [pw192 + (-1.3) * (x - 19.2) if x >= 18.0
              else pw192 + (-1.3) * (18.0 - 19.2) + (-2.0) * (x - 18.0) for x in a]

    fams = dict(single_PL=single, PW14_spline=pw14, broken_PL=broken)
    priors = [J.SubLLSPrior(list(a), v, list(ANCHOR_SIG)) for v in fams.values()]
    return priors, fams


def _band_from_shapes(fwd, cfg, drop, shapes, lam_spline, n_lap):
    return J.lls_shape_marginalized_band(fwd, cfg, drop, shapes, lam_spline=lam_spline, n_lap=n_lap)


def _band_both(fwd, cfg, drop, shapes, lam_spline, n_lap, q=(16, 50, 84)):
    """Mirror of lls_shape_marginalized_band, but reduce EACH Laplace draw to ell on BOTH bands
    (per-draw ratio, not the MAP ratio). ell[17.5,19.5) = ell_reduce[17.2,19.5) * I(17.5)/I(17.2),
    exact because pspline f is separable in (N,z) (asserted in run())."""
    rng = np.random.default_rng(0)
    ceil = float(getattr(cfg, "v3_lls_ell_ceiling", 50.0))
    e172, e175 = [], []
    for i, sp in enumerate(shapes):
        lam_i = float(lam_spline)
        for _ in range(5):
            res = J.fit_joint(fwd, "pspline", cfg, drop, sub_lls=sp, lam_spline=lam_i, seed=i)
            m = float(v3x_reduce(cfg, res["theta_map"], fwd["fine"], "pspline",
                                 fwd.get("M_meta"))["ell_lls_extrap"])
            if np.isfinite(m) and m <= ceil:
                break
            lam_i *= 2.0
        lap = J.joint_laplace(res["theta_map"], fwd, "pspline", cfg, drop, sub_lls=sp,
                              n_draw=n_lap, rng=rng, lam_spline=lam_i)
        for th in lap["draws"]:
            a = float(v3x_reduce(cfg, th, fwd["fine"], "pspline", fwd.get("M_meta"))["ell_lls_extrap"])
            if not np.isfinite(a) or a > ceil:
                continue
            i172 = _ell_integral(th, cfg, 17.2, 19.5)
            if i172 <= 0:
                continue
            e172.append(a)
            e175.append(a * _ell_integral(th, cfg, 17.5, 19.5) / i172)
    return (np.percentile(e172, q).tolist(), np.percentile(e175, q).tolist(),
            int(len(e172)))


def _scaled_fp(fwd, aF):
    g = dict(fwd)
    g["lam_fp"] = fwd["lam_fp"] * aF
    g["mu_fp"] = fwd["mu_fp"] * aF
    return g


# ---------------------------------------------------------------------------
def run(n_lap, n_lap_sweep, lam_spline):
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg, fwd, tr, ing, lo, hi, Nb, dNb = _build_forward()
    fine = fwd["fine"]
    f_truth = np.asarray(tr["f_truth"], float)
    om = float(getattr(cfg, "Omega_m", 0.279))
    cosmo = LYC.Cosmology(Om=om)

    # ---- truth on both bands (snap to actual bin edges) ----
    def _true_ell(a, b):
        sel = (lo >= a - 1e-9) & (hi <= b + 1e-9)
        return float(np.nansum(f_truth[sel] * dNb[sel])), sel.sum()
    true_172, n172 = _true_ell(17.2, 19.5)
    true_175, n175 = _true_ell(17.5, 19.5)
    edge_ok = bool(np.min(np.abs(lo - 17.5)) < 1e-6)
    print(f"[truth] ell[17.2,19.5)={true_172:.4f} ({n172} bins)   "
          f"ell[17.5,19.5)={true_175:.4f} ({n175} bins)  edge17.5_exact={edge_ok}", flush=True)

    tau_hat, n_sl, z_qso_eff = _physical_drop()
    sigma = np.maximum(0.10 * tau_hat, 0.02)
    drop = J.DropData(Z912, tau_hat, sigma, z_qso_eff, sigma912=LYC.SIGMA_912, beta=BETA, logN_lo=17.2)
    shapes = _shape_priors(f_truth, lo, hi)

    # ================= P1: drop headline =================
    fj = J.fit_joint(fwd, "pspline", cfg, drop, sub_lls=shapes[1], lam_spline=lam_spline, seed=1)
    th_map = fj["theta_map"]
    tau_est = J.drop_tau_model(th_map, "pspline", cfg, Z912, z_qso_eff,
                               sigma912=LYC.SIGMA_912, beta=BETA, logN_lo=17.2)
    chi2 = float(np.sum(((tau_est - tau_hat) / sigma) ** 2)) / Z912.size
    kap_t = LYC.fit_kappa(Z912, tau_hat, z_qso_eff, beta=BETA, cosmo=cosmo)
    kap_e = LYC.fit_kappa(Z912, tau_est, z_qso_eff, beta=BETA, cosmo=cosmo)
    lam_t = LYC.lambda_mfp_from_kappa(kap_t, z_qso_eff, beta=BETA, cosmo=cosmo)
    lam_e = LYC.lambda_mfp_from_kappa(kap_e, z_qso_eff, beta=BETA, cosmo=cosmo)
    r0_lam = lam_e / lam_t
    print(f"[P1] lambda_mfp truth={lam_t:.1f} est={lam_e:.1f} R0={r0_lam:.3f} chi2/dof={chi2:.3f}", flush=True)

    # separability self-check: the [17.5,19.5)/[17.2,19.5) ratio must not depend on z0
    r_lo = _ell_integral(th_map, cfg, 17.5, 19.5, z0=2.5) / _ell_integral(th_map, cfg, 17.2, 19.5, z0=2.5)
    r_hi = _ell_integral(th_map, cfg, 17.5, 19.5, z0=3.5) / _ell_integral(th_map, cfg, 17.2, 19.5, z0=3.5)
    sep_ok = bool(abs(r_lo - r_hi) < 1e-6)
    print(f"[check] band-ratio z-separability: {r_lo:.6f} vs {r_hi:.6f} -> {sep_ok}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].errorbar(Z912, tau_hat, yerr=sigma, fmt="o", color="k", label=r"measured $\hat\tau$ (HCD truth)")
    ax[0].plot(Z912, tau_est, "-", color="crimson", label=r"MAP model $\tau(\theta)$")
    ax[0].set_xlabel(r"$z_{912}$"); ax[0].set_ylabel(r"$\tau_{\rm eff,LL}$")
    ax[0].set_title(rf"P1 drop: $\chi^2/{{\rm dof}}={chi2:.2f}$"); ax[0].legend(frameon=False)
    ax[1].bar([0, 1], [lam_t, lam_e], color=["0.4", "crimson"], width=.55)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["truth", "estimated"])
    ax[1].set_ylabel(r"$\lambda_{\rm mfp}$ [proper Mpc]")
    ax[1].set_title(rf"$R_0 = {r0_lam:.3f}$  (prior-free headline)")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/P1_drop_headline.png", dpi=140); plt.close(fig)

    # ================= P2: ell recovery + the CDDF shape behind it =================
    band = _band_from_shapes(fwd, cfg, drop, shapes, lam_spline, n_lap)
    b172_only = np.array(band["band"])
    b172_l, b175_l, n_draw_ok = _band_both(fwd, cfg, drop, shapes, lam_spline, n_lap)
    b172, b175 = np.array(b172_l), np.array(b175_l)
    map_172 = float(v3x_reduce(cfg, th_map, fine, "pspline", fwd.get("M_meta"))["ell_lls_extrap"])
    map_175 = map_172 * r_hi
    cov172 = bool(b172[0] <= true_172 <= b172[2])
    cov175 = bool(b175[0] <= true_175 <= b175[2])
    # where does ell live inside the band? truth vs model
    frac_lo_truth = 1.0 - true_175 / true_172          # fraction of ell in [17.2,17.5)
    frac_lo_model = 1.0 - r_hi
    print(f"[P2] [17.2,19.5): band={np.round(b172,4)} truth={true_172:.4f} cover={cov172} R0={b172[1]/true_172:.3f}",
          flush=True)
    print(f"[P2] [17.5,19.5): band={np.round(b175,4)} truth={true_175:.4f} cover={cov175} R0={b175[1]/true_175:.3f}",
          flush=True)
    print(f"[P2] ell fraction in [17.2,17.5):  truth={frac_lo_truth:.3f}   MODEL={frac_lo_model:.3f}  "
          f"<- floor pile-up  (n_draw={n_draw_ok})", flush=True)

    # left: recovered f(N) vs truth (model rescaled to the reduce normalization -> shape+level honest).
    # START AT THE KNOT FLOOR: below the lowest pspline knot the B-spline basis is all-zeros, so
    # log f = basis@coeffs = 0 -> f = 1 (spurious, theta-independent). Plotting below 17.2 would
    # draw that artifact as if it were the model (it is exactly what _clamp_drop_grid suppresses).
    knot_lo = float(J._v3x_knot_span("pspline", cfg)[0])
    gg = np.linspace(knot_lo, 19.8, 300)
    scale = map_172 / _ell_integral(th_map, cfg, 17.2, 19.5)
    f_mod = v3x_f_of_N(gg[:, None], np.array([[3.0]]), th_map, "pspline", cfg)[:, 0] * scale
    mid = 0.5 * (lo + hi)
    msk = (mid >= knot_lo) & (mid < 19.8) & (f_truth > 0)

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax[0].step(mid[msk], np.log10(f_truth[msk]), where="mid", color="k", lw=2, label="mock truth $f(N)$")
    ax[0].plot(gg, np.log10(np.clip(f_mod, 1e-300, None)), color="crimson", lw=2, label="MAP $f(N)$")
    ax[0].axvline(17.2, color="0.4", ls="--", lw=1, label="fit floor / knot floor (17.2)")
    ax[0].axvline(17.498, color="0.4", ls=":", lw=1, label=r"$\tau=2$ (17.50)")
    ax[0].axvspan(17.2, 17.498, color="0.8", alpha=.5)
    ax[0].set_xlabel(r"$\log_{10} N_{\rm HI}$"); ax[0].set_ylabel(r"$\log_{10} f(N,X)$")
    ax[0].set_title("P2a  recovered CDDF vs truth: pile-up at the floor")
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")
    ax[0].annotate(f"ℓ in [17.2,17.5):\ntruth {100*frac_lo_truth:.0f}%   model {100*frac_lo_model:.0f}%\n"
                   f"(no sub-LLS term ⇒ the drop's\n<17.2 opacity is dumped at the floor)",
                   (.30, .06), xycoords="axes fraction", fontsize=8.5,
                   bbox=dict(boxstyle="round", fc="w", ec="0.6"))
    ax[0].text(.02, .02, "model undefined below the knot floor (f≡1 there)",
               transform=ax[0].transAxes, fontsize=7, color="0.45")

    for k, (bb, tv) in enumerate([(b172, true_172), (b175, true_175)]):
        ax[1].errorbar([k], [bb[1]], yerr=[[bb[1] - bb[0]], [bb[2] - bb[1]]], fmt="s",
                       color="crimson", capsize=6, ms=8,
                       label="estimated band (q16/50/84)" if k == 0 else None)
        ax[1].plot([k - .18, k + .18], [tv, tv], "-", lw=2.4, color="k",
                   label="mock truth" if k == 0 else None)
        ax[1].annotate(f"$R_0$={bb[1]/tv:.2f}\ncovers={'YES' if bb[0]<=tv<=bb[2] else 'NO'}",
                       (k + .22, bb[1]), fontsize=9, va="center")
    ax[1].set_xlim(-.5, 1.7); ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels([r"$[17.2,19.5)$", r"$[17.5,19.5)$  ($\tau\geq2$)"])
    ax[1].set_ylabel(r"$\ell(X)$")
    ax[1].set_title("P2b  ℓ recovery (FP + sub-LLS width NOT in the band)")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/P2_ell_recovery.png", dpi=140); plt.close(fig)

    # ================= P3: sub-LLS bias curve (DECISIVE) =================
    tsub_unit = _tau_sub_unit(Z912, z_qso_eff, om, LYC.SIGMA_912, BETA)
    iref = len(Z912) // 2
    p3 = []
    for phi in PHI_GRID:
        amp = phi * tau_hat[iref] / tsub_unit[iref]
        tau_resid = tau_hat - amp * tsub_unit
        if np.any(tau_resid <= 0):
            print(f"  [P3] phi={phi}: residual drop <=0, stop", flush=True); break
        d = J.DropData(Z912, tau_resid, np.maximum(0.10 * tau_resid, 0.02), z_qso_eff,
                       sigma912=LYC.SIGMA_912, beta=BETA, logN_lo=17.2)
        f = J.fit_joint(fwd, "pspline", cfg, d, sub_lls=shapes[1], lam_spline=lam_spline, seed=1)
        e = float(v3x_reduce(cfg, f["theta_map"], fine, "pspline", fwd.get("M_meta"))["ell_lls_extrap"])
        p3.append(dict(phi=phi, map_ell=e, bias_vs_phi0=np.nan))
        print(f"  [P3] phi={phi:.2f}  ell_MAP={e:.4f}", flush=True)
    e0 = p3[0]["map_ell"]
    for r in p3:
        r["bias_vs_phi0"] = e0 / r["map_ell"] - 1.0 if r["map_ell"] > 0 else np.nan

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ph = [r["phi"] for r in p3]; el = [r["map_ell"] for r in p3]
    ax.axvspan(0.4, 0.6, color="gold", alpha=.22, label=r"literature $\phi$ (Fumagalli/Rudie)")
    ax.plot(ph, el, "o-", color="crimson",
            label=r"$\ell$ when a fraction $\phi$ of the drop is correctly given to sub-LLS")
    ax.axhline(e0, color="crimson", ls="--", lw=1.2,
               label=r"what the CURRENT estimator reports (it forces $\phi=0$)")
    ax.plot([0], [true_172], "*", ms=16, color="k", label=r"mock truth (valid only at $\phi=0$)")
    ax.annotate("the mock is HCD-only,\nso it sits at φ=0 by construction\n— it cannot test this axis",
                (0.02, e0), textcoords="offset points", xytext=(24, -34), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=.9))
    for r in p3:
        if r["phi"] in (0.3, 0.5):
            ax.annotate(rf"$\times${e0/r['map_ell']:.1f} over-report", (r["phi"], r["map_ell"]),
                        textcoords="offset points", xytext=(6, 10), fontsize=8.5, color="crimson")
    ax.set_xlabel(r"$\phi = \tau(<17.2)\,/\,\tau_{\rm total}$  (true sub-LLS share of the drop)")
    ax.set_ylabel(r"$\ell(X)[17.2,19.5)$")
    ax.set_title("P3  sub-LLS mis-attribution — the dominant, mock-untestable systematic")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right"); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/P3_sublls_bias.png", dpi=140); plt.close(fig)

    # ================= P4: FP sensitivity =================
    p4 = []
    for aF in ALPHA_F_GRID:
        g = _scaled_fp(fwd, aF)
        bb = _band_from_shapes(g, cfg, drop, shapes, lam_spline, n_lap_sweep)["band"]
        p4.append(dict(alpha_F=aF, band=list(map(float, bb))))
        print(f"  [P4] aF={aF:.2f}  band={np.round(bb,4)}", flush=True)
    q50 = [r["band"][1] for r in p4]
    fp_swing = (max(q50) - min(q50)) / true_172
    band_w = (b172[2] - b172[0]) / true_172

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    aFs = [r["alpha_F"] for r in p4]
    lo_ = [r["band"][0] for r in p4]; hi_ = [r["band"][2] for r in p4]
    ax.fill_between(aFs, lo_, hi_, color="crimson", alpha=.2, label="reported band (no FP width)")
    ax.plot(aFs, q50, "o-", color="crimson", label=r"$\ell$ median")
    ax.axhline(true_172, color="k", lw=2.2, label="mock truth")
    ax.axvline(1.0, color="0.5", ls=":", label=r"committed $\alpha_F=1$ (loa0)")
    ax.set_xlabel(r"FP intensity scale $\alpha_F$"); ax.set_ylabel(r"$\ell(X)[17.2,19.5)$")
    ax.set_title(f"P4  FP swing {100*fp_swing:.0f}% of truth  vs  reported band width {100*band_w:.0f}%")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/P4_fp_sensitivity.png", dpi=140); plt.close(fig)

    # ================= P5: slopes vs genuine families =================
    fam_priors, fam_anchor = _family_priors(f_truth, lo, hi)
    band_fam = _band_from_shapes(fwd, cfg, drop, fam_priors, lam_spline, n_lap)
    bf = np.array(band_fam["band"])
    # like-for-like: both bands from lls_shape_marginalized_band (b172_only), not the _band_both variant
    w_slopes = b172_only[2] - b172_only[0]
    w_fams = bf[2] - bf[0]
    print(f"[P5] width slopes={w_slopes:.4f} (spread {band['map_spread']:.4f})  "
          f"families={w_fams:.4f} (spread {band_fam['map_spread']:.4f})", flush=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for k, (bb, lab) in enumerate([(b172_only, f"3 single-PL slopes\n{SHAPE_SLOPES}"),
                                   (bf, "3 genuine families\nsingle-PL / PW14 / broken-PL")]):
        ax.errorbar([k], [bb[1]], yerr=[[bb[1] - bb[0]], [bb[2] - bb[1]]], fmt="s",
                    color="crimson", capsize=6, ms=8)
        ax.annotate(f"width={bb[2]-bb[0]:.4f}", (k + .12, bb[1]), fontsize=9, va="center")
    ax.axhline(true_172, color="k", lw=2.2, label="mock truth")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["3 slopes\n(committed)", "3 families\n(genuine)"])
    ax.set_ylabel(r"$\ell(X)[17.2,19.5)$")
    ax.set_title("P5  does 'shape-marginalized' buy width?")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/P5_shape_families.png", dpi=140); plt.close(fig)

    # ================= P6: opacity weight / band choice =================
    g = np.linspace(16.5, 20.0, 400)
    w = 1.0 - np.exp(-(10.0 ** g) * LYC.SIGMA_912)
    logN_t1 = float(np.log10(1.0 / LYC.SIGMA_912))
    logN_t2 = float(np.log10(2.0 / LYC.SIGMA_912))

    def _kappa_over_ell(a, b, n=300):
        gg = np.linspace(a, b, n); NN = 10.0 ** gg
        f = v3x_f_of_N(gg[:, None], np.array([[3.0]]), th_map, "pspline", cfg)[:, 0]
        ww = 1.0 - np.exp(-NN * LYC.SIGMA_912)
        num = np.trapezoid(f * NN * LN10 * ww, gg)
        den = np.trapezoid(f * NN * LN10, gg)
        return float(num / den)
    kl_172 = _kappa_over_ell(17.2, 19.5)
    kl_175 = _kappa_over_ell(17.5, 19.5)
    print(f"[P6] kappa/ell  [17.2,19.5)={kl_172:.4f}   [17.5,19.5)={kl_175:.4f}  "
          f"(tau=1 @ {logN_t1:.3f}, tau=2 @ {logN_t2:.3f})", flush=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(g, w, color="crimson", lw=2)
    ax.axvspan(17.2, 17.5, color="0.75", alpha=.5, label=r"$[17.2,17.5)$: drop+prior only,"
                                                         "\nunreachable by break counting")
    ax.axvline(logN_t1, color="k", ls="--", lw=1, label=rf"$\tau=1$ ({logN_t1:.2f})")
    ax.axvline(logN_t2, color="k", ls=":", lw=1, label=rf"$\tau=2$ ({logN_t2:.2f})")
    ax.set_xlabel(r"$\log_{10} N_{\rm HI}$"); ax.set_ylabel(r"opacity weight $1-e^{-N\sigma_{912}}$")
    ax.set_title(rf"P6  $\kappa/\ell$: {kl_172:.3f} on $[17.2,19.5)$ vs {kl_175:.3f} on $[17.5,19.5)$")
    ax.legend(frameon=False, loc="lower right"); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/P6_band_choice.png", dpi=140); plt.close(fig)

    # ================= P7: drop-normalization regression guard =================
    tau_wide, n_wide = _physical_drop_wide()
    r_narrow = tau_hat / tau_est
    r_wide = tau_wide / tau_est
    print(f"[P7] narrow/model mean={r_narrow.mean():.3f} std={r_narrow.std():.3f}  |  "
          f"wide/model mean={r_wide.mean():.3f} std={r_wide.std():.3f}  (n_wide={n_wide})", flush=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(Z912, r_narrow, "o-", color="crimson", label=rf"narrow $z_{{\rm qso}}\in[3.4,3.6]$ (n={n_sl})")
    ax.plot(Z912, r_wide, "s--", color="steelblue", label=rf"wide $z_{{\rm qso}}\geq3$ (n={n_wide}) — THE BUG")
    ax.axhline(1.0, color="k", lw=1.2)
    ax.set_xlabel(r"$z_{912}$"); ax.set_ylabel(r"$\hat\tau\,/\,\tau_{\rm model}$")
    ax.set_title("P7  drop normalization: coverage dilution is z-dependent (regression guard)")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/P7_drop_normalization.png", dpi=140); plt.close(fig)

    return dict(
        wall_s=round(time.time() - t0, 1),
        checks=dict(band_ratio_z_separable=sep_ok, bin_edge_17p5_exact=edge_ok,
                    ell_ratio_175_over_172=float(r_hi)),
        truth=dict(ell_172_195=true_172, ell_175_195=true_175),
        P1_headline=dict(lambda_mfp_truth_Mpc=lam_t, lambda_mfp_est_Mpc=lam_e, r0_lambda_mfp=r0_lam,
                         kappa912_truth=kap_t, kappa912_est=kap_e, drop_chi2_per_dof=chi2,
                         tau_hat=tau_hat.tolist(), tau_est=tau_est.tolist(), z912=Z912.tolist(),
                         z_qso_eff=z_qso_eff, n_sl_narrow=int(n_sl)),
        P2_ell=dict(band_172=b172.tolist(), band_175=b175.tolist(),
                    band_172_committed_api=b172_only.tolist(),
                    map_172=map_172, map_175=map_175, n_draw_ok=n_draw_ok,
                    r0_172=b172[1] / true_172, r0_175=b175[1] / true_175,
                    covers_truth_172=cov172, covers_truth_175=cov175,
                    ell_frac_below_17p5_truth=frac_lo_truth,
                    ell_frac_below_17p5_model=frac_lo_model,
                    note="model piles ell against the 17.2 floor because it has no sub-LLS(<17.2) "
                         "term; R0=0.81 on [17.2,19.5) is two errors partly cancelling, and the "
                         "tau>=2 band [17.5,19.5) exposes it (R0~0.48)."),
        theta_map=np.asarray(th_map, float).tolist(),
        P3_sublls=dict(beta_sub=BETA_SUB, phi_grid=list(PHI_GRID), points=p3,
                       over_report_factor={str(r["phi"]): (e0 / r["map_ell"]) for r in p3 if r["map_ell"] > 0},
                       note="fit has NO sub-LLS term (forces phi=0); phi = fraction of the measured "
                            "drop truly from <17.2. over_report_factor = ell(phi=0)/ell(phi) = the "
                            "factor by which the current estimator over-reports ell on real data."),
        P4_fp=dict(points=p4, fp_swing_frac_of_truth=fp_swing, band_width_frac_of_truth=band_w,
                   n_lap=n_lap_sweep),
        P5_shape=dict(slopes=dict(band=b172.tolist(), width=float(w_slopes),
                                  map_spread=float(band["map_spread"])),
                      families=dict(band=bf.tolist(), width=float(w_fams),
                                    map_spread=float(band_fam["map_spread"]),
                                    anchors_logf=fam_anchor, anchor_logN=list(ANCHORS))),
        P6_band=dict(kappa_over_ell_172=kl_172, kappa_over_ell_175=kl_175,
                     logN_tau1=logN_t1, logN_tau2=logN_t2),
        P7_norm=dict(ratio_narrow=r_narrow.tolist(), ratio_wide=r_wide.tolist(),
                     n_sl_narrow=int(n_sl), n_sl_wide=int(n_wide)),
    )


def main(a):
    t = time.time()
    n_lap = 60 if a.quick else a.n_lap
    n_lap_sweep = 40 if a.quick else a.n_lap_sweep
    res = run(n_lap, n_lap_sweep, a.lam_spline)
    out = dict(metadata=dict(
        what="LLS joint estimator: measurement-vs-truth figure set (7 panels) on the 2LPT-0 mock",
        mock="2LPT-0 (loa-124); MOCK values (public-OK), NOT real-LOA",
        code_commit=_git_commit("CDDF_analysis/diagnostics/lls/lls_recovery_figures.py"),
        wallclock_s=round(time.time() - t, 1),
        quick=bool(a.quick), n_lap=n_lap, n_lap_sweep=n_lap_sweep, lam_spline=a.lam_spline,
        figures=OUT_DIR,
        rederive="python CDDF_analysis/diagnostics/lls/lls_recovery_figures.py --force",
    ), result=res)
    if os.path.exists(a.out) and not a.force:
        print(f"[skip-json] {a.out} exists (pass --force).")
    else:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"[saved-json] {a.out}  code_commit={out['metadata']['code_commit']}")
    print(f"[figures] {OUT_DIR}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true", help="small n_lap for a fast smoke run")
    ap.add_argument("--n-lap", type=int, default=400, dest="n_lap")
    ap.add_argument("--n-lap-sweep", type=int, default=150, dest="n_lap_sweep")
    ap.add_argument("--lam-spline", type=float, default=60.0, dest="lam_spline")
    main(ap.parse_args())
