"""lyc_inject_closure.py — inject a KNOWN Lyman-continuum opacity into the 2LPT-0 mock
(from the HCD truth catalog) and close the loop on the drop estimator.

The 2LPT-0 mock has NO bound-free Lyman-continuum opacity (quickquasars injected only the
Lyman-SERIES lines) — verified. So the drop -> lambda_mfp estimator cannot be validated on
it as-is. Here we ADD the continuum ourselves, per sightline, from the discrete HCD truth
(log N_HI >= 17.2), giving a mock with a KNOWN LyC drop:

  for each foreground HCD k (z_k < z_QSO, from hcd_truth_cat: N_k, z_k),
      flux(lambda_obs) *= exp[ -N_k * sigma912 * (lambda_obs / (912*(1+z_k)))^3 ]  for
      lambda_obs < 912*(1+z_k)   (bound-free cross-section ~ nu^-3 above threshold).

Estimator fixes (from the 2026-07-05 double referee panel): the composite is the
INVERSE-VARIANCE-WEIGHTED MEAN transmission <F>/<C> (flux and continuum stacked SEPARATELY,
using the mock TRUE_CONT), NOT a sigma-clip median (which biases <T> and deletes the LLS
signal). The pure LyC transmission is isolated EMPIRICALLY as T_inj / T_base (dividing out
the common Lyman-series forest = empirical series subtraction).

Closure: does T_LyC_measured = T_inj/T_base recover the ANALYTIC mean injected transmission
computed straight from the HCD catalog? And does the derived tau_eff,LL / lambda_mfp match?

Run (smoke): python examples/lyc_inject_closure.py --limit-healpix 40 --out /tmp/lyc_closure
Requires the gpdla env (desispec + healpy + fitsio).
"""
from __future__ import annotations
# ---------------------------------------------------------------------------
# Physics cross-check (Prochaska + Fumagalli series, web-verified 2026-07-05):
# the injection below is VERIFIED correct (sigma ~ nu^-3, per-absorber tau, mean-of-exp).
# The inj/base mock closure is weighting- and Lyman-series-invariant. The ABSOLUTE
# tau_eff -> lambda_mfp fit (D3, real data) MUST additionally adopt, per Fumagalli+2013 /
# PWO09 / Worseck14:
#   1. EQUAL-WEIGHT arithmetic mean <F/C> (NOT inverse-var, NOT median) + a QSO S/N floor.
#   2. proximity cut dv<=3000 km/s of z_QSO; fit window rest 830-905 A (red edge avoids the
#      proximity zone -> else MFP biased high).
#   3. fix the Lyman-series tau_eff^Ly(z) from f(N), fit only the LL term (on the mock the
#      series lives in the baseline -> inj/base removes it empirically).
#   4. continuum = Telfer broken power law (real data); TRUE_CONT (mock).
#   5. inversion exponent consistent with sigma: nu^-3 => tau_eff ~ (1+z912)^3 int (1+z')^-5.5,
#      redshift exponent -4.5 (Worseck's -4.25 is for nu^-2.75); lambda_mfp := where tau_eff=1.
#   6. KEY: logN>=17.5 is only ~40% of the LyC opacity (Fumagalli+2013) -> an HCD-only (>=17.2)
#      injection captures ~40-45% of the real MFP; the diffuse sub-LLS dominates.
# See notes/2026-07-05_lls_drop_walkthrough.md Step 6.
# ---------------------------------------------------------------------------

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table
import fitsio

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from preload_spectra.preload_2lpt_simple import (
    _read_one_healpix_file, _spec_path, _healpix_for_radec, _build_targetid_filter)

SIGMA_912 = 6.35e-18  # HI photoionization cross-section at 1 Ryd (cm^2), Verner et al. 1996
# Bound-free cross-section index sigma(nu) ~ (nu/nu_912)^-BETA_LL below the limit:
#   3.0 = the standard near-threshold hydrogenic value (PW09, Verner asymptotic);
#   2.75 = the Worseck+2014 effective index over nu_912..~4 nu_912.
# The "Lyman-limit recovery" of flux below 912 A is governed by (i) this cross-section
# power law AND (ii) the INTRINSIC quasar continuum, which is itself a power law with an
# FUV softening below ~1000 A (alpha_lambda - 0.72; Telfer+2002 / O'Meara+13 / Romano+19).
# On the mock we divide by the exact TRUE_CONT so (ii) is handled; a real-data run must
# model that power-law + break continuum (D1).
BETA_LL = 3.0
LYMAN_LIMIT = 911.76
DEF_MOCKDIR = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
               "v2.8.5/mock-0/loa-124")
Z_QSO_BINS = [(3.0, 3.3), (3.3, 3.6), (3.6, 3.9)]
REST_GRID = np.arange(840.0, 1220.0, 0.5)
TRUE_CONT_WAVE = 3500.0 + 2.0 * np.arange(3251)   # observed grid of the TRUE_CONT HDU


def lyc_optical_depth(wave_obs, z_abs, nhi, beta=BETA_LL):
    """Sum bound-free LyC optical depth over a sightline's HCDs, on the observed grid.
    tau(lambda) = sum_k N_k sigma912 (lambda/(912(1+z_k)))^beta  for lambda < 912(1+z_k).
    (lambda/edge)^beta == (nu/nu_912)^-beta; beta=3 standard, 2.75 = Worseck+2014."""
    tau = np.zeros_like(wave_obs)
    for zk, nk in zip(np.atleast_1d(z_abs), np.atleast_1d(nhi)):
        edge = LYMAN_LIMIT * (1.0 + zk)              # observed Lyman limit of this absorber
        below = wave_obs < edge
        N = 10.0 ** nk if nk < 30 else nk            # accept log or linear N
        tau[below] += N * SIGMA_912 * (wave_obs[below] / edge) ** beta
    return tau


def load_hcd_by_tid(mockdir: Path, nhi_min=17.2):
    hcd = Table.read(mockdir / "hcd_truth_cat.fits")
    N = np.asarray(hcd["NHI"], float)
    z = np.asarray(hcd["Z"], float)
    tid = np.asarray(hcd["TARGETID"])
    keep = N >= nhi_min
    N, z, tid = N[keep], z[keep], tid[keep]
    order = np.argsort(tid)
    tid, z, N = tid[order], z[order], N[order]
    uniq, start = np.unique(tid, return_index=True)
    end = np.r_[start[1:], len(tid)]
    return {int(t): (z[s:e], N[s:e]) for t, s, e in zip(uniq, start, end)}


def _bin(z):
    for i, (lo, hi) in enumerate(Z_QSO_BINS):
        if lo <= z < hi:
            return i
    return -1


def build(mockdir: Path, limit_healpix, exclude_bal=True):
    zcat = Table.read(mockdir / "zcat.fits")
    z = np.asarray(zcat["Z"], float)
    zw = np.asarray(zcat["ZWARN"], float) if "ZWARN" in zcat.colnames else np.zeros(len(zcat))
    keep = (z >= 3.0) & (z < 3.9) & (zw == 0)
    keep &= _build_targetid_filter(zcat, mockdir, exclude_hcd=False, exclude_bal=exclude_bal)
    zcat = zcat[keep]
    tid = np.asarray(zcat["TARGETID"]); zq = np.asarray(zcat["Z"], float)
    hpx = _healpix_for_radec(np.asarray(zcat["TARGET_RA"], float),
                             np.asarray(zcat["TARGET_DEC"], float))
    zof = dict(zip(tid.tolist(), zq.tolist()))
    print(f"[zcat] {len(zcat)} QSOs (z 3.0-3.9, non-BAL)")
    hcd_by_tid = load_hcd_by_tid(mockdir)
    print(f"[hcd] {len(hcd_by_tid)} sightlines carry >=1 HCD (logN>=17.2)")

    ng = REST_GRID.size
    nb = len(Z_QSO_BINS)
    # weighted sums: base flux, injected flux, continuum, per bin
    swFb = np.zeros((nb, ng)); swFi = np.zeros((nb, ng)); swC = np.zeros((nb, ng)); npx = np.zeros((nb, ng))
    # analytic injected LyC transmission (mean over sightlines), per bin
    aT = np.zeros((nb, ng)); aN = np.zeros((nb, ng))

    uniq = np.unique(hpx)
    if limit_healpix:
        uniq = uniq[:limit_healpix]
    t0 = time.time(); nqso = 0
    for k, hp in enumerate(uniq):
        sf = _spec_path(mockdir, int(hp))
        if not sf.exists():
            continue
        tf = str(sf).replace("spectra-16-", "truth-16-")
        try:
            tc = fitsio.read(tf, ext="TRUE_CONT")
        except Exception:
            continue
        tcmap = {int(t): np.asarray(tc["TRUE_CONT"])[i] for i, t in enumerate(np.asarray(tc["TARGETID"]))}
        for (t, wave, flux, ivar, mask) in _read_one_healpix_file(sf, tid[hpx == hp].tolist()):
            zz = zof.get(int(t)); b = _bin(zz) if zz is not None else -1
            if b < 0 or int(t) not in tcmap:
                continue
            cont = np.interp(wave, TRUE_CONT_WAVE, tcmap[int(t)], left=np.nan, right=np.nan)
            good = (mask == 0) & (ivar > 0) & np.isfinite(flux) & np.isfinite(cont) & (cont > 0)
            if good.sum() < 50:
                continue
            # LyC injection for this sightline
            zk, nk = hcd_by_tid.get(int(t), (np.array([]), np.array([])))
            zk = zk[zk < zz]; nk = nk[:zk.size] if zk.size else nk[:0]
            tau = lyc_optical_depth(wave, zk, nk) if zk.size else np.zeros_like(wave)
            Tlyc = np.exp(-tau)
            rest = wave / (1.0 + zz)
            def onto(a):
                return np.interp(REST_GRID, rest[good], a[good], left=np.nan, right=np.nan)
            Fb = onto(flux); Fi = onto(flux * Tlyc); C = onto(cont); W = onto(ivar); TL = onto(Tlyc)
            m = np.isfinite(Fb) & np.isfinite(C) & np.isfinite(W) & (W > 0)
            swFb[b, m] += W[m] * Fb[m]; swFi[b, m] += W[m] * Fi[m]; swC[b, m] += W[m] * C[m]; npx[b, m] += 1
            ma = np.isfinite(TL)
            aT[b, ma] += TL[ma]; aN[b, ma] += 1
            nqso += 1
        if (k + 1) % 25 == 0:
            print(f"  ...{k+1}/{len(uniq)} healpix, {nqso} QSOs ({time.time()-t0:.0f}s)")

    ok = (swC > 0) & (npx >= 10)
    Tbase = np.where(ok, swFb / swC, np.nan)
    Tinj = np.where(ok, swFi / swC, np.nan)
    Tlyc_meas = np.where(ok, Tinj / Tbase, np.nan)          # empirical series-subtracted LyC
    Tlyc_truth = np.where(aN >= 10, aT / np.maximum(aN, 1), np.nan)  # analytic injected mean
    return dict(rest=REST_GRID, Tbase=Tbase, Tinj=Tinj, Tlyc_meas=Tlyc_meas,
                Tlyc_truth=Tlyc_truth, npx=npx, nqso=nqso)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mockdir", default=DEF_MOCKDIR)
    ap.add_argument("--out", default="/tmp/lyc_closure")
    ap.add_argument("--limit-healpix", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = build(Path(args.mockdir), args.limit_healpix)
    np.savez(os.path.join(args.out, "lyc_closure.npz"), z_bins=np.array(Z_QSO_BINS), **R)

    rest = R["rest"]
    print("\n== CLOSURE: measured (T_inj/T_base) vs analytic injected LyC, sub-912 windows ==")
    for b, (lo, hi) in enumerate(Z_QSO_BINS):
        for wlo, whi in [(895, 910), (860, 895), (845, 875)]:
            w = (rest >= wlo) & (rest < whi)
            meas = np.nanmean(R["Tlyc_meas"][b][w]); tru = np.nanmean(R["Tlyc_truth"][b][w])
            if np.isfinite(tru):
                print(f"  z_Q {lo:.1f}-{hi:.1f}  [{wlo},{whi}): measured={meas:.3f}  truth={tru:.3f}  "
                      f"ratio={meas/tru:.3f}")

    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for b, (lo, hi) in enumerate(Z_QSO_BINS):
        ax = axes[b]; z = (rest > 845) & (rest < 1050)
        ax.plot(rest[z], R["Tbase"][b][z], color="#9aa1b2", lw=1, label="baseline (no inject)")
        ax.plot(rest[z], R["Tinj"][b][z], color="#4b45c4", lw=1, label="injected")
        ax.plot(rest[z], R["Tlyc_meas"][b][z], color="#2f7d5b", lw=1.6, label="LyC measured =inj/base")
        ax.plot(rest[z], R["Tlyc_truth"][b][z], color="#b04a3f", lw=1.6, ls="--", label="LyC analytic truth")
        ax.axvline(LYMAN_LIMIT, color="k", ls=":", lw=1)
        ax.set_title(f"z_QSO {lo:.1f}-{hi:.1f}"); ax.set_xlabel("QSO rest (Å)"); ax.set_ylim(0, 1.1); ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean transmission"); axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("LyC injection closure: does inj/base recover the analytic injected Lyman-continuum drop?", fontsize=11)
    fig.tight_layout()
    out_png = os.path.join(args.out, "lyc_closure.png"); fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[saved] {out_png}\n[saved] {os.path.join(args.out,'lyc_closure.npz')}  ({R['nqso']} QSOs)")


if __name__ == "__main__":
    main()
