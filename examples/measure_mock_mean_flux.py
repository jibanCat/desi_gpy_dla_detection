"""
examples/measure_mock_mean_flux.py
==================================
Measure the Lyα-forest mean flux F̄(z) (→ effective optical depth
τ_eff(z) = −ln F̄) directly from DESI mock spectra, using the TRUE continuum
saved by quickquasars (--save-continuum, truth-16 TRUE_CONT HDU).

For each QSO (z_qso in [2.1, 4.0]) and the B+R cameras: F = flux / true_cont in
the forest rest range [1040, 1185] Å; accumulate Σflux and Σcont per absorption
redshift bin (z_abs = λ_obs/1215.67 − 1). F̄(z) = Σflux/Σcont (continuum-weighted
mean transmission, noise-robust), τ_eff = −ln F̄.

Overlays the Turner+2024 Lyα term that GP inference assumes:
τ_eff^Lyα(z) = tau_0·(1+z)^beta  (tau_0=0.00246, beta=3.62).
If the mock curve diverges from Turner, the inference mean-flux model is
mismatched to the mock → biases the GP continuum (and NHI).

Usage:
    python examples/measure_mock_mean_flux.py \
        --mock London:/path/to/london/.../spectra-16 \
        --mock 2lpt:/path/to/lyacolore_2lpt/.../spectra-16 \
        --mock Saclay:/path/to/saclay/.../spectra-16 \
        --n-files 2 --out-prefix mock_mean_flux
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LYA = 1215.67
FOREST_RF = (1040.0, 1185.0)
TURNER_TAU0, TURNER_BETA = 0.00246, 3.62


import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpy_dla_detection.effective_optical_depth import effective_optical_depth


def accumulate_mock(spectra_dir, n_files, zbins):
    sumF = np.zeros(len(zbins) - 1)
    sumC = np.zeros(len(zbins) - 1)
    # GP-predicted mean-flux suppression accumulated EXACTLY as null_gp applies it:
    # per-QSO z_qso (so higher Lyman lines from z>z_qso are correctly excluded),
    # continuum-weighted like the mock so the two are apples-to-apples.
    sumGP3 = np.zeros(len(zbins) - 1)
    sumGP31 = np.zeros(len(zbins) - 1)
    nqso = 0
    specs = sorted(glob.glob(os.path.join(spectra_dir, "*", "*", "spectra-16-*.fits")))[:n_files]
    for sp in specs:
        truth = sp.replace("spectra-16-", "truth-16-")
        if not os.path.exists(truth):
            print(f"  [skip] no truth for {os.path.basename(sp)}"); continue
        # true continuum grid + per-target cont
        with fitsio.FITS(truth) as t:
            tc = t["TRUE_CONT"]; hdr = tc.read_header()
            cw = hdr["WMIN"] + hdr["DWAVE"] * np.arange(
                int(round((hdr["WMAX"] - hdr["WMIN"]) / hdr["DWAVE"])) + 1)
            d = tc.read()
            cont_by = {int(tid): c for tid, c in zip(d["TARGETID"], d["TRUE_CONT"])}
            tr = t["TRUTH"].read(columns=["TARGETID", "Z"])
            z_by = {int(tid): float(z) for tid, z in zip(tr["TARGETID"], tr["Z"])}
        with fitsio.FITS(sp) as f:
            tid = np.asarray(f["FIBERMAP"].read(columns=["TARGETID"])["TARGETID"]).astype(int)
            cams = {}
            for c in ("B", "R"):
                cams[c] = (np.asarray(f[f"{c}_WAVELENGTH"].read()),
                           f[f"{c}_FLUX"].read(), f[f"{c}_IVAR"].read())
        for i, t in enumerate(tid):
            zq = z_by.get(t)
            if zq is None or not (2.1 <= zq <= 4.0) or t not in cont_by:
                continue
            cont_arr = np.asarray(cont_by[t], float)
            used = False
            for c in ("B", "R"):
                w, flux, ivar = cams[c]
                fl, iv = flux[i], ivar[i]
                rest = w / (1 + zq)
                m = (rest >= FOREST_RF[0]) & (rest <= FOREST_RF[1]) & (iv > 0)
                if not m.any():
                    continue
                conti = np.interp(w[m], cw, cont_arr)
                g = conti > 0.01
                if not g.any():
                    continue
                zabs = w[m][g] / LYA - 1.0
                idx = np.digitize(zabs, zbins) - 1
                ok = (idx >= 0) & (idx < len(sumF))
                cg = conti[g]
                flg = fl[m][g]
                wobs = w[m][g]
                # GP-predicted suppression at these pixels, THIS QSO's z_qso
                fgp3 = np.exp(-np.nansum(
                    effective_optical_depth(wobs, TURNER_BETA, TURNER_TAU0, zq, 3), axis=1))
                fgp31 = np.exp(-np.nansum(
                    effective_optical_depth(wobs, TURNER_BETA, TURNER_TAU0, zq, 31), axis=1))
                np.add.at(sumF, idx[ok], flg[ok])
                np.add.at(sumC, idx[ok], cg[ok])
                np.add.at(sumGP3, idx[ok], (cg * fgp3)[ok])
                np.add.at(sumGP31, idx[ok], (cg * fgp31)[ok])
                used = True
            nqso += used
    return sumF, sumC, sumGP3, sumGP31, nqso


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="append", required=True,
                    help="LABEL:/path/to/spectra-16  (repeatable)")
    ap.add_argument("--n-files", type=int, default=2)
    ap.add_argument("--out-prefix", default="mock_mean_flux")
    args = ap.parse_args()

    zbins = np.arange(2.0, 4.0 + 1e-9, 0.1)
    zmid = 0.5 * (zbins[:-1] + zbins[1:])
    plt.figure(figsize=(8.5, 5.2))
    results = {}
    fits = {}
    gp3_ref = gp31_ref = None
    def tau_of(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            F = np.where(den > 0, num / den, np.nan)
            tau = -np.log(np.clip(F, 1e-6, None))
        tau[~np.isfinite(F) | (den <= 0)] = np.nan
        return tau
    for spec in args.mock:
        label, d = spec.split(":", 1)
        sumF, sumC, sumGP3, sumGP31, nq = accumulate_mock(d, args.n_files, zbins)
        tau = tau_of(sumF, sumC)
        results[label] = (zmid, sumF / np.where(sumC > 0, sumC, np.nan), tau)
        gp3_ref = tau_of(sumGP3, sumC)    # GP curves coincide across mocks (same z_qso dist)
        gp31_ref = tau_of(sumGP31, sumC)
        # fit tau_eff = tau_0 (1+z)^beta  via log-linear regression on finite bins
        fin = np.isfinite(tau) & (tau > 0)
        beta_f, lnt0 = np.polyfit(np.log(1 + zmid[fin]), np.log(tau[fin]), 1)
        tau0_f = np.exp(lnt0)
        fits[label] = (tau0_f, beta_f, int(nq))
        print(f"[{label}] {nq} QSOs  -> fit tau_0={tau0_f:.5f} beta={beta_f:.3f}  "
              f"(Turner 0.00246/3.62; ratio tau0={tau0_f/TURNER_TAU0:.2f}x, dbeta={beta_f-TURNER_BETA:+.2f})")
        plt.plot(zmid, tau, "o-", lw=1.6, ms=4, label=f"{label} (n={nq}): "
                 rf"$\tau_0$={tau0_f:.4f}, $\beta$={beta_f:.2f}")
    zt = np.linspace(2.0, 4.0, 100)
    plt.plot(zt, TURNER_TAU0 * (1 + zt) ** TURNER_BETA, "k--", lw=2,
             label=r"Turner24 Ly$\alpha$ only: $0.00246(1+z)^{3.62}$")
    if gp3_ref is not None:
        plt.plot(zmid, gp3_ref, "k:", lw=2, label="GP mean-flux model, 3-line (per-QSO z_qso)")
        plt.plot(zmid, gp31_ref, color="gray", lw=1.5, label="GP mean-flux model, 31-line (training)")
    plt.xlabel("absorption redshift z"); plt.ylabel(r"$\tau_{\rm eff}(z) = -\ln\bar F$")
    plt.title("Mock mean-flux vs GP model (correct per-QSO z_qso)")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(args.out_prefix + ".png", dpi=130)
    np.savez(args.out_prefix + ".npz", zmid=zmid, gp3=gp3_ref, gp31=gp31_ref,
             turner_tau0=TURNER_TAU0, turner_beta=TURNER_BETA,
             **{f"tau_{k}": v[2] for k, v in results.items()},
             **{f"fit_{k}": np.array(fits[k][:2]) for k in fits})
    print("\n=== effective (tau_0, beta) per mock vs Turner24 (0.00246, 3.620) ===")
    print(f"{'mock':>14} {'tau_0':>9} {'beta':>7} {'tau0/Turner':>12} {'beta-Turner':>12}")
    for k, (t0, bf, nq) in fits.items():
        print(f"{k:>14} {t0:>9.5f} {bf:>7.3f} {t0/TURNER_TAU0:>11.2f}x {bf-TURNER_BETA:>+12.2f}")
    print(f"\n[out] {args.out_prefix}.png + .npz")


if __name__ == "__main__":
    main()
