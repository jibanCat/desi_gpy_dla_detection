"""lyc_mfp_inversion.py — D3 of the LLS drop estimator.

Invert the composite Lyman-limit decrement to the ionizing mean free path and close the loop
against the injected truth. Uses the injected-mock composite from lyc_inject_closure.py.

Method (PW09 / Worseck14, with the exponent matched to OUR injection sigma ~ nu^-BETA, BETA=3):
  tau_eff,LL(z912, z_QSO) = kappa912 * (c/H0) * (1+z912)^BETA * INT_{z912}^{z_QSO} (1+z')^-(BETA+2.5) dz'
  with z912(lambda) = (1+z_QSO)*(lambda_rest/912) - 1  (QSO-rest lambda below 912).
  Fit the single normalization kappa912 to the measured tau_eff(z912) in the fit window
  (rest 850-905 A, avoiding the >905 proximity edge per Fumagalli+13). Then
  lambda_mfp := the PROPER distance from z_QSO to the z912 where tau_eff = 1.
CLOSURE: lambda_mfp from the MEASURED composite (T_inj/T_base) vs from the injected TRUTH
transmission (both fit the same way) — tests the inversion machinery, not just the stack.
NOTE: on the HCD-only (>=17.2) injection the population is optically THICK, so the power-law
kernel is an imperfect model (a documented caveat); the sub-LLS that a physical lambda_mfp
needs is below the injection floor (Fumagalli ~40%).
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

C_KMS = 299792.458
H0 = 70.0           # km/s/Mpc
OM, OL = 0.3, 0.7
LYLIM = 911.76
BETA = 3.0          # our injection cross-section index (sigma ~ nu^-3)


def Ez(z):
    return np.sqrt(OM * (1 + z) ** 3 + OL)


def proper_distance(z_lo, z_hi):
    """Proper (physical) distance between z_lo<z_hi, Mpc: INT c/(H0 (1+z) E(z)) dz."""
    zz = np.linspace(z_lo, z_hi, 400)
    integ = C_KMS / (H0 * (1 + zz) * Ez(zz))
    return float(np.sum(0.5 * (integ[1:] + integ[:-1]) * np.diff(zz)))


def kernel_basis(z912, z_q):
    """The tau_eff shape (everything except kappa912): (1+z912)^BETA * INT (1+z')^-(BETA+2.5)."""
    e = BETA + 2.5
    integral = ((1 + z912) ** (1 - e) - (1 + z_q) ** (1 - e)) / (e - 1)   # INT_{z912}^{z_q}
    return (C_KMS / H0) * (1 + z912) ** BETA * integral                    # units: Mpc (times kappa912 [1/Mpc])


def lambda_mfp_from_kappa(kappa, z_q):
    """Proper distance from z_q to the z912 where tau_eff=1, given kappa912."""
    zg = np.linspace(z_q - 1.5, z_q - 1e-3, 3000)
    tau = kappa * kernel_basis(zg, z_q)
    if tau[0] < 1:                     # never reaches 1 in range
        return np.nan
    z_tau1 = np.interp(1.0, tau[::-1], zg[::-1])   # tau increases as zg decreases
    return proper_distance(z_tau1, z_q)


def fit_kappa(z912, tau, z_q):
    """Least-squares kappa912 for tau = kappa * basis (through origin)."""
    b = kernel_basis(z912, z_q)
    good = np.isfinite(tau) & np.isfinite(b) & (b > 0)
    if good.sum() < 4:
        return np.nan
    return float(np.sum(b[good] * tau[good]) / np.sum(b[good] ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--closure", default="/tmp/lyc_closure_smoke/lyc_closure.npz")
    ap.add_argument("--out", default="/tmp/lyc_closure_smoke")
    args = ap.parse_args()
    Z = np.load(args.closure)
    rest, zb = Z["rest"], Z["z_bins"]
    Tm, Tt = Z["Tlyc_meas"], Z["Tlyc_truth"]

    fitwin = (rest >= 850) & (rest < 905)     # Fumagalli fit window; red edge 905 avoids proximity
    print(f"{'z_QSO bin':>14} | {'kappa912 (meas)':>16} {'kappa912 (truth)':>16} | "
          f"{'lmfp meas':>10} {'lmfp truth':>10} {'ratio':>7}")
    print("-" * 92)
    rows = []
    for b in range(len(zb)):
        z_q = 0.5 * (zb[b][0] + zb[b][1])
        z912 = (1 + z_q) * (rest / LYLIM) - 1.0
        sel = fitwin & (rest < LYLIM) & np.isfinite(Tm[b]) & (Tm[b] > 0) & (Tm[b] < 1) \
              & np.isfinite(Tt[b]) & (Tt[b] > 0) & (Tt[b] < 1)
        if sel.sum() < 5:
            continue
        tau_m = -np.log(Tm[b][sel]); tau_t = -np.log(Tt[b][sel]); zg = z912[sel]
        km = fit_kappa(zg, tau_m, z_q); kt = fit_kappa(zg, tau_t, z_q)
        lm = lambda_mfp_from_kappa(km, z_q); lt = lambda_mfp_from_kappa(kt, z_q)
        ratio = lm / lt if (lt and np.isfinite(lt)) else np.nan
        print(f"  {zb[b][0]:.2f}-{zb[b][1]:.2f}     | {km:16.5g} {kt:16.5g} | "
              f"{lm:10.2f} {lt:10.2f} {ratio:7.3f}")
        rows.append(dict(z_q=z_q, kappa_meas=km, kappa_truth=kt, lmfp_meas=lm, lmfp_truth=lt))
    # Worseck14 reference (nu^-2.75 fit) for scale context (real IGM, not this HCD-only mock)
    for r in rows:
        w = 37.0 * ((1 + r["z_q"]) / 5.0) ** (-5.4)
        print(f"    [ref] Worseck14 lambda_mfp(z={r['z_q']:.2f}) = {w:.1f} h70^-1 Mpc "
              f"(real IGM; our HCD-only mock is thick-dominated, not directly comparable)")
    np.savez(os.path.join(args.out, "lyc_mfp_inversion.npz"), rows=rows)
    print(f"\n[note] closure = lmfp_meas/lmfp_truth ratio (should be ~1 -> inversion recovers the "
          f"injected truth). Absolute lambda_mfp is HCD-only (>=17.2), not the physical IGM value.")


if __name__ == "__main__":
    main()
