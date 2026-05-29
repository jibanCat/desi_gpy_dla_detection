"""
examples/prototype_dla_clustering.py
====================================
PROTOTYPE (design-review only, not the production module) for the DLA
velocity-separation clustering prior. Computes the analytic clustering
function ξ_DLA(Δv,z) = b²·D(z)²·ξ_matter(r), r = Δv(1+z)/H(z), with a
self-contained Eisenstein-Hu (1998) NO-WIGGLE linear P(k) (no camb/classy),
σ8-normalized, FT'd to ξ_matter(r). Overlays the empirical 1+ξ(Δv) measured
from the mock truth (the b=2 cross-check) and the per-model normalization.

Sanity prints (verify before trusting the figure): reproduced σ8, ξ_matter at
8 Mpc/h (z=0), growth D(2.5)/D(0).

Generates a 4-panel figure: (1) ξ_DLA(Δv) analytic vs empirical; (2) ξ_matter(r)
+ r↔Δv mapping; (3) per-sample log ρ_k weight + additive-vs-multiplicative;
(4) ⟨ξ⟩_window + Z_k normalization.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

# --- Cosmology: LyaCoLoRe input = Planck 2015 XIII Table 3 base-ΛCDM TT+lowP
# (Farr+2019 §4.1; confirmed by the cosmology referee 2026-05-22). Om=0.3156,
# Ωb h²=0.02222 (→Ob0=0.0491 at h=0.673), H0=67.3, ns=0.9645, σ8=0.831.
Om0, Ob0, H0, ns, SIGMA8 = 0.3156, 0.0491, 67.31, 0.9645, 0.831
TCMB = 2.7255
B_DLA = 2.0
cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, Tcmb0=TCMB)
h = H0 / 100.0
LYA = 1215.67


# --- Eisenstein-Hu 1998 no-wiggle transfer function (k in h/Mpc) -----------------
def T_nowiggle(k):
    om_m = Om0 * h**2
    om_b = Ob0 * h**2
    theta = TCMB / 2.7
    s = 44.5 * np.log(9.83 / om_m) / np.sqrt(1.0 + 10.0 * om_b**0.75)  # Mpc
    fb = om_b / om_m
    alpha = (1.0 - 0.328 * np.log(431.0 * om_m) * fb
             + 0.38 * np.log(22.3 * om_m) * fb**2)
    ks = k * s * h  # k[h/Mpc]*s[Mpc] -> need k in 1/Mpc => k*h
    gamma_eff = Om0 * h * (alpha + (1.0 - alpha) / (1.0 + (0.43 * ks) ** 4))
    q = k * (theta**2 / gamma_eff)  # k in h/Mpc
    L0 = np.log(2.0 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)


def Pk_unnorm(k):
    return k**ns * T_nowiggle(k) ** 2


def _sigma2(R, norm=1.0):
    # σ²(R) = (1/2π²) ∫ P(k) k² W(kR)² dk, W = 3(sin x - x cos x)/x³, R in Mpc/h
    def integ(lnk):
        k = np.exp(lnk)
        x = k * R
        w = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
        return norm * Pk_unnorm(k) * k**3 * w**2 / (2.0 * np.pi**2)  # k³ from dk=k dlnk
    val, _ = quad(integ, np.log(1e-4), np.log(1e3), limit=200)
    return val


# normalize P(k) so σ8 = SIGMA8
NORM = SIGMA8**2 / _sigma2(8.0, norm=1.0)


def Pk(k):
    return NORM * Pk_unnorm(k)


def _xi_matter_one(rr):
    # ξ(r) = (1/2π²) ∫ P(k) k² j0(kr) dk, j0 = sin(kr)/(kr); r in Mpc/h, k in h/Mpc
    def integ(lnk):
        k = np.exp(lnk)
        x = k * rr
        j0 = np.sin(x) / x
        damp = np.exp(-((k / 50.0) ** 2))  # mild smoothing to suppress ringing
        return Pk(k) * k**3 * j0 / (2.0 * np.pi**2) * damp
    val, _ = quad(integ, np.log(1e-4), np.log(1e3), limit=300)
    return val


# build a cached log-r interpolator once (the quad is the bottleneck)
_R_GRID = np.logspace(-1.0, 2.6, 300)  # 0.1 .. ~400 Mpc/h
_XI_GRID = np.array([_xi_matter_one(rr) for rr in _R_GRID])
from scipy.interpolate import interp1d
_XI_INTERP = interp1d(np.log(_R_GRID), _XI_GRID, kind="cubic",
                      bounds_error=False, fill_value=(_XI_GRID[0], 0.0))


def xi_matter_z0(r):
    r = np.atleast_1d(r).astype(float)
    return _XI_INTERP(np.log(np.clip(r, _R_GRID[0], _R_GRID[-1])))


# --- Linear growth D(z), normalized D(0)=1 --------------------------------------
def growth_D(z):
    def Ea(a):
        return np.sqrt(Om0 * a**-3 + (1.0 - Om0))
    def integ(a):
        return 1.0 / (a * Ea(a)) ** 3
    def D_unnorm(a):
        val, _ = quad(integ, 1e-6, a, limit=200)
        return Ea(a) * val
    a = 1.0 / (1.0 + np.atleast_1d(z))
    D0 = D_unnorm(1.0)
    return np.array([D_unnorm(ai) for ai in a]) / D0


# --- r(Δv,z) and ξ_DLA ----------------------------------------------------------
def r_of_dv(dv_kms, z):
    # comoving LOS separation [Mpc/h] for a velocity split Δv at redshift z
    Hz = cosmo.H(z).value  # km/s/Mpc
    r_mpc = dv_kms * (1.0 + z) / Hz  # Mpc
    return r_mpc * h  # -> Mpc/h


def xi_dla(dv_kms, z, b=B_DLA):
    r = r_of_dv(dv_kms, z)
    Dz = growth_D(z)[0]
    return b**2 * Dz**2 * xi_matter_z0(r)


# --- ⟨ξ⟩_window and Z_k ----------------------------------------------------------
def mean_xi_window(L_kms, z, b=B_DLA, n=400):
    # mean of ξ over the triangular pair-separation pdf p(d)=2(L-d)/L², d∈[0,L]
    d = np.linspace(1.0, L_kms, n)
    pdf = 2.0 * (L_kms - d) / L_kms**2
    xi = xi_dla(d, z, b)
    trapz = getattr(np, "trapezoid", np.trapz)
    return trapz(xi * pdf, d)


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    z0 = 2.5

    # --- sanity prints ---
    s8 = np.sqrt(_sigma2(8.0, norm=NORM))
    Dz = growth_D(z0)[0]
    print(f"[sanity] reproduced sigma8 = {s8:.4f}  (target {SIGMA8})")
    print(f"[sanity] xi_matter(8 Mpc/h, z=0) = {xi_matter_z0(8.0)[0]:.4f}")
    print(f"[sanity] D({z0})/D(0) = {Dz:.4f}   (D^2 = {Dz**2:.4f})")
    print(f"[sanity] H({z0}) = {cosmo.H(z0).value:.1f} km/s/Mpc")
    for dv in (200, 500, 800, 1500, 3000):
        r = r_of_dv(dv, z0)
        print(f"[map] dv={dv:5d} km/s -> r={r:6.2f} Mpc/h ; xi_DLA={xi_dla(dv,z0)[0]:.3f} ; 1+xi={1+xi_dla(dv,z0)[0]:.3f}")

    # --- empirical 1+xi from the mock truth ---
    emp_path = "/scratch/cavestru_root/cavestru0/mfho/dla_clustering_london0_nhi203.npz"
    emp = None
    if os.path.exists(emp_path):
        d = np.load(emp_path)
        print(f"[emp] keys: {list(d.files)}")
        emp = d

    dv = np.linspace(50, 4000, 200)
    xi = xi_dla(dv, z0)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (1) 1+xi_DLA(dv): analytic vs empirical
    a0 = ax[0, 0]
    a0.plot(dv, 1 + xi, "b-", lw=2, label=f"analytic b={B_DLA}: 1+ξ_DLA(Δv), z={z0}")
    if emp is not None:
        # try common key layouts
        keys = {k.lower(): k for k in emp.files}
        xk = keys.get("dv") or keys.get("dv_mid") or keys.get("dv_bins") or keys.get("v") or emp.files[0]
        yk = keys.get("one_plus_xi") or keys.get("1+xi") or keys.get("xi") or keys.get("oneplusxi")
        try:
            xv = emp[xk]; yv = emp[yk] if yk else None
            if yv is not None:
                if "xi" == (yk or "").lower() and yv.min() < 0.5:
                    yv = 1 + yv
                a0.plot(xv[:len(yv)], yv, "ks", ms=5, label="empirical mock truth (NHI≥20.3)")
        except Exception as e:
            print(f"[emp] overplot skipped: {e}")
    for f, lab in [(812, "MIN_Z_SEP=3000 (Δz=.01)"), (160, "→160 km/s")]:
        a0.axvline(f, ls=":", color="gray"); a0.text(f, a0.get_ylim()[1]*0.9, lab, rotation=90, fontsize=7, va="top")
    a0.set_xlim(0, 3500); a0.set_ylim(0.8, None)
    a0.set_xlabel("Δv [km/s]"); a0.set_ylabel("1 + ξ_DLA"); a0.legend(fontsize=8); a0.grid(alpha=0.3)
    a0.set_title("(1) Clustering weight: analytic b=2 vs empirical mock")

    # (2) xi_matter(r) + r<->dv
    a1 = ax[0, 1]
    rr = np.logspace(-0.3, 1.78, 200)  # ~0.5..60 Mpc/h (the forest Δv<3000 range)
    a1.loglog(rr, xi_matter_z0(rr), "g-", lw=2, label="ξ_matter(r, z=0)")
    a1.loglog(rr, B_DLA**2 * Dz**2 * xi_matter_z0(rr), "b--", lw=2, label=f"b²D²(z={z0}) ξ_matter")
    a1.set_xlabel("r [Mpc/h]"); a1.set_ylabel("ξ"); a1.legend(fontsize=8); a1.grid(alpha=0.3, which="both")
    a1.set_title("(2) Matter & DLA correlation vs comoving r")

    # (3) per-sample log rho_k weight + additive vs multiplicative at k=3
    a2 = ax[1, 0]
    a2.plot(dv, np.log1p(xi), "b-", lw=2, label="k=2: log(1+ξ) (additive≡mult)")
    # k=3 equal-spaced triple: pairs (d, d, 2d)
    xi_d = xi_dla(dv, z0); xi_2d = xi_dla(2*dv, z0)
    add3 = np.log1p(2*xi_d + xi_2d)
    mult3 = 2*np.log1p(xi_d) + np.log1p(xi_2d)
    a2.plot(dv, add3, "r-", lw=2, label="k=3 additive log(1+Σξ)")
    a2.plot(dv, mult3, "r:", lw=2, label="k=3 multiplicative Σlog(1+ξ) [overcounts]")
    a2.set_xlabel("Δv [km/s] (pair spacing)"); a2.set_ylabel("log ρ_k"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
    a2.set_title("(3) Per-sample log-weight; additive vs multiplicative")

    # (4) <xi>_window and Z_k vs window width
    a3 = ax[1, 1]
    Lw = np.linspace(500, 12000, 40)
    mxi = np.array([mean_xi_window(L, z0) for L in Lw])
    a3.plot(Lw, mxi, "m-", lw=2, label="⟨ξ⟩_window")
    for k in (2, 3, 4):
        Ck2 = k * (k - 1) / 2
        a3.plot(Lw, np.log1p(Ck2 * mxi), lw=1.6, label=f"log Z_{k} = log(1+{int(Ck2)}⟨ξ⟩)")
    a3.set_xlabel("z-DLA window width L [km/s]"); a3.set_ylabel("⟨ξ⟩ / log Z_k"); a3.legend(fontsize=8); a3.grid(alpha=0.3)
    a3.set_title("(4) Normalization: ⟨ξ⟩_window and log Z_k")

    plt.tight_layout()
    out = os.path.join(outdir, "prototype_dla_clustering.png")
    plt.savefig(out, dpi=130)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
