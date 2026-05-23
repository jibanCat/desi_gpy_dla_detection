"""
gpy_dla_detection/dla_clustering.py
===================================
DLA two-point clustering prior for the multi-DLA evidence (gated, default-off).

xi_DLA(dv, z) = b_DLA^2 * [D(z)/D(0)]^2 * xi_matter(r), r = dv(1+z)/H(z) [-> Mpc/h],
with a small-scale cap (linear-bias xi->inf as r->0 is unphysical). The per-k weight
is the additive (leading-order) log rho_k = log(1 + sum_{i<j} xi_DLA(dv_ij)), floored.

Cosmology = LyaCoLoRe's Planck-2015 input (Farr+2019 sec 4.1; cosmology referee
2026-05-22): Om=0.3156, Ob h^2=0.02222, H0=67.31, ns=0.9645, sigma8=0.831. P(k) is
the Eisenstein-Hu 1998 no-wiggle transfer function (no camb/classy), sigma8-normalized.
See docs/superpowers/specs/2026-05-22-dla-clustering-prior-design.md sec 4-5.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

_C_KMS = 299792.458
_OM0, _OB0, _H0, _NS, _SIGMA8, _TCMB = 0.3156, 0.0491, 67.31, 0.9645, 0.831, 2.7255


class DLAClusteringPrior:
    """Analytic DLA clustering prior. b_DLA=2 is the LyaCoLoRe planted value."""

    def __init__(self, b_dla: float = 2.0, r_cut_mpch: float = 0.5,
                 eps: float = 1e-3, Om0=_OM0, Ob0=_OB0, H0=_H0,
                 ns=_NS, sigma8=_SIGMA8):
        self.b_dla = float(b_dla)
        self.r_cut = float(r_cut_mpch)
        self.eps = float(eps)
        self.ns = float(ns)
        self._sigma8 = float(sigma8)
        self.h = H0 / 100.0
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, Tcmb0=_TCMB)
        self._Om0, self._Ob0 = Om0, Ob0
        self._norm = sigma8 ** 2 / self._sigma2(8.0, 1.0)
        rg = np.logspace(-1.0, 2.6, 300)
        self._xi_interp = interp1d(np.log(rg),
                                   np.array([self._xi_matter_one(r) for r in rg]),
                                   kind="cubic", bounds_error=False,
                                   fill_value=(self._xi_matter_one(rg[0]), 0.0))
        self._r_grid = rg
        zg = np.linspace(0.0, 6.0, 200)
        self._growth_interp = interp1d(zg, np.array([self._growth_one(z) for z in zg]),
                                       kind="cubic")

    def _T_nowiggle(self, k):
        om_m, om_b = self._Om0 * self.h**2, self._Ob0 * self.h**2
        theta = _TCMB / 2.7
        s = 44.5 * np.log(9.83 / om_m) / np.sqrt(1.0 + 10.0 * om_b**0.75)
        fb = om_b / om_m
        alpha = (1.0 - 0.328 * np.log(431.0 * om_m) * fb
                 + 0.38 * np.log(22.3 * om_m) * fb**2)
        ks = k * s * self.h
        gamma_eff = self._Om0 * self.h * (alpha + (1.0 - alpha) / (1.0 + (0.43 * ks) ** 4))
        q = k * (theta**2 / gamma_eff)
        L0 = np.log(2.0 * np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        return L0 / (L0 + C0 * q**2)

    def _Pk(self, k):
        return self._norm * k**self.ns * self._T_nowiggle(k) ** 2

    def _sigma2(self, R, norm):
        def integ(lnk):
            k = np.exp(lnk); x = k * R
            w = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
            return norm * (k**self.ns * self._T_nowiggle(k) ** 2) * k**3 * w**2 / (2 * np.pi**2)
        return quad(integ, np.log(1e-4), np.log(1e3), limit=200)[0]

    def _xi_matter_one(self, r):
        def integ(lnk):
            k = np.exp(lnk); x = k * r
            return self._Pk(k) * k**3 * (np.sin(x) / x) / (2 * np.pi**2) * np.exp(-((k / 50.0) ** 2))
        return quad(integ, np.log(1e-4), np.log(1e3), limit=300)[0]

    def _growth_one(self, z):
        def Ea(a): return np.sqrt(self._Om0 * a**-3 + (1.0 - self._Om0))
        def integ(a): return 1.0 / (a * Ea(a)) ** 3
        def Du(a): return Ea(a) * quad(integ, 1e-6, a, limit=200)[0]
        a = 1.0 / (1.0 + z)
        return Du(a) / Du(1.0)

    def sigma8_check(self):
        return np.sqrt(self._sigma2(8.0, self._norm))

    def xi_matter_z0(self, r_mpch):
        r = np.atleast_1d(r_mpch).astype(float)
        return self._xi_interp(np.log(np.clip(r, self._r_grid[0], self._r_grid[-1])))

    def growth_D(self, z):
        return self._growth_interp(np.clip(np.asarray(z, float), 0.0, 6.0))

    def xi_dla(self, dv_kms, z):
        dv_kms = np.atleast_1d(dv_kms).astype(float)
        z = np.atleast_1d(z).astype(float)
        Hz = self.cosmo.H(z).value
        r = dv_kms * (1.0 + z) / Hz * self.h
        r = np.maximum(r, self.r_cut)
        return self.b_dla**2 * self.growth_D(z) ** 2 * self.xi_matter_z0(r)

    def log_rho(self, all_z_dlas: np.ndarray) -> np.ndarray:
        all_z_dlas = np.atleast_2d(all_z_dlas)
        k, N = all_z_dlas.shape
        if k < 2:
            return np.zeros(N)
        sum_xi = np.zeros(N)
        for a in range(k):
            za = all_z_dlas[a]
            for b in range(a + 1, k):
                zb = all_z_dlas[b]
                zbar = 0.5 * (za + zb)
                dv = _C_KMS * np.abs(za - zb) / (1.0 + zbar)
                sum_xi += self.xi_dla(dv, zbar)
        return np.log(np.maximum(1.0 + sum_xi, self.eps))
