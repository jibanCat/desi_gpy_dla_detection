"""VALIDATION-ONLY: the per-draw scalar vectors the campaign's criteria are evaluated on.

Two families, exactly as the campaign PREDECLARATION sec.3 enumerates them:

  * the 30 REPORTED SCALARS, built with the PAPER'S OWN reduction weights
    (paper_figures/hbi_reduction.py, read-only import) so that every definition -- the
    open-topped dN/dX threshold weight, the 21.6-closed Omega weight, the path-weighted
    overlap z weight, the locked reporting bins -- is byte-for-byte the reported one.  The
    quantity set is identical to the candidate's reductions/tools/reduce.py (R-042a);
  * every SAMPLED NUISANCE SITE, component by component (574 on the C1 pack).

Nothing here reduces anything for publication; it only supplies per-draw vectors to the
convergence estimators.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/mfho/Latex/gp_dla_desi_y3/paper_figures")
import hbi_reduction as HR          # noqa: E402  (read-only import)

SAMPLED_SITES = ("sigma_N", "sigma_z", "theta_level", "theta_slope", "eps_N",
                 "eps_z", "psi_c", "fp_lam_total", "fp_shape_v", "t")


class Q:
    """Per-draw scalars for the 30 reported quantities of reduce.py."""

    def __init__(self, fdraws_path, pack_path):
        d = np.load(fdraws_path)
        pk = np.load(pack_path, allow_pickle=True)
        self.f = np.asarray(d["f"], float)                      # (D, B, Kf)
        self.n_edges = np.asarray(d["ntrue_edges"], float)
        self.z_edges = np.asarray(d["zf_edges"], float)
        self.dX = np.asarray(pk["dX"], float).sum(axis=1)
        if not np.allclose(self.z_edges, np.asarray(pk["zf_edges"], float)):
            raise SystemExit("BLOCKED: draws/pack redshift grids disagree")
        self._P = HR.Posterior.__new__(HR.Posterior)   # borrow the weight functions only
        self._P.f = self.f
        self._P.n_edges = self.n_edges
        self._P.z_edges = self.z_edges
        self._P.dX = self.dX

    def _reduce(self, nhi_w, z_lo, z_hi):
        zw = self._P._z_weight(z_lo, z_hi)
        if zw.sum() <= 0.0:
            return None
        return np.einsum("dbk,b,k->d", self.f, nhi_w, zw) / zw.sum()

    def vectors(self):
        """dict name -> (D,) per-draw scalars, keys identical to reduce.py's."""
        out = {}
        zlo, zhi = HR.LOWZ_SUPPORT
        for key, thr in HR.THRESHOLDS.items():
            w = self._P._nhi_weight(thr)
            out[f"dndx_{key}_allz"] = self._reduce(w, zlo, zhi)
            for name, lo, hi in HR.LOWZ_BINS:
                if HR.coverage(lo, hi) <= 0.0:
                    continue
                v = self._reduce(w, lo, hi)
                if v is None:
                    continue
                out[f"dndx_{key}_{name}"] = v
                if key == "20p3" and name != "B5":
                    ow = self._P._omega_weight(thr, HR.REPORT_NHI[1])
                    out[f"omega_20p3_21p6_{name}"] = (HR.OMEGA_PREFACTOR_CM2
                                                      * self._reduce(ow, lo, hi))
        ow = self._P._omega_weight(*HR.OMEGA_NHI)
        out["omega_20p3_21p6_allz"] = HR.OMEGA_PREFACTOR_CM2 * self._reduce(ow, zlo, zhi)
        zw = self._P._z_weight(zlo, zhi)
        lo_e, hi_e = self.n_edges[:-1], self.n_edges[1:]
        for i, (a, b) in enumerate(zip(lo_e, hi_e)):
            if b <= HR.FN_REPORT_NHI[0] or a >= HR.FN_REPORT_NHI[1]:
                continue
            out[f"cddf_{a:.1f}_{b:.1f}"] = (np.einsum("dk,k->d", self.f[:, i, :], zw)
                                            / zw.sum())
        return out


def nuisance_components(bychain_path):
    """dict name -> (chains, draws) matrices, one per SAMPLED scalar component."""
    z = np.load(bychain_path)
    out = {}
    for s in SAMPLED_SITES:
        a = np.asarray(z[s], float)
        m, n = a.shape[0], a.shape[1]
        flat = a.reshape(m, n, -1)
        if flat.shape[2] == 1:
            out[s] = flat[:, :, 0]
            continue
        ev = np.asarray(z[s]).shape[2:]
        for j in range(flat.shape[2]):
            idx = np.unravel_index(j, ev)
            out[s + "[" + ",".join(str(i) for i in idx) + "]"] = flat[:, :, j]
    return out


def by_chain(v, n_chains):
    """(D,) -> (m, n).  numpyro get_samples(group_by_chain=False) is CHAIN-MAJOR."""
    v = np.asarray(v)
    D = v.shape[0]
    assert D % n_chains == 0, (D, n_chains)
    return v.reshape(n_chains, D // n_chains)
