"""
gpy_dla_detection.postprocess.lyb_veto
=======================================
Flag DLAs whose redshift coincides with the Lyβ-shifted redshift of
another (typically stronger, higher-z) DLA on the same line of sight.

Convention
----------
A real DLA at z_real produces Lyα absorption at λ_obs = (1 + z_real) ·
λ_Lyα. Its Lyβ line, at the SAME observed wavelength, is interpreted by
a Lyα-only matching scheme as a fake Lyα line at apparent redshift

    z_lybeta_apparent = (λ_Lyβ / λ_Lyα) · (1 + z_real) − 1
                       ≈ 0.84366 · (1 + z_real) − 1

If a multi-DLA finder reports a DLA at z_app and a stronger DLA at
z_real exists on the LOS such that z_app ≈ z_lybeta_apparent(z_real),
the lower-z DLA is most likely a Lyβ misidentification.

Why this happens even though the production Voigt model includes the
Lyβ + Lyγ Lyman lines:
- The forward model for an N-DLA hypothesis sums optical depth contributions
  from each absorber (each with its own Lyα + Lyβ + Lyγ).
- For an N=2 hypothesis with one absorber at z_real and one at
  z_lybeta_apparent, the "extra" absorber's Lyα coincides — at the same
  observed wavelength — with the first absorber's Lyβ. The summed model
  *over-predicts* the absorption at that wavelength.
- The fitter compensates by lowering the second absorber's NHI. The
  best-fit "second DLA" therefore lands at low NHI, near the prior edge
  (~20.3).
- Whether the M_DLA(2) Bayes factor still beats M_DLA(1) depends on the
  evidence integral over the QMC samples, not just the MAP. If FILTER=1
  truncates low-likelihood samples (see voigt_v2 / FILTER=1 algorithm
  notes), it tends to suppress this confused mode and reduce the spurious
  rate, but it does not eliminate it.
- A clean catalog-time veto is therefore worth running regardless of
  FILTER setting.

API
---

    flag_lybeta(dla_table, dz_match=0.005) -> astropy.table.Table

Adds three columns to a DLA catalog:
    LYBETA_FLAG        bool,  True if this DLA is a likely Lyβ misidentification
    LYBETA_PARENT_TID  int64, TARGETID this row was matched to (same LOS)
    LYBETA_PARENT_Z    float, the parent DLA's z (the high-z one)

The function is conservative: it only flags a DLA at z_low when there
exists another DLA on the same LOS with z_high > z_low and
|z_low − z_lybeta_apparent(z_high)| ≤ dz_match. Both DLAs must be in
the same input table.
"""

from __future__ import annotations

import numpy as np


LYA_REST = 1215.6701   # Å
LYB_REST = 1025.7228   # Å
LYG_REST = 972.5368    # Å


def lybeta_apparent_z(z_real: np.ndarray | float) -> np.ndarray | float:
    """Apparent z of a DLA's Lyβ line if it were mistaken for Lyα."""
    return (LYB_REST / LYA_REST) * (1.0 + np.asarray(z_real)) - 1.0


def lygamma_apparent_z(z_real: np.ndarray | float) -> np.ndarray | float:
    """Apparent z of a DLA's Lyγ line if it were mistaken for Lyα.
    Provided for symmetry; in practice Lyγ is rarely the dominant
    confusion source because Lyγ falls below the GP search range for
    most z_qso values."""
    return (LYG_REST / LYA_REST) * (1.0 + np.asarray(z_real)) - 1.0


def flag_lybeta(
    dla_table,
    *,
    targetid_col: str = "TARGETID",
    z_col: str = "Z_DLA",
    nhi_col: str = "LOG_NHI",
    dz_match: float = 0.005,
    require_higher_nhi_parent: bool = True,
):
    """Add LYBETA_FLAG / LYBETA_PARENT_TID / LYBETA_PARENT_Z columns
    to ``dla_table`` and return it.

    Parameters
    ----------
    dla_table : astropy.table.Table
        Per-DLA catalog with at least the columns named by ``targetid_col``,
        ``z_col``, ``nhi_col``. Multiple rows per TARGETID are expected
        for spectra with multiple MAP DLAs.
    dz_match : float
        Tolerance in z for matching ``z_lybeta_apparent(z_high)`` to
        ``z_low`` (default 0.005, ≈ 7 px at DESI dlambda=0.15).
    require_higher_nhi_parent : bool
        If True (default), only flag z_low if the parent (z_high) has
        a strictly higher logNHI. The "Lyβ from a real DLA" failure
        mode produces a *lower-NHI* fake DLA.

    Returns
    -------
    The same ``dla_table`` with three new columns. Original columns are
    not modified. If the columns already exist, they're overwritten.
    """
    from astropy.table import Table  # local import keeps astropy a soft dep

    if not isinstance(dla_table, Table):
        dla_table = Table(dla_table)

    n = len(dla_table)
    flag = np.zeros(n, dtype=bool)
    parent_tid = np.full(n, -1, dtype=np.int64)
    parent_z = np.full(n, np.nan, dtype=np.float64)

    tids = np.asarray(dla_table[targetid_col])
    zs = np.asarray(dla_table[z_col], dtype=np.float64)
    nhis = np.asarray(dla_table[nhi_col], dtype=np.float64)

    # Group rows by TARGETID
    unique_tids, inverse = np.unique(tids, return_inverse=True)
    for k, tid in enumerate(unique_tids):
        mask = inverse == k
        idxs = np.where(mask)[0]
        if idxs.size < 2:
            continue
        # Sort within LOS by descending z (parent candidates first)
        order = idxs[np.argsort(-zs[idxs])]
        for i_high in order:
            for i_low in order:
                if i_low == i_high:
                    continue
                if zs[i_low] >= zs[i_high]:
                    continue
                if require_higher_nhi_parent and nhis[i_high] <= nhis[i_low]:
                    continue
                z_lyb = lybeta_apparent_z(zs[i_high])
                if abs(zs[i_low] - z_lyb) <= dz_match:
                    flag[i_low] = True
                    parent_tid[i_low] = tids[i_high]
                    parent_z[i_low] = zs[i_high]

    dla_table["LYBETA_FLAG"] = flag
    dla_table["LYBETA_PARENT_TID"] = parent_tid
    dla_table["LYBETA_PARENT_Z"] = parent_z
    return dla_table
