"""
gpy_dla_detection.postprocess.lls_cross_reference
==================================================
Cross-reference a DLA-mode (multi-DLA, NHI prior 20.0–23.0) catalog
against an LLS-mode (single-absorber, PW14 prior NHI 17.2–22.0) catalog
on the same TARGETIDs.

Why
---
The multi-DLA model is known to inflate sub-threshold absorbers (LLS,
sub-DLAs, strong forest) to log NHI ≈ 20.3 (the conventional DLA
boundary, and, with the prior at 20.0, near the boundary of where the
posterior gets confidently above 20.3). Running the same spectra
through LLS-mode with a sample grid covering NHI ∈ [17.2, 22.0] gives
a posterior that *can* place mass below 20.3, which is the right
diagnostic for "is this really a DLA, or just a strong LLS?".

For each MAP DLA in the DLA-mode catalog this routine looks up the
LLS-mode posterior at the same (TARGETID, z) (within a tolerance) and
adds:

    LLS_LOG_NHI         float, MAP NHI from LLS-mode  (NaN if no match)
    LLS_P_ABSORBER      float, p(absorber) from LLS-mode
    LLS_DOWNGRADE_FLAG  bool,  True if LLS-mode prefers logNHI < 20.3
                                AND p(LLS absorber) > p(DLA-mode DLA)/2

The flag is intentionally conservative — it does NOT remove the DLA
from the catalog, it only marks it for review / downstream filtering.

API
---

    cross_reference_lls(dla_catalog, lls_catalog,
                        dz_match=0.01, lls_threshold=20.3)

Both catalogs must be ``astropy.table.Table`` (or convertible) with at
least TARGETID, Z_DLA, LOG_NHI, MODEL_P columns. Returns the augmented
DLA catalog.
"""

from __future__ import annotations

import numpy as np


def cross_reference_lls(
    dla_catalog,
    lls_catalog,
    *,
    dz_match: float = 0.01,
    lls_threshold: float = 20.3,
    targetid_col: str = "TARGETID",
    z_col: str = "Z_DLA",
    nhi_col: str = "LOG_NHI",
    p_col: str = "MODEL_P",
):
    """Annotate ``dla_catalog`` with LLS-mode cross-reference flags."""
    from astropy.table import Table

    if not isinstance(dla_catalog, Table):
        dla_catalog = Table(dla_catalog)
    if not isinstance(lls_catalog, Table):
        lls_catalog = Table(lls_catalog)

    n = len(dla_catalog)
    nhi_lls = np.full(n, np.nan, dtype=np.float64)
    p_lls = np.full(n, np.nan, dtype=np.float64)
    downgrade = np.zeros(n, dtype=bool)

    # Group LLS rows by TARGETID for fast lookup
    lls_by_tid: dict[int, list[tuple[float, float, float]]] = {}
    for r in lls_catalog:
        lls_by_tid.setdefault(int(r[targetid_col]), []).append(
            (float(r[z_col]), float(r[nhi_col]), float(r[p_col]))
        )

    for i, row in enumerate(dla_catalog):
        tid = int(row[targetid_col])
        z_dla = float(row[z_col])
        p_dla = float(row[p_col])
        candidates = lls_by_tid.get(tid, [])
        if not candidates:
            continue
        # nearest LLS detection on the same LOS within dz_match
        best = None
        best_dz = dz_match
        for z_lls, nhi_lls_val, p_lls_val in candidates:
            dz = abs(z_lls - z_dla)
            if dz <= best_dz:
                best = (z_lls, nhi_lls_val, p_lls_val)
                best_dz = dz
        if best is None:
            continue
        nhi_lls[i] = best[1]
        p_lls[i] = best[2]
        # Downgrade if LLS-mode prefers a sub-DLA / LLS classification AND
        # the LLS absorber is at least half as confident as the DLA detection.
        if best[1] < lls_threshold and best[2] > p_dla / 2:
            downgrade[i] = True

    dla_catalog["LLS_LOG_NHI"] = nhi_lls
    dla_catalog["LLS_P_ABSORBER"] = p_lls
    dla_catalog["LLS_DOWNGRADE_FLAG"] = downgrade
    return dla_catalog
