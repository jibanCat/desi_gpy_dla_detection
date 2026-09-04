"""tools/r041_mock_campaign.native_selection — truth-level multiplicity selection for the P1 native multi-HCD arm
(PI ruling 2026-09-02 §5): m_true counted from absorbers >= 20.0 INSIDE the high-z-emulation bin window only; multi
(m >= 2) sightlines kept; single reference = exactly one absorber (>= 20.0) in window and it is >= 20.3, stratum-matched
by singles_per_multi; BAL / low-SNR / low-z_qso sightlines excluded; truth rows carry nearest-neighbour dv and m_true."""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, REPO)
MC = pytest.importorskip("r041_mock_campaign")


def test_native_selection_rules():
    zq = 3.3
    zlo, zhi, lo, hi = MC.window(zq, 1.0)
    assert hi > lo
    mid = 0.5 * (lo + hi)
    truth = {1: [(mid, 20.4), (mid + 0.02, 20.1)],            # multi: two absorbers >= 20.0 in window
             2: [(mid, 20.5)],                                 # single reference (>= 20.3)
             3: [(mid, 20.1)],                                 # single but < 20.3 -> not a reference
             4: [(mid, 20.4), (hi + 0.5, 21.0)],               # second absorber OUTSIDE the window -> counts as single
             5: [(mid, 20.4), (mid + 0.01, 20.4)],             # multi but BAL -> excluded
             6: [(mid, 20.4), (mid + 0.01, 20.4)],             # multi but SNR <= 2 -> excluded
             7: [(mid, 20.4), (mid + 0.03, 19.5)]}             # companion < 20.0 -> single
    tids = np.array(list(truth)); z = np.full(tids.size, zq)
    snr = {t: 5.0 for t in truth}; snr[6] = 1.5
    pop, tr = MC.native_selection(tids, z, snr, {5}, truth, dz=1.0, zqso_min=3.0, singles_per_multi=1.0, rng=np.random.default_rng(1))
    by = {r["TARGETID"]: r for r in pop}
    assert by[1]["m_true"] == 2 and 5 not in by and 6 not in by and 3 not in by
    singles = [t for t, r in by.items() if r["m_true"] == 1]
    assert len(singles) == 1 and singles[0] in (2, 4, 7)                    # one reference per multi sightline (stratum-matched)
    rows = [t for t in tr if t["TARGETID"] == 1]
    assert len(rows) == 2 and all(t["m_true"] == 2 for t in rows) and all(t["pair_class"] == "native" for t in rows)
    dv = float(rows[0]["dv_kms"]); assert abs(dv - 299792.458 * 0.02 / (1 + mid)) < 1.0
    # window bookkeeping for the scorer
    assert abs(by[1]["zlo"] - lo) < 1e-12 and abs(by[1]["zhi"] - hi) < 1e-12
