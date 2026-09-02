"""tools/r041_mock_campaign.system_arms_geometry — the multi-HCD clustering-control geometry (gate §2): syscluster keeps truth
positions; sysrandom applies ONE rigid shift per system inside the window preserving every internal separation; sysshuffle keeps the
first absorber, redraws companion separations from the pooled truth distribution with |dv| >= 200 km/s inside the window; all arms
preserve m_true and the N multiset; the construction is deterministic per TARGETID."""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, REPO)
MC = pytest.importorskip("r041_mock_campaign")
C = 299792.458


def systems():
    return [dict(TARGETID=1, zlo_bin=3.0, zhi_bin=3.6, absorbers=[(3.20, 20.4), (3.25, 20.8)], salt="t"),
            dict(TARGETID=2, zlo_bin=3.0, zhi_bin=3.6, absorbers=[(3.10, 21.0), (3.11, 20.3), (3.40, 20.5)], salt="t"),
            dict(TARGETID=3, zlo_bin=3.0, zhi_bin=3.05, absorbers=[(3.00, 20.4), (3.05, 20.4)], salt="t")]     # span == window: no valid shift range


def test_geometry_rules():
    S = systems(); rng = np.random.default_rng(0)
    clu = MC.system_arms_geometry(S, "syscluster", rng); ran = MC.system_arms_geometry(S, "sysrandom", rng); shf = MC.system_arms_geometry(S, "sysshuffle", rng)
    for g, s in zip(clu, S):
        assert g["z"] == [a[0] for a in s["absorbers"]] and g["logN"] == [a[1] for a in s["absorbers"]] and g["shift_ok"]
    for g, s in zip(ran[:2], S[:2]):
        zt = np.array([a[0] for a in s["absorbers"]]); zr = np.array(g["z"])
        assert g["shift_ok"] and abs(g["shift"]) > 0 and np.allclose(np.diff(zr), np.diff(zt))          # rigid shift, separations preserved
        assert zr.min() >= s["zlo_bin"] - 1e-12 and zr.max() <= s["zhi_bin"] + 1e-12 and g["logN"] == [a[1] for a in s["absorbers"]]
    assert ran[2]["shift_ok"] is False and ran[2]["z"] == [3.00, 3.05]                                   # no valid range -> truth kept, flagged
    for g, s in zip(shf, S):
        z = np.array(g["z"]); assert z[0] == s["absorbers"][0][0] and len(z) == len(s["absorbers"]) and g["logN"] == [a[1] for a in s["absorbers"]]
        assert z.min() >= s["zlo_bin"] - 1e-12 and z.max() <= s["zhi_bin"] + 1e-12
        seps = [C * abs(z[i] - z[j]) / (1 + min(z[i], z[j])) for i in range(len(z)) for j in range(i)]
        assert all(v >= 200.0 - 1e-6 for v in seps) or not g["shift_ok"]
    # determinism per TARGETID (independent of the outer rng)
    ran2 = MC.system_arms_geometry(S, "sysrandom", np.random.default_rng(99))
    assert ran2[0]["z"] == ran[0]["z"] and ran2[1]["shift"] == ran[1]["shift"]
    # the shuffle breaks the truth separation of system 2 (3.10/3.11 at ~730 km/s would only survive by chance)
    assert MC.system_arms_geometry(S, "sysshuffle", rng)[1]["z"] != [a[0] for a in S[1]["absorbers"]]
