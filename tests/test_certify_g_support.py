"""certify_g_support — the certification checks for the g(N,z) support
correction (PI ruling 2026-08-21: CP-1 acceptance (a)-(d)). Pure array
checks; MOCK/synthetic only. Loads the module file-directly (jax-free)."""
import importlib.util as _ilu
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_spec = _ilu.spec_from_file_location(
    "certify_g_support_mod",
    os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "certify_g_support.py"))
CG = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(CG)


def _arrays(inflate=1.0, rows_floor=19.0):
    # molly rows: 17.2 17.5 18.0 18.5 19.0 19.5 20.0 ... ; truth rows from 19.0
    molly_edges = np.array([17.2, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, np.inf])
    ntrue_edges = np.array([19.0, 19.2, 19.5, 19.7, 19.9, 20.1, 20.3, 20.5, 20.7])
    K = 15
    rng = np.random.default_rng(0)
    tc = rng.integers(10, 100, size=(len(ntrue_edges) - 1, K)).astype(float)
    # build a g_occupancy whose rows >= 19.0 reproduce tc per z exactly,
    # distributing tc's per-z totals over the >=19.0 molly rows
    g = np.zeros((len(molly_edges) - 1, K))
    g[:4] = rng.integers(100, 300, size=(4, K))          # sub-19.0 rows: free
    per_z = tc.sum(axis=0)
    g[4] = tc[:2].sum(axis=0)                             # 19.0-19.5
    g[5] = tc[2:5].sum(axis=0) * 0.0 + tc[2:4].sum(axis=0)
    g[6] = tc[4:6].sum(axis=0)
    g[7] = per_z - g[4] - g[5] - g[6]
    g[4:] *= inflate
    return g, molly_edges, tc, ntrue_edges


def test_g_support_identity_passes_on_a_consistent_support():
    g, me, tc, ne = _arrays()
    r = CG.g_support_identity(g, me, tc, ne, floor=19.0, tol_rows=2)
    assert r["passed"] is True
    assert r["max_abs_diff"] == 0
    assert r["total_g"] == r["total_truth"]


def test_g_support_identity_fails_on_an_inflated_denominator():
    """The deployed pack's signature: g rows >= 19.0 total ~1.9x the truth
    support, z-dependently. Must FAIL, and must report the per-z excess."""
    g, me, tc, ne = _arrays(inflate=1.9)
    r = CG.g_support_identity(g, me, tc, ne, floor=19.0, tol_rows=2)
    assert r["passed"] is False
    assert r["max_abs_diff"] > 2
    assert len(r["per_z_diff"]) == 15 and all(d > 0 for d in r["per_z_diff"])


def test_g_support_identity_tolerates_the_documented_one_row_bundle_difference():
    g, me, tc, ne = _arrays()
    g[5, 3] += 1.0                                        # one extra truth row
    r = CG.g_support_identity(g, me, tc, ne, floor=19.0, tol_rows=2)
    assert r["passed"] is True and r["max_abs_diff"] == 1


def test_array_identity_reports_only_the_changed_keys():
    a = dict(x=np.arange(4.0), y=np.ones(3), g_grid=np.ones(2))
    b = dict(x=np.arange(4.0), y=np.ones(3), g_grid=np.full(2, 1.5))
    assert CG.array_identity(a, b) == ["g_grid"]
    assert CG.array_identity(a, b, skip=("g_grid",)) == []
    c = dict(a); c.pop("y")
    assert "y" in CG.array_identity(a, c)                 # missing key = changed
