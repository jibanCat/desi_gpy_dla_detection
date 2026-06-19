"""test_tilted_manifest.py — WALL-1 tilted-f(N) manifest sampler (pure logic, no data).

Checks (design §5.1 TDD note):
  * the recovered logN histogram's slope matches the injected (2LPT × tilt) slope,
  * one injection per target (no reuse),
  * every emitted z sits inside the host sightline's GP search window,
  * MANIFEST_FIELDS schema is complete and write_campaign-consumable.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "injection")):
    if p not in sys.path:
        sys.path.insert(0, p)

from injection.campaign_grid import (  # noqa: E402
    build_tilted_manifest, MANIFEST_FIELDS, tilt_weight, LOGN_KNEE,
    _per_sightline_forest_window,
)


def _toy_clean(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    zq = rng.uniform(2.5, 3.6, n)             # hosts with non-empty GP windows
    return dict(
        target_id=np.arange(10**18, 10**18 + n, dtype=np.int64),
        healpix=rng.integers(0, 100, n).astype(np.int64),
        z_qso=zq,
        native_snr=rng.uniform(2.5, 12.0, n),
    )


def _flat_pdf(logN):
    """Flat 2LPT shape so the recovered slope is PURELY the tilt 10^(Δα(N−pivot))."""
    return np.ones_like(np.asarray(logN, dtype=float))


def test_one_injection_per_target():
    clean = _toy_clean(n=3000)
    rows = build_tilted_manifest(
        clean, dalpha=0.5, n_inj=2000, logn_pdf_2lpt=_flat_pdf, seed=7)
    tids = [r["target_id"] for r in rows]
    assert len(tids) == len(set(tids)), "a target was injected more than once"
    assert len(rows) == 2000


def test_manifest_schema_complete():
    clean = _toy_clean(n=500)
    rows = build_tilted_manifest(
        clean, dalpha=-0.5, n_inj=300, logn_pdf_2lpt=_flat_pdf, seed=3)
    assert rows, "no rows emitted"
    for r in rows:
        for k in MANIFEST_FIELDS:
            assert k in r, f"manifest row missing required field {k!r}"
        assert r["control"] is False
        assert 19.5 - 1e-9 <= r["logN_true"] <= 22.5 + 1e-9


def test_z_inside_gp_window():
    clean = _toy_clean(n=1000)
    rows = build_tilted_manifest(
        clean, dalpha=0.5, n_inj=800, logn_pdf_2lpt=_flat_pdf, seed=11)
    for r in rows:
        z_lo, z_hi = _per_sightline_forest_window(r["z_qso"])
        assert z_lo <= r["z_true"] <= z_hi, (
            f"z_true {r['z_true']} outside GP window [{z_lo},{z_hi}] for "
            f"z_qso {r['z_qso']}")


def test_recovered_slope_matches_tilt():
    """With a FLAT 2LPT shape, the recovered logN density slope ≈ Δα (in dex^-1·ln10).

    f_tilt(N) ∝ 10^(Δα·(N−pivot)) ⇒ log10 density is linear in N with slope Δα.
    Fit log10(hist) vs N over [20.0, 22.0] (well inside support, good statistics).
    """
    clean = _toy_clean(n=40000, seed=2)
    for dalpha in (0.5, -0.5):
        rows = build_tilted_manifest(
            clean, dalpha=dalpha, n_inj=40000, logn_pdf_2lpt=_flat_pdf, seed=99)
        logn = np.array([r["logN_true"] for r in rows])
        edges = np.arange(20.0, 22.0 + 1e-9, 0.2)
        counts, _ = np.histogram(logn, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ok = counts > 0
        slope = np.polyfit(centers[ok], np.log10(counts[ok]), 1)[0]
        assert abs(slope - dalpha) < 0.1, (
            f"recovered slope {slope:.3f} != injected Δα {dalpha} (flat 2LPT shape)")


def test_tilt_weight_pivot_is_unity():
    assert abs(tilt_weight(LOGN_KNEE, 0.5) - 1.0) < 1e-12
    assert abs(tilt_weight(LOGN_KNEE, -0.5) - 1.0) < 1e-12
