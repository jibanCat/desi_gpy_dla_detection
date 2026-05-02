"""Smoke + correctness tests for the Voigt-sweep target picker and analyzer.

Builds tiny synthetic mock catalogs in tmpdir, runs the picker against
them, and verifies the output schema + filtering logic.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

# Make repo imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.pick_voigt_sweep_targets import pick_for_mock, NHI_REGIMES, MOCK_PATHS


def _build_synthetic_mock(tmp_path: Path, *, n_qsos: int = 200) -> Path:
    """Build a minimal 2LPT-shaped mock dir with zcat + hcd_truth_cat."""
    rng = np.random.default_rng(0)

    mock_dir = tmp_path / "mock-0/loa-124"
    mock_dir.mkdir(parents=True)

    # zcat: TARGETID, Z, ZWARN, TARGET_RA, TARGET_DEC
    targets = rng.integers(low=10**12, high=10**13, size=n_qsos, dtype=np.int64)
    z_qsos = rng.uniform(2.5, 3.5, n_qsos).astype(np.float32)
    zcat = Table({
        "TARGETID": targets,
        "Z": z_qsos,
        "ZWARN": np.zeros(n_qsos, dtype=np.int32),
        "TARGET_RA": rng.uniform(0, 360, n_qsos),
        "TARGET_DEC": rng.uniform(-30, 60, n_qsos),
    })
    zcat.write(mock_dir / "zcat.fits", overwrite=True)

    # hcd_truth_cat: TARGETID, Z (= z_dla), NHI (logNHI), SNR
    # Place absorbers spread across all 3 NHI regimes.
    n_abs = 100
    abs_tids = rng.choice(targets, size=n_abs, replace=False)
    # Place in z_qso - 0.3 ± 0.05 so the mid-forest cut passes.
    z_dlas = []
    for tid in abs_tids:
        zq = float(zcat[zcat["TARGETID"] == tid]["Z"][0])
        z_dlas.append(zq - rng.uniform(0.05, 0.5))
    z_dlas = np.asarray(z_dlas, dtype=np.float32)
    nhis = np.concatenate([
        rng.uniform(17.5, 18.9, n_abs // 3),     # LLS
        rng.uniform(19.1, 20.2, n_abs // 3),     # sub-DLA
        rng.uniform(20.5, 22.0, n_abs - 2 * (n_abs // 3)),  # DLA
    ]).astype(np.float32)
    rng.shuffle(nhis)
    hcd = Table({
        "TARGETID": abs_tids,
        "Z": z_dlas,
        "NHI": nhis,
        "SNR": rng.uniform(2.0, 10.0, n_abs).astype(np.float32),
    })
    hcd.write(mock_dir / "hcd_truth_cat.fits", overwrite=True)

    # Fake spectra-16 layout — we just need the directory structure to
    # exist; pick_for_mock checks file existence with .exists().
    for tid in abs_tids:
        # Compute healpix from RA/DEC
        ra = float(zcat[zcat["TARGETID"] == tid]["TARGET_RA"][0])
        dec = float(zcat[zcat["TARGETID"] == tid]["TARGET_DEC"][0])
        import healpy as hp
        theta = np.deg2rad(90.0 - dec)
        phi = np.deg2rad(ra)
        hpx = int(hp.ang2pix(16, theta, phi, nest=True))
        spec_subdir = mock_dir / "spectra-16" / str(hpx // 100) / str(hpx)
        spec_subdir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_subdir / f"spectra-16-{hpx}.fits"
        spec_path.touch()

    return mock_dir


def test_pick_for_mock_returns_all_three_regimes(tmp_path):
    mock_dir = _build_synthetic_mock(tmp_path)
    # Override the 2lpt mock_dir for this test.
    MOCK_PATHS["2lpt"]["mock_dir"] = str(mock_dir)
    rows = pick_for_mock("2lpt", mock_dir, n_per_bin=3, snr_min=1.0, seed=0)

    # Should have >= 3 rows per regime if synth has enough candidates.
    regimes = {r["nhi_regime"] for r in rows}
    assert "LLS" in regimes, f"missing LLS in {regimes}"
    assert "sub-DLA" in regimes, f"missing sub-DLA in {regimes}"
    assert "DLA" in regimes, f"missing DLA in {regimes}"

    # Schema
    expected_keys = {"mock", "target_id", "z_qso", "truth_z_dla",
                     "truth_log_nhi", "nhi_regime", "spec_path", "zcat_path"}
    assert set(rows[0].keys()) == expected_keys

    # Each row's truth_log_nhi must be in the regime band it claims.
    for r in rows:
        log_nhi = float(r["truth_log_nhi"])
        regime = r["nhi_regime"]
        if regime == "LLS":
            assert 17.2 <= log_nhi < 19.0, f"LLS row out of band: {log_nhi}"
        elif regime == "sub-DLA":
            assert 19.0 <= log_nhi < 20.3
        elif regime == "DLA":
            assert 20.3 <= log_nhi < 23.0

    # Mid-forest cut: z_qso - 0.5 ≤ z_dla ≤ z_qso - 0.05
    for r in rows:
        z_qso = float(r["z_qso"])
        z_dla = float(r["truth_z_dla"])
        assert z_qso - 0.5 <= z_dla <= z_qso - 0.05, (
            f"mid-forest cut violated: z_qso={z_qso}, z_dla={z_dla}"
        )


def test_pick_skips_missing_mock_dir():
    """Should silently skip a mock whose mock_dir doesn't exist."""
    rows = pick_for_mock("2lpt", Path("/no/such/dir"), n_per_bin=3,
                         snr_min=1.0, seed=0)
    assert rows == []
