"""Synthetic tests for `gpy_dla_detection.postprocess.lyb_veto`."""

from __future__ import annotations

import numpy as np
import pytest

from astropy.table import Table

from gpy_dla_detection.postprocess.lyb_veto import (
    flag_lybeta,
    lybeta_apparent_z,
    LYA_REST,
    LYB_REST,
)


def _make_table(rows):
    return Table(rows=rows, names=["TARGETID", "Z_DLA", "LOG_NHI", "MODEL_P"])


def test_lybeta_apparent_z_matches_wavelength_definition():
    """For a DLA at z_real, Lyα and Lyβ are at the same observed
    wavelength; verify the apparent z formula."""
    z_real = 2.7
    lam_obs_lyb = LYB_REST * (1 + z_real)
    z_app = lam_obs_lyb / LYA_REST - 1
    assert np.isclose(z_app, lybeta_apparent_z(z_real))


def test_flags_only_child_in_a_lybeta_pair():
    z_parent = 2.700
    z_child  = lybeta_apparent_z(z_parent)
    rows = [
        (10001, z_parent, 21.30, 0.99),  # parent (high z, high NHI)
        (10001, z_child,  20.32, 0.40),  # spurious child near 20.3
        (10002, 2.500,    21.00, 0.95),  # unrelated DLA on a different LOS
    ]
    cat = _make_table(rows)
    out = flag_lybeta(cat, dz_match=0.005)
    flags = list(out["LYBETA_FLAG"])
    assert flags == [False, True, False]
    # parent_tid only set for the flagged row
    assert out["LYBETA_PARENT_TID"][1] == 10001
    assert np.isclose(out["LYBETA_PARENT_Z"][1], z_parent, atol=1e-6)


def test_does_not_flag_when_parent_nhi_lower():
    """The Lyβ-of-real-DLA confusion gives the spurious DLA a LOWER
    NHI than the parent. If the parent has lower NHI, don't flag —
    the configuration isn't the known failure mode."""
    z_parent = 2.7
    z_child  = lybeta_apparent_z(z_parent)
    rows = [
        (10003, z_parent, 20.40, 0.5),
        (10003, z_child,  21.50, 0.99),  # NHI HIGHER than parent — should NOT be flagged
    ]
    cat = _make_table(rows)
    out = flag_lybeta(cat)
    assert list(out["LYBETA_FLAG"]) == [False, False]


def test_does_not_flag_when_dz_outside_tolerance():
    z_parent = 2.7
    z_child  = lybeta_apparent_z(z_parent) + 0.02   # 4× the default tol
    rows = [
        (10004, z_parent, 21.40, 0.99),
        (10004, z_child,  20.30, 0.30),
    ]
    cat = _make_table(rows)
    out = flag_lybeta(cat, dz_match=0.005)
    assert list(out["LYBETA_FLAG"]) == [False, False]


def test_keeps_real_two_dla_los():
    """Two genuinely separate DLAs at uncorrelated z's must not be flagged."""
    rows = [
        (10005, 2.30, 21.0, 0.99),
        (10005, 2.85, 20.5, 0.95),    # not at 0.844*(1+2.30)-1 = 1.79
    ]
    cat = _make_table(rows)
    out = flag_lybeta(cat)
    assert list(out["LYBETA_FLAG"]) == [False, False]
