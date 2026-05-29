"""Regression tests for the legacy-bit-numbering refusal in
`tools/postprocess/add_dla_flags.py`.

The refusal protects pre-2026-05-15 dlacats (where DLAFLAG bits 3/4/5 meant
POTENTIAL_BAL / BAD_ZFIT / BAD_NHIFIT, now bits 0/1/2) from being silently
corrupted by `flag &= ~_ALL_POSTPROCESS_BITS_TO_CLEAR`.

The first-cut implementation (commit b8f00d6) probed `tbl.colnames` from
inside `_update_dlaflag_bitmask`. By that point `process_one` has already
called `_add_lybeta` / `_add_bal_flag` / `_add_nhi_consistency` /
`_add_pdla_saturated`, all of which unconditionally write their boolean
columns onto `tbl`. So the column-presence check ALWAYS evaluated False in
production, and the refusal was dead code. The fix snapshots the input
column names BEFORE the `_add_*` helpers run, and passes that snapshot into
`_update_dlaflag_bitmask`.

These tests pin both code paths:
  (A) the direct-call path the prior unit test exercised, and
  (B) the production `process_one` path that the prior test missed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.postprocess.add_dla_flags import (  # noqa: E402
    _update_dlaflag_bitmask,
    process_one,
)


# Legacy schema: bits 3/4/5 = POTENTIAL_BAL / BAD_ZFIT / BAD_NHIFIT.
LEGACY_POTENTIAL_BAL = np.int64(1 << 3)
LEGACY_BAD_ZFIT      = np.int64(1 << 4)
LEGACY_BAD_NHIFIT    = np.int64(1 << 5)


# ---------------------------------------------------------------------------
# Direct-call path (A)
# ---------------------------------------------------------------------------

def test_direct_legacy_catalog_refused():
    """Bare table with bits 3/4/5 set and no postprocess cols → refuse."""
    tbl = Table({"DLAFLAG": np.array(
        [0, LEGACY_POTENTIAL_BAL, LEGACY_BAD_ZFIT, LEGACY_BAD_NHIFIT, 0],
        dtype=np.int64)})
    with pytest.raises(RuntimeError, match="pre-2026-05-15 legacy"):
        _update_dlaflag_bitmask(tbl)


def test_direct_current_schema_passes():
    """Current-schema fresh-from-inference catalog (bits 0/1/2 only) passes."""
    tbl = Table({"DLAFLAG": np.array([0, 1, 2, 4, 0], dtype=np.int64)})
    n_changed = _update_dlaflag_bitmask(tbl)
    # Nothing to fold in, bits 3-5 already clear → no changes.
    assert n_changed == 0
    assert list(tbl["DLAFLAG"]) == [0, 1, 2, 4, 0]


def test_direct_post_reshuffle_with_lybeta_col_passes():
    """If LYBETA_FLAG is present in the (bare) table, refusal does not fire."""
    tbl = Table({
        "DLAFLAG":     np.array([0, LEGACY_POTENTIAL_BAL, 0, 0, 0], dtype=np.int64),
        "LYBETA_FLAG": np.array([False, True, False, False, False]),
    })
    n_changed = _update_dlaflag_bitmask(tbl)
    # Bit 3 originally set by legacy, then cleared, then set again because
    # LYBETA_FLAG is True on the same row → net change for row 1 is zero
    # bits. Row 1 final value = bit 3 (LYBETA_MISID).
    assert int(tbl["DLAFLAG"][1]) & (1 << 3) != 0


# ---------------------------------------------------------------------------
# Production `process_one` path (B) — the path the first-cut implementation
# bypassed, leaving the refusal as dead code.
# ---------------------------------------------------------------------------

class _Args:
    """Minimal stand-in for the argparse Namespace `process_one` consumes."""
    no_lyb_veto = False
    no_bal_flag = True            # skip — no bal_tids fixture
    no_nhi_consistency = False
    no_pdla_saturated = False
    no_bf_band = True             # skip — no bf_spec fixture
    lyb_veto_dz = 0.001
    nhi_consistency_k = 1.0
    nhi_consistency_floor = 19.5
    pdla_saturation = 1e-4


def _write_legacy_fits(path: Path, n: int = 5) -> None:
    """Write a tiny legacy-shaped dlacat: DLAFLAG bits 3/4/5 set, no
    postprocess boolean columns. Includes the columns the kept-on _add_*
    helpers need to run (NHI, NHI_ERR, P_DLA, Z_DLA, Z_QSO, TARGETID)."""
    tbl = Table({
        "TARGETID": np.arange(n, dtype=np.int64),
        "Z_QSO":    np.full(n, 3.0, dtype=np.float64),
        "Z_DLA":    np.full(n, 2.5, dtype=np.float64),
        "NHI":      np.full(n, 20.5, dtype=np.float64),
        "NHI_ERR":  np.full(n, 0.1, dtype=np.float64),
        "P_DLA":    np.full(n, 0.95, dtype=np.float64),
        "DLAFLAG":  np.array(
            [0, LEGACY_POTENTIAL_BAL, LEGACY_BAD_ZFIT, LEGACY_BAD_NHIFIT, 0],
            dtype=np.int64,
        ),
    })
    tbl.write(str(path), format="fits", overwrite=True)


def test_process_one_refuses_legacy_catalog(tmp_path):
    """REGRESSION: feeding a legacy dlacat to the production `process_one`
    path must refuse, NOT silently clear bits 3/4/5.

    Before the input-colnames-snapshot fix this test failed: `_add_lybeta`
    etc. wrote LYBETA_FLAG/NHI_CONSISTENCY_FLAG/PDLA_SATURATED_FLAG onto
    `tbl` before `_update_dlaflag_bitmask` ran, so the column-presence
    probe always saw the cols and the refusal was dead code. After the fix
    the snapshot is taken BEFORE the helpers run, so the refusal fires.
    """
    fits_path = tmp_path / "dlacat-legacy.fits"
    _write_legacy_fits(fits_path)
    args = _Args()
    with pytest.raises(RuntimeError, match="pre-2026-05-15 legacy"):
        process_one(fits_path, args, bal_tids=None,
                    bf_spec=None, bf_lognhi=None)
