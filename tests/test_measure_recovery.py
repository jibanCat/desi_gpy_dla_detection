"""
tests/test_measure_recovery.py
==============================
TDD tests for the recovery-assembly seam of ``injection.measure_recovery``.

``_assemble_recovered`` matches GP-processed rows (TARGETID → p_dla / N_HI / z)
back to the injection manifest by ``inj_id``.  This is the NON-CIRCULAR scoring
join (recovered vs INJECTED truth).  The campaign's correctness depends on the
join being one-to-one: if the manifest reused a target_id the join would COLLAPSE
several inj_ids onto one and score recovery against the wrong injected truth.
These tests pin that the seam (a) assembles a distinct recovered record per
matched GP row, (b) tolerates partial GP coverage, and (c) REFUSES a collapsing
manifest / duplicated GP output instead of silently corrupting the measurement.
"""
import numpy as np
import pytest

from injection.measure_recovery import (
    _assemble_recovered,
    _pool_completeness_by_logN,
    _pool_bias_by_logN,
    _dlacat_rows_from_table,
)
from injection.measurements import detection_completeness, nhi_bias


def _manifest(n=4, start_tid=1000):
    # One injection per sightline (the post-blocker-fix invariant): unique tids.
    return [
        {"inj_id": i, "target_id": start_tid + i, "logN_true": 18.0 + 0.1 * i}
        for i in range(n)
    ]


def test_assemble_one_record_per_matched_gp_row():
    man = _manifest(4)
    rows = [(r["target_id"], 0.9, 20.0, 3.0) for r in man]  # all 4 processed
    rec = _assemble_recovered(man, rows)
    assert set(rec.keys()) == {0, 1, 2, 3}
    assert rec[2]["p_dla"] == 0.9 and rec[2]["logN_rec"] == 20.0


def test_assemble_tolerates_partial_coverage():
    # Only 2 of 4 sightlines were processed (e.g. a truncated / un-run healpix).
    man = _manifest(4)
    rows = [(man[0]["target_id"], 0.8, 19.5, 2.8),
            (man[3]["target_id"], 0.2, 17.6, 3.2)]
    rec = _assemble_recovered(man, rows)
    assert set(rec.keys()) == {0, 3}          # partial, no crash
    assert len(rec) == 2


def test_assemble_ignores_gp_rows_absent_from_manifest():
    man = _manifest(3)
    rows = [(man[1]["target_id"], 0.5, 20.1, 3.0),
            (999999, 0.99, 21.0, 3.1)]          # not in the manifest
    rec = _assemble_recovered(man, rows)
    assert set(rec.keys()) == {1}


def test_assemble_raises_on_manifest_targetid_collapse():
    # A manifest that reuses a target_id across two inj_ids would collapse the join.
    man = [
        {"inj_id": 0, "target_id": 1000, "logN_true": 18.0},
        {"inj_id": 1, "target_id": 1000, "logN_true": 20.5},  # SAME target_id
    ]
    rows = [(1000, 0.9, 19.0, 3.0)]
    with pytest.raises(ValueError, match="reuses target_id"):
        _assemble_recovered(man, rows)


def test_assemble_raises_on_duplicate_gp_output_row():
    # The same sightline appearing twice in the GP output would overwrite a record.
    man = _manifest(2)
    rows = [(man[0]["target_id"], 0.9, 20.0, 3.0),
            (man[0]["target_id"], 0.1, 17.5, 3.0)]  # duplicate target_id in GP output
    with pytest.raises(ValueError, match="more than once"):
        _assemble_recovered(man, rows)


# --------------------------------------------------------------------------- #
# pooling helpers — consume the REAL estimator schema (the blank-figure guard).
# Regression test for the schema mismatch where measure_recovery expected a
# {"cells":[...]} structure the estimators never produce → silently empty figures.
# --------------------------------------------------------------------------- #
def test_pool_completeness_consumes_real_estimator_schema():
    # A manifest with two logN values, each across two SNR bins; recovery scored by
    # detection_completeness (real parallel-array dict), then pooled by logN.
    manifest = []
    iid = 0
    # logN=18: SNR-bin 0 → 2/4 recovered; SNR-bin 1 → 3/4 recovered  → pooled 5/8
    # logN=20.3: all 4 recovered in each bin → pooled 8/8
    plan = [(18.0, 0, [1, 1, 0, 0]), (18.0, 1, [1, 1, 1, 0]),
            (20.3, 0, [1, 1, 1, 1]), (20.3, 1, [1, 1, 1, 1])]
    recovered = {}
    for logN, snr_bin, flags in plan:
        for det in flags:
            manifest.append({"inj_id": iid, "target_id": 1000 + iid, "control": False,
                             "logN_true": logN, "z_true": 3.0, "snr_bin": snr_bin})
            recovered[iid] = {"inj_id": iid, "p_dla": 0.9 if det else 0.1,
                              "logN_rec": logN + 0.2, "z_rec": 3.0}
            iid += 1
    cdet = detection_completeness(recovered, manifest, p_dla_thresh=0.5)
    xs, C, lo, hi, n = _pool_completeness_by_logN(cdet)
    assert list(xs) == [18.0, 20.3]
    assert n.tolist() == [8, 8]
    assert C[0] == pytest.approx(5 / 8)
    assert C[1] == pytest.approx(1.0)
    # CI is ordered and in [0,1] (at k=n the binomial point 1.0 may exceed the Beta
    # upper quantile — the documented one-sided-uncertainty behaviour — so we don't
    # require the interval to bracket the MLE point).
    assert np.all(lo <= hi) and np.all(lo >= 0) and np.all(hi <= 1)


def test_pool_bias_consumes_real_estimator_schema():
    # Two logN cells; recovered logN offset by a known bias; pooled over SNR.
    manifest, recovered, iid = [], {}, 0
    for logN, snr_bin, offset in [(17.4, 0, +2.0), (17.4, 1, +1.0), (20.5, 0, +0.05)]:
        for _ in range(5):
            manifest.append({"inj_id": iid, "target_id": 2000 + iid, "control": False,
                             "logN_true": logN, "z_true": 3.0, "snr_bin": snr_bin})
            recovered[iid] = {"inj_id": iid, "p_dla": 0.9,
                              "logN_rec": logN + offset, "z_rec": 3.0}
            iid += 1
    bias = nhi_bias(recovered, manifest, p_dla_thresh=0.5)
    xs, b, w = _pool_bias_by_logN(bias)
    assert list(xs) == [17.4, 20.5]
    # logN 17.4 pools (+2.0)*5 and (+1.0)*5 over 10 survivors → +1.5
    assert b[0] == pytest.approx(1.5)
    assert b[1] == pytest.approx(0.05)
    assert w.tolist() == [10, 5]


def test_dlacat_rows_extracts_recovery_fields():
    # The mock GP run writes a dlacat FITS (one row per detected absorber); the
    # reader must yield (TARGETID, P_DLA, NHI, Z_DLA) per the README §5 schema.
    Table = pytest.importorskip("astropy.table").Table
    tbl = Table({
        "TARGETID": np.array([100, 101, 102], np.int64),
        "P_DLA": [0.99, 0.60, 0.20],
        "NHI": [20.5, 19.1, 17.8],
        "Z_DLA": [3.0, 2.8, 2.5],
    })
    rows = list(_dlacat_rows_from_table(tbl))
    assert rows == [(100, 0.99, 20.5, 3.0), (101, 0.60, 19.1, 2.8), (102, 0.20, 17.8, 2.5)]


def test_dlacat_rows_dedup_keeps_highest_pdla():
    # A multi-row TARGETID (e.g. a multi-DLA dlacat) collapses to its BEST detection
    # so the single-absorber recovery join stays one-to-one (no spurious collapse).
    Table = pytest.importorskip("astropy.table").Table
    tbl = Table({
        "TARGETID": np.array([100, 100, 101], np.int64),
        "P_DLA": [0.40, 0.95, 0.7],   # 100 appears twice; keep the 0.95 row
        "NHI": [19.0, 20.6, 20.1],
        "Z_DLA": [2.9, 3.1, 2.7],
    })
    rows = sorted(_dlacat_rows_from_table(tbl))
    assert rows == [(100, 0.95, 20.6, 3.1), (101, 0.70, 20.1, 2.7)]


def test_pool_helpers_handle_empty_input():
    # No recovered absorbers → empty estimator arrays → pooling returns empty, no crash
    # (the figure section guards on .size before plotting).
    empty_manifest = [{"inj_id": 0, "target_id": 1, "control": False,
                       "logN_true": 18.0, "z_true": 3.0, "snr_bin": 0}]
    cdet = detection_completeness({}, empty_manifest, p_dla_thresh=0.5)
    xs, C, lo, hi, n = _pool_completeness_by_logN(cdet)
    assert xs.tolist() == [18.0] and n.tolist() == [1] and C[0] == pytest.approx(0.0)
    bias = nhi_bias({}, empty_manifest, p_dla_thresh=0.5)
    bx, bb, bw = _pool_bias_by_logN(bias)
    assert bx.size == 0  # no survivors → nothing to report
