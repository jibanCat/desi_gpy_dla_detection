"""tools/r041_analyze.py under multi-row (MAX_DLAS=4) catalogues — MAX4 repair cycle (PI ruling
2026-08-28 item 3): nearest-|dz| matching unchanged, order-independent accepted-row summaries,
multiplicity columns, and pair-mode counting of two matched accepted rows. Synthetic inputs only;
the single-row case must reproduce the old (MAX_DLAS=1) columns exactly."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

fitsio = pytest.importorskip("fitsio")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import r041_analyze as RA  # noqa: E402

OLD_COLS = ["TARGETID", "wave", "inj_idx", "logN", "z_inj", "stratum", "snr", "has_cand_ge20", "pair_class", "dv_kms",
            "pair_logN", "method", "meanflux_model", "detected", "nhat", "p", "dz", "n_rows_sightline", "any_accepted",
            "accepted_nhat", "accepted_z", "zbin"]


def _row(tid, z, nhi, p=0.999, flag=0, snr=3.0, n=0):
    return (tid, 10.0, 20.0, 4.8, 2.0, snr, f"{tid}00{n}", z, 1e-4, nhi, 0.05, flag, p, 1 - p, -1.0, -5.0, p)


def _write_cat(path, rows):
    dt = [("TARGETID", "i8"), ("RA", "f8"), ("DEC", "f8"), ("Z_QSO", "f8"), ("SNR_FOREST", "f8"), ("SNR_REDSIDE", "f8"),
          ("DLAID", "S24"), ("Z_DLA", "f8"), ("Z_DLA_ERR", "f8"), ("NHI", "f8"), ("NHI_ERR", "f8"), ("DLAFLAG", "i8"),
          ("P_DLA", "f8"), ("P_NULL", "f8"), ("LOGP_DLA", "f8"), ("LOGP_NULL", "f8"), ("MODEL_P", "f8")]
    fitsio.write(path, np.array(rows, dtype=dt), clobber=True)


def _write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)


POP_COLS = ["TARGETID", "z_qso", "snr", "zlo", "zhi", "zlo_bin", "zhi_bin", "dX_bin", "stratum", "n_cand", "has_cand_ge20", "cand"]
TRUTH_COLS = ["TARGETID", "wave", "inj_idx", "logN", "z_inj", "stratum", "snr", "has_cand_ge20", "pair_class", "dv_kms",
              "pair_logN", "method", "meanflux_model"]


def _pop(tid, stratum=2):
    return dict(TARGETID=tid, z_qso=4.8, snr=3.0, zlo=3.9, zhi=4.7, zlo_bin=3.9, zhi_bin=4.7, dX_bin=3.0, stratum=stratum,
                n_cand=0, has_cand_ge20=0, cand="")


def _truth(tid, idx, logN, z, pair_class="", dv="", pair_logN=""):
    return dict(TARGETID=tid, wave=0, inj_idx=idx, logN=logN, z_inj=z, stratum=2, snr=3.0, has_cand_ge20=0,
                pair_class=pair_class, dv_kms=dv, pair_logN=pair_logN, method="variance_preserving", meanflux_model="fiducial")


def _run(tmp_path, cat_rows, truth_rows, pop_rows, label):
    odir = tmp_path / "outputs"; odir.mkdir()
    _write_cat(str(odir / "dlacat-loa-main-dark-hpx-0-1.fits"), cat_rows)
    truth = tmp_path / "truth.csv"; _write_csv(truth, TRUTH_COLS, truth_rows)
    pop = tmp_path / "population.csv"; _write_csv(pop, POP_COLS, pop_rows)
    out = tmp_path / f"analysis_{label}.json"
    RA.main(["--truth", str(truth), "--outputs", str(odir), "--population", str(pop), "--out", str(out), "--label", label])
    per = list(csv.DictReader(open(str(out)[:-5] + "_per_injection.csv")))
    return json.load(open(out)), per, out


def test_multirow_sightline_nearest_match_and_multiplicity(tmp_path):
    # 4 rows on one sightline: accepted-near (small N-hat), accepted-far (large N-hat), rejected by P, rejected by DLAFLAG
    rows = [_row(1, 4.60, 21.0, n=0), _row(1, 4.301, 20.2, n=1), _row(1, 4.45, 20.9, p=0.5, n=2), _row(1, 4.50, 21.5, flag=8, n=3)]
    j, per, _ = _run(tmp_path, rows, [_truth(1, 0, 20.4, 4.30)], [_pop(1)], "multi")
    r = per[0]
    assert r["detected"] == "True" and float(r["nhat"]) == 20.2            # nearest-|dz| accepted row wins the match
    assert abs(float(r["dz"]) - 0.001 / 5.30) < 1e-12
    assert int(r["n_rows_sightline"]) == 4 and int(r["n_accepted"]) == 2
    assert float(r["accepted_nhat_max"]) == 21.0 and float(r["accepted_nhat"]) == 21.0 and float(r["accepted_z"]) == 4.60
    assert r["accepted_nhats"] == "21.0;20.2" and r["accepted_zs"] == "4.6;4.301"
    assert list(r.keys())[:len(OLD_COLS)] == OLD_COLS                       # old column order is a prefix
    assert j["tables"]["per_logN_point_all"][0]["k"] == 1


def test_multirow_summary_is_file_order_independent(tmp_path):
    rows_a = [_row(1, 4.60, 21.0, n=0), _row(1, 4.301, 20.2, n=1)]
    rows_b = [_row(1, 4.301, 20.2, n=0), _row(1, 4.60, 21.0, n=1)]
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    _, pa, _ = _run(tmp_path / "a", rows_a, [_truth(1, 0, 20.4, 4.30)], [_pop(1)], "a")
    _, pb, _ = _run(tmp_path / "b", rows_b, [_truth(1, 0, 20.4, 4.30)], [_pop(1)], "b")
    for k in ("detected", "nhat", "dz", "accepted_nhat", "accepted_z", "accepted_nhat_max", "n_accepted"):
        assert pa[0][k] == pb[0][k], k


def test_single_row_reproduces_the_old_columns_exactly(tmp_path):
    # a MAX_DLAS=1-like output: exactly one (accepted) row -> the legacy columns are the old cands[0] values
    rows = [_row(2, 4.201, 20.7, p=0.995, snr=2.5, n=0)]
    _, per, _ = _run(tmp_path, rows, [_truth(2, 0, 20.6, 4.20)], [_pop(2)], "single")
    r = per[0]
    old = dict(TARGETID="2", wave="0", inj_idx="0", logN="20.6", z_inj="4.2", stratum="2", snr="3.0", has_cand_ge20="0",
               pair_class="", dv_kms="", pair_logN="", method="variance_preserving", meanflux_model="fiducial",
               detected="True", nhat="20.7", p="0.995", dz=repr(abs(4.201 - 4.20) / (1.0 + 4.20)), n_rows_sightline="1",
               any_accepted="True", accepted_nhat="20.7", accepted_z="4.201", zbin="0")
    assert {k: r[k] for k in OLD_COLS} == old
    assert int(r["n_accepted"]) == 1 and float(r["accepted_nhat_max"]) == 20.7 and r["accepted_nhats"] == "20.7"


def test_no_row_and_rejected_only(tmp_path):
    rows = [_row(3, 4.30, 20.9, p=0.5, n=0)]                                   # rejected by P; sightline 4 has no rows at all
    _, per, _ = _run(tmp_path, rows, [_truth(3, 0, 20.4, 4.30), _truth(4, 0, 20.4, 4.30)], [_pop(3), _pop(4)], "none")
    for r in per:
        assert r["detected"] == "False" and int(r["n_accepted"]) == 0 and r["accepted_nhats"] == ""
        assert math.isnan(float(r["accepted_nhat"])) and math.isnan(float(r["accepted_nhat_max"]))
    assert int(per[0]["n_rows_sightline"]) == 1 and int(per[1]["n_rows_sightline"]) == 0


def test_pair_mode_counts_two_matched_rows(tmp_path):
    # pair on TID 5: two accepted rows, one near each absorber -> n_matched 2 (only possible with MAX_DLAS > 1)
    # pair on TID 6: one accepted row near absorber 1 only -> n_matched 1 (the MAX1-style outcome)
    rows = [_row(5, 4.301, 20.4, n=0), _row(5, 4.401, 20.9, n=1), _row(6, 4.402, 20.6, n=0)]
    truth = [_truth(5, 0, 20.5, 4.30, "wide", "5000", "20.8"), _truth(5, 1, 20.8, 4.40, "wide", "5000", "20.5"),
             _truth(6, 0, 20.5, 4.30, "wide", "5000", "20.8"), _truth(6, 1, 20.8, 4.40, "wide", "5000", "20.5")]
    j, per, out = _run(tmp_path, rows, truth, [_pop(5), _pop(6)], "pairs")
    s = j["tables"]["pairs_summary"]["wide"]
    assert s["n_pairs"] == 2 and s["frac_two"] == 0.5 and s["frac_ge2_accepted"] == 0.5
    assert s["frac_one"] == 0.5 and s["frac_zero"] == 0.0 and s["frac_one_matched"] == 0.5
    pairs = list(csv.DictReader(open(str(out)[:-5] + "_pairs.csv")))
    p5 = [p for p in pairs if p["TARGETID"] == "5"][0]; p6 = [p for p in pairs if p["TARGETID"] == "6"][0]
    assert int(p5["n_accepted"]) == 2 and int(p5["n_matched"]) == 2 and float(p5["nhat_abs0"]) == 20.4 and float(p5["nhat_abs1"]) == 20.9
    assert float(p5["accepted_nhat"]) == 20.9 and p5["winner"] == "1"       # legacy summary = the row of largest N-hat
    assert int(p6["n_accepted"]) == 1 and int(p6["n_matched"]) == 1 and p6["nhat_abs0"] == "" and float(p6["nhat_abs1"]) == 20.6
    assert p6["winner"] == "1" and p6["matched_within_tol"] == "True"
