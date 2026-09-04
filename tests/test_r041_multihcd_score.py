"""tools/r041_multihcd_score.py — the predeclared multi-HCD scoring rules on a synthetic fixture (written before any P1
pairs output was read): one-to-one greedy matching; resolvable pairs (dv >= 3000 km/s) scored per absorber with
captured / split labels; unresolvable pairs (dv < 3000) scored as ONE system with N_sys = log10(N1 + N2); singles;
truth outside the window ignored; verdict path executes. Mutation check: a wrong tolerance changes the matches."""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "tools", "r041_multihcd_score.py")
C = 299792.458


def dv_to_dz(z, dv):
    return dv / C * (1.0 + z)


def make(tmp):
    # population: five sightlines, window [3.5, 4.4]
    pop = os.path.join(tmp, "pop.csv")
    with open(pop, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["TARGETID", "zlo", "zhi", "stratum"]); w.writeheader()
        for t in (1, 2, 3, 4, 5, 6):
            w.writerow(dict(TARGETID=t, zlo=3.5, zhi=4.4, stratum=2))
    # truth: 1 = wide pair both found; 2 = wide pair, one captured (single row near both -> nearest claims it), 3 = unresolvable
    # blend pair -> one system matched; 4 = wide pair with a split extra row; 5 = single missed; 6 = truth outside window (ignored)
    rows = [dict(TARGETID=1, wave=0, inj_idx=0, logN=20.4, z_inj=3.80, stratum=2, snr=5.0, pair_class="wide", dv_kms=10000, pair_logN="20.4+20.4"),
            dict(TARGETID=1, wave=0, inj_idx=1, logN=20.4, z_inj=3.80 + dv_to_dz(3.80, 10000), stratum=2, snr=5.0, pair_class="wide", dv_kms=10000, pair_logN="20.4+20.4"),
            dict(TARGETID=2, wave=0, inj_idx=0, logN=20.4, z_inj=3.80, stratum=2, snr=5.0, pair_class="partial", dv_kms=3500, pair_logN="20.4+20.4"),
            dict(TARGETID=2, wave=0, inj_idx=1, logN=20.4, z_inj=3.80 + dv_to_dz(3.80, 3500), stratum=2, snr=5.0, pair_class="partial", dv_kms=3500, pair_logN="20.4+20.4"),
            dict(TARGETID=3, wave=0, inj_idx=0, logN=20.4, z_inj=3.90, stratum=2, snr=5.0, pair_class="blend", dv_kms=500, pair_logN="20.4+20.4"),
            dict(TARGETID=3, wave=0, inj_idx=1, logN=20.4, z_inj=3.90 + dv_to_dz(3.90, 500), stratum=2, snr=5.0, pair_class="blend", dv_kms=500, pair_logN="20.4+20.4"),
            dict(TARGETID=4, wave=0, inj_idx=0, logN=20.7, z_inj=3.70, stratum=2, snr=5.0, pair_class="wide", dv_kms=12000, pair_logN="20.7+20.7"),
            dict(TARGETID=4, wave=0, inj_idx=1, logN=20.7, z_inj=3.70 + dv_to_dz(3.70, 12000), stratum=2, snr=5.0, pair_class="wide", dv_kms=12000, pair_logN="20.7+20.7"),
            dict(TARGETID=5, wave=0, inj_idx=0, logN=20.4, z_inj=4.0, stratum=2, snr=5.0, pair_class="", dv_kms="", pair_logN=""),
            dict(TARGETID=6, wave=0, inj_idx=0, logN=20.4, z_inj=3.0, stratum=2, snr=5.0, pair_class="", dv_kms="", pair_logN="")]
    rows1 = [dict(TARGETID=1, wave=1, inj_idx=0, logN=20.4, z_inj=4.10, stratum=2, snr=5.0, pair_class="blend", dv_kms=500, pair_logN="20.4+20.4"),
             dict(TARGETID=1, wave=1, inj_idx=1, logN=20.4, z_inj=4.10 + dv_to_dz(4.10, 500), stratum=2, snr=5.0, pair_class="blend", dv_kms=500, pair_logN="20.4+20.4")]
    truth = os.path.join(tmp, "truth.csv")
    with open(truth, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    truth1 = os.path.join(tmp, "truth_wave1.csv")
    with open(truth1, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows1[0])); w.writeheader(); w.writerows(rows1)
    # accepted rows
    acc = [(1, 3.8005, 20.45), (1, rows[1]["z_inj"] + 0.001, 20.35),                       # both found
           (2, 3.8 + 0.5 * dv_to_dz(3.80, 3500) + 0.003, 20.5),                            # resolvable (3500 >= 3000 km/s) pair, ONE row between them within tolerance of both (0.028 each), nearer the second -> second matched, first captured
           (3, 3.9 + 0.5 * dv_to_dz(3.9, 500), 20.75),                                      # blend -> one system row with N ~ log10(2*10^20.4)=20.70
           (4, 3.7005, 20.6), (4, rows[7]["z_inj"] - 0.001, 20.8), (4, 3.7 + 0.5 * dv_to_dz(3.7, 12000), 20.1)]   # both found + a split extra row between them
    outdir = os.path.join(tmp, "out"); os.makedirs(outdir)
    t = np.array([(a, z, n, 1.0) for a, z, n in acc], dtype=[("TARGETID", "i8"), ("Z_DLA", "f8"), ("NHI", "f8"), ("P_DLA", "f8")])
    fits.BinTableHDU(t, name="DLACAT").writeto(os.path.join(outdir, "dlacat-x-hpx-0-1.fits"))
    outdir1 = os.path.join(tmp, "out1"); os.makedirs(outdir1)          # wave 1: the blend on sightline 1 reported as one row (system matched)
    t1 = np.array([(1, 4.10 + 0.5 * dv_to_dz(4.10, 500), 20.7, 1.0)], dtype=t.dtype)
    fits.BinTableHDU(t1, name="DLACAT").writeto(os.path.join(outdir1, "dlacat-x-hpx-0-1.fits"))
    # reference (m=1) CSV: 40 candidate-free rows at 20.4 stratum 2, 30 detected
    ref = os.path.join(tmp, "ref.csv")
    with open(ref, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["TARGETID", "wave", "inj_idx", "logN", "z_inj", "stratum", "has_cand_ge20", "detected", "nhat"]); w.writeheader()
        for i in range(40):
            w.writerow(dict(TARGETID=1000 + i, wave=0, inj_idx=0, logN=20.4, z_inj=3.9, stratum=2, has_cand_ge20=0, detected=str(i < 30), nhat=(20.45 if i < 30 else "")))
        for i in range(40):
            w.writerow(dict(TARGETID=2000 + i, wave=0, inj_idx=0, logN=20.7, z_inj=3.9, stratum=2, has_cand_ge20=0, detected=str(i < 36), nhat=(20.7 if i < 36 else "")))
    weights = os.path.join(tmp, "w.json")
    json.dump(dict(g_cell={"1": 0.00312, "2": 0.00653, "3": 0.0025}, s_stratum=[0.133, 0.144, 0.119, 0.194, 0.41], q_cand=[0.6] * 5), open(weights, "w"))
    return pop, truth, outdir, ref, weights, truth1, outdir1


def run(tmp, tol="0.01"):
    os.makedirs(tmp, exist_ok=True)
    pop, truth, outdir, ref, weights, truth1, outdir1 = make(tmp)
    out = os.path.join(tmp, "res.json")
    subprocess.run([sys.executable, TOOL, "--truth", truth, truth1, "--outputs", outdir, outdir1, "--reference", ref, "--population", pop, "--weights", weights,
                    "--n-boot", "50", "--out", out, "--tol", tol], check=True, capture_output=True)
    units = list(csv.DictReader(open(out[:-5] + "_units.csv"))); absorbers = list(csv.DictReader(open(out[:-5] + "_absorbers.csv")))
    return json.load(open(out)), units, absorbers


def test_scoring_rules(tmp_path):
    res, units, absorbers = run(str(tmp_path))
    by = {}
    for u in units:
        if int(u["wave"]) == 0: by.setdefault(int(u["TARGETID"]), []).append(u)
    w1 = [u for u in units if int(u["wave"]) == 1]
    assert len(w1) == 1 and w1[0]["kind"] == "system" and w1[0]["matched"] == "True" and int(w1[0]["m_true"]) == 2   # wave 1 scored on its own spectrum, not pooled with wave 0
    assert res["n_sightline_spectra"] == 6 and res["n_sightlines"] == 5 and res["waves"] == [0, 1]
    assert 6 not in by and res["n_truth_outside_window"] == 1                      # outside the window: ignored
    assert [u["kind"] for u in by[1]] == ["absorber", "absorber"] and all(u["matched"] == "True" for u in by[1])
    a2 = {int(x["inj_idx"]): x for x in absorbers if int(x["TARGETID"]) == 2}
    assert (a2[0]["matched"], a2[1]["matched"]) == ("False", "True") and a2[0]["captured"] == "True"   # one row, claimed by the nearer absorber
    s3 = by[3]; assert len(s3) == 1 and s3[0]["kind"] == "system" and s3[0]["matched"] == "True"
    assert abs(float(s3[0]["logN"]) - np.log10(2 * 10 ** 20.4)) < 1e-6 and s3[0]["sep_class"] == "close"   # N_sys = log10(N1+N2)
    assert all(u["matched"] == "True" for u in by[4]) and all(int(u["split"]) == 1 for u in by[4])         # both found + one split component
    assert by[5][0]["kind"] == "single" and by[5][0]["matched"] == "False"
    assert res["merge_split"]["wide"]["n"] == 2 and abs(res["merge_split"]["wide"]["split"] - 0.5) < 1e-9 and res["merge_split"]["wide"]["any_captured"] == 0.0
    assert res["merge_split"]["moderate"]["n"] == 1 and res["merge_split"]["moderate"]["any_captured"] == 1.0 and res["merge_split"]["moderate"]["pair_recovery"] == 0.0
    assert res["verdict"]["tier"] in ("PASS", "BOUNDED", "FAIL", "INCONCLUSIVE") and res["primary"]["dC_w_multi"] is not None
    # mutation: a tolerance too small to reach the rows must lose the matches
    res2, units2, _ = run(str(tmp_path / "m"), tol="0.00001")
    assert sum(u["matched"] == "True" for u in units2) < sum(u["matched"] == "True" for u in units)
