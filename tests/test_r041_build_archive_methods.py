"""tools/r041_build_archive.py method dispatch (MAX4 repair cycle follow-up, 2026-09-01): the three
--method choices run end-to-end on a tiny synthetic LoaArchive; each injected flux equals the
corresponding injection primitive called directly (prescription A = the injector's
variance_preserving with its default seed 0 for every sightline; B = residual_preserving; OLD =
multiplicative); the build summary records `method` as passed plus `injection_prescription`; and
a second build of the same plan reproduces the flux arrays bit-for-bit. Synthetic inputs only."""
import csv
import json
import os
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
fitsio = pytest.importorskip("fitsio")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, REPO)
import r041_build_archive as B  # noqa: E402
from injection import noise_preserving as NP  # noqa: E402

N_PIX = 1600
TIDS = [101, 102, 103]
PLAN = [dict(TARGETID=101, wave=0, inj_idx=0, logN=20.5, z_inj=4.00, stratum=1, snr=3.0, has_cand_ge20=0),
        dict(TARGETID=102, wave=0, inj_idx=0, logN=21.0, z_inj=4.10, stratum=2, snr=4.5, has_cand_ge20=1),
        dict(TARGETID=103, wave=0, inj_idx=0, logN=20.3, z_inj=3.92, stratum=0, snr=2.4, has_cand_ge20=0)]


def make_inputs(root):
    """synthetic source archive (schema used by the builder), plan CSV, QSO catalogue -> paths"""
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(11)
    wave = np.linspace(3600.0, 6800.0, N_PIX)
    n = len(TIDS)
    cont = 1.0 + 0.2 * np.sin(wave / 300.0)
    flux = np.empty((n, N_PIX), np.float32); ivar = np.empty((n, N_PIX), np.float32); mask = np.zeros((n, N_PIX), np.uint32)
    for i in range(n):
        forest = np.clip(1.0 - 0.6 * rng.random(N_PIX) * (wave < 6200.0), 0.05, 1.0)
        S = cont * forest; sigma = S.mean() / (3.0 + i)
        flux[i] = S + rng.standard_normal(N_PIX) * sigma; ivar[i] = 1.0 / sigma ** 2
        mask[i, 50 + 7 * i:60 + 7 * i] = 1; ivar[i, 300 + i] = 0.0
    cat = np.array([(t, 4.6 + 0.1 * i, 10.0 + i, 20.0 + i) for i, t in enumerate(TIDS)],
                   dtype=[("TARGETID", "i8"), ("Z", "f8"), ("RA", "f8"), ("DEC", "f8")])
    arch = os.path.join(root, "source_archive.h5")
    with h5py.File(arch, "w") as h:
        h.attrs["schema_version"] = 1
        h.create_dataset("wavelength", data=wave.astype(np.float64)); h.create_dataset("catalog", data=cat)
        h.create_dataset("flux", data=flux); h.create_dataset("ivar", data=ivar); h.create_dataset("mask", data=mask)
        h.create_dataset("fwhm_pix", data=np.full((n, N_PIX), 1.1, np.float32))
    plan = os.path.join(root, "plan.csv")
    with open(plan, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(PLAN[0])); w.writeheader(); w.writerows(PLAN)
    qso = np.array([(t, 4.6 + 0.1 * i, 10.0 + i, 20.0 + i, 7000 + i) for i, t in enumerate(TIDS)],
                   dtype=[("TARGETID", "i8"), ("Z", "f8"), ("RA", "f8"), ("DEC", "f8"), ("HPXPIXEL", "i8")])
    qsocat = os.path.join(root, "qsocat.fits"); fitsio.write(qsocat, qso, clobber=True)
    return dict(archive=arch, plan=plan, qsocat=qsocat, wave=wave, flux=flux, ivar=ivar, mask=mask, cat=cat)


def build(inputs, out_dir, method, tag="t"):
    B.main(["--plan", inputs["plan"], "--wave", "0", "--archive", inputs["archive"], "--qsocat", inputs["qsocat"],
            "--out-dir", out_dir, "--tag", tag, "--method", method])
    h5 = os.path.join(out_dir, f"r041_{tag}_wave0.h5")
    with h5py.File(h5, "r") as h:
        out = dict(flux=h["flux"][:], ivar=h["ivar"][:], mask=h["mask"][:], tids=h["catalog"][:]["TARGETID"], attrs=dict(h.attrs))
    out["summary"] = json.load(open(h5 + ".build_summary.json"))
    out["truth"] = list(csv.DictReader(open(h5 + ".truth.csv")))
    out["h5"] = h5
    return out


def _expected(inputs, method):
    wave = inputs["wave"]; exp = np.empty_like(inputs["flux"])
    for i, t in enumerate(TIDS):
        r = [p for p in PLAN if p["TARGETID"] == t][0]
        ab = [{"nhi": 10.0 ** r["logN"], "z_dla": r["z_inj"], "num_lines": 3}]
        fl = inputs["flux"][i].astype(np.float64)
        if method == "multiplicative":
            e = NP.inject_multiplicative(wave, fl, ab, 3)
        else:
            e = NP.inject_noise_preserving(wave, fl, inputs["ivar"][i], inputs["mask"][i], ab, z_qso=float(inputs["cat"][i]["Z"]),
                                           r=None, seed=0, num_lines=3, method=B.METHODS[method][0])
        exp[i] = e.astype(np.float32)
    return exp


@pytest.mark.parametrize("method", ["noise_preserving", "residual_preserving", "multiplicative"])
def test_builder_dispatch_matches_the_primitive(tmp_path, method):
    inputs = make_inputs(str(tmp_path / "in"))
    out = build(inputs, str(tmp_path / method), method)
    assert np.array_equal(out["flux"], _expected(inputs, method))              # exactly the primitive's result (float32)
    assert np.array_equal(out["ivar"], inputs["ivar"]) and np.array_equal(out["mask"], inputs["mask"])   # copied unchanged
    assert list(out["tids"]) == TIDS
    assert out["summary"]["method"] == method and out["attrs"]["r041_method"] == method
    assert out["summary"]["injection_prescription"] == B.METHODS[method][1] and "F'" in out["summary"]["injection_prescription"]
    assert out["summary"]["injector_method"] == B.METHODS[method][0]
    assert all(t["method"] == method for t in out["truth"]) and len(out["truth"]) == 3


def test_prescription_A_is_seed_zero_for_every_sightline(tmp_path):
    # documents the archive route's deterministic state: no per-sightline seed -> the injector default 0 everywhere
    inputs = make_inputs(str(tmp_path / "in"))
    out = build(inputs, str(tmp_path / "a"), "noise_preserving")
    assert "seed 0 for every sightline" in out["summary"]["noise_seed_policy"]
    wave = inputs["wave"]; r = PLAN[1]; i = 1
    ab = [{"nhi": 10.0 ** r["logN"], "z_dla": r["z_inj"], "num_lines": 3}]
    with_seed_1 = NP.inject_noise_preserving(wave, inputs["flux"][i].astype(np.float64), inputs["ivar"][i], inputs["mask"][i], ab,
                                             z_qso=float(inputs["cat"][i]["Z"]), seed=1).astype(np.float32)
    assert not np.array_equal(out["flux"][i], with_seed_1)                     # a different seed would have changed the bytes
    with_seed_0 = NP.inject_noise_preserving(wave, inputs["flux"][i].astype(np.float64), inputs["ivar"][i], inputs["mask"][i], ab,
                                             z_qso=float(inputs["cat"][i]["Z"]), seed=0).astype(np.float32)
    assert np.array_equal(out["flux"][i], with_seed_0)                         # ... and seed 0 is what the builder used


def test_residual_preserving_and_A_differ_only_inside_the_profile(tmp_path):
    inputs = make_inputs(str(tmp_path / "in"))
    a = build(inputs, str(tmp_path / "a"), "noise_preserving"); b = build(inputs, str(tmp_path / "b"), "residual_preserving")
    assert not np.array_equal(a["flux"], b["flux"])
    assert b["summary"]["noise_seed_policy"] == "no synthetic noise"


def test_rebuilding_the_same_plan_is_bit_identical(tmp_path):
    inputs = make_inputs(str(tmp_path / "in"))
    for method in ("noise_preserving", "residual_preserving"):
        one = build(inputs, str(tmp_path / f"{method}_1"), method); two = build(inputs, str(tmp_path / f"{method}_2"), method)
        assert np.array_equal(one["flux"], two["flux"]) and one["flux"].dtype == np.float32
        assert open(one["h5"] + ".truth.csv", "rb").read() == open(two["h5"] + ".truth.csv", "rb").read()
        assert one["summary"]["truth_sha256"] == two["summary"]["truth_sha256"]


def test_unknown_method_is_refused(tmp_path):
    inputs = make_inputs(str(tmp_path / "in"))
    with pytest.raises(SystemExit):
        B.main(["--plan", inputs["plan"], "--wave", "0", "--archive", inputs["archive"], "--qsocat", inputs["qsocat"],
                "--out-dir", str(tmp_path / "x"), "--tag", "x", "--method", "no_such_method"])
    with pytest.raises(ValueError):
        B.inject_sightline("no_such_method", None, None, None, None, [], z_qso=4.5, alt=None, fid=None, num_lines=3, median_px=9, sigma_px=2.5)
