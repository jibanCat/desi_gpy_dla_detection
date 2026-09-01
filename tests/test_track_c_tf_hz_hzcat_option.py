"""MAX4 repair cycle (PI ruling 2026-08-28 item 3; MAX4 CHECKPOINT 0 §D): track_c_tf_hz.py takes
--hz-cat / --hz-mockdir (defaults = the run-of-record constants, so the recorded invocations are
unchanged) and stamps the consumed catalogue, tables and the catalogue's finder configuration
(from its BASELINE.env) into the output metadata. Synthetic inputs only."""
import inspect
import os

import pytest

HZ = pytest.importorskip("CDDF_analysis.hbi.track_c_tf_hz")


def test_parser_defaults_are_the_run_of_record():
    a = HZ.build_parser().parse_args([])
    assert a.hz_cat == HZ.HZ_CAT and a.hz_mockdir == HZ.HZ_MOCKDIR
    # every pre-existing default unchanged
    assert (a.variant, a.fp, a.window, a.envelope, a.gap_treatment) == ("h2cal", "loa0", "lya", "none", "frozen")
    assert a.gap_c is None and a.gap_c_neff is None and a.r041_analysis is None
    assert a.zbins == "3.8,4.25,4.5,5.0" and a.n_mc == 120
    assert a.out_json is None and a.force is False and a.finite_snr_only is False
    assert a.work_root is None and a.dump_npz is None


def test_parser_accepts_overrides(tmp_path):
    a = HZ.build_parser().parse_args(["--hz-cat", str(tmp_path / "cat"), "--hz-mockdir", str(tmp_path / "md")])
    assert a.hz_cat == str(tmp_path / "cat") and a.hz_mockdir == str(tmp_path / "md")


def test_finder_config_from_synthetic_baseline_env(tmp_path):
    cat = tmp_path / "max4_cat"
    cat.mkdir()
    (cat / "BASELINE.env").write_text(
        "# Resolved env for run launched (synthetic)\n# config: x.env\n"
        "CODE_COMMIT=0123456789abcdef0123456789abcdef01234567\nCODE_BRANCH=synthetic\n"
        "MAX_DLAS=4\nSINGLE_ABSORBER_MODEL=1\nFILTER_LOW_LIKELIHOOD=1\nNUM_DLA_SAMPLES=50000\n"
        "DLA_SAMPLES_FILE=/x/pw_samples_a3_172_225_50000.mat\nPAIR_PRIOR_MODE=(unset)\n")
    fc = HZ.read_finder_config(cat)
    assert fc["MAX_DLAS"] == "4" and fc["SINGLE_ABSORBER_MODEL"] == "1"
    assert fc["FILTER_LOW_LIKELIHOOD"] == "1" and fc["NUM_DLA_SAMPLES"] == "50000"
    assert fc["CODE_COMMIT"] == "0123456789abcdef0123456789abcdef01234567"
    assert fc["baseline_env"] == str(cat / "BASELINE.env")
    assert set(fc) == set(HZ.FINDER_CONFIG_KEYS) | {"baseline_env"}
    # a missing key is None, not an error
    (cat / "BASELINE.env").write_text("MAX_DLAS=1\n")
    fc = HZ.read_finder_config(cat)
    assert fc["MAX_DLAS"] == "1" and fc["FILTER_LOW_LIKELIHOOD"] is None
    # no BASELINE.env at all
    assert HZ.read_finder_config(tmp_path / "nowhere") == "unavailable"


def test_metadata_stamp_block(tmp_path):
    cat = tmp_path / "cat"
    cat.mkdir()
    (cat / "BASELINE.env").write_text("MAX_DLAS=4\nSINGLE_ABSORBER_MODEL=1\nFILTER_LOW_LIKELIHOOD=1\n"
                                      "NUM_DLA_SAMPLES=50000\nCODE_COMMIT=abc\n")
    md = tmp_path / "md"
    stamp = HZ.hz_input_stamp(cat, md)
    assert stamp["hz_cat"] == str(cat) and stamp["hz_mockdir"] == str(md)
    assert stamp["finder_config"]["MAX_DLAS"] == "4" and stamp["finder_config"]["CODE_COMMIT"] == "abc"
    assert HZ.hz_input_stamp(tmp_path / "none", md)["finder_config"] == "unavailable"


def test_main_routes_the_options_and_stamps_metadata():
    # wiring check without running the measurement: the options feed args.loa_* and the
    # metadata block is merged from hz_input_stamp(a.hz_cat, a.hz_mockdir).
    src = inspect.getsource(HZ.main)
    assert "args.loa_cat = a.hz_cat" in src and "args.loa_mockdir = a.hz_mockdir" in src
    assert 'os.path.join(a.hz_mockdir, "dla_cat.fits")' in src and 'os.path.join(a.hz_mockdir, "bal_cat.fits")' in src
    assert "**hz_input_stamp(a.hz_cat, a.hz_mockdir)" in src
    assert "HZ_CAT\n" not in src.replace("default=HZ_CAT", "")  # no remaining hard-wired use in main
