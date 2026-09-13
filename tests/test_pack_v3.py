"""Tests for the corrected mock scan packs v3 (PI ruling 2026-09-13c §8).

Two layers:

* **machinery** — no data, no jax, runs anywhere: the v3 verifier must FAIL
  CLOSED on a deliberately mismatched product (the mutation test the ruling
  asks for), on a product whose bytes changed after stamping, and on an
  ``fp_counts`` block that has been given the survey support it must not have;

* **delivered artefacts** — skipped when the products are absent: the counting
  arguments are asserted on the real v3 products, namely that the sentinel
  filter moved exactly the predeclared number of rows, all in the top S/N
  stratum, that the v3 pack differs from the A0 pack in ``counts`` ALONE, and
  that every plane of the pack / census / operators shares one row
  ``support_id`` (collar consistency, counted).

Every numeric assertion uses an exact comparison or an explicit tolerance;
none relies on ``np.allclose``'s default atol.
"""
import json
import os
import shutil
import sys

import numpy as np
import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TESTS)
_SUP = os.path.join(_REPO, "validation", "absorber_ladder", "support")
sys.path.insert(0, _SUP)

from support_contract import (                                     # noqa: E402
    NO_TRUTH_SIDE, ROW_SELECTION_FIELDS, SUPPORT_FIELDS,
    SupportContractError, stamp, stamp_path, support_id,
)
import verify_pack_v3 as V                                         # noqa: E402

V3_DIR = ("/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/"
          "support_v3")
A0_DIR = ("/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/"
          "support")
FAMILIES = ("2lpt0", "london0", "saclay0")

BASE = dict(
    collar_kms=3300.0, snr_min=2.0, z_window=(2.0, 4.25), p_dla_min=0.99,
    lya_only_lam_min=1025.0, lam_rf_max=1216.0, z_cut_columns="Z_DLA",
    quality_cut="DLAFLAG==0", truth_host_floor=NO_TRUTH_SIDE,
    bal_policy="mock:drop_all_TARGETID_in_bal_cat", catalogue_id="1f:deadbeef",
    truth_catalogue_sha256="0" * 64)


def _fields(**over):
    f = dict(BASE)
    f.update(over)
    return f


def _triple(tmp_path, pack_over=None, census_over=None, ops_over=None):
    """A pack / census / ops triple with stamped planes, like the real v3 set."""
    paths = {}
    spec = (("pack", ("counts", "dX", "fp_E_alloc", "truth_counts",
                      "truth_counts_bks"), pack_over or {}),
            ("census", ("counts_all", "hostless"), census_over or {}),
            ("ops", ("C_true_bKs", "N_match_cksb"), ops_over or {}))
    for name, planes, over in spec:
        p = str(tmp_path / f"{name}.npz")
        np.savez(p, x=np.arange(3))
        base = _fields(**over)
        sid = support_id(**base)
        plane_fields = {}
        for pl in planes:
            f = dict(base)
            if pl in ("truth_counts", "truth_counts_bks", "C_true_bKs",
                      "N_match_cksb"):
                f["truth_host_floor"] = 19.0
            elif name == "census":
                f["truth_host_floor"] = 17.2
            plane_fields[pl] = f
        stamp(p, sid, extra=dict(planes=plane_fields))
        paths[name] = p
    return paths


# ---------------------------------------------------------------------------
# MACHINERY — the verifier must fail closed
# ---------------------------------------------------------------------------
def test_row_gate_passes_on_a_consistent_triple(tmp_path):
    p = _triple(tmp_path)
    rec = V.check_support_gate(p["pack"], p["census"], p["ops"])
    assert rec["PASS"] is True
    assert rec["row"]["status"] == "PASS"
    # nine planes, one id
    assert len(rec["row"]["planes"]) == 9
    assert len(set(rec["row"]["planes"].values())) == 3      # 3 floors, 1 row id
    # the floor is REPORTED, never hidden
    assert sorted(set(map(str, rec["row"]["truth_host_floor"].values()))) == \
        ["17.2", "19.0", "n/a"]


@pytest.mark.parametrize("field,bad", [
    ("collar_kms", 3000.0),                 # the A0 defect (instance #7)
    ("z_cut_columns", "minmax(Z_DLA,Z_TRUE)"),   # instance #8
    ("snr_min", 3.0),
    ("p_dla_min", 0.9),
    ("lya_only_lam_min", 911.0),
    ("lam_rf_max", 1200.0),
    ("z_window", (2.0, 4.0)),
    ("quality_cut", "none"),
    ("bal_policy", "none"),
    ("catalogue_id", "1f:cafe"),
    ("truth_catalogue_sha256", "1" * 64),
])
def test_row_gate_fails_closed_on_every_mutated_field(tmp_path, field, bad):
    """MUTATION TEST: one differing row-selection field anywhere in the triple
    must fail the gate, and the failure must NAME the field."""
    p = _triple(tmp_path, census_over={field: bad})
    rec = V.check_support_gate(p["pack"], p["census"], p["ops"])
    assert rec["PASS"] is False
    assert rec["row"]["status"] == "FAIL"
    assert field in rec["row"]["error"]


def test_row_gate_is_not_fooled_by_a_floor_difference(tmp_path):
    """``truth_host_floor`` is the ONE field the row level tolerates — and the
    full level must still record it."""
    p = _triple(tmp_path)
    rec = V.check_support_gate(p["pack"], p["census"], p["ops"])
    assert rec["row"]["status"] == "PASS"
    assert rec["full"]["status"] == "FAIL"
    assert rec["full_level_fails_only_on_truth_host_floor"] is True


def test_unstamped_product_fails_closed(tmp_path):
    p = _triple(tmp_path)
    os.remove(stamp_path(p["census"]))
    rec = V.check_support_gate(p["pack"], p["census"], p["ops"])
    assert rec["PASS"] is False
    assert "NO support stamp" in rec["row"]["error"]


def test_product_bytes_changed_after_stamping_fails_closed(tmp_path):
    p = _triple(tmp_path)
    ok = V.check_stamps(p)
    assert all(r["PASS"] for r in ok.values())
    np.savez(p["pack"], x=np.arange(4))          # silently regenerate
    bad = V.check_stamps(p)
    assert bad["pack"]["PASS"] is False
    assert bad["pack"]["UNMODIFIED"] is False
    assert bad["census"]["PASS"] is True


def test_stamp_with_a_drifted_support_id_fails_closed(tmp_path):
    p = _triple(tmp_path)
    sp = stamp_path(p["ops"])
    rec = json.load(open(sp))
    rec["support_id"] = "f" * 64
    json.dump(rec, open(sp, "w"))
    out = V.check_stamps({"ops": p["ops"]})
    assert out["ops"]["PASS"] is False
    assert "INTERNALLY INCONSISTENT" in out["ops"]["error"]


def test_fp_counts_must_not_share_the_survey_support(tmp_path):
    """The loa-0 FP block is a SECOND contract instance; if it is ever stamped
    with the survey support that is a defect, not a pass."""
    survey = support_id(**BASE)
    good = str(tmp_path / "fp_ok.npz")
    np.savez(good, x=np.arange(2))
    own = support_id(**_fields(collar_kms="none", bal_policy="none",
                               quality_cut="none"))
    stamp(good, own, extra=dict(disclosed_difference={"collar": "89 vs 87"}))
    assert V.check_fp_counts_second_instance(
        good, survey.row_sha256)["PASS"] is True

    bad = str(tmp_path / "fp_bad.npz")
    np.savez(bad, x=np.arange(2))
    stamp(bad, survey, extra=dict(disclosed_difference={"collar": "89 vs 87"}))
    assert V.check_fp_counts_second_instance(
        bad, survey.row_sha256)["PASS"] is False


def test_fp_counts_without_a_disclosure_fails(tmp_path):
    survey = support_id(**BASE)
    p = str(tmp_path / "fp.npz")
    np.savez(p, x=np.arange(2))
    stamp(p, support_id(**_fields(collar_kms="none")))       # no disclosure
    assert V.check_fp_counts_second_instance(p, survey.row_sha256)["PASS"] \
        is False


# ---------------------------------------------------------------------------
# DELIVERED ARTEFACTS — the counting arguments
# ---------------------------------------------------------------------------
def _have(fam):
    return (os.path.exists(os.path.join(V3_DIR, f"scanpack_{fam}_b300_v3.npz"))
            and os.path.exists(os.path.join(A0_DIR,
                                            f"scanpack_{fam}_b300_A0.npz")))


@pytest.mark.parametrize("fam", FAMILIES)
def test_sentinel_filter_is_counted(fam):
    """The ONLY difference between the v3 pack and the A0 pack is ``counts``,
    and it is exactly the predeclared sentinel rows, all in stratum s = 7."""
    if not _have(fam):
        pytest.skip(f"v3 / A0 products for {fam} not on disk")
    rec = V.check_a0_equivalence(V3_DIR, fam, A0_DIR)
    assert rec["keys_differing_from_A0"] == ["counts"]
    assert rec["counts_delta"] == V.EXPECTED_SENTINEL_DELTA[fam]
    assert rec["max_abs_cell_difference"] == 1
    assert rec["n_cells_differing"] == abs(V.EXPECTED_SENTINEL_DELTA[fam])
    per_s = rec["delta_per_snr_stratum"]
    assert per_s[V.EXPECTED_SENTINEL_STRATUM] == V.EXPECTED_SENTINEL_DELTA[fam]
    assert [v for i, v in enumerate(per_s)
            if i != V.EXPECTED_SENTINEL_STRATUM] == [0] * (len(per_s) - 1)


@pytest.mark.parametrize("fam", FAMILIES)
def test_counting_identity_counts_equals_census_counts_all(fam):
    """THE decisive counting argument: the pack's numerator and the matched
    census are one row set, bit-exactly."""
    if not _have(fam):
        pytest.skip(f"v3 products for {fam} not on disk")
    prov = json.load(open(os.path.join(
        V3_DIR, f"scanpack_{fam}_b300_v3.provenance.json")))
    ci = prov["findings"]["counting_identity_counts_vs_census"]
    assert ci["EXACT"] is True
    assert ci["n_cells_differing"] == 0
    assert ci["v3_counts_total"] == ci["census_counts_all_total"]
    ctl = ci["control_without_the_sentinel_filter"]
    assert ctl["residual_vs_census"] == -V.EXPECTED_SENTINEL_DELTA[fam]


@pytest.mark.parametrize("fam", FAMILIES)
def test_collar_consistency_is_counted(fam):
    """Every plane of pack / census / operators shares ONE row support_id, and
    the truth planes moved by the collar ratio the forensics predicted."""
    if not _have(fam):
        pytest.skip(f"v3 products for {fam} not on disk")
    rec = V.check_support_gate(
        os.path.join(V3_DIR, f"scanpack_{fam}_b300_v3.npz"),
        os.path.join(V3_DIR, f"fp_census_{fam}_v3.npz"),
        os.path.join(V3_DIR, f"empirical_ops_{fam}_v3.npz"))
    assert rec["row"]["status"] == "PASS"
    assert len(rec["row"]["planes"]) == 18
    assert len(set(rec["row"]["planes"].values())) == 3       # three floors
    assert rec["row"]["fields"]["collar_kms"] == 3300.0
    assert rec["row"]["fields"]["z_cut_columns"] == "Z_DLA"

    prov = json.load(open(os.path.join(
        V3_DIR, f"scanpack_{fam}_b300_v3.provenance.json")))
    tc = prov["findings"]["truth_counting_argument"]
    assert 0.9950 < tc["collar_ratio"] < 0.9960
    assert tc["ladder"][3]["NOOP"] is True                     # idempotence
    steps = {d["step"]: d["n_rows"] for d in tc["ladder"]}
    assert steps["& collar-3300 window"] < list(steps.values())[0]


@pytest.mark.parametrize("fam", FAMILIES)
def test_packs_of_record_are_untouched(fam):
    if not _have(fam):
        pytest.skip(f"v3 products for {fam} not on disk")
    assert V.check_immutability(V3_DIR, fam)["PASS"] is True


def test_end_to_end_verifier_fails_on_a_mutated_copy(tmp_path):
    """MUTATION TEST, end to end: copy the delivered directory, change ONE
    support field in ONE stamp, and the verifier must refuse the whole set."""
    if not _have("2lpt0"):
        pytest.skip("v3 products not on disk")
    dst = str(tmp_path / "support_v3")
    shutil.copytree(V3_DIR, dst, ignore=shutil.ignore_patterns("_work"))
    rec, ok = V.verify(dst, ("2lpt0",), A0_DIR, consumability=False)
    assert ok is True, rec["families"]["2lpt0"]["FAILURES"]

    sp = os.path.join(dst, "fp_census_2lpt0_v3.support.json")
    st = json.load(open(sp))
    st["fields"]["collar_kms"] = 3000.0
    for blk in st["extra"]["planes"].values():
        blk["collar_kms"] = 3000.0
    st.pop("support_id")            # otherwise the self-consistency check fires
    json.dump(st, open(sp, "w"))
    rec2, ok2 = V.verify(dst, ("2lpt0",), A0_DIR, consumability=False)
    assert ok2 is False
    assert "support_gate" in rec2["families"]["2lpt0"]["FAILURES"]
    assert "collar_kms" in rec2["families"]["2lpt0"]["support_gate"]["row"][
        "error"]
