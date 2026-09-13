"""Unit tests for the machine-enforced SUPPORT INVARIANT
(``validation/absorber_ladder/support/support_contract.py``; PI ruling
2026-09-13b §3).

The contract exists because the project's recurring bug class is a numerator and
a denominator built on different supports (``feedback_one_sided_support_bug_class``;
instance #7 = the mock scan packs' collar-3000 ``truth_counts`` against
collar-3300 ``counts``/``dX``).  These tests pin the three properties the ruling
asks for — identical supports PASS, any single differing field FAILS, a missing
field FAILS CLOSED — plus the fail-closed behaviour on unknown/None/NaN values,
on an unstamped product, and on an internally inconsistent stamp.

Fast: no data, no jax, no sampler; runs under either environment.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "validation", "absorber_ladder", "support"))

from support_contract import (                                     # noqa: E402
    NO_TRUTH_SIDE, PI_MINIMUM_FIELDS, SUPPORT_FIELDS,
    SUPPORT_SCHEMA_VERSION, SupportContractError, SupportID, SupportMismatch,
    assert_same_support, catalogue_identity, check_support_consistency,
    read_stamp, stamp, stamp_array, stamp_path, support_from_json, support_id,
)

# A complete, valid declaration: the mock scan-pack data plane at collar 3300.
BASE = dict(
    collar_kms=3300.0,
    snr_min=2.0,
    z_window=(2.0, 4.25),
    p_dla_min=0.99,
    lya_only_lam_min=1025.0,
    lam_rf_max=1216.0,
    z_cut_columns="Z_DLA",
    quality_cut="DLAFLAG==0",
    truth_host_floor=19.0,
    bal_policy="mock:drop_all_bal_cat",
    catalogue_id="1f:abcdef0123456789abcdef01",
    truth_catalogue_sha256="0" * 64,
)

#: A per-field perturbation that must change the support id.
PERTURB = dict(
    collar_kms=3000.0,
    snr_min=3.0,
    z_window=(2.0, 4.0),
    p_dla_min=0.95,
    lya_only_lam_min=911.0,
    lam_rf_max=1215.67,
    z_cut_columns="minmax(Z_DLA,Z_TRUE)",
    quality_cut="DLAFLAG==0 & ~lyb_veto",
    truth_host_floor=17.2,
    bal_policy="real:BI_CIV>0",
    catalogue_id="1f:0123456789abcdef01234567",
    truth_catalogue_sha256="1" * 64,
)


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
def test_schema_is_closed_and_covers_the_pi_minimum():
    assert set(PI_MINIMUM_FIELDS) <= set(SUPPORT_FIELDS)
    assert len(set(SUPPORT_FIELDS)) == len(SUPPORT_FIELDS)
    assert set(BASE) == set(SUPPORT_FIELDS) == set(PERTURB)


def test_canonical_string_carries_the_schema_version():
    s = support_id(**BASE)
    assert json.loads(s.canonical)["schema"] == SUPPORT_SCHEMA_VERSION
    assert len(s.sha256) == 64 and s.short == s.sha256[:16]


# --------------------------------------------------------------------------
# identical supports PASS
# --------------------------------------------------------------------------
def test_identical_supports_pass():
    a, b = support_id(**BASE), support_id(**dict(BASE))
    assert a.sha256 == b.sha256 and a == b
    assert assert_same_support(dict(counts=a, dX=b, truth_counts=a,
                                    fp_census=b)) == a.sha256


def test_key_order_and_numeric_spelling_do_not_change_the_id():
    shuffled = {k: BASE[k] for k in reversed(SUPPORT_FIELDS)}
    assert support_id(**shuffled).sha256 == support_id(**BASE).sha256
    # int 3300 vs float 3300.0 vs np.float64 must canonicalise identically
    for v in (3300, 3300.0, np.float64(3300.0), np.int64(3300)):
        assert support_id(**{**BASE, "collar_kms": v}).sha256 == \
            support_id(**BASE).sha256
    # tuple vs list vs ndarray for the z window
    for v in ((2.0, 4.25), [2.0, 4.25], np.array([2.0, 4.25])):
        assert support_id(**{**BASE, "z_window": v}).sha256 == \
            support_id(**BASE).sha256


# --------------------------------------------------------------------------
# ANY single differing field FAILS
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field", SUPPORT_FIELDS)
def test_any_single_differing_field_changes_the_id_and_fails(field):
    ref = support_id(**BASE)
    other = support_id(**{**BASE, field: PERTURB[field]})
    assert other.sha256 != ref.sha256, f"{field} is not support-defining!"
    with pytest.raises(SupportMismatch) as exc:
        assert_same_support({"counts": ref, "truth_counts": other})
    msg = str(exc.value)
    assert field in msg, f"the diff must name the differing field {field}"
    assert "SUPPORT MISMATCH" in msg


def test_the_real_defect_is_caught_collar_3000_truth_vs_3300_counts():
    """Instance #7 verbatim: truth_counts @3000 against counts/dX @3300."""
    counts = support_id(**{**BASE, "collar_kms": 3300.0,
                           "truth_host_floor": NO_TRUTH_SIDE})
    dX = support_id(**{**BASE, "collar_kms": 3300.0,
                       "truth_host_floor": NO_TRUTH_SIDE})
    truth = support_id(**{**BASE, "collar_kms": 3000.0})
    assert assert_same_support({"counts": counts, "dX": dX}) == counts.sha256
    with pytest.raises(SupportMismatch) as exc:
        assert_same_support({"counts": counts, "dX": dX,
                             "truth_counts": truth})
    assert "collar_kms" in str(exc.value)


# --------------------------------------------------------------------------
# fail-closed
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field", SUPPORT_FIELDS)
def test_missing_field_fails_closed(field):
    fields = {k: v for k, v in BASE.items() if k != field}
    with pytest.raises(SupportContractError) as exc:
        support_id(**fields)
    assert "missing support field" in str(exc.value)
    assert field in str(exc.value)


def test_unknown_field_fails_closed():
    with pytest.raises(SupportContractError) as exc:
        support_id(**BASE, molly_tsv="/some/matrix.tsv")
    assert "unknown support field" in str(exc.value)
    assert "CLOSED" in str(exc.value)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "", "   "])
def test_unknown_or_blank_values_fail_closed(bad):
    with pytest.raises(SupportContractError):
        support_id(**{**BASE, "collar_kms": bad})


def test_bool_and_odd_types_fail_closed():
    with pytest.raises(SupportContractError):
        support_id(**{**BASE, "collar_kms": True})
    with pytest.raises(SupportContractError):
        support_id(**{**BASE, "z_window": {"lo": 2.0}})
    with pytest.raises(SupportContractError):
        support_id(**{**BASE, "z_window": ()})


def test_empty_object_set_is_not_a_pass():
    with pytest.raises(SupportContractError):
        assert_same_support({})


# --------------------------------------------------------------------------
# stamping / reading
# --------------------------------------------------------------------------
def test_stamp_round_trip(tmp_path):
    p = tmp_path / "prod.npz"
    np.savez(p, a=np.arange(3))
    sid = support_id(**BASE)
    sp = stamp(str(p), sid, extra=dict(note="A0 rebuild"))
    assert sp == stamp_path(str(p)) == str(tmp_path / "prod.support.json")
    rec = json.loads(open(sp).read())
    assert rec["support_id"] == sid.sha256
    assert rec["product_sha256"] is not None
    assert rec["extra"]["note"] == "A0 rebuild"
    back = read_stamp(str(p))
    assert back.sha256 == sid.sha256 and back.fields["collar_kms"] == 3300.0


def test_unstamped_product_fails_closed(tmp_path):
    p = tmp_path / "bare.npz"
    np.savez(p, a=np.arange(3))
    with pytest.raises(SupportContractError) as exc:
        read_stamp(str(p))
    assert "NO support stamp" in str(exc.value)
    assert read_stamp(str(p), required=False) is None
    with pytest.raises(SupportContractError):
        check_support_consistency(str(p))


def test_embedded_npz_key_is_accepted_as_a_sha_only_stamp(tmp_path):
    p = tmp_path / "emb.npz"
    sid = support_id(**BASE)
    np.savez(p, a=np.arange(3), support_id=stamp_array(sid))
    got = read_stamp(str(p))
    assert got.sha256 == sid.sha256 and got.fields == {}
    assert assert_same_support({"a": got, "b": sid}) == sid.sha256


def test_tampered_stamp_fails_closed(tmp_path):
    p = tmp_path / "prod.npz"
    np.savez(p, a=np.arange(3))
    sp = stamp(str(p), support_id(**BASE))
    rec = json.loads(open(sp).read())
    rec["fields"]["collar_kms"] = 3000.0          # fields edited, sha left alone
    open(sp, "w").write(json.dumps(rec))
    with pytest.raises(SupportContractError) as exc:
        read_stamp(str(p))
    assert "INTERNALLY INCONSISTENT" in str(exc.value)


def test_stamp_schema_drift_is_refused(tmp_path):
    rec = support_id(**BASE).to_json()
    rec["schema"] = "support_contract/v99"
    with pytest.raises(SupportContractError):
        support_from_json(rec)


# --------------------------------------------------------------------------
# the gate the ladder runner calls
# --------------------------------------------------------------------------
def _product(tmp_path, name, sid):
    p = tmp_path / name
    np.savez(p, a=np.arange(3))
    stamp(str(p), sid)
    return str(p)


def test_check_support_consistency_pass_and_fail(tmp_path):
    good = support_id(**BASE)
    bad = support_id(**{**BASE, "collar_kms": 3000.0})
    pack = _product(tmp_path, "pack.npz", good)
    cen_ok = _product(tmp_path, "census_ok.npz", good)
    ops_ok = _product(tmp_path, "ops_ok.npz", good)
    rec = check_support_consistency(pack, cen_ok, ops_ok)
    assert rec["status"] == "PASS" and rec["support_id"] == good.sha256
    assert set(rec["planes"]) == {"pack", "fp_census", "empirical_ops"}
    cen_bad = _product(tmp_path, "census_bad.npz", bad)
    with pytest.raises(SupportMismatch) as exc:
        check_support_consistency(pack, cen_bad)
    assert "collar_kms" in str(exc.value)


def test_cli_exit_codes(tmp_path, capsys):
    import support_contract as SC
    good = support_id(**BASE)
    bad = support_id(**{**BASE, "snr_min": 3.0})
    pack = _product(tmp_path, "pack.npz", good)
    cen_ok = _product(tmp_path, "c_ok.npz", good)
    cen_bad = _product(tmp_path, "c_bad.npz", bad)
    assert SC.main(["--pack", pack, "--census", cen_ok]) == 0
    ok = json.loads(capsys.readouterr().out)
    assert ok["status"] == "PASS" and ok["support_id"] == good.sha256
    assert SC.main(["--pack", pack, "--census", cen_bad, "--json"]) == 1
    bad_out = json.loads(capsys.readouterr().out)
    assert bad_out["status"] == "FAIL" and "snr_min" in bad_out["error"]


# --------------------------------------------------------------------------
# catalogue identity
# --------------------------------------------------------------------------
def test_catalogue_identity_changes_with_content(tmp_path):
    d = tmp_path / "cat"
    d.mkdir()
    (d / "dlacat-a.fits").write_bytes(b"one")
    a = catalogue_identity(str(d))
    (d / "dlacat-a.fits").write_bytes(b"two")
    assert catalogue_identity(str(d)) != a
    (d / "dlacat-b.fits").write_bytes(b"three")
    assert catalogue_identity(str(d)).startswith("2f:")
    with pytest.raises(SupportContractError):
        catalogue_identity(str(tmp_path / "nope"))


# --------------------------------------------------------------------------
# field subsets (the row-selection relaxation) and plane-level stamps
# --------------------------------------------------------------------------
def test_row_selection_subset_excludes_only_the_truth_floor():
    from support_contract import ROW_SELECTION_FIELDS
    assert set(SUPPORT_FIELDS) - set(ROW_SELECTION_FIELDS) == {"truth_host_floor"}
    a = support_id(**BASE)
    b = support_id(**{**BASE, "truth_host_floor": 17.2})
    assert a.sha256 != b.sha256                      # full level: DIFFERENT
    assert a.row_sha256 == b.row_sha256              # row level:  SAME
    with pytest.raises(SupportMismatch):
        assert_same_support({"pack": a, "census": b})
    assert assert_same_support({"pack": a, "census": b},
                               fields=ROW_SELECTION_FIELDS) == a.row_sha256
    # a row-selection difference still fails at the relaxed level
    c = support_id(**{**BASE, "truth_host_floor": 17.2, "collar_kms": 3000.0})
    with pytest.raises(SupportMismatch):
        assert_same_support({"pack": a, "census": c},
                            fields=ROW_SELECTION_FIELDS)


def test_row_sha_is_domain_separated_from_the_full_sha():
    s = support_id(**BASE)
    assert s.row_sha256 != s.sha256


def test_empty_or_unknown_comparison_subset_fails_closed():
    a = support_id(**BASE)
    with pytest.raises(SupportContractError):
        assert_same_support({"x": a}, fields=())
    with pytest.raises(SupportContractError):
        assert_same_support({"x": a}, fields=("collar_kms", "nope"))


def test_sha_only_stamp_cannot_be_compared_at_a_subset_level(tmp_path):
    p = tmp_path / "emb.npz"
    sid = support_id(**BASE)
    np.savez(p, a=np.arange(3), support_id=stamp_array(sid))
    from support_contract import ROW_SELECTION_FIELDS
    with pytest.raises(SupportContractError):
        assert_same_support({"a": read_stamp(str(p)), "b": sid},
                            fields=ROW_SELECTION_FIELDS)


def test_plane_level_stamp_catches_defect_7_inside_one_pack(tmp_path):
    """A pack whose truth plane is at collar 3000 and data plane at 3300."""
    p = tmp_path / "pack.npz"
    np.savez(p, a=np.arange(3))
    data = {**BASE, "collar_kms": 3300.0, "truth_host_floor": NO_TRUTH_SIDE}
    truth_bad = {**BASE, "collar_kms": 3000.0}
    stamp(str(p), support_id(**data), extra=dict(planes=dict(
        counts=data, dX=data, truth_counts=truth_bad)))
    with pytest.raises(SupportMismatch) as exc:
        check_support_consistency(str(p))
    msg = str(exc.value)
    assert "pack.counts" in msg and "pack.truth_counts" in msg
    assert "collar_kms" in msg
    # the A0 repair (truth rebuilt at 3300) makes the same gate pass
    truth_ok = {**BASE, "collar_kms": 3300.0}
    stamp(str(p), support_id(**data), extra=dict(planes=dict(
        counts=data, dX=data, truth_counts=truth_ok)))
    from support_contract import ROW_SELECTION_FIELDS
    rec = check_support_consistency(str(p), fields=ROW_SELECTION_FIELDS)
    assert rec["status"] == "PASS"
    assert set(rec["planes"]) == {"pack.counts", "pack.dX", "pack.truth_counts"}
