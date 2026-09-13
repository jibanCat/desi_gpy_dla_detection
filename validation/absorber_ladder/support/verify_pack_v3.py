#!/usr/bin/env python
"""verify_pack_v3.py — the FAIL-CLOSED verifier for the corrected mock packs v3.

PI ruling 2026-09-13c §8: *"Every corrected product carries a machine-readable
``support_id`` and fails closed on mismatch."*

WHAT IT CHECKS (every one of these can FAIL the run; exit code != 0):

 1. STAMPED           every v3 product carries a ``.support.json``; a missing
                      stamp is a failure, never a pass (``read_stamp`` is
                      fail-closed).
 2. SELF-CONSISTENT   the support_id is RE-DERIVED from the product's own 11/12
                      selection fields and must equal the recorded sha256
                      (``support_from_json`` refuses a drifted stamp).
 3. UNMODIFIED        the sha256 recorded in the stamp must equal the product's
                      sha256 on disk (a silently regenerated or truncated
                      product is caught here).
 4. ROW-LEVEL GATE    ``check_support_consistency(pack, census, ops,
                      fields=ROW_SELECTION_FIELDS)`` over all 18 stamped planes
                      — pack ``counts`` / ``dX`` / ``fp_E_alloc`` /
                      ``truth_counts`` / ``truth_counts_bks``, the 7 census
                      blocks and the 6 operator contract arrays must share ONE
                      row support_id.  ``truth_host_floor`` is reported
                      alongside, never hidden.
 5. SECOND INSTANCE   ``fp_counts_<fam>_v3.npz`` must carry a support_id that
                      DIFFERS from the survey one (it is a different selection
                      by nature) and must never enter the survey gate.
 6. MANIFEST          ``support/MANIFEST.json`` must agree with the files:
                      sha256 and support_id per product.
 7. A0 EQUIVALENCE    the v3 pack must equal the A0 pack BIT-EXACTLY on every
                      array except ``counts``, and ``counts`` must differ by
                      exactly the sentinel rows, all in SNR stratum s = 7; the
                      v3 census / operators must equal A0v2 / A0 bit-exactly.
 8. IMMUTABILITY      the packs of record, the adopted packs, the census and the
                      operators of record must be byte-identical to the sha256
                      recorded in the v3 provenance.
 9. CONSUMABILITY     (``gpdla-hbi`` only) ``pack.load_pack`` accepts the v3
                      pack and ``cc_posterior_validation.build_cc_tensors`` runs on it.

NO SAMPLER IS RUN.  Nothing is written except the verification JSON.

Usage
-----
    python validation/absorber_ladder/support/verify_pack_v3.py \
        --out /scratch/.../absorber_ladder_2026-09-13/support_v3
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_HERE, os.path.join(_REPO, "validation", "absorber_diag"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from support_contract import (                                   # noqa: E402
    ROW_SELECTION_FIELDS, SUPPORT_FIELDS, SupportContractError,
    check_support_consistency, file_sha256, read_stamp, stamp_path,
)

FAMILIES = ("2lpt0", "london0", "saclay0")
A0_DIR = ("/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/"
          "support")

#: the expected sentinel-filter effect, from the A0 report §5.1 (predeclared)
EXPECTED_SENTINEL_DELTA = {"2lpt0": -4, "london0": -5, "saclay0": -5}
EXPECTED_SENTINEL_STRATUM = 7


def _products(out_dir, fam):
    return dict(
        pack=os.path.join(out_dir, f"scanpack_{fam}_b300_v3.npz"),
        census=os.path.join(out_dir, f"fp_census_{fam}_v3.npz"),
        ops=os.path.join(out_dir, f"empirical_ops_{fam}_v3.npz"),
        fp_counts=os.path.join(out_dir, f"fp_counts_{fam}_v3.npz"))


# ---------------------------------------------------------------------------
def check_stamps(paths: dict) -> dict:
    """(1) stamped, (2) self-consistent, (3) unmodified — all fail-closed."""
    out = {}
    for name, p in paths.items():
        rec = dict(path=p)
        try:
            if not os.path.exists(p):
                raise SupportContractError(f"product absent: {p}")
            sid = read_stamp(p)                      # (1) + (2)
            rec["support_id"] = sid.sha256
            rec["row_support_id"] = sid.row_sha256
            sp = stamp_path(p)
            with open(sp) as fh:
                st = json.load(fh)
            disk = file_sha256(p)
            rec["product_sha256_on_disk"] = disk
            rec["product_sha256_in_stamp"] = st.get("product_sha256")
            rec["UNMODIFIED"] = bool(st.get("product_sha256") == disk)   # (3)
            if not rec["UNMODIFIED"]:
                rec["error"] = ("the product's sha256 differs from the one its "
                                "own stamp records — the product was modified "
                                "or regenerated after stamping")
            rec["PASS"] = rec["UNMODIFIED"]
        except Exception as exc:
            rec["PASS"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
        out[name] = rec
    return out


def check_support_gate(pack, census, ops) -> dict:
    """(4) the row-level fail-closed gate, + the full level for the record."""
    out = {}
    for label, lvl in (("row", ROW_SELECTION_FIELDS), ("full", SUPPORT_FIELDS)):
        try:
            out[label] = check_support_consistency(pack, census, ops,
                                                   fields=lvl)
        except SupportContractError as exc:
            out[label] = dict(status="FAIL", error=str(exc))
    out["PASS"] = out["row"].get("status") == "PASS"
    out["full_level_fails_only_on_truth_host_floor"] = bool(
        out["full"].get("status") == "FAIL"
        and "truth_host_floor" in out["full"].get("error", "")
        and sum(f"DIFFERS  {f}" in out["full"].get("error", "")
                for f in SUPPORT_FIELDS) == 1)
    return out


def check_fp_counts_second_instance(fp_path, survey_row_sha) -> dict:
    """(5) the loa-0 FP block is a SEPARATE contract instance."""
    rec = dict(path=fp_path)
    try:
        sid = read_stamp(fp_path)
        rec["support_id"] = sid.sha256
        rec["row_support_id"] = sid.row_sha256
        rec["differs_from_the_survey_row_support"] = bool(
            sid.row_sha256 != survey_row_sha)
        with open(stamp_path(fp_path)) as fh:
            st = json.load(fh)
        rec["disclosed_difference"] = (st.get("extra", {})
                                       .get("disclosed_difference"))
        rec["PASS"] = bool(rec["differs_from_the_survey_row_support"]
                           and rec["disclosed_difference"])
        if not rec["PASS"]:
            rec["error"] = ("fp_counts either shares the survey row support "
                            "(it must not) or carries no disclosure")
    except Exception as exc:
        rec["PASS"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def check_manifest(out_dir) -> dict:
    """(6) support/MANIFEST.json agrees with the products on disk."""
    p = os.path.join(out_dir, "support", "MANIFEST.json")
    rec = dict(path=p)
    try:
        with open(p) as fh:
            man = json.load(fh)
        bad = []
        for name, blk in man["products"].items():
            f = os.path.join(out_dir, name)
            if not os.path.exists(f):
                bad.append(f"{name}: absent")
                continue
            if file_sha256(f) != blk["sha256"]:
                bad.append(f"{name}: sha256 drift")
            if read_stamp(f).sha256 != blk["support_id"]:
                bad.append(f"{name}: support_id drift")
        rec["n_products"] = len(man["products"])
        rec["mismatches"] = bad
        rec["PASS"] = not bad
    except Exception as exc:
        rec["PASS"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def check_a0_equivalence(out_dir, fam, a0_dir=A0_DIR) -> dict:
    """(7) v3 == A0 bit-exactly except the sentinel filter on ``counts``."""
    rec = {}
    try:
        v3 = np.load(os.path.join(out_dir, f"scanpack_{fam}_b300_v3.npz"),
                     allow_pickle=False)
        a0 = np.load(os.path.join(a0_dir, f"scanpack_{fam}_b300_A0.npz"),
                     allow_pickle=False)
        diffk = sorted(k for k in a0.files
                       if not np.array_equal(np.asarray(v3[k]),
                                             np.asarray(a0[k])))
        d = (np.asarray(v3["counts"], np.int64)
             - np.asarray(a0["counts"], np.int64))
        per_s = [int(d[:, :, s].sum()) for s in range(d.shape[2])]
        rec["keys_differing_from_A0"] = diffk
        rec["only_counts_differs"] = bool(diffk == ["counts"])
        rec["counts_delta"] = int(d.sum())
        rec["counts_delta_expected"] = EXPECTED_SENTINEL_DELTA[fam]
        rec["n_cells_differing"] = int(np.count_nonzero(d))
        rec["max_abs_cell_difference"] = int(np.abs(d).max()) if d.any() else 0
        rec["delta_per_snr_stratum"] = per_s
        rec["all_in_top_snr_stratum"] = bool(
            all(v == 0 for i, v in enumerate(per_s)
                if i != EXPECTED_SENTINEL_STRATUM))
        rec["delta_matches_predeclared"] = bool(
            rec["counts_delta"] == rec["counts_delta_expected"])

        cv = np.load(os.path.join(out_dir, f"fp_census_{fam}_v3.npz"),
                     allow_pickle=True)
        ca = np.load(os.path.join(a0_dir, f"fp_census_{fam}_A0v2.npz"),
                     allow_pickle=True)
        rec["census_blocks_equal_A0v2"] = {
            k: bool(np.array_equal(np.asarray(cv[k], float),
                                   np.asarray(ca[k], float)))
            for k in ("counts_all", "hostless", "host_17p2_19p0",
                      "host_19p0_19p5", "host_19p5_19p7", "host_19p7_21p6",
                      "host_ge_21p6")}
        ov = np.load(os.path.join(out_dir, f"empirical_ops_{fam}_v3.npz"),
                     allow_pickle=True)
        oa = np.load(os.path.join(a0_dir, f"empirical_ops_{fam}_A0.npz"),
                     allow_pickle=True)
        rec["ops_contract_arrays_equal_A0"] = {
            k: bool(np.array_equal(np.asarray(ov[k], float),
                                   np.asarray(oa[k], float)))
            for k in ("C_true_bKs", "C_true_bs", "M_true_sKcb", "E_true_cKsb",
                      "N_match_cksb", "P6b_cks")}
        # the operators must divide by the pack's OWN truth plane
        rec["ops_truth_plane_is_the_pack_truth_plane"] = bool(np.array_equal(
            np.asarray(ov["truth_counts_bks"], float),
            np.asarray(v3["truth_counts_bks"], float)))
        rec["PASS"] = bool(
            rec["only_counts_differs"] and rec["delta_matches_predeclared"]
            and rec["all_in_top_snr_stratum"]
            and all(rec["census_blocks_equal_A0v2"].values())
            and all(rec["ops_contract_arrays_equal_A0"].values())
            and rec["ops_truth_plane_is_the_pack_truth_plane"])
    except Exception as exc:
        rec["PASS"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def check_immutability(out_dir, fam) -> dict:
    """(8) every input of record is byte-identical to the build-time sha256."""
    rec = {}
    try:
        with open(os.path.join(out_dir,
                               f"scanpack_{fam}_b300_v3.provenance.json")) as fh:
            prov = json.load(fh)
        bad = []
        for k, blk in prov["inputs"].items():
            if not os.path.exists(blk["path"]):
                bad.append(f"{k}: absent")
            elif file_sha256(blk["path"]) != blk["sha256"]:
                bad.append(f"{k}: CHANGED")
        rec["inputs_checked"] = sorted(prov["inputs"])
        rec["changed"] = bad
        rec["PASS"] = not bad
    except Exception as exc:
        rec["PASS"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def check_consumability(pack_path) -> dict:
    """(9) ``load_pack`` + ``build_cc_tensors`` (needs ``gpdla-hbi``)."""
    rec = dict(pack=pack_path)
    try:
        from CDDF_analysis.hbi_mcmc.pack import load_pack
        pk = load_pack(pack_path)
        rec["load_pack_accepts"] = True
        rec["counts_total"] = float(np.asarray(pk.counts).sum())
        rec["truth_counts_total"] = float(np.asarray(pk.truth_counts).sum())
        rec["truth_counts_equals_bks_sum"] = bool(np.allclose(
            np.asarray(pk.truth_counts),
            np.asarray(pk.truth_counts_bks).sum(axis=2), atol=0.0))
        from CDDF_analysis.hbi_mcmc.cc_posterior_validation import (
            build_cc_tensors)
        _consts, Mg = build_cc_tensors(pk)
        rec["build_cc_tensors_ok"] = True
        rec["Mg_shape"] = list(np.asarray(Mg).shape) if Mg is not None else None
        rec["PASS"] = bool(rec["load_pack_accepts"]
                           and rec["truth_counts_equals_bks_sum"]
                           and rec["build_cc_tensors_ok"])
    except ImportError as exc:
        rec["PASS"] = None                 # not available in this environment
        rec["skipped"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        rec["PASS"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


# ---------------------------------------------------------------------------
def verify(out_dir, families=FAMILIES, a0_dir=A0_DIR, consumability=True):
    """Run every check; return ``(record, ok)``. ``ok`` False => exit non-zero."""
    out_dir = os.path.abspath(out_dir)
    rec = dict(schema="pack_v3_verification/v1", directory=out_dir,
               checked_utc=datetime.datetime.utcnow().isoformat() + "Z",
               families={})
    ok = True
    for fam in families:
        p = _products(out_dir, fam)
        f = {}
        f["stamps"] = check_stamps(p)
        f["support_gate"] = check_support_gate(p["pack"], p["census"], p["ops"])
        survey_row = (f["support_gate"]["row"].get("support_id")
                      if f["support_gate"]["row"].get("status") == "PASS"
                      else None)
        f["fp_counts_second_instance"] = check_fp_counts_second_instance(
            p["fp_counts"], survey_row)
        f["a0_equivalence"] = check_a0_equivalence(out_dir, fam, a0_dir)
        f["immutability"] = check_immutability(out_dir, fam)
        if consumability:
            f["consumability"] = check_consumability(p["pack"])
        fails = []
        for k, v in f.items():
            if k == "stamps":
                fails += [f"stamps.{n}" for n, r in v.items()
                          if not r.get("PASS")]
            elif v.get("PASS") is False:
                fails.append(k)
        f["FAILURES"] = fails
        f["PASS"] = not fails
        ok &= f["PASS"]
        rec["families"][fam] = f
    rec["manifest"] = check_manifest(out_dir)
    ok &= bool(rec["manifest"]["PASS"])
    rec["PASS"] = bool(ok)
    return rec, bool(ok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", nargs="+", default=list(FAMILIES))
    ap.add_argument("--a0-dir", default=A0_DIR)
    ap.add_argument("--no-consumability", action="store_true")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)
    rec, ok = verify(a.out, a.family, a.a0_dir,
                     consumability=not a.no_consumability)
    dest = a.json_out or os.path.join(a.out, "V3_VERIFICATION.json")
    with open(dest, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    for fam, f in rec["families"].items():
        print(f"{fam}: {'PASS' if f['PASS'] else 'FAIL ' + str(f['FAILURES'])}"
              f"  row support_id="
              f"{f['support_gate']['row'].get('support_id_short')}"
              f"  counts delta={f['a0_equivalence'].get('counts_delta')}")
    print(f"manifest: {'PASS' if rec['manifest']['PASS'] else 'FAIL'}")
    print(f"OVERALL: {'PASS' if ok else 'FAIL'} -> {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
