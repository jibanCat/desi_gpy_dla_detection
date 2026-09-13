#!/usr/bin/env python
"""verify_a0_products.py — acceptance checks for the A0 support products.

VALIDATION-ONLY.  NO SAMPLER IS RUN: the ladder runner is only imported (dry),
never executed.  Nothing under ``CDDF_analysis/`` is modified.

Checks
------
  1. ``CDDF_analysis.hbi_mcmc.pack.load_pack`` accepts every A0 pack;
  2. whether an extra ``support_id`` NPZ key is admissible (it is NOT — the
     schema is a closed contract — which is WHY the A0 packs carry their
     support in the ``.support.json`` sidecar only);
  3. ``validation/fp_ladder/run_ladder.py`` imports and its own MOCK gate +
     tensor build accept the A0 pack (``build_cc_tensors`` + the census shape
     check the ORACLE arm performs), with no sampling;
  4. the committed truth reduction on the A0 pack vs the pack of record:
     ``truth_f`` -> ``reduce_f_posterior`` -> dN/dX(>=20.0) and (>=20.3), i.e.
     the MOCK ESTIMAND the A0 repair moves;
  5. the support invariant on the pairs the ladder actually forms.

ENV: ``gpdla-hbi`` (jax; ``run_ladder`` imports numpyro).

    python validation/absorber_ladder/support/verify_a0_products.py \
        --dir /scratch/.../absorber_ladder_2026-09-13/support
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

from support_contract import (                                   # noqa: E402
    ROW_SELECTION_FIELDS, SUPPORT_FIELDS, assert_same_support, read_stamp,
    stamp_array, stamp_path, support_id, SupportContractError,
)

FAMILIES = ("2lpt0", "london0", "saclay0")
PACK_OF_RECORD_DIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                      "real_pack_v2_20260821")
CENSUS_OF_RECORD_DIR = ("/scratch/cavestru_root/cavestru0/mfho/"
                        "fp_ladder_2026-09-12/census")


def check_load_pack(a0_pack: str) -> dict:
    from CDDF_analysis.hbi_mcmc.pack import load_pack, PackSchemaError
    out = {}
    pk = load_pack(a0_pack)
    out["load_pack_accepts_A0_pack"] = True
    out["truth_counts_total"] = float(np.asarray(pk.truth_counts).sum())
    out["truth_counts_bks_total"] = float(np.asarray(pk.truth_counts_bks).sum())
    out["counts_total"] = int(np.asarray(pk.counts).sum())
    out["dX_total"] = float(np.asarray(pk.dX).sum())
    out["truth_counts_is_bks_marginal"] = bool(np.array_equal(
        np.asarray(pk.truth_counts, float),
        np.asarray(pk.truth_counts_bks, float).sum(axis=2)))
    out["sidecar_provenance_attached"] = pk.provenance is not None
    # 2. would an embedded support_id key be accepted?
    tmp = tempfile.mkdtemp(prefix="a0_supportkey_")
    try:
        probe = os.path.join(tmp, os.path.basename(a0_pack))
        raw = dict(np.load(a0_pack, allow_pickle=False))
        sid = read_stamp(a0_pack)
        raw["support_id"] = stamp_array(
            support_id(**{k: sid.fields[k] for k in SUPPORT_FIELDS}))
        np.savez_compressed(probe, **raw)
        try:
            load_pack(probe)
            out["embedded_support_id_key_accepted"] = True
            out["embedded_support_id_error"] = None
        except PackSchemaError as exc:
            out["embedded_support_id_key_accepted"] = False
            out["embedded_support_id_error"] = str(exc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def check_runner_dry(a0_pack: str, a0_census: str) -> dict:
    """Import the ladder runner and exercise everything it does BEFORE MCMC."""
    sys.path.insert(0, os.path.join(_REPO, "validation", "fp_ladder"))
    import importlib
    out = {}
    mod = importlib.import_module("run_ladder")
    out["run_ladder_imported"] = True
    out["run_ladder_file"] = mod.__file__
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
    pk = load_pack(a0_pack)
    tc = np.asarray(pk.truth_counts)
    out["mock_gate_passes"] = bool(tc.size > 0 and tc.sum() > 0)
    consts, Mg = build_cc_tensors(pk)
    out["build_cc_tensors_ok"] = True
    out["Mg_shape"] = list(np.asarray(Mg).shape)
    cz = np.load(a0_census, allow_pickle=True)
    mu = np.asarray(cz["hostless"], float)
    out["ORACLE_census_shape_matches_counts"] = bool(
        mu.shape == tuple(np.asarray(pk.counts).shape))
    out["ORACLE_mu_FP_total"] = float(mu.sum())
    out["extra_fixed_host_17p2_19p0_total"] = float(
        np.asarray(cz["host_17p2_19p0"], float).sum())
    return out


def check_truth_estimand(pack_path: str, label: str) -> dict:
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior
    from CDDF_analysis.hbi_mcmc.forward_selftest import truth_f
    pk = load_pack(pack_path)
    ft = np.asarray(truth_f(pk), float)
    red = reduce_f_posterior(ft[None, :, :], pk)
    return {label: dict(
        pack=pack_path,
        truth_counts_total=float(np.asarray(pk.truth_counts).sum()),
        dX_total=float(np.asarray(pk.dX).sum()),
        truth_dndx_ge20p0=float(np.asarray(red["dndx_dla_20p0_allz"])[0]),
        truth_dndx_ge20p3=float(np.asarray(red["dndx_dla_20p3_allz"])[0]))}


def check_invariant(a0_pack: str, a0_census: str) -> dict:
    """The support invariant on the pairs the ladder actually forms."""
    def planes(path, label):
        with open(stamp_path(path)) as fh:
            rec = json.load(fh)
        return {f"{label}.{n}": support_id(**{k: blk[k] for k in SUPPORT_FIELDS})
                for n, blk in rec["extra"]["planes"].items()}

    pp, cp = planes(a0_pack, "pack"), planes(a0_census, "census")
    out = {}
    for name, objs, lvl in (
            ("pack data+truth planes (row)",
             pp, ROW_SELECTION_FIELDS),
            ("ORACLE pin: counts vs census.hostless (row)",
             {"pack.counts": pp["pack.counts"],
              "pack.truth_counts": pp["pack.truth_counts"],
              "census.hostless": cp["census.hostless"]}, ROW_SELECTION_FIELDS),
            ("A0 sub-floor term: counts vs census.host_17p2_19p0 (row)",
             {"pack.counts": pp["pack.counts"],
              "census.host_17p2_19p0": cp["census.host_17p2_19p0"]},
             ROW_SELECTION_FIELDS)):
        try:
            out[name] = dict(status="PASS",
                             support_id=assert_same_support(objs, fields=lvl)[:16])
        except SupportContractError as exc:
            out[name] = dict(status="FAIL", error=str(exc).splitlines()[0],
                             differs=[l.strip() for l in str(exc).splitlines()
                                      if l.strip().startswith("DIFFERS")])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="the A0 product directory")
    ap.add_argument("--family", nargs="+", default=list(FAMILIES))
    ap.add_argument("--out", default=None, help="write the record as JSON here")
    a = ap.parse_args(argv)
    rep = {}
    for fam in a.family:
        a0p = os.path.join(a.dir, f"scanpack_{fam}_b300_A0.npz")
        a0c = os.path.join(a.dir, f"fp_census_{fam}_A0.npz")
        r = dict(family=fam)
        r["load_pack"] = check_load_pack(a0p)
        r["runner_dry"] = check_runner_dry(a0p, a0c)
        r["estimand"] = {}
        r["estimand"].update(check_truth_estimand(
            os.path.join(PACK_OF_RECORD_DIR, f"scanpack_{fam}_b300.npz"),
            "of_record_collar3000_truth"))
        r["estimand"].update(check_truth_estimand(a0p, "A0_collar3300_truth"))
        o, n = (r["estimand"]["of_record_collar3000_truth"],
                r["estimand"]["A0_collar3300_truth"])
        r["estimand"]["ratio_A0_over_of_record"] = dict(
            ge20p0=round(n["truth_dndx_ge20p0"] / o["truth_dndx_ge20p0"], 6),
            ge20p3=round(n["truth_dndx_ge20p3"] / o["truth_dndx_ge20p3"], 6))
        r["invariant"] = check_invariant(a0p, a0c)
        rep[fam] = r
        print(f"\n===== {fam} =====")
        print(json.dumps(r, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=1, default=str)
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
