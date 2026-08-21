#!/usr/bin/env python
"""certify_g_support.py — certification of the g(N,z) support correction
(finding N1 2026-08-20; PI rulings 2026-08-21 items 1-4, 10-11: CP-1
acceptance (a)-(d) + the permanent per-z gate). jax-free.

Checks (all fail-closed; the record is the JSON this writes):
  (a) COUNTING IDENTITY on the calibration pack (2LPT-0): per fine-z cell,
      g_occupancy summed over the molly rows >= `floor` equals truth_counts
      summed over the basis rows >= `floor` (the fold's S2N_RED > snr_min
      support) to within `tol_rows` rows. The deployed surface failed this by
      +505..+13,099 rows per cell; the documented padded-bundle vs
      detection-bundle truth-match difference is <= 1 row per cell.
  (b) NON-g BYTE IDENTITY: every array of each regenerated pack other than
      g_grid / g_occupancy is np.array_equal to its predecessor.
  (c) g BLOCK IDENTITY: the regenerated g_grid / g_occupancy equal the
      Battery-3-validated DIAGNOSTIC arrays (so Battery 3 is the validation of
      the production packs), and are identical across every pack (one frozen
      surface).
  (d) PER-Z GATE: perz_gate v2 PASS on the validation JSONs run on the
      regenerated production packs (when given).

Usage:
  python -m CDDF_analysis.hbi_mcmc.certify_g_support \
      --new PACK.npz [...] --old OLDPACK.npz [...] [--diag DIAGPACK.npz [...]]
      [--calib-index 0] [--floor 19.0] [--tol-rows 2]
      [--validation-json J.json [...]] --out CERT.json
Exit 0 = every check passed; 1 = a check failed (the JSON still records it).
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

G_KEYS = ("g_grid", "g_occupancy")


def g_support_identity(g_occupancy, molly_nhi_edges, truth_counts,
                       ntrue_edges, floor=19.0, tol_rows=2):
    """Per fine-z counting identity between g's denominator and the fold's
    truth support, on the rows both grids cover (>= floor)."""
    g = np.asarray(g_occupancy, float)
    tc = np.asarray(truth_counts, float)
    me = np.asarray(molly_nhi_edges, float)[:-1]
    ne = np.asarray(ntrue_edges, float)[:-1]
    gz = g[me >= floor - 1e-9].sum(axis=0)
    tz = tc[ne >= floor - 1e-9].sum(axis=0)
    d = gz - tz
    return dict(floor=float(floor), tol_rows=int(tol_rows),
                per_z_g=[float(x) for x in gz], per_z_truth=[float(x) for x in tz],
                per_z_diff=[float(x) for x in d],
                max_abs_diff=float(np.max(np.abs(d))),
                total_g=float(gz.sum()), total_truth=float(tz.sum()),
                passed=bool(np.max(np.abs(d)) <= tol_rows))


def array_identity(a, b, skip=()):
    """Keys whose arrays differ (or are missing on either side), minus `skip`."""
    keys = sorted(set(a.keys()) | set(b.keys()))
    out = []
    for k in keys:
        if k in skip:
            continue
        if k not in a or k not in b or not np.array_equal(a[k], b[k]):
            out.append(k)
    return out


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--old", nargs="+", default=[])
    ap.add_argument("--diag", nargs="+", default=[])
    ap.add_argument("--calib-index", type=int, default=0,
                    help="index into --new of the calibration (2LPT-0) pack")
    ap.add_argument("--floor", type=float, default=19.0)
    ap.add_argument("--tol-rows", type=int, default=2)
    ap.add_argument("--validation-json", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rec = dict(role="CERTIFICATION RECORD: g(N,z) support correction (CP-1 (a)-(d))",
               checks={}, inputs=dict(new=a.new, old=a.old, diag=a.diag,
                                      validation_json=a.validation_json))
    new = [_load(p) for p in a.new]
    ok_all = True

    # (a) counting identity on the calibration pack
    cal = new[a.calib_index]
    ident = g_support_identity(cal["g_occupancy"], cal["molly_nhi_edges"],
                               cal["truth_counts"], cal["ntrue_edges"],
                               floor=a.floor, tol_rows=a.tol_rows)
    ident["pack"] = a.new[a.calib_index]
    rec["checks"]["a_counting_identity"] = ident
    ok_all &= ident["passed"]

    # (b) non-g byte identity vs predecessors
    if a.old:
        assert len(a.old) == len(a.new), "--old must pair with --new"
        b = []
        for i, (pn, po) in enumerate(zip(a.new, a.old)):
            old_arrays = _load(po)
            diff = array_identity(old_arrays, new[i], skip=G_KEYS)
            g_changed = array_identity(old_arrays, new[i])
            b.append(dict(new=pn, old=po, non_g_changed=diff,
                          g_changed=[k for k in g_changed if k in G_KEYS],
                          passed=(diff == [])))
        rec["checks"]["b_non_g_byte_identity"] = b
        ok_all &= all(x["passed"] for x in b)

    # (c) g block identity vs the validated diagnostic arrays + across packs
    c = dict(across_packs=[], vs_diag=[])
    for i, pn in enumerate(a.new):
        same = all(np.array_equal(new[i][k], cal[k]) for k in G_KEYS)
        c["across_packs"].append(dict(pack=pn, identical_to_calib=bool(same)))
    if a.diag:
        for pd in a.diag:
            d = _load(pd)
            same = all(np.array_equal(d[k], cal[k]) for k in G_KEYS)
            c["vs_diag"].append(dict(diag=pd, identical=bool(same)))
    c["passed"] = (all(x["identical_to_calib"] for x in c["across_packs"])
                   and all(x["identical"] for x in c["vs_diag"]))
    rec["checks"]["c_g_block_identity"] = c
    ok_all &= c["passed"]

    # (d) per-z gate on the production-pack validation runs
    if a.validation_json:
        sys.path.insert(0, os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..")))
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "perz_gate_mod", os.path.join(os.path.dirname(__file__), "perz_gate.py"))
        PG = ilu.module_from_spec(spec)
        spec.loader.exec_module(PG)
        runs = [PG.gate_one(json.load(open(p))) | {"file": os.path.basename(p)}
                for p in a.validation_json]
        packs_ok = [r["pack"] in a.new for r in runs]
        d = dict(criteria=PG.CRIT, runs=runs,
                 all_runs_on_new_packs=bool(all(packs_ok)),
                 passed=bool(all(r["status"] == "PASS" for r in runs)
                             and all(packs_ok)))
        rec["checks"]["d_perz_gate"] = d
        ok_all &= d["passed"]

    rec["passed"] = bool(ok_all)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    for k, v in rec["checks"].items():
        st = v["passed"] if isinstance(v, dict) else all(x["passed"] for x in v)
        print(f"{'PASS' if st else 'FAIL'} {k}")
    print("CERTIFICATION", "PASS" if ok_all else "FAIL", "->", a.out)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
