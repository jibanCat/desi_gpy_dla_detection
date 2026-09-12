#!/usr/bin/env python
"""build_fp_census.py — VALIDATION-ONLY builder for the MOCK FALSE-POSITIVE
TRUTH CENSUS per (c, k, s) cell, for the three mock families 2LPT-0 /
London-0 / Saclay-0.

WHAT THIS IS.  The HBI packs label a detection ``is_TP`` against a truth table
that has ALREADY been floored at ``truth_nhi_floor = mm.nhi_edges[0] = 19.5``
(``cddf_catalog_hbi.py:576``).  A detection whose genuine absorber lies below
19.5 is therefore labelled NOT is_TP -- the contract's own
``TRUTH_FLOOR_ASYMMETRY_IN_is_TP`` contradiction.  Re-cutting the SAME
catalogue against the SAME op mask but with the truth table floored at 17.2
(the molly172 sub-floor) recovers most of those hosts, and what is STILL
hostless at 17.2 is the only defensible FP-like census on disk.

This script rebuilds, per family and per (c, k, s) cell:

  A. the pack's own 19.5-floor accounting  -- counts / counts_tp / unmatched,
     GATED against a pack on disk (exact elementwise equality, no tolerance);
  B. the 17.2-floor census -- counts_all, hostless, and the five host slots
     [17.2,19.0) / [19.0,19.5) / [19.5,19.7) / [19.7,21.6) / >=21.6.

The 17.2-floor ``hostless`` array is the census of record: it is
P4_FOREST_FP (+) P6_RESIDUAL sub-slot (c).  It is NOT a measurement of P4, and
NOT a ceiling on the forward FP term -- see README.md.

Recipe provenance: reproduces, line for line, the verified recipe of
OPUSH_REPORT.md sec.Q1.4 (2026-09-12), which itself reproduced every control
total in ``CDDF_analysis/hbi_mcmc/matching_contract.py``.  Every control total
is HARD-ASSERTED here (see ``CONTROLS``); the script writes nothing if any
assert fails.

ENV: ``gpdla`` (jax-free).  ``extract_pack.py`` is loaded FILE-DIRECTLY: the
``hbi_mcmc`` package ``__init__`` imports jax, which ``gpdla`` does not have.
``matching_contract.py`` cannot be imported at all under ``gpdla`` (its
top-level ``from CDDF_analysis.hbi_mcmc import reporting`` executes that same
__init__), so its constants are cross-read with ``ast`` instead -- no import,
no retyping.

NO SAMPLER IS RUN.  Nothing under ``CDDF_analysis/`` is modified.

Usage
-----
    python validation/fp_ladder/build_fp_census.py \
        --family 2lpt0 --out /path/to/census/ [--pack PACK.npz] [--work DIR]
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.util as ilu
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_EXTRACT_PACK = os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "extract_pack.py")
_CONTRACT = os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "matching_contract.py")

FAMILIES = ("2lpt0", "london0", "saclay0")

#: The pack of record (collar-scan variants, b = 300 km/s).  This is the
#: DEFAULT gate target because it is the pack the posterior runs consume.
PACK_OF_RECORD_DIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                      "real_pack_v2_20260821")


def default_pack(family: str) -> str:
    return os.path.join(PACK_OF_RECORD_DIR, f"scanpack_{family}_b300.npz")


# ---------------------------------------------------------------------------
# grid / slot helpers  (PURE -- no heavy imports; unit-tested in
# tests/test_fp_census_grid.py, and cross-checked against extract_pack._idx on
# the real arrays at run time by ``_assert_index_convention``)
# ---------------------------------------------------------------------------

#: Half-open slot comparisons carry this tolerance, EXACTLY as the verified
#: recipe ran it: ``(n >= lo - EPS) & (n < hi - EPS)``.  Note the sign: a value
#: within EPS BELOW an edge is placed in the UPPER slot.  The slots stay
#: mutually exclusive and (over finite n >= 17.2 - EPS) exhaustive.
SLOT_EPS = 1e-9

#: True-host N_HI slots, in declaration order.  Contract classes:
#:   [17.2, 19.0)  P6_RESIDUAL sub-slot (b) -- below the basis floor, NO support
#:   [19.0, 19.5)  P1_SCATTER_IN, invisible to the pack's 19.5-floored truth
#:   [19.5, 19.7)  P1_SCATTER_IN
#:   [19.7, 21.6)  P2_IN_WINDOW -- the reported estimand
#:   >= 21.6       P6_RESIDUAL sub-slot (a) -- scatter-down from above ceiling
HOST_SLOTS = (
    ("host_17p2_19p0", 17.2, 19.0),
    ("host_19p0_19p5", 19.0, 19.5),
    ("host_19p5_19p7", 19.5, 19.7),
    ("host_19p7_21p6", 19.7, 21.6),
    ("host_ge_21p6", 21.6, np.inf),
)
HOST_SLOT_NAMES = tuple(n for n, _, _ in HOST_SLOTS)

#: The census truth floor, and the pack's own truth floor.
CENSUS_TRUTH_FLOOR = 17.2
PACK_TRUTH_FLOOR = 19.5


def bin_index(edges, x):
    """The pack's grid index convention: half-open [lo, hi), right-searchsorted.

    Identical to ``extract_pack._idx``; duplicated here ONLY so the convention
    is unit-testable without importing the data-plane stack.  The builder
    asserts the two agree on the actual arrays before using either.
    """
    return np.searchsorted(np.asarray(edges, float),
                           np.asarray(x, float), side="right") - 1


def host_slot_masks(nhi_true):
    """Split candidates by their matched host's true N_HI.

    Returns an ordered dict with ``hostless`` (NHI_TRUE not finite -- no host
    even at truth floor 17.2) followed by the five ``HOST_SLOTS``.  The masks
    partition the input exactly when every finite NHI_TRUE is >= 17.2 - EPS;
    the builder asserts that partition.
    """
    n = np.asarray(nhi_true, float)
    host = np.isfinite(n)
    out = {"hostless": ~host}
    for name, lo, hi in HOST_SLOTS:
        if np.isinf(hi):
            out[name] = host & (n >= lo - SLOT_EPS)
        else:
            out[name] = host & (n >= lo - SLOT_EPS) & (n < hi - SLOT_EPS)
    return out


# ---------------------------------------------------------------------------
# control totals -- HARD ASSERTS
# ---------------------------------------------------------------------------
#: Sources:
#:  * ``matching_contract.MATCHING["measured_2lpt0_19p5_floor_bundle"]``
#:    (matching_contract.py:339-344) -- the 19.5-floor 2LPT-0 row;
#:  * ``matching_contract.FP_CEILING_MEASURED`` (matching_contract.py:1653-1690)
#:    -- the 17.2-floor n_on_grid / unmatched / P1 / P2 / P6 rows, all three
#:    families;
#:  * OPUSH_REPORT.md sec.Q1.4 (2026-09-12) -- the London-0 / Saclay-0 19.5-floor
#:    totals and the finer [19.0,19.5) / [19.5,19.7) split of the contract's
#:    combined P1_true_19p0_to_19p7.
#: The first two groups are additionally cross-read out of matching_contract.py
#: with ``ast`` (``_contract_constants_via_ast``) so these literals cannot drift
#: away from the contract silently.
CONTROLS = {
    "2lpt0": dict(
        n_cat_cut=582855, n_op=495553,                     # contract
        n_on_pack_grid=88071, n_is_TP=63890, n_unmatched=24181,   # contract
        n_on_grid_172=88053, hostless_172=13860,           # contract
        host_17p2_19p0=3200,                               # contract P6_below_19p0
        host_19p0_19p5=7106, host_19p5_19p7=8332,          # OPUSH Q1.4 split
        host_19p0_19p7=15438,                              # contract P1
        host_19p7_21p6=55058,                              # contract P2
        host_ge_21p6=497,                                  # contract P6_above
    ),
    "london0": dict(
        n_on_pack_grid=87840, n_is_TP=68643, n_unmatched=19197,   # OPUSH Q1.4
        n_on_grid_172=87831, hostless_172=9598,
        host_17p2_19p0=2611,
        host_19p0_19p5=6983, host_19p5_19p7=8851,
        host_19p0_19p7=15834,
        host_19p7_21p6=59186,
        host_ge_21p6=602,
    ),
    "saclay0": dict(
        n_on_pack_grid=86763, n_is_TP=66538, n_unmatched=20225,   # OPUSH Q1.4
        n_on_grid_172=86745, hostless_172=10592,
        host_17p2_19p0=2668,
        host_19p0_19p5=6949, host_19p5_19p7=8784,
        host_19p0_19p7=15733,
        host_19p7_21p6=57213,
        host_ge_21p6=539,
    ),
}

#: control key -> (contract container, contract field)
_CONTRACT_XREF = {
    "n_cat_cut": ("MATCHING_2LPT0", "n_cat_cut"),
    "n_op": ("MATCHING_2LPT0", "n_op"),
    "n_on_pack_grid": ("MATCHING_2LPT0", "n_on_pack_grid"),
    "n_is_TP": ("MATCHING_2LPT0", "n_is_TP"),
    "n_unmatched": ("MATCHING_2LPT0", "n_unmatched"),
    "n_on_grid_172": ("FP_CEILING", "n_on_grid"),
    "hostless_172": ("FP_CEILING", "unmatched"),
    "host_19p0_19p7": ("FP_CEILING", "P1_true_19p0_to_19p7"),
    "host_19p7_21p6": ("FP_CEILING", "P2_true_19p7_to_21p6"),
    "host_ge_21p6": ("FP_CEILING", "P6_true_above_21p6"),
    "host_17p2_19p0": ("FP_CEILING", "P6_true_below_19p0"),
}


def _dict_call_ints(node):
    """{keyword: int} from an ast ``dict(...)`` call or ``{...}`` literal."""
    out = {}
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
        items = [(kw.arg, kw.value) for kw in node.keywords]
    elif isinstance(node, ast.Dict):
        items = [(k.value if isinstance(k, ast.Constant) else None, v)
                 for k, v in zip(node.keys, node.values)]
    else:
        return out
    for key, val in items:
        if key is not None and isinstance(val, ast.Constant) and isinstance(val.value, int):
            out[key] = val.value
    return out


def _contract_constants_via_ast(path=_CONTRACT):
    """Read the contract's control integers WITHOUT importing it.

    ``matching_contract`` is un-importable under ``gpdla`` (top-level
    ``from CDDF_analysis.hbi_mcmc import reporting`` -> package __init__ ->
    jax).  Parsing is the only jax-free way to hold the literals above
    accountable to the contract file.
    """
    tree = ast.parse(open(path).read(), filename=path)
    got = {"MATCHING_2LPT0": {}, "FP_CEILING": {}}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "MATCHING" in names:
            # the block is nested (MATCHING -> is_TP -> ...), so walk
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.keyword) and \
                        sub.arg == "measured_2lpt0_19p5_floor_bundle":
                    got["MATCHING_2LPT0"] = _dict_call_ints(sub.value)
        if "FP_CEILING_MEASURED" in names and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant):
                    got["FP_CEILING"][k.value] = _dict_call_ints(v)
    return got


def assert_controls_match_contract(family):
    """HARD ASSERT: the literals in ``CONTROLS`` still equal the contract's."""
    got = _contract_constants_via_ast()
    checked = {}
    for key, (container, field) in _CONTRACT_XREF.items():
        if key not in CONTROLS[family]:
            continue
        if container == "MATCHING_2LPT0":
            if family != "2lpt0":
                continue
            src = got["MATCHING_2LPT0"]
        else:
            src = got["FP_CEILING"].get(family, {})
        if field not in src:
            raise AssertionError(
                f"CONTRACT X-REF FAILED: {container}[{field!r}] not found in "
                f"{_CONTRACT} -- the contract moved; re-verify CONTROLS by hand.")
        if int(src[field]) != int(CONTROLS[family][key]):
            raise AssertionError(
                f"CONTRACT X-REF FAILED for {family} {key}: CONTROLS says "
                f"{CONTROLS[family][key]}, matching_contract.py says {src[field]}.")
        checked[key] = int(src[field])
    return checked


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
def _load_ep():
    """Load extract_pack.py FILE-DIRECTLY (never via the jax-importing package)."""
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    spec = ilu.spec_from_file_location("_fpladder_ep", _EXTRACT_PACK)
    mod = ilu.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo=_REPO):
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                       stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
            stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo,
            stderr=subprocess.DEVNULL).decode().strip()
        return dict(commit=head, dirty=bool(dirty), branch=branch)
    except Exception as exc:                                   # pragma: no cover
        return dict(commit="unknown", dirty=None, branch=None, error=str(exc))


def _assert_index_convention(ep, nhat, zhat, snr):
    """The local pure helper MUST agree with the committed ``extract_pack._idx``
    on the actual arrays -- otherwise the unit test guards nothing."""
    for edges, x, label in ((ep.NHAT_EDGES, nhat, "NHAT"),
                            (ep.ZF_EDGES, zhat, "ZF"),
                            (ep.SNR_EDGES, snr, "SNR")):
        a = bin_index(edges, x)
        b = ep._idx(np.asarray(edges, float), np.asarray(x, float))
        if not np.array_equal(a, b):
            raise AssertionError(
                f"INDEX CONVENTION DRIFT on {label}: validation/fp_ladder's "
                "bin_index disagrees with extract_pack._idx.")


# ---------------------------------------------------------------------------
# the two passes
# ---------------------------------------------------------------------------
def pass_19p5(ep, family, work_dir):
    """The pack's own bundle: truth floor 19.5, is_TP as the pack sees it."""
    t0 = time.time()
    b = ep.load_mock_bundle(family, work_dir)
    cat, op = b["cat_cut"], b["op_mask"]
    nhat = np.asarray(cat["NHI"], float)[op]
    zhat = np.asarray(cat["Z_DLA"], float)[op]
    snr = np.asarray(cat["S2N_RED"], float)[op]
    ntr = np.asarray(cat["NHI_TRUE"], float)[op]
    _assert_index_convention(ep, nhat, zhat, snr)
    is_TP = np.isfinite(ntr)
    counts, _ = ep.bin_counts_cks(nhat, zhat, snr)
    counts_tp, _ = ep.bin_counts_cks(nhat[is_TP], zhat[is_TP], snr[is_TP])
    min_true = float(np.nanmin(ntr[is_TP])) if is_TP.any() else float("nan")
    return dict(
        counts=counts, counts_tp=counts_tp, counts_unmatched=counts - counts_tp,
        n_cat_cut=int(len(cat)), n_op=int(op.sum()),
        n_is_TP_all_op=int(is_TP.sum()), min_nhi_true=min_true,
        truth_floor=float(b["mm"].nhi_edges[0]), seconds=time.time() - t0,
        cfg=b["cfg"])


def pass_17p2(ep, family, work_dir, floor=CENSUS_TRUTH_FLOOR):
    """The census: SAME catalogue, SAME op mask, truth table floored at 17.2."""
    from CDDF_analysis.hbi.cddf_catalog_hbi import (
        load_and_cut_catalog, load_molly_matrix, _build_qso_lookup)
    from CDDF_analysis.hbi import track_c_tf_saclay as TS

    t0 = time.time()
    cfg = ep._make_cfg(family, work_dir)
    mm = load_molly_matrix(cfg.molly_tsv)
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, _is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=floor, qso_lookup=qso_lookup,
        host_truth_floor=floor)
    # committed interior-edge tie-break (saclay0 has one NHI_TRUE == 20.0 row;
    # NO-OP on 2lpt0 / london0)
    TS._snap_off_molly_edges(cat_cut, truth_cut, mm)
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask

    nhat = np.asarray(cat_cut["NHI"], float)[op]
    zhat = np.asarray(cat_cut["Z_DLA"], float)[op]
    snr = np.asarray(cat_cut["S2N_RED"], float)[op]
    ntr = np.asarray(cat_cut["NHI_TRUE"], float)[op]
    _assert_index_convention(ep, nhat, zhat, snr)

    masks = host_slot_masks(ntr)
    # partition guard: the slots must tile the op rows exactly
    stacked = np.vstack([masks[k] for k in ("hostless",) + HOST_SLOT_NAMES])
    per_row = stacked.sum(axis=0)
    if not np.all(per_row == 1):
        bad = int(np.sum(per_row != 1))
        raise AssertionError(
            f"SLOT PARTITION FAILED on {family}: {bad} of {len(ntr)} op rows are "
            "in 0 or >1 slots (a finite NHI_TRUE below the census floor?). "
            f"min finite NHI_TRUE = {np.nanmin(ntr):.6f}")

    counts_all, n_in = ep.bin_counts_cks(nhat, zhat, snr)
    blocks = {}
    for name in ("hostless",) + HOST_SLOT_NAMES:
        m = masks[name]
        blocks[name], _ = ep.bin_counts_cks(nhat[m], zhat[m], snr[m])
    return dict(counts_all=counts_all, n_in_window=int(n_in),
                n_cat_cut=int(len(cat_cut)), n_op=int(op.sum()),
                min_nhi_true=float(np.nanmin(ntr)) if np.isfinite(ntr).any() else float("nan"),
                truth_floor=float(floor), seconds=time.time() - t0,
                cfg=cfg, meta=meta, **blocks)


# ---------------------------------------------------------------------------
# the pack gate
# ---------------------------------------------------------------------------
def pack_gate(counts, pack_path, label):
    """Elementwise equality against a pack's ``counts``.  NO tolerance."""
    z = np.load(pack_path, allow_pickle=True)
    pc = np.asarray(z["counts"], np.int64)
    if pc.shape != counts.shape:
        raise AssertionError(f"PACK GATE shape mismatch {pc.shape} vs {counts.shape}")
    diff = counts.astype(np.int64) - pc
    rec = dict(label=label, pack=pack_path, pack_sha256=_sha256(pack_path),
               pack_counts_total=int(pc.sum()), bundle_counts_total=int(counts.sum()),
               total_difference=int(diff.sum()),
               n_cells_differing=int(np.count_nonzero(diff)),
               max_abs_cell_difference=int(np.abs(diff).max()),
               EQUAL=bool(np.array_equal(counts.astype(np.int64), pc)))
    if not rec["EQUAL"]:
        rec["difference_per_coarse_z_K"] = [
            int(diff[:, k * 5:(k + 1) * 5, :].sum()) for k in range(3)]
        rec["difference_per_snr_s"] = [int(diff[:, :, s].sum()) for s in range(diff.shape[2])]
    return rec


def _stop_on_gate(rec, pack_path):
    print("\n" + "=" * 78, file=sys.stderr)
    print("PACK GATE FAILED -- STOPPING, NOTHING WRITTEN.", file=sys.stderr)
    print(json.dumps(rec, indent=1), file=sys.stderr)
    prov = pack_path[:-4] + ".provenance.json"
    if os.path.exists(prov):
        try:
            p = json.load(open(prov))
            print("\npack provenance recipe: %s" % p.get("recipe"), file=sys.stderr)
            print("pack provenance role  : %s" % p.get("role"), file=sys.stderr)
            print("pack provenance src   : %s" % p.get("this", {}).get("src"),
                  file=sys.stderr)
        except Exception:
            pass
    print("=" * 78 + "\n", file=sys.stderr)
    raise SystemExit(3)


# ---------------------------------------------------------------------------
def build(family, out_dir, pack_path=None, work_dir=None):
    t_start = time.time()
    pack_path = pack_path or default_pack(family)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    work_dir = work_dir or os.path.join(out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)

    xref = assert_controls_match_contract(family)
    print(f"[{family}] contract x-ref OK ({len(xref)} constants read from "
          f"matching_contract.py by ast)")

    ep = _load_ep()
    ctl = CONTROLS[family]
    # the coarse-z blocking used by every per-K rollup below is the pack's own
    if not np.array_equal(np.asarray(ep.KZ_TO_K), np.repeat([0, 1, 2], 5)):
        raise AssertionError(f"KZ_TO_K changed: {np.asarray(ep.KZ_TO_K)!r} -- "
                             "the k*5 coarse-z slicing in this script is stale.")

    # ---- A. 19.5-floor pass + pack gate -----------------------------------
    a = pass_19p5(ep, family, work_dir)
    gate = pack_gate(a["counts"], pack_path, "gate")
    print(f"[{family}] 19.5-floor: n_cat_cut={a['n_cat_cut']} n_op={a['n_op']} "
          f"on_grid={int(a['counts'].sum())} is_TP={int(a['counts_tp'].sum())} "
          f"unmatched={int(a['counts_unmatched'].sum())} ({a['seconds']:.1f}s)")
    # informational: always record the delta against the pack of record too
    gates = [gate]
    if os.path.abspath(pack_path) != os.path.abspath(default_pack(family)):
        gates.append(pack_gate(a["counts"], default_pack(family),
                               "pack_of_record_informational"))
    if not gate["EQUAL"]:
        _stop_on_gate(gate, pack_path)
    print(f"[{family}] PACK GATE PASSED against {pack_path}")

    assert a["truth_floor"] == PACK_TRUTH_FLOOR, a["truth_floor"]
    assert int(a["counts"].sum()) == ctl["n_on_pack_grid"], (
        f"19.5 control n_on_pack_grid: {int(a['counts'].sum())} != {ctl['n_on_pack_grid']}")
    assert int(a["counts_tp"].sum()) == ctl["n_is_TP"], (
        f"19.5 control n_is_TP: {int(a['counts_tp'].sum())} != {ctl['n_is_TP']}")
    assert int(a["counts_unmatched"].sum()) == ctl["n_unmatched"], (
        f"19.5 control n_unmatched: {int(a['counts_unmatched'].sum())} != {ctl['n_unmatched']}")
    if "n_cat_cut" in ctl:
        assert a["n_cat_cut"] == ctl["n_cat_cut"], (a["n_cat_cut"], ctl["n_cat_cut"])
        assert a["n_op"] == ctl["n_op"], (a["n_op"], ctl["n_op"])
    assert a["min_nhi_true"] >= PACK_TRUTH_FLOOR - SLOT_EPS, (
        f"is_TP row below the 19.5 truth floor: {a['min_nhi_true']}")
    print(f"[{family}] 19.5-floor controls PASSED")

    # ---- B. 17.2-floor census ---------------------------------------------
    b = pass_17p2(ep, family, work_dir)
    tot = {k: int(b[k].sum()) for k in ("counts_all",) + ("hostless",) + HOST_SLOT_NAMES}
    print(f"[{family}] 17.2-floor: " + " ".join(f"{k}={v}" for k, v in tot.items())
          + f" ({b['seconds']:.1f}s)")

    assert tot["counts_all"] == ctl["n_on_grid_172"], (
        f"17.2 control n_on_grid: {tot['counts_all']} != {ctl['n_on_grid_172']}")
    assert tot["hostless"] == ctl["hostless_172"], (
        f"17.2 control hostless: {tot['hostless']} != {ctl['hostless_172']}")
    for key in ("host_17p2_19p0", "host_19p0_19p5", "host_19p5_19p7",
                "host_19p7_21p6", "host_ge_21p6"):
        assert tot[key] == ctl[key], f"17.2 control {key}: {tot[key]} != {ctl[key]}"
    assert tot["host_19p0_19p5"] + tot["host_19p5_19p7"] == ctl["host_19p0_19p7"], (
        "the fine [19.0,19.5)+[19.5,19.7) split must sum to the contract's "
        "P1_true_19p0_to_19p7")
    assert sum(tot[k] for k in ("hostless",) + HOST_SLOT_NAMES) == tot["counts_all"], (
        "17.2 slots do not sum to counts_all on the grid")
    print(f"[{family}] 17.2-floor controls PASSED")

    # ---- C. derived cross-checks (reported, not asserted) ------------------
    implied_19p5_unmatched = (tot["hostless"] + tot["host_17p2_19p0"]
                              + tot["host_19p0_19p5"])
    cat_cut_perturbation = int(a["counts"].sum()) - tot["counts_all"]

    # ---- D. write ----------------------------------------------------------
    prov = dict(
        role="MOCK false-positive TRUTH CENSUS per (c,k,s): 19.5-floor pack "
             "accounting + 17.2-floor hostless/host-slot decomposition",
        census_meaning="hostless @17.2 = P4_FOREST_FP (+) P6_RESIDUAL sub-slot (c); "
                       "NOT a measurement of P4 and NOT a ceiling on the forward "
                       "FP term -- see validation/fp_ladder/README.md",
        family=family,
        recipe="validation/fp_ladder/build_fp_census.py",
        recipe_source="OPUSH_REPORT.md sec.Q1.4 (2026-09-12), verified against "
                      "matching_contract.py control totals",
        code=_git_head(),
        extract_pack_path=_EXTRACT_PACK,
        extract_pack_sha256=_sha256(_EXTRACT_PACK),
        matching_contract_path=_CONTRACT,
        matching_contract_sha256=_sha256(_CONTRACT),
        catalog_dir=str(b["cfg"].catalog_dir),
        truth_path=str(b["cfg"].truth_path),
        bal_cat_path=str(b["cfg"].bal_cat_path),
        molly_tsv=str(b["cfg"].molly_tsv),
        mockdir=str(b["cfg"].mockdir),
        pack_gates=gates,
        pack_truth_floor=PACK_TRUTH_FLOOR,
        census_truth_floor=CENSUS_TRUTH_FLOOR,
        host_slots=[[n, lo, (None if np.isinf(hi) else hi)] for n, lo, hi in HOST_SLOTS],
        slot_eps=SLOT_EPS,
        op_cut="S2N_RED > snr_min AND P_DLA > p_dla_min AND good_mask "
               "(DLAFLAG==0, BAL veto, z_qso + lambda_rf window)",
        snr_min=float(b["cfg"].snr_min), p_dla_min=float(b["cfg"].p_dla_min),
        contract_xref=xref,
        controls=CONTROLS[family],
        env=dict(python=platform.python_version(), numpy=np.__version__,
                 conda_prefix=os.environ.get("CONDA_PREFIX"),
                 host=platform.node()),
        timestamps=dict(started_utc=datetime.datetime.utcfromtimestamp(t_start)
                        .isoformat() + "Z",
                        finished_utc=datetime.datetime.utcnow().isoformat() + "Z"),
        seconds=dict(pass_19p5=a["seconds"], pass_17p2=b["seconds"]),
    )
    totals = dict(
        family=family,
        floor_19p5=dict(n_cat_cut=a["n_cat_cut"], n_op=a["n_op"],
                        n_on_grid=int(a["counts"].sum()),
                        n_is_TP=int(a["counts_tp"].sum()),
                        n_unmatched=int(a["counts_unmatched"].sum()),
                        n_is_TP_all_op=a["n_is_TP_all_op"],
                        min_nhi_true=a["min_nhi_true"]),
        floor_17p2=dict(n_cat_cut=b["n_cat_cut"], n_op=b["n_op"],
                        min_nhi_true=b["min_nhi_true"], **tot),
        derived=dict(
            implied_19p5_unmatched=implied_19p5_unmatched,
            measured_19p5_unmatched=int(a["counts_unmatched"].sum()),
            cat_cut_perturbation_19p5_minus_17p2=cat_cut_perturbation,
            genuine_absorbers_in_19p5_unmatched=int(a["counts_unmatched"].sum())
            - tot["hostless"],
            hostless_per_coarse_z_K=[int(b["hostless"][:, k * 5:(k + 1) * 5, :].sum())
                                     for k in range(3)],
            counts_all_per_coarse_z_K=[int(b["counts_all"][:, k * 5:(k + 1) * 5, :].sum())
                                       for k in range(3)],
            hostless_per_snr_s=[int(b["hostless"][:, :, s].sum())
                                for s in range(b["hostless"].shape[2])],
        ),
        provenance=prov,
    )

    npz_path = os.path.join(out_dir, f"fp_census_{family}.npz")
    json_path = os.path.join(out_dir, f"fp_census_{family}.json")
    np.savez(
        npz_path,
        counts_19p5=a["counts"], counts_tp_19p5=a["counts_tp"],
        counts_unmatched_19p5=a["counts_unmatched"],
        counts_all=b["counts_all"], hostless=b["hostless"],
        **{k: b[k] for k in HOST_SLOT_NAMES},
        nhat_edges=ep.NHAT_EDGES, zf_edges=ep.ZF_EDGES, snr_edges=ep.SNR_EDGES,
        zc_edges=ep.ZC_EDGES, kz_to_K=ep.KZ_TO_K,
        host_slot_names=np.array(HOST_SLOT_NAMES),
        host_slot_lo=np.array([lo for _, lo, _ in HOST_SLOTS]),
        host_slot_hi=np.array([hi for _, _, hi in HOST_SLOTS]),
        provenance=np.array(json.dumps(prov, default=str)),
    )
    with open(json_path, "w") as fh:
        json.dump(totals, fh, indent=1, default=str)
    print(f"[{family}] wrote {npz_path}")
    print(f"[{family}] wrote {json_path}")
    print(f"[{family}] TOTAL {time.time() - t_start:.1f}s")
    return totals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--pack", default=None,
                    help="pack whose counts[c,k,s] the 19.5-floor bundle must "
                         "reproduce EXACTLY (default: the collar-scan pack of "
                         "record scanpack_<fam>_b300.npz)")
    ap.add_argument("--work", default=None, help="scratch dir for the loaders "
                                                 "(default: <out>/_work)")
    args = ap.parse_args(argv)
    build(args.family, args.out, pack_path=args.pack, work_dir=args.work)


if __name__ == "__main__":
    main()
