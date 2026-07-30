#!/usr/bin/env python
"""Build the committed tombstone records for retired HBI result artifacts.

A tombstone retires an artifact *identity*. See ``SCHEMA.md`` in this directory for the
schema and the hard rules; the two that constrain this builder are:

  * a tombstone carries NO retired science value -- every number it writes is an int or a
    hex digest string (``_assert_no_floats`` enforces this before writing), and
  * regeneration post-B16 is a NEW MEASUREMENT with a NEW ARTIFACT IDENTITY, never a
    provenance repair of the retired one.

The retired artifacts are UNTRACKED scratch files: they were never committed, so this
builder must be pointed at a worktree that still physically holds them
(``--source-worktree``, default the primary worktree).

Usage
-----
    python CDDF_analysis/hbi/tombstones/build_tombstones.py [--source-worktree DIR] [--force]

Refuses to stamp from a dirty tree unless ``--allow-dirty`` is given (a dirty stamp is
precisely the defect three of these four artifacts are being retired for).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DEFAULT_SOURCE = "/home/mfho/desi_gpy_dla_detection"

SCHEMA = "hbi-artifact-tombstone"
SCHEMA_VERSION = 1
RETIRED_UNDER = "PI decision 7 (2026-07-29): retire with stamped tombstone records"

# --------------------------------------------------------------------------------------
# defect texts (shared)
# --------------------------------------------------------------------------------------
_DIRTY_DETAIL = (
    "metadata.code_commit ends in '-dirty': the artifact was produced by uncommitted "
    "modifications to its own diagnostics script. The working tree that produced it was "
    "never committed, so `git cat-file`/`git checkout` cannot recover the producing "
    "source. This is not a labelling defect that a re-stamp could fix -- the information "
    "is gone. Provenance class: worse than ORPHANED, since even the routine text is "
    "unrecoverable."
)
_B16_DETAIL = (
    "B16 leaky truth: the truth f(N) numerator was integrated with NO z-mask while the "
    "pathlength denominator Delta X WAS masked. The correct criterion is provenance, not "
    "units -- anything built from tr['f_truth'] is biased regardless of what it is "
    "called, including ell(X), lambda_mfp anchors and per-bin f_tru. The measured leak is "
    "1.0557-1.0606 integrated but 1.0000-1.1818 per fine bin, i.e. NOT a scalar: "
    "RE-DERIVE, never rescale."
)
_LLS_DETAIL = (
    "The artifact's leaves ARE LLS population measurements (ell(X) over [17.2,19.5) and "
    "[17.5,19.5), lambda_mfp, kappa_912 / tau_eff_LL, and the LLS drop chi2). Standing "
    "project policy bars LLS population values as results; ell(X) is additionally "
    "structurally prior-limited (its two largest errors sit OUTSIDE its band, which is a "
    "statistical width only). Committing the leaves would be committing the retired "
    "result, so this tombstone records the identity and the defect and nothing else."
)

_NEW_IDENTITY = (
    "Regeneration post-B16 is a NEW MEASUREMENT with a NEW ARTIFACT IDENTITY, not a "
    "provenance repair. Re-running the retired routine today does not restore this "
    "artifact's provenance: (i) the producing source is unrecoverable, so the new run is "
    "not the same computation, and (ii) the B16 fix changes the estimand itself (the "
    "truth reductions the ratios are taken against), so the new number answers a "
    "different question. A successor MUST be written to a different path, stamped afresh "
    "against its own clean HEAD, and MUST NOT reuse this identity."
)

# --------------------------------------------------------------------------------------
# the retirement table
# --------------------------------------------------------------------------------------
RETIRED = {
    "lls_recovery_figures.json": dict(
        path="CDDF_analysis/hbi/lls_recovery_figures.json",
        what_it_was=(
            "LLS joint-estimator measurement-vs-truth figure set (7 panels: P1 the "
            "lambda_mfp / kappa_912 headline, P2 the ell(X) band, P3 the sub-LLS slope "
            "probe, P4 the FP swing, P6 the kappa/ell band, P7 the narrow-vs-wide "
            "normalisation) on the 2LPT-0 (loa-124) mock. MOCK values."
        ),
        defects=[
            ("LLS_POPULATION_CONTENT", _LLS_DETAIL),
            ("B16_LEAKY_TRUTH", _B16_DETAIL + " This artifact's /truth/ell_* anchors and "
             "every r0_* taken against them are f_truth integrals, so they are in the "
             "blast radius."),
        ],
        recoverable_from_git=True,
        recovery_note=(
            "The stamp is CLEAN (a bare 40-char sha that exists in this repo), so the "
            "producing source IS recoverable via `git cat-file`. This artifact is retired "
            "for CONTENT (LLS population values + B16 truth contamination), not for a "
            "provenance hole."
        ),
        requirements=[
            "Re-derive the truth side with the z-masked truth reductions (B16 fix), never "
            "rescale the stored numbers by any leak factor -- the leak is per-bin, not scalar.",
            "Obtain explicit PI sign-off before any LLS population quantity (ell(X), "
            "lambda_mfp, tau_eff_LL, Omega_HI(LLS)) is written to a committed artifact at all.",
            "Report ell over [17.5,19.5) and state the prior-limitation as a limit, not a band.",
            "Write to a NEW path; stamp a clean 40-char HEAD; paper_facing stays false.",
        ],
    ),
    "subdla_edge_systematic.json": dict(
        path="CDDF_analysis/hbi/subdla_edge_systematic.json",
        what_it_was=(
            "Decoupled basis-padding bracket isolating geometric edge-mass recovery at the "
            "19.5 sub-DLA floor: integrated [19.5,20.3) recovery ratios (dN/dX and Omega) "
            "and per-0.1-dex ratios for headline floor-19.5 (no pad) versus basis pads at "
            "19.2 and 19.0, plus the padding-band over-recovery and a 5-criterion gate. "
            "2LPT-0 (loa-124) MOCK recovery ratios with an MC band."
        ),
        defects=[
            ("DIRTY_STAMP_NOT_REDERIVABLE", _DIRTY_DETAIL),
            ("B16_LEAKY_TRUTH", _B16_DETAIL + " Every recovery ratio here is est/truth "
             "with the truth side an f_truth integral, so every Omega ratio is biased."),
            ("STALE_FP_RESAMPLE_SEMANTICS",
             "Its MC band predates the Loa0FP.resample fix; the pre-fix resample "
             "manufactured false positives in empty cells, so FP-resampled BANDS (not "
             "points) were not safe."),
        ],
        recoverable_from_git=False,
        recovery_note=(
            "NOT recoverable. The routine's producing text was uncommitted; the parked "
            "note points at lls-subdla-cddf @1a39493 as the *tree state*, but the "
            "modifications on top of it were never committed."
        ),
        requirements=[
            "COMMIT the diagnostics routine FIRST, then run, so the stamp is a clean 40-char sha.",
            "Re-derive the truth reductions under the B16 z-mask; do not rescale.",
            "Re-run the MC band on the fixed Loa0FP.resample semantics.",
            "Re-express the pad question under PI decision 4 (pad floor 19.0, molly172 "
            "convention) and propagate the measured convention dependence as a SYSTEMATIC.",
            "Write to a NEW path; the successor answers PI decision 4, not the retired question.",
        ],
    ),
    "subdla_floor_mc_band.json": dict(
        path="CDDF_analysis/hbi/subdla_floor_mc_band.json",
        what_it_was=(
            "Joint-MC error band on the sub-DLA integrated [19.5,20.3) recovery ratios "
            "(dN/dX and Omega) and the per-0.1-dex ratios, for the floor-19.5 (headline) "
            "and floor-19.0 (rebuild) configurations on the loa0 FP estimator. 2LPT-0 "
            "(loa-124) MOCK recovery ratios with an MC band."
        ),
        defects=[
            ("DIRTY_STAMP_NOT_REDERIVABLE", _DIRTY_DETAIL),
            ("B16_LEAKY_TRUTH", _B16_DETAIL),
            ("STALE_FP_RESAMPLE_SEMANTICS",
             "The band is the whole product here, and it was built on the pre-fix "
             "Loa0FP.resample that manufactured FP in empty cells -- one such band went "
             "negative and was masked by recenter_band_on_point. A band-valued artifact "
             "built on that resample is unusable end to end."),
        ],
        recoverable_from_git=False,
        recovery_note=(
            "NOT recoverable. Same dirty-tree hole as subdla_edge_systematic.json; both "
            "were produced in the same uncommitted working tree."
        ),
        requirements=[
            "COMMIT the routine FIRST; stamp clean.",
            "Rebuild on the fixed Loa0FP.resample; verify the band cannot go negative "
            "without recenter_band_on_point hiding it.",
            "Re-derive truth under the B16 z-mask.",
            "Report the measured mis-scaling detection curve: truth-containment is "
            "monotone in band width and therefore cannot fail an over-wide band.",
            "Write to a NEW path.",
        ],
    ),
    "subdla_mock_headline.json": dict(
        path="CDDF_analysis/hbi/subdla_mock_headline.json",
        what_it_was=(
            "sub-DLA-band catalog-HBI headline measurement (dN/dX, Omega, and the CDDF "
            "f(N|z)) over [19.5,20.3) on the 2LPT-0 (loa-124) MOCK, produced as a "
            "config-only override of the DLA loa0 headline recipe (frozen ingredients "
            "identical; only report_limits and fp_estimator changed)."
        ),
        defects=[
            ("DIRTY_STAMP_NOT_REDERIVABLE", _DIRTY_DETAIL),
            ("B16_LEAKY_TRUTH", _B16_DETAIL + " The measurement side is an estimate, but "
             "the artifact is only ever consumed against truth reductions, and its "
             "estimand is stamped DIAGNOSTIC_RECENTERED -- a recentred band is not the "
             "estimand a headline reports."),
            ("RETIRED_REPORTING_WINDOW",
             "Its window is [19.5,20.3) with a 19.5 fit floor. PI decision 1 restricts "
             "the primary reporting window to 19.7 <= log NHI <= 21.6 and PI decision 4 "
             "moves the pad floor to 19.0 under the molly172 convention, so this "
             "artifact's window is no longer the reported one under any successor."),
        ],
        recoverable_from_git=False,
        recovery_note=(
            "NOT recoverable. Uncommitted producing tree, same as the other two sub-DLA "
            "artifacts."
        ),
        requirements=[
            "COMMIT the routine FIRST; stamp clean.",
            "Re-derive under PI decision 1 (report only 19.7 <= log NHI <= 21.6), decision "
            "3 (0.2-dex latent basis; a 0.1-dex plotting grid is NOT independent "
            "information resolution) and decision 4 (pad floor 19.0, molly172, pad is a "
            "latent nuisance only).",
            "Carry an estimand that is the reported estimand, not DIAGNOSTIC_RECENTERED.",
            "Re-run closure and coverage; the successor is gated on the decision-8 "
            "framework (fail-closed, matched-configuration SBC, chi2/dof <= 3).",
            "Write to a NEW path.",
        ],
        # this artifact was load-bearing for a TEST, not for a result -- see SCHEMA.md
        tripwire=dict(
            consumer=(
                "tests/test_subdla_forward_headline.py::"
                "test_two_independent_forward_derivations_agree_bitforbit "
                "(and the secondary cross-check in "
                "test_band_is_cumulative_difference_not_direct_integral)"
            ),
            hazard=(
                "The consumer read this file from the WORKING TREE and did "
                "`if head is None or xm is None: pytest.skip(...)`. Deleting the file "
                "therefore turned two abs=1e-12 assertions into a silent skip -- it would "
                "have disarmed the corroboration instead of retiring the artifact."
            ),
            resolution=(
                "The head-side values are replaced by COMMITMENTS: sha256(repr(float)) at "
                "the exact JSON pointers the consumer read. A commitment is not a value "
                "-- it cannot be quoted, plotted or integrated -- so the no-science-values "
                "rule holds. The consumer now hashes crossmock_transfer_loa0.json at the "
                "matching pointers and compares against these commitments, so the "
                "bit-for-bit head-vs-crossmock agreement is still certified with the "
                "retired file gone."
            ),
            corroborating_artifact="CDDF_analysis/hbi/crossmock_transfer_loa0.json",
            commitment_algorithm="sha256(repr(float(value)).encode('utf-8')).hexdigest()",
            # (pointer in the retired headline, matching pointer in the corroborating file)
            pointer_pairs=[
                ("/measurement/19.5/dndx/integrated/MAP",
                 "/self_recovery_baseline_2lpt0/cumulative_map/dndx/19.5"),
                ("/measurement/20.3/dndx/integrated/MAP",
                 "/self_recovery_baseline_2lpt0/cumulative_map/dndx/20.3"),
                ("/measurement/19.5/omega/integrated/MAP",
                 "/self_recovery_baseline_2lpt0/cumulative_map/omega/19.5"),
                ("/measurement/20.3/omega/integrated/MAP",
                 "/self_recovery_baseline_2lpt0/cumulative_map/omega/20.3"),
            ],
            derived_commitments=[dict(
                note=(
                    "Secondary consumer check (test_band_is_cumulative_difference_not_"
                    "direct_integral, step ii): the sub-DLA band plus the DLA tier must "
                    "equal the retired headline's cum(19.5) in dN/dX, i.e. the band is "
                    "cum(19.5)-cum(20.3) and not a mislabelled cumulative. Both summands "
                    "live in the COMMITTED forward artifact, so committing the digest of "
                    "their sum re-anchors this check on committed data only."
                ),
                from_committed_artifact=(
                    "CDDF_analysis/hbi/subdla_mock_validation_forward.json"),
                sum_of_pointers=[
                    "/integrated/loa0/dndx_est_195_203",
                    "/integrated/loa0/dndx_est_203",
                ],
                equals_pointer="/measurement/19.5/dndx/integrated/MAP",
            )],
        ),
    ),
}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _git(args, cwd=_REPO):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)


def _head_sha():
    r = _git(["rev-parse", "HEAD"])
    if r.returncode != 0:
        raise SystemExit("cannot resolve HEAD: " + r.stderr.strip())
    return r.stdout.strip()


def _tree_is_dirty():
    return bool(_git(["status", "--porcelain"]).stdout.strip())


def _committed_json_at_head(relpath, ref="HEAD"):
    r = _git(["show", f"{ref}:{relpath}"])
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)


def resolve_pointer(doc, pointer):
    """Minimal RFC-6901 JSON pointer resolution ('/a/b/c'). Returns None if absent."""
    cur = doc
    for tok in pointer.split("/")[1:]:
        tok = tok.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            try:
                cur = cur[int(tok)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            if tok not in cur:
                return None
            cur = cur[tok]
        else:
            return None
    return cur


def commit_float(value):
    """The commitment: sha256 of the exact Python repr of the float. NOT the value."""
    return hashlib.sha256(repr(float(value)).encode("utf-8")).hexdigest()


def _assert_no_floats(obj, path="$"):
    """Hard rule 1: a tombstone may hold ints and hex digests, never a float."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        raise AssertionError(
            f"tombstone would carry a float at {path} -- tombstones must not carry "
            "retired science values (SCHEMA.md hard rule 1)")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_floats(v, f"{path}[{i}]")


def _stamp_class(code_commit):
    if code_commit is None:
        return "MISSING"
    return "DIRTY" if code_commit.endswith("-dirty") else "CLEAN"


def build_one(name, spec, source_worktree, head_sha, now):
    src = os.path.join(source_worktree, spec["path"])
    if not os.path.exists(src):
        raise SystemExit(
            f"{src} not found. Point --source-worktree at a tree that still holds the "
            "retired artifact (it was never committed, so git cannot supply it).")
    raw = open(src, "rb").read()
    doc = json.loads(raw.decode("utf-8"))
    meta = doc.get("metadata", {})
    code_commit = meta.get("code_commit")

    tomb = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "artifact": {
            "path": spec["path"],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "stamped_code_commit": code_commit,
            "stamp_class": _stamp_class(code_commit),
            "was_tracked_at_git_head": _git(
                ["cat-file", "-e", f"HEAD:{spec['path']}"]).returncode == 0,
            "what_it_was": spec["what_it_was"],
            "rederive_command_as_stamped": meta.get("rederive"),
            "read_from_worktree": source_worktree,
        },
        "retirement": {
            "retired_utc": now,
            "retired_under": RETIRED_UNDER,
            "defects": [{"code": c, "detail": d} for c, d in spec["defects"]],
            "recoverable_from_git": spec["recoverable_from_git"],
            "recovery_note": spec["recovery_note"],
        },
        "successor_policy": {
            "regeneration_is_a_new_measurement": True,
            "statement": _NEW_IDENTITY,
            "requirements": spec["requirements"],
            "must_not_reuse_identity": True,
            "successor_identity_rule": (
                "A successor MUST be written to a path that is not "
                f"'{spec['path']}'. Committing anything at that path is a resurrection "
                "and is caught by tests/test_tombstones.py."
            ),
        },
        "values_policy": {
            "carries_science_values": False,
            "note": (
                "This record deliberately holds no dN/dX, Omega, f(N), ell(X), "
                "lambda_mfp, tau_eff_LL or recovery ratio. Every number below is an "
                "integer or a hex digest. Enforced by "
                "tests/test_tombstones.py::test_tombstones_carry_no_science_values."
            ),
        },
        "metadata": {
            "code_commit": head_sha,
            # A tombstone is itself a committed artifact, so it must satisfy the repo-wide
            # audit (CDDF_analysis/unblind/audit.py): a clean full-sha stamp PLUS an
            # identifiable generating routine. Its routine is this builder -- NOT the
            # retired artifact's routine, which is recorded separately (and read-only) at
            # /artifact/rederive_command_as_stamped.
            "routine": "CDDF_analysis/hbi/tombstones/build_tombstones.py",
            "rederive": (
                "python CDDF_analysis/hbi/tombstones/build_tombstones.py --force "
                f"--source-worktree {source_worktree}"
            ),
            "builder": "CDDF_analysis/hbi/tombstones/build_tombstones.py",
            "generated_utc": now,
            "paper_facing": False,
        },
    }

    tw = spec.get("tripwire")
    if tw is not None:
        # The corroborating artifact is ALSO untracked, so a commitment taken only on the
        # retired side would leave the agreement unverifiable once both files are gone.
        # Commit BOTH sides' commitments and assert their equality HERE, at build time:
        # for IEEE-754 doubles, repr() is round-trip-exact, so
        #     sha256(repr(a)) == sha256(repr(b))  <=>  a == b  bit-for-bit.
        # The committed pair of digests is therefore a permanent, value-free certificate
        # of the bit-for-bit agreement the retired artifact used to demonstrate.
        corro_rel = tw["corroborating_artifact"]
        corro_path = os.path.join(source_worktree, corro_rel)
        if not os.path.exists(corro_path):
            raise SystemExit(
                f"{corro_rel} not found under {source_worktree}: cannot certify the "
                "tripwire agreement without both sides. Refusing to write a half-armed "
                "tombstone.")
        corro_doc = json.loads(open(corro_path, "rb").read().decode("utf-8"))
        commitments = []
        for head_ptr, corro_ptr in tw["pointer_pairs"]:
            val = resolve_pointer(doc, head_ptr)
            if val is None:
                raise SystemExit(f"{spec['path']}: pointer {head_ptr} did not resolve")
            cval = resolve_pointer(corro_doc, corro_ptr)
            if cval is None:
                raise SystemExit(f"{corro_rel}: pointer {corro_ptr} did not resolve")
            h_a, h_b = commit_float(val), commit_float(cval)
            if h_a != h_b:
                raise SystemExit(
                    f"tripwire FAILS at build time: {spec['path']}{head_ptr} and "
                    f"{corro_rel}{corro_ptr} do NOT agree bit-for-bit. The retired "
                    "artifact's corroboration was never true; do not tombstone over it.")
            commitments.append({
                "pointer": head_ptr,
                "corroborating_pointer": corro_ptr,
                "sha256_of_repr": h_a,
                "corroborating_sha256_of_repr": h_b,
            })
        # derived commitment: the secondary consumer check (band + DLA-tier == cum(19.5))
        # was ALSO read off the retired headline. Re-anchor it on the COMMITTED forward
        # artifact so it no longer needs any untracked file at all.
        derived = []
        for d in tw.get("derived_commitments", []):
            fwd = _committed_json_at_head(d["from_committed_artifact"])
            if fwd is None:
                raise SystemExit(
                    f"{d['from_committed_artifact']} is not committed at HEAD; cannot "
                    "re-anchor the derived tripwire commitment.")
            terms = [resolve_pointer(fwd, p) for p in d["sum_of_pointers"]]
            if any(t is None for t in terms):
                raise SystemExit(
                    f"{d['from_committed_artifact']}: a derived pointer did not resolve "
                    f"({d['sum_of_pointers']})")
            total = float(sum(float(t) for t in terms))
            target = resolve_pointer(doc, d["equals_pointer"])
            if target is None:
                raise SystemExit(f"{spec['path']}: pointer {d['equals_pointer']} absent")
            if commit_float(total) != commit_float(target):
                raise SystemExit(
                    "derived tripwire FAILS at build time: "
                    f"sum{d['sum_of_pointers']} != {d['equals_pointer']}. Do not "
                    "tombstone over a false corroboration.")
            derived.append({
                "note": d["note"],
                "from_committed_artifact": d["from_committed_artifact"],
                "sum_of_pointers": d["sum_of_pointers"],
                "equals_pointer_in_retired_artifact": d["equals_pointer"],
                "sha256_of_repr": commit_float(total),
            })
        tomb["tripwire"] = {
            "consumer": tw["consumer"],
            "hazard": tw["hazard"],
            "resolution": tw["resolution"],
            "corroborating_artifact": corro_rel,
            "corroborating_artifact_is_committed": _git(
                ["cat-file", "-e", f"HEAD:{corro_rel}"]).returncode == 0,
            "commitment_algorithm": tw["commitment_algorithm"],
            "equality_of_digests_certifies_bit_equality": (
                "repr() of an IEEE-754 double is round-trip exact, so equality of the two "
                "digests is equivalent to bit-for-bit equality of the two floats. The "
                "consumer test asserts digest equality UNCONDITIONALLY -- it needs neither "
                "the retired file nor the corroborating file to be present."
            ),
            "commitments": commitments,
            "derived_commitments": derived,
        }

    _assert_no_floats(tomb)
    return tomb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-worktree", default=DEFAULT_SOURCE)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing tombstones")
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and diff against what is committed; write nothing")
    a = ap.parse_args(argv)

    if _tree_is_dirty() and not a.allow_dirty and not a.check:
        raise SystemExit(
            "refusing to stamp from a dirty tree -- a dirty stamp is the very defect three "
            "of these four artifacts are retired for. Commit first, or pass --allow-dirty.")

    head_sha = _head_sha()
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rc = 0
    for name, spec in RETIRED.items():
        tomb = build_one(name, spec, a.source_worktree, head_sha, now)
        out = os.path.join(_HERE, name.replace(".json", "") + ".tombstone.json")
        if a.check:
            if not os.path.exists(out):
                print(f"MISSING {out}")
                rc = 1
                continue
            old = json.load(open(out))
            volatile = ("metadata", "retirement")
            a_cmp = {k: v for k, v in tomb.items() if k not in volatile}
            b_cmp = {k: v for k, v in old.items() if k not in volatile}
            status = "OK" if a_cmp == b_cmp else "DRIFT"
            if status == "DRIFT":
                rc = 1
            print(f"{status} {out}")
            continue
        if os.path.exists(out) and not a.force:
            print(f"exists, skipping (use --force): {out}")
            continue
        with open(out, "w") as fh:
            json.dump(tomb, fh, indent=2, sort_keys=False)
            fh.write("\n")
        print(f"wrote {out}  ({tomb['artifact']['bytes']} B retired, "
              f"sha256={tomb['artifact']['sha256'][:12]}…, stamp={head_sha})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
