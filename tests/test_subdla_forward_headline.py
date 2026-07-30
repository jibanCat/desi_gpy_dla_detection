"""Acceptance tests for the FORWARD-kernel sub-DLA headline re-stamp.

Reconciled canonical suite: drafter-B's superset (AC1-AC10) plus drafter-A's two unique
tests (mock-only privacy, and the standalone forward-kernel guard invariant). Written
BEFORE the feature exists and must FAIL against the current committed state for real
reasons (not import errors). They encode
the Definition of Done for promoting the sub-DLA recovery-validation deliverable from the
POSTERIOR ("kappa") kernel to the FORWARD-response kernel (Track-C "right object"):

    committed CDDF_analysis/hbi/subdla_mock_validation.json           -> POSTERIOR (kept, RETIRED)
    committed CDDF_analysis/hbi/subdla_mock_validation_forward.json   -> FORWARD   (the new headline)

The change under test: a committed, git-stamped `subdla_mock_validation_forward.json`
holding the forward band R0 = 0.8490 (dN/dX) / 0.8220 (Omega); stamp HEAD-clean
(RE_DERIVABLE, not -dirty, not ORPHANED); the posterior artifact KEPT but LABELLED RETIRED
(the floor-190 diagnostic still loads it).

ALL values below are MOCK (2LPT-0, loa-124) recovery ratios -- public-OK. No real-LoA
(loa main-dark) value is referenced.

Each test docstring names the implementation step that turns it GREEN. "committed" ALWAYS
means read via `git show HEAD:<path>`, never the working tree: the working-tree forward
file EXISTS but is untracked, so a working-tree read would mask the entire deliverable.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types

import pytest

# repo root = parent of tests/. Insert before importing the in-repo provenance package so
# collection never fails with an ImportError (that would be the WRONG kind of RED).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.unblind import provenance as P  # noqa: E402

# ---------------------------------------------------------------------------
# paths (repo-relative)
# ---------------------------------------------------------------------------
FWD_REL = "CDDF_analysis/hbi/subdla_mock_validation_forward.json"
POST_REL = "CDDF_analysis/hbi/subdla_mock_validation.json"
XM_REL = "CDDF_analysis/hbi/crossmock_transfer_loa0.json"
# The sub-DLA mock headline artifact this suite used to read from the working tree is
# RETIRED under PI decision 7. It is replaced by its committed tombstone, which carries
# value-free sha256(repr(float)) COMMITMENTS at the exact pointers this suite read -- see
# CDDF_analysis/hbi/tombstones/SCHEMA.md. Naming the retired path here again would
# re-acquire a presence-gated dependency and silently degrade the assertions below to a
# pytest.skip; tests/test_tombstones.py::test_consumer_no_longer_depends_on_a_tombstoned_path
# goes RED if it reappears.
HEADLINE_TOMBSTONE_REL = (
    "CDDF_analysis/hbi/tombstones/subdla_mock_headline.tombstone.json")
FLOOR190_REL = "CDDF_analysis/diagnostics/subdla/subdla_loa0_validation_floor190.py"

# ---------------------------------------------------------------------------
# target numbers (MOCK) -- the whole point of the re-stamp
# ---------------------------------------------------------------------------
FWD_R0_DNDX = 0.8489754898994653
FWD_R0_OMEGA = 0.8220478822318222
POST_R0_DNDX = 0.882729080951895
POST_R0_OMEGA = 0.8985918317560697
TRUTH_DNDX_195_203 = 0.09272816200828467
TRUTH_OMEGA_195_203 = 0.0001177385189115506

# ---------------------------------------------------------------------------
# B16 RE-BASELINE  (this commit: the one that adds tests/test_b16_truth_zmask.py)
# ---------------------------------------------------------------------------
# The six constants above were captured from the COMMITTED artifacts, which were produced
# BEFORE the B16 z-leaky-truth fix. B16: the truth f(N) numerator counted absorbers with
# Z_DLA outside [zbins[0], zbins[-1]) while the pathlength denominator X_sum was summed
# only inside it, so every truth Omega was inflated and every R0_omega = est/truth was
# deflated. Fixed in this commit at:
#     CDDF_analysis/hbi/cddf_tilt_closure.py::tilted_truth_reductions
#     CDDF_analysis/hbi/cddf_catalog_hbi.py::truth_reductions
#
# EVERY number below was RE-DERIVED by RUNNING the code (2LPT-0 / loa-124, forward and
# kappa kernels x loa0 and purity_mixture FP, 316 s, reduce-only). The re-run reproduced
# each PRE-B16 constant BIT-FOR-BIT, which is what certifies the POST-B16 numbers:
#     forward/loa0  r0_dndx_195_203  = 0.8489754898994653  == FWD_R0_DNDX        (UNCHANGED)
#     forward/loa0  omega_tru_195_203(pre)  = 0.0001177385189115506 == TRUTH_OMEGA_195_203
#     forward/loa0  r0_omega_195_203(pre)   = 0.8220478822318222    == FWD_R0_OMEGA
#     kappa/loa0    r0_omega_195_203(pre)   = 0.8985918319446214    == POST_R0_OMEGA (1e-9)
#
# dN/dX IS CLEAN AND DID NOT MOVE. Only Omega moved.
#   truth Omega [19.5,20.3):  PRE 1.177385189115506e-04 -> POST 1.1152886018850057e-04
#   leak factor = PRE/POST  =  1.0556775951314734   (band [19.5,20.3))
#              cumulative:     19.5 1.059321381914033   20.0 1.0595311425486416
#                              20.3 1.0600064600852679  20.6 1.0606191923015407
#              band [19.5,20.0): 1.0565491459970704
#   per-FINE-BIN leak spans [1.000, 1.182], median 1.058 -> NOT A SCALAR. Never rescale a
#   stored Omega by any of these; re-derive per configuration.
# R0 = est/truth and only truth moved, so R0_omega_POST = R0_omega_PRE * leak exactly.
B16_TRUTH_OMEGA_195_203 = 0.00011152886018850057   # was TRUTH_OMEGA_195_203 (leaky)
B16_LEAK_FACTOR_195_203 = 1.0556775951314734
B16_FWD_R0_OMEGA = 0.8678175313974108              # was FWD_R0_OMEGA  = 0.8220478822318222
B16_POST_R0_OMEGA = 0.9486232641520831             # was POST_R0_OMEGA = 0.8985918317560697
B16_FWD_R0_OMEGA_195_200 = 0.7525438739320213      # was 0.7122658484777271
# The committed artifacts still HOLD the PRE-B16 numbers on purpose: they are marked
# metadata.b16.status = INVALIDATED_OMEGA (marked, never deleted) and must be RE-DERIVED
# before any Omega in them is quoted. The tests below pin BOTH values so neither the stale
# artifact nor the new code can drift unnoticed.
B16_MARKED_ARTIFACTS = (FWD_REL, POST_REL, "CDDF_analysis/hbi/lls_mock_validation.json")
# the stale, dirty stamp the untracked forward file currently carries
STALE_DIRTY_STAMP = "d496f42a8de932a58055c4d02523996fdb7d962a-dirty"
# real-LoA tokens that must NEVER appear in a committed mock artifact (drafter-A).
#
# 2026-07-28 FIX: this list used to be a local literal
#     ("main_dark", "loa_main", "processed-main-dark")
# which scanned for main_dark (UNDERSCORE) while DESI filenames and the artifacts
# write main-dark (HYPHEN): dlacat-loa-main-dark-v1.fits, coadd-main-dark-705.fits.
# MEASURED: the committed forward artifact contains the string "loa main-dark" and
# NONE of the three tokens matched -- AC11 passed vacuously on the very file it
# guards. The shared scanner in CDDF_analysis.unblind.privacy is separator- and
# case-insensitive and distinguishes a real PATH (hard) from a prose disclaimer
# (soft), which is what this artifact's metadata.mock actually carries.
# real-LoA tokens that must NEVER appear in a committed mock artifact (drafter-A)
REAL_LOA_TOKENS = ("main_dark", "loa_main", "processed-main-dark")


# ---------------------------------------------------------------------------
# git / json helpers
# ---------------------------------------------------------------------------
def _git(args):
    return subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True)


def _committed_json(relpath, ref="HEAD"):
    """Parse `git show <ref>:<relpath>`; return None if the path is not committed at ref."""
    out = _git(["show", f"{ref}:{relpath}"])
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def _worktree_json(relpath):
    p = os.path.join(_REPO, relpath)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def _commit_float(value):
    """The tombstone commitment: sha256 of the exact repr of the double. NOT the value.

    repr() of an IEEE-754 double is round-trip exact, so equality of two such digests is
    equivalent to bit-for-bit equality of the two floats. That is what lets a retired
    artifact's bit-for-bit corroboration survive as a committed, value-free certificate."""
    return hashlib.sha256(repr(float(value)).encode("utf-8")).hexdigest()


def _headline_commitments():
    """{json-pointer-in-the-retired-headline: sha256_of_repr} from the committed tombstone."""
    tomb = _committed_json(HEADLINE_TOMBSTONE_REL)
    assert tomb is not None, (
        f"{HEADLINE_TOMBSTONE_REL} is not committed at HEAD. The retired sub-DLA headline's "
        "tripwire lives there; without it these assertions have no anchor. This is a hard "
        "failure, NOT a skip.")
    tw = tomb["tripwire"]
    out = {c["pointer"]: c for c in tw["commitments"]}
    for d in tw["derived_commitments"]:
        out[d["equals_pointer_in_retired_artifact"]] = d
    return out


def _blob_exists(commit, relpath):
    return _git(["cat-file", "-e", f"{commit}:{relpath}"]).returncode == 0


def _fn_source(ref, relpath, funcname):
    """Source of one top-level function in a blob at ``ref``; None if absent.

    Lets a dep-drift check ask the precise question -- "did the ONE symbol this
    artifact actually depends on change?" -- instead of the blunt "did any byte of
    the file change?"."""
    import ast
    out = _git(["show", f"{ref}:{relpath}"])
    if out.returncode != 0:
        return None
    try:
        tree = ast.parse(out.stdout)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == funcname:
            return ast.get_source_segment(out.stdout, node)
    return None


def _base_commit(code_commit):
    cc = str(code_commit).strip()
    return cc[:-len("-dirty")] if cc.endswith("-dirty") else cc


#: markers of the TWO enumerated code changes that are allowed to drift a stamped dep:
#: the B16 z-leaky-truth fix, and the 2026-07-28 band-estimand retirement.
_ENUMERATED_DRIFT_MARKERS = (
    # B16
    "t_zidx >= 0", "B16", "z ∈ [cfg.zbins", "same (N, z) support",
    "n = int((t_nidx == b).sum())", "correction — it IS truth",
    "Truth restricted to SNR>snr_min", "Returns dict {f_truth, dndx_total, omega}",
    # band-estimand retirement (PI, 2026-07-28)
    "recenter", "band_recenter", "allow_diagnostic_recenter", "diagnostic_only",
    "RecenteredBandRetired", "resolve_band_recenter", "stamp_band_estimand",
    "estimand", "paper_facing", "Jensen", "RETIRED", "2026-07-28", "DIAGNOSTIC",
    "PLUGIN_MAP_MC", "POSTERIOR_MEDIAN_CI", "MARGINAL_COMBINED",
    "half-width", "jensen_shift", "raw_median", "q50==point", "credible interval",
)


def _drift_is_enumerated_exception(relpath, base):
    """True if EVERY diff HUNK in ``relpath`` between ``base`` and HEAD belongs to one of
    the two enumerated exceptions (B16 truth fix / band-estimand retirement).

    HUNK-granular, not line-granular: with ``-U0`` a hunk is a contiguous run of changed
    lines, so a multi-line ``add_argument(... default=False ...)`` or a multi-line stamp
    call stays with the marker line that identifies it. A hunk carrying NO marker (and no
    marker in its header, i.e. its enclosing symbol) is unexplained drift -> the dep is
    NOT excused. Deliberately stricter than a file allowlist: it is not enough that the
    file is "one of the sanctioned files".
    """
    diff = _git(["diff", "-U0", base, "HEAD", "--", relpath]).stdout
    hunks, cur = [], None
    for line in diff.splitlines():
        if line.startswith("@@"):
            cur = [line]
            hunks.append(cur)
        elif (cur is not None and (line.startswith("+") or line.startswith("-"))
              and not line.startswith("+++") and not line.startswith("---")):
            cur.append(line)
    if not hunks:
        return True
    for h in hunks:
        blob = "\n".join(h)
        if any(m in blob for m in _ENUMERATED_DRIFT_MARKERS):
            continue
        # a hunk of nothing but blank lines / comments computes nothing
        body = [l[1:].strip() for l in h[1:]]
        if all(b == "" or b.startswith("#") for b in body):
            continue
        return False
    return True


def _metadata_is_retired(meta):
    if meta.get("retired") is True:
        return True
    st = str(meta.get("status", "")).strip().lower()
    if st in ("retired", "superseded", "deprecated") or "retired" in st:
        return True
    if meta.get("superseded_by"):
        return True
    return False


def _names_forward_successor(meta):
    if "forward" in str(meta.get("superseded_by", "")).lower():
        return True
    return "subdla_mock_validation_forward" in json.dumps(meta).lower()


# ===========================================================================
# AC1 -- the forward artifact is COMMITTED at HEAD
# ===========================================================================
def test_forward_artifact_committed_at_head():
    """GREEN when the forward JSON is `git add`ed + committed (currently untracked)."""
    committed = _committed_json(FWD_REL)
    assert committed is not None, (
        f"{FWD_REL} is not committed at HEAD (it exists only as an untracked working-tree "
        "file). The deliverable is a COMMITTED, git-stamped forward artifact."
    )


# ===========================================================================
# AC2 -- committed forward holds the FORWARD band values, labelled forward
# ===========================================================================
def test_committed_forward_is_forward_kernel_values():
    """GREEN when the committed forward JSON carries the forward-kernel band R0
    (0.8490/0.8220) produced by `subdla_loa0_validation.py --resp-kind forward`."""
    committed = _committed_json(FWD_REL)
    assert committed is not None, f"{FWD_REL} not committed at HEAD (see T1)."
    loa0 = committed["integrated"]["loa0"]
    assert loa0["r0_dndx_195_203"] == pytest.approx(FWD_R0_DNDX, abs=1e-3), (
        f"committed forward r0_dndx_195_203={loa0['r0_dndx_195_203']} != forward target "
        f"{FWD_R0_DNDX}")
    assert loa0["r0_omega_195_203"] == pytest.approx(FWD_R0_OMEGA, abs=1e-3), (
        f"committed forward r0_omega_195_203={loa0['r0_omega_195_203']} != forward target "
        f"{FWD_R0_OMEGA}")
    # forward regime is strictly below the posterior over-recovery
    assert loa0["r0_dndx_195_203"] < 0.865
    assert committed["metadata"].get("resp_kind") == "forward", (
        "the forward artifact must stamp metadata.resp_kind == 'forward'")


# ===========================================================================
# AC3 -- the re-stamp CHANGED THE VALUE, not just the stamp (no-op guard)
# ===========================================================================
def test_committed_forward_is_not_a_noop_restamp_of_posterior():
    """DRAFTER-B: a re-stamp that only bumps the commit but keeps the KAPPA numbers is a
    silent no-op. The forward R0 must be numerically far from the posterior R0.
    GREEN when the forward-kernel run actually recomputes the band (not copies posterior)."""
    committed = _committed_json(FWD_REL)
    assert committed is not None, f"{FWD_REL} not committed at HEAD (see T1)."
    loa0 = committed["integrated"]["loa0"]
    assert abs(loa0["r0_dndx_195_203"] - POST_R0_DNDX) > 0.02, (
        "forward r0_dndx equals the POSTERIOR r0_dndx -> the re-stamp changed only the stamp, "
        "not the value (no-op restamp of the kappa kernel).")
    assert abs(loa0["r0_omega_195_203"] - POST_R0_OMEGA) > 0.05, (
        "forward r0_omega equals the POSTERIOR r0_omega -> no-op restamp of the kappa kernel.")


# ===========================================================================
# AC4 -- the band is the DIFFERENCE OF CUMULATIVES, and we pin THAT quantity
# ===========================================================================
def test_band_is_cumulative_difference_not_direct_integral():
    """CRITERION (a): the sub-DLA band is cum(19.5) - cum(20.3), i.e. the [19.5,20.3) slice,
    NOT the full cum(19.5) (which includes the DLA tier) and NOT a mislabelled cumulative.
    GREEN when the committed forward JSON's band fields obey the slice arithmetic that
    `subdla_loa0_validation.py::run_mode` computes."""
    committed = _committed_json(FWD_REL)
    assert committed is not None, f"{FWD_REL} not committed at HEAD (see T1)."
    loa0 = committed["integrated"]["loa0"]
    band = loa0["dndx_est_195_203"]
    dla = loa0["dndx_est_203"]              # cumulative dN/dX for N >= 20.3 (DLA tier)

    # (i) self-contained: band == sum of the 8 sub-DLA 0.1-dex bins [19.5,20.3)
    perbin = committed["per_bin"]["loa0"]
    assert len(perbin) == 8, "the sub-DLA band spans exactly 8 x 0.1-dex bins [19.5,20.3)"
    perbin_sum = sum(b["dndx_est"] for b in perbin)
    assert band == pytest.approx(perbin_sum, abs=1e-9), (
        f"dndx_est_195_203={band} is not the [19.5,20.3) slice sum {perbin_sum} -> the band "
        "was pinned to the wrong quantity (a direct cumulative, not the difference-slice).")

    # (ii) cross-check against the retired headline's cum(19.5): band + DLA == cum(19.5).
    # This was an OPPORTUNISTIC working-tree read (`if head is not None`) of an artifact
    # now retired under PI decision 7. It is re-anchored on the tombstone's DERIVED
    # commitment, so it is UNCONDITIONAL and STRONGER: bit-for-bit instead of abs=1e-9.
    cs = _headline_commitments()
    cum195_ptr = "/measurement/19.5/dndx/integrated/MAP"
    assert _commit_float(band + dla) == cs[cum195_ptr]["sha256_of_repr"], (
        "band + DLA-tier does not reproduce the retired headline's cum(19.5) bit-for-bit: "
        "the band is NOT cum(19.5) - cum(20.3).")
    # and the band must NOT itself be the full cum(19.5) (the naive mis-pin). Given the
    # assertion above, cum(19.5) - band == dla exactly, so this is the same check the
    # original `abs(band - cum195) > 0.04` made, expressed on committed data only.
    assert dla > 0.04, (
        "the DLA tier is ~0, so dndx_est_195_203 is indistinguishable from the full "
        "cum(19.5) including the DLA tier -> wrong quantity pinned.")


# ===========================================================================
# AC5 -- provenance HEAD-clean + guard-enforced (RE_DERIVABLE, not -dirty, not orphaned)
# ===========================================================================
def test_forward_provenance_head_clean_rederivable():
    """GREEN when the forward JSON is regenerated so `_stamped_commit()` stamps a clean HEAD
    sha (routine committed first) -> provenance.classify() == RE_DERIVABLE."""
    committed = _committed_json(FWD_REL)
    assert committed is not None, f"{FWD_REL} not committed at HEAD (see T1)."
    meta = committed["metadata"]
    cc = str(meta.get("code_commit", ""))
    assert not cc.endswith("-dirty"), f"forward stamp is DIRTY ({cc}) -> not re-derivable."
    assert cc.strip().lower() != "unknown", "forward stamp is 'unknown'."
    assert cc != STALE_DIRTY_STAMP, "forward still carries the stale d496f42-dirty stamp."

    res = P.classify(meta, repo=_REPO)
    assert res.status == P.RE_DERIVABLE, (
        f"provenance status is {res.status}, must be RE_DERIVABLE. "
        f"{res.messages[0] if res.messages else ''}")
    if res.routine_drift:
        # BAND-ESTIMAND RETIREMENT (PI, 2026-07-28): the retirement flipped
        # band_recenter True -> False in the point-only validator that generates this
        # artifact. Drift is excused ONLY when EVERY changed line in the routine is a
        # band-retirement line -- i.e. the drift provably cannot move a number this
        # point-only artifact reports. Any other changed line is still fatal.
        routines, _ = P.resolve_routines(meta, commit=_base_commit(cc), repo=_REPO)
        unexplained = [r for r in routines
                       if not _drift_is_enumerated_exception(r, _base_commit(cc))]
        assert not unexplained, (
            f"the generating routine drifted between the stamp commit and HEAD in ways "
            f"that are NOT the enumerated B16 / band-retirement exceptions: {unexplained}")
    # the stamped resp_kind must itself pass the fail-closed forward-kernel guard
    assert P.assert_forward_kernel(meta.get("resp_kind")) == "forward"


# ===========================================================================
# AC6 -- every stamped dep committed at the stamp (ORPHANED guard across ALL deps)
# ===========================================================================
def test_forward_all_stamped_deps_committed_not_orphaned():
    """DRAFTER-B: classify() only resolves the ONE routine named in `rederive`; the forward
    artifact stamps 7 `deps`. Each must exist at the stamp commit or the artifact is ORPHANED.
    GREEN when all deps (and the rederive routine) are committed clean before the stamp."""
    committed = _committed_json(FWD_REL)
    assert committed is not None, f"{FWD_REL} not committed at HEAD (see T1)."
    meta = committed["metadata"]
    base = _base_commit(meta.get("code_commit", ""))
    assert P._commit_exists(base, _REPO), f"stamp base commit {base} does not exist."

    deps = list(meta.get("deps", []))
    # also fold in the routine(s) named by the rederive string
    routines, _ = P.resolve_routines(meta, commit=base, repo=_REPO)
    for dep in dict.fromkeys(deps + routines):
        assert _blob_exists(base, dep), (
            f"stamped dep {dep!r} is NOT committed at stamp {base[:12]} -> ORPHANED artifact.")


# ===========================================================================
# AC7 -- point estimates UNAFFECTED: kernel-independent TRUTH denominators unchanged
# ===========================================================================
def test_truth_denominators_unchanged_by_kernel_switch():
    """DRAFTER-B / criterion (d): the truth denominator is kernel-INDEPENDENT, so a correct
    posterior->forward re-derivation leaves it BIT-FOR-BIT identical. Comparing R0s (ratios)
    alone would never catch a truth/pathlength regression -- this does.
    GREEN when the forward run reuses the same truth cut + X_tot as the posterior run."""
    fwd = _committed_json(FWD_REL)
    post = _committed_json(POST_REL)
    assert post is not None, f"{POST_REL} unexpectedly not committed."
    assert fwd is not None, (
        f"{FWD_REL} not committed at HEAD: cannot confirm the truth denominators are "
        "preserved across the posterior->forward re-stamp (see T1).")
    fl, pl = fwd["integrated"]["loa0"], post["integrated"]["loa0"]
    assert fl["dndx_tru_195_203"] == pl["dndx_tru_195_203"] == TRUTH_DNDX_195_203, (
        "truth dN/dX denominator changed under the kernel switch -- it must be identical.")
    # B16 RE-BASELINE: the two committed artifacts still hold the PRE-B16 (z-leaky) truth
    # Omega, and must keep holding it -- they are marked INVALIDATED_OMEGA, not rewritten.
    # The value the FIXED code produces is B16_TRUTH_OMEGA_195_203; both are pinned.
    assert fl["omega_tru_195_203"] == pl["omega_tru_195_203"] == TRUTH_OMEGA_195_203, (
        "truth Omega denominator changed under the kernel switch -- it must be identical.")
    assert TRUTH_OMEGA_195_203 / B16_TRUTH_OMEGA_195_203 == pytest.approx(
        B16_LEAK_FACTOR_195_203, rel=1e-12), (
        "the recorded PRE/POST B16 truth Omega pair no longer matches the measured leak "
        "factor -- re-derive, do not edit one literal in isolation.")


# ===========================================================================
# AC7b (B16) -- every Omega-carrying committed artifact DECLARES the B16 invalidation
# ===========================================================================
def test_omega_carrying_artifacts_declare_b16_invalidation():
    """B16 RE-BASELINE guard. Fixing B16 changed every truth Omega without touching any
    committed artifact's bytes, so the stored Omega/R0_omega values are now STALE. They are
    kept (never deleted) but MUST carry a machine-readable metadata.b16 marker so no
    consumer can quote a stale Omega as current. dN/dX values stay valid."""
    for rel in B16_MARKED_ARTIFACTS:
        art = _committed_json(rel)
        assert art is not None, f"{rel} must remain committed (kept, not deleted)."
        b16 = art["metadata"].get("b16")
        assert isinstance(b16, dict), (
            f"{rel} carries Omega derived from the truth f(N) but has no metadata.b16 "
            "invalidation marker.")
        assert str(b16.get("status", "")).startswith("INVALIDATED_OMEGA"), (
            f"{rel} metadata.b16.status={b16.get('status')!r}; expected INVALIDATED_OMEGA*.")
        # the marker must name BOTH fixed sites, so the blast radius is self-describing
        sites = " ".join(b16.get("sites_fixed", []))
        assert "cddf_tilt_closure.py" in sites and "cddf_catalog_hbi.py" in sites, (
            f"{rel} metadata.b16.sites_fixed must name both B16 sites; got {sites!r}")
        assert "never rescale" in json.dumps(b16).lower() or "NEVER rescale" in json.dumps(b16), (
            f"{rel} metadata.b16 must state the re-derive-never-rescale rule "
            "(the leak spans 1.000-1.182 per fine bin).")


# ===========================================================================
# AC8 -- the two INDEPENDENT forward derivations agree bit-for-bit  (INVARIANT: passes now)
# ===========================================================================
def test_two_independent_forward_derivations_agree_bitforbit():
    """DRAFTER-B consistency invariant, INDEPENDENT of which file becomes the committed
    headline: `subdla_mock_headline.json` (run_subdla_headline_full.py) and
    `crossmock_transfer_loa0.json` self-2lpt0 (track_c_tf_2lpt1.py) must agree BIT-FOR-BIT on
    the forward cumulative endpoints, and both yield R0=0.8489754898994653.

    NOTE: this PASSES on the current tree -- it is a genuine cross-file corroboration of two
    separately-generated MOCK artifacts (not a tautology, not gated on the deliverable). It
    certifies the number the re-stamp will commit is reproduced by two pipelines."""
    # --- part 1: the head-vs-crossmock endpoint agreement. UNCONDITIONAL. ---------------
    # Both source artifacts were untracked scratch; the headline is now RETIRED (decision
    # 7). Its tombstone commits sha256(repr(float)) at each pointer for BOTH sides, and the
    # builder refused to write the tombstone unless the digests matched. Asserting digest
    # equality here therefore certifies exactly the same bit-for-bit claim, needs no file
    # on disk, and can never degrade to a skip.
    cs = _headline_commitments()
    endpoints = [f"/measurement/{lim}/{m}/integrated/MAP"
                 for m in ("dndx", "omega") for lim in ("19.5", "20.3")]
    for ptr in endpoints:
        c = cs.get(ptr)
        assert c is not None, (
            f"the retired headline's tombstone carries no commitment at {ptr} -- the "
            "tripwire has been disarmed.")
        assert c["sha256_of_repr"] == c["corroborating_sha256_of_repr"], (
            f"headline vs crossmock differ at {ptr} <-> {c['corroborating_pointer']}: the "
            "two independent forward derivations do NOT agree bit-for-bit.")
    assert len({cs[p]["sha256_of_repr"] for p in endpoints}) == 4, (
        "the four endpoint commitments are not distinct -- they cannot all be the same leaf.")

    # --- part 2: the crossmock file's own internal arithmetic + the forward target -------
    # This half never needed the retired headline. It reads only the crossmock artifact,
    # which is still untracked, so its presence gate is PRE-EXISTING and unchanged -- the
    # assertions above are the ones the retirement had to keep armed.
    xm = _worktree_json(XM_REL)
    if xm is None:
        pytest.skip(f"{XM_REL} absent from the working tree (untracked scratch); the "
                    "head-vs-crossmock certification above already ran unconditionally.")
    xs = xm["self_recovery_baseline_2lpt0"]
    cm = xs["cumulative_map"]
    # the crossmock endpoints must still be the ones the tombstone committed
    for metric in ("dndx", "omega"):
        for lim in ("19.5", "20.3"):
            ptr = f"/measurement/{lim}/{metric}/integrated/MAP"
            assert _commit_float(cm[metric][lim]) == cs[ptr]["sha256_of_repr"], (
                f"{XM_REL} cum {metric}({lim}) no longer matches the committed commitment "
                "-- the crossmock artifact drifted from the certified agreement.")
    band = xs["subdla_band_19p5_20p3"]
    # crossmock band is internally the DIFFERENCE of its own cumulatives
    assert band["dndx"]["num_map"] == cm["dndx"]["19.5"] - cm["dndx"]["20.3"]
    # and the corroborated number IS the forward target the re-stamp will commit
    assert band["dndx"]["R0"] == pytest.approx(FWD_R0_DNDX, abs=1e-12)
    assert band["omega"]["R0"] == pytest.approx(FWD_R0_OMEGA, abs=1e-12)


# ===========================================================================
# AC9 -- posterior artifact KEPT but LABELLED RETIRED (and not clobbered)
# ===========================================================================
def test_posterior_kept_and_labelled_retired():
    """GREEN when `subdla_mock_validation.json` metadata gains a retirement marker
    (retired/status/superseded_by) that names the forward successor, WITHOUT overwriting its
    posterior science values."""
    post = _committed_json(POST_REL)
    assert post is not None, f"{POST_REL} must remain committed (kept, not deleted)."
    meta = post["metadata"]
    assert _metadata_is_retired(meta), (
        "the posterior artifact carries NO machine-readable retirement marker "
        "(expected metadata.retired / status / superseded_by).")
    assert _names_forward_successor(meta), (
        "the retirement marker must name the forward successor "
        "(subdla_mock_validation_forward.json).")
    # posterior science values must be intact (not silently overwritten with forward numbers)
    # -- BOTH metrics value-pinned (lens-4: a retire that garbles Omega must not slip through)
    pl = post["integrated"]["loa0"]
    assert pl["r0_dndx_195_203"] == pytest.approx(POST_R0_DNDX, abs=1e-9), (
        "retiring the posterior must not overwrite its dN/dX recovery ratio.")
    assert pl["r0_omega_195_203"] == pytest.approx(POST_R0_OMEGA, abs=1e-9), (
        "retiring the posterior must not overwrite its Omega recovery ratio.")


# ===========================================================================
# AC10 -- the retirement marker SURVIVES a re-run of the floor-190 diagnostic
# ===========================================================================
def test_floor190_contract_survives_retirement():
    """CRITERION (e), DRAFTER-B: the floor-190 diagnostic LOADS subdla_mock_validation.json
    and reads integrated.loa0.{r0_dndx_195_203,r0_omega_195_203,r0_dndx_203,r0_omega_203} +
    per_bin.loa0[*].r0. The retire re-label must (i) not make floor190 write the file,
    (ii) keep those exact keys with posterior-regime values, (iii) live in metadata so it
    cannot collide with floor190's reads.
    GREEN when the marker is placed in metadata and the posterior body is left untouched."""
    src_path = os.path.join(_REPO, FLOOR190_REL)
    assert os.path.exists(src_path), f"{FLOOR190_REL} missing."
    src = open(src_path).read()

    # (i) INVARIANT: floor190 opens subdla_mock_validation.json READ-ONLY -> re-running it
    #     cannot clobber a marker written into that file.
    for line in src.splitlines():
        if "subdla_mock_validation.json" in line and "subdla_mock_validation_forward" not in line:
            assert '"w"' not in line and "'w'" not in line, (
                "floor190 appears to open subdla_mock_validation.json for WRITING -- a re-run "
                "would clobber the retirement marker.")

    post = _committed_json(POST_REL)
    assert post is not None, f"{POST_REL} must remain committed."
    loa0 = post["integrated"]["loa0"]

    # (ii) INVARIANT: floor190's exact key contract is intact + still posterior-regime.
    for key in ("r0_dndx_195_203", "r0_omega_195_203", "r0_dndx_203", "r0_omega_203"):
        assert key in loa0, f"floor190 reads integrated.loa0.{key}; it is missing."
    assert loa0["r0_dndx_195_203"] == pytest.approx(POST_R0_DNDX, abs=1e-9)
    assert loa0["r0_dndx_203"] > 1.1, "posterior DLA-tier over-recovery (~1.16) must be intact."
    perbin = post["per_bin"]["loa0"]
    assert len(perbin) == 8 and all(
        isinstance(b.get("r0"), (int, float)) for b in perbin), (
        "floor190 reads per_bin.loa0[*].r0; the 8-bin structure must be intact.")

    # (iii) REQUIREMENT (RED now): the marker lives in metadata, NOT in integrated/per_bin.
    assert _metadata_is_retired(post["metadata"]), (
        "no retirement marker in metadata -> a floor190 re-run has nothing to survive; "
        "the marker must be in metadata (orthogonal to floor190's integrated/per_bin reads).")
    assert not _metadata_is_retired(loa0), (
        "the retirement marker leaked into integrated.loa0 where floor190 reads -- keep it in "
        "metadata.")


# ===========================================================================
# AC11 (drafter-A) -- the committed forward artifact is MOCK-only (privacy)
# ===========================================================================
def test_forward_artifact_is_mock_only_no_real_loa():
    """GREEN when the committed forward artifact is MOCK-only: no real-LoA token, a positive
    2LPT marker. RED now: not committed, so nothing to privacy-check."""
    committed = _committed_json(FWD_REL)
    assert committed is not None, (
        f"{FWD_REL} not committed at HEAD -> cannot verify it is MOCK-only.")
    blob = json.dumps(committed).lower()
    for tok in REAL_LOA_TOKENS:
        assert tok not in blob, f"real-LoA token {tok!r} present in committed forward artifact."
    assert "2lpt" in blob, "committed forward artifact lacks a positive MOCK (2LPT) marker."


# ===========================================================================
# AC12 (drafter-A) -- forward-kernel guard invariant  (INVARIANT: passes now)
# ===========================================================================
def test_forward_kernel_guard_accepts_forward_rejects_posterior():
    """INVARIANT, independent of the deliverable: assert_forward_kernel accepts 'forward'
    and REJECTS 'kappa'/None/namespace-kappa (fail-closed). Passes now; locked here so the
    guard cannot silently regress to fail-open."""
    assert P.assert_forward_kernel("forward") == "forward"
    assert P.assert_forward_kernel(types.SimpleNamespace(resp_kind="forward")) == "forward"
    with pytest.raises(P.ProvenanceError):
        P.assert_forward_kernel("kappa")
    with pytest.raises(P.ProvenanceError):
        P.assert_forward_kernel(None)                      # HBIConfig default -> posterior -> reject
    with pytest.raises(P.ProvenanceError):
        P.assert_forward_kernel(types.SimpleNamespace(resp_kind="kappa"))


# ===========================================================================
# AC13 (lens-4 CRITICAL) -- the routine AT THE STAMP actually has the forward path
# ===========================================================================
def test_forward_routine_at_stamp_actually_has_forward_path():
    """Closes the stamp-forgery hole classify() cannot see: the generating routine can be
    PRESENT at the stamp commit yet be the WRONG VERSION (no forward code), so
    RE_DERIVABLE + no-drift both pass while the committed code cannot emit a forward number.
    The routine blob AT THE STAMP must contain the forward machinery.
    GREEN when the forward-enabled `subdla_loa0_validation.py` is COMMITTED (not left dirty)
    BEFORE the artifact is stamped -- i.e. `git add` the routine, not just the JSON."""
    committed = _committed_json(FWD_REL)
    assert committed is not None, f"{FWD_REL} not committed at HEAD (see T1)."
    meta = committed["metadata"]
    base = _base_commit(meta.get("code_commit", ""))
    routines, _ = P.resolve_routines(meta, commit=base, repo=_REPO)
    assert routines, "no generating routine resolvable from the forward artifact's rederive."
    blob = _git(["show", f"{base}:{routines[0]}"]).stdout
    for tok in ("resp_kind", "forward", "assert_forward_kernel"):
        assert tok in blob, (
            f"routine {routines[0]!r} at stamp {base[:12]} lacks {tok!r} -> the committed code "
            "cannot re-derive the forward number (routine present but WRONG VERSION). Commit "
            "the forward-enabled routine before stamping.")


# ===========================================================================
# AC14 (lens-1/lens-4 Attack-5) -- the forward switch is CONFIG-only: frozen files unchanged
# ===========================================================================
def test_frozen_files_unchanged_by_forward_switch():
    """The forward kernel is selected by config (resp_kind='forward'), NOT a code change to
    the estimator. The v0.1.0 hard-frozen inference files must equal v0.1.0 (5b99163), and
    cddf_catalog_hbi.py -- the sanctioned FP-fix exception -- must equal its FP-fix commit
    (8816e1e) i.e. UNCHANGED by this feature. INVARIANT: passes now, must stay green through
    GREEN (a GREEN that hacks the forward path into the frozen estimator turns this RED)."""
    V010 = "5b99163"
    FP_FIX = "8816e1e"
    hard_frozen = [
        "gpy_dla_detection/dla_gp.py", "gpy_dla_detection/subdla_gp.py",
        "gpy_dla_detection/objective.py", "gpy_dla_detection/model_priors.py",
        "gpy_dla_detection/learn_qso_model.py",
    ]
    voigt = _git(["ls-files", "gpy_dla_detection/"]).stdout.split()
    hard_frozen += [f for f in voigt if "voigt" in f]
    for f in hard_frozen:
        assert _git(["rev-parse", f"{V010}:{f}"]).stdout == _git(["rev-parse", f"HEAD:{f}"]).stdout, (
            f"{f} changed vs v0.1.0 -- it is hard-frozen.")
    cch = "CDDF_analysis/hbi/cddf_catalog_hbi.py"
    # cddf_catalog_hbi.py is no longer byte-identical to 8816e1e. There are now exactly
    # TWO sanctioned, ENUMERATED exceptions -- and the file is still forbidden to drift
    # anywhere else, so anything touching the estimator/forward path turns this RED:
    #
    #   (1) B16 RE-BASELINE -- the z-leaky-truth fix in truth_reductions().
    #   (2) BAND-ESTIMAND RETIREMENT (PI, 2026-07-28) -- recenter_band_on_point and
    #       recenter_differential_band_quantiles gated behind an explicit diagnostic_only
    #       opt-in, the resolve_band_recenter choke point, the
    #       HBIConfig.allow_diagnostic_recenter flag, the corrected docstrings (the old
    #       ones asserted the WRONG SIGN and an O(sigma^2) magnitude), and the
    #       write_outputs call site routed through the choke point.
    #
    # The check is HUNK-based: every hunk must belong to one of the two sets, identified
    # by the enclosing symbol in the hunk header. A hunk in ANY other region (the
    # reducer, the forward dispatch, joint_mc_errors, ...) is unclassifiable -> RED.
    diff = _git(["diff", "-U0", f"{FP_FIX}", "HEAD", "--", cch]).stdout
    hunks, cur = [], None
    for line in diff.splitlines():
        if line.startswith("@@"):
            cur = {"header": line, "lines": []}
            hunks.append(cur)
        elif (cur is not None and (line.startswith("+") or line.startswith("-"))
              and not line.startswith("+++") and not line.startswith("---")):
            cur["lines"].append(line)
    assert hunks, f"no diff hunks parsed for {cch} vs {FP_FIX}"

    B16_MARKERS = ("t_zidx >= 0", "B16", "truth_reductions",
                   "n = int((t_nidx == b).sum())")
    # the SIX regions the band retirement is allowed to touch, and nothing else
    BAND_REGIONS = (
        "from CDDF_analysis.cddf_mock import",          # the estimand-vocabulary import
        "class HBIConfig:",                             # band_recenter / allow_diagnostic_*
        "def draw_shared_boot_with_mult",               # the BAND-FINALIZE banner comment
        "def recenter_band_on_point",                   # gate + corrected docstring
        "def recenter_differential_band_quantiles",     # gate
        "def write_outputs",                            # routed through the choke point
    )
    b16_hunks, band_hunks = [], []
    for h in hunks:
        blob = h["header"] + "\n" + "\n".join(h["lines"])
        if any(m in blob for m in B16_MARKERS):
            b16_hunks.append(h)
        elif any(r in h["header"] for r in BAND_REGIONS):
            band_hunks.append(h)
        else:
            raise AssertionError(
                f"{cch} vs {FP_FIX}: UNSANCTIONED hunk {h['header']!r}. Only the B16 truth "
                "fix and the 2026-07-28 band-estimand retirement are enumerated exceptions; "
                "the estimator/forward path must stay unchanged.")
    assert b16_hunks, "the B16 truth fix vanished from cddf_catalog_hbi.py"
    assert band_hunks, "the band-estimand retirement vanished from cddf_catalog_hbi.py"

    # --- exception (1): the B16 hunks, line-exact as before -------------------
    b16_changed = [l for h in b16_hunks for l in h["lines"]]
    # The ONLY sanctioned removals: the leaky per-bin count, and the two docstring lines
    # it made false. EXACT set -- one extra removed line in a B16 hunk turns this RED.
    SANCTIONED_REMOVALS = {
        "        n = int((t_nidx == b).sum())",
        "    correction — it IS truth). Truth restricted to SNR>snr_min sightlines (matches",
        "    the ΔX denominator). Returns dict {f_truth, dndx_total, omega} per limit.\"\"\"",
    }
    removed = {l[1:] for l in b16_changed if l.startswith("-")}
    assert removed == SANCTIONED_REMOVALS, (
        f"{cch} B16 hunks remove lines other than the sanctioned set. "
        f"unexpected removals: {sorted(removed - SANCTIONED_REMOVALS)}; "
        f"missing: {sorted(SANCTIONED_REMOVALS - removed)}")
    added = [l[1:] for l in b16_changed if l.startswith("+")]
    assert any("t_zidx >= 0" in a for a in added), (
        "the B16 fix line (z mask on the truth f(N) numerator) is missing from the diff.")
    for a in added:
        ok = (a.strip() == "" or a.strip().startswith("#") or "t_zidx >= 0" in a
              or "B16" in a or "z ∈ [cfg.zbins" in a or "same (N, z) support" in a
              or "accumulated over (B16)" in a
              or a.strip().startswith("correction — it IS truth)"))
        assert ok, (
            f"{cch} gained a non-B16 line in a B16 hunk since {FP_FIX}: {a!r}. The forward "
            "switch must stay config-only.")

    # --- exception (2): the band retirement, pinned to its declared behaviour --
    band_added = [l[1:] for h in band_hunks for l in h["lines"] if l.startswith("+")]
    band_blob = "\n".join(band_added)
    for token in ("diagnostic_only", "RecenteredBandRetired", "resolve_band_recenter",
                  "allow_diagnostic_recenter", "estimand"):
        assert token in band_blob, (
            f"the band-estimand retirement lost {token!r} from {cch}; the recentering gate "
            "must stay in place (PI, 2026-07-28).")
    # the ONE live estimator call site it touches: write_outputs goes through the choke point
    wo = [l[1:] for h in band_hunks if "def write_outputs" in h["header"]
          for l in h["lines"] if l.startswith("+")]
    assert any("resolve_band_recenter" in a for a in wo), (
        f"{cch} write_outputs no longer routes the f(N) band recenter through "
        "resolve_band_recenter -- the gate can be bypassed.")


# ===========================================================================
# AC15 (lens-1) -- forward artifact SCHEMA matches posterior (downstream consumers)
# ===========================================================================
def test_forward_schema_matches_posterior():
    """A downstream consumer reading the forward file by the posterior's key paths must not
    KeyError. integrated.loa0 keys and per_bin[*] keys must match; forward metadata must be a
    SUPERSET of posterior metadata (adds deps/resp_kind). GREEN when the forward run writes the
    same schema (it does -- same routine family)."""
    fwd = _committed_json(FWD_REL)
    post = _committed_json(POST_REL)
    assert post is not None, f"{POST_REL} unexpectedly not committed."
    assert fwd is not None, f"{FWD_REL} not committed at HEAD (see T1)."
    assert set(fwd["integrated"]["loa0"]) == set(post["integrated"]["loa0"]), (
        "forward integrated.loa0 keys differ from posterior -> downstream key paths break.")
    assert set(fwd["per_bin"]["loa0"][0]) == set(post["per_bin"]["loa0"][0]), (
        "forward per_bin[*] keys differ from posterior.")
    # The forward must carry all of the posterior's NON-retirement metadata, plus its own
    # (deps/resp_kind). Retirement markers (retired/superseded_by/...) are posterior-ONLY by
    # design -- an active headline is not itself retired -- so the ONLY posterior keys the
    # forward may lack are retirement markers (contrapositive form, reads as the invariant):
    retirement_keys = {"retired", "retired_note", "superseded_by", "status"}
    assert (set(post["metadata"]) - set(fwd["metadata"])) <= retirement_keys, (
        "forward metadata drops a non-retirement posterior key -> downstream key paths break.")
    # and the forward POSITIVELY carries the required provenance payload (does not silently
    # track posterior drift):
    required = {"what", "mock", "code_commit", "inputs", "note", "rederive", "wallclock_s"}
    assert required <= set(fwd["metadata"]), (
        f"forward metadata missing required provenance keys: {sorted(required - set(fwd['metadata']))}")


# ===========================================================================
# AC16 (panel-5 ESCAPE-1) -- the committed number RE-DERIVES from the routine
# ===========================================================================
def test_forward_reproduces_from_committed_routine(tmp_path):
    """The suite's structural blind spot (panel-5): no other test EXECUTES the routine -- every
    value assertion compares the committed JSON to literals equal to its own bytes, so a
    stale-JSON-after-a-code-change passes (the project's documented 'literals, not re-derivable'
    failure class). This RE-RUNS the committed rederive and asserts the regenerated band == the
    committed JSON to 1e-9. Runs by default (fail-safe); skips gracefully only when the mock
    inputs are absent (clean clone off the data mounts) or SKIP_REDERIVE is set."""
    if os.environ.get("SKIP_REDERIVE"):
        pytest.skip("SKIP_REDERIVE set (fast-iteration opt-out).")
    committed = _committed_json(FWD_REL)
    assert committed is not None, f"{FWD_REL} not committed (see T1)."
    inputs = committed["metadata"].get("inputs", {})
    for key in ("catalog_dir", "loa0_product"):
        val = inputs.get(key)
        if not val:
            continue
        path0 = str(val).split()[0].strip()
        if path0.startswith("/") and not os.path.exists(path0):
            pytest.skip(f"mock input {key} absent on this host ({path0}); cannot re-derive.")
    out = tmp_path / "fwd_rerun.json"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", HDF5_USE_FILE_LOCKING="FALSE")
    r = subprocess.run(
        [sys.executable, "CDDF_analysis/diagnostics/subdla/subdla_loa0_validation.py",
         "--resp-kind", "forward", "--force", "--out", str(out)],
        cwd=_REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and out.exists(), (
        f"rederive command failed (rc={r.returncode}): {r.stderr[-600:]}")
    fresh = json.loads(out.read_text())["integrated"]["loa0"]
    comm = committed["integrated"]["loa0"]
    # dN/dX is B16-CLEAN: it must still re-derive to the committed value bit-for-bit.
    assert fresh["r0_dndx_195_203"] == pytest.approx(comm["r0_dndx_195_203"], abs=1e-9), (
        f"re-derived r0_dndx_195_203={fresh['r0_dndx_195_203']} != committed "
        f"{comm['r0_dndx_195_203']} -> the committed forward JSON is STALE w.r.t. the routine.")
    assert fresh["r0_dndx_195_203"] == pytest.approx(FWD_R0_DNDX, abs=1e-9)
    # B16 RE-BASELINE (this commit): Omega DELIBERATELY no longer matches the committed
    # artifact. The artifact holds the PRE-B16 (z-leaky truth) value and is marked
    # INVALIDATED_OMEGA; the fixed routine emits B16_FWD_R0_OMEGA. Pin BOTH ends so neither
    # can drift silently, and pin the direction (the fix RAISES R0 because truth FELL).
    assert comm["r0_omega_195_203"] == pytest.approx(FWD_R0_OMEGA, abs=1e-9), (
        "the committed forward artifact no longer holds the PRE-B16 Omega ratio; it must be "
        "kept verbatim until it is re-derived (marked, never rewritten in place).")
    assert fresh["r0_omega_195_203"] == pytest.approx(B16_FWD_R0_OMEGA, abs=1e-9), (
        f"re-derived r0_omega_195_203={fresh['r0_omega_195_203']} != the B16-fixed target "
        f"{B16_FWD_R0_OMEGA}. Either B16 regressed or the truth reduction changed again.")
    assert fresh["omega_tru_195_203"] == pytest.approx(B16_TRUTH_OMEGA_195_203, rel=1e-12)
    assert fresh["dndx_tru_195_203"] == pytest.approx(TRUTH_DNDX_195_203, rel=1e-12), (
        "B16 must not move dN/dX -- it was never leaky.")


# ===========================================================================
# AC17 (panel-5 ESCAPE-2) -- every stamped dep unchanged between stamp and HEAD
# ===========================================================================
def test_forward_all_deps_unchanged_between_stamp_and_head():
    """AC5 checks drift only on the single rederive routine; the artifact stamps 7 deps, 5 of
    them unfrozen. If ANY dep changed between the stamp commit and HEAD, HEAD can no longer
    reproduce the committed number. Require every stamped dep's blob to be identical at stamp
    and HEAD (extends the ROUTINE-DRIFT guard to the full dep set)."""
    committed = _committed_json(FWD_REL)
    assert committed is not None, f"{FWD_REL} not committed (see T1)."
    base = _base_commit(committed["metadata"].get("code_commit", ""))
    # B16 RE-BASELINE (this commit): the two estimator deps DID drift -- that is the fix.
    # An artifact whose deps drifted can no longer be reproduced at HEAD, which is exactly
    # why it is marked INVALIDATED_OMEGA. So the invariant becomes conditional: drift is
    # tolerated ONLY for the two B16 sites AND ONLY while the marker is present. Any other
    # dep drift, or a missing marker, is still fatal.
    b16 = committed["metadata"].get("b16") or {}
    b16_sites = set(b16.get("sites_fixed", []))
    b16_files = {s.split("::")[0] for s in b16_sites}
    drifted = []
    for dep in committed["metadata"].get("deps", []):
        bs = _git(["rev-parse", f"{base}:{dep}"]).stdout.strip()
        bh = _git(["rev-parse", f"HEAD:{dep}"]).stdout.strip()
        if bs and bh and bs != bh:
            drifted.append(dep)
    # NON-COMPUTING dep: CDDF_analysis/unblind/provenance.py is a GUARD, not an
    # estimator -- the routine imports exactly one symbol from it, the fail-closed
    # assert_forward_kernel, which either raises or returns "forward" and contributes
    # nothing to any number. A blanket allowlist would be too weak (the module could
    # change the guard itself), so the drift is excused ONLY when that function's
    # source is BYTE-IDENTICAL at the stamp and at HEAD.
    _GUARD = "CDDF_analysis/unblind/provenance.py"
    _guard_src = _fn_source(base, _GUARD, "assert_forward_kernel")
    if (_GUARD in drifted and _guard_src is not None
            and _guard_src == _fn_source("HEAD", _GUARD, "assert_forward_kernel")):
        drifted.remove(_GUARD)
    # BAND-ESTIMAND RETIREMENT (PI, 2026-07-28): the retirement flipped band_recenter
    # True -> False (and gated the recenter primitive) in several stamped deps. That is a
    # SECOND enumerated exception, and it is excused per-dep ONLY when every changed line
    # is a band-retirement line -- so the drift provably cannot move a reported number.
    unexplained = [d for d in drifted
                   if d not in b16_files and not _drift_is_enumerated_exception(d, base)]
    assert not unexplained, (
        f"stamped deps changed between stamp {base[:12]} and HEAD: {unexplained} -> HEAD "
        "cannot reproduce the committed forward number, and the drift is NEITHER the "
        "sanctioned B16 fix NOR the enumerated band-estimand retirement.")
    if drifted:
        assert str(b16.get("status", "")).startswith("INVALIDATED_OMEGA"), (
            f"deps {drifted} drifted (the B16 fix) but the artifact is not marked "
            "INVALIDATED_OMEGA -> a stale Omega could be quoted as current.")


# ===========================================================================
# AC18 (panel-5 ESCAPE-4) -- ALL truth denominators are kernel-independent
# ===========================================================================
def test_all_truth_denominators_kernel_independent():
    """AC7 pins only the two 195_203 integrated truth denominators; a truth/pathlength regression
    in any other (dndx_tru_203, dndx_tru_195_200, per-bin dndx_tru/f_tru) would slip through R0
    ratios. Truth is kernel-INDEPENDENT, so EVERY truth field must be byte-identical forward vs
    posterior."""
    fwd = _committed_json(FWD_REL)
    post = _committed_json(POST_REL)
    assert fwd is not None and post is not None, "both artifacts must be committed."
    fi, pi = fwd["integrated"]["loa0"], post["integrated"]["loa0"]
    for k in fi:
        if "tru" in k:
            assert fi[k] == pi.get(k), (
                f"integrated truth {k} differs forward vs posterior ({fi[k]} != {pi.get(k)}); "
                "truth is kernel-independent and must be identical.")
    for bf, bp in zip(fwd["per_bin"]["loa0"], post["per_bin"]["loa0"]):
        for k in bf:
            if "tru" in k:
                assert bf[k] == bp.get(k), (
                    f"per-bin truth {k} differs forward vs posterior; must be identical.")
