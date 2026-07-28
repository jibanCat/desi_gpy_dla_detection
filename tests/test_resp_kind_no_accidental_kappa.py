"""No paper-facing entry point may resolve resp_kind to the GP-posterior ('kappa') kernel.

WHY
---
``HBIConfig.resp_kind`` defaults to ``"kappa"`` -- the GP-POSTERIOR kernel, which Track-C
established is the WRONG OBJECT for the catalog-HBI CDDF (DLA-tier R0>=20.3 ~1.16
posterior vs ~1.04 forward).  An audit of both worktrees found that **not one** of the 22
``HBIConfig(...)`` construction sites passes ``resp_kind``: the forward kernel is never a
construction-time choice, always a post-hoc mutation.  So "forgot the mutation" ==
"silently kappa", and that is exactly how the RETIRED sub-DLA anchor (0.883/0.899) was
produced.

This file is the standing tripwire.  It has two halves:

  PART A (static, no heavy imports) -- an AST audit of every place in CDDF_analysis where
  ``resp_kind`` can be RESOLVED WITHOUT THE CALLER STATING IT: an argparse ``default=``,
  a ``getattr(x, "resp_kind", <fallback>)`` fallback, a function-signature default, and a
  dataclass field default.  Every such site that resolves to ``'kappa'`` must be on
  KAPPA_DEFAULT_ALLOWLIST with a written reason.  A NEW one fails the test.

  An EXPLICIT call-site literal (e.g. ``validate_against_committed(..., resp_kind="kappa")``
  in the --gate path) is deliberately NOT flagged: a stated kappa diagnostic is legitimate;
  an unstated one is not.  The distinction this test enforces is stated-vs-silent.

  PART B (runtime) -- the fail-closed guards in CDDF_analysis/unblind/resp_kind.py actually
  refuse: no default, kappa rejected when paper_facing=True, and an artifact that does not
  self-declare metadata['resp_kind'] is rejected rather than assumed forward.

All values referenced here are MOCK (2LPT-0) recovery ratios -- public-OK.
"""
from __future__ import annotations

import ast
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.unblind import resp_kind as RK              # noqa: E402
from CDDF_analysis.unblind.provenance import ProvenanceError   # noqa: E402

_PKG = os.path.join(_REPO, "CDDF_analysis")

# ---------------------------------------------------------------------------
# The PAPER-FACING entry points.  A resolution of 'kappa' in any of these is the
# defect this file exists to prevent -- they are never allowlisted.
# ---------------------------------------------------------------------------
PAPER_FACING_ENTRY_POINTS = (
    "CDDF_analysis/hbi/track_c_tf_loa.py",                                  # DLA headline
    "CDDF_analysis/diagnostics/subdla/run_subdla_headline_full.py",         # sub-DLA headline
    "CDDF_analysis/diagnostics/bal_metal_fp/arbiter/run_loa0_headline_full.py",
    "CDDF_analysis/diagnostics/bal_metal_fp/arbiter/apply_broadtrough_veto_headline.py",
)

# ---------------------------------------------------------------------------
# The ONLY files still permitted to resolve resp_kind to 'kappa' by default, each
# with the reason it cannot simply be changed.  Allowlisted BY FILE, not by line:
# these files are edited concurrently by other workstreams and line numbers move.
# ---------------------------------------------------------------------------
KAPPA_DEFAULT_ALLOWLIST = {
    "CDDF_analysis/hbi/cddf_catalog_hbi.py": (
        "HBIConfig.resp_kind + the three internal dispatch fallbacks. This file is pinned "
        "BYTE-FOR-BYTE at commit 8816e1e by tests/test_subdla_forward_headline.py::"
        "test_frozen_files_unchanged_by_forward_switch (the forward switch must stay "
        "CONFIG-ONLY and touch no estimator code). Flipping the default to 'forward' would "
        "also make all 22 HBIConfig() construction sites raise on the missing "
        "kernel_forward_model, breaking every legitimate kappa diagnostic while making "
        "zero paper-facing paths more correct -- none of them rely on the default, they "
        "all mutate. Enforcement lives at the entry-point/stamp layer instead "
        "(CDDF_analysis/unblind/resp_kind.py)."
    ),
    "CDDF_analysis/diagnostics/subdla/subdla_loa0_validation.py": (
        "This routine is the committed re-deriver for BOTH sub-DLA validation artifacts. "
        "Its kappa default is what makes the RETIRED subdla_mock_validation.json's own "
        "stamped `rederive` command reproduce it; changing the default would silently "
        "break the retired artifact's re-derivability. It is ALSO routine-drift-pinned: "
        "subdla_mock_validation_forward.json names this file in its `rederive`, and "
        "tests/test_subdla_forward_headline.py asserts routine_drift is False, so any "
        "COMMITTED edit here invalidates the committed forward headline's provenance. "
        "It is safe because the routine STAMPS metadata['resp_kind'] either way, so its "
        "output always self-declares, and no paper-facing path calls it without an "
        "explicit --resp-kind."
    ),
}


# ---------------------------------------------------------------------------
# PART A -- static audit
# ---------------------------------------------------------------------------
def _iter_py():
    for dp, _, fns in os.walk(_PKG):
        for fn in sorted(fns):
            if fn.endswith(".py"):
                yield os.path.join(dp, fn)


def _silent_resolution_sites():
    """Every site where resp_kind is resolved WITHOUT the caller stating it.

    Returns a list of (kind, repo_rel_path, lineno, resolved_value).
    """
    hits = []
    for path in _iter_py():
        rel = os.path.relpath(path, _REPO)
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except SyntaxError:                                  # pragma: no cover
            continue
        for node in ast.walk(tree):
            # (a) argparse:  add_argument("--resp-kind", ..., default=X)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
                if any(str(f).lstrip("-").replace("-", "_") == "resp_kind" for f in flags):
                    for kw in node.keywords:
                        if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                            hits.append(("argparse-default", rel, node.lineno, kw.value.value))
            # (b) getattr(x, "resp_kind", FALLBACK)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr" and len(node.args) == 3
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "resp_kind"
                    and isinstance(node.args[2], ast.Constant)):
                hits.append(("getattr-fallback", rel, node.lineno, node.args[2].value))
            # (c) def f(..., resp_kind=X)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                pairs = list(zip(a.args[len(a.args) - len(a.defaults):], a.defaults))
                pairs += [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
                for arg, dflt in pairs:
                    if arg.arg == "resp_kind" and isinstance(dflt, ast.Constant):
                        hits.append(("signature-default", rel, node.lineno, dflt.value))
            # (d) dataclass field:  resp_kind: str = X
            if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                    and node.target.id == "resp_kind" and isinstance(node.value, ast.Constant)):
                hits.append(("field-default", rel, node.lineno, node.value.value))
    return hits


def test_no_new_silent_kappa_default():
    """THE tripwire: no file outside the written allowlist may resolve resp_kind to
    'kappa' without the caller stating it."""
    offenders = [h for h in _silent_resolution_sites()
                 if h[3] == RK.RESP_KIND_KAPPA and h[1] not in KAPPA_DEFAULT_ALLOWLIST]
    assert not offenders, (
        "new SILENT kappa default(s) -- resp_kind resolves to the WRONG (GP-posterior) "
        "kernel with nobody saying so:\n"
        + "\n".join(f"  {k:18} {p}:{ln} -> {v!r}" for k, p, ln, v in offenders)
        + "\n\nState the kernel explicitly at the call site, or route the resolution "
          "through CDDF_analysis/unblind/resp_kind.resolve_resp_kind(..., paper_facing=...) "
          "which has NO default. If a file genuinely must keep a kappa default, add it to "
          "KAPPA_DEFAULT_ALLOWLIST in this test WITH the reason.")


@pytest.mark.parametrize("entry", PAPER_FACING_ENTRY_POINTS)
def test_paper_facing_entry_point_never_resolves_to_kappa(entry):
    """No PAPER-FACING entry point may resolve resp_kind to 'kappa' by default.

    This is the assertion the task names directly: it FAILS if any paper-facing entry
    point resolves resp_kind to 'kappa'.  Explicit, stated `resp_kind="kappa"` call-site
    literals (the --gate config-drift tripwire against the retired artifact) are NOT
    flagged -- only silent resolutions are.
    """
    assert os.path.exists(os.path.join(_REPO, entry)), f"entry point missing: {entry}"
    assert entry not in KAPPA_DEFAULT_ALLOWLIST, (
        f"{entry} is paper-facing; it must never be allowlisted for a kappa default.")
    bad = [h for h in _silent_resolution_sites()
           if h[1] == entry and h[3] == RK.RESP_KIND_KAPPA]
    assert not bad, (
        f"PAPER-FACING entry point {entry} resolves resp_kind to 'kappa':\n"
        + "\n".join(f"  {k:18} line {ln} -> {v!r}" for k, _, ln, v in bad))


def test_allowlist_entries_are_real_and_still_needed():
    """The allowlist is a ledger, not a graveyard: every entry must name a file that
    exists AND still actually carries a kappa default (otherwise delete the entry)."""
    sites = _silent_resolution_sites()
    for rel, reason in KAPPA_DEFAULT_ALLOWLIST.items():
        assert os.path.exists(os.path.join(_REPO, rel)), f"allowlisted file gone: {rel}"
        assert len(reason) > 80, f"allowlist entry {rel} lacks a real reason"
        still = [h for h in sites if h[1] == rel and h[3] == RK.RESP_KIND_KAPPA]
        assert still, (
            f"{rel} no longer has any kappa default -- remove it from "
            f"KAPPA_DEFAULT_ALLOWLIST so the ledger stays honest.")


def test_validate_against_committed_has_no_resp_kind_default():
    """The sub-DLA committed-artifact gate must REQUIRE the kernel.

    Its resp_kind used to default to 'kappa', so `validate_against_committed(modes)`
    silently gated the run against the RETIRED posterior artifact and reported PASS."""
    src = open(os.path.join(
        _REPO, "CDDF_analysis/diagnostics/subdla/run_subdla_headline_full.py"),
        encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "validate_against_committed")
    assert "resp_kind" in [a.arg for a in fn.args.kwonlyargs], (
        "resp_kind must be a KEYWORD-ONLY parameter of validate_against_committed")
    i = [a.arg for a in fn.args.kwonlyargs].index("resp_kind")
    assert fn.args.kw_defaults[i] is None, (
        "validate_against_committed.resp_kind must have NO default -- it previously "
        "defaulted to 'kappa', silently targeting the RETIRED artifact.")
    assert not any(a.arg == "resp_kind" for a in fn.args.args), (
        "resp_kind must not be positional (a positional could be filled by accident).")


# ---------------------------------------------------------------------------
# PART B -- the runtime guards actually refuse
# ---------------------------------------------------------------------------
def test_resolve_resp_kind_has_no_default_at_all():
    with pytest.raises(ProvenanceError):
        RK.resolve_resp_kind(context="unit", paper_facing=True)
    with pytest.raises(ProvenanceError):
        RK.resolve_resp_kind(context="unit", paper_facing=False)
    with pytest.raises(ProvenanceError):
        RK.resolve_resp_kind(None, context="unit", paper_facing=False)


def test_resolve_resp_kind_refuses_kappa_when_paper_facing():
    assert RK.resolve_resp_kind("forward", context="unit", paper_facing=True) == "forward"
    assert RK.resolve_resp_kind("kappa", context="unit", paper_facing=False) == "kappa"
    with pytest.raises(ProvenanceError, match="PAPER-FACING"):
        RK.resolve_resp_kind("kappa", context="unit", paper_facing=True)
    with pytest.raises(ProvenanceError, match="unknown resp_kind"):
        RK.resolve_resp_kind("posterior", context="unit", paper_facing=False)


def test_resolve_reads_a_cfg_like_object():
    import types
    cfg_f = types.SimpleNamespace(resp_kind="forward")
    cfg_k = types.SimpleNamespace(resp_kind="kappa")
    cfg_u = types.SimpleNamespace()                      # never mutated -> unset
    assert RK.resolve_resp_kind(cfg_f, context="unit", paper_facing=True) == "forward"
    with pytest.raises(ProvenanceError):
        RK.resolve_resp_kind(cfg_k, context="unit", paper_facing=True)
    with pytest.raises(ProvenanceError):
        # a bare namespace carries no resp_kind; it must NOT silently become 'kappa'
        RK.resolve_resp_kind(cfg_u, context="unit", paper_facing=False)


def test_kernel_metadata_self_declares_and_refuses_kappa_paper_facing():
    md = RK.kernel_metadata("forward", context="unit", paper_facing=True)
    assert md["resp_kind"] == "forward" and md["paper_facing"] is True
    assert "FORWARD" in md["kernel_note"]
    md = RK.kernel_metadata("kappa", context="unit", paper_facing=False)
    assert md["resp_kind"] == "kappa" and md["paper_facing"] is False
    assert "NOT PAPER-FACING" in md["kernel_note"]
    with pytest.raises(ProvenanceError, match="PAPER-FACING"):
        RK.kernel_metadata("kappa", context="unit", paper_facing=True)


def test_assert_artifact_kernel_rejects_undeclared_artifacts():
    with pytest.raises(ProvenanceError, match="self-declare"):
        RK.assert_artifact_kernel({"what": "some artifact"}, context="unit")
    with pytest.raises(ProvenanceError, match="self-declare"):
        RK.assert_artifact_kernel({"resp_kind": None}, context="unit")
    # a labelled kappa diagnostic reads fine ...
    assert RK.assert_artifact_kernel(
        {"resp_kind": "kappa", "paper_facing": False}, context="unit") == "kappa"
    # ... but cannot be promoted into a paper-facing number
    with pytest.raises(ProvenanceError):
        RK.assert_artifact_kernel({"resp_kind": "kappa", "paper_facing": False},
                                  context="unit", require_paper_facing=True)
    # and a self-contradictory artifact (paper_facing on kappa) is refused outright
    with pytest.raises(ProvenanceError, match="PAPER-FACING"):
        RK.assert_artifact_kernel({"resp_kind": "kappa", "paper_facing": True},
                                  context="unit")
    assert RK.assert_artifact_kernel(
        {"resp_kind": "forward", "paper_facing": True},
        context="unit", require_paper_facing=True) == "forward"


def test_retired_kappa_artifact_is_rejected_by_the_read_guard():
    """The RETIRED subdla_mock_validation.json carries NO metadata.resp_kind, so the
    read guard must reject it rather than let a consumer assume forward.  Its live
    successor must pass.  (MOCK 2LPT-0 values; public-OK.)"""
    import json
    retired = os.path.join(_REPO, "CDDF_analysis/hbi/subdla_mock_validation.json")
    forward = os.path.join(_REPO, "CDDF_analysis/hbi/subdla_mock_validation_forward.json")
    if not (os.path.exists(retired) and os.path.exists(forward)):
        pytest.skip("sub-DLA validation artifacts not present")
    md_r = json.load(open(retired))["metadata"]
    assert md_r.get("retired") is True, "the posterior artifact must stay labelled retired"
    with pytest.raises(ProvenanceError, match="self-declare"):
        RK.assert_artifact_kernel(md_r, context="retired sub-DLA posterior artifact")
    md_f = json.load(open(forward))["metadata"]
    assert RK.assert_artifact_kernel(
        md_f, context="forward sub-DLA anchor") == "forward"


# ---------------------------------------------------------------------------
# PART C -- the kappa DIAGNOSTICS are labelled and cannot self-promote
# ---------------------------------------------------------------------------
KAPPA_DIAGNOSTICS = (
    "CDDF_analysis/diagnostics/subdla/subdla_basis_pad_bracket.py",
    "CDDF_analysis/diagnostics/subdla/subdla_floor_mc_band.py",
)


@pytest.mark.parametrize("rel", KAPPA_DIAGNOSTICS)
def test_kappa_diagnostics_declare_themselves(rel):
    """Both sub-DLA MC diagnostics run on the kappa kernel. They must say so in the
    module banner, STATE the kernel on cfg (not inherit it), and stamp
    paper_facing=False via RK.kernel_metadata -- which raises if anyone flips
    PAPER_FACING to True while RESP_KIND stays 'kappa'."""
    src = open(os.path.join(_REPO, rel), encoding="utf-8").read()
    assert "KAPPA-KERNEL DIAGNOSTIC" in src, f"{rel} lacks the kappa-diagnostic banner"
    assert "NOT PAPER-FACING" in src, f"{rel} does not declare itself non-paper-facing"
    assert "RESP_KIND = RK.RESP_KIND_KAPPA" in src, f"{rel} does not STATE its kernel"
    assert "PAPER_FACING = False" in src, f"{rel} does not pin PAPER_FACING = False"
    assert "cfg.resp_kind = RESP_KIND" in src, (
        f"{rel} still inherits the HBIConfig default instead of stating it on cfg")
    assert "RK.kernel_metadata(" in src, f"{rel} does not stamp the kernel into metadata"
    # the guard is real: this is the call those modules make, with the flag flipped.
    with pytest.raises(ProvenanceError, match="PAPER-FACING"):
        RK.kernel_metadata(RK.RESP_KIND_KAPPA, context=rel, paper_facing=True)
