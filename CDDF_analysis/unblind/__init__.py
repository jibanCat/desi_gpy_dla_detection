"""CDDF_analysis.unblind -- shared foundation for the DLA / sub-DLA unblinding notebooks.

Thin, testable helpers behind ``notebooks/UNBLIND_00_guards_and_data.ipynb``:

  provenance   -- the RE_DERIVABLE provenance guard (refuses unstamped/dirty/orphaned).
  schema       -- structural validator for the headline JSON (shapes only, no values).
  loader       -- guarded, tidy loader -> HeadlineData (+ 3 distinct per-z regime flags).
  systematics  -- carried DLA-tier systematics table, as data, VERIFIED/UNVERIFIED.
  privacy      -- self-check that no real-LOA numerics survive in committed notebook
                  outputs OR committed JSON artifacts (TARGETID magnitude, real path
                  tokens, real-value co-occurrence).
  audit        -- repo-wide provenance audit over every committed JSON artifact in
                  BOTH Paper-1 worktrees: `python -m CDDF_analysis.unblind.audit`.
  estimand     -- the band-estimand vocabulary + classifier behind the PI's 2026-07-28
                  retirement (only a joint-posterior credible interval whose median IS
                  the reported point may be paper-facing).

Nothing in this package prints a real-LOA science value.
"""

from __future__ import annotations

from . import audit, estimand, loader, privacy, provenance, schema, systematics
from .audit import AuditRow, audit_worktree, render_table, summarize
from .estimand import (
    DIAGNOSTIC_RECENTERED,
    ESTIMAND_DETAIL_KEY,
    ESTIMAND_VOCABULARY,
    MARGINAL_COMBINED,
    PAPER_FACING_ESTIMANDS,
    PLUGIN_MAP_MC,
    POINT_ONLY,
    POSTERIOR_MEDIAN_CI,
    UNKNOWN,
    RecenteredBandRetired,
    assert_paper_facing,
    band_estimand,
    classify_estimand,
    is_paper_facing,
    mark_retired,
    normalize_estimand_metadata,
    normalize_estimand_stamp,
    stamp_band_estimand,
    BAND_BEARING_ESTIMANDS,
    PAPER_FACING_REFUSED_KEY,
    DECLARATION_ABSENT,
    DECLARATION_FALSE,
    DECLARATION_TRUE,
    DECLARATION_UNPARSEABLE,
    parse_paper_facing_declaration,
)
from .loader import (
    DEFAULT_LOA0_ARTIFACT,
    DEFAULT_LOA0_ROUTINE,
    DEFAULT_PURITY_ARTIFACT,
    DEFAULT_PURITY_ROUTINE,
    HeadlineData,
    ZBinFlags,
    load_headline,
)
from .privacy import (
    PrivacyError,
    RealDataHit,
    assert_json_artifact_mock_only,
    assert_no_outputs,
    scan_json_artifact,
    scan_notebook_outputs,
)
from .provenance import (
    ProvenanceError,
    ProvenanceResult,
    assert_forward_kernel,
    check_artifact,
    classify,
    load_stamp_block,
    stamp_block,
    stamp_kind,
    ABBREVIATED_SHA,
    ALL_STATUSES,
    CANONICAL_STAMP_KEY,
    MOVABLE_REF,
    RE_DERIVABLE,
    NOT_STAMPED,
    DIRTY,
    ORPHANED,
    COMMIT_NOT_FOUND,
    NO_ROUTINE,
    NOT_ANCESTOR,
)
from .schema import SchemaError, SchemaReport, validate_headline_schema
from .systematics import Systematic, carried_systematics

__all__ = [
    "provenance", "schema", "loader", "systematics", "privacy", "audit", "estimand",
    "classify", "check_artifact", "assert_forward_kernel",
    # -- band-estimand vocabulary + classifier (PI retirement, 2026-07-28) --
    "classify_estimand", "band_estimand", "stamp_band_estimand",
    "assert_paper_facing", "is_paper_facing", "mark_retired",
    "normalize_estimand_stamp", "normalize_estimand_metadata",
    "RecenteredBandRetired", "ESTIMAND_VOCABULARY", "ESTIMAND_DETAIL_KEY",
    "PAPER_FACING_ESTIMANDS",
    "POSTERIOR_MEDIAN_CI", "PLUGIN_MAP_MC", "DIAGNOSTIC_RECENTERED",
    "MARGINAL_COMBINED", "POINT_ONLY", "UNKNOWN",
    # -- fail-closed hardening of the producer veto (referee, 2026-07-29) --
    "parse_paper_facing_declaration", "BAND_BEARING_ESTIMANDS",
    "PAPER_FACING_REFUSED_KEY", "DECLARATION_TRUE", "DECLARATION_FALSE",
    "DECLARATION_ABSENT", "DECLARATION_UNPARSEABLE",
    "load_stamp_block", "stamp_block", "stamp_kind",
    "ProvenanceResult", "ProvenanceError",
    "RE_DERIVABLE", "NOT_STAMPED", "DIRTY", "ORPHANED", "COMMIT_NOT_FOUND",
    "NO_ROUTINE", "NOT_ANCESTOR", "MOVABLE_REF", "ABBREVIATED_SHA",
    "ALL_STATUSES", "CANONICAL_STAMP_KEY",
    "AuditRow", "audit_worktree", "render_table", "summarize",
    "PrivacyError", "RealDataHit",
    "assert_json_artifact_mock_only", "scan_json_artifact",
    "validate_headline_schema", "SchemaError", "SchemaReport",
    "load_headline", "HeadlineData", "ZBinFlags",
    "DEFAULT_LOA0_ARTIFACT", "DEFAULT_LOA0_ROUTINE",
    "DEFAULT_PURITY_ARTIFACT", "DEFAULT_PURITY_ROUTINE",
    "carried_systematics", "Systematic",
    "assert_no_outputs", "scan_notebook_outputs",
]
