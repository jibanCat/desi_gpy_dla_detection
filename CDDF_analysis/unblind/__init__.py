"""CDDF_analysis.unblind -- shared foundation for the DLA / sub-DLA unblinding notebooks.

Thin, testable helpers behind ``notebooks/UNBLIND_00_guards_and_data.ipynb``:

  provenance   -- the RE_DERIVABLE provenance guard (refuses unstamped/dirty/orphaned).
  schema       -- structural validator for the headline JSON (shapes only, no values).
  loader       -- guarded, tidy loader -> HeadlineData (+ 3 distinct per-z regime flags).
  systematics  -- carried DLA-tier systematics table, as data, VERIFIED/UNVERIFIED.
  privacy      -- self-check that no real-LOA numerics survive in committed nb outputs.

Nothing in this package prints a real-LOA science value.
"""

from __future__ import annotations

from . import loader, privacy, provenance, schema, systematics
from .loader import (
    DEFAULT_LOA0_ARTIFACT,
    DEFAULT_LOA0_ROUTINE,
    DEFAULT_PURITY_ARTIFACT,
    DEFAULT_PURITY_ROUTINE,
    HeadlineData,
    ZBinFlags,
    load_headline,
)
from .privacy import assert_no_outputs, scan_notebook_outputs
from .provenance import (
    ProvenanceError,
    ProvenanceResult,
    assert_forward_kernel,
    check_artifact,
    classify,
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
    "provenance", "schema", "loader", "systematics", "privacy",
    "classify", "check_artifact", "assert_forward_kernel",
    "ProvenanceResult", "ProvenanceError",
    "RE_DERIVABLE", "NOT_STAMPED", "DIRTY", "ORPHANED", "COMMIT_NOT_FOUND",
    "NO_ROUTINE", "NOT_ANCESTOR",
    "validate_headline_schema", "SchemaError", "SchemaReport",
    "load_headline", "HeadlineData", "ZBinFlags",
    "DEFAULT_LOA0_ARTIFACT", "DEFAULT_LOA0_ROUTINE",
    "DEFAULT_PURITY_ARTIFACT", "DEFAULT_PURITY_ROUTINE",
    "carried_systematics", "Systematic",
    "assert_no_outputs", "scan_notebook_outputs",
]
