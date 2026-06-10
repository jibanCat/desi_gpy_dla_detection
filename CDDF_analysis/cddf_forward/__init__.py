"""``cddf_forward`` — forward-model plumbing for the DLA CDDF pipeline.

M0 ships:

* the shared search-window spec (``WindowSpec``), the single source of truth for
  the DLA search window applied identically to measurement / truth / injection
  (``window.py``); and
* the FILTER-off guard (``assert_filter_off``), refusing FILTER-on catalogs for
  which the CDDF is invalid (``filter_guard.py``); and
* the deterministic, TARGETID-keyed train/test split (``split.py``) that keeps the
  response-matrix BUILD set disjoint from the closure-validation HELDOUT set, with
  a no-leakage guard so non-circularity is enforced, not merely intended.

M1 ships the O1 end-to-end CDDF driver (``driver.py``): a faithful wrapper over the
Pathway-A estimator (``calc_cddf.DLACatalogue``) that emits the raw probabilistic
CDDF f(N), dN/dX, and Omega_DLA (no selection correction yet) and saves text tables.
"""

from .window import WindowSpec
from .filter_guard import (
    assert_filter_off,
    read_filter_flag,
    assert_filter_off_from_file,
)
from .split import (
    sightline_role,
    assign_roles,
    split_masks,
    SplitProvenance,
    assert_no_leakage,
)
from .driver import (
    compute_o1_products,
    save_o1_products,
    compute_o3_products,
    heldout_closure,
    save_o3_products,
    plot_o3_products,
)
from .soft_completeness import (
    estimate_diagonal_completeness,
    estimate_false_positive_deposit,
    apply_diagonal_correction,
    toy_count_mock,
    sbc_coverage,
)
from .diagonal_deposit import (
    build_truth_map,
    TruthMap,
    DiagonalSoftDeposit,
)
from .streaming import (
    compute_o1_products_streaming,
    compute_o3_products_streaming,
    heldout_closure_streaming,
)

__all__ = [
    "WindowSpec",
    "assert_filter_off",
    "read_filter_flag",
    "assert_filter_off_from_file",
    "sightline_role",
    "assign_roles",
    "split_masks",
    "SplitProvenance",
    "assert_no_leakage",
    "compute_o1_products",
    "save_o1_products",
    # O3 diagonal soft-completeness (M2)
    "compute_o3_products",
    "heldout_closure",
    "save_o3_products",
    "plot_o3_products",
    "estimate_diagonal_completeness",
    "estimate_false_positive_deposit",
    "apply_diagonal_correction",
    "toy_count_mock",
    "sbc_coverage",
    "build_truth_map",
    "TruthMap",
    "DiagonalSoftDeposit",
    # no-combine streaming pipeline
    "compute_o1_products_streaming",
    "compute_o3_products_streaming",
    "heldout_closure_streaming",
]
