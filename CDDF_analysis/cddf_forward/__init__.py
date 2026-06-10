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
from .filter_guard import assert_filter_off
from .split import (
    sightline_role,
    assign_roles,
    split_masks,
    SplitProvenance,
    assert_no_leakage,
)
from .driver import compute_o1_products, save_o1_products

__all__ = [
    "WindowSpec",
    "assert_filter_off",
    "sightline_role",
    "assign_roles",
    "split_masks",
    "SplitProvenance",
    "assert_no_leakage",
    "compute_o1_products",
    "save_o1_products",
]
