"""Tutorial-fixture resolver shim (additive, optional store path).

The HBI tutorial notebooks (NB0–NB5) read their inputs from the committed,
version-controlled mock fixtures under ``CDDF_analysis/hbi/tutorial_data/``.
That is the default, byte-identical, no-scratch-floating-files behaviour and it
is what every notebook used through 7 review rounds.

This shim adds a *purely optional* second source: a fresh recompute in a keyed
results store (``$CDDF_STORE``; see ``CDDF_analysis/results_store.py``). When
``$CDDF_STORE`` is set AND a matching committed leaf exists for a fixture, the
shim points the notebook at the freshly-recomputed copy in the store leaf;
otherwise — and ALWAYS when ``$CDDF_STORE`` is unset — it returns the committed
``tutorial_data/`` path.

Design contract
---------------
* **Default mode (``$CDDF_STORE`` unset) returns the committed path, always.**
  The notebooks load byte-identically to today; this module changes nothing.
* The store is *optional and best-effort*: ANY failure to resolve a store leaf
  (no ``$CDDF_STORE``; ``ResultStore`` import/construction error; ``LookupError``
  on 0 / >1 matches; the named file missing inside the leaf) falls back silently
  to the committed fixture.
* No heavy producers are imported. Only ``CDDF_analysis.results_store`` (stdlib
  sqlite/json) is touched, and only when ``$CDDF_STORE`` is set.

Each committed fixture maps to a store ``stage`` (and, where the leaf stores the
file under a different name, the in-leaf filename). Fixtures that have no
in-session store leaf (the validation / diagnostic tables) always resolve to the
committed path.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["tutorial_fixture", "TUTORIAL_DATA_DIR"]

# This module lives in CDDF_analysis/hbi/tutorial_data/; the committed fixture
# directory is exactly this module's directory.
TUTORIAL_DATA_DIR = Path(__file__).resolve().parent


# Map each committed fixture filename -> (store stage, in-leaf filename).
# The in-leaf filename differs from the committed name where the producer writes
# a generic name into the leaf (e.g. molly_matrix.tsv, loa0_fp_product.npz).
# Fixtures NOT in this map have no in-session store leaf and always resolve to
# the committed copy.
_FIXTURE_STAGE = {
    "forward_response_2lpt0.npz":        ("kernel",       "forward_response_2lpt0.npz"),
    "znz_2lpt0.npz":                     ("kernel_znz",   "znz_2lpt0.npz"),
    "molly_matrix_nhi195_lyaonly.tsv":   ("completeness", "molly_matrix.tsv"),
    "loa0_fp_product_lyaonly1025.npz":   ("fp",           "loa0_fp_product.npz"),
}

# Default dataset for the in-session leaves (2LPT-0 mock recompute).
_DEFAULT_DATASET = "2lpt0"


def _committed_path(filename: str) -> str:
    """Absolute path to the committed tutorial_data fixture."""
    return str(TUTORIAL_DATA_DIR / filename)


def _store_path(filename: str, dataset, stage, selectors) -> str | None:
    """Try to resolve ``filename`` from a ``$CDDF_STORE`` leaf.

    Returns the absolute path to the file inside the resolved leaf, or ``None``
    on any miss/ambiguity/error (caller falls back to the committed copy). Never
    raises; never imports a heavy producer.
    """
    if not os.environ.get("CDDF_STORE"):
        return None  # default mode: no store configured.

    # Resolve the stage + in-leaf name. Explicit stage= wins; else use the map.
    mapped = _FIXTURE_STAGE.get(filename)
    if stage is None:
        if mapped is None:
            return None  # no in-session leaf for this fixture.
        stage = mapped[0]
    in_leaf_name = mapped[1] if mapped is not None else filename

    ds = dataset if dataset is not None else _DEFAULT_DATASET

    try:
        # Local import so module load never requires the store package; this is
        # stdlib-only (sqlite/json) and imports no producers.
        from CDDF_analysis.results_store import ResultStore

        store = ResultStore()  # reads $CDDF_STORE
        # ResultStore.get accepts (dataset, stage, selection). Pass selection
        # only if the caller supplied one; the single-leaf-per-stage layout
        # resolves uniquely without it.
        selection = selectors.get("selection") if selectors else None
        leaf = store.get(dataset=ds, stage=stage, selection=selection)
    except Exception:
        # No $CDDF_STORE, import failure, LookupError (0 / >1 matches), bad
        # manifest — fall back silently to the committed fixture.
        return None

    leaf_file = leaf.path(in_leaf_name)
    if not os.path.exists(leaf_file):
        return None  # leaf exists but this file is not in it -> committed copy.
    return leaf_file


def tutorial_fixture(filename, *, dataset=None, stage=None, **selectors):
    """Resolve a tutorial input to an absolute path.

    Parameters
    ----------
    filename : str
        The committed fixture filename (e.g. ``"forward_response_2lpt0.npz"``).
    dataset : str, optional
        Store dataset (defaults to ``"2lpt0"`` for the in-session leaves).
    stage : str, optional
        Store stage. If omitted, inferred from ``filename`` via the built-in map.
    **selectors
        Forwarded to ``ResultStore.get`` (only ``selection=`` is consulted).

    Returns
    -------
    str
        ``<leaf.dir>/<in_leaf_name>`` when ``$CDDF_STORE`` is set AND a matching
        committed leaf (with the file) exists; otherwise the committed
        ``CDDF_analysis/hbi/tutorial_data/<filename>``.

    In default mode (``$CDDF_STORE`` unset) this ALWAYS returns the committed
    path — byte-identical to today.
    """
    sp = _store_path(filename, dataset, stage, selectors)
    if sp is not None:
        return sp
    return _committed_path(filename)
