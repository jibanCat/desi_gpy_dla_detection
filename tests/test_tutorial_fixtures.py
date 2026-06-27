"""Tests for the tutorial-fixture resolver shim
(``CDDF_analysis/hbi/tutorial_data/fixtures.py``).

Contract under test:
  * Default mode (``$CDDF_STORE`` unset) ALWAYS returns the committed
    ``tutorial_data/`` path — byte-identical behaviour to before the shim.
  * With ``$CDDF_STORE`` set AND a matching committed leaf, returns the leaf path
    (handling the in-leaf filename rename).
  * Any store miss / ambiguity / missing-file falls back silently to committed.

None of these require the real scratch store to exist: the store-hit test builds
a minimal fake store via the public ``ResultStore`` API in a ``tmp_path``.
"""
from __future__ import annotations

import os

import pytest

from CDDF_analysis.hbi.tutorial_data.fixtures import (
    tutorial_fixture,
    TUTORIAL_DATA_DIR,
)


# --------------------------------------------------------------------------- #
# default mode: $CDDF_STORE unset -> always the committed path                 #
# --------------------------------------------------------------------------- #
def test_default_mode_returns_committed_path(monkeypatch):
    monkeypatch.delenv("CDDF_STORE", raising=False)
    for fn in (
        "forward_response_2lpt0.npz",
        "znz_2lpt0.npz",
        "molly_matrix_nhi195_lyaonly.tsv",
        "loa0_fp_product_lyaonly1025.npz",
        "recovery_mock_data.csv",  # unmapped fixture
    ):
        got = tutorial_fixture(fn)
        expected = str(TUTORIAL_DATA_DIR / fn)
        assert got == expected, f"default mode must return committed path for {fn}"


def test_default_mode_committed_files_exist(monkeypatch):
    """Sanity: the committed fixtures the shim points at actually exist."""
    monkeypatch.delenv("CDDF_STORE", raising=False)
    for fn in ("forward_response_2lpt0.npz", "znz_2lpt0.npz"):
        assert os.path.exists(tutorial_fixture(fn))


# --------------------------------------------------------------------------- #
# store-hit: $CDDF_STORE set + a real leaf -> the leaf path                    #
# --------------------------------------------------------------------------- #
def _make_fake_store(root):
    """Build a minimal but real store via the public ResultStore API, with a
    single 2lpt0/kernel leaf carrying forward_response_2lpt0.npz and a
    2lpt0/completeness leaf carrying the renamed molly_matrix.tsv."""
    from CDDF_analysis.results_store import ResultStore

    store = ResultStore(root=str(root))

    # kernel leaf: in-leaf filename == committed filename.
    leaf = store.new(
        dataset="2lpt0", stage="kernel", producer="fake",
        config={"kind": "forward"}, inputs=[], privacy="mock",
    )
    with open(leaf.path("forward_response_2lpt0.npz"), "wb") as f:
        f.write(b"FAKE-FORWARD")
    store.commit_leaf(
        leaf, what="fake forward kernel", cli="x", regen_cmd="x",
        outputs=[("forward_response_2lpt0.npz", "fake")],
    )

    # completeness leaf: in-leaf filename is the RENAMED molly_matrix.tsv.
    leaf2 = store.new(
        dataset="2lpt0", stage="completeness", producer="fake",
        config={"snr_min": 2.0}, inputs=[], privacy="mock",
    )
    with open(leaf2.path("molly_matrix.tsv"), "w") as f:
        f.write("FAKE-MOLLY\n")
    store.commit_leaf(
        leaf2, what="fake molly", cli="x", regen_cmd="x",
        outputs=[("molly_matrix.tsv", "fake")],
    )
    return store


def test_store_hit_returns_leaf_path(monkeypatch, tmp_path):
    store_root = tmp_path / "cddf_store"
    _make_fake_store(store_root)
    monkeypatch.setenv("CDDF_STORE", str(store_root))

    # mapped fixture, same in-leaf name -> store path.
    got = tutorial_fixture("forward_response_2lpt0.npz")
    assert str(store_root) in got
    assert got.endswith("forward_response_2lpt0.npz")
    assert os.path.exists(got)
    with open(got, "rb") as f:
        assert f.read() == b"FAKE-FORWARD"


def test_store_hit_handles_rename(monkeypatch, tmp_path):
    """The committed name molly_matrix_nhi195_lyaonly.tsv maps to the in-leaf
    molly_matrix.tsv — the shim must return the renamed file in the leaf."""
    store_root = tmp_path / "cddf_store"
    _make_fake_store(store_root)
    monkeypatch.setenv("CDDF_STORE", str(store_root))

    got = tutorial_fixture("molly_matrix_nhi195_lyaonly.tsv")
    assert str(store_root) in got
    assert got.endswith("molly_matrix.tsv")
    assert os.path.exists(got)


# --------------------------------------------------------------------------- #
# fallbacks: store set but the leaf is missing / file absent / unmapped        #
# --------------------------------------------------------------------------- #
def test_store_miss_falls_back_to_committed(monkeypatch, tmp_path):
    """$CDDF_STORE points at an EMPTY store -> no matching leaf -> committed."""
    store_root = tmp_path / "empty_store"
    monkeypatch.setenv("CDDF_STORE", str(store_root))

    got = tutorial_fixture("forward_response_2lpt0.npz")
    assert got == str(TUTORIAL_DATA_DIR / "forward_response_2lpt0.npz")
    assert os.path.exists(got)


def test_unmapped_fixture_with_store_falls_back(monkeypatch, tmp_path):
    """A fixture not in the stage map (a validation/diagnostic table) always
    resolves to committed, even with a populated store."""
    store_root = tmp_path / "cddf_store"
    _make_fake_store(store_root)
    monkeypatch.setenv("CDDF_STORE", str(store_root))

    got = tutorial_fixture("recovery_mock_data.csv")
    assert got == str(TUTORIAL_DATA_DIR / "recovery_mock_data.csv")


def test_leaf_present_but_file_absent_falls_back(monkeypatch, tmp_path):
    """A matching leaf exists but does not contain the requested file -> the
    shim must fall back to the committed copy, not return a non-existent path."""
    from CDDF_analysis.results_store import ResultStore

    store_root = tmp_path / "cddf_store"
    store = ResultStore(root=str(store_root))
    # fp leaf exists but we DON'T write loa0_fp_product.npz into it.
    leaf = store.new(
        dataset="2lpt0", stage="fp", producer="fake",
        config={"snr_min": 2.0}, inputs=[], privacy="mock",
    )
    store.commit_leaf(
        leaf, what="empty fp", cli="x", regen_cmd="x",
        outputs=[("loa0_fp_product.npz", "fake")],
    )
    monkeypatch.setenv("CDDF_STORE", str(store_root))

    got = tutorial_fixture("loa0_fp_product_lyaonly1025.npz")
    assert got == str(TUTORIAL_DATA_DIR / "loa0_fp_product_lyaonly1025.npz")


def test_real_loa_leaf_falls_back_to_committed(monkeypatch, tmp_path):
    """REGRESSION (Fix 5): a store leaf for the queried (dataset, stage) whose
    privacy is real-LOA must NEVER be handed to a notebook — the shim drops to the
    committed mock fixture even though the leaf exists AND contains the file."""
    from CDDF_analysis.results_store import ResultStore

    store_root = tmp_path / "cddf_store"
    store = ResultStore(root=str(store_root))
    # A 2lpt0/kernel leaf declared real-LOA (e.g. contagion from a real input).
    leaf = store.new(
        dataset="2lpt0", stage="kernel", producer="fake",
        config={"kind": "forward"}, inputs=[], privacy="real-LOA",
    )
    with open(leaf.path("forward_response_2lpt0.npz"), "wb") as f:
        f.write(b"REAL-LOA-SECRET")
    store.commit_leaf(
        leaf, what="real kernel", cli="x", regen_cmd="x",
        outputs=[("forward_response_2lpt0.npz", "real")],
    )
    monkeypatch.setenv("CDDF_STORE", str(store_root))

    got = tutorial_fixture("forward_response_2lpt0.npz")
    committed = str(TUTORIAL_DATA_DIR / "forward_response_2lpt0.npz")
    assert got == committed, "must not return the real-LOA leaf path"
    # and confirm it never points at the real_loa subtree.
    assert "real_loa" not in got
    with open(got, "rb") as f:
        assert f.read() != b"REAL-LOA-SECRET"


def test_real_loa_partition_path_not_returned(monkeypatch, tmp_path):
    """Even if a (typo'd) leaf class read 'mock' but the leaf physically lives
    under the real_loa/ partition, the shim still refuses it (path component
    check)."""
    from CDDF_analysis.results_store import ResultStore

    store_root = tmp_path / "cddf_store"
    store = ResultStore(root=str(store_root))
    leaf = store.new(
        dataset="2lpt0", stage="kernel", producer="fake",
        config={"kind": "forward"}, inputs=[], privacy="real-LOA",
    )
    with open(leaf.path("forward_response_2lpt0.npz"), "wb") as f:
        f.write(b"SECRET")
    store.commit_leaf(
        leaf, what="real kernel", cli="x", regen_cmd="x",
        outputs=[("forward_response_2lpt0.npz", "real")],
    )
    # the leaf must have landed under real_loa/ (sanity for the test premise).
    assert "real_loa" in leaf.dir
    monkeypatch.setenv("CDDF_STORE", str(store_root))

    got = tutorial_fixture("forward_response_2lpt0.npz")
    assert got == str(TUTORIAL_DATA_DIR / "forward_response_2lpt0.npz")
    assert "real_loa" not in got


def test_ambiguous_leaves_fall_back(monkeypatch, tmp_path):
    """>1 leaf for (dataset, stage) -> ResultStore.get raises LookupError -> the
    shim falls back silently to committed (does not propagate the error)."""
    from CDDF_analysis.results_store import ResultStore

    store_root = tmp_path / "cddf_store"
    store = ResultStore(root=str(store_root))
    for cfg in ({"kind": "a"}, {"kind": "b"}):
        leaf = store.new(
            dataset="2lpt0", stage="kernel", producer="fake",
            config=cfg, inputs=[], privacy="mock",
        )
        with open(leaf.path("forward_response_2lpt0.npz"), "wb") as f:
            f.write(b"X")
        store.commit_leaf(
            leaf, what="dup", cli="x", regen_cmd="x",
            outputs=[("forward_response_2lpt0.npz", "fake")],
        )
    monkeypatch.setenv("CDDF_STORE", str(store_root))

    got = tutorial_fixture("forward_response_2lpt0.npz")
    assert got == str(TUTORIAL_DATA_DIR / "forward_response_2lpt0.npz")
