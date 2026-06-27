"""Unit tests for ``CDDF_analysis/results_store.py`` — the addressable, keyed,
write-once intermediate-results store + sqlite manifest (Task 2 of the HBI
results-store pipeline; implements ``CDDF_analysis/RESULTS_STORE_PLAN.md`` §1/§3/§4).

Every test roots the store at ``tmp_path`` (via ``CDDF_STORE`` env or the ``root=``
arg) — NEVER a real scratch path or a production ``$CDDF_STORE``.
"""
import json
import os

import pytest

from CDDF_analysis.results_store import ResultStore, ResultLeaf


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A ResultStore rooted under tmp_path, via the CDDF_STORE env var."""
    root = tmp_path / "cddf_store"
    monkeypatch.setenv("CDDF_STORE", str(root))
    return ResultStore()


_DEFAULTS = {"snr_min": 0.0, "no_bal": False, "zbins": [2.0, 2.5, 3.0, 3.5, 4.0]}


def _make_and_commit(store, *, dataset, stage, producer, config, inputs,
                     privacy=None, status="current", selection_what="reduction"):
    """Helper: new leaf → write a result.json → commit. Returns the ResultLeaf."""
    leaf = store.new(dataset=dataset, stage=stage, producer=producer,
                     config=config, inputs=inputs, privacy=privacy,
                     producer_defaults=_DEFAULTS)
    with open(leaf.path("result.json"), "w") as fh:
        json.dump({"dndx": [1, 2, 3], "config": config}, fh)
    store.commit_leaf(
        leaf,
        what=f"test {selection_what} leaf",
        cli="python -m pipeline.run_pipeline --dataset %s --stage %s" % (dataset, stage),
        outputs=[("result.json", "the per-z reduction")],
        regen_cmd="python -m pipeline.run_pipeline --dataset %s --stage %s" % (dataset, stage),
        status=status,
    )
    return leaf


# --------------------------------------------------------------------------- #
# init / env                                                                   #
# --------------------------------------------------------------------------- #
def test_init_requires_root(monkeypatch):
    monkeypatch.delenv("CDDF_STORE", raising=False)
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        ResultStore()


def test_init_creates_root_and_manifest(tmp_path):
    root = tmp_path / "store"
    s = ResultStore(root=str(root))
    assert root.exists()
    assert (root / "MANIFEST.sqlite").exists()


# --------------------------------------------------------------------------- #
# leaf_path determinism + privacy split                                        #
# --------------------------------------------------------------------------- #
def test_leaf_path_deterministic(store):
    cfg = {"snr_min": 2.0, "no_bal": True, "zbins": [2.0, 2.5, 3.0, 3.5]}
    p1 = store.leaf_path("2lpt0", "reduction", "track_c_tf_loa", cfg,
                         producer_defaults=_DEFAULTS)
    p2 = store.leaf_path("2lpt0", "reduction", "track_c_tf_loa", cfg,
                         producer_defaults=_DEFAULTS)
    assert p1 == p2
    # contains slug + hash, under the mock privacy subdir + dataset + stage.
    assert "mock" in p1
    assert "2lpt0" in p1
    assert "reduction" in p1
    assert "snr2_nobal_z2-3.5" in p1


def test_leaf_path_real_loa_subdir(store):
    cfg = {"snr_min": 2.0}
    p = store.leaf_path("real_loa", "measurement", "track_c_tf_loa", cfg,
                        privacy="real-LOA", producer_defaults=_DEFAULTS)
    assert "real_loa" in p
    assert "mock" not in p.replace("real_loa", "")  # not under mock/


# --------------------------------------------------------------------------- #
# new -> commit -> get round-trip                                              #
# --------------------------------------------------------------------------- #
def test_new_commit_get_roundtrip(store):
    cfg = {"snr_min": 2.0, "no_bal": True, "zbins": [2.0, 2.5, 3.0, 3.5]}
    leaf = _make_and_commit(store, dataset="2lpt0", stage="reduction",
                            producer="track_c_tf_loa", config=cfg, inputs=[])
    got = store.get(dataset="2lpt0", stage="reduction")
    assert isinstance(got, ResultLeaf)
    assert got.id == leaf.id
    assert os.path.exists(got.path("result.json"))
    payload = json.load(open(got.path("result.json")))
    assert payload["config"]["snr_min"] == 2.0
    # provenance + privacy populated.
    assert got.config["snr_min"] == 2.0
    assert got.status == "current"
    assert got.privacy["class"] == "mock"
    assert got.commit  # a non-empty commit-stamp dict/short
    assert got.provenance["schema_version"] == "cddf-provenance/1"


def test_commit_writes_provenance_and_readme(store):
    cfg = {"snr_min": 1.0}
    leaf = _make_and_commit(store, dataset="2lpt0", stage="kernel_remp",
                            producer="run_remp_kernel", config=cfg, inputs=[])
    assert os.path.exists(leaf.path("provenance.json"))
    assert os.path.exists(leaf.path("README.md"))
    # manifest json mirror written.
    assert (store.root / "MANIFEST.json").exists()


# --------------------------------------------------------------------------- #
# get strictness                                                               #
# --------------------------------------------------------------------------- #
def test_get_zero_match_raises_with_candidates(store):
    # nothing committed for this (dataset, stage).
    _make_and_commit(store, dataset="2lpt0", stage="reduction",
                     producer="p", config={"snr_min": 2.0}, inputs=[])
    with pytest.raises(LookupError) as exc:
        store.get(dataset="2lpt0", stage="band")  # no band leaf
    # error lists candidate ids (or says none) — must be informative.
    msg = str(exc.value)
    assert "band" in msg or "candidate" in msg.lower() or "no" in msg.lower()


def test_get_two_match_raises_listing_candidates(store):
    # two leaves sharing (dataset, stage, selection) — ambiguous. The slug only
    # surfaces first/last of a list, so these two zbins collapse to the SAME slug
    # (z2-4.0) but differ in the middle edge -> different config hash -> two leaves.
    cfg_a = {"snr_min": 2.0, "no_bal": True, "zbins": [2.0, 2.5, 4.0]}
    cfg_b = {"snr_min": 2.0, "no_bal": True, "zbins": [2.0, 3.0, 4.0]}
    a = _make_and_commit(store, dataset="2lpt0", stage="reduction",
                         producer="p", config=cfg_a, inputs=[])
    b = _make_and_commit(store, dataset="2lpt0", stage="reduction",
                         producer="p", config=cfg_b, inputs=[])
    assert a.id != b.id  # genuinely two leaves
    sel = "snr2_nobal_z2-4"
    with pytest.raises(LookupError) as exc:
        store.get(dataset="2lpt0", stage="reduction", selection=sel)
    msg = str(exc.value)
    assert a.id in msg and b.id in msg


def test_get_with_selection_disambiguates(store):
    cfg_a = {"snr_min": 2.0, "no_bal": True}
    cfg_b = {"snr_min": 0.0, "no_bal": False}
    _make_and_commit(store, dataset="2lpt0", stage="reduction",
                     producer="p", config=cfg_a, inputs=[])
    _make_and_commit(store, dataset="2lpt0", stage="reduction",
                     producer="p", config=cfg_b, inputs=[])
    got = store.get(dataset="2lpt0", stage="reduction", selection="snr2_nobal")
    assert got.config["snr_min"] == 2.0


# --------------------------------------------------------------------------- #
# by_id pins exactly                                                           #
# --------------------------------------------------------------------------- #
def test_by_id_pins(store):
    leaf = _make_and_commit(store, dataset="2lpt0", stage="reduction",
                            producer="p", config={"snr_min": 2.0}, inputs=[])
    got = store.by_id(leaf.id)
    assert got.id == leaf.id
    assert os.path.exists(got.path("result.json"))


def test_by_id_missing_raises(store):
    with pytest.raises(LookupError):
        store.by_id("mock/2lpt0/reduction/nope__00000000")


# --------------------------------------------------------------------------- #
# list                                                                         #
# --------------------------------------------------------------------------- #
def test_list_filters(store):
    _make_and_commit(store, dataset="2lpt0", stage="reduction",
                     producer="track_c_tf_loa", config={"snr_min": 2.0}, inputs=[])
    _make_and_commit(store, dataset="2lpt0", stage="reduction",
                     producer="track_c_tf_loa", config={"snr_min": 0.0}, inputs=[])
    _make_and_commit(store, dataset="2lpt1", stage="band",
                     producer="track_c_band", config={"snr_min": 0.0}, inputs=[])
    assert len(store.list()) == 3
    assert len(store.list(producer="track_c_tf_loa")) == 2
    assert len(store.list(dataset="2lpt1")) == 1
    assert len(store.list(status="current")) == 3
    assert len(store.list(status="superseded")) == 0


# --------------------------------------------------------------------------- #
# privacy contagion through inputs                                             #
# --------------------------------------------------------------------------- #
def test_privacy_contagion_through_inputs(store):
    # a mock kernel leaf...
    kernel = _make_and_commit(store, dataset="2lpt0", stage="kernel_remp",
                              producer="run_remp_kernel", config={"k": 1}, inputs=[])
    # ...and a downstream leaf that takes a real-LOA external input -> real-LOA.
    leaf = store.new(
        dataset="real_loa", stage="measurement", producer="track_c_tf_loa",
        config={"snr_min": 2.0},
        inputs=[kernel.id,
                {"id": "external/loa_dlacat", "privacy": {"class": "real-LOA"}}],
        producer_defaults=_DEFAULTS,
    )
    with open(leaf.path("result.json"), "w") as fh:
        json.dump({"x": 1}, fh)
    store.commit_leaf(leaf, what="real measurement",
                      cli="cli", outputs=[("result.json", "x")], regen_cmd="cli")
    got = store.by_id(leaf.id)
    assert got.privacy["class"] == "real-LOA"
    assert got.privacy["shareable"] is False
    # and it landed under the real_loa/ subtree, not mock/.
    assert "real_loa" in got.dir


# --------------------------------------------------------------------------- #
# rebuild_manifest reconstructs from leaves alone                              #
# --------------------------------------------------------------------------- #
def test_rebuild_manifest_from_leaves(store):
    a = _make_and_commit(store, dataset="2lpt0", stage="reduction",
                         producer="p", config={"snr_min": 2.0}, inputs=[])
    b = _make_and_commit(store, dataset="2lpt1", stage="band",
                         producer="q", config={"snr_min": 0.0}, inputs=[])
    before = {r.id: (r.dataset_stage()) for r in store.list()}

    # nuke the sqlite manifest entirely — provenance.json files remain.
    sqlite_path = store.root / "MANIFEST.sqlite"
    sqlite_path.unlink()
    assert not sqlite_path.exists()

    store.rebuild_manifest()
    assert sqlite_path.exists()
    after = {r.id: (r.dataset_stage()) for r in store.list()}
    assert before == after
    # get still resolves after a rebuild-from-leaves.
    got = store.get(dataset="2lpt0", stage="reduction")
    assert got.id == a.id
    assert os.path.exists(got.path("result.json"))
    # both leaves present.
    assert {a.id, b.id} == set(after)


def test_rebuild_manifest_rows_identical(store):
    """The rebuilt rows match the originally-committed rows field-by-field."""
    _make_and_commit(store, dataset="2lpt0", stage="reduction",
                     producer="p", config={"snr_min": 2.0, "no_bal": True}, inputs=[])
    rows_before = {r.id: r.provenance for r in store.list()}
    (store.root / "MANIFEST.sqlite").unlink()
    store.rebuild_manifest()
    rows_after = {r.id: r.provenance for r in store.list()}
    assert rows_before == rows_after
