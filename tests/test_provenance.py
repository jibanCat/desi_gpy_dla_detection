"""Unit tests for ``CDDF_analysis/hbi/provenance.py`` — the shared commit-stamp
+ provenance-emit layer (Task 1 of the HBI results-store pipeline; implements
``CDDF_analysis/RESULTS_STORE_PLAN.md`` §2/§4).

Pure-stdlib module: ``git_stamp``, ``config_hash``, ``make_slug``,
``privacy_class``, ``write_provenance``. Tests use ``tmp_path`` only; the one
git-touching test (``git_stamp``) reads THIS repo (read-only) and only asserts
shape, never a specific SHA value.
"""
import json

import pytest

from CDDF_analysis.hbi.provenance import (
    git_stamp,
    config_hash,
    make_slug,
    privacy_class,
    write_provenance,
    _atomic_write,
)


# --------------------------------------------------------------------------- #
# git_stamp                                                                    #
# --------------------------------------------------------------------------- #
def test_git_stamp_real_repo_shape():
    """In this repo, git_stamp returns a real short SHA + a bool dirty flag."""
    s = git_stamp()
    assert isinstance(s, dict)
    for key in ("commit_short", "commit_long", "branch", "dirty", "diff_sha256"):
        assert key in s
    # commit_short is a real abbreviated SHA (hex, 4..40 chars) in this repo.
    assert s["commit_short"] != "unknown"
    assert all(c in "0123456789abcdef" for c in s["commit_short"])
    assert 4 <= len(s["commit_short"]) <= 40
    # commit_long is the full 40-char SHA.
    assert len(s["commit_long"]) == 40
    assert isinstance(s["dirty"], bool)
    # diff_sha256 is None when clean, a 64-hex string when dirty.
    if s["dirty"]:
        assert isinstance(s["diff_sha256"], str)
        assert len(s["diff_sha256"]) == 64
    else:
        assert s["diff_sha256"] is None


def test_git_stamp_non_repo_is_safe(tmp_path):
    """Pointed at a non-git dir, git_stamp returns 'unknown'/safe defaults, never raises."""
    s = git_stamp(repo_root=str(tmp_path))
    assert s["commit_short"] == "unknown"
    assert s["commit_long"] == "unknown"
    assert s["branch"] == "unknown"
    assert s["dirty"] is False
    assert s["diff_sha256"] is None


# --------------------------------------------------------------------------- #
# config_hash                                                                  #
# --------------------------------------------------------------------------- #
def test_config_hash_idempotent():
    cfg = {"snr_min": 2.0, "no_bal": True, "zbins": [2.0, 2.5, 3.0]}
    h1 = config_hash(cfg)
    h2 = config_hash(cfg)
    assert h1 == h2
    assert len(h1) == 8
    assert all(c in "0123456789abcdef" for c in h1)


def test_config_hash_key_order_irrelevant():
    a = {"snr_min": 2.0, "no_bal": True, "zbins": [2.0, 2.5]}
    b = {"zbins": [2.0, 2.5], "no_bal": True, "snr_min": 2.0}
    assert config_hash(a) == config_hash(b)


def test_config_hash_distinct_configs_differ():
    a = {"snr_min": 2.0, "no_bal": True}
    b = {"snr_min": 0.0, "no_bal": True}
    assert config_hash(a) != config_hash(b)


def test_config_hash_handles_nonjson_via_default_str():
    """default=str lets non-JSON-native values (e.g. tuples / paths) hash deterministically."""
    from pathlib import Path

    cfg = {"path": Path("/a/b"), "t": (1, 2, 3)}
    assert config_hash(cfg) == config_hash(dict(cfg))


# --------------------------------------------------------------------------- #
# make_slug                                                                    #
# --------------------------------------------------------------------------- #
def test_make_slug_example_from_plan():
    """The canonical example from the spec."""
    defaults = {"snr_min": 0.0, "no_bal": False, "zbins": [2.0, 2.5, 3.0, 3.5, 4.0]}
    cfg = {"snr_min": 2.0, "no_bal": True, "zbins": [2.0, 2.5, 3.0, 3.5]}
    slug = make_slug(cfg, defaults)
    assert slug == "snr2_nobal_z2-3.5"


def test_make_slug_omits_default_valued_keys():
    """A key equal to its default contributes nothing to the slug."""
    defaults = {"snr_min": 0.0, "no_bal": False}
    cfg = {"snr_min": 0.0, "no_bal": True}
    slug = make_slug(cfg, defaults)
    assert "snr" not in slug
    assert slug == "nobal"


def test_make_slug_deterministic_and_fs_safe():
    defaults = {"snr_min": 0.0, "no_bal": False, "zbins": [2.0, 4.0]}
    cfg = {"snr_min": 2.5, "no_bal": True, "zbins": [2.0, 4.25]}
    s1 = make_slug(cfg, defaults)
    s2 = make_slug(cfg, defaults)
    assert s1 == s2
    # filesystem-safe: no spaces, no slashes, no path-traversal chars.
    for bad in (" ", "/", "\\", "\n", "\t"):
        assert bad not in s1


def test_make_slug_all_defaults_is_base():
    """Nothing differs → a stable non-empty base slug (not an empty string)."""
    defaults = {"snr_min": 0.0, "no_bal": False}
    cfg = dict(defaults)
    slug = make_slug(cfg, defaults)
    assert slug  # non-empty
    assert " " not in slug and "/" not in slug


def test_make_slug_list_of_strings_does_not_crash():
    """REGRESSION (Fix 1): a list/tuple-of-strings config value (e.g.
    {"models": ["null", "dla"]}) must NOT raise ValueError from float() — it
    renders a clean, fs-safe slug by joining the stringified elements."""
    slug = make_slug({"models": ["null", "dla"]}, {})
    assert isinstance(slug, str) and slug
    # fs-safe, and the element names survive.
    for bad in (" ", "/", "\\", "\n", "\t"):
        assert bad not in slug
    assert "null" in slug and "dla" in slug


def test_make_slug_tuple_of_strings_and_mixed_lists():
    """A tuple-of-strings and a list with non-numeric entries both slug cleanly;
    an all-numeric list still uses the compact <first>-<last> form."""
    # tuple of strings.
    s_tuple = make_slug({"models": ("null", "dla", "2dla")}, {})
    assert "null" in s_tuple and "2dla" in s_tuple
    # mixed (contains a non-numeric) -> join branch, never float().
    s_mixed = make_slug({"edges": ["lo", 2.0, "hi"]}, {})
    assert "lo" in s_mixed and "hi" in s_mixed
    # all-numeric list keeps the compact first-last rendering.
    s_num = make_slug({"zbins": [2.0, 2.5, 3.0, 3.5]}, {})
    assert s_num == "z2-3.5"


def test_make_slug_list_of_strings_via_store_new_does_not_raise(tmp_path, monkeypatch):
    """REGRESSION (Fix 1): store.new(config={"models": [...]}) no longer raises.
    Drives the slug path through the real ResultStore allocation."""
    monkeypatch.setenv("CDDF_STORE", str(tmp_path / "store"))
    from CDDF_analysis.results_store import ResultStore

    store = ResultStore()
    leaf = store.new(
        dataset="2lpt0", stage="kernel", producer="p",
        config={"models": ["null", "dla"]}, inputs=[], privacy="mock",
    )
    assert "null" in leaf.dir and "dla" in leaf.dir


# --------------------------------------------------------------------------- #
# privacy_class                                                                #
# --------------------------------------------------------------------------- #
def test_privacy_class_all_mock():
    inputs = [
        {"privacy": {"class": "mock"}},
        {"privacy": {"class": "mock"}},
    ]
    p = privacy_class(inputs)
    assert p["class"] == "mock"
    assert p["shareable"] is True


def test_privacy_class_contagion():
    """One real-LOA input makes the whole result real-LOA + not shareable."""
    inputs = [
        {"privacy": {"class": "mock"}},
        {"privacy": {"class": "real-LOA"}},
    ]
    p = privacy_class(inputs)
    assert p["class"] == "real-LOA"
    assert p["shareable"] is False


def test_privacy_class_empty_is_mock():
    p = privacy_class([])
    assert p["class"] == "mock"
    assert p["shareable"] is True


@pytest.mark.parametrize("typo", ["Real-LOA", "real_loa", "REAL-LOA", "reel-loa",
                                  "real loa", " real-loa ", "secret"])
def test_privacy_class_fail_closed_on_typo(typo):
    """REGRESSION (Fix 2): an unrecognized / mis-spelled input class is FAIL-CLOSED
    — it must classify the result as real-LOA + non-shareable, never silently mock."""
    p = privacy_class([{"privacy": {"class": typo}}])
    assert p["class"] == "real-LOA", f"{typo!r} must not pass as shareable"
    assert p["shareable"] is False


def test_privacy_class_mock_spellings_are_shareable():
    """Recognized mock spellings (case/space-insensitive) stay shareable."""
    for spelling in ("mock", "Mock", "MOCK", " mock "):
        p = privacy_class([{"privacy": {"class": spelling}}])
        assert p["class"] == "mock"
        assert p["shareable"] is True


def test_privacy_class_no_signal_inputs_default_mock():
    """Inputs that carry no privacy signal at all (no privacy dict / no class /
    empty class) don't taint — a pure-mock seed stays shareable."""
    inputs = [{"id": "seed"}, {"privacy": {}}, {"privacy": {"class": ""}}]
    p = privacy_class(inputs)
    assert p["class"] == "mock"
    assert p["shareable"] is True


def test_privacy_class_typo_contagion_in_mixed_inputs():
    """One typo'd input among real mocks still poisons the whole result."""
    p = privacy_class([
        {"privacy": {"class": "mock"}},
        {"privacy": {"class": "Real_LOA"}},  # typo
    ])
    assert p["class"] == "real-LOA"
    assert p["shareable"] is False


# --------------------------------------------------------------------------- #
# write_provenance                                                             #
# --------------------------------------------------------------------------- #
def _minimal_kwargs(tmp_path, code_commit):
    return dict(
        what="2LPT-0 Track-C reduction (test fixture)",
        status="current",
        privacy={"class": "mock", "shareable": True},
        producer="track_c_tf_loa",
        config={"snr_min": 2.0, "no_bal": True},
        inputs=[{"id": "mock/2lpt0/kernel/remp__x__abcd1234",
                 "privacy": {"class": "mock"}}],
        cli="python -m pipeline.run_pipeline --dataset 2lpt0 --stage reduction",
        outputs=[("result.json", "per-z dN/dX + Omega + f(N|z)")],
        regen_cmd="python -m pipeline.run_pipeline --dataset 2lpt0 --stage reduction",
        code_commit=code_commit,
    )


def test_write_provenance_writes_both_files(tmp_path):
    cc = {"commit_short": "deadbee", "commit_long": "deadbee" * 5 + "abcde",
          "branch": "feat", "dirty": False, "diff_sha256": None}
    rec = write_provenance(str(tmp_path), **_minimal_kwargs(tmp_path, cc))
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "provenance.json").exists()
    assert isinstance(rec, dict)
    assert rec["schema_version"] == "cddf-provenance/1"


def test_write_provenance_json_roundtrips(tmp_path):
    cc = {"commit_short": "abc1234", "commit_long": "abc1234" * 5 + "fffff",
          "branch": "main", "dirty": False, "diff_sha256": None}
    rec = write_provenance(str(tmp_path), **_minimal_kwargs(tmp_path, cc))
    on_disk = json.loads((tmp_path / "provenance.json").read_text())
    assert on_disk == rec
    # required schema fields present
    for key in ("schema_version", "id", "dataset", "stage", "producer",
                "config_hash", "config", "slug", "code_commit", "date_utc",
                "inputs", "outputs", "cli", "regen_cmd", "status", "privacy",
                "supersedes", "superseded_by"):
        assert key in on_disk
    assert on_disk["supersedes"] is None
    assert on_disk["superseded_by"] is None
    # config_hash is consistent with the standalone hasher.
    assert on_disk["config_hash"] == config_hash(on_disk["config"])


def test_write_provenance_dirty_banner(tmp_path):
    cc = {"commit_short": "abc1234", "commit_long": "abc1234" * 5 + "fffff",
          "branch": "wip", "dirty": True, "diff_sha256": "f" * 64}
    write_provenance(str(tmp_path), **_minimal_kwargs(tmp_path, cc))
    readme = (tmp_path / "README.md").read_text()
    assert "DIRTY" in readme
    # clean stamp: no dirty banner.
    other = tmp_path / "clean"
    other.mkdir()
    cc_clean = dict(cc, dirty=False, diff_sha256=None)
    write_provenance(str(other), **_minimal_kwargs(other, cc_clean))
    assert "DIRTY" not in (other / "README.md").read_text()


def test_write_provenance_readme_has_outputs_table_and_regen(tmp_path):
    cc = {"commit_short": "abc1234", "commit_long": "abc1234" * 5 + "fffff",
          "branch": "main", "dirty": False, "diff_sha256": None}
    write_provenance(str(tmp_path), **_minimal_kwargs(tmp_path, cc))
    readme = (tmp_path / "README.md").read_text()
    assert "result.json" in readme           # outputs table row
    assert "per-z dN/dX" in readme            # the "what it is" text
    assert "run_pipeline" in readme           # regen command rendered
    assert "abc1234" in readme                # commit stamp


def test_write_provenance_default_code_commit_is_git_stamp(tmp_path):
    """Omitting code_commit falls back to git_stamp() of the repo (real dict, never raises)."""
    kw = _minimal_kwargs(tmp_path, code_commit=None)
    kw.pop("code_commit")
    rec = write_provenance(str(tmp_path), **kw)
    assert isinstance(rec["code_commit"], dict)
    assert "commit_short" in rec["code_commit"]


# --------------------------------------------------------------------------- #
# _atomic_write — unique tmp suffix, concurrent-write safe                     #
# --------------------------------------------------------------------------- #
def test_atomic_write_unique_tmp_suffix(tmp_path):
    """REGRESSION (Fix 3): the tmp leaf is unique per call (not a fixed
    '<name>.tmp'), so it embeds the pid + a uuid hex and leaves no fixed-name
    artifact behind."""
    target = tmp_path / "provenance.json"
    _atomic_write(target, '{"ok": true}')
    assert json.loads(target.read_text()) == {"ok": True}
    # the old fixed tmp name must not linger.
    assert not (tmp_path / "provenance.json.tmp").exists()
    # no stray tmp files at all after a clean write.
    assert not list(tmp_path.glob("provenance.json.tmp*"))


def test_atomic_write_concurrent_same_leaf_no_race(tmp_path):
    """REGRESSION (Fix 3, best-effort): many threads writing the SAME file
    concurrently must not raise (no shared fixed-name tmp -> no os.replace race),
    and the final file is valid JSON written by exactly one of them."""
    import threading

    target = tmp_path / "provenance.json"
    n = 16
    errors: list[BaseException] = []
    barrier = threading.Barrier(n)

    def worker(i: int) -> None:
        try:
            barrier.wait()
            for _ in range(20):
                _atomic_write(target, json.dumps({"writer": i}))
        except BaseException as exc:  # noqa: BLE001 - record any race failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent _atomic_write raised: {errors!r}"
    # final file is valid JSON from one writer; no stray tmp files survive.
    data = json.loads(target.read_text())
    assert "writer" in data and 0 <= data["writer"] < n
    assert not list(tmp_path.glob("provenance.json.tmp*"))
