"""Tests for the HBI pipeline stage catalog + DAG driver (task 5 of
``pipeline/IMPLEMENTATION_PLAN.md``).

These tests touch ONLY the registry + topo-sort + ``--dry-run`` — they NEVER run the
heavy producers (no GP inference, no catalog I/O). The store, when used, is rooted at
``tmp_path``.
"""
import os
import subprocess
import sys

import pytest

from pipeline import stages as ST
from pipeline import datasets as DS


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# registry well-formedness                                                     #
# --------------------------------------------------------------------------- #
def test_registry_nonempty_and_has_required_stages():
    for required in ("kernel_fwd", "reduction", "completeness_molly", "kernel_sir"):
        assert required in ST.STAGES, f"missing required stage {required!r}"


def test_every_stage_has_a_callable_run_and_valid_deps():
    for name, st in ST.STAGES.items():
        assert st.name == name
        assert callable(st.run), f"stage {name} has no run callable"
        assert st.producer, f"stage {name} has no producer id"
        for dep in st.deps:
            assert dep in ST.STAGES, f"stage {name} depends on unknown stage {dep!r}"


def test_kernel_sir_is_cluster_only():
    assert ST.STAGES["kernel_sir"].cluster_only is True
    # the other in-session stages are NOT cluster-only.
    assert ST.STAGES["kernel_fwd"].cluster_only is False
    assert ST.STAGES["reduction"].cluster_only is False


def test_heavy_flags():
    # the heavy in-session producers are flagged so callers don't run them casually.
    for name in ("kernel_remp", "fit_map", "band"):
        assert ST.STAGES[name].heavy is True, f"{name} should be heavy"
    # the fast demo stages are not heavy.
    assert ST.STAGES["kernel_fwd"].heavy is False
    assert ST.STAGES["reduction"].heavy is False


# --------------------------------------------------------------------------- #
# topo sort                                                                     #
# --------------------------------------------------------------------------- #
def test_stage_order_full_is_a_valid_topo_sort():
    order = ST.stage_order()
    assert set(order) == set(ST.STAGES)
    pos = {n: i for i, n in enumerate(order)}
    for n in ST.STAGES:
        for dep in ST.STAGES[n].deps:
            assert pos[dep] < pos[n], f"{dep} must precede {n} in topo order"


def test_stage_order_for_reduction_lists_dependencies_first():
    order = ST.stage_order(targets=["reduction"])
    # reduction depends (transitively) on kernel_fwd + completeness_molly.
    assert "reduction" in order
    assert "kernel_fwd" in order
    assert "completeness_molly" in order
    pos = {n: i for i, n in enumerate(order)}
    assert pos["completeness_molly"] < pos["kernel_fwd"] < pos["reduction"]
    # reduction is LAST in its sub-DAG.
    assert order[-1] == "reduction"
    # the sub-DAG does not pull in unrelated heavy/cluster stages.
    assert "kernel_sir" not in order
    assert "band" not in order


# --------------------------------------------------------------------------- #
# dry-run CLI (runs nothing)                                                    #
# --------------------------------------------------------------------------- #
def _run_cli(argv, env=None):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "pipeline.run_pipeline", *argv],
        cwd=REPO, env=full_env, capture_output=True, text=True,
    )


def test_dry_run_reduction_lists_dependency_order(tmp_path):
    out = _run_cli(
        ["--dataset", "2lpt0", "--stage", "reduction", "--dry-run"],
        env={"CDDF_STORE": str(tmp_path / "store")},
    )
    assert out.returncode == 0, out.stderr
    text = out.stdout
    assert "topo order" in text
    # the dependency chain appears, in order, before reduction.
    i_comp = text.index("completeness_molly")
    i_fwd = text.index("kernel_fwd")
    i_red = text.index("reduction", i_fwd)  # the run line, after kernel_fwd
    assert i_comp < i_fwd < i_red
    # dry-run creates NO leaves.
    store_dir = tmp_path / "store"
    if store_dir.exists():
        leaves = [p for p in store_dir.rglob("provenance.json")]
        assert not leaves, "dry-run must not create any leaf"


def test_dry_run_all_marks_cluster_only(tmp_path):
    out = _run_cli(
        ["--dataset", "2lpt0", "--stage", "all", "--dry-run"],
        env={"CDDF_STORE": str(tmp_path / "store")},
    )
    assert out.returncode == 0, out.stderr
    assert "kernel_sir" in out.stdout
    assert "cluster_only" in out.stdout


def test_cluster_emit_defers_descendants_instead_of_crashing(tmp_path):
    # --cluster-emit on the band path emits the kernel_sir sbatch, then SKIPS its
    # cluster-blocked descendants (fit_map, band) rather than crashing on the missing
    # cluster output. Exit 0, no in-session leaf created (no heavy producer invoked).
    out = _run_cli(
        ["--dataset", "2lpt0", "--stage", "band", "--cluster-emit"],
        env={"CDDF_STORE": str(tmp_path / "store")},
    )
    assert out.returncode == 0, out.stderr
    assert "sbatch" in out.stdout                       # emitted the cluster command
    assert "deferred" in out.stdout.lower()             # descendants deferred
    assert "fit_map" in out.stdout and "band" in out.stdout
    store_dir = tmp_path / "store"
    if store_dir.exists():
        assert not list(store_dir.rglob("provenance.json")), \
            "cluster-emit must not create any in-session leaf for the deferred path"


def test_dry_run_works_without_store_env(tmp_path):
    # --dry-run should not require a real store root.
    env = dict(os.environ)
    env.pop("CDDF_STORE", None)
    out = subprocess.run(
        [sys.executable, "-m", "pipeline.run_pipeline",
         "--dataset", "2lpt0", "--stage", "kernel_fwd", "--dry-run"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "kernel_fwd" in out.stdout
