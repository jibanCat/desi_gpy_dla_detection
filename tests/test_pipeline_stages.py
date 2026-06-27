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
    # reduction depends (transitively) on kernel_fwd ONLY. completeness_molly is a
    # standalone reportable stage, NOT a build-dependency of reduction (the forward
    # fit is non-circular + reduction uses the committed nhi19.5 molly fixture).
    assert "reduction" in order
    assert "kernel_fwd" in order
    assert "completeness_molly" not in order
    pos = {n: i for i, n in enumerate(order)}
    assert pos["kernel_fwd"] < pos["reduction"]
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
    # the dependency chain appears, in order, before reduction. completeness_molly is
    # NOT a reduction dependency (dropped — non-circular kernel + nhi19.5 fixture).
    assert "completeness_molly" not in text
    i_fwd = text.index("kernel_fwd")
    i_red = text.index("reduction", i_fwd)  # the run line, after kernel_fwd
    assert i_fwd < i_red
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


# --------------------------------------------------------------------------- #
# config-aware --resume (fix #1) — exercised WITHOUT running any producer       #
# --------------------------------------------------------------------------- #
from CDDF_analysis.results_store import ResultStore   # noqa: E402
from pipeline import run_pipeline as RP                # noqa: E402


def _commit_synthetic_leaf(store, ds, sc, *, config=None):
    """Commit a leaf for stage-config ``sc`` using ``config`` (defaults to sc.config)
    WITHOUT invoking the producer — a stand-in for 'this stage already ran'."""
    cfg = sc.config if config is None else config
    leaf = store.new(dataset=ds.name, stage=sc.store_stage, producer=sc.producer,
                     config=cfg, inputs=[], privacy=sc.privacy,
                     producer_defaults=sc.producer_defaults)
    # write a stub payload so the leaf dir is non-empty (mirrors a real producer).
    with open(leaf.path("result.json"), "w") as fh:
        fh.write("{}")
    store.commit_leaf(leaf, what="synthetic test leaf", cli="(test)",
                      outputs=[("result.json", "stub")], regen_cmd="(test)")
    return leaf


def test_resume_skips_only_the_exact_config_leaf(tmp_path):
    # SAME config → resume skips; a CHANGED knob → a different hash → does NOT skip.
    store = ResultStore(root=str(tmp_path / "store"))
    ds = DS.dataset_inputs("2lpt0")
    sc = ST.stage_config("reduction", ds)

    # nothing committed yet → not done.
    assert RP._already_done(store, ds, "reduction") is False

    # commit the EXACT current config → resume sees it as done.
    _commit_synthetic_leaf(store, ds, sc)
    assert RP._already_done(store, ds, "reduction") is True

    # now simulate an OLD leaf built from a CHANGED knob (different n_mc): a stale leaf
    # with a different hash. The current config's leaf id no longer matches it, so a
    # fresh store containing ONLY that stale leaf is NOT 'done'.
    store2 = ResultStore(root=str(tmp_path / "store2"))
    stale_cfg = dict(sc.config)
    stale_cfg["n_mc"] = sc.config["n_mc"] + 7   # a science-knob edit
    _commit_synthetic_leaf(store2, ds, sc, config=stale_cfg)
    # a leaf EXISTS for (2lpt0, measurement) in store2, but with the wrong hash.
    assert store2.get(dataset=ds.name, stage="measurement") is not None
    # config-aware resume must NOT treat the stale-config leaf as done.
    assert RP._already_done(store2, ds, "reduction") is False


def test_resume_config_aware_for_fast_stage(tmp_path):
    # the same property holds for a fast deterministic stage (kernel_fwd).
    store = ResultStore(root=str(tmp_path / "store"))
    ds = DS.dataset_inputs("2lpt0")
    sc = ST.stage_config("kernel_fwd", ds)
    assert RP._already_done(store, ds, "kernel_fwd") is False
    _commit_synthetic_leaf(store, ds, sc)
    assert RP._already_done(store, ds, "kernel_fwd") is True

    store2 = ResultStore(root=str(tmp_path / "store2"))
    stale = dict(sc.config); stale["fit_floor"] = 19.7   # changed science knob
    _commit_synthetic_leaf(store2, ds, sc, config=stale)
    assert RP._already_done(store2, ds, "kernel_fwd") is False


def test_resume_cluster_only_stage_never_done(tmp_path):
    # cluster-only stages have no in-session leaf to resume on.
    store = ResultStore(root=str(tmp_path / "store"))
    ds = DS.dataset_inputs("2lpt0")
    assert ST.stage_config("kernel_sir", ds) is None
    assert ST.stage_leaf_id(store, "kernel_sir", ds) is None
    assert RP._already_done(store, ds, "kernel_sir") is False


# --------------------------------------------------------------------------- #
# config honesty + completeness (fix #3)                                        #
# --------------------------------------------------------------------------- #
def test_reduction_config_pins_science_knobs(tmp_path):
    # editing a pinned science knob must change the leaf hash (no stale-leaf reuse).
    from CDDF_analysis.hbi.provenance import config_hash
    ds = DS.dataset_inputs("2lpt0")
    base = ST.stage_config("reduction", ds).config
    # the science knobs the reduction hardcodes must be in the config now.
    for knob in ("fit_floor", "lambda_bspbody", "sigma_slope", "slope_edge",
                 "band_recenter", "seed", "host_truth_floor", "family", "fp_estimator"):
        assert knob in base, f"reduction config missing pinned knob {knob!r}"
    h0 = config_hash(base)
    for knob, newval in [("fit_floor", 19.7), ("sigma_slope", 0.4),
                         ("lambda_bspbody", 25.0), ("band_recenter", False)]:
        changed = dict(base); changed[knob] = newval
        assert config_hash(changed) != h0, f"changing {knob} did not re-hash"


def test_phase3d_config_does_not_record_unpassed_n_mc(tmp_path):
    # fit_map/band do NOT pass --n-mc/--fp-estimator, so the config must not LIE by
    # recording a fixed n_mc the producer default actually governs.
    ds = DS.dataset_inputs("2lpt0")
    for stage in ("fit_map", "band"):
        cfg = ST.stage_config(stage, ds).config
        assert "n_mc" not in cfg, f"{stage} records an n_mc it never passes"
        assert "fp_estimator" not in cfg, f"{stage} records an fp_estimator it never passes"


def test_reduction_records_n_mc_it_actually_passes(tmp_path):
    # reduction DOES pass n_mc (60), so recording it is honest, and it must equal the
    # value the wrapper actually uses.
    ds = DS.dataset_inputs("2lpt0")
    cfg = ST.stage_config("reduction", ds).config
    assert cfg["n_mc"] == ST._RED_N_MC == 60


# --------------------------------------------------------------------------- #
# producer failure cleans up the orphan leaf dir (fix #6)                        #
# --------------------------------------------------------------------------- #
def test_run_or_cleanup_removes_uncommitted_leaf_on_failure(tmp_path):
    store = ResultStore(root=str(tmp_path / "store"))
    ds = DS.dataset_inputs("2lpt0")
    sc = ST.stage_config("kernel_fwd", ds)
    leaf = store.new(dataset=ds.name, stage=sc.store_stage, producer=sc.producer,
                     config=sc.config, inputs=[], privacy=sc.privacy,
                     producer_defaults=sc.producer_defaults)
    assert os.path.isdir(leaf.dir)   # store.new mkdir'd it.

    def _boom():
        # producer writes a partial file then dies before commit_leaf.
        with open(leaf.path("partial.npz"), "w") as fh:
            fh.write("x")
        raise RuntimeError("producer blew up")

    with pytest.raises(RuntimeError, match="blew up"):
        ST._run_or_cleanup(leaf, _boom)
    # the orphaned (uncommitted) leaf dir is gone — no litter.
    assert not os.path.isdir(leaf.dir)
    # and no manifest row was written (get() stays safe).
    with pytest.raises(LookupError):
        store.get(dataset=ds.name, stage=sc.store_stage)


def test_run_or_cleanup_keeps_dir_on_success(tmp_path):
    store = ResultStore(root=str(tmp_path / "store"))
    ds = DS.dataset_inputs("2lpt0")
    sc = ST.stage_config("kernel_fwd", ds)
    leaf = store.new(dataset=ds.name, stage=sc.store_stage, producer=sc.producer,
                     config=sc.config, inputs=[], privacy=sc.privacy,
                     producer_defaults=sc.producer_defaults)
    ST._run_or_cleanup(leaf, lambda: open(leaf.path("ok.npz"), "w").close())
    assert os.path.isdir(leaf.dir)
    assert os.path.exists(leaf.path("ok.npz"))


# --------------------------------------------------------------------------- #
# no phantom report.md output (fix #4)                                          #
# --------------------------------------------------------------------------- #
def test_reduction_does_not_advertise_report_md():
    # the reduction wrapper must not list a report.md output it never writes. Grep the
    # source for the offending outputs row (cheap, no producer run).
    src = open(os.path.join(REPO, "pipeline", "stages.py")).read()
    # the only commit_leaf for reduction is in run_reduction; report.md must not be an
    # advertised output anywhere in the stage outputs.
    assert '("report.md"' not in src, "stages.py still advertises a phantom report.md output"


# --------------------------------------------------------------------------- #
# deferred message no longer references a non-existent --ingest flag (fix #5)    #
# --------------------------------------------------------------------------- #
def test_deferred_message_has_no_ingest_flag(tmp_path):
    out = _run_cli(
        ["--dataset", "2lpt0", "--stage", "band", "--cluster-emit"],
        env={"CDDF_STORE": str(tmp_path / "store")},
    )
    assert out.returncode == 0, out.stderr
    assert "deferred" in out.stdout.lower()
    assert "--ingest" not in out.stdout, "deferred message references a non-existent --ingest flag"
