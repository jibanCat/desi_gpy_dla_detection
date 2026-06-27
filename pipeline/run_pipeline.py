"""HBI results-store pipeline — DAG driver + CLI (task 5 of
``pipeline/IMPLEMENTATION_PLAN.md``).

Topologically sorts the stage dependency graph (``pipeline.stages.STAGES``) and runs
the requested stage(s) for a dataset, writing each into a write-once
``ResultStore`` leaf with a provenance stamp.

CLI::

    python -m pipeline.run_pipeline --dataset 2lpt0 --stage kernel_fwd
    python -m pipeline.run_pipeline --dataset 2lpt0 --stage all --dry-run
    python -m pipeline.run_pipeline --dataset real_loa --stage reduction --store /path

Flags:
  --dataset      one of pipeline.datasets.DATASETS (2lpt0, real_loa, 2lpt1, london0)
  --stage        a single stage name, or ``all`` (the whole DAG, topo-sorted)
  --store        the store root (else $CDDF_STORE)
  --resume       skip a stage ONLY when ITS exact config-hash leaf is already
                 committed; a changed science knob → new hash → the stage re-runs
  --dry-run      print the topo order + the leaf ids it WOULD create; run nothing
  --cluster-emit make cluster_only stages print their sbatch line instead of raising
"""
from __future__ import annotations

import argparse
import sys
import time

from CDDF_analysis.results_store import ResultStore
from pipeline import datasets as DS
from pipeline import stages as ST


# --------------------------------------------------------------------------- #
# dry-run leaf-id preview                                                      #
# --------------------------------------------------------------------------- #
def _preview_leaf_id(store: ResultStore, ds: DS.DatasetInputs, stage_name: str) -> str:
    """Best-effort: the EXACT leaf id a stage WOULD create, without running it.
    Cluster-only stages have no leaf (return a marker)."""
    stage = ST.get_stage(stage_name)
    if stage.cluster_only:
        return "<cluster-only: no in-session leaf>"
    # the stage exposes the config it would build, so we can compute the exact
    # slug/hash addressed leaf id (the same one --resume checks).
    try:
        leaf_id = ST.stage_leaf_id(store, stage_name, ds)
        if leaf_id is not None:
            return leaf_id
    except Exception:
        pass  # fall back to the prefix form if config-building is unavailable.
    store_stage = _store_stage_for(stage_name)
    sub = "real_loa" if ds.privacy == "real-LOA" else "mock"
    return f"{sub}/{ds.name}/{store_stage}/<slug>__<hash8>"


# the store-stage path component each pipeline stage writes under (some differ from
# the stage name, e.g. completeness_molly -> completeness).
_STORE_STAGE = {
    "completeness_molly": "completeness",
    "kernel_fwd": "kernel",
    "kernel_znz": "kernel_znz",
    "fp_loa0": "fp",
    "kernel_remp": "kernel_remp",
    "kernel_sir": "kernel_sir",
    "fit_map": "fit_map",
    "band": "band",
    "reduction": "measurement",
}


def _store_stage_for(stage_name: str) -> str:
    return _STORE_STAGE.get(stage_name, stage_name)


def _already_done(store: ResultStore, ds: DS.DatasetInputs, stage_name: str) -> bool:
    """True iff THIS stage's exact config-hash leaf already exists + is committed.

    Used by --resume. Config-aware: it computes the leaf id the stage WOULD create
    for the current config (via ``stages.stage_leaf_id``) and resume-skips ONLY when
    that exact ``<slug>__<hash>`` leaf is committed. Editing a science knob changes
    the hash → a different (not-yet-built) leaf id → NOT done → the stage re-runs into
    a fresh leaf. (A stale leaf from the OLD config still exists, but it no longer
    matches, so resume never silently reuses it.)"""
    leaf_id = ST.stage_leaf_id(store, stage_name, ds)
    if leaf_id is None:  # cluster-only stage: never an in-session leaf to resume.
        return False
    try:
        leaf = store.by_id(leaf_id)
    except LookupError:
        return False
    return leaf.status in ("current", "superseded")


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="dataset id: " + ", ".join(sorted(DS.DATASETS)))
    p.add_argument("--stage", required=True,
                   help="a single stage name, or 'all' for the whole topo-sorted DAG")
    p.add_argument("--store", default=None,
                   help="store root (else $CDDF_STORE)")
    p.add_argument("--resume", action="store_true",
                   help="skip a stage only when its exact config-hash leaf is committed "
                        "(a changed knob re-hashes → the stage re-runs)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the topo order + would-create leaf ids; run nothing")
    p.add_argument("--cluster-emit", action="store_true",
                   help="cluster_only stages print their sbatch line instead of raising")
    args = p.parse_args(argv)

    ds = DS.dataset_inputs(args.dataset)

    # resolve the stage list (topo-sorted).
    if args.stage == "all":
        order = ST.stage_order()
    else:
        ST.get_stage(args.stage)  # validate
        order = ST.stage_order(targets=[args.stage])

    # ---- dry-run: print the plan, run nothing ------------------------------
    if args.dry_run:
        print(f"[dry-run] dataset={args.dataset} stage={args.stage}")
        print(f"[dry-run] topo order ({len(order)} stage(s)):")
        # the dry-run preview does not need a real store dir; build one only if a
        # root is available (so --dry-run works without $CDDF_STORE).
        store = None
        root = args.store or __import__("os").environ.get("CDDF_STORE")
        if root:
            store = ResultStore(root=root)
        for i, name in enumerate(order, 1):
            st = ST.get_stage(name)
            flags = []
            if st.cluster_only:
                flags.append("cluster_only")
            if st.heavy:
                flags.append("heavy")
            flag_txt = (" [" + ",".join(flags) + "]") if flags else ""
            leaf_id = (_preview_leaf_id(store, ds, name) if store
                       else f"{_store_stage_for(name)}/<slug>__<hash8>")
            print(f"  {i}. {name}{flag_txt}  ->  would create: {leaf_id}")
        return 0

    # ---- real run ----------------------------------------------------------
    store = ResultStore(root=args.store)
    overall_t0 = time.time()
    # stages whose in-session output does NOT exist this run because they were
    # cluster-emitted (or transitively depend on one). Their descendants cannot run
    # in-session either — skip them as deferred instead of crashing on a missing leaf.
    deferred: set[str] = set()
    for name in order:
        stage = ST.get_stage(name)
        blocking = set(stage.deps) & deferred
        if blocking:
            print(f"[deferred] SKIP {name} — needs cluster output of "
                  f"{', '.join(sorted(blocking))} (run those on the cluster; the produced "
                  f"leaf is then registered in the store)")
            deferred.add(name)
            continue
        if args.resume and _already_done(store, ds, name):
            print(f"[resume] SKIP {name} (committed leaf already exists for {args.dataset})")
            continue
        print(f"\n=== RUN stage={name} dataset={args.dataset} "
              f"({'cluster_only' if stage.cluster_only else ('heavy' if stage.heavy else 'in-session')}) ===")
        t0 = time.time()
        try:
            leaf = stage.run(store, ds, resume=args.resume, cluster_emit=args.cluster_emit)
        except ST.ClusterOnlyStage as e:
            print(f"[cluster-only] {name}: submit on the cluster with:\n  {e.sbatch_cmd}")
            print("[cluster-only] (pass --cluster-emit to emit the sbatch line and continue)")
            return 2
        wall = time.time() - t0
        if leaf is None:  # cluster_only under --cluster-emit
            print(f"[{name}] emitted (no in-session leaf)  wall={wall:.1f}s")
            deferred.add(name)  # its descendants are now cluster-blocked too
        else:
            print(f"[{name}] DONE  leaf={leaf.id}  wall={wall:.1f}s")
    if deferred:
        print(f"\n[pipeline] {len(deferred)} stage(s) deferred to the cluster: "
              f"{', '.join(sorted(deferred))}")
    print(f"\n[pipeline] in-session stages done  total_wall={time.time()-overall_t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
