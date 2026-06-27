"""The HBI pipeline stage catalog — thin wrappers around the existing producers
(task 4 of ``pipeline/IMPLEMENTATION_PLAN.md``).

Each stage is a ``Stage`` record (name, dependency list, producer id, in-session/
heavy/cluster flags) plus a ``run(store, ds, *, resume, cluster_emit) -> ResultLeaf``
function that:

  1. resolves upstream leaves via ``store.get(...)`` (or, for the frozen-calibration
     fixtures, the committed ``tutorial_data/`` payloads — the no-scratch fallback)
     and the dataset primary inputs (``ds.catalog_dir`` etc.);
  2. allocates a fresh write-once leaf with ``store.new(...)`` carrying the
     result-affecting config + the input descriptors (for privacy contagion);
  3. invokes the EXISTING producer, pointing its output at ``leaf.dir`` (never
     re-implementing the science);
  4. stamps provenance with ``store.commit_leaf(...)`` (README + provenance.json +
     manifest row).

This module rewrites NO science. Every producer is called via its real entry point
(``main(argv)`` / a ``build_*`` subcommand / the NB5 reduction functions). Cluster-
only stages raise ``ClusterOnlyStage`` carrying the exact sbatch line (never run it).

The wrappers DO NOT import the heavy producers at module load — each is imported
inside its ``run_*`` body so that ``--dry-run`` / the registry tests stay light and
never pull in the inference stack.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from CDDF_analysis.results_store import ResultStore, ResultLeaf
from pipeline.datasets import DatasetInputs

__all__ = [
    "Stage",
    "STAGES",
    "ClusterOnlyStage",
    "stage_order",
    "get_stage",
    "TUTORIAL_DATA",
    "external_input",
]

# the committed frozen-calibration fixtures (the no-scratch fallback the reduction
# stage grafts in — same files NB5 uses).
TUTORIAL_DATA = Path(__file__).resolve().parents[1] / "CDDF_analysis" / "hbi" / "tutorial_data"


class ClusterOnlyStage(Exception):
    """Raised by a ``cluster_only`` stage's ``run`` to signal the work must be done
    on the cluster. Carries the exact ``sbatch`` command in ``.sbatch_cmd`` so the
    driver can print it (with ``--cluster-emit``) instead of pretending to run it."""

    def __init__(self, stage: str, sbatch_cmd: str):
        self.stage = stage
        self.sbatch_cmd = sbatch_cmd
        super().__init__(
            f"stage {stage!r} is cluster-only; submit it with:\n  {sbatch_cmd}"
        )


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
# Config keys that are verbose / non-distinguishing (paths, the producer id, the
# kernel descriptor) and should NOT clutter the human-readable slug. The config
# itself still carries them (and they feed the hash); they're only dropped from the
# slug token list. _slug_defaults(config) returns a producer_defaults dict in which
# exactly these keys equal the config value (so make_slug skips them) and every other
# key is absent (so the science knobs DO render).
_SLUG_NOISE_KEYS = ("producer", "catalog", "kernel")


def _slug_defaults(config: dict) -> dict:
    """A producer_defaults dict that hides the noisy path/producer keys from the slug
    while keeping the science knobs (snr/nhi/n_mc/zbins/...) visible."""
    return {k: config[k] for k in _SLUG_NOISE_KEYS if k in config}


def external_input(path: str, *, role: str, privacy: str = "mock") -> dict:
    """Build an external-input descriptor for ``store.new(inputs=[...])``.

    External (non-leaf) inputs — the GP-DLA catalog dir, truth/bal FITS, the GP
    kernel — are recorded as dicts (vs upstream-leaf id strings). The ``privacy``
    propagates by contagion: a real-LOA catalog input → a real-LOA result.
    """
    return {"path": path, "role": role, "privacy": {"class": privacy}}


def _ds_external_inputs(ds: DatasetInputs, *roles: str) -> list:
    """The dataset's primary inputs (catalog/truth/bal) as external descriptors,
    each tagged with the dataset privacy so contagion works."""
    table = {
        "catalog": ds.catalog_dir,
        "truth": ds.truth,
        "bal": ds.bal,
    }
    return [external_input(table[r], role=r, privacy=ds.privacy) for r in roles]


def _upstream_or_none(store: ResultStore, *, dataset: str, stage: str):
    """Resolve an upstream leaf id if it exists in the store, else None (the stage
    falls back to the committed fixture / the dataset primary input)."""
    try:
        return store.get(dataset=dataset, stage=stage).id
    except LookupError:
        return None


# --------------------------------------------------------------------------- #
# stage record                                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stage:
    """One pipeline stage. ``run`` is the wrapper that does the actual work.

    Flags:
      heavy        — runs in-session but is slow (~10-30 min); a notebook/CI should
                     not call it casually, but the driver will if asked.
      cluster_only — cannot run in-session; ``run`` raises ``ClusterOnlyStage``.
    """

    name: str
    deps: tuple = ()
    producer: str = ""
    heavy: bool = False
    cluster_only: bool = False
    run: callable = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
# stage wrappers                                                               #
# --------------------------------------------------------------------------- #
def run_completeness_molly(store, ds, *, resume=False, cluster_emit=False):
    """completeness_molly → examples/molly_faithful_pc_plots.py (writes
    molly_matrix.tsv). Lyα-only window, BAL-excluded, the matched headline cut."""
    config = {
        "producer": "molly_faithful_pc_plots",
        "lam_rf_min": 1025.0, "lam_rf_max": 1216.0,
        "snr_min": 2.0, "nhi_min": 20.3, "gp_conf": 0.99,
        "no_bal": True, "dz_rel": 0.01, "z_qso_min": 2.0, "z_qso_max": 4.25,
    }
    inputs = _ds_external_inputs(ds, "catalog", "truth", "bal")
    leaf = store.new(dataset=ds.name, stage="completeness", producer="molly_faithful_pc_plots",
                     config=config, inputs=inputs, privacy=ds.privacy,
                     producer_defaults=_slug_defaults(config))
    argv = [
        "--catalog-dir", ds.catalog_dir,
        "--truth", ds.truth,
        "--bal-cat", ds.bal, "--no-bal",
        "--mockdir", ds.resolved_mockdir,
        "--lam-rf-min", "1025.0",
        "--snr-min", "2.0", "--nhi-min", "20.3", "--gp-conf", "0.99",
        "--out", leaf.dir,
    ]
    cli = "python examples/molly_faithful_pc_plots.py " + " ".join(argv)
    # molly main() reads sys.argv (no argv param) — set it transiently.
    from examples import molly_faithful_pc_plots as MOLLY  # noqa: E402
    _argv0 = sys.argv
    try:
        sys.argv = ["molly_faithful_pc_plots.py", *argv]
        MOLLY.main()
    finally:
        sys.argv = _argv0
    store.commit_leaf(
        leaf, what="2LPT-0 molly purity/completeness C/ρ matrix (Lyα-only, BAL-excl).",
        cli=cli,
        outputs=[("molly_matrix.tsv", "per-(SNR, logN) purity & completeness matrix"),
                 ("molly_summary.tsv", "headline purity/completeness summary")],
        regen_cmd=f"python -m pipeline.run_pipeline --dataset {ds.name} --stage completeness_molly",
    )
    return leaf


def run_kernel_fwd(store, ds, *, resume=False, cluster_emit=False):
    """kernel_fwd → znz_kernel.build_forward_cache (writes forward_response_2lpt0.npz).
    Deterministic Track-C forward-response kernel; reads x̂/N_true/SNR/z_QSO only."""
    out_npz = os.path.join("$LEAF", "forward_response_2lpt0.npz")  # for the README
    config = {
        "producer": "znz_kernel.build-forward-cache",
        "kind": "forward", "family": "empirical",
        "lam_rf_min": 1025.0, "fit_floor": 19.5, "deg_N": 2,
        "host_truth_floor": 19.0,
    }
    molly_id = _upstream_or_none(store, dataset=ds.name, stage="completeness")
    inputs = _ds_external_inputs(ds, "catalog", "truth", "bal")
    if molly_id:
        inputs.append(molly_id)
    leaf = store.new(dataset=ds.name, stage="kernel", producer="znz_kernel.build-forward-cache",
                     config=config, inputs=inputs, privacy=ds.privacy,
                     producer_defaults=_slug_defaults(config))
    out_path = leaf.path("forward_response_2lpt0.npz")
    argv = [
        "--catalog-dir", ds.catalog_dir,
        "--truth", ds.truth,
        "--bal-cat", ds.bal,
        "--out", out_path,
    ]
    cli = "python -m CDDF_analysis.hbi.znz_kernel build-forward-cache " + " ".join(argv)
    from CDDF_analysis.hbi import znz_kernel as ZK  # noqa: E402
    ZK.build_forward_cache(argv)
    store.commit_leaf(
        leaf, what="Track-C forward-response kernel R(x̂→N | SNR, z) on 2LPT-0 (deterministic).",
        cli=cli,
        outputs=[("forward_response_2lpt0.npz",
                  "forward-response model (mu/sig/skew coefs + emp ρ grid) consumed by run_measurement")],
        regen_cmd=f"python -m pipeline.run_pipeline --dataset {ds.name} --stage kernel_fwd",
    )
    return leaf


def run_kernel_znz(store, ds, *, resume=False, cluster_emit=False):
    """kernel_znz → znz_kernel.build_cache (writes znz_2lpt0.npz)."""
    config = {
        "producer": "znz_kernel.build-cache",
        "lam_rf_min": 1025.0, "fit_floor": 19.5, "deg_xhat": 1, "deg_z": 2,
        "host_truth_floor": 19.0,
    }
    inputs = _ds_external_inputs(ds, "catalog", "truth", "bal")
    leaf = store.new(dataset=ds.name, stage="kernel_znz", producer="znz_kernel.build-cache",
                     config=config, inputs=inputs, privacy=ds.privacy,
                     producer_defaults=_slug_defaults(config))
    out_path = leaf.path("znz_2lpt0.npz")
    argv = [
        "--catalog-dir", ds.catalog_dir, "--truth", ds.truth,
        "--bal-cat", ds.bal, "--out", out_path,
    ]
    cli = "python -m CDDF_analysis.hbi.znz_kernel build-cache " + " ".join(argv)
    from CDDF_analysis.hbi import znz_kernel as ZK  # noqa: E402
    ZK.build_cache(argv)
    store.commit_leaf(
        leaf, what="Track-C stage-0 znz kernel cache on 2LPT-0.",
        cli=cli,
        outputs=[("znz_2lpt0.npz", "stage-0 z–N kernel cache")],
        regen_cmd=f"python -m pipeline.run_pipeline --dataset {ds.name} --stage kernel_znz",
    )
    return leaf


def run_fp_loa0(store, ds, *, resume=False, cluster_emit=False):
    """fp_loa0 → build_loa0_fp_product.main (writes loa0_fp_product.npz). The loa-0
    forest false-positive product (Lyα-only λ_rest>=1025 to match the headline)."""
    config = {
        "producer": "build_loa0_fp_product",
        "snr_min": 2.0, "p_dla_min": 0.99, "lya_only_lam_rf_min": 1025.0,
    }
    inputs = _ds_external_inputs(ds, "catalog", "truth", "bal")
    leaf = store.new(dataset=ds.name, stage="fp", producer="build_loa0_fp_product",
                     config=config, inputs=inputs, privacy=ds.privacy,
                     producer_defaults=_slug_defaults(config))
    out_path = leaf.path("loa0_fp_product.npz")
    argv = [
        "--prod-cat", ds.catalog_dir,
        "--prod-mockdir", ds.resolved_mockdir,
        "--prod-bal", ds.bal,
        "--snr-min", "2.0", "--p-dla-min", "0.99",
        "--lya-only-lam-rf-min", "1025",
        "--out", out_path,
    ]
    cli = "python -m CDDF_analysis.hbi.build_loa0_fp_product " + " ".join(argv)
    from CDDF_analysis.hbi import build_loa0_fp_product as FP  # noqa: E402
    FP.main(argv)
    store.commit_leaf(
        leaf, what="loa-0 forest false-positive product (Lyα-only λ_rest>=1025).",
        cli=cli,
        outputs=[("loa0_fp_product.npz", "per-cell FP λ_FP product consumed by the loa0 FP path")],
        regen_cmd=f"python -m pipeline.run_pipeline --dataset {ds.name} --stage fp_loa0",
    )
    return leaf


def run_kernel_remp(store, ds, *, resume=False, cluster_emit=False):
    """kernel_remp → run_remp_kernel.py --stage build (~20 min; heavy in-session).
    The R_emp posterior kernel (no processed-h5 needed)."""
    config = {
        "producer": "run_remp_kernel", "stage": "build",
        "lam_rf_min": 911.0, "dalpha": 0.5, "host_truth_floor": 19.0,
    }
    inputs = _ds_external_inputs(ds, "catalog", "truth", "bal")
    leaf = store.new(dataset=ds.name, stage="kernel_remp", producer="run_remp_kernel",
                     config=config, inputs=inputs, privacy=ds.privacy,
                     producer_defaults=_slug_defaults(config))
    argv = [
        "--stage", "build",
        "--catalog-dir", ds.catalog_dir, "--truth", ds.truth, "--bal-cat", ds.bal,
        "--mockdir", ds.resolved_mockdir,
        "--out", leaf.dir,
    ]
    cli = "python CDDF_analysis/hbi/run_remp_kernel.py " + " ".join(argv)
    from CDDF_analysis.hbi import run_remp_kernel as REMP  # noqa: E402
    REMP.main(argv)
    store.commit_leaf(
        leaf, what="R_emp posterior kernel on 2LPT-0 (~20 min, in-session heavy).",
        cli=cli,
        outputs=[("posterior_kernel.npz", "R_emp posterior kernel")],
        regen_cmd=f"python -m pipeline.run_pipeline --dataset {ds.name} --stage kernel_remp",
    )
    return leaf


def _phase3d_stage(store, ds, *, store_stage, sub_stage, what, out_desc, resume, cluster_emit):
    """Shared wrapper for run_phase3d_postkernel.py --stage {2,3} (fit_map / band).
    Heavy in-session; consumes the SIR kernel produced on the cluster (stage 1)."""
    config = {
        "producer": "run_phase3d_postkernel", "stage": sub_stage,
        "fp_estimator": "purity_mixture", "n_mc": 200,
    }
    inputs = _ds_external_inputs(ds, "catalog", "truth", "bal")
    sir_id = _upstream_or_none(store, dataset=ds.name, stage="kernel_sir")
    if sir_id:
        inputs.append(sir_id)
    leaf = store.new(dataset=ds.name, stage=store_stage, producer="run_phase3d_postkernel",
                     config=config, inputs=inputs, privacy=ds.privacy,
                     producer_defaults=_slug_defaults(config))
    argv = [
        "--stage", sub_stage,
        "--catalog-dir", ds.catalog_dir, "--truth", ds.truth, "--bal-cat", ds.bal,
        "--mockdir", ds.resolved_mockdir,
        "--out", leaf.dir,
    ]
    cli = "python CDDF_analysis/hbi/run_phase3d_postkernel.py " + " ".join(argv)
    from CDDF_analysis.hbi import run_phase3d_postkernel as P3D  # noqa: E402
    P3D.main(argv)
    store.commit_leaf(
        leaf, what=what, cli=cli,
        outputs=[out_desc],
        regen_cmd=f"python -m pipeline.run_pipeline --dataset {ds.name} --stage "
                  + ("fit_map" if sub_stage == "2" else "band"),
    )
    return leaf


def run_fit_map(store, ds, *, resume=False, cluster_emit=False):
    """fit_map → run_phase3d_postkernel.py --stage 2 (~10 min; heavy in-session)."""
    return _phase3d_stage(
        store, ds, store_stage="fit_map", sub_stage="2",
        what="phase3d post-kernel MAP point fit on 2LPT-0 (~10 min, in-session heavy).",
        out_desc=("point_kernel.npz", "MAP point-kernel fit"),
        resume=resume, cluster_emit=cluster_emit)


def run_band(store, ds, *, resume=False, cluster_emit=False):
    """band → run_phase3d_postkernel.py --stage 3 (~30 min; heavy in-session)."""
    return _phase3d_stage(
        store, ds, store_stage="band", sub_stage="3",
        what="phase3d post-kernel MC band on 2LPT-0 (~30 min, in-session heavy).",
        out_desc=("band.npz", "MC band npz"),
        resume=resume, cluster_emit=cluster_emit)


_SIR_SBATCH = (
    "sbatch --export=ALL,STAGE=1 "
    "slurm/greatlakes/production/phase3d_postkernel_staged.sbatch"
)


def run_kernel_sir(store, ds, *, resume=False, cluster_emit=False):
    """kernel_sir → run_phase3d_postkernel.py --stage 1: the SIR kernel (1.6 GB, 1150
    processed-h5, 3-5 h). CLUSTER-ONLY — never run in-session. Raises
    ClusterOnlyStage carrying the exact sbatch line (or prints it under
    --cluster-emit)."""
    if cluster_emit:
        print(f"[cluster-emit] kernel_sir for dataset {ds.name!r}:\n  {_SIR_SBATCH}")
        return None
    raise ClusterOnlyStage("kernel_sir", _SIR_SBATCH)


def run_reduction(store, ds, *, resume=False, cluster_emit=False):
    """reduction → track_c_tf_loa build_frozen_calibration / build_loa_ingredients /
    run_measurement (the SAME path NB5 uses). The frozen 2LPT-0 calibration is the
    committed tutorial_data/ fixtures; the catalog being reduced is ds.catalog_dir.
    Writes result.json (per-z dN/dX & Ω MAP + band)."""
    import json
    import types
    import numpy as np

    # frozen calibration = the committed tutorial_data/ fixtures (NB5's public mode).
    FROZEN_FORWARD = str(TUTORIAL_DATA / "forward_response_2lpt0.npz")
    FROZEN_MOLLY = str(TUTORIAL_DATA / "molly_matrix_nhi195_lyaonly.tsv")
    FROZEN_FP = str(TUTORIAL_DATA / "loa0_fp_product_lyaonly1025.npz")

    # a frozen-calibration upstream leaf takes precedence over the committed fixture
    # when one exists in the store (so a recomputed kernel feeds the reduction).
    fwd_leaf = None
    try:
        fwd_leaf = store.get(dataset="2lpt0", stage="kernel")
        cand = fwd_leaf.path("forward_response_2lpt0.npz")
        if os.path.exists(cand):
            FROZEN_FORWARD = cand
    except LookupError:
        fwd_leaf = None

    from CDDF_analysis.hbi import track_c_tf_loa as TF  # noqa: E402
    from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB  # noqa: E402

    # n_mc small for the demo; matches NB5's notebook-sized setting.
    N_MC, WORKERS, ZBINS = 60, 4, "2.0,2.5,3.0,3.5"
    config = {
        "producer": "track_c_tf_loa", "kernel": "forward_response (frozen)",
        "fp_estimator": "purity_mixture", "n_mc": N_MC, "zbins": ZBINS,
        "snr_min": 2.0, "no_bal": True, "report_limits": "20.0,20.3",
        "catalog": ds.catalog_dir,
    }
    # frozen-calibration leg (2LPT-0 mock) is always a mock input; the privacy of the
    # RESULT is set by the catalog being reduced (ds.privacy).
    inputs = [
        external_input(FROZEN_FORWARD, role="frozen_forward_kernel", privacy="mock"),
        external_input(FROZEN_MOLLY, role="frozen_molly", privacy="mock"),
        external_input(AB.DEF_CAT, role="calibration_catalog", privacy="mock"),
        *_ds_external_inputs(ds, "catalog", "bal"),
    ]
    if fwd_leaf is not None:
        inputs.append(fwd_leaf.id)
    leaf = store.new(dataset=ds.name, stage="measurement", producer="track_c_tf_loa",
                     config=config, inputs=inputs, privacy=ds.privacy,
                     producer_defaults=_slug_defaults(config))

    # assemble the args namespace exactly as NB5 does.
    a = types.SimpleNamespace()
    a.forward_model = FROZEN_FORWARD
    a.molly_tsv = FROZEN_MOLLY
    a.loa0_product = FROZEN_FP
    # 2LPT-0 calibration leg (frozen g(N,z) + molly counts come from these)
    a.catalog_dir, a.truth, a.bal_cat = AB.DEF_CAT, AB.DEF_TRUTH, AB.DEF_BAL
    # DATA leg = the catalog being reduced
    a.loa_cat, a.loa_truth = ds.catalog_dir, ds.truth
    a.loa_bal, a.loa_mockdir = ds.bal, ds.resolved_mockdir
    a.loa_kernel = None
    a.loa_processed_glob = ""
    a.loa_pw_samples = ""
    a.kernel = AB.DEF_KERNEL
    a.resp_family, a.resp_kind = "empirical", "forward"
    a.out = leaf.dir
    a.report_out = os.path.join(leaf.dir, "report.md")
    a.zbins, a.v2_z_fit_hi, a.report_limits = ZBINS, 3.5, "20.0,20.3"
    a.family, a.fit_floor, a.fit_ceil = "bspbody", 19.5, 99.0
    a.lambda_bspbody, a.lam_rf_min, a.edge_slope_lam = 30.0, 1025.0, 40.0
    a.gl_nodes, a.host_truth_floor = 1, 19.0
    a.n_mc, a.workers, a.seed, a.cz_min_count = N_MC, WORKERS, 0, 30.0
    a.band_recenter = True
    a.omega_slope_extrap = True
    a.omega_slope_extrap_integrated = True
    a.slope_edge, a.slope_fit_dex, a.sigma_slope = 21.2, 0.6, 0.5
    os.makedirs(a.out, exist_ok=True)
    LIMITS = tuple(float(x) for x in a.report_limits.split(","))
    a._limits = LIMITS   # private field build_frozen_calibration reads (NB5 sets it too)

    frozen = TF.build_frozen_calibration(a)
    a.molly_tsv = frozen["molly_tsv"]
    ing = TF.build_loa_ingredients(a, frozen)
    res = TF.run_measurement(a, ing, LIMITS, a.seed, frozen=frozen)

    # serialize the per-z result into result.json (no numpy in the JSON).
    ZB = [float(z) for z in res["zbins"]]
    n_zc = int(res["n_zc"])
    ZC = [0.5 * (ZB[k] + ZB[k + 1]) for k in range(n_zc)]

    def _perz(node, key="MAP", scale=1.0):
        return [scale * p[key] for p in node["perz"]]

    out = {
        "dataset": ds.name,
        "kernel": "forward_response (frozen, Track-C)",
        "fp_estimator": "purity_mixture",
        "n_mc": int(res["n_mc"]),
        "zbins": ZB,
        "z_centres": ZC,
        "n_op_detections": int(res.get("n_op_detections", -1)),
        "n_op_sl": int(res.get("n_op_sl", -1)),
        "max_truth_z": float(res.get("max_truth_z", float("nan"))),
        "truth_counts_perz": (list(map(int, res["truth_counts_perz"]))
                              if res.get("truth_counts_perz") is not None else None),
        "limits": list(LIMITS),
        "perz": {},
        "integrated": {},
    }
    for lim in LIMITS:
        out["perz"][f"{lim:g}"] = {
            "dndx": {"MAP": _perz(res["dndx"][lim]),
                     "q16": _perz(res["dndx"][lim], "q16"),
                     "q84": _perz(res["dndx"][lim], "q84")},
            "omega": {"MAP": _perz(res["omega"][lim]),
                      "q16": _perz(res["omega"][lim], "q16"),
                      "q84": _perz(res["omega"][lim], "q84")},
        }
        out["integrated"][f"{lim:g}"] = {
            "dndx_MAP": float(res["dndx"][lim]["integrated"]["MAP"]),
            "omega_MAP": float(res["omega"][lim]["integrated"]["MAP"]),
        }

    with open(leaf.path("result.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    cli = (f"python -m pipeline.run_pipeline --dataset {ds.name} --stage reduction "
           f"# (track_c_tf_loa build_frozen_calibration -> build_loa_ingredients -> run_measurement)")
    store.commit_leaf(
        leaf,
        what=("Per-z dN/dX & Ω_DLA CDDF reduction (frozen Track-C forward kernel + "
              f"mock-calibration), catalog={ds.name}."),
        cli=cli,
        outputs=[("result.json", "per-z + integrated dN/dX & Ω MAP and statistical band"),
                 ("report.md", "track_c_tf_loa text report")],
        regen_cmd=f"python -m pipeline.run_pipeline --dataset {ds.name} --stage reduction",
    )
    return leaf


# --------------------------------------------------------------------------- #
# the registry                                                                 #
# --------------------------------------------------------------------------- #
# DAG: completeness_molly is a precursor; kernel_fwd depends on it; reduction depends
# on the frozen forward kernel (kernel_fwd) + completeness, but FALLS BACK to the
# committed tutorial_data/ fixtures so it runs standalone (the NB5 public path).
STAGES: dict[str, Stage] = {}


def _reg(stage: Stage) -> None:
    STAGES[stage.name] = stage


_reg(Stage(name="completeness_molly", deps=(), producer="molly_faithful_pc_plots",
           run=run_completeness_molly))
_reg(Stage(name="kernel_znz", deps=(), producer="znz_kernel.build-cache",
           run=run_kernel_znz))
_reg(Stage(name="kernel_fwd", deps=("completeness_molly",),
           producer="znz_kernel.build-forward-cache", run=run_kernel_fwd))
_reg(Stage(name="fp_loa0", deps=(), producer="build_loa0_fp_product",
           run=run_fp_loa0))
_reg(Stage(name="kernel_remp", deps=(), producer="run_remp_kernel", heavy=True,
           run=run_kernel_remp))
_reg(Stage(name="kernel_sir", deps=(), producer="run_phase3d_postkernel",
           cluster_only=True, run=run_kernel_sir))
_reg(Stage(name="fit_map", deps=("kernel_sir",), producer="run_phase3d_postkernel",
           heavy=True, run=run_fit_map))
_reg(Stage(name="band", deps=("fit_map",), producer="run_phase3d_postkernel",
           heavy=True, run=run_band))
_reg(Stage(name="reduction", deps=("kernel_fwd", "completeness_molly"),
           producer="track_c_tf_loa", run=run_reduction))


def get_stage(name: str) -> Stage:
    """Return the ``Stage`` for ``name``; raise KeyError listing the catalog
    otherwise."""
    try:
        return STAGES[name]
    except KeyError:
        known = ", ".join(STAGES)
        raise KeyError(f"unknown stage {name!r}; known stages: {known}.") from None


def stage_order(targets=None) -> list:
    """Topologically sort the stage DAG (Kahn's algorithm), returning the stage names
    in dependency order. If ``targets`` is given, return only the sub-DAG needed to
    reach those targets (their transitive deps + themselves, topo-sorted)."""
    if targets is None:
        wanted = set(STAGES)
    else:
        wanted = set()
        frontier = list(targets)
        while frontier:
            n = frontier.pop()
            if n in wanted:
                continue
            wanted.add(n)
            frontier.extend(STAGES[n].deps)

    # Kahn over the induced subgraph.
    indeg = {n: 0 for n in wanted}
    for n in wanted:
        for d in STAGES[n].deps:
            if d in wanted:
                indeg[n] += 1
    # deterministic order: process ready nodes in registry order.
    order = []
    ready = [n for n in STAGES if n in wanted and indeg[n] == 0]
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in STAGES:
            if m in wanted and n in STAGES[m].deps:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
    if len(order) != len(wanted):
        cyclic = wanted - set(order)
        raise ValueError(f"stage DAG has a cycle among: {sorted(cyclic)}")
    return order
