"""Voigt LSF + num_lines hypothesis-test sweep across 3 mocks × 3 NHI bins.

This is the production runner for the Bayesian-correctness Step 1
experiment described in `docs/notes/2026-04-27_bayesian_correctness_plan.md`.

Goal: empirically attribute the +0.37 dex N_HI bias on Y3 mocks to
(a) LSF mismatch (BOSS-shaped kernel applied to DESI linear-λ data),
(b) too few Lyman series lines (production uses num_lines=3),
(c) something else (data physics / mock-generator differences).

Sweep design::

    mocks ∈ {2lpt, saclay, london}
    NHI regimes ∈ {LLS [17.2, 19), sub-DLA [19, 20.3), DLA [20.3, 23]}
    Voigt configs:
       A: production       (kernel='boss-log-r2000', num_lines=3)
       B: DESI LSF fix     (kernel='desi-linear-r3000', num_lines=3)
       C: DESI LSF + lines (kernel='desi-linear-r3000', num_lines=6)
       D: no LSF           (kernel='none', num_lines=3)  [diagnostic]

Each (mock, regime, config) → 5–10 truth sightlines → 5×3×3×3 = ~135
to ~270 inferences. Measures ΔlogNHI and Δz vs truth.

Usage::

    python examples/voigt_lsf_sweep.py \\
        --picked-targets out/voigt_sweep/targets.tsv \\
        --out-dir out/voigt_sweep/runs/ \\
        --configs A,B,C,D

The kernel swap requires a fresh Python process per (target × config)
because ``voigt_v2_inject.inject()`` mutates a module global. We use
``multiprocessing.spawn`` to isolate.

Companion: examples/pick_voigt_sweep_targets.py picks the targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Voigt configuration registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VoigtConfig:
    tag: str
    kernel: str           # 'boss-log-r2000' | 'desi-linear-r3000' | 'none'
    num_lines: int
    description: str


CONFIGS: dict[str, VoigtConfig] = {
    "A": VoigtConfig("A", "boss-log-r2000", 3,
                     "Production (BOSS-shaped LSF on DESI grid, 3 Lyman lines)"),
    "B": VoigtConfig("B", "desi-linear-r3000", 3,
                     "DESI LSF fix (R=3000 Gaussian on DESI grid, 3 Lyman lines)"),
    "C": VoigtConfig("C", "desi-linear-r3000", 6,
                     "DESI LSF + more lines (R=3000 + Lyα–Ly-VI)"),
    "D": VoigtConfig("D", "none", 3,
                     "No LSF convolution — bare Voigt (diagnostic only)"),
}


# ---------------------------------------------------------------------------
# Per-target inference (runs in a child process so kernel swap is isolated)
# ---------------------------------------------------------------------------
def _run_one_inference(
    config_tag: str,
    spec_path: str,
    zcat_path: str,
    target_id: int,
    truth_z_dla: float,
    truth_log_nhi: float,
    nhi_regime: str,
    mock_name: str,
    data_root: str,
    dla_samples_file: str,
    sub_dla_samples_file: str,
    out_h5: str,
):
    """Run multi-DLA inference on one spectrum under one Voigt config.

    Imports happen INSIDE this function so the kernel swap (which mutates
    a module-level global) lives in a fresh child interpreter.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import time
    import numpy as np

    cfg = CONFIGS[config_tag]

    # Inject before any dla_gp / DLAHolder imports.
    from gpy_dla_detection.voigt_v2_inject import inject
    inject(kernel=cfg.kernel, num_lines=cfg.num_lines)

    # Now safe to import the inference machinery.
    from examples.smoke_one_spectrum import (
        load_one_desi_spectrum, lookup_z_qso, PRESETS,
    )
    import h5py

    t0 = time.perf_counter()
    try:
        wave, flux, noise_var, mask = load_one_desi_spectrum(spec_path, target_id)
        z_qso = lookup_z_qso(zcat_path, target_id)
    except Exception as e:
        return {"error": f"load: {e}"}

    # Single-DLA mode at first — multi-DLA is overkill for this study.
    # The bias signature shows up on the 1-DLA MAP NHI.
    preset = PRESETS["y3"]
    learned_file = os.path.join(data_root, preset.learned_file)
    catalog_name = os.path.join(data_root, "data/dr12q/processed/catalog.mat")
    los_catalog = os.path.join(data_root, "data/dla_catalogs/dr9q_concordance/processed/los_catalog")
    dla_catalog = os.path.join(data_root, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog")

    try:
        from gpy_dla_detection.set_parameters import Parameters
        from run_bayes_select import DLAHolder

        common = dict(
            loading_min_lambda=preset.loading_min_lambda,
            loading_max_lambda=preset.loading_max_lambda,
            normalization_min_lambda=preset.normalization_min_lambda,
            normalization_max_lambda=preset.normalization_max_lambda,
            min_lambda=preset.min_lambda,
            max_lambda=preset.max_lambda,
            dlambda=preset.dlambda,
            k=preset.k,
            max_noise_variance=9.0,
            num_lines=cfg.num_lines,
            max_z_cut=3000.0,
            min_z_cut=3000.0,
            num_forest_lines=preset.num_forest_lines,
        )
        params = Parameters(num_dla_samples=100000, **common)
        params_subdla = Parameters(num_dla_samples=100000, **common)

        holder = DLAHolder(
            learned_file=learned_file,
            catalog_name=catalog_name,
            los_catalog=los_catalog,
            dla_catalog=dla_catalog,
            dla_samples_file=dla_samples_file,
            sub_dla_samples_file=sub_dla_samples_file,
            params=params,
            params_subdla=params_subdla,
            min_z_separation=3000.0,
            prev_tau_0=preset.prev_tau_0,
            prev_beta=preset.prev_beta,
            max_dlas=4,
            broadening=True,
            plot_figures=False,
            # Match production multi-DLA settings (per user): 8 workers, 12500
            # batch. 8 batches over 8 workers — 1 round, minimal dispatch
            # overhead.
            max_workers=8,
            batch_size=12500,
            figure_dir="/tmp",
            single_absorber_model=False,
            filter_low_likelihood=True,
        )
        holder.initialize_results(1)
        holder.process_qso(
            idx=0,
            target_id=str(target_id),
            wavelengths=wave,
            flux=flux,
            noise_variance=noise_var,
            pixel_mask=mask,
            z_qso=z_qso,
        )
        res = holder.results
        map_z = float(res["MAP_z_dlas"][0, 0])
        map_log_nhi = float(res["MAP_log_nhis"][0, 0])
        p_dla = float(res["p_dlas"][0])
    except Exception as e:
        import traceback
        return {"error": f"infer: {e}\n{traceback.format_exc()[:500]}"}
    finally:
        wall = time.perf_counter() - t0

    # Save the per-(target, config) result for downstream analysis.
    Path(out_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.attrs["target_id"] = int(target_id)
        f.attrs["mock"] = mock_name
        f.attrs["config_tag"] = cfg.tag
        f.attrs["kernel"] = cfg.kernel
        f.attrs["num_lines"] = int(cfg.num_lines)
        f.attrs["nhi_regime"] = nhi_regime
        f.attrs["truth_z_dla"] = float(truth_z_dla)
        f.attrs["truth_log_nhi"] = float(truth_log_nhi)
        f.attrs["map_z_dla"] = float(map_z)
        f.attrs["map_log_nhi"] = float(map_log_nhi)
        f.attrs["p_dla"] = float(p_dla)
        f.attrs["wall_s"] = float(wall)

    return {
        "target_id": int(target_id), "mock": mock_name,
        "config_tag": cfg.tag, "kernel": cfg.kernel,
        "num_lines": int(cfg.num_lines), "nhi_regime": nhi_regime,
        "truth_z_dla": float(truth_z_dla), "truth_log_nhi": float(truth_log_nhi),
        "map_z_dla": float(map_z), "map_log_nhi": float(map_log_nhi),
        "p_dla": float(p_dla), "delta_log_nhi": float(map_log_nhi - truth_log_nhi),
        "delta_z_dla": float(map_z - truth_z_dla),
        "wall_s": float(wall),
    }


# ---------------------------------------------------------------------------
# Process wrapper: drops the dict result onto a queue so the parent can read
# it. Used because we launch via ctx.Process (not Pool) to keep the child
# non-daemonic — daemonic children can't spawn the inner ProcessPoolExecutor.
# ---------------------------------------------------------------------------
def _proc_target(q, *args):
    result = _run_one_inference(*args)
    q.put(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _read_targets(path: Path) -> list[dict]:
    """TSV with columns: mock target_id z_qso truth_z_dla truth_log_nhi
    nhi_regime spec_path zcat_path"""
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--picked-targets", required=True, type=Path,
                   help="TSV from examples/pick_voigt_sweep_targets.py")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--configs", default="A,B,C,D",
                   help="Comma-separated config tags (default A,B,C,D)")
    p.add_argument("--data-root", default=os.environ.get(
        "DATA_ROOT", "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection",
    ))
    p.add_argument("--dla-samples-file",
                   default="data/dr12q/processed/dla_samples_a03_100000.mat")
    p.add_argument("--sub-dla-samples-file",
                   default="data/dr12q/processed/subdla_samples_a03_191_200_100000.mat")
    p.add_argument("--max-targets-per-bin", type=int, default=None,
                   help="Cap on (mock, regime) for quick iteration")
    args = p.parse_args()

    targets = _read_targets(args.picked_targets)
    config_tags = [t.strip() for t in args.configs.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve sample-file paths under data_root.
    dla_samples_path = os.path.join(args.data_root, args.dla_samples_file)
    sub_dla_samples_path = os.path.join(args.data_root, args.sub_dla_samples_file)

    # Optionally cap.
    if args.max_targets_per_bin:
        from collections import defaultdict
        binned = defaultdict(list)
        for t in targets:
            binned[(t["mock"], t["nhi_regime"])].append(t)
        targets = []
        for k, v in binned.items():
            targets.extend(v[: args.max_targets_per_bin])

    print(f"[sweep] {len(targets)} targets × {len(config_tags)} configs = "
          f"{len(targets) * len(config_tags)} inferences")

    # Run sequentially in spawned processes (kernel swap requires isolation).
    # Use ctx.Process (non-daemonic) so the child can itself spawn the inner
    # ProcessPoolExecutor used by parallel_log_model_evidences. ctx.Pool would
    # make the worker daemonic and forbid grandchild processes.
    ctx = get_context("spawn")
    rows: list[dict] = []
    t_start = time.perf_counter()
    for t_idx, t in enumerate(targets):
        for c_tag in config_tags:
            tag = f"{t['mock']}_{t['nhi_regime']}_tid{t['target_id']}_{c_tag}"
            out_h5 = args.out_dir / f"{tag}.h5"
            args_tuple = (
                c_tag, t["spec_path"], t["zcat_path"], int(t["target_id"]),
                float(t["truth_z_dla"]), float(t["truth_log_nhi"]),
                t["nhi_regime"], t["mock"], args.data_root,
                dla_samples_path, sub_dla_samples_path, str(out_h5),
            )
            print(f"[sweep] [{t_idx + 1}/{len(targets)}] {tag}", flush=True)
            q = ctx.Queue()
            p = ctx.Process(target=_proc_target, args=(q,) + args_tuple)
            p.start()
            p.join()
            if q.empty():
                result = {"error": f"child exited without result (exitcode={p.exitcode})"}
            else:
                result = q.get_nowait()
            rows.append(result)
            if "error" in result:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  truth_logNHI={result['truth_log_nhi']:.2f} "
                      f"map_logNHI={result['map_log_nhi']:.2f} "
                      f"Δ={result['delta_log_nhi']:+.3f} "
                      f"p(DLA)={result['p_dla']:.3f} "
                      f"({result['wall_s']:.0f}s)")

    # Write the master CSV.
    csv_path = args.out_dir / "master.csv"
    with csv_path.open("w") as f:
        keys = sorted({k for r in rows if "error" not in r for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            if "error" not in r:
                w.writerow({k: r.get(k) for k in keys})
    print(f"\n[sweep] {(time.perf_counter() - t_start) / 60:.1f} min total wall")
    print(f"[sweep] master CSV → {csv_path}")
    print(f"[sweep] per-(target,config) H5s → {args.out_dir}/*.h5")


if __name__ == "__main__":
    main()
