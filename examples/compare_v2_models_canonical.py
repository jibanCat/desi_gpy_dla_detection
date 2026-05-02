"""Compare v2-normalized GP models (and the v1 baseline) on the canonical
2lpt-mock-0 TID 120046865.

This is the cheapest discriminating test we have: ~20s per model on CPU,
shows whether mock-trained vs LOA-trained models give different
``p_dla`` / ``MAP_log_nhi`` / ``model_posteriors`` on the SAME spectrum.

Usage::

    python examples/compare_v2_models_canonical.py \\
        --models /path/to/v1.h5,/path/to/v2_a.h5,/path/to/v2_b.h5 \\
        --out comparison.md

If ``--out`` ends in ``.md`` the result is a Markdown table; ``.tsv``
gives a tab-separated table. Default = stdout-only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CANONICAL_TID = 120046865
CANONICAL_SPEC = (
    "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
    "mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits"
)
CANONICAL_ZCAT = (
    "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
    "mock-0/loa-124/zcat.fits"
)
DATA_ROOT = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"


def _load_canonical_spectrum() -> dict:
    """Load TID 120046865 from the 2lpt mock-0 / loa-124 spectra-16 file.
    Mirrors examples.smoke_one_spectrum.load_one_desi_spectrum."""
    from examples.smoke_one_spectrum import load_one_desi_spectrum, lookup_z_qso
    if not os.path.exists(CANONICAL_SPEC):
        raise SystemExit(f"canonical spec not on disk: {CANONICAL_SPEC}")
    if not os.path.exists(CANONICAL_ZCAT):
        raise SystemExit(f"canonical zcat not on disk: {CANONICAL_ZCAT}")
    wave, flux, nv, mask = load_one_desi_spectrum(CANONICAL_SPEC, CANONICAL_TID)
    z_qso = lookup_z_qso(CANONICAL_ZCAT, CANONICAL_TID)
    return dict(wave=wave, flux=flux, nv=nv, mask=mask, z_qso=z_qso)


def _build_holder(learned_file: str, num_dla_samples: int = 10000):
    """Build a fresh DLAHolder pointed at the given learned_file. Uses the
    y3 production preset (matching examples.smoke_one_spectrum.PRESETS['y3'])
    apart from the learned_file override."""
    from gpy_dla_detection.set_parameters import Parameters
    from run_bayes_select import DLAHolder

    common = dict(
        loading_min_lambda=910.0, loading_max_lambda=1550.0,
        normalization_min_lambda=1425.0, normalization_max_lambda=1475.0,
        min_lambda=911.75, max_lambda=1216.75, dlambda=0.15, k=30,
        max_noise_variance=9.0, num_lines=3,
        max_z_cut=3000.0, min_z_cut=3000.0, num_forest_lines=3,
    )
    params = Parameters(num_dla_samples=num_dla_samples, **common)
    params_subdla = Parameters(num_dla_samples=num_dla_samples, **common)
    holder = DLAHolder(
        learned_file=learned_file,
        catalog_name=os.path.join(DATA_ROOT, "data/dr12q/processed/catalog.mat"),
        los_catalog=os.path.join(
            DATA_ROOT,
            "data/dla_catalogs/dr9q_concordance/processed/los_catalog"),
        dla_catalog=os.path.join(
            DATA_ROOT,
            "data/dla_catalogs/dr9q_concordance/processed/dla_catalog"),
        dla_samples_file=os.path.join(
            DATA_ROOT, "data/dr12q/processed/dla_samples_a03.mat"),
        sub_dla_samples_file=os.path.join(
            DATA_ROOT, "data/dr12q/processed/subdla_samples.mat"),
        params=params, params_subdla=params_subdla,
        min_z_separation=3000.0,
        prev_tau_0=0.00246, prev_beta=3.62,
        max_dlas=3, broadening=True, plot_figures=False,
        max_workers=1, batch_size=1,
        single_absorber_model=False,
    )
    holder.initialize_results(1)
    return holder, params


def _run_one(model_path: str, spectrum: dict, num_dla_samples: int) -> dict:
    """Build a holder, run process_qso once, return a dict of summary numbers."""
    label = Path(model_path).parent.name + "/" + Path(model_path).name
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    holder, params = _build_holder(model_path, num_dla_samples=num_dla_samples)
    t_load = time.time() - t0

    t0 = time.time()
    holder.process_qso(
        idx=0, target_id=CANONICAL_TID,
        wavelengths=spectrum["wave"].astype(np.float64),
        flux=spectrum["flux"].astype(np.float64),
        noise_variance=spectrum["nv"].astype(np.float64),
        pixel_mask=spectrum["mask"].astype(bool),
        z_qso=float(spectrum["z_qso"]),
    )
    t_inf = time.time() - t0

    res = holder.results
    return dict(
        label=label,
        norm_min=params.normalization_min_lambda,
        norm_max=params.normalization_max_lambda,
        p_no_dlas=float(res["p_no_dlas"][0]),
        p_dlas=float(res["p_dlas"][0]),
        map_z_dla=float(res["MAP_z_dlas"][0, 0])
                  if not np.isnan(res["MAP_z_dlas"][0, 0]) else float("nan"),
        map_log_nhi=float(res["MAP_log_nhis"][0, 0])
                    if not np.isnan(res["MAP_log_nhis"][0, 0]) else float("nan"),
        model_posteriors=np.asarray(res["model_posteriors"][0]).tolist(),
        t_load=t_load, t_inf=t_inf,
    )


def _format_md(rows: list[dict]) -> str:
    lines = []
    lines.append(f"# Canonical-TID comparison — TID {CANONICAL_TID}\n")
    lines.append(f"Spectrum: 2lpt mock-0 / loa-124 / spectra-16-789.fits\n")
    lines.append(f"z_qso (zcat): {rows[0].get('z_qso', 'see per-model'):.4f}\n"
                 if isinstance(rows[0].get('z_qso'), float) else "")
    lines.append("| Model | norm window | p_no_dla | p_dla | MAP z | MAP logNHI | "
                 "model_posteriors | t_load | t_inf |")
    lines.append("|---|---|---:|---:|---:|---:|---|---:|---:|")
    for r in rows:
        mp_str = "[" + ", ".join(f"{p:.3f}" for p in r["model_posteriors"]) + "]"
        lines.append(
            f"| {r['label']} "
            f"| [{r['norm_min']:.0f}, {r['norm_max']:.0f}] "
            f"| {r['p_no_dlas']:.4f} "
            f"| {r['p_dlas']:.4f} "
            f"| {r['map_z_dla']:.4f} "
            f"| {r['map_log_nhi']:.3f} "
            f"| {mp_str} "
            f"| {r['t_load']:.1f}s "
            f"| {r['t_inf']:.1f}s |"
        )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True,
                   help="Comma-separated absolute paths to model .h5 files")
    p.add_argument("--out", default=None,
                   help="Output path (.md or .tsv). Default = stdout-only.")
    p.add_argument("--num-dla-samples", type=int, default=10000,
                   help="Number of DLA samples per inference (default 10000 for "
                        "production-realistic Bayes factor).")
    args = p.parse_args()

    model_paths = [m.strip() for m in args.models.split(",") if m.strip()]
    missing = [m for m in model_paths if not os.path.exists(m)]
    if missing:
        raise SystemExit(f"missing model files: {missing}")

    print(f"[main] loading canonical spectrum (TID {CANONICAL_TID})")
    spectrum = _load_canonical_spectrum()
    print(f"  z_qso = {float(spectrum['z_qso']):.4f}  n_pix = {len(spectrum['wave'])}")

    rows = []
    for m in model_paths:
        rows.append(_run_one(m, spectrum, args.num_dla_samples))

    md = _format_md(rows)
    print("\n" + md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"[main] wrote {args.out}")


if __name__ == "__main__":
    main()
