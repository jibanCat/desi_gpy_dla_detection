"""Inference consistency check: vec vs per-spec DR16-trained models on the
canonical 2lpt TID.

This is the inference-side complement to the kernel-level comparison in
`docs/notes/2026-05-11_vec_vs_perspec_full_comparison.md`. Where that doc
showed M·M^T agrees to 1.7% Frobenius (corr to 0.95%), this script asks:
  → Does that translate to the same p_DLA, MAP_z, MAP_log_NHI on a real
    spectrum, end-to-end through DLAHolder?

Cross-domain caveat: the models are SDSS-DR16-trained, the test spectrum
is DESI 2lpt. Absolute p_DLA and MAP values depend on that mismatch and
should not be over-interpreted as scientific results — what matters is
that **both models, applied identically, produce the same numbers**.

Usage:
    python examples/compare_inference_vec_vs_perspec.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTES = REPO / "docs" / "notes"

MODELS = [
    ("vec_full",
     str(NOTES / "2026-05-08_matlab_dr16_validation_vec_full" / "phase2_result.h5")),
    ("per_spec",
     str(NOTES / "2026-05-08_matlab_dr16_validation_per_spec" / "phase2_result.h5")),
]
TARGET_ID = 120046865
TRUTH_LOG_NHI = 21.263  # for context only; not the metric
SPEC_PATH = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits"
ZCAT_PATH = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits"
DATA_ROOT = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"
OUT_DIR = NOTES / "2026-05-11_vec_vs_perspec_inference"


def _run_one(name: str, model_path: str, out_dir: Path) -> dict:
    sys.path.insert(0, str(REPO))
    from examples.smoke_one_spectrum import (
        load_one_desi_spectrum, lookup_z_qso, PRESETS,
    )
    from gpy_dla_detection.set_parameters import Parameters
    from run_bayes_select import DLAHolder

    print(f"\n=== {name} ===")
    print(f"  model: {model_path}")
    if not os.path.exists(model_path):
        print(f"  SKIP: not found")
        return None

    wave, flux, nv, mask = load_one_desi_spectrum(SPEC_PATH, TARGET_ID)
    z_qso = lookup_z_qso(ZCAT_PATH, TARGET_ID)
    preset = PRESETS["y3"]

    # The DR16-trained models have M with k=20 components (SDSS DR16
    # convention). Y3 preset assumes k=30. We override k to match the
    # trained model, otherwise the M_interpolator assertion fails.
    import h5py as _h5
    with _h5.File(model_path, "r") as _f:
        k_trained = int(_f["M"].shape[1])
        rest_min = float(_f["rest_wavelengths"][0])
        rest_max = float(_f["rest_wavelengths"][-1])
        d_lambda = float(_f["rest_wavelengths"][1] - _f["rest_wavelengths"][0])
    print(f"  trained model: k={k_trained}, rest=[{rest_min:.2f}, {rest_max:.2f}], dλ={d_lambda:.4f}")

    # Restrict inference to the trained model's rest range so the
    # interpolators don't get queried out-of-domain. dlambda also matches
    # the model grid.
    common = dict(
        loading_min_lambda=preset.loading_min_lambda,
        loading_max_lambda=preset.loading_max_lambda,
        normalization_min_lambda=preset.normalization_min_lambda,
        normalization_max_lambda=preset.normalization_max_lambda,
        min_lambda=rest_min, max_lambda=rest_max,
        dlambda=d_lambda, k=k_trained,
        max_noise_variance=9.0, num_lines=3,
        max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=preset.num_forest_lines,
    )
    params = Parameters(num_dla_samples=100000, **common)
    params_subdla = Parameters(num_dla_samples=100000, **common)

    holder = DLAHolder(
        learned_file=model_path,
        catalog_name=os.path.join(DATA_ROOT, "data/dr12q/processed/catalog.mat"),
        los_catalog=os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/los_catalog"),
        dla_catalog=os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog"),
        dla_samples_file=os.path.join(DATA_ROOT, "data/dr12q/processed/dla_samples_a03_100000.mat"),
        sub_dla_samples_file=os.path.join(DATA_ROOT, "data/dr12q/processed/subdla_samples_a03_191_200_100000.mat"),
        params=params, params_subdla=params_subdla,
        min_z_separation=3000.0,
        prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta,
        max_dlas=4, broadening=True,
        plot_figures=False, max_workers=8, batch_size=12500,
        figure_dir="/tmp",
        single_absorber_model=False,
        filter_low_likelihood=True,
    )
    holder.initialize_results(1)
    t0 = time.time()
    holder.process_qso(idx=0, target_id=str(TARGET_ID),
                       wavelengths=wave, flux=flux,
                       noise_variance=nv, pixel_mask=mask, z_qso=z_qso)
    dt = time.time() - t0
    res = holder.results
    p_dla = float(res["p_dlas"][0])
    map_z = float(res["MAP_z_dlas"][0, 0])
    map_nhi = float(res["MAP_log_nhis"][0, 0])
    posteriors = [float(x) for x in res["model_posteriors"][0]]
    print(f"  p_DLA       = {p_dla:.6f}")
    print(f"  MAP z_DLA   = {map_z:.6f}  (z_qso = {z_qso:.4f})")
    print(f"  MAP log NHI = {map_nhi:.6f}  (Δ vs truth = {map_nhi - TRUTH_LOG_NHI:+.3f} dex)")
    print(f"  posteriors  = {posteriors}")
    print(f"  elapsed     = {dt:.1f} s")
    out = dict(
        model=name, model_path=model_path, target_id=TARGET_ID, z_qso=z_qso,
        p_dla=p_dla, map_z_dla=map_z, map_log_nhi=map_nhi,
        model_posteriors=posteriors, elapsed_s=dt,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name, mp in MODELS:
        try:
            r = _run_one(name, mp, OUT_DIR)
            if r is not None:
                results.append(r)
        except Exception as ex:
            print(f"  ERROR in {name}: {ex!r}")
            import traceback; traceback.print_exc()

    if len(results) != 2:
        raise SystemExit(f"expected 2 results, got {len(results)}")

    a, b = results  # vec_full, per_spec
    md = [
        f"# Inference consistency: vec vs per-spec on canonical TID {TARGET_ID}",
        "",
        f"Cross-domain test: SDSS-DR16-trained model on a DESI 2lpt spectrum.",
        f"What we're checking: do **both DR16 retrains** (vec full and per-spec",
        f"full) produce the same DLAHolder output? Absolute values are not the",
        f"point — relative agreement is.",
        "",
        f"Truth: TID {TARGET_ID} log_NHI = {TRUTH_LOG_NHI} (z_qso = {a['z_qso']:.4f}).",
        "",
        "| metric | vec_full | per_spec | Δ |",
        "|---|---:|---:|---:|",
        f"| p_DLA | {a['p_dla']:.6f} | {b['p_dla']:.6f} | {a['p_dla']-b['p_dla']:+.2e} |",
        f"| MAP z_DLA | {a['map_z_dla']:.6f} | {b['map_z_dla']:.6f} | {a['map_z_dla']-b['map_z_dla']:+.2e} |",
        f"| MAP log NHI | {a['map_log_nhi']:.6f} | {b['map_log_nhi']:.6f} | {a['map_log_nhi']-b['map_log_nhi']:+.2e} |",
        f"| elapsed (s) | {a['elapsed_s']:.1f} | {b['elapsed_s']:.1f} | — |",
        "",
        "## Model posteriors (per absorber count)",
        "",
        "Layout: [Null, SubDLA, 1DLA, 2DLA, 3DLA, 4DLA] (max_dlas=4, "
        "single_absorber_model=False).",
        "",
        "| idx | vec_full | per_spec | Δ |",
        "|---|---:|---:|---:|",
    ]
    for i in range(len(a['model_posteriors'])):
        va, vb = a['model_posteriors'][i], b['model_posteriors'][i]
        md.append(f"| {i} | {va:.6e} | {vb:.6e} | {va-vb:+.2e} |")

    summary = OUT_DIR / "summary.md"
    summary.write_text("\n".join(md) + "\n")
    print(f"\n[saved] {summary}")


if __name__ == "__main__":
    main()
