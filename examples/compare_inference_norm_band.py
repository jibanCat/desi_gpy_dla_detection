"""Does the per-spectrum normalization band [1310, 1325] vs [1425, 1475]
matter for DLA detection in inference?

Motivation: the multi-dataset PCA-init plot
(`docs/notes/2026-05-12_2lpt_models_vs_v1_analysis/corr_pca_init_multi_dataset.png`)
shows the [1310, 1325] norm band produces visibly stronger off-diagonal
correlation in the Lyα-forest region of corr(M·M^T) than [1425, 1475].
This script tests whether that propagates to inference on a canonical
DLA target.

Models: `2lpt_loa124_nohcd_nobal_wide_g` (norm [1310, 1325]) vs
        `2lpt_loa124_nohcd_nobal_wide_m` (norm [1425, 1475]).
Both trained on the same `2lpt_loa124_nohcd_nobal_wide_v2_*` preload,
same priors (strict Turner+2024), same 1500 Adam iters, same k=30,
PCA init. The ONLY difference is the normalization band.

Target: 2lpt loa-124 mock-0 TID 120046865, truth log_NHI = 21.263.
This is the canonical target used in
`tests/fixtures/2lpt_frozen/short_retrain/canonical_tid_summary.md`,
where v1 / v3.5 trainers both gave p_DLA = 0.989, MAP_log_NHI ≈ 21.63.

The training data and inference data are both 2lpt-loa-124, so this is
an *in-domain* comparison (vs the cross-domain DR16-on-2lpt test in
`compare_inference_vec_vs_perspec.py`).

Output:
  docs/notes/2026-05-13_norm_band_inference/{g,m}.json
  docs/notes/2026-05-13_norm_band_inference/summary.md
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
    ("g_norm_1310_1325",
     str(NOTES / "2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_g" / "phase2_result.h5")),
    ("m_norm_1425_1475",
     str(NOTES / "2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_m" / "phase2_result.h5")),
]
TARGET_ID = 120046865
TRUTH_LOG_NHI = 21.263
SPEC_PATH = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits"
ZCAT_PATH = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits"
DATA_ROOT = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"
OUT_DIR = NOTES / "2026-05-13_norm_band_inference"


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

    # Match the trained model's k, rest range, dλ. The loader
    # mutates params.normalization_*_lambda from the .h5 fields,
    # so we don't need to pass the right ones — but pass the y3
    # preset as a starting point.
    import h5py as _h5
    with _h5.File(model_path, "r") as _f:
        k_trained = int(_f["M"].shape[1])
        rest_min = float(_f["rest_wavelengths"][0])
        rest_max = float(_f["rest_wavelengths"][-1])
        d_lambda = float(_f["rest_wavelengths"][1] - _f["rest_wavelengths"][0])
        norm_min_h5 = float(_f["normalization_min_lambda"][()]) if "normalization_min_lambda" in _f else None
        norm_max_h5 = float(_f["normalization_max_lambda"][()]) if "normalization_max_lambda" in _f else None
    print(f"  trained model: k={k_trained}, rest=[{rest_min:.2f}, {rest_max:.2f}], "
          f"dλ={d_lambda:.4f}, norm=[{norm_min_h5}, {norm_max_h5}]")

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
    # After holder init the .h5 has been read and params mutated:
    print(f"  after load: params.normalization=[{params.normalization_min_lambda}, "
          f"{params.normalization_max_lambda}]")

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
    print(f"  MAP log NHI = {map_nhi:.6f}  (truth = {TRUTH_LOG_NHI}; "
          f"Δ = {map_nhi - TRUTH_LOG_NHI:+.3f} dex)")
    print(f"  posteriors  = {posteriors}")
    print(f"  elapsed     = {dt:.1f} s")
    out = dict(
        model=name, model_path=model_path, target_id=TARGET_ID, z_qso=z_qso,
        norm_min_h5=norm_min_h5, norm_max_h5=norm_max_h5,
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

    a, b = results
    md = [
        f"# Norm-band inference test on canonical TID {TARGET_ID}",
        "",
        f"In-domain test: 2lpt-loa-124-trained models on a 2lpt-loa-124",
        f"spectrum. Both models trained on the same preload with the same",
        f"priors and 1500 Adam iters; the only difference is the per-spectrum",
        f"normalization band.",
        "",
        f"Truth: TID {TARGET_ID}, z_qso = {a['z_qso']:.4f}, log_NHI = {TRUTH_LOG_NHI}.",
        f"Reference (DR16-trained v1 trainer on same target):",
        f"  p_DLA = 0.9897, MAP_log_NHI = 21.628, Δ = +0.365 dex",
        f"  (`tests/fixtures/2lpt_frozen/short_retrain/canonical_tid_summary.md`)",
        "",
        "| metric | g (norm [1310, 1325]) | m (norm [1425, 1475]) | Δ |",
        "|---|---:|---:|---:|",
        f"| p_DLA | {a['p_dla']:.6f} | {b['p_dla']:.6f} | {a['p_dla']-b['p_dla']:+.2e} |",
        f"| MAP z_DLA | {a['map_z_dla']:.6f} | {b['map_z_dla']:.6f} | {a['map_z_dla']-b['map_z_dla']:+.2e} |",
        f"| MAP log NHI | {a['map_log_nhi']:.6f} | {b['map_log_nhi']:.6f} | {a['map_log_nhi']-b['map_log_nhi']:+.3f} |",
        f"| log p(noDLA) | {a['model_posteriors'][0]:.3f} | {b['model_posteriors'][0]:.3f} | {a['model_posteriors'][0]-b['model_posteriors'][0]:+.3f} |",
        f"| log p(subDLA) | {a['model_posteriors'][1]:.3f} | {b['model_posteriors'][1]:.3f} | {a['model_posteriors'][1]-b['model_posteriors'][1]:+.3f} |",
        f"| log p(1DLA) | {a['model_posteriors'][2]:.3f} | {b['model_posteriors'][2]:.3f} | {a['model_posteriors'][2]-b['model_posteriors'][2]:+.3f} |",
        "",
        f"## Interpretation",
        "",
        f"- p_DLA delta: |Δ| = {abs(a['p_dla']-b['p_dla']):.2e}",
        f"- Compared to the **kernel** difference (1.7% Frobenius from",
        f"  vec-vs-per-spec) which gave Δp_DLA = 2.9e-3, this norm-band",
        f"  difference is a {'larger' if abs(a['p_dla']-b['p_dla']) > 0.003 else 'smaller'}",
        f"  effect.",
    ]
    summary = "\n".join(md) + "\n"
    (OUT_DIR / "summary.md").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
