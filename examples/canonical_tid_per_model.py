"""Run canonical-TID 120046865 inference under each of the 6 _corrected
v2 retrains and emit a comparison table.

Truth: 2lpt mock-0 loa-124, TID 120046865, log_NHI = 21.263 (strong DLA).
Historic v1 production bias on this target was +0.34 dex.

This is fast — DLAHolder.process_qso on a single spectrum takes ~10 s
per model with FILTER=True. We use FILTER=True (post-fix #5 path) since
on the canonical target it matches BASELINE p_DLA to ~0.001.

Outputs JSON per model and a summary markdown.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


MODELS = [
    ("loa_no_dla_no_bal_corrected",
     "/scratch/cavestru_root/cavestru0/mfho/gl_outputs/GP_trained/loa_no_dla_no_bal_corrected/model_epoch_1499.h5"),
    ("loa_no_hcd_with_bal_corrected",
     "/scratch/cavestru_root/cavestru0/mfho/gl_outputs/GP_trained/loa_no_hcd_with_bal_corrected/model_epoch_1499.h5"),
    ("2lpt_loa0_corrected",
     "/scratch/cavestru_root/cavestru0/mfho/gl_outputs/v2_runs/2lpt_loa0_corrected/model_epoch_1499.h5"),
    ("2lpt_loa124_nohcd_nobal_corrected",
     "/scratch/cavestru_root/cavestru0/mfho/gl_outputs/v2_runs/2lpt_loa124_nohcd_nobal_corrected/model_epoch_1499.h5"),
    ("saclay_mock0_nohcd_nobal_corrected",
     "/scratch/cavestru_root/cavestru0/mfho/gl_outputs/v2_runs/saclay_mock0_nohcd_nobal_corrected/model_epoch_1499.h5"),
    ("2lpt_bal_only_corrected",
     "/scratch/cavestru_root/cavestru0/mfho/gl_outputs/v2_runs/2lpt_bal_only_corrected/model_epoch_1499.h5"),
]

TRUTH_LOG_NHI = 21.263
TARGET_ID = 120046865
SPEC_PATH = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits"
ZCAT_PATH = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits"


def _run_one(name, model_path, data_root, out_dir, filter_low_likelihood=True):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
    common = dict(
        loading_min_lambda=preset.loading_min_lambda,
        loading_max_lambda=preset.loading_max_lambda,
        normalization_min_lambda=preset.normalization_min_lambda,
        normalization_max_lambda=preset.normalization_max_lambda,
        min_lambda=preset.min_lambda, max_lambda=preset.max_lambda,
        dlambda=preset.dlambda, k=preset.k,
        max_noise_variance=9.0, num_lines=3,
        max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=preset.num_forest_lines,
    )
    params = Parameters(num_dla_samples=100000, **common)
    params_subdla = Parameters(num_dla_samples=100000, **common)

    holder = DLAHolder(
        learned_file=model_path,
        catalog_name=os.path.join(data_root, "data/dr12q/processed/catalog.mat"),
        los_catalog=os.path.join(data_root, "data/dla_catalogs/dr9q_concordance/processed/los_catalog"),
        dla_catalog=os.path.join(data_root, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog"),
        dla_samples_file=os.path.join(data_root, "data/dr12q/processed/dla_samples_a03_100000.mat"),
        sub_dla_samples_file=os.path.join(data_root, "data/dr12q/processed/subdla_samples_a03_191_200_100000.mat"),
        params=params, params_subdla=params_subdla,
        min_z_separation=3000.0,
        prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta,
        max_dlas=4, broadening=True,
        plot_figures=False, max_workers=8, batch_size=12500,
        figure_dir="/tmp",
        single_absorber_model=False,
        filter_low_likelihood=filter_low_likelihood,
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
    delta_nhi = map_nhi - TRUTH_LOG_NHI
    print(f"  p_DLA       = {p_dla:.4f}")
    print(f"  MAP z_DLA   = {map_z:.4f} (z_qso = {z_qso:.4f})")
    print(f"  MAP log NHI = {map_nhi:.4f}  (Δ = {delta_nhi:+.3f} dex)")
    print(f"  elapsed     = {dt:.1f} s")
    out = dict(
        model=name, model_path=model_path, target_id=TARGET_ID,
        truth_log_nhi=TRUTH_LOG_NHI, z_qso=z_qso,
        p_dla=p_dla, map_z_dla=map_z, map_log_nhi=map_nhi,
        delta_log_nhi=delta_nhi, elapsed_s=dt,
        filter_low_likelihood=filter_low_likelihood,
    )
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  [saved] {out_path}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", default="/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection")
    p.add_argument("--out-dir", default="docs/notes/2026-05-06_corrected_model_validation/canonical_tid")
    p.add_argument("--filter", action="store_true", default=True,
                   help="use FILTER=True (post-fix path; default)")
    args = p.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, model_path in MODELS:
        try:
            r = _run_one(name, model_path, args.data_root, out_dir,
                         filter_low_likelihood=args.filter)
            if r is not None:
                results.append(r)
        except Exception as ex:
            print(f"  ERROR: {ex!r}")
            import traceback; traceback.print_exc()

    # markdown summary
    md = [f"# Canonical TID {TARGET_ID} per model (truth log_NHI = {TRUTH_LOG_NHI})",
          "",
          "| model | p_DLA | MAP z_DLA | MAP log NHI | Δ NHI | elapsed |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        md.append(f"| {r['model']} | {r['p_dla']:.4f} | {r['map_z_dla']:.4f} | "
                  f"{r['map_log_nhi']:.3f} | {r['delta_log_nhi']:+.3f} | "
                  f"{r['elapsed_s']:.1f} s |")
    summary = out_dir.parent / "canonical_tid_summary.md"
    summary.write_text("\n".join(md) + "\n")
    print(f"\n[saved] {summary}")


if __name__ == "__main__":
    main()
