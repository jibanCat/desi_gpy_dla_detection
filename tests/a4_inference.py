"""Step A.4 — canonical TID inference per A.3 lane.

For each lane in {v1, v3.5, lbfgs, matlab}:
  1. Load that lane's trained params from short_retrain/<lane>.{npz,mat}.
  2. Save as .h5 matching the schema gpy_dla_detection's NullGPMAT/DLAGPMAT
     /SubDLAGPMAT loaders expect.
  3. Run canonical TID 120046865 (truth log NHI = 21.263) inference.
  4. Save p_DLA, MAP_z, MAP_log_NHI per lane.

Outputs:
  tests/fixtures/2lpt_frozen/short_retrain/h5/<lane>.h5
  tests/fixtures/2lpt_frozen/short_retrain/canonical_tid/<lane>.json
  tests/fixtures/2lpt_frozen/short_retrain/canonical_tid_summary.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIX = Path(__file__).resolve().parent / "fixtures" / "2lpt_frozen"
SR  = FIX / "short_retrain"
H5_DIR = SR / "h5"
TID_DIR = SR / "canonical_tid"

# Canonical fixture target: 2lpt mock-0 loa-124, TID 120046865, log_NHI=21.263
TARGET_ID = 120046865
TRUTH_LOG_NHI = 21.263
SPEC_PATH = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits"
ZCAT_PATH = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits"
DATA_ROOT = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"


def _load_lane(lane):
    """Returns dict with M, mu, log_omega, log_c_0, log_tau_0, log_beta, rest_wavelengths."""
    if lane == "matlab":
        m = loadmat(SR / "matlab.mat")
        return dict(
            M=np.asarray(m["M_final"]),
            mu=np.asarray(m["mu"]).squeeze(),
            log_omega=np.asarray(m["log_omega_final"]).squeeze(),
            log_c_0=float(np.asarray(m["log_c_0_final"]).squeeze()),
            log_tau_0=float(np.asarray(m["log_tau_0_final"]).squeeze()),
            log_beta=float(np.asarray(m["log_beta_final"]).squeeze()),
            rest_wavelengths=np.asarray(m["rest_wavelengths"]).squeeze(),
        )
    p = SR / f"{lane}.npz"
    if not p.exists():
        return None
    n = np.load(p)
    return dict(
        M=np.asarray(n["M_final"]),
        mu=np.asarray(n["mu"]),
        log_omega=np.asarray(n["log_omega_final"]),
        log_c_0=float(n["log_c_0_final"]),
        log_tau_0=float(n["log_tau_0_final"]),
        log_beta=float(n["log_beta_final"]),
        rest_wavelengths=np.asarray(n["rest_wavelengths"]),
    )


def _save_h5(lane, params):
    """Schema matches v1 model_epoch_*.h5: scalar log_c_0/log_tau_0/log_beta,
    1D mu/log_omega/rest_wavelengths, 2D M (n_pix, k). max_noise_variance =
    9.0 from v1 preset (Parameters.max_noise_variance default for DESI Y3).

    Does NOT write normalization_min/max_lambda — DLAHolder falls back to
    the preset's [1425, 1475] window which is what v1 production uses. The
    A.3 trainer didn't apply per-spectrum normalize (matching v1
    desi_learn_qsos_model.py:97-104 which has it commented out), so the
    inference will normalize while training did not. The resulting
    flux-scale mismatch on canonical TID is the documented mean-flux
    issue addressed at inference time by τ-EB (PR #5; out of scope here).
    """
    H5_DIR.mkdir(parents=True, exist_ok=True)
    out = H5_DIR / f"{lane}.h5"
    rw = params["rest_wavelengths"]
    with h5py.File(out, "w") as f:
        f.create_dataset("rest_wavelengths", data=rw.astype(np.float64))
        f.create_dataset("mu", data=params["mu"].astype(np.float64))
        f.create_dataset("log_omega", data=params["log_omega"].astype(np.float64))
        # M shape: (n_pix, k) with n_pix matching rest_wavelengths
        M = params["M"].astype(np.float64)
        if M.shape[0] != rw.shape[0]:
            M = M.T
        f.create_dataset("M", data=M)
        f.create_dataset("log_c_0", data=np.float64(params["log_c_0"]))
        f.create_dataset("log_tau_0", data=np.float64(params["log_tau_0"]))
        f.create_dataset("log_beta", data=np.float64(params["log_beta"]))
        f.create_dataset("max_noise_variance", data=np.float64(9.0))  # v1 preset
    return out


def _run_canonical_tid_inference(learned_h5, lane):
    from examples.smoke_one_spectrum import (
        load_one_desi_spectrum, lookup_z_qso, PRESETS,
    )
    from gpy_dla_detection.set_parameters import Parameters
    from run_bayes_select import DLAHolder

    print(f"\n=== {lane} (model: {learned_h5.name}) ===")
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
        learned_file=str(learned_h5),
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
        figure_dir="/tmp", single_absorber_model=False,
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
    delta = map_nhi - TRUTH_LOG_NHI

    print(f"  p_DLA       = {p_dla:.4f}")
    print(f"  MAP z_DLA   = {map_z:.4f}  (z_qso = {z_qso:.4f})")
    print(f"  MAP log NHI = {map_nhi:.4f}  Δ = {delta:+.3f} dex (truth = {TRUTH_LOG_NHI})")
    print(f"  elapsed     = {dt:.1f}s")
    out = dict(
        lane=lane,
        learned_h5=str(learned_h5),
        target_id=TARGET_ID, truth_log_nhi=TRUTH_LOG_NHI, z_qso=z_qso,
        p_dla=p_dla, map_z_dla=map_z, map_log_nhi=map_nhi,
        delta_log_nhi=delta, elapsed_s=dt,
    )
    TID_DIR.mkdir(parents=True, exist_ok=True)
    p_out = TID_DIR / f"{lane}.json"
    p_out.write_text(json.dumps(out, indent=2))
    print(f"  [saved] {p_out}")
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lanes", nargs="+",
                   default=["v1", "v3.5", "lbfgs", "matlab"])
    args = p.parse_args()

    results = []
    for lane in args.lanes:
        params = _load_lane(lane)
        if params is None:
            print(f"\n[skip] {lane}: trained params not found")
            continue
        h5_path = _save_h5(lane, params)
        try:
            r = _run_canonical_tid_inference(h5_path, lane)
            results.append(r)
        except Exception as ex:
            print(f"  ERROR: {ex!r}")
            import traceback; traceback.print_exc()

    md = [f"# Step A.4 canonical TID {TARGET_ID} per A.3 lane",
          f"truth log_NHI = {TRUTH_LOG_NHI}",
          "",
          "| lane | p_DLA | MAP z_DLA | MAP log NHI | Δ NHI | elapsed |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        md.append(f"| {r['lane']} | {r['p_dla']:.4f} | {r['map_z_dla']:.4f} | "
                  f"{r['map_log_nhi']:.3f} | {r['delta_log_nhi']:+.3f} | "
                  f"{r['elapsed_s']:.1f} s |")
    summary = SR / "canonical_tid_summary.md"
    summary.write_text("\n".join(md) + "\n")
    print(f"\n[saved] {summary}")


if __name__ == "__main__":
    main()
