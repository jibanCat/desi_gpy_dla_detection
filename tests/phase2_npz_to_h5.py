"""Convert phase2_result.npz → phase2_result.h5 in DESI learned-model schema.

The DESI inference loader (`null_gp.py:453-468`) expects an HDF5 file
with these keys at top level:

  M                          (n_pixels, k)        float
  mu                         (n_pixels,)          float
  log_omega                  (n_pixels,)          float
  log_c_0                    scalar               float
  log_tau_0                  scalar               float
  log_beta                   scalar               float
  rest_wavelengths           (n_pixels,)          float
  max_noise_variance         scalar               float
  normalization_min_lambda   scalar               float
  normalization_max_lambda   scalar               float

The earlier production retrains (vec_full / per_spec / vec_smoke /
Phase-1) wrote `.npz` files instead of `.h5`. This converter back-fills
the .h5 (one-off) so they can be loaded by `DLAHolder` for inference
testing. Going forward, `phase2_train_dr16.py` writes both formats.

Usage:
    python tests/phase2_npz_to_h5.py PATH/TO/phase2_result.npz
    # writes PATH/TO/phase2_result.h5

The three "required for inference loader" scalars
(max_noise_variance, normalization_min_lambda, normalization_max_lambda)
match the Parameters defaults in use during DR16 training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

# DR16 training-time scalars (must match what the trainer used)
MAX_NOISE_VARIANCE = 9.0  # tests/phase2_train_dr16.py:68 MAX_NV
NORM_MIN_LAMBDA = 1310.0  # gpy_dla_detection/set_parameters.py:31 default
NORM_MAX_LAMBDA = 1325.0  # gpy_dla_detection/set_parameters.py:32 default


def npz_to_h5(npz_path: Path, h5_path: Path | None = None) -> Path:
    if h5_path is None:
        h5_path = npz_path.with_suffix(".h5")
    n = np.load(npz_path)

    with h5py.File(h5_path, "w") as f:
        # learned-model schema (DESI branch in null_gp.load_learned_qso_model)
        f.create_dataset("M", data=np.asarray(n["M"], dtype=np.float64))
        f.create_dataset("mu", data=np.asarray(n["mu"], dtype=np.float64))
        f.create_dataset("log_omega", data=np.asarray(n["log_omega"], dtype=np.float64))
        f.create_dataset("log_c_0", data=np.float64(n["log_c_0"]))
        f.create_dataset("log_tau_0", data=np.float64(n["log_tau_0"]))
        f.create_dataset("log_beta", data=np.float64(n["log_beta"]))
        f.create_dataset("rest_wavelengths",
                         data=np.asarray(n["rest_wavelengths"], dtype=np.float64))
        # required for inference loader (training-time scalars)
        f.create_dataset("max_noise_variance", data=np.float64(MAX_NOISE_VARIANCE))
        f.create_dataset("normalization_min_lambda", data=np.float64(NORM_MIN_LAMBDA))
        f.create_dataset("normalization_max_lambda", data=np.float64(NORM_MAX_LAMBDA))
        # provenance — not read by the inference loader, useful for audit
        if "n_spectra" in n.files:
            f.attrs["n_spectra"] = int(n["n_spectra"])
        if "n_iters" in n.files:
            f.attrs["n_iters"] = int(n["n_iters"])
        if "lr" in n.files:
            f.attrs["lr"] = float(n["lr"])
        f.attrs["source_npz"] = str(npz_path)

    return h5_path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("npz", type=Path, nargs="+",
                   help="One or more phase2_result.npz files to convert.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output .h5 path (only valid for single input). "
                        "Default: alongside input with .h5 suffix.")
    args = p.parse_args()

    if args.out and len(args.npz) > 1:
        raise SystemExit("--out only valid for a single input")

    for npz in args.npz:
        if not npz.exists():
            raise SystemExit(f"not found: {npz}")
        out = args.out if args.out else npz.with_suffix(".h5")
        h5 = npz_to_h5(npz, out)
        print(f"[converted] {npz}  →  {h5}")
        # quick sanity readback
        with h5py.File(h5, "r") as f:
            print(f"  keys: {sorted(f.keys())}")
            print(f"  M shape: {f['M'].shape}   "
                  f"log_c_0={float(f['log_c_0'][()]):.6f}   "
                  f"log_tau_0={float(f['log_tau_0'][()]):.6f}   "
                  f"log_beta={float(f['log_beta'][()]):.6f}")


if __name__ == "__main__":
    main()
