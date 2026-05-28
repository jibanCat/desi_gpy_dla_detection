"""Convert a phase2_train_desi checkpoint .pt → DESI learned-model .h5.

Use this when:
  - A SLURM job walltime-killed mid-training and you want the partial model
    in production-loadable .h5 form (you'd otherwise need to resume + finish).
  - An old checkpoint is missing rest_wavelengths (for .pt files saved
    before commit X — we now always save it inside _save_checkpoint).
    Pass --rest-wavelengths-from <preload.h5> to recover the grid from
    the original training preload.

Output schema matches `null_gp.NullGPMAT.__init__` (DESI branch):
  M, mu, log_omega, log_c_0, log_tau_0, log_beta, rest_wavelengths,
  max_noise_variance, normalization_min_lambda, normalization_max_lambda

Usage::

    # Modern checkpoint (rest_wavelengths included)
    python tests/phase2_pt_to_h5.py \\
        /scratch/.../phase2_desi_checkpoint_final_iter1499.pt \\
        --out trained_model.h5

    # Old checkpoint missing rest_wavelengths — recover from the preload
    python tests/phase2_pt_to_h5.py \\
        /scratch/.../phase2_desi_checkpoint_final_iter1499.pt \\
        --out trained_model.h5 \\
        --rest-wavelengths-from /nfs/.../v2_runs/2lpt_loa0_wide_v2_*/trainset.h5

    # Last-resort: explicit linspace (NOT recommended; verify against the actual preload)
    python tests/phase2_pt_to_h5.py \\
        old.pt --out trained_model.h5 \\
        --rest-min 850.75 --rest-max 1700.0 --n-pix 5662
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


# DR16 / DESI training-time scalars (must match what the trainer used).
MAX_NOISE_VARIANCE = 9.0
NORM_MIN_LAMBDA = 1310.0
NORM_MAX_LAMBDA = 1325.0


def _resolve_rest_wavelengths(ckpt: dict, args: argparse.Namespace) -> np.ndarray:
    """Pick the best available source for rest_wavelengths."""
    n_pix = int(np.asarray(ckpt["M"]).shape[0])
    # 1) Embedded in the checkpoint (modern path)
    if ckpt.get("rest_wavelengths") is not None:
        rw = np.asarray(ckpt["rest_wavelengths"], dtype=np.float64)
        if rw.shape[0] != n_pix:
            raise ValueError(f"checkpoint rest_wavelengths has shape {rw.shape}, "
                             f"M expects {n_pix}")
        print(f"[rest] using rest_wavelengths from checkpoint "
              f"(n={n_pix}, [{rw[0]:.2f}, {rw[-1]:.2f}], dλ={rw[1]-rw[0]:.4f})")
        return rw
    # 2) From the original preload .h5 (--rest-wavelengths-from)
    if args.rest_wavelengths_from is not None:
        with h5py.File(args.rest_wavelengths_from, "r") as f:
            if "rest_wavelengths" in f:
                rwd = f["rest_wavelengths"]
                rw = (rwd[0] if rwd.ndim == 2 else rwd[:]).astype(np.float64)
            elif "rest_wavelength_list" in f:
                rwd = f["rest_wavelength_list"]
                rw = (rwd[0] if rwd.ndim == 2 else rwd[:]).astype(np.float64)
            else:
                raise KeyError(f"no rest_wavelengths in {args.rest_wavelengths_from}")
        if rw.shape[0] != n_pix:
            raise ValueError(
                f"preload rest_wavelengths has shape {rw.shape}, M expects {n_pix}. "
                f"Did you point at the wrong preload?"
            )
        print(f"[rest] using rest_wavelengths from {args.rest_wavelengths_from} "
              f"(n={n_pix}, [{rw[0]:.2f}, {rw[-1]:.2f}], dλ={rw[1]-rw[0]:.4f})")
        return rw
    # 3) Manual linspace (last resort)
    if args.rest_min is not None and args.rest_max is not None and args.n_pix is not None:
        if args.n_pix != n_pix:
            raise ValueError(f"--n-pix {args.n_pix} does not match M.shape[0] {n_pix}")
        rw = np.linspace(args.rest_min, args.rest_max, n_pix, dtype=np.float64)
        print(f"[rest] WARNING: using linspace({args.rest_min}, {args.rest_max}, {n_pix}). "
              f"Verify this matches the preload that produced this checkpoint!")
        return rw
    raise SystemExit(
        "rest_wavelengths not in checkpoint. Pass --rest-wavelengths-from <preload.h5> "
        "(recommended) or --rest-min/--rest-max/--n-pix (last resort)."
    )


def pt_to_h5(pt_path: Path, h5_path: Path, *,
             rest_wavelengths: np.ndarray | None = None,
             max_noise_variance: float = MAX_NOISE_VARIANCE,
             norm_min_lambda: float = NORM_MIN_LAMBDA,
             norm_max_lambda: float = NORM_MAX_LAMBDA) -> Path:
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    M = np.asarray(ckpt["M"], dtype=np.float64)
    mu = np.asarray(ckpt["mu"], dtype=np.float64)
    log_omega = np.asarray(ckpt["log_omega"], dtype=np.float64)
    log_c_0 = float(np.asarray(ckpt["log_c_0"]))
    log_tau_0 = float(np.asarray(ckpt["log_tau_0"]))
    log_beta = float(np.asarray(ckpt["log_beta"]))
    if rest_wavelengths is None:
        raise ValueError("rest_wavelengths required (can't infer from .pt)")

    # Sanity: shapes
    n_pix, k = M.shape
    assert mu.shape == (n_pix,), f"mu shape {mu.shape} vs n_pix {n_pix}"
    assert log_omega.shape == (n_pix,), f"log_omega shape {log_omega.shape} vs n_pix {n_pix}"
    assert rest_wavelengths.shape == (n_pix,), \
        f"rest_wavelengths shape {rest_wavelengths.shape} vs n_pix {n_pix}"

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("M", data=M)
        f.create_dataset("mu", data=mu)
        f.create_dataset("log_omega", data=log_omega)
        f.create_dataset("log_c_0", data=np.float64(log_c_0))
        f.create_dataset("log_tau_0", data=np.float64(log_tau_0))
        f.create_dataset("log_beta", data=np.float64(log_beta))
        f.create_dataset("rest_wavelengths", data=rest_wavelengths)
        f.create_dataset("max_noise_variance", data=np.float64(max_noise_variance))
        f.create_dataset("normalization_min_lambda", data=np.float64(norm_min_lambda))
        f.create_dataset("normalization_max_lambda", data=np.float64(norm_max_lambda))
        f.attrs["source_pt"] = str(pt_path)
        f.attrs["iter_completed"] = int(np.asarray(ckpt.get("iter_completed", -1)))
    return h5_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("pt", type=Path, help="phase2_desi_checkpoint_*.pt input")
    p.add_argument("--out", type=Path, required=True, help="output .h5 path")
    p.add_argument("--rest-wavelengths-from", type=Path, default=None,
                   help="Original training preload .h5 to copy rest_wavelengths from "
                        "(used when the .pt was saved before rest_wavelengths was embedded).")
    p.add_argument("--rest-min", type=float, default=None,
                   help="Last-resort manual linspace lower edge (verify against preload).")
    p.add_argument("--rest-max", type=float, default=None)
    p.add_argument("--n-pix", type=int, default=None,
                   help="Last-resort manual linspace n_pix (must match M.shape[0]).")
    p.add_argument("--max-noise-variance", type=float, default=MAX_NOISE_VARIANCE)
    p.add_argument("--norm-min-lambda", type=float, default=NORM_MIN_LAMBDA)
    p.add_argument("--norm-max-lambda", type=float, default=NORM_MAX_LAMBDA)
    args = p.parse_args()

    if not args.pt.exists():
        raise SystemExit(f"not found: {args.pt}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.pt, map_location="cpu", weights_only=False)
    rw = _resolve_rest_wavelengths(ckpt, args)
    out = pt_to_h5(args.pt, args.out,
                   rest_wavelengths=rw,
                   max_noise_variance=args.max_noise_variance,
                   norm_min_lambda=args.norm_min_lambda,
                   norm_max_lambda=args.norm_max_lambda)
    print(f"[converted] {args.pt}  →  {out}")
    with h5py.File(out, "r") as f:
        print(f"  keys: {sorted(f.keys())}")
        print(f"  M shape: {f['M'].shape}   "
              f"log_c_0={float(f['log_c_0'][()]):.6f}   "
              f"log_tau_0={float(f['log_tau_0'][()]):.6f}   "
              f"log_beta={float(f['log_beta'][()]):.6f}")


if __name__ == "__main__":
    main()
