"""Layer 4: byte-stable parity between Python spectrum_loss and the
DR16Q-public MATLAB reference at
``/home/mfho/gp_dla_detection_dr16q_public/spectrum_loss.m``.

Both implementations evaluate

    -log p(y | M, ω, c₀, τ₀, β, num_forest_lines, lya_1pz, zqso_1pz)

and the analytical gradients (dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta).
This driver runs each on the same fixed-seed synthetic input and reports the
maximum absolute / relative discrepancy.

Pass criterion (defaulting to float64): each output element matches to
``rtol = atol = 1e-12`` — well below the autograd parity already established
in tests/test_objective_math.py.

Run::

    python tests/parity/matlab_parity_check.py [--out tests/parity/results.md]

Requires:
    - PYTHON: torch (already installed) + scipy.io for .mat round-trip
    - MATLAB: ``module load matlab/R2024b`` (or any R2019+) on GreatLakes,
      or set MATLAB_BIN to a different binary.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

# Make repo root importable when invoked as a script.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from gpy_dla_detection.objective import spectrum_loss  # noqa: E402
from gpy_dla_detection.voigt import (  # noqa: E402
    transition_wavelengths as TRANSITION_WAVELENGTHS_NP,
    oscillator_strengths as OSCILLATOR_STRENGTHS_NP,
)


DTYPE = torch.float64
N = 32
K = 4
NUM_FOREST_LINES = 5


def _make_inputs(seed: int = 1234):
    """Same kind of inputs as Layer 1 test_objective_math.py but with NUM_FOREST_LINES=5
    and a slightly larger (n, k) so the Woodbury path gets exercised non-trivially.
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(N, generator=g, dtype=DTYPE) * 0.4
    noise_variance = 0.01 + 0.05 * torch.rand(N, generator=g, dtype=DTYPE)

    # Force all Lyman series lines to land below z_qso so the indicator
    # is always 1.
    one_pz_lya = 2.6 + 0.2 * torch.rand(N, generator=g, dtype=DTYPE)  # in (2.6, 2.8)
    zqso_1pz = torch.tensor(4.0, dtype=DTYPE)

    log_omega = torch.randn(N, generator=g, dtype=DTYPE) * 0.5
    log_c_0 = torch.tensor(math.log(0.05), dtype=DTYPE)
    log_tau_0 = torch.tensor(math.log(0.0025), dtype=DTYPE)
    log_beta = torch.tensor(math.log(3.6), dtype=DTYPE)

    M = torch.randn(N, K, generator=g, dtype=DTYPE)

    omega2 = torch.exp(2 * log_omega)
    c_0 = torch.exp(log_c_0)
    tau_0 = torch.exp(log_tau_0)
    beta = torch.exp(log_beta)

    transition_wavelengths = torch.tensor(TRANSITION_WAVELENGTHS_NP, dtype=DTYPE)
    oscillator_strengths = torch.tensor(OSCILLATOR_STRENGTHS_NP, dtype=DTYPE)

    return dict(
        y=y, noise_variance=noise_variance, lya_1pz=one_pz_lya, zqso_1pz=zqso_1pz,
        log_omega=log_omega, log_c_0=log_c_0, log_tau_0=log_tau_0, log_beta=log_beta,
        M=M, omega2=omega2, c_0=c_0, tau_0=tau_0, beta=beta,
        transition_wavelengths=transition_wavelengths,
        oscillator_strengths=oscillator_strengths,
        num_forest_lines=NUM_FOREST_LINES,
    )


def _run_python(inp):
    nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta = spectrum_loss(
        inp["y"], inp["lya_1pz"], inp["noise_variance"],
        inp["M"], inp["omega2"], inp["c_0"], inp["tau_0"], inp["beta"],
        inp["num_forest_lines"],
        inp["transition_wavelengths"], inp["oscillator_strengths"],
        inp["zqso_1pz"],
    )
    return {
        "nlog_p": nlog_p.detach().cpu().numpy(),
        "dM": dM.detach().cpu().numpy(),
        "dlog_omega": dlog_omega.detach().cpu().numpy(),
        "dlog_c_0": dlog_c_0.detach().cpu().numpy(),
        "dlog_tau_0": dlog_tau_0.detach().cpu().numpy(),
        "dlog_beta": dlog_beta.detach().cpu().numpy(),
    }


def _save_input_for_matlab(inp, path: Path):
    """Save inputs as a v7 .mat for MATLAB. Cast scalars and reshape vectors
    so MATLAB's column-vector conventions are honored (n×1, not n)."""
    def _col(x):
        a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        return a.reshape(-1, 1) if a.ndim == 1 else a

    def _scalar(x):
        a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        return float(a) if a.ndim == 0 else float(a.item())

    payload = {
        "y": _col(inp["y"]),
        "lya_1pz": _col(inp["lya_1pz"]),
        "noise_variance": _col(inp["noise_variance"]),
        "M": inp["M"].detach().cpu().numpy(),
        "omega2": _col(inp["omega2"]),
        "c_0": _scalar(inp["c_0"]),
        "tau_0": _scalar(inp["tau_0"]),
        "beta": _scalar(inp["beta"]),
        "num_forest_lines": int(inp["num_forest_lines"]),
        "transition_wavelengths": _col(inp["transition_wavelengths"]),
        "oscillator_strengths": _col(inp["oscillator_strengths"]),
        "zqso_1pz": _scalar(inp["zqso_1pz"]),
    }
    savemat(str(path), payload, do_compression=False, format="5")


def _run_matlab(matlab_bin: str, input_path: Path, output_path: Path) -> str:
    """Spawn MATLAB to run matlab_parity_check.m and return its stderr/stdout."""
    if shutil.which(matlab_bin) is None:
        raise RuntimeError(
            f"MATLAB binary not on PATH: {matlab_bin!r}. "
            f"Run `module load matlab/R2024b` first."
        )

    cmd = [
        matlab_bin,
        "-batch",
        f"addpath('{_HERE}'); matlab_parity_check('{input_path}', '{output_path}'); exit",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"MATLAB exited {res.returncode}.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    return res.stdout + res.stderr


def _load_matlab(path: Path) -> dict:
    raw = loadmat(str(path))
    return {
        "nlog_p": np.asarray(raw["nlog_p"]).reshape(()).astype(np.float64),
        "dM": np.asarray(raw["dM"]).astype(np.float64),
        "dlog_omega": np.asarray(raw["dlog_omega"]).reshape(-1).astype(np.float64),
        "dlog_c_0": np.asarray(raw["dlog_c_0"]).reshape(()).astype(np.float64),
        "dlog_tau_0": np.asarray(raw["dlog_tau_0"]).reshape(()).astype(np.float64),
        "dlog_beta": np.asarray(raw["dlog_beta"]).reshape(()).astype(np.float64),
    }


def _compare(py: dict, ml: dict, rtol: float = 1e-12, atol: float = 1e-12):
    """Return list of dicts, one per output, with max abs diff and rel diff."""
    rows = []
    for key in ["nlog_p", "dM", "dlog_omega", "dlog_c_0", "dlog_tau_0", "dlog_beta"]:
        p = np.asarray(py[key]).astype(np.float64)
        m = np.asarray(ml[key]).astype(np.float64)
        if p.shape != m.shape:
            # MATLAB's column-vector convention: dM shape (n, k); dlog_omega (n,1) vs (n,).
            # Reshape MATLAB output to match Python.
            m = m.reshape(p.shape)
        diff = p - m
        max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
        max_rel = float(
            np.max(np.abs(diff) / np.maximum(np.abs(p), np.abs(m)))
        ) if diff.size and np.any(np.abs(p) > 0) else 0.0
        ok = np.allclose(p, m, rtol=rtol, atol=atol)
        rows.append({
            "name": key,
            "shape": str(p.shape),
            "max_abs": max_abs,
            "max_rel": max_rel,
            "ok": ok,
        })
    return rows


def _format_results(rows, rtol: float, atol: float) -> str:
    md = []
    md.append("# Layer 4 — MATLAB ↔ Python parity for spectrum_loss\n")
    md.append("Reference: `/home/mfho/gp_dla_detection_dr16q_public/spectrum_loss.m`")
    md.append("Python:    `gpy_dla_detection/objective.py::spectrum_loss`\n")
    md.append(f"Tolerance: rtol = atol = {rtol:.0e}\n")
    md.append("| output | shape | max\\|Δ\\| | max rel\\|Δ\\| | pass |")
    md.append("|---|---|---:|---:|:---:|")
    for r in rows:
        md.append(
            f"| `{r['name']}` | {r['shape']} | {r['max_abs']:.3e} | "
            f"{r['max_rel']:.3e} | {'✅' if r['ok'] else '❌'} |"
        )
    overall = all(r["ok"] for r in rows)
    md.append("")
    md.append(f"**Overall: {'PASS' if overall else 'FAIL'}**\n")
    return "\n".join(md)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matlab-bin", default=os.environ.get("MATLAB_BIN", "matlab"))
    p.add_argument("--out", default=str(_HERE / "results.md"))
    p.add_argument("--rtol", type=float, default=1e-12)
    p.add_argument("--atol", type=float, default=1e-12)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    inp = _make_inputs(seed=args.seed)
    py_out = _run_python(inp)

    input_mat = _HERE / "_input.mat"
    matlab_out = _HERE / "_matlab_outputs.mat"
    _save_input_for_matlab(inp, input_mat)

    print(f"[parity] Running MATLAB: {args.matlab_bin}", flush=True)
    log = _run_matlab(args.matlab_bin, input_mat, matlab_out)
    print("[parity] MATLAB done.", flush=True)
    if log.strip():
        for line in log.splitlines()[-5:]:
            print(f"   matlab> {line}")

    ml_out = _load_matlab(matlab_out)
    rows = _compare(py_out, ml_out, rtol=args.rtol, atol=args.atol)
    md = _format_results(rows, rtol=args.rtol, atol=args.atol)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print()
    print(md)
    print(f"\n[parity] wrote {out_path}")

    # Cleanup intermediate files unless KEEP_MAT=1.
    if not os.environ.get("KEEP_MAT"):
        for f in (input_mat, matlab_out):
            try:
                f.unlink()
            except FileNotFoundError:
                pass

    sys.exit(0 if all(r["ok"] for r in rows) else 1)


if __name__ == "__main__":
    main()
