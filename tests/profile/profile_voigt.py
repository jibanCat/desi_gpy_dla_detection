"""Profile Voigt forward-model variants.

Measures wall time per Voigt evaluation across:

  - v1 production C extension via voigt_fast.VoigtProfile
    (gpy_dla_detection/_voigt.so)
  - v2 pure-Python via voigt_v2.voigt_absorption with each kernel:
      • boss-log-r2000  (parity to v1)
      • desi-linear-r3000
      • desi-linear-r5000
      • none
  - GPU-port: torch + scipy.special.wofz on GPU? wofz is not in torch,
    so the GPU port uses Faddeeva via Humlicek's algorithm. See
    `voigt_v2_torch.py` companion (TODO; for now this profile exercises
    just CPU variants).

Usage::

    python tests/profile/profile_voigt.py [--num-eval 10000]
                                           [--n-pix 600]
                                           [--out tests/profile/results/voigt_profile.md]

Output: a markdown table written to OUT, plus stdout summary.

What this is for: in inference (run_bayes_select.process_qso), each
spectrum requires N_DLA_samples × num_lines Voigt evaluations
(~100k × 3 = 300k per spectrum), so per-evaluation cost dominates.
The C extension was the production answer; v2 is a pure-Python
alternative for studying LSF mismatches. Need to measure whether
the v2 cost per call is acceptable (~5-50× slower expected).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))


def _make_grid(n_pix: int = 600) -> np.ndarray:
    """A representative DESI rest-grid window: λ_obs ≈ 4500 Å, dλ=0.15 Å."""
    return np.linspace(4500.0, 4500.0 + 0.15 * (n_pix - 1), n_pix)


def time_callable(fn: Callable, n_iter: int, warmup: int = 5) -> tuple[float, float]:
    """Run fn() n_iter times after warmup. Return (mean_us, std_us)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    return float(np.mean(times)), float(np.std(times))


def profile_v1_c_extension(wave: np.ndarray, n_iter: int):
    """Production voigt_fast (C extension via ctypes)."""
    try:
        from gpy_dla_detection.voigt_fast import VoigtProfile
    except (ImportError, OSError) as e:
        return {"name": "v1_voigt_fast", "skip": str(e)}
    vp = VoigtProfile()
    nhi = 10 ** 21.0
    z = 2.5
    out = vp.compute_voigt_profile(wave, nhi=nhi, z_dla=z, num_lines=3)
    fn = lambda: vp.compute_voigt_profile(wave, nhi=nhi, z_dla=z, num_lines=3)
    mean, std = time_callable(fn, n_iter)
    return {
        "name": "v1_voigt_fast (C ext)",
        "kernel": "boss-log-r2000",
        "num_lines": 3,
        "out_len": int(len(out)),
        "mean_us": mean, "std_us": std,
    }


def profile_v2(wave: np.ndarray, kernel: str, n_iter: int, num_lines: int = 3):
    """Pure-Python voigt_v2."""
    from gpy_dla_detection.voigt_v2 import voigt_absorption
    fn = lambda: voigt_absorption(
        wave, log_nhi=21.0, z_dla=2.5,
        num_lines=num_lines, kernel=kernel, dlambda_A=float(np.diff(wave)[0]),
    )
    out = fn()
    mean, std = time_callable(fn, n_iter)
    return {
        "name": f"v2 ({kernel})",
        "kernel": kernel,
        "num_lines": num_lines,
        "out_len": int(len(out)),
        "mean_us": mean, "std_us": std,
    }


def profile_v2_torch_cpu(wave: np.ndarray, n_iter: int, num_lines: int = 3,
                         kernel: str = "none"):
    """v2 ported to torch, running on CPU. Useful for measuring the
    overhead of moving to the differentiable framework before testing
    on GPU. NOTE: torch's complex-valued wofz isn't built-in; this
    uses scipy.special.wofz wrapped in a torch context."""
    try:
        import torch
        from scipy.special import wofz
        from gpy_dla_detection.voigt_v2 import (
            _C_CGS, _GAMMAS_CM_S, _LEADING_CONSTS_CM2, _SIGMA, _TRANS_WAV_CM,
        )
    except ImportError as e:
        return {"name": f"v2_torch_cpu ({kernel})", "skip": str(e)}

    device = torch.device("cpu")
    wave_t = torch.tensor(wave, dtype=torch.float64, device=device)
    z_dla = 2.5
    log_nhi = 21.0

    def _fn():
        wave_cm = wave_t * 1e-8
        total = torch.zeros_like(wave_cm)
        for j in range(num_lines):
            lam_line = _TRANS_WAV_CM[j]
            vel = wave_cm * (_C_CGS / (lam_line * (1 + z_dla))) - _C_CGS
            # Faddeeva via scipy → numpy → torch round-trip (for now).
            z = (vel.cpu().numpy() + 1j * _GAMMAS_CM_S[j]) / (_SIGMA * np.sqrt(2.0))
            voigt = np.real(wofz(z)) / (_SIGMA * np.sqrt(2 * np.pi))
            total = total - _LEADING_CONSTS_CM2[j] * torch.tensor(
                voigt, dtype=torch.float64, device=device,
            )
        profile = torch.exp((10.0 ** log_nhi) * total)
        return profile

    out = _fn()
    fn_no_ret = lambda: _fn().cpu().numpy()
    mean, std = time_callable(fn_no_ret, n_iter)
    return {
        "name": f"v2_torch_cpu ({kernel})",
        "kernel": kernel,
        "num_lines": num_lines,
        "out_len": int(len(out)),
        "mean_us": mean, "std_us": std,
        "note": "torch wraps numpy/scipy wofz — bottleneck unchanged",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--num-eval", type=int, default=2000,
                   help="Number of timed evaluations per variant (default 2000)")
    p.add_argument("--n-pix", type=int, default=600,
                   help="DESI rest-grid window size in pixels (default 600)")
    p.add_argument("--out", default="tests/profile/results/voigt_profile.md")
    args = p.parse_args()

    wave = _make_grid(args.n_pix)
    print(f"[profile] n_pix={args.n_pix} num_eval={args.num_eval}")

    rows = []
    rows.append(profile_v1_c_extension(wave, n_iter=args.num_eval))
    for kernel in ("boss-log-r2000", "desi-linear-r3000", "desi-linear-r5000", "none"):
        rows.append(profile_v2(wave, kernel=kernel, n_iter=args.num_eval, num_lines=3))
    # num_lines comparison at fixed kernel ('none' isolates Voigt cost).
    for nl in (1, 6, 12, 31):
        rows.append(profile_v2(wave, kernel="none", n_iter=args.num_eval, num_lines=nl))
    # Torch CPU port (mostly to see overhead — wofz isn't accelerated).
    rows.append(profile_v2_torch_cpu(wave, n_iter=max(50, args.num_eval // 10),
                                      num_lines=3, kernel="none"))

    md = []
    md.append(f"# Voigt-variant profiling\n")
    md.append(f"- n_pix = {args.n_pix} (DESI rest grid, ~600 pix per spectrum is typical)")
    md.append(f"- num_eval = {args.num_eval} per variant")
    md.append(f"- All times reported in **microseconds per Voigt call**.\n")
    md.append("## Kernel comparison (num_lines=3)\n")
    md.append("| variant | kernel | num_lines | output len | mean (μs) | std (μs) | note |")
    md.append("|---|---|---:|---:|---:|---:|---|")
    for r in rows:
        if r.get("skip"):
            md.append(f"| `{r['name']}` | — | — | — | (skipped: {r['skip'][:60]}) | — | — |")
        else:
            md.append(
                f"| `{r['name']}` | `{r['kernel']}` | {r['num_lines']} | "
                f"{r['out_len']} | **{r['mean_us']:.1f}** | {r['std_us']:.1f} | "
                f"{r.get('note', '')} |"
            )
    md.append("")

    # Per-spectrum projection.
    # Production multi-DLA inference: 100k DLA samples × 3 num_lines × Voigt
    # The dominant Voigt call inside dla_gp.py multiplies forward-modelled
    # absorption per QMC sample. Project total time per spectrum.
    md.append("## Per-spectrum projection (multi-DLA mode, FILTER=1)\n")
    md.append("Production inference per spectrum: roughly 100,000 QMC samples × 1–4 DLAs × num_lines Voigt evaluations. ")
    md.append("With FILTER=1 the truncated set is ~10–20 % of full QMC. ")
    md.append("Effective Voigt evaluations per spectrum: 10,000 × ~3 × num_lines ≈ 30k (low estimate, 1 DLA + filter) to 1.2M (4 DLAs no filter).\n")
    md.append("| variant | μs/call | s/spectrum (30k Voigt) | s/spectrum (300k Voigt) |")
    md.append("|---|---:|---:|---:|")
    for r in rows:
        if r.get("skip") or r["kernel"] != "none":
            continue
        ms_per_30k = r["mean_us"] * 30_000 / 1e6
        ms_per_300k = r["mean_us"] * 300_000 / 1e6
        md.append(
            f"| `{r['name']}` | {r['mean_us']:.1f} | {ms_per_30k:.1f} | {ms_per_300k:.1f} |"
        )
    md.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))
    print()
    print("\n".join(md[:25]))
    print(f"...\n[profile] full report → {out_path}")


if __name__ == "__main__":
    main()
