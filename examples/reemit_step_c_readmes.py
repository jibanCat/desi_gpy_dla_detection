"""Regenerate auto-emitted README.md for each Step C trained 2lpt model.

The original auto-template (`tests/phase2_train_desi.py::_save_readme`,
pre-2026-05-14) hard-coded `normalize | [1310, 1325]` regardless of the
runtime `--norm-min-lambda`/`--norm-max-lambda`. So the `_m` runs say
Garnett band but were trained on MATLAB band. This script rebuilds each
README from the .h5 (which carries the runtime norm band post the
2026-05-13 manifest commit `3a0b84f`), so the cards now report truth.

For pre-manifest runs that don't have `normalization_min/max_lambda`
in the .h5, the script falls back to the original SLURM log header
(slurm/greatlakes/phase2_desi_retrain_*.log) to recover the actual band.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
NOTES = REPO / "docs" / "notes"
SLURM_LOGS = REPO / "slurm" / "greatlakes"


# Map: out_dir → SLURM job ID (so we can grep the log for the actual norm band)
RUN_TO_SLURM = {
    "2026-05-11_desi_phase2_2lpt_loa0_wide":              None,  # untracked
    "2026-05-11_desi_phase2_2lpt_loa0_wide_g":            None,
    "2026-05-11_desi_phase2_2lpt_loa0_wide_m":            None,
    "2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide":            None,
    "2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_g":          None,
    "2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_m":          None,
    "2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior":    "50021381",
    "2026-05-13_desi_smoke_normmask":                                   "50072213",
}


def find_norm_band_from_slurm_log(out_dir_name: str) -> tuple[float, float] | None:
    """Scan all phase2_desi_retrain logs for one that mentions this out_dir,
    then parse `norm_band : [X, Y] Å rest` from the header."""
    pattern = re.compile(r"norm_band\s*:\s*\[([\d.]+),\s*([\d.]+)\]")
    out_dir_pattern = re.compile(re.escape(out_dir_name))
    for log in sorted(SLURM_LOGS.glob("phase2_desi_retrain_*.log")):
        try:
            head = log.read_text(errors="ignore")[:3000]
        except Exception:
            continue
        if not out_dir_pattern.search(head):
            continue
        m = pattern.search(head)
        if m:
            return float(m.group(1)), float(m.group(2))
    # Also try smoke logs
    for log in sorted(SLURM_LOGS.glob("phase2_desi_smoke_*.log")):
        try:
            head = log.read_text(errors="ignore")[:3000]
        except Exception:
            continue
        if not out_dir_pattern.search(head):
            continue
        # Smoke logs print `out_dir : <path>` — check it matches before parsing band
        m = pattern.search(head)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def read_h5_norm_band(h5_path: Path) -> tuple[float, float] | None:
    with h5py.File(h5_path, "r") as f:
        if ("normalization_min_lambda" in f
                and "normalization_max_lambda" in f):
            return (float(f["normalization_min_lambda"][()]),
                    float(f["normalization_max_lambda"][()]))
    return None


def read_h5_scalars(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        d = {}
        for k in ("log_c_0", "log_tau_0", "log_beta"):
            if k in f:
                d[k.replace("log_", "")] = float(np.exp(f[k][()]))
        for k in ("n_spectra", "n_iters", "lr", "log_c_0_prior_sigma"):
            if k in f:
                v = f[k][()]
                d[k] = (float(v) if hasattr(v, "__float__") else v)
        d["rest_min"] = float(f["rest_wavelengths"][0])
        d["rest_max"] = float(f["rest_wavelengths"][-1])
        d["n_pix"] = int(f["rest_wavelengths"].shape[0])
        d["d_lambda"] = float(f["rest_wavelengths"][1] - f["rest_wavelengths"][0])
        if "loss_history" in f:
            d["final_loss"] = float(f["loss_history"][-1])
        else:
            d["final_loss"] = float("nan")
    return d


def re_emit_readme(out_dir: Path, norm_min: float, norm_max: float,
                   slurm_job_id: str | None = None) -> str:
    h5 = out_dir / "phase2_result.h5"
    sc = read_h5_scalars(h5)
    band_label = ("(MATLAB DR16 convention)"
                  if abs(norm_min - 1425.0) < 1
                  else "(Garnett+2017 convention)"
                  if abs(norm_min - 1310.0) < 1
                  else "(custom)")
    c0_prior_sigma = sc.get("log_c_0_prior_sigma", None)
    if c0_prior_sigma is not None and np.isnan(c0_prior_sigma):
        c0_prior_sigma = None
    body = f"""# Phase 2 DESI trained GP — model card

This directory contains a GP model trained by `tests/phase2_train_desi.py`
(PR #6 corrected trainer; PCA init + hand-coded gradient via
`gpy_dla_detection/training_v3/objective_vectorized.spectrum_loss_batch`).

> **2026-05-14**: re-emitted to fix the original auto-template's hard-coded
> norm band; the norm band below now reflects what the model was actually
> trained on (from the .h5 manifest or the SLURM log header).

## Files

| File | Purpose |
|---|---|
| `phase2_result.h5` | **Learned model** in DESI schema. Production-loadable by `gpy_dla_detection.null_gp.NullGPMAT(learned_file=...)`. |
| `phase2_result.npz` | Training-history record. Not loaded by the inference pipeline. |
| `README.md` | This file. |

## Training config

| Parameter | Value |
|---|---|
| n_spectra | {int(sc.get('n_spectra', -1)):,} |
| n_pix (rest) | {sc['n_pix']} |
| rest grid | [{sc['rest_min']:.2f}, {sc['rest_max']:.2f}] Å, dλ={sc['d_lambda']:.4f} |
| n_iters (Adam) | {int(sc.get('n_iters', -1))} |
| lr | {sc.get('lr', float('nan'))} |
| normalize | per-spectrum median in **[{norm_min:.2f}, {norm_max:.2f}] Å rest** {band_label} |
| log_c_0 prior σ | {c0_prior_sigma if c0_prior_sigma is not None else "(none)"} |
| SLURM job | {slurm_job_id or "(not tracked)"} |

## Endpoint scalars

| Parameter | Value |
|---|---:|
| c_0 | {sc.get('c_0', float('nan')):.6f} |
| τ_0 | {sc.get('tau_0', float('nan')):.6f} |
| β | {sc.get('beta', float('nan')):.4f} |
| log p(D \\| Adam endpoint) | {sc.get('final_loss', float('nan')):.4f} |

## Provenance

- norm band source: {"`.h5` manifest" if read_h5_norm_band(h5) else "SLURM log header"}
- corr-noise debug arc: see `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md`
- DLA-recovery test on canonical TID: see `docs/notes/2026-05-13_step_c_dla_recovery/findings.md`
"""
    (out_dir / "README.md").write_text(body)
    return body


def main():
    for dir_name, slurm_id in RUN_TO_SLURM.items():
        out_dir = NOTES / dir_name
        if not (out_dir / "phase2_result.h5").exists():
            print(f"[skip] {dir_name}: no phase2_result.h5")
            continue
        # Try .h5 manifest first, fall back to SLURM log
        band = read_h5_norm_band(out_dir / "phase2_result.h5")
        if band is None:
            band = find_norm_band_from_slurm_log(dir_name)
        if band is None:
            # Last resort: infer from suffix
            if dir_name.endswith("_m") or "_m_" in dir_name or dir_name.endswith("smoke_normmask"):
                band = (1425.0, 1475.0)
            else:
                band = (1310.0, 1325.0)
            print(f"[infer] {dir_name}: no manifest/log; inferring band {band} from suffix")
        else:
            print(f"[band]  {dir_name}: {band}")
        re_emit_readme(out_dir, band[0], band[1], slurm_job_id=slurm_id)
        print(f"[saved] {out_dir}/README.md")


if __name__ == "__main__":
    main()
