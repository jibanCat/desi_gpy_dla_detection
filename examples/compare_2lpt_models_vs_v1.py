"""Compare the two 2lpt-trained DESI models (PR #6 Step C) against v1 production.

Three models:
  - v1 production: /nfs/turbo/.../learnlogs/model_epoch_920.h5 (3798 pix, k=30)
  - 2lpt loa-0 wide: docs/notes/2026-05-11_desi_phase2_2lpt_loa0_wide/phase2_result.h5
  - 2lpt loa-124 nohcd-nobal wide: docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide/phase2_result.h5

Outputs (under `docs/notes/2026-05-12_2lpt_models_vs_v1_analysis/`):
  - endpoint_scalars.md
  - corr_matrices.png       (3-panel corr(M·M^T), per model, full rest range each)
  - mu_log_omega_overlay.png  (μ + log_omega per model on shared rest range)

Different rest grids (v1: 911-1500-ish, ours: 850.75-1700) → no direct
M·M^T diff. We compare structure side by side.

Usage:
    python examples/compare_2lpt_models_vs_v1.py
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
NOTES = REPO / "docs" / "notes"
OUT = NOTES / "2026-05-12_2lpt_models_vs_v1_analysis"

MODELS = {
    "v1_production": "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5",
    "2lpt_loa0_wide": str(NOTES / "2026-05-11_desi_phase2_2lpt_loa0_wide" / "phase2_result.h5"),
    "2lpt_loa124_nohcd_nobal_wide": str(NOTES / "2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide" / "phase2_result.h5"),
}


def _load(p: str) -> dict:
    with h5py.File(p, "r") as f:
        return dict(
            M=np.asarray(f["M"][:]),
            mu=np.asarray(f["mu"][:]),
            log_omega=np.asarray(f["log_omega"][:]),
            log_c_0=float(f["log_c_0"][()]),
            log_tau_0=float(f["log_tau_0"][()]),
            log_beta=float(f["log_beta"][()]),
            rest_wavelengths=np.asarray(f["rest_wavelengths"][:]),
        )


def _corr(M: np.ndarray) -> np.ndarray:
    K = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    return np.clip(K / np.outer(d, d), -1.0, 1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = {name: _load(path) for name, path in MODELS.items()}

    # --- 1) Endpoint scalars table ---
    md = ["# 2lpt trained models vs v1 production — endpoint scalars",
          "",
          f"v1 production: `{MODELS['v1_production']}`  ",
          "  Trained on real DESI Y3 LOA spectra (Y3 pipeline; epoch 920).",
          "",
          "2lpt models: from PR #6 Step C, trained 2026-05-11 on the v2 wide preload",
          "(rest grid [850.75, 1700] @ dλ=0.15, k=30, 1500 Adam iter, Turner+2024 priors).",
          "",
          "| Param | v1 production | 2lpt loa-0 wide | 2lpt loa-124 nohcd-nobal wide |",
          "|---|---:|---:|---:|"]
    for label, key, fn in [
        ("c_0", "log_c_0", np.exp),
        ("τ_0", "log_tau_0", np.exp),
        ("β", "log_beta", np.exp),
        ("log_c_0", "log_c_0", lambda x: x),
        ("log_τ_0", "log_tau_0", lambda x: x),
        ("log_β", "log_beta", lambda x: x),
    ]:
        row = [label]
        for name in MODELS:
            row.append(f"{fn(models[name][key]):.6f}")
        md.append("| " + " | ".join(row) + " |")

    md += ["",
           "## Rest grid",
           "",
           "| Model | n_pix | min_lambda | max_lambda | dlambda |",
           "|---|---:|---:|---:|---:|"]
    for name, m in models.items():
        rw = m["rest_wavelengths"]
        md.append(f"| {name} | {len(rw)} | {rw[0]:.2f} | {rw[-1]:.2f} | {rw[1]-rw[0]:.4f} |")

    md += ["",
           "## Observations",
           "",
           "- **c_0 differs by ~30-40×** between v1 and 2lpt models. v1 c_0 ≈ 0.17 (continuum scale around Lyα). 2lpt c_0 ≈ 0.004-0.006. Likely reflects different absolute flux normalization in the lyacolore mocks vs real DESI spectra.",
           "- **τ_0 and β** in 2lpt models are ~3-4× below Turner+2024 priors (0.00246, 3.62), while v1 also lands below prior but less aggressively. The 2lpt mocks may have been constructed with weaker effective optical depth than Turner+2024.",
           "- **Rest grid** is wider for the 2lpt models ([850.75, 1700] vs v1's [~911, ~1500]), reflecting the v2 wide_v2 preload format — captures more red-side continuum.",
           "",
           "**Caveat**: scalar differences alone do not say the 2lpt models are wrong. The trained model fit the *2lpt mock data*, which is its own statistical realization. The DLA-detection capability is the actual test, see the inference comparison.",
          ]
    (OUT / "endpoint_scalars.md").write_text("\n".join(md) + "\n")
    print(f"[saved] {OUT / 'endpoint_scalars.md'}")

    # --- 2) corr(M·M^T) per model ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    for i, (name, m) in enumerate(models.items()):
        rw = m["rest_wavelengths"]
        C = _corr(m["M"])
        extent = [rw[0], rw[-1], rw[-1], rw[0]]
        im = axes[i].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1,
                            extent=extent, aspect="auto")
        axes[i].set_title(f"({chr(ord('a')+i)}) corr(M·M$^T$) — {name}\n"
                          f"shape={C.shape}  range [{C.min():+.3f}, {C.max():+.3f}]")
        axes[i].set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
        if i == 0:
            axes[i].set_ylabel(r"$\lambda_\mathrm{rest}$ [Å]")
        plt.colorbar(im, ax=axes[i], fraction=0.046, label="correlation")
    fig.suptitle("Trained-GP correlation matrix corr(M·M$^T$): v1 production vs 2lpt models",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT / "corr_matrices.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    # --- 3) μ + log_omega side by side ---
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    # Top: μ
    for name, m in models.items():
        axes[0].plot(m["rest_wavelengths"], m["mu"], lw=1.5, alpha=0.8, label=name)
    axes[0].set_ylabel(r"$\mu(\lambda_\mathrm{rest})$ — mean continuum")
    axes[0].set_title("Mean continuum and per-pixel pixel noise log-variance")
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].grid(alpha=0.3)
    # Bottom: log_omega
    for name, m in models.items():
        axes[1].plot(m["rest_wavelengths"], m["log_omega"], lw=1.5, alpha=0.8, label=name)
    axes[1].set_ylabel(r"$\log\,\omega^2(\lambda_\mathrm{rest})$")
    axes[1].set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = OUT / "mu_log_omega_overlay.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    print()
    print("=== summary ===")
    for name, m in models.items():
        print(f"  {name}: c_0={np.exp(m['log_c_0']):.6f}  "
              f"τ_0={np.exp(m['log_tau_0']):.6f}  β={np.exp(m['log_beta']):.4f}  "
              f"M.shape={m['M'].shape}")


if __name__ == "__main__":
    main()
