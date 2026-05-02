"""Compare the 5 trained GP models we have on disk.

Extracts hyperparameters from each model's ``.h5`` (μ, M-norm, log τ_0,
log β, log c_0, log ω) and produces a 4-panel comparison figure plus a
hyperparameter table that the model-comparison story doc embeds.

Models (in order):
  PROD_y3_LOA                   — original Y3 production model on real LOA
  LOA_no_dla_no_bal             — NERSC retrain on real LOA, no DLAs/BALs
  LOA_no_hcd_with_bal           — NERSC retrain on real LOA, no HCDs but BALs
  MOCK_2lpt_loa0                — GreatLakes train on 2lpt mock loa-0 (forest only)
  MOCK_2lpt_loa124_nohcd_nobal  — GreatLakes train on 2lpt mock loa-124 (no HCD/BAL)

Run::
    python examples/compare_trained_gp_models.py \\
        --out-dir docs/story_figures
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


MODELS = [
    ("PROD_y3_LOA",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5",
     None,  # loss_history is baked into the .h5 for this one
     "real LOA Y3 (current production)"),
    ("LOA_no_dla_no_bal",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/GP_trained/loa_no_dla_no_bal_52198069/model_epoch_1499.h5",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/GP_trained/loa_no_dla_no_bal_52198069/loss_history.json",
     "real LOA, no DLAs no BALs (cleanest LOA)"),
    ("LOA_no_hcd_with_bal",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/GP_trained/loa_no_hcd_with_bal_52198070/model_epoch_1499.h5",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/GP_trained/loa_no_hcd_with_bal_52198070/loss_history.json",
     "real LOA, no HCDs but BALs kept"),
    ("MOCK_2lpt_loa0",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_48938765/model_epoch_0799.h5",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_48938765/loss_history.json",
     "2lpt mock loa-0 (forest-only by construction)"),
    ("MOCK_2lpt_loa124_nohcd_nobal",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa124_nohcd_nobal_48938766/model_epoch_0799.h5",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa124_nohcd_nobal_48938766/loss_history.json",
     "2lpt mock loa-124, no HCDs no BALs"),
]


def _extract(path: str, loss_path: str | None):
    """Extract relevant fields from a learned .h5 (and optional loss JSON)."""
    info = {}
    with h5py.File(path, "r") as h:
        info["mu"] = np.asarray(h["mu"])
        info["M"] = np.asarray(h["M"])
        info["log_omega"] = np.asarray(h["log_omega"])
        info["rest_wavelengths"] = np.asarray(h["rest_wavelengths"])
        info["log_tau_0"] = float(h["log_tau_0"][()])
        info["log_beta"] = float(h["log_beta"][()])
        info["log_c_0"] = float(h["log_c_0"][()])
        # tau_0/beta/c_0 in physical units
        info["tau_0"] = float(np.exp(info["log_tau_0"]))
        info["beta"] = float(np.exp(info["log_beta"]))
        info["c_0"] = float(np.exp(info["log_c_0"]))
        info["omega_mean"] = float(np.exp(info["log_omega"]).mean())
        # loss history: either inside h5 (legacy) or in sibling json (v2)
        if "loss_history" in h:
            info["loss_history"] = np.asarray(h["loss_history"])
        elif loss_path and os.path.exists(loss_path):
            info["loss_history"] = np.array(json.load(open(loss_path)))
        else:
            info["loss_history"] = None
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out-dir", default="docs/story_figures")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting hyperparameters from each model:")
    extracted = {}
    for name, path, loss_path, _ in MODELS:
        if not os.path.exists(path):
            print(f"  ! {name}: not found at {path}")
            continue
        extracted[name] = _extract(path, loss_path)
        info = extracted[name]
        print(f"  {name:<32}  τ₀={info['tau_0']:.5f}  β={info['beta']:.2f}  "
              f"c₀={info['c_0']:.4f}  ω̄={info['omega_mean']:.3f}  "
              f"L_end={info['loss_history'][-1] if info['loss_history'] is not None else 'N/A'}")

    # ============================================================
    # Figure 1 — split into v1 (production) vs v2 (new) panels
    # because the trainers use different normalization conventions:
    #   v1: per-spectrum-median-normalized (μ fluctuates around 1)
    #   v2: inverse-variance-weighted-population-mean centered (μ
    #       is the absolute mean flux per pixel, not learned but
    #       computed from data; centered data has mean ≈ 0)
    # ============================================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 11),
                             gridspec_kw=dict(height_ratios=[2, 2, 1.5]))

    color_map = {
        "PROD_y3_LOA": "C0",
        "LOA_no_dla_no_bal": "C2",
        "LOA_no_hcd_with_bal": "C1",
        "MOCK_2lpt_loa0": "C3",
        "MOCK_2lpt_loa124_nohcd_nobal": "C4",
    }
    v1_models = ["PROD_y3_LOA"]
    v2_models = [n for n in extracted if n not in v1_models]

    def _plot_panel(ax, key, models, ylabel, title, log=False):
        for n in models:
            if n not in extracted: continue
            info = extracted[n]
            rest = info["rest_wavelengths"]
            y = info[key] if key != "omega" else np.exp(info["log_omega"])
            ax.plot(rest, y, color=color_map.get(n, "0.4"), lw=0.9, label=n)
        ax.set_xlabel("rest wavelength [Å]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3, which="both" if log else "major")
        if log:
            ax.set_yscale("log")
        ax.axvline(1215.67, color="0.7", lw=0.5, ls="--")

    _plot_panel(axes[0, 0], "mu", v1_models,
                "μ (per-spectrum median-normalized)",
                "(A1) v1 trainer — μ centered around 1.0 (QSO emission ≥ 1)")
    _plot_panel(axes[0, 1], "mu", v2_models,
                "μ (population inverse-variance-weighted mean)",
                "(A2) v2 trainer — μ in absolute flux units")
    _plot_panel(axes[1, 0], "omega", v1_models,
                "ω (normalized-flux units)",
                "(B1) v1 ω", log=True)
    _plot_panel(axes[1, 1], "omega", v2_models,
                "ω (absolute-flux units)",
                "(B2) v2 ω", log=True)

    # Loss panels — also split (v1 absolute log-likelihood vs v2 normalized)
    for ax, models, title in [
        (axes[2, 0], v1_models, "(C1) v1 loss (absolute log-likelihood scale)"),
        (axes[2, 1], v2_models, "(C2) v2 loss (per-pixel normalized scale)"),
    ]:
        for n in models:
            if n not in extracted: continue
            info = extracted[n]
            if info["loss_history"] is None: continue
            ax.plot(np.arange(len(info["loss_history"])),
                    info["loss_history"],
                    color=color_map.get(n, "0.4"), lw=0.8, label=n)
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("Trained GP models — v1 (production) vs v2 (new) trainer\n"
                 "Note: v1 / v2 normalization conventions differ; values are "
                 "NOT directly comparable across trainers.",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = out_dir / "trained_gp_models_compare.png"
    fig.savefig(out1, dpi=130)
    print(f"\n[saved] {out1}")

    # ============================================================
    # Figure 2 — hyperparameter bar chart (τ_0, β, c_0, mean ω)
    # ============================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 6))
    names = list(extracted.keys())
    xs = np.arange(len(names))
    colors = [color_map.get(n, "0.4") for n in names]

    def _bar(ax, key, title, ref=None, ylabel=None):
        vals = [extracted[n][key] for n in names]
        bars = ax.bar(xs, vals, color=colors, alpha=0.85, edgecolor="0.2")
        ax.set_xticks(xs)
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
        ax.set_ylabel(ylabel or key)
        ax.set_title(title, fontsize=10)
        if ref is not None:
            ax.axhline(ref, color="C3", lw=1.0, ls="--",
                       label=f"Turner+2024 = {ref}")
            ax.legend(fontsize=8)
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.4f}" if v < 1 else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax.grid(alpha=0.3, axis="y")

    _bar(axes2[0, 0], "tau_0", "(A) Learned τ₀ (mean-flux opacity)",
         ref=0.00246)
    _bar(axes2[0, 1], "beta", "(B) Learned β (forest opacity power-law)",
         ref=3.62)
    _bar(axes2[1, 0], "c_0", "(C) Learned c₀ (small-scale noise)")
    _bar(axes2[1, 1], "omega_mean", "(D) Mean ω (per-pixel forest noise)")

    fig2.tight_layout()
    out2 = out_dir / "trained_gp_models_hyperparameters.png"
    fig2.savefig(out2, dpi=130)
    print(f"[saved] {out2}")

    # ============================================================
    # Hyperparameter summary table (markdown for embedding)
    # ============================================================
    md = ["| model | τ₀ | β | c₀ | ω̄ | loss[-1] | epochs |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name, info in extracted.items():
        loss = info["loss_history"]
        loss_s = f"{loss[-1]:.1f}" if loss is not None else "N/A"
        epoch_s = f"{len(loss)}" if loss is not None else "N/A"
        md.append(f"| {name} | {info['tau_0']:.5f} | {info['beta']:.2f} | "
                  f"{info['c_0']:.4f} | {info['omega_mean']:.3f} | "
                  f"{loss_s} | {epoch_s} |")
    md_path = out_dir / "trained_gp_models_table.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
