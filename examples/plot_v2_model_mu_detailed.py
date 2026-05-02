"""Detailed per-model μ comparison: each model on its own y-scale plus
a difference panel against the v1 baseline. Surfaces the SHAPE of the
emission-line structure model-by-model.

The shared-y-axis version is in plot_v2_model_diagnostics.py; this one
is for diagnosing why LOA-trained μ looks 'flatter' visually even
though the Lyα peak sharpness is 4+σ.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


LINES = {"Lyα": 1215.67, "Lyβ": 1025.72, "OVI": 1031.93,
         "Lyγ": 972.54, "NV": 1240.0, "CIII*": 1175.7}


def _load_mu(path):
    with h5py.File(path, "r") as f:
        rw = f["rest_wavelengths"][...]; rw = rw[:,0] if rw.ndim==2 else rw
        mu = f["mu"][...]; mu = mu[:,0] if mu.ndim==2 else mu
        log_omega = f["log_omega"][...]; log_omega = log_omega[:,0] if log_omega.ndim==2 else log_omega
    return rw, mu, log_omega


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--baseline-idx", type=int, default=0,
                   help="Index of the model to use as 'baseline' for difference panels (default 0=first)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    paths = [m.strip() for m in args.models.split(",")]
    labels = [l.strip() for l in args.labels.split(",")]
    assert len(paths) == len(labels)

    models = [_load_mu(p_) for p_ in paths]
    rw_base, mu_base, _ = models[args.baseline_idx]

    # Figure: 2 columns × N rows. Left = per-model μ on own y-scale. Right = (μ - μ_baseline) on common scale.
    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(13, 1.7 * n), sharex=True,
                             gridspec_kw=dict(width_ratios=[1.4, 1.0], hspace=0.18, wspace=0.15))

    # Common diff y-range
    diffs = []
    for i, (rw, mu, _) in enumerate(models):
        if i == args.baseline_idx:
            continue
        # interpolate mu_base onto rw if needed
        if rw.shape == rw_base.shape and np.allclose(rw, rw_base):
            d = mu - mu_base
        else:
            d = mu - np.interp(rw, rw_base, mu_base)
        diffs.append(d)
    if diffs:
        d_max = max(np.abs(d).max() for d in diffs)
    else:
        d_max = 0.5

    for i, ((rw, mu, log_omega), label) in enumerate(zip(models, labels)):
        ax_mu = axes[i, 0]
        ax_mu.plot(rw, mu, color="C0", lw=0.8)
        # Mark emission lines
        for lname, lwave in LINES.items():
            if rw[0] < lwave < rw[-1]:
                ax_mu.axvline(lwave, color="0.7", ls=":", lw=0.5, alpha=0.6)
                if i == 0:
                    ax_mu.text(lwave, ax_mu.get_ylim()[1] * 0.92, lname,
                               fontsize=6, ha="center", color="0.5")
        ax_mu.set_title(label, fontsize=8, loc="left")
        ax_mu.set_ylabel("μ", fontsize=9)
        ax_mu.grid(alpha=0.25)
        # Annotate Lyα peak height + continuum ratio
        in_lya = (rw > 1213) & (rw < 1218)
        in_cont = (rw > 1100) & (rw < 1180)
        if in_lya.sum() > 0 and in_cont.sum() > 0:
            lya_peak = mu[in_lya].max()
            cont_med = np.median(mu[in_cont])
            ax_mu.text(0.99, 0.95, f"peak={lya_peak:.2f}, cont={cont_med:.2f}, ratio={lya_peak/cont_med:.2f}×",
                       transform=ax_mu.transAxes, fontsize=7, ha="right", va="top",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.85))

        ax_diff = axes[i, 1]
        if i == args.baseline_idx:
            ax_diff.text(0.5, 0.5, f"(baseline:\n{label})", transform=ax_diff.transAxes,
                        fontsize=9, ha="center", va="center", color="0.5")
            ax_diff.set_yticks([])
        else:
            if rw.shape == rw_base.shape and np.allclose(rw, rw_base):
                d = mu - mu_base
            else:
                d = mu - np.interp(rw, rw_base, mu_base)
            ax_diff.fill_between(rw, 0, d, where=(d > 0), color="C2", alpha=0.4, lw=0)
            ax_diff.fill_between(rw, 0, d, where=(d < 0), color="C3", alpha=0.4, lw=0)
            ax_diff.plot(rw, d, color="0.2", lw=0.5)
            ax_diff.axhline(0, color="0.6", lw=0.4, ls="--")
            ax_diff.set_ylim(-d_max * 1.1, d_max * 1.1)
            ax_diff.set_ylabel(f"μ − μ[{labels[args.baseline_idx]}]", fontsize=7)
            for lname, lwave in LINES.items():
                if rw[0] < lwave < rw[-1]:
                    ax_diff.axvline(lwave, color="0.7", ls=":", lw=0.4, alpha=0.4)
        ax_diff.grid(alpha=0.25)

    axes[-1, 0].set_xlabel("rest wavelength [Å]")
    axes[-1, 1].set_xlabel("rest wavelength [Å]")
    fig.suptitle("Per-model μ on individual y-scale + difference vs baseline",
                 fontsize=11, y=0.998)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[main] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
