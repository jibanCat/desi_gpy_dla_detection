"""Per-model GP diagnostic plot: μ, log_omega (and ω = exp(log_omega)),
and the correlation matrix derived from K = M·M^T + diag(ω²).

For each model passed via ``--models``, produces a 4-row figure:
  row 1: μ(λ_rest)             — emission-line continuum
  row 2: log_omega(λ_rest)     — per-pixel absorption noise scale
  row 3: ω(λ_rest)             — same in linear scale (for direct comparison)
  row 4: |corr(K)|             — correlation matrix abs value, log-color

If multiple models are passed they are arranged side-by-side so the same
row uses the same y-axis range (forced via shared ax) for direct
comparison. The correlation panel uses a single global colorbar.

Usage::

    python examples/plot_v2_model_diagnostics.py \\
        --models /path/v1.h5,/path/v2_a.h5,/path/v2_b.h5 \\
        --labels "v1,v2-LOA,v2-saclay" \\
        --out figs/v2_model_diagnostics.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def _load_model(path: str) -> dict:
    """Pull the GP fields we need from a .h5 (handles both v1 and v2 schemas)."""
    with h5py.File(path, "r") as f:
        # rest_wavelengths can be either 1-D (v2) or (n_pix, 1) (v1).
        rw = f["rest_wavelengths"][...]
        rw = rw[:, 0] if rw.ndim == 2 else rw
        mu = f["mu"][...]
        mu = mu[:, 0] if mu.ndim == 2 else mu
        log_omega = f["log_omega"][...]
        log_omega = log_omega[:, 0] if log_omega.ndim == 2 else log_omega
        # M can be (n_pix, k) (v2) or (k, n_pix) (v1) — distinguish by shape.
        M = f["M"][...]
        if M.shape[0] != rw.shape[0]:
            M = M.T
        # Scalars
        log_c_0 = float(np.asarray(f["log_c_0"]).flatten()[0])
        log_tau_0 = float(np.asarray(f["log_tau_0"]).flatten()[0])
        log_beta = float(np.asarray(f["log_beta"]).flatten()[0])
        # v2 normalization fields (NaN means trained without normalization)
        norm = None
        if "normalization_min_lambda" in f:
            n_min = float(f["normalization_min_lambda"][()])
            n_max = float(f["normalization_max_lambda"][()])
            norm = (n_min, n_max)
    return dict(
        rest_wavelengths=rw, mu=mu, log_omega=log_omega, M=M,
        log_c_0=log_c_0, log_tau_0=log_tau_0, log_beta=log_beta,
        norm=norm,
    )


def _compute_correlation(M: np.ndarray, log_omega: np.ndarray,
                         downsample: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """K = M·M^T + diag(ω²); return (K_corr, sub_indices) downsampled by `downsample`.
    Downsampling keeps memory + render time reasonable for ~3800-px grids.
    Correlation matrix = K / sqrt(diag(K) ⊗ diag(K))."""
    M_ds = M[::downsample]
    omega2_ds = np.exp(2 * log_omega[::downsample])
    K = M_ds @ M_ds.T + np.diag(omega2_ds)
    diag = np.diag(K)
    sigma = np.sqrt(np.maximum(diag, 1e-30))
    Corr = K / np.outer(sigma, sigma)
    return Corr, np.arange(0, M.shape[0], downsample)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True,
                   help="Comma-separated absolute paths to model .h5 files")
    p.add_argument("--labels", default=None,
                   help="Comma-separated labels for the models (default = filename stems)")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--corr-downsample", type=int, default=4,
                   help="Downsample the correlation matrix by this factor "
                        "(default 4 → ~950×950 from 3801 px)")
    args = p.parse_args()

    paths = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.labels:
        labels = [l.strip() for l in args.labels.split(",")]
    else:
        labels = [Path(m).parent.name + "/" + Path(m).stem for m in paths]
    if len(labels) != len(paths):
        raise SystemExit("--labels count must match --models count")

    print(f"[main] loading {len(paths)} models")
    models = []
    for p_ in paths:
        m = _load_model(p_)
        print(f"  {Path(p_).parent.name}/{Path(p_).name}: "
              f"n_pix={m['rest_wavelengths'].shape[0]} k={m['M'].shape[1]} "
              f"log_tau_0={m['log_tau_0']:.4f} log_beta={m['log_beta']:.4f} "
              f"norm={m['norm']}")
        models.append(m)

    n_models = len(models)
    fig, axes = plt.subplots(
        4, n_models,
        figsize=(4.5 * n_models, 13),
        squeeze=False,
        gridspec_kw=dict(height_ratios=[1, 1, 1, 1.6]),
    )

    # Shared y-ranges per row, computed across all models for fair compare.
    mu_min = min(m["mu"].min() for m in models)
    mu_max = max(m["mu"].max() for m in models)
    lo_min = min(m["log_omega"].min() for m in models)
    lo_max = max(m["log_omega"].max() for m in models)
    om_min = 0.0
    om_max = max(np.exp(m["log_omega"]).max() for m in models)

    # Compute correlations once + a global vmin/vmax (use abs corr since we
    # want to highlight anti-correlation too).
    corrs = [_compute_correlation(m["M"], m["log_omega"], args.corr_downsample) for m in models]

    for i, (m, label, (Corr, sub_idx)) in enumerate(zip(models, labels, corrs)):
        rw = m["rest_wavelengths"]

        ax_mu = axes[0, i]
        ax_mu.plot(rw, m["mu"], lw=0.8, color="C0")
        ax_mu.set_ylim(mu_min - 0.1, mu_max + 0.1)
        ax_mu.set_title(label, fontsize=10)
        if i == 0:
            ax_mu.set_ylabel("μ (continuum)")
        ax_mu.axvline(1215.67, color="0.7", lw=0.5, ls="--", alpha=0.6)
        ax_mu.axvline(1025.72, color="0.7", lw=0.5, ls="--", alpha=0.6)
        if m["norm"] is not None and not np.isnan(m["norm"][0]):
            ax_mu.axvspan(m["norm"][0], m["norm"][1], color="C1", alpha=0.15,
                          label=f"norm [{m['norm'][0]:.0f}, {m['norm'][1]:.0f}]")
            ax_mu.legend(fontsize=7, loc="upper right")

        ax_lo = axes[1, i]
        ax_lo.plot(rw, m["log_omega"], lw=0.6, color="C2")
        ax_lo.set_ylim(lo_min - 0.2, lo_max + 0.2)
        if i == 0:
            ax_lo.set_ylabel("log ω (absorption noise)")

        ax_om = axes[2, i]
        ax_om.plot(rw, np.exp(m["log_omega"]), lw=0.6, color="C3")
        ax_om.set_ylim(om_min, om_max * 1.05)
        if i == 0:
            ax_om.set_ylabel("ω = exp(log_ω)")
        ax_om.set_xlabel("rest wavelength [Å]")

        ax_corr = axes[3, i]
        # Plot |corr| in log scale (highlights both strong + weak structure).
        # vmin tied across panels via globally normalized log range.
        im = ax_corr.imshow(
            np.abs(Corr),
            origin="lower",
            extent=(rw[sub_idx[0]], rw[sub_idx[-1]],
                    rw[sub_idx[0]], rw[sub_idx[-1]]),
            aspect="auto",
            cmap="viridis",
            norm=mpl.colors.LogNorm(vmin=1e-3, vmax=1.0),
        )
        if i == 0:
            ax_corr.set_ylabel("|corr K| (rest λ Å)")
        ax_corr.set_xlabel("rest λ [Å]")
        if i == n_models - 1:
            cbar = fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
            cbar.set_label("|correlation|", fontsize=8)

    fig.suptitle("Per-model GP diagnostics: μ, ω, K-correlation",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[main] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
