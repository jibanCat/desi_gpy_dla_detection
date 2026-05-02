"""Trained-GP calibration check.

For a trained model + its trainset, compute the *expected* vs *observed*
distribution of:

  1. Mahalanobis χ² per spectrum: (y-μ)^T K^-1 (y-μ).
     Expected χ² ~ Chi-square(n_valid).
  2. Per-pixel standardized residual: r_i = (y_i - μ_i) / sqrt(K_ii).
     Expected r ~ N(0, 1) (under joint Gaussian; this is the *prior*
     normalization, not the conditional, but is a meaningful sanity
     test of whether the per-pixel σ is right).
  3. Log evidence per spectrum.

Histograms get overlaid with the theoretical distributions so you
can eyeball under/over training:

  - If χ² is too big (right-shifted) → residuals don't fit → undertrained
    or wrong noise model.
  - If χ² is too small (left-shifted) → model fits "too well" → overfit.
  - If the per-pixel r distribution is wider than N(0,1) → ω² is too
    small (model overconfident).
  - If narrower than N(0,1) → ω² is too big (model under-confident).

K^-1 (y-μ) is computed via the Woodbury identity using the trained
M and ω² (no full matrix inversion). Same math as
``gpy_dla_detection/objective.py``.

Usage::

    python examples/check_v2_model_calibration.py \\
        --model /path/to/model.h5 \\
        --trainset /path/to/trainset.h5 \\
        --n-spectra 500 \\
        --out figs/calibration_<model>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def _load_model(path):
    with h5py.File(path, "r") as f:
        rw = f["rest_wavelengths"][...]; rw = rw[:,0] if rw.ndim==2 else rw
        mu = f["mu"][...]; mu = mu[:,0] if mu.ndim==2 else mu
        log_omega = f["log_omega"][...]; log_omega = log_omega[:,0] if log_omega.ndim==2 else log_omega
        M = f["M"][...]
        if M.shape[0] != rw.shape[0]:
            M = M.T
    return rw, mu, log_omega, M


def _woodbury_solve(M, omega2, y_centered, valid_mask):
    """Compute z = K^-1 (y - μ) and chi² = (y-μ)^T K^-1 (y-μ) using
    K = diag(ω²) + M M^T. Pixels where valid_mask=False contribute 0."""
    n = y_centered.shape[0]
    # d_inv with masked pixels zeroed
    d_inv = np.where(valid_mask, 1.0 / np.where(omega2 > 0, omega2, 1.0), 0.0)
    # K_inv y = d_inv y - d_inv M (I + M^T d_inv M)^-1 M^T d_inv y
    Mty = M.T @ (d_inv * y_centered)              # (k,)
    DiM = d_inv[:, None] * M                       # (n, k)
    B = M.T @ DiM + np.eye(M.shape[1])             # (k, k)
    z_solve = np.linalg.solve(B, Mty)              # (k,)
    K_inv_y = d_inv * y_centered - DiM @ z_solve   # (n,)
    chi2 = float(y_centered @ K_inv_y)
    n_valid = int(valid_mask.sum())
    # log|K| = sum_valid log(ω²) + log|B|
    sign, logabsdet_B = np.linalg.slogdet(B)
    log_det_K = float(np.log(omega2[valid_mask]).sum() + sign * logabsdet_B)
    log_evidence = -0.5 * (chi2 + log_det_K + n_valid * np.log(2 * np.pi))
    return chi2, log_evidence, n_valid


def _per_pixel_predicted_var(M, omega2):
    """Diagonal of K = ω² + sum_k M[:, k]**2."""
    return omega2 + (M ** 2).sum(axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--trainset", required=True)
    p.add_argument("--n-spectra", type=int, default=500)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"[main] loading model: {args.model}")
    rw_mod, mu, log_omega, M = _load_model(args.model)
    omega2 = np.exp(2 * log_omega)
    n_pix_model, k = M.shape
    print(f"  n_pix={n_pix_model}  k={k}  trace_MMT={float((M**2).sum()):.2e}  trace_omega²={float(omega2.sum()):.2e}")

    print(f"[main] loading trainset: {args.trainset}")
    with h5py.File(args.trainset, "r") as f:
        keys = list(f.keys())
        print(f"  keys: {keys}")
        # The trainset.h5 has fluxes already centered + deforested, so y_centered = flux.
        n_total = f["fluxes"].shape[0]
        n_pix_data = f["fluxes"].shape[1]
        if n_pix_data != n_pix_model:
            raise SystemExit(f"trainset n_pix ({n_pix_data}) != model n_pix ({n_pix_model})")
        rng = np.random.default_rng(args.seed)
        n = min(args.n_spectra, n_total)
        idx = rng.choice(n_total, size=n, replace=False)
        idx.sort()
        # Read in chunks
        fluxes = f["fluxes"][idx]
        nv = f["noise_variance"][idx]

    print(f"[main] selected {n} of {n_total} spectra")

    # IMPORTANT: trainset fluxes are *centered* (μ already subtracted). So
    # y_centered = fluxes; we pass fluxes directly into Woodbury.
    # Verify by checking median: should be near 0 if centered.
    sample_med = float(np.nanmedian(fluxes))
    print(f"  trainset flux median = {sample_med:.3f}  (~0 if centered, ~1 if not)")
    centered = abs(sample_med) < 0.1
    print(f"  → assuming {'centered' if centered else 'not centered'}; "
          f"{'using as-is' if centered else 'subtracting μ'}")
    if not centered:
        fluxes = fluxes - mu[None, :]

    # The trainset noise_variance is the data noise; the model also has its
    # own ω² from training. For calibration, K = M·M^T + diag(ω² + nv) for
    # this spectrum (data noise + absorption noise). Combine pixel-wise.
    chi2_list = []
    log_ev_list = []
    n_valid_list = []
    standardized_resids = []
    for i in range(n):
        y = fluxes[i].astype(np.float64)
        nv_i = nv[i].astype(np.float64)
        valid = np.isfinite(y) & np.isfinite(nv_i) & (nv_i > 0)
        if valid.sum() < 100:
            continue
        # K_diag = ω²_model + nv_data (per-pixel total noise budget).
        d = omega2 + np.where(valid, nv_i, 0.0)
        # Sanitize for masked pixels
        y_safe = np.where(valid, y, 0.0)
        chi2, log_ev, n_valid = _woodbury_solve(M, d, y_safe, valid)
        chi2_list.append(chi2)
        log_ev_list.append(log_ev)
        n_valid_list.append(n_valid)
        # Per-pixel standardized residual using DIAGONAL approx (not the
        # full K^-1; this is the prior σ at each pixel)
        K_diag = d + (M ** 2).sum(axis=1)
        sigma = np.sqrt(np.maximum(K_diag, 1e-30))
        r = (y_safe / sigma)[valid]   # masked pixels excluded
        # Subsample to keep memory bounded
        if r.size > 200:
            r = r[::max(1, r.size // 200)]
        standardized_resids.append(r)
    chi2_arr = np.array(chi2_list)
    log_ev_arr = np.array(log_ev_list)
    n_valid_arr = np.array(n_valid_list)
    std_resid = np.concatenate(standardized_resids)

    print(f"[main] computed chi² for {len(chi2_arr)} spectra")

    # Standardized chi²: (chi² - n_valid) / sqrt(2 * n_valid). Should be N(0, 1).
    z_chi2 = (chi2_arr - n_valid_arr) / np.sqrt(2 * n_valid_arr)
    print(f"  chi² z-score: mean={z_chi2.mean():.3f}  std={z_chi2.std():.3f}  "
          f"(target: mean~0, std~1)")
    print(f"  per-pixel resid r: mean={std_resid.mean():.3f}  std={std_resid.std():.3f}  "
          f"(target: mean~0, std~1)")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    model_short = Path(args.model).parent.name + "/" + Path(args.model).stem
    fig.suptitle(f"Calibration check: {model_short}", fontsize=11)

    # Top-left: chi²/n_valid vs 1
    ax = axes[0, 0]
    chi2_per_n = chi2_arr / n_valid_arr
    ax.hist(chi2_per_n, bins=40, density=True, color="C0", edgecolor="white", alpha=0.85)
    ax.axvline(1.0, color="C3", ls="--", lw=1, label="expected = 1")
    ax.axvline(chi2_per_n.mean(), color="C2", ls="-", lw=1,
              label=f"observed mean = {chi2_per_n.mean():.3f}")
    ax.set_xlabel("χ² / n_valid")
    ax.set_ylabel("density")
    ax.set_title("Per-spectrum χ² (Mahalanobis) per dof")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Top-right: standardized chi² z-score
    ax = axes[0, 1]
    ax.hist(z_chi2, bins=40, density=True, color="C1", edgecolor="white", alpha=0.85)
    xs = np.linspace(z_chi2.min() - 1, z_chi2.max() + 1, 200)
    ax.plot(xs, stats.norm.pdf(xs), color="C3", lw=1.2, label="N(0, 1) target")
    ax.set_xlabel("(χ² - n_valid) / sqrt(2 n_valid)")
    ax.set_title(f"χ² z-score: mean={z_chi2.mean():.3f} std={z_chi2.std():.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-left: per-pixel standardized residual
    ax = axes[1, 0]
    ax.hist(std_resid, bins=80, density=True, range=(-6, 6), color="C2",
            edgecolor="white", alpha=0.85)
    xs = np.linspace(-6, 6, 200)
    ax.plot(xs, stats.norm.pdf(xs), color="C3", lw=1.2, label="N(0, 1)")
    ax.set_xlabel("(y - μ) / σ_pred  per pixel")
    ax.set_title(f"Per-pixel standardized residual: mean={std_resid.mean():.3f} std={std_resid.std():.3f}")
    ax.set_xlim(-6, 6)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-right: log evidence per spectrum
    ax = axes[1, 1]
    ax.hist(log_ev_arr, bins=40, density=True, color="C4", edgecolor="white", alpha=0.85)
    ax.set_xlabel("log evidence per spectrum")
    ax.set_title(f"log evidence: mean={log_ev_arr.mean():.1f} std={log_ev_arr.std():.1f}")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[main] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")

    # Print verdict
    print()
    print(f"=== CALIBRATION VERDICT ===")
    if abs(chi2_per_n.mean() - 1.0) < 0.1 and 0.9 < std_resid.std() < 1.1:
        print("  PASS: well-calibrated (χ²/n ≈ 1, std_resid ≈ 1)")
    elif chi2_per_n.mean() > 1.5 or std_resid.std() > 1.5:
        print("  FAIL: UNDER-FIT — residuals larger than predicted σ")
        print(f"       χ²/n_valid mean = {chi2_per_n.mean():.2f} (>>1)")
        print(f"       std_resid       = {std_resid.std():.2f} (>>1)")
    elif chi2_per_n.mean() < 0.5 or std_resid.std() < 0.5:
        print("  FAIL: OVER-FIT — residuals smaller than predicted σ")
        print(f"       χ²/n_valid mean = {chi2_per_n.mean():.2f} (<<1)")
        print(f"       std_resid       = {std_resid.std():.2f} (<<1)")
    else:
        print("  MARGINAL: somewhat off-calibration but not pathological")
        print(f"       χ²/n_valid mean = {chi2_per_n.mean():.2f}")
        print(f"       std_resid       = {std_resid.std():.2f}")


if __name__ == "__main__":
    main()
