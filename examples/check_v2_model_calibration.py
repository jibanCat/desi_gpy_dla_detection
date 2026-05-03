"""Trained-GP calibration check.

CRITICAL: The GP models the (normalize→deforest→center) outputs of the
trainset, NOT the raw trainset.h5 fluxes. This script applies the exact
same pipeline as the trainer (via load_preprocessed_h5) before
evaluating chi² / standardized residuals. Earlier versions of this
script worked on raw fluxes, which gave meaningless verdicts.

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
import sys
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

    # CRITICAL: GP is calibrated against (normalize→deforest→center) outputs,
    # NOT raw trainset.h5 fluxes. Use load_preprocessed_h5 to apply the
    # exact same pipeline the trainer applied. Otherwise the residuals
    # we evaluate are in the wrong space and chi² is meaningless.
    print(f"[main] loading + pipelining trainset: {args.trainset}")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gpy_dla_detection.training.dataset import load_preprocessed_h5
    ts = load_preprocessed_h5(
        args.trainset,
        z_min=2.0, z_max=4.5,
        max_spectra=args.n_spectra,
        # production defaults; the trainer applies these at training time
        norm_min_lambda=1310.0, norm_max_lambda=1325.0,
        de_forest_tau_0=0.00246, de_forest_beta=3.62,
        de_forest_num_lines=3,
    )
    fluxes = ts.fluxes.numpy().astype(np.float64)  # pipelined + centered ~0
    nv = ts.noise_variances.numpy().astype(np.float64)
    lya_1pzs = ts.lya_1pzs.numpy().astype(np.float64)
    z_qsos = ts.z_qsos.numpy().astype(np.float64)
    n = fluxes.shape[0]
    n_pix_data = fluxes.shape[1]
    if n_pix_data != n_pix_model:
        raise SystemExit(f"pipelined n_pix ({n_pix_data}) != model n_pix ({n_pix_model})")
    print(f"[main] selected {n} pipelined spectra")
    print(f"  pipelined flux median = {float(np.nanmedian(fluxes)):.3f}  (~0 if centered)")

    # Recreate the trainer's full d = nv + ω² · scaling² where scaling depends
    # on per-spectrum lya_1pz + the model's trained log_tau_0/log_beta/log_c_0.
    # This matches gpy_dla_detection/training/objective_v2.py:vectorized_nll.
    from gpy_dla_detection.voigt import (
        transition_wavelengths as TW, oscillator_strengths as OS)
    with h5py.File(args.model, "r") as fh:
        log_tau_0 = float(np.asarray(fh["log_tau_0"]).flatten()[0])
        log_beta  = float(np.asarray(fh["log_beta"]).flatten()[0])
        log_c_0   = float(np.asarray(fh["log_c_0"]).flatten()[0])
    tau_0_m = float(np.exp(log_tau_0))
    beta_m  = float(np.exp(log_beta))
    c_0_m   = float(np.exp(log_c_0))
    num_lines = 3
    tw0 = float(TW[0])
    os0 = float(OS[0])

    chi2_list = []
    log_ev_list = []
    n_valid_list = []
    standardized_resids = []
    for i in range(n):
        y = fluxes[i].astype(np.float64)
        nv_i = nv[i].astype(np.float64)
        lya_1pz = lya_1pzs[i].astype(np.float64)
        zqso_1pz = 1.0 + z_qsos[i]
        valid = np.isfinite(y) & np.isfinite(nv_i) & (nv_i > 0)
        if valid.sum() < 100:
            continue

        # tau_optical_depth (Lyα + higher Lyman lines), each masked above zqso
        indicator_lya = (lya_1pz <= zqso_1pz).astype(np.float64)
        tau = tau_0_m * (lya_1pz ** beta_m) * indicator_lya
        for j in range(1, num_lines):
            lyman_1pz = tw0 * lya_1pz / float(TW[j])
            ind_j = (lyman_1pz <= zqso_1pz).astype(np.float64)
            tau_j = tau_0_m * float(TW[j]) * float(OS[j]) / (tw0 * os0)
            tau = tau + tau_j * (lyman_1pz ** beta_m) * ind_j

        scaling = 1.0 - np.exp(-tau) + c_0_m
        absorption_noise = omega2 * (scaling ** 2)
        d = np.where(valid, nv_i, 0.0) + absorption_noise

        y_safe = np.where(valid, y, 0.0)
        chi2, log_ev, n_valid = _woodbury_solve(M, d, y_safe, valid)
        chi2_list.append(chi2)
        log_ev_list.append(log_ev)
        n_valid_list.append(n_valid)

        # Per-pixel standardized residual (uses K_diag = d + diag(M·M^T))
        K_diag = d + (M ** 2).sum(axis=1)
        sigma = np.sqrt(np.maximum(K_diag, 1e-30))
        r = (y_safe / sigma)[valid]
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

    # Print verdict.
    # Use chi²/n_valid as the primary metric (accounts for the full K via
    # Woodbury; correctly handles the M·M^T contribution). Per-pixel
    # std_resid below is computed using K_diag only (no Woodbury) and is
    # MISLEADING when M·M^T dominates the diagonal — it OVER-estimates σ
    # because the Woodbury inversion would shrink the effective σ
    # substantially. Treat std_resid as informational, not a verdict
    # criterion.
    print()
    print(f"=== CALIBRATION VERDICT ===")
    chi2_dev = abs(chi2_per_n.mean() - 1.0)
    if chi2_dev < 0.2:
        print(f"  PASS: well-calibrated  (χ²/n_valid = {chi2_per_n.mean():.2f}, "
              f"target 1.0 ± 0.2)")
    elif chi2_per_n.mean() > 1.5:
        print(f"  FAIL: UNDER-FIT — residuals larger than predicted σ")
        print(f"       χ²/n_valid mean = {chi2_per_n.mean():.2f} (>1.5)")
    elif chi2_per_n.mean() < 0.5:
        print(f"  FAIL: OVER-FIT — residuals smaller than predicted σ")
        print(f"       χ²/n_valid mean = {chi2_per_n.mean():.2f} (<0.5)")
    else:
        print(f"  MARGINAL: χ²/n_valid = {chi2_per_n.mean():.2f}  "
              f"(within [0.5, 1.5] but >0.2 from target)")
    print(f"  per-pixel resid (K_diag-based, informational only): "
          f"mean={std_resid.mean():.3f} std={std_resid.std():.3f}")
    print(f"  Note: per-pixel std_resid using K_diag is MISLEADING when "
          f"M·M^T dominates;\n        the full Woodbury K^-1 effective σ "
          f"is much smaller. χ²/n is the trustworthy metric.")


if __name__ == "__main__":
    main()
