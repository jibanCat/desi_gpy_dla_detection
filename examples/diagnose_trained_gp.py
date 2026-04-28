#!/usr/bin/env python
"""Diagnose trained GP models — sensibility, OOD, BAL-vs-non-BAL selection,
and HCD-impact analysis.

Three sub-commands:

  1) ``visualize``  — model-vs-model overlay of μ(λ), ω(λ), eigenspectra,
                      and a hyperparameter summary table. No test data
                      needed; useful for quickly comparing a freshly
                      trained model against the legacy production GP.

  2) ``score``      — for a sample of test spectra, compute the NullGP
                      log marginal likelihood (``log p(y | M_no-DLA)``)
                      under each provided model. Reports per-pixel NLL
                      distributions, model-ranking statistics, and
                      a CSV with per-spectrum scores.

  3) ``classify-bal``  — given two models tagged "non-BAL" and "BAL", and
                      a sample with known ``BI_CIV`` flags, predict which
                      model fits each spectrum better (Bayes-factor) and
                      report accuracy + confusion matrix. Tests whether
                      the trained ω(λ) is sensitive enough to BAL features
                      to be useful as a model-selection signal.

The HCD-impact diagnosis falls out naturally from running ``score`` on
HCD-bearing vs HCD-free held-out spectra: a model trained on a clean
sample should give *worse* NLL on HCD-bearing spectra than on clean
ones, while a model trained on the full sample should be more uniform.

Companion to ``notebooks/Visualize Model.ipynb`` — the visualize
sub-command produces matplotlib figures matching the notebook style
but in non-interactive form so they can be embedded in reports / CI.

Usage::

    # 1) Visualize μ, ω, eigenspectra for several models
    python examples/diagnose_trained_gp.py visualize \\
        --model y3_legacy:/.../learnlogs/model_epoch_920.h5 \\
        --model v2_2lpt_loa0:/.../learnlogs_v2/2lpt_loa0_X/model_epoch_0199.h5 \\
        --out-dir out/diagnostics/visualize

    # 2) Score on 200 random LOA spectra
    python examples/diagnose_trained_gp.py score \\
        --model y3_legacy:/.../model_epoch_920.h5 \\
        --model v2_loa:/.../model_epoch_0799.h5 \\
        --qsocat /.../altbal.fits \\
        --specdir /.../spectro/redux/loa \\
        --n-spectra 200 \\
        --out-dir out/diagnostics/score

    # 3) BAL-vs-non-BAL classifier accuracy
    python examples/diagnose_trained_gp.py classify-bal \\
        --nonbal-model y3_legacy:/.../model_epoch_920.h5 \\
        --bal-model v2_bal_aware:/.../bal_model.h5 \\
        --qsocat /.../altbal.fits \\
        --specdir /.../spectro/redux/loa \\
        --n-bal 100 --n-nonbal 100 \\
        --out-dir out/diagnostics/classify_bal
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Model loading (mirrors notebooks/Visualize Model.ipynb's QSOModelLoader,
# but auto-detects schema and computes the correlation matrix C = K / sqrt(diag).)
# ---------------------------------------------------------------------------
@dataclass
class TrainedModel:
    tag: str
    path: Path
    rest_wavelengths: np.ndarray   # (n_pix,)
    mu: np.ndarray                 # (n_pix,)
    M: np.ndarray                  # (n_pix, k)
    log_omega: np.ndarray          # (n_pix,)
    log_c_0: float
    log_tau_0: float
    log_beta: float
    max_noise_variance: float

    @property
    def K_emission(self) -> np.ndarray:
        return self.M @ self.M.T

    @property
    def correlation(self) -> np.ndarray:
        K = self.K_emission
        diag = np.diag(K)
        return K / np.sqrt(np.outer(diag, diag) + 1e-30)


def load_model(path: Path | str, tag: Optional[str] = None) -> TrainedModel:
    """Load a trained GP H5 (legacy or v2 schema). Auto-detects orientation."""
    path = Path(path)
    if tag is None:
        tag = path.stem

    with h5py.File(path, "r") as f:
        # MATLAB legacy stores log_tau_0 as 2D [0,0]; Python v2 saves as 0-dim.
        log_tau_0_raw = f["log_tau_0"][()]
        is_legacy_matlab = np.asarray(log_tau_0_raw).ndim > 0

        if is_legacy_matlab and np.asarray(log_tau_0_raw).ndim == 2:
            rest_wavelengths = f["rest_wavelengths"][:, 0]
            mu = f["mu"][:, 0]
            M = f["M"][()].T
            log_omega = f["log_omega"][:, 0]
            log_c_0 = float(f["log_c_0"][0, 0])
            log_tau_0 = float(f["log_tau_0"][0, 0])
            log_beta = float(f["log_beta"][0, 0])
        else:
            rest_wavelengths = f["rest_wavelengths"][:].astype(np.float64)
            mu = f["mu"][:].astype(np.float64)
            M = f["M"][()].astype(np.float64)
            log_omega = f["log_omega"][:].astype(np.float64)
            log_c_0 = float(np.asarray(f["log_c_0"][()]).reshape(-1)[0])
            log_tau_0 = float(np.asarray(f["log_tau_0"][()]).reshape(-1)[0])
            log_beta = float(np.asarray(f["log_beta"][()]).reshape(-1)[0])
            if M.ndim == 2 and M.shape[0] != rest_wavelengths.shape[0]:
                M = M.T

        max_noise_variance = float(
            np.asarray(f["max_noise_variance"][()]).reshape(-1)[0]
        ) if "max_noise_variance" in f else 9.0

    return TrainedModel(
        tag=tag, path=path,
        rest_wavelengths=rest_wavelengths, mu=mu, M=M,
        log_omega=log_omega, log_c_0=log_c_0, log_tau_0=log_tau_0,
        log_beta=log_beta, max_noise_variance=max_noise_variance,
    )


def parse_model_arg(s: str) -> tuple[str, Path]:
    """Parse a ``tag:path`` argument; if no colon, use stem as tag."""
    if ":" in s:
        tag, path = s.split(":", 1)
        return tag, Path(path)
    p = Path(s)
    return p.stem, p


# ---------------------------------------------------------------------------
# Sub-command 1: visualize
# ---------------------------------------------------------------------------
def cmd_visualize(args):
    """Plot μ(λ), ω(λ), top-k eigenspectra, hyperparameter table, and
    correlation matrix overlays for the provided models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [load_model(parse_model_arg(s)[1], parse_model_arg(s)[0])
              for s in args.model]
    print(f"[visualize] loaded {len(models)} models: {[m.tag for m in models]}")

    # --- Hyperparameter table ---
    table = []
    for m in models:
        table.append({
            "tag": m.tag,
            "n_pix": int(m.rest_wavelengths.size),
            "k": int(m.M.shape[1]),
            "λ_rest_min": float(m.rest_wavelengths.min()),
            "λ_rest_max": float(m.rest_wavelengths.max()),
            "tau_0": float(np.exp(m.log_tau_0)),
            "beta": float(np.exp(m.log_beta)),
            "c_0": float(np.exp(m.log_c_0)),
            "max_noise_variance": float(m.max_noise_variance),
        })
    with (out_dir / "hyperparameters.json").open("w") as f:
        json.dump(table, f, indent=2)
    print(f"[visualize] wrote hyperparameters.json")

    # --- mu(λ) overlay ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for m in models:
        axes[0].plot(m.rest_wavelengths, m.mu, label=m.tag, alpha=0.85)
    axes[0].set_ylabel(r"$\mu(\lambda_{\rm rest})$  (normalised flux)")
    axes[0].set_title("Mean flux $\\mu$ across trained GP models")
    axes[0].legend(loc="best", fontsize=10)
    axes[0].axvline(1215.67, color="grey", ls=":", alpha=0.5)
    axes[0].axvline(1025.72, color="grey", ls=":", alpha=0.5)
    axes[0].axvline(911.76,  color="grey", ls=":", alpha=0.5)
    for m in models:
        axes[1].plot(m.rest_wavelengths, np.exp(m.log_omega), label=m.tag, alpha=0.85)
    axes[1].set_ylabel(r"$\omega(\lambda_{\rm rest})$  (per-pixel σ)")
    axes[1].set_xlabel(r"$\lambda_{\rm rest}$ (Å)")
    axes[1].set_yscale("log")
    axes[1].set_title("Per-pixel residual scale $\\omega$")
    axes[1].legend(loc="best", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "mu_omega_overlay.png", dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote mu_omega_overlay.png")

    # --- log10(omega ratio) for first model vs the rest, showing where they differ ---
    if len(models) >= 2:
        from scipy.interpolate import interp1d
        ref = models[0]
        fig, ax = plt.subplots(figsize=(14, 5))
        for m in models[1:]:
            # Always interpolate m onto ref's grid (handles different n_pix).
            ref_omega = np.exp(ref.log_omega)
            if m.rest_wavelengths.shape == ref.rest_wavelengths.shape and \
               np.allclose(m.rest_wavelengths, ref.rest_wavelengths):
                m_omega_on_ref = np.exp(m.log_omega)
            else:
                m_omega_on_ref = interp1d(
                    m.rest_wavelengths, np.exp(m.log_omega),
                    bounds_error=False, fill_value=np.nan,
                )(ref.rest_wavelengths)
            with np.errstate(divide="ignore", invalid="ignore"):
                omega_ratio = m_omega_on_ref / np.maximum(ref_omega, 1e-30)
            ax.plot(ref.rest_wavelengths,
                    np.log10(np.maximum(omega_ratio, 1e-6)),
                    label=f"ω[{m.tag}] / ω[{ref.tag}]")
        ax.axhline(0, color="grey", ls="--", alpha=0.6)
        ax.set_xlabel(r"$\lambda_{\rm rest}$ (Å)")
        ax.set_ylabel(r"log$_{10}$ $\omega$ ratio")
        ax.set_title(f"Per-pixel ω(λ) ratio relative to {ref.tag}")
        ax.legend(loc="best", fontsize=10)
        plt.tight_layout()
        fig.savefig(out_dir / "omega_ratio.png", dpi=150)
        plt.close(fig)
        print(f"[visualize] wrote omega_ratio.png")

    # --- Top-k eigenspectra ---
    n_eig = min(args.n_eigenspectra, max(m.M.shape[1] for m in models))
    fig, axes = plt.subplots(n_eig, 1, figsize=(14, 2.5 * n_eig), sharex=True)
    if n_eig == 1:
        axes = [axes]
    for i in range(n_eig):
        for m in models:
            if i < m.M.shape[1]:
                axes[i].plot(m.rest_wavelengths, m.M[:, i], label=m.tag, alpha=0.85)
        axes[i].set_ylabel(f"M[:, {i}]")
        axes[i].axvline(1215.67, color="grey", ls=":", alpha=0.5)
        axes[i].legend(loc="best", fontsize=9)
    axes[-1].set_xlabel(r"$\lambda_{\rm rest}$ (Å)")
    axes[0].set_title(f"Top-{n_eig} M eigenspectra across models")
    plt.tight_layout()
    fig.savefig(out_dir / "eigenspectra.png", dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote eigenspectra.png")

    # --- Correlation matrix per model (one figure each) ---
    for m in models:
        C = m.correlation
        fig, ax = plt.subplots(figsize=(7, 6))
        extent = [m.rest_wavelengths.min(), m.rest_wavelengths.max(),
                  m.rest_wavelengths.min(), m.rest_wavelengths.max()]
        im = ax.imshow(C, origin="lower", extent=extent, cmap="RdBu_r",
                        vmin=-1, vmax=1, aspect="auto")
        ax.set_xlabel(r"$\lambda_{\rm rest}$ (Å)")
        ax.set_ylabel(r"$\lambda_{\rm rest}$ (Å)")
        ax.set_title(f"Emission-feature correlation matrix — {m.tag}")
        plt.colorbar(im, ax=ax, label="Pearson correlation")
        plt.tight_layout()
        fig.savefig(out_dir / f"correlation_{m.tag}.png", dpi=150)
        plt.close(fig)
        print(f"[visualize] wrote correlation_{m.tag}.png")

    # --- Markdown summary ---
    md = ["# GP model visualization\n"]
    md.append("## Hyperparameters\n")
    md.append("| tag | n_pix | k | τ₀ | β | c₀ |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in table:
        md.append(f"| `{r['tag']}` | {r['n_pix']} | {r['k']} | {r['tau_0']:.3e} | {r['beta']:.3f} | {r['c_0']:.3e} |")
    md.append("")
    md.append("## Figures\n")
    md.append("- `mu_omega_overlay.png` — μ(λ) and ω(λ) vs rest wavelength")
    md.append("- `omega_ratio.png` — log10 ω-ratio relative to the first listed model")
    md.append("- `eigenspectra.png` — top-k M eigenvectors")
    md.append("- `correlation_<tag>.png` — emission correlation matrix per model")
    (out_dir / "report.md").write_text("\n".join(md))
    print(f"[visualize] wrote report.md")


# ---------------------------------------------------------------------------
# Helpers shared by score / classify-bal
# ---------------------------------------------------------------------------
def _load_qso_subset(qsocat_path: Path, *, n: int, z_min: float, z_max: float,
                     bal_filter: Optional[str] = None, seed: int = 0,
                     bal_col: str = "BI_CIV", bal_min: float = 0.0):
    """Pick a random subset of TARGETIDs from the altbal catalog.

    bal_filter ∈ {None, "bal", "nonbal"} — None for any, "bal" for
    BAL_COL > BAL_MIN, "nonbal" for BAL_COL == 0.
    """
    from astropy.table import Table
    t = Table.read(str(qsocat_path))
    keep = (t["Z"] >= z_min) & (t["Z"] <= z_max)
    if "ZWARN" in t.colnames:
        keep &= (t["ZWARN"] == 0)
    if bal_filter is not None and bal_col in t.colnames:
        if bal_filter == "bal":
            keep &= (t[bal_col] > bal_min)
        elif bal_filter == "nonbal":
            keep &= (t[bal_col] <= bal_min)
        else:
            raise ValueError(f"bal_filter must be None|bal|nonbal, got {bal_filter}")
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise RuntimeError(f"No rows match filter (n_kept=0)")
    rng = np.random.default_rng(seed)
    idx = np.where(keep)[0]
    if len(idx) > n:
        idx = rng.choice(idx, size=n, replace=False)
    return t[idx]


def _read_one_coadd_loa(specdir: Path, healpix: int, target_ids: list[int]):
    """Same fallback chain as preload_loa_real.py."""
    import fitsio
    from desispec.io import read_spectra
    from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log

    spec_path = (specdir / "healpix" / "main" / "dark"
                 / str(healpix // 100) / str(healpix)
                 / f"coadd-main-dark-{healpix}.fits")
    if not spec_path.exists():
        return []
    spectra = read_spectra(str(spec_path), targetids=target_ids)
    coadded = False
    band = "brz"
    try:
        s = coadd_cameras(spectra)
        if "brz" in s.wave or "b" in s.wave:
            spectra = s
            band = "brz" if "brz" in s.wave else list(s.wave.keys())[0]
            coadded = True
    except Exception:
        coadded = False
    if not coadded:
        if spectra.resolution_data is None:
            return []
        wave_min = float(np.min(spectra.wave["b"]))
        wave_max = float(np.max(spectra.wave["z"]))
        spectra = resample_spectra_lin_or_log(
            spectra, linear_step=0.8, wave_min=wave_min, wave_max=wave_max, fast=True,
        )
        spectra = coadd_cameras(spectra)
        band = "brz" if "brz" in spectra.wave else list(spectra.wave.keys())[0]
    wave = spectra.wave[band].astype(np.float64)
    flux = spectra.flux[band].astype(np.float64)
    ivar = spectra.ivar[band].astype(np.float64)
    mask = spectra.mask[band].astype(bool)
    fibermap_tids = np.asarray(spectra.fibermap["TARGETID"])
    out = []
    for tid in target_ids:
        idx = np.where(fibermap_tids == tid)[0]
        if idx.size == 0:
            continue
        i = int(idx[0])
        out.append((tid, wave, flux[i], ivar[i], mask[i]))
    return out


def _gp_marginal_loglik(model: TrainedModel, wave: np.ndarray, flux: np.ndarray,
                         noise_variance: np.ndarray, mask: np.ndarray, z_qso: float,
                         prev_tau_0: float = 0.00246, prev_beta: float = 3.62,
                         num_forest_lines: int = 3) -> tuple[float, int]:
    """Compute the GP log marginal likelihood log p(y | M_no-DLA) for one
    spectrum under one trained model.

    Reuses the legacy ``NullGPMAT`` so we get the exact same likelihood that
    is used at inference time — no parallel implementation drift.

    Returns (log_p, n_valid_pixels).
    """
    from gpy_dla_detection.null_gp import NullGP
    from gpy_dla_detection.set_parameters import Parameters

    # Map the spectrum onto the model's rest-frame grid.
    rest_wave = wave / (1.0 + z_qso)

    # Spectral mask: drop bad pixels, masked, or out-of-rest-range.
    valid = (
        np.isfinite(flux) & np.isfinite(noise_variance)
        & (~mask) & (noise_variance > 0)
        & (rest_wave >= model.rest_wavelengths.min())
        & (rest_wave <= model.rest_wavelengths.max())
    )
    n_valid = int(valid.sum())
    if n_valid < 50:
        return float("nan"), n_valid

    rest_wave_v = rest_wave[valid]
    flux_v = flux[valid]
    nv_v = noise_variance[valid]

    # Interpolate model μ, M, log_ω onto the spectrum's rest grid.
    from scipy.interpolate import interp1d
    mu_interp = interp1d(model.rest_wavelengths, model.mu, kind="linear")(rest_wave_v)
    omega_interp = interp1d(model.rest_wavelengths, np.exp(model.log_omega),
                             kind="linear")(rest_wave_v)
    M_interp = np.empty((n_valid, model.M.shape[1]))
    for j in range(model.M.shape[1]):
        M_interp[:, j] = interp1d(model.rest_wavelengths, model.M[:, j],
                                   kind="linear")(rest_wave_v)

    # Mean-flux suppression a_F = exp(-τ_eff(λ_obs))
    obs_wave_v = rest_wave_v * (1.0 + z_qso)
    from gpy_dla_detection.effective_optical_depth import effective_optical_depth
    tau = effective_optical_depth(obs_wave_v, beta=prev_beta, tau_0=prev_tau_0,
                                   z_qso=z_qso, num_forest_lines=num_forest_lines)
    a_F = np.exp(-np.sum(tau, axis=1))

    # Apply the absorption sandwich (Ho 2020 eq. 41 with no DLA):
    #     C = A_F (K + Ω) A_F + V
    this_mu = a_F * mu_interp
    this_M = a_F[:, None] * M_interp
    c_0 = float(np.exp(model.log_c_0))
    scaling = 1.0 - a_F + c_0
    this_omega2 = (omega_interp ** 2) * (scaling ** 2)
    d = this_omega2 + nv_v

    return float(NullGP.log_mvnpdf_low_rank(flux_v, this_mu, this_M, d)), n_valid


# ---------------------------------------------------------------------------
# Sub-command 2: score
# ---------------------------------------------------------------------------
def cmd_score(args):
    """Compute per-spectrum NullGP log marginal likelihood under each
    provided model on a random sample of LOA spectra. Writes a CSV and a
    Markdown report with comparison statistics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [load_model(parse_model_arg(s)[1], parse_model_arg(s)[0])
              for s in args.model]
    print(f"[score] loaded {len(models)} models: {[m.tag for m in models]}")

    qcat = _load_qso_subset(
        Path(args.qsocat), n=args.n_spectra,
        z_min=args.z_min, z_max=args.z_max,
        bal_filter=None, seed=args.seed,
    )
    print(f"[score] picked {len(qcat)} test spectra")

    # Group by healpix.
    by_hpx: dict[int, list[tuple[int, float]]] = {}
    for h, tid, z in zip(qcat["HPXPIXEL"], qcat["TARGETID"], qcat["Z"]):
        by_hpx.setdefault(int(h), []).append((int(tid), float(z)))

    rows: list[dict] = []
    t0 = time.time()
    n_processed = 0
    n_skipped = 0
    for hpx_idx, (healpix, target_pairs) in enumerate(sorted(by_hpx.items())):
        target_ids = [tid for tid, _ in target_pairs]
        z_qso_dict = {tid: z for tid, z in target_pairs}
        try:
            results = _read_one_coadd_loa(Path(args.specdir), healpix, target_ids)
        except Exception:
            n_skipped += len(target_pairs)
            continue
        for tid, wave, flux, ivar, mask in results:
            z_qso = z_qso_dict[tid]
            with np.errstate(divide="ignore", invalid="ignore"):
                noise_variance = np.where(
                    (ivar > 0) & np.isfinite(ivar), 1.0 / ivar, np.nan
                )
            row = {"target_id": int(tid), "z_qso": z_qso}
            for m in models:
                try:
                    logp, n_valid = _gp_marginal_loglik(
                        m, wave, flux, noise_variance, mask, z_qso,
                    )
                except Exception as e:
                    logp, n_valid = float("nan"), 0
                row[f"logp_{m.tag}"] = logp
                row[f"n_valid_{m.tag}"] = n_valid
                # Per-pixel NLL (lower = better fit per pixel).
                row[f"nll_per_pix_{m.tag}"] = -logp / max(n_valid, 1) if np.isfinite(logp) else float("nan")
            rows.append(row)
            n_processed += 1
        if (hpx_idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"[score] hpx {hpx_idx+1}/{len(by_hpx)}, "
                  f"{n_processed} spectra ({n_processed/max(elapsed,1e-3):.1f}/s, "
                  f"skipped {n_skipped})")

    print(f"[score] done: {n_processed} scored, {n_skipped} skipped, "
          f"{(time.time()-t0)/60:.1f} min")
    if not rows:
        sys.exit("[error] no spectra scored")

    # CSV
    csv_path = out_dir / "scores.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[score] wrote {csv_path}")

    # Stats per model
    summary = []
    for m in models:
        nll = np.array([r[f"nll_per_pix_{m.tag}"] for r in rows], dtype=float)
        finite = np.isfinite(nll)
        summary.append({
            "tag": m.tag,
            "n_finite": int(finite.sum()),
            "nll_per_pix_median": float(np.median(nll[finite])) if finite.any() else float("nan"),
            "nll_per_pix_p16": float(np.percentile(nll[finite], 16)) if finite.any() else float("nan"),
            "nll_per_pix_p84": float(np.percentile(nll[finite], 84)) if finite.any() else float("nan"),
        })
    print(f"[score] summary:")
    for s in summary:
        print(f"  {s['tag']:>20s}: median NLL/pix = {s['nll_per_pix_median']:.3f}  "
              f"[{s['nll_per_pix_p16']:.3f}, {s['nll_per_pix_p84']:.3f}]")

    # Histogram per model
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in models:
        nll = np.array([r[f"nll_per_pix_{m.tag}"] for r in rows], dtype=float)
        nll = nll[np.isfinite(nll)]
        if nll.size:
            ax.hist(nll, bins=40, histtype="step", linewidth=2, label=m.tag)
    ax.set_xlabel("NLL per pixel  (lower = better fit)")
    ax.set_ylabel("count")
    ax.set_title(f"GP NullModel NLL/pix distribution ({n_processed} spectra)")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_dir / "nll_per_pix_hist.png", dpi=150)
    plt.close(fig)

    # Pairwise scatter (only when 2 models)
    if len(models) == 2:
        a, b = models[0], models[1]
        x = np.array([r[f"nll_per_pix_{a.tag}"] for r in rows])
        y = np.array([r[f"nll_per_pix_{b.tag}"] for r in rows])
        finite = np.isfinite(x) & np.isfinite(y)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(x[finite], y[finite], s=8, alpha=0.5)
        lo = min(x[finite].min(), y[finite].min())
        hi = max(x[finite].max(), y[finite].max())
        ax.plot([lo, hi], [lo, hi], "r-", alpha=0.4, label="y=x")
        ax.set_xlabel(f"NLL/pix [{a.tag}]")
        ax.set_ylabel(f"NLL/pix [{b.tag}]")
        ax.set_title(f"Per-spectrum NLL/pix: {a.tag} vs {b.tag}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / f"nll_scatter_{a.tag}_vs_{b.tag}.png", dpi=150)
        plt.close(fig)

    md = ["# Score diagnostic\n"]
    md.append(f"Test spectra: {n_processed} ({n_skipped} skipped, "
              f"z ∈ [{args.z_min}, {args.z_max}])")
    md.append(f"Source: {args.qsocat}\n")
    md.append("## Per-model NLL/pix\n")
    md.append("| tag | n_finite | median | p16 | p84 |")
    md.append("|---|---:|---:|---:|---:|")
    for s in summary:
        md.append(f"| `{s['tag']}` | {s['n_finite']} | "
                  f"{s['nll_per_pix_median']:.4f} | "
                  f"{s['nll_per_pix_p16']:.4f} | "
                  f"{s['nll_per_pix_p84']:.4f} |")
    md.append("\nLower NLL/pix = the model's GP describes the spectrum better.\n")
    md.append("## Figures\n")
    md.append("- `nll_per_pix_hist.png` — distribution overlay across models")
    if len(models) == 2:
        md.append(f"- `nll_scatter_{models[0].tag}_vs_{models[1].tag}.png` — pairwise per-spectrum scatter")
    (out_dir / "report.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# Sub-command 3: classify-bal
# ---------------------------------------------------------------------------
def cmd_classify_bal(args):
    """Bayes-factor classifier: pick the model with higher GP marginal
    likelihood for each spectrum. Compute accuracy + confusion matrix
    against the ground-truth ``BI_CIV`` flag."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nonbal_tag, nonbal_path = parse_model_arg(args.nonbal_model)
    bal_tag, bal_path = parse_model_arg(args.bal_model)
    nonbal = load_model(nonbal_path, nonbal_tag)
    bal = load_model(bal_path, bal_tag)
    print(f"[classify] non-BAL model: {nonbal.tag}; BAL model: {bal.tag}")

    qcat_bal = _load_qso_subset(
        Path(args.qsocat), n=args.n_bal,
        z_min=args.z_min, z_max=args.z_max,
        bal_filter="bal", seed=args.seed,
    )
    qcat_nonbal = _load_qso_subset(
        Path(args.qsocat), n=args.n_nonbal,
        z_min=args.z_min, z_max=args.z_max,
        bal_filter="nonbal", seed=args.seed + 1,
    )
    print(f"[classify] BAL sample: {len(qcat_bal)}; non-BAL sample: {len(qcat_nonbal)}")

    rows: list[dict] = []
    for qcat, label in ((qcat_bal, "bal"), (qcat_nonbal, "nonbal")):
        by_hpx: dict[int, list[tuple[int, float]]] = {}
        for h, tid, z in zip(qcat["HPXPIXEL"], qcat["TARGETID"], qcat["Z"]):
            by_hpx.setdefault(int(h), []).append((int(tid), float(z)))
        for healpix, target_pairs in sorted(by_hpx.items()):
            target_ids = [tid for tid, _ in target_pairs]
            z_dict = {tid: z for tid, z in target_pairs}
            try:
                res = _read_one_coadd_loa(Path(args.specdir), healpix, target_ids)
            except Exception:
                continue
            for tid, wave, flux, ivar, mask in res:
                with np.errstate(divide="ignore", invalid="ignore"):
                    nv = np.where((ivar > 0) & np.isfinite(ivar), 1.0 / ivar, np.nan)
                z_qso = z_dict[tid]
                logp_n, n_valid_n = _gp_marginal_loglik(nonbal, wave, flux, nv, mask, z_qso)
                logp_b, n_valid_b = _gp_marginal_loglik(bal, wave, flux, nv, mask, z_qso)
                rows.append({
                    "target_id": int(tid), "z_qso": z_qso, "truth_label": label,
                    f"logp_{nonbal.tag}": logp_n, f"logp_{bal.tag}": logp_b,
                    f"n_valid_{nonbal.tag}": n_valid_n,
                    f"n_valid_{bal.tag}": n_valid_b,
                })

    print(f"[classify] scored {len(rows)} spectra")
    if not rows:
        sys.exit("[error] no spectra scored")

    csv_path = out_dir / "classify_scores.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Predict the model with higher logp; compare to truth_label.
    correct = 0
    total = 0
    confusion = {"bal": {"bal": 0, "nonbal": 0},
                 "nonbal": {"bal": 0, "nonbal": 0}}
    for r in rows:
        if not (np.isfinite(r[f"logp_{nonbal.tag}"]) and np.isfinite(r[f"logp_{bal.tag}"])):
            continue
        pred = "bal" if r[f"logp_{bal.tag}"] > r[f"logp_{nonbal.tag}"] else "nonbal"
        confusion[r["truth_label"]][pred] += 1
        total += 1
        if pred == r["truth_label"]:
            correct += 1
    accuracy = correct / max(total, 1)

    md = [f"# BAL vs non-BAL Bayes-factor classifier\n"]
    md.append(f"- non-BAL model: `{nonbal.tag}` from `{nonbal.path}`")
    md.append(f"- BAL model: `{bal.tag}` from `{bal.path}`")
    md.append(f"- BAL truth = `BI_CIV > 0` in altbal\n")
    md.append(f"## Accuracy: **{accuracy*100:.1f}%** ({correct} / {total})\n")
    md.append("## Confusion matrix\n")
    md.append("| truth ↓  /  predicted → | bal | nonbal |")
    md.append("|---|---:|---:|")
    for truth in ("bal", "nonbal"):
        md.append(f"| **{truth}** | {confusion[truth]['bal']} | {confusion[truth]['nonbal']} |")
    md.append("")

    # Histogram of log Bayes factor per truth class
    bf = np.array([r[f"logp_{bal.tag}"] - r[f"logp_{nonbal.tag}"] for r in rows])
    finite = np.isfinite(bf)
    truth_arr = np.array([r["truth_label"] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 5))
    if (truth_arr == "bal").any():
        ax.hist(bf[finite & (truth_arr == "bal")], bins=30, alpha=0.6, label="truth=BAL", color="C1")
    if (truth_arr == "nonbal").any():
        ax.hist(bf[finite & (truth_arr == "nonbal")], bins=30, alpha=0.6, label="truth=nonBAL", color="C0")
    ax.axvline(0, color="grey", ls="--", label="decision boundary")
    ax.set_xlabel(f"log p(BAL model) − log p(nonBAL model)")
    ax.set_ylabel("count")
    ax.set_title(f"Bayes-factor distribution by BAL truth (acc {accuracy*100:.1f}%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "log_bayes_factor_hist.png", dpi=150)
    plt.close(fig)

    md.append("## Figures\n")
    md.append("- `log_bayes_factor_hist.png` — log Bayes factor `(BAL − nonBAL)` "
              "split by truth class. Cleanly separated peaks ⇒ high accuracy.")
    (out_dir / "report.md").write_text("\n".join(md))
    print(f"[classify] accuracy: {accuracy*100:.2f}%, wrote {out_dir}/report.md")


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("visualize", help="μ/ω/M overlays, no spectra needed")
    pv.add_argument("--model", action="append", required=True,
                    help="`tag:path` (repeat for each model)")
    pv.add_argument("--out-dir", required=True)
    pv.add_argument("--n-eigenspectra", type=int, default=5)
    pv.set_defaults(func=cmd_visualize)

    ps = sub.add_parser("score", help="GP NLL on a sample of test spectra")
    ps.add_argument("--model", action="append", required=True,
                    help="`tag:path` (repeat for each model)")
    ps.add_argument("--qsocat", required=True)
    ps.add_argument("--specdir", required=True)
    ps.add_argument("--n-spectra", type=int, default=200)
    ps.add_argument("--z-min", type=float, default=2.0)
    ps.add_argument("--z-max", type=float, default=4.25)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--out-dir", required=True)
    ps.set_defaults(func=cmd_score)

    pc = sub.add_parser("classify-bal", help="Bayes-factor BAL classifier")
    pc.add_argument("--nonbal-model", required=True, help="tag:path")
    pc.add_argument("--bal-model", required=True, help="tag:path")
    pc.add_argument("--qsocat", required=True)
    pc.add_argument("--specdir", required=True)
    pc.add_argument("--n-bal", type=int, default=100)
    pc.add_argument("--n-nonbal", type=int, default=100)
    pc.add_argument("--z-min", type=float, default=2.0)
    pc.add_argument("--z-max", type=float, default=4.25)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--out-dir", required=True)
    pc.set_defaults(func=cmd_classify_bal)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
