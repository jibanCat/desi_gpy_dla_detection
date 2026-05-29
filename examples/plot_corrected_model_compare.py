"""Side-by-side comparison of the 6 _corrected v2 retrains.

For each model we extract from the trained ``.h5``:
  - μ(λ): the GP null mean
  - ω(λ): per-pixel forest-noise scale (= exp(log_omega))
  - corr(M·M^T)(λ, λ'): correlation matrix of the low-rank covariance
    component, normalized by its own diagonal — i.e.
       corr_ij = (M·M^T)_ij / sqrt((M·M^T)_ii · (M·M^T)_jj)
    This is the "shape" of the GP's structural covariance, with the
    per-pixel amplitude divided out.

Three figures are written:
  trained_corrected_compare_mu.png          μ overlay (6 lines)
  trained_corrected_compare_omega.png       ω overlay (6 lines, log y)
  trained_corrected_corr_grid.png           2x3 grid of corr matrices
  trained_corrected_summary_table.md        χ²/n + trace/eig diagnostics

The third figure is the discriminator: v1's trace_omega²/trace(K) is
~0.84 (most of K lives in per-pixel noise) and the previous v2 trained
~0.002–0.034 (most of K lives in M·M^T basis, which corresponds to
strongly-correlated structure across pixels). The corrected runs test
whether the new flag combo prevents that collapse.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


# Each entry: (display_name, model.h5 path, optional metrics.json path)
MODELS = [
    ("loa_no_dla_no_bal_corrected",
     "/scratch/cavestru_root/cavestru0/mfho/gl_outputs/GP_trained/loa_no_dla_no_bal_corrected/model_epoch_1499.h5",
     "real LOA, no DLAs no BALs"),
    ("loa_no_hcd_with_bal_corrected",
     "/scratch/cavestru_root/cavestru0/mfho/gl_outputs/GP_trained/loa_no_hcd_with_bal_corrected/model_epoch_1499.h5",
     "real LOA, no HCDs but BALs kept"),
    ("2lpt_loa0_corrected",
     "/scratch/cavestru_root/cavestru0/mfho/gl_outputs/v2_runs/2lpt_loa0_corrected/model_epoch_1499.h5",
     "2lpt mock loa-0 (forest-only)"),
    ("2lpt_loa124_nohcd_nobal_corrected",
     "/scratch/cavestru_root/cavestru0/mfho/gl_outputs/v2_runs/2lpt_loa124_nohcd_nobal_corrected/model_epoch_1499.h5",
     "2lpt mock loa-124, no HCDs no BALs"),
    ("saclay_mock0_nohcd_nobal_corrected",
     "/scratch/cavestru_root/cavestru0/mfho/gl_outputs/v2_runs/saclay_mock0_nohcd_nobal_corrected/model_epoch_1499.h5",
     "saclay mock-0, no HCDs no BALs"),
    ("2lpt_bal_only_corrected",
     "/scratch/cavestru_root/cavestru0/mfho/gl_outputs/v2_runs/2lpt_bal_only_corrected/model_epoch_1499.h5",
     "2lpt mock loa-124, BALs only (BI_CIV>0)"),
]


def _load(path):
    with h5py.File(path, "r") as f:
        rw = np.asarray(f["rest_wavelengths"]); rw = rw[:,0] if rw.ndim==2 else rw
        mu = np.asarray(f["mu"]); mu = mu[:,0] if mu.ndim==2 else mu
        log_omega = np.asarray(f["log_omega"]); log_omega = log_omega[:,0] if log_omega.ndim==2 else log_omega
        M = np.asarray(f["M"])
        if M.shape[0] != rw.shape[0]:
            M = M.T
        log_tau_0 = float(np.asarray(f["log_tau_0"]).flatten()[0])
        log_beta = float(np.asarray(f["log_beta"]).flatten()[0])
        log_c_0 = float(np.asarray(f["log_c_0"]).flatten()[0])
    return dict(rest=rw, mu=mu, omega=np.exp(log_omega), M=M,
                tau_0=float(np.exp(log_tau_0)),
                beta=float(np.exp(log_beta)),
                c_0=float(np.exp(log_c_0)))


def _correlation_matrix(M):
    """corr(M·M^T) normalized by its own diagonal.

    Returns an n_pix × n_pix matrix with diag=1 and off-diag in [-1, 1].
    """
    K = M @ M.T  # (n_pix, n_pix)
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    C = K / np.outer(d, d)
    return np.clip(C, -1.0, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out-dir", default="docs/notes/2026-05-06_corrected_model_validation/figs")
    p.add_argument("--metrics-dir", default="docs/notes/2026-05-06_corrected_model_validation/metrics")
    p.add_argument("--include-prod-y3", action="store_true",
                   help="Also include the v1 production model "
                        "(model_epoch_920.h5) for reference. Note μ/ω are "
                        "in different units (v1 normalized vs v2 absolute).")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_dir)

    extracted = {}
    for name, path, _ in MODELS:
        if not os.path.exists(path):
            print(f"[skip] {name}: not at {path}")
            continue
        extracted[name] = _load(path)
        e = extracted[name]
        n_pix, k = e["M"].shape
        trace_K = float((e["omega"]**2).sum() + (e["M"]**2).sum())
        trace_w = float((e["omega"]**2).sum())
        ratio = trace_w / trace_K
        print(f"  {name:<40}  τ₀={e['tau_0']:.5f} β={e['beta']:.2f} "
              f"trace_ω²/K={ratio:.3f}  n_pix={n_pix} k={k}")

    if args.include_prod_y3:
        prod_path = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5"
        if os.path.exists(prod_path):
            extracted["PROD_y3_v1"] = _load(prod_path)

    color_map = plt.cm.tab10(np.linspace(0, 1, max(10, len(extracted))))
    color_for = {n: color_map[i % len(color_map)] for i, n in enumerate(extracted)}

    # ---- μ overlay ----
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for n, e in extracted.items():
        ax.plot(e["rest"], e["mu"], lw=0.9, color=color_for[n], label=n)
    ax.axvline(1215.67, color="0.6", lw=0.6, ls="--", label="Lyα")
    ax.set_xlabel("rest wavelength [Å]")
    ax.set_ylabel("μ (centered, ~0 baseline)")
    ax.set_title("GP null mean μ across the 6 _corrected retrains")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_mu = out_dir / "trained_corrected_compare_mu.png"
    fig.savefig(out_mu, dpi=130); plt.close(fig)
    print(f"[saved] {out_mu}")

    # ---- ω overlay ----
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for n, e in extracted.items():
        ax.plot(e["rest"], e["omega"], lw=0.9, color=color_for[n], label=n)
    ax.axvline(1215.67, color="0.6", lw=0.6, ls="--")
    ax.set_xlabel("rest wavelength [Å]")
    ax.set_ylabel("ω (per-pixel noise scale)")
    ax.set_title("ω across the 6 _corrected retrains (log y)")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_omega = out_dir / "trained_corrected_compare_omega.png"
    fig.savefig(out_omega, dpi=130); plt.close(fig)
    print(f"[saved] {out_omega}")

    # ---- corr(M·M^T) grid ----
    n_models = len(extracted)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             squeeze=False)
    for i, (name, e) in enumerate(extracted.items()):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        C = _correlation_matrix(e["M"])
        rest = e["rest"]
        extent = [rest[0], rest[-1], rest[-1], rest[0]]
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, extent=extent,
                       interpolation="nearest", aspect="auto")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("λ′ [Å]")
        ax.set_ylabel("λ [Å]")
        ax.axhline(1215.67, color="0.2", lw=0.4, alpha=0.5)
        ax.axvline(1215.67, color="0.2", lw=0.4, alpha=0.5)
    # turn off any extra panels
    for j in range(n_models, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")
    fig.suptitle("corr(M·M^T) per model — diagonal-normalized covariance shape\n"
                 "(red = positively correlated, blue = anti-correlated)",
                 fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                 label="correlation")
    out_corr = out_dir / "trained_corrected_corr_grid.png"
    fig.savefig(out_corr, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_corr}")

    # ---- summary table (markdown) ----
    rows = ["| model | τ₀ | β | c₀ | trace_ω²/K | top_eig_MMT | eig1/eig2 | χ²/n | n_evaluated |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, e in extracted.items():
        # Read metrics JSON if available
        m_path = metrics_dir / f"{name}.json"
        m = json.loads(m_path.read_text()) if m_path.exists() else {}
        chi2_n = m.get("chi2_per_n_mean")
        n_eval = m.get("n_evaluated")
        top_eig = m.get("top_eig_MMT", [None])
        ratio_w = m.get("trace_omega2_over_trace_K")
        eig_ratio = m.get("top_eig_ratio")
        # Fall back to computing if metrics not present
        if ratio_w is None:
            trace_K = float((e["omega"]**2).sum() + (e["M"]**2).sum())
            ratio_w = float((e["omega"]**2).sum() / trace_K)
        if top_eig and top_eig[0] is None:
            s = np.linalg.svd(e["M"], compute_uv=False)
            top_eig = (s ** 2).tolist()
            eig_ratio = top_eig[0] / top_eig[1]
        rows.append(
            f"| {name} | {e['tau_0']:.5f} | {e['beta']:.2f} | {e['c_0']:.4f} | "
            f"{ratio_w:.3f} | {(top_eig[0] if top_eig else float('nan')):.2e} | "
            f"{(eig_ratio if eig_ratio is not None else float('nan')):.2f} | "
            f"{chi2_n if chi2_n is not None else 'N/A':<6} | "
            f"{n_eval if n_eval is not None else 'N/A'} |"
        )
    out_md = out_dir.parent / "summary_table.md"
    out_md.write_text("\n".join(rows) + "\n")
    print(f"[saved] {out_md}")


if __name__ == "__main__":
    main()
