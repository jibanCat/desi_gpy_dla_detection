"""Step A.3 comparison: v1 vs v3.5 vs MATLAB short-retrain endpoints.

Loads:
  tests/fixtures/2lpt_frozen/short_retrain/v1.npz
  tests/fixtures/2lpt_frozen/short_retrain/v3.5.npz
  tests/fixtures/2lpt_frozen/short_retrain/matlab.mat   (optional)

Produces:
  tests/fixtures/2lpt_frozen/short_retrain/comparison.png
        4-panel: loss curve, β trajectory, τ_0 trajectory, c_0 trajectory
  tests/fixtures/2lpt_frozen/short_retrain/mu_omega_M.png
        μ overlay, ω overlay, eigenspectrum-of-M comparison
  tests/fixtures/2lpt_frozen/short_retrain/corr_grid.png
        corr(M·M^T) per lane
  tests/fixtures/2lpt_frozen/short_retrain/SUMMARY.md
        markdown table of converged values + inter-lane Δ
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "fixtures" / "2lpt_frozen" / "short_retrain"


def _load(lane):
    if lane in ("v1", "v3.5"):
        p = OUT / f"{lane}.npz"
        if not p.exists():
            return None
        d = np.load(p)
        return {k: d[k] for k in d.files}
    if lane == "matlab":
        p = OUT / "matlab.mat"
        if not p.exists():
            return None
        m = loadmat(p)
        # squeeze MATLAB's (N,1) shapes
        return {k: np.asarray(v).squeeze() for k, v in m.items() if not k.startswith("__")}
    raise ValueError(lane)


def _corr_matrix(M):
    K = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    return np.clip(K / np.outer(d, d), -1.0, 1.0)


def main():
    lanes = {n: _load(n) for n in ["v1", "v3.5", "matlab"]}
    available = [n for n, d in lanes.items() if d is not None]
    if not available:
        print("no lanes found")
        return 1
    print(f"available: {available}")
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Trajectories ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    color = {"v1": "C0", "v3.5": "C2", "matlab": "C3"}
    for ax, key, title in [
        (axes[0, 0], "loss_history", "loss"),
        (axes[0, 1], "log_beta_history", "log β"),
        (axes[1, 0], "log_tau_0_history", "log τ_0"),
        (axes[1, 1], "log_c_0_history", "log c_0"),
    ]:
        for n in available:
            d = lanes[n]
            if key not in d:
                continue
            y = np.asarray(d[key]).flatten()
            ax.plot(y, color=color[n], lw=1.0, label=n)
        ax.set_xlabel("iteration")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Step A.3 trajectories: v1 (approx dlog_β) vs v3.5 (strict) vs MATLAB",
                 fontsize=11)
    fig.tight_layout()
    out_traj = OUT / "comparison.png"
    fig.savefig(out_traj, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_traj}")

    # ── μ, ω, M-eigenspectrum overlays ─────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    for n in available:
        d = lanes[n]
        rest = np.asarray(d.get("rest_wavelengths", np.arange(d["mu"].shape[0]))).flatten()
        mu = np.asarray(d["mu"]).flatten()
        log_omega = np.asarray(d["log_omega_final"]).flatten()
        M = np.asarray(d["M_final"])
        if M.ndim == 1: M = M.reshape(-1, 30)
        if M.shape[0] != rest.shape[0]:
            M = M.T
        s = np.linalg.svd(M, compute_uv=False)

        axes[0].plot(rest, mu, color=color[n], lw=0.9, label=n)
        axes[1].plot(rest, np.exp(log_omega), color=color[n], lw=0.9, label=n)
        axes[2].plot(np.arange(1, len(s) + 1), s ** 2, "o-", color=color[n], label=n, ms=4)

    axes[0].set_xlabel("rest λ [Å]"); axes[0].set_ylabel("μ"); axes[0].axvline(1215.67, color="0.6", ls="--", lw=0.5)
    axes[0].set_title("μ"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("rest λ [Å]"); axes[1].set_ylabel("ω"); axes[1].set_yscale("log")
    axes[1].set_title("ω (log y)"); axes[1].grid(alpha=0.3, which="both"); axes[1].legend(fontsize=8)
    axes[2].set_xlabel("eigenvalue rank"); axes[2].set_ylabel("σ² (eigenvalue of M·M^T)")
    axes[2].set_yscale("log"); axes[2].set_title("M·M^T eigenspectrum (final)")
    axes[2].grid(alpha=0.3, which="both"); axes[2].legend(fontsize=8)
    fig.tight_layout()
    out_mom = OUT / "mu_omega_M.png"
    fig.savefig(out_mom, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_mom}")

    # ── corr(M·M^T) grid ───────────────────────────────────────
    fig, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 4.5),
                             squeeze=False)
    for ax, n in zip(axes[0], available):
        d = lanes[n]
        M = np.asarray(d["M_final"])
        if M.ndim == 1: M = M.reshape(-1, 30)
        rest = np.asarray(d.get("rest_wavelengths", np.arange(M.shape[0]))).flatten()
        if M.shape[0] != rest.shape[0]:
            M = M.T
        C = _corr_matrix(M)
        extent = [rest[0], rest[-1], rest[-1], rest[0]]
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, extent=extent,
                       interpolation="nearest", aspect="auto")
        ax.set_title(n, fontsize=10)
        ax.set_xlabel("λ′ [Å]"); ax.set_ylabel("λ [Å]")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="correlation")
    fig.suptitle("corr(M·M^T) per lane (final)", fontsize=11)
    out_corr = OUT / "corr_grid.png"
    fig.savefig(out_corr, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out_corr}")

    # ── Markdown summary ───────────────────────────────────────
    rows = ["| metric | v1 | v3.5 | MATLAB | Δ(v3.5−v1) | Δ(MATLAB−v1) |",
            "|---|---:|---:|---:|---:|---:|"]
    def _scalar(name, get_fn, fmt="{:.6f}"):
        vals = {n: get_fn(lanes[n]) if lanes[n] is not None else None for n in ["v1", "v3.5", "matlab"]}
        cells = [fmt.format(vals[n]) if vals[n] is not None else "—" for n in ["v1", "v3.5", "matlab"]]
        deltas = []
        for n in ["v3.5", "matlab"]:
            if vals[n] is None or vals["v1"] is None:
                deltas.append("—")
            else:
                deltas.append(fmt.format(vals[n] - vals["v1"]))
        rows.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {deltas[0]} | {deltas[1]} |")

    _scalar("loss[final]", lambda d: float(np.asarray(d["loss_history"]).flatten()[-1]), "{:.2f}")
    _scalar("c_0", lambda d: float(d["c_0_final"]), "{:.6f}")
    _scalar("τ_0", lambda d: float(d["tau_0_final"]), "{:.6f}")
    _scalar("β", lambda d: float(d["beta_final"]), "{:.4f}")
    _scalar("log τ_0", lambda d: float(d["log_tau_0_final"]), "{:.4f}")
    _scalar("log β", lambda d: float(d["log_beta_final"]), "{:.4f}")

    out_md = OUT / "SUMMARY.md"
    out_md.write_text("# Step A.3 short-retrain endpoint comparison\n\n" + "\n".join(rows) + "\n")
    print(f"[saved] {out_md}")
    print("\n--- summary ---")
    for r in rows: print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
