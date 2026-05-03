"""Plot loss curves for v2 training runs (and v1 if loss_history exists).

Reads loss_history.json from each RUN_DIR. Produces a 2-panel figure:
  - linear y-axis (per-epoch NLL)
  - log y-axis with running mean (convergence behavior)

Helps surface: stuck local optima, divergence, premature LR-anneal
collapse.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def _load_loss(run_dir: Path) -> np.ndarray | None:
    p = run_dir / "loss_history.json"
    if not p.exists():
        return None
    with p.open() as f:
        h = json.load(f)
    return np.asarray(h, dtype=np.float64)


def _v1_loss_history(model_h5: Path) -> np.ndarray | None:
    """Legacy v1 .h5 stores loss_history as an h5 dataset (per
    learn_qso_model.py:save). Return it if present."""
    try:
        with h5py.File(model_h5, "r") as f:
            if "loss_history" in f:
                return np.asarray(f["loss_history"][:]).flatten()
    except Exception:
        pass
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True,
                   help="Comma-separated run-dir paths (each must contain loss_history.json) "
                        "or model.h5 paths (will look for sibling loss_history.json or "
                        "embedded loss_history dataset)")
    p.add_argument("--labels", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    paths = [Path(s.strip()) for s in args.runs.split(",")]
    labels = args.labels.split(",") if args.labels else [str(p.name) for p in paths]
    assert len(labels) == len(paths)

    histories = []
    for p_, label in zip(paths, labels):
        if p_.is_dir():
            h = _load_loss(p_)
        else:
            # Try sibling loss_history.json first
            sib = p_.parent / "loss_history.json"
            if sib.exists():
                with sib.open() as f:
                    h = np.asarray(json.load(f), dtype=np.float64)
            else:
                h = _v1_loss_history(p_)
        if h is None:
            print(f"  [skip] {label}: no loss history at {p_}")
            continue
        h = h[np.isfinite(h)]
        print(f"  {label:55s}  n_epochs={len(h):4d}  start={h[0]:.3e}  end={h[-1]:.3e}  "
              f"min={h.min():.3e} @ epoch {int(np.argmin(h))}")
        histories.append((label, h))

    if not histories:
        raise SystemExit("no loss histories found")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    cmap = plt.get_cmap("tab10")

    # Panel 1: linear y-axis, all curves overlaid
    ax = axes[0]
    for i, (label, h) in enumerate(histories):
        ax.plot(np.arange(len(h)), h, lw=0.8, color=cmap(i % 10), label=label, alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (NLL)")
    ax.set_title("Loss curves — linear y-axis")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 2: per-epoch fractional change, log scale (convergence rate)
    ax = axes[1]
    for i, (label, h) in enumerate(histories):
        if len(h) < 2:
            continue
        # Smooth: 10-epoch rolling mean
        if len(h) > 20:
            kernel = np.ones(10) / 10
            h_smooth = np.convolve(h, kernel, mode="valid")
        else:
            h_smooth = h
        # Plot |Δloss / loss| as a measure of convergence rate
        delta_rel = np.abs(np.diff(h_smooth)) / np.maximum(np.abs(h_smooth[:-1]), 1e-30)
        if (delta_rel > 0).any():
            ax.semilogy(np.arange(len(delta_rel)), delta_rel, lw=0.8,
                        color=cmap(i % 10), label=label, alpha=0.85)
    ax.set_xlabel("epoch (10-epoch smoothed)")
    ax.set_ylabel("|Δloss| / loss   (per epoch)")
    ax.set_title("Convergence rate — log scale (lower = converged)")
    ax.axhline(1e-6, color="0.6", ls=":", lw=0.5, label="1e-6 threshold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[main] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
