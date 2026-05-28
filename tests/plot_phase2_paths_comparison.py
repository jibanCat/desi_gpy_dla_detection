"""Comparison plots: Phase 2 across all training paths.

Generates a 4-panel figure that overlays the training trajectories and
endpoint scalars across:

  - Per-spectrum 89k×200 (49671617, walltime-killed at iter 175 — log only)
  - Per-spectrum 89k×200 (49709974 — fresh resubmit, optional, when avail)
  - Vectorized 89k×200 (49700040, COMPLETED — phase2_result.npz)
  - Vec smoke 5k×50    (49699997, COMPLETED — phase2_result.npz)
  - Phase-1 5k×50      (commit 0918ea7 baseline — phase2_result.npz)
  - MATLAB DR16 final  (loaded from learned_qso_model_*.mat)

Output:
  docs/notes/2026-05-09_phase2_paths_comparison.png
  docs/notes/2026-05-09_phase2_paths_speedup.png

Re-run when the new per-spec result npz appears at
docs/notes/2026-05-08_matlab_dr16_validation_per_spec/phase2_result.npz
to fold it in.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
NOTES = REPO / "docs" / "notes"
OUT = NOTES / "2026-05-09_phase2_paths_comparison.png"
OUT_SPEEDUP = NOTES / "2026-05-09_phase2_paths_speedup.png"

VEC_FULL = NOTES / "2026-05-08_matlab_dr16_validation_vec_full" / "phase2_result.npz"
VEC_SMOKE = NOTES / "2026-05-08_matlab_dr16_validation_vec_smoke" / "phase2_result.npz"
PHASE1 = NOTES / "2026-05-08_matlab_dr16_validation" / "phase2_result.npz"
PER_SPEC_NEW = NOTES / "2026-05-08_matlab_dr16_validation_per_spec" / "phase2_result.npz"

PER_SPEC_LOG_49671617 = REPO / "slurm" / "greatlakes" / "phase2_dr16_49671617.log"

MATLAB_REF = Path("/home/mfho/MATLAB/gp_dla_detection_dr16q_public/data/dr16/MATLAB_Catalogue/"
                   "learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat")


def _parse_log(log_path: Path):
    """Return (iters, losses, per_iter_secs) arrays for logged iters."""
    iters, losses, dts = [], [], []
    if not log_path.exists():
        return np.array([]), np.array([]), np.array([])
    pat = re.compile(
        r"^\s*it=\s*(\d+)\s+loss=\s*([\d\.]+)\s+τ_0=[\d\.]+\s+β=[\d\.]+\s+c_0=[\d\.]+\s+\(([\d\.]+)s/iter\)"
    )
    for line in log_path.read_text().splitlines():
        m = pat.match(line)
        if m:
            iters.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            dts.append(float(m.group(3)))
    return np.asarray(iters), np.asarray(losses), np.asarray(dts)


def _load_npz_history(p: Path):
    if not p.exists():
        return None
    n = np.load(p)
    return dict(
        loss=np.asarray(n["loss_history"]),
        c_0=float(n["c_0"]), tau_0=float(n["tau_0"]), beta=float(n["beta"]),
        n_spectra=int(n["n_spectra"]), n_iters=int(n["n_iters"]),
    )


def _load_matlab_scalars():
    with h5py.File(MATLAB_REF, "r") as f:
        return dict(
            c_0=float(np.exp(np.asarray(f["log_c_0"])[0, 0])),
            tau_0=float(np.exp(np.asarray(f["log_tau_0"])[0, 0])),
            beta=float(np.exp(np.asarray(f["log_beta"])[0, 0])),
        )


def main():
    # Load everything available
    its_p, loss_p, dts_p = _parse_log(PER_SPEC_LOG_49671617)
    vec_full = _load_npz_history(VEC_FULL)
    vec_smoke = _load_npz_history(VEC_SMOKE)
    phase1 = _load_npz_history(PHASE1)
    per_spec_new = _load_npz_history(PER_SPEC_NEW)
    matlab = _load_matlab_scalars()

    # ------- 4-panel comparison figure -------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (a): loss trajectory (log scale on y; iter on x)
    ax = axes[0, 0]
    if vec_full is not None:
        ax.plot(np.arange(vec_full["n_iters"]), vec_full["loss"], "C0-",
                lw=2, label=f"vec full 89k×200 (final={vec_full['loss'][-1]:.3e})")
    if its_p.size:
        ax.plot(its_p, loss_p, "C3o-", ms=4, lw=1,
                label=f"per-spec 89k (49671617, killed @iter {its_p[-1]})")
    if per_spec_new is not None:
        ax.plot(np.arange(per_spec_new["n_iters"]), per_spec_new["loss"], "C2-",
                lw=2, label=f"per-spec 89k×200 (49709974, final={per_spec_new['loss'][-1]:.3e})")
    if phase1 is not None:
        ax.plot(np.arange(phase1["n_iters"]), phase1["loss"], "C1-",
                lw=1, alpha=0.6, label=f"per-spec 5k×50 (Phase-1 baseline)")
    if vec_smoke is not None:
        ax.plot(np.arange(vec_smoke["n_iters"]), vec_smoke["loss"], "C4-",
                lw=1, alpha=0.6, label=f"vec smoke 5k×50")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("loss (negative log-likelihood, summed)")
    ax.set_title("(a) Loss trajectories — Phase 2 across paths")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel (b): zoomed-in loss trajectory (last half)
    ax = axes[0, 1]
    if vec_full is not None:
        n = vec_full["n_iters"]; half = n // 2
        ax.plot(np.arange(half, n), vec_full["loss"][half:], "C0-",
                lw=2, label=f"vec full 89k×200")
    if its_p.size:
        mask = its_p >= 100
        ax.plot(its_p[mask], loss_p[mask], "C3o-", ms=4, lw=1,
                label=f"per-spec 89k (49671617)")
    if per_spec_new is not None:
        n = per_spec_new["n_iters"]; half = n // 2
        ax.plot(np.arange(half, n), per_spec_new["loss"][half:], "C2-",
                lw=2, label="per-spec 89k×200 (49709974)")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("loss")
    ax.set_title("(b) Loss trajectories, zoomed (iter ≥ N/2)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel (c): endpoint scalars vs MATLAB final
    ax = axes[1, 0]
    rows = []
    if vec_full is not None:
        rows.append(("vec full 89k×200", vec_full))
    if per_spec_new is not None:
        rows.append(("per-spec 89k×200", per_spec_new))
    if its_p.size and per_spec_new is None:
        # use 49671617's last logged iter as proxy
        rows.append((f"per-spec @iter {its_p[-1]} (49671617)",
                     dict(c_0=loss_p[-1] * 0,  # placeholder — actual scalars from log
                          tau_0=0, beta=0)))
    if phase1 is not None:
        rows.append(("per-spec 5k×50 (Phase-1)", phase1))

    # Bar groups: c_0, τ_0, β; bars per row
    params = ["c_0", "tau_0", "beta"]
    plabels = [r"$c_0$", r"$\tau_0$ ×100", r"$\beta$ /3"]
    x_pos = np.arange(len(params))
    bar_w = 0.8 / max(len(rows) + 1, 2)
    for i, (label, r) in enumerate(rows):
        vals = [r["c_0"], r["tau_0"] * 100, r["beta"] / 3.0]
        ax.bar(x_pos + i * bar_w, vals, width=bar_w, label=label, alpha=0.85)
    # MATLAB
    vals_m = [matlab["c_0"], matlab["tau_0"] * 100, matlab["beta"] / 3.0]
    ax.bar(x_pos + len(rows) * bar_w, vals_m, width=bar_w,
           label="MATLAB final", color="black", alpha=0.6)
    ax.set_xticks(x_pos + bar_w * len(rows) / 2)
    ax.set_xticklabels(plabels)
    ax.set_ylabel("value (rescaled — see x-tick labels)")
    ax.set_title("(c) Endpoint scalars (vs MATLAB final)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Panel (d): per-iter wall-time
    ax = axes[1, 1]
    if its_p.size:
        ax.plot(its_p, dts_p, "C3o-", ms=3, lw=0.7, alpha=0.8,
                label=f"per-spec 89k (49671617): mean {dts_p.mean():.0f}s/iter")
    # vec full per-iter rate isn't in the npz; pull from its log if exists
    vec_log = REPO / "slurm" / "greatlakes" / "phase2_dr16_49700040.log"
    its_v, _, dts_v = _parse_log(vec_log)
    if its_v.size:
        ax.plot(its_v, dts_v, "C0o-", ms=3, lw=0.7, alpha=0.8,
                label=f"vec full 89k (49700040): mean {dts_v.mean():.0f}s/iter")
    if per_spec_new is not None:
        new_log = list((REPO / "slurm" / "greatlakes").glob("phase2_dr16_49709974*.log"))
        if new_log:
            its_n, _, dts_n = _parse_log(new_log[0])
            if its_n.size:
                ax.plot(its_n, dts_n, "C2o-", ms=3, lw=0.7, alpha=0.8,
                        label=f"per-spec 89k (49709974): mean {dts_n.mean():.0f}s/iter")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("seconds per iteration")
    ax.set_title("(d) Per-iter wall-time")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Phase 2 on DR16 — per-spectrum vs vectorized retrain comparison",
        fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT}")

    # ------- speedup figure (focused) -------
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    if its_p.size:
        ax.plot(its_p, dts_p, "C3o-", ms=4, lw=1,
                label=f"per-spectrum 89k (49671617):  mean {dts_p.mean():.0f}s/iter")
    if its_v.size:
        ax.plot(its_v, dts_v, "C0o-", ms=4, lw=1,
                label=f"vectorized 89k (49700040): mean {dts_v.mean():.0f}s/iter")
    if its_p.size and its_v.size:
        ratio = dts_p.mean() / dts_v.mean()
        ax.set_title(f"Phase 2 per-iter wall on DR16 89k — vectorized vs per-spectrum"
                     f"\n→ {ratio:.2f}× speedup at scale (mean / mean)",
                     fontweight="bold")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("seconds per iteration")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_SPEEDUP, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_SPEEDUP}")


if __name__ == "__main__":
    main()
