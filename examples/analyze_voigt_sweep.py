"""Analyze the Voigt LSF + num_lines sweep results.

Reads master.csv from `examples/voigt_lsf_sweep.py`, slices by
(mock × nhi_regime × config), computes ΔlogNHI / Δz statistics,
writes a markdown report with figures.

Usage::

    python examples/analyze_voigt_sweep.py \\
        --master out/voigt_sweep/runs/master.csv \\
        --out-dir docs/notes/voigt_sweep_<date>/

Produces:
    docs/notes/voigt_sweep_<date>/
        report.md
        delta_log_nhi_by_config_per_mock.png
        delta_log_nhi_by_regime.png
        per_target_scatter.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np


def _read_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "median": float("nan"), "p16": float("nan"),
                "p84": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--master", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    rows = _read_csv(args.master)
    if not rows:
        raise RuntimeError(f"no rows in {args.master}")

    # Coerce numeric columns.
    for r in rows:
        for k in ("truth_log_nhi", "map_log_nhi", "delta_log_nhi",
                  "truth_z_dla", "map_z_dla", "delta_z_dla", "p_dla", "wall_s"):
            if k in r and r[k] != "":
                r[k] = float(r[k])
        r["num_lines"] = int(r.get("num_lines", 3))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Per-(mock, regime, config) summary -----
    grouped = defaultdict(list)
    for r in rows:
        if isinstance(r.get("delta_log_nhi"), float):
            grouped[(r["mock"], r["nhi_regime"], r["config_tag"])].append(r["delta_log_nhi"])

    md = ["# Voigt LSF + num_lines sweep — analysis\n"]
    md.append(f"Source: `{args.master}` ({len(rows)} inferences)\n")
    md.append("## Configurations\n")
    md.append("| tag | kernel | num_lines |")
    md.append("|---|---|---:|")
    cfg_seen = sorted({r["config_tag"] for r in rows})
    for tag in cfg_seen:
        sample = next(r for r in rows if r["config_tag"] == tag)
        md.append(f"| **{tag}** | `{sample['kernel']}` | {sample['num_lines']} |")
    md.append("")

    # ----- Headline table: median ΔlogNHI per (mock, regime, config) -----
    md.append("## Median ΔlogNHI = MAP − truth, per (mock × regime × config)\n")
    mocks_seen = sorted({r["mock"] for r in rows})
    regimes_seen = sorted({r["nhi_regime"] for r in rows},
                          key=lambda x: {"LLS": 0, "sub-DLA": 1, "DLA": 2}.get(x, 99))

    for mock in mocks_seen:
        md.append(f"### Mock: `{mock}`\n")
        md.append("| regime | " + " | ".join(f"{tag}: ΔlogNHI [p16, p84]" for tag in cfg_seen) + " |")
        md.append("|---|" + "---|" * len(cfg_seen))
        for regime in regimes_seen:
            line = f"| {regime}"
            for tag in cfg_seen:
                vals = grouped.get((mock, regime, tag), [])
                if not vals:
                    line += " | —"
                else:
                    s = _summary_stats(vals)
                    line += f" | {s['median']:+.3f} [{s['p16']:+.3f}, {s['p84']:+.3f}] (n={s['n']})"
            line += " |"
            md.append(line)
        md.append("")

    # ----- Plots -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        md.append("(matplotlib not available; figures skipped)")
        plt = None

    if plt is not None:
        # 1) Box plot of ΔlogNHI by config, per mock × regime
        for mock in mocks_seen:
            fig, axes = plt.subplots(1, len(regimes_seen),
                                       figsize=(4 * len(regimes_seen), 5),
                                       sharey=True)
            if len(regimes_seen) == 1:
                axes = [axes]
            for ax, regime in zip(axes, regimes_seen):
                data = [grouped.get((mock, regime, tag), []) for tag in cfg_seen]
                ax.boxplot(data, tick_labels=cfg_seen, showmeans=True)
                ax.axhline(0, ls="--", color="grey", alpha=0.6)
                ax.set_title(f"{mock} / {regime}")
                ax.set_ylabel(r"$\Delta\log N_{\rm HI}$ (MAP − truth)")
                ax.grid(alpha=0.3)
            plt.suptitle(f"Mock: {mock}")
            plt.tight_layout()
            fig_path = args.out_dir / f"delta_log_nhi_box_{mock}.png"
            fig.savefig(fig_path, dpi=140)
            plt.close(fig)
            md.append(f"\n![Δ log NHI box plot for {mock}](./{fig_path.name})\n")

        # 2) Per-target scatter: ΔlogNHI vs truth NHI, coloured by config.
        fig, axes = plt.subplots(1, len(mocks_seen),
                                   figsize=(5 * len(mocks_seen), 5), sharey=True)
        if len(mocks_seen) == 1:
            axes = [axes]
        colours = {"A": "C0", "B": "C1", "C": "C2", "D": "C3"}
        for ax, mock in zip(axes, mocks_seen):
            for tag in cfg_seen:
                xs, ys = [], []
                for r in rows:
                    if r["mock"] != mock or r["config_tag"] != tag:
                        continue
                    if not isinstance(r.get("delta_log_nhi"), float):
                        continue
                    xs.append(r["truth_log_nhi"])
                    ys.append(r["delta_log_nhi"])
                ax.scatter(xs, ys, label=tag, alpha=0.6,
                           color=colours.get(tag, None), s=40)
            ax.axhline(0, ls="--", color="grey", alpha=0.6)
            ax.set_title(mock)
            ax.set_xlabel(r"truth $\log N_{\rm HI}$")
            ax.set_ylabel(r"$\Delta \log N_{\rm HI}$")
            ax.legend()
            ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = args.out_dir / "per_target_scatter.png"
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
        md.append(f"\n![Per-target ΔlogNHI scatter](./{fig_path.name})\n")

    # ----- Interpretation cheat sheet -----
    md.append("## Reading these results\n")
    md.append(
        "- **Config A** is production. Its bias is the baseline. Y3 mocks have "
        "shown +0.37 dex on the canonical regression target (TID 120046865)."
    )
    md.append(
        "- **Config B** isolates the LSF effect. If B's bias < A's, the BOSS-"
        "shaped kernel on a DESI grid is contributing."
    )
    md.append(
        "- **Config C** adds higher-order Lyman lines on top of the DESI "
        "kernel. If C's bias < B's, the production num_lines=3 is also "
        "contributing."
    )
    md.append(
        "- **Config D** is bare Voigt (no LSF). If D's bias matches B's, the "
        "LSF wasn't the dominant effect; mock-physics or QMC are."
    )
    md.append("")
    md.append("## Per-mock comparison")
    md.append(
        "- Differences across mocks reveal **mock-physics** effects, not "
        "inference effects. London is known to use approximate Lyman series "
        "scaling (rescale by oscillator strength rather than per-line Voigt) "
        "— if config A on London disagrees with config A on 2LPT/Saclay, that "
        "could be a mock generator artefact rather than the GP's fault."
    )
    md.append("")
    md.append("## Per-NHI-regime comparison")
    md.append(
        "- **DLA regime (logNHI ≥ 20.3)**: damping-wing-dominated. LSF "
        "matters most here — if the model trough is too narrow, the fitter "
        "compensates with high NHI."
    )
    md.append(
        "- **sub-DLA / LLS regimes**: Doppler-core-dominated. LSF effect "
        "is smaller; bias here is more likely from prior-edge or "
        "QMC-density effects."
    )

    (args.out_dir / "report.md").write_text("\n".join(md))
    print(f"[analyze] wrote {args.out_dir}/report.md")


if __name__ == "__main__":
    main()
