"""Aggregate the FILTER × num_dla_samples sweep into one comparison table.

Reads each per-condition summary.tsv produced by finalize_smoke_batch.py,
computes detection rate, NHI bias median/std, and fraction of spurious
multi-DLA selections per condition. Writes a Markdown table that's
copy-pasteable into a notes file.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics as st
from typing import Iterable

import numpy as np


def load_summary(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def stats(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    n_total = len(rows)
    detected = [r for r in rows if float(r["p_dla"]) > 0.5]
    n_det = len(detected)
    multi = [r for r in detected if int(r["selected_dlas"]) >= 2]
    clean = [r for r in detected if int(r["selected_dlas"]) == 1]
    # Use only "clean" 1-DLA fits for unbiased NHI bias measurement
    if clean:
        dn_clean = [float(r["dlogNHI"]) for r in clean if r["dlogNHI"] != "-"]
        median_clean = float(np.median(dn_clean)) if dn_clean else float("nan")
        std_clean = float(np.std(dn_clean)) if dn_clean else float("nan")
    else:
        median_clean = std_clean = float("nan")
    # All detected (including multi-DLA: take the MAP closest to truth z)
    return {
        "N_total": n_total,
        "N_detected": n_det,
        "N_multi": len(multi),
        "N_clean": len(clean),
        "median_dlogNHI_clean": median_clean,
        "std_dlogNHI_clean": std_clean,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="out/smoke/batch",
                   help="root containing per-condition <preset>_filter<F>_n<N> dirs")
    p.add_argument("--preset", default="eboss",
                   help="preset prefix to compare conditions for")
    p.add_argument("--out", default="docs/notes/2026-04-25_filter_samples_sweep.md")
    return p.parse_args()


def main():
    args = parse_args()
    pattern = os.path.join(args.root, f"{args.preset}_filter*_n*", "summary.tsv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f"no summary.tsv found under {pattern}")

    print(f"[sweep] aggregating {len(matches)} conditions for preset={args.preset}")

    table_rows = []
    for path in matches:
        m = re.search(rf"{args.preset}_filter(\d)_n(\d+)", path)
        if not m:
            continue
        f, n = int(m.group(1)), int(m.group(2))
        rows = load_summary(path)
        s = stats(rows)
        table_rows.append((f, n, s))
        print(f"  filter={f}  N={n:>6}  detected={s['N_detected']}/{s['N_total']}"
              f"  multi-DLA={s['N_multi']}  bias(clean1)="
              f"{s['median_dlogNHI_clean']:+.3f}±{s['std_dlogNHI_clean']:.3f}")

    md_lines = [
        "# FILTER × num_dla_samples sweep — eBOSS multi-DLA",
        "",
        "20 high-SNR strong-DLA truth targets (10 Saclay juraLy8-124 + 10 2LPT loa-124).",
        "Multi-DLA mode (`--single-absorber-model 0 --max-dlas 4`),",
        "`MAX_WORKERS=8`, `BATCH_SIZE=1250`. eBOSS DR16Q model "
        "(`dlambda=0.25`, `k=20`, `prev_tau_0=0.0023`, `prev_beta=3.65`).",
        "",
        "Bias is computed only on the *clean* 1-DLA detections (i.e. the model "
        "selected exactly k=1, matching the truth). Multi-DLA selections (k≥2) "
        "are reported separately as a purity-loss diagnostic.",
        "",
        "| FILTER | N_DLA samples | detected/total | multi-DLA selections | clean 1-DLA fits | median ΔlogN_HI (clean) | σ ΔlogN_HI (clean) |",
        "|:------:|--------------:|:--------------:|:--------------------:|:----------------:|:----------------------:|:------------------:|",
    ]
    for f, n, s in sorted(table_rows):
        md_lines.append(
            f"| {f} | {n:,} | {s['N_detected']}/{s['N_total']} | {s['N_multi']} | "
            f"{s['N_clean']} | {s['median_dlogNHI_clean']:+.3f} | {s['std_dlogNHI_clean']:.3f} |"
        )

    md_lines += [
        "",
        "## Reading the table",
        "",
        "- **detected/total** answers the user's *completeness* question: did the "
        "  model find a DLA when truth says one is there?",
        "- **multi-DLA selections** answers the *purity* question: did the model "
        "  identify only one DLA (truth)? Or did it spuriously add a second one "
        "  fitting an LLS or strong forest absorption? Lower is better for purity.",
        "- **median ΔlogN_HI on clean 1-DLA fits** answers the *NHI bias* question. "
        "  Restricting to k=1 matches the truth setup so the bias is not "
        "  confounded by the 2-DLA splitting effect.",
        "",
        "## Recommendation",
        "",
        "(filled in once the sweep finishes)",
        "",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
