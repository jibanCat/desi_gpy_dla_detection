"""Per-truth-DLA matching + Lyβ misdetection analysis.

For a smoke batch directory:

  1. For each TARGETID, read the saved .pkl (MAP_z_dlas, MAP_log_nhis).
  2. Match each MAP DLA to the nearest truth DLA in the LOS-truth list
     (within Δz_max). Truth DLAs with no match are "missed". MAP DLAs
     with no truth match are "spurious".
  3. For each spurious MAP DLA, check whether its z corresponds to the
     Lyβ-shifted z of any truth DLA on the same LOS:

         z_lyb_apparent = (λ_Lyβ / λ_Lyα) · (1 + z_truth) − 1
                        = (1025.7 / 1215.7) · (1 + z_truth) − 1
                        ≈ 0.8437 · (1 + z_truth) − 1

  4. Produce:
       - completeness as a function of truth log NHI bin
       - purity (fraction of MAP DLAs with a truth match)
       - fraction of spurious detections explainable as Lyβ
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle

import numpy as np


LYA = 1215.67
LYB = 1025.7228


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-dir", required=True)
    p.add_argument("--targets", default="out/smoke/targets100.tsv")
    p.add_argument("--dz-match", type=float, default=0.005,
                   help="max |Δz| for a truth↔MAP DLA match (default 0.005)")
    p.add_argument("--dz-lybeta", type=float, default=0.005,
                   help="max |Δz| for spurious-DLA ↔ Lyβ-of-truth match")
    p.add_argument("--report-md", default=None,
                   help="Output a Markdown report to this path")
    return p.parse_args()


def lyb_apparent_z(z_truth: float) -> float:
    """z at which a Lyβ line from a DLA at z_truth lies if mistaken for Lyα."""
    return (LYB / LYA) * (1.0 + z_truth) - 1.0


def main():
    args = parse_args()

    with open(args.targets) as f:
        targets = list(csv.DictReader(f, delimiter="\t"))
    truth_index = {(r["mock"], r["target_id"]): r for r in targets}

    # Per-truth-DLA records
    truth_records = []   # one row per truth DLA: (mock, tid, z_t, n_t, matched, dlogN, dz)
    map_records = []     # one row per MAP DLA: (mock, tid, z_m, n_m, matched, lyb_match)

    for r in targets:
        mock, tid = r["mock"], r["target_id"]
        pkl = os.path.join(args.batch_dir, f"{mock}_{tid}.pkl")
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as f:
            res = pickle.load(f)

        # All truth absorbers on LOS
        all_z = [float(x) for x in r["all_truth_z"].split(",") if x]
        all_n = [float(x) for x in r["all_truth_nhi"].split(",") if x]
        truth_dla_idx = [i for i, n in enumerate(all_n) if n >= 20.3]
        truth_dla_z = [all_z[i] for i in truth_dla_idx]
        truth_dla_n = [all_n[i] for i in truth_dla_idx]

        # MAP DLAs (drop NaN slots)
        map_z = np.asarray(res["MAP_z_dlas"])[0]
        map_n = np.asarray(res["MAP_log_nhis"])[0]
        finite = np.isfinite(map_z) & np.isfinite(map_n)
        map_z = map_z[finite].tolist()
        map_n = map_n[finite].tolist()

        # Greedy nearest-neighbour matching by |Δz|
        used_map = set()
        for tz, tn in zip(truth_dla_z, truth_dla_n):
            if not map_z:
                truth_records.append((mock, tid, tz, tn, False, np.nan, np.nan))
                continue
            best_j, best_dz = None, np.inf
            for j, mz in enumerate(map_z):
                if j in used_map: continue
                dz = abs(mz - tz)
                if dz < best_dz:
                    best_dz, best_j = dz, j
            if best_j is not None and best_dz <= args.dz_match:
                used_map.add(best_j)
                truth_records.append((mock, tid, tz, tn, True,
                                       map_n[best_j] - tn, map_z[best_j] - tz))
            else:
                truth_records.append((mock, tid, tz, tn, False, np.nan, np.nan))

        # Spurious MAP DLAs: those not in used_map
        for j, (mz, mn) in enumerate(zip(map_z, map_n)):
            matched = j in used_map
            lyb_match = False
            if not matched:
                # check if this z matches Lyβ of any truth absorber on LOS
                for tz_any in all_z:
                    if abs(lyb_apparent_z(tz_any) - mz) <= args.dz_lybeta:
                        lyb_match = True
                        break
            map_records.append((mock, tid, mz, mn, matched, lyb_match))

    # ---- aggregate ----
    nhi_bins = [(20.3, 20.6), (20.6, 21.0), (21.0, 21.5), (21.5, 23.5)]
    lines = []
    lines.append("# Purity / completeness analysis")
    lines.append("")
    lines.append(f"Batch: `{args.batch_dir}`")
    lines.append(f"Targets file: `{args.targets}`")
    lines.append(f"Match thresholds: Δz_truth-match = {args.dz_match}, "
                 f"Δz_lybeta = {args.dz_lybeta}")
    lines.append("")

    # Completeness per NHI bin
    lines.append("## Completeness — fraction of truth DLAs matched")
    lines.append("")
    lines.append("| log N_HI bin | total truth DLAs | matched | completeness |")
    lines.append("|:------------:|----------------:|---------:|:-------------:|")
    for lo, hi in nhi_bins:
        sel = [r for r in truth_records if lo <= r[3] < hi]
        n_total = len(sel)
        n_matched = sum(1 for r in sel if r[4])
        rate = n_matched / n_total if n_total else 0.0
        lines.append(f"| [{lo}, {hi}) | {n_total} | {n_matched} | {rate:.1%} |")
    n_all = len(truth_records)
    n_match_all = sum(1 for r in truth_records if r[4])
    lines.append(f"| **all** | **{n_all}** | **{n_match_all}** | "
                 f"**{(n_match_all/n_all if n_all else 0):.1%}** |")
    lines.append("")

    # NHI bias on matched truth DLAs
    matched = [r for r in truth_records if r[4]]
    if matched:
        dn = np.array([r[5] for r in matched])
        dz = np.array([r[6] for r in matched])
        lines.append("## ΔlogN_HI on matched DLAs")
        lines.append("")
        lines.append(f"- N matched = {len(matched)}")
        lines.append(f"- median ΔlogN_HI = **{np.nanmedian(dn):+.3f}**")
        lines.append(f"- σ ΔlogN_HI = {np.nanstd(dn):.3f}")
        lines.append(f"- median Δz = {np.nanmedian(dz):+.5f}")
        lines.append(f"- σ Δz = {np.nanstd(dz):.5f}")
        lines.append("")
        lines.append("| log N_HI bin | N matched | median ΔlogN_HI | σ |")
        lines.append("|:------------:|---------:|:---------------:|:---:|")
        for lo, hi in nhi_bins:
            sub = [r for r in matched if lo <= r[3] < hi]
            if sub:
                d = np.array([r[5] for r in sub])
                lines.append(f"| [{lo}, {hi}) | {len(sub)} | "
                             f"{np.nanmedian(d):+.3f} | {np.nanstd(d):.3f} |")
        lines.append("")

    # Purity / spurious / Lyβ
    n_map_total = len(map_records)
    n_map_matched = sum(1 for r in map_records if r[4])
    spurious = [r for r in map_records if not r[4]]
    n_lyb = sum(1 for r in spurious if r[5])
    purity = n_map_matched / n_map_total if n_map_total else 0.0
    lines.append("## Purity — what fraction of MAP DLAs match a truth DLA?")
    lines.append("")
    lines.append(f"- N MAP DLAs total = {n_map_total}")
    lines.append(f"- N matched to truth = {n_map_matched}  (purity = **{purity:.1%}**)")
    lines.append(f"- N spurious (no truth match) = {len(spurious)}")
    lines.append(f"  - of which **{n_lyb} ({n_lyb/len(spurious):.1%} of spurious)** "
                 "match the Lyβ-shifted z of a truth DLA on the same LOS.")
    lines.append(f"  - This suggests that the model misidentifies Lyβ"
                 " absorption from a real DLA as an additional Lyα DLA.")
    lines.append("")

    # Spurious-NHI distribution
    if spurious:
        sp_nhi = np.array([r[3] for r in spurious])
        lines.append(f"- Spurious MAP NHI distribution: median={np.median(sp_nhi):.2f}, "
                     f"min={sp_nhi.min():.2f}, max={sp_nhi.max():.2f}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.report_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_md)),
                    exist_ok=True)
        with open(args.report_md, "w") as f:
            f.write(text + "\n")
        print(f"\n[saved] {args.report_md}")


if __name__ == "__main__":
    main()
