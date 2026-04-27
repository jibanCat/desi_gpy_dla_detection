"""Scan over P_DLA cut + BAL exclusion on a production catalog,
reporting purity / completeness vs full truth at each operating point.

The user's recollection is that historic GP-DLA purity ≈ 0.78 and
completeness ≈ 0.8 on London mocks excluding BAL LOS. This script
reproduces that operating-point sweep so we can find which P_DLA cut
matches the historical numbers, before drawing any conclusions about
whether the production catalog has degraded.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import fitsio
import numpy as np
from astropy.table import Table, vstack


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog-dir", required=True)
    p.add_argument("--truth-dla", required=True,
                   help="dla_cat_mask_20.30.fits (DLA truth, NHI≥20.3)")
    p.add_argument("--truth-full", default=None,
                   help="(optional) full HCD truth (dla_cat.fits) for "
                        "recovery-rate-against-anything-real metric")
    p.add_argument("--bal-cat", default=None,
                   help="bal_cat.fits — exclude TARGETIDs with BI_CIV>0 "
                        "if --no-bal is set")
    p.add_argument("--no-bal", action="store_true",
                   help="Drop MAP DLAs whose TARGETID has BI_CIV>0")
    p.add_argument("--lls-dir", default=None)
    p.add_argument("--p-cuts", default="0.5,0.9,0.99,0.999")
    p.add_argument("--dz-rel", type=float, default=0.01)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _load_dir(d):
    parts = []
    for f in sorted(glob.glob(os.path.join(d, "dlacat-*.fits"))):
        parts.append(Table(fitsio.read(f, ext=1, columns=[
            "TARGETID", "Z_DLA", "NHI", "P_DLA"
        ])))
    return vstack(parts)


def _match(map_cat, truth, dz_rel):
    """For each MAP DLA, mark whether it matches any truth row at the
    same TARGETID with |Δz|/(1+z_truth) ≤ dz_rel."""
    matched = np.zeros(len(map_cat), dtype=bool)
    by_tid = {}
    t_z = np.asarray(truth["Z_DLA"] if "Z_DLA" in truth.colnames
                     else truth["Z"], dtype=float)
    t_tid = np.asarray(truth["TARGETID"])
    for i, t in enumerate(t_tid):
        by_tid.setdefault(int(t), []).append(t_z[i])

    m_tid = np.asarray(map_cat["TARGETID"])
    m_z = np.asarray(map_cat["Z_DLA"], dtype=float)
    for j, (mt, mz) in enumerate(zip(m_tid, m_z)):
        for tz in by_tid.get(int(mt), []):
            if abs(mz - tz) / (1 + tz) <= dz_rel:
                matched[j] = True
                break
    return matched


def _truth_match(truth, map_cat, dz_rel):
    """For each truth DLA, mark whether ≥1 MAP DLA at same TARGETID
    matches within dz_rel."""
    matched = np.zeros(len(truth), dtype=bool)
    by_tid = {}
    m_tid = np.asarray(map_cat["TARGETID"])
    m_z = np.asarray(map_cat["Z_DLA"], dtype=float)
    for j, mt in enumerate(m_tid):
        by_tid.setdefault(int(mt), []).append(m_z[j])

    t_z = np.asarray(truth["Z_DLA"], dtype=float)
    t_tid = np.asarray(truth["TARGETID"])
    for i, (tt, tz) in enumerate(zip(t_tid, t_z)):
        for mz in by_tid.get(int(tt), []):
            if abs(mz - tz) / (1 + tz) <= dz_rel:
                matched[i] = True
                break
    return matched


def main():
    args = parse_args()
    p_cuts = [float(x) for x in args.p_cuts.split(",")]

    print("[load] truth DLA", flush=True)
    truth = Table(fitsio.read(args.truth_dla))
    print(f"  {len(truth)} truth DLAs (NHI>=20.3)")

    print("[load] catalog", flush=True)
    cat = _load_dir(args.catalog_dir)
    print(f"  {len(cat)} MAP DLAs total", flush=True)

    # Restrict truth denominator to processed TIDs
    cat_tids = set(int(t) for t in np.asarray(cat["TARGETID"]))
    in_proc = np.array([int(t) in cat_tids for t in np.asarray(truth["TARGETID"])])
    truth_proc = truth[in_proc]
    print(f"  truth on processed TIDs: {len(truth_proc)}")

    # BAL handling
    bal_tids = set()
    if args.bal_cat:
        bal = fitsio.read(args.bal_cat, ext=1, columns=["TARGETID", "BI_CIV"])
        bal_tids = set(int(r["TARGETID"]) for r in bal if r["BI_CIV"] > 0)
        print(f"  {len(bal_tids)} BAL TIDs (BI_CIV>0)")

    if args.no_bal:
        cat_kept_mask = ~np.isin(np.asarray(cat["TARGETID"]),
                                  list(bal_tids))
        truth_kept_mask = ~np.isin(np.asarray(truth_proc["TARGETID"]),
                                    list(bal_tids))
        cat = cat[cat_kept_mask]
        truth_proc = truth_proc[truth_kept_mask]
        print(f"  after BAL exclusion: {len(cat)} MAP, {len(truth_proc)} truth")

    truth_full = None
    if args.truth_full:
        truth_full = Table(fitsio.read(args.truth_full,
                                        columns=["TARGETID", "Z_DLA", "NHI"]))
        if args.no_bal:
            truth_full = truth_full[
                ~np.isin(np.asarray(truth_full["TARGETID"]), list(bal_tids))
            ]
        print(f"  full truth (DLA+subDLA+LLS): {len(truth_full)}")

    rows = []
    for pc in p_cuts:
        keep = np.asarray(cat["P_DLA"]) >= pc
        cat_pc = cat[keep]
        if len(cat_pc) == 0:
            rows.append(dict(p_cut=pc, n_map=0,
                             completeness=0.0, strict_purity=0.0,
                             recovery_rate=0.0))
            continue

        # Strict purity: matches DLA truth (NHI≥20.3) by (TARGETID, dz)
        m_strict = _match(cat_pc, truth_proc, args.dz_rel)
        n_match_strict = int(m_strict.sum())
        strict_purity = n_match_strict / len(cat_pc)

        # Completeness: truth DLAs with at least one MAP match
        m_truth = _truth_match(truth_proc, cat_pc, args.dz_rel)
        compl = m_truth.sum() / len(truth_proc) if len(truth_proc) else 0.0

        rec_rate = None
        if truth_full is not None:
            m_any = _match(cat_pc, truth_full, args.dz_rel)
            rec_rate = int(m_any.sum()) / len(cat_pc)

        rows.append(dict(
            p_cut=pc, n_map=len(cat_pc),
            n_match_strict=n_match_strict,
            completeness=float(compl),
            strict_purity=float(strict_purity),
            recovery_rate=float(rec_rate) if rec_rate is not None else None,
            n_truth_proc=int(len(truth_proc)),
            n_truth_matched=int(m_truth.sum()),
        ))
        rec_str = f", recovery={rec_rate:.3f}" if rec_rate is not None else ""
        print(f"  P_DLA>={pc}: n={len(cat_pc)}, completeness={compl:.3f}, "
              f"strict_purity={strict_purity:.3f}{rec_str}")

    # Markdown
    title = "P_DLA cut sweep"
    if args.no_bal: title += " (excluding BAL LOS)"
    lines = [
        f"# {title}",
        "",
        f"- Catalog dir: `{args.catalog_dir}`",
        f"- Truth DLA: `{args.truth_dla}` (NHI ≥ 20.3, "
        f"{len(truth_proc) if not args.no_bal else len(truth_proc) + 0} on processed TIDs)",
        f"- BAL excluded: {args.no_bal}",
        f"- Match: |Δz|/(1+z_truth) ≤ {args.dz_rel}",
        "",
        "| P_DLA cut | N MAP | N matched | completeness | strict purity | recovery rate (anything real) |",
        "|:--------:|------:|---------:|:-----------:|:-------------:|:-----------------------------:|",
    ]
    for r in rows:
        rec = (f"{r['recovery_rate']:.3f}" if r["recovery_rate"] is not None
               else "—")
        lines.append(
            f"| ≥ {r['p_cut']:.3f} | {r['n_map']:,} | "
            f"{r['n_match_strict']:,} | {r['completeness']:.3f} | "
            f"{r['strict_purity']:.3f} | {rec} |"
        )

    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
