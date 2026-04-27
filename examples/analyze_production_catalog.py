"""Production-scale purity/completeness analysis for a London-mock GP-DLA run.

Reads:
- A directory of `dlacat-*.fits` chunks produced by the production
  pipeline (DLA-mode multi-DLA OR LLS-mode single-absorber).
- The London truth catalog `dla_cat_mask_20.30.fits` from the mock
  directory.
- Optionally, the LLS-mode catalog directory (used for the LLS
  cross-reference post-processing).

Computes:
- TRUTH-DLA matching by (TARGETID, |Δz|/(1+z) < dz_match_rel) — same
  metric as the existing molly notebook (Δv < 3000 km/s).
- Completeness = (N matched truth DLAs) / (N truth DLAs). Per NHI bin.
- Purity = (N matched MAP DLAs) / (N MAP DLAs). Per p(DLA) cut.
- Lyβ-veto post-processing: re-run completeness/purity AFTER applying
  `gpy_dla_detection.postprocess.lyb_veto.flag_lybeta` to the catalog.
- LLS cross-reference: re-run AFTER applying `lls_cross_reference`.

Output:
- A markdown report with before/after tables.
- Optional FITS dump of the post-processed catalog.

Usage:
  python examples/analyze_production_catalog.py \
      --catalog-dir /nfs/turbo/.../desi-mock-gpdla-...-filter \
      --truth /nfs/turbo/.../jura-124/dla_cat_mask_20.30.fits \
      --zcat  /nfs/turbo/.../jura-124/zcat.fits \
      --lls-dir /nfs/turbo/.../desi-mock-gpdla-...-lls_run-nhi172 \
      --bal-cat /nfs/turbo/.../jura-124/bal_cat.fits \
      --out docs/notes/2026-04-27_london_production_pc.md
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import fitsio
import numpy as np
from astropy.table import Table, vstack


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog-dir", required=True,
                   help="dir of dlacat-*.fits (multi-DLA mode catalog)")
    p.add_argument("--truth", required=True,
                   help="dla_cat_mask_20.30.fits (truth DLAs with NHI>=20.3)")
    p.add_argument("--zcat", required=True,
                   help="zcat.fits with TARGETID, Z, ZWARN")
    p.add_argument("--lls-dir", default=None,
                   help="(optional) dir of LLS-mode dlacat-*.fits")
    p.add_argument("--bal-cat", default=None,
                   help="(optional) bal_cat.fits — used to flag BAL targets")
    p.add_argument("--no-bal", action="store_true",
                   help="Exclude TARGETIDs with BI_CIV>0 from BOTH the catalog "
                        "and the truth denominator. Matches the molly-notebook "
                        "convention for production purity/completeness reports.")
    p.add_argument("--dz-match-rel", type=float, default=0.01,
                   help="|Δz|/(1+z_truth) tolerance for truth matching")
    p.add_argument("--p-dla-cut", type=float, default=0.5,
                   help="p(DLA) threshold for considering a MAP DLA detected")
    p.add_argument("--out", required=True, help="markdown report path")
    return p.parse_args()


def _load_catalog_dir(d: str) -> Table:
    files = sorted(glob.glob(os.path.join(d, "dlacat-*.fits")))
    print(f"[load] {d}: {len(files)} chunks", flush=True)
    tbls = []
    for f in files:
        try:
            tbls.append(Table(fitsio.read(f, ext=1)))
        except Exception as e:
            print(f"  [warn] {os.path.basename(f)}: {e}")
    if not tbls:
        return Table()
    return vstack(tbls)


def _per_nhi_bin_stats(matched: np.ndarray, nhis: np.ndarray) -> list[dict]:
    bins = [(20.3, 20.6), (20.6, 21.0), (21.0, 21.5), (21.5, 23.5)]
    rows = []
    for lo, hi in bins:
        m = (nhis >= lo) & (nhis < hi)
        n_total = int(m.sum())
        n_matched = int(matched[m].sum())
        rate = n_matched / n_total if n_total else 0.0
        rows.append(dict(bin=f"[{lo}, {hi})", total=n_total,
                         matched=n_matched, rate=rate))
    rows.append(dict(bin="all", total=int(matched.size),
                     matched=int(matched.sum()),
                     rate=matched.sum() / matched.size if matched.size else 0.0))
    return rows


def _match_truth_to_map(truth: Table, mp: Table, dz_rel: float
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy nearest-z matching by TARGETID. Returns:
    - matched_truth: bool array length len(truth) — strict 1-to-1
    - matched_map_strict: bool array length len(mp) — strict 1-to-1
    - matched_map_any: bool array length len(mp) — every MAP DLA that has
      ANY truth DLA at same TID within dz_rel (regardless of whether
      another MAP already claimed it). Used for the looser "matches a real
      DLA on LOS" purity metric that matches the molly-notebook style.
    """
    matched_truth = np.zeros(len(truth), dtype=bool)
    matched_map = np.zeros(len(mp), dtype=bool)
    matched_map_any = np.zeros(len(mp), dtype=bool)

    # Index MAP DLAs by TARGETID
    tid_arr = np.asarray(mp["TARGETID"])
    z_arr = np.asarray(mp["Z_DLA"], dtype=float)

    map_by_tid: dict[int, list[int]] = {}
    for i, t in enumerate(tid_arr):
        map_by_tid.setdefault(int(t), []).append(i)

    truth_tid = np.asarray(truth["TARGETID"])
    truth_z = np.asarray(truth["Z_DLA"], dtype=float)
    truth_nhi = np.asarray(truth["NHI"], dtype=float)

    # Greedy: per TARGETID, sort truth DLAs by descending NHI (priority to
    # strong DLAs), match each to nearest unused MAP within dz_rel.
    by_tid: dict[int, list[int]] = {}
    for i, t in enumerate(truth_tid):
        by_tid.setdefault(int(t), []).append(i)

    for tid, truth_idxs in by_tid.items():
        cand = map_by_tid.get(tid, [])
        if not cand:
            continue
        # Strict 1-to-1
        order = sorted(truth_idxs, key=lambda i: -truth_nhi[i])
        for ti in order:
            best, best_dz = None, np.inf
            for mi in cand:
                if matched_map[mi]: continue
                dz = abs(z_arr[mi] - truth_z[ti]) / (1 + truth_z[ti])
                if dz < best_dz:
                    best_dz, best = dz, mi
            if best is not None and best_dz <= dz_rel:
                matched_truth[ti] = True
                matched_map[best] = True
        # Loose: every MAP at this TID that's near *any* truth DLA on the LOS
        for mi in cand:
            for ti in truth_idxs:
                if abs(z_arr[mi] - truth_z[ti]) / (1 + truth_z[ti]) <= dz_rel:
                    matched_map_any[mi] = True
                    break

    return matched_truth, matched_map, matched_map_any


def _format_table(rows: list[dict], header: str) -> str:
    keys = list(rows[0].keys())
    out = ["", "## " + header, "",
           "| " + " | ".join(keys) + " |",
           "|" + "|".join([":-:"] * len(keys)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) and 0 <= v <= 1 else str(v)
            for v in r.values()) + " |")
    return "\n".join(out)


def main():
    args = parse_args()

    print("[load] truth", flush=True)
    truth = Table(fitsio.read(args.truth))
    print(f"  truth DLAs (NHI>=20.3): {len(truth)}", flush=True)

    cat = _load_catalog_dir(args.catalog_dir)
    print(f"[load] catalog rows: {len(cat)}", flush=True)
    if "Z_DLA" not in cat.colnames or "TARGETID" not in cat.colnames:
        raise SystemExit("catalog missing Z_DLA / TARGETID columns")

    # Apply p(DLA) cut to the catalog
    if "P_DLA" in cat.colnames:
        cat["_KEEP"] = cat["P_DLA"] >= args.p_dla_cut
    elif "MODEL_P" in cat.colnames:
        cat["_KEEP"] = cat["MODEL_P"] >= args.p_dla_cut
    else:
        cat["_KEEP"] = np.ones(len(cat), dtype=bool)
    cat_pass = cat[cat["_KEEP"]]
    print(f"  after p(DLA)>={args.p_dla_cut}: {len(cat_pass)} entries",
          flush=True)

    # Optional BAL flagging / exclusion
    bal_tids = set()
    if args.bal_cat:
        bal = fitsio.read(args.bal_cat, ext=1, columns=["TARGETID", "BI_CIV"])
        bal_tids = set(int(r["TARGETID"]) for r in bal if r["BI_CIV"] > 0)
        print(f"[load] {len(bal_tids)} BAL targets (BI_CIV>0)", flush=True)
    if args.no_bal and bal_tids:
        keep_cat = ~np.isin(np.asarray(cat_pass["TARGETID"]), list(bal_tids))
        cat_pass = cat_pass[keep_cat]
        keep_truth = ~np.isin(np.asarray(truth["TARGETID"]), list(bal_tids))
        truth = truth[keep_truth]
        print(f"[no-bal] {keep_cat.sum()} cat, {keep_truth.sum()} truth "
              f"after BAL exclusion", flush=True)
        # also drop BAL TIDs from the all-cat list used for "processed TIDs"
        cat_for_processed_tid = cat[~np.isin(np.asarray(cat["TARGETID"]),
                                              list(bal_tids))]
    else:
        cat_for_processed_tid = cat

    # Restrict truth to TARGETIDs that were ACTUALLY processed by the
    # pipeline (i.e. appear in the catalog at any p_dla). The full truth
    # catalog includes spectra that were filtered out upstream of the GP
    # (z<2.5, ZWARN!=0, BAL exclusion, etc.); counting them in the
    # completeness denominator makes the production look much worse than
    # it actually is.
    cat_all_tids = set(int(t) for t in np.asarray(cat_for_processed_tid["TARGETID"]))
    in_cat_mask = np.array([int(t) in cat_all_tids
                            for t in np.asarray(truth["TARGETID"])])
    truth_processed = truth[in_cat_mask]
    print(f"[truth] {in_cat_mask.sum()} of {len(truth)} truth DLAs are on "
          f"TARGETIDs that were processed by the pipeline", flush=True)

    # Match truth ↔ MAP — BEFORE post-processing
    print("[match] truth ↔ map (raw)", flush=True)
    m_truth, m_map, m_map_any = _match_truth_to_map(
        truth_processed, cat_pass, args.dz_match_rel)

    rows_compl_raw = _per_nhi_bin_stats(
        m_truth, np.asarray(truth_processed["NHI"]))
    purity_raw = m_map.sum() / len(cat_pass) if len(cat_pass) else 0.0
    purity_raw_loose = m_map_any.sum() / len(cat_pass) if len(cat_pass) else 0.0

    # ---- Lyβ veto post-processing ----
    sys.path.insert(0, os.getcwd())
    from gpy_dla_detection.postprocess.lyb_veto import flag_lybeta
    cat_lyb = flag_lybeta(cat_pass.copy(),
                          targetid_col="TARGETID", z_col="Z_DLA",
                          nhi_col="NHI", dz_match=0.005)
    print(f"[lyβ] flagged {int(cat_lyb['LYBETA_FLAG'].sum())} as Lyβ misIDs",
          flush=True)
    cat_lyb_clean = cat_lyb[~cat_lyb["LYBETA_FLAG"]]
    m_truth_lyb, m_map_lyb, m_map_any_lyb = _match_truth_to_map(
        truth_processed, cat_lyb_clean, args.dz_match_rel)
    rows_compl_lyb = _per_nhi_bin_stats(
        m_truth_lyb, np.asarray(truth_processed["NHI"]))
    purity_lyb = m_map_lyb.sum() / len(cat_lyb_clean) if len(cat_lyb_clean) else 0.0
    purity_lyb_loose = m_map_any_lyb.sum() / len(cat_lyb_clean) if len(cat_lyb_clean) else 0.0

    # ---- LLS cross-reference (optional) ----
    rows_compl_lls = None
    purity_lls = None
    cat_lls_clean = None
    if args.lls_dir:
        lls_cat = _load_catalog_dir(args.lls_dir)
        print(f"[load] LLS catalog: {len(lls_cat)} rows", flush=True)
        if "P_DLA" in lls_cat.colnames:
            lls_keep = lls_cat[lls_cat["P_DLA"] >= 0.5]
        elif "MODEL_P" in lls_cat.colnames:
            lls_keep = lls_cat[lls_cat["MODEL_P"] >= 0.5]
        else:
            lls_keep = lls_cat
        print(f"  after p>=0.5: {len(lls_keep)}", flush=True)

        from gpy_dla_detection.postprocess.lls_cross_reference import (
            cross_reference_lls,
        )
        # Use the NHI from LLS-mode (not capped at 20.3)
        try:
            cat_lls = cross_reference_lls(
                cat_lyb.copy(), lls_keep, dz_match=0.01,
                lls_threshold=20.3,
                targetid_col="TARGETID", z_col="Z_DLA",
                nhi_col="NHI", p_col="P_DLA" if "P_DLA" in cat_lyb.colnames else "MODEL_P",
            )
            cat_lls_clean = cat_lls[
                ~cat_lls["LYBETA_FLAG"] & ~cat_lls["LLS_DOWNGRADE_FLAG"]
            ]
            n_downgrade = int(cat_lls["LLS_DOWNGRADE_FLAG"].sum())
            print(f"[lls-xref] downgraded {n_downgrade} additional MAP DLAs "
                  f"as likely sub-DLA / LLS", flush=True)
            m_truth_lls, m_map_lls, m_map_any_lls = _match_truth_to_map(
                truth_processed, cat_lls_clean, args.dz_match_rel)
            rows_compl_lls = _per_nhi_bin_stats(
                m_truth_lls, np.asarray(truth_processed["NHI"]))
            purity_lls = (m_map_lls.sum() / len(cat_lls_clean)
                          if len(cat_lls_clean) else 0.0)
            purity_lls_loose = (m_map_any_lls.sum() / len(cat_lls_clean)
                                if len(cat_lls_clean) else 0.0)
        except Exception as exc:
            print(f"[warn] LLS xref failed: {exc}", flush=True)

    # ---- Write report ----
    lines = [
        f"# London production catalog — purity / completeness",
        "",
        f"- Multi-DLA catalog: `{args.catalog_dir}`",
        f"- LLS catalog:       `{args.lls_dir or '(none)'}`",
        f"- Truth:             `{args.truth}` "
        f"(N truth DLAs with NHI≥20.3 = {len(truth)}; "
        f"{len(truth_processed)} on TARGETIDs the pipeline actually processed)",
        f"- Match metric: |Δz|/(1+z_truth) ≤ {args.dz_match_rel}",
        f"- p(DLA) cut: {args.p_dla_cut}",
        f"- N MAP DLAs after cut: raw={len(cat_pass)}",
        f"- N MAP DLAs after Lyβ veto: {len(cat_lyb_clean)}"
        f"  (removed {len(cat_pass) - len(cat_lyb_clean)})",
    ]
    if cat_lls_clean is not None:
        lines.append(
            f"- N MAP DLAs after Lyβ + LLS xref: {len(cat_lls_clean)}"
            f"  (removed {len(cat_pass) - len(cat_lls_clean)} total)"
        )
    lines += [
        "",
        f"## Headline numbers",
        "",
        "| stage                       | completeness | strict purity (1-to-1) | loose purity (≥1 truth match) |",
        "|:----------------------------|:------------:|:----------------------:|:-----------------------------:|",
        f"| raw catalog                 | {rows_compl_raw[-1]['rate']:.1%} | {purity_raw:.1%} | {purity_raw_loose:.1%} |",
        f"| + Lyβ veto                  | {rows_compl_lyb[-1]['rate']:.1%} | {purity_lyb:.1%} | {purity_lyb_loose:.1%} |",
    ]
    if rows_compl_lls is not None:
        lines.append(f"| + LLS cross-reference       | {rows_compl_lls[-1]['rate']:.1%} | {purity_lls:.1%} | {purity_lls_loose:.1%} |")

    lines.append(_format_table(rows_compl_raw, "Completeness — raw"))
    lines.append(_format_table(rows_compl_lyb, "Completeness — after Lyβ veto"))
    if rows_compl_lls is not None:
        lines.append(_format_table(
            rows_compl_lls, "Completeness — after Lyβ + LLS cross-reference"))

    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
