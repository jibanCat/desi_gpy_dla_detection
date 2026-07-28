#!/usr/bin/env python
"""loa_matterhorn_consistency.py — PER-TARGET consistency of two GP-DLA catalogs
produced by the SAME finder from DIFFERENT reductions of the same survey.

WHY THIS EXISTS
---------------
Purity and completeness are structurally impossible on real spectra: there is no
truth catalog.  The strongest *deployment* statement the project can make is
therefore a REPEATABILITY statement — run the identical finder (identical learned
model, identical configuration) over two independent reductions of an overlapping
quasar sample and ask how often it returns the same absorber.

This routine implements exactly that comparison and NOTHING else.  It is a
reduce-only consumer of two packaged absorber catalogs.  It runs NO GP inference
and imports nothing from ``gpy_dla_detection/``.

WHAT IT IS NOT
--------------
* It is NOT a purity or completeness measurement.  Neither catalog is truth.
  A disagreement tells you the pair is unstable under redux; it does NOT tell
  you which of the two is right, and an AGREEMENT does not establish that either
  is correct — two runs of the same deterministic finder on correlated inputs
  share every systematic the finder has.
* It is NOT a population measurement.  No dN/dX, no f(N), no Omega.
* The "present in one catalog only" bucket is NOT "missed by the other finder"
  unless you can independently show the sightline was PROCESSED by the other run.
  Packaged absorber catalogs contain only sightlines with >=1 candidate, so the
  processed-but-empty case and the never-processed case are indistinguishable
  from the catalogs alone.  Supply ``--loa-processed-tids`` /
  ``--mh-processed-tids`` (newline- or whitespace-separated TARGETID lists) to
  split that bucket; without them the routine reports the bucket UNSPLIT and
  marks ``processed_denominator_available=False``.  DO NOT quote a "miss rate"
  from an unsplit run.

PRIVACY (load-bearing)
----------------------
On REAL survey catalogs every number this routine emits — counts, match rates,
Delta N_HI / Delta z distributions, SNR trends — is a REAL-DATA result.  The
routine and its tests are mock-safe and may be committed; its OUTPUT MUST NOT be
committed to the code repo and MUST NOT appear in a commit message.  Real-data
outputs belong in the private notes repo only.  The routine refuses to write
inside the code repository unless ``--allow-in-repo-output`` is passed, and it
prints a REAL-DATA banner whenever it sees survey-magnitude TARGETIDs
(mock O(1e3-1e8) vs real DESI O(1e16); see ``--real-tid-threshold``).

MATCHING CONVENTION
-------------------
Absorbers are matched per shared TARGETID by redshift proximity, greedily in
order of increasing |dz|/(1+z), with a maximum separation ``--dz-rel`` (default
0.01 — the same tolerance every purity/completeness product in this project uses
to match a detection to its truth counterpart).  Matching is on redshift ONLY;
N_HI is never used to match, so the recovered Delta N_HI distribution is an
independent output rather than a consequence of the pairing rule.

Usage
-----
    python CDDF_analysis/diagnostics/deployment/loa_matterhorn_consistency.py \
        --cat-a <catalogA.fits> --label-a loa \
        --cat-b <catalogB.fits> --label-b matterhorn \
        --nhi-min 20.3 --snr-min 2.0 --gp-conf 0.99 \
        --out /path/OUTSIDE/the/code/repo/consistency.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

_PROVENANCE_DEPS = ["CDDF_analysis/diagnostics/deployment/loa_matterhorn_consistency.py"]

#: TARGETID magnitude above which a catalog is treated as REAL DESI survey data.
#: Mock TARGETIDs are O(1e3-1e8); real DESI TARGETIDs are O(1e16).
DEFAULT_REAL_TID_THRESHOLD = 1e12


# -----------------------------------------------------------------------------
# provenance
# -----------------------------------------------------------------------------
def _git_commit(deps=None) -> str:
    """40-char HEAD SHA, suffixed ``-dirty`` if any dependency is untracked or
    modified relative to HEAD.  A ``-dirty`` stamp means the artifact is NOT
    third-party re-derivable: commit the routine, then re-run."""
    deps = list(deps if deps is not None else _PROVENANCE_DEPS)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                      stderr=subprocess.DEVNULL).decode().strip()
        for f in deps:
            tracked = subprocess.call(
                ["git", "ls-files", "--error-unmatch", f], cwd=_REPO,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
            modified = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", f],
                                       cwd=_REPO) != 0
            if not tracked or modified:
                return f"{sha}-dirty"
        return sha
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] _git_commit() failed ({type(e).__name__}: {e})", file=sys.stderr)
        return "unknown"


# -----------------------------------------------------------------------------
# selection
# -----------------------------------------------------------------------------
def select_rows(cat, *, nhi_min, snr_min, gp_conf, z_qso_min, z_qso_max,
                lam_rf_min, lam_rf_max, require_flag_zero=True):
    """Apply the packaged catalogs' documented 'recommended selection'.

    Returns a boolean mask over ``cat`` rows.  ``lam_rf_*`` gate Z_DLA to the
    rest-frame forest window of its own quasar, matching every other product in
    this project (LYA_REST = 1215.67 A).
    """
    LYA = 1215.67
    m = np.ones(len(cat["TARGETID"]), dtype=bool)
    if require_flag_zero and "DLAFLAG" in cat:
        m &= (np.asarray(cat["DLAFLAG"]) == 0)
    if gp_conf is not None and "P_DLA" in cat:
        m &= (np.asarray(cat["P_DLA"], dtype=float) > gp_conf)
    if snr_min is not None and "SNR_REDSIDE" in cat:
        m &= (np.asarray(cat["SNR_REDSIDE"], dtype=float) > snr_min)
    if nhi_min is not None:
        m &= (np.asarray(cat["NHI"], dtype=float) >= nhi_min)
    zq = np.asarray(cat["Z_QSO"], dtype=float)
    zd = np.asarray(cat["Z_DLA"], dtype=float)
    if z_qso_min is not None:
        m &= (zq >= z_qso_min)
    if z_qso_max is not None:
        m &= (zq <= z_qso_max)
    lam_rf = LYA * (1.0 + zd) / (1.0 + zq)
    if lam_rf_min is not None:
        m &= (lam_rf >= lam_rf_min)
    if lam_rf_max is not None:
        m &= (lam_rf <= lam_rf_max)
    m &= np.isfinite(zd) & np.isfinite(np.asarray(cat["NHI"], dtype=float))
    return m


def _group_by_tid(tid):
    """Return dict tid -> np.array of row indices, built with one argsort."""
    order = np.argsort(tid, kind="stable")
    st = tid[order]
    bounds = np.flatnonzero(np.r_[True, st[1:] != st[:-1], True])
    out = {}
    for i in range(len(bounds) - 1):
        sl = order[bounds[i]:bounds[i + 1]]
        out[st[bounds[i]]] = sl
    return out


def greedy_match(za, zb, dz_rel):
    """Greedy nearest-redshift matching between two absorber lists of one sightline.

    Returns a list of (ia, ib) index pairs.  Pairs are consumed in order of
    increasing |za-zb|/(1+za); each absorber is used at most once; pairs beyond
    ``dz_rel`` are never formed.  Matching uses redshift ONLY.
    """
    if len(za) == 0 or len(zb) == 0:
        return []
    d = np.abs(za[:, None] - zb[None, :]) / (1.0 + za[:, None])
    pairs = []
    used_a, used_b = set(), set()
    flat = np.argsort(d, axis=None)
    na, nb = d.shape
    for k in flat:
        ia, ib = divmod(int(k), nb)
        if d[ia, ib] > dz_rel:
            break
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        pairs.append((ia, ib))
        if len(used_a) == na or len(used_b) == nb:
            break
    return pairs


def _dist(x):
    """Summary of a 1-D distribution.  Empty input -> all-None (never NaN-in-JSON)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "mad": None,
                "q02.5": None, "q16": None, "q84": None, "q97.5": None,
                "frac_within_0.1dex": None, "frac_within_0.3dex": None}
    med = float(np.median(x))
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": med,
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "mad": float(np.median(np.abs(x - med))),
        "q02.5": float(np.quantile(x, 0.025)),
        "q16": float(np.quantile(x, 0.16)),
        "q84": float(np.quantile(x, 0.84)),
        "q97.5": float(np.quantile(x, 0.975)),
        "frac_within_0.1dex": float(np.mean(np.abs(x) <= 0.1)),
        "frac_within_0.3dex": float(np.mean(np.abs(x) <= 0.3)),
    }


def compare(cat_a, cat_b, *, dz_rel=0.01, sel_kw=None, snr_bins=None,
            processed_a=None, processed_b=None, restrict_to_common=False):
    """Core comparison.  ``cat_a``/``cat_b`` are dict-like column mappings.

    ``restrict_to_common`` first intersects the two catalogs on RAW TARGETID
    (before any quality cut) and keeps only that intersection.  Every sightline
    in the intersection carries >=1 candidate in BOTH files, so it was certainly
    processed by both runs; within that population "absent from the other
    catalog" means "the other run reported nothing passing the cut", not "the
    other run never looked".  This is the only sub-population on which a
    disagreement rate is interpretable without the processed-sightline lists.

    Returns a plain-JSON-able dict.  Deterministic; no RNG.
    """
    sel_kw = dict(sel_kw or {})
    ma = select_rows(cat_a, **sel_kw)
    mb = select_rows(cat_b, **sel_kw)
    if restrict_to_common:
        common_raw = np.intersect1d(np.asarray(cat_a["TARGETID"]),
                                    np.asarray(cat_b["TARGETID"]))
        ma &= np.isin(np.asarray(cat_a["TARGETID"]), common_raw)
        mb &= np.isin(np.asarray(cat_b["TARGETID"]), common_raw)

    tid_a = np.asarray(cat_a["TARGETID"])[ma]
    tid_b = np.asarray(cat_b["TARGETID"])[mb]
    z_a = np.asarray(cat_a["Z_DLA"], dtype=float)[ma]
    z_b = np.asarray(cat_b["Z_DLA"], dtype=float)[mb]
    n_a = np.asarray(cat_a["NHI"], dtype=float)[ma]
    n_b = np.asarray(cat_b["NHI"], dtype=float)[mb]
    snr_a = np.asarray(cat_a["SNR_REDSIDE"], dtype=float)[ma] if "SNR_REDSIDE" in cat_a \
        else np.full(tid_a.size, np.nan)
    snr_b = np.asarray(cat_b["SNR_REDSIDE"], dtype=float)[mb] if "SNR_REDSIDE" in cat_b \
        else np.full(tid_b.size, np.nan)

    ga = _group_by_tid(tid_a)
    gb = _group_by_tid(tid_b)
    set_a, set_b = set(ga), set(gb)
    shared = sorted(set_a & set_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)

    d_nhi, d_z_rel, pair_snr, pair_nhi_a = [], [], [], []
    unmatched_a_shared = unmatched_b_shared = 0
    count_pairs = {}   # (n_a, n_b) -> n sightlines
    n_same_count = 0

    for t in shared:
        ia, ib = ga[t], gb[t]
        key = (int(len(ia)), int(len(ib)))
        count_pairs[key] = count_pairs.get(key, 0) + 1
        if key[0] == key[1]:
            n_same_count += 1
        pairs = greedy_match(z_a[ia], z_b[ib], dz_rel)
        unmatched_a_shared += len(ia) - len(pairs)
        unmatched_b_shared += len(ib) - len(pairs)
        for pa, pb in pairs:
            d_nhi.append(n_b[ib[pb]] - n_a[ia[pa]])
            d_z_rel.append((z_b[ib[pb]] - z_a[ia[pa]]) / (1.0 + z_a[ia[pa]]))
            pair_snr.append(snr_a[ia[pa]])
            pair_nhi_a.append(n_a[ia[pa]])

    d_nhi = np.asarray(d_nhi, dtype=float)
    d_z_rel = np.asarray(d_z_rel, dtype=float)
    pair_snr = np.asarray(pair_snr, dtype=float)
    pair_nhi_a = np.asarray(pair_nhi_a, dtype=float)

    n_det_a = int(tid_a.size)
    n_det_b = int(tid_b.size)
    n_pairs = int(d_nhi.size)

    # absorber-level taxonomy over the SHARED sightlines only (where "absent in
    # the other catalog" is unambiguous, because that sightline was certainly
    # processed by both runs -- it carries >=1 detection in both).
    det_a_shared = int(sum(len(ga[t]) for t in shared))
    det_b_shared = int(sum(len(gb[t]) for t in shared))

    res = {
        "selection": sel_kw,
        "restrict_to_common_sightlines": bool(restrict_to_common),
        "dz_rel": float(dz_rel),
        "sightlines": {
            "n_a": len(set_a), "n_b": len(set_b),
            "n_shared": len(shared),
            "n_only_a": len(only_a), "n_only_b": len(only_b),
            "shared_frac_of_a": (len(shared) / len(set_a)) if set_a else None,
            "shared_frac_of_b": (len(shared) / len(set_b)) if set_b else None,
            "jaccard": (len(shared) / len(set_a | set_b)) if (set_a or set_b) else None,
        },
        "absorbers": {
            "n_a": n_det_a, "n_b": n_det_b,
            "n_a_on_shared": det_a_shared, "n_b_on_shared": det_b_shared,
            "n_matched_pairs": n_pairs,
            "n_unmatched_a_on_shared": int(unmatched_a_shared),
            "n_unmatched_b_on_shared": int(unmatched_b_shared),
            "match_rate_a_on_shared": (n_pairs / det_a_shared) if det_a_shared else None,
            "match_rate_b_on_shared": (n_pairs / det_b_shared) if det_b_shared else None,
            "match_rate_a_all": (n_pairs / n_det_a) if n_det_a else None,
            "match_rate_b_all": (n_pairs / n_det_b) if n_det_b else None,
        },
        "count_agreement": {
            "n_shared_sightlines": len(shared),
            "n_same_absorber_count": int(n_same_count),
            "frac_same_absorber_count": (n_same_count / len(shared)) if shared else None,
            "confusion": {f"{k[0]}->{k[1]}": v
                          for k, v in sorted(count_pairs.items())},
        },
        "delta_nhi_b_minus_a": _dist(d_nhi),
        "delta_z_rel_b_minus_a": _dist(d_z_rel),
        "taxonomy": {
            "note": ("Buckets are disjoint and exhaust every selected absorber. "
                     "'only_in_one_catalog_sightline' CANNOT be read as a finder miss "
                     "unless the processed-sightline lists are supplied."),
            "matched_pair": n_pairs,
            "same_sightline_no_z_counterpart_a": int(unmatched_a_shared),
            "same_sightline_no_z_counterpart_b": int(unmatched_b_shared),
            "sightline_absent_from_b": int(n_det_a - det_a_shared),
            "sightline_absent_from_a": int(n_det_b - det_b_shared),
        },
        "processed_denominator_available": bool(processed_a is not None
                                                and processed_b is not None),
    }

    if processed_a is not None and processed_b is not None:
        pa, pb = set(np.asarray(processed_a).tolist()), set(np.asarray(processed_b).tolist())
        both_proc = pa & pb
        res["processed"] = {
            "n_processed_a": len(pa), "n_processed_b": len(pb),
            "n_processed_both": len(both_proc),
            "n_only_a_and_processed_by_b": len(set(only_a) & both_proc),
            "n_only_b_and_processed_by_a": len(set(only_b) & both_proc),
            "n_only_a_not_processed_by_b": len(set(only_a) - both_proc),
            "n_only_b_not_processed_by_a": len(set(only_b) - both_proc),
        }

    # ---- SNR trend -----------------------------------------------------------
    if snr_bins is None:
        snr_bins = [2.0, 3.0, 4.0, 6.0, 9.0, 15.0, 1e9]
    snr_bins = list(map(float, snr_bins))
    rows = []
    # per-absorber (catalog A) match rate + Delta scatter, binned on A's SNR_REDSIDE
    a_snr_all = snr_a
    a_is_shared = np.isin(tid_a, np.asarray(shared)) if shared else np.zeros(tid_a.size, bool)
    for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
        in_bin_all = (a_snr_all >= lo) & (a_snr_all < hi)
        in_bin_shared = in_bin_all & a_is_shared
        in_bin_pair = (pair_snr >= lo) & (pair_snr < hi)
        n_shared_det = int(in_bin_shared.sum())
        n_pair = int(in_bin_pair.sum())
        rows.append({
            "snr_lo": lo, "snr_hi": (None if hi >= 1e9 else hi),
            "n_absorbers_a": int(in_bin_all.sum()),
            "n_absorbers_a_on_shared_sightlines": n_shared_det,
            "n_matched": n_pair,
            "match_rate_on_shared": (n_pair / n_shared_det) if n_shared_det else None,
            "delta_nhi": _dist(d_nhi[in_bin_pair]),
            "delta_z_rel": _dist(d_z_rel[in_bin_pair]),
        })
    res["by_snr"] = rows

    # ---- N_HI trend ----------------------------------------------------------
    nhi_edges = [19.5, 20.0, 20.3, 20.6, 21.0, 21.5, 23.0]
    nrows = []
    for lo, hi in zip(nhi_edges[:-1], nhi_edges[1:]):
        sel = (pair_nhi_a >= lo) & (pair_nhi_a < hi)
        nrows.append({"nhi_lo": lo, "nhi_hi": hi,
                      "delta_nhi": _dist(d_nhi[sel]),
                      "delta_z_rel": _dist(d_z_rel[sel])})
    res["by_nhi_a"] = nrows
    return res


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def _read_cat(path):
    import fitsio
    cols = ["TARGETID", "Z_QSO", "SNR_REDSIDE", "Z_DLA", "NHI", "DLAFLAG", "P_DLA"]
    d = fitsio.read(path, ext=1)
    have = set(d.dtype.names)
    return {c: d[c] for c in cols if c in have}


def _read_tids(path):
    if path is None:
        return None
    with open(path) as fh:
        txt = fh.read().replace(",", " ")
    return np.array([int(x) for x in txt.split() if x.strip()], dtype=np.int64)


def _looks_real(*tid_arrays, threshold=DEFAULT_REAL_TID_THRESHOLD):
    for t in tid_arrays:
        t = np.asarray(t)
        if t.size and float(np.nanmax(np.abs(t.astype(float)))) >= threshold:
            return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat-a", required=True)
    ap.add_argument("--cat-b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dz-rel", type=float, default=0.01)
    ap.add_argument("--nhi-min", type=float, default=20.3)
    ap.add_argument("--snr-min", type=float, default=2.0)
    ap.add_argument("--gp-conf", type=float, default=0.99)
    ap.add_argument("--z-qso-min", type=float, default=2.0)
    ap.add_argument("--z-qso-max", type=float, default=4.25)
    ap.add_argument("--lam-rf-min", type=float, default=1025.0)
    ap.add_argument("--lam-rf-max", type=float, default=1216.0)
    ap.add_argument("--loa-processed-tids", default=None,
                    help="optional TARGETID list processed by run A")
    ap.add_argument("--mh-processed-tids", default=None,
                    help="optional TARGETID list processed by run B")
    ap.add_argument("--real-tid-threshold", type=float,
                    default=DEFAULT_REAL_TID_THRESHOLD)
    ap.add_argument("--allow-in-repo-output", action="store_true",
                    help="permit --out inside the code repo (MOCK catalogs only)")
    ap.add_argument("--restrict-to-common-sightlines", action="store_true",
                    help=("keep only TARGETIDs present in BOTH raw catalogs, so every "
                          "sightline compared was certainly processed by both runs"))
    args = ap.parse_args(argv)

    out_abs = os.path.abspath(args.out)
    in_repo = os.path.commonpath([out_abs, _REPO]) == _REPO

    t0 = time.time()
    A = _read_cat(args.cat_a)
    B = _read_cat(args.cat_b)
    real = _looks_real(A["TARGETID"], B["TARGETID"], threshold=args.real_tid_threshold)

    if in_repo and not args.allow_in_repo_output:
        raise SystemExit(
            f"REFUSING to write inside the code repo ({out_abs}).\n"
            "Real-data outputs belong in the private notes repo. Pass "
            "--allow-in-repo-output only for MOCK catalogs.")
    if real and in_repo:
        raise SystemExit(
            "REFUSING: survey-magnitude TARGETIDs detected AND --out is inside the "
            "code repo. Real-data results must never enter this repository.")
    if real:
        print("=" * 78)
        print("REAL-DATA RUN. Every number below is a REAL DESI result.")
        print("Do NOT commit this output or quote it in a commit message.")
        print("=" * 78)

    sel_kw = dict(nhi_min=args.nhi_min, snr_min=args.snr_min, gp_conf=args.gp_conf,
                  z_qso_min=args.z_qso_min, z_qso_max=args.z_qso_max,
                  lam_rf_min=args.lam_rf_min, lam_rf_max=args.lam_rf_max)
    res = compare(A, B, dz_rel=args.dz_rel, sel_kw=sel_kw,
                  processed_a=_read_tids(args.loa_processed_tids),
                  processed_b=_read_tids(args.mh_processed_tids),
                  restrict_to_common=args.restrict_to_common_sightlines)

    res["metadata"] = {
        "what": ("per-TARGETID consistency of two GP-DLA catalogs from the same finder "
                 "on different reductions; a REPEATABILITY statement, NOT purity or "
                 "completeness"),
        "label_a": args.label_a, "label_b": args.label_b,
        "cat_a": args.cat_a, "cat_b": args.cat_b,
        "data_class": "REAL-SURVEY (private)" if real else "MOCK (public-OK)",
        "code_commit": _git_commit(),
        "routine": _PROVENANCE_DEPS[0],
        "rederive": (
            "python CDDF_analysis/diagnostics/deployment/loa_matterhorn_consistency.py "
            f"--cat-a {args.cat_a} --cat-b {args.cat_b} --label-a {args.label_a} "
            f"--label-b {args.label_b} --nhi-min {args.nhi_min} --snr-min {args.snr_min} "
            f"--gp-conf {args.gp_conf} --dz-rel {args.dz_rel} --out <out>"),
        "wallclock_s": time.time() - t0,
    }

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    with open(out_abs, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {out_abs}  ({res['metadata']['wallclock_s']:.1f}s)")
    return res


if __name__ == "__main__":
    main()
