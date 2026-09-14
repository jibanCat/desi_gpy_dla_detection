#!/usr/bin/env python
"""run_base_refit_check.py -- sealed section 6A reweighting-invariance audit
for candidate E with the R1c-form BASE REFITTED under each reweighting.

DIAGNOSTIC ONLY.  Calibration side.  No HBI/MCMC run, no real data, no mock
closure number, no model development, no tracked file modified.

Why: in the sealed run (run_candidates.antitaut_A) E's base was fitted ONCE
unweighted and held fixed across the three reweightings, while the split-half
noise denominator DID refit the base (mathematics review F1/F2).  So the sealed
"E is invariant" is a statement about the CORRECTION GIVEN A FIXED BASE.  Here
the same audit is rerun with the base refitted, and the total is decomposed.

Everything is imported from run_candidates / candlib / candmetrics / respfit;
the only new code is the weighted expression of the SAME R1c estimator
(``wbase.py``), which respfit does not provide.

Paths / arms (all against the delivered native objects rows_<name>.npz):
  E_fixed_base  : base = R1c(full, unweighted), E refit with w   [reproduction]
  E_base_refit  : base = R1c(full, w),          E refit with w   [new]
  base_only     : R1c(full, w) vs rows_R1c.npz                   [new]
Denominators: split-half (TARGETID % 4) sampling KL, base refit in both halves
for the E arms (exactly as run_candidates does), R1c split-half for base_only.
B (no base) equalised is rerun as the harness control.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAND = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_HERE, _CAND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_candidates as RC                                    # noqa: E402
import candlib as CL                                           # noqa: E402
import build_variants as BV                                    # noqa: E402
import wbase as WB                                             # noqa: E402

PRODUCTS = os.path.join(RC.SCRATCH, "response_review", "candidates")
KINDS = ("steep", "flat", "equalised")


# ---------------------------------------------------------------------------
def setup(events=None):
    geom = CL.load_geom(os.path.join(RC.SUPPORT,
                                     "empirical_ops_2lpt0_A0.npz"),
                        os.path.join(RC.PACKS, "scanpack_2lpt0_b300.npz"))
    ev = CL.load_events(events or RC.EVENTS, geom)
    N_ref = float(np.load(RC.ADOPTED_NPZ, allow_pickle=True)["N_ref"])
    return geom, ev, N_ref


def kl_block(rows_native, rows_alt, kl_samp, cnt, big):
    """The sealed section 6A row statistics + a count-weighted mean."""
    kl = RC.row_kl(rows_native, rows_alt)
    ratio = kl[big] / np.maximum(kl_samp[big], 1e-12)
    exc = ratio > 3.0
    wts = cnt[big]
    return dict(
        kl_median=RC._safe(np.median, kl[big]),
        kl_p95=RC._safe(np.percentile, kl[big], 95),
        kl_count_weighted_mean=(float(np.sum(wts * kl[big]) / wts.sum())
                                if wts.size else float("nan")),
        ratio_median=RC._safe(np.median, ratio),
        ratio_count_weighted_mean=(float(np.sum(wts * ratio) / wts.sum())
                                   if wts.size else float("nan")),
        n_rows_over_3x=int(exc.sum()),
        frac_rows_over_3x=(float(exc.mean()) if exc.size else float("nan")),
        verdict=("OCCUPANCY-IMPRINTED" if exc.mean() > 0.20
                 else "invariant within sampling noise"))


def r1c_spec():
    """The base spec AS DELIVERED: build_variants module defaults
    (marg_deg=4, r1c_estimator='sample').  NOT changed here."""
    return BV.specs()["R1c"]


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        _HERE, "E_base_refit_reweighting_check.json"))
    ap.add_argument("--products", default=PRODUCTS)
    ap.add_argument("--skip-b", action="store_true")
    a = ap.parse_args(argv)

    t_all = time.time()
    geom, ev, N_ref = setup()
    spec = r1c_spec()
    full = np.ones(ev["n"], bool)
    tid = np.asarray(ev["tid"], np.int64)
    h1, h2 = (tid % 4) < 2, (tid % 4) >= 2
    cnt = CL.held_out_counts(ev, full, geom).sum(axis=-1)
    big = cnt >= 200
    print(f"[brc] n={ev['n']} rows>=200: {int(big.sum())}", flush=True)

    def load_native(name):
        return np.asarray(np.load(os.path.join(a.products, f"rows_{name}.npz"),
                                  allow_pickle=True)["rows"], float)

    nat_E = load_native("E")
    nat_R1c = load_native("R1c")

    out = dict(
        provenance=dict(
            events=RC.EVENTS, products=a.products, git=RC._git(),
            base_spec={k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in dict(spec).items()},
            note=("base spec is build_variants.specs() with the MODULE "
                  "DEFAULTS marg_deg=4, r1c_estimator='sample' -- the object "
                  "of record as delivered; deliberately NOT changed.")),
        n_rows_ge200=int(big.sum()))

    # ---- shared denominators --------------------------------------------
    t0 = time.time()
    base_full = RC.fit_candidate("R1c", ev, full, geom, N_ref)["rows"]
    b1 = RC.fit_candidate("R1c", ev, h1, geom, N_ref)["rows"]
    b2 = RC.fit_candidate("R1c", ev, h2, geom, N_ref)["rows"]
    kl_samp_base = 0.5 * (RC.row_kl(b1, b2) + RC.row_kl(b2, b1))
    s1 = RC.fit_candidate("E", ev, h1, geom, N_ref, base_rows=b1)["rows"]
    s2 = RC.fit_candidate("E", ev, h2, geom, N_ref, base_rows=b2)["rows"]
    kl_samp_E = 0.5 * (RC.row_kl(s1, s2) + RC.row_kl(s2, s1))
    out["sampling_kl_median_E"] = RC._safe(np.median, kl_samp_E[big])
    out["sampling_kl_median_base"] = RC._safe(np.median, kl_samp_base[big])
    print(f"[brc] denominators in {time.time()-t0:.1f}s  "
          f"E={out['sampling_kl_median_E']:.6g} "
          f"base={out['sampling_kl_median_base']:.6g}", flush=True)

    # ---- the three arms --------------------------------------------------
    out["E_fixed_base"] = {}
    out["E_base_refit"] = {}
    out["base_only"] = {}
    for kind in KINDS:
        t0 = time.time()
        w = RC.slope_weights(ev, geom, kind)
        r_fix = RC.fit_candidate("E", ev, full, geom, N_ref, w=w,
                                 base_rows=base_full)["rows"]
        out["E_fixed_base"][kind] = kl_block(nat_E, r_fix, kl_samp_E, cnt, big)
        base_w, _obj = WB.base_rows_weighted(ev, full, geom, spec, N_ref, w=w)
        out["base_only"][kind] = kl_block(nat_R1c, base_w, kl_samp_base,
                                          cnt, big)
        r_ref = RC.fit_candidate("E", ev, full, geom, N_ref, w=w,
                                 base_rows=base_w)["rows"]
        out["E_base_refit"][kind] = kl_block(nat_E, r_ref, kl_samp_E, cnt, big)
        # decomposition: how much of the total is moved by refitting the base
        kl_fix = RC.row_kl(nat_E, r_fix)
        kl_ref = RC.row_kl(nat_E, r_ref)
        kl_bb = RC.row_kl(r_fix, r_ref)       # base-refit displacement of E
        out["E_base_refit"][kind]["decomposition"] = dict(
            kl_total_median=RC._safe(np.median, kl_ref[big]),
            kl_correction_only_median=RC._safe(np.median, kl_fix[big]),
            kl_base_displacement_median=RC._safe(np.median, kl_bb[big]),
            ratio_base_displacement=RC._safe(
                np.median, kl_bb[big] / np.maximum(kl_samp_E[big], 1e-12)),
            total_over_correction=(
                RC._safe(np.median, kl_ref[big]) /
                max(RC._safe(np.median, kl_fix[big]), 1e-300)))
        print(f"[brc] {kind}: fixed-base ratio_med="
              f"{out['E_fixed_base'][kind]['ratio_median']:.4g} "
              f"n>3x={out['E_fixed_base'][kind]['n_rows_over_3x']} | "
              f"base-refit ratio_med="
              f"{out['E_base_refit'][kind]['ratio_median']:.4g} "
              f"n>3x={out['E_base_refit'][kind]['n_rows_over_3x']} | "
              f"base-only ratio_med="
              f"{out['base_only'][kind]['ratio_median']:.4g} "
              f"n>3x={out['base_only'][kind]['n_rows_over_3x']} "
              f"[{time.time()-t0:.1f}s]", flush=True)
        with open(a.out, "w") as fh:
            json.dump(RC._jsonable(out), fh, indent=1)

    # ---- harness control: B, equalised ----------------------------------
    if not a.skip_b:
        t0 = time.time()
        nat_B = load_native("B")
        w = RC.slope_weights(ev, geom, "equalised")
        rb = RC.fit_candidate("B", ev, full, geom, N_ref, w=w)["rows"]
        q1 = RC.fit_candidate("B", ev, h1, geom, N_ref)["rows"]
        q2 = RC.fit_candidate("B", ev, h2, geom, N_ref)["rows"]
        kl_samp_B = 0.5 * (RC.row_kl(q1, q2) + RC.row_kl(q2, q1))
        out["B_equalised_control"] = kl_block(nat_B, rb, kl_samp_B, cnt, big)
        out["B_equalised_control"]["sampling_kl_median"] = RC._safe(
            np.median, kl_samp_B[big])
        print(f"[brc] B equalised control ratio_med="
              f"{out['B_equalised_control']['ratio_median']:.4g} "
              f"n>3x={out['B_equalised_control']['n_rows_over_3x']} "
              f"[{time.time()-t0:.1f}s]", flush=True)

    out["wall_s"] = time.time() - t_all
    with open(a.out, "w") as fh:
        json.dump(RC._jsonable(out), fh, indent=1)
    print(f"[brc] written {a.out} in {out['wall_s']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
