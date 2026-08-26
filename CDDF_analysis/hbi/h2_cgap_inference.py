#!/usr/bin/env python3
"""h2_cgap_inference.py — the committed producer of H2_CGAP_INFERENCE.json (PI ruling
2026-08-16 item 2), reconstructed 2026-08-26 during the Paper-1 code review: the number
C_gap = 0.496 [0.407, 0.593] that defines the BH artifact of record previously existed
only as prose inside its own output.

Inference (verbatim from the recorded assumptions):
  A1  C_det(N) is monotone non-decreasing on [20.0, 21.0] (supported by every H2 cell).
  A2  the gap average C_gap = mean C over [20.3, 20.5) lies bracket-uniform between the
      two H2-measured endpoints:  C_gap = C_lo + U (C_hi - C_lo),  U ~ Uniform(0, 1),
      C_lo ~ Beta(k+1/2, n-k+1/2) from the canonical arm-B molly cell [20.0, 20.3)
      (k = 63, n = 175, nobal), C_hi ~ Beta(k+1/2, n-k+1/2) from the pure N = 20.5
      stratum nre[20.3, 20.7) (k = 49, n = 77).  Jeffreys posteriors.
The dN/dX(>= 20.3) component is the recorded MAP response map of the full estimator
(loa0 FP, h2cal C elsewhere; kernel fixed) evaluated at the C_gap draws by monotone
interpolation.  Monte Carlo with a fixed seed; the original seed is not on record, so
reproduction is to the printed precision of the recorded quantiles, not bitwise.

    python CDDF_analysis/hbi/h2_cgap_inference.py --canonical <h2_canonical_armB_lya_nobal.json>
        --record <H2_CGAP_INFERENCE.json> [--out new.json] [--n 200000] [--seed 20260816]
"""
import argparse, json, pathlib, sys
import numpy as np

LO_STRATUM = "molly_nhi:[20.0,20.3)"
HI_STRATUM = "nre:20.3-20.7"


def strata_counts(canonical: dict) -> dict:
    return {s["stratum"]: (int(s["k"]), int(s["n"])) for s in canonical["detection_strata"]
            if s.get("k") is not None and s.get("n") is not None}


def cgap_draws(k_lo, n_lo, k_hi, n_hi, n=200_000, seed=20260816):
    rng = np.random.default_rng(seed)
    c_lo = rng.beta(k_lo + 0.5, n_lo - k_lo + 0.5, n)
    c_hi = rng.beta(k_hi + 0.5, n_hi - k_hi + 0.5, n)
    u = rng.uniform(0.0, 1.0, n)
    return c_lo + u * (c_hi - c_lo)


def infer(canonical: dict, record: dict, n=200_000, seed=20260816) -> dict:
    kn = strata_counts(canonical)
    k_lo, n_lo = kn[LO_STRATUM]
    k_hi, n_hi = kn[HI_STRATUM]
    # the record names its inputs in prose; refuse silently different counts
    assert f"Beta({k_lo + 0.5},{n_lo - k_lo + 0.5})" in record["h2_inputs"]["C_lo"], (k_lo, n_lo, record["h2_inputs"]["C_lo"])
    assert f"Beta({k_hi + 0.5},{n_hi - k_hi + 0.5})" in record["h2_inputs"]["C_hi"], (k_hi, n_hi, record["h2_inputs"]["C_hi"])
    c = cgap_draws(k_lo, n_lo, k_hi, n_hi, n, seed)
    q = np.percentile(c, [16, 50, 84])
    grid = np.array(record["response_map"]["C_grid"], float)
    dndx = np.array(record["response_map"]["dndx_ge20_3"], float)
    # monotone decreasing map: interpolate on the reversed axis
    d = np.interp(c, grid, dndx)
    qd = np.percentile(d, [16, 50, 84])
    return {"C_lo": {"stratum": LO_STRATUM, "k": k_lo, "n": n_lo},
            "C_hi": {"stratum": HI_STRATUM, "k": k_hi, "n": n_hi},
            "n_draws": n, "seed": seed,
            "C_gap_p16_50_84": [round(float(x), 4) for x in q],
            "dndx_ge20_3_p16_50_84": [round(float(x), 4) for x in qd],
            "recorded_C_gap_p16_50_84": record["posterior"]["C_gap_p16_50_84"],
            "recorded_dndx_ge20_3_p16_50_84": record["posterior"]["dndx_ge20_3_p16_50_84"],
            "max_abs_diff_C_gap": float(np.max(np.abs(q - np.array(record["posterior"]["C_gap_p16_50_84"])))),
            "max_abs_diff_dndx": float(np.max(np.abs(qd - np.array(record["posterior"]["dndx_ge20_3_p16_50_84"]))))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--tol", type=float, default=5e-3,
                    help="reproduction tolerance on the quantiles: the recorded values carry the original run's own "
                         "Monte-Carlo noise (draw count not on record), ~2e-3 on the outer percentiles")
    a = ap.parse_args(argv)
    canonical = json.loads(pathlib.Path(a.canonical).read_text())
    record = json.loads(pathlib.Path(a.record).read_text())
    r = infer(canonical, record, a.n, a.seed)
    r["role"] = "committed re-derivation of H2_CGAP_INFERENCE.json (Paper-1 code review 2026-08-26); the recorded artifact stays the artifact of record"
    print(json.dumps({k: v for k, v in r.items() if k != "role"}, indent=1))
    ok = r["max_abs_diff_C_gap"] <= a.tol and r["max_abs_diff_dndx"] <= a.tol
    print("REPRODUCES recorded quantiles to %.0e: %s" % (a.tol, ok))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(r, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
