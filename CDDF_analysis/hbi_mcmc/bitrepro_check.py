#!/usr/bin/env python3
"""bitrepro_check.py — the CP-2 bit-reproduction check as a committed routine (it lived as
an inline heredoc in run_cp2_collect.sbatch with a silent OUT fall-through; Paper-1 code
review 2026-08-26). Compares each family's production-pack validation run against its
Battery-2/3 reference record: thresholds identical, per-z medians identical, max |Δ| on
the all-z bias, divergences. Fails closed on a missing file.

    python -m CDDF_analysis.hbi_mcmc.bitrepro_check --new-dir <cp2_validation> --ref-dir <perz_20260820>
        --families 2lpt0 london0 saclay0 --new-pattern cp2_ln_w1500_{fam}_s20260811.json
        --ref-pattern perz_gcons_ln_w1500_{fam}_s20260811.json --out bitrepro.json
"""
import argparse, json, os, sys
from CDDF_analysis.hbi_mcmc.provenance_util import sha256


def compare(new: dict, ref: dict) -> dict:
    same_thr = new["thresholds"] == ref["thresholds"]
    pa, pb = new["perz_recovery"]["estimand"], ref["perz_recovery"]["estimand"]
    same_perz = all(x["median_bias_pct"] == y["median_bias_pct"] for t in ("ge20.0", "ge20.3")
                    for x, y in zip(pa[t]["paper1_bins"], pb[t]["paper1_bins"]))
    dmax = max(abs(new["thresholds"][t]["median_bias_pct"] - ref["thresholds"][t]["median_bias_pct"]) for t in ("ge20.0", "ge20.3"))
    return dict(thresholds_identical=bool(same_thr), perz_identical=bool(same_perz), max_abs_diff_allz_pct=float(dmax),
                divergences_new=new.get("divergences"), divergences_ref=ref.get("divergences"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", required=True); ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--families", nargs="+", default=["2lpt0", "london0", "saclay0"])
    ap.add_argument("--new-pattern", default="cp2_ln_w1500_{fam}_s20260811.json")
    ap.add_argument("--ref-pattern", default="perz_gcons_ln_w1500_{fam}_s20260811.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rec, allok = {}, True
    for fam in a.families:
        pn, pr = os.path.join(a.new_dir, a.new_pattern.format(fam=fam)), os.path.join(a.ref_dir, a.ref_pattern.format(fam=fam))
        for p in (pn, pr):
            if not os.path.isfile(p):
                raise SystemExit(f"bitrepro_check: missing {p} — refusing to report a PASS on absent files")
        r = compare(json.load(open(pn)), json.load(open(pr)))
        r.update(new=pn, new_sha256=sha256(pn), ref=pr, ref_sha256=sha256(pr))
        rec[fam] = r; allok &= r["thresholds_identical"] and r["perz_identical"]
        print(f"{fam}: thresholds identical={r['thresholds_identical']} perz medians identical={r['perz_identical']} "
              f"max|d allz|={r['max_abs_diff_allz_pct']:.3e} div new/ref={r['divergences_new']}/{r['divergences_ref']}")
    rec["all_bit_reproduced"] = bool(allok)
    print("BITREPRO", "PASS" if allok else "FAIL")
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
