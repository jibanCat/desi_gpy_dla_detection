"""wall1_explain_partB.py — WALL-1 tilt-closure data + tilt-mechanism arrays for
the explanatory doc (reduce-only, NO MC, NO inference, NO SLURM).

The HEADLINE WALL-1 closure numbers (Delta-alpha=+/-0.5) are CACHED at the
calibrated experiment dir (wall1_result.tsv / wall1_pulls_*.csv from
run_phase3d_postkernel --stage 3). This script:
  (1) extracts the cached closure arrays into a single npz the figures consume;
  (2) recomputes the TRUTH-SIDE tilt mechanism arrays (the injected tilted-truth
      f(N) and dN/dX for Delta-alpha=+0.5, -0.5, and the realistic 0.015) — these
      are pure reductions of the truth catalog reweighted by w(N)=10^(da*(N-20.3)),
      computed via cddf_tilt_closure.tilted_truth_reductions. No detections, no MC.

The tilt FORMULAS (verified against cddf_tilt_closure.py):
  * weight              w(logN) = 10^(Delta-alpha * (logN - 20.3)), NaN host -> 1.0
  * TRUTH side          n_true^tilt[b] = sum_{true absorbers in bin b} w(logN_true)
  * DETECTION (measure) each op detection gets w(N_host_true) (truth-host logN,
                        NHI_TILT_HOST floor 19.0); hostless (forest FP) -> 1.0; this
                        multiplies BOTH the 1/C numerator AND the (1-rho) FP term in
                        the v1/v3 estimator (boot_weights / tilt_weights_op).
  * FROZEN              C, rho, b_FP (forest props, slope-independent), the kernel.
  * baseline R0         untilted est0/truth0 (the absolute Eddington bias; divided out)
  * closure pull        (est^tilt - R0*truth^tilt)/sigma_MC   (the GATED statistic)
  * raw pull            (est^tilt - truth^tilt)/sigma_MC      (re-measures abs bias)
  * gate (per limit)    integrated dN/dX & Omega |pull|<=3 on BOTH tilts AND closure
                        prediction in 95% MC band AND no opposite-sign coherent pull.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.cddf_tilt_closure import (
    tilt_weight, tilted_truth_reductions, LOGN_PIVOT,
)
from CDDF_analysis.ab_loa0_fp_baseline import build_ingredients


def parse_wall1_result(tsv_path):
    d = {}
    with open(tsv_path) as fh:
        for line in fh:
            if "\t" not in line:
                continue
            k, v = line.rstrip("\n").split("\t", 1)
            d[k] = v
    return d


def parse_wall1_pulls_fN(csv_path):
    rows = []
    with open(csv_path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("logN_lo"):
                continue
            rows.append([float(x) if x not in ("nan", "") else np.nan
                         for x in line.rstrip("\n").split(",")])
    cols = ("logN_lo logN_hi R0 ftrue_plus fpred_plus fest_plus fstd_plus "
            "pull_closure_plus pull_raw_plus ftrue_minus fpred_minus fest_minus "
            "fstd_minus pull_closure_minus pull_raw_minus gated").split()
    arr = np.array(rows)
    return {c: arr[:, i] for i, c in enumerate(cols)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    DEF_EXPER = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                 "phase3d_experiments/mollynhi195_lyaonly1025_broaden012")
    p.add_argument("--exper-dir", default=DEF_EXPER)
    p.add_argument("--catalog-dir",
                   default=("/scratch/cavestru_root/cavestru0/mfho/"
                            "gl_prod_2lpt0_v1_20260526/combined_catalog/"))
    p.add_argument("--truth",
                   default=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                            "qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits"))
    p.add_argument("--bal-cat",
                   default=("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                            "qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"))
    p.add_argument("--molly-tsv", default=None)
    p.add_argument("--kernel",
                   default=("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                            "phase3d_experiments/mollynhi195_lyaonly1025_broaden012/"
                            "posterior_kernel_2lpt0.npz"))
    p.add_argument("--loa0-product",
                   default=("/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/"
                            "outputs/loa0_fp_product_lyaonly1025.npz"))
    p.add_argument("--out",
                   default="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                           "wall1_explain_partB")
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3,20.6")
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--fit-ceil", type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes", type=int, default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    limits = tuple(float(x) for x in args.report_limits.split(","))

    # ---- (1) extract cached WALL-1 closure (Delta-alpha=+/-0.5) ----
    res = parse_wall1_result(os.path.join(args.exper_dir, "wall1_result.tsv"))
    fN = parse_wall1_pulls_fN(os.path.join(args.exper_dir, "wall1_pulls_fN.csv"))
    print("=" * 70)
    print("CACHED WALL-1 (Delta-alpha=+/-0.5, calibrated kernel + lya_only nhi195 molly)")
    print("=" * 70)
    print(f"  VERDICT = {res['WALL1_VERDICT']}")
    print(f"  classifier = {res['WALL1_FAIL_CLASSIFICATION'][:90]}...")
    for l in limits:
        print(f"  dN/dX(>={l}) closure pull: +tilt={res[f'dndx_total_closure_pull_{l}_plus']}, "
              f"-tilt={res[f'dndx_total_closure_pull_{l}_minus']}")
        print(f"  Omega(>={l}) closure pull: +tilt={res[f'omega_closure_pull_{l}_plus']}, "
              f"-tilt={res[f'omega_closure_pull_{l}_minus']}")

    # ---- (2) recompute the TRUTH-SIDE tilt mechanism (fast, no MC) ----
    print("\n[partB] recompute tilted-truth reductions (truth-side mechanism)")
    ing = build_ingredients(args, "purity_mixture")  # tilt is wired for PM; truth-side is FP-agnostic
    cfg = ing["cfg"]; cfg.report_logN_limits = limits
    logN_lo = ing["logN_lo"]; logN_hi = ing["logN_hi"]
    N_b = ing["N_b"]; dN_b = ing["dN_b"]; X_tot = ing["X_tot"]; truth_cut = ing["truth_cut"]
    mid = 0.5 * (logN_lo + logN_hi)

    da_vals = {"plus": +0.5, "minus": -0.5, "real": 0.015, "zero": 0.0}
    ttr = {}
    for tag, da in da_vals.items():
        ttr[tag] = tilted_truth_reductions(cfg, truth_cut, logN_lo, logN_hi,
                                           N_b, dN_b, X_tot, da)
    # the tilt weight curve over the grid (schematic annotation)
    w_plus = tilt_weight(mid, +0.5)
    w_minus = tilt_weight(mid, -0.5)
    w_real = tilt_weight(mid, 0.015)

    savez = dict(
        logN_lo=logN_lo, logN_hi=logN_hi, mid=mid, N_b=N_b, dN_b=dN_b,
        zbins=np.asarray(cfg.zbins, float), report_limits=np.asarray(limits),
        pivot=LOGN_PIVOT,
        # tilt weight curves
        w_plus=w_plus, w_minus=w_minus, w_real=w_real,
        # truth-side f(N) at each tilt
        f_truth_zero=ttr["zero"]["f_truth"],
        f_truth_plus=ttr["plus"]["f_truth"],
        f_truth_minus=ttr["minus"]["f_truth"],
        f_truth_real=ttr["real"]["f_truth"],
        # closure arrays from the cached fit (Delta-alpha=+/-0.5)
        cached_R0=fN["R0"],
        cached_ftrue_plus=fN["ftrue_plus"], cached_fpred_plus=fN["fpred_plus"],
        cached_fest_plus=fN["fest_plus"], cached_fstd_plus=fN["fstd_plus"],
        cached_pull_closure_plus=fN["pull_closure_plus"],
        cached_pull_raw_plus=fN["pull_raw_plus"],
        cached_ftrue_minus=fN["ftrue_minus"], cached_fpred_minus=fN["fpred_minus"],
        cached_fest_minus=fN["fest_minus"], cached_fstd_minus=fN["fstd_minus"],
        cached_pull_closure_minus=fN["pull_closure_minus"],
        cached_pull_raw_minus=fN["pull_raw_minus"],
        cached_gated=fN["gated"],
    )
    for tag in ("plus", "minus", "real", "zero"):
        for l in limits:
            savez[f"truth_dndx_{tag}_{l}"] = float(ttr[tag]["dndx_total"][l])
            savez[f"truth_omega_{tag}_{l}"] = float(ttr[tag]["omega"][l])
        savez[f"truth_dndx_z_{tag}"] = np.array([ttr[tag]["dndx_z"][l] for l in limits])
    # cached integrated closure pulls + verdict (scalars as strings -> floats)
    for l in limits:
        for q in ("dndx_total", "omega"):
            for sgn in ("plus", "minus"):
                key = f"{q}_closure_pull_{l}_{sgn}"
                savez[f"cached_{key}"] = float(res[key])
                rawkey = f"{q}_raw_pull_{l}_{sgn}" if q == "dndx_total" else None
            savez[f"cached_baseline_R0_dndx_{l}"] = float(res[f"baseline_R0_dndx_{l}"])
            savez[f"cached_baseline_R0_omega_{l}"] = float(res[f"baseline_R0_omega_{l}"])
    savez["cached_verdict"] = res["WALL1_VERDICT"]
    savez["cached_classifier"] = res["WALL1_FAIL_CLASSIFICATION"]
    out_npz = os.path.join(args.out, "partB_tilt.npz")
    np.savez(out_npz, **savez)
    print(f"\n[partB] saved -> {out_npz}")

    # console: integrated tilt mechanism numbers
    print("\nTRUTH-side integrated dN/dX under tilt (injected target before R0):")
    for l in limits:
        print(f"  >={l}: zero={ttr['zero']['dndx_total'][l]:.5f}  "
              f"+0.5={ttr['plus']['dndx_total'][l]:.5f}  "
              f"-0.5={ttr['minus']['dndx_total'][l]:.5f}  "
              f"real(0.015)={ttr['real']['dndx_total'][l]:.5f}")
    return out_npz


if __name__ == "__main__":
    main()
