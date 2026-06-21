#!/usr/bin/env python
"""track_c_td_band.py — Track-C T-D DELIVERABLE: the forward-empirical kernel in the
MARGINALIZED BAND with the kernel-calibration uncertainty carry.

On the 2LPT-0 PM (purity_mixture) bundle, builds the marginalized band for dN/dX(z),
Ω(z) and f(N) at NHI > 20 (>=20.0 / >=20.3) with:
  * the forward-empirical kernel (resp_kind=forward, resp_family=empirical),
  * Stage I  = inner Laplace (mc_inner=laplace),
  * Stage II = shared truth-match bootstrap (mc_nuisance=shared_boot),
  * Stage III (T-D) = per-draw RE-FIT of the ForwardResponseModel on the SAME shared
    boot_mult (mc_response=marginalize) — the kernel-calibration uncertainty carry.

Reports per limit: the MAP point R0 (the T-BC headline; dN/dX≈1, Ω≈0.885), the band
68/95, whether truth sits in the band (the on-mock coverage question, scoped NHI>20),
the deep-tail [21.3+] Ω band (should be WIDER — data-starvation as honest uncertainty),
and the band WIDTH WITH vs WITHOUT the kernel-uncertainty carry (mc_response
marginalize vs frozen) — the toy's flagged widening.

DISCIPLINE: analysis-side only; inference (gpy_dla_detection/) byte-frozen. The default
(resp_kind=kappa) band is byte-identical; this driver is the forward-band path only.

Run:
    python -m CDDF_analysis.track_c_td_band \
        --forward-model /scratch/.../track_c/stage0/forward_response_2lpt0.npz \
        --n-mc 120 --out /scratch/.../track_c/td_band
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis import ab_loa0_fp_baseline as AB
from CDDF_analysis.ab_loa0_fp_baseline import build_ingredients, run_baseline
from CDDF_analysis.cddf_catalog_hbi import (
    joint_mc_errors, make_v3x_refit_fn, build_truth_match_resample,
    omega_hi_prefactor)

_DEF_FORWARD = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                "track_c/stage0/forward_response_2lpt0.npz")


def _band(samp):
    s = np.asarray(samp, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return dict(q025=np.nan, q16=np.nan, q50=np.nan, q84=np.nan, q975=np.nan,
                    std=np.nan, n=0)
    return dict(q025=float(np.percentile(s, 2.5)), q16=float(np.percentile(s, 16.0)),
                q50=float(np.percentile(s, 50.0)), q84=float(np.percentile(s, 84.0)),
                q975=float(np.percentile(s, 97.5)), std=float(np.std(s)), n=int(s.size))


def _cover(band, truth):
    """does truth sit inside the 68%/95% band?"""
    return dict(in68=bool(band["q16"] <= truth <= band["q84"]),
                in95=bool(band["q025"] <= truth <= band["q975"]))


def _set_forward_band_cfg(cfg, forward_model, resp_family, carry):
    """Configure the forward-empirical marginalized band on cfg."""
    cfg.resp_kind = "forward"
    cfg.resp_family = resp_family
    cfg.kernel_forward_model = forward_model
    cfg.mc_inner = "laplace"
    cfg.mc_nuisance = "shared_boot"
    cfg.mc_response = "marginalize" if carry else "frozen"


def run_forward_band(args, limits, carry, seed):
    """Build the PM forward-empirical marginalized band; return point R0 + band samples.

    ``carry`` True => Stage-III kernel-calibration uncertainty ON (mc_response=marginalize);
    False => the same band with the forward kernel FROZEN at the point (mc_response=frozen),
    so the WIDTH difference isolates the kernel-uncertainty carry."""
    t0 = time.time()
    ing = build_ingredients(args, "purity_mixture")
    cfg = ing["cfg"]
    cfg.report_logN_limits = limits
    cfg._wall1_estimator = "v3"
    cfg.n_mc = args.n_mc
    _set_forward_band_cfg(cfg, args.forward_model, args.resp_family, carry)

    # ---- single-source point + truth + R0 (the forward MAP headline) ----
    base = run_baseline(ing)
    e0 = base["e0"]; t0r = base["t0"]
    point = e0
    logN_lo = ing["logN_lo"]; logN_hi = ing["logN_hi"]
    mid = 0.5 * (logN_lo + logN_hi)

    # ---- the marginalized band (PM wired parametric joint_mc_errors + forward Stage III) ----
    # build the SHARED truth-match resample ONCE so the joint_mc_errors C/ρ/g draws AND the
    # Stage-III forward refit use the SAME tmr.uniq_tids (the joint correlation).
    tmr = build_truth_match_resample(
        ing["mm"], ing["cat_cut"], ing["is_TP"], ing["truth_cut"], ing["good_mask"], cfg)
    refit_fn = make_v3x_refit_fn(cfg, point["_v3x"], ing["mm"],
                                 cat_cut=ing["cat_cut"], good_mask=ing["good_mask"], tmr=tmr)
    mc = joint_mc_errors(
        ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["mm"], ing["fp_model"],
        ing["X_tot"], logN_lo, logN_hi, ing["N_b"], ing["dN_b"], ing["truth_cut"],
        cfg, np.random.default_rng(seed + 4), refit_fn=refit_fn, tmr=tmr)
    print(f"    [carry={carry}] band done ({time.time()-t0:.0f}s)")

    return dict(
        mid=mid, logN_lo=logN_lo, logN_hi=logN_hi, N_b=ing["N_b"], dN_b=ing["dN_b"],
        point_dndx={l: float(e0["dndx_total"][l]) for l in limits},
        point_omega={l: float(e0["omega"][l]) for l in limits},
        truth_dndx={l: float(t0r["dndx_total"][l]) for l in limits},
        truth_omega={l: float(t0r["omega"][l]) for l in limits},
        R0_dndx={l: float(base["R0_dndx_total"][l]) for l in limits},
        R0_omega={l: float(base["R0_omega"][l]) for l in limits},
        f_b_point=np.asarray(e0["f_b"], float),
        f_truth=np.asarray(t0r["f_truth"], float),
        dndx_samples={l: np.asarray(mc["_samples"]["dndx_total"][l], float) for l in limits},
        omega_samples={l: np.asarray(mc["_samples"]["omega"][l], float) for l in limits},
        f_b_samples=np.asarray(mc["_samples"]["f_b"], float),
        cfg=cfg, H0=cfg.H0,
    )


def _deep_tail_omega_samples(res, lo=21.3):
    """Per-draw Ω restricted to NHI >= lo, from the per-bin f_b samples (the data-starved
    deep tail). Ω = K Σ N f(N) dN over bins with logN_lo >= lo."""
    K = omega_hi_prefactor(res["H0"])
    lo_b = res["logN_lo"]; N_b = res["N_b"]; dN_b = res["dN_b"]
    sel = lo_b >= lo - 1e-9
    fb = res["f_b_samples"]                      # (n_mc, n_nbins)
    om = K * np.nansum(N_b[None, sel] * fb[:, sel] * dN_b[None, sel], axis=1)
    om_truth = K * float(np.nansum(N_b[sel] * res["f_truth"][sel] * dN_b[sel]))
    om_point = K * float(np.nansum(N_b[sel] * res["f_b_point"][sel] * dN_b[sel]))
    return om, om_truth, om_point


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir", default=AB.DEF_CAT)
    p.add_argument("--truth", default=AB.DEF_TRUTH)
    p.add_argument("--bal-cat", default=AB.DEF_BAL)
    p.add_argument("--molly-tsv", default=AB.DEF_LYAONLY_MOLLY)
    p.add_argument("--kernel", default=AB.DEF_KERNEL)
    p.add_argument("--loa0-product", default=AB.DEF_LOA0_PRODUCT)
    p.add_argument("--forward-model", default=_DEF_FORWARD,
                   help="ForwardResponseModel NPZ (save_forward_response / T-A build).")
    p.add_argument("--resp-family", default="empirical",
                   choices=["skewnorm", "empirical"])
    p.add_argument("--out",
                   default="/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                           "track_c/td_band")
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3")
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--fit-ceil", type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes", type=int, default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    p.add_argument("--n-mc", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deep-tail-lo", type=float, default=21.3)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    limits = tuple(float(x) for x in args.report_limits.split(","))

    print("=" * 74)
    print("TRACK-C T-D — forward-empirical MARGINALIZED band (PM, 2LPT-0)")
    print(f"  forward model: {args.forward_model}  family={args.resp_family}")
    print(f"  Stage I=laplace  Stage II=shared_boot  Stage III=marginalize(forward refit)")
    print(f"  n_mc={args.n_mc}")
    print("=" * 74)
    print("[1/2] WITH kernel-calibration uncertainty carry (mc_response=marginalize)")
    res_carry = run_forward_band(args, limits, carry=True, seed=args.seed)
    print("[2/2] WITHOUT carry (mc_response=frozen — forward kernel frozen at point)")
    res_frozen = run_forward_band(args, limits, carry=False, seed=args.seed)

    # ---------- assemble + print the band table ----------
    rows = []
    print("\n" + "=" * 74)
    print("MARGINALIZED BAND (forward-empirical kernel + carry) — dN/dX & Ω, NHI>20")
    print("=" * 74)
    print(f"{'q':>6} {'lim':>5} | {'MAP R0':>8} | {'68% band':>21} | "
          f"{'95% band':>21} | cover68 cover95")
    print("-" * 92)
    summary = dict(metadata=dict(
        forward_model=args.forward_model, resp_family=args.resp_family,
        n_mc=args.n_mc, seed=args.seed, limits=list(limits),
        kernel=args.kernel, molly=args.molly_tsv, truth=args.truth,
        deep_tail_lo=args.deep_tail_lo,
        stages="I=laplace, II=shared_boot, III=marginalize(forward refit)"))
    for kind, samp_key, pt_key, tr_key, r0_key in (
            ("dndx", "dndx_samples", "point_dndx", "truth_dndx", "R0_dndx"),
            ("omega", "omega_samples", "point_omega", "truth_omega", "R0_omega")):
        for l in limits:
            band = _band(res_carry[samp_key][l])
            truth = res_carry[tr_key][l]
            cov = _cover(band, truth)
            r0 = res_carry[r0_key][l]
            print(f"{kind:>6} {l:>5} | {r0:>8.3f} | "
                  f"[{band['q16']:.4g}, {band['q84']:.4g}] | "
                  f"[{band['q025']:.4g}, {band['q975']:.4g}] | "
                  f"{str(cov['in68']):>7} {str(cov['in95']):>7}")
            band_frozen = _band(res_frozen[samp_key][l])
            w_carry = band["q84"] - band["q16"]
            w_frozen = band_frozen["q84"] - band_frozen["q16"]
            summary.setdefault(kind, {})[str(l)] = dict(
                MAP_R0=float(r0), point=float(res_carry[pt_key][l]),
                truth=float(truth), band68=[band["q16"], band["q84"]],
                band95=[band["q025"], band["q975"]],
                cover68=cov["in68"], cover95=cov["in95"],
                width68_carry=float(w_carry), width68_frozen=float(w_frozen),
                carry_widens=bool(w_carry >= w_frozen))

    # deep-tail Ω band (data-starvation as honest uncertainty)
    om_dt, om_dt_truth, om_dt_point = _deep_tail_omega_samples(res_carry, args.deep_tail_lo)
    om_dt_band = _band(om_dt)
    om_dt_cov = _cover(om_dt_band, om_dt_truth)
    om_dt_f, _, _ = _deep_tail_omega_samples(res_frozen, args.deep_tail_lo)
    om_dt_band_f = _band(om_dt_f)
    print("-" * 92)
    print(f"DEEP-TAIL Ω (NHI>={args.deep_tail_lo}):  MAP={om_dt_point:.4g} "
          f"truth={om_dt_truth:.4g}  68%=[{om_dt_band['q16']:.4g},{om_dt_band['q84']:.4g}]"
          f"  cover68={om_dt_cov['in68']}")
    print(f"  width68 carry={om_dt_band['q84']-om_dt_band['q16']:.4g}  "
          f"frozen={om_dt_band_f['q84']-om_dt_band_f['q16']:.4g}  "
          f"(carry widens deep tail = honest data-starvation uncertainty)")
    summary["omega_deep_tail"] = dict(
        lo=float(args.deep_tail_lo), MAP=float(om_dt_point), truth=float(om_dt_truth),
        band68=[om_dt_band["q16"], om_dt_band["q84"]],
        band95=[om_dt_band["q025"], om_dt_band["q975"]],
        cover68=om_dt_cov["in68"], cover95=om_dt_cov["in95"],
        width68_carry=float(om_dt_band["q84"] - om_dt_band["q16"]),
        width68_frozen=float(om_dt_band_f["q84"] - om_dt_band_f["q16"]))

    # ---- carry-widening summary ----
    print("\n" + "=" * 74)
    print("KERNEL-UNCERTAINTY CARRY — 68% band WIDTH (carry vs frozen)")
    print("=" * 74)
    print(f"{'q':>6} {'lim':>5} | {'width carry':>12} | {'width frozen':>12} | widens?")
    print("-" * 56)
    for kind in ("dndx", "omega"):
        for l in limits:
            s = summary[kind][str(l)]
            print(f"{kind:>6} {l:>5} | {s['width68_carry']:>12.5g} | "
                  f"{s['width68_frozen']:>12.5g} | {s['carry_widens']}")

    out_json = os.path.join(args.out, "td_band.json")
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    # f(N) per-bin band (carry) for the differential figure
    mid = res_carry["mid"]
    fb_samp = res_carry["f_b_samples"]
    fN = []
    for b in range(len(mid)):
        if not (20.0 - 1e-9 <= mid[b] <= 22.0 + 1e-9):
            continue
        col = fb_samp[:, b]
        fb_band = _band(col)
        fN.append(dict(logN_mid=float(mid[b]), hbi=float(res_carry["f_b_point"][b]),
                       truth=float(res_carry["f_truth"][b]),
                       band68=[fb_band["q16"], fb_band["q84"]],
                       band95=[fb_band["q025"], fb_band["q975"]]))
    with open(os.path.join(args.out, "td_band_fN.json"), "w") as fh:
        json.dump(dict(metadata=summary["metadata"], f_b=fN), fh, indent=2)
    print(f"\nsaved -> {out_json}")
    print(f"saved -> {os.path.join(args.out, 'td_band_fN.json')}")
    return summary


if __name__ == "__main__":
    main()
