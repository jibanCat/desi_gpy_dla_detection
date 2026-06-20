#!/usr/bin/env python
"""track_c_tbc_smoke.py — Track-C T-BC SMOKE: forward-response vs kappa deconvolution.

Runs the v3 (bspbody) UNTILTED point estimate on the 2LPT-0 broaden012 bundle TWICE:

  (1) resp_kind="kappa"   — the DEFAULT (the cached GP posterior kappa2d kernel), and
  (2) resp_kind="forward" — the T-BC forward-LIKELIHOOD deconvolution kernel A built from
      the T-A ForwardResponseModel skew-normal density p(x̂_i | N, SNR_i, z_i).

…then prints the recovered dN/dX and Ω against 2LPT-0 truth (R0 = est/truth) per report
limit. This is a SMOKE (a rough R0 sanity), NOT the full WALL-1/SBC validation (that is T-F).

The certified expectation (notes/2026-06-20_track_c_forward_toy_certificate.md): the narrow
kappa posterior OVER-recovers the high-N tail (the +9% / 21.0–21.5 shoulder, Ω over); the
forward kernel should move the high-N over-recovery TOWARD truth.

DISCIPLINE: analysis-side only; inference (gpy_dla_detection/) byte-frozen. The forward
path is fully gated — the kappa run is byte-identical to the historical headline.

Run:
    python -m CDDF_analysis.track_c_tbc_smoke \
        --forward-model /scratch/.../track_c/stage0/forward_response_2lpt0.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis import ab_loa0_fp_baseline as AB

_DEF_FORWARD = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                "track_c/stage0/forward_response_2lpt0.npz")


def _run_one(args, resp_kind, forward_model, limits):
    ing = AB.build_ingredients(args, "purity_mixture")
    cfg = ing["cfg"]
    cfg.resp_kind = resp_kind
    cfg.kernel_forward_model = forward_model if resp_kind == "forward" else None
    cfg.resp_family = args.resp_family
    base = AB.run_baseline(ing)
    return dict(
        R0_dndx={lim: float(base["R0_dndx_total"][lim]) for lim in limits},
        R0_omega={lim: float(base["R0_omega"][lim]) for lim in limits},
        dndx_est={lim: float(base["e0"]["dndx_total"][lim]) for lim in limits},
        dndx_truth={lim: float(base["t0"]["dndx_total"][lim]) for lim in limits},
        omega_est={lim: float(base["e0"]["omega"][lim]) for lim in limits},
        omega_truth={lim: float(base["t0"]["omega"][lim]) for lim in limits},
        f_b=np.asarray(base["e0"]["f_b"]),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir", default=AB.DEF_CAT)
    p.add_argument("--truth", default=AB.DEF_TRUTH)
    p.add_argument("--bal-cat", default=AB.DEF_BAL)
    p.add_argument("--molly-tsv", default=None)
    p.add_argument("--kernel", default=AB.DEF_KERNEL)
    p.add_argument("--loa0-product", default=AB.DEF_LOA0_PRODUCT)
    p.add_argument("--forward-model", default=_DEF_FORWARD,
                   help="ForwardResponseModel NPZ (save_forward_response / T-A build).")
    p.add_argument("--resp-family", default="skewnorm",
                   choices=["skewnorm", "empirical"])
    p.add_argument("--out", default="/tmp/track_c_tbc_smoke")
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3,20.6,21.0")
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

    print("=" * 72)
    print("[1/2] resp_kind=kappa (DEFAULT, byte-identical headline)")
    print("=" * 72)
    res_k = _run_one(args, "kappa", None, limits)
    print("=" * 72)
    print(f"[2/2] resp_kind=forward (T-BC; family={args.resp_family})")
    print(f"      forward model: {args.forward_model}")
    print("=" * 72)
    res_f = _run_one(args, "forward", args.forward_model, limits)

    print("\n" + "=" * 72)
    print("TRACK-C T-BC SMOKE — dN/dX & Ω recovery (R0=est/truth), v3 bspbody, 2LPT-0")
    print("=" * 72)
    print(f"{'limit':>6} | {'dN/dX R0':>22} | {'Omega R0':>22}")
    print(f"{'':>6} | {'kappa    forward':>22} | {'kappa    forward':>22}")
    print("-" * 60)
    for lim in limits:
        print(f"{lim:>6} | {res_k['R0_dndx'][lim]:>9.4f} {res_f['R0_dndx'][lim]:>11.4f} "
              f"| {res_k['R0_omega'][lim]:>9.4f} {res_f['R0_omega'][lim]:>11.4f}")
    print("-" * 60)
    print("interpretation: the certified expectation is that forward MOVES the high-N "
          "over-recovery\n(kappa R0>1 at >=20.3/Omega) TOWARD 1. Finite f and a reduced "
          "over-recovery = smoke PASS.")

    # finiteness guard
    assert np.all(np.isfinite(res_f["f_b"])), "forward f_b has non-finite entries!"
    np.savez(os.path.join(args.out, "tbc_smoke.npz"),
             limits=np.asarray(limits),
             kappa_R0_dndx=np.array([res_k["R0_dndx"][l] for l in limits]),
             forward_R0_dndx=np.array([res_f["R0_dndx"][l] for l in limits]),
             kappa_R0_omega=np.array([res_k["R0_omega"][l] for l in limits]),
             forward_R0_omega=np.array([res_f["R0_omega"][l] for l in limits]),
             kappa_dndx_est=np.array([res_k["dndx_est"][l] for l in limits]),
             forward_dndx_est=np.array([res_f["dndx_est"][l] for l in limits]),
             dndx_truth=np.array([res_k["dndx_truth"][l] for l in limits]),
             forward_f_b=res_f["f_b"], kappa_f_b=res_k["f_b"])
    print(f"\nsaved -> {args.out}/tbc_smoke.npz")


if __name__ == "__main__":
    main()
