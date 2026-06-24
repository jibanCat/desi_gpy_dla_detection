"""subdla_loa0_validation_floor190.py — floor-19.0 variant of the sub-DLA edge
recovery validation (reduce-only, cached kernel, NO inference, NO SLURM, NO tilt).

IDENTICAL to subdla_loa0_validation.py EXCEPT the three repointed knobs the task
specifies:
  * kernel       -> floor190_lyaonly1025_broaden012/posterior_kernel_2lpt0.npz
  * molly_tsv    -> figures_molly_nhi190/molly_matrix.tsv  (floor-19.0 lya_only)
  * fit_floor    -> 19.0  (gives sub-floor headroom below the 19.5 report edge)

Everything else (cat_cut, frozen C/ρ, pathlength, loa-0 FP product, family bspbody,
σ_add already baked into the kernel build, lam_rf_min=1025, report band [19.5,20.3))
is held verbatim against subdla_loa0_validation.py so the comparison is apples-to-apples.

The floor-19.5 reference numbers (from the prior run) are hardcoded for the
side-by-side table.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
from CDDF_analysis.hbi.cddf_tilt_closure import baseline_recovery

# the floor-19.0 rebuild inputs (the task's three repointed knobs)
FLOOR190_KERNEL = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                   "phase3d_experiments/floor190_lyaonly1025_broaden012/"
                   "posterior_kernel_2lpt0.npz")
FLOOR190_MOLLY = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
                  "figures_molly_nhi190/molly_matrix.tsv")
LOA0_PRODUCT = ("/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/"
                "outputs/loa0_fp_product_lyaonly1025.npz")

OUT_DIR = "/tmp/subdla_loa0_validation_floor190"

# cumulative report limits: 19.5 floor + 0.1-dex steps through 20.3, then the DLA tier
REPORT_LIMITS = (19.5, 19.6, 19.7, 19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.6)

# floor-19.5 reference (from the prior validation run) — for the side-by-side table
F195_PER_BIN_R0 = [0.454, 0.717, 0.893, 0.932, 0.901, 0.903, 0.923, 1.028]
F195_DNDX_195_203 = 0.883
F195_OMEGA_195_203 = 0.899
F195_DNDX_203 = 1.159
F195_OMEGA_203 = 1.114


class _Args:
    """Mirror ab_loa0_fp_baseline argparse defaults, repointed to floor-19.0."""
    def __init__(self):
        self.catalog_dir = AB.DEF_CAT
        self.truth = AB.DEF_TRUTH
        self.bal_cat = AB.DEF_BAL
        self.molly_tsv = FLOOR190_MOLLY     # <-- repointed: floor-19.0 lya_only molly
        self.kernel = FLOOR190_KERNEL       # <-- repointed: floor-19.0 kernel
        self.loa0_product = LOA0_PRODUCT
        self.out = OUT_DIR
        self.mockdir = None
        self.zbins = "2.0,2.5,3.0,3.5"
        self.report_limits = ",".join(f"{x:g}" for x in REPORT_LIMITS)
        self.family = "bspbody"
        self.fit_floor = 19.0               # <-- repointed: sub-floor headroom
        self.fit_ceil = 99.0
        self.lambda_bspbody = 30.0
        self.lam_rf_min = 1025.0
        self.edge_slope_lam = 40.0
        self.gl_nodes = 1
        self.host_truth_floor = 19.0


def run_mode(mode: str) -> dict:
    args = _Args()
    os.makedirs(args.out, exist_ok=True)
    print("=" * 78)
    print(f"[sub-DLA validation FLOOR-19.0] fp_estimator = {mode}")
    print("=" * 78)
    ing = AB.build_ingredients(args, mode, loa0_product=args.loa0_product)
    cfg = ing["cfg"]
    cfg._wall1_estimator = "v3"
    base = baseline_recovery(
        cfg, ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["truth_cut"],
        ing["C_interp"], ing["fp_model"], ing["X_tot"],
        ing["logN_lo"], ing["logN_hi"], ing["N_b"], ing["dN_b"],
        estimator_fn=ing["estimator_fn"])

    logN_lo = np.asarray(ing["logN_lo"], float)
    logN_hi = np.asarray(ing["logN_hi"], float)
    dN_b = np.asarray(ing["dN_b"], float)
    f_est = np.asarray(base["e0"]["f_b"], float)
    f_tru = np.asarray(base["t0"]["f_truth"], float)

    bins = [(round(19.5 + 0.1 * k, 1), round(19.6 + 0.1 * k, 1)) for k in range(8)]
    per_bin = []
    for blo, bhi in bins:
        sel = (logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)
        fe = float(np.nansum(f_est[sel]))
        ft = float(np.nansum(f_tru[sel]))
        dndx_e = float(np.nansum(f_est[sel] * dN_b[sel]))
        dndx_t = float(np.nansum(f_tru[sel] * dN_b[sel]))
        r0 = (dndx_e / dndx_t) if dndx_t > 0 else np.nan
        per_bin.append(dict(blo=blo, bhi=bhi, f_est=fe, f_tru=ft,
                            dndx_est=dndx_e, dndx_tru=dndx_t, r0=r0))

    dndx_e_195_203 = (base["e0"]["dndx_total"][19.5] - base["e0"]["dndx_total"][20.3])
    dndx_t_195_203 = (base["t0"]["dndx_total"][19.5] - base["t0"]["dndx_total"][20.3])
    om_e_195_203 = (base["e0"]["omega"][19.5] - base["e0"]["omega"][20.3])
    om_t_195_203 = (base["t0"]["omega"][19.5] - base["t0"]["omega"][20.3])
    dndx_e_195_200 = (base["e0"]["dndx_total"][19.5] - base["e0"]["dndx_total"][20.0])
    dndx_t_195_200 = (base["t0"]["dndx_total"][19.5] - base["t0"]["dndx_total"][20.0])
    om_e_195_200 = (base["e0"]["omega"][19.5] - base["e0"]["omega"][20.0])
    om_t_195_200 = (base["t0"]["omega"][19.5] - base["t0"]["omega"][20.0])

    return dict(
        mode=mode, n_sl=int(ing["n_sl"]),
        per_bin=per_bin,
        dndx_est_195_203=dndx_e_195_203, dndx_tru_195_203=dndx_t_195_203,
        r0_dndx_195_203=(dndx_e_195_203 / dndx_t_195_203) if dndx_t_195_203 > 0 else np.nan,
        omega_est_195_203=om_e_195_203, omega_tru_195_203=om_t_195_203,
        r0_omega_195_203=(om_e_195_203 / om_t_195_203) if om_t_195_203 > 0 else np.nan,
        dndx_est_195_200=dndx_e_195_200, dndx_tru_195_200=dndx_t_195_200,
        r0_dndx_195_200=(dndx_e_195_200 / dndx_t_195_200) if dndx_t_195_200 > 0 else np.nan,
        omega_est_195_200=om_e_195_200, omega_tru_195_200=om_t_195_200,
        r0_omega_195_200=(om_e_195_200 / om_t_195_200) if om_t_195_200 > 0 else np.nan,
        dndx_est_203=base["e0"]["dndx_total"][20.3], dndx_tru_203=base["t0"]["dndx_total"][20.3],
        r0_dndx_203=base["R0_dndx_total"][20.3], r0_omega_203=base["R0_omega"][20.3],
        r0_dndx_200=base["R0_dndx_total"][20.0], r0_omega_200=base["R0_omega"][20.0],
    )


def main():
    res = {m: run_mode(m) for m in ("purity_mixture", "loa0")}

    def _fmt(x, w=10, p=4):
        return f"{x:>{w}.{p}f}" if np.isfinite(x) else f"{'nan':>{w}}"

    lo = res["loa0"]["per_bin"]
    pm = res["purity_mixture"]["per_bin"]

    # ---- side-by-side per-0.1-dex: floor-19.5 vs floor-19.0 (loa0 estimator) ----
    print("\n" + "=" * 78)
    print("PER-0.1-dex R0 (loa0) : FLOOR-19.5 (prior) vs FLOOR-19.0 (this run)")
    print("=" * 78)
    print(f"{'bin':>14} | {'floor19.5 R0':>14} | {'floor19.0 R0':>14} | {'delta':>10}")
    print("-" * 78)
    for k, bl in enumerate(lo):
        lab = f"[{bl['blo']:.1f},{bl['bhi']:.1f})"
        r195 = F195_PER_BIN_R0[k]
        d = bl["r0"] - r195
        print(f"{lab:>14} | {r195:>14.3f} | {_fmt(bl['r0'],14,3)} | {d:>+10.3f}")

    print("\n" + "=" * 78)
    print("PER-0.1-dex R0 (floor-19.0) : purity_mixture vs loa0")
    print("=" * 78)
    print(f"{'bin':>14} | {'truth dndx':>12} | {'pm R0':>10} | {'loa0 R0':>10}")
    print("-" * 78)
    for bp, bl in zip(pm, lo):
        lab = f"[{bp['blo']:.1f},{bp['bhi']:.1f})"
        print(f"{lab:>14} | {_fmt(bp['dndx_tru'],12,6)} | "
              f"{_fmt(bp['r0'],10,3)} | {_fmt(bl['r0'],10,3)}")

    # ---- integrated bands: floor-19.5 vs floor-19.0 (loa0) ----
    L = res["loa0"]
    print("\n" + "=" * 78)
    print("INTEGRATED R0 (loa0) : FLOOR-19.5 (prior) vs FLOOR-19.0 (this run)")
    print("=" * 78)
    print(f"{'band/metric':>26} | {'floor19.5':>12} | {'floor19.0':>12} | {'delta':>10}")
    print("-" * 78)
    rows195 = [
        ("dN/dX [19.5,20.3)", F195_DNDX_195_203, L["r0_dndx_195_203"]),
        ("Omega [19.5,20.3)", F195_OMEGA_195_203, L["r0_omega_195_203"]),
        ("dN/dX>=20.3 (DLA)", F195_DNDX_203, L["r0_dndx_203"]),
        ("Omega>=20.3 (DLA)", F195_OMEGA_203, L["r0_omega_203"]),
    ]
    for lab, r195, r190 in rows195:
        print(f"{lab:>26} | {r195:>12.4f} | {_fmt(r190,12,4)} | {(r190-r195):>+10.4f}")
    # the [19.5,20.0) band has no floor-19.5 reference number given; show it standalone
    print(f"{'dN/dX [19.5,20.0)':>26} | {'(n/a)':>12} | {_fmt(L['r0_dndx_195_200'],12,4)} |")
    print(f"{'Omega [19.5,20.0)':>26} | {'(n/a)':>12} | {_fmt(L['r0_omega_195_200'],12,4)} |")

    print("\n" + "=" * 78)
    print("INTEGRATED BANDS — recovered vs truth, R0, both FP estimators (floor-19.0)")
    print("=" * 78)
    for band, ek, tk, rk in (
        ("dN/dX [19.5,20.3)", "dndx_est_195_203", "dndx_tru_195_203", "r0_dndx_195_203"),
        ("Omega [19.5,20.3)", "omega_est_195_203", "omega_tru_195_203", "r0_omega_195_203"),
        ("dN/dX [19.5,20.0)", "dndx_est_195_200", "dndx_tru_195_200", "r0_dndx_195_200"),
        ("Omega [19.5,20.0)", "omega_est_195_200", "omega_tru_195_200", "r0_omega_195_200"),
    ):
        t = res["purity_mixture"][tk]
        print(f"\n--- {band} ---  truth = {t:.6g}")
        for m in ("purity_mixture", "loa0"):
            r = res[m]
            print(f"    {m:>16}: est={r[ek]:.6g}  R0={r[rk]:.4f}")

    print("\n" + "=" * 78)
    print("DLA-TIER CONTEXT (>=20.3 / >=20.0 cumulative) — FP~=0, should be unchanged")
    print("=" * 78)
    for m in ("purity_mixture", "loa0"):
        r = res[m]
        print(f"    {m:>16}: R0_dndx(>=20.3)={r['r0_dndx_203']:.4f}  "
              f"R0_omega(>=20.3)={r['r0_omega_203']:.4f}  "
              f"R0_dndx(>=20.0)={r['r0_dndx_200']:.4f}  "
              f"R0_omega(>=20.0)={r['r0_omega_200']:.4f}  n_sl={r['n_sl']}")

    # ---- persist TSV (floor-19.5 vs floor-19.0 side by side, loa0 estimator) ----
    out_tsv = os.path.join(OUT_DIR, "subdla_validation_floor190.tsv")
    with open(out_tsv, "w") as fh:
        fh.write("metric\tbin\ttruth\tfloor195_loa0\tfloor190_loa0\tfloor190_pm\n")
        for k, (bp, bl) in enumerate(zip(pm, lo)):
            lab = f"[{bp['blo']:.1f},{bp['bhi']:.1f})"
            fh.write(f"r0_dndx_bin\t{lab}\t1.0\t{F195_PER_BIN_R0[k]:.6g}\t"
                     f"{bl['r0']:.6g}\t{bp['r0']:.6g}\n")
        fh.write(f"R0_dndx_195_203\t-\t1.0\t{F195_DNDX_195_203:.6g}\t"
                 f"{L['r0_dndx_195_203']:.6g}\t{res['purity_mixture']['r0_dndx_195_203']:.6g}\n")
        fh.write(f"R0_omega_195_203\t-\t1.0\t{F195_OMEGA_195_203:.6g}\t"
                 f"{L['r0_omega_195_203']:.6g}\t{res['purity_mixture']['r0_omega_195_203']:.6g}\n")
        fh.write(f"R0_dndx_195_200\t-\t1.0\t(n/a)\t"
                 f"{L['r0_dndx_195_200']:.6g}\t{res['purity_mixture']['r0_dndx_195_200']:.6g}\n")
        fh.write(f"R0_omega_195_200\t-\t1.0\t(n/a)\t"
                 f"{L['r0_omega_195_200']:.6g}\t{res['purity_mixture']['r0_omega_195_200']:.6g}\n")
        fh.write(f"R0_dndx_203_DLAtier\t-\t1.0\t{F195_DNDX_203:.6g}\t"
                 f"{L['r0_dndx_203']:.6g}\t{res['purity_mixture']['r0_dndx_203']:.6g}\n")
        fh.write(f"R0_omega_203_DLAtier\t-\t1.0\t{F195_OMEGA_203:.6g}\t"
                 f"{L['r0_omega_203']:.6g}\t{res['purity_mixture']['r0_omega_203']:.6g}\n")
    print(f"\n[saved] {out_tsv}")
    return res


if __name__ == "__main__":
    main()
