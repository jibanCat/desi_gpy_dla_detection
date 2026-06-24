"""subdla_loa0_validation.py — SUB-DLA-tier validation of the corrected loa-0 forest-FP
catalog-HBI estimator against the 2LPT-0 truth (reduce-only, cached kernel, NO inference,
NO SLURM, NO tilt).

Reuses ab_loa0_fp_baseline.build_ingredients / run_baseline VERBATIM (same cat_cut /
frozen molly C/ρ / pathlength / cached 2-D posterior kernel that
run_phase3d_postkernel.py stage 2/3 uses), but:

  * reports over the SUB-DLA band [19.5, 20.3) (+ the band [19.5, 20.0)),
  * extracts PER-0.1-dex-bin R0 = est/truth across [19.5,19.6),...,[20.2,20.3) from the
    SAME baseline_recovery e0["f_b"] / t0["f_truth"] (apples-to-apples pathlength — both
    use the SNR>2 truth restriction + the same X_tot denominator),
  * compares the corrected ``loa0`` FP against ``purity_mixture`` in this band,
  * keeps the DLA tier [20.3+] for context (FP≈0 there → unchanged ~1.16 overshoot).

VERDICT: does corrected-loa0 recover the true sub-DLA dN/dX (R0≈1) and land closer to
truth than purity_mixture (which over-subtracts sub-DLA→DLA migration as FP)?
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


# cumulative report limits: 19.5 floor + 0.1-dex steps through 20.3, then the DLA tier
REPORT_LIMITS = (19.5, 19.6, 19.7, 19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.6)


class _Args:
    """Mirror ab_loa0_fp_baseline argparse defaults, but with sub-DLA report limits."""
    def __init__(self):
        self.catalog_dir = AB.DEF_CAT
        self.truth = AB.DEF_TRUTH
        self.bal_cat = AB.DEF_BAL
        self.molly_tsv = None            # -> _resolve_molly fallback to verified lya_only-195
        self.kernel = AB.DEF_KERNEL
        self.loa0_product = AB.DEF_LOA0_PRODUCT
        self.out = "/tmp/subdla_loa0_validation"
        self.mockdir = None
        self.zbins = "2.0,2.5,3.0,3.5"
        self.report_limits = ",".join(f"{x:g}" for x in REPORT_LIMITS)
        self.family = "bspbody"
        self.fit_floor = 19.5            # parametric f(N) fit spans the sub-DLA band
        self.fit_ceil = 99.0
        self.lambda_bspbody = 30.0
        self.lam_rf_min = 1025.0         # lyaonly1025 (matches the kernel + lya_only molly)
        self.edge_slope_lam = 40.0
        self.gl_nodes = 1
        self.host_truth_floor = 19.0


def run_mode(mode: str) -> dict:
    args = _Args()
    os.makedirs(args.out, exist_ok=True)
    print("=" * 78)
    print(f"[sub-DLA validation] fp_estimator = {mode}")
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
    f_est = np.asarray(base["e0"]["f_b"], float)        # estimator f(N), z-marginal
    f_tru = np.asarray(base["t0"]["f_truth"], float)    # truth f(N), z-marginal (SNR>2)

    # per 0.1-dex bin across [19.5, 20.3): exact bin edges (half-open [lo,hi))
    bins = [(round(19.5 + 0.1 * k, 1), round(19.6 + 0.1 * k, 1)) for k in range(8)]
    per_bin = []
    for blo, bhi in bins:
        sel = (logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)
        # exactly one fine bin per 0.1-dex (dlogN=0.1)
        fe = float(np.nansum(f_est[sel]))
        ft = float(np.nansum(f_tru[sel]))
        dndx_e = float(np.nansum(f_est[sel] * dN_b[sel]))
        dndx_t = float(np.nansum(f_tru[sel] * dN_b[sel]))
        r0 = (dndx_e / dndx_t) if dndx_t > 0 else np.nan
        per_bin.append(dict(blo=blo, bhi=bhi, f_est=fe, f_tru=ft,
                            dndx_est=dndx_e, dndx_tru=dndx_t, r0=r0))

    # integrated band [19.5, 20.3) = cumulative(19.5) - cumulative(20.3)
    def _band(lo, hi, key):
        return base[key][lo] - base[key][hi]

    dndx_e_195_203 = _band(19.5, 20.3, "e0") if False else (
        base["e0"]["dndx_total"][19.5] - base["e0"]["dndx_total"][20.3])
    dndx_t_195_203 = (base["t0"]["dndx_total"][19.5] - base["t0"]["dndx_total"][20.3])
    om_e_195_203 = (base["e0"]["omega"][19.5] - base["e0"]["omega"][20.3])
    om_t_195_203 = (base["t0"]["omega"][19.5] - base["t0"]["omega"][20.3])
    # band [19.5, 20.0)
    dndx_e_195_200 = (base["e0"]["dndx_total"][19.5] - base["e0"]["dndx_total"][20.0])
    dndx_t_195_200 = (base["t0"]["dndx_total"][19.5] - base["t0"]["dndx_total"][20.0])
    om_e_195_200 = (base["e0"]["omega"][19.5] - base["e0"]["omega"][20.0])
    om_t_195_200 = (base["t0"]["omega"][19.5] - base["t0"]["omega"][20.0])

    return dict(
        mode=mode, n_sl=int(ing["n_sl"]),
        per_bin=per_bin,
        # integrated band [19.5,20.3)
        dndx_est_195_203=dndx_e_195_203, dndx_tru_195_203=dndx_t_195_203,
        r0_dndx_195_203=(dndx_e_195_203 / dndx_t_195_203) if dndx_t_195_203 > 0 else np.nan,
        omega_est_195_203=om_e_195_203, omega_tru_195_203=om_t_195_203,
        r0_omega_195_203=(om_e_195_203 / om_t_195_203) if om_t_195_203 > 0 else np.nan,
        # integrated band [19.5,20.0)
        dndx_est_195_200=dndx_e_195_200, dndx_tru_195_200=dndx_t_195_200,
        r0_dndx_195_200=(dndx_e_195_200 / dndx_t_195_200) if dndx_t_195_200 > 0 else np.nan,
        omega_est_195_200=om_e_195_200, omega_tru_195_200=om_t_195_200,
        r0_omega_195_200=(om_e_195_200 / om_t_195_200) if om_t_195_200 > 0 else np.nan,
        # DLA tier context (cumulative >=20.3)
        dndx_est_203=base["e0"]["dndx_total"][20.3], dndx_tru_203=base["t0"]["dndx_total"][20.3],
        r0_dndx_203=base["R0_dndx_total"][20.3], r0_omega_203=base["R0_omega"][20.3],
        r0_dndx_200=base["R0_dndx_total"][20.0], r0_omega_200=base["R0_omega"][20.0],
    )


def main():
    res = {m: run_mode(m) for m in ("purity_mixture", "loa0")}

    def _fmt(x, w=10, p=4):
        return f"{x:>{w}.{p}f}" if np.isfinite(x) else f"{'nan':>{w}}"

    print("\n" + "=" * 78)
    print("PER-0.1-dex-BIN R0 = recovered dN/dX / truth dN/dX  (sub-DLA band)")
    print("=" * 78)
    print(f"{'bin':>14} | {'truth dndx':>12} | {'pm dndx':>10} {'pm R0':>8} | "
          f"{'loa0 dndx':>10} {'loa0 R0':>8}")
    print("-" * 78)
    pm = res["purity_mixture"]["per_bin"]
    lo = res["loa0"]["per_bin"]
    for bp, bl in zip(pm, lo):
        lab = f"[{bp['blo']:.1f},{bp['bhi']:.1f})"
        print(f"{lab:>14} | {_fmt(bp['dndx_tru'],12,6)} | "
              f"{_fmt(bp['dndx_est'],10,6)} {_fmt(bp['r0'],8,3)} | "
              f"{_fmt(bl['dndx_est'],10,6)} {_fmt(bl['r0'],8,3)}")

    print("\n" + "=" * 78)
    print("INTEGRATED BANDS — recovered vs truth, R0, both FP estimators")
    print("=" * 78)
    for band, ek, tk, rk in (
        ("dN/dX [19.5,20.3)", "dndx_est_195_203", "dndx_tru_195_203", "r0_dndx_195_203"),
        ("Omega [19.5,20.3)", "omega_est_195_203", "omega_tru_195_203", "r0_omega_195_203"),
        ("dN/dX [19.5,20.0)", "dndx_est_195_200", "dndx_tru_195_200", "r0_dndx_195_200"),
        ("Omega [19.5,20.0)", "omega_est_195_200", "omega_tru_195_200", "r0_omega_195_200"),
    ):
        t = res["purity_mixture"][tk]  # truth identical across modes
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

    # persist a tsv
    out_tsv = "/tmp/subdla_loa0_validation/subdla_validation.tsv"
    with open(out_tsv, "w") as fh:
        fh.write("metric\tbin\ttruth\tpurity_mixture\tloa0\n")
        for bp, bl in zip(pm, lo):
            lab = f"[{bp['blo']:.1f},{bp['bhi']:.1f})"
            fh.write(f"r0_dndx_bin\t{lab}\t1.0\t{bp['r0']:.6g}\t{bl['r0']:.6g}\n")
        for band, ek, tk, rk in (
            ("dndx_195_203", "dndx_est_195_203", "dndx_tru_195_203", "r0_dndx_195_203"),
            ("omega_195_203", "omega_est_195_203", "omega_tru_195_203", "r0_omega_195_203"),
            ("dndx_195_200", "dndx_est_195_200", "dndx_tru_195_200", "r0_dndx_195_200"),
            ("omega_195_200", "omega_est_195_200", "omega_tru_195_200", "r0_omega_195_200"),
        ):
            fh.write(f"R0_{band}\t-\t1.0\t{res['purity_mixture'][rk]:.6g}\t{res['loa0'][rk]:.6g}\n")
        fh.write(f"R0_dndx_203\t-\t1.0\t{res['purity_mixture']['r0_dndx_203']:.6g}\t"
                 f"{res['loa0']['r0_dndx_203']:.6g}\n")
        fh.write(f"R0_omega_203\t-\t1.0\t{res['purity_mixture']['r0_omega_203']:.6g}\t"
                 f"{res['loa0']['r0_omega_203']:.6g}\n")
    print(f"\n[saved] {out_tsv}")
    return res


if __name__ == "__main__":
    main()
