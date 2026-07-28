"""lls_loa0_validation.py — LLS-tier [17.2, 19.5) validation of the marked-Poisson
loa0-FP catalog-HBI on the 2LPT-0 mock (reduce-only, cached kernel, NO inference,
NO SLURM, NO tilt).

Mirrors ``subdla_loa0_validation.py`` but for the LLS band, per the 2026-07-05 consultant
recipe (notes ``2026-07-05_lls_method_survey.md``):

  * catalog = the STANDARD ``combined_catalog`` (AB.DEF_CAT) — it already spans to NHI=17.2
    and matches the cached posterior-kappa kernel's op_base (the lls_run-nhi172 shard dir
    would break the kernel assert). fit_floor=17.2 admits the existing LLS kernel rows.
  * molly = an nhi172 template (floor 17.2). truth_nhi_floor is DERIVED from the molly's
    lowest edge (17.2) inside build_ingredients — so the LLS truth is non-empty.
    NOTE: the POINT estimate reads the molly TSV C/ρ ratios directly, so a **lya_only**
    nhi172 molly (matching the lya_only kernel + loa0 FP) is required for correct numbers;
    the default full-forest ``figures_molly_nhi172/molly_matrix.tsv`` is a PLUMBING-ONLY
    fallback (carry a ~100%% lya_only-vs-full C/ρ systematic if used for numbers).
  * reports the wide LLS bands [17.2,19.5) and [17.5,19.5) (τ≥1 / τ≥2 thresholds) + the
    0.5-dex per-bin R0, and keeps the DLA tier [20.3+] as an FP≈0 sanity anchor.
  * runs BOTH FP estimators (loa0 vs purity_mixture) AND BOTH estimator paths: v3x
    (posterior-kappa deconvolution, primary) and the kernel-free v1 1/Vmax cross-check
    (``baseline_recovery`` with estimator_fn=None) — agreement certifies the wide-band
    integral is kernel- and shape-prior-independent.

VERDICT target: does loa0 recover the true LLS dN/dX[17.2,19.5) (R0≈1) where the band is a
small residual of a large FP subtraction, and does v3x agree with the kernel-free v1?

.. note::

   **B16 split for this routine's outputs** (audited 2026-07-28). ``_extract`` reads two
   different truth objects out of ``baseline_recovery``, and only one of them is clean:

     * ``dndx_tru_*``, ``r0_dndx_*``, ``per_bin[].dndx_tru`` -- from ``t0["dndx_total"]``
       (:107,:119), which ``cddf_tilt_closure.py::tilted_truth_reductions:168-169`` masks
       with ``zidx >= 0``. **CLEAN.**
     * ``omega_tru_*``, ``r0_omega_*`` -- from ``t0["omega"]`` (:109), built from the
       z-leaky ``f_truth`` at ``cddf_tilt_closure.py:144-146``. **LEAKY**, truth inflated
       x1.05739 on [17.2,19.5): ``omega_tru_172_195`` 2.158191e-05 -> 2.041060e-05, so
       ``r0_omega_172_195`` (loa0/v1) 2.9659 -> 3.1361 -- it moves AWAY from 1.

   This routine's ``dndx_tru_172_195`` = 0.24877424826307443 is the reference value that
   proves the LLS ell(X) truth in ``CDDF_analysis/hbi/joint_mock_validation.json`` is
   contaminated: the same integral built from the leaky ``f_truth`` gives 0.2628520.
   See ``tests/test_b16_ell_contamination.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# repo root = 4 dirnames up (lls -> diagnostics -> CDDF_analysis -> <repo>).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
from CDDF_analysis.hbi.cddf_tilt_closure import baseline_recovery

# committed, git-stamped MOCK deliverable (2LPT-0 recovery ratios — public-OK).
DEFAULT_OUT_JSON = os.path.join(_REPO, "CDDF_analysis", "hbi", "lls_mock_validation.json")

# nhi172 molly — VERIFIED lya_only floor-17.2 (molly_summary title "2lpt0_v1 floor17.2
# lya_only (G0 LLS-coverage)", lam_rf_min=1025), matching the lya_only kernel + loa0 FP.
NHI172_MOLLY = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
                "figures_molly_nhi172/molly_matrix.tsv")

# floor-17.2 posterior-kappa kernel (op_base 494962, matches the LLS config) — enables the
# v3x deconvolution cross-check. Built + broadened (sigma=0.12) by
# phase3d_floor172_robustness.sbatch. NOTE its WALL-1 tilt-closure FAILED
# (V3_KERNEL_SLOPE_DEPENDENCE): v3x here carries a slope-dependent Eddington bias, so it is
# a CAVEATED cross-check to the kernel-free v1 primary, not the headline.
FLOOR172_KERNEL = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phase3d_experiments/"
                   "floor172_lyaonly1025_broaden012/posterior_kernel_2lpt0.npz")

# cumulative report limits: 17.2 floor + 0.5-dex steps through 19.5, then the DLA tier.
REPORT_LIMITS = (17.2, 17.5, 18.0, 18.5, 19.0, 19.5, 20.3)
# 0.5-dex LLS bins for the per-bin R0.
LLS_BINS = [(17.2, 17.5), (17.5, 18.0), (18.0, 18.5), (18.5, 19.0), (19.0, 19.5)]


def _git_commit():
    """Repo HEAD hash for provenance. Never crash; warn loudly instead of a silent 'unknown'."""
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] _git_commit() failed ({type(e).__name__}: {e}); "
              f"code_commit will be stamped 'unknown' (cwd={_REPO}).", file=sys.stderr)
        return "unknown"


class _Args:
    """Mirror ab_loa0_fp_baseline argparse defaults, but with the LLS band + 17.2 floor."""
    def __init__(self, molly_tsv):
        self.catalog_dir = AB.DEF_CAT          # STANDARD combined_catalog (kernel op_base match)
        self.truth = AB.DEF_TRUTH
        self.bal_cat = AB.DEF_BAL
        self.molly_tsv = molly_tsv             # nhi172 template (floor 17.2 -> truth_floor=17.2)
        self.kernel = FLOOR172_KERNEL          # floor-17.2 kernel (op_base 494962 -> v3x enabled)
        self.loa0_product = AB.DEF_LOA0_PRODUCT
        self.out = "/tmp/lls_loa0_validation"
        self.mockdir = None
        self.zbins = "2.0,2.5,3.0,3.5"
        self.report_limits = ",".join(f"{x:g}" for x in REPORT_LIMITS)
        self.family = "bspbody"
        self.fit_floor = 17.2                  # include ALL LLS rows (documented dual-floor arm)
        self.fit_ceil = 99.0
        self.lambda_bspbody = 30.0
        self.lam_rf_min = 1025.0               # lyaonly1025 (kernel + lya_only molly + op_base)
        self.edge_slope_lam = 40.0
        self.gl_nodes = 1
        self.host_truth_floor = 19.0


def _extract(base):
    """Pull the LLS bands + per-bin + DLA-tier context out of a baseline_recovery result."""
    d = {}
    for lo, hi, tag in ((17.2, 19.5, "172_195"), (17.5, 19.5, "175_195")):
        de = base["e0"]["dndx_total"][lo] - base["e0"]["dndx_total"][hi]
        dt = base["t0"]["dndx_total"][lo] - base["t0"]["dndx_total"][hi]
        oe = base["e0"]["omega"][lo] - base["e0"]["omega"][hi]
        ot = base["t0"]["omega"][lo] - base["t0"]["omega"][hi]
        d[f"dndx_est_{tag}"] = de
        d[f"dndx_tru_{tag}"] = dt
        d[f"r0_dndx_{tag}"] = (de / dt) if dt > 0 else np.nan
        d[f"omega_est_{tag}"] = oe
        d[f"omega_tru_{tag}"] = ot
        d[f"r0_omega_{tag}"] = (oe / ot) if ot > 0 else np.nan
    per_bin = []
    for blo, bhi in LLS_BINS:
        de = base["e0"]["dndx_total"][blo] - base["e0"]["dndx_total"][bhi]
        dt = base["t0"]["dndx_total"][blo] - base["t0"]["dndx_total"][bhi]
        per_bin.append(dict(blo=blo, bhi=bhi, dndx_est=de, dndx_tru=dt,
                            r0=(de / dt) if dt > 0 else np.nan))
    d["per_bin"] = per_bin
    # DLA-tier context (FP~=0 there -> should reproduce the DLA-tier R0)
    d["r0_dndx_203"] = base["R0_dndx_total"][20.3]
    d["r0_omega_203"] = base["R0_omega"][20.3]
    return d


def run_mode(mode: str, molly_tsv: str) -> dict:
    """Build ingredients ONCE, run both v3x (deconvolution) and v1 (kernel-free) points."""
    args = _Args(molly_tsv)
    os.makedirs(args.out, exist_ok=True)
    print("=" * 78)
    print(f"[LLS validation] fp_estimator = {mode}  molly = {os.path.basename(os.path.dirname(molly_tsv))}")
    print("=" * 78)
    ing = AB.build_ingredients(args, mode, loa0_product=args.loa0_product)

    # provenance guard (arbiter parity): loa0 FP volume-scale must use the live op n_sl,
    # not the product's stored mock n_sl_prod.
    if mode == "loa0":
        nsp = getattr(ing["fp_model"], "n_sl_prod", None)
        assert nsp == ing["n_sl"], (
            f"n_sl_prod mismatch {nsp} != {ing['n_sl']} — loa0 FP volume-scale wrong.")

    cfg = ing["cfg"]
    cfg._wall1_estimator = "v3"
    common = (cfg, ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["truth_cut"],
              ing["C_interp"], ing["fp_model"], ing["X_tot"],
              ing["logN_lo"], ing["logN_hi"], ing["N_b"], ing["dN_b"])

    # v1 kernel-free 1/Vmax is the LLS PRIMARY (wide-bin, no posterior kernel). v3x
    # (posterior-kappa deconvolution) is BEST-EFFORT: the cached kernel's op_base was
    # built at host_truth_floor=19.0, but the nhi172 molly drives host_truth_floor->17.2,
    # shifting op_base -> the op_base assert fires. v3x for the LLS band needs a kernel
    # rebuilt at host_truth_floor=17.2 (not a config change); we surface that, not crash.
    base_v1 = baseline_recovery(*common)  # default estimator_fn=estimate_f_b (kernel-free 1/Vmax)

    # sanity: LLS truth must be non-empty (else the nhi172 molly floor wasn't applied)
    dt = base_v1["t0"]["dndx_total"][17.2] - base_v1["t0"]["dndx_total"][19.5]
    assert dt > 0, f"empty LLS truth (dt={dt}) — molly floor not 17.2?"

    v3x = None
    v3x_note = "ok"
    try:
        base_v3x = baseline_recovery(*common, estimator_fn=ing["estimator_fn"])
        v3x = _extract(base_v3x)
    except AssertionError as e:
        v3x_note = f"unavailable: {e}"
        print(f"  [v3x unavailable] {e}")

    out = dict(mode=mode, n_sl=int(ing["n_sl"]),
               truth_floor=float(ing["meta"].get("truth_floor", np.nan)),
               v3x_note=v3x_note, v1=_extract(base_v1), v3x=v3x)
    return out


def _fmt(x, w=9, p=4):
    return f"{x:>{w}.{p}f}" if np.isfinite(x) else f"{'nan':>{w}}"


def main(args):
    t_start = time.time()
    res = {m: run_mode(m, args.molly) for m in ("purity_mixture", "loa0")}
    wall = time.time() - t_start

    # v1 always present; v3x only if it ran for BOTH fp modes (kernel op_base match).
    paths = ["v1"]
    if res["purity_mixture"]["v3x"] is not None and res["loa0"]["v3x"] is not None:
        paths.append("v3x")
    else:
        print(f"\n[note] v3x path unavailable ({res['loa0']['v3x_note']}); "
              f"reporting the kernel-free v1 only.")
    for path in paths:
        print("\n" + "=" * 78)
        print(f"PER-0.5-dex-BIN R0 (LLS band) — estimator path = {path}")
        print("=" * 78)
        print(f"{'bin':>14} | {'truth dndx':>12} | {'pm R0':>9} | {'loa0 R0':>9}")
        print("-" * 78)
        for bp, bl in zip(res["purity_mixture"][path]["per_bin"], res["loa0"][path]["per_bin"]):
            lab = f"[{bp['blo']:.1f},{bp['bhi']:.1f})"
            print(f"{lab:>14} | {_fmt(bp['dndx_tru'],12,6)} | "
                  f"{_fmt(bp['r0'],9,3)} | {_fmt(bl['r0'],9,3)}")
        print("\nINTEGRATED bands (R0 = est/truth):")
        for tag, name in (("172_195", "dN/dX [17.2,19.5) τ≥1"), ("175_195", "dN/dX [17.5,19.5) τ≥2")):
            t = res["purity_mixture"][path][f"dndx_tru_{tag}"]
            print(f"  {name}:  truth={t:.6g}  "
                  f"pm R0={_fmt(res['purity_mixture'][path][f'r0_dndx_{tag}'])}  "
                  f"loa0 R0={_fmt(res['loa0'][path][f'r0_dndx_{tag}'])}")
        print(f"  DLA-tier sanity R0_dndx(>=20.3): pm={_fmt(res['purity_mixture'][path]['r0_dndx_203'])} "
              f"loa0={_fmt(res['loa0'][path]['r0_dndx_203'])}")

    out_json = dict(
        metadata=dict(
            what="LLS-tier catalog-HBI recovery validation on the 2LPT-0 mock "
                 "(loa0 vs purity_mixture FP; v3x posterior-kappa vs v1 kernel-free), "
                 "band [17.2,19.5)",
            mock="2LPT-0 (loa-124); MOCK recovery ratios, not real-LOA",
            code_commit=_git_commit(),
            wallclock_s=round(wall, 1),
            molly=args.molly,
            molly_is_lya_only=("lya_only" in args.molly or "lyaonly" in args.molly),
            rederive=f"python CDDF_analysis/diagnostics/lls/lls_loa0_validation.py --molly {args.molly} --force",
            report_limits=list(REPORT_LIMITS),
            note="Reduce-only (cached posterior-kappa kernel, no inference/SLURM/tilt). "
                 "LLS dN/dX is a small residual of a large FP subtraction -> the "
                 "loa0-vs-purity_mixture spread is the headline systematic; v3x-vs-v1 "
                 "agreement certifies kernel/shape independence. If molly_is_lya_only is "
                 "false, carry the lya_only-vs-full-forest C/ρ systematic.",
        ),
        results=res,
    )
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and not args.force:
        print(f"\n[skip-json] {out_path} exists (pass --force to overwrite).")
    else:
        with open(out_path, "w") as fh:
            json.dump(out_json, fh, indent=2, default=float)
        print(f"\n[saved-json] {out_path}  code_commit={out_json['metadata']['code_commit']}  "
              f"molly_lya_only={out_json['metadata']['molly_is_lya_only']}  ({wall:.0f}s)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--molly", default=NHI172_MOLLY,
                    help="nhi172 molly TSV (VERIFIED lya_only floor-17.2 by default).")
    ap.add_argument("--out", default=DEFAULT_OUT_JSON, help="stamped JSON deliverable path.")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists.")
    main(ap.parse_args())
