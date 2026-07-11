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

import argparse
import json
import os
import sys
import time

import numpy as np

# repo root = 4 dirnames up (subdla -> diagnostics -> CDDF_analysis -> <repo>).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
from CDDF_analysis.hbi.cddf_tilt_closure import baseline_recovery
# forward-response kernel path (Track-C): reuse the frozen-2LPT-0 self-recovery driver
from CDDF_analysis.hbi import track_c_perz_band as PZ
from CDDF_analysis.hbi import track_c_tf_2lpt1 as TF1
from CDDF_analysis.unblind.provenance import assert_forward_kernel

# committed, git-stamped MOCK deliverable (2LPT-0 recovery ratios — public-OK, no real-LOA
# values). The real-LOA sub-DLA dN/dX/Omega/f(N) numbers are private (notes repo only).
DEFAULT_OUT_JSON = os.path.join(_REPO, "CDDF_analysis", "hbi", "subdla_mock_validation.json")
# the FORWARD-response cross-check deliverable (separate file — never overwrites the kappa one)
DEFAULT_OUT_JSON_FORWARD = os.path.join(
    _REPO, "CDDF_analysis", "hbi", "subdla_mock_validation_forward.json")


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


def _stamped_commit():
    """HEAD sha with '-dirty' appended iff THIS routine file is modified/untracked
    (feedback_headline_provenance_routine: stamp -dirty on the ROUTINE, not the whole tree)."""
    base = _git_commit()
    if base == "unknown":
        return base
    import subprocess
    rel = os.path.relpath(os.path.abspath(__file__), _REPO)
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--", rel],
                                      cwd=_REPO, stderr=subprocess.DEVNULL).decode().strip()
        return base + ("-dirty" if out else "")
    except Exception:  # noqa: BLE001
        return base


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


# forward-response kernel (FROZEN 2LPT-0 Track-C stage-0 model). Same file the committed
# forward artifacts (subdla_mock_headline.json / crossmock_transfer_loa0.json) use.
_DEF_FORWARD_MODEL = TF1._DEF_FORWARD


class _FwdArgs:
    """Args for the FORWARD-response kernel path, mirroring track_c_tf_2lpt1's argparse
    defaults but with the held-out catalog/truth/mockdir SELF-POINTED at 2LPT-0 (mock-0)
    — i.e. the on-mock self-recovery FLOOR under the forward kernel, the exact leg that
    produced crossmock_transfer_loa0.json's self-2lpt0 numbers.  Fields cover the union
    read by build_frozen_calibration + build_heldout_ingredients + PZ._set_forward_cfg."""
    def __init__(self, fp_estimator: str):
        # frozen 2LPT-0 calibration inputs (build_frozen_calibration)
        self.molly_tsv = None            # -> AB._resolve_molly fallback (lya_only-195)
        self.kernel = AB.DEF_KERNEL      # unused on the forward path (kept for parity)
        self.forward_model = _DEF_FORWARD_MODEL
        self.resp_family = "empirical"
        # held-out = 2LPT-0 ITSELF (self-recovery), mock-0 paths
        self.heldout_cat = AB.DEF_CAT
        self.heldout_truth = AB.DEF_TRUTH
        self.heldout_bal = AB.DEF_BAL
        self.heldout_mockdir = os.path.dirname(AB.DEF_TRUTH)
        # FP estimator for the held-out POINT
        self.fp_estimator = fp_estimator
        self.loa0_product = AB.DEF_LOA0_PRODUCT
        # shared HBIConfig knobs (identical to the posterior _Args + the TF driver)
        self.out = "/tmp/subdla_loa0_validation_forward"
        self.mockdir = None
        self.zbins = "2.0,2.5,3.0,3.5"
        self.report_limits = ",".join(f"{x:g}" for x in REPORT_LIMITS)
        self.family = "bspbody"
        self.fit_floor = 19.5
        self.fit_ceil = 99.0
        self.lambda_bspbody = 30.0
        self.lam_rf_min = 1025.0
        self.edge_slope_lam = 40.0
        self.gl_nodes = 1
        self.host_truth_floor = 19.0
        self.cz_min_count = 30.0
        # band-finalize knobs (read by _set_forward_cfg; INERT for the POINT R0 that
        # baseline_recovery computes — no MC band here — but must exist on the namespace)
        self.band_recenter = True
        self.omega_slope_extrap = True
        self.omega_slope_extrap_integrated = True
        self.slope_edge = 21.2
        self.slope_fit_dex = 0.6
        self.sigma_slope = 0.5


def _build_base(mode: str, resp_kind: str, frozen=None):
    """Build (base, ing) for FP `mode` on the chosen response-kernel path.

    resp_kind='kappa'  (DEFAULT, byte-identical to the committed diagnostic): the GP-
        POSTERIOR kernel via ab_loa0_fp_baseline.build_ingredients.  This is a labelled
        DIAGNOSTIC path and does NOT call the forward-kernel guard.
    resp_kind='forward': the FROZEN 2LPT-0 forward-response kernel via the
        track_c_tf_2lpt1 self-recovery machinery (build_frozen_calibration +
        build_heldout_ingredients self-pointed at 2LPT-0 + _set_forward_cfg).  The
        forward-kernel guard fires here — a "forward" artifact can never be stamped on
        the posterior kernel.
    """
    if resp_kind == "forward":
        fa = _FwdArgs(mode)
        os.makedirs(fa.out, exist_ok=True)
        if frozen is None:
            frozen = TF1.build_frozen_calibration(fa)
        ing = TF1.build_heldout_ingredients(fa, frozen, "A")   # variant A = fully frozen
        cfg = ing["cfg"]
        cfg.report_logN_limits = tuple(REPORT_LIMITS)
        cfg._wall1_estimator = "v3"
        cfg.n_mc = 0
        PZ._set_forward_cfg(cfg, fa)      # sets resp_kind='forward' + kernel_forward_model
        # FAIL-CLOSED: refuse to proceed to a stamped "forward" artifact on the posterior
        # kernel. Fires iff _set_forward_cfg did not actually engage the forward path.
        assert_forward_kernel(cfg, context=f"subdla forward validation (fp={mode})",
                              require_kernel_model=True)
    else:
        fa = _Args()
        os.makedirs(fa.out, exist_ok=True)
        ing = AB.build_ingredients(fa, mode, loa0_product=fa.loa0_product)
        cfg = ing["cfg"]
        cfg._wall1_estimator = "v3"       # posterior (kappa) diagnostic path — no guard
    base = baseline_recovery(
        cfg, ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["truth_cut"],
        ing["C_interp"], ing["fp_model"], ing["X_tot"],
        ing["logN_lo"], ing["logN_hi"], ing["N_b"], ing["dN_b"],
        estimator_fn=ing["estimator_fn"])
    return base, ing


def run_mode(mode: str, resp_kind: str = "kappa", frozen=None) -> dict:
    print("=" * 78)
    print(f"[sub-DLA validation] fp_estimator = {mode}   kernel = {resp_kind}")
    print("=" * 78)
    base, ing = _build_base(mode, resp_kind, frozen=frozen)

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


def main(args):
    resp_kind = getattr(args, "resp_kind", "kappa")
    t_start = time.time()
    frozen = None
    if resp_kind == "forward":
        # build the FROZEN 2LPT-0 calibration ONCE (mode-independent) and reuse for both FPs
        frozen = TF1.build_frozen_calibration(_FwdArgs("purity_mixture"))
    res = {m: run_mode(m, resp_kind=resp_kind, frozen=frozen)
           for m in ("purity_mixture", "loa0")}
    wall = time.time() - t_start

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
    _tsv_dir = "/tmp/subdla_loa0_validation" + ("_forward" if resp_kind == "forward" else "")
    os.makedirs(_tsv_dir, exist_ok=True)
    out_tsv = os.path.join(_tsv_dir, "subdla_validation.tsv")
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

    # ---- committed, git-stamped JSON deliverable (mock recovery ratios; public-OK) ----
    if resp_kind == "forward":
        fa = _FwdArgs("loa0")
        inputs = dict(catalog_dir=fa.heldout_cat, truth=fa.heldout_truth,
                      bal_cat=fa.heldout_bal, forward_model=fa.forward_model,
                      loa0_product=fa.loa0_product, resp_family=fa.resp_family,
                      molly="<_resolve_molly fallback: AB.DEF_LYAONLY_MOLLY (nhi195 lya_only)>",
                      report_limits=fa.report_limits, fit_floor=fa.fit_floor,
                      lam_rf_min=fa.lam_rf_min, family=fa.family,
                      host_truth_floor=fa.host_truth_floor)
        what = ("sub-DLA-tier catalog-HBI recovery validation on the 2LPT-0 mock, "
                "FORWARD-RESPONSE kernel (Track-C 'right object'), band [19.5,20.3)")
        note = ("Forward-response kernel (resp_kind='forward'; FROZEN 2LPT-0 "
                "forward_response_2lpt0.npz + z-resolved g(N,z)) via the track_c_tf_2lpt1 "
                "self-recovery machinery self-pointed at 2LPT-0 (build_frozen_calibration + "
                "build_heldout_ingredients variant A + _set_forward_cfg). baseline_recovery "
                "POINT R0 = est/truth (no MC band). Reproduces bit-for-bit two independent "
                "forward derivations (subdla_mock_headline.json + crossmock_transfer_loa0.json, "
                "both currently UNTRACKED; this validation artifact stands on its own committed "
                "routine, not on those cross-checks) at loa0 band R0 ~ 0.849/0.822. The "
                "Adopted as the sub-DLA headline for single-kernel COHERENCE with the DLA "
                "tier (which is already forward). The forward-vs-posterior discriminator is "
                "validated at the DLA tier (>=20.3), where truth completeness ~ 1 so R0 SHOULD "
                "be 1: posterior R0=1.16 (over-recovers, SBC-fails) vs forward R0=1.04. That "
                "anchor does NOT exist below 20.3 (true completeness ~ 0.2-1.0, and on-mock R0 "
                "is a tautology), so the sub-DLA switch is a CONSISTENCY choice, not an "
                "independently-validated one -- the forward is shown stable across recipes "
                "below 20.3 but not shown to BEAT the posterior there. CARRIED SYSTEMATIC: the "
                "forward<->posterior kernel-object gap on this band is 3.8% (dN/dX) / 8.5% "
                "(Omega) (kappa 0.883/0.899 vs forward 0.849/0.822), UNRESOLVED at the sub-DLA "
                "tier -- do NOT quote the 0.849/0.822 point without it. NB the lower forward R0 "
                "means a LARGER 1/R0 up-correction, i.e. it raises Omega_HI(sub-DLA). "
                "Fail-closed forward-kernel guard (unblind.provenance.assert_forward_kernel) "
                "fires before this artifact is stamped.")
        rederive = ("python CDDF_analysis/diagnostics/subdla/subdla_loa0_validation.py "
                    "--resp-kind forward --force")
        deps = ["CDDF_analysis/diagnostics/subdla/subdla_loa0_validation.py",
                "CDDF_analysis/hbi/track_c_tf_2lpt1.py",
                "CDDF_analysis/hbi/track_c_perz_band.py",
                "CDDF_analysis/hbi/ab_loa0_fp_baseline.py",
                "CDDF_analysis/hbi/cddf_tilt_closure.py",
                "CDDF_analysis/hbi/cddf_catalog_hbi.py",
                "CDDF_analysis/unblind/provenance.py"]
    else:
        a = _Args()
        inputs = dict(catalog_dir=a.catalog_dir, truth=a.truth, bal_cat=a.bal_cat,
                      kernel=a.kernel, loa0_product=a.loa0_product,
                      molly="<_resolve_molly fallback: AB.DEF_LYAONLY_MOLLY (nhi195 lya_only)>",
                      report_limits=a.report_limits, fit_floor=a.fit_floor,
                      lam_rf_min=a.lam_rf_min, family=a.family,
                      host_truth_floor=a.host_truth_floor)
        what = ("sub-DLA-tier catalog-HBI recovery validation on the 2LPT-0 mock "
                "(loa0 vs purity_mixture FP), band [19.5,20.3)")
        note = ("Reduce-only (cached POSTERIOR/kappa kernel, no inference/SLURM/tilt). "
                "R0 = est/truth. DIAGNOSTIC path (posterior kernel; superseded as the sub-DLA "
                "headline by the forward artifact subdla_mock_validation_forward.json for "
                "single-kernel coherence -- the kappa<->forward gap is a carried systematic, "
                "and kappa is demonstrably wrong only at the DLA tier, not below 20.3). "
                "The [19.5,19.7) edge is formally non-identifiable on a 19.5-floored "
                "catalog (Track A closed); report the [19.7,20.3) f(N) diff + the "
                "[19.5,20.3] integrated band with an edge-migration systematic.")
        rederive = "python CDDF_analysis/diagnostics/subdla/subdla_loa0_validation.py --force"
        deps = ["CDDF_analysis/diagnostics/subdla/subdla_loa0_validation.py",
                "CDDF_analysis/hbi/ab_loa0_fp_baseline.py",
                "CDDF_analysis/hbi/cddf_tilt_closure.py",
                "CDDF_analysis/hbi/cddf_catalog_hbi.py"]
    out_json = dict(
        metadata=dict(
            what=what,
            mock="2LPT-0 (loa-124); values are MOCK recovery ratios, not real-LOA. "
                 "No real-LOA (loa main-dark) data was read.",
            resp_kind=resp_kind,
            code_commit=_stamped_commit(),
            deps=deps,
            wallclock_s=round(wall, 1),
            rederive=rederive,
            inputs=inputs,
            note=note,
        ),
        per_bin={m: res[m]["per_bin"] for m in ("purity_mixture", "loa0")},
        integrated={m: {k: res[m][k] for k in res[m] if k not in ("per_bin",)}
                    for m in ("purity_mixture", "loa0")},
    )
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and not args.force:
        print(f"[skip-json] {out_path} exists (pass --force to overwrite).")
    else:
        with open(out_path, "w") as fh:
            json.dump(out_json, fh, indent=2, default=float)
        print(f"[saved-json] {out_path}  code_commit={out_json['metadata']['code_commit']}  "
              f"({wall:.0f}s)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resp-kind", dest="resp_kind", default="kappa",
                    choices=["kappa", "forward"],
                    help="response-kernel OBJECT. 'kappa' (DEFAULT, byte-identical to the "
                         "committed subdla_mock_validation.json): the GP-POSTERIOR kernel — a "
                         "labelled DIAGNOSTIC (superseded as headline; over-recovers high-N, "
                         "demonstrably wrong at the DLA tier, a consistency call below 20.3). "
                         "'forward': the Track-C forward-response kernel (the right object / "
                         "headline) via the frozen-2LPT-0 self-recovery path.")
    ap.add_argument("--out", default=None,
                    help="stamped JSON deliverable path. Default depends on --resp-kind: "
                         "subdla_mock_validation.json (kappa) / "
                         "subdla_mock_validation_forward.json (forward).")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out if it already exists (default: refuse).")
    _a = ap.parse_args()
    if _a.out is None:
        _a.out = DEFAULT_OUT_JSON_FORWARD if _a.resp_kind == "forward" else DEFAULT_OUT_JSON
    main(_a)
