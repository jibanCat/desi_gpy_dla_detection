"""subdla_basis_pad_bracket.py — the DECOUPLED BASIS-PADDING bracket test for the
sub-DLA low edge (2026-06-17, PI-greenlit).

################################################################################
# KAPPA-KERNEL DIAGNOSTIC — NOT PAPER-FACING.  resp_kind = 'kappa'.
#
# This routine runs on the GP-POSTERIOR ('kappa') response kernel, via
# ab_loa0_fp_baseline.build_ingredients, which constructs its HBIConfig without
# passing resp_kind and therefore inherits the dataclass default.  Until 2026-07-28
# that inheritance was IMPLICIT (indistinguishable from an oversight); it is now
# STATED (`cfg.resp_kind = RESP_KIND` in run_one, and an explicit `resp_kind=`
# argument at the padding_truth_dndx HBIConfig construction) and the artifact
# SELF-DECLARES metadata['resp_kind']='kappa' + metadata['paper_facing']=False,
# enforced by CDDF_analysis/unblind/resp_kind.kernel_metadata (which raises if
# anyone flips PAPER_FACING to True while RESP_KIND stays 'kappa').
#
# WHY IT IS NOT PORTED TO THE FORWARD KERNEL.  Two reasons, the second decisive:
#   1. the forward path is a different ingredient builder
#      (track_c_tf_2lpt1.build_frozen_calibration + build_heldout_ingredients +
#      track_c_perz_band._set_forward_cfg) returning a different dict shape --
#      a rewrite plus a full re-run, not a knob flip; and
#   2. the PRE-REGISTERED PASS GATE below is written in KAPPA numbers.  Criterion 5
#      ("DLA tier R0(dN/dX>=20.3) within 1.159 +- 0.014") is precisely the posterior
#      over-recovery value; the forward kernel gives ~1.04 there, so a ported run
#      would fail its own pre-registered gate BY CONSTRUCTION.  Re-porting therefore
#      requires re-registering the gate with the PI, which is not a code change.
#
# CONSEQUENCE, stated plainly: the R0 LEVELS in this artifact carry the
# kappa<->forward kernel-object gap (sub-DLA band: kappa 0.883/0.899 vs forward
# 0.849/0.822).  The BRACKET (pad-19.0 / pad-19.2 vs the no-pad baseline) is the
# intended observable and is a within-kernel difference.
################################################################################

Tests cfg.basis_pad_floor: extend ONLY the deconvolution basis + the marked-Poisson
normalizer support below the 19.5 fit floor (so an edge object's broadened-kernel mass
that leaks below 19.5 has BASIS columns to land in), while KEEPING the detection set,
the molly C/ρ, and the FP μ_FP/λ_fp at 19.5 (NO FP-heavy [pad,19.5) detections; FP
normalizer matched to the 19.5 detection set). Report ≥19.5 only; the padding band
[basis_pad_floor, 19.5) is unreported deconvolution support.

This is the clean isolation of the GEOMETRIC edge-mass recovery WITHOUT the floor-19.0
failure's triple coupling (which ALSO admitted FP-heavy [19.0,19.5) detections and
re-gridded everything via v3_logN_fit_floor).

REDUCE-ONLY: cached FLOOR-19.5 kernel (NO rebuild), NO inference, NO SLURM, NO tilt
(untilted Δα=0 baseline recovery). Reuses ab_loa0_fp_baseline.build_ingredients +
the joint-MC band machinery from subdla_floor_mc_band.py VERBATIM.

Configs (all on the SAME floor-19.5 kernel + nhi195 lya_only molly + loa0 FP):
  * floor19.5  : basis_pad_floor=None  (== 19.5, byte-identical baseline)
  * pad19.0    : basis_pad_floor=19.0  (aggressive; carries the leaked [19.0,19.5) mass)
  * pad19.2    : basis_pad_floor=19.2  (less aggressive; carries [19.2,19.5))

Pre-registered PASS gate (ALL 5 required; see 2026-06-17_subdla_floor190_mc_band.md §3):
  1. [19.5,19.6)/[19.6,19.7) rise from below with q84 <= 1.02 (no overshoot).
  2. every [19.7,20.3) bin >= (floor-19.5 q16 − 0.01); binding [19.7,19.8)>=0.86,
     [19.8,19.9)>=0.91, [19.9,20.0)>=0.88, [20.0,20.3) within (0.88,1.04).
  3. integrated [19.5,20.3) R0 > 0.90 AND <= 1.0.
  4. total recovered dN/dX[19.5,20.3) rises (vs floor-19.5) AND padding
     dN/dX[19.0,19.5) <= ~2× truth there (<= ~0.12; floor-19.0's 0.192 was the failure).
  5. DLA tier R0(dN/dX>=20.3) within 1.159 ± 0.014.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
from CDDF_analysis.hbi.cddf_tilt_closure import baseline_recovery, tilted_truth_reductions
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    _draw_beta_cell, _slice_active_unitC, _rescale_unitC_active,
    _apply_C_to_M, v3x_fit_map, v3x_reduce, C_FLOOR,
    load_and_cut_catalog, _build_qso_lookup, HBIConfig,
)
from CDDF_analysis.unblind import resp_kind as RK

# ---- RESPONSE-KERNEL DECLARATION (see the module banner) --------------------
# STATED, not inherited. RK.kernel_metadata() below refuses to stamp
# paper_facing=True while RESP_KIND is 'kappa', so this pair cannot drift apart.
RESP_KIND = RK.RESP_KIND_KAPPA
PAPER_FACING = False

# the cached FLOOR-19.5 kernel (NOT a floor-19.0 rebuild — basis_pad_floor needs NO rebuild)
KERNEL = AB.DEF_KERNEL                       # mollynhi195_lyaonly1025_broaden012
MOLLY = AB.DEF_LYAONLY_MOLLY                 # figures_molly_nhi195/lya_only
LOA0_PRODUCT = AB.DEF_LOA0_PRODUCT           # loa0_fp_product_lyaonly1025.npz

CONFIGS = (
    dict(name="floor19.5", basis_pad_floor=None),
    dict(name="pad19.0", basis_pad_floor=19.0),
    dict(name="pad19.2", basis_pad_floor=19.2),
)
REPORT_LIMITS = (19.0, 19.5, 19.6, 19.7, 19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.6)
PER_BINS = [(round(19.5 + 0.1 * k, 1), round(19.6 + 0.1 * k, 1)) for k in range(8)]
PAD_BINS = [(round(19.0 + 0.1 * k, 1), round(19.1 + 0.1 * k, 1)) for k in range(5)]  # [19.0,19.5)

# committed, git-stamped MOCK deliverable (2LPT-0 edge-mass bracket — public-OK, no real-LOA).
DEFAULT_OUT_JSON = os.path.join(_REPO, "CDDF_analysis", "hbi", "subdla_edge_systematic.json")
# sibling artifact carrying the floor-19.0 rebuild comparison (cross-referenced, NOT recomputed
# here: this routine only sweeps basis-padding on the floor-19.5 kernel).
SIBLING_FLOOR_MC_JSON = os.path.join(_REPO, "CDDF_analysis", "hbi", "subdla_floor_mc_band.json")


def _git_commit(routine=None):
    """HEAD, suffixed `-dirty` iff the ROUTINE that produced this artifact (or a diagnostic
    module it imports) is untracked or modified.

    A `-dirty` stamp means the artifact is NOT third-party re-derivable: the named commit
    does not contain (this version of) the routine. Checking the routine specifically --
    rather than `git status`, which is dirtied by the artifact's own untracked output -- is
    what makes the marker meaningful. The estimator modules (ab_loa0_fp_baseline /
    cddf_tilt_closure) are listed because without them the stamped `rederive` command cannot
    reproduce the numbers.
    """
    deps = [routine or os.path.relpath(os.path.abspath(__file__), _REPO),
            "CDDF_analysis/hbi/ab_loa0_fp_baseline.py",
            "CDDF_analysis/hbi/cddf_tilt_closure.py"]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                      stderr=subprocess.DEVNULL).decode().strip()
        for f in deps:
            tracked = subprocess.call(["git", "ls-files", "--error-unmatch", f], cwd=_REPO,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
            modified = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", f], cwd=_REPO) != 0
            if not tracked or modified:
                return f"{sha}-dirty"
        return sha
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] _git_commit() failed ({type(e).__name__}: {e})", file=sys.stderr)
        return "unknown"


def _floor190_reference():
    """Cross-load the floor-19.0 rebuild R0 from the sibling subdla_floor_mc_band.json (which
    IS the routine that computes it). Returns a dict with the values + the sibling's own
    code_commit as provenance, or a null-with-note if the sibling has not been produced yet.
    This routine does NOT recompute floor-19.0 (it sweeps basis-padding on the 19.5 kernel);
    the floor-19.0 comparison is reported so a reader can construct the edge-migration
    systematic across {headline 19.5, basis-pad brackets, floor-19.0 rebuild}."""
    if not os.path.exists(SIBLING_FLOOR_MC_JSON):
        return dict(available=False, source=os.path.relpath(SIBLING_FLOOR_MC_JSON, _REPO),
                    note="sibling subdla_floor_mc_band.json not found; run it first to populate "
                         "the floor-19.0 comparison.")
    try:
        with open(SIBLING_FLOOR_MC_JSON) as fh:
            sib = json.load(fh)
        f190 = sib["result"]["floor19_0"]
        return dict(
            available=True, source=os.path.relpath(SIBLING_FLOOR_MC_JSON, _REPO),
            sibling_code_commit=sib["metadata"].get("code_commit"),
            r0_dndx_195_203=f190["dndx"]["r0_band_195_203"],
            r0_omega_195_203=f190["omega"]["r0_band_195_203"],
            dndx_est0_band_195_203=f190["dndx"]["est0_band_195_203"],
        )
    except Exception as exc:  # noqa: BLE001
        return dict(available=False, source=os.path.relpath(SIBLING_FLOOR_MC_JSON, _REPO),
                    note=f"failed to read sibling ({type(exc).__name__}: {exc}).")


class _Args:
    def __init__(self, name):
        self.catalog_dir = AB.DEF_CAT
        self.truth = AB.DEF_TRUTH
        self.bal_cat = AB.DEF_BAL
        self.molly_tsv = MOLLY
        self.kernel = KERNEL
        self.loa0_product = LOA0_PRODUCT
        self.out = "/tmp/subdla_basis_pad_bracket_" + name
        self.mockdir = None
        self.zbins = "2.0,2.5,3.0,3.5"
        self.report_limits = ",".join(f"{x:g}" for x in REPORT_LIMITS)
        self.family = "bspbody"
        self.fit_floor = 19.5            # DETECTIONS + molly + μ_FP at 19.5 (UNCHANGED)
        self.fit_ceil = 99.0
        self.lambda_bspbody = 30.0
        self.lam_rf_min = 1025.0
        self.edge_slope_lam = 40.0
        self.gl_nodes = 1
        self.host_truth_floor = 19.0


def _band_dndx(red_dndx_total, lo=19.5, hi=20.3):
    return red_dndx_total[lo] - red_dndx_total[hi]


def _bin_dndx(f_b, logN_lo, logN_hi, dN_b, bins):
    out = np.zeros(len(bins))
    for bi, (blo, bhi) in enumerate(bins):
        sel = (logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)
        out[bi] = np.nansum(f_b[sel] * dN_b[sel])
    return out


def run_one(knobs, n_mc, seed):
    args = _Args(knobs["name"])
    os.makedirs(args.out, exist_ok=True)
    print("=" * 78)
    print(f"[bracket] {knobs['name']}  basis_pad_floor={knobs['basis_pad_floor']}  "
          f"(fit_floor=19.5, kernel=floor-19.5)  n_mc={n_mc}")
    print("=" * 78)
    ing = AB.build_ingredients(args, "loa0", loa0_product=args.loa0_product)
    cfg = ing["cfg"]
    # EXPLICIT KERNEL DECLARATION (2026-07-28) — stated, not inherited from the
    # HBIConfig dataclass default. See the module banner.
    cfg.resp_kind = RESP_KIND
    cfg._wall1_estimator = "v3"
    cfg.v3_mc_n_restart = 2
    # THE knob: decoupled basis padding (None => byte-identical to floor19.5)
    cfg.basis_pad_floor = knobs["basis_pad_floor"]

    base = baseline_recovery(
        cfg, ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["truth_cut"],
        ing["C_interp"], ing["fp_model"], ing["X_tot"],
        ing["logN_lo"], ing["logN_hi"], ing["N_b"], ing["dN_b"],
        estimator_fn=ing["estimator_fn"])
    e0 = base["e0"]; t0 = base["t0"]
    logN_lo = np.asarray(ing["logN_lo"], float)
    logN_hi = np.asarray(ing["logN_hi"], float)
    dN_b = np.asarray(ing["dN_b"], float)

    f_tru = np.asarray(t0["f_truth"], float)
    f_est0 = np.asarray(e0["f_b"], float)
    dndx_tru_bin = _bin_dndx(f_tru, logN_lo, logN_hi, dN_b, PER_BINS)
    dndx_est0_bin = _bin_dndx(f_est0, logN_lo, logN_hi, dN_b, PER_BINS)
    dndx_tru_band = _band_dndx(t0["dndx_total"])
    dndx_est0_band = _band_dndx(e0["dndx_total"])
    # padding-band diagnostic [19.0,19.5) (UNREPORTED support; gate criterion #4)
    pad_tru_bin = _bin_dndx(f_tru, logN_lo, logN_hi, dN_b, PAD_BINS)
    pad_est0_bin = _bin_dndx(f_est0, logN_lo, logN_hi, dN_b, PAD_BINS)
    pad_tru = float(pad_tru_bin.sum()); pad_est0 = float(pad_est0_bin.sum())
    r0_band_point = dndx_est0_band / dndx_tru_band
    r0_bin_point = dndx_est0_bin / dndx_tru_bin
    r0_203_point = base["R0_dndx_total"][20.3]
    om_e_band = (e0["omega"][19.5] - e0["omega"][20.3])
    om_t_band = (t0["omega"][19.5] - t0["omega"][20.3])

    # ---- MC band (joint: C/ρ Beta + NHI_ERR width + sightline bootstrap; loa0 FROZEN) ----
    fwd = e0["_v3x"]["fwd"]; family = e0["_v3x"]["family"]; fine = e0["_v3x"]["fine"]
    M_meta = e0["_v3x"]["M_meta"]; theta_map = e0["_v3x"]["theta_map"]
    A_meta = fwd["A_meta"]; cat_op = fwd["cat_op"]
    lam_fp_frozen = fwd["lam_fp"]; mu_fp_frozen = fwd["mu_fp"]
    active_flat = fwd["active_flat"]; keep_in_base = fwd["keep_in_base"]
    z_edges_fine = fine[4]
    n_flat = len(logN_lo) * (len(z_edges_fine) - 1)
    unitC = _slice_active_unitC(A_meta, np.arange(n_flat), np.ones(A_meta["n_obs"], bool))

    mm = ing["mm"]; cat_cut = ing["cat_cut"]; good_mask = ing["good_mask"]
    s2n_all = np.asarray(cat_cut["S2N_RED"], float)
    pdla_all = np.asarray(cat_cut["P_DLA"], float)
    op_base = (s2n_all > cfg.snr_min) & (pdla_all > cfg.p_dla_min) & good_mask
    nhi0_base = np.asarray(cat_cut["NHI"], float)[op_base]
    nhi_err_base = np.asarray(cat_cut["NHI_ERR"], float)[op_base]
    nhi_err_base = np.where(np.isfinite(nhi_err_base) & (nhi_err_base > 0), nhi_err_base, 0.0)
    tids_base = np.asarray(cat_cut["TARGETID"], np.int64)[op_base]
    uniq_tids, inv = np.unique(tids_base, return_inverse=True)
    n_uniq = len(uniq_tids)

    rng = np.random.default_rng(seed)
    mc_band = np.full(n_mc, np.nan)
    mc_omega = np.full(n_mc, np.nan)
    mc_bin = np.full((n_mc, len(PER_BINS)), np.nan)
    mc_r0_203 = np.full(n_mc, np.nan)
    mc_pad = np.full(n_mc, np.nan)

    for m in range(n_mc):
        C_draw = _draw_beta_cell(rng, mm.cmp_nfound, mm.cmp_nfid)
        rho_draw = _draw_beta_cell(rng, mm.pur_ntp, mm.pur_ntot)
        C_draw = np.where((mm.cmp_nfid > 0), C_draw, C_FLOOR)
        rho_draw = np.where((mm.pur_ntot > 0), rho_draw, 0.0)
        mult = rng.multinomial(n_uniq, np.full(n_uniq, 1.0 / n_uniq))
        boot_w = mult[inv].astype(float)[keep_in_base]
        A_draw = _rescale_unitC_active(unitC, C_draw)
        M_draw = np.where(active_flat, _apply_C_to_M(M_meta, C_draw), 0.0)
        fit = v3x_fit_map(A_draw, M_draw, lam_fp_frozen, mu_fp_frozen, fine, family, cfg,
                          obj_weights=boot_w, theta0=theta_map, n_restart=2,
                          rng=np.random.default_rng(seed * 100003 + m), lit_start=False)
        rr = v3x_reduce(cfg, fit["theta_map"], fine, family, M_meta)
        f_b = np.asarray(rr["f_b"], float)
        mc_band[m] = _band_dndx(rr["dndx_total"])
        mc_omega[m] = rr["omega"][19.5] - rr["omega"][20.3]   # post-proc; consumes no RNG
        mc_bin[m] = _bin_dndx(f_b, logN_lo, logN_hi, dN_b, PER_BINS)
        mc_pad[m] = float(_bin_dndx(f_b, logN_lo, logN_hi, dN_b, PAD_BINS).sum())
        mc_r0_203[m] = rr["dndx_total"][20.3] / t0["dndx_total"][20.3]
        if (m + 1) % 25 == 0:
            print(f"    draw {m+1}/{n_mc}")

    return dict(
        name=knobs["name"], basis_pad_floor=knobs["basis_pad_floor"], n_sl=int(ing["n_sl"]),
        dndx_tru_band=float(dndx_tru_band), dndx_tru_bin=dndx_tru_bin,
        dndx_est0_band=float(dndx_est0_band), dndx_est0_bin=dndx_est0_bin,
        r0_band_point=float(r0_band_point), r0_bin_point=r0_bin_point,
        r0_203_point=float(r0_203_point),
        om_e_band=float(om_e_band), om_t_band=float(om_t_band),
        r0_omega_band=float(om_e_band / om_t_band) if om_t_band > 0 else np.nan,
        pad_tru=pad_tru, pad_est0=pad_est0,
        pad_tru_bin=pad_tru_bin, pad_est0_bin=pad_est0_bin,
        mc_band=mc_band, mc_omega=mc_omega, mc_bin=mc_bin, mc_r0_203=mc_r0_203, mc_pad=mc_pad,
    )


def _qstats(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (np.nan,) * 5
    return (np.nanmean(a), np.nanstd(a),
            np.nanpercentile(a, 16), np.nanpercentile(a, 50), np.nanpercentile(a, 84))


def evaluate_gate(base195, cfg_res, tru_band, tru_bin):
    """Apply the 5-criterion pre-registered gate to a candidate config vs floor-19.5."""
    crit = {}
    # per-bin R0 q16/q84 for the candidate
    r0_bin_q16 = np.array([np.nanpercentile(cfg_res["mc_bin"][:, bi] / tru_bin[bi], 16)
                           for bi in range(len(PER_BINS))])
    r0_bin_q84 = np.array([np.nanpercentile(cfg_res["mc_bin"][:, bi] / tru_bin[bi], 84)
                           for bi in range(len(PER_BINS))])
    r0_bin_pt = cfg_res["r0_bin_point"]
    f195_q16 = np.array([np.nanpercentile(base195["mc_bin"][:, bi] / tru_bin[bi], 16)
                         for bi in range(len(PER_BINS))])

    # criterion 1: bottom two bins rise from below, q84 <= 1.02
    c1 = (r0_bin_q84[0] <= 1.02) and (r0_bin_q84[1] <= 1.02) \
        and (r0_bin_pt[0] > base195["r0_bin_point"][0] - 0.005) \
        and (r0_bin_pt[1] > base195["r0_bin_point"][1] - 0.005)
    crit["1_edge_rise_no_overshoot"] = dict(
        passed=bool(c1),
        edge0_pt=float(r0_bin_pt[0]), edge0_q84=float(r0_bin_q84[0]),
        edge1_pt=float(r0_bin_pt[1]), edge1_q84=float(r0_bin_q84[1]),
        f195_edge0=float(base195["r0_bin_point"][0]),
        f195_edge1=float(base195["r0_bin_point"][1]))

    # criterion 2: mid-band [19.7,20.3) >= f195_q16 - 0.01 ; binding thresholds
    binding = {2: 0.86, 3: 0.91, 4: 0.88}  # [19.7,19.8),[19.8,19.9),[19.9,20.0)
    c2_ok = True; c2_detail = []
    for bi in range(2, 8):  # [19.7,19.8)...[20.2,20.3)
        thr = max(f195_q16[bi] - 0.01, binding.get(bi, 0.0))
        ok = r0_bin_pt[bi] >= thr - 1e-9
        # also enforce the [20.0,20.3) window upper bound 1.04
        if bi >= 5:
            ok = ok and (r0_bin_pt[bi] <= 1.04 + 1e-9)
        c2_ok = c2_ok and ok
        c2_detail.append(dict(bin=PER_BINS[bi], r0=float(r0_bin_pt[bi]),
                              thr=float(thr), passed=bool(ok)))
    crit["2_midband_holds"] = dict(passed=bool(c2_ok), detail=c2_detail)

    # criterion 3: integrated [19.5,20.3) R0 in (0.90, 1.0]
    r0b = cfg_res["r0_band_point"]
    c3 = (r0b > 0.90) and (r0b <= 1.0 + 1e-9)
    crit["3_integrated_band"] = dict(passed=bool(c3), r0=float(r0b),
                                     f195_r0=float(base195["r0_band_point"]))

    # criterion 4: total recovered dN/dX rises AND padding <= 2x truth (<=~0.12)
    rises = cfg_res["dndx_est0_band"] > base195["dndx_est0_band"] - 1e-9
    pad_ok = cfg_res["pad_est0"] <= 0.12 + 1e-9
    pad_ratio = cfg_res["pad_est0"] / cfg_res["pad_tru"] if cfg_res["pad_tru"] > 0 else np.nan
    c4 = rises and pad_ok
    crit["4_count_book_keeping"] = dict(
        passed=bool(c4), band_rises=bool(rises),
        band_est=float(cfg_res["dndx_est0_band"]), f195_band_est=float(base195["dndx_est0_band"]),
        pad_est=float(cfg_res["pad_est0"]), pad_tru=float(cfg_res["pad_tru"]),
        pad_ratio=float(pad_ratio), pad_ok=bool(pad_ok))

    # criterion 5: DLA tier R0(>=20.3) within 1.159 +- 0.014
    r5 = cfg_res["r0_203_point"]
    c5 = abs(r5 - 1.159) <= 0.014 + 1e-9
    crit["5_dla_tier_unmoved"] = dict(passed=bool(c5), r0_203=float(r5))

    overall = all(crit[k]["passed"] for k in crit)
    return dict(passed=bool(overall), criteria=crit)


def padding_truth_dndx():
    """The TRUE dN/dX in the unreported padding band [19.0,19.5), config-independent.
    The bracket's truth_cut is floored at the molly's lowest edge (19.5) so it carries
    NO sub-floor truth; reload the truth catalog floored at 19.0 to get the padding
    truth (over the SAME SNR>2 sightlines / X_tot). Used ONLY for the gate-#4 padding
    over-recovery ratio (floor-19.0's failure dumped 0.192 here, truth ≈ 0.060 → R0≈3.2)."""
    args = _Args("padtruth")
    os.makedirs(args.out, exist_ok=True)
    cfg = HBIConfig(
        catalog_dir=args.catalog_dir, truth_path=args.truth, bal_cat_path=args.bal_cat,
        molly_tsv=args.molly_tsv, out_dir=args.out,
        mockdir=args.mockdir or os.path.dirname(args.truth),
        zbins=tuple(float(x) for x in args.zbins.split(",")),
        report_logN_limits=(19.0, 19.5), no_bal=True, resp_kind=RESP_KIND,
        lam_rf_min=args.lam_rf_min, rng_seed=0)
    ql = _build_qso_lookup(cfg)
    cat_cut, truth_cut19, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=19.0, qso_lookup=ql, host_truth_floor=19.0)
    from CDDF_analysis.hbi.cddf_catalog_hbi import build_pathlength, build_fine_grid
    X_tot, n_sl = build_pathlength(cfg, qso_lookup=ql)
    lo, hi, N_b, dN_b = build_fine_grid(cfg)
    t = tilted_truth_reductions(cfg, truth_cut19, lo, hi, N_b, dN_b, X_tot, dalpha=0.0)
    return float(t["dndx_total"][19.0] - t["dndx_total"][19.5])


def main(args):
    t_start = time.time()
    n_mc = args.n_mc
    seed = args.seed
    out = {}
    for knobs in CONFIGS:
        out[knobs["name"]] = run_one(knobs, n_mc, seed)

    base195 = out["floor19.5"]
    tru_band = base195["dndx_tru_band"]
    tru_bin = base195["dndx_tru_bin"]
    # config-independent padding-band truth for the gate-#4 over-recovery ratio
    try:
        pad_tru_real = padding_truth_dndx()
    except Exception as exc:
        print(f"[warn] padding_truth_dndx failed ({exc}); pad ratio diagnostic only")
        pad_tru_real = float("nan")
    for name in out:
        out[name]["pad_tru"] = pad_tru_real
    base195 = out["floor19.5"]

    print("\n" + "#" * 84)
    print("# RESULT 1 — per-0.1-dex R0 (point + MC q16,q84) across the sub-DLA band")
    print("#" * 84)
    hdr = f"{'bin':>14} | {'truth dndx':>11} |"
    for name in ("floor19.5", "pad19.2", "pad19.0"):
        hdr += f" {name+' R0(q16,q84)':>24} |"
    print(hdr); print("-" * len(hdr))
    for bi, (blo, bhi) in enumerate(PER_BINS):
        row = f"[{blo:.1f},{bhi:.1f})".rjust(14) + f" | {tru_bin[bi]:>11.5g} |"
        for name in ("floor19.5", "pad19.2", "pad19.0"):
            r = out[name]
            pt = r["r0_bin_point"][bi]
            q16 = np.nanpercentile(r["mc_bin"][:, bi] / tru_bin[bi], 16)
            q84 = np.nanpercentile(r["mc_bin"][:, bi] / tru_bin[bi], 84)
            row += f" {pt:>6.3f} ({q16:.3f},{q84:.3f}) |"
        print(row)

    print("\n" + "#" * 84)
    print("# RESULT 2 — integrated [19.5,20.3) dN/dX & Ω R0 with MC band; DLA tier")
    print("#" * 84)
    for name in ("floor19.5", "pad19.2", "pad19.0"):
        r = out[name]
        mu, sd, q16, q50, q84 = _qstats(r["mc_band"] / tru_band)
        print(f"\n{name} (basis_pad_floor={r['basis_pad_floor']}):")
        print(f"   dN/dX[19.5,20.3) R0 point={r['r0_band_point']:.4f}  "
              f"MC mean={mu:.4f} std={sd:.4f} q16={q16:.4f} q84={q84:.4f}")
        print(f"   Ω[19.5,20.3)     R0 point={r['r0_omega_band']:.4f}")
        a203 = r["mc_r0_203"][np.isfinite(r["mc_r0_203"])]
        print(f"   DLA tier R0(>=20.3) point={r['r0_203_point']:.4f}  "
              f"MC mean={np.nanmean(a203):.4f} std={np.nanstd(a203):.4f}")

    print("\n" + "#" * 84)
    print("# RESULT 3 — padding [19.0,19.5) recovery (UNREPORTED support; gate #4)")
    print("#" * 84)
    print(f"   truth dN/dX[19.0,19.5) = {base195['pad_tru']:.5g}  "
          f"(floor-19.0's failure dumped 0.192 here → R0≈3.2)")
    for name in ("floor19.5", "pad19.2", "pad19.0"):
        r = out[name]
        ratio = r["pad_est0"] / r["pad_tru"] if r["pad_tru"] > 0 else np.nan
        print(f"   {name:>10}: dN/dX[19.0,19.5)={r['pad_est0']:.5g}  R0={ratio:.3f}  "
              f"(<=0.12 and <=2× truth ? {'YES' if r['pad_est0']<=0.12 else 'NO'})")

    print("\n" + "#" * 84)
    print("# RESULT 4 — count conservation: total recovered dN/dX[19.5,20.3)")
    print("#" * 84)
    print(f"   truth total = {tru_band:.6g}")
    for name in ("floor19.5", "pad19.2", "pad19.0"):
        r = out[name]
        print(f"   {name:>10}: recovered = {r['dndx_est0_band']:.6g}  "
              f"(Δ vs floor-19.5 = {r['dndx_est0_band']-base195['dndx_est0_band']:+.6g})")

    print("\n" + "#" * 84)
    print("# RESULT 5 — PRE-REGISTERED GATE (5 criteria; ALL must hold)")
    print("#" * 84)
    gates = {}
    for name in ("pad19.0", "pad19.2"):
        g = evaluate_gate(base195, out[name], tru_band, tru_bin)
        gates[name] = g
        print(f"\n--- {name} : {'PASS' if g['passed'] else 'FAIL'} ---")
        for k, v in g["criteria"].items():
            print(f"   [{ 'PASS' if v['passed'] else 'FAIL' }] {k}")
            for kk, vv in v.items():
                if kk in ("passed", "detail"):
                    continue
                print(f"          {kk} = {vv}")
            if "detail" in v:
                for d in v["detail"]:
                    print(f"          {d['bin']}: r0={d['r0']:.3f} thr={d['thr']:.3f} "
                          f"{'OK' if d['passed'] else 'FAIL'}")

    # persist TSV
    out_tsv = "/tmp/subdla_basis_pad_bracket.tsv"
    with open(out_tsv, "w") as fh:
        fh.write("metric\tbin\ttruth\tfloor19.5\tpad19.2\tpad19.0\n")
        for bi, (blo, bhi) in enumerate(PER_BINS):
            lab = f"[{blo:.1f},{bhi:.1f})"
            fh.write(f"r0_dndx_bin\t{lab}\t1.0\t"
                     f"{out['floor19.5']['r0_bin_point'][bi]:.6g}\t"
                     f"{out['pad19.2']['r0_bin_point'][bi]:.6g}\t"
                     f"{out['pad19.0']['r0_bin_point'][bi]:.6g}\n")
        fh.write(f"R0_dndx_195_203\t-\t1.0\t{out['floor19.5']['r0_band_point']:.6g}\t"
                 f"{out['pad19.2']['r0_band_point']:.6g}\t{out['pad19.0']['r0_band_point']:.6g}\n")
        fh.write(f"R0_omega_195_203\t-\t1.0\t{out['floor19.5']['r0_omega_band']:.6g}\t"
                 f"{out['pad19.2']['r0_omega_band']:.6g}\t{out['pad19.0']['r0_omega_band']:.6g}\n")
        fh.write(f"R0_dndx_203\t-\t1.0\t{out['floor19.5']['r0_203_point']:.6g}\t"
                 f"{out['pad19.2']['r0_203_point']:.6g}\t{out['pad19.0']['r0_203_point']:.6g}\n")
        fh.write(f"dndx_pad_190_195\t-\t{base195['pad_tru']:.6g}\t{out['floor19.5']['pad_est0']:.6g}\t"
                 f"{out['pad19.2']['pad_est0']:.6g}\t{out['pad19.0']['pad_est0']:.6g}\n")
        fh.write(f"dndx_total_195_203\t-\t{tru_band:.6g}\t{out['floor19.5']['dndx_est0_band']:.6g}\t"
                 f"{out['pad19.2']['dndx_est0_band']:.6g}\t{out['pad19.0']['dndx_est0_band']:.6g}\n")
        for name in ("pad19.0", "pad19.2"):
            fh.write(f"GATE_{name}\t-\t-\t{'PASS' if gates[name]['passed'] else 'FAIL'}\t-\t-\n")
    print(f"\n[saved] {out_tsv}")
    np.savez("/tmp/subdla_basis_pad_bracket_result.npz",
             tru_band=tru_band, tru_bin=tru_bin, pad_tru=base195["pad_tru"],
             **{f"{n}_mc_band": out[n]["mc_band"] for n in out},
             **{f"{n}_mc_bin": out[n]["mc_bin"] for n in out},
             **{f"{n}_r0_band_point": out[n]["r0_band_point"] for n in out},
             **{f"{n}_r0_bin_point": out[n]["r0_bin_point"] for n in out},
             **{f"{n}_pad_est0": out[n]["pad_est0"] for n in out},
             **{f"{n}_r0_203": out[n]["r0_203_point"] for n in out})
    print("[saved] /tmp/subdla_basis_pad_bracket_result.npz")

    # ---- committed, git-stamped JSON deliverable (edge-mass bracket; mock; public-OK) ----
    def _cfg_block(r):
        tb = r["dndx_tru_band"]; to = r["om_t_band"]; tbin = r["dndx_tru_bin"]
        mc_r0_dndx = _qstats(r["mc_band"] / tb)          # (mean,std,q16,q50,q84)
        mc_r0_om = _qstats(r["mc_omega"] / to)
        per_bin_q16 = [float(np.nanpercentile(r["mc_bin"][:, bi] / tbin[bi], 16))
                       for bi in range(len(PER_BINS))]
        per_bin_q84 = [float(np.nanpercentile(r["mc_bin"][:, bi] / tbin[bi], 84))
                       for bi in range(len(PER_BINS))]
        a203 = r["mc_r0_203"][np.isfinite(r["mc_r0_203"])]
        return dict(
            basis_pad_floor=r["basis_pad_floor"], n_sl=int(r["n_sl"]),
            dndx=dict(
                r0_band_195_203=dict(
                    point=float(r["r0_band_point"]), mean=float(mc_r0_dndx[0]),
                    std=float(mc_r0_dndx[1]), q16=float(mc_r0_dndx[2]),
                    q50=float(mc_r0_dndx[3]), q84=float(mc_r0_dndx[4])),
                r0_per_bin_point=[float(x) for x in r["r0_bin_point"]],
                r0_per_bin_q16=per_bin_q16, r0_per_bin_q84=per_bin_q84,
                est0_band_195_203=float(r["dndx_est0_band"]),
                est0_per_bin=[float(x) for x in r["dndx_est0_bin"]]),
            omega=dict(
                r0_band_195_203=dict(
                    point=float(r["r0_omega_band"]), mean=float(mc_r0_om[0]),
                    std=float(mc_r0_om[1]), q16=float(mc_r0_om[2]),
                    q50=float(mc_r0_om[3]), q84=float(mc_r0_om[4])),
                est0_band_195_203=float(r["om_e_band"])),
            padding_190_195=dict(
                est0=float(r["pad_est0"]), truth=float(r["pad_tru"]),
                ratio=float(r["pad_est0"] / r["pad_tru"]) if r["pad_tru"] > 0 else float("nan")),
            dla_tier=dict(
                r0_dndx_203_point=float(r["r0_203_point"]),
                r0_dndx_203_mc=dict(mean=float(np.nanmean(a203)), std=float(np.nanstd(a203)))),
        )

    inputs = dict(
        n_mc=int(n_mc), seed=int(seed),
        catalog_dir=AB.DEF_CAT, truth=AB.DEF_TRUTH, bal_cat=AB.DEF_BAL,
        kernel=KERNEL, molly=MOLLY, loa0_product=LOA0_PRODUCT,
        fit_floor=19.5, lam_rf_min=1025.0, family="bspbody", host_truth_floor=19.0,
        report_limits=",".join(f"{x:g}" for x in REPORT_LIMITS),
        basis_pad_configs=[dict(name=c["name"], basis_pad_floor=c["basis_pad_floor"])
                           for c in CONFIGS],
    )
    # FAIL-CLOSED kernel self-declaration. Raises if RESP_KIND/PAPER_FACING are ever
    # edited into the forbidden combination (kappa + paper_facing=True).
    kernel_md = RK.kernel_metadata(
        RESP_KIND, context="sub-DLA basis-pad bracket (kappa diagnostic)",
        paper_facing=PAPER_FACING,
        extra_note=("The BRACKET (pad-19.0 / pad-19.2 vs no-pad) is the intended "
                    "observable and is a within-kernel difference; the R0 LEVELS carry "
                    "the kappa<->forward kernel gap. Pre-registered gate criterion 5 "
                    "(R0>=20.3 = 1.159 +- 0.014) is itself a KAPPA number."))
    out_json = dict(
        metadata=dict(
            what="Decoupled basis-padding bracket isolating the geometric edge-mass recovery at "
                 "the 19.5 sub-DLA floor: sub-DLA integrated [19.5,20.3) R0 (dN/dX + Omega) and "
                 "per-0.1-dex R0 for the headline (floor-19.5, no pad) vs basis-pad 19.2 / 19.0, "
                 "with the [19.0,19.5) padding-band over-recovery and the 5-criterion gate.",
            mock="2LPT-0 (loa-124); values are MOCK recovery ratios + MC bands, not real-LOA",
            **kernel_md,
            code_commit=_git_commit(),
            wallclock_s=round(time.time() - t_start, 1),
            rederive=("python CDDF_analysis/diagnostics/subdla/subdla_basis_pad_bracket.py "
                      "--force --n-mc 150 --seed 0"),
            inputs=inputs,
            note="Reduce-only (cached FLOOR-19.5 kernel, NO rebuild; NO inference/SLURM/tilt; "
                 "loa0 FP FROZEN per draw). basis_pad_floor lowers ONLY the deconvolution basis "
                 "+ marked-Poisson normalizer support below 19.5, keeping the detection set / "
                 "molly C,rho / mu_FP at 19.5. R0 = est/truth; per-bin R0 is dN/dX. This routine "
                 "reports the BRACKET; it does NOT prescribe the edge-migration systematic number "
                 "(the review panel adjudicates the construction across {headline 19.5, basis-pad "
                 "19.2, basis-pad 19.0, floor-19.0 rebuild}). floor-19.0 is cross-referenced from "
                 "the sibling subdla_floor_mc_band.json (this routine does not recompute it).",
        ),
        result=dict(
            per_bins=[[blo, bhi] for blo, bhi in PER_BINS],
            truth=dict(
                dndx_band_195_203=float(tru_band),
                omega_band_195_203=float(base195["om_t_band"]),
                dndx_per_bin=[float(x) for x in tru_bin],
                padding_190_195=float(base195["pad_tru"])),
            headline_floor19_5=_cfg_block(out["floor19.5"]),
            basis_pad_19_2=_cfg_block(out["pad19.2"]),
            basis_pad_19_0=_cfg_block(out["pad19.0"]),
            floor19_0_reference=_floor190_reference(),
            gate=dict((name, dict(passed=bool(gates[name]["passed"]),
                                  criteria=gates[name]["criteria"]))
                      for name in gates),
        ),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out) and not args.force:
        print(f"[skip-json] {args.out} exists (pass --force to overwrite).")
    else:
        with open(args.out, "w") as fh:
            json.dump(out_json, fh, indent=2, default=float)
        print(f"[saved-json] {args.out}  code_commit={out_json['metadata']['code_commit']}  "
              f"({out_json['metadata']['wallclock_s']:.0f}s)")
    return out, gates


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT_JSON,
                    help="stamped JSON deliverable path (default: committed mock artifact).")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out if it already exists (default: refuse).")
    ap.add_argument("--n-mc", type=int, default=int(os.environ.get("N_MC", "150")),
                    dest="n_mc", help="MC draws per config (default 150; env N_MC honored).")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")),
                    help="base RNG seed (default 0; env SEED honored).")
    main(ap.parse_args())
