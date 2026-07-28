"""subdla_floor_mc_band.py — MC error bar on the sub-DLA integrated [19.5,20.3) R0
and per-0.1-dex R0, for floor-19.5 (current) vs floor-19.0 (rebuild).

################################################################################
# KAPPA-KERNEL DIAGNOSTIC — NOT PAPER-FACING.  resp_kind = 'kappa'.
#
# This routine runs on the GP-POSTERIOR ('kappa') response kernel, via
# ab_loa0_fp_baseline.build_ingredients, which constructs its HBIConfig without
# passing resp_kind and therefore inherits the dataclass default.  Until 2026-07-28
# that inheritance was IMPLICIT (indistinguishable from an oversight); it is now
# STATED (`cfg.resp_kind = RESP_KIND` in run_one) and the artifact SELF-DECLARES
# metadata['resp_kind']='kappa' + metadata['paper_facing']=False, enforced by
# CDDF_analysis/unblind/resp_kind.kernel_metadata (which raises if anyone flips
# PAPER_FACING to True while RESP_KIND stays 'kappa').
#
# WHY IT IS NOT PORTED TO THE FORWARD KERNEL.  The forward path is a different
# ingredient builder (track_c_tf_2lpt1.build_frozen_calibration +
# build_heldout_ingredients + track_c_perz_band._set_forward_cfg) returning a
# different dict shape, and the floor-19.5-vs-19.0 comparison this routine exists
# to make is a COUNT-CONSERVATION diagnostic of the deconvolution basis, not a
# paper number.  Porting is a rewrite plus a full re-run, not a knob flip.
#
# CONSEQUENCE, stated plainly: every R0 in this artifact carries the kappa<->forward
# kernel-object gap (sub-DLA band: kappa 0.883/0.899 vs forward 0.849/0.822, i.e.
# 3.8% in dN/dX and 8.5% in Omega).  The floor-19.5-vs-19.0 DIFFERENCE is the
# intended observable and is far less kernel-sensitive than either level.
################################################################################

Reduce-only, cached kernel, NO inference, NO SLURM, NO tilt (Delta_alpha=0, the
UNTILTED baseline recovery, same as subdla_loa0_validation*.py). The point R0 is
reported with NO error bar in the prior docs; this puts the joint-MC band on it.

The MC band contains EXACTLY three variance channels (plus MAP multistart jitter).
Each draw re-MAPs theta warm-started at the point MAP and reduces (v3x_reduce), so
the band is the parametric-refit band (NOT a v1 1/Vmax fallback):
  * completeness C: Wilson/Jeffreys-Beta resample per molly cell (_draw_beta_cell)
  * sightline bootstrap: TID-multiplicity multinomial -> per-op-row weight
  * loa0 FP normalization: the Gehrels Gamma(n_FP+1/2, 1/ell_eff) resample of the
    loa-0 forest-FP counts (Loa0FP.resample, "FIX 3"), on an INDEPENDENT RNG stream
    so it does not perturb the C/bootstrap draws. This is the DOMINANT channel: the
    sub-DLA loa-0 FP rests on ~89 loa-0 detections (~10.6% Poisson), so FREEZING it
    (the prior defect) collapsed the band ~7.5x AND inverted the floor conclusion
    (frozen sigma ~+-0.7% -> FP-resampled ~+-6%).

--freeze-fp (default OFF) restores the prior FROZEN-FP behaviour BYTE-IDENTICALLY
(same C/bootstrap stream; loa0 FP held at the point value) as a regression guard.

spec §7 clarification: §7 forbids resampling the forest FP *coupled to the catalog
purity* (the circular purity-mixture form). It does NOT forbid resampling the loa-0
FP by its OWN Gamma -- exactly what Loa0FP.resample implements and what
PurityMixtureFP's docstring calls "the loa-0 model's job". The prior freeze
conflated the two and dropped the dominant variance channel.

NOT in the band (documented here, and in the JSON `band_channels_excluded`; NOT
silently dropped):
  * NHI_ERR (sigma_i) Eddington edge scatter: A/M are rebuilt from the CACHED kernel
    per draw and are NOT re-sliced on perturbed NHI, so this channel feeds nothing.
    Wiring it would require a per-draw kernel re-slice + sparse-A rebuild (recomputing
    which detections clear the floor and which (N,SNR) cell each lands in) -- outside
    the reduce-only budget. Dropped honestly, not faked with a dead code path.
  * rho purity-mixture term: the loa0 FP estimator does not use rho.

Run BOTH configs (floor-19.5, floor-19.0), BOTH on the loa0 estimator, n_mc draws
each, and report:
  * integrated [19.5,20.3) R0: point, q16/q50/q84, std, and the z-score for the
    floor-19.5 vs floor-19.0 DIFFERENCE (paired same-truth, so the difference band
    is the per-draw difference distribution).
  * per-0.1-dex R0 bands across [19.5,20.3), so the mid-band drops can be tested.
  * total recovered dN/dX(19.5,20.3) per config (count-conservation / see-saw test).
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
    _draw_beta_cell, _cell_index, _slice_active_unitC, _rescale_unitC_active,
    _apply_C_to_M, v3x_fit_map, v3x_reduce, C_FLOOR,
)
from CDDF_analysis.unblind import resp_kind as RK

# ---- RESPONSE-KERNEL DECLARATION (see the module banner) --------------------
# STATED, not inherited. RK.kernel_metadata() below refuses to stamp
# paper_facing=True while RESP_KIND is 'kappa', so this pair cannot drift apart.
RESP_KIND = RK.RESP_KIND_KAPPA
PAPER_FACING = False

# ---- config knobs (the three repointed for floor-19.0) ----------------------
F195 = dict(
    name="floor19.5",
    kernel=AB.DEF_KERNEL,                       # mollynhi195_lyaonly1025_broaden012
    molly=AB.DEF_LYAONLY_MOLLY,                 # figures_molly_nhi195/lya_only
    fit_floor=19.5,
)
F190 = dict(
    name="floor19.0",
    kernel=("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
            "phase3d_experiments/floor190_lyaonly1025_broaden012/"
            "posterior_kernel_2lpt0.npz"),
    molly=("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
           "figures_molly_nhi190/molly_matrix.tsv"),
    fit_floor=19.0,
)
LOA0_PRODUCT = ("/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/"
                "outputs/loa0_fp_product_lyaonly1025.npz")
REPORT_LIMITS = (19.5, 19.6, 19.7, 19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.6)
PER_BINS = [(round(19.5 + 0.1 * k, 1), round(19.6 + 0.1 * k, 1)) for k in range(8)]

# committed, git-stamped MOCK deliverable (2LPT-0 recovery ratios + MC bands — public-OK,
# no real-LOA values).
DEFAULT_OUT_JSON = os.path.join(_REPO, "CDDF_analysis", "hbi", "subdla_floor_mc_band.json")


def _git_commit(routine=None):
    """HEAD, suffixed `-dirty` iff the ROUTINE that produced this artifact (or a diagnostic
    module it imports) is untracked or modified.

    A `-dirty` stamp means the artifact is NOT third-party re-derivable: the named commit
    does not contain (this version of) the routine. Commit the routine first, then re-run.
    Checking the routine specifically -- rather than `git status`, which is dirtied by the
    artifact's own untracked output -- is what makes the marker meaningful. The estimator
    modules (ab_loa0_fp_baseline / cddf_tilt_closure) are listed because without them the
    stamped `rederive` command cannot reproduce the numbers.
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


class _Args:
    def __init__(self, knobs):
        self.catalog_dir = AB.DEF_CAT
        self.truth = AB.DEF_TRUTH
        self.bal_cat = AB.DEF_BAL
        self.molly_tsv = knobs["molly"]
        self.kernel = knobs["kernel"]
        self.loa0_product = LOA0_PRODUCT
        self.out = "/tmp/subdla_floor_mc_band_" + knobs["name"]
        self.mockdir = None
        self.zbins = "2.0,2.5,3.0,3.5"
        self.report_limits = ",".join(f"{x:g}" for x in REPORT_LIMITS)
        self.family = "bspbody"
        self.fit_floor = knobs["fit_floor"]
        self.fit_ceil = 99.0
        self.lambda_bspbody = 30.0
        self.lam_rf_min = 1025.0
        self.edge_slope_lam = 40.0
        self.gl_nodes = 1
        self.host_truth_floor = 19.0


def _band_dndx(red_dndx_total, lo=19.5, hi=20.3):
    """integrated dN/dX over [lo,hi) from the cumulative dndx_total dict."""
    return red_dndx_total[lo] - red_dndx_total[hi]


def run_one(knobs, n_mc, seed, freeze_fp=False):
    """One config's point + MC band. freeze_fp=True holds the loa0 FP at its point
    value (the prior defect; regression guard); default resamples it (Gehrels Gamma).

    Variance channels (default): completeness C (Beta), sightline bootstrap, loa0 FP
    normalization (Gehrels Gamma). The rho Beta + NHI_ERR width draws are drawn ONLY
    to keep the `rng` stream byte-aligned with the frozen baseline (so --freeze-fp
    reproduces it exactly); both contribute ZERO variance in loa0 FP mode and are NOT
    band channels. The loa0 FP resample runs on an INDEPENDENT rng_fp stream so the
    frozen path stays bit-reproducible regardless of freeze_fp.
    """
    args = _Args(knobs)
    os.makedirs(args.out, exist_ok=True)
    print("=" * 78)
    print(f"[MC band] {knobs['name']}  fit_floor={knobs['fit_floor']}  n_mc={n_mc}"
          f"  FP={'FROZEN' if freeze_fp else 'RESAMPLED (Gehrels Gamma)'}")
    print("=" * 78)
    ing = AB.build_ingredients(args, "loa0", loa0_product=args.loa0_product)
    cfg = ing["cfg"]
    # EXPLICIT KERNEL DECLARATION (2026-07-28). AB.build_ingredients constructs an
    # HBIConfig without passing resp_kind, so this routine used to run on the kappa
    # kernel purely by INHERITING the dataclass default -- indistinguishable from an
    # oversight. State it. See RESP_KIND / the module banner for why it stays kappa.
    cfg.resp_kind = RESP_KIND
    cfg._wall1_estimator = "v3"
    cfg.v3_mc_n_restart = 2  # warm-started per draw; cheap

    # ---- point estimate (loa0, untilted) ----
    base = baseline_recovery(
        cfg, ing["cat_cut"], ing["is_TP"], ing["good_mask"], ing["truth_cut"],
        ing["C_interp"], ing["fp_model"], ing["X_tot"],
        ing["logN_lo"], ing["logN_hi"], ing["N_b"], ing["dN_b"],
        estimator_fn=ing["estimator_fn"])
    e0 = base["e0"]; t0 = base["t0"]
    logN_lo = np.asarray(ing["logN_lo"], float)
    logN_hi = np.asarray(ing["logN_hi"], float)
    dN_b = np.asarray(ing["dN_b"], float)

    # truth (FIXED across draws) — per-bin dN/dX truth and integrated
    f_tru = np.asarray(t0["f_truth"], float)
    dndx_tru_bin = np.array([
        np.nansum(f_tru[(logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)]
                  * dN_b[(logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)])
        for blo, bhi in PER_BINS])
    dndx_tru_band = _band_dndx(t0["dndx_total"])

    # point per-bin / integrated dN/dX (est) and R0
    f_est0 = np.asarray(e0["f_b"], float)
    dndx_est0_bin = np.array([
        np.nansum(f_est0[(logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)]
                  * dN_b[(logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)])
        for blo, bhi in PER_BINS])
    dndx_est0_band = _band_dndx(e0["dndx_total"])
    r0_band_point = dndx_est0_band / dndx_tru_band
    r0_bin_point = dndx_est0_bin / dndx_tru_bin
    r0_203_point = base["R0_dndx_total"][20.3]
    # integrated Omega band [19.5,20.3) (point) — same cumulative-difference recipe as dN/dX
    om_e0_band = float(e0["omega"][19.5] - e0["omega"][20.3])
    om_t_band = float(t0["omega"][19.5] - t0["omega"][20.3])
    r0_omega_band_point = (om_e0_band / om_t_band) if om_t_band > 0 else np.nan

    # ---- MC: reuse the point fwd; resample C + bootstrap + loa0 FP (Gehrels Gamma).
    # rho/sigma are DRAWN each iter only to keep the `rng` stream byte-aligned with the
    # frozen baseline (see run_one docstring); they feed NOTHING in loa0 mode.
    fwd = e0["_v3x"]["fwd"]
    family = e0["_v3x"]["family"]
    fine = e0["_v3x"]["fine"]
    M_meta = e0["_v3x"]["M_meta"]
    theta_map = e0["_v3x"]["theta_map"]
    A_meta = fwd["A_meta"]; cat_op = fwd["cat_op"]
    lam_fp_frozen = fwd["lam_fp"]; mu_fp_frozen = fwd["mu_fp"]   # point FP (freeze-fp path)
    logN_fit_floor = fwd["logN_fit_floor"]                       # loa0 mu_FP support
    active_flat = fwd["active_flat"]
    keep_in_base = fwd["keep_in_base"]
    snr_op = cat_op["snr"]; i_snr0 = cat_op["i_snr"]
    z_edges_fine = fine[4]
    n_flat = len(logN_lo) * (len(z_edges_fine) - 1)
    unitC = _slice_active_unitC(A_meta, np.arange(n_flat), np.ones(A_meta["n_obs"], bool))

    mm = ing["mm"]
    cat_cut = ing["cat_cut"]; good_mask = ing["good_mask"]
    s2n_all = np.asarray(cat_cut["S2N_RED"], float)
    pdla_all = np.asarray(cat_cut["P_DLA"], float)
    op_base = (s2n_all > cfg.snr_min) & (pdla_all > cfg.p_dla_min) & good_mask
    nhi0_base = np.asarray(cat_cut["NHI"], float)[op_base]   # length only (stream-align draw)
    tids_base = np.asarray(cat_cut["TARGETID"], np.int64)[op_base]
    # bootstrap over the FULL op_base sightlines (then slice to floored subset, exactly
    # as joint_mc_errors/make_v3x_refit_fn: boot_w op_base-ordered -> [keep_in_base])
    uniq_tids, inv = np.unique(tids_base, return_inverse=True)
    n_uniq = len(uniq_tids)

    rng = np.random.default_rng(seed)
    # loa0 FP resample stream — INDEPENDENT of `rng` (derived deterministically from
    # seed) so the C/rho/nhi/bootstrap draws are untouched and the --freeze-fp path
    # stays bit-reproducible. Advanced once per draw via Loa0FP.resample(rng_fp).
    rng_fp = np.random.default_rng([int(seed), 202607081])
    mc_band = np.full(n_mc, np.nan)        # integrated dN/dX [19.5,20.3) (est)
    mc_omega = np.full(n_mc, np.nan)       # integrated Omega [19.5,20.3) (est)
    mc_bin = np.full((n_mc, len(PER_BINS)), np.nan)  # per-bin dN/dX (est)
    mc_r0_203 = np.full(n_mc, np.nan)

    for m in range(n_mc):
        # CHANNEL 1 — completeness C (per-molly-cell Beta resample) --------------
        C_draw = _draw_beta_cell(rng, mm.cmp_nfound, mm.cmp_nfid)
        # STREAM-ALIGN (NOT a channel): rho Beta is the purity-mixture FP term; the
        # loa0 estimator never uses rho. Drawn only to keep `rng` byte-aligned with
        # the frozen baseline (--freeze-fp regression guard). Consumes rng, feeds nothing.
        _draw_beta_cell(rng, mm.pur_ntp, mm.pur_ntot)
        C_draw = np.where((mm.cmp_nfid > 0), C_draw, C_FLOOR)
        # STREAM-ALIGN (NOT a channel): NHI_ERR (sigma_i) Eddington edge scatter. A/M
        # are rebuilt from the CACHED kernel below and are NOT re-sliced on perturbed
        # NHI, so this feeds nothing (see run_one/module docstring). Same rng draw as
        # the frozen baseline (len(nhi0_base) normals) so byte-identity is preserved.
        rng.normal(0.0, 1.0, len(nhi0_base))
        # CHANNEL 2 — sightline bootstrap (TID multiplicity) --------------------
        mult = rng.multinomial(n_uniq, np.full(n_uniq, 1.0 / n_uniq))
        boot_w_base = mult[inv].astype(float)
        boot_w = boot_w_base[keep_in_base]        # slice to floored op subset
        # CHANNEL 3 — loa0 FP normalization (Gehrels Gamma), on rng_fp (independent) -
        # The dominant channel. --freeze-fp holds it at the point value byte-identically.
        if freeze_fp:
            lam_fp_d, mu_fp_d = lam_fp_frozen, mu_fp_frozen
        else:
            loa0_draw = cfg._loa0_fp.resample(rng_fp)     # Gamma(n_FP+1/2, 1/ell_eff)
            lam_fp_d = loa0_draw.lam_fp_per_obj(cat_op["xhat"], cat_op["snr"]).astype(float)
            mu_fp_d = loa0_draw.mu_fp_scalar(logN_fit_floor=logN_fit_floor)
        # C-rescale A/M (same as make_v3x_refit_fn)
        A_draw = _rescale_unitC_active(unitC, C_draw)
        M_draw = np.where(active_flat, _apply_C_to_M(M_meta, C_draw), 0.0)
        fit = v3x_fit_map(A_draw, M_draw, lam_fp_d, mu_fp_d, fine, family, cfg,
                          obj_weights=boot_w, theta0=theta_map, n_restart=2,
                          rng=np.random.default_rng(seed * 100003 + m), lit_start=False)
        rr = v3x_reduce(cfg, fit["theta_map"], fine, family, M_meta)
        f_b = np.asarray(rr["f_b"], float)
        mc_band[m] = _band_dndx(rr["dndx_total"])
        mc_omega[m] = rr["omega"][19.5] - rr["omega"][20.3]   # post-proc; consumes no RNG
        for bi, (blo, bhi) in enumerate(PER_BINS):
            sel = (logN_lo >= blo - 1e-6) & (logN_hi <= bhi + 1e-6)
            mc_bin[m, bi] = np.nansum(f_b[sel] * dN_b[sel])
        mc_r0_203[m] = rr["dndx_total"][20.3] / t0["dndx_total"][20.3]
        if (m + 1) % 25 == 0:
            print(f"    draw {m+1}/{n_mc}")

    return dict(
        name=knobs["name"], n_sl=int(ing["n_sl"]),
        dndx_tru_band=float(dndx_tru_band), dndx_tru_bin=dndx_tru_bin,
        dndx_est0_band=float(dndx_est0_band), dndx_est0_bin=dndx_est0_bin,
        r0_band_point=float(r0_band_point), r0_bin_point=r0_bin_point,
        r0_203_point=float(r0_203_point),
        om_e0_band=om_e0_band, om_t_band=om_t_band,
        r0_omega_band_point=float(r0_omega_band_point),
        mc_band=mc_band, mc_omega=mc_omega, mc_bin=mc_bin, mc_r0_203=mc_r0_203,
    )


def main(args):
    t_start = time.time()
    n_mc = args.n_mc
    seed = args.seed
    freeze_fp = bool(getattr(args, "freeze_fp", False))
    out = {}
    for knobs in (F195, F190):
        out[knobs["name"]] = run_one(knobs, n_mc, seed, freeze_fp=freeze_fp)

    r195 = out["floor19.5"]; r190 = out["floor19.0"]
    tru_band = r195["dndx_tru_band"]
    tru_bin = r195["dndx_tru_bin"]

    def _q(a):
        a = a[np.isfinite(a)]
        return (np.nanmean(a), np.nanstd(a),
                np.nanpercentile(a, 16), np.nanpercentile(a, 50), np.nanpercentile(a, 84))

    print("\n" + "#" * 78)
    print("# RESULT 1 — integrated [19.5,20.3) R0 with MC band")
    print("#" * 78)
    for r in (r195, r190):
        mc_r0 = r["mc_band"] / tru_band
        mu, sd, q16, q50, q84 = _q(mc_r0)
        print(f"\n{r['name']}: point R0 = {r['r0_band_point']:.4f}")
        print(f"   MC: mean={mu:.4f} std={sd:.4f} q16={q16:.4f} q50={q50:.4f} q84={q84:.4f}")
        print(f"   truth dN/dX[19.5,20.3) = {tru_band:.6g}; "
              f"point est dN/dX = {r['dndx_est0_band']:.6g}")

    # paired difference (same truth, INDEPENDENT draws -> conservative; also report the
    # naive quadrature combine). The cleaner test: is |R0_195 - R0_190| > combined sigma?
    mc_r0_195 = r195["mc_band"] / tru_band
    mc_r0_190 = r190["mc_band"] / tru_band
    n = min(len(mc_r0_195), len(mc_r0_190))
    # paired per-draw difference (shared C/rho/bootstrap draws via shared seed) — the
    # two configs use DIFFERENT kernels/molly so draws are not literally the same cells,
    # but the same RNG seed makes the bootstrap multiplicities correlated. Report the
    # per-draw difference band AND the independent-quadrature band.
    diff_paired = mc_r0_195[:n] - mc_r0_190[:n]
    dpm, dps = np.nanmean(diff_paired), np.nanstd(diff_paired)
    s195 = np.nanstd(mc_r0_195); s190 = np.nanstd(mc_r0_190)
    s_quad = np.hypot(s195, s190)
    dpoint = r195["r0_band_point"] - r190["r0_band_point"]
    print("\n--- DIFFERENCE: floor-19.5 minus floor-19.0 (integrated [19.5,20.3) R0) ---")
    print(f"   point diff           = {dpoint:+.4f}")
    print(f"   per-config sigma      : 19.5 {s195:.4f}, 19.0 {s190:.4f}")
    print(f"   independent-quadrature sigma = {s_quad:.4f}  ->  z = {dpoint / s_quad:.2f}")
    print(f"   paired-draw diff      : mean={dpm:+.4f} std={dps:.4f}  ->  z = {dpoint / dps:.2f}")

    print("\n" + "#" * 78)
    print("# RESULT 1b — integrated [19.5,20.3) OMEGA R0 with MC band")
    print("#" * 78)
    for r in (r195, r190):
        tru_om = r["om_t_band"]
        mc_r0_om = r["mc_omega"] / tru_om
        mu, sd, q16, q50, q84 = _q(mc_r0_om)
        print(f"\n{r['name']}: point Omega R0 = {r['r0_omega_band_point']:.4f}")
        print(f"   MC: mean={mu:.4f} std={sd:.4f} q16={q16:.4f} q50={q50:.4f} q84={q84:.4f}")
        print(f"   truth Omega[19.5,20.3) = {tru_om:.6g}; "
              f"point est Omega = {r['om_e0_band']:.6g}")

    print("\n" + "#" * 78)
    print("# RESULT 2 — per-0.1-dex R0 with MC band (floor-19.5 vs floor-19.0)")
    print("#" * 78)
    print(f"{'bin':>14} | {'truth dndx':>11} | {'f19.5 R0 (q16,q84)':>26} | "
          f"{'f19.0 R0 (q16,q84)':>26} | {'z(diff)':>8}")
    print("-" * 116)
    for bi, (blo, bhi) in enumerate(PER_BINS):
        tb = tru_bin[bi]
        r0_195 = r195["mc_bin"][:, bi] / tb
        r0_190 = r190["mc_bin"][:, bi] / tb
        p195 = r195["r0_bin_point"][bi]; p190 = r190["r0_bin_point"][bi]
        s1 = np.nanstd(r0_195); s2 = np.nanstd(r0_190)
        sq = np.hypot(s1, s2)
        z = (p195 - p190) / sq if sq > 0 else np.nan
        lab = f"[{blo:.1f},{bhi:.1f})"
        print(f"{lab:>14} | {tb:>11.5g} | "
              f"{p195:>6.3f} ({np.nanpercentile(r0_195,16):.3f},{np.nanpercentile(r0_195,84):.3f}) | "
              f"{p190:>6.3f} ({np.nanpercentile(r0_190,16):.3f},{np.nanpercentile(r0_190,84):.3f}) | "
              f"{z:>+8.2f}")

    print("\n" + "#" * 78)
    print("# RESULT 3 — count conservation / see-saw: total recovered dN/dX [19.5,20.3)")
    print("#" * 78)
    print(f"   truth total dN/dX[19.5,20.3)      = {tru_band:.6g}")
    print(f"   floor-19.5 recovered dN/dX (point) = {r195['dndx_est0_band']:.6g}  "
          f"(R0 {r195['r0_band_point']:.4f})")
    print(f"   floor-19.0 recovered dN/dX (point) = {r190['dndx_est0_band']:.6g}  "
          f"(R0 {r190['r0_band_point']:.4f})")
    print(f"   delta total recovered (19.0 - 19.5) = "
          f"{r190['dndx_est0_band'] - r195['dndx_est0_band']:+.6g}")
    print("\n   per-bin recovered dN/dX (point), and per-bin redistribution:")
    print(f"{'bin':>14} | {'truth':>11} | {'f19.5 est':>11} | {'f19.0 est':>11} | {'delta(19.0-19.5)':>17}")
    print("-" * 78)
    tot195 = tot190 = tott = 0.0
    for bi, (blo, bhi) in enumerate(PER_BINS):
        e195 = r195["dndx_est0_bin"][bi]; e190 = r190["dndx_est0_bin"][bi]
        tb = tru_bin[bi]
        tot195 += e195; tot190 += e190; tott += tb
        lab = f"[{blo:.1f},{bhi:.1f})"
        print(f"{lab:>14} | {tb:>11.5g} | {e195:>11.5g} | {e190:>11.5g} | {e190 - e195:>+17.5g}")
    print("-" * 78)
    print(f"{'SUM[19.5,20.3)':>14} | {tott:>11.5g} | {tot195:>11.5g} | {tot190:>11.5g} | "
          f"{tot190 - tot195:>+17.5g}")
    print(f"\n   fractional change in TOTAL recovered counts (19.0 vs 19.5) = "
          f"{(tot190 - tot195) / tot195 * 100:+.1f}%")

    print("\n" + "#" * 78)
    print("# RESULT 4 — DLA-tier (>=20.3) R0 MC band (should stay ~1.16, error-bar size)")
    print("#" * 78)
    for r in (r195, r190):
        a = r["mc_r0_203"][np.isfinite(r["mc_r0_203"])]
        print(f"   {r['name']}: point R0(>=20.3) = {r['r0_203_point']:.4f}  "
              f"MC mean={np.nanmean(a):.4f} std={np.nanstd(a):.4f} "
              f"q16={np.nanpercentile(a,16):.4f} q84={np.nanpercentile(a,84):.4f}")

    # persist
    np.savez("/tmp/subdla_floor_mc_band_result.npz",
             tru_band=tru_band, tru_bin=tru_bin,
             f195_mc_band=r195["mc_band"], f190_mc_band=r190["mc_band"],
             f195_mc_bin=r195["mc_bin"], f190_mc_bin=r190["mc_bin"],
             f195_r0_band_point=r195["r0_band_point"],
             f190_r0_band_point=r190["r0_band_point"],
             f195_dndx_est0_bin=r195["dndx_est0_bin"],
             f190_dndx_est0_bin=r190["dndx_est0_bin"])
    print("\n[saved] /tmp/subdla_floor_mc_band_result.npz")

    # ---- committed, git-stamped JSON deliverable (mock recovery ratios + MC bands) ----
    def _bandstats(a):
        """(point-independent) mean/std/q16/q50/q84 of a finite MC sample -> plain floats."""
        mu, sd, q16, q50, q84 = _q(a)
        return dict(mean=float(mu), std=float(sd),
                    q16=float(q16), q50=float(q50), q84=float(q84))

    def _config_block(r):
        tb = r["dndx_tru_band"]; to = r["om_t_band"]; tbin = r["dndx_tru_bin"]
        mc_r0_dndx = r["mc_band"] / tb
        mc_r0_om = r["mc_omega"] / to
        # per-0.1-dex R0 bands (dN/dX)
        per_bin_q16 = []; per_bin_q50 = []; per_bin_q84 = []
        for bi in range(len(PER_BINS)):
            ratio = r["mc_bin"][:, bi] / tbin[bi]
            ratio = ratio[np.isfinite(ratio)]
            per_bin_q16.append(float(np.nanpercentile(ratio, 16)))
            per_bin_q50.append(float(np.nanpercentile(ratio, 50)))
            per_bin_q84.append(float(np.nanpercentile(ratio, 84)))
        a203 = r["mc_r0_203"][np.isfinite(r["mc_r0_203"])]
        return dict(
            fit_floor=(19.5 if r["name"] == "floor19.5" else 19.0),
            n_sl=int(r["n_sl"]),
            dndx=dict(
                r0_band_195_203=dict(point=float(r["r0_band_point"]),
                                     **_bandstats(mc_r0_dndx)),
                r0_per_bin_point=[float(x) for x in r["r0_bin_point"]],
                r0_per_bin_q16=per_bin_q16, r0_per_bin_q50=per_bin_q50,
                r0_per_bin_q84=per_bin_q84,
                est0_band_195_203=float(r["dndx_est0_band"]),
                est0_per_bin=[float(x) for x in r["dndx_est0_bin"]],
            ),
            omega=dict(
                r0_band_195_203=dict(point=float(r["r0_omega_band_point"]),
                                     **_bandstats(mc_r0_om)),
                est0_band_195_203=float(r["om_e0_band"]),
            ),
            dla_tier=dict(
                r0_dndx_203_point=float(r["r0_203_point"]),
                r0_dndx_203_mc=dict(mean=float(np.nanmean(a203)), std=float(np.nanstd(a203)),
                                    q16=float(np.nanpercentile(a203, 16)),
                                    q84=float(np.nanpercentile(a203, 84))),
            ),
        )

    inputs = dict(
        n_mc=int(n_mc), seed=int(seed), fp_frozen=bool(freeze_fp),
        catalog_dir=AB.DEF_CAT, truth=AB.DEF_TRUTH, bal_cat=AB.DEF_BAL,
        loa0_product=LOA0_PRODUCT, lam_rf_min=1025.0, family="bspbody",
        host_truth_floor=19.0, report_limits=",".join(f"{x:g}" for x in REPORT_LIMITS),
        floor19_5=dict(kernel=F195["kernel"], molly=F195["molly"], fit_floor=F195["fit_floor"]),
        floor19_0=dict(kernel=F190["kernel"], molly=F190["molly"], fit_floor=F190["fit_floor"]),
    )
    # FAIL-CLOSED kernel self-declaration. Raises if RESP_KIND/PAPER_FACING are ever
    # edited into the forbidden combination (kappa + paper_facing=True).
    kernel_md = RK.kernel_metadata(
        RESP_KIND, context="sub-DLA floor MC band (kappa diagnostic)",
        paper_facing=PAPER_FACING,
        extra_note=("The floor-19.5-vs-19.0 DIFFERENCE is the intended observable; the "
                    "R0 LEVELS carry the kappa<->forward kernel gap (3.8% dN/dX, 8.5% "
                    "Omega on this band) and are not paper-facing."))
    out_json = dict(
        metadata=dict(
            what="Joint-MC error band on the sub-DLA integrated [19.5,20.3) R0 (dN/dX and "
                 "Omega) and the per-0.1-dex R0, for the floor-19.5 (headline) and floor-19.0 "
                 "(rebuild) configs, on the loa0 FP estimator.",
            mock="2LPT-0 (loa-124); values are MOCK recovery ratios + MC bands, not real-LOA",
            **kernel_md,
            code_commit=_git_commit(),
            wallclock_s=round(time.time() - t_start, 1),
            rederive=("python CDDF_analysis/diagnostics/subdla/subdla_floor_mc_band.py "
                      "--force --n-mc 150 --seed 0"),
            inputs=inputs,
            fp_frozen=bool(freeze_fp),
            band_channels=[
                "completeness_C_beta: Wilson/Jeffreys-Beta resample per molly cell (_draw_beta_cell)",
                "sightline_bootstrap: TID-multiplicity multinomial -> per-op-row weight",
                ("loa0_FP_normalization: Gehrels Gamma(n_FP+1/2, 1/ell_eff) resample of the "
                 "loa-0 forest-FP counts (Loa0FP.resample) on an independent rng_fp stream -- "
                 "the DOMINANT channel (~10.6% Poisson on ~89 loa-0 detections)")
                if not freeze_fp else
                "loa0_FP FROZEN at the point value (--freeze-fp regression guard; NOT resampled)",
                "map_multistart_jitter: 2 warm-started L-BFGS-B restarts (numerical)",
            ],
            band_channels_excluded=[
                ("NHI_ERR (sigma_i) Eddington edge scatter: A/M rebuilt from the CACHED kernel "
                 "per draw are NOT re-sliced on perturbed NHI, so it feeds nothing -> ZERO "
                 "variance. Wiring it needs a per-draw kernel re-slice + sparse-A rebuild "
                 "(outside the reduce-only budget). The sigma_i draw is retained ONLY for "
                 "RNG stream byte-alignment with the frozen baseline."),
                ("rho purity-mixture term: unused by the loa0 FP estimator. The rho Beta draw "
                 "is retained ONLY for RNG stream byte-alignment."),
            ] + (["loa0_FP Gehrels Gamma normalization: FROZEN by --freeze-fp (default OFF resamples it)"]
                 if freeze_fp else []),
            note="Reduce-only (cached kernel, no inference/SLURM/tilt). MC band = completeness "
                 "C (Beta) + sightline bootstrap + loa0 FP Gehrels-Gamma resample (the DOMINANT "
                 "channel; --freeze-fp holds it fixed = the prior defect, which collapsed the "
                 "band ~7.5x and inverted the floor conclusion). rho/NHI_ERR are drawn only to "
                 "keep the RNG stream aligned and contribute ZERO variance (see band_channels_"
                 "excluded). Each draw re-MAPs theta warm-started at the point MAP and reduces. "
                 "R0 = est/truth. Per-bin R0 is dN/dX. The floor-19.0 vs floor-19.5 difference "
                 "is the count-conservation / edge-migration diagnostic (see the 'difference' "
                 "block); its integrated-band z-scores are printed to stdout.",
        ),
        result=dict(
            per_bins=[[blo, bhi] for blo, bhi in PER_BINS],
            truth=dict(
                dndx_band_195_203=float(tru_band),
                omega_band_195_203=float(r195["om_t_band"]),
                dndx_per_bin=[float(x) for x in tru_bin],
            ),
            floor19_5=_config_block(r195),
            floor19_0=_config_block(r190),
            difference_dndx_195_203=dict(
                point_diff=float(dpoint), sigma_195=float(s195), sigma_190=float(s190),
                sigma_quadrature=float(s_quad), z_quadrature=float(dpoint / s_quad),
                paired_mean=float(dpm), paired_std=float(dps), z_paired=float(dpoint / dps),
            ),
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
    return out


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
    ap.add_argument("--freeze-fp", action="store_true", dest="freeze_fp",
                    help="FREEZE the loa0 FP at its point value (the prior defect / "
                         "regression guard). Default OFF = resample it by its Gehrels "
                         "Gamma (the correct, dominant variance channel).")
    main(ap.parse_args())
