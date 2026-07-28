#!/usr/bin/env python
"""run_subdla_headline_full.py — produce the FULL sub-DLA headline JSON in the SAME
schema as track_c_tf_loa.json / the DLA loa0 headline (measurement[19.5..20.3] per-z +
integrated with an MC band + perz_fN differential f(N|z) arrays + zbins + metadata), for
the paper figures + provenance.  This is the SUB-DLA analogue of
``bal_metal_fp/arbiter/run_loa0_headline_full.py``.

CONFIG-ONLY OVERRIDE (no estimator edit, no re-inference).  It reuses
track_c_tf_loa.py's own build_frozen_calibration / build_loa_ingredients /
run_measurement on the IDENTICAL frozen ingredients as the DLA loa0 headline (same
FROZEN 2LPT-0 forward-response kernel, z-resolved completeness g(N,z), lya_only-nhi195
molly C/rho, lam_rf_min=1025, loa0 forest-FP product, real dlacat, cut bundle), and the
ONLY things that differ from the DLA loa0 headline are:

  * report_limits  "20.0,20.3"  ->  "19.5,19.6,...,20.3"   (the sub-DLA band, 0.1-dex)

Everything else is byte-identical to the committed DLA loa0 headline path:
  * fit_floor        = 19.5   (ALREADY 19.5 in the DLA headline — the fine grid already
                               spans the sub-DLA band; only the *reported* limits change)
  * molly            = the lya_only-nhi195 matrix (AB.DEF_LYAONLY_MOLLY, resolved via
                       track_c_tf_loa.build_frozen_calibration -> AB._resolve_molly)
  * lam_rf_min       = 1025.0 (lyaonly1025)
  * fp_estimator     = loa0   (headline), loa0 product = AB.DEF_LOA0_PRODUCT (lyaonly1025)
  * forward kernel   = forward_response_2lpt0.npz (FROZEN)
  * zbins            = 2.0,2.5,3.0,3.5,4.0,4.25 ; v2_z_fit_hi = 3.5  (== DLA job 52266001)

gpy_dla_detection/ and cddf_catalog_hbi.py are UNTOUCHED.

  🔴 REAL-LOA is GATED.  Building + mock-validating this routine is what this file does.
     The real-LOA run is gated on PI approval and requires an explicit --real-loa flag; the
     integrator runs it.  Real-LOA numbers go to SCRATCH only (never the code repo), and the
     routine refuses to write anywhere under the repo.  No real-LOA result value is printed
     or committed.

MOCK (2LPT-0) values are PUBLIC — the default run is the 2LPT-0 self-application, which
produces a public full-schema JSON (schema + determinism + z/N masks demonstration) and
validates the config against the committed CDDF_analysis/hbi/subdla_mock_validation.json.

Env: conda gpdla; OMP/OPENBLAS/MKL_NUM_THREADS=1; HDF5_USE_FILE_LOCKING=FALSE.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

# repo root = 4 dirnames up (subdla -> diagnostics -> CDDF_analysis -> <repo>).
# (5 dirnames overshoots to /home/mfho and silently stamps code_commit="unknown" — the
#  exact bug the DLA headline routine carried; see _assert_repo_root / task #1.)
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import track_c_tf_loa as TF          # noqa: E402
from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB      # noqa: E402
from CDDF_analysis.hbi.cddf_catalog_hbi import (             # noqa: E402
    make_fp_model, make_rho_interpolator)
from CDDF_analysis.unblind import resp_kind as RK            # noqa: E402
from CDDF_analysis.unblind import estimand as _EST           # noqa: E402

# The committed, git-stamped MOCK deliverables this routine's config is gated against.
# KAPPA twin = the RETIRED posterior-kernel artifact (kept for config-drift detection only;
# its numbers are NOT the headline). FORWARD = the headline anchor (0.849/0.822 band R0).
COMMITTED_MOCK_JSON = os.path.join(_REPO, "CDDF_analysis", "hbi", "subdla_mock_validation.json")
COMMITTED_FORWARD_JSON = os.path.join(
    _REPO, "CDDF_analysis", "hbi", "subdla_mock_validation_forward.json")

# The sub-DLA report band: the 19.5 floor + 0.1-dex steps through 20.3.
SUBDLA_REPORT_LIMITS = "19.5,19.6,19.7,19.8,19.9,20.0,20.1,20.2,20.3"
# The [19.5,19.7) differential f(N) edge is formally non-identifiable on a 19.5-floored
# catalog (Track A closed): the two lowest 0.1-dex bins straddle the fit floor and cannot
# be separated from edge migration.  This is a per-logN mask, INDEPENDENT of the z masks.
NONIDENT_EDGE = 19.7
NONIDENT_REASON = (
    "logN_nonidentifiable = (logN center < 19.7): the sub-DLA DIFFERENTIAL f(N) is "
    "identifiable only over [19.7,20.3). Centers < 19.7 are non-identifiable — either "
    "BELOW the 19.5 fit floor (parametric extrapolation, not a measurement) or in the "
    "[19.5,19.7) EDGE-MIGRATION zone (the two lowest 0.1-dex bins straddle the floor and "
    "cannot be separated from edge migration on a 19.5-floored catalog; Track A closed). "
    "The deliverable is the [19.7,20.3) f(N) diff + the INTEGRATED [19.5,20.3] band with "
    "an edge-migration systematic; do NOT present any center < 19.7 as a measured "
    "differential bin. This N mask is INDEPENDENT of the z masks.")

# The DLA loa0 headline scratch output tree (NEVER the repo). Real-LOA + mock go under here.
_SCRATCH_ROOT = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_loa/subdla"
DEFAULT_MOCK_OUT = os.path.join(_SCRATCH_ROOT, "subdla_headline_mock.json")
DEFAULT_REAL_OUT = os.path.join(_SCRATCH_ROOT, "subdla_headline_loa0.json")

# provenance dep set: the files WITHOUT WHICH the stamped `rederive` command cannot run
# and that carry the science/config logic (mirrors break_census.py, which also stamps its
# imported survival.py). cddf_catalog_hbi.py is FROZEN so it is deliberately not listed.
_PROVENANCE_DEPS = [
    "CDDF_analysis/diagnostics/subdla/run_subdla_headline_full.py",
    "CDDF_analysis/hbi/track_c_tf_loa.py",              # the measurement engine
    "CDDF_analysis/diagnostics/subdla/subdla_loa0_validation.py",  # the gate twin
    "CDDF_analysis/hbi/ab_loa0_fp_baseline.py",         # shared ingredient builder
    "CDDF_analysis/unblind/resp_kind.py",               # the fail-closed kernel gate
]


def _git_commit(deps=None):
    """Repo HEAD, suffixed ``-dirty`` iff any dep ROUTINE is untracked or modified.

    New (break_census.py) semantics: a ``-dirty`` stamp means the artifact is NOT
    third-party re-derivable — the named commit does not contain (this version of) the
    routine(s) that produced it.  Commit them first, then re-run.  Checking the ROUTINE
    files specifically — rather than ``git status``, which is dirtied by the artifact's
    own untracked scratch output — is what makes the marker meaningful.

    Returns "unknown" ONLY on a hard git failure (git missing / not a checkout); callers
    MUST fail loudly rather than ship an "unknown" stamp (task #1).
    """
    deps = list(deps if deps is not None else _PROVENANCE_DEPS)
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
        print(f"  [WARN] _git_commit() failed ({type(e).__name__}: {e}); would stamp "
              f"'unknown' (cwd={_REPO}).", file=sys.stderr)
        return "unknown"


def _assert_provenance():
    """Task #1: refuse to run unless (a) the resolved repo root actually contains a .git
    directory (the DLA headline resolved to /home/mfho and silently stamped 'unknown'),
    and (b) _git_commit() would not silently return 'unknown'.  A '<sha>-dirty' stamp is
    ALLOWED (it just means this new/edited routine is not yet committed)."""
    git_dir = os.path.join(_REPO, ".git")
    if not os.path.isdir(git_dir):
        raise SystemExit(
            f"[PROVENANCE] resolved repo root has no .git directory: {_REPO}\n"
            f"  The repo-root dirname count is wrong (the DLA headline bug: one dirname too "
            f"many resolved to /home/mfho and stamped code_commit='unknown'). Refusing to run.")
    commit = _git_commit()
    if commit == "unknown":
        raise SystemExit(
            f"[PROVENANCE] _git_commit() returned 'unknown' (git failure at cwd={_REPO}). "
            f"Refusing to stamp an un-provenanced artifact (task #1).")
    return commit


# --- reuse the committed loa0 provenance guard (imported from the arbiter helper) --------
def _load_preflight():
    """Load preflight_loa0_product from the committed DLA-headline arbiter helper so the
    loa0 product/molly provenance guard is SHARED (not a drifting copy)."""
    p = os.path.join(_REPO, "CDDF_analysis", "diagnostics", "bal_metal_fp", "arbiter",
                     "apply_broadtrough_veto_headline.py")
    spec = importlib.util.spec_from_file_location("bt_helper_for_subdla", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.preflight_loa0_product


# ---------------------------------------------------------------------------
# argument namespace (mirror track_c_tf_loa.main() defaults; sub-DLA report band)
# ---------------------------------------------------------------------------
def build_args(a, dataset):
    """Build the track_c_tf_loa arg namespace for the sub-DLA headline.  `dataset` in
    {'mock','real-loa'} only swaps the CATALOG paths (2LPT-0 self-application vs the real
    LOA catalog); every recipe knob is identical to the DLA loa0 headline."""
    import types
    if dataset == "mock":
        loa_cat, loa_truth, loa_bal = TF._C0_CAT, TF._C0_TRUTH, TF._C0_BAL
        loa_mockdir = os.path.dirname(TF._C0_TRUTH)
    elif dataset == "real-loa":
        loa_cat, loa_truth, loa_bal, loa_mockdir = (
            TF._LOA_CAT, TF._LOA_TRUTH, TF._LOA_BAL, TF._LOA_MOCKDIR)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    args = types.SimpleNamespace(
        # frozen 2LPT-0 calibration inputs (IDENTICAL to the DLA headline)
        catalog_dir=TF._C0_CAT, truth=TF._C0_TRUTH, bal_cat=TF._C0_BAL,
        molly_tsv=None,                                  # -> AB._resolve_molly (lya_only-195)
        kernel=AB.DEF_KERNEL, forward_model=TF._DEF_FORWARD,
        resp_family="empirical", resp_kind="forward", loa_kernel=None,
        loa_processed_glob=("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
                            "loa_main_dark_v1/processed/processed-main-dark-*.h5"),
        loa_pw_samples=("/scratch/cavestru_root/cavestru0/mfho/DESI/"
                        "desi_gpy_dla_detection/data/dr12q/processed/"
                        "pw_samples_a3_172_225_50000.mat"),
        # the DATA catalog (mock self-app or real LOA)
        loa_cat=loa_cat, loa_truth=loa_truth, loa_bal=loa_bal, loa_mockdir=loa_mockdir,
        out=os.path.dirname(a.out) or _SCRATCH_ROOT,
        report_out=os.path.join(os.path.dirname(a.out) or _SCRATCH_ROOT, "_subdla_report.md"),
        # ---- recipe knobs (== DLA loa0 headline / job 52266001) ----
        zbins=a.zbins, v2_z_fit_hi=a.v2_z_fit_hi,
        report_limits=SUBDLA_REPORT_LIMITS,             # <-- the ONE sub-DLA change
        family="bspbody", fit_floor=19.5, fit_ceil=99.0, lambda_bspbody=30.0,
        lam_rf_min=1025.0, edge_slope_lam=40.0, gl_nodes=1, host_truth_floor=19.0,
        n_mc=a.n_mc, workers=a.workers, seed=a.seed, cz_min_count=30.0,
        # RETIRED for paper-facing output (PI, 2026-07-28): the sub-DLA headline band
        # must NOT be an MC cloud slid onto the plug-in MAP. Measured displacement on
        # this very path (subdla_mock_headline.json, 2LPT-0, n_mc=2000, seed 0):
        # raw MC median is +70.00% (dN/dX) / +55.20% (Omega) ABOVE the MAP, i.e.
        # +16.67 / +19.02 68%-band half-widths. The POINT is untouched by this flag.
        band_recenter=False, allow_diagnostic_recenter=False,
        omega_slope_extrap=True,
        omega_slope_extrap_integrated=True, slope_edge=21.2, slope_fit_dex=0.6,
        sigma_slope=0.5,
    )
    args._limits = tuple(float(x) for x in args.report_limits.split(","))
    return args


# ---------------------------------------------------------------------------
# the sub-DLA measurement: config-only loa0 (or purity_mixture) override
# ---------------------------------------------------------------------------
def run_headline(args, fp_estimator, loa0_product):
    """Config-only FP override on the frozen sub-DLA ingredients.  Returns the
    track_c_tf_loa run_measurement `res` dict (the SAME object the DLA headline dumps)."""
    # FAIL-CLOSED, BEFORE any compute: this is a paper-facing measurement, so refuse the
    # GP-posterior kernel here rather than after an hours-long run (the stamp guard in
    # assemble_out_json is the backstop, not the first line of defence).
    RK.resolve_resp_kind(getattr(args, "resp_kind", None),
                         context=f"sub-DLA headline run (fp={fp_estimator})",
                         paper_facing=True)
    frozen = TF.build_frozen_calibration(args)
    args.molly_tsv = frozen["molly_tsv"]

    ing = TF.build_loa_ingredients(args, frozen)
    cfg = ing["cfg"]
    if fp_estimator == "loa0":
        cfg.fp_estimator = "loa0"
        cfg.loa0_product_path = loa0_product
        # provenance guards (task #6/#7 of the DLA headline): the loa0 product must be the
        # lya-only (1025) headline product AND the resolved molly the lya_only-nhi195 matrix.
        _load_preflight()(loa0_product, cfg, args.molly_tsv)
        rho = make_rho_interpolator(ing["mm"])
        loa0_model, _ = make_fp_model(cfg, ing["cat_cut"], ing["op_mask"], rho)
        ing["fp_model"] = loa0_model
        assert getattr(cfg, "_loa0_fp", None) is not None, "loa0 FP not attached"
        # the FP volume-scale must use the REAL op sightline count (cfg.n_sl_prod set by
        # build_loa_ingredients), not the product's stored mock n_sl_prod fallback.
        assert loa0_model.n_sl_prod == ing["n_sl"], (
            f"n_sl_prod mismatch {loa0_model.n_sl_prod} != {ing['n_sl']} — loa0 FP volume-"
            f"scale did not pick up the op sightline count.")
        print(f"  [FP=loa0] n_sl_loa0={loa0_model.n_sl_loa0:.0f} "
              f"n_sl_prod={loa0_model.n_sl_prod:.0f} vol_scale={loa0_model.vol_scale:.3f}")
    elif fp_estimator == "purity_mixture":
        pass                                             # build_loa_ingredients default
    else:
        raise ValueError(f"unknown fp_estimator {fp_estimator!r}")

    # Capture the estimator's OWN joint_mc_errors samples (in the loa0 path the forest FP is
    # RESAMPLED per draw inside run_measurement via Loa0FP.resample — the Gehrels Gamma rate
    # draw) so we can form the WINDOW [19.5,20.3) band = per-draw difference of the
    # cumulative(19.5) - cumulative(20.3) draws.  run_measurement returns only per-limit
    # cumulative bands; the sub-DLA deliverable is the WINDOWED band, a CORRELATED difference
    # that cannot be reconstructed from the two marginal bands.  We CAPTURE (do not re-run) so
    # the window band uses the IDENTICAL draws the per-limit bands come from.
    _cap = {}
    _orig_jme = TF.joint_mc_errors
    def _capture_jme(*a, **k):
        mc = _orig_jme(*a, **k)
        _cap["mc"] = mc
        return mc
    TF.joint_mc_errors = _capture_jme
    try:
        res = TF.run_measurement(args, ing, args._limits, args.seed, frozen=frozen)
    finally:
        TF.joint_mc_errors = _orig_jme
    res["_frozen"] = frozen
    res["window_band_195_203"] = _window_band(args, ing, res, _cap.get("mc"))
    return res


def _window_band(args, ing, res, mc):
    """Authoritative sub-DLA WINDOW [19.5,20.3) band (dN/dX + Omega) from the estimator's own
    joint_mc_errors samples (loa0 FP RESAMPLED per draw).  The window = cumulative(19.5) -
    cumulative(20.3); its band is the per-draw difference of the SAME draws (correlated),
    exactly the recipe run_measurement uses for the per-limit bands.

    ESTIMAND (PI, 2026-07-28): the POINT is a plug-in MAP and the BAND is the MC/bootstrap
    ensemble around a DIFFERENT centre -> class PLUGIN_MAP_MC, NOT a posterior credible
    interval.  ``raw_median`` and ``jensen_shift`` below record the gap explicitly: on
    2LPT-0 (n_mc=2000, seed 0) the raw MC median sits +70.00% (dN/dX) and +55.20% (Omega)
    ABOVE the MAP, i.e. +16.67 and +19.02 68%-band half-widths.  Recentering the cloud onto
    the point (cfg.band_recenter) is RETIRED to diagnostic-only; it is now routed through
    resolve_band_recenter and defaults OFF.

    A cross-check asserts that re-forming each per-limit CUMULATIVE band from the captured
    draws reproduces run_measurement's stored per-limit band to < 1e-9 — proving the window
    band is built from the identical draws.  MOCK 2LPT-0 recovery values (public).
    """
    if mc is None or "_samples" not in mc:
        return None
    cfg = ing["cfg"]
    # PI 2026-07-28 choke point: raises if band_recenter=True without the explicit
    # allow_diagnostic_recenter opt-in; False on every headline invocation.
    _recenter = TF.resolve_band_recenter(cfg, where="run_subdla_headline_full._window_band")
    limits = args._limits
    lo, hi = limits[0], limits[-1]          # 19.5, 20.3
    samp = mc["_samples"]
    # committed 2LPT-0 truth (estimator-independent: same mock/grid/SNR cut/pathlength) so
    # R0 = est/truth is reported without recomputing truth in this truth-free routine.
    truth = {"dndx": None, "omega": None}
    try:
        with open(COMMITTED_MOCK_JSON) as fh:
            ci = json.load(fh)["integrated"]["loa0"]
        truth = {"dndx": float(ci["dndx_tru_195_203"]),
                 "omega": float(ci["omega_tru_195_203"])}
    except Exception:  # noqa: BLE001
        pass
    out = {"window": [float(lo), float(hi)],
           "note": ("integrated [19.5,20.3) = cumulative(19.5) - cumulative(20.3); band = the "
                    "per-draw difference of the SAME joint_mc_errors draws (loa0 forest FP "
                    "RESAMPLED per draw via Loa0FP.resample Gehrels Gamma).  R0 vs committed "
                    "estimator-independent 2LPT-0 truth.  MOCK values — public."),
           "estimand": _EST.band_estimand(band_recenter=_recenter, posterior_sampled=False),
           "raw_median_note": ("raw_median = the UN-recentered MC ensemble median; "
                               "jensen_shift = MAP - raw_median. A large |jensen_shift| "
                               "relative to (q84-q16)/2 means the point and the band are "
                               "DIFFERENT ESTIMANDS.")}
    for kind in ("dndx", "omega"):
        key = "dndx_total" if kind == "dndx" else "omega"
        raw_lo = np.asarray(samp[key][lo], float)
        raw_hi = np.asarray(samp[key][hi], float)
        map_lo = float(res[kind][lo]["integrated"]["MAP"])
        map_hi = float(res[kind][hi]["integrated"]["MAP"])
        # cross-check: reproduce each per-limit cumulative band exactly (identical draws).
        for L, mp in ((lo, map_lo), (hi, map_hi)):
            chk = TF.PZ._band(np.asarray(samp[key][L], float), point=mp,
                           recenter=_recenter)
            ref = res[kind][L]["integrated"]
            for q in ("q16", "q84", "q025", "q975", "std"):
                d = abs(float(chk[q]) - float(ref[q]))
                assert d < 1e-9, (f"window-band cross-check FAILED: {kind} cum>={L} {q} "
                                  f"|Δ|={d:.2e} (reproduced band != run_measurement band).")
        map_win = map_lo - map_hi
        raw_win = raw_lo - raw_hi
        b = TF.PZ._band(raw_win, point=map_win, recenter=_recenter)
        finite = raw_win[np.isfinite(raw_win)]
        raw_med = float(np.median(finite)) if finite.size else float("nan")
        rec = dict(MAP=map_win, q16=b["q16"], q50=b["q50"], q84=b["q84"],
                   q025=b["q025"], q975=b["q975"], std=b["std"], n_finite=int(finite.size),
                   raw_median=raw_med, jensen_shift=(map_win - raw_med))
        t = truth[kind]
        if t is not None and t > 0:
            rec["truth_committed"] = t
            for q in ("MAP", "q16", "q50", "q84", "q025", "q975", "std"):
                rec[f"R0_{q}"] = rec[q] / t
        out[kind] = rec
    return out


# ---------------------------------------------------------------------------
# z / N masks + JSON assembly (SAME schema as the DLA headline, plus the masks)
# ---------------------------------------------------------------------------
def _z_masks(zbins, v2_z_fit_hi, max_truth_z, z_extrapolated):
    """Per coarse-z-bin, the THREE booleans the PI unblinding requires (config-only,
    no re-derivation for the consumer):
      * beyond_calibration    := z_extrapolated (frozen g(N,z) has NO truth support here)
      * beyond_v2_fit         := zbin_lo >= v2_z_fit_hi (above the mean-flux/tau_eff ceiling)
      * partial_truth_support := zbin_lo < max_truth_z < zbin_hi (truth support ends mid-bin)
    These do NOT coincide: the [3.5,4.0] bin is beyond_v2_fit and partial_truth_support but
    NOT beyond_calibration; the [4.0,4.25] bin is all three.  The DLA headline JSON recorded
    only z_extrapolated, so a plotter reading it alone mis-presents [3.5,4.0] as validated."""
    zb = np.asarray(zbins, float)
    n_zc = len(zb) - 1
    beyond_calib = np.asarray(z_extrapolated, bool)
    beyond_v2 = np.array([zb[k] >= v2_z_fit_hi - 1e-9 for k in range(n_zc)], bool)
    partial = np.array(
        [(zb[k] < max_truth_z < zb[k + 1]) if np.isfinite(max_truth_z) else False
         for k in range(n_zc)], bool)
    return beyond_calib, beyond_v2, partial


def assemble_out_json(res, args, limits, wall, fp_estimator, loa0_product, dataset, code_commit):
    """Emit the SAME top-level structure as track_c_tf_loa.json / the DLA headline
    (['measurement','metadata','perz_fN','zbins']), plus the z/N masks. Any field the
    sub-DLA band cannot populate is emitted EXPLICITLY as null with a documented reason."""
    zbins = np.asarray(res["zbins"], float)
    n_zc = res["n_zc"]
    v2_hi = float(args.v2_z_fit_hi)
    max_tz = float(res.get("max_truth_z", float("nan")))
    z_extrap = [bool(x) for x in np.asarray(res.get("z_extrapolated", []))]
    z_thin = [bool(x) for x in np.asarray(res.get("z_thin", []))]
    beyond_calib, beyond_v2, partial = _z_masks(zbins, v2_hi, max_tz, z_extrap)

    # per-logN non-identifiability mask (INDEPENDENT of the z masks; distinct name).
    mid = np.asarray(res["mid"], float)
    logN_nonident = [bool(m < NONIDENT_EDGE - 1e-9) for m in mid]

    # FAIL-CLOSED KERNEL STAMP (2026-07-28): this routine emits a PAPER-FACING headline,
    # so the kernel is resolved with NO DEFAULT and 'kappa' is refused outright -- the
    # artifact can never be written on the GP-posterior kernel, and it SELF-DECLARES
    # metadata['resp_kind'] / ['paper_facing'] / ['kernel_note'].  Previously this line
    # was `resp_kind=getattr(args, "resp_kind", "forward")`, i.e. a stamp that would
    # happily record whatever it was handed (including 'kappa') with nothing refusing it.
    _kernel_md = RK.kernel_metadata(
        getattr(args, "resp_kind", None),
        context=f"sub-DLA {dataset} headline stamp (fp={fp_estimator})",
        paper_facing=True)

    out = dict(
        metadata=dict(
            what="sub-DLA-band catalog-HBI headline measurement (dN/dX, Omega, CDDF f(N|z)) "
                 "over [19.5,20.3), config-only loa0 override of the DLA headline recipe.",
            dataset=dataset,
            fp_estimator=fp_estimator,
            loa0_product=(loa0_product if fp_estimator == "loa0" else None),
            provenance=("config-only FP+report-limits override of the DLA loa0 headline "
                        "(track_c_tf_loa build_frozen_calibration/build_loa_ingredients/"
                        "run_measurement); frozen ingredients identical, only report_limits "
                        "-> the sub-DLA band (+ fp_estimator=loa0)."),
            rederive=("python CDDF_analysis/diagnostics/subdla/run_subdla_headline_full.py"
                      + (" --real-loa" if dataset == "real-loa" else "")
                      + f" --fp-estimator {fp_estimator} --n-mc {args.n_mc} "
                        f"--seed {args.seed} --force"),
            n_mc=int(args.n_mc), seed=int(args.seed),
            limits=list(limits), report_limits=SUBDLA_REPORT_LIMITS,
            **_kernel_md,                                  # resp_kind/paper_facing/kernel_note
            loa_kernel=res.get("_kernel_built_path"),       # null on the forward path
            forward_model=args.forward_model, molly_tsv=args.molly_tsv,
            loa_cat=(args.loa_cat if dataset == "mock" else "<REAL-LOA — path withheld>"),
            n_op_detections=res["n_op_detections"], n_op_sl=res["n_op_sl"],
            consistency_err=res["consistency_err"], v2_z_fit_hi=v2_hi,
            max_truth_z=max_tz, support_limit=float(res.get("support_limit", max(limits))),
            truth_counts_perz=res.get("truth_counts_perz"),
            # z-bin masks (task from PI): all three, explicitly, in metadata AND per-bin.
            z_extrapolated=z_extrap, z_thin=z_thin,
            beyond_calibration=[bool(x) for x in beyond_calib],
            beyond_v2_fit=[bool(x) for x in beyond_v2],
            partial_truth_support=[bool(x) for x in partial],
            # N-bin (differential f(N)) non-identifiability mask — INDEPENDENT of z.
            logN_nonidentifiable_edge=float(NONIDENT_EDGE),
            logN_nonidentifiable_reason=NONIDENT_REASON,
            wallclock_s=float(wall), code_commit=code_commit,
            privacy=("MOCK (2LPT-0) values — public." if dataset == "mock"
                     else "REAL-LOA values — PRIVATE (scratch/notes only; never the repo)."),
        ),
        measurement={
            str(l): dict(
                dndx=dict(
                    perz=[res["dndx"][l]["perz"][k] for k in range(n_zc)],
                    integrated=res["dndx"][l]["integrated"]),
                omega=dict(
                    perz=[res["omega"][l]["perz"][k] for k in range(n_zc)],
                    integrated=res["omega"][l]["integrated"]),
            ) for l in limits},
        zbins=list(map(float, zbins)),
    )

    # perz_fN: reuse the FROZEN production assembler, then AUGMENT with the z/N masks
    # (post-process only — track_c_tf_loa.py is NOT edited).
    perz_fN = TF.assemble_perz_fN(res, limits)
    perz_fN["v2_z_fit_hi"] = v2_hi
    perz_fN["max_truth_z"] = max_tz
    perz_fN["beyond_calibration"] = [bool(x) for x in beyond_calib]
    perz_fN["beyond_v2_fit"] = [bool(x) for x in beyond_v2]
    perz_fN["partial_truth_support"] = [bool(x) for x in partial]
    perz_fN["logN_nonidentifiable"] = logN_nonident       # per logN center (distinct name)
    perz_fN["logN_nonidentifiable_edge"] = float(NONIDENT_EDGE)
    perz_fN["logN_nonidentifiable_reason"] = NONIDENT_REASON
    for k, rec in enumerate(perz_fN["perz"]):
        rec["beyond_calibration"] = bool(beyond_calib[k])
        rec["beyond_v2_fit"] = bool(beyond_v2[k])
        rec["partial_truth_support"] = bool(partial[k])
    out["perz_fN"] = perz_fN
    # the authoritative sub-DLA WINDOW [19.5,20.3) band (dN/dX + Omega), FP-resampled, the
    # actual sub-DLA deliverable (run_measurement emits only per-limit CUMULATIVE bands).
    out["window_band_195_203"] = res.get("window_band_195_203")
    # ESTIMAND SELF-DECLARATION (PI, 2026-07-28). The point is a plug-in MAP and the band is
    # an MC/bootstrap ensemble around a DIFFERENT centre -> PLUGIN_MAP_MC, so the artifact is
    # NOT paper-facing as an uncertainty until the faithful joint-posterior route lands.
    # ``paper_facing`` is ANDed with the resp_kind/kernel gate above: a gate may only veto.
    _EST.stamp_band_estimand(
        out["metadata"],
        band_recenter=bool(getattr(args, "band_recenter", False)),
        posterior_sampled=False)
    return out


# ---------------------------------------------------------------------------
# config-only gate: reproduce the committed mock validation R0 (baseline_recovery twin)
# ---------------------------------------------------------------------------
def _load_validation_twin():
    p = os.path.join(_HERE, "subdla_loa0_validation.py")
    spec = importlib.util.spec_from_file_location("subdla_loa0_validation_twin", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_against_committed(modes, tol=1e-6, *, resp_kind, committed_json=None):
    """Reproduce a committed sub-DLA mock artifact's integrated R0 for the given FP
    modes by re-running the committed baseline-recovery twin (subdla_loa0_validation.
    run_mode with the requested kernel). Asserts |Delta| <= tol per checked number.

    ``resp_kind`` is a REQUIRED KEYWORD with NO DEFAULT (2026-07-28).  It previously
    defaulted to 'kappa', which meant the validator's own default silently targeted the
    RETIRED posterior artifact -- a caller writing `validate_against_committed(modes)`
    got a PASS against a superseded number and had no way to notice.  There is no
    defensible default here: the two kernels answer different questions.

    resp_kind='forward' → vs subdla_mock_validation_forward.json (the HEADLINE anchor,
                          0.849/0.822; this is the gate that matters scientifically).
    resp_kind='kappa'   → vs subdla_mock_validation.json (the RETIRED posterior-kernel
                          artifact; a config-drift tripwire ONLY — proves the config-only
                          override changed nothing but the FP model. NOT the headline,
                          and never admissible as a paper-facing gate).
    Returns a table."""
    resp_kind = RK.resolve_resp_kind(
        resp_kind, context="sub-DLA committed-artifact gate", paper_facing=False)
    if committed_json is None:
        committed_json = (COMMITTED_FORWARD_JSON if resp_kind == "forward"
                          else COMMITTED_MOCK_JSON)
    if not os.path.exists(committed_json):
        raise SystemExit(f"committed mock artifact not found: {committed_json}")
    with open(committed_json) as fh:
        committed = json.load(fh)["integrated"]
    twin = _load_validation_twin()
    keys = ("r0_dndx_195_203", "r0_omega_195_203",
            "r0_dndx_195_200", "r0_omega_195_200")
    rows = []
    name = os.path.basename(committed_json)
    for mode in modes:
        print("=" * 78)
        print(f"[gate] re-running committed sub-DLA baseline-recovery twin: "
              f"fp={mode} kernel={resp_kind}")
        print("=" * 78)
        got = twin.run_mode(mode, resp_kind=resp_kind)
        ref = committed[mode]
        for k in keys:
            g = float(got[k]); r = float(ref[k]); d = abs(g - r)
            rows.append((mode, resp_kind, k, r, g, d))
            print(f"  {mode:>15} {k:>18}: committed={r:.6f}  reproduced={g:.6f}  |D|={d:.2e}")
            assert d <= tol, (
                f"gate FAILED: {mode}.{k} |Delta|={d:.2e} > {tol:.0e} vs committed "
                f"{name} (kernel={resp_kind}) — the sub-DLA config drifted from the "
                f"validated one (report_limits/floor/molly/loa0/lam_rf_min/kernel).")
    print(f"[gate] PASS — reproduced {name} R0 for {modes} (kernel={resp_kind}) "
          f"to <= {tol:.0e}.")
    return rows


# ---------------------------------------------------------------------------
def _refuse_repo_write(out_path):
    ap = os.path.abspath(out_path)
    if ap == os.path.abspath(_REPO) or ap.startswith(os.path.abspath(_REPO) + os.sep):
        raise SystemExit(
            f"[SAFETY] refusing to write the measurement JSON under the code repo:\n  {ap}\n"
            f"  Measurement JSONs (mock AND real-LOA) go to SCRATCH only. Pass --out under "
            f"/scratch (real-LOA outputs are PRIVATE).")


def _print_config(a, dataset, fp_estimator, loa0_product, code_commit):
    args = build_args(a, dataset)
    print("=" * 78)
    print(f"[dry-run] resolved sub-DLA headline config (dataset={dataset}, fp={fp_estimator})")
    print("=" * 78)
    for k in ("loa_cat", "loa_truth", "loa_bal", "loa_mockdir", "forward_model",
              "report_limits", "zbins", "v2_z_fit_hi", "fit_floor", "lam_rf_min",
              "family", "n_mc", "seed"):
        print(f"    {k:>16} = {getattr(args, k)}")
    print(f"    {'loa0_product':>16} = {loa0_product}")
    print(f"    {'out':>16} = {a.out}")
    print(f"    {'code_commit':>16} = {code_commit}")
    print(f"    {'committed_mock':>16} = {COMMITTED_MOCK_JSON}")
    print(f"    {'nonident_edge':>16} = {NONIDENT_EDGE} (logN differential [19.5,{NONIDENT_EDGE}))")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-loa", action="store_true",
                    help="RUN ON REAL DESI LOA (gated on PI approval; PRIVATE outputs). "
                         "Default OFF = the public 2LPT-0 self-application.")
    ap.add_argument("--fp-estimator", choices=["loa0", "purity_mixture"], default="loa0",
                    help="DLA/sub-DLA-tier FP estimator (headline = loa0).")
    ap.add_argument("--loa0-product", default=AB.DEF_LOA0_PRODUCT,
                    help="loa0 forest-FP product (lyaonly1025; == the DLA headline product).")
    ap.add_argument("--zbins", default="2.0,2.5,3.0,3.5,4.0,4.25",
                    help="coarse z report bins (default the extended grid exposing z>4).")
    ap.add_argument("--v2-z-fit-hi", dest="v2_z_fit_hi", type=float, default=3.5,
                    help="mean-flux/tau_eff fit ceiling (== DLA job 52266001; a DIFFERENT "
                         "quantity from the completeness truth support).")
    ap.add_argument("--n-mc", type=int, default=120, help="MC band draws (MAP is n_mc-indep).")
    ap.add_argument("--seed", type=int, default=0, help="MC seed (determinism).")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="output JSON (SCRATCH only; repo writes refused).")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists.")
    ap.add_argument("--gate", action="store_true",
                    help="ALSO reproduce subdla_mock_validation.json for BOTH FP modes "
                         "(single-knob proof). Implies the loa0 validation.")
    ap.add_argument("--no-validate", action="store_true",
                    help="(mock only) skip the R0 validation against the committed artifact.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved config and exit (no data touched).")
    a = ap.parse_args()

    # --- provenance FIRST (task #1): fail loudly rather than stamp 'unknown' ---
    code_commit = _assert_provenance()
    dataset = "real-loa" if a.real_loa else "mock"
    if a.out is None:
        a.out = DEFAULT_REAL_OUT if a.real_loa else DEFAULT_MOCK_OUT

    if a.dry_run:
        _print_config(a, dataset, a.fp_estimator, a.loa0_product, code_commit)
        return

    if a.real_loa:
        print("#" * 78)
        print("# 🔴 REAL DESI LOA sub-DLA headline — OUTPUTS ARE PRIVATE.")
        print("#   dN/dX / Omega / f(N) values must NOT be committed to the code repo,")
        print("#   pasted into a shared transcript, or quoted. Scratch + private notes only.")
        print("#   (This routine writes to scratch and refuses any repo path.)")
        print("#" * 78)

    _refuse_repo_write(a.out)
    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(a.out) and not a.force:
        raise SystemExit(f"refusing to overwrite existing {a.out} (pass --force).")

    # --- config-only gate / mock validation (baseline-recovery twin) ---
    # forward = the headline anchor (0.849/0.822); kappa = retired-artifact config-drift
    # tripwire. --gate runs both; the default pre-run validation checks the kernel this
    # routine actually measures with (forward).
    if a.gate:
        validate_against_committed(("purity_mixture", "loa0"), resp_kind="forward")
        validate_against_committed(("purity_mixture", "loa0"), resp_kind="kappa")
    elif dataset == "mock" and not a.no_validate:
        validate_against_committed(("loa0",), resp_kind="forward")

    # --- the full-schema sub-DLA headline measurement (run_measurement) ---
    t0 = time.time()
    args = build_args(a, dataset)
    res = run_headline(args, a.fp_estimator, a.loa0_product)
    wall = time.time() - t0

    out_json = assemble_out_json(res, args, args._limits, wall, a.fp_estimator,
                                 a.loa0_product, dataset, code_commit)
    with open(a.out, "w") as fh:
        json.dump(out_json, fh, indent=2, default=float)
    print(f"\n[subdla] wrote {a.out}  ({wall:.0f}s)  code_commit={code_commit}  "
          f"dataset={dataset} fp={a.fp_estimator}")

    # aggregate echo — MOCK values are public; on real-loa these ARE private (operator sees
    # them at the console but they are never committed).
    if dataset == "mock":
        for l in (19.5, 20.0, 20.3):
            if l in res["dndx"]:
                di = res["dndx"][l]["integrated"]["MAP"]; oi = res["omega"][l]["integrated"]["MAP"]
                print(f"  [mock] >= {l}: integ dN/dX={di:.4f}  1e3*Om={1e3*oi:.3f}")
        wb = res.get("window_band_195_203")
        if wb:
            for kind, sc, u in (("dndx", 1.0, ""), ("omega", 1e3, "1e3*")):
                r = wb[kind]
                r0 = f" R0={r.get('R0_MAP'):.4f} [{r.get('R0_q16'):.4f},{r.get('R0_q84'):.4f}]" \
                    if "R0_MAP" in r else ""
                print(f"  [mock] WINDOW [19.5,20.3) {u}{kind}: MAP={sc*r['MAP']:.5g} "
                      f"q16={sc*r['q16']:.5g} q84={sc*r['q84']:.5g} "
                      f"q025={sc*r['q025']:.5g} q975={sc*r['q975']:.5g} std={sc*r['std']:.5g}"
                      f"{r0}  (raw_median={sc*r['raw_median']:.5g}, "
                      f"jensen_shift={sc*r['jensen_shift']:+.3g})")
    else:
        print("  [real-loa] aggregate values written to scratch (PRIVATE; not echoed here).")


if __name__ == "__main__":
    main()
