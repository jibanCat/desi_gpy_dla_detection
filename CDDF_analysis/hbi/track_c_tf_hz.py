#!/usr/bin/env python
"""track_c_tf_hz.py — the BH-candidate high-z (z_QSO>4.25) real-LOA measurement.

CONFIG-ONLY extension of track_c_tf_loa.py to the 3,022-QSO high-z production
catalog (gl_cddf_loa_hz_v1_20260813), following the run_loa0_headline_full.py
override pattern: reuses build_frozen_calibration / build_loa_ingredients /
run_measurement UNCHANGED. No estimator edit; no re-inference; raw production
catalog read-only.

The three config differences vs the low-z headline, each mechanical:
  1. catalog/mockdir/out point at the high-z production + its staged
     snr_cat/zcat/bal_cat (same BI_CIV>0 bal convention as the low-z stage);
  2. the QSO window is the high-z production universe: z_qso in (4.25, 7.0)
     (the low-z analysis uses (2.0, 4.25) strict, so the two partition
     cleanly). HBIConfig does not expose z_qso_* to the caller, so this
     runner wraps the dataclass DURING build_loa_ingredients only (the
     frozen 2LPT-0 calibration build keeps its own mock window untouched);
     the fine z-fit grid keeps lo=2.0 (grid-identity with the frozen
     g(N,z)); columns without high-z pathlength carry zero weight.
  3. zbins default to the H2-v2 predeclared absorber-z bins
     3.8,4.25,4.5,5.0 (the BH candidate [3.8,5.0) + its sub-bins).

Completeness variants (--variant):
  frozenC : the unchanged frozen 2LPT-0 molly C (every BH bin is
            z-extrapolated under the frozen g(N,z) — the mechanical
            'as-wired' run, DIAGNOSTIC of what the old calibration gives).
  h2cal   : the PI-ADOPTED (2026-08-15) H2-v2 arm-B in-regime calibration:
            the molly C ratio cells for SNR>=2 rows are replaced by the
            CANONICAL arm-B DETECTION completeness per truth-NHI molly bin
            (contract v1; detection — NOT reporting — completeness, because
            migration lives in the forward kernel; no double-count), and
            the Jeffreys count denominators by the H2 counts (SNR-marginal
            counts per row — documented approximation). Cells with NO H2
            truth support ([20.3,20.5) — a design-grid gap — [22.0,inf),
            and SNR<2 rows) KEEP the frozen molly value and are flagged.
  --envelope plus|minus applies the adopted A/B transport envelope
            (|dC|+se per NHI domain, canonical nobal values) to the h2cal
            C table — bounded sensitivity, separate from the stat band.

FP (--fp): loa0 (headline product, lya-only 1025 — refuses the lyab window)
           or pm (purity_mixture from the frozen molly rho).
Window (--window): lya (P1_PRIMARY_LYA) or lyab (P1_SENS_LYAB: lam_rf_min
           911 + the established figures_molly_nhi195/lya_lyb matrix).

Output: same schema as track_c_tf_loa.json + variant/calibration metadata.
Estimand stamp: DIAGNOSTIC_RECENTERED / paper_facing=false — the band
machinery is the recentered-MC one retired for paper use on 2026-07-28;
its successor (run_posterior.py) refuses real-survey packs pending forward
closure. BH results are therefore CANDIDATE / PI-ADOPTION-PENDING.

Real-LOA numbers go to SCRATCH only. Env: gpdla; *_NUM_THREADS=1.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_ARB = os.path.join(_REPO, "CDDF_analysis", "diagnostics", "bal_metal_fp", "arbiter")
_spec = importlib.util.spec_from_file_location(
    "bt_helper", os.path.join(_ARB, "apply_broadtrough_veto_headline.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
TF = H.TF

HZ_ROOT = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/tf_hz"
HZ_CAT = ("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
          "gl_cddf_loa_hz_v1_20260813/outputs")
HZ_MOCKDIR = os.path.join(HZ_ROOT, "mockdir")
LYAB_TSV = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
            "figures_molly_nhi195/lya_lyb/molly_matrix.tsv")
H2_CANON = "/scratch/cavestru_root/cavestru0/mfho/loa_hz_production/h2_exec"

# adopted A/B transport envelope (canonical, nobal, P1_PRIMARY_LYA):
# |dC| + se per truth-NHI domain (h2_canonical_ab_transport regeneration)
ENVELOPE = {"19.5-20.0": 0.10, "20.0": 0.0604, "20.3": 0.0761}

# MAX4 repair cycle (PI ruling 2026-08-28 item 3; MAX4 CHECKPOINT 0 §D, 2026-09-01): the
# catalogue directory and the per-QSO snr/z/bal tables are CLI-overridable so a
# configuration-matched (MAX_DLAS=4) high-z catalogue can feed this track without editing
# module constants. Defaults = the constants above (run of record byte-identical). The
# finder configuration of the catalogue actually consumed is stamped into the output
# metadata from its BASELINE.env so a MAX1 / MAX4 product can never be confused downstream.
FINDER_CONFIG_KEYS = ("MAX_DLAS", "SINGLE_ABSORBER_MODEL", "FILTER_LOW_LIKELIHOOD",
                      "NUM_DLA_SAMPLES", "CODE_COMMIT")


def read_finder_config(cat_dir, keys=FINDER_CONFIG_KEYS):
    """Read the finder configuration of a catalogue directory from its `BASELINE.env`
    (the launcher's resolved `KEY=VALUE` record). Returns a dict with one entry per key
    (None when the key is absent) plus `baseline_env` (the file read), or the string
    "unavailable" when the directory has no BASELINE.env."""
    path = os.path.join(str(cat_dir), "BASELINE.env")
    if not os.path.isfile(path):
        return "unavailable"
    found = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            found[k.strip()] = v.strip()
    out = {k: found.get(k) for k in keys}
    out["baseline_env"] = path
    return out


def hz_input_stamp(hz_cat, hz_mockdir):
    """Metadata block naming the catalogue / per-QSO tables consumed and the catalogue's
    finder configuration (`read_finder_config`)."""
    return dict(hz_cat=str(hz_cat), hz_mockdir=str(hz_mockdir),
                finder_config=read_finder_config(hz_cat))


def finite_snr_guard(lookup, drop=False):
    """Guard for the NaN-SNR hole in build_pathlength's sightline test (R-039 closure,
    2026-08-28): quasars whose SNR_REDSIDE is not finite pass `snr <= snr_min` and are
    counted in n_op_sl / the loa-0 FP volume scale although the row-level detection mask
    can never accept a candidate on them. Returns (lookup, n_nonfinite); with drop=True the
    non-finite entries are removed, otherwise the lookup is returned unchanged and a
    warning is printed. Default behaviour of the run of record is unchanged."""
    bad = [t for t, v in lookup.items() if not np.isfinite(v[0])]
    if drop:
        print(f"  [finite-snr-only] dropping {len(bad)} quasars with non-finite SNR_REDSIDE "
              f"from the sightline population before ingest")
        return {t: v for t, v in lookup.items() if np.isfinite(v[0])}, len(bad)
    if bad:
        print(f"  [WARN] {len(bad)} quasars with non-finite SNR_REDSIDE are in the sightline "
              f"population (counted by build_pathlength; carry no candidates); pass "
              f"--finite-snr-only to drop them")
    return lookup, len(bad)


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def h2_c_table(window, gap_treatment="frozen"):
    """Canonical arm-B DETECTION completeness per molly truth-NHI bin + counts.

    gap_treatment='h2coarse' (PI item 7 sensitivity, 2026-08-16): fill the
    H2 design-grid gap cell [20.3,20.5) with the canonical detection C of
    the coarser PREDECLARED stratum nre:[20.3,20.7) (which covers the cell)
    instead of the frozen 2LPT value. Bounded, defensible alternative —
    no redesign, no top-up; both treatments reported."""
    sfx = "lya" if window == "lya" else "lyab"
    j = json.load(open(f"{H2_CANON}/h2_canonical_armB_{sfx}_nobal.json"))
    out = {}
    coarse = None
    for s in j["detection_strata"]:
        if s["stratum"].startswith("molly_nhi:"):
            key = s["stratum"].split(":", 1)[1]
            out[key] = (s["detection_C"], s["k"], s["n"])
        if s["stratum"] == "nre:20.3-20.7":
            coarse = (s["detection_C"], s["k"], s["n"])
    if gap_treatment == "h2coarse" and coarse is not None:
        for key in list(out):
            if key.startswith("[20.3"):
                out[key] = coarse
    return out, j["contract"]


def _neff_from_cgap_record(path=os.path.join(HZ_ROOT, "H2_CGAP_INFERENCE.json")):
    """Effective Beta trial count reproducing the C_gap inference's 68 % half-width:
    for Beta(mean m, n_eff) var ~ m(1-m)/(n_eff+1) -> n_eff = m(1-m)/var - 1."""
    rec = json.load(open(path))
    p16, p50, p84 = rec["posterior"]["C_gap_p16_50_84"]
    sd = 0.5 * (p84 - p16)
    return max(p50 * (1.0 - p50) / sd ** 2 - 1.0, 1.0)


def patch_mm_with_h2(mm, window, envelope, gap_treatment="frozen", gap_c=None, gap_c_neff=None):
    """In-place ADOPTED-calibration override of the molly C cells (SNR>=2 rows)."""
    tab, contract = h2_c_table(window, gap_treatment)
    if gap_c is not None:
        # PI item 2 (2026-08-16): explicit H2-inference value for the gap
        # cell [20.3,20.5) — used by the C_gap response-mapping grid.
        for key in list(tab):
            if key.startswith("[20.3"):
                tab[key] = (float(gap_c), None, None)
    nhi_edges = np.asarray(mm.nhi_edges, float)
    snr_edges = np.asarray(mm.snr_edges, float)
    patched, kept = [], []
    for jcell in range(len(nhi_edges) - 1):
        lo, hi = nhi_edges[jcell], (nhi_edges[jcell + 1]
                                    if jcell + 2 <= len(nhi_edges) - 1 else np.inf)
        key = f"[{lo},{hi if np.isfinite(hi) else 'inf'})"
        # analyzer wrote keys like "[19.5,20.0)"; normalize
        cand = [k for k in tab if abs(float(k.split(",")[0][1:]) - lo) < 1e-9]
        C = k_ = n_ = None
        if cand and tab[cand[0]][0] is not None:
            t0 = tab[cand[0]]
            if t0[2] is None or t0[2] > 0:     # explicit gap_c has n=None
                C, k_, n_ = t0
        if C is None:
            kept.append(key)
            continue
        if k_ is None:            # explicit gap_c
            # A5 fix (2026-08-28, R-041A): the MC band draws C per cell from
            # Beta(cmp_nfound + 0.5, cmp_nfid - cmp_nfound + 0.5) (joint_mc_errors ->
            # _draw_beta_cell); leaving the FROZEN counts in place while overriding the
            # point value made the uncertainty draw centre on the frozen cell's k/n, not on
            # gap_c. Now the counts are set to (gap_c * n_eff, n_eff) so the draw is
            # centred on the adopted value with the H2-inference width (n_eff from the
            # inference record's 68 % interval unless overridden).
            n_eff = float(gap_c_neff) if gap_c_neff is not None else _neff_from_cgap_record()
            for i in range(len(snr_edges) - 1):
                if snr_edges[i] < 2.0:
                    continue
                mm.completeness[i, jcell] = C
                mm.cmp_nfound[i, jcell] = C * n_eff
                mm.cmp_nfid[i, jcell] = n_eff
            patched.append(dict(cell=key, C=C, k=C * n_eff, n=n_eff,
                                note="explicit gap_c (PI item 2); counts set to (gap_c*n_eff, n_eff) so the Beta draw is centred on gap_c (A5 fix 2026-08-28)"))
            continue
        if envelope:
            dom = ("19.5-20.0" if lo < 20.0 - 1e-9 else
                   "20.0" if lo < 20.3 - 1e-9 else "20.3")
            C = float(np.clip(C + (ENVELOPE[dom] if envelope == "plus"
                                   else -ENVELOPE[dom]), 0.02, 0.98))
        for i in range(len(snr_edges) - 1):
            if snr_edges[i] < 2.0:      # H2 sample is SNR_REDSIDE>2: keep frozen
                continue
            mm.completeness[i, jcell] = C
            mm.cmp_nfound[i, jcell] = k_
            mm.cmp_nfid[i, jcell] = n_
        patched.append(dict(cell=key, C=C, k=k_, n=n_))
    return dict(patched=patched, kept_frozen=kept, h2_contract=contract)


def patch_mm_with_r041(mm, analysis_json, strata_edges=(2.0, 3.0, 4.0, 5.0, 7.0, np.inf), min_n=5):
    """R-041A (2026-08-28): in-place override of the molly C cells (SNR>=2 rows) with the
    corrected real-spectrum single-injection calibration — the analyzer's
    per_molly_cell_x_stratum (k, n) counts. Every SNR row of the molly grid is mapped to the
    R-041 SNR_REDSIDE stratum that contains its centre ([2,3),[3,4),[4,5),[5,7),[7,inf));
    a cell with n < min_n keeps the frozen value and is flagged. The counts ARE the real
    (k, n), so the MC Beta draw is centred on the measured C with its true width (A5)."""
    tab = json.load(open(analysis_json))["tables"]["per_molly_cell_x_stratum"]
    by = {}
    for c in tab:
        key = c["key"]
        lo = float(key["n_lo"]) if "n_lo" in key else float(key["lo"])      # analyzer key: molly_cell, n_lo, n_hi, stratum
        by[(round(lo, 3), int(key["stratum"]))] = c
    nhi_edges = np.asarray(mm.nhi_edges, float); snr_edges = np.asarray(mm.snr_edges, float)
    se = np.asarray(strata_edges, float)
    patched, kept = [], []
    for jcell in range(len(nhi_edges) - 1):
        lo = nhi_edges[jcell]
        for i in range(len(snr_edges) - 1):
            if snr_edges[i] < 2.0:
                kept.append(dict(cell=f"[{lo},...)", snr_row=i, reason="SNR<2 row: outside the calibration population")); continue
            centre = 0.5 * (snr_edges[i] + (snr_edges[i + 1] if np.isfinite(snr_edges[i + 1]) else snr_edges[i] + 2.0))
            s = int(np.digitize(centre, se) - 1)
            c = by.get((round(lo, 3), s))
            if c is None or c["n"] < min_n or c["C"] is None:
                kept.append(dict(cell=f"[{lo},...)", snr_row=i, stratum=s, n=(c["n"] if c else 0), reason="no R-041 support")); continue
            mm.completeness[i, jcell] = float(c["C"]); mm.cmp_nfound[i, jcell] = float(c["k"]); mm.cmp_nfid[i, jcell] = float(c["n"])
            patched.append(dict(cell=f"[{lo},...)", snr_row=i, stratum=s, C=float(c["C"]), k=int(c["k"]), n=int(c["n"])))
    return dict(patched=patched, kept_frozen=kept, source=analysis_json, strata_edges=[float(x) if np.isfinite(x) else "inf" for x in se])


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["frozenC", "h2cal", "r041cal"], default="h2cal")
    ap.add_argument("--r041-analysis", default=None, help="r041_analyze.py JSON (variant r041cal)")
    ap.add_argument("--fp", choices=["loa0", "pm"], default="loa0")
    ap.add_argument("--window", choices=["lya", "lyab"], default="lya")
    ap.add_argument("--envelope", choices=["none", "plus", "minus"], default="none")
    ap.add_argument("--gap-treatment", choices=["frozen", "h2coarse"], default="frozen")
    ap.add_argument("--gap-c", type=float, default=None,
                    help="explicit C value for the [20.3,20.5) gap cell (item-2 response mapping)")
    ap.add_argument("--gap-c-neff", type=float, default=None,
                    help="effective Beta trial count for the gap cell's MC draw (A5 fix); default = "
                         "from H2_CGAP_INFERENCE.json's 68 %% interval")
    ap.add_argument("--zbins", default="3.8,4.25,4.5,5.0")
    ap.add_argument("--n-mc", type=int, default=120)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--force", action="store_true")
    # 2026-08-28 (paper1 requests, R-039 closure): build_pathlength's sightline test is
    # `snr <= snr_min: skip`, which a NON-FINITE SNR_REDSIDE passes, so such quasars are
    # counted in n_op_sl and in the loa-0 FP volume scale although the row-level detection
    # mask (SNR > 2) can never accept a candidate on them. Default OFF keeps the artifact of
    # record bit-for-bit; the default path now only PRINTS how many such quasars it counted.
    ap.add_argument("--finite-snr-only", action="store_true",
                    help="drop quasars with non-finite SNR_REDSIDE from the sightline "
                         "population before ingest (guarded variant; default off)")
    ap.add_argument("--work-root", default=None,
                    help="directory for the run's side files (report, out_dir); default = the "
                         "frozen HZ_ROOT, unchanged for the run of record")
    ap.add_argument("--dump-npz", default=None,
                    help="also save the MAP f(N), the MC f(N) samples, the grids and X_tot "
                         "(diagnostic carrier for the Omega tail study; default off)")
    # MAX4 repair cycle: catalogue / per-QSO-table overrides (defaults = the run of record).
    ap.add_argument("--hz-cat", default=HZ_CAT,
                    help="high-z catalogue directory (dlacat-*.fits [+ BASELINE.env]); default = "
                         "the MAX_DLAS=1 run of record (HZ_CAT)")
    ap.add_argument("--hz-mockdir", default=HZ_MOCKDIR,
                    help="directory holding snr_cat/zcat/bal_cat/dla_cat.fits of the high-z "
                         "sample; default = the run of record (HZ_MOCKDIR)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)

    if a.window == "lyab" and a.fp == "loa0":
        raise SystemExit("loa0 FP product is lya-only-1025; use --fp pm for lyab.")

    tag = (f"{a.variant}_{a.fp}_{a.window}"
           + (f"_env{a.envelope}" if a.envelope != "none" else "")
           + (f"_gap{a.gap_treatment}" if a.gap_treatment != "frozen" else "")
           + (f"_gapc{a.gap_c:g}" if a.gap_c is not None else ""))
    root = a.work_root or HZ_ROOT
    out_path = a.out_json or os.path.join(root, f"track_c_tf_hz_{tag}.json")
    os.makedirs(root, exist_ok=True)
    if os.path.exists(out_path) and not a.force:
        raise SystemExit(f"refusing to overwrite {out_path} (pass --force).")

    args = H.default_args()
    args.loa_cat = a.hz_cat
    args.loa_mockdir = a.hz_mockdir
    args.loa_truth = os.path.join(a.hz_mockdir, "dla_cat.fits")
    args.loa_bal = os.path.join(a.hz_mockdir, "bal_cat.fits")
    args.out = root
    args.report_out = os.path.join(root, f"_report_{tag}.md")
    args.zbins = a.zbins
    args.v2_z_fit_hi = 5.0
    args.n_mc = a.n_mc
    limits = (20.0, 20.3)
    args._limits = limits
    args.report_limits = "20.0,20.3"
    if a.window == "lyab":
        args.lam_rf_min = 911.0
        args.molly_tsv = LYAB_TSV

    t0 = time.time()
    frozen = TF.build_frozen_calibration(args)     # 2LPT-0 window untouched
    args.molly_tsv = frozen["molly_tsv"]

    # ---- the ONLY structural override: the QSO window of the real catalog ----
    _HBIConfig = TF.HBIConfig

    def _HZConfig(**kw):
        # v2_z_fit_lo stays 2.0: the frozen g(N,z) grid and the forward fine-z
        # grid must match (30 cols); empty low-z columns carry zero pathlength.
        c = _HBIConfig(**kw)
        c.z_qso_min = 4.25
        c.z_qso_max = 7.0
        return c

    TF.HBIConfig = _HZConfig
    _orig_lookup = TF._build_qso_lookup
    _nonfinite = {"n": 0}

    def _guarded_lookup(cfg):
        lk, n_bad = finite_snr_guard(_orig_lookup(cfg), drop=a.finite_snr_only)
        _nonfinite["n"] = n_bad
        return lk

    TF._build_qso_lookup = _guarded_lookup
    try:
        ing = TF.build_loa_ingredients(args, frozen)
    finally:
        TF.HBIConfig = _HBIConfig
        TF._build_qso_lookup = _orig_lookup
    cfg = ing["cfg"]
    assert cfg.z_qso_min == 4.25 and cfg.z_qso_max == 7.0

    cal_meta = dict(variant=a.variant, envelope=a.envelope)
    if a.variant == "h2cal":
        from CDDF_analysis.hbi.cddf_catalog_hbi import make_C_interpolator
        env = None if a.envelope == "none" else a.envelope
        cal_meta["gap_treatment"] = a.gap_treatment
        cal_meta["h2_patch"] = patch_mm_with_h2(ing["mm"], a.window, env, a.gap_treatment, a.gap_c, a.gap_c_neff)
        ing["C_interp"] = make_C_interpolator(ing["mm"])
    elif a.variant == "r041cal":
        from CDDF_analysis.hbi.cddf_catalog_hbi import make_C_interpolator
        if not a.r041_analysis:
            raise SystemExit("--variant r041cal needs --r041-analysis")
        cal_meta["r041_patch"] = patch_mm_with_r041(ing["mm"], a.r041_analysis)
        ing["C_interp"] = make_C_interpolator(ing["mm"])

    if a.fp == "loa0":
        from CDDF_analysis.hbi.cddf_catalog_hbi import (
            make_fp_model, make_rho_interpolator)
        cfg.fp_estimator = "loa0"
        cfg.loa0_product_path = H.LOA0_LYAONLY
        H.preflight_loa0_product(H.LOA0_LYAONLY, cfg, args.molly_tsv)
        rho = make_rho_interpolator(ing["mm"])
        loa0_model, _ = make_fp_model(cfg, ing["cat_cut"], ing["op_mask"], rho)
        ing["fp_model"] = loa0_model
        assert loa0_model.n_sl_prod == ing["n_sl"], "loa0 n_sl_prod guard"
        cal_meta["loa0"] = dict(n_sl_loa0=float(loa0_model.n_sl_loa0),
                                n_sl_prod=float(loa0_model.n_sl_prod),
                                vol_scale=float(loa0_model.vol_scale),
                                transport_flag=("loa0 FP background measured on a "
                                                "LOW-z HCD-free twin mock; volume-"
                                                "scaled to the high-z op sample — "
                                                "transport assumption, flagged"))

    res = TF.run_measurement(args, ing, limits, args.seed, frozen=frozen)
    wall = time.time() - t0
    if a.dump_npz:
        np.savez(a.dump_npz, map_fb=res["map_fb"], fb_samp=res["fb_samp"],
                 map_fbk=res["map_fbk"], logN_lo=res["logN_lo"], logN_hi=res["logN_hi"],
                 N_b=res["N_b"], dN_b=res["dN_b"], K=float(res["K"]),
                 X_tot=np.asarray(res["X_tot"], float), zbins=np.asarray(res["zbins"], float),
                 n_op_sl=int(res["n_op_sl"]), n_op_detections=int(res["n_op_detections"]),
                 band_recenter=bool(cfg.band_recenter), finite_snr_only=bool(a.finite_snr_only),
                 n_nonfinite_snr_in_population=int(_nonfinite["n"]), seed=int(args.seed),
                 n_mc=int(args.n_mc), argv=np.array(sys.argv))
        print(f"  [dump] wrote {a.dump_npz}")

    out_json = dict(
        metadata=dict(
            estimand="DIAGNOSTIC_RECENTERED", paper_facing=False,
            status="CANDIDATE / PI-ADOPTION-PENDING (BH high-z bin)",
            sample=("P1_PRIMARY_LYA" if a.window == "lya" else "P1_SENS_LYAB"),
            contract="CANONICAL_PURITY_COMPLETENESS_CONTRACT v1",
            fp_estimator=cfg.fp_estimator, calibration=cal_meta,
            z_qso_window=[4.25, 7.0], lam_rf_min=float(args.lam_rf_min),
            n_mc=args.n_mc, seed=args.seed, limits=list(limits),
            resp_kind="forward", forward_model=args.forward_model,
            molly_tsv=args.molly_tsv, loa_cat=args.loa_cat,
            n_op_detections=res["n_op_detections"], n_op_sl=res["n_op_sl"],
            consistency_err=res["consistency_err"],
            v2_z_fit=[2.0, float(args.v2_z_fit_hi)],
            z_extrapolated=[bool(x) for x in np.asarray(res.get("z_extrapolated", []))],
            truth_counts_perz=res.get("truth_counts_perz"),
            max_truth_z=float(res.get("max_truth_z", float("nan"))),
            wallclock_s=float(wall), code_commit=_git_commit(),
            **({"finite_snr_only": True,
                "n_nonfinite_snr_dropped": int(_nonfinite["n"])} if a.finite_snr_only else {}),
            # MAX4 repair cycle: the catalogue / tables consumed and the finder configuration
            # of that catalogue (from its BASELINE.env; "unavailable" when absent).
            **hz_input_stamp(a.hz_cat, a.hz_mockdir),
            # Paper-1 code review 2026-08-26: the invocation is part of the record (the
            # frozen artifact of record was produced with --n-mc 2000 against a CLI
            # default of 120, and no launch script existed).
            argv=list(sys.argv)),
        measurement={
            str(l): dict(
                dndx=dict(
                    perz=[res["dndx"][l]["perz"][k] for k in range(res["n_zc"])],
                    integrated=res["dndx"][l]["integrated"]),
                omega=dict(
                    perz=[res["omega"][l]["perz"][k] for k in range(res["n_zc"])],
                    integrated=res["omega"][l]["integrated"]),
            ) for l in limits},
        zbins=list(map(float, res["zbins"])))
    out_json["perz_fN"] = TF.assemble_perz_fN(res, limits)
    with open(out_path, "w") as fh:
        json.dump(out_json, fh, indent=2, default=float)
    print(f"[tf_hz:{tag}] wrote {out_path} ({wall:.0f}s)")
    for l in limits:
        di = res["dndx"][l]["integrated"]["MAP"]
        oi = res["omega"][l]["integrated"]["MAP"]
        print(f"  >= {l}: integ dN/dX={di:.4f}  1e3*Om={1e3*oi:.3f}")


if __name__ == "__main__":
    main()
