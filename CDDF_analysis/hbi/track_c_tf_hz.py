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


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def h2_c_table(window):
    """Canonical arm-B DETECTION completeness per molly truth-NHI bin + counts."""
    sfx = "lya" if window == "lya" else "lyab"
    j = json.load(open(f"{H2_CANON}/h2_canonical_armB_{sfx}_nobal.json"))
    out = {}
    for s in j["detection_strata"]:
        if s["stratum"].startswith("molly_nhi:"):
            key = s["stratum"].split(":", 1)[1]
            out[key] = (s["detection_C"], s["k"], s["n"])
    return out, j["contract"]


def patch_mm_with_h2(mm, window, envelope):
    """In-place ADOPTED-calibration override of the molly C cells (SNR>=2 rows)."""
    tab, contract = h2_c_table(window)
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
        if cand and tab[cand[0]][0] is not None and tab[cand[0]][2] > 0:
            C, k_, n_ = tab[cand[0]]
        if C is None:
            kept.append(key)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["frozenC", "h2cal"], default="h2cal")
    ap.add_argument("--fp", choices=["loa0", "pm"], default="loa0")
    ap.add_argument("--window", choices=["lya", "lyab"], default="lya")
    ap.add_argument("--envelope", choices=["none", "plus", "minus"], default="none")
    ap.add_argument("--zbins", default="3.8,4.25,4.5,5.0")
    ap.add_argument("--n-mc", type=int, default=120)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.window == "lyab" and a.fp == "loa0":
        raise SystemExit("loa0 FP product is lya-only-1025; use --fp pm for lyab.")

    tag = f"{a.variant}_{a.fp}_{a.window}" + (f"_env{a.envelope}"
                                              if a.envelope != "none" else "")
    out_path = a.out_json or os.path.join(HZ_ROOT, f"track_c_tf_hz_{tag}.json")
    os.makedirs(HZ_ROOT, exist_ok=True)
    if os.path.exists(out_path) and not a.force:
        raise SystemExit(f"refusing to overwrite {out_path} (pass --force).")

    args = H.default_args()
    args.loa_cat = HZ_CAT
    args.loa_mockdir = HZ_MOCKDIR
    args.loa_truth = os.path.join(HZ_MOCKDIR, "dla_cat.fits")
    args.loa_bal = os.path.join(HZ_MOCKDIR, "bal_cat.fits")
    args.out = HZ_ROOT
    args.report_out = os.path.join(HZ_ROOT, f"_report_{tag}.md")
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
    try:
        ing = TF.build_loa_ingredients(args, frozen)
    finally:
        TF.HBIConfig = _HBIConfig
    cfg = ing["cfg"]
    assert cfg.z_qso_min == 4.25 and cfg.z_qso_max == 7.0

    cal_meta = dict(variant=a.variant, envelope=a.envelope)
    if a.variant == "h2cal":
        from CDDF_analysis.hbi.cddf_catalog_hbi import make_C_interpolator
        env = None if a.envelope == "none" else a.envelope
        cal_meta["h2_patch"] = patch_mm_with_h2(ing["mm"], a.window, env)
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
            wallclock_s=float(wall), code_commit=_git_commit()),
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
