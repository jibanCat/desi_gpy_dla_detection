#!/usr/bin/env python
"""build_lane_l_matched_artifact.py — the Lane-L matched alpha-vs-HBI
comparison artifact (spec: notes 2026-08-12 handoff §3/§6.1 + WP-1b
+ REVIEW_PACKAGE_ADDENDUM; PI ruling 2026-08-15/16 item 5).

Three arms per (z-zone × threshold in {20.0, 20.3}) on 2LPT-0, all on the
SAME matched population (the committed pack universe: 374,177 sightlines,
build_pathlength(no_bal=True) whitelist):

  raw FF        : posterior-weighted detected counts (ff_matched_2lpt0_long
                  .json, job 57261601; cross-commit byte-identical receipt
                  vs job 57250729) — a PLUG-IN SAMPLING point, not a
                  credible interval;
  alpha-corr FF : raw FF × alpha, alpha = 1/R0 from the COMMITTED
                  full-population artifact calccddf_vs_hbi.json (the
                  matched-population R0 does not exist — the FF driver
                  writes no truth counts — so alpha is TRANSFERRED from
                  the unmatched population and labeled as such; the
                  on-mock alpha=1/R0 tautology caveat applies);
  fold arm      : the RATIFIED P1 empirical-kernel fold
                  (p1_refold_fold: build_fold + build_p1_kernel +
                  mu_sig_p1 + mu_fp + below-floor net migration), zone-
                  reduced on the pack's fine-z grid.

Both dX conventions are recorded per zone (FF search-window dX from the
harness vs the pack build_pathlength dX) — the choice moves the headline
by ~60%, so the artifact records both and picks NEITHER.

Migration z-resolution: the frozen load_migration returns no per-migrant
z, so this builder replicates the frozen net-migrant selection with Z_DLA
attached and GATES the replication on the committed per-group totals
(exact, fail-loud) before zone-resolving it.

Env: gpdla-hbi (jax; the fold rebuild guard needs it). Output committed
on hbi/forward-2026-08 (mock-only content; no real values).
"""
from __future__ import annotations
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_P1 = os.path.join(_REPO, "diagnostics_phaseC", "p1_completeness")
for p in (_REPO, _P1):
    if p not in sys.path:
        sys.path.insert(0, p)

FF_JSON = ("/scratch/cavestru_root/cavestru0/mfho/wp2prime_2026_08_12/"
           "ff_matched_2lpt0_long.json")
VS_HBI = os.path.join(_HERE, "calccddf_vs_hbi.json")
SPLITS = os.path.join(_HERE, "calccddf_2lpt0_splits.json")
OUT = os.path.join(_HERE, "lane_l_matched_alpha_vs_hbi.json")
Z_ZONES = [(2.0, 2.4), (2.4, 2.6), (2.6, 3.0), (3.0, 3.5)]
THRESH = (20.0, 20.3)
FLOOR = 19.5
P_DLA_MIN = 0.99


def main():
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    import p1_refold_fold as F
    from p1_refold_fold import (PACK, CACHE172, CACHE195,
                                MIGRATION_GROUPS_REF, P1RefoldGuardError)

    ff = json.load(open(FF_JSON))
    assert ff["schema"] == "ff_matched_window/v1"
    vs = json.load(open(VS_HBI))
    R0_full = vs["mocks"]["2lpt0"]["dndx"]["R0_calccddf"]
    hbi_fwd = vs["hbi_forward"]["2lpt0"]
    splits = json.load(open(SPLITS)) if os.path.exists(SPLITS) else None

    pk = load_pack(PACK)
    fold = F.build_fold(pk)
    E, truth, sparse, art, cache = F.load_kernel_events()
    K_P1, kinfo = F.build_p1_kernel(E, fold, sparse)
    mu_sig = np.asarray(F.mu_sig_p1(K_P1, fold))            # (C,Kf,S)
    mu_fp = np.asarray(fold["mu_fp"])                       # (C,Kf,S)
    obs = np.asarray(pk.counts, float)                      # (C,Kf,S)
    ne = np.asarray(pk.nhat_edges, float)
    dX = np.asarray(pk.dX, float)                           # (Kf,S)
    kcent = 2.05 + 0.1 * np.arange(dX.shape[0])

    # z-resolved below-floor net migration: frozen selection REPLICATED with
    # Z_DLA attached; gated on the committed group totals (fail-loud).
    mig_ref = F.load_migration(ne)                          # frozen totals gate
    d = np.load(CACHE172)
    d5 = np.load(CACHE195)
    sel = ((d["cat_P_DLA"] > P_DLA_MIN) & d["cat_good"]
           & (d["cat_S2N"] > 2.0) & (d["cat_NHI"] > FLOOR))
    sel5 = ((d5["cat_P_DLA"] > P_DLA_MIN) & d5["cat_good"]
            & (d5["cat_S2N"] > 2.0) & (d5["cat_NHI"] > FLOOR))
    tp195 = set(zip(d5["cat_TARGETID"][d5["cat_is_TP"] & sel5].tolist(),
                    np.round(d5["cat_Z_DLA"][d5["cat_is_TP"] & sel5], 6).tolist()))
    rowkeys = list(zip(d["cat_TARGETID"].tolist(),
                       np.round(d["cat_Z_DLA"], 6).tolist()))
    in195 = np.array([k in tp195 for k in rowkeys])
    net = sel & d["cat_is_TP"] & (d["cat_NHI_TRUE"] < FLOOR) & ~in195
    mig_nhat = np.asarray(d["cat_NHI"][net], float)
    mig_z = np.asarray(d["cat_Z_DLA"][net], float)
    if int(net.sum()) != mig_ref["n_net_total"]:
        raise P1RefoldGuardError("migration replication drift (total)")
    from CDDF_analysis.hbi_mcmc.gate_covariance import PRIMARY_GROUP_EDGES
    for (glo, ghi), g in zip(PRIMARY_GROUP_EDGES, ("G1", "G2", "G3")):
        if int(((mig_nhat >= glo) & (mig_nhat < ghi)).sum()) != MIGRATION_GROUPS_REF[g]:
            raise P1RefoldGuardError(f"migration replication drift ({g})")

    def fold_zone(thr, zlo, zhi):
        cm = ne[:-1] >= thr - 1e-9
        km = (kcent >= zlo) & (kcent < zhi)
        mu = float(mu_sig[cm][:, km, :].sum() + mu_fp[cm][:, km, :].sum())
        mmig = int(((mig_nhat >= thr) & (mig_z >= zlo) & (mig_z < zhi)).sum())
        o = float(obs[cm][:, km, :].sum())
        return mu + mmig, o, mmig

    def pack_dX_zone(zlo, zhi):
        km = (kcent >= zlo) & (kcent < zhi)
        return float(dX[km, :].sum())

    zones_out = []
    n_edges = np.asarray(ff["n_edges"], float)

    # robust zone-key discovery in the FF JSON
    def ff_zone(zlo, zhi):
        if zlo == 2.0 and zhi == 3.5:
            return ff.get("integrated") or ff["zones"]["integrated"]
        for k, v in ff["zones"].items():
            if "," not in k:
                continue
            nums = [float(x) for x in k.strip("[)").replace("(", "").split(",")]
            if abs(nums[0] - zlo) < 1e-9 and abs(nums[1] - zhi) < 1e-9:
                return v
        raise KeyError(f"zone [{zlo},{zhi}) not in FF JSON")

    for zlo, zhi in Z_ZONES + [(2.0, 3.5)]:
        fz = ff_zone(zlo, zhi)
        dx_ff = float(fz["dX"])
        dx_pack = pack_dX_zone(zlo, zhi)
        row = dict(zone=f"[{zlo},{zhi})" if (zlo, zhi) != (2.0, 3.5) else "integrated",
                   dX_ff_search=dx_ff, dX_pack=dx_pack,
                   coverage_ratio=round(dx_ff / dx_pack, 4))
        counts = np.asarray(fz["counts_per_Nbin"], float)
        for thr in THRESH:
            csel = n_edges[:-1] >= thr - 1e-9
            n_det = float(counts[csel[:len(counts)]].sum())
            alpha = 1.0 / float(R0_full[f"{thr:.1f}"] if f"{thr:.1f}" in R0_full
                                else R0_full[str(thr)])
            mu, o, mmig = fold_zone(thr, zlo, zhi)
            row[f"ge{thr}"] = dict(
                ff_raw_dndx_ffdX=round(n_det / dx_ff, 6),
                ff_raw_dndx_packdX=round(n_det / dx_pack, 6),
                ff_alpha_corr_dndx_ffdX=round(alpha * n_det / dx_ff, 6),
                ff_alpha_corr_dndx_packdX=round(alpha * n_det / dx_pack, 6),
                alpha=round(alpha, 6),
                fold_mu_counts=round(mu, 2), fold_obs_counts=int(o),
                fold_mu_dndx_packdX=round(mu / dx_pack, 6),
                fold_obs_dndx_packdX=round(o / dx_pack, 6),
                fold_migration_counts=mmig)
        zones_out.append(row)

    def _git():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=_REPO).decode().strip()
        except Exception:
            return "unknown"

    out = dict(
        schema="lane_l_matched_alpha_vs_hbi/v1",
        code_commit=_git(),
        inputs=dict(ff=FF_JSON, ff_schema=ff["schema"],
                    ff_code_commit=ff.get("code_commit"),
                    pack=PACK, vs_hbi=VS_HBI,
                    ff_jobs="57261601 (+57250729 byte-identical cross-commit receipt)"),
        population=dict(
            matched_universe_n_targetids=ff["universe"]["n_targetids"],
            n_sightlines_ff_matched=ff["universe"]["n_sightlines_ff_matched"],
            note=("matched = pack build_pathlength(no_bal=True) whitelist "
                  "(374,177); the FF harness additionally requires TARGETID "
                  "in hcd_truth_cat, giving z-dependent coverage (dX ratio "
                  "0.48-0.90) — BOTH n definitions and BOTH dX conventions "
                  "recorded per zone; neither picked. The committed "
                  "calccddf_vs_hbi n_sightlines=921,027 is a ROW count, "
                  "not comparable.")),
        estimand=dict(
            ff="PLUG-IN posterior-weighted detected counts (sampling point; "
               "NOT a credible interval); single-absorber restriction ~-7% "
               "applies to the FF route",
            alpha=("alpha = 1/R0 TRANSFERRED from the committed FULL-"
                   "population artifact (matched-population R0 does not "
                   "exist: the FF driver writes no truth counts); on-mock "
                   "alpha=1/R0 is a TAUTOLOGY — claim is structural "
                   "(z-only scalar cannot reproduce an N-resolved "
                   "response), never 'HBI closes better on-mock'"),
            fold="RATIFIED P1 empirical-kernel fold (refold machinery, "
                 "clamp=both) + mu_fp + z-resolved below-floor net "
                 "migration (frozen selection replicated, gated on the "
                 "committed G1/G2/G3 totals)"),
        caveats=dict(
            z_resolved_warning=("the FF estimator manufactures a residual "
                                "z-tilt (R0(>=20.3) 0.908->1.052->1.189); "
                                "report z-marginalised unless controlled"),
            wp2prime_z_tilt="family-universal -0.150/-0.153 per unit z",
            matching_rule="plug-in point vs posterior point are different "
                          "estimands",
            zone_grid=("Z1-Z4 [2.0,2.4/2.6/3.0/3.5) — the FROZEN matched-FF "
                       "aggregation grid; the FINAL Paper-1 low-z bins are "
                       "B1-B5 [2.15,2.35/2.56/2.96,3.40,3.80): re-cutting "
                       "the FF arm to B-bins needs a driver re-run (~3.5 h) "
                       "— recorded, not silently done")),
        reference_hbi_forward=dict(
            n_sightlines=hbi_fwd["n_sightlines"],
            R0_dndx=hbi_fwd["R0_dndx"], dndx_est=hbi_fwd.get("dndx_est"),
            note="integrated-only committed HBI forward arm (pack universe)"),
        alpha_perz_advisory=(splits and {k: v.get("R0_dndx")
                                         for k, v in splits.get("splits", {}).items()}),
        zones=zones_out)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[lane-l] wrote {OUT}")
    for r in zones_out:
        g = r["ge20.3"]
        print(f"  {r['zone']:12s} ff_raw(ffdX)={g['ff_raw_dndx_ffdX']:.4f} "
              f"alpha(ffdX)={g['ff_alpha_corr_dndx_ffdX']:.4f} "
              f"fold_mu(packdX)={g['fold_mu_dndx_packdX']:.4f} "
              f"fold_obs(packdX)={g['fold_obs_dndx_packdX']:.4f}")


if __name__ == "__main__":
    main()
