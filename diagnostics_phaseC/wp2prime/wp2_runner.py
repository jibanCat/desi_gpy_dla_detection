#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WP-2′ — held-out family fold-transfer + candidate-floor gate.

Implements docs/WP2PRIME_PROTOCOL.md (v1, PI-approved 2026-08-12) with the
committed p1_refold discipline: `--phase predict` writes the FULL frozen
prediction (no observed count enters the file); `--phase close` refuses to
run without the committed prediction, re-derives it in memory, verifies
equality, evaluates the pre-registered statistics ONCE and refuses reruns;
`--phase decide` applies the pre-registered decision rule to the committed
closures of BOTH mocks.

Frozen scientific content (all fixed BEFORE any close):
  * operator: the ratified 2LPT-0 natural-pair (C,K) kernel + 2LPT-0
    kernel-event set (SHARED — that is the transport claim); per-mock pack
    (phaseB, winlya_only) supplies truth allocation, dX, FP counts,
    t_sigma and observed counts; per-mock M(<19.5) from the committed
    chain-bridge definition on the mock's OWN caches (totals recorded, not
    gated — the 4088/144/0 gate is a 2LPT-0 record);
  * domains (0.2-dex paired reported grain): control [20.3,21.5);
    candidate [20.0,21.5); aligned [20.1,21.5);
  * per-z zones at N-hat >= 20.3, fold-grid aligned: Z1 [2.0,2.4),
    Z2 [2.4,2.6), Z3 [2.6,3.0), Z4 [3.0,3.5);
  * per-SNR diagnostic of observed [19.9,20.1) (labeled DIAGNOSTIC);
  * Layer-B statistic, three variants, all defined here:
      - frozen_construction  : survey Poisson + FP-count Poisson
      - total_with_CKM       : + Sigma_CKM (2LPT-0 operator covariance,
        rescaled per group by s_i = G_pred_mock_i / G_pred_2lpt0_i,
        proportional-operator-error model, declared here)
      - total_with_CKM_and_t : + per-K lognormal FP-transfer factors,
        eps_K ~ N(0, t_sigma[K]^2) (pack-frozen widths) — THE GATE VARIANT
    frozen seeds/sizes from gate_covariance (41001/43001, B=2000/2000);
  * pre-registered expectation: cross-family FP over-supply ~1.45x
    (London) / ~1.31x (Saclay), Phase-A record — direction disclosed, the
    mean is NOT shifted;
  * decision rule (--phase decide):
      1. candidate PASS on BOTH mocks AND control PASS on BOTH
         => floor 20.0 primary;
      2. any candidate FAIL, control PASS on both => floor 20.3 primary;
      3. control FAIL anywhere => HARD STOP, PI checkpoint.
    control PASS  := paired chi2/dof <= 3 on [20.3,21.5) AND
                     Layer-B (gate variant) p >= 0.01;
    candidate PASS:= Holm (alpha 0.01) over the 4 upper-tail chi2 p-values
                     {2 mocks} x {candidate, aligned} rejects NONE.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_P1 = os.path.join(_REPO, "diagnostics_phaseC", "p1_completeness")
sys.path.insert(0, os.path.join(_REPO, "injection"))
sys.path.insert(0, _P1)
sys.path.insert(0, _REPO)

PACKS = {
    "london0": ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                "phaseB_packs/"
                "modelA_pack_london0_winlya_only_pad19p0_molly172_bw0p2.npz"),
    "saclay0": ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                "phaseB_packs/"
                "modelA_pack_saclay0_winlya_only_pad19p0_molly172_bw0p2.npz"),
}
# the frozen 2LPT-0 prediction record (committed 19f60ab) — rescale anchor
G_PRED_2LPT0 = np.array([38158.879, 22997.897, 6196.036])
EXPECTED_FP_OVERSUPPLY = {"london0": 1.45, "saclay0": 1.31}

Z_ZONES = [(2.0, 2.4), (2.4, 2.6), (2.6, 3.0), (3.0, 3.5)]
DOMAINS = {"control_20p3": (20.3, 21.5), "candidate_20p0": (20.0, 21.5),
           "aligned_20p1": (20.1, 21.5)}


def wp2_load_migration(cache195_path, cache172_path, nhat_edges):
    """The committed p1_migration net definition on per-mock caches
    (identical selection/attribution to p1_refold_fold.load_migration;
    totals RECORDED per mock rather than gated against the 2LPT-0 record)."""
    d = np.load(cache172_path)
    d5 = np.load(cache195_path)
    FLOOR = 19.5
    P_DLA_MIN = 0.99
    sel = ((d["cat_P_DLA"] > P_DLA_MIN) & d["cat_good"]
           & (d["cat_S2N"] > 2.0) & (d["cat_NHI"] > FLOOR))
    sel5 = ((d5["cat_P_DLA"] > P_DLA_MIN) & d5["cat_good"]
            & (d5["cat_S2N"] > 2.0) & (d5["cat_NHI"] > FLOOR))
    tp195 = set(zip(d5["cat_TARGETID"][d5["cat_is_TP"] & sel5].tolist(),
                    np.round(d5["cat_Z_DLA"][d5["cat_is_TP"] & sel5],
                             6).tolist()))
    keys = list(zip(d["cat_TARGETID"].tolist(),
                    np.round(d["cat_Z_DLA"], 6).tolist()))
    in195 = np.array([k in tp195 for k in keys])
    net = sel & d["cat_is_TP"] & (d["cat_NHI_TRUE"] < FLOOR) & ~in195
    nhat = d["cat_NHI"][net]
    ne = np.asarray(nhat_edges, float)
    ci = np.digitize(nhat, ne) - 1
    in_grid = (ci >= 0) & (ci < len(ne) - 1) & (nhat < ne[-1])
    M_c = np.bincount(ci[in_grid], minlength=len(ne) - 1).astype(float)
    return dict(M_c=M_c, NHAT=np.asarray(nhat, float),
                Z=np.asarray(d["cat_Z_DLA"][net], float),
                S2N=np.asarray(d["cat_S2N"][net], float),
                n_net_total=int(net.sum()),
                n_out_of_grid=int(np.sum(~in_grid)))


def assemble(mock, cachedir):
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    import p1_refold_fold as F
    from p1_ckm_cov import load_p1_ckm_cov

    pk = load_pack(PACKS[mock])
    fold = F.build_fold(pk)
    E, truth, sparse, art, cache = F.load_kernel_events()   # 2LPT-0 SHARED
    K_P1, kinfo = F.build_p1_kernel(E, fold, sparse)
    mu_sig = F.mu_sig_p1(K_P1, fold)                        # (C,Kf,S)
    mu_sig_c = F.c_marginal(mu_sig)
    fp_ck = fold["mu_fp"]
    fp_c = F.c_marginal(fp_ck)
    mig = wp2_load_migration(
        os.path.join(cachedir, f"wp2_{mock}_completeness_cache.npz"),
        os.path.join(cachedir, f"wp2_{mock}_completeness_cache_172.npz"),
        fold["nhat_edges"])
    M_c = mig["M_c"]
    mu_c = mu_sig_c + M_c + fp_c
    A = fold["A"]
    G_pred = A @ mu_c
    ckm = load_p1_ckm_cov()
    Sigma_G0 = np.asarray(ckm["Sigma_G"], float)
    s = G_pred / G_PRED_2LPT0
    Sigma_G = Sigma_G0 * np.outer(s, s)
    return dict(pk=pk, fold=fold, mu_sig=mu_sig, mu_sig_c=mu_sig_c,
                fp_ck=fp_ck, fp_c=fp_c, mig=mig, M_c=M_c, mu_c=mu_c,
                A=A, G_pred=G_pred, Sigma_G=Sigma_G, Sigma_scale=s)


def paired_mu(fold, mu_c, lo, hi):
    ne = fold["nhat_edges"]
    pairs = []
    x = lo
    while x + 0.2 <= hi + 1e-9:
        j0 = int(np.searchsorted(ne, x + 1e-9) - 1)
        pairs.append(dict(lo=round(x, 2), hi=round(x + 0.2, 2),
                          mu=float(mu_c[j0] + mu_c[j0 + 1]),
                          j=(j0, j0 + 2)))
        x = round(x + 0.2, 10)
    return pairs


def zone_mu(a):
    fold = a["fold"]
    ne = fold["nhat_edges"]
    cm = ne[:-1] >= 20.3 - 1e-9
    kcent = 2.05 + 0.1 * np.arange(a["mu_sig"].shape[1])
    zones = []
    for zlo, zhi in Z_ZONES:
        km = (kcent >= zlo) & (kcent < zhi)
        mu = float(a["mu_sig"][cm][:, km, :].sum()
                   + a["fp_ck"][cm][:, km, :].sum())
        mM = float(np.sum((a["mig"]["NHAT"] >= 20.3)
                          & (a["mig"]["Z"] >= zlo) & (a["mig"]["Z"] < zhi)))
        zones.append(dict(z=f"[{zlo},{zhi})", mu=mu + mM, mu_sig_fp=mu,
                          mu_M=mM))
    return zones


def snr_diag_mu(a):
    fold = a["fold"]
    ne = fold["nhat_edges"]
    sl = (ne[:-1] >= 19.9 - 1e-9) & (ne[1:] <= 20.1 + 1e-9)
    mu_s = (a["mu_sig"][sl].sum(axis=(0, 1))
            + a["fp_ck"][sl].sum(axis=(0, 1)))
    snr_edges = np.asarray(a["pk"].molly_snr_edges, float)
    mreg = (a["mig"]["NHAT"] >= 19.9) & (a["mig"]["NHAT"] < 20.1)
    si = np.digitize(a["mig"]["S2N"][mreg], snr_edges) - 1
    M_s = np.bincount(np.clip(si, 0, len(mu_s) - 1),
                      minlength=len(mu_s)).astype(float)
    return [dict(stratum=int(i),
                 snr=f"[{snr_edges[i]},{snr_edges[i+1]})",
                 mu_sig_fp=float(mu_s[i]), mu_M=float(M_s[i]))
            for i in range(len(mu_s))]


def pred_path(outdir, mock):
    return os.path.join(outdir, f"wp2_{mock}_prediction.json")


def close_path(outdir, mock):
    return os.path.join(outdir, f"wp2_{mock}_closure.json")


def phase_predict(mock, cachedir, outdir):
    if os.path.exists(close_path(outdir, mock)):
        raise SystemExit("REFUSED: closure exists — predictions are never "
                         "regenerated after a close (order rule).")
    a = assemble(mock, cachedir)
    t_sigma = np.asarray(a["pk"].t_sigma, float).tolist() \
        if hasattr(a["pk"], "t_sigma") else None
    out = {
        "schema": "wp2prime_prediction/v1",
        "protocol": "docs/WP2PRIME_PROTOCOL.md v1 (PI-approved 2026-08-12)",
        "date": time.strftime("%Y-%m-%d"),
        "mock": mock, "pack": PACKS[mock],
        "operator": "p1_natpair_ck/v1 (2LPT-0, ratified, READ-ONLY) + "
                    "2LPT-0 kernel events; truth support >= 19.5 + "
                    "per-mock M(<19.5)",
        "note": "CALIBRATION-SIDE PREDICTION; no observed count in this "
                "file.",
        "migration": dict(n_net_total=a["mig"]["n_net_total"],
                          n_out_of_grid=a["mig"]["n_out_of_grid"],
                          group_totals=[float(x) for x in
                                        (a["A"] @ a["M_c"])]),
        "per_bin_mu": dict(nhat_edges=a["fold"]["nhat_edges"].tolist(),
                           mu_signal_CK=a["mu_sig_c"].tolist(),
                           mu_migration_M=a["M_c"].tolist(),
                           mu_fp=a["fp_c"].tolist(),
                           mu_total=a["mu_c"].tolist()),
        "groups_mu_total": a["G_pred"].tolist(),
        "Sigma_CKM_rescaled": a["Sigma_G"].tolist(),
        "Sigma_scale_s": a["Sigma_scale"].tolist(),
        "t_sigma": t_sigma,
        "expected_fp_oversupply_direction":
            EXPECTED_FP_OVERSUPPLY[mock],
        "domains": {name: paired_mu(a["fold"], a["mu_c"], lo, hi)
                    for name, (lo, hi) in DOMAINS.items()},
        "z_zones_ge20p3": zone_mu(a),
        "snr_diagnostic_19p9_20p1": snr_diag_mu(a),
        "decision_rule": "see module docstring / protocol doc (frozen)",
    }
    os.makedirs(outdir, exist_ok=True)
    with open(pred_path(outdir, mock), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"prediction committed-side file written: "
          f"{pred_path(outdir, mock)}")


def phase_close(mock, cachedir, outdir):
    from CDDF_analysis.hbi_mcmc import gate_covariance as GC
    import p1_refold_fold as F
    from scipy import stats

    pp = pred_path(outdir, mock)
    cp = close_path(outdir, mock)
    if not os.path.exists(pp):
        raise SystemExit("REFUSED: prediction absent (order guard).")
    if os.path.exists(cp):
        raise SystemExit("REFUSED: closure exists — ONE read (order rule).")
    pred = json.load(open(pp))
    a = assemble(mock, cachedir)
    if not np.allclose(pred["groups_mu_total"], a["G_pred"], rtol=0,
                       atol=1e-9):
        raise SystemExit("IMPLEMENTATION-INVALID: in-memory prediction != "
                         "committed prediction")

    fold, A = a["fold"], a["A"]
    obs_cks = fold["obs_counts"]
    obs_c = F.c_marginal(obs_cks)
    G_obs = A @ obs_c
    d_obs = G_obs - a["G_pred"]

    # ---- domains: paired chi2/dof + upper-tail p ------------------------
    domains = {}
    for name, plist in pred["domains"].items():
        chis, rows = [], []
        for p in plist:
            j0, j1 = p["j"]
            o = float(obs_c[j0:j1].sum())
            z = (o - p["mu"]) / np.sqrt(p["mu"])
            chis.append(z * z)
            rows.append(dict(bin=f"[{p['lo']},{p['hi']})", mu=p["mu"],
                             obs=o, z=float(z)))
        n = len(chis)
        chi2 = float(np.sum(chis))
        domains[name] = dict(pairs=rows, chi2=chi2, dof=n,
                             chi2_dof=chi2 / n,
                             p_upper=float(stats.chi2.sf(chi2, n)))

    # ---- per-z zones ----------------------------------------------------
    ne = fold["nhat_edges"]
    cm = ne[:-1] >= 20.3 - 1e-9
    kcent = 2.05 + 0.1 * np.arange(a["mu_sig"].shape[1])
    zrows = []
    for zp, (zlo, zhi) in zip(pred["z_zones_ge20p3"], Z_ZONES):
        km = (kcent >= zlo) & (kcent < zhi)
        o = float(obs_cks[cm][:, km, :].sum())
        zres = (o - zp["mu"]) / np.sqrt(zp["mu"])
        zrows.append(dict(z=zp["z"], mu=zp["mu"], obs=o,
                          z_resid=float(zres)))

    # ---- SNR diagnostic (labeled; never a gate) -------------------------
    sl = (ne[:-1] >= 19.9 - 1e-9) & (ne[1:] <= 20.1 + 1e-9)
    obs_s = obs_cks[sl].sum(axis=(0, 1))
    srows = []
    for sp in pred["snr_diagnostic_19p9_20p1"]:
        i = sp["stratum"]
        mu = sp["mu_sig_fp"] + sp["mu_M"]
        if mu <= 0 and obs_s[i] <= 0:
            continue
        srows.append(dict(stratum=i, snr=sp["snr"], mu=mu,
                          obs=float(obs_s[i]),
                          z_resid=float((obs_s[i] - mu)
                                        / np.sqrt(max(mu, 1)))))

    # ---- Layer-B, three pre-registered variants -------------------------
    mu_sig_c, M_c, mu_c = a["mu_sig_c"], a["M_c"], a["mu_c"]
    Sigma_G = a["Sigma_G"]
    pk = a["pk"]
    n0 = np.asarray(pk.fp_counts, float)
    _, fp_fold, live3 = GC._fold_parts(pk, resp_clamp="both")
    kz_to_K = np.asarray(fold["consts"].kz_to_K)
    t_sig = np.asarray(pred["t_sigma"], float) if pred["t_sigma"] else None

    def fp_c_of(n0v, eps=None):
        fp3 = np.where(live3, fp_fold(n0v), 0.0)
        if eps is not None:
            fp3 = fp3 * np.exp(eps[kz_to_K])[None, :, None]
        return F.c_marginal(fp3)

    def draws_fn(rng, with_t):
        y_star = rng.poisson(np.clip(mu_c, 0, None))
        n0_star = rng.poisson(n0)
        eps = (rng.normal(0.0, t_sig) if (with_t and t_sig is not None)
               else None)
        return A @ y_star - A @ (mu_sig_c + M_c + fp_c_of(n0_star, eps))

    layerB = {}
    for tag, add_ckm, with_t in (("frozen_construction", False, False),
                                 ("total_with_CKM", True, False),
                                 ("total_with_CKM_and_t", True, True)):
        rng = np.random.default_rng(GC.SEED_COV)
        D = np.stack([draws_fn(rng, with_t)
                      for _ in range(GC.N_COV_DRAWS)])
        C = np.cov(D, rowvar=False)
        Sig = C + (Sigma_G if add_ckm else 0.0)
        ev = np.linalg.eigvalsh(Sig)
        cond = float(ev[-1] / max(ev[0], 1e-300))
        z_g = d_obs / np.sqrt(np.diag(Sig))
        if cond > GC.MAX_CONDITION_NUMBER:
            layerB[tag] = dict(T_obs=float(np.max(np.abs(z_g))),
                               p_value=None, fallback_1d=True,
                               condition_number=cond,
                               residual_z=z_g.tolist())
            continue
        Sinv = np.linalg.inv(Sig)
        T_obs = float(d_obs @ Sinv @ d_obs)
        rng_n = np.random.default_rng(GC.SEED_NULL)
        L = (np.linalg.cholesky(Sigma_G + 1e-12 * np.eye(3)
                                * max(Sigma_G.max(), 1e-300))
             if add_ckm else None)
        T_null = np.empty(GC.N_NULL_DRAWS)
        for r in range(GC.N_NULL_DRAWS):
            d = draws_fn(rng_n, with_t)
            if add_ckm:
                d = d - L @ rng_n.standard_normal(3)
            T_null[r] = float(d @ Sinv @ d)
        n_ex = int(np.sum(T_null >= T_obs))
        p = (1 + n_ex) / (GC.N_NULL_DRAWS + 1)
        layerB[tag] = dict(T_obs=T_obs, p_value=p,
                           p_is_bound=(n_ex == 0),
                           condition_number=cond, fallback_1d=False,
                           residual_z=z_g.tolist(),
                           covariance=[[float(v) for v in row]
                                       for row in Sig])

    gate = layerB["total_with_CKM_and_t"]
    out = {
        "schema": "wp2prime_closure/v1", "mock": mock,
        "date": time.strftime("%Y-%m-%d"),
        "groups": dict(pred=a["G_pred"].tolist(), obs=G_obs.tolist(),
                       resid=d_obs.tolist()),
        "domains": domains, "z_zones_ge20p3": zrows,
        "snr_diagnostic_19p9_20p1_LABELED_DIAGNOSTIC_ONLY": srows,
        "layerB": layerB,
        "gate_variant": "total_with_CKM_and_t",
        "layerB_gate_p": gate.get("p_value"),
        "control_chi2_dof": domains["control_20p3"]["chi2_dof"],
        "control_pass": bool(domains["control_20p3"]["chi2_dof"] <= 3.0
                             and (gate.get("p_value") or 0) >= 0.01),
        "wall_s": None,
    }
    with open(cp, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("mock", "layerB_gate_p", "control_chi2_dof",
                       "control_pass")}, indent=1))
    for name in ("candidate_20p0", "aligned_20p1"):
        d = domains[name]
        print(f"  {name}: chi2/dof {d['chi2_dof']:.3f} "
              f"p_upper {d['p_upper']:.4f}")


def phase_decide(outdir):
    mocks = ("london0", "saclay0")
    cls = {m: json.load(open(close_path(outdir, m))) for m in mocks}
    control_ok = all(cls[m]["control_pass"] for m in mocks)
    tests = []
    for m in mocks:
        for name in ("candidate_20p0", "aligned_20p1"):
            tests.append((f"{m}:{name}",
                          cls[m]["domains"][name]["p_upper"]))
    tests.sort(key=lambda t: t[1])
    alpha = 0.01
    holm = []
    reject_any = False
    for i, (name, p) in enumerate(tests):
        thr = alpha / (len(tests) - i)
        rej = p < thr
        reject_any |= rej
        holm.append(dict(test=name, p=p, holm_threshold=thr,
                         reject=bool(rej)))
    if not control_ok:
        verdict = ("HARD STOP — certified-domain control FAILED on a "
                   "held-out family; PI checkpoint required")
        floor = None
    elif not reject_any:
        verdict = "candidate PASS on both families — floor 20.0 PRIMARY"
        floor = 20.0
    else:
        verdict = ("candidate FAIL — floor 20.3 PRIMARY; [20.0,20.3) "
                   "demoted to labeled diagnostics")
        floor = 20.3
    out = dict(schema="wp2prime_decision/v1",
               date=time.strftime("%Y-%m-%d"),
               control=({m: cls[m]["control_pass"] for m in mocks}),
               layerB_gate_p={m: cls[m]["layerB_gate_p"] for m in mocks},
               holm=holm, verdict=verdict, paper1_primary_floor=floor)
    with open(os.path.join(outdir, "wp2_decision.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=("predict", "close", "decide"))
    ap.add_argument("--mock", choices=sorted(PACKS))
    ap.add_argument("--cachedir", default="/scratch/cavestru_root/"
                    "cavestru0/mfho/wp2prime_2026_08_12")
    ap.add_argument("--outdir", default=_HERE)
    args = ap.parse_args()
    if args.phase == "decide":
        phase_decide(args.outdir)
    else:
        if not args.mock:
            raise SystemExit("--mock required for predict/close")
        (phase_predict if args.phase == "predict"
         else phase_close)(args.mock, args.cachedir, args.outdir)
