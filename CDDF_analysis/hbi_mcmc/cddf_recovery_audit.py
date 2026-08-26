#!/usr/bin/env python
"""cddf_recovery_audit.py — per-0.2-dex-bin POSTERIOR recovery of f(N) vs the pack's
own truth, all-z and per redshift slice, from SAVED validation draws (mock-only).

PI instruction 2026-08-26 (systematics-visualization checkpoint, item 1): re-verify
the per-bin CDDF recovery claim from source artifacts with a committed script.

Statistic (identical to cc_posterior_validation's `reporting_bins`, generalised to a
z slice with the validator's overlap weights): for each reporting bin [e0, e1) on the
latent (ntrue) grid and z slice [z_lo, z_hi),
    F_d = sum_k w_k sum_{b in bin} f[d, b, k] dN_b / sum_k w_k,   w_k = dX_k |cell_k ∩ slice| / |cell_k|
    T   = the same reduction of truth_f (= truth_counts / (dX_tot dN)),
    median_bias_pct = 100 (median_d F_d / T - 1); stat half-widths from the [16, 84] quantiles.
This is the POSTERIOR-RECOVERY residual — distinct from the forward truth-fold
residual (kernel_uncertainty_closure / cc_fold) that the CKPT5/8 per-bin lines quote.

Inputs: the `*_fdraws.npz` written by cc_posterior_validation --save-fdraws
(keys f, truth_f, ntrue_edges, zf_edges, dX_k). Real-data mode (`--real`) applies the
same reduction to a pooled draws file and, optionally, to a named chain of a run's
draws (the mirror configuration) — statistical half-widths and the coherent
alternative-configuration curve at the same (N, z) aggregation. Nothing is fitted,
tuned or written into any posterior product.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, re, subprocess
import numpy as np

REDGES = np.round(np.arange(19.7, 21.7 + 1e-9, 0.2), 3)


def _overlap_w(zf, dX_k, lo, hi):
    ov = np.clip(np.minimum(zf[1:], hi) - np.maximum(zf[:-1], lo), 0.0, None)
    return dX_k * ov / np.diff(zf)


def bin_recovery(f, truth, ntrue, zf, dX_k, redges=REDGES, z_lo=None, z_hi=None, truth_counts=None):
    ntrue = np.asarray(ntrue, float); zf = np.asarray(zf, float); dX_k = np.asarray(dX_k, float)
    dN = np.diff(ntrue)
    lo = zf[0] if z_lo is None else float(z_lo); hi = zf[-1] if z_hi is None else float(z_hi)
    w = _overlap_w(zf, dX_k, lo, hi)
    if w.sum() <= 0:
        return []
    rows = []
    for e0, e1 in zip(redges[:-1], redges[1:]):
        m = (ntrue[:-1] >= e0 - 1e-9) & (ntrue[1:] <= e1 + 1e-9)
        if not m.any():
            continue
        Fd = ((f[:, m, :] * dN[None, m, None]).sum(axis=1) * w[None, :]).sum(axis=1) / w.sum()
        T = float(((truth[m, :] * dN[m, None]).sum(axis=0) * w).sum() / w.sum())
        q = np.percentile(Fd, [2.5, 16, 50, 84, 97.5])
        row = dict(bin=[float(round(e0, 1)), float(round(e1, 1))], z=[lo, hi], dX=float(w.sum()), truth=T,
                   post_p2p5_16_50_84_97p5=[float(x) for x in q],
                   median_bias_pct=(100.0 * (q[2] / T - 1.0)) if T > 0 else None,
                   stat_halfwidth_pct_lo=(100.0 * (1.0 - q[1] / q[2])) if q[2] > 0 else None,
                   stat_halfwidth_pct_hi=(100.0 * (q[3] / q[2] - 1.0)) if q[2] > 0 else None,
                   truth_in_68=bool(q[1] <= T <= q[3]), truth_in_95=bool(q[0] <= T <= q[4]))
        if truth_counts is not None:
            tc = np.asarray(truth_counts, float)
            ov = np.clip(np.minimum(zf[1:], hi) - np.maximum(zf[:-1], lo), 0.0, None) / np.diff(zf)
            row["n_truth_support"] = float((tc[m, :] * ov[None, :]).sum())
        rows.append(row)
    return rows


def family_summary(rows_by_run, family_of):
    out = {}
    bins = sorted({tuple(r["bin"]) for rows in rows_by_run.values() for r in rows})
    for b in bins:
        out[b] = {}
        vals_all = []
        for fam in sorted(set(family_of.values())):
            vals = [(run, r["median_bias_pct"]) for run, rows in rows_by_run.items() if family_of[run] == fam
                    for r in rows if tuple(r["bin"]) == b and r["median_bias_pct"] is not None]
            if vals:
                v = np.array([x[1] for x in vals])
                out[b][fam] = dict(n_runs=int(len(v)), min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
                                   runs={k: float(x) for k, x in vals})
                vals_all += vals
        if vals_all:
            v = np.array([x[1] for x in vals_all]); i = int(np.argmin(v)); j = int(np.argmax(v))
            out[b]["all"] = dict(n_runs=int(len(v)), min=float(v.min()), max=float(v.max()),
                                 extreme_run=vals_all[i][0] if abs(v.min()) >= abs(v.max()) else vals_all[j][0])
    return out


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", help="validation *_fdraws.npz files (mock)")
    ap.add_argument("--run-json", nargs="*", default=[], help="matching run JSONs (for pack path + provenance)")
    ap.add_argument("--slices", default="", help="z edges, e.g. 2.0,2.5,3.0,3.5 (all-z always included)")
    ap.add_argument("--real", default=None, help="pooled real fdraws (f, ntrue_edges, zf_edges) for stat half-widths")
    ap.add_argument("--real-pack", default=None, help="real pack (dX) for --real")
    ap.add_argument("--mirror", default=None, help="a run's fdraws whose chain --mirror-chain is the alternative configuration")
    ap.add_argument("--mirror-chain", type=int, default=0)
    ap.add_argument("--mirror-chains", type=int, default=2)
    ap.add_argument("--redges", default="", help="reporting edges, e.g. 19.7,19.9,...,22.4 (default 19.7:0.2:21.7)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    edges = [float(x) for x in a.slices.split(",")] if a.slices else []
    redges = np.array([float(x) for x in a.redges.split(",")]) if a.redges else REDGES
    slices = [(None, None)] + [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "UNKNOWN"
    out = {"role": "per-0.2-dex-bin POSTERIOR recovery audit (mock validation draws) + real stat half-widths / mirror curve at the same aggregation; diagnostic; PI 2026-08-26",
           "code_commit": commit, "redges": redges.tolist(), "slices": [list(s) for s in slices], "mock": {}, "real": {}}
    json_by_stem = {os.path.basename(j)[:-5]: j for j in a.run_json}
    if a.runs:
        for s in slices:
            key = "allz" if s[0] is None else f"z{s[0]}-{s[1]}"
            rows_by_run, fam_of, prov = {}, {}, {}
            for f in a.runs:
                z = np.load(f)
                stem = os.path.basename(f).replace("_fdraws.npz", "")
                fam = re.search(r"_(2lpt0|london0|saclay0)_", stem).group(1)
                tc = None
                pack_path = None
                if stem in json_by_stem:
                    jj = json.load(open(json_by_stem[stem])); pack_path = jj.get("pack")
                    try:
                        pk = np.load(pack_path, allow_pickle=True); tc = pk["truth_counts"]
                    except Exception:
                        tc = None
                rows = bin_recovery(z["f"], z["truth_f"], z["ntrue_edges"], z["zf_edges"], z["dX_k"], redges=redges,
                                    z_lo=s[0], z_hi=s[1], truth_counts=tc)
                rows_by_run[stem] = rows; fam_of[stem] = fam
                prov[stem] = dict(fdraws=f, fdraws_sha256=_sha(f), run_json=json_by_stem.get(stem),
                                  run_json_sha256=_sha(json_by_stem[stem]) if stem in json_by_stem else None,
                                  pack=pack_path, pack_sha256=_sha(pack_path) if pack_path and os.path.exists(pack_path) else None,
                                  n_draws=int(z["f"].shape[0]))
            out["mock"][key] = dict(rows_by_run=rows_by_run, family_summary={str(k): v for k, v in family_summary(rows_by_run, fam_of).items()},
                                    provenance=prov)
    if a.real:
        R = np.load(a.real); pk = np.load(a.real_pack, allow_pickle=True); dX_k = np.asarray(pk["dX"], float).sum(axis=1)
        fake_truth = np.ones_like(R["f"][0])
        for s in slices:
            key = "allz" if s[0] is None else f"z{s[0]}-{s[1]}"
            rows = bin_recovery(R["f"], fake_truth, R["ntrue_edges"], R["zf_edges"], dX_k, redges=redges, z_lo=s[0], z_hi=s[1])
            rec = {"pooled": [{k: v for k, v in r.items() if k not in ("truth", "median_bias_pct", "truth_in_68", "truth_in_95")} for r in rows]}
            if a.mirror:
                M = np.load(a.mirror); n = M["f"].shape[0] // a.mirror_chains
                fm = M["f"][a.mirror_chain * n:(a.mirror_chain + 1) * n]
                mrows = bin_recovery(fm, fake_truth, M["ntrue_edges"], M["zf_edges"], dX_k, redges=redges, z_lo=s[0], z_hi=s[1])
                rec["mirror_over_pooled_minus1_pct"] = [100.0 * (m["post_p2p5_16_50_84_97p5"][2] / p["post_p2p5_16_50_84_97p5"][2] - 1.0)
                                                       for m, p in zip(mrows, rows)]
            out["real"][key] = rec
        out["real"]["provenance"] = dict(pooled=a.real, pooled_sha256=_sha(a.real), pack=a.real_pack, pack_sha256=_sha(a.real_pack),
                                         mirror=a.mirror, mirror_sha256=_sha(a.mirror) if a.mirror else None, mirror_chain=a.mirror_chain)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
