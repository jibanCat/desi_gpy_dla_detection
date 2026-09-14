#!/usr/bin/env python
"""PI inspection packet (2026-09-14) — READ-ONLY deterministic read-outs.

Nothing here fits, refits, retunes or reopens any frozen object.  Every number
is a percentile of STORED posterior draws under the frozen reductions
(validation/real_c1/reduce_truthfree.py and the paper's read-only
paper_figures/hbi_reduction.py), or a verbatim copy of a stored JSON field.

Outputs a single JSON blob consumed by the figure/section writers.
"""
from __future__ import annotations
import glob, hashlib, json, os, sys
import numpy as np

sys.path.insert(0, "/home/mfho/wt_abs_diag_2026-09")
from validation.real_c1 import reduce_truthfree as RT          # frozen reduction
from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior   # frozen reduction
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import PAPER1_LOWZ_BINS, _overlap_w
sys.path.insert(0, "/home/mfho/Latex/gp_dla_desi_y3/paper_figures")
import hbi_reduction as HR                                      # READ-ONLY

REAL = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/real_c1"
FINAL = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/final/runs"
REAL_PACK = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/real_c1_inputs/C1_pack.npz"
OLD_DIR = "/nfs/turbo/lsa-cavestru/mfho/paper1_science_handoff/LOWZ_CLEAN_C1_DURABLE_CANDIDATE_2026-09-11"
OLD_DRAWS = f"{OLD_DIR}/inference/POOLED_ln_clean_C1_fdraws.npz"
OLD_SUMMARY = f"{OLD_DIR}/inference/POOLED_ln_clean_C1.json"
OLD_REDUCED = f"{OLD_DIR}/reductions/reduced_CLEAN_C1_pooled.json"
OLD_PACK = "/home/mfho/lowz_clean_work_2026-09-10/opusC/pack/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.npz"
OLD_C1REF = "/home/mfho/lowz_clean_work_2026-09-12/posterior_campaign/out/C1_POOL_REFERENCE.json"

SOURCES = {}


def sha8(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    d = h.hexdigest()
    SOURCES[p] = d[:8]
    return d[:8]


def q(a, ps=(2.5, 16, 50, 84, 97.5)):
    return [float(x) for x in np.percentile(np.asarray(a), list(ps))]


# --------------------------------------------------------------- packs/draws
def load_pack_arrays(p):
    d = np.load(p, allow_pickle=True)
    return dict(ntrue=np.asarray(d["ntrue_edges"], float),
                zf=np.asarray(d["zf_edges"], float),
                zc=np.asarray(d["zc_edges"], float),
                kz=np.asarray(d["kz_to_K"]).astype(int),
                nhat=np.asarray(d["nhat_edges"], float),
                dX_k=np.asarray(d["dX"], float).sum(axis=1),
                counts=np.asarray(d["counts"], float))


def pooled_new_f():
    fs = []
    for p in sorted(glob.glob(os.path.join(REAL, "runs", "RUN_REALC1_*_fdraws.npz"))):
        sha8(p)
        fs.append(np.load(p)["f"])
    return np.concatenate(fs, axis=0)


# -------------------------------------------------------- generic reductions
def thr_weight(ntrue, nhat0, thr):
    reported = 0.5 * (ntrue[:-1] + ntrue[1:]) >= nhat0 - 1e-9
    return np.where(reported, np.clip(ntrue[1:] - np.maximum(ntrue[:-1], thr), 0.0, None), 0.0)


def dndx_per_bin(f, u, dX_k):
    """per-latent-bin contribution to dN/dX(>=thr): (D, B)."""
    return np.einsum("dbk,k->db", f, dX_k) * u[None, :] / dX_k.sum()


def rep_bins(f, ntrue, dX_k):
    """frozen 0.2-dex reporting bins (dN/dX per bin), VERBATIM RT.reporting_bins_0p2dex."""
    dN = np.diff(ntrue)
    out = []
    for e0, e1 in zip(RT.REDGES[:-1], RT.REDGES[1:]):
        m = (ntrue[:-1] >= e0 - 1e-9) & (ntrue[1:] <= e1 + 1e-9)
        if not m.any():
            continue
        dr = ((f[:, m, :] * dN[None, m, None]).sum(axis=1) * dX_k[None, :]).sum(axis=1) / dX_k.sum()
        out.append(dict(bin=[round(e0, 1), round(e1, 1)], q=q(dr)))
    return out


def zbins(f, ntrue, zf, dX_k, nhat0, thr):
    u = thr_weight(ntrue, nhat0, thr)
    per_k = np.einsum("dbk,b->dk", f, u)
    out = []
    for name, lo, hi in PAPER1_LOWZ_BINS:
        w = _overlap_w(zf, dX_k, lo, hi)
        if w.sum() <= 0:
            out.append(dict(bin=name, z=[lo, hi], available=False)); continue
        pd = (per_k * w[None, :]).sum(axis=1) / w.sum()
        out.append(dict(bin=name, z=[lo, hi], available=True, dX=float(w.sum()),
                        coverage=float(np.clip(min(hi, zf[-1]) - max(lo, zf[0]), 0, None) / (hi - lo)),
                        n_cells_with_dX=int((w > 0).sum()), q=q(pd)))
    return out


def omega_allz(f, ntrue, zf, dX_k):
    P = HR.Posterior.__new__(HR.Posterior)
    P.f, P.n_edges, P.z_edges, P.dX = f, ntrue, zf, dX_k
    ow = P._omega_weight(*HR.OMEGA_NHI); zw = P._z_weight(*HR.LOWZ_SUPPORT)
    post = float(HR.OMEGA_PREFACTOR_CM2) * np.einsum("dbk,b,k->d", f, ow, zw) / zw.sum()
    per_bin = (float(HR.OMEGA_PREFACTOR_CM2) * np.einsum("dbk,k->db", f, zw) * ow[None, :] / zw.sum())
    return q(post), ow, per_bin


def hr_dndx(f, ntrue, zf, dX_k, thr, z_lo=None, z_hi=None):
    """the OLD package's own reduction path (paper hbi_reduction), for cross-checks."""
    P = HR.Posterior.__new__(HR.Posterior)
    P.f, P.n_edges, P.z_edges, P.dX = f, ntrue, zf, dX_k
    return P.dndx(thr, z_lo, z_hi)["quantiles"]


def main():
    out = {}
    npk = load_pack_arrays(REAL_PACK); sha8(REAL_PACK)
    opk = load_pack_arrays(OLD_PACK); sha8(OLD_PACK)

    # ---- grid / data identity between the two packages -------------------
    out["grid_identity"] = dict(
        ntrue_equal=bool(np.array_equal(npk["ntrue"], opk["ntrue"])),
        zf_equal=bool(np.array_equal(npk["zf"], opk["zf"])),
        zc_equal=bool(np.array_equal(npk["zc"], opk["zc"])),
        nhat_equal=bool(np.array_equal(npk["nhat"], opk["nhat"])),
        dX_max_absdiff=float(np.abs(npk["dX_k"] - opk["dX_k"]).max()),
        counts_equal=bool(np.array_equal(npk["counts"], opk["counts"])),
        counts_total=float(npk["counts"].sum()),
        zf_edges=[float(x) for x in npk["zf"]],
        dX_k=[float(x) for x in npk["dX_k"]],
        ntrue_edges=[float(x) for x in npk["ntrue"]],
        nhat_min=float(npk["nhat"][0]),
        lowz_support=[float(x) for x in HR.LOWZ_SUPPORT],
        omega_nhi=[float(x) for x in HR.OMEGA_NHI],
        h_reporting=float(HR.H_REPORTING))

    # ---- OLD: reproduce the package's own headline numbers ---------------
    old = np.load(OLD_DRAWS); sha8(OLD_DRAWS); sha8(OLD_SUMMARY)
    fo = np.asarray(old["f"], float)
    old_red = json.load(open(OLD_REDUCED)); sha8(OLD_REDUCED)
    sha8(OLD_C1REF)
    repro = {}
    for key, thr in (("20p0", 20.0), ("20p3", 20.3)):
        mine = hr_dndx(fo, opk["ntrue"], opk["zf"], opk["dX_k"], thr)
        ref = old_red["quantities"][f"dndx_{key}_allz"]
        repro[f"dndx_{key}_allz"] = dict(stored=ref, recomputed=mine,
                                         max_rel=float(max(abs(a / b - 1) for a, b in zip(mine, ref))))
    Po = HR.Posterior.__new__(HR.Posterior)
    Po.f, Po.n_edges, Po.z_edges, Po.dX = fo, opk["ntrue"], opk["zf"], opk["dX_k"]
    om_o = Po.omega(*HR.OMEGA_NHI)["quantiles"]
    ref = old_red["quantities"]["omega_20p3_21p6_allz"]
    repro["omega_20p3_21p6_allz"] = dict(stored=ref, recomputed=om_o,
                                         max_rel=float(max(abs(a / b - 1) for a, b in zip(om_o, ref))))
    for name, lo, hi in PAPER1_LOWZ_BINS:
        for key, thr in (("20p0", 20.0), ("20p3", 20.3)):
            k = f"dndx_{key}_{name}"
            if k not in old_red["quantities"]:
                continue
            mine = hr_dndx(fo, opk["ntrue"], opk["zf"], opk["dX_k"], thr, lo, hi)
            ref = old_red["quantities"][k]
            repro[k] = dict(stored=ref, recomputed=mine,
                            max_rel=float(max(abs(a / b - 1) for a, b in zip(mine, ref))))
    out["old_reproduction_check"] = dict(
        n_draws=int(fo.shape[0]), worst_max_rel=float(max(v["max_rel"] for v in repro.values())),
        detail=repro)
    out["old_quantities"] = old_red["quantities"]
    out["old_cddf_bins"] = old_red["cddf_bins"]

    # ---- NEW pooled draws -------------------------------------------------
    fn = pooled_new_f()
    pooled = json.load(open(os.path.join(REAL, "REAL_C1_POOLED.json")))
    sha8(os.path.join(REAL, "REAL_C1_POOLED.json"))
    out["new_n_draws"] = int(fn.shape[0])
    out["new_pooled_stored"] = pooled["pools"]["all"]
    out["new_per_run"] = pooled["per_run"]
    out["new_spreads"] = pooled["spreads"]
    out["new_health"] = pooled["health"]

    # cross-check: the pooled concatenation reproduces the stored pooled read-out
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    pkobj = load_pack(REAL_PACK)
    red = reduce_f_posterior(fn, pkobj)
    thr_new = RT.thresholds_allz(fn, pkobj, red=red)
    chk = {}
    for k in ("ge20.0", "ge20.3"):
        a = thr_new[k]["post_p16_50_84"]; b = pooled["pools"]["all"]["thresholds_allz"][k]["post_p16_50_84"]
        chk[k] = float(max(abs(x / y - 1) for x, y in zip(a, b)))
    # and the paper's own reduction applied to the NEW draws (same function on both sides)
    for k, thr in (("ge20.0", 20.0), ("ge20.3", 20.3)):
        hr = hr_dndx(fn, npk["ntrue"], npk["zf"], npk["dX_k"], thr)
        b = pooled["pools"]["all"]["thresholds_allz"][k]
        ref5 = [b["post_p2p5_97p5"][0]] + b["post_p16_50_84"] + [b["post_p2p5_97p5"][1]]
        chk[k + "_vs_paper_reduction"] = float(max(abs(x / y - 1) for x, y in zip(hr, ref5)))
    out["new_reproduction_check"] = chk

    # ---- SEC 2/3: 0.2-dex bins, both models -------------------------------
    out["new_rep_bins"] = rep_bins(fn, npk["ntrue"], npk["dX_k"])
    # OLD, same frozen reduction, on the old draws (identical grid => directly comparable)
    out["old_rep_bins"] = rep_bins(fo, opk["ntrue"], opk["dX_k"])

    # ---- SEC 2/4: Paper-1 z bins, both models -----------------------------
    out["new_zbins"] = {f"ge{t}": zbins(fn, npk["ntrue"], npk["zf"], npk["dX_k"], npk["nhat"][0], t)
                        for t in (20.0, 20.3)}
    out["old_zbins"] = {f"ge{t}": zbins(fo, opk["ntrue"], opk["zf"], opk["dX_k"], opk["nhat"][0], t)
                        for t in (20.0, 20.3)}

    # ---- SEC 3: per-bin contributions to the thresholds and to Omega ------
    contrib = {}
    for t in (20.0, 20.3):
        u = thr_weight(npk["ntrue"], npk["nhat"][0], t)
        pb = dndx_per_bin(fn, u, npk["dX_k"])                     # (D, B)
        tot = pb.sum(axis=1)
        contrib[f"ge{t}"] = dict(
            u=[float(x) for x in u],
            per_bin_q=[q(pb[:, b]) for b in range(pb.shape[1])],
            frac_median=[float(np.median(pb[:, b] / tot)) for b in range(pb.shape[1])],
            total_q=q(tot))
    out["new_threshold_contrib"] = contrib
    om_q, ow, om_pb = omega_allz(fn, npk["ntrue"], npk["zf"], npk["dX_k"])
    out["new_omega"] = dict(q=om_q, weights=[float(x) for x in ow],
                            per_bin_q=[q(om_pb[:, b]) for b in range(om_pb.shape[1])],
                            frac_median=[float(np.median(om_pb[:, b] / om_pb.sum(axis=1)))
                                         for b in range(om_pb.shape[1])])
    om_qo, _, om_pbo = omega_allz(fo, opk["ntrue"], opk["zf"], opk["dX_k"])
    out["old_omega"] = dict(q=om_qo)

    # ---- SEC 8: Lambda / t_K ridge ---------------------------------------
    def run_row(p, mock=False):
        j = json.load(open(p)); sha8(p)
        d = j.get("diagnostics") or {}
        lc = j.get("lam_cut") or d.get("lam_cut") or {}
        fpt = j.get("fp_totals") or d.get("fp_by_block") or {}
        th = (j.get("estimands") or {}).get("thresholds_allz") or j.get("thresholds")
        med = {}
        for k in ("ge20.0", "ge20.3"):
            tt = th[k]; pp = tt.get("post_p16_50_84") or tt.get("p16_50_84")
            med[k] = dict(median=float(pp[1]), hw=float(0.5 * (pp[2] - pp[0])))
        tp = j.get("t_posterior") or {}
        tmean = tp.get("mean") or d.get("t_post_mean")
        pm = j.get("predictive_marginals") or d.get("predictive_marginals") or {}
        return dict(file=os.path.basename(p), seed=(j.get("run_config") or {}).get("seed"),
                    j=lc.get("j"), lam=lc.get("lam_fixed"), t_mean=tmean,
                    t_sd=tp.get("sd"), thresholds=med,
                    mu_fp_total=fpt.get("mu_fp_total_p16_50_84"),
                    mu_fp_block=fpt.get("mu_fp_block_p16_50_84"),
                    mu_fp_nhat_group=fpt.get("mu_fp_nhat_group_p16_50_84"),
                    counts_block=fpt.get("counts_block"),
                    counts_nhat_group=fpt.get("counts_nhat_group"),
                    fp_lam_total_over_naive=(j.get("fp_totals") or {}).get("fp_lam_total_over_naive")
                    or d.get("fp_lam_total_over_naive"),
                    predictive_fp_share=pm.get("predictive_fp_share") or d.get("predictive_fp_share"),
                    predictive_total_ratio=pm.get("predictive_total_ratio") or d.get("predictive_total_ratio"),
                    omega=((j.get("estimands") or {}).get("omega_20p3_21p6_allz") or {}).get("post_p16_50_84"))
    out["real_runs"] = [run_row(p) for p in sorted(glob.glob(os.path.join(REAL, "runs", "RUN_REALC1_*.json")))]
    out["mock_2lpt0_J8"] = [run_row(p, True) for p in sorted(glob.glob(os.path.join(
        FINAL, "B-phi2lpt-C1nsadd-M1CUTJ8", "RUN_*_2lpt0_s20260811_J8j*.json")))]

    # ---- SEC 4: ORACLE F1 mock transfer residuals -------------------------
    orac = {}
    for fam in ("2lpt0", "london0", "saclay0"):
        for seed in (20260811, 20260812):
            p = os.path.join(FINAL, "B-phi2lpt-C1nsadd",
                             f"RUN_B-phi2lpt-C1nsadd_{fam}_s{seed}.json")
            if not os.path.exists(p):
                continue
            j = json.load(open(p)); sha8(p)
            pz = j["perz_recovery"]["estimand"]
            orac[f"{fam}_s{seed}"] = {t: [dict(bin=b["bin"], z=b.get("z"),
                                               median_bias_pct=b.get("median_bias_pct"),
                                               truth_in_68=b.get("truth_in_68"),
                                               available=b.get("available"))
                                          for b in pz[t]["paper1_bins"]]
                                      for t in ("ge20.0", "ge20.3")}
    out["oracle_f1_zbins"] = orac
    m1 = {}
    for fam in ("2lpt0", "london0", "saclay0"):
        rows = []
        for p in sorted(glob.glob(os.path.join(FINAL, "B-phi2lpt-C1nsadd-M1CUTJ8",
                                               f"RUN_*_{fam}_s20260811_J8j*.json"))):
            j = json.load(open(p)); sha8(p)
            pz = j["perz_recovery"]["estimand"]
            rows.append({t: [b.get("median_bias_pct") for b in pz[t]["paper1_bins"]]
                         for t in ("ge20.0", "ge20.3")})
        m1[fam] = {t: [float(np.mean([r[t][i] for r in rows])) for i in range(5)]
                   for t in ("ge20.0", "ge20.3")}
        m1[fam]["n_j"] = len(rows)
    out["m1cut_J8_zbins_mean_over_j"] = m1

    # ---- FP attribution, old side ----------------------------------------
    oldfp = {}
    for p in sorted(glob.glob(os.path.join(OLD_DIR, "inference", "REAL_ln_*.json"))):
        if "_ppcrun" in p:
            continue
        j = json.load(open(p))
        g = (j.get("guards") or {}).get("G_A_real_mode") or {}
        d = j.get("diagnostics") or {}
        if not g:
            continue
        oldfp[os.path.basename(p)] = dict(fp_share=g.get("fp_share"),
                                          predictive_level=g.get("predictive_level"),
                                          fp_lam_total_over_naive=d.get("fp_lam_total_over_naive"),
                                          t_post_mean=d.get("t_post_mean"))
        sha8(p)
    out["old_fp_by_seed"] = oldfp
    out["old_pool_members"] = json.load(open(OLD_SUMMARY)).get("inputs")
    out["sources"] = SOURCES
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
    print("wrote", sys.argv[1])
    print("grid_identity", json.dumps({k: v for k, v in out["grid_identity"].items()
                                       if not isinstance(v, list)}))
    print("old reproduction worst rel", out["old_reproduction_check"]["worst_max_rel"])
    print("new reproduction check", out["new_reproduction_check"])


if __name__ == "__main__":
    main()
