#!/usr/bin/env python
"""R-041 response population-dependence study (mock-only; PI ruling 2026-09-02 evening §3–§13).

Frozen specification: notes governance/MAX4_RESPONSE_POPULATION_DEPENDENCE_STUDY_2026-09-02.md (private).
Stages (run in order; every stage writes JSON/CSV under --out):
  match    full-support truth/recovery matching of the native arms (2LPT loa-124, London jura-124) down to the 19.0 latent floor
  tables   conditional response tables (dN = N_hat - N_true on detected truth) per sample x N bin x S/N group x z block x class
  reweight production-style forward-response fits (fit_forward_response, importance weights) under CDDF reweightings
  context  2LPT natives reweighted to the London context distribution (class, S/N group, z block, companion N); before/after
  unfold   population-prior stress closure: split-half calibration under beta, evaluation under alpha, EM unfold, biases
  summary  criteria of the frozen spec -> classification inputs

No real-data value is read or written: the injection tables carry only injection recovery flags and N-hat of injected profiles.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

C_KMS = 299792.458
ROOT = os.environ.get("ROOT_MAX4", "/scratch/cavestru_root/cavestru0/mfho/r041_max4_highz_2026-09")
HBI_REPO = os.environ.get("HBI_REPO", "/home/mfho/wt_hbi_validation_2026_09")
MOCKS = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks"
NATIVE = {
    "N2": dict(arm="2lpt", pop=f"{ROOT}/p1/mock_native/2lpt/population_native.csv", outputs=f"{ROOT}/p1/mock_native/2lpt/native_outputs",
               truth=f"{MOCKS}/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits", ncol="NHI", zcol="Z"),
    "NL": dict(arm="london", pop=f"{ROOT}/p1/mock_native/london/population_native.csv", outputs=f"{ROOT}/p1/mock_native/london/native_outputs",
               truth=f"{MOCKS}/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits", ncol="NHI", zcol="Z_DLA"),
}
INJ = {
    "IR": f"{ROOT}/fid_max4/analysis/analysis_fid_MAX4_per_injection.csv",
    "I2": f"{ROOT}/p1/reductions/analysis_mock_2lpt_random_MAX4_per_injection.csv",
    "IL": f"{ROOT}/p1/reductions/analysis_mock_london_random_MAX4_per_injection.csv",
    "I2c": f"{ROOT}/p1/reductions/analysis_mock_2lpt_clustered_MAX4_per_injection.csv",
}
P_MIN, SNR_MIN, DZ_REL = 0.99, 2.0, 0.01
FLOOR_LAT, XHAT_FLOOR = 19.0, 19.5
EDGES = np.round(np.arange(19.5, 22.3 + 1e-9, 0.2), 2)                     # observed / truth 0.2-dex grid
TRUTH_EDGES = np.concatenate([[FLOOR_LAT], EDGES])                          # + [19.0,19.5) truth bin
ZB_NATIVE = (2.8, 3.2, 3.5, 4.0)
ZB_EMUL = (3.8, 4.2, 4.5, 5.0)
SNRGROUP = {0: "low", 1: "low", 2: "mid", 3: "mid", 4: "high"}
CLASSES = [("close", 0.0, 1000.0), ("moderate", 1000.0, 4000.0), ("gap", 4000.0, 8000.0), ("wide", 8000.0, np.inf)]
FIT_SNR_EDGES = (2.0, 3.5, 6.5, np.inf)
TILT = 0.4


def sep_class(dv):
    for name, lo, hi in CLASSES:
        if lo <= dv < hi:
            return name
    return "wide"


def zblock(z):
    zb = ZB_EMUL if z >= 3.8 else ZB_NATIVE
    i = int(np.clip(np.searchsorted(zb, z, side="right") - 1, 0, 2))
    return i


def match(truth, rows, tol=DZ_REL):
    """One-to-one greedy closest-first matching (the multi-HCD scorer's rule). truth: list of dict(z); rows: [(z, N, P)]."""
    cand = []
    for i, t in enumerate(truth):
        for j, (z, N, P) in enumerate(rows):
            d = abs(z - t["z"]) / (1.0 + t["z"])
            if d <= tol:
                cand.append((d, i, j))
    cand.sort()
    used_t, used_r, out = set(), set(), {}
    for d, i, j in cand:
        if i in used_t or j in used_r:
            continue
        used_t.add(i); used_r.add(j); out[i] = dict(row=j, dz=d, Nhat=rows[j][1], P=rows[j][2])
    for i in range(len(truth)):
        out.setdefault(i, dict(row=None, dz=None, Nhat=None, P=None))
    return out, set(range(len(rows))) - used_r


# ----------------------------------------------------------------------------------------------------------------------- match
def stage_match(sample, out_dir):
    from astropy.io import fits
    cfg = NATIVE[sample]
    pop = {int(r["TARGETID"]): r for r in csv.DictReader(open(cfg["pop"]))}
    t = fits.open(cfg["truth"])[1].data
    tid = np.asarray(t["TARGETID"]).astype(np.int64); N = np.asarray(t[cfg["ncol"]], float)
    N = np.log10(N) if np.nanmax(N) > 100 else N
    Z = np.asarray(t[cfg["zcol"]], float)
    sel = np.isin(tid, np.fromiter(pop.keys(), dtype=np.int64))
    truth_by = {}
    for ti, n, z in zip(tid[sel], N[sel], Z[sel]):
        p = pop[int(ti)]
        if float(p["zlo"]) <= z <= float(p["zhi"]):
            truth_by.setdefault(int(ti), []).append((float(z), float(n)))
    acc = {}
    n_rows_all = 0
    for f in sorted(glob.glob(os.path.join(cfg["outputs"], "dlacat-*.fits"))):
        d = fits.open(f)[1].data
        for r in d:
            ti = int(r["TARGETID"]); p = pop.get(ti)
            if p is None:
                continue
            n_rows_all += 1
            if (float(r["P_DLA"]) > P_MIN and int(r["DLAFLAG"]) == 0 and float(r["SNR_REDSIDE"]) > SNR_MIN
                    and float(p["zlo"]) < float(r["Z_DLA"]) < float(p["zhi"])):
                acc.setdefault(ti, []).append((float(r["Z_DLA"]), float(r["NHI"]), float(r["P_DLA"])))
    rows = []; n_subfloor = 0; n_unmatched = 0; n_acc_total = 0
    for ti, p in pop.items():
        tl = sorted(truth_by.get(ti, []))
        hi = [dict(z=z, N=n) for z, n in tl if n >= FLOOR_LAT]
        lo = [dict(z=z, N=n) for z, n in tl if n < FLOOR_LAT]
        rws = acc.get(ti, []); n_acc_total += len(rws)
        m1, unused = match(hi, rws)
        rest_idx = sorted(unused); rest = [rws[j] for j in rest_idx]
        m2, unused2 = match(lo, rest)
        n_subfloor += sum(1 for v in m2.values() if v["row"] is not None); n_unmatched += len(unused2)
        m20 = sum(1 for a in hi if a["N"] >= 20.0); m19 = len(hi)
        for i, a in enumerate(hi):
            others = [(abs(C_KMS * (b["z"] - a["z"]) / (1 + a["z"])), b["N"]) for j, b in enumerate(hi) if j != i]
            dv_nn, N_nn = (min(others) if others else (np.nan, np.nan))
            oth20 = [abs(C_KMS * (b["z"] - a["z"]) / (1 + a["z"])) for j, b in enumerate(hi) if j != i and b["N"] >= 20.0]
            dv20 = min(oth20) if oth20 else np.nan
            sub20 = any(abs(C_KMS * (b["z"] - a["z"]) / (1 + a["z"])) < 4000 for j, b in enumerate(hi) if j != i and b["N"] < 20.0)
            cls = "single" if not oth20 else sep_class(dv20)
            isolated = (not oth20) and ((not others) or dv_nn >= 8000)
            wide_unb = bool(oth20) and cls == "wide" and not sub20
            mm = m1[i]; det = mm["row"] is not None
            rows.append(dict(sample=sample, TARGETID=ti, z=a["z"], logN=a["N"], stratum=int(p["stratum"]), snr=float(p["snr"]), z_qso=float(p["z_qso"]),
                             zblock=zblock(a["z"]), matched=int(det), Nhat=(mm["Nhat"] if det else np.nan), dz=(mm["dz"] if det else np.nan),
                             P=(mm["P"] if det else np.nan), n_acc=len(rws), m20=m20, m19=m19, dv_nn=dv_nn, N_nn=N_nn, dv_nn20=dv20, class_20=cls,
                             sub20_within_4000=int(sub20), isolated=int(isolated), wide_unblended=int(wide_unb)))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"matches_{sample}_native_full.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    n_ge20 = sum(1 for r in rows if r["logN"] >= 20.0); n_ge19 = len(rows)
    summ = dict(sample=sample, sightlines=len(pop), truth_ge19=n_ge19, truth_ge20=n_ge20, truth_19_20=n_ge19 - n_ge20,
                matched_ge19=sum(r["matched"] for r in rows), matched_ge20=sum(r["matched"] for r in rows if r["logN"] >= 20.0),
                matched_19_20=sum(r["matched"] for r in rows if r["logN"] < 20.0), accepted_rows=n_acc_total, catalogue_rows_in_pop=n_rows_all,
                rows_matched_to_subfloor_host=n_subfloor, rows_unmatched=n_unmatched, acceptance="P_DLA>0.99, DLAFLAG==0, SNR_REDSIDE>2, in window",
                matching="one-to-one greedy closest-first |dz|/(1+z)<=0.01; pass 1 truth>=19.0, pass 2 truth [17.2,19.0)")
    json.dump(summ, open(os.path.join(out_dir, f"matches_{sample}_summary.json"), "w"), indent=1)
    print(json.dumps(summ))
    return out_csv


# ----------------------------------------------------------------------------------------------------------------------- events
def load_events(sample, out_dir):
    """Uniform event arrays: logN, z, snr, stratum, zblock, matched, Nhat, class_20, isolated, wide_unblended, sub20, N_nn, TARGETID."""
    if sample in NATIVE:
        rows = list(csv.DictReader(open(os.path.join(out_dir, f"matches_{sample}_native_full.csv"))))
        f = lambda k, typ=float: np.array([typ(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
        ev = dict(logN=f("logN"), z=f("z"), snr=f("snr"), stratum=f("stratum", int), zblock=f("zblock", int), matched=f("matched", int).astype(bool),
                  Nhat=f("Nhat"), class_20=np.array([r["class_20"] for r in rows]), isolated=f("isolated", int).astype(bool),
                  wide_unblended=f("wide_unblended", int).astype(bool), sub20=f("sub20_within_4000", int).astype(bool), N_nn=f("N_nn"),
                  TARGETID=f("TARGETID", int), m20=f("m20", int))
    else:
        rows = list(csv.DictReader(open(INJ[sample])))
        f = lambda k, typ=float: np.array([typ(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
        det = np.array([r["detected"] == "True" for r in rows])
        ev = dict(logN=f("logN"), z=f("z_inj"), snr=f("snr"), stratum=f("stratum", int), matched=det, Nhat=f("nhat"),
                  class_20=np.array(["injection"] * len(rows)), isolated=np.ones(len(rows), bool), wide_unblended=np.zeros(len(rows), bool),
                  sub20=np.zeros(len(rows), bool), N_nn=np.full(len(rows), np.nan), TARGETID=f("TARGETID", int), m20=np.ones(len(rows), int))
        ev["zblock"] = np.array([zblock(z) for z in ev["z"]])
    ev["snrgroup"] = np.array([SNRGROUP[int(s)] for s in ev["stratum"]])
    ev["dN"] = ev["Nhat"] - ev["logN"]
    return ev


def dstats(dN, w=None):
    dN = np.asarray(dN, float); ok = np.isfinite(dN); dN = dN[ok]
    if w is None:
        w = np.ones(len(dN))
    else:
        w = np.asarray(w, float)[ok]
    n = float(w.sum())
    if len(dN) == 0 or n <= 0:
        return dict(n=0)
    mean = float((w * dN).sum() / n); sd = float(np.sqrt((w * (dN - mean) ** 2).sum() / n))
    order = np.argsort(dN); cw = np.cumsum(w[order]) / n
    q = lambda p: float(dN[order][np.searchsorted(cw, p)])
    return dict(n=int(len(dN)), n_eff=round(n, 1), mean=round(mean, 4), median=round(q(0.5), 4), sd=round(sd, 4), p16=round(q(0.16), 4), p84=round(q(0.84), 4),
                p_gt_0p1=round(float((w * (dN > 0.1)).sum() / n), 4), p_gt_0p2=round(float((w * (dN > 0.2)).sum() / n), 4),
                p_lt_m0p2=round(float((w * (dN < -0.2)).sum() / n), 4))


def crossing(ev, mask, w=None):
    """U = P(Nhat>=20.3 | N in [20.0,20.3), det), D = P(Nhat<20.3 | N in [20.3,20.6), det); also all-truth versions (x completeness)."""
    w = np.ones(len(ev["logN"])) if w is None else np.asarray(w, float)
    out = {}
    for key, lo, hi, cond in (("U", 20.0, 20.3, lambda x: x >= 20.3), ("D", 20.3, 20.6, lambda x: x < 20.3)):
        sel = mask & (ev["logN"] >= lo) & (ev["logN"] < hi)
        det = sel & ev["matched"] & np.isfinite(ev["Nhat"])
        nt = w[sel].sum(); nd = w[det].sum()
        out[key] = round(float(w[det & cond(np.nan_to_num(ev["Nhat"], nan=-99))].sum() / nd), 4) if nd > 0 else None
        out[key + "_alltruth"] = round(float(w[det & cond(np.nan_to_num(ev["Nhat"], nan=-99))].sum() / nt), 4) if nt > 0 else None
        out[key + "_n"] = int(sel.sum())
    return out


def bin_table(ev, mask, edges=TRUTH_EDGES, w=None, min_n=20):
    w_all = np.ones(len(ev["logN"])) if w is None else np.asarray(w, float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = mask & (ev["logN"] >= lo) & (ev["logN"] < hi)
        n = int(sel.sum())
        if n < min_n:
            continue
        det = sel & ev["matched"] & np.isfinite(ev["Nhat"])
        rec = dict(bin=[float(lo), float(hi)], n_truth=n, C=round(float(w_all[det].sum() / w_all[sel].sum()), 4))
        rec.update(dstats(ev["dN"][det], w_all[det]))
        out.append(rec)
    return out


def stage_tables(out_dir):
    res = {}
    for s in list(NATIVE) + list(INJ):
        ev = load_events(s, out_dir); allm = np.ones(len(ev["logN"]), bool)
        rec = dict(n_events=int(len(ev["logN"])), n_detected=int(ev["matched"].sum()),
                   all=dict(bins=bin_table(ev, allm), crossing=crossing(ev, allm)), by_snrgroup={}, by_zblock={}, by_class={}, by_class_x_snrgroup={})
        for g in ("low", "mid", "high"):
            m = ev["snrgroup"] == g
            rec["by_snrgroup"][g] = dict(bins=bin_table(ev, m), crossing=crossing(ev, m))
        for zb in range(3):
            m = ev["zblock"] == zb
            rec["by_zblock"][str(zb)] = dict(bins=bin_table(ev, m), crossing=crossing(ev, m))
        classes = ["isolated", "single", "wide_unblended", "wide", "gap", "moderate", "close"] if s in NATIVE else ["injection"]
        for c in classes:
            m = ev["isolated"] if c == "isolated" else ev["wide_unblended"] if c == "wide_unblended" else (ev["class_20"] == c)
            rec["by_class"][c] = dict(n=int(m.sum()), bins=bin_table(ev, m), crossing=crossing(ev, m))
            rec["by_class_x_snrgroup"][c] = {g: dict(bins=bin_table(ev, m & (ev["snrgroup"] == g)), crossing=crossing(ev, m & (ev["snrgroup"] == g)))
                                             for g in ("low", "mid", "high")}
        res[s] = rec
    json.dump(res, open(os.path.join(out_dir, "conditional_response_tables.json"), "w"), indent=1)
    # compact print: mean dN per bin for the key rows
    for s in res:
        print(f"== {s}: events {res[s]['n_events']} detected {res[s]['n_detected']}; all-bins mean dN:",
              [(b["bin"][0], b["mean"], b["p_gt_0p2"]) for b in res[s]["all"]["bins"]], "| crossing", res[s]["all"]["crossing"])
    return res


# ----------------------------------------------------------------------------------------------------------------------- reweight
def hist_density(N, edges=EDGES, w=None):
    h, _ = np.histogram(N, edges, weights=w); return h.astype(float)


def scheme_weights(N, scheme, own_hist, other_hist):
    """Importance weights per truth absorber (normalised to mean 1 over the events given)."""
    N = np.asarray(N, float); idx = np.clip(np.searchsorted(EDGES, N, side="right") - 1, 0, len(EDGES) - 2)
    below = N < EDGES[0]
    if scheme == "W1_unit":
        w = np.ones(len(N))
    elif scheme == "W2_flat":
        dens = own_hist / max(own_hist.max(), 1); w = 1.0 / np.maximum(dens[idx], 1e-3); w = np.minimum(w, 20.0)
    elif scheme == "W3_other_shape":
        ratio = np.where(own_hist > 0, other_hist / np.maximum(own_hist, 1), 0.0); w = ratio[idx]
    elif scheme == "W5_tilt_minus":
        w = 10 ** (-TILT * (N - 20.5))
    elif scheme == "W5_tilt_plus":
        w = 10 ** (+TILT * (N - 20.5))
    elif scheme == "W5_tilt_minus_0p8":
        w = 10 ** (-2 * TILT * (N - 20.5))
    elif scheme == "W5_tilt_plus_0p8":
        w = 10 ** (+2 * TILT * (N - 20.5))
    else:
        raise ValueError(scheme)
    w = np.where(below, w[0] if scheme.startswith("W2") or scheme.startswith("W3") else w, w)   # [19.0,19.5): carry the first bin's weight
    if scheme in ("W2_flat", "W3_other_shape"):
        w = np.where(below, (1.0 / max(own_hist[0] / max(own_hist.max(), 1), 1e-3) if scheme == "W2_flat" else (other_hist[0] / max(own_hist[0], 1))), w)
        w = np.minimum(w, 20.0)
    return w / max(w.mean(), 1e-12)


def fit_events(ev, mask):
    """Events for the production-style fit: detected TPs with Nhat >= 19.5 (the observed-grid floor)."""
    det = mask & ev["matched"] & np.isfinite(ev["Nhat"]) & (ev["Nhat"] >= XHAT_FLOOR)
    z_edges = ZB_EMUL if np.nanmedian(ev["z"][det]) >= 3.8 else ZB_NATIVE
    meas = dict(N_true=ev["logN"][det], snr=ev["snr"][det], zqso=ev["z"][det], dx=ev["dN"][det], xhat=ev["Nhat"][det], z_covariate="zdla")
    return meas, det, np.asarray(z_edges, float)


def fit_under(meas, z_edges, w=None):
    sys.path.insert(0, HBI_REPO)
    from CDDF_analysis.hbi.znz_kernel import fit_forward_response, _empirical_forward_cells
    frm = fit_forward_response(meas, snr_edges=FIT_SNR_EDGES, z_edges=z_edges, deg_N=2, n_N_cells=7, min_count=40, N_skew_collapse=21.0,
                               N_skew_ramp_width=0.5, N_ref=None, build_empirical=False, weights=w)
    cells = _empirical_forward_cells(meas["N_true"], meas["snr"], meas["zqso"], meas["dx"], np.asarray(FIT_SNR_EDGES, float), z_edges,
                                     n_N_cells=7, min_count=40, weights=w)
    n_snr, n_z = len(FIT_SNR_EDGES) - 1, len(z_edges) - 1
    rng = np.full((n_snr, n_z, 2), np.nan)
    for (a, b), cl in cells.items():
        if cl:
            cs = [c[0] for c in cl]; rng[a, b] = (min(cs), max(cs))
    return frm, rng, cells


def kernel_from_fit(frm, rng, truth_edges=TRUTH_EDGES, obs_edges=EDGES):
    """Skew-normal bin-to-bin masses M[s, z, c, b] mirroring count_conserving_fold.surface_masses (bin centres, fit-range clamp, skew ramp)."""
    from scipy.stats import skewnorm
    sys.path.insert(0, HBI_REPO)
    from CDDF_analysis.hbi.znz_kernel import _moment_to_skewnormal_vec
    Nc = 0.5 * (truth_edges[:-1] + truth_edges[1:]); n_snr, n_z = rng.shape[:2]
    M = np.zeros((n_snr, n_z, len(obs_edges) - 1, len(Nc))); mu = np.zeros((n_snr, n_z, len(Nc))); sd_ = np.zeros_like(mu); sk_ = np.zeros_like(mu)
    snr_rep = [0.5 * (FIT_SNR_EDGES[i] + (FIT_SNR_EDGES[i + 1] if np.isfinite(FIT_SNR_EDGES[i + 1]) else FIT_SNR_EDGES[i] + 2)) for i in range(n_snr)]
    z_rep = [0.5 * (frm.z_edges[j] + frm.z_edges[j + 1]) for j in range(n_z)]
    # cells without populated sub-bins carry the fitter's POOLED constant surfaces (mu_coef[a,b] = [pooled, 0, 0]); their covariate range is the
    # global event range (no clamp effect on a constant). Marked in K["pooled"] so the metrics can exclude them.
    pooled = np.zeros((n_snr, n_z), bool)
    for a in range(n_snr):
        for b in range(n_z):
            if not np.isfinite(rng[a, b, 0]):
                pooled[a, b] = True; rng = rng.copy(); rng[a, b] = (float(Nc.min()), float(Nc.max()))
            Ncl = np.clip(Nc, rng[a, b, 0], rng[a, b, 1])
            s = np.full(len(Nc), snr_rep[a]); zz = np.full(len(Nc), z_rep[b])
            m = frm.mu_b(Ncl, s, zz); sdv = frm.sigma(Ncl, s, zz)
            g = frm._eval_surface(frm.skew_coef, Ncl, frm._i_snr(s), frm._i_z(zz)); g = np.clip(g, -0.995 * 0.99, 0.995 * 0.99)
            ramp = np.clip((Nc - frm.N_skew_collapse) / frm.N_skew_ramp_width, 0, 1); g = g * (1 - ramp)
            xi, om, al = _moment_to_skewnormal_vec(Nc + m, sdv, g)
            cdf = np.stack([skewnorm.cdf(e, al, loc=xi, scale=om) for e in obs_edges])
            M[a, b] = np.clip(np.diff(cdf, axis=0), 0, 1); mu[a, b] = m; sd_[a, b] = sdv; sk_[a, b] = g
    return dict(M=M, mu=mu, sd=sd_, skew=sk_, Nc=Nc, pooled=pooled)


def kernel_metrics(K, truth_edges=TRUTH_EDGES, obs_edges=EDGES):
    """U, D, tail from the kernel per cell: U from truth bin [20.1,20.3), D from [20.3,20.5), tail = P(Nhat>=21.1 | [20.7,20.9))."""
    Nc = K["Nc"]; M = K["M"]; ob = obs_edges[:-1]
    def frac(bin_lo, cond):
        b = int(np.argmin(np.abs(Nc - (bin_lo + 0.1)))); m = M[:, :, :, b]; phi = np.nansum(m, axis=2)
        return np.where(phi > 0, np.nansum(m[:, :, cond], axis=2) / np.maximum(phi, 1e-12), np.nan)
    return dict(U=frac(20.1, ob >= 20.3 - 1e-9), D=frac(20.3, ob < 20.3 - 1e-9), tail=frac(20.7, ob >= 21.1 - 1e-9))


def stage_reweight(out_dir):
    evs = {s: load_events(s, out_dir) for s in NATIVE}
    hists = {s: hist_density(evs[s]["logN"]) for s in NATIVE}
    schemes = ["W1_unit", "W2_flat", "W3_other_shape", "W5_tilt_minus", "W5_tilt_plus", "W5_tilt_minus_0p8", "W5_tilt_plus_0p8"]
    Ngrid = np.round(np.arange(20.0, 21.5 + 1e-9, 0.1), 2)
    res = {}
    for s in NATIVE:
        other = "NL" if s == "N2" else "N2"
        ev = evs[s]; meas, det, z_edges = fit_events(ev, np.ones(len(ev["logN"]), bool))
        res[s] = dict(n_fit_events=int(det.sum()), z_edges=list(map(float, z_edges)), schemes={})
        base = None
        for sc in schemes:
            w = scheme_weights(meas["N_true"], sc, hists[s], hists[other])
            frm, rng, cells = fit_under(meas, z_edges, w)
            K = kernel_from_fit(frm, rng); km = kernel_metrics(K)
            # implied mu_b on the N grid per cell (clamped)
            mu_grid = np.full((rng.shape[0], rng.shape[1], len(Ngrid)), np.nan)
            for a in range(rng.shape[0]):
                for b in range(rng.shape[1]):
                    if np.isfinite(rng[a, b, 0]):
                        Ncl = np.clip(Ngrid, rng[a, b, 0], rng[a, b, 1]); s_ = np.full(len(Ngrid), 0.5 * (FIT_SNR_EDGES[a] + (FIT_SNR_EDGES[a + 1] if np.isfinite(FIT_SNR_EDGES[a + 1]) else FIT_SNR_EDGES[a] + 2)))
                        zz = np.full(len(Ngrid), 0.5 * (z_edges[b] + z_edges[b + 1])); mu_grid[a, b] = frm.mu_b(Ncl, s_, zz)
            # weighted narrow-bin (0.2-dex) means (criterion C2)
            nb = []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                m = (meas["N_true"] >= lo) & (meas["N_true"] < hi)
                if m.sum() >= 20:
                    nb.append(dict(bin=[float(lo), float(hi)], n=int(m.sum()), mean_w=round(float((w[m] * meas["dx"][m]).sum() / w[m].sum()), 4), mean_unit=round(float(meas["dx"][m].mean()), 4)))
            rec = dict(weights=dict(mean=1.0, sd=round(float(w.std()), 3), max=round(float(w.max()), 3), n_eff=round(float(w.sum() ** 2 / (w ** 2).sum()), 1)),
                       mu_coef=frm.mu_coef.tolist(), sig_coef=frm.sig_coef.tolist(), skew_coef=frm.skew_coef.tolist(), N_ref=float(frm.N_ref),
                       fit_range=np.round(rng, 3).tolist(), mu_grid=np.round(mu_grid, 4).tolist(), Ngrid=Ngrid.tolist(),
                       kernel_U=np.round(km["U"], 4).tolist(), kernel_D=np.round(km["D"], 4).tolist(), kernel_tail=np.round(km["tail"], 4).tolist(), narrow_bins=nb)
            if base is None:
                base = dict(mu_grid=mu_grid, U=km["U"], D=km["D"], tail=km["tail"])
                rec["vs_W1"] = dict(d_mu=0.0, d_U=0.0, d_D=0.0, d_tail=0.0)
            else:
                d_mu = float(np.nanmax(np.abs(mu_grid - base["mu_grid"]))); d_U = float(np.nanmax(np.abs(km["U"] - base["U"]))); d_D = float(np.nanmax(np.abs(km["D"] - base["D"]))); d_t = float(np.nanmax(np.abs(km["tail"] - base["tail"])))
                rec["vs_W1"] = dict(d_mu=round(d_mu, 4), d_U=round(d_U, 4), d_D=round(d_D, 4), d_tail=round(d_t, 4),
                                    d_mu_by_cell=np.round(np.nanmax(np.abs(mu_grid - base["mu_grid"]), axis=2), 4).tolist())
            res[s]["schemes"][sc] = rec
            print(f"{s} {sc}: n_eff {rec['weights']['n_eff']} vs_W1 {rec['vs_W1'] if 'd_mu_by_cell' not in rec['vs_W1'] else {k: v for k, v in rec['vs_W1'].items() if k != 'd_mu_by_cell'}}")
    json.dump(res, open(os.path.join(out_dir, "reweighting_fits.json"), "w"), indent=1)
    return res


# ----------------------------------------------------------------------------------------------------------------------- context
def context_cells(ev):
    nn = np.where(~np.isfinite(ev["N_nn"]), "none", np.where(ev["N_nn"] < 20.0, "lt20", np.where(ev["N_nn"] < 20.5, "20-20.5", "ge20.5")))
    return np.array([f"{c}|{g}|{z}|{n}" for c, g, z, n in zip(ev["class_20"], ev["snrgroup"], ev["zblock"], nn)])


def stage_context(out_dir):
    e2 = load_events("N2", out_dir); eL = load_events("NL", out_dir)
    m2 = e2["logN"] >= 19.5; mL = eL["logN"] >= 19.5
    c2 = context_cells(e2); cL = context_cells(eL)
    cells = sorted(set(c2[m2]) | set(cL[mL])); n2 = {c: int((c2[m2] == c).sum()) for c in cells}; nL = {c: int((cL[mL] == c).sum()) for c in cells}
    w = np.zeros(len(e2["logN"]))
    dropped = []
    for c in cells:
        if n2[c] >= 5:
            w[c2 == c] = nL[c] / n2[c]
        else:
            dropped.append((c, n2[c], nL[c]))
    w = w * m2; w = w / max(w[m2].mean(), 1e-12)
    res = dict(n_cells=len(cells), dropped_cells=dropped, n_eff_after=round(float(w[m2].sum() ** 2 / (w[m2] ** 2).sum()), 1), groups={})
    def compare(mask2, maskL, key):
        before = bin_table(e2, mask2); after = bin_table(e2, mask2, w=w); ref = bin_table(eL, maskL)
        rows = []
        for r in ref:
            b0 = next((x for x in before if x["bin"] == r["bin"]), None); a0 = next((x for x in after if x["bin"] == r["bin"]), None)
            if b0 and a0 and r.get("n", 0) >= 20 and b0.get("n", 0) >= 20:
                rows.append(dict(bin=r["bin"], NL_mean=r["mean"], N2_before=b0["mean"], N2_after=a0["mean"], d_before=round(b0["mean"] - r["mean"], 4), d_after=round(a0["mean"] - r["mean"], 4),
                                 NL_p02=r["p_gt_0p2"], N2_before_p02=b0["p_gt_0p2"], N2_after_p02=a0["p_gt_0p2"], n_NL=r["n"], n_N2=b0["n"]))
        res["groups"][key] = dict(rows=rows, crossing_NL=crossing(eL, maskL), crossing_N2_before=crossing(e2, mask2), crossing_N2_after=crossing(e2, mask2, w=w))
    compare(m2, mL, "all")
    for g in ("low", "mid", "high"):
        compare(m2 & (e2["snrgroup"] == g), mL & (eL["snrgroup"] == g), f"snr_{g}")
    for c in ("single", "wide", "gap", "moderate", "close"):
        compare(m2 & (e2["class_20"] == c), mL & (eL["class_20"] == c), f"class_{c}")
    json.dump(res, open(os.path.join(out_dir, "context_matching.json"), "w"), indent=1)
    for k, v in res["groups"].items():
        print(k, [(r["bin"][0], r["d_before"], r["d_after"]) for r in v["rows"]])
    return res


# ----------------------------------------------------------------------------------------------------------------------- unfold
def em_unfold(A, n_obs, n_iter=200):
    """D'Agostini EM: A[c,b] = P(observed in c | truth in b) incl. completeness (columns need not sum to 1)."""
    eff = A.sum(axis=0); f = np.full(A.shape[1], n_obs.sum() / max(A.shape[1], 1))
    for _ in range(n_iter):
        pred = A @ f
        pred = np.where(pred > 0, pred, 1e-12)
        f = f * (A.T @ (n_obs / pred)) / np.maximum(eff, 1e-12)
        f = np.where(eff > 0, f, 0.0)
    return f


def stage_unfold(out_dir, seed=20260902, tag=""):
    evs = {s: load_events(s, out_dir) for s in NATIVE}
    hists = {s: hist_density(evs[s]["logN"]) for s in NATIVE}
    schemes = ["W1_unit", "W5_tilt_minus", "W5_tilt_plus", "W3_other_shape"]
    res = {}
    for s in NATIVE:
        other = "NL" if s == "N2" else "N2"; ev = evs[s]
        rng = np.random.default_rng(seed); tids = np.unique(ev["TARGETID"]); half = set(rng.choice(tids, len(tids) // 2, replace=False).tolist())
        in_cal = np.array([t in half for t in ev["TARGETID"]]); in_eval = ~in_cal
        ok = ev["logN"] >= FLOOR_LAT
        det = ev["matched"] & np.isfinite(ev["Nhat"]) & (ev["Nhat"] >= XHAT_FLOOR)
        tb = np.clip(np.searchsorted(TRUTH_EDGES, ev["logN"], side="right") - 1, 0, len(TRUTH_EDGES) - 2)
        ob = np.clip(np.searchsorted(EDGES, np.nan_to_num(ev["Nhat"], nan=0.0), side="right") - 1, 0, len(EDGES) - 2)
        out = dict(n_cal=int(in_cal.sum()), n_eval=int(in_eval.sum()), grid=dict(truth=TRUTH_EDGES.tolist(), obs=EDGES.tolist()), cases={})
        meas_cal, det_cal_mask, z_edges = fit_events(ev, in_cal)
        meas_all, _, _ = fit_events(ev, np.ones(len(ev["logN"]), bool))
        for beta in schemes:
            w_b = scheme_weights(ev["logN"], beta, hists[s], hists[other])
            # empirical operator on H_cal: A[c,b] = sum w (truth b, det, obs c) / sum w (truth b)
            A_emp = np.zeros((len(EDGES) - 1, len(TRUTH_EDGES) - 1)); denom = np.zeros(len(TRUTH_EDGES) - 1)
            for b in range(len(TRUTH_EDGES) - 1):
                m = in_cal & ok & (tb == b); denom[b] = w_b[m].sum()
                for c in range(len(EDGES) - 1):
                    A_emp[c, b] = w_b[m & det & (ob == c)].sum()
            A_emp = np.where(denom > 0, A_emp / np.maximum(denom, 1e-12), 0.0)
            # parametric operator (spec Amendment 2): the production-style fit needs >= 7 x 40 events per (SNR, z) cell, which the half sample does not
            # provide (every cell fell back to constants); it is therefore fitted on ALL events under w_b, while completeness and the truth mix stay on H_cal
            # and the evaluation population stays on H_eval. The empirical operator remains strictly split-half.
            wfit = scheme_weights(meas_all["N_true"], beta, hists[s], hists[other])
            frm, rngc, _ = fit_under(meas_all, z_edges, wfit); K = kernel_from_fit(frm, rngc)
            isnr = np.clip(np.searchsorted(np.asarray(FIT_SNR_EDGES), ev["snr"], side="right") - 1, 0, len(FIT_SNR_EDGES) - 2)
            iz = np.clip(np.searchsorted(z_edges, ev["z"], side="right") - 1, 0, len(z_edges) - 2)
            A_par = np.zeros_like(A_emp)
            for b in range(len(TRUTH_EDGES) - 1):
                m = in_cal & ok & (tb == b); tot = w_b[m].sum()
                if tot <= 0:
                    continue
                for a in range(len(FIT_SNR_EDGES) - 1):
                    for zc in range(len(z_edges) - 1):
                        mc = m & (isnr == a) & (iz == zc); nt = w_b[mc].sum()
                        if nt <= 0 or not np.all(np.isfinite(K["M"][a, zc, :, b])):
                            continue
                        C_cell = w_b[mc & det].sum() / nt
                        A_par[:, b] += (nt / tot) * C_cell * K["M"][a, zc, :, b]
            for alpha in schemes:
                w_a = scheme_weights(ev["logN"], alpha, hists[s], hists[other])
                m_ev = in_eval & ok
                truth_hist = np.array([w_a[m_ev & (tb == b)].sum() for b in range(len(TRUTH_EDGES) - 1)])
                n_obs = np.array([w_a[m_ev & det & (ob == c)].sum() for c in range(len(EDGES) - 1)])
                case = {}
                for name, A in (("empirical", A_emp), ("parametric", A_par)):
                    if not np.any(A > 0):
                        case[name] = dict(error="operator has no support"); continue
                    f_hat = em_unfold(A, n_obs); f_hat_1000 = em_unfold(A, n_obs, n_iter=1000)
                    ge = lambda x, thr: float(x[TRUTH_EDGES[:-1] >= thr - 1e-9].sum())
                    tail = lambda x: float(x[TRUTH_EDGES[:-1] >= 21.1 - 1e-9].sum())
                    case[name] = dict(bias_ge20p0_pct=round(100 * (ge(f_hat, 20.0) / ge(truth_hist, 20.0) - 1), 2), bias_ge20p3_pct=round(100 * (ge(f_hat, 20.3) / ge(truth_hist, 20.3) - 1), 2),
                                      bias_tail_ge21p1_pct=round(100 * (tail(f_hat) / max(tail(truth_hist), 1e-9) - 1), 2),
                                      bias_by_bin_pct=[round(100 * (fh / th - 1), 1) if th > 0 else None for fh, th in zip(f_hat, truth_hist)],
                                      truth=np.round(truth_hist, 2).tolist(), f_hat=np.round(f_hat, 2).tolist(),
                                      bias_ge20p3_pct_1000iter=round(100 * (ge(f_hat_1000, 20.3) / ge(truth_hist, 20.3) - 1), 2),
                                      bias_ge20p0_pct_1000iter=round(100 * (ge(f_hat_1000, 20.0) / ge(truth_hist, 20.0) - 1), 2),
                                      n_par_cells_with_support=int((~K["pooled"]).sum()) if name == "parametric" else None)
                out["cases"][f"alpha={alpha}|beta={beta}"] = case
                ce, cp = case["empirical"], case["parametric"]
                print(f"{s} alpha={alpha} beta={beta}: emp ≥20.3 {ce.get('bias_ge20p3_pct')} (1000it {ce.get('bias_ge20p3_pct_1000iter')}) | par ≥20.3 {cp.get('bias_ge20p3_pct')} (1000it {cp.get('bias_ge20p3_pct_1000iter')}) cells {cp.get('n_par_cells_with_support')} | emp ≥20.0 {ce.get('bias_ge20p0_pct')} par {cp.get('bias_ge20p0_pct')}")
        res[s] = out
    json.dump(res, open(os.path.join(out_dir, f"prior_stress_unfold{tag}.json"), "w"), indent=1)
    return res


# ----------------------------------------------------------------------------------------------------------------------- summary
def stage_summary(out_dir):
    T = json.load(open(os.path.join(out_dir, "conditional_response_tables.json")))
    R = json.load(open(os.path.join(out_dir, "reweighting_fits.json")))
    Cx = json.load(open(os.path.join(out_dir, "context_matching.json")))
    U = json.load(open(os.path.join(out_dir, "prior_stress_unfold.json")))
    summ = dict(C1_material_fit_weight={}, C2_narrow_bin_stable={}, D1_agreement={}, E1_material_prior={})
    for s in R:
        trig = {sc: (r["vs_W1"]["d_mu"] > 0.03 or r["vs_W1"]["d_U"] > 0.03 or r["vs_W1"]["d_D"] > 0.03) for sc, r in R[s]["schemes"].items() if sc != "W1_unit" and not sc.endswith("0p8")}
        summ["C1_material_fit_weight"][s] = dict(triggered=any(trig.values()), by_scheme=trig, metrics={sc: {k: v for k, v in r["vs_W1"].items() if k != "d_mu_by_cell"} for sc, r in R[s]["schemes"].items()})
        nbmax = {sc: max((abs(b["mean_w"] - b["mean_unit"]) for b in r["narrow_bins"]), default=0.0) for sc, r in R[s]["schemes"].items()}
        summ["C2_narrow_bin_stable"][s] = dict(stable=all(v <= 0.02 for v in nbmax.values()), max_abs_shift_by_scheme={k: round(v, 4) for k, v in nbmax.items()})
    for g in ("snr_low", "snr_high", "all"):
        rows = Cx["groups"][g]["rows"]
        rows_in = [r for r in rows if 20.0 <= r["bin"][0] < 21.5]
        summ["D1_agreement"][g] = dict(before=all(abs(r["d_before"]) <= 0.03 and abs(r["N2_before_p02"] - r["NL_p02"]) <= 0.05 for r in rows_in),
                                       after=all(abs(r["d_after"]) <= 0.03 and abs(r["N2_after_p02"] - r["NL_p02"]) <= 0.05 for r in rows_in),
                                       rows=[dict(bin=r["bin"], d_before=r["d_before"], d_after=r["d_after"], dp02_before=round(r["N2_before_p02"] - r["NL_p02"], 3), dp02_after=round(r["N2_after_p02"] - r["NL_p02"], 3)) for r in rows_in])
    for s in U:
        cases = U[s]["cases"]; e1 = {}
        for key, c in cases.items():
            a, b = key.split("|"); a = a.split("=")[1]; b = b.split("=")[1]
            ref = cases[f"alpha={a}|beta={a}"]
            for op in ("empirical", "parametric"):
                e1[f"{key}|{op}"] = round(c[op]["bias_ge20p3_pct"] - ref[op]["bias_ge20p3_pct"], 2)
        summ["E1_material_prior"][s] = dict(excess_bias_ge20p3_pct=e1, triggered_empirical=any(abs(v) > 2 for k, v in e1.items() if k.endswith("empirical") and k.split("|")[0] != k.split("|")[1].replace("beta", "alpha")),
                                            triggered_parametric=any(abs(v) > 2 for k, v in e1.items() if k.endswith("parametric") and k.split("|")[0] != k.split("|")[1].replace("beta", "alpha")))
    json.dump(summ, open(os.path.join(out_dir, "criteria_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1)[:6000])
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["match", "tables", "reweight", "context", "unfold", "summary"])
    ap.add_argument("--out", default=f"{ROOT}/response_study")
    ap.add_argument("--sample", default=None, help="match: N2 or NL (default both)")
    ap.add_argument("--seed", type=int, default=20260902, help="unfold: split-half seed (extra seeds quantify the metric scatter)")
    ap.add_argument("--tag", default="", help="unfold: output suffix for extra seeds")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.stage == "match":
        for s in ([a.sample] if a.sample else list(NATIVE)):
            stage_match(s, a.out)
    elif a.stage == "tables":
        stage_tables(a.out)
    elif a.stage == "reweight":
        stage_reweight(a.out)
    elif a.stage == "context":
        stage_context(a.out)
    elif a.stage == "unfold":
        stage_unfold(a.out, seed=a.seed, tag=a.tag)
    else:
        stage_summary(a.out)


if __name__ == "__main__":
    main()
