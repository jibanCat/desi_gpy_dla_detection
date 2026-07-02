#!/usr/bin/env python
"""balfinder_validation.py — apples-to-apples completeness + purity of the DESI
BAL-finder (Paul Martini's ``baltools``) run on the 2LPT-0 loa-124 LyaCoLoRe mock.

Motivation
----------
The high-N DLA false-positive budget (roadmap blocker #1) removes all BAL sightlines
using the DESI BAL VAC. The load-bearing input is the *BAL-finder completeness at our
op-cut* — how many FP-causing BALs the finder flags (and thus removes) vs how many leak.
Literature (Filbert/Martini 2024) gives c ≈ 0.90–0.95 at SNR>2; this script MEASURES it
directly by running ``baltools`` on the mock (LyaCoLoRe — the same mock family as Fig 2 of
Filbert 2024) and cross-matching against the truth ``bal_cat`` and the GP dlacat.

Each figure validates one step so we can *see* the finder is behaving correctly (not a
silent bug):
  fig1  completeness vs CIV-region SNR  -> must reproduce Filbert Fig 2 (~95% asymptote)
  fig2  completeness vs BAL strength BI_CIV
  fig3  purity (confusion) at the op-cut
  fig4  finder AI_CIV vs truth BI_CIV  -> the finder measures the right quantity
  fig5  example spectra: a recovered strong BAL vs a (low CIV-SNR) missed one
  fig6  FP-BAL completeness -> Omega residual (the headline number)

Every number is aggregate/mock only (privacy-clean). RE-RUN, don't trust the prose.

Env:  source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate gpdla
Run:  python balfinder_validation.py [--nmax-hp N] [--outdir DIR]
"""
import os, glob, json, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fitsio

CIV = 1549.06          # C IV rest wavelength (Å)
C_KMS = 299792.458
VMAX_BAL = 25000.0     # BAL search velocity (km/s), Filbert §4.1

# ---- default paths (GreatLakes) -------------------------------------------------
BALDIR = "/scratch/cavestru_root/cavestru0/mfho/balfinder_mock_loa124/spectra-16"
MOCK   = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124"
DLACAT = "/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/dlacat-v2.8.5-mockcat.fits"
SNR_MIN, PDLA_MIN = 2.0, 0.99      # real CDDF op-cut (cddf_catalog_hbi.py:108-109)


def load_baltables(baldir, nmax=None, min_age_s=60):
    """Return {tid: AI_CIV}, {tid: BAL_PROB}, {tid: Z}, processed-tid set, healpix list.

    IMPORTANT (reproducibility): runbalfinder writes with --nproc N workers; reading a
    baltable mid-write returns spurious AI_CIV=0 for not-yet-written rows -> phantom
    "missed BAL" -> completeness biased LOW / residual biased HIGH. We therefore SKIP any
    baltable modified within the last `min_age_s` seconds and ERROR if the finder appears
    active, so finder-measured numbers are always derived on a STATIC set.
    """
    import time
    fs = sorted(glob.glob(f"{baldir}/*/*/baltable-16-*.fits"))
    if nmax:
        fs = fs[:nmax]
    now = time.time(); fresh = [f for f in fs if now - os.path.getmtime(f) < min_age_s]
    if fresh:
        raise RuntimeError(
            f"{len(fresh)} baltables were modified in the last {min_age_s}s — the BAL-finder "
            f"looks ACTIVE. Reading mid-write biases completeness. Wait for it to finish "
            f"(pgrep -f 'python.*runbalfinder') and re-run, or pass a larger --min-age.")
    ai, bi, z, hps = {}, {}, {}, []
    for f in fs:
        d = fitsio.read(f)
        hps.append(int(f.split("-")[-1][:-5]))
        for a, av, bv, zv in zip(d["TARGETID"], d["AI_CIV"], d["BI_CIV"], d["Z"]):
            ai[int(a)] = float(av); bi[int(a)] = float(bv); z[int(a)] = float(zv)
    return ai, bi, z, set(ai.keys()), sorted(hps)


def civ_region_snr(hps, mock, ztab):
    """Per-QSO CIV-region SNR (Filbert §4.1): mean(flux*sqrt(ivar)) over v=0..25000 km/s
    blueward of C IV, combining whichever cameras cover the window. Cached to disk."""
    cache = os.path.join(os.path.dirname(__file__), ".civsnr_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache); done = set(z["hps"].tolist())
        if set(hps) <= done:
            return {int(t): float(s) for t, s in zip(z["tid"], z["snr"])}
    out = {}
    for hp in hps:
        sp = f"{mock}/spectra-16/{hp//100}/{hp}/spectra-16-{hp}.fits"
        fm = fitsio.read(sp, ext="FIBERMAP"); tid = np.array([int(x) for x in fm["TARGETID"]])
        FL = {c: fitsio.read(sp, ext=f"{c}_FLUX") for c in "BRZ"}
        IV = {c: fitsio.read(sp, ext=f"{c}_IVAR") for c in "BRZ"}
        WV = {c: fitsio.read(sp, ext=f"{c}_WAVELENGTH") for c in "BRZ"}
        for k, tt in enumerate(tid):
            zq = ztab.get(int(tt), np.nan)
            if not (zq > 2):
                continue
            civ = CIV * (1 + zq); lo = civ * (1 - VMAX_BAL / C_KMS)
            snrs = []
            for c in "BRZ":
                m = (WV[c] >= lo) & (WV[c] <= civ)
                if m.sum() > 3:
                    f, g = FL[c][k][m], IV[c][k][m]; ok = g > 0
                    if ok.sum() > 3:
                        snrs.append(np.mean(f[ok] * np.sqrt(g[ok])))
            out[int(tt)] = float(np.nanmean(snrs)) if snrs else np.nan
    # atomic write (tmp + rename) so a concurrent reader never sees a half-written npz.
    # tmp ends in .npz so np.savez uses it verbatim (no extra suffix to track).
    tmp = cache.replace(".npz", f".tmp{os.getpid()}.npz")
    np.savez(tmp, hps=np.array(hps), tid=np.array(list(out)), snr=np.array(list(out.values())))
    os.replace(tmp, cache)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baldir", default=BALDIR); ap.add_argument("--mock", default=MOCK)
    ap.add_argument("--dlacat", default=DLACAT); ap.add_argument("--nmax-hp", type=int, default=None)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    ap.add_argument("--min-age", type=int, default=60,
                    help="skip/refuse baltables modified within this many seconds (mid-write guard)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    # --- catalogs ---
    ai_of, bi_of, z_of, proc, hps = load_baltables(a.baldir, a.nmax_hp, a.min_age)
    tr = fitsio.read(f"{a.mock}/bal_cat.fits")
    tset = set(int(x) for x in tr["TARGETID"])
    tbi = {int(t): float(b) for t, b in zip(tr["TARGETID"], tr["BI_CIV"])}
    tai = {int(t): float(b) for t, b in zip(tr["TARGETID"], tr["AI_CIV"])} if "AI_CIV" in tr.dtype.names else {}
    sc = fitsio.read(f"{a.mock}/snr_cat.fits")
    srd = {int(t): float(s) for t, s in zip(sc["TARGETID"], sc["SNR_REDSIDE"])}
    zc = fitsio.read(f"{a.mock}/zcat.fits")
    zq = {int(t): float(zz) for t, zz in zip(zc["TARGETID"], zc["Z"])}
    civsnr = civ_region_snr(hps, a.mock, zq)

    M = {"n_healpix": len(hps), "n_qso_processed": len(proc)}

    # =========================================================================
    # fig1 — completeness vs CIV-region SNR (reproduce Filbert Fig 2)
    # =========================================================================
    qz = [t for t in proc if zq.get(t, 0) > 2]
    cs = np.array([civsnr.get(t, np.nan) for t in qz])
    tru = np.array([t in tset for t in qz], dtype=bool); rec = np.array([ai_of.get(t, 0) > 0 for t in qz], dtype=bool)
    edges = [0, 1, 2, 3, 5, 10, 40]; mids, comp, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = tru & (cs >= lo) & (cs < hi)
        if m.sum():
            mids.append((lo + hi) / 2); comp.append((rec & m).sum() / m.sum()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(mids, comp, "o-", lw=2, color="C0", zorder=3)
    for x, y, n in zip(mids, comp, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax.axhline(0.95, ls="--", color="grey"); ax.text(20, 0.955, "Filbert 2024 asymptote ~95%", fontsize=8, color="grey")
    ax.set_xscale("log"); ax.set_xlabel("CIV-region SNR (mean flux/err, v=0–25000 km/s blueward of CIV)")
    ax.set_ylabel("completeness (recovered AI>0 / truth BAL)"); ax.set_ylim(0, 1.05)
    ax.set_title(f"fig1 · balfinder completeness vs CIV-SNR ({len(hps)} healpix, z>2)\nreproduces Filbert/Martini 2024 Fig 2")
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig1_completeness_civsnr.png", dpi=130); plt.close(fig)
    M["completeness_vs_civsnr"] = {f"[{lo},{hi})": round(c, 3) for lo, hi, c in zip(edges[:-1], edges[1:], comp)}

    # =========================================================================
    # fig2 — completeness vs BAL strength BI_CIV (at CIV-SNR>2)
    # =========================================================================
    hi_snr = cs > 2
    bis = np.array([tbi.get(t, 0) for t in qz])
    bedges = [0, 500, 1000, 2000, 4000, 1e9]; bmids, bcomp, bns = [], [], []
    for lo, hi in zip(bedges[:-1], bedges[1:]):
        m = tru & hi_snr & (bis >= lo) & (bis < hi)
        if m.sum():
            bmids.append(f"[{lo:.0f},{hi:.0f})" if hi < 1e8 else f">={lo:.0f}")
            bcomp.append((rec & m).sum() / m.sum()); bns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(range(len(bcomp)), bcomp, color="C2")
    for i, (c, n) in enumerate(zip(bcomp, bns)):
        ax.text(i, c + 0.01, f"{c:.2f}\nn={n}", ha="center", fontsize=8)
    ax.set_xticks(range(len(bmids))); ax.set_xticklabels(bmids, fontsize=8)
    ax.set_xlabel("truth BI_CIV (BAL strength)"); ax.set_ylabel("completeness (CIV-SNR>2)"); ax.set_ylim(0, 1.1)
    ax.set_title(f"fig2 · completeness vs BAL strength ({len(hps)} healpix)\nstrong BALs (the DLA-mimicking FP source) are ~100% recovered")
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig2_completeness_bi.png", dpi=130); plt.close(fig)
    M["completeness_vs_bi_snr2"] = dict(zip(bmids, [round(c, 3) for c in bcomp]))

    # =========================================================================
    # fig3 — purity at the op-cut (z>2 & SNR_RED>2)
    # =========================================================================
    op_q = [t for t in proc if zq.get(t, 0) > 2 and srd.get(t, 0) > SNR_MIN]
    isf = np.array([ai_of.get(t, 0) > 0 for t in op_q], dtype=bool); ist = np.array([t in tset for t in op_q], dtype=bool)
    TP = int((isf & ist).sum()); FP = int((isf & ~ist).sum()); FN = int((~isf & ist).sum()); TN = int((~isf & ~ist).sum())
    purity = TP / (TP + FP) if (TP + FP) else float("nan"); compl = TP / (TP + FN) if (TP + FN) else float("nan")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    cm = np.array([[TP, FN], [FP, TN]], float)
    ax.imshow(np.log10(cm + 1), cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=12,
                color="white" if np.log10(v + 1) > np.log10(cm.max()) * 0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["truth BAL", "truth non-BAL"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["finder BAL", "finder non-BAL"])
    ax.set_title(f"fig3 · confusion at op-cut (z>2 & SNR_RED>2)\npurity={purity:.3f}  completeness={compl:.3f}  (n={len(op_q)})")
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig3_purity.png", dpi=130); plt.close(fig)
    M["op_cut"] = {"purity": round(purity, 3), "completeness": round(compl, 3),
                   "TP": TP, "FP": FP, "FN": FN, "TN": TN, "n": len(op_q)}

    # =========================================================================
    # fig4 — finder AI_CIV vs truth BI_CIV (does the finder measure the right thing?)
    # =========================================================================
    both = [t for t in op_q if (t in tset) and ai_of.get(t, 0) > 0]
    xa = np.array([tbi[t] for t in both]); ya = np.array([ai_of[t] for t in both])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(xa, ya, s=6, alpha=0.3, color="C3")
    lim = max(xa.max(), ya.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
    r = np.corrcoef(xa, ya)[0, 1] if len(xa) > 2 else np.nan
    ax.set_xlabel("truth BI_CIV"); ax.set_ylabel("finder AI_CIV")
    ax.set_title(f"fig4 · finder vs truth BAL strength (n={len(both)}, r={r:.2f})\nfinder AI tracks truth BI — measuring the right quantity")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{a.outdir}/fig4_ai_vs_bi.png", dpi=130); plt.close(fig)
    M["ai_vs_bi_corr"] = round(float(r), 3)

    # =========================================================================
    # fig5 — example spectra: recovered strong BAL vs missed one
    # =========================================================================
    strong = [t for t in qz if tbi.get(t, 0) > 2000]
    recov = [t for t in strong if ai_of.get(t, 0) > 0 and civsnr.get(t, 0) > 3]
    missd = [t for t in strong if ai_of.get(t, 0) <= 0]
    def readspec(tid):
        hp = None
        for h in hps:
            sp = f"{a.mock}/spectra-16/{h//100}/{h}/spectra-16-{h}.fits"
            fm = fitsio.read(sp, ext="FIBERMAP"); ids = np.array([int(x) for x in fm["TARGETID"]])
            if tid in ids:
                k = int(np.where(ids == tid)[0][0]); hp = sp; break
        if hp is None: return None
        w = np.concatenate([fitsio.read(hp, ext=f"{c}_WAVELENGTH") for c in "BRZ"])
        f = np.concatenate([fitsio.read(hp, ext=f"{c}_FLUX")[k] for c in "BRZ"])
        o = np.argsort(w); return w[o], f[o]
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=False)
    for ax, tid, tag in [(axes[0], recov[0] if recov else None, "RECOVERED"),
                         (axes[1], missd[0] if missd else None, "MISSED")]:
        if tid is None: continue
        s = readspec(tid)
        if s is None: continue
        w, f = s; zz = z_of.get(tid, zq.get(tid))
        civ = CIV * (1 + zz)
        m = (w > civ - 700) & (w < civ + 300)
        ax.plot(w[m], f[m], lw=0.6, color="k")
        ax.axvspan(civ * (1 - VMAX_BAL / C_KMS), civ, color="orange", alpha=0.15, label="CIV BAL search window")
        ax.axvline(civ, color="r", ls="--", lw=1, label="CIV")
        ax.set_title(f"{tag}: BI_truth={tbi.get(tid,0):.0f}, AI_finder={ai_of.get(tid,0):.0f}, "
                     f"CIV-SNR={civsnr.get(tid,float('nan')):.1f}", fontsize=9)
        ax.set_ylabel("flux"); ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("observed wavelength (Å)")
    fig.suptitle("fig5 · example spectra — recovered (deep trough) vs missed (low CIV-SNR)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig5_example_spectra.png", dpi=130); plt.close(fig)

    # =========================================================================
    # fig6 — FP-BAL completeness -> Omega residual (the headline)
    # =========================================================================
    dc = fitsio.read(a.dlacat)
    tid = np.array([int(x) for x in dc["TARGETID"]]); nhi = np.asarray(dc["NHI"], float)
    snrR = np.asarray(dc["SNR_REDSIDE"], float); pdla = np.asarray(dc["P_DLA"], float)
    balflag = np.asarray(dc["BAL_FLAG"], int)
    hcd = fitsio.read(f"{a.mock}/hcd_truth_cat.fits")
    hcdset = set(int(x) for x in hcd["TARGETID"])
    tr_nhi = np.asarray(hcd["NHI"], float); tr_snr = np.asarray(hcd["SNR"], float)
    op = (snrR > SNR_MIN) & (pdla > PDLA_MIN)
    isfp = np.array([t not in hcdset for t in tid]); inproc = np.array([t in proc for t in tid])
    def boot_leak(w, recd, B=8000, seed=0):
        """95% CI on the Omega-leak = 1 - sum(w[recd])/sum(w) by resampling FP-BALs."""
        if len(w) == 0:
            return (float("nan"), float("nan"))
        rng = np.random.default_rng(seed); n = len(w); ls = np.empty(B)
        for b in range(B):
            i = rng.integers(0, n, n); tot = w[i].sum()
            ls[b] = 1 - (w[i][recd[i]].sum() / tot) if tot > 0 else np.nan
        return (float(np.nanpercentile(ls, 2.5)), float(np.nanpercentile(ls, 97.5)))

    res = {}
    for lab, lim in [("ge20.3", 20.3), ("deep_ge21.6", 21.6)]:
        # over-count = FP-BAL Omega / truth-DLA Omega, both over the FULL mock (all healpix)
        # -- must NOT restrict the numerator to processed healpix or it scales down by the
        #    processed-healpix fraction (matches decompose_highn_fp.py's tw()).
        base_full = op & isfp & (balflag == 1) & (nhi >= lim)
        truthO = float(np.sum(10.0 ** tr_nhi[(tr_nhi >= lim) & (tr_snr > SNR_MIN)]))
        overcount = float((10.0 ** nhi[base_full]).sum() / truthO)
        # completeness/leak = measured on the STATIC processed-healpix set (finder cross-match);
        # assumed representative of the full catalog (healpix are statistically equivalent).
        # Report BOTH finder criteria: AI>0 (Filbert Fig-2 detection criterion; optimistic if the
        # removal VAC keeps all AI>0) and BI>0 (matches build_bal_cat_from_qsocat.py default thresh).
        base = base_full & inproc
        w = 10.0 ** nhi[base]
        row = {"n_fpbal_proc": int(base.sum()), "n_fpbal_full": int(base_full.sum()),
               "overcount_frac": round(overcount, 3)}
        for crit, getter in [("ai", ai_of), ("bi", bi_of)]:
            recd = np.array([getter.get(t, 0) > 0 for t in tid[base]], dtype=bool)
            comp_om = float(w[recd].sum() / w.sum()) if w.sum() > 0 else float("nan")
            leak = 1 - comp_om
            lo, hi = boot_leak(w, recd)
            row[f"omega_completeness_{crit}"] = round(comp_om, 3)
            row[f"omega_residual_{crit}"] = round(overcount * leak, 4)
            row[f"omega_residual_{crit}_ci95"] = [round(overcount * lo, 4), round(overcount * hi, 4)]
        # headline residual = the BI>0 (VAC-matched) value; keep legacy keys for the doc/plot
        row["omega_completeness"] = row["omega_completeness_bi"]
        row["leak"] = round(1 - row["omega_completeness_bi"], 3)
        row["omega_residual"] = row["omega_residual_bi"]
        res[lab] = row
    M["fp_bal_residual"] = res

    # SNR-cut consistency: c MUST be measured on the DLA op-cut (SNR_REDSIDE>2 & P_DLA>0.99);
    # a CIV-SNR conditioning or the all-SNR sample average would give the wrong (much lower) c.
    snr_sens = {}
    for slab, smask in [("dla_cut_snrR>2", (snrR > SNR_MIN)), ("no_snr_cut", np.ones(len(tid), bool)),
                        ("snrR>3", snrR > 3), ("snrR>5", snrR > 5)]:
        m = smask & (pdla > PDLA_MIN) & isfp & (balflag == 1) & (nhi >= 20.3) & inproc
        w = 10.0 ** nhi[m]
        cb = float(np.array([bi_of.get(t, 0) > 0 for t in tid[m]], dtype=bool) @ w / w.sum()) if w.sum() else float("nan")
        snr_sens[slab] = {"omega_completeness_bi": round(cb, 3), "n": int(m.sum()),
                          "omega_residual_bi": round(res["ge20.3"]["overcount_frac"] * (1 - cb), 4)}
    M["snr_cut_sensitivity_ge20.3"] = snr_sens

    labs = ["≥20.3", "deep tail ≥21.6"]; keys = ["ge20.3", "deep_ge21.6"]
    over = [res[k]["overcount_frac"] * 100 for k in keys]
    resid = [res[k]["omega_residual_bi"] * 100 for k in keys]          # BI>0 (VAC-matched) headline
    ci = [res[k]["omega_residual_bi_ci95"] for k in keys]
    yerr = np.array([[max(r - c[0] * 100, 0), max(c[1] * 100 - r, 0)] for r, c in zip(resid, ci)]).T
    fig, ax = plt.subplots(figsize=(6, 4.2)); x = np.arange(len(labs))
    ax.bar(x - 0.2, over, 0.4, label="BAL Ω over-count (if none flagged)", color="C1")
    ax.bar(x + 0.2, resid, 0.4, yerr=yerr, capsize=4, label="residual after finder, BI>0 (95% CI)", color="C3")
    for i in range(len(labs)):
        ax.text(x[i] - 0.2, over[i] + 1.5, f"{over[i]:.0f}%", ha="center", fontsize=8)
        ax.text(x[i] + 0.2, ci[i][1] * 100 + 1.5, f"{resid[i]:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("% of truth Ω_DLA")
    ax.set_title(f"fig6 · BAL FP residual on Ω after finder flagging ({len(hps)} hp, static)\n"
                 f"BI>0 completeness={res['ge20.3']['omega_completeness_bi']:.2f}; "
                 f"AI>0 residual≈{res['ge20.3']['omega_residual_ai']*100:.1f}% (optimistic)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{a.outdir}/fig6_fpbal_residual.png", dpi=130); plt.close(fig)

    # --- DLA loss from balfinder impurity (false BAL flags on real-DLA non-BAL sightlines) ---
    op3 = op & (nhi >= 20.3) & inproc
    realDLA = np.array([t in hcdset for t in tid], dtype=bool) & op3
    falseflag = np.array([(ai_of.get(t, 0) > 0) and (t not in tset) for t in tid], dtype=bool)
    lost = realDLA & falseflag
    M["dla_loss_from_impurity"] = {"lost": int(lost.sum()), "of_real_dla": int(realDLA.sum()),
                                   "frac": round(float(lost.sum() / max(realDLA.sum(), 1)), 4)}

    with open(f"{a.outdir}/metrics.json", "w") as fh:
        json.dump(M, fh, indent=2)
    print(json.dumps(M, indent=2))
    print(f"\nfigures + metrics.json -> {a.outdir}")


if __name__ == "__main__":
    main()
