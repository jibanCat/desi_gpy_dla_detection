#!/usr/bin/env python
"""veto_snr_sweep.py — the (BAL-veto x SNR) operating-point sweep on the REAL LOA CDDF
(P0 action #2 of the real-LOA BAL arbiter, notes/2026-07-02_real_loa_bal_arbiter_design.md rev 3).

For each candidate BAL veto x SNR_REDSIDE threshold, tabulate on the real high-N DLA sample:
  - ΔX cost      : fraction of forest search path removed (real-DLA statistical/normalization cost)
  - Ω removed    : fraction of high-N Ω on veto-flagged sightlines
  - residual leak: broad-trough (>2000 km/s) BAL Ω LEFT in the retained set (the CIV-detectable
                   contamination the veto misses; on-VAC — the CIV-weak/FeLoBAL tail is E1-limited)
  - net Ω×       : Ω(veto)/Ω(no-veto) = (retained Ω / retained ΔX) normalised

Candidate vetoes (increasing completeness & cost; each keys on the QSO-frame CIV trough):
  BI>0 (production)   : BI_CIV>0 = the DESI altbal v2 BAL_FLAG in dlacat-loa-main-dark-v1.fits
  broad-trough>2000   : DERIVED (NOT a native column) = max over the VMIN/VMAX_CIV_450 troughs of
                        |VMAX-VMIN| > 2000 km/s + significant AI. The physically DLA-mimicking population
                        BI misses on onset velocity. CAVEAT: the native >=2000 column NCIV_2000 is
                        IDENTICAL to BI>0 (no broad-at-any-onset flag exists); this 450-derived proxy
                        matches BI imperfectly (450-troughs fragment; match-rate value in the private notes,
                        notes/2026-07-02_real_loa_bal_arbiter_design.md) -> likely a slight UNDER-count.
                        A clean veto needs (a) this + a ~5% fragmentation systematic, or (b) a balfinder
                        re-run with a lowered BI onset floor. Physical finding robust; implementation open.
  AI>0(sig)           : any significant AI-CIV trough (>=450 km/s) — includes narrow non-DLA-mimicking

NOTE (2026-07-04): the numbers below are FULL-FOREST frontier illustrations. The headline ADOPTS the
broad-trough veto but applies it LYA-ONLY (lam_rf>=1025) — see lya_only_rerun.py Part A for the lya-only
leak/veto-cost and the headline re-derivation for the vetoed dN/dX & Ω. Under lya-only these full-forest
percentages shrink substantially (the excluded Lyβ-overlap region is the more BAL-contaminated part).

Key finding (SNR>2, >=20.3, FULL FOREST): broad-trough is the frontier knee — it removes the CIV-detectable
residual at a small ΔX cost over production, well below AI>0's cost for the same residual. Production BI>0
leaves a substantial broad-BAL leak (larger in the deep tail), and adopting broad-trough reduces high-N Ω
materially. Quantitative percentages are in the private notes: notes/2026-07-02_real_loa_bal_arbiter_design.md.

Env: source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate gpdla
Aggregate-only (real-LOA privacy). VAC = v2 (production BAL_FLAG source).
"""
import os, json, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fitsio

C_KMS = 299792.458; OMEGA_M = 0.279; LYA = 1215.6701; LYB = 1025.7222  # cddf_mock.py convention
V2 = "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v2-altbal.fits"
DLACAT = "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1/dlacat-loa-main-dark-v1.fits"


def _Xz():
    zg = np.linspace(0, 6, 60001)
    ig = (1 + zg) ** 2 / np.sqrt(OMEGA_M * (1 + zg) ** 3 + (1 - OMEGA_M))
    Xg = np.concatenate([[0.0], np.cumsum(0.5 * (ig[:-1] + ig[1:]) * np.diff(zg))])
    return lambda z: np.interp(z, zg, Xg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vac", default=V2); ap.add_argument("--dlacat", default=DLACAT)
    ap.add_argument("--vprox", type=float, default=10000.0, help="proximity velocity cut (km/s)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    X = _Xz()

    v = fitsio.read(a.vac); vt = np.asarray(v["TARGETID"], np.int64); vz = np.asarray(v["Z"], float)
    BI = np.asarray(v["BI_CIV"], float); AI = np.asarray(v["AI_CIV"], float); eAI = np.asarray(v["ERR_AI_CIV"], float)
    vmn = np.asarray(v["VMIN_CIV_450"], float); vmx = np.asarray(v["VMAX_CIV_450"], float)
    wid = np.abs(vmx - vmn); wid[(vmn == 0) & (vmx == 0)] = 0; widest = wid.max(axis=1)
    sig = (AI > 0) & (eAI > 0) & (AI > 3 * eAI)
    vetoes = {"BI>0 (production)": (BI > 0),
              "broad-trough>2000": sig & (widest > 2000),
              "AI>0(sig, any width)": sig}
    vset = {k: set(vt[m].tolist()) for k, m in vetoes.items()}

    # ΔX denominator: searched sightlines ~ VAC z>2.1
    srch = vz > 2.1
    zmax = vz - (1 + vz) * (a.vprox / C_KMS); zmin = (1 + vz) * LYB / LYA - 1
    dxsl = np.clip(X(zmax) - X(zmin), 0, None)
    DXtot = float(dxsl[srch].sum())
    DXrm = {k: float(sum(dxsl[i] for i in np.where(srch)[0] if int(vt[i]) in vset[k])) for k in vset}

    d = fitsio.read(a.dlacat); tid = np.asarray(d["TARGETID"], np.int64); nhi = np.asarray(d["NHI"], float)
    snr = np.asarray(d["SNR_REDSIDE"], float); p = np.asarray(d["P_DLA"], float)
    dlahost = set(tid[(snr > 2) & (p > 0.99) & (nhi >= 20.3)].tolist())
    field = srch & np.array([int(t) not in dlahost for t in vt])
    fld_broad = float(100 * (vetoes["broad-trough>2000"] & field).sum() / field.sum())

    M = {"searched_z2p1": int(srch.sum()), "DeltaX_total": DXtot, "field_broad_bal_rate_pct": round(fld_broad, 2),
         "sweep": {}}
    for lim in (20.3, 21.6):
        M["sweep"][f"ge{lim}"] = {}
        for vlab in vset:
            rows = []
            for s in (2, 3, 5):
                base = (snr > s) & (p > 0.99) & (nhi >= lim); idx = np.where(base)[0]
                w = 10.0 ** nhi[idx]; Wb = w.sum()
                flg = np.array([tid[i] in vset[vlab] for i in idx]); ret = ~flg; Wr = w[ret].sum()
                broad_ret = np.array([tid[i] in vset["broad-trough>2000"] for i in idx]) & ret
                resid = float(100 * w[broad_ret].sum() / Wr) if Wr else 0.0
                fDX = 100 * DXrm[vlab] / DXtot
                netO = (Wr / Wb) / ((DXtot - DXrm[vlab]) / DXtot)
                rows.append({"snr_min": s, "dX_removed_pct": round(fDX, 1), "omega_removed_pct": round(100 * (1 - Wr / Wb), 1),
                             "resid_broad_pct": round(resid, 1), "net_omega_mult": round(float(netO), 3), "n_retained": int(ret.sum())})
            M["sweep"][f"ge{lim}"][vlab] = rows

    with open(f"{a.outdir}/veto_snr_sweep.json", "w") as fh: json.dump(M, fh, indent=2)

    # --- frontier figure: residual leak vs ΔX cost (per veto), + net Ω× bars ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {"BI>0 (production)": "C1", "broad-trough>2000": "C2", "AI>0(sig, any width)": "C3"}
    for lim, mk in [(20.3, "o"), (21.6, "s")]:
        for vlab in vset:
            r = M["sweep"][f"ge{lim}"][vlab][0]  # SNR>2
            ax[0].scatter(r["dX_removed_pct"], r["resid_broad_pct"], s=90, marker=mk, color=colors[vlab],
                          label=f"{vlab} (≥{lim})" if lim == 20.3 else None, zorder=3, edgecolor="k", linewidth=0.5)
    ax[0].set_xlabel("ΔX cost — forest path removed (%)"); ax[0].set_ylabel("residual broad-BAL leak in retained set (%)")
    ax[0].set_title("Decision frontier (SNR>2): broad-trough is the knee\n○ ≥20.3  □ ≥21.6")
    ax[0].legend(fontsize=7, loc="upper right"); ax[0].grid(alpha=0.3)
    ax[0].annotate("broad-trough:\nfrontier knee", (7.8, 0), textcoords="offset points", xytext=(10, 25), fontsize=8,
                   arrowprops=dict(arrowstyle="->", color="C2"))
    x = np.arange(3); wbar = 0.25
    for j, s in enumerate((2, 3, 5)):
        vals = [M["sweep"]["ge20.3"][v][j]["net_omega_mult"] for v in vset]
        ax[1].bar(x + (j - 1) * wbar, vals, wbar, label=f"SNR>{s}")
    ax[1].axhline(1.0, ls="--", color="grey"); ax[1].set_xticks(x); ax[1].set_xticklabels(["BI>0", "broad>2000", "AI>0"], fontsize=8)
    ax[1].set_ylabel("net Ω(≥20.3) × vs no-veto"); ax[1].set_title("Ω(≥20.3) after veto (lower = more BAL removed)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/veto_snr_frontier.png", dpi=130); plt.close(fig)
    print(json.dumps(M["sweep"]["ge20.3"], indent=1))
    print(f"\nfield broad-BAL rate={fld_broad:.2f}%  ->  figures/veto_snr_frontier.png + veto_snr_sweep.json")


if __name__ == "__main__":
    main()
