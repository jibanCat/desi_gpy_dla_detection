#!/usr/bin/env python
"""Raw (uncorrected) CDDF f(N,z), dN/dX(z) and histogram diagnostics for a
finished GP-DLA catalog.

RAW = direct catalog measurement at the baseline operating point, with NO
completeness/alpha correction. Denominator (absorption distance ΔX) uses the
catalog's OWN per-sightline search window (min_z_dlas, max_z_dlas) read from the
processed-*.h5 parent sample — so numerator and denominator are self-consistent.

Baseline selection (the documented operating point):
    DLAFLAG == 0  &  SNR_REDSIDE > 2  &  P_DLA > 0.99
CDDF / dN/dX use NHI >= 20 on top of that.

Reuses CDDF_analysis.cddf_mock for the LCDM path-length (Omega_m=0.279) and the
per-z-bin ΔX accumulation (the tested engine).

Usage:
    python examples/cddf_dndx_raw_diagnostics.py \
        --dlacat <dlacat.fits> --procdir <run>/outputs/figures/processed \
        --out <bundle>/diagnostics/raw_cddf_dndx --label "matterhorn (real)"
"""
import os, sys, glob, argparse, multiprocessing as mp
import numpy as np
import h5py, fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CDDF_analysis.cddf_mock import AbsorptionDistance, total_DeltaX_in_zbins
from CDDF_analysis.dla_data import dla_data   # literature overlays (sbird/DLA_data)

# ----------------------------- parent sample (h5) ---------------------------
_KEYS = ("target_ids", "z_qsos", "min_z_dlas", "max_z_dlas")

def _read_one(fp):
    try:
        with h5py.File(fp, "r") as H:
            if not all(k in H for k in _KEYS):
                return None
            return (np.asarray(H["target_ids"][()]).ravel(),
                    np.asarray(H["z_qsos"][()]).ravel(),
                    np.asarray(H["min_z_dlas"][()]).ravel(),
                    np.asarray(H["max_z_dlas"][()]).ravel())
    except Exception:
        return None

def load_parent(procdir, nproc):
    files = sorted(glob.glob(os.path.join(procdir, "processed-*.h5")))
    print(f"[parent] reading {len(files)} h5 with {nproc} workers ...", flush=True)
    with mp.Pool(nproc) as pool:
        res = [r for r in pool.map(_read_one, files, chunksize=16) if r is not None]
    tid = np.concatenate([r[0] for r in res]).astype(np.int64)
    zq  = np.concatenate([r[1] for r in res]).astype(float)
    mn  = np.concatenate([r[2] for r in res]).astype(float)
    mx  = np.concatenate([r[3] for r in res]).astype(float)
    # dedupe by TARGETID (healpix are disjoint; guard anyway), keep first
    _, uidx = np.unique(tid, return_index=True)
    tid, zq, mn, mx = tid[uidx], zq[uidx], mn[uidx], mx[uidx]
    good = np.isfinite(zq) & np.isfinite(mn) & np.isfinite(mx) & (mx > mn)
    print(f"[parent] {len(tid)} unique sightlines ({int(good.sum())} with valid window)", flush=True)
    return tid[good], zq[good], mn[good], mx[good]

# ------------------------------ main ----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlacat", required=True)
    ap.add_argument("--procdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--omega-m", type=float, default=0.279)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- parent sample + per-sightline search window ----
    s_tid, s_zq, s_mn, s_mx = load_parent(a.procdir, a.nproc)
    order = np.argsort(s_tid)
    s_tid_s, s_mn_s, s_mx_s = s_tid[order], s_mn[order], s_mx[order]

    # ---- absorber catalog + baseline cut ----
    d = fitsio.read(a.dlacat, ext=1,
                    columns=["TARGETID", "DLAFLAG", "SNR_REDSIDE", "P_DLA", "NHI", "Z_DLA"])
    dflag = np.asarray(d["DLAFLAG"]); snr = np.asarray(d["SNR_REDSIDE"], float)
    pdla = np.asarray(d["P_DLA"], float); nhi = np.asarray(d["NHI"], float)
    zdla = np.asarray(d["Z_DLA"], float); atid = np.asarray(d["TARGETID"]).astype(np.int64)
    base = (dflag == 0) & (snr > 2.0) & (pdla > 0.99)
    print(f"[abs] rows={len(d)} baseline={int(base.sum())} (+NHI>=20: {int((base&(nhi>=20)).sum())})", flush=True)

    # restrict baseline absorbers to within their sightline search window (numerator==denominator path)
    b = np.where(base)[0]
    pos = np.searchsorted(s_tid_s, atid[b])
    pos = np.clip(pos, 0, len(s_tid_s) - 1)
    in_sample = s_tid_s[pos] == atid[b]
    win_ok = in_sample & (zdla[b] >= s_mn_s[pos]) & (zdla[b] <= s_mx_s[pos])
    keep = b[win_ok]
    print(f"[abs] baseline within search window: {len(keep)} (dropped {int(base.sum())-len(keep)} off-window/orphan)", flush=True)
    z_b, nhi_b = zdla[keep], nhi[keep]

    # ---- cosmology / absorption distance ----
    Xcalc = AbsorptionDistance(zmax=float(np.max(s_mx)), Omega_m=a.omega_m)

    # =========================== dN/dX vs z =================================
    zbins = np.round(np.arange(2.0, 4.4 + 1e-9, 0.2), 3)
    zmid = 0.5 * (zbins[:-1] + zbins[1:])
    Xtot = total_DeltaX_in_zbins(zbins, s_mn, s_mx, Xcalc)   # ΔX per z-bin (whole parent path)
    series = {"NHI≥20": nhi_b >= 20.0, "NHI≥20.3": nhi_b >= 20.3, "NHI≥21": nhi_b >= 21.0}
    fig, ax = plt.subplots(figsize=(7, 5))
    rows = []
    for lab, m in series.items():
        N, _ = np.histogram(z_b[m], bins=zbins)
        dndx = np.where(Xtot > 0, N / Xtot, np.nan)
        err = np.where(Xtot > 0, np.sqrt(N) / Xtot, np.nan)
        ax.errorbar(zmid, dndx, yerr=err, marker="o", ms=4, capsize=2, label=lab)
        if lab == "NHI≥20":
            for i in range(len(zmid)):
                rows.append((zmid[i], zbins[i], zbins[i+1], int(N[i]), Xtot[i],
                             dndx[i], err[i]))
    plt.sca(ax)                                  # literature dN/dX overlays
    dla_data.dndx_not()                          # Noterdaeme+2012 (N12)
    dla_data.dndx_pro()                          # Prochaska & Wolfe 2009 (PW09)
    ax.set_xlabel("z"); ax.set_ylabel(r"$dN/dX$")
    ax.set_title(f"Raw dN/dX (no completeness correction) vs literature\n{a.label}")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "dndx_vs_z.png"), dpi=130); plt.close(fig)

    # =========================== CDDF f(N,z) ================================
    # CDDF on linear-N log-log axes to match the dla_data literature overlays
    # (noterdaeme_12_data / ho21_cddf plot f(N) vs N in cm^-2).
    logN_bins = np.round(np.arange(20.0, 22.6 + 1e-9, 0.2), 3)
    N_edges = 10.0 ** logN_bins
    dN = np.diff(N_edges)
    N_mid = np.sqrt(N_edges[:-1] * N_edges[1:])           # linear bin center
    logN_mid = np.log10(N_mid)
    cddf_zbins = np.array([2.0, 2.5, 3.0, 3.5, 4.4])
    cddf_rows = []

    # (a) per-z-bin CDDF + N12 reference
    fig, ax = plt.subplots(figsize=(7.6, 5.7))
    for k in range(len(cddf_zbins) - 1):
        zlo, zhi = cddf_zbins[k], cddf_zbins[k+1]
        Xk = total_DeltaX_in_zbins(np.array([zlo, zhi]), s_mn, s_mx, Xcalc)[0]
        m = (z_b >= zlo) & (z_b < zhi) & (nhi_b >= 20.0)
        counts, _ = np.histogram(nhi_b[m], bins=logN_bins)
        fN = np.where(counts > 0, counts / (Xk * dN), np.nan)
        ferr = np.where(counts > 0, np.sqrt(counts) / (Xk * dN), np.nan)
        ax.errorbar(N_mid, fN, yerr=ferr, marker="o", ms=4, capsize=2,
                    label=f"{zlo:.1f}≤z<{zhi:.1f} (N={int(counts.sum())})")
        for i in range(len(N_mid)):
            cddf_rows.append((zlo, zhi, logN_mid[i], int(counts[i]), Xk,
                              fN[i] if counts[i] > 0 else np.nan))
    plt.sca(ax); dla_data.noterdaeme_12_data()            # N12 (z=2-3.5) reference
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8e19, 5e22)
    ax.set_xlabel(r"$N_{HI}\ [\mathrm{cm}^{-2}]$"); ax.set_ylabel(r"$f(N_{HI}, X)$")
    ax.set_title(f"Raw CDDF by z, NHI≥20 (no corr.) vs N12\n{a.label}")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "cddf_fN_zbins.png"), dpi=130); plt.close(fig)

    # (b) combined-z CDDF vs literature (N12 + Ho21)
    Xall = total_DeltaX_in_zbins(np.array([2.0, 4.4]), s_mn, s_mx, Xcalc)[0]
    mall = (z_b >= 2.0) & (z_b < 4.4) & (nhi_b >= 20.0)
    callc, _ = np.histogram(nhi_b[mall], bins=logN_bins)
    fNall = np.where(callc > 0, callc / (Xall * dN), np.nan)
    fErrall = np.where(callc > 0, np.sqrt(callc) / (Xall * dN), np.nan)
    fig, ax = plt.subplots(figsize=(7.6, 5.7))
    ax.errorbar(N_mid, fNall, yerr=fErrall, marker="o", ms=5, capsize=2, color="C0",
                label="this work (2.0≤z<4.4, raw)")
    plt.sca(ax); dla_data.noterdaeme_12_data(); dla_data.ho21_cddf(redshift=-1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8e19, 5e22)
    ax.set_xlabel(r"$N_{HI}\ [\mathrm{cm}^{-2}]$"); ax.set_ylabel(r"$f(N_{HI}, X)$")
    ax.set_title(f"Raw CDDF NHI≥20 (no corr.) vs N12 + Ho21\n{a.label}")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "cddf_fN_vs_literature.png"), dpi=130); plt.close(fig)

    # =========================== histograms =================================
    snr_ok = (dflag == 0) & (snr > 2.0)            # pre-pDLA-cut, for P_DLA panel
    flag0  = (dflag == 0)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.hist(nhi_b, bins=np.arange(19.5, 22.8, 0.1), color="steelblue")
    ax.axvline(20.0, color="k", ls="--", lw=1, label="NHI=20.0")
    ax.axvline(20.3, color="grey", ls=":", lw=1, label="NHI=20.3")
    ax.set_xlabel(r"$\log_{10} N_{HI}$"); ax.set_ylabel("count")
    ax.set_title(f"NHI (baseline, in-window)\n{a.label}"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "hist_nhi.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.hist(z_b[nhi_b >= 20.0], bins=np.arange(1.9, 4.5, 0.1), color="indianred")
    ax.set_xlabel("z_DLA"); ax.set_ylabel("count")
    ax.set_title(f"z_DLA (baseline, NHI≥20)\n{a.label}"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "hist_zdla.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.hist(pdla[snr_ok], bins=np.linspace(0, 1, 51), color="seagreen")
    ax.axvline(0.99, color="k", ls="--", lw=1, label="P_DLA=0.99 cut")
    ax.set_yscale("log"); ax.set_xlabel("P_DLA"); ax.set_ylabel("count")
    ax.set_title(f"P_DLA (DLAFLAG=0 & SNR_RED>2)\n{a.label}"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "hist_pdla.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sv = snr[flag0]; sv = sv[(sv > -5) & (sv < 30)]
    ax.hist(sv, bins=np.arange(-5, 30, 0.5), color="slateblue")
    ax.axvline(2.0, color="k", ls="--", lw=1, label="SNR_RED=2 cut")
    ax.set_xlabel("SNR_REDSIDE"); ax.set_ylabel("count")
    ax.set_title(f"SNR_REDSIDE (DLAFLAG=0)\n{a.label}"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "hist_snr.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.hist(s_zq, bins=np.arange(1.9, 4.5, 0.05), color="darkgray", label="parent QSO z")
    ax.hist(z_b[nhi_b >= 20.0], bins=np.arange(1.9, 4.5, 0.05), color="indianred",
            alpha=.7, label="DLA z (NHI≥20)")
    ax.set_xlabel("z"); ax.set_ylabel("count"); ax.legend()
    ax.set_title(f"Parent QSO z vs DLA z\n{a.label}"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "hist_zqso_vs_zdla.png"), dpi=130); plt.close(fig)

    # DLAs per sightline (baseline NHI>=20)
    kt = atid[keep][nhi_b >= 20.0]
    _, cps = np.unique(kt, return_counts=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.hist(cps, bins=np.arange(0.5, max(6, cps.max() + 1.5), 1.0), color="teal")
    ax.set_xlabel("# DLAs per sightline (NHI≥20)"); ax.set_ylabel("# sightlines")
    ax.set_yscale("log"); ax.set_title(f"DLAs per sightline\n{a.label}"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "hist_ndla_per_sightline.png"), dpi=130); plt.close(fig)

    # 2D NHI vs z
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    m20 = nhi_b >= 20.0
    hb = ax.hist2d(z_b[m20], nhi_b[m20], bins=[np.arange(1.9, 4.45, 0.1), np.arange(20.0, 22.6, 0.1)],
                   cmap="viridis")
    fig.colorbar(hb[3], ax=ax, label="count")
    ax.set_xlabel("z_DLA"); ax.set_ylabel(r"$\log_{10} N_{HI}$")
    ax.set_title(f"NHI vs z (baseline, NHI≥20)\n{a.label}")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "nhi_vs_z_2d.png"), dpi=130); plt.close(fig)

    # =========================== summary tsv ================================
    with open(os.path.join(a.out, "summary.tsv"), "w") as f:
        f.write(f"# RAW CDDF/dNdX diagnostics — {a.label}\n")
        f.write(f"# catalog: {a.dlacat}\n")
        f.write("# selection: DLAFLAG==0 & SNR_REDSIDE>2 & P_DLA>0.99 ; CDDF/dNdX add NHI>=20 ; NO completeness correction\n")
        f.write(f"# parent_sightlines\t{len(s_tid)}\n")
        f.write(f"# baseline_absorbers\t{int(base.sum())}\tin_window\t{len(keep)}\tNHI>=20\t{int((nhi_b>=20).sum())}\n")
        f.write("## dN/dX (NHI>=20)\nz_mid\tz_lo\tz_hi\tN_abs\tX_tot\tdNdX\terr\n")
        for r in rows:
            f.write("\t".join(f"{x:.6g}" for x in r) + "\n")
        f.write("## CDDF f(N,z) (NHI>=20)\nz_lo\tz_hi\tlogN_mid\tcount\tX_tot\tfN\n")
        for r in cddf_rows:
            f.write("\t".join(("{:.6g}".format(x) if isinstance(x, float) else str(x)) for x in r) + "\n")
    print(f"[done] wrote plots + summary.tsv to {a.out}", flush=True)

if __name__ == "__main__":
    main()
