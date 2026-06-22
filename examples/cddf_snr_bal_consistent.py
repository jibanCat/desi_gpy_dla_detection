#!/usr/bin/env python
"""Fully selection-consistent raw CDDF/dN/dX/Omega for LOA + matterhorn.

Both SNR and BAL are SIGHTLINE properties, so they must be applied to the parent
path ΔX as well as the numerator detections (else dN/dX is biased low):

  numerator (always): DLAFLAG==0 (excludes BAL bit4 + Lyβ bit3) & P_DLA>0.99
  SNR>2  : N(SNR>2 dets) / ΔX(sightlines: SNR>2 AND not-BAL)
  no-SNR : N(all dets)   / ΔX(sightlines: not-BAL, any SNR)

BAL sightlines (TARGETID in the BI_CIV>0 BAL catalog) are removed from BOTH the
detections (already, via DLAFLAG bit4) and the path ΔX. Raw = no completeness
correction. Per-sightline z_qso/min_z/max_z/snr/target_id from processed-*.h5
(cached). Supersedes cddf_snr_consistent.py (which dropped BAL only from the
numerator, not the path).
"""
import os, sys, glob, argparse, multiprocessing as mp
import numpy as np, h5py, fitsio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CDDF_analysis.cddf_mock import AbsorptionDistance, total_DeltaX_in_zbins, omega_hi_prefactor
from CDDF_analysis.dla_data import dla_data

# Set these to your own dlacat / processed-h5 dir / BAL-cat paths (or via env vars).
# Each CFG entry is (label, color, dlacat.fits, processed-h5-dir, bal-cat.fits).
# No real paths are committed.
CFG = [
    ("LOA", "C0",
     os.environ.get("LOA_DLACAT", "SET_loa_dlacat.fits"),
     os.environ.get("LOA_PROCDIR", "SET_loa_processed_dir"),
     os.environ.get("LOA_BALCAT", "SET_loa_bal_cat.fits")),
    ("matterhorn", "C3",
     os.environ.get("MH_DLACAT", "SET_matterhorn_dlacat.fits"),
     os.environ.get("MH_PROCDIR", "SET_matterhorn_processed_dir"),
     os.environ.get("MH_BALCAT", "SET_matterhorn_bal_cat.fits")),
]
OUT = os.environ.get("OUT_DIR", "SET_OUT_DIR")
CACHE = os.environ.get("PARENT_CACHE", "SET_PARENT_CACHE")

zbins = np.round(np.arange(2.0, 4.4 + 1e-9, 0.2), 3)
zmid = 0.5 * (zbins[:-1] + zbins[1:])
logN_bins = np.round(np.arange(20.0, 22.6 + 1e-9, 0.2), 3)
N_edges = 10.0 ** logN_bins; dN = np.diff(N_edges)
N_mid = np.sqrt(N_edges[:-1] * N_edges[1:]); logN_mid = np.log10(N_mid)
K = omega_hi_prefactor(70.0)

_KEYS = ("target_ids", "z_qsos", "min_z_dlas", "max_z_dlas", "snrs")
def _read(fp):
    try:
        with h5py.File(fp, "r") as H:
            if not all(k in H for k in _KEYS): return None
            return tuple(np.asarray(H[k][()]).ravel() for k in _KEYS)
    except Exception:
        return None

def parent(procdir, tag, nproc):
    os.makedirs(CACHE, exist_ok=True)
    cf = os.path.join(CACHE, tag.split()[0] + "_parent_v2.npz")   # v2 = keeps target_id
    if os.path.exists(cf):
        z = np.load(cf); print(f"[{tag}] cache {cf}", flush=True); return z["tid"], z["mn"], z["mx"], z["snr"]
    fs = sorted(glob.glob(procdir + "/processed-*.h5"))
    print(f"[{tag}] reading {len(fs)} h5 ({nproc} workers)...", flush=True)
    with mp.Pool(nproc) as p:
        r = [x for x in p.map(_read, fs, chunksize=16) if x is not None]
    tid = np.concatenate([x[0] for x in r]).astype(np.int64)
    mn = np.concatenate([x[2] for x in r]).astype(float)
    mx = np.concatenate([x[3] for x in r]).astype(float)
    sn = np.concatenate([x[4] for x in r]).astype(float)
    _, u = np.unique(tid, return_index=True)
    tid, mn, mx, sn = tid[u], mn[u], mx[u], sn[u]
    g = np.isfinite(mn) & np.isfinite(mx) & np.isfinite(sn) & (mx > mn)
    tid, mn, mx, sn = tid[g], mn[g], mx[g], sn[g]
    np.savez(cf, tid=tid, mn=mn, mx=mx, snr=sn)
    return tid, mn, mx, sn

def load_abs(fits):
    d = fitsio.read(fits, ext=1, columns=["DLAFLAG", "SNR_REDSIDE", "P_DLA", "NHI", "Z_DLA"])
    k = (np.asarray(d["DLAFLAG"]) == 0) & (np.asarray(d["P_DLA"], float) > 0.99)   # BAL-free (bit4)
    return np.asarray(d["Z_DLA"], float)[k], np.asarray(d["NHI"], float)[k], np.asarray(d["SNR_REDSIDE"], float)[k]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=32); a = ap.parse_args()
    D = []
    for label, color, fits, procdir, balcat in CFG:
        tid, mn, mx, sn = parent(procdir, label, a.nproc)
        bal = np.unique(fitsio.read(balcat, ext=1, columns=["TARGETID"])["TARGETID"].astype(np.int64))
        not_bal = ~np.isin(tid, bal)
        Xc = AbsorptionDistance(zmax=float(mx.max()), Omega_m=0.279)
        hi = (sn > 2) & not_bal          # SNR>2 AND non-BAL sightlines
        nb = not_bal                     # non-BAL, any SNR
        Xhi = total_DeltaX_in_zbins(zbins, mn[hi], mx[hi], Xc)
        Xall = total_DeltaX_in_zbins(zbins, mn[nb], mx[nb], Xc)
        z, nhi, snr = load_abs(fits)
        D.append(dict(label=label, color=color, z=z, nhi=nhi, snr=snr, Xhi=Xhi, Xall=Xall,
                      Xhi_full=float(Xhi.sum()), Xall_full=float(Xall.sum())))
        frac_bal = 1 - not_bal.mean()
        print(f"[{label}] BAL sightlines={100*frac_bal:.1f}% | non-BAL path | "
              f"SNR>2&nonBAL path frac of nonBAL={Xhi.sum()/Xall.sum():.3f} | dets(BAL-free)={len(z)}", flush=True)
    loa, mh = D

    def dndx(d, thr, sel):
        if sel == "hi":
            m = (d["nhi"] >= thr) & (d["snr"] > 2); X = d["Xhi"]
        else:
            m = (d["nhi"] >= thr); X = d["Xall"]
        N, _ = np.histogram(d["z"][m], bins=zbins)
        return N, np.where(X > 0, N / X, np.nan), np.where(X > 0, np.sqrt(N) / X, np.nan)

    # 1. dN/dX (SNR>2 vs no-SNR; non-BAL) + literature
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for d in D:
        for sel, ls, al, tag in [("hi", "-", 1.0, "SNR>2"), ("all", "--", 0.5, "no-SNR")]:
            _, dx, de = dndx(d, 20.3, sel)
            ax.errorbar(zmid, dx, yerr=de, color=d["color"], ls=ls, alpha=al, marker="o", ms=3,
                        capsize=2, label=f"{d['label']}, {tag}")
    plt.sca(ax); dla_data.dndx_not(); dla_data.dndx_pro()
    ax.set_xlabel("z"); ax.set_ylabel(r"$dN/dX$")
    ax.set_title("Raw dN/dX (NHI ≥ 20.3, non-BAL) — LOA & matterhorn vs N12/PW09")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "dndx_snr_bal_consistent.png"), dpi=140); plt.close(fig)

    # 2. CDDF (SNR>2 vs no-SNR; non-BAL) + literature
    def cddf(d, sel):
        if sel == "hi":
            m = (d["z"] >= 2.0) & (d["z"] < 4.4) & (d["nhi"] >= 20.0) & (d["snr"] > 2); X = d["Xhi_full"]
        else:
            m = (d["z"] >= 2.0) & (d["z"] < 4.4) & (d["nhi"] >= 20.0); X = d["Xall_full"]
        c, _ = np.histogram(d["nhi"][m], bins=logN_bins)
        return c, X
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for d in D:
        for sel, ls, al, tag in [("hi", "-", 1.0, "SNR>2"), ("all", "--", 0.5, "no-SNR")]:
            c, X = cddf(d, sel)
            fN = np.where(c > 0, c / (X * dN), np.nan); fe = np.where(c > 0, np.sqrt(c) / (X * dN), np.nan)
            ax.errorbar(N_mid, fN, yerr=fe, color=d["color"], ls=ls, alpha=al, marker="o", ms=4,
                        capsize=2, label=f"{d['label']}, {tag}")
    plt.sca(ax); dla_data.noterdaeme_12_data(); dla_data.ho21_cddf(redshift=-1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8e19, 5e22)
    ax.set_xlabel(r"$N_{HI}\ [\mathrm{cm}^{-2}]$"); ax.set_ylabel(r"$f(N_{HI},X)$")
    ax.set_title("Raw CDDF (NHI ≥ 20, non-BAL) — LOA & matterhorn vs N12/Ho21")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cddf_snr_bal_consistent.png"), dpi=140); plt.close(fig)

    # 3. Omega (SNR>2 & non-BAL proper vs no-SNR) + literature
    sel_n = (logN_mid >= 20.3) & (logN_mid <= 22.0)
    def omega(d, sel):
        if sel == "hi":
            m = d["snr"] > 2; X = d["Xhi"]
        else:
            m = np.ones(len(d["z"]), bool); X = d["Xall"]
        C2, _, _ = np.histogram2d(d["z"][m], d["nhi"][m], bins=[zbins, logN_bins])
        om = np.full(len(zmid), np.nan); oe = np.full(len(zmid), np.nan)
        for k in range(len(zmid)):
            if X[k] > 0:
                c = C2[k][sel_n]; Nm = N_mid[sel_n]
                om[k] = K * np.sum(Nm * c) / X[k] * 1000.0
                oe[k] = K * np.sqrt(np.sum(Nm ** 2 * c)) / X[k] * 1000.0
        return om, oe
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for d in D:
        for sel, ls, al, tag in [("hi", "-", 1.0, "SNR>2"), ("all", "--", 0.5, "no-SNR")]:
            om, oe = omega(d, sel)
            ax.errorbar(zmid, om, yerr=oe, color=d["color"], ls=ls, alpha=al, marker="o", ms=3,
                        capsize=2, label=f"{d['label']}, {tag}")
    plt.sca(ax); dla_data.omegahi_not(); dla_data.omegahi_pro(); dla_data.crighton_omega(); dla_data.xq100_omega()
    ax.set_xlabel("z"); ax.set_ylabel(r"$\Omega_{\rm DLA}\times10^{3}$")
    ax.set_xlim(1.9, 5.1); ax.set_ylim(0, 2.2)
    ax.set_title("Raw Ω_HI(z) (non-BAL) — LOA & matterhorn vs literature")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "omega_snr_bal_consistent.png"), dpi=140); plt.close(fig)

    with open(os.path.join(OUT, "snr_bal_consistent_summary.txt"), "w") as f:
        for d in D:
            f.write(f"{d['label']}: ΔX_nonBAL_all={d['Xall_full']:.4g}  ΔX_nonBAL_SNR2={d['Xhi_full']:.4g}  "
                    f"(SNR2 frac of nonBAL={d['Xhi_full']/d['Xall_full']:.3f})\n")
            for i,thr in enumerate((20.0, 20.3)):
                _, dh, _ = dndx(d, thr, "hi")
                f.write(f"   NHI>={thr} dN/dX(SNR>2,nonBAL) per z: " +
                        " ".join(f"{v:.4f}" for v in dh) + "\n")
    print("[done] wrote SNR+BAL-consistent plots + summary to", OUT, flush=True)


if __name__ == "__main__":
    main()
