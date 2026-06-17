#!/usr/bin/env python
"""Cross-method raw CDDF / dN/dX comparison on matterhorn: GP-DLA vs DLA-toolkit
vs CNN. Common ΔX (matterhorn parent path), so only the numerator (each method's
detections) differs. Raw, no completeness correction.

Common footing for all three methods:
  - parent: matterhorn GP-processed sightlines (v3 cache), restricted to SNR>2 &
    non-BAL (SNR = red-side, a per-spectrum property shared by all methods; BAL =
    TARGETID in the matterhorn BI_CIV>0 catalog).
  - ΔX over those sightlines' GP search window [min_z_dla, max_z_dla], or the
    Lyα-only sub-window [max(min_z, (λLyβ/λLyα)(1+z_qso)-1), max_z] with --lya.
  - each method's detections kept only if: TARGETID in that parent, z_dla within
    the (same) window, NHI>=20.3 (dN/dX) / >=20 (CDDF), + the method's own cut.

Method cuts (first-pass; tune via --cnn-conf):
  GP        : DLAFLAG==0 & P_DLA>0.99
  DLAtoolkit: the published "good" catalog as-is
  CNN       : ABSORBER_TYPE=='DLA' & DLA_CONFIDENCE > cnn_conf (default 0.5)

Usage:  python examples/cddf_method_compare.py [--cnn-conf 0.5] [--lya]
"""
import os, sys, argparse
import numpy as np, fitsio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CDDF_analysis.cddf_mock import AbsorptionDistance, total_DeltaX_in_zbins
from CDDF_analysis.dla_data import dla_data

# Set these to your own parent-cache / BAL-cat / per-method dlacat / output paths
# (or via the corresponding environment variables). No real paths are committed.
CACHE  = os.environ.get("PARENT_CACHE", "SET_PARENT_CACHE.npz")
BALCAT = os.environ.get("BAL_CAT", "SET_BAL_CAT.fits")
GP   = os.environ.get("GP_DLACAT", "SET_GP_DLACAT.fits")
DT   = os.environ.get("DT_DLACAT", "SET_DLATOOLKIT_DLACAT.fits")
CNN  = os.environ.get("CNN_DLACAT", "SET_CNN_DLACAT.fits")
OUT  = os.environ.get("OUT_DIR", "SET_OUT_DIR")

LYA = 1215.6701; LYB = 1025.7223; R = LYB / LYA            # Lyα-forest blue edge factor
zbins = np.round(np.arange(2.0, 4.4 + 1e-9, 0.2), 3); zmid = 0.5 * (zbins[:-1] + zbins[1:])
logN_bins = np.round(np.arange(20.0, 22.6 + 1e-9, 0.2), 3)
N_edges = 10.0 ** logN_bins; dN = np.diff(N_edges); N_mid = np.sqrt(N_edges[:-1] * N_edges[1:])


def load_method(name, conf, dt_dc2):
    if name == "GP":
        d = fitsio.read(GP, ext=1, columns=["TARGETID", "DLAFLAG", "P_DLA", "NHI", "Z_DLA"])
        k = (d["DLAFLAG"] == 0) & (d["P_DLA"] > 0.99)
    elif name == "DLAtoolkit":
        # "good" = DLAFLAG==0; recommended cut adds DELTACHI2>0.3 (Brodzeller+2025).
        d = fitsio.read(DT, ext=1, columns=["TARGETID", "NHI", "Z_DLA", "DELTACHI2"])
        k = d["DELTACHI2"] > dt_dc2
    elif name == "CNN":
        # DLA + SUBDLA (exclude LYB = Lyβ-misIDs). CNN labels NHI<20.3 as SUBDLA, so
        # this is needed for the CDDF to include CNN's NHI=20-20.3 absorbers; the
        # NHI>=20.3 dN/dX is unaffected (only DLAs survive that floor).
        d = fitsio.read(CNN, ext=1, columns=["TARGETID", "NHI", "Z_DLA", "DLA_CONFIDENCE", "ABSORBER_TYPE"])
        k = (d["ABSORBER_TYPE"] != "LYB") & (d["DLA_CONFIDENCE"] > conf)
    return (np.asarray(d["TARGETID"])[k].astype(np.int64),
            np.asarray(d["Z_DLA"], float)[k], np.asarray(d["NHI"], float)[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-conf", type=float, default=0.5)
    ap.add_argument("--dt-dc2", type=float, default=0.3,
                    help="DLA-toolkit DELTACHI2 cut (Brodzeller+2025 recommends 0.3 for DR1)")
    ap.add_argument("--lya", action="store_true", help="Lyα-only window (blue edge at QSO Lyβ emission)")
    ap.add_argument("--nhi-min", type=float, default=20.3,
                    help="dN/dX NHI floor (default 20.3 = DLA def; DLA-toolkit searches 20.1-22.5)")
    ap.add_argument("--dt-nothr", action="store_true",
                    help="also overlay the DLA-toolkit with NO DELTACHI2 cut as a dashed line")
    a = ap.parse_args()

    def label(name):
        if name == "CNN": return f"CNN (conf>{a.cnn_conf:g})"
        if name == "DLAtoolkit": return f"DLA-toolkit (Δχ²>{a.dt_dc2:g})"
        return name
    os.makedirs(OUT, exist_ok=True)
    sfx = ("_lya" if a.lya else "") + f"_dt{a.dt_dc2:g}" + ("" if a.nhi_min == 20.3 else f"_nhi{a.nhi_min:g}") + ("_wnothr" if a.dt_nothr else "")
    wtag = "Lyα-only, " if a.lya else ""

    # ---- common parent + ΔX (SNR>2 & non-BAL; window: full or Lyα-only) ----
    P = np.load(CACHE); tid = P["tid"]; zq = P["zq"]; mn = P["mn"]; mx = P["mx"]; snr = P["snr"]
    bal = np.unique(fitsio.read(BALCAT, ext=1, columns=["TARGETID"])["TARGETID"].astype(np.int64))
    not_bal = ~np.isin(tid, bal)
    wlo = np.maximum(mn, R * (1 + zq) - 1) if a.lya else mn       # blue edge
    keepS = (snr > 2) & not_bal & (mx > wlo)
    Xc = AbsorptionDistance(zmax=float(mx.max()), Omega_m=0.279)
    Xz = total_DeltaX_in_zbins(zbins, wlo[keepS], mx[keepS], Xc)
    Xfull = float(Xz.sum())
    o = np.argsort(tid); st = tid[o]; swlo = wlo[o]; smx = mx[o]; ssn = snr[o]; snb = not_bal[o]
    print(f"parent SNR>2 & non-BAL: {int(keepS.sum())} sightlines | window={'Lyα-only' if a.lya else 'full'} | ΔX_full={Xfull:.4g}", flush=True)

    def restrict(t, z, nhi):
        pos = np.clip(np.searchsorted(st, t), 0, len(st) - 1)
        keep = (st[pos] == t) & (ssn[pos] > 2) & snb[pos] & (z >= swlo[pos]) & (z <= smx[pos])
        return z[keep], nhi[keep]

    METH = [("GP", "C0"), ("DLAtoolkit", "C1"), ("CNN", "C2")]
    data = {}
    for name, col in METH:
        t, z, nhi = load_method(name, a.cnn_conf, a.dt_dc2)
        z, nhi = restrict(t, z, nhi)
        data[name] = (z, nhi, col)
        print(f"[{name}] in common parent+window: {len(z)} (NHI>=20.3: {int((nhi>=20.3).sum())})", flush=True)

    # optional: DLA-toolkit with NO DELTACHI2 cut, as a dashed reference line
    nothr = None
    if a.dt_nothr:
        t, z, nhi = load_method("DLAtoolkit", a.cnn_conf, 0.0)
        nothr = restrict(t, z, nhi)
        print(f"[DLAtoolkit no-dchi2] in common parent+window: {len(nothr[0])} "
              f"(NHI>=20.3: {int((nothr[1] >= 20.3).sum())})", flush=True)

    # ---- dN/dX (NHI>=20.3) ----
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for name, col in METH:
        z, nhi, _ = data[name]
        N, _ = np.histogram(z[nhi >= a.nhi_min], bins=zbins)
        dx = np.where(Xz > 0, N / Xz, np.nan); de = np.where(Xz > 0, np.sqrt(N) / Xz, np.nan)
        lab = label(name)
        ax.errorbar(zmid, dx, yerr=de, color=col, marker="o", ms=4, capsize=2, label=lab)
    if nothr is not None:
        zz, nn = nothr; N, _ = np.histogram(zz[nn >= a.nhi_min], bins=zbins)
        ax.errorbar(zmid, np.where(Xz > 0, N / Xz, np.nan), color="C1", ls="--", alpha=0.55,
                    marker="o", ms=3, label="DLA-toolkit (no Δχ² cut)")
    plt.sca(ax); dla_data.dndx_not(); dla_data.dndx_pro()
    ax.set_xlabel("z"); ax.set_ylabel(r"$dN/dX$")
    ax.set_title(f"Raw dN/dX (matterhorn, NHI ≥ {a.nhi_min:g}, {wtag}SNR>2 & non-BAL, common ΔX)\nGP vs DLA-toolkit vs CNN, + N12/PW09", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"dndx_method_compare{sfx}.png"), dpi=140); plt.close(fig)

    # ---- CDDF (NHI>=20) ----
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for name, col in METH:
        z, nhi, _ = data[name]
        # DLA-toolkit floors at NHI=20.1, so the standard [20.0,20.2] bin is half-empty
        # and artificially low — bin the toolkit from 20.1 (remaining edges aligned).
        lb = np.concatenate([[20.1], logN_bins[1:]]) if name == "DLAtoolkit" else logN_bins
        Ne = 10.0 ** lb; dNl = np.diff(Ne); Nm = np.sqrt(Ne[:-1] * Ne[1:])
        m = (z >= 2.0) & (z < 4.4) & (nhi >= lb[0])
        c, _ = np.histogram(nhi[m], bins=lb)
        fN = np.where(c > 0, c / (Xfull * dNl), np.nan); fe = np.where(c > 0, np.sqrt(c) / (Xfull * dNl), np.nan)
        ax.errorbar(Nm, fN, yerr=fe, color=col, marker="o", ms=4, capsize=2, label=label(name))
    if nothr is not None:
        zz, nn = nothr; lb = np.concatenate([[20.1], logN_bins[1:]])
        Ne = 10.0 ** lb; dNl = np.diff(Ne); Nm2 = np.sqrt(Ne[:-1] * Ne[1:])
        c, _ = np.histogram(nn[(zz >= 2.0) & (zz < 4.4) & (nn >= 20.1)], bins=lb)
        ax.errorbar(Nm2, np.where(c > 0, c / (Xfull * dNl), np.nan), color="C1", ls="--", alpha=0.55,
                    marker="o", ms=3, label="DLA-toolkit (no Δχ² cut)")
    plt.sca(ax); dla_data.noterdaeme_12_data(); dla_data.ho21_cddf(redshift=-1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8e19, 5e22)
    ax.set_xlabel(r"$N_{HI}\ [\mathrm{cm}^{-2}]$"); ax.set_ylabel(r"$f(N_{HI},X)$")
    ax.set_title(f"Raw CDDF (matterhorn, NHI ≥ 20, {wtag}SNR>2 & non-BAL, common ΔX)\nGP vs DLA-toolkit vs CNN, + N12/Ho21", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"cddf_method_compare{sfx}.png"), dpi=140); plt.close(fig)

    with open(os.path.join(OUT, f"method_compare_summary{sfx}.txt"), "w") as f:
        f.write(f"matterhorn DLA-method comparison (raw, common ΔX SNR>2 & non-BAL, "
                f"{'Lyα-only' if a.lya else 'full'} window)\n")
        f.write(f"cuts: GP DLAFLAG==0 & P_DLA>0.99 | DLAtoolkit DELTACHI2>{a.dt_dc2} | "
                f"CNN DLA+SUBDLA(excl LYB) DLA_CONFIDENCE>{a.cnn_conf} | dN/dX floor NHI>={a.nhi_min:g}\n")
        f.write(f"ΔX_full(2.0-4.4)={Xfull:.5g}\n")
        for name, col in METH:
            z, nhi, _ = data[name]
            N20, _ = np.histogram(z[nhi >= a.nhi_min], bins=zbins)
            f.write(f"{name}: NHI>={a.nhi_min:g} in-window = {int((nhi>=a.nhi_min).sum())} | dN/dX per z: "
                    + " ".join(f"{v:.4f}" for v in np.where(Xz > 0, N20 / Xz, np.nan)) + "\n")
    print(f"[done] wrote method-comparison plots{sfx} + summary to", OUT, flush=True)


if __name__ == "__main__":
    main()
