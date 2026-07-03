#!/usr/bin/env python
"""stack_ai_minibal.py — is the un-vetoed AI>0 mini-BAL high-N population causal FP or benign?

Stacks the clean (BI=0, DLAFLAG==0) lya-only high-N (logN>=20.3) DLA detections, split by whether
the sightline is an AI>0 mini-BAL (our own VAC) vs AI=0 control, in TWO rest frames:
  - DLA frame  (λ_rest = λ_obs/(1+Z_DLA)): a REAL DLA shows a damped Lyα Voigt + metals at the DLA z;
    a BAL-caused FP does NOT (no real absorber there).
  - QSO frame  (λ_rest = λ_obs/(1+Z_QSO)): a BAL shows a broad blueshifted CIV/SiIV trough near the
    QSO emission; a real DLA's CIV is at the DLA z and smears out.
Plus a z-SCRAMBLED control (shuffled Z_DLA): a real absorber's feature vanishes under scrambling; a
QSO-frame BAL feature survives.

Spectra from the compressed LOA archive (pre-coadded 3600-9824 A). Aggregate-only (stacked profiles).
Env: source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate gpdla
"""
import os, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import h5py, fitsio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

LYA = 1215.67
ARCHIVE = "/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5"
OUR_VAC = "/scratch/cavestru_root/cavestru0/mfho/our_loa_bal_vac/our_loa_bal_vac_v1.fits"
REAL_DLA = "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1/dlacat-loa-main-dark-v1.fits"


def norm_flux(wobs, flx, ivr, zdla):
    """continuum-normalise by the median flux redward of the DLA Lyα (line-free-ish)."""
    red = (wobs > LYA * (1 + zdla) + 30) & (wobs < LYA * (1 + zdla) + 200) & (ivr > 0)
    c = np.median(flx[red]) if red.sum() > 20 else np.nan
    return flx / c if (c and np.isfinite(c) and c > 0) else None


def stack_frame(rows, spec, wave, zsel, rest_grid, scramble=False, rng=None):
    """stack normalised flux on a common rest grid; rest = wobs/(1+z). spec = {tid:(flux,ivar)}."""
    acc = np.zeros(len(rest_grid)); wsum = np.zeros(len(rest_grid))
    zs = [z for (_, z, zd) in rows]
    if scramble:
        zs = list(np.asarray([zd for (_, _, zd) in rows])[rng.permutation(len(rows))])  # scramble z_DLA
    for (tid, zq, zd), zuse in zip(rows, zs):
        fi = spec.get(int(tid))
        if fi is None: continue
        flx, ivr = fi
        nf = norm_flux(wave, flx, ivr, zd)
        if nf is None: continue
        zframe = zq if zsel == "qso" else (zuse if scramble else zd)
        rest = wave / (1 + zframe)
        good = ivr > 0
        vals = np.interp(rest_grid, rest[good], nf[good], left=np.nan, right=np.nan)
        m = np.isfinite(vals); acc[m] += vals[m]; wsum[m] += 1
    return np.where(wsum > 20, acc / np.maximum(wsum, 1), np.nan), wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=4000, help="per-group subsample")
    ap.add_argument("--nhi-min", type=float, default=20.3)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(0)

    ours = fitsio.read(OUR_VAC)
    aiset = {int(t) for t, x in zip(ours["TARGETID"], ours["AI_CIV"]) if x > 0}
    biset = {int(t) for t, x in zip(ours["TARGETID"], ours["BI_CIV"]) if x > 0}
    d = fitsio.read(REAL_DLA)
    tid = np.asarray(d["TARGETID"], np.int64); nhi = np.asarray(d["NHI"], float)
    zd = np.asarray(d["Z_DLA"], float); zq = np.asarray(d["Z_QSO"], float)
    snr = np.asarray(d["SNR_REDSIDE"], float); p = np.asarray(d["P_DLA"], float); fl = np.asarray(d["DLAFLAG"], int)
    lam = LYA * (1 + zd) / (1 + zq)
    clean = (snr > 2) & (p > 0.99) & (lam >= 1025) & (fl == 0) & (nhi >= a.nhi_min)
    ii = np.where(clean)[0]
    is_ai = np.array([int(tid[i]) in aiset for i in ii])
    grp = {"AI>0 (mini-BAL)": ii[is_ai], "AI=0 (control DLA)": ii[~is_ai]}

    rest_dla = np.arange(1180, 1265, 0.4)   # Lyα damped profile region (DLA frame)
    rest_civ = np.arange(1480, 1580, 0.4)   # CIV region (QSO frame)
    grp_rows = {}
    with h5py.File(ARCHIVE, "r") as H:
        wave = H["wavelength"][:]
        arc_tid = {int(t): k for k, t in enumerate(H["catalog"]["TARGETID"])}
        # collect the subsampled rows per group + the archive indices to read
        need = {}
        for gname, gidx in grp.items():
            sub = gidx if len(gidx) <= a.nmax else rng.choice(gidx, a.nmax, replace=False)
            grp_rows[gname] = [(int(tid[i]), float(zq[i]), float(zd[i])) for i in sub]
            for (t, _, _) in grp_rows[gname]:
                j = arc_tid.get(int(t))
                if j is not None: need[int(t)] = j
        order = sorted(need.items(), key=lambda kv: kv[1])   # sort by archive index for fast fancy-read
        tids = [t for t, _ in order]; jj = np.array([j for _, j in order])
        print(f"  batch-reading {len(jj)} spectra (sorted) ...", flush=True)
        FL = H["flux"][jj]; IV = H["ivar"][jj]                # ONE sorted fancy-index read each
        spec = {t: (FL[k], IV[k]) for k, t in enumerate(tids)}
    stacks = {}
    for gname, rows in grp_rows.items():
        print(f"  {gname}: stacking {len(rows)} spectra ...", flush=True)
        sdla, _ = stack_frame(rows, spec, wave, "dla", rest_dla)
        sciv, _ = stack_frame(rows, spec, wave, "qso", rest_civ)
        sscr, _ = stack_frame(rows, spec, wave, "dla", rest_dla, scramble=True, rng=rng)
        stacks[gname] = (sdla, sciv, sscr)

    # --- figures ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    cols = {"AI>0 (mini-BAL)": "C3", "AI=0 (control DLA)": "C0"}
    for g, (sdla, sciv, sscr) in stacks.items():
        ax[0].plot(rest_dla, sdla, color=cols[g], lw=1.5, label=g)
        ax[1].plot(rest_civ, sciv, color=cols[g], lw=1.5, label=g)
        ax[2].plot(rest_dla, sscr, color=cols[g], lw=1.2, ls="--", label=g + " (z-scrambled)")
    ax[0].axvline(LYA, color="grey", ls=":"); ax[0].set_title("DLA-frame Lyα (damped Voigt?)\nreal DLA = deep damped core + wings")
    ax[0].set_xlabel("rest λ [Å], DLA frame"); ax[0].set_ylabel("normalised flux"); ax[0].legend(fontsize=8)
    for x, lbl in [(1548.2, "CIV"), (1550.8, "")]:
        ax[1].axvline(x, color="grey", ls=":")
    ax[1].set_title("QSO-frame CIV (BAL trough?)\nAI>0 excess CIV absorption = BAL"); ax[1].set_xlabel("rest λ [Å], QSO frame"); ax[1].legend(fontsize=8)
    ax[2].axvline(LYA, color="grey", ls=":"); ax[2].set_title("z-SCRAMBLED control (DLA frame)\nreal absorber → flat; QSO feature → survives")
    ax[2].set_xlabel("rest λ [Å], scrambled DLA frame"); ax[2].legend(fontsize=7)
    fig.suptitle(f"Stack of clean lya-only high-N (logN≥{a.nhi_min}) DLA detections: AI>0 mini-BAL vs AI=0 control", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/stack_ai_minibal.png", dpi=130); plt.close(fig)

    # quantify the damped-core depth (DLA-frame, ±5 Å of Lyα) — real DLA is deep
    for g, (sdla, sciv, sscr) in stacks.items():
        core = np.nanmean(sdla[np.abs(rest_dla - LYA) < 5])
        civd = np.nanmean(sciv[(rest_civ > 1546) & (rest_civ < 1552)])
        print(f"  {g}: Lyα-core depth(DLA frame)={core:.3f}  CIV depth(QSO frame)={civd:.3f}")
    print(f"\nfig -> {a.outdir}/stack_ai_minibal.png")


if __name__ == "__main__":
    main()
