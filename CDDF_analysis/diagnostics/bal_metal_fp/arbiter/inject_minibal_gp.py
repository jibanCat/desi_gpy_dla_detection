#!/usr/bin/env python
"""inject_minibal_gp.py — does a BAL-like broad forest trough fool the GP-DLA search into a
high-N false positive? Direct causation test for the AI-mini-BAL causal-vs-benign question.

Injects three things into real CLEAN (non-BAL, non-DLA) LOA archive spectra and runs the UNMODIFIED
GP-DLA inference (DLAHolder.process_qso, y3 = the LOA production config) on each:
  (0) NOTHING            — negative control (should stay ~no absorber)
  (1) a Voigt DLA logN=20.5 — positive control (should recover logN~20.5, P_DLA~1)
  (2) a BROAD Gaussian trough (BAL-like: no damped wings) at a forest position, width W km/s, depth D
      — THE TEST: if it gets flagged P_DLA>0.99 & logN>=20.3, broad forest absorption CAUSES DLA FPs
      (→ AI-mini-BALs are causal, ~10%); if not, the AI excess is benign selection (~2%).

Env: source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate gpdla
     export PYTHONPATH=/home/mfho/desi_gpy_dla_detection:$PYTHONPATH
Aggregate-only (FP rates, not per-object real spectra).

COMMITTED figures/inject_minibal.npz was produced with (NOT the CLI defaults):
    python inject_minibal_gp.py --widths 1000,2000 --depths 0.5 --n 15
  → none=6.7% / voigt20.5=40% / broad_W1000_D0.5=6.7% / broad_W2000_D0.5=6.7% (paired McNemar: 0 incremental FP).
4-LENS REVIEW (2026-07-03) follow-up (recommended, not yet run): re-run at the PRODUCTION config
(max_dlas=4, single_absorber_model=True→multi, filter_low_likelihood=True), n>=100, deeper/wider troughs,
and a stronger positive control (the Voigt-20.5 control here fires only ~33-40%, capping statistical power).
The Voigt insensitivity is width/EW-bounded: at W=5000 km/s / D=0.9 the broad trough DOES produce logN~21.2
fits (EW rivals a DLA) — those DLA-mimicking broad troughs are what the BI>0 veto removes; the leaked
narrow shallow (<=2000 km/s, D~0.5) mini-BALs give the negative-control rate.
"""
import os, sys, argparse, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import h5py, fitsio

DATA_ROOT = "/scratch/cavestru_root/cavestru0/mfho/DESI/desi_gpy_dla_detection"
MODEL_LEGACY = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5"
# production LOA model (loa_main_dark_v1 BASELINE.env): 2lpt_loa124_nohcd_nobal_wide_m
MODEL_PROD = "/scratch/cavestru_root/cavestru0/mfho/phase2_desi/2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5"
ARCHIVE = "/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5"
OUR_VAC = "/scratch/cavestru_root/cavestru0/mfho/our_loa_bal_vac/our_loa_bal_vac_v1.fits"
REAL_DLA = "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1/dlacat-loa-main-dark-v1.fits"
LYA = 1215.67; C = 299792.458


def broad_trough(wave, flux, z_qso, lam_rf, width_kms, depth):
    """multiply flux by a Gaussian absorption trough (BAL-like, NO damped wings) at rest lam_rf."""
    lam0 = lam_rf * (1 + z_qso)
    sig = (width_kms / C) * lam0 / 2.355
    return flux * (1.0 - depth * np.exp(-0.5 * ((wave - lam0) / sig) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="# clean spectra")
    ap.add_argument("--config", choices=["production", "legacy"], default="production",
                    help="production = the loa_main_dark_v1 config (phase2 model, max_dlas=4, filter, tau_eb null)")
    ap.add_argument("--widths", default="1000,2000,3000,5000", help="BAL trough widths (km/s)")
    ap.add_argument("--depths", default="0.5,0.7,0.9", help="BAL trough depths")
    ap.add_argument("--pos-nhi", default="20.5,21.0", help="positive-control Voigt DLA logN(s)")
    ap.add_argument("--inject-lamrf", type=float, default=1100.0, help="rest-λ to place the trough / DLA")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figures"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, DATA_ROOT); sys.path.insert(0, "/home/mfho/desi_gpy_dla_detection")
    from gpy_dla_detection.set_parameters import Parameters
    from gpy_dla_detection.inject_absorber import inject_voigt
    from run_bayes_select import DLAHolder

    prod = a.config == "production"
    MODEL = MODEL_PROD if prod else MODEL_LEGACY
    # config: production = loa_main_dark_v1 BASELINE.env (MAX_DLAS=4, SINGLE_ABSORBER_MODEL=1,
    # FILTER_LOW_LIKELIHOOD=1, ENABLE_TAU_EB=1/null, MAX_LAMBDA=1250, LOADING_MAX_LAMBDA=1550)
    common = dict(loading_min_lambda=910.0, loading_max_lambda=1550.0 if prod else 1217.0,
                  normalization_min_lambda=1425.0, normalization_max_lambda=1475.0, min_lambda=911.75,
                  max_lambda=1250.0 if prod else 1215.75, dlambda=0.15, k=30, max_noise_variance=9.0,
                  num_lines=3, max_z_cut=3000.0, min_z_cut=3000.0, num_forest_lines=3)
    dr = lambda r: os.path.join(DATA_ROOT, r)
    holder = DLAHolder(
        learned_file=MODEL, catalog_name=dr("data/dr12q/processed/catalog.mat"),
        los_catalog=dr("data/dla_catalogs/dr9q_concordance/processed/los_catalog"),
        dla_catalog=dr("data/dla_catalogs/dr9q_concordance/processed/dla_catalog"),
        dla_samples_file=dr("data/dr12q/processed/pw_samples_a3_172_220_50000.mat"),
        sub_dla_samples_file=dr("data/dr12q/processed/subdla_samples_a03_191_200_100000.mat"),
        params=Parameters(num_dla_samples=50000, **common),
        params_subdla=Parameters(num_dla_samples=100000, **common),
        min_z_separation=3000.0, prev_tau_0=0.00246, prev_beta=3.62,
        max_dlas=4 if prod else 1, broadening=True, plot_figures=False, max_workers=1, batch_size=100,
        filter_low_likelihood=prod, single_absorber_model=True,
        enable_tau_eb=prod, tau_eb_objective="null")
    print(f"[config={a.config}] model={os.path.basename(MODEL)} max_dlas={4 if prod else 1} "
          f"filter={prod} tau_eb={prod}", flush=True)

    # pick CLEAN spectra: z>2, not in our BAL VAC (AI=0 & BI=0), no high-N DLA detection
    ours = fitsio.read(OUR_VAC); balset = {int(t) for t, a2, b2 in zip(ours["TARGETID"], ours["AI_CIV"], ours["BI_CIV"]) if a2 > 0 or b2 > 0}
    dc = fitsio.read(REAL_DLA); dlahost = {int(t) for t, nh, pp in zip(dc["TARGETID"], dc["NHI"], dc["P_DLA"]) if nh >= 20.0 and pp > 0.5}
    with h5py.File(ARCHIVE, "r") as H:
        wave = H["wavelength"][:]; cat = H["catalog"][:]
        ct = np.asarray(cat["TARGETID"], np.int64); cz = np.asarray(cat["Z"], float)
        cand = [k for k in range(len(ct)) if cz[k] > 2.3 and cz[k] < 3.5 and int(ct[k]) not in balset and int(ct[k]) not in dlahost]
        rng = np.random.default_rng(0); sel = rng.choice(cand, min(a.n, len(cand)), replace=False)
        sel = np.sort(sel)
        FL = H["flux"][sel]; IV = H["ivar"][sel]; MK = H["mask"][sel]; ZQ = cz[sel]; TT = ct[sel]

    widths = [float(x) for x in a.widths.split(",")]; depths = [float(x) for x in a.depths.split(",")]
    pos_nhis = [float(x) for x in a.pos_nhi.split(",")]

    def run(wave, flux, ivar, mask, zq, tid):
        nv = np.where(ivar > 0, 1.0 / np.maximum(ivar, 1e-8), 1e10)
        pm = (mask != 0) | (ivar <= 0)
        holder.initialize_results(1)
        holder.process_qso(idx=0, target_id=str(int(tid)), wavelengths=wave.copy(), flux=flux.copy(),
                           noise_variance=nv.copy(), pixel_mask=pm.copy(), z_qso=float(zq))
        r = holder.results
        return float(r["p_dlas"][0]), float(r["MAP_log_nhis"][0, 0])

    import collections
    res = collections.defaultdict(list)
    t0 = time.time()
    for i in range(len(sel)):
        fl = FL[i]; iv = IV[i]; mk = MK[i]; zq = ZQ[i]; tid = TT[i]
        # (0) negative control
        p, n = run(wave, fl, iv, mk, zq, tid); res["none"].append((p, n))
        # (1) positive controls: Voigt DLA(s) at inject-lamrf
        zdla = a.inject_lamrf * (1 + zq) / LYA - 1
        for NH in pos_nhis:
            flv = inject_voigt(wave, fl, 10 ** NH, zdla, num_lines=3)
            p, n = run(wave, flv, iv, mk, zq, tid); res[f"voigt{NH:g}"].append((p, n))
        # (2) test: broad BAL-like troughs (narrow shallow = leaked mini-BAL; wide/deep = the BI-vetoed regime)
        for W in widths:
            for D in depths:
                flb = broad_trough(wave, fl, zq, a.inject_lamrf, W, D)
                p, n = run(wave, flb, iv, mk, zq, tid); res[f"broad_W{int(W)}_D{D}"].append((p, n))
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(sel)} spectra ({(time.time()-t0)/(i+1):.1f}s each)", flush=True)

    def cp(k, n, alpha=0.05):
        from scipy.stats import beta
        if n == 0: return (np.nan, np.nan)
        lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
        return lo, hi

    print(f"\n=== INJECTION-RECOVERY [config={a.config}]: high-N DLA FP rate (P_DLA>0.99 & logN>=20.3) ===")
    summ = {}; ci = {}
    for k, v in res.items():
        arr = np.array(v); nfp = int(np.sum((arr[:, 0] > 0.99) & (arr[:, 1] >= 20.3))); n = len(arr)
        fp = nfp / n; summ[k] = fp; ci[k] = cp(nfp, n)
        print(f"  {k:18s}: FP = {100*fp:5.1f}% [{100*ci[k][0]:.1f},{100*ci[k][1]:.1f}] ({nfp}/{n})  "
              f"med P_DLA={np.median(arr[:,0]):.3f}  med logN(det)={np.median(arr[arr[:,0]>0.5,1]) if (arr[:,0]>0.5).any() else float('nan'):.2f}")
    tag = a.config
    np.savez(f"{a.out}/inject_minibal_{tag}.npz", **{k: np.array(v) for k, v in res.items()})
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    posks = sorted([k for k in res if k.startswith("voigt")])
    ks = ["none"] + posks + [k for k in res if k.startswith("broad")]
    yerr = np.array([[100 * (summ[k] - ci[k][0]) for k in ks], [100 * (ci[k][1] - summ[k]) for k in ks]])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    cols = ["grey"] + ["C2"] * len(posks) + ["C3"] * (len(ks) - 1 - len(posks))
    ax.bar(range(len(ks)), [100 * summ[k] for k in ks], color=cols, yerr=yerr, capsize=3)
    for i, k in enumerate(ks): ax.text(i, 100 * summ[k] + 1, f"{100*summ[k]:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("high-N DLA FP rate [%]  (P_DLA>0.99 & logN≥20.3)")
    ax.set_title(f"Injection [config={a.config}]: does a broad forest trough fool the GP into a high-N DLA?\n"
                 "grey=neg control, green=Voigt-DLA pos controls, red=broad troughs (narrow shallow=leaked; wide/deep=BI-vetoed)")
    fig.tight_layout(); fig.savefig(f"{a.out}/inject_minibal_{tag}.png", dpi=130); plt.close(fig)
    print(f"\nfig -> {a.out}/inject_minibal_{tag}.png   (Clopper-Pearson 95% CIs)")


if __name__ == "__main__":
    main()
