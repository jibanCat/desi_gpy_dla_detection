"""break_census.py — how many LLS can a 912 A break counter actually count, and what is the
TRUE ell(z)[tau>=2], on the 2LPT-0 mock HCD truth catalog. Reduce-only (NO inference, NO SLURM).

This closes a provenance gap: the break-countability numbers below previously existed only as
bare literals in a doc (from a since-deleted script). It is also the GROUND TRUTH that Build B
(the break-counting incidence pipeline) will be validated against.

An LLS is counted by its Lyman-limit BREAK, and a break counter -- scanning a sightline from the
QSO downward -- sees only the FIRST (highest-z) tau>=2 system (blocking). This routine:

  1. derives the blue-cutoff redshift (912 A break above the 3600 A DESI cutoff) and the tau>=2
     column threshold log10(2/sigma_912) -- NOT hard-coded;
  2. counts QSOs total and with z_qso >= 3.3 (the break-counting sample: a full searchable window);
  3. of tau>=2 absorbers on those sightlines, the fraction that are break-observable
     (z_abs > cutoff and z_abs < z_qso minus a 3000 km/s proximity zone);
  4. the number of COUNTABLE first-breaks after blocking (one per sightline);
  5. the TRUE ell(z)[tau>=2], computed BOTH as
        (a) the direct incidence from ALL observable tau>=2 absorbers over the full window, and
        (b) via the Nelson-Aalen estimator (CDDF_analysis.lyc.survival.ell_nelson_aalen) applied
            to the BLOCKED first-break census.
     (a) and (b) must agree -- a self-consistency check that Nelson-Aalen recovers the true
     incidence despite blocking. The ratio is asserted and stamped;
     NOTE on the residual per-bin scatter of (a)/(b) (~1%): this is ORDINARY SAMPLING NOISE, not
     absorber clustering. A coupled sightline bootstrap gives pooled a/b = 0.998 +- 0.004 (-0.5
     sigma) and chi2 = 9.04/6 (p = 0.17) against a/b = 1; a clean inhomogeneous-Poisson null on
     the SAME sightline windows returns per-bin a/b ~ 1.000 (so it is not a binning artifact
     either); and the mock's measured LOS correlation INT xi dr ~ 1e-4 predicts a clustering
     bias of ~0.02% -- 50-100x too small to make 1%. (Pooled a/b < 1 while per-bin a/b > 1 is a
     benign pooling/Simpson effect.) Real-data clustering IS a carried systematic -- see the
     `survival.py` module docstring; this mock cannot bound it.

TWO CONVENTIONS THAT CHANGE THE HEADLINE COUNTS -- state them whenever the numbers are quoted:
  * PROXIMITY: with the 3000 km/s exclusion, break-observable = 38.87% and countable = 22,704.
    With NO proximity cut these become 43.70% and 24,412 (and drop-window overlap 69.3% -> 61.6%).
    The cut removes 7.0% of countable breaks and lowers the sample's median z by 0.036.
  * The break-observable / countable counts are a GEOMETRIC (redshift-window) UPPER BOUND, NOT a
    detectability or completeness. They ignore blue-edge S/N (the mock's ivar is ~2.7x worse at
    3600-3650 A than at 3800-4000 A; an S/N-limited effective cutoff of 3650-3700 A costs a
    further 7-15% of the sample), accumulated Lyman-valley/Lyman-series opacity, and cumulative
    attenuation from sub-tau=2 partial systems stacked above a break. Build B must measure
    completeness AS A FUNCTION OF COLUMN, not adopt these as achievable detections.

MOCK-vs-LITERATURE (truth level): the 2LPT-0 recipe OVER-produces tau>=2 LLS by ~1.25x versus a
multi-anchor literature trend (O'Meara+13 z=2.25, Fumagalli+13 z=2.8, POW10 z=3.5), and by up to
~1.9x versus POW10's gamma=5.2 power law extrapolated down to z=3.0 (that extrapolation is
unreliable -- POW10 fit only z>3.3). Any earlier note claiming the mock sits "20-40% BELOW" the
literature refers to the RECOVERED (drop+count fitted) ell, which is biased low, NOT to the
injected truth. Build-B validation against this mock is therefore a RECIPE-ROBUSTNESS test, not
an absolute-abundance check.
  6. the z-overlap with the drop channel's window z912 in [2.6, 3.35]: what fraction of countable
     first-breaks fall inside it.

MOCK (2LPT-0) values are public-OK; NOT real-LOA. Stamped git-provenanced to break_census.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# repo root = 4 dirnames up (lls -> diagnostics -> CDDF_analysis -> <repo>).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from astropy.table import Table  # noqa: E402

from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB  # noqa: E402
from CDDF_analysis.lyc import survival as SV  # noqa: E402
from CDDF_analysis.lyc import opacity as LYC  # noqa: E402

DEFAULT_OUT_JSON = os.path.join(_REPO, "CDDF_analysis", "hbi", "break_census.json")

# --- census geometry (derived where possible; see run()) ---
WAVE_OBS_MIN = 3600.0        # DESI blue cutoff (Angstrom)
Z_QSO_MIN = 3.3              # break-counting sample: sightlines with a full searchable window
PROXIMITY_DV_KMS = 3000.0    # QSO proximity velocity zone excluded from the window
OMEGA_M = 0.279              # repo's single absorption-distance cosmology (path_length_int)
# ell(z) grid: from the blue cutoff up to where z_qso>=3.3 sightlines still contribute exposure.
Z_EDGES = np.round(np.arange(2.95, 3.56, 0.10), 2)   # [2.95,3.05,...,3.55] -> 6 bins
DROP_WINDOW = (2.6, 3.35)   # the drop channel's z912 window (opacity.py drop estimator)
N_BOOT = 500


def _git_commit(routine: str | None = None):
    """HEAD, suffixed `-dirty` iff the ROUTINE that produced this artifact is untracked or modified.

    A `-dirty` stamp means the artifact is NOT third-party re-derivable: the named commit does not
    contain (this version of) the routine. Commit the routine first, then re-run. Checking the
    routine specifically -- rather than `git status`, which is dirtied by the artifact's own
    untracked output -- is what makes the marker meaningful. Also covers `survival.py`, which the
    census imports and without which the stamped `rederive` command cannot run.
    """
    deps = [routine or os.path.relpath(os.path.abspath(__file__), _REPO),
            "CDDF_analysis/lyc/survival.py"]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                      stderr=subprocess.DEVNULL).decode().strip()
        for f in deps:
            tracked = subprocess.call(["git", "ls-files", "--error-unmatch", f], cwd=_REPO,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
            modified = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", f], cwd=_REPO) != 0
            if not tracked or modified:
                return f"{sha}-dirty"
        return sha
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] _git_commit() failed ({type(e).__name__}: {e})", file=sys.stderr)
        return "unknown"


def run() -> dict:
    t0 = time.time()

    # --- 1. derived constants ---
    cutoff = SV.blue_cutoff_z(WAVE_OBS_MIN)                 # 3600/911.76 - 1
    logN_tau2 = float(np.log10(2.0 / LYC.SIGMA_912))       # tau=2 <=> N = 2/sigma_912
    print(f"[1] blue-cutoff z (912 A > {WAVE_OBS_MIN:.0f} A) = {cutoff:.4f}  "
          f"(LYMAN_LIMIT={LYC.LYMAN_LIMIT} A)")
    print(f"    tau>=2 threshold  log10 N_HI = log10(2/sigma_912) = {logN_tau2:.4f}  "
          f"(sigma_912={LYC.SIGMA_912:.3g} cm^2)")

    # --- data handles (mirror joint_drop_count_validation._physical_drop) ---
    hcd = Table.read(AB.DEF_TRUTH)
    zc = Table.read(os.path.join(os.path.dirname(AB.DEF_TRUTH), "zcat.fits"))
    zq_all = np.asarray(zc["Z"], float)
    tid_all = np.asarray(zc["TARGETID"])

    # --- 2. QSO counts ---
    n_qso = int(zq_all.size)
    samp = zq_all >= Z_QSO_MIN
    n_qso_hi = int(samp.sum())
    frac_hi = n_qso_hi / n_qso if n_qso else np.nan
    print(f"[2] QSOs total = {n_qso}   with z_qso >= {Z_QSO_MIN} = {n_qso_hi}   "
          f"fraction = {frac_hi:.4f}")

    # break-counting sample: {targetid -> z_qso} for z_qso >= 3.3
    zq_map = {int(t): float(z) for t, z in zip(tid_all[samp], zq_all[samp])}

    # --- tau>=2 absorbers on the sample sightlines ---
    nhi = np.asarray(hcd["NHI"], float)
    za = np.asarray(hcd["Z"], float)
    ta = np.asarray(hcd["TARGETID"])
    is_tau2 = nhi >= logN_tau2
    in_samp = np.array([int(t) in zq_map for t in ta])
    sel = is_tau2 & in_samp
    n_tau2_samp = int(sel.sum())
    sl_a = ta[sel]
    z_a = za[sel]

    # --- 3. break-observable fraction (blocking-independent per-absorber test) ---
    census = SV.build_break_census(sl_a, z_a, zq_map, cutoff, proximity_dv_kms=PROXIMITY_DV_KMS)
    n_obs_abs = int(census["obs_mask"].sum())
    frac_obs = n_obs_abs / n_tau2_samp if n_tau2_samp else np.nan
    print(f"[3] tau>=2 absorbers on z_qso>={Z_QSO_MIN} sightlines = {n_tau2_samp}   "
          f"break-observable = {n_obs_abs}   fraction = {frac_obs:.4f}")

    # --- 4. countable first-breaks after blocking ---
    n_break = int(census["has_break"].sum())
    print(f"[4] countable first-breaks after blocking (1 per sightline) = {n_break}   "
          f"(observable-absorbers/countable-breaks = {n_obs_abs / n_break:.3f} extra per sightline)")

    # --- 5. TRUE ell(z)[tau>=2]: (a) direct all absorbers, (b) Nelson-Aalen on blocked census ---
    row_of = {s: i for i, s in enumerate(census["sl"].tolist())}
    obs = census["obs_mask"]
    abs_rows = np.array([row_of[int(t)] for t in sl_a[obs]], int)
    abs_z = z_a[obs]
    z_start_full = census["z_cut"]        # (a): full observable window, no blocking truncation
    z_stop = census["z_stop"]
    di = SV.ell_direct_incidence(abs_z, abs_rows, z_start_full, z_stop, Z_EDGES,
                                 n_boot=N_BOOT, seed=0)
    na = SV.ell_nelson_aalen(census["z_detect"], census["z_start"], z_stop, Z_EDGES,
                             n_boot=N_BOOT, seed=0)

    ell_a = np.asarray(di["ell"], float)
    ell_b = np.asarray(na["ell"], float)
    z_mid = np.asarray(na["z_mid"], float)
    ell_a_dX = SV.ell_per_dz_to_dX(ell_a, z_mid, Omega_m=OMEGA_M)
    ell_b_dX = SV.ell_per_dz_to_dX(ell_b, z_mid, Omega_m=OMEGA_M)
    pooled_a = float(di["n_det"].sum() / di["exposure"].sum())
    pooled_b = float(na["n_det"].sum() / na["exposure"].sum())
    ratio_pooled = pooled_a / pooled_b
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_bin = ell_a / ell_b

    print(f"[5] TRUE ell(z)[tau>=2]  (a)=direct all-absorbers  (b)=Nelson-Aalen on blocked census")
    print("    z_mid   ell_a[/dz]  ell_b[/dz]  a/b     ell_b[/dX]")
    for i in range(z_mid.size):
        print(f"    {z_mid[i]:.2f}   {ell_a[i]:9.4f}  {ell_b[i]:9.4f}  {ratio_bin[i]:.4f}  "
              f"{ell_b_dX[i]:9.4f}")
    print(f"    POOLED ell_a={pooled_a:.4f}  ell_b={pooled_b:.4f}  ratio(a/b)={ratio_pooled:.4f} "
          f"[/dz]")
    # self-consistency assertion: Nelson-Aalen must recover the direct incidence
    assert abs(ratio_pooled - 1.0) < 0.05, (
        f"Nelson-Aalen vs direct incidence disagree: pooled a/b={ratio_pooled:.4f} "
        f"(expected ~1). Blocking correction or exposure geometry is wrong.")

    # --- 6. overlap with the drop window ---
    zd = census["z_detect"][census["has_break"]]
    in_drop = (zd >= DROP_WINDOW[0]) & (zd <= DROP_WINDOW[1])
    frac_drop = float(in_drop.mean())
    n_drop = int(in_drop.sum())
    print(f"[6] countable first-breaks in drop window z912 in [{DROP_WINDOW[0]},{DROP_WINDOW[1]}] "
          f"= {n_drop}/{n_break} = {frac_drop:.4f}")
    print(f"[wall {time.time()-t0:.1f}s]")

    return dict(
        wall_s=round(time.time() - t0, 1),
        constants=dict(
            blue_cutoff_z=cutoff, wave_obs_min=WAVE_OBS_MIN, lyman_limit=LYC.LYMAN_LIMIT,
            logN_tau2=logN_tau2, sigma_912=LYC.SIGMA_912, proximity_dv_kms=PROXIMITY_DV_KMS,
            omega_m=OMEGA_M, z_qso_min=Z_QSO_MIN),
        qso_counts=dict(n_qso_total=n_qso, n_qso_zq_ge_3p3=n_qso_hi, frac_zq_ge_3p3=frac_hi),
        observable=dict(n_tau2_on_sample=n_tau2_samp, n_break_observable=n_obs_abs,
                        frac_observable=frac_obs),
        countable=dict(n_first_breaks=n_break,
                       obs_absorbers_per_countable_break=n_obs_abs / n_break),
        true_ell=dict(
            z_edges=Z_EDGES.tolist(), z_mid=z_mid.tolist(),
            ell_direct_dz=ell_a.tolist(), ell_direct_err_dz=np.asarray(di["ell_err"]).tolist(),
            ell_na_dz=ell_b.tolist(), ell_na_err_dz=np.asarray(na["ell_err"]).tolist(),
            ell_direct_dX=ell_a_dX.tolist(), ell_na_dX=ell_b_dX.tolist(),
            n_det_direct=di["n_det"].tolist(), n_det_na=na["n_det"].tolist(),
            exposure_direct_dz=di["exposure"].tolist(), exposure_na_dz=na["exposure"].tolist(),
            ratio_bin_a_over_b=ratio_bin.tolist(),
            pooled_ell_direct_dz=pooled_a, pooled_ell_na_dz=pooled_b,
            ratio_pooled_a_over_b=ratio_pooled),
        drop_overlap=dict(window=list(DROP_WINDOW), n_in_window=n_drop,
                          frac_in_window=frac_drop),
    )


def main(a):
    t = time.time()
    res = run()
    out = dict(
        metadata=dict(
            what="Break-countability census + TRUE ell(z)[tau>=2] on the 2LPT-0 mock HCD truth "
                 "catalog: how many LLS a 912 A break counter can count (blocking), and the true "
                 "incidence recovered by Nelson-Aalen (self-consistency vs the direct count).",
            mock="2LPT-0 (loa-124) HCD truth catalog; MOCK values (public-OK), NOT real-LOA",
            code_commit=_git_commit(),
            wallclock_s=round(time.time() - t, 1),
            rederive="python CDDF_analysis/diagnostics/lls/break_census.py --force",
            estimator="CDDF_analysis/lyc/survival.py (build_break_census / ell_nelson_aalen / "
                      "ell_direct_incidence): model-free Nelson-Aalen exposure incidence.",
            truth_catalog=AB.DEF_TRUTH,
            note="Ground truth for Build B validation. ell(z) is per unit z (dN/dz) and per unit "
                 "absorption distance dX (path_length_int, Omega_m=0.279). (a) direct-all-absorbers "
                 "== (b) Nelson-Aalen-on-blocked-census is the check that blocking does not bias "
                 "the incidence.",
        ),
        result=res,
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if os.path.exists(a.out) and not a.force:
        print(f"[skip-json] {a.out} exists (pass --force to overwrite).")
    else:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"[saved-json] {a.out}  code_commit={out['metadata']['code_commit']}  "
              f"({out['metadata']['wallclock_s']:.0f}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT_JSON)
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
