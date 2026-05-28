"""Real-LOA DLA-recovery test — in-distribution validation of the
post-reorder LOA-trained GP model.

The 2lpt canonical-TID recovery test (`dla_recovery_step_c.py`) validated
the 2lpt `_m_normmask` models in-distribution, but the LOA-trained
`loa_no_dla_no_bal_wide_m_normmask_3000iter` model was only tested
OUT-of-distribution (on a 2lpt mock target). This script does the
in-distribution test: real DESI LOA spectra.

Reference ("truth"): v1 production. The catalog `dlacat-loa-main-dark.fits`
IS v1 production's output (epoch_920 GP-DLA run). We select strong DLAs
v1 confidently detected, run the NEW LOA model on the same real spectra
(pulled from the LoaArchive), and check it agrees — detection rate and
MAP log N_HI bias vs v1.

Caveat: the catalog was produced in LLS single-absorber mode; this script
runs the standard multi-DLA config (max_dlas=4) — for STRONG DLAs the
p_DLA and MAP N_HI are directly comparable across the two modes.

Output: docs/notes/2026-05-15_dla_recovery_real_loa/<target>.json
        docs/notes/2026-05-15_dla_recovery_real_loa/findings.md
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from astropy.io import fits

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
NOTES = REPO / "docs" / "notes"
OUT_DIR = NOTES / "2026-05-15_dla_recovery_real_loa"

DLACAT = "/scratch/cavestru_root/cavestru0/mfho/nersc/desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/dlacat-loa-main-dark.fits"
LOA_ARCHIVE = "/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5"
DATA_ROOT = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"

# Model under test — the post-reorder LOA candidate.
NEW_MODEL = str(NOTES / "2026-05-13_desi_phase2_loa_no_dla_no_bal_wide_m_normmask_3000iter"
                / "phase2_result.h5")

# Target selection — strong DLAs v1 confidently detected.
NHI_MIN, NHI_MAX = 20.3, 22.0
P_DLA_MIN = 0.99
SNR_FOREST_MIN = 2.0
N_TARGETS = 100        # stratified across NHI; override with --limit


def _peek_k(model_path: str) -> int:
    import h5py
    with h5py.File(model_path, "r") as f:
        return int(f["M"].shape[1])


def select_targets(limit: int) -> np.ndarray:
    """Strong DLAs v1 confidently detected, stratified across NHI so the
    NHI-bias measurement has even coverage."""
    print(f"loading catalog: {DLACAT}", flush=True)
    with fits.open(DLACAT) as f:
        cat = np.array(f["DLACAT"].data)
    keep = (
        (cat["NHI"] >= NHI_MIN) & (cat["NHI"] <= NHI_MAX)
        & (cat["P_DLA"] >= P_DLA_MIN)
        & (cat["SNR_FOREST"] > SNR_FOREST_MIN)
        & (cat["DLAFLAG"] == 0)
        & (cat["Z_DLA"] < cat["Z_QSO"] - 0.05)   # intervening, not proximate
    )
    sub = cat[keep]
    print(f"  {len(sub)} strong DLAs pass selection "
          f"(NHI∈[{NHI_MIN},{NHI_MAX}], P_DLA≥{P_DLA_MIN}, "
          f"SNR_forest>{SNR_FOREST_MIN})", flush=True)
    # Stratify: sort by NHI, take `limit` evenly-spaced rows.
    order = np.argsort(sub["NHI"])
    sub = sub[order]
    if len(sub) > limit:
        idx = np.linspace(0, len(sub) - 1, limit).round().astype(int)
        sub = sub[np.unique(idx)]
    print(f"  selected {len(sub)} targets (stratified across NHI)", flush=True)
    return sub


def build_holder(model_path: str):
    """Create the DLAHolder once — model + sample grids load here."""
    sys.path.insert(0, str(REPO))
    from examples.smoke_one_spectrum import PRESETS
    from gpy_dla_detection.set_parameters import Parameters
    from run_bayes_select import DLAHolder

    preset = PRESETS["y3"]
    k = _peek_k(model_path)
    common = dict(
        loading_min_lambda=preset.loading_min_lambda,
        loading_max_lambda=preset.loading_max_lambda,
        normalization_min_lambda=preset.normalization_min_lambda,
        normalization_max_lambda=preset.normalization_max_lambda,
        min_lambda=preset.min_lambda, max_lambda=preset.max_lambda,
        dlambda=preset.dlambda, k=k,
        max_noise_variance=9.0, num_lines=3,
        max_z_cut=3000.0, min_z_cut=3000.0,
        num_forest_lines=preset.num_forest_lines,
    )
    # 10k-sample QMC grids — 10× faster than the 100k production files.
    # Per-target MAP N_HI carries a little extra QMC scatter, but the
    # aggregate detection rate and median NHI bias over ~100 targets are
    # robust to it.
    params = Parameters(num_dla_samples=10000, **common)
    params_subdla = Parameters(num_dla_samples=10000, **common)
    holder = DLAHolder(
        learned_file=model_path,
        catalog_name=os.path.join(DATA_ROOT, "data/dr12q/processed/catalog.mat"),
        los_catalog=os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/los_catalog"),
        dla_catalog=os.path.join(DATA_ROOT, "data/dla_catalogs/dr9q_concordance/processed/dla_catalog"),
        dla_samples_file=os.path.join(DATA_ROOT, "data/dr12q/processed/dla_samples_a03.mat"),
        sub_dla_samples_file=os.path.join(DATA_ROOT, "data/dr12q/processed/subdla_samples.mat"),
        params=params, params_subdla=params_subdla,
        min_z_separation=3000.0,
        prev_tau_0=preset.prev_tau_0, prev_beta=preset.prev_beta,
        max_dlas=3, broadening=True,
        plot_figures=False, max_workers=8, batch_size=12500,
        figure_dir="/tmp",
        single_absorber_model=False,
        filter_low_likelihood=True,
    )
    return holder


def run_one(holder, archive, row) -> dict:
    """Run inference on one target; compare to the catalog's v1 values."""
    tid = int(row["TARGETID"])
    v1 = dict(p_dla=float(row["P_DLA"]), z_dla=float(row["Z_DLA"]),
              nhi=float(row["NHI"]), z_qso=float(row["Z_QSO"]),
              snr_forest=float(row["SNR_FOREST"]))
    try:
        spec = archive.get_spectrum(tid)
    except KeyError:
        return dict(target_id=tid, status="not_in_archive", v1=v1)

    wavelengths = np.asarray(spec.wavelength, dtype=np.float64)
    flux = np.asarray(spec.flux, dtype=np.float64)
    noise_variance = np.asarray(spec.noise_variance, dtype=np.float64)
    pixel_mask = np.asarray(spec.mask, dtype=np.uint32) != 0
    z_qso = float(row["Z_QSO"])

    try:
        holder.initialize_results(1)
        t0 = time.time()
        holder.process_qso(idx=0, target_id=str(tid),
                           wavelengths=wavelengths, flux=flux,
                           noise_variance=noise_variance,
                           pixel_mask=pixel_mask, z_qso=z_qso)
        dt = time.time() - t0
    except Exception as ex:
        return dict(target_id=tid, status="process_qso_failed",
                    error=repr(ex), traceback=traceback.format_exc(), v1=v1)

    res = holder.results
    p_dla = float(res["p_dlas"][0])
    posteriors = [float(x) for x in res["model_posteriors"][0]]
    # Multi-DLA inference can find several DLAs along the sightline; the
    # catalog row is ONE DLA at a specific z. Match the detected DLA
    # closest in redshift to the catalog DLA and compare THAT one's NHI
    # (taking index [0] blindly would compare unrelated absorbers).
    map_z_arr = np.asarray(res["MAP_z_dlas"][0], dtype=float)
    map_nhi_arr = np.asarray(res["MAP_log_nhis"][0], dtype=float)
    finite = np.isfinite(map_z_arr) & np.isfinite(map_nhi_arr)
    if finite.any():
        cand_z = map_z_arr[finite]
        cand_nhi = map_nhi_arr[finite]
        j = int(np.argmin(np.abs(cand_z - v1["z_dla"])))
        matched_z, matched_nhi = float(cand_z[j]), float(cand_nhi[j])
        n_dla_found = int(finite.sum())
    else:
        matched_z = matched_nhi = float("nan")
        n_dla_found = 0
    return dict(
        target_id=tid, status="ok",
        new_model=dict(p_dla=p_dla, matched_z_dla=matched_z,
                       matched_log_nhi=matched_nhi, n_dla_found=n_dla_found,
                       all_map_z=[float(x) for x in map_z_arr],
                       all_map_nhi=[float(x) for x in map_nhi_arr],
                       model_posteriors=posteriors),
        v1=v1,
        delta_p_dla=p_dla - v1["p_dla"],
        delta_log_nhi=matched_nhi - v1["nhi"],
        delta_z_dla=matched_z - v1["z_dla"],
        elapsed_s=dt,
    )


def main():
    limit = N_TARGETS
    for a in sys.argv[1:]:
        if a.startswith("--limit"):
            limit = int(a.split("=")[1] if "=" in a else sys.argv[sys.argv.index(a) + 1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # `--findings-only` rebuilds findings.md from the cached per-target
    # JSONs — no inference re-run.
    if "--findings-only" in sys.argv:
        results = [json.loads(jf.read_text())
                   for jf in sorted(OUT_DIR.glob("*.json"))]
        print(f"loaded {len(results)} cached per-target results", flush=True)
        write_findings(results)
        return

    targets = select_targets(limit)

    from gpy_dla_detection.loa_archive import LoaArchive
    print(f"building DLAHolder (model: {Path(NEW_MODEL).parent.name})", flush=True)
    holder = build_holder(NEW_MODEL)

    results = []
    with LoaArchive(LOA_ARCHIVE) as archive:
        for i, row in enumerate(targets):
            r = run_one(holder, archive, row)
            results.append(r)
            (OUT_DIR / f"{r['target_id']}.json").write_text(json.dumps(r, indent=2))
            if r["status"] == "ok":
                nm = r["new_model"]
                print(f"  [{i+1}/{len(targets)}] TID {r['target_id']}: "
                      f"new p_DLA={nm['p_dla']:.3f} (v1={r['v1']['p_dla']:.3f}), "
                      f"matched NHI={nm['matched_log_nhi']:.2f} "
                      f"(v1={r['v1']['nhi']:.2f}, Δ={r['delta_log_nhi']:+.2f} dex, "
                      f"Δz={r['delta_z_dla']:+.4f}), "
                      f"{nm['n_dla_found']} DLA(s), {r['elapsed_s']:.0f}s",
                      flush=True)
            else:
                print(f"  [{i+1}/{len(targets)}] TID {r['target_id']}: "
                      f"{r['status']}", flush=True)

    write_findings(results)


def write_findings(results):
    ok = [r for r in results if r["status"] == "ok"]
    md = ["# Real-LOA DLA-recovery test — in-distribution validation", ""]
    md.append(f"Date: 2026-05-15. Model under test: "
              f"`loa_no_dla_no_bal_wide_m_normmask_3000iter`.")
    md.append("")
    md.append("## Setup")
    md.append("")
    md.append(f"- Targets: {len(results)} strong DLAs v1 production confidently "
              f"detected (NHI∈[{NHI_MIN},{NHI_MAX}], P_DLA≥{P_DLA_MIN}, "
              f"SNR_forest>{SNR_FOREST_MIN}), stratified across NHI.")
    md.append(f"- Reference: v1 production = the `dlacat-loa-main-dark.fits` "
              f"catalog (epoch_920 GP-DLA run).")
    md.append(f"- Spectra: real DESI LOA, from the LoaArchive "
              f"`loa_full_z2_noR_v2.h5`.")
    md.append(f"- Inference: multi-DLA mode (max_dlas=3, single_absorber=False), "
              f"num_dla_samples=10000 (10k QMC grid; aggregate stats robust, "
              f"per-target MAP N_HI carries minor QMC scatter). The new "
              f"model's detected DLAs are matched to the catalog DLA by "
              f"closest redshift before comparing MAP log N_HI.")
    md.append(f"- Caveat: the v1 catalog was produced in LLS single-absorber "
              f"mode; this run is multi-DLA. For isolated strong DLAs the "
              f"p_DLA and MAP N_HI are comparable across modes.")
    md.append(f"- Status counts: " +
              ", ".join(f"{s}={sum(1 for r in results if r['status']==s)}"
                        for s in sorted({r['status'] for r in results})))
    md.append("")

    if ok:
        p_new = np.array([r["new_model"]["p_dla"] for r in ok])
        d_nhi = np.array([r["delta_log_nhi"] for r in ok])
        d_z = np.array([r["delta_z_dla"] for r in ok])
        det_05 = int((p_new > 0.5).sum())
        det_97 = int((p_new > 0.97).sum())
        # "Well-matched" = the new model's matched DLA lands within 0.01 in
        # redshift of the catalog DLA (same absorber, not a neighbour).
        # NHI bias is only meaningful for these.
        well = np.abs(d_z) < 0.01
        d_nhi_w = d_nhi[well]
        md.append("## Verdict")
        md.append("")
        md.append(f"- **Detection agreement**: the new model recovers "
                  f"{det_05}/{len(ok)} ({100*det_05/len(ok):.0f}%) at p_DLA > 0.5, "
                  f"{det_97}/{len(ok)} ({100*det_97/len(ok):.0f}%) at p_DLA > 0.97 — "
                  f"on DLAs v1 detected at P_DLA ≥ {P_DLA_MIN}.")
        md.append(f"- **Redshift match**: {int(well.sum())}/{len(ok)} targets have "
                  f"the new model's matched DLA within |Δz| < 0.01 of the "
                  f"catalog DLA (same absorber). NHI bias below is for these.")
        if len(d_nhi_w):
            md.append(f"- **MAP log N_HI bias** (new − v1, well-matched): median "
                      f"{np.median(d_nhi_w):+.3f} dex, mean {np.mean(d_nhi_w):+.3f}, "
                      f"scatter (MAD) "
                      f"{np.median(np.abs(d_nhi_w-np.median(d_nhi_w)))*1.4826:.3f} dex.")
        md.append(f"- MAP log N_HI bias (all finite matches, incl. ambiguous): "
                  f"median {np.nanmedian(d_nhi):+.3f} dex.")
        md.append("")
        md.append("## Per-target results")
        md.append("")
        md.append("| TID | v1 NHI | v1 P_DLA | new p_DLA | matched NHI | Δ NHI | Δ z | n_DLA | t (s) |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in ok:
            nm = r["new_model"]
            md.append(f"| {r['target_id']} | {r['v1']['nhi']:.2f} | "
                      f"{r['v1']['p_dla']:.3f} | {nm['p_dla']:.3f} | "
                      f"{nm['matched_log_nhi']:.2f} | {r['delta_log_nhi']:+.2f} | "
                      f"{r['delta_z_dla']:+.4f} | {nm['n_dla_found']} | "
                      f"{r['elapsed_s']:.0f} |")
    md.append("")
    (OUT_DIR / "findings.md").write_text("\n".join(md) + "\n")
    print(f"\n[saved] {OUT_DIR / 'findings.md'}", flush=True)
    if ok:
        print(f"detection: {det_05}/{len(ok)} at p>0.5, {det_97}/{len(ok)} at p>0.97; "
              f"NHI bias (well-matched) median "
              f"{np.median(d_nhi_w):+.3f} dex" if len(d_nhi_w) else "", flush=True)


if __name__ == "__main__":
    main()
