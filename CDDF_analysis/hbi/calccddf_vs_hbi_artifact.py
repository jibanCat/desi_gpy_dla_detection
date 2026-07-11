# -*- coding: utf-8 -*-
"""Aggregate per-mock calc_cddf-vs-HBI JSONs into the stamped artifact
CDDF_analysis/hbi/calccddf_vs_hbi.json.  UNTRACKED / do-not-commit.  MOCK-ONLY."""
import os
import json
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

# HBI reference (forward-path loa0; crossmock_transfer_loa0.json self baseline + legs)
HBI_FORWARD = {
    "2lpt0": dict(dndx={"20.3": 1.0381, "20.0": 1.0041, "19.5": 0.9189, "band_195_203": 0.8490},
                  omega={"20.3": 0.9593, "20.0": 0.9547, "19.5": 0.9377, "band_195_203": 0.8220}),
    "london0": dict(dndx={"20.3": 1.0347, "20.0": 1.0190, "19.5": 0.8829, "band_195_203": 0.7892},
                    omega={"20.3": 0.9464, "20.0": 0.9469, "19.5": 0.9270, "band_195_203": 0.8168}),
    "saclay0": dict(dndx={"20.3": 1.0452, "20.0": 1.0165, "19.5": 0.8898, "band_195_203": 0.7977},
                    omega={"20.3": 0.9420, "20.0": 0.9423, "19.5": 0.9228, "band_195_203": 0.8183}),
}
# HBI truth cumulative constants (2lpt0, estimator-independent; crossmock self baseline)
HBI_TRUTH_2LPT0 = dict(dndx={"20.3": 0.0543430, "20.0": 0.0858851, "19.5": 0.1470712},
                       omega={"20.3": 0.00062879, "20.0": 0.00069416, "19.5": 0.00074653})


def git_commit():
    try:
        c = subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"]).decode().strip()
        dirty = subprocess.call(["git", "-C", HERE, "diff", "--quiet"]) != 0
        return c + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True, help="mock=json ...")
    ap.add_argument("--out", default=os.path.join(HERE, "calccddf_vs_hbi.json"))
    args = ap.parse_args()

    mocks = {}
    total_wall = 0.0
    for spec in args.inp:
        m, jp = spec.split("=", 1)
        if not os.path.exists(jp):
            print("missing:", jp); continue
        d = json.load(open(jp))
        total_wall += d.get("wallclock_s", 0.0)
        mocks[m] = dict(
            n_files=d["n_files"], n_sightlines=d["n_sightlines"], dX_total=d["dX_total"],
            grid=d["grid"], second=d["second"], z_range=d["z_range"], snr_min=d["snr_min"],
            calccddf=d["cumulative"]["calccddf"], truth=d["cumulative"]["truth"],
            R0_calccddf=d["cumulative"]["R0_calccddf"],
            hbi_forward_R0=HBI_FORWARD.get(m),
        )

    out = dict(
        metadata=dict(
            what="LITERAL calc_cddf (Bird-2017 posterior CDDF) vs catalog-HBI, head-to-head on the "
                 "same mocks vs the same injected truth. MOCK-ONLY, public-OK. No GP re-inference, "
                 "no alpha, no hard P_DLA cut (posterior-weighted). DLA sample grid = the SAME "
                 "pw_samples_a3_172_225 grid the inference used (support [17.2,22.5]), so the DLA(1..k) "
                 "posterior reaches the sub-DLA band natively.",
            MOCK="ALL values are MOCK recovery ratios (2LPT-0 / Saclay-0 / London-0). NO real-LOA "
                 "(loa main-dark) data was read. R0 = est/truth.",
            code_commit=git_commit(),
            deps=["CDDF_analysis/hbi/calccddf_vs_hbi.py", "CDDF_analysis/calc_cddf.py",
                  "CDDF_analysis/cddf_forward/window.py"],
            rederive="conda activate gpdla; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                     "MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE; "
                     "python CDDF_analysis/hbi/calccddf_vs_hbi.py --mock 2lpt0 --second 0 --out full_2lpt0.json "
                     "(repeat --mock saclay0 / london0); then "
                     "python CDDF_analysis/hbi/calccddf_vs_hbi_artifact.py --in 2lpt0=full_2lpt0.json "
                     "saclay0=full_saclay0.json london0=full_london0.json",
            conventions=dict(
                z_range=[2.0, 3.5], snr_cut="SNR_REDSIDE > 2.0 (matches HBI)",
                window="Lya-only (WindowSpec z_min_lyb=True): blue edge = lymanbeta(z_qso); "
                       "no proximity/tail re-cut (stored min/max_z_dla already encode it)",
                p_dla="posterior-weighted, NO hard P_DLA cut (HBI uses P_DLA>0.99)",
                multi_dla="second=0 (DLA(1)-model N-shape carries the TOTAL DLA posterior once per "
                          "sightline); ~7-8% of injected DLAs are the 2nd/3rd in a sightline and are "
                          "NOT separately counted -> calc R0 is ~7% conservative (LOW) at the DLA tier",
                nan_handling="DESI processed files store NaN for negligible/invalid DLA posteriors & "
                             "samples; NaN->0 (posteriors) / NaN->-inf (samples), faithful to calc_cddf's "
                             "own multi-DLA path. calc_cddf's DLA(1) path does NOT do this -> literal "
                             "calc_cddf does not run out-of-the-box on these files.",
                truth="injected HCD truth catalog, windowed IDENTICALLY to the estimator (same Lyb edge, "
                      "same stored [min_z_dla,max_z_dla], same SNR>2 sightline set, same dX).",
            ),
            hbi_reference="forward-path loa0 (CDDF_analysis/hbi/crossmock_transfer_loa0.json); "
                          "2LPT-0 truth constants CDDF_analysis/hbi/subdla_mock_validation.json.",
            hbi_truth_2lpt0=HBI_TRUTH_2LPT0,
            wallclock_s=total_wall,
        ),
        mocks=mocks,
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)
    # quick console table
    for m, d in mocks.items():
        print(f"\n=== {m} (nfiles={d['n_files']} nSL={d['n_sightlines']} dX={d['dX_total']:.0f}) ===")
        for lim in ["20.3", "20.0", "19.5", "band_195_203"]:
            rc = d["R0_calccddf"]["dndx"][lim]; rh = d["hbi_forward_R0"]["dndx"][lim] if d["hbi_forward_R0"] else float("nan")
            print(f"  dNdX {lim:>12}: calc_cddf R0={rc:.3f}   HBI R0={rh:.3f}")


if __name__ == "__main__":
    main()
