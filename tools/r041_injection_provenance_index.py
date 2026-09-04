#!/usr/bin/env python3
"""r041_injection_provenance_index.py — ONE ROW PER INJECTION provenance index (PI ruling 2026-09-01 late §32):
injection_id, arm, TARGETID, healpix, source archive + sha, truth z, truth log N, S/N stratum, candidate-status stratum, plan
row, builder commit, prescription, output (injected) archive sha, finder run ID, finder commit, finder config
(MAX_DLAS/SINGLE/FILTER/QMC from the run's BASELINE.env), accepted?, n_accepted, matched z, matched N_HI.
Inputs per arm: --arm LABEL:build_summary.json:outputs_dir:analysis_per_injection.csv:run_id. Private output.
"""
import argparse, csv, glob, json, os, sys


def baseline(outdir):
    p = os.path.join(outdir, "BASELINE.env"); d = {}
    if os.path.isfile(p):
        for line in open(p):
            if "=" in line and not line.startswith("#"):
                k, v = line.rstrip("\n").split("=", 1); d[k.strip()] = v.strip().strip('"')
    return d


def hpx_of(outdir):
    m = {}
    try:
        from astropy.io import fits
        for f in glob.glob(os.path.join(outdir, "figures", "processed", "processed-*.h5")):
            import h5py
            hp = int(os.path.basename(f).split("-")[-1].split(".")[0])
            with h5py.File(f, "r") as h:
                for t in h["target_ids"][...]:
                    m[int(t)] = hp
    except Exception:
        pass
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="LABEL:build_summary.json:outputs_dir:per_injection.csv:run_id")
    ap.add_argument("--plan-label", default="cmp"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = []
    for spec in a.arm:
        lab, bs, od, pi, rid = spec.split(":")
        B = json.load(open(bs)); base = baseline(od); hp = hpx_of(od)
        truth = {(int(r["TARGETID"]), int(r["inj_idx"])): r for r in csv.DictReader(open(B["truth"]))}
        for r in csv.DictReader(open(pi)):
            k = (int(r["TARGETID"]), int(r["inj_idx"])); t = truth.get(k, {})
            rows.append(dict(injection_id=f"{a.plan_label}:{r['wave']}:{r['TARGETID']}:{r['inj_idx']}", arm=lab, TARGETID=r["TARGETID"], healpix=hp.get(int(r["TARGETID"])),
                             source_archive=B["source_archive"], source_archive_sha256=B["source_archive_sha256"], truth_z=r["z_inj"], truth_logN=r["logN"], snr_stratum=r["stratum"], cand_stratum=r.get("has_cand_ge20"),
                             plan=B["plan"], plan_sha256=B["plan_sha256"], plan_row=f"{r['wave']}:{r['inj_idx']}", builder="tools/r041_build_archive.py", builder_commit=B["code_commit"],
                             prescription=B.get("injection_prescription", B["method"]), injector_method=B.get("injector_method", B["method"]), noise_seed_policy=B.get("noise_seed_policy", "see ledger"),
                             output_archive=B["injected_archive"], output_archive_sha256=B["injected_archive_sha256"], truth_csv_sha256=B["truth_sha256"],
                             finder_run_id=rid, finder_commit=base.get("CODE_COMMIT"), finder_config=f"MAX_DLAS={base.get('MAX_DLAS')}/SINGLE={base.get('SINGLE_ABSORBER_MODEL')}/FILTER={base.get('FILTER_LOW_LIKELIHOOD')}/QMC={base.get('NUM_DLA_SAMPLES')}",
                             finder_archive_recorded=base.get("GPDLA_SPECTRA_ARCHIVE", "NOT RECORDED (pre-255664d writer)"),
                             accepted=r["detected"], n_accepted=r.get("n_accepted"), matched_z=(float(r["z_inj"]) + float(r["dz"]) if r.get("dz") not in (None, "", "nan") else None), matched_NHI=r.get("nhat"), accepted_nhats=r.get("accepted_nhats"), p_dla=r.get("p")))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(len(rows), "rows ->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
