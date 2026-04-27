"""Read all per-target .pkl files in a smoke batch, build a clean summary
TSV, and (optionally) generate the unified v2 plot for each target.

Usage:
  python examples/finalize_smoke_batch.py \
      --batch-dir out/smoke/batch/eboss \
      --targets   out/smoke/targets.tsv \
      --plot
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import sys

import numpy as np


PRESET_FROM_DIRNAME = {"eboss": "eboss", "y3": "y3", "london": "london"}

UNCONTAMINATED = {
    "saclay": "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/jura-0",
    "2lpt":   "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-0",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-dir", required=True,
                   help="e.g. out/smoke/batch/eboss")
    p.add_argument("--targets", required=True,
                   help="targets.tsv produced by pick_smoke_targets.py")
    p.add_argument("--plot", action="store_true",
                   help="generate the unified v2 plot per target via plot_smoke_v2.py")
    p.add_argument("--data-root",
                   default="/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection")
    return p.parse_args()


def main():
    args = parse_args()
    dirname = os.path.basename(args.batch_dir.rstrip("/"))
    # Accept either "<preset>" or "<preset>_filter<F>_n<N>" directory names.
    preset = dirname.split("_", 1)[0]
    if preset not in PRESET_FROM_DIRNAME:
        raise SystemExit(f"unrecognised preset {preset}")

    # Load targets
    with open(args.targets) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        targets = list(rdr)

    summary_path = os.path.join(args.batch_dir, "summary.tsv")
    fields = ["mock", "tid", "z_qso", "z_dla_truth", "log_nhi_truth", "snr",
              "p_dla", "MAP_z", "z_err", "MAP_logNHI", "logNHI_err",
              "dlogNHI", "dz", "selected_dlas"]
    rows = []

    for t in targets:
        mock = t["mock"]; tid = t["target_id"]
        pkl = os.path.join(args.batch_dir, f"{mock}_{tid}.pkl")
        if not os.path.exists(pkl):
            print(f"[skip] no pkl for {mock} {tid}")
            continue
        with open(pkl, "rb") as f:
            r = pickle.load(f)

        z_qso = float(r["z_qsos"][0])
        p_dla = float(r["p_dlas"][0])
        map_z = float(r["MAP_z_dlas"][0, 0])
        z_err = float(r["z_dla_errs"][0, 0])
        map_n = float(r["MAP_log_nhis"][0, 0])
        n_err = float(r["log_nhi_errs"][0, 0])
        # selected k-DLA: argmax of finite model_posteriors[2:]
        mp = np.asarray(r["model_posteriors"])[0]
        # offset = 1 + num_subdla; for multi-DLA runs num_subdla=1, so DLA cols start at 2
        k = int(np.nanargmax(mp[2:])) + 1 if np.isfinite(mp[2:]).any() else 0

        z_t = float(t["z_dla"]); n_t = float(t["log_nhi"])
        d_n = map_n - n_t if np.isfinite(map_n) else float("nan")
        d_z = map_z - z_t if np.isfinite(map_z) else float("nan")

        rows.append({
            "mock": mock, "tid": tid,
            "z_qso": f"{z_qso:.4f}",
            "z_dla_truth": f"{z_t:.4f}",
            "log_nhi_truth": f"{n_t:.3f}",
            "snr": t["snr"],
            "p_dla": f"{p_dla:.3f}",
            "MAP_z": f"{map_z:.4f}" if np.isfinite(map_z) else "-",
            "z_err": f"{z_err:.4f}" if np.isfinite(z_err) else "-",
            "MAP_logNHI": f"{map_n:.3f}" if np.isfinite(map_n) else "-",
            "logNHI_err": f"{n_err:.3f}" if np.isfinite(n_err) else "-",
            "dlogNHI": f"{d_n:+.3f}" if np.isfinite(d_n) else "-",
            "dz": f"{d_z:+.4f}" if np.isfinite(d_z) else "-",
            "selected_dlas": str(k),
        })

        if args.plot:
            uncont_root = UNCONTAMINATED.get(mock)
            uncont_spec = None
            if uncont_root:
                # Map contaminated spec path → uncontaminated equivalent
                cont = t["spec_path"]
                if "juraLy8-124" in cont:
                    uncont_spec = cont.replace("juraLy8-124", "jura-0")
                elif "loa-124" in cont:
                    uncont_spec = cont.replace("loa-124", "loa-0")
            if uncont_spec and not os.path.exists(uncont_spec):
                uncont_spec = None

            out_png = f"figures/smoke_v2/{mock}_{tid}/v2_{preset}.png"
            cmd = [
                sys.executable, "examples/plot_smoke_v2.py",
                "--pkl", pkl,
                "--specfile", t["spec_path"],
                "--zcat", t["zcat_path"],
                "--target-id", tid,
                "--data-root", args.data_root,
                "--preset", preset,
                "--truth-z", t["z_dla"],
                "--truth-nhi", t["log_nhi"],
                "--snr", t["snr"],
                "--title", f"{preset.upper()} — {mock} {tid}  (truth z={t['z_dla']}, logNHI={t['log_nhi']}, SNR={t['snr']})",
                "--out", out_png,
            ]
            if uncont_spec:
                cmd += ["--specfile-uncontaminated", uncont_spec]
            print(f"[plot] {mock} {tid}")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  failed: {r.stderr.strip().splitlines()[-1] if r.stderr else 'unknown'}")

    # Write summary
    with open(summary_path, "w") as f:
        f.write("\t".join(fields) + "\n")
        for r in rows:
            f.write("\t".join(r[k] for k in fields) + "\n")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
