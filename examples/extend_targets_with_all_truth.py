"""Take a legacy 9-column targets.tsv and add the all_truth_z / all_truth_nhi
columns by re-reading the per-mock hcd_truth_cat.fits.
"""

import argparse, csv, os, fitsio


MOCKS = {
    "saclay": "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124",
    "2lpt":   "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    truth = {}
    for mock, root in MOCKS.items():
        hcd = fitsio.read(os.path.join(root, "hcd_truth_cat.fits"), ext=1)
        for h in hcd:
            truth.setdefault((mock, int(h["TARGETID"])), []).append(
                (float(h["Z"]), float(h["NHI"])))

    with open(args.src) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        rows = list(rdr)

    out_fields = list(rdr.fieldnames or []) + ["all_truth_z", "all_truth_nhi"]
    with open(args.out, "w") as f:
        f.write("\t".join(out_fields) + "\n")
        for r in rows:
            key = (r["mock"], int(r["target_id"]))
            entries = sorted(truth.get(key, []), key=lambda zn: zn[0])
            r["all_truth_z"]   = ",".join(f"{z:.4f}" for z, _ in entries)
            r["all_truth_nhi"] = ",".join(f"{n:.3f}" for _, n in entries)
            f.write("\t".join(r[k] for k in out_fields) + "\n")
    print(f"[saved] {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
