"""dash_omega.py — READ-ONLY: Omega[20.3,21.6] all-z bias for every stored final-ladder
run, using the COMMITTED helper validation.fp_ladder.ladder_table.paper_omega_20p3_21p6
(which imports the paper's own reduction weights read-only).

Motivation: the run JSONs' `thresholds.omega_allz` is the SUB-DLA window [19.5,20.3),
not the paper's Omega[20.3,21.6]; the released systematics table carries no Omega column
at all.  This is a pure read-out of stored f-draws: no fit, no sampling, nothing written
into any frozen product.
"""
import json, glob, os, re, sys
import numpy as np
sys.path.insert(0, "/home/mfho/wt_abs_diag_2026-09")
from validation.fp_ladder.ladder_table import paper_omega_20p3_21p6

R = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/final/runs"
OUT = ("/tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/"
       "323b5500-b134-4e5f-8dd1-1292b65f6a96/scratchpad/dash_omega.json")


def main():
    rows = []
    for f in sorted(glob.glob(R + "/*/*_fdraws.npz")):
        base = os.path.basename(f)[:-len("_fdraws.npz")]
        arm = os.path.basename(os.path.dirname(f))
        js = os.path.join(os.path.dirname(f), base + ".json")
        if not os.path.exists(js):
            continue
        j = json.load(open(js))
        pk = j["pack"]
        om = paper_omega_20p3_21p6(f, pack=None)
        if om is None or "unavailable" in om:
            print("UNAVAILABLE", base, om); continue
        fam = [x for x in ("2lpt0", "london0", "saclay0") if x in base][0]
        rows.append(dict(arm=arm, file=base, family=fam,
                         seed=int(re.search(r"_s(\d{8})", base).group(1)),
                         ladder=j["diagnostics"].get("ladder"),
                         omega_bias_pct=om["median_bias_pct"],
                         omega_hw68_pct=100 * 0.5 *
                         (om["post_p16_50_84"][2] - om["post_p16_50_84"][0]) /
                         om["post_p16_50_84"][1],
                         subdla_omega_bias_pct=(j["thresholds"].get("omega_allz") or {}).get("median_bias_pct"),
                         subdla_key=(j["thresholds"].get("omega_allz") or {}).get("key"),
                         a0=j["run_config"].get("a0") if "run_config" in j else None,
                         argv=" ".join(j["run_config"].get("argv", []))))
        print(rows[-1]["arm"], rows[-1]["file"], rows[-1]["omega_bias_pct"])
    json.dump(rows, open(OUT, "w"), indent=1)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
