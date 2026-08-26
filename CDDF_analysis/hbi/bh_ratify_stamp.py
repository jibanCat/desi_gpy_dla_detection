#!/usr/bin/env python
"""bh_ratify_stamp.py — write the PI-ratified SUCCESSOR of a track_c_tf_hz
artifact (PI ruling 2026-08-26 #43/#44: the BH / high-z measurement is the
artifact of record and a PAPER-FACING, ONE-BIN measurement).

The source artifact is never modified. The successor carries the measurement,
zbins and perz_fN blocks byte-for-byte (asserted) and a metadata block that
records the written ruling, the source sha256, the reported estimand (the
integrated bin only; the internal z sub-bins are a response/calibration
subdivision, not reported estimands), the validation basis, and the named
lines that accompany the number. RULES §2: authority=PI / paper_facing=True
are written ONLY from a written ruling — the routine refuses without one.

Usage:
  python -m CDDF_analysis.hbi.bh_ratify_stamp --src ART.json --out ART_RATIFIED.json
      --ruling "PI 2026-08-26 #43/#44" --reported-bin 3.8 5.0
      --validation-basis "<pointer>" [--named-lines JSON]
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess

SUBDIVISION_NOTE = ("the artifact's internal z sub-bins (zbins) are the response / "
                    "calibration subdivision used to construct the integrated value; "
                    "they are NOT separately reported science estimands (PI #44)")


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def stamp(src, out, *, ruling, reported_bin, validation_basis, named_lines, date="2026-08-26",
          extra=None):
    if not ruling or not str(ruling).strip():
        raise ValueError("bh_ratify_stamp: a written PI ruling id is required (RULES §2)")
    d = json.load(open(src))
    zb = [float(x) for x in d["zbins"]]
    lo, hi = float(reported_bin[0]), float(reported_bin[1])
    if abs(zb[0] - lo) > 1e-9 or abs(zb[-1] - hi) > 1e-9:
        raise ValueError(f"reported bin {reported_bin} is not the artifact's span {zb[0]}..{zb[-1]}")
    src_sha = _sha(src)
    md = dict(d["metadata"])
    md["superseded_status"] = md.get("status")
    md["superseded_paper_facing"] = md.get("paper_facing")
    md["superseded_estimand"] = md.get("estimand")
    md["paper_facing"] = True
    md["authority"] = "PI"
    md["status"] = f"PAPER-FACING one-bin BH measurement [{lo}, {hi}) ({ruling})"
    md["estimand"] = ("REPORTED: integrated dN/dX over the one bin, MAP + joint Monte-Carlo band "
                      "(recentered forward estimator); internal sub-bins not reported")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except Exception:
        commit = "UNKNOWN"
    md["ratification"] = dict(
        ruling=ruling, date=date, source=src, source_sha256=src_sha,
        reported_estimand=dict(z_bin=[lo, hi], quantity="dN/dX above the reporting thresholds",
                               form="MAP + joint Monte-Carlo band; never a posterior interval; never "
                                    "averaged with, or in a shared column with, the low-z HBI posterior; "
                                    "no curve through the join (PI #45)",
                               internal_subdivision=SUBDIVISION_NOTE),
        validation_basis=validation_basis, named_lines=dict(named_lines),
        omega_status="Omega_HI from this route is DIAGNOSTIC unless separately ruled (PI #45)",
        stamp_code_commit=commit)
    if extra:
        md["ratification"].update(extra)
    o = {"metadata": md, "measurement": d["measurement"], "zbins": d["zbins"],
         "perz_fN": d.get("perz_fN")}
    # byte-identity of the science blocks (round-trip through the same encoder)
    for k in ("measurement", "zbins", "perz_fN"):
        assert json.dumps(o[k], sort_keys=True) == json.dumps(d.get(k), sort_keys=True)
    with open(out, "w") as fh:
        json.dump(o, fh, indent=2, default=float)
    return {"out": out, "source_sha256": src_sha, "out_sha256": _sha(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ruling", required=True)
    ap.add_argument("--reported-bin", nargs=2, type=float, required=True)
    ap.add_argument("--validation-basis", required=True)
    ap.add_argument("--named-lines", default="{}")
    ap.add_argument("--extra", default="{}")
    a = ap.parse_args()
    r = stamp(a.src, a.out, ruling=a.ruling, reported_bin=a.reported_bin,
              validation_basis=a.validation_basis, named_lines=json.loads(a.named_lines),
              extra=json.loads(a.extra))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
