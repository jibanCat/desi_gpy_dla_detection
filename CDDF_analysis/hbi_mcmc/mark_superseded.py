#!/usr/bin/env python
"""mark_superseded.py — stamp SUPERSEDED provenance onto old artifacts
(PI ruling 2026-08-21 #9: supersede with provenance metadata; do not delete
or hide; retain for reproducibility and historical comparison).

Only the ``.provenance.json`` sidecar is written (created if absent). The
NPZ is never opened for writing. Idempotent per successor.

  python CDDF_analysis/hbi_mcmc/mark_superseded.py --by NEW.npz --reason ... \
      --ruling "PI 2026-08-21 #9" --date 2026-08-21 OLD.npz [OLD2.npz ...]
"""
from __future__ import annotations
import argparse
import json
import os


def supersede_record(prov, *, superseded_by, reason, ruling, date):
    out = dict(prov or {})
    hist = list(out.get("superseded") or [])
    if not any(h.get("superseded_by") == superseded_by for h in hist):
        hist.append(dict(superseded_by=superseded_by, reason=reason,
                         ruling=ruling, date=date, retained=True,
                         note=("artifact retained unchanged for reproducibility "
                               "and historical comparison; not for new inference")))
    out["superseded"] = hist
    out["status"] = "SUPERSEDED"
    return out


def apply(npz_path, *, superseded_by, reason, ruling, date):
    side = npz_path[:-4] + ".provenance.json"
    prov = json.load(open(side)) if os.path.exists(side) else {}
    out = supersede_record(prov, superseded_by=superseded_by, reason=reason,
                           ruling=ruling, date=date)
    with open(side, "w") as f:
        json.dump(out, f, indent=1)
    return side


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old", nargs="+")
    ap.add_argument("--by", required=True, nargs="+",
                    help="successor path(s); one, or one per OLD in order")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--ruling", required=True)
    ap.add_argument("--date", required=True)
    a = ap.parse_args(argv)
    by = a.by if len(a.by) == len(a.old) else [a.by[0]] * len(a.old)
    for old, new in zip(a.old, by):
        print("superseded:", apply(old, superseded_by=new, reason=a.reason,
                                   ruling=a.ruling, date=a.date))


if __name__ == "__main__":
    main()
