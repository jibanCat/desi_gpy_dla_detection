#!/usr/bin/env python
"""Build a CONTROL-ONLY tree (clean, forest-hostable, NO injection) to measure b_FP.

After the control-NaN-poisoning fix, write_campaign skips control rows → the control
fibers stay the clean SOURCE flux, so running the GP on this tree measures the GENUINE
false-positive rate on absorber-free sightlines (the b_FP the campaigns need, which
the original runs fabricated to 0 by crashing every control).
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from _gen_common import load_clean_sightlines, finalize_tree
from campaign_grid import build_control_rows, validate_manifest, default_zqso_bins

DEFAULT_MOCK = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                "qq_desi_y3/v2.8.5/mock-0/loa-124")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mockdir", default=DEFAULT_MOCK)
    ap.add_argument("--n_controls", type=int, default=4000)
    ap.add_argument("--n_healpix", type=int, default=200)
    ap.add_argument("--snr_cut", type=float, default=2.0)
    ap.add_argument("--num_lines", type=int, default=31)
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--snr_bins", type=float, nargs="+", default=[2.0, 4.0, 8.0, 1e9])
    a = ap.parse_args()

    clean, csl = load_clean_sightlines(a.mockdir, snr_cut=a.snr_cut, n_healpix=a.n_healpix)
    ctrl = build_control_rows(csl, snr_bins=a.snr_bins, target_controls=a.n_controls,
                              seed=a.seed, zqso_bins=list(default_zqso_bins()))
    validate_manifest(ctrl)
    if not ctrl:
        raise SystemExit("[manifest] ERROR: zero forest-hostable controls built.")
    zq = np.array([r["z_qso"] for r in ctrl])
    print(f"[manifest] {len(ctrl)} CONTROLS (no injection) on "
          f"{len(set(int(r['healpix']) for r in ctrl))} healpix; "
          f"z_QSO [{zq.min():.2f}, {zq.max():.2f}] (all forest-hostable)", flush=True)
    finalize_tree(ctrl, clean, out_root=a.out, mockdir=a.mockdir, num_lines=a.num_lines)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
