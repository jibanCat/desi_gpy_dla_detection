#!/usr/bin/env python
"""Campaign B — CLOSE-PAIR injectable tree (blending / R-nonlinearity systematic).

Injects TWO absorbers per sightline at varying velocity separation Δv and column
offset ΔN, so the response matrix's non-linearity (two real absorbers recovered as
one, or as a wrong-N single) can be measured against the KNOWN pair truth.  Reuses
the validated clean-select + tree-write path (`_gen_common`); only the grid builder
differs (`build_close_pair_grid`, which now injects BOTH absorbers).  See README §2.
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from _gen_common import load_clean_sightlines, finalize_tree, report_restframe
from campaign_grid import (build_close_pair_grid, build_control_rows, validate_manifest,
                           default_zqso_bins, default_z_grid)

DEFAULT_MOCK = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                "qq_desi_y3/v2.8.5/mock-0/loa-124")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mockdir", default=DEFAULT_MOCK)
    ap.add_argument("--n_per_cell", type=int, default=4,
                    help="distinct sightlines per (logN×z×zqso×snr) cell PER pair config")
    ap.add_argument("--n_controls", type=int, default=1500)
    ap.add_argument("--n_healpix", type=int, default=150)
    ap.add_argument("--snr_cut", type=float, default=2.0)
    ap.add_argument("--num_lines", type=int, default=31)
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--snr_bins", type=float, nargs="+", default=[2.0, 4.0, 8.0, 1e9])
    # First-absorber column grid (focus on sub-DLA/DLA where pairs matter most).
    ap.add_argument("--logN_grid", type=float, nargs="+", default=[19.5, 20.3, 21.0])
    # Velocity separations: below ~400 km/s the single-absorber GP can't resolve the pair.
    ap.add_argument("--dv_kms", type=float, nargs="+", default=[100., 200., 400., 800., 1500.])
    # Second-absorber column offset (logN_true2 = logN_true + dlogN).
    ap.add_argument("--dlogN", type=float, nargs="+", default=[-0.5, 0.0, 0.5])
    a = ap.parse_args()

    zqso_bins = list(default_zqso_bins())
    clean, csl = load_clean_sightlines(a.mockdir, snr_cut=a.snr_cut, n_healpix=a.n_healpix)
    inj = build_close_pair_grid(
        csl, logN_grid=a.logN_grid, z_grid=list(default_z_grid()),
        dv_kms_grid=a.dv_kms, dlogN_grid=a.dlogN, snr_bins=a.snr_bins,
        zqso_bins=zqso_bins, n_per_cell=a.n_per_cell, seed=a.seed, num_lines=a.num_lines)
    if not inj:
        raise SystemExit("[manifest] ERROR: zero pairs built — widen healpix/grid.")
    ctrl = build_control_rows(
        csl, snr_bins=a.snr_bins, target_controls=a.n_controls, seed=a.seed + 1,
        inj_id_start=len(inj), exclude_target_ids={int(r["target_id"]) for r in inj},
        zqso_bins=zqso_bins)
    manifest = list(inj) + list(ctrl)
    validate_manifest(manifest)

    dv = np.array([r["dv_kms"] for r in inj])
    print(f"[manifest] {len(inj)} pairs + {len(ctrl)} ctrl on "
          f"{len(set(int(r['healpix']) for r in inj))} healpix", flush=True)
    print(f"[pairs] Δv km/s {sorted(set(np.round(dv).astype(int).tolist()))}; "
          f"ΔlogN {sorted(set(np.round([r.get('_dlogN', 0.0) for r in inj], 2).tolist()))}",
          flush=True)
    report_restframe(inj, zqso_bins)
    finalize_tree(manifest, clean, out_root=a.out, mockdir=a.mockdir, num_lines=a.num_lines)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
