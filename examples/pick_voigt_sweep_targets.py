"""Pick stratified targets for the Voigt LSF/num_lines sweep.

Picks, per mock (2lpt, saclay, london), N_PER_BIN truth sightlines in
each of three NHI regimes:

    LLS:       17.2 ≤ logNHI < 19.0
    sub-DLA:   19.0 ≤ logNHI < 20.3
    DLA:       20.3 ≤ logNHI < 23.0

Selection criteria (per sightline):
  - z_qso such that the truth absorber is mid-forest (z_qso − 0.5 ≤ z_dla
    ≤ z_qso − 0.05) so it's not at the edge of the search window.
  - SNR (red-side, from snr_cat.fits if available) ≥ a threshold.
  - Exactly ONE truth absorber in the search window (so the bias measurement
    isn't confounded by multi-DLA mixing).

Mock truth catalogs:
  2lpt:    `hcd_truth_cat.fits` (NHI in column LOG_NHI or NHI)
  saclay:  same
  london:  same

Output: TSV with columns the sweep runner expects:
    mock target_id z_qso truth_z_dla truth_log_nhi nhi_regime
    spec_path zcat_path

Usage::

    python examples/pick_voigt_sweep_targets.py \\
        --out out/voigt_sweep/targets.tsv \\
        --n-per-bin 5

NB: You'll need GreatLakes paths for 2LPT (`/nfs/turbo/...`) and
NERSC paths for Saclay / London (`/global/cfs/projectdirs/...` or
`/global/cfs/cdirs/desicollab/...`). Override via flags or the script
will use the GreatLakes mirror defaults.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.table import Table


# ---------------------------------------------------------------------------
# Mock registry — data paths and column names
# ---------------------------------------------------------------------------
MOCK_PATHS = {
    "2lpt": {
        "mock_dir": "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124",
        "spec_dir_relative": "spectra-16",
        "spec_layout": "{healpix//100}/{healpix}/spectra-16-{healpix}.fits",
    },
    "saclay": {
        # NERSC: /global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124
        # GreatLakes mirror: /nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124
        "mock_dir": "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124",
        "spec_dir_relative": "spectra-16",
        "spec_layout": "{healpix//100}/{healpix}/spectra-16-{healpix}.fits",
    },
    "london": {
        # NERSC: /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124
        # No GreatLakes mirror by default — user may need to add one.
        "mock_dir": "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/london/qq_desi_y3/v5.9.5/mock-0/jura-124",
        "spec_dir_relative": "spectra-16",
        "spec_layout": "{healpix//100}/{healpix}/spectra-16-{healpix}.fits",
    },
}


NHI_REGIMES = [
    ("LLS",     17.2, 19.0),
    ("sub-DLA", 19.0, 20.3),
    ("DLA",     20.3, 23.0),
]


def _spec_path_for(mock_dir: Path, healpix: int, layout: str,
                   spec_dir_relative: str) -> Path:
    sub = layout.format(healpix=healpix, **{"healpix//100": healpix // 100})
    return mock_dir / spec_dir_relative / sub


def _radec_to_healpix(ra: np.ndarray, dec: np.ndarray, nside: int = 16) -> np.ndarray:
    """nside=16 NESTED healpix index from RA/DEC, in degrees."""
    import healpy as hp
    theta = np.deg2rad(90.0 - np.asarray(dec))
    phi = np.deg2rad(np.asarray(ra))
    return hp.ang2pix(nside, theta, phi, nest=True)


def pick_for_mock(mock: str, mock_dir: Path, n_per_bin: int,
                   snr_min: float, seed: int) -> list[dict]:
    """Return list of selected targets across all 3 NHI regimes."""
    rng = np.random.default_rng(seed)

    # Load mock truth catalog + zcat.
    hcd_path = mock_dir / "hcd_truth_cat.fits"
    zcat_path = mock_dir / "zcat.fits"
    if not hcd_path.exists() or not zcat_path.exists():
        print(f"[skip {mock}] missing files: {hcd_path.exists()=} {zcat_path.exists()=}")
        return []

    print(f"[mock={mock}] reading truth + zcat from {mock_dir}")
    hcd = Table.read(hcd_path)
    zcat = Table.read(zcat_path)

    # Normalise the NHI column name.
    if "LOG_NHI" in hcd.colnames:
        nhi_col = "LOG_NHI"
    elif "NHI" in hcd.colnames:
        nhi_col = "NHI"
    else:
        raise KeyError(f"{hcd_path}: no LOG_NHI or NHI column")
    print(f"[mock={mock}] hcd col={nhi_col}, n={len(hcd)}")

    # Index zcat by TARGETID for quick lookup.
    zcat_by_tid = {int(r["TARGETID"]): r for r in zcat}

    # Compute healpix for each TARGETID using zcat RA/DEC.
    print(f"[mock={mock}] computing healpix from zcat RA/DEC")

    rows = []
    for regime_name, nhi_lo, nhi_hi in NHI_REGIMES:
        # Filter HCDs to this NHI bin.
        m = (hcd[nhi_col] >= nhi_lo) & (hcd[nhi_col] < nhi_hi)
        bin_hcd = hcd[m]
        # Group by TARGETID, count absorbers per LOS.
        from collections import Counter
        tid_counts = Counter(int(t) for t in bin_hcd["TARGETID"])
        # Single-absorber LOS only (avoid multi-DLA confounds).
        single_tids = {t for t, c in tid_counts.items() if c == 1}
        # Cross-reference with zcat (must have z_qso, RA, DEC).
        candidates: list[dict] = []
        for row in bin_hcd:
            tid = int(row["TARGETID"])
            if tid not in single_tids or tid not in zcat_by_tid:
                continue
            zrow = zcat_by_tid[tid]
            z_qso = float(zrow["Z"])
            z_dla = float(row["Z"])
            log_nhi = float(row[nhi_col])
            # Mid-forest cut: z_qso - 0.5 ≤ z_dla ≤ z_qso - 0.05.
            if not (z_qso - 0.5 <= z_dla <= z_qso - 0.05):
                continue
            # SNR cut (skip if no SNR column available).
            if "SNR" in row.colnames and row["SNR"] < snr_min:
                continue
            # Healpix from RA/DEC.
            try:
                hpx = int(_radec_to_healpix(zrow["TARGET_RA"], zrow["TARGET_DEC"]))
            except Exception:
                continue
            # Build spec path.
            mock_info = MOCK_PATHS[mock]
            sp = _spec_path_for(
                Path(mock_info["mock_dir"]), hpx,
                mock_info["spec_layout"], mock_info["spec_dir_relative"],
            )
            if not sp.exists():
                continue
            candidates.append({
                "mock": mock,
                "target_id": tid,
                "z_qso": z_qso,
                "truth_z_dla": z_dla,
                "truth_log_nhi": log_nhi,
                "nhi_regime": regime_name,
                "spec_path": str(sp),
                "zcat_path": str(zcat_path),
            })
        # Sample n_per_bin (deterministic order).
        if len(candidates) > n_per_bin:
            idx = rng.choice(len(candidates), size=n_per_bin, replace=False)
            idx.sort()
            candidates = [candidates[i] for i in idx]
        print(f"[mock={mock}, regime={regime_name}, NHI∈[{nhi_lo},{nhi_hi})]: "
              f"picked {len(candidates)} of N candidates")
        rows.extend(candidates)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--mocks", default="2lpt,saclay,london",
                   help="Comma-separated subset of {2lpt, saclay, london}")
    p.add_argument("--n-per-bin", type=int, default=5,
                   help="Targets per (mock, NHI regime) — total = mocks × 3 × n")
    p.add_argument("--snr-min", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mock-dir-2lpt", default=None)
    p.add_argument("--mock-dir-saclay", default=None)
    p.add_argument("--mock-dir-london", default=None)
    args = p.parse_args()

    if args.mock_dir_2lpt:
        MOCK_PATHS["2lpt"]["mock_dir"] = args.mock_dir_2lpt
    if args.mock_dir_saclay:
        MOCK_PATHS["saclay"]["mock_dir"] = args.mock_dir_saclay
    if args.mock_dir_london:
        MOCK_PATHS["london"]["mock_dir"] = args.mock_dir_london

    all_rows: list[dict] = []
    for mock in args.mocks.split(","):
        mock = mock.strip()
        if mock not in MOCK_PATHS:
            print(f"[skip] unknown mock: {mock}")
            continue
        mock_dir = Path(MOCK_PATHS[mock]["mock_dir"])
        if not mock_dir.exists():
            print(f"[skip {mock}] mock_dir not found: {mock_dir}")
            continue
        rows = pick_for_mock(mock, mock_dir, args.n_per_bin, args.snr_min, args.seed)
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("[error] no targets picked")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(all_rows[0].keys())
    with args.out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\n[picked] {len(all_rows)} targets total → {args.out}")


if __name__ == "__main__":
    main()
