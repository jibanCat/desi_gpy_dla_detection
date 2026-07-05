"""stack_qso_rest_lyman_limit.py — D0 of the Lyman-limit-drop LLS estimator.

Build a QSO-REST-frame composite of quasar spectra, binned by z_QSO, so the mean
flux decrement blueward of the rest-frame 911.76 Å Lyman limit can be measured (the
PW09 / Worseck+2014 mean-free-path method). This is the STACK step only (D0); the
continuum model, Lyman-series removal, and tau_eff / lambda_mfp fit are later steps.

Reuses the PR#8 spectrum machinery VERBATIM (no change to shared code):
  * preload_spectra.preload_2lpt_simple._read_one_healpix_file  — mock coadd reader
  * examples.stack_real_loa_dlas._resample_spectrum             — mask/shift/normalize/resample
  * examples.stack_real_loa_dlas._sigma_clip_median             — 3-sigma-clip median stack
The ONLY change vs the PR#8 absorber-rest stack is the frame: we pass z = z_QSO (not
z_absorber), so `rest = wave_obs/(1+z_QSO)` and the composite is in the quasar rest
frame. `_resample_spectrum` normalizes in its [1410,1520] Å rest window — the line-free
shelf that brackets 1450 Å in the QSO frame (PW09's normalization anchor).

MOCK NOTE (2LPT-0): z_QSO caps at ~3.81, so the highest bin is [3.6,3.8]; rest 912 Å is
in-band (observed >= 3600 Å) for z_QSO >= 2.95.

Run (smoke, ~20 healpix):
  python examples/stack_qso_rest_lyman_limit.py --limit-healpix 20 --out /tmp/qso_rest_d0
Full:
  python examples/stack_qso_rest_lyman_limit.py --out /tmp/qso_rest_d0
Requires the gpdla/desi env (desispec + healpy).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from preload_spectra.preload_2lpt_simple import (
    _read_one_healpix_file, _spec_path, _healpix_for_radec, _build_targetid_filter)
from examples.stack_real_loa_dlas import _resample_spectrum, _sigma_clip_median

LYMAN_LIMIT = 911.76  # H I rest-frame Lyman limit (Å)

DEF_MOCKDIR = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
               "v2.8.5/mock-0/loa-124")
# z_QSO bins tuned to the mock's real n(z) (caps ~3.81 -> top bin is [3.6,3.8]).
Z_QSO_BINS = [(2.95, 3.0), (3.0, 3.3), (3.3, 3.6), (3.6, 3.9)]
# QSO-rest resample grid (reuse PR#8's expression); 700 floor covers below the limit.
REST_GRID = 10.0 ** np.arange(np.log10(700.0), np.log10(1600.0), 1e-4)


def _bin_index(z):
    for i, (lo, hi) in enumerate(Z_QSO_BINS):
        if lo <= z < hi:
            return i
    return -1


def build_composite(mockdir: Path, limit_healpix: int | None, exclude_bal: bool = True):
    """Stream mock spectra, resample to QSO rest, sigma-clip-median per z_QSO bin."""
    zcat = Table.read(mockdir / "zcat.fits")
    z = np.asarray(zcat["Z"], float)
    zwarn = np.asarray(zcat["ZWARN"], float) if "ZWARN" in zcat.colnames else np.zeros(len(zcat))
    keep = (z >= 2.95) & (z < 3.9) & (zwarn == 0)
    keep &= _build_targetid_filter(zcat, mockdir, exclude_hcd=False, exclude_bal=exclude_bal)
    zcat = zcat[keep]
    print(f"[zcat] {len(zcat)} QSOs after z_QSO in [2.95,3.9), ZWARN==0, "
          f"{'non-BAL' if exclude_bal else 'BAL-in'}")

    tid = np.asarray(zcat["TARGETID"])
    zq = np.asarray(zcat["Z"], float)
    hpx = _healpix_for_radec(np.asarray(zcat["TARGET_RA"], float),
                             np.asarray(zcat["TARGET_DEC"], float))
    zq_of = dict(zip(tid.tolist(), zq.tolist()))

    uniq = np.unique(hpx)
    if limit_healpix is not None:
        uniq = uniq[:limit_healpix]
    print(f"[hpx] {len(uniq)} healpix files to read")

    n_grid = REST_GRID.size
    stacks = [[] for _ in Z_QSO_BINS]   # list of resampled arrays per z bin
    n_read = n_used = n_skip = 0
    t0 = time.time()
    for k, hp in enumerate(uniq):
        specfile = _spec_path(mockdir, int(hp))
        if not specfile.exists():
            continue
        tids_here = tid[hpx == hp].tolist()
        try:
            rows = _read_one_healpix_file(specfile, tids_here)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] healpix {hp} read failed: {type(e).__name__}: {e}")
            continue
        for (t, wave, flux, ivar, mask) in rows:
            n_read += 1
            zz = zq_of.get(int(t))
            if zz is None:
                continue
            b = _bin_index(zz)
            if b < 0:
                continue
            res = _resample_spectrum(flux.copy(), ivar, mask, wave, zz, REST_GRID)
            if res is None:
                n_skip += 1
                continue
            stacks[b].append(res)
            n_used += 1
        if (k + 1) % 25 == 0:
            print(f"  ...{k+1}/{len(uniq)} healpix, {n_used} spectra stacked "
                  f"({time.time()-t0:.0f}s)")

    curves, counts = [], []
    for b, (lo, hi) in enumerate(Z_QSO_BINS):
        if stacks[b]:
            arr = np.vstack(stacks[b])
            curve, cnt = _sigma_clip_median(arr)
        else:
            curve = np.full(n_grid, np.nan)
            cnt = np.zeros(n_grid, int)
        curves.append(curve)
        counts.append(cnt)
        print(f"[bin z_QSO {lo:.2f}-{hi:.2f}] {len(stacks[b])} spectra")
    meta = dict(n_read=n_read, n_used=n_used, n_skip_norm=n_skip,
                n_healpix=int(len(uniq)), exclude_bal=exclude_bal)
    return REST_GRID, np.array(curves), np.array(counts), meta


def rebin(rest_grid, curve, counts, dlam=1.0, lam_lo=820.0, lam_hi=1300.0):
    """Count-weighted rebin of a composite onto uniform dlam-Å bins (noise reduction)."""
    edges = np.arange(lam_lo, lam_hi + dlam, dlam)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(rest_grid, edges) - 1
    out = np.full(ctr.size, np.nan)
    for j in range(ctr.size):
        m = (idx == j) & (counts >= 50) & np.isfinite(curve)
        if m.sum() >= 3:
            out[j] = np.average(curve[m], weights=counts[m])
    return ctr, out


def plot_composite(rest_grid, curves, counts, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for b, (lo, hi) in enumerate(Z_QSO_BINS):
        ctr, c = rebin(rest_grid, curves[b], counts[b], dlam=1.0)
        if np.isfinite(c).sum() < 10:
            continue
        ax.plot(ctr, c, lw=1.5, label=f"z$_Q$ {lo:.2f}-{hi:.2f}  (N={int(np.nanmax(counts[b]))})")
    ax.axvline(LYMAN_LIMIT, color="k", ls="--", lw=1)
    ax.text(LYMAN_LIMIT + 4, 0.05, "912 Å\nLyman limit", fontsize=9)
    ax.axvline(1215.67, color="gray", ls=":", lw=0.8); ax.text(1200, 0.9, "Ly$\\alpha$", fontsize=8, color="gray")
    ax.axvline(1025.72, color="gray", ls=":", lw=0.8); ax.text(1010, 0.9, "Ly$\\beta$", fontsize=8, color="gray")
    ax.set_xlabel("QSO rest wavelength (Å)"); ax.set_ylabel("mean normalized flux (1450 Å = 1)")
    ax.set_title("D0: QSO-rest composite — the Lyman-limit decrement below 912 Å (2LPT-0)")
    ax.set_ylim(-0.05, 1.15); ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=135, bbox_inches="tight")
    print(f"[saved] {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mockdir", default=DEF_MOCKDIR)
    ap.add_argument("--out", default="/tmp/qso_rest_d0")
    ap.add_argument("--limit-healpix", type=int, default=None,
                    help="stack only the first N healpix (fast smoke).")
    ap.add_argument("--bal-in", action="store_true", help="do NOT exclude BALs (systematic test).")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rest_grid, curves, counts, meta = build_composite(
        Path(args.mockdir), args.limit_healpix, exclude_bal=not args.bal_in)
    print(f"[meta] {meta}")
    npz = os.path.join(args.out, "qso_rest_composite.npz")
    np.savez(npz, rest_grid=rest_grid, curves=curves, counts=counts,
             z_bins=np.array(Z_QSO_BINS), **meta)
    print(f"[saved] {npz}")
    plot_composite(rest_grid, curves, counts, os.path.join(args.out, "qso_rest_composite.png"))
    # D0 unit checks
    ok_break = False
    for b in range(len(Z_QSO_BINS)):
        c = curves[b].copy(); c[counts[b] < 50] = np.nan
        red = np.nanmedian(c[(rest_grid > 1250) & (rest_grid < 1400)])
        blue = np.nanmedian(c[(rest_grid > 880) & (rest_grid < 905)])
        if np.isfinite(red) and np.isfinite(blue):
            print(f"[check] z-bin {b}: <flux> 1250-1400={red:.3f}  880-905(sub-LL)={blue:.3f}  "
                  f"decrement={1-blue/red:.2f}")
            if blue < red:
                ok_break = True
    print(f"[D0 unit check] Lyman-limit decrement present in >=1 bin: {ok_break}")


if __name__ == "__main__":
    main()
