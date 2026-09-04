#!/usr/bin/env python
"""r041_mock_meanflux.py — measure a mock's own Lyα-forest mean flux F̄_mock(z) from the
truth continuum (truth-16 TRUE_CONT) on HCD-free, BAL-free sightlines, so the R-041C
mean-flux-only high-z extrapolation F_high = F_low · F̄_high(z+Δz) / F̄_low(z) uses the
substrate's MEASURED F̄_low rather than an assumed law. Also reports F̄_mock against the
finder's fiducial τ_eff law (0.00246 (1+z)^3.62) as the transfer diagnostic.

Per pixel: z = λ/1215.67 − 1 inside the rest-frame window 1045–1195 Å (away from the Lyβ
and Lyα emission regions), unmasked, finite ivar; F̄(z) = Σ F / Σ C in bins of Δz = 0.05
(ratio of sums → unbiased under symmetric noise). Output JSON: bins, F̄, τ_eff = −ln F̄,
counts, sightline/file provenance.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess

import numpy as np
from astropy.io import fits

LYA = 1215.67


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mockdir", required=True)
    ap.add_argument("--exclude-cats", nargs="*", default=[], help="FITS catalogs whose TARGETIDs are excluded (HCD truth, BAL)")
    ap.add_argument("--zqso-min", type=float, default=2.6)
    ap.add_argument("--n-files", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.mockdir, "spectra-16", "*", "*", "spectra-16-*.fits")))
    rng = np.random.default_rng(a.seed)
    files = [files[i] for i in sorted(rng.choice(len(files), size=min(a.n_files, len(files)), replace=False))]
    excl = set()
    for c in a.exclude_cats:
        excl |= set(np.asarray(fits.open(c)[1].data["TARGETID"], dtype=np.int64).tolist())
    edges = np.arange(1.9, 3.85, 0.05)
    sF = np.zeros(edges.size - 1); sC = np.zeros(edges.size - 1); n = np.zeros(edges.size - 1, int)
    nsl = 0
    for f in files:
        sp = fits.open(f); tr = fits.open(f.replace("spectra-16-", "truth-16-"))
        fm = sp["FIBERMAP"].data; tid = np.asarray(fm["TARGETID"], dtype=np.int64)
        truth = tr["TRUTH"].data; zq_by = dict(zip(np.asarray(truth["TARGETID"], dtype=np.int64), truth["Z"]))
        cont = tr["TRUE_CONT"].data; cont_by = dict(zip(np.asarray(cont["TARGETID"], dtype=np.int64), range(len(cont))))
        # TRUE_CONT grid from the extension header (quickquasars: WMIN=3500, DWAVE=2 A)
        hdr = tr["TRUE_CONT"].header
        cwave = float(hdr["WMIN"]) + float(hdr["DWAVE"]) * np.arange(cont["TRUE_CONT"].shape[1])
        for cam in ("B", "R", "Z"):
            w = sp[f"{cam}_WAVELENGTH"].data; F = sp[f"{cam}_FLUX"].data; IV = sp[f"{cam}_IVAR"].data; M = sp[f"{cam}_MASK"].data
            for i, t in enumerate(tid):
                if t in excl or t not in zq_by or t not in cont_by:
                    continue
                zq = float(zq_by[t])
                if zq < a.zqso_min:
                    continue
                C = np.interp(w, cwave, cont["TRUE_CONT"][cont_by[t]])
                rest = w / (1.0 + zq)
                ok = (rest > 1045.0) & (rest < 1195.0) & (M[i] == 0) & np.isfinite(IV[i]) & (IV[i] > 0) & (C > 0)
                if not ok.any():
                    continue
                z = w[ok] / LYA - 1.0
                k = np.digitize(z, edges) - 1
                good = (k >= 0) & (k < edges.size - 1)
                np.add.at(sF, k[good], F[i][ok][good]); np.add.at(sC, k[good], C[ok][good]); np.add.at(n, k[good], 1)
                if cam == "B":
                    nsl += 1
    Fbar = np.where(sC > 0, sF / np.maximum(sC, 1e-30), np.nan)
    zc = 0.5 * (edges[:-1] + edges[1:])
    fid = np.exp(-0.00246 * (1 + zc) ** 3.62)
    out = dict(mockdir=a.mockdir, files=files, n_sightlines=nsl, z_center=zc.tolist(), z_edges=edges.tolist(), n_pix=n.tolist(),
               Fbar=Fbar.tolist(), taueff=(-np.log(Fbar)).tolist(), Fbar_finder_fiducial=fid.tolist(),
               ratio_mock_over_fiducial=(Fbar / fid).tolist(), exclude_cats=a.exclude_cats,
               method="sum(F)/sum(TRUE_CONT) per z bin over rest 1045-1195 A, unmasked, ivar>0; HCD/BAL sightlines excluded",
               generator_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip())
    json.dump(out, open(a.out, "w"), indent=1)
    sel = (n > 2000)
    print(json.dumps({"n_sightlines": nsl, "z": np.round(zc[sel], 2).tolist(), "Fbar": np.round(Fbar[sel], 3).tolist(), "mock/fiducial": np.round((Fbar / fid)[sel], 3).tolist()}))


if __name__ == "__main__":
    main()
