#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate observed_tilt_shape.npz — the ε=1 power-alternative shape.

Committed generator (review finding F1): the per-bin fractional residual
(obs − μ)/μ of the twin pack under the committed selftest fold at
resp_clamp="both" — the same fold the closure table uses. Deterministic;
the committed npz must reproduce bit-for-bit.

Usage:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python diagnostics_phaseC/threshold_study/make_observed_tilt_shape.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

from CDDF_analysis.hbi_mcmc.pack import load_pack                    # noqa: E402
from CDDF_analysis.hbi_mcmc import forward_selftest as FS            # noqa: E402

PACK = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/"
        "modelA_pack_2lpt0_winlya_only_pad19p0_molly172_bw0p2.npz")


def main():
    pk = load_pack(PACK)
    st = FS.selftest(pk, resp_clamp="both")
    live = (np.asarray(pk.dX, float) > 0)[None, :, :]
    mu_c = np.where(live, st["mu"], 0).sum(axis=(1, 2))
    obs_c = np.where(live, st["counts"], 0).sum(axis=(1, 2))
    frac = np.where(mu_c > 0, (obs_c - mu_c) / np.maximum(mu_c, 1e-30), 0.0)
    out = os.path.join(_HERE, "observed_tilt_shape.npz")
    if os.path.exists(out):
        old = np.load(out)["frac_resid_c"]
        err = float(np.max(np.abs(old - frac)))
        print(f"existing file max|delta| = {err:.2e} "
              f"({'REPRODUCED' if err == 0.0 else 'DIFFERS — investigate'})")
    np.savez(out, frac_resid_c=frac, nhat_edges=np.asarray(pk.nhat_edges),
             note=("(obs-mu)/mu per N-hat bin, twin pack, resp_clamp=both, "
                   "selftest fold; the epsilon=1 power alternative"))
    print("wrote", out)


if __name__ == "__main__":
    main()
