"""Regenerate the committed O1 golden snapshot (``o1_golden.npz``).

The golden file regression-locks the O1 CDDF + dN/dX arrays the driver produces on
the SEEDED synthetic fixture (``build_synthetic_cddf`` with its default ``seed=0``),
using the SAME estimator args as ``tests/test_cddf_driver.py``.  Run from the repo
root after an intentional, reviewed change to the estimator output:

    python tests/fixtures/cddf/make_o1_golden.py

It writes ``o1_golden.npz`` next to this file.  The matching test
(``TestGoldenSnapshot::test_golden_matches``) asserts ``np.allclose(rtol=1e-10)``.
"""
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_HERE, "..", "..", "..")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from CDDF_analysis.cddf_forward.driver import compute_o1_products  # noqa: E402
from build_synthetic_cddf_fixture import build_synthetic_cddf  # noqa: E402

# MUST match the constants in tests/test_cddf_driver.py.
_Z_MIN, _Z_MAX = 2.4, 3.1
_LNHI_MIN, _LNHI_MAX, _LNHI_NBINS = 20.3, 22.5, 4
_HUBBLE = 0.7
_DLACAT_KWARGS = dict(sub_dla=False, snr=-2, lowzcut=False, highzcut=False)


def main():
    with tempfile.TemporaryDirectory() as td:
        synth = build_synthetic_cddf(td)
        prod = compute_o1_products(
            synth["processed_file"],
            synth["sample_file"],
            synth["catalog_file"],
            z_min=_Z_MIN,
            z_max=_Z_MAX,
            lnhi_min=_LNHI_MIN,
            lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS,
            hubble=_HUBBLE,
            filter_low_likelihood=0,
            **_DLACAT_KWARGS,
        )
    out = os.path.join(_HERE, "o1_golden.npz")
    np.savez(
        out,
        cddf_logN=prod["cddf"]["logN"],
        cddf_f=prod["cddf"]["f"],
        dndx_z=prod["dndx"]["z"],
        dndx=prod["dndx"]["dndx"],
    )
    print(f"Wrote golden snapshot: {out}")


if __name__ == "__main__":
    main()
