"""A catalog_file that is a SUBSET of the processed target_ids (e.g. a
SNR_REDSIDE / BAL-filtered selection) must NOT KeyError in DLACatalogue.__init__;
processed targets absent from the catalog map to real_index=-1 and are excluded
via `condition`. Byte-identical when the catalog covers every processed target.
"""
import os, sys
import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "fixtures", "cddf"))
from build_synthetic_cddf_fixture import build_synthetic_cddf  # noqa: E402
from CDDF_analysis.calc_cddf import DLACatalogue  # noqa: E402


def test_subset_catalog_excludes_missing_targets(tmp_path):
    from astropy.table import Table
    fx = build_synthetic_cddf(str(tmp_path), n_spec=4, n_samples=128,
                              p_dla=(1.0, 1.0, 1.0, 0.0),
                              peak_logN=(20.5, 21.0, 21.5, None),
                              peak_z=(2.6, 2.8, 3.0, None))
    import h5py
    with h5py.File(fx["processed_file"], "r") as f:
        all_tids = np.asarray(f["target_ids"][:]).astype(np.int64)
    # catalog with the FIRST target DROPPED (subset of processed targets)
    sub = all_tids[1:]
    subcat = str(tmp_path / "subset_catalog.fits")
    Table({"TARGETID": sub, "Z": np.full(sub.size, 3.5)}).write(subcat, overwrite=True)

    cat = DLACatalogue(processed_file=fx["processed_file"], sample_file=fx["sample_file"],
                       catalog_file=subcat, sub_dla=0, snr=-2, high_nhi_cut_value=21.9)
    # the dropped target maps to -1 and is excluded; no KeyError raised
    assert cat.real_index[0] == -1
    assert not cat.condition[0]
    assert np.all(cat.condition[1:])  # the rest are kept
