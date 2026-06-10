"""Synthetic 2-DLA CDDF fixture for the O3 multi-DLA deposit tests (contract C4).

Extends ``build_synthetic_cddf`` to the MAX_DLAS=2 (``second_dla=1``) layout that
``calc_cddf.DLACatalogue`` reads when ``second=1``: a 3-column ``model_posteriors``
(``[Null, DLA1, DLA2]`` for ``sub_dla=False``), a ``(N, S, 2)``
``sample_log_likelihoods_dla`` (column 0 = DLA1, column 1 = DLA2), a ``(N, 2)``
``log_likelihoods_dla``, and a ``(N, 1, S)`` ``base_sample_inds`` (the per-spectrum
permutation the 2nd-DLA samples are looked up through, ``k-2 == 0`` axis for
MAX_DLAS=2).

The DLA1 and DLA2 sample log-likelihoods are each made delta-like at a chosen
sample so the per-second deposit has a known, checkable peak.  The 2nd-DLA peak is
addressed THROUGH ``base_sample_inds`` (since ``_get_sample_params(second=1)``
re-indexes the grid by it), so the 2nd DLA can be steered into the SAME (logN, z)
cell as the 1st to make a recoverable close pair.

This is ADDITIVE test scaffolding only — it writes the exact on-disk schema the
estimator reads; no estimator code is touched.
"""
import numpy as np
import h5py
from astropy.table import Table
from scipy.special import logsumexp


def build_synthetic_multidla_cddf(
    out_dir,
    *,
    n_samples=32,
    z_qso=3.5,
    z_min=2.4,
    z_max=3.3,
    snr=5.0,
    lnhi_lo=20.3,
    lnhi_hi=21.9,
    fill_loglike=-60.0,
    # single close-pair sightline (tid 1000): DLA1 + DLA2 both in the SAME cell.
    dla1_logN=20.5,
    dla1_z=2.52,
    dla2_logN=20.6,
    dla2_z=2.58,
    p_exactly_1=0.0,
    p_exactly_2=1.0,
):
    """Build a 2-DLA (MAX_DLAS=2) close-pair fixture; return file paths + truth.

    One ACTIVE sightline (TARGETID 1000) hosts two recovered DLAs: DLA1 at
    ``(dla1_logN, dla1_z)`` and DLA2 at ``(dla2_logN, dla2_z)``.  With
    ``model_posteriors[:, 2] = p_dla2`` the 2nd-DLA model carries probability mass
    so a ``second_dla=1`` deposit sums both into the close-pair cell.

    NOTE on the estimator's 2nd-DLA path: ``_get_prob_dla_this_bin(second=k)``
    indexes ``model_posteriors[sample_index, col]`` (a quirk of the proven
    inference code we must NOT change), so the number of spectra ROWS must be
    ``>=`` the number of samples ``S`` for the sample indices to stay in-bounds.
    We therefore pad with ``S`` inactive (p_dla=0, filtered-out) sightlines.  Those
    padding rows carry the SAME 2nd-DLA model probability column value so the
    indexed lookup returns a constant ``p_dla2``.
    """
    out_dir = str(out_dir)
    S = int(n_samples)
    N = S + 1  # 1 active close-pair + S inactive padding rows (>= S, in-bounds)

    log_nhi_samples = np.linspace(lnhi_lo, lnhi_hi, S)
    offset_samples = np.linspace(0.02, 0.98, S)
    z_of_sample = z_min + (z_max - z_min) * offset_samples

    def _peak_index(logN, z):
        d2 = (log_nhi_samples - logN) ** 2 + (z_of_sample - z) ** 2
        return int(np.argmin(d2))

    j1 = _peak_index(dla1_logN, dla1_z)
    # The estimator's 2nd-DLA probability lookup reads model_posteriors[j2, 2]
    # (sample-index-as-row quirk); pin j2 = 0 so it hits the ACTIVE row (row 0),
    # which is the only row carrying p_dla2.  DLA2's (logN, z) is therefore the
    # GRID sample 0 -> (lnhi_lo, z at offset 0.02), still inside the close-pair
    # cell (0,0) for the test's bin edges.
    j2 = 0

    # sample_log_likelihoods_dla: (N, S, 2). Column 0 = DLA1 delta at j1.
    sll = np.full((N, S, 2), float(fill_loglike), dtype=float)
    sll[0, j1, 0] = 0.0
    # Column 1 = DLA2 delta at j2=0 (identity base_sample_inds -> same grid).
    sll[0, j2, 1] = 0.0

    # log_likelihoods_dla: (N, 2): normalizer per DLA model so that
    # sum_j exp(log_norm_like) == 1 (the assert in calc_cddf) for the active row.
    lld = np.zeros((N, 2), dtype=float)
    lld[0, 0] = logsumexp(sll[0, :, 0]) - np.log(S)
    lld[0, 1] = logsumexp(sll[0, :, 1]) - np.log(S)

    # base_sample_inds: (N, 1, S) for MAX_DLAS=2 (axis k-2 == 0). 1-INDEXED on disk
    # (calc_cddf subtracts 1). Identity permutation -> 2nd-DLA grid == 1st-DLA grid.
    base_sample_inds = np.tile(np.arange(1, S + 1), (N, 1, 1)).astype(np.int64)

    # model_posteriors [Null, P(exactly 1 DLA), P(exactly 2 DLAs)] (sub_dla=False).
    # Only the ACTIVE row 0 carries DLA mass; padding rows are Null (inactive,
    # filtered out). Row 0's column 2 = P(exactly 2) is what the j2=0 lookup returns
    # for the 2nd-DLA deposit; p_dla = col1+col2 drives the single-DLA deposit.
    mp = np.zeros((N, 3), dtype=float)
    mp[0, 1] = float(p_exactly_1)
    mp[0, 2] = float(p_exactly_2)
    mp[0, 0] = 1.0 - mp[0, 1] - mp[0, 2]
    mp[1:, 0] = 1.0  # padding rows: pure Null -> filtered out

    target_ids = np.concatenate(
        [np.array([1000], dtype=np.int64), np.arange(5000, 5000 + S, dtype=np.int64)]
    )

    processed_file = f"{out_dir}/processed_multidla.h5"
    with h5py.File(processed_file, "w") as f:
        f["min_z_dlas"] = np.full(N, z_min, dtype=float)
        f["max_z_dlas"] = np.full(N, z_max, dtype=float)
        f["z_qsos"] = np.full(N, z_qso, dtype=float)
        f["target_ids"] = target_ids
        f["snrs"] = np.full(N, snr, dtype=float)
        f["model_posteriors"] = mp
        f["sample_log_likelihoods_dla"] = sll
        f["log_likelihoods_dla"] = lld
        f["base_sample_inds"] = base_sample_inds

    sample_file = f"{out_dir}/samples_multidla.h5"
    with h5py.File(sample_file, "w") as f:
        f["offset_samples"] = offset_samples.reshape(S, 1)
        f["log_nhi_samples"] = log_nhi_samples.reshape(S, 1)

    catalog_file = f"{out_dir}/catalog_multidla.fits"
    Table({"TARGETID": target_ids, "Z": np.full(N, z_qso, dtype=float)}).write(
        catalog_file, overwrite=True
    )

    return {
        "processed_file": processed_file,
        "sample_file": sample_file,
        "catalog_file": catalog_file,
        "n_spec": N,
        "n_samples": S,
        "z_min": z_min,
        "z_max": z_max,
        "dla1": (dla1_logN, dla1_z),
        "dla2": (dla2_logN, dla2_z),
        "log_nhi_samples": log_nhi_samples,
    }
