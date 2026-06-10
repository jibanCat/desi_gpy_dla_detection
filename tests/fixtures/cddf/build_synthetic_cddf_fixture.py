"""Synthetic fixture builder for the Pathway-A CDDF estimator (``calc_cddf.DLACatalogue``).

Writes a tiny ``processed`` HDF5 + QMC-sample HDF5 (``.mat``-style) + FITS QSO catalog
into a directory, using the exact schema ``DLACatalogue`` reads for the single-absorber
(``sub_dla=False``, MAX_DLAS=1) case that the 2LPT-0 FILTER-off run uses.

The per-spectrum sample likelihoods can be made "concentrated" (a delta at a chosen
(z, logN) sample) so the probabilistic per-bin count has a known closed form: an active
spectrum with ``p_dla`` deposits exactly ``p_dla`` of expected count into the bin holding
its peak sample. The load-bearing invariant enforced here is

    log_likelihoods_dla[i, 0] = logsumexp(sample_log_likelihoods_dla[i, :, 0]) - log(S)

which makes ``DLACatalogue._log_norm_like`` satisfy ``sum_j exp(log_norm_like_j) == 1``
(the ``assert 0.95 < ... < 1.05`` in calc_cddf.py).
"""
import numpy as np
import h5py
from astropy.table import Table
from scipy.special import logsumexp


def build_synthetic_cddf(
    out_dir,
    *,
    n_spec=4,
    n_samples=256,
    p_dla=(1.0, 1.0, 1.0, 0.0),
    peak_logN=(20.5, 21.0, 21.5, None),
    peak_z=(2.6, 2.8, 3.0, None),
    z_qso=3.5,
    z_min=2.2,
    z_max=3.3,
    snr=5.0,
    lnhi_lo=20.3,
    lnhi_hi=21.9,
    fill_loglike=-60.0,
    convention="mean",
    seed=0,
):
    """Build a synthetic single-absorber CDDF fixture; return a dict of file paths + truth.

    Parameters mirror the knobs a test needs: number of spectra/samples, the per-spectrum
    DLA probability ``p_dla``, and the injected ``(peak_z, peak_logN)`` of each active
    spectrum's delta-like sample. Inactive spectra (``p_dla<=0``) get a diffuse low
    likelihood and are filtered out by the estimator's probability threshold.
    """
    out_dir = str(out_dir)
    N, S = int(n_spec), int(n_samples)
    assert len(p_dla) == N and len(peak_logN) == N and len(peak_z) == N

    # shared QMC sample grid (coordinates only; values set per-spectrum below)
    log_nhi_samples = np.linspace(lnhi_lo, lnhi_hi, S)
    offset_samples = np.linspace(0.02, 0.98, S)  # Halton-like offsets in [0, 1]
    z_of_sample = z_min + (z_max - z_min) * offset_samples

    # per-spectrum sample log-likelihoods: a delta at the peak for active spectra
    sll = np.full((N, S, 1), float(fill_loglike), dtype=float)
    peak_idx = np.full(N, -1, dtype=int)
    for i in range(N):
        if p_dla[i] <= 0:
            continue
        d2 = (log_nhi_samples - peak_logN[i]) ** 2 + (z_of_sample - peak_z[i]) ** 2
        j = int(np.argmin(d2))
        peak_idx[i] = j
        sll[i, j, 0] = 0.0

    # The stored marginal `log_likelihoods_dla` follows one of two conventions:
    #   * "mean" (pre-PR#7): log-mean-exp = logsumexp(sll) - log(S);
    #   * "sum"  (post-PR#7 +log(N) evidence fix): log-sum-exp = logsumexp(sll).
    # A convention-agnostic (self-normalizing) estimator must give sum_j
    # exp(log_norm_like_j) == 1 for BOTH.
    _lse = logsumexp(sll[:, :, 0], axis=1)
    if convention == "sum":
        lld = _lse.reshape(N, 1)
    elif convention == "mean":
        lld = (_lse - np.log(S)).reshape(N, 1)
    else:
        raise ValueError(f"convention must be 'mean' or 'sum', got {convention!r}")

    # model posteriors [Null, DLA(1)] for the single-absorber (sub_dla=False) layout
    mp = np.zeros((N, 2), dtype=float)
    mp[:, 1] = np.asarray(p_dla, dtype=float)
    mp[:, 0] = 1.0 - np.asarray(p_dla, dtype=float)

    target_ids = (1000 + np.arange(N)).astype(np.int64)

    # snr may be a scalar (applied to all) or a per-spectrum array (length N) so a
    # test can make a specific sightline fail the SNR cut (contract C3 active-set).
    snr_arr = np.full(N, float(snr)) if np.ndim(snr) == 0 else np.asarray(snr, float)
    assert snr_arr.shape == (N,)

    processed_file = f"{out_dir}/processed_synth.h5"
    with h5py.File(processed_file, "w") as f:
        f["min_z_dlas"] = np.full(N, z_min, dtype=float)
        f["max_z_dlas"] = np.full(N, z_max, dtype=float)
        f["z_qsos"] = np.full(N, z_qso, dtype=float)
        f["target_ids"] = target_ids
        f["snrs"] = snr_arr
        f["model_posteriors"] = mp
        f["sample_log_likelihoods_dla"] = sll
        f["log_likelihoods_dla"] = lld

    sample_file = f"{out_dir}/samples_synth.h5"
    with h5py.File(sample_file, "w") as f:
        f["offset_samples"] = offset_samples.reshape(S, 1)
        f["log_nhi_samples"] = log_nhi_samples.reshape(S, 1)

    catalog_file = f"{out_dir}/catalog_synth.fits"
    Table({"TARGETID": target_ids, "Z": np.full(N, z_qso, dtype=float)}).write(
        catalog_file, overwrite=True
    )

    return {
        "processed_file": processed_file,
        "sample_file": sample_file,
        "catalog_file": catalog_file,
        "n_spec": N,
        "n_samples": S,
        "n_active": int(np.sum(np.asarray(p_dla) > 0)),
        "p_dla": np.asarray(p_dla, dtype=float),
        "peak_logN": peak_logN,
        "peak_z": peak_z,
        "peak_idx": peak_idx,
        "z_min": z_min,
        "z_max": z_max,
        "log_nhi_samples": log_nhi_samples,
    }
