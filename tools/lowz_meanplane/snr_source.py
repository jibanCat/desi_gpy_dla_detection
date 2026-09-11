"""Canonical per-sightline S/N lookup for the low-z injection campaigns.

DEFECT THIS REPAIRS (2026-09-11, contract C1)
---------------------------------------------
The H2-M / cleanreal real substrates were drawn and stratified on the archive
catalogue column ``RED_SNR`` (``src_archive_catalog.npy`` <- ``loa_full_z2_noR_v2.h5::catalog``
<- QSO_cat v3-altbal ``SNR_REDSIDE``), which is the **median** of flux*sqrt(ivar)
over rest-frame [1420, 1480] A.  Every mock calibration block the campaign is
compared against is indexed on the **mean** of the same pixels -- the finder's own
``SNR_REDSIDE`` (``dlasearch.py:670-676`` @1fd4828).  Same column name, different
statistic: 0 / 926,128 archive sightlines are bit-equal, r = 0.9914, and the >2
cut disagrees on 9,614 sightlines.

The archive ``catalog`` compound dtype carries only ``BLUE_SNR`` / ``RED_SNR``
(``gpy_dla_detection/loa_archive.py:122-123``); there is no mean-S/N column
anywhere in the archive schema, and the archive is 75 GB.  This module therefore
supplies the mean by an int64 TARGETID join against the vendored searched-population
table, and the 75 GB archive is never rewritten.

RULING: the canonical source for path-leg S/N is
``processed-main-dark-<HPX>.h5::snrs``, vendored as ``pack/searched_population.npz``
(C1 candidate handoff 2026-09-11 §9).  ``RED_SNR`` /
``src_archive_catalog.npy`` must NOT be used as an S/N source.

SCIENCE-PRODUCING: this helper defines the systematics substrate.
"""
from __future__ import annotations

import numpy as np

#: The C1 candidate pack's vendored searched population (sha256 81ef9338...).
DEFAULT_PACK = (
    "/nfs/turbo/lsa-cavestru/mfho/paper1_science_handoff/"
    "LOWZ_CLEAN_C1_CANDIDATE_2026-09-10/pack/searched_population.npz"
)

#: ``--snr-source`` choices.  ``archive-median`` is the DEFAULT and reproduces the
#: frozen (defective) plan exactly; ``finder-mean`` is the C1 repair.
ARCHIVE_MEDIAN = "archive-median"
FINDER_MEAN = "finder-mean"
SNR_SOURCE_CHOICES = (ARCHIVE_MEDIAN, FINDER_MEAN)


def load_finder_mean(pack_path: str = DEFAULT_PACK):
    """Return ``(tid_sorted, snr_sorted)`` -- the finder's mean ``SNR_REDSIDE``
    keyed by int64 TARGETID, sorted for ``searchsorted``."""
    sp = np.load(pack_path)
    tid = np.asarray(sp["TARGETID"], dtype=np.int64)
    snr = np.asarray(sp["snr_redside"], dtype=float)
    assert tid.dtype == np.int64, "TARGETID must be int64 (float64 join bug, CKPT10)"
    order = np.argsort(tid, kind="stable")
    return tid[order], snr[order]


def mean_snr_for(targetids, pack_path: str = DEFAULT_PACK):
    """Mean ``SNR_REDSIDE`` for each TARGETID; ``nan`` where the finder did not
    search the sightline (no canonical mean exists -> the sightline is INELIGIBLE).

    Returns ``(snr, found)``.
    """
    tids = np.asarray(targetids, dtype=np.int64)
    st, ss = load_finder_mean(pack_path)
    pos = np.searchsorted(st, tids)
    pos_c = np.clip(pos, 0, len(st) - 1)
    found = st[pos_c] == tids
    snr = np.where(found, ss[pos_c], np.nan)
    return snr, found


def resolve_snr(targetids, archive_median, snr_source=ARCHIVE_MEDIAN,
                pack_path: str = DEFAULT_PACK):
    """Return ``(snr_used, eligible, meta)`` for the requested source.

    ``archive-median`` (default) returns ``archive_median`` untouched, so the
    frozen plan is bit-reproducible.  ``finder-mean`` returns the canonical mean
    and marks sightlines without one ineligible.
    """
    if snr_source not in SNR_SOURCE_CHOICES:
        raise ValueError(f"unknown --snr-source {snr_source!r}; "
                         f"choose from {SNR_SOURCE_CHOICES}")
    med = np.asarray(archive_median, dtype=float)
    if snr_source == ARCHIVE_MEDIAN:
        return med, np.isfinite(med), {"snr_source": ARCHIVE_MEDIAN,
                                       "statistic": "median",
                                       "provenance": "archive catalog RED_SNR"}
    snr, found = mean_snr_for(targetids, pack_path)
    eligible = found & np.isfinite(snr)
    meta = {
        "snr_source": FINDER_MEAN,
        "statistic": "mean",
        "provenance": f"{pack_path}::snr_redside (processed-*.h5::snrs)",
        "n_queried": int(len(snr)),
        "n_with_finder_mean": int(found.sum()),
        "n_without_finder_mean": int((~found).sum()),
        "n_nonfinite_mean": int((found & ~np.isfinite(snr)).sum()),
    }
    return snr, eligible, meta
