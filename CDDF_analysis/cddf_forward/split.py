"""Deterministic sightline train/test split for the CDDF pipeline.

WHY
---
The CDDF response matrix ``R`` is *measured* by recovered-vs-truth matching on a
mock run, and the forward inference is then *validated* by closure.  If ``R`` is
built and validated on the SAME sightlines, the closure is circular — it proves
only that the estimator reproduces the data it was tuned on.  To break the
circularity we partition sightlines into two disjoint roles:

  * ``BUILD``   — used to measure the response matrix ``R``;
  * ``HELDOUT`` — reserved for closure validation; never touches the R-build.

The partition is keyed on the immutable QSO ``TARGETID``, NOT on array position:
loaders (2LPT / London / Saclay / LOA) iterate spectra in different orders, so a
position-keyed split would assign the same physical sightline different roles
across loaders and break reproducibility.  By hashing the TARGETID value, the
role of a given sightline is identical everywhere and reproducible from
``(seed, frac_build)`` alone.

Determinism is paramount, so we use ``hashlib.blake2b`` (a stable, unsalted
cryptographic hash) rather than Python's per-process-salted ``hash()``, and we
operate on a canonical big-endian int64 byte representation of the TARGETID so
that a Python ``int`` and a ``numpy.int64`` of the same value map to the same
role.

A ``SplitProvenance`` record stamps an R artifact with the ``(seed, frac_build)``
and a stable hash of the BUILD TARGETIDs, and ``assert_no_leakage`` is the guard
that makes non-circularity *enforced* (raises on any BUILD∩HELDOUT overlap)
rather than merely intended.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

__all__ = [
    "sightline_role",
    "assign_roles",
    "split_masks",
    "SplitProvenance",
    "assert_no_leakage",
]

_DEFAULT_SEED = 20260609
_DEFAULT_FRAC_BUILD = 0.7

# blake2b: keyed/personalized so the mapping depends on ``seed`` deterministically
# (unlike Python's salted ``hash()``).  digest_size=8 -> 64-bit digest.
_DIGEST_SIZE = 8
_MAX_U64 = float(1 << 64)  # normaliser to map the 64-bit digest into [0, 1)


def _seed_personalization(seed: int) -> bytes:
    """Canonical 16-byte personalization string derived from ``seed``.

    ``blake2b``'s ``person=`` field is at most 16 bytes; we use the seed's
    big-endian int representation, zero-padded.  This makes the hash a pure,
    reproducible function of the seed.
    """
    # Mask to 128 bits so any int (incl. numpy ints) gives <=16 bytes.
    return (int(seed) & ((1 << 128) - 1)).to_bytes(16, "big")


def _target_id_bytes(target_id: int) -> bytes:
    """Canonical 8-byte big-endian representation of an int64 TARGETID.

    Accepts a Python ``int`` or ``numpy.int64`` (anything ``int()``-coercible).
    Both map to identical bytes, so they receive identical roles.  TARGETIDs are
    DESI int64; we mask to 64 bits (two's-complement) so negative/edge values
    serialise unambiguously.
    """
    return (int(target_id) & ((1 << 64) - 1)).to_bytes(8, "big")


def _uniform01(target_id: int, seed: int) -> float:
    """Map a TARGETID to a deterministic uniform float in [0, 1)."""
    h = hashlib.blake2b(
        _target_id_bytes(target_id),
        digest_size=_DIGEST_SIZE,
        person=_seed_personalization(seed),
    )
    return int.from_bytes(h.digest(), "big") / _MAX_U64


def sightline_role(
    target_id: int, *, seed: int = _DEFAULT_SEED, frac_build: float = _DEFAULT_FRAC_BUILD
) -> str:
    """Return ``"BUILD"`` or ``"HELDOUT"`` for a single TARGETID.

    Pure function of ``(target_id, seed, frac_build)``: the int64 TARGETID is
    hashed with keyed ``blake2b`` into a uniform draw ``u ∈ [0, 1)``; the role is
    ``"BUILD"`` if ``u < frac_build`` else ``"HELDOUT"``.

    Parameters
    ----------
    target_id : int
        Immutable QSO ``TARGETID`` (Python ``int`` or ``numpy.int64``).
    seed : int, optional
        Split seed; personalizes the hash so a different seed reshuffles roles.
    frac_build : float, optional
        Target BUILD fraction in ``[0, 1]``.  ``1.0`` -> all BUILD, ``0.0`` ->
        all HELDOUT.
    """
    return "BUILD" if _uniform01(target_id, seed) < frac_build else "HELDOUT"


def assign_roles(
    target_ids,
    *,
    seed: int = _DEFAULT_SEED,
    frac_build: float = _DEFAULT_FRAC_BUILD,
) -> np.ndarray:
    """Vectorized ``sightline_role`` over an array of TARGETIDs.

    Returns an array of dtype ``<U7`` of ``"BUILD"`` / ``"HELDOUT"`` labels,
    elementwise-equivalent to :func:`sightline_role`.  Keyed on TARGETID value,
    so the result is invariant under permutation of ``target_ids``.
    """
    tids = np.asarray(target_ids, dtype=np.int64).ravel()
    roles = np.empty(tids.shape, dtype="<U7")
    for i, tid in enumerate(tids):
        roles[i] = (
            "BUILD" if _uniform01(int(tid), seed) < frac_build else "HELDOUT"
        )
    return roles


def split_masks(
    target_ids,
    *,
    seed: int = _DEFAULT_SEED,
    frac_build: float = _DEFAULT_FRAC_BUILD,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(build_mask, heldout_mask)`` boolean arrays.

    The two masks are disjoint and their union covers every input element
    (``heldout_mask == ~build_mask``).  Aligned positionally with the input
    ``target_ids`` so they can index parallel arrays.
    """
    roles = assign_roles(target_ids, seed=seed, frac_build=frac_build)
    build_mask = roles == "BUILD"
    heldout_mask = ~build_mask
    return build_mask, heldout_mask


@dataclass(frozen=True)
class SplitProvenance:
    """Immutable record of a split, to stamp an R artifact.

    Records the ``seed`` and ``frac_build`` that defined the split plus a stable
    hash of the sorted BUILD TARGETIDs, so a response-matrix artifact can record
    exactly which sightlines built it (and a reader can detect a mismatch).
    """

    seed: int
    frac_build: float
    build_tids_hash: str

    @staticmethod
    def _hash_build_tids(build_target_ids) -> str:
        """Stable ``blake2b`` hex of the sorted, de-duplicated BUILD int64 bytes.

        Sorting (and uniquing) internally makes the hash invariant under input
        permutation and dependent only on the SET of BUILD sightlines.
        """
        tids = np.asarray(build_target_ids, dtype=np.int64).ravel()
        canonical = np.unique(tids)  # sorted ascending + de-duplicated
        # Canonical big-endian int64 bytes so the digest is platform-stable
        # regardless of native endianness.
        payload = canonical.astype(">i8", copy=False).tobytes()
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    @classmethod
    def from_target_ids(
        cls,
        target_ids,
        *,
        seed: int = _DEFAULT_SEED,
        frac_build: float = _DEFAULT_FRAC_BUILD,
    ) -> "SplitProvenance":
        """Build provenance directly from the full TARGETID list and split params."""
        tids = np.asarray(target_ids, dtype=np.int64).ravel()
        build_mask, _ = split_masks(tids, seed=seed, frac_build=frac_build)
        return cls(
            seed=int(seed),
            frac_build=float(frac_build),
            build_tids_hash=cls._hash_build_tids(tids[build_mask]),
        )


def assert_no_leakage(build_target_ids, heldout_target_ids, *, ctx: str = "") -> None:
    """Raise ``ValueError`` if BUILD and HELDOUT share any TARGETID.

    This is the guard that makes the train/test split's non-circularity
    *enforced*: an R built on BUILD must never be validated on a sightline that
    also appears in BUILD.

    Parameters
    ----------
    build_target_ids, heldout_target_ids : array-like of int
        The two role sets to check for overlap.
    ctx : str, optional
        Caller-side context prepended to the error message.

    Raises
    ------
    ValueError
        If ``set(build) & set(heldout)`` is non-empty.  The message reports the
        overlap count and the offending IDs.
    """
    build = set(int(t) for t in np.asarray(build_target_ids, dtype=np.int64).ravel())
    held = set(int(t) for t in np.asarray(heldout_target_ids, dtype=np.int64).ravel())
    overlap = sorted(build & held)
    if overlap:
        prefix = f"{ctx}: " if ctx else ""
        shown = overlap[:20]
        more = "" if len(overlap) <= 20 else f" (+{len(overlap) - 20} more)"
        raise ValueError(
            prefix
            + f"BUILD/HELDOUT leakage: {len(overlap)} TARGETID(s) appear in both "
            + "sets, which makes the CDDF closure circular. Offending IDs: "
            + f"{shown}{more}. The split must be keyed on TARGETID so BUILD and "
            + "HELDOUT are disjoint."
        )
