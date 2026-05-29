"""
Mask bit definitions for DLA fit warnings + postprocess catalog flags.

Bits 0-2 are inference-time fitter warnings (set by dlasearch.py during
GP fit / sample post-processing). Bits 3+ are postprocess catalog flags
added by tools/postprocess/add_dla_flags.py after inference.

With all bits used as intended,

    cat[cat["DLAFLAG"] == 0]

selects DLAs that passed every inference-time AND postprocess check —
the "clean" production catalog for population statistics, dN/dX, f(N,z),
Ω_HI, etc.

For finer-grained filtering, the individual boolean flag columns
(LYBETA_FLAG, BAL_FLAG, NHI_CONSISTENCY_FLAG, …) are kept alongside
the bitmask. PDLA_SATURATED_FLAG and NHI_CONSISTENCY_FLAG are
informational — they are *selection knobs*, NOT quality warnings — and
are deliberately NOT folded into DLAFLAG, so `DLAFLAG == 0` stays a
meaningful "clean catalog" (it is not swamped by a tunable cut).

Schema history:
  prior to 2026-05-15: bits 0-2 reserved for template-DLA-fit boundary
    warnings (ZBOUNDARY_COARSE, ZBOUNDARY_REFINE, NHIBOUNDARY_REFINE)
    that were never actually set by the GP-DLA inference code. Removed
    in 2026-05-15 reshuffle; bits renumbered compactly.
  2026-05-17: NHI_INCONSISTENT (bit 5) removed from DLAFLAG. The
    2026-05-17 NHI-flag investigation found it is a purity-vs-completeness
    selection knob, not a quality defect (it flagged ~79-86% of rows and
    silently NHI-gated the headline P/C). It is now informational-only,
    surfaced solely via the NHI_CONSISTENCY_FLAG column. See
    docs/notes/2026-05-17_nhi_flag_investigation.md.
"""


class DLAFLAG(object):
    # ---- Inference-time fitter warnings (set by dlasearch.py) ----
    POTENTIAL_BAL = 2**0      # DLA solution overlaps with Lyα or NV BAL,
                              # potential false positive (inference-time
                              # geometric check; differs from BAL_CAT_OVERLAP
                              # below which is a TARGETID lookup against
                              # bal_cat.fits)
    BAD_ZFIT = 2**1           # bad parabola fit to chi2(refined z) surface,
                              # also raised on np.linalg.LinAlgError during
                              # GP processing of this QSO
    BAD_NHIFIT = 2**2         # bad parabola fit to chi2(refined nhi) surface,
                              # also raised on All-NaN slice during processing

    # ---- Postprocess flags folded into DLAFLAG (tools/postprocess/add_dla_flags.py) ----
    LYBETA_MISID = 2**3       # likely Lyβ misidentification of a higher-z DLA
                              # on the same LOS
                              # (gpy_dla_detection.postprocess.lyb_veto)
    BAL_CAT_OVERLAP = 2**4    # TARGETID is in the mock/real bal_cat.fits.
                              # Stricter than POTENTIAL_BAL (which is the
                              # inference-time region overlap); this is the
                              # "drop-all-bal_cat-TIDs" molly recipe.

    # ---- Informational flag — NOT folded into DLAFLAG (its own column) ----
    NHI_INCONSISTENT = 2**5   # NHI - k * NHI_ERR < 20.3 (default k=0.5):
                              # the DLA sits near the 20.3 catalog floor.
                              # This is a *selection knob* (a purity↔
                              # completeness trade), NOT a quality defect, so
                              # since 2026-05-17 it is NOT OR'd into DLAFLAG —
                              # it is surfaced only as the NHI_CONSISTENCY_FLAG
                              # column. The bit value is retained so
                              # add_dla_flags.py can clear it from catalogs
                              # stamped under the pre-2026-05-17 schema.

    # Postprocess bits OR'd into DLAFLAG (NHI_INCONSISTENT deliberately excluded).
    _POSTPROCESS_MASK = LYBETA_MISID | BAL_CAT_OVERLAP

    # Bits add_dla_flags.py clears before re-stamping, so a catalog produced
    # under an older schema is cleaned up on re-postprocess. Includes:
    #   - NHI_INCONSISTENT (bit 5): folded into DLAFLAG pre-2026-05-17.
    #   - bits 6,7,8: the 2026-05-15 first schema's LYBETA/BAL/NHI positions,
    #     left set after the renumber to bits 3,4,5.
    _LEGACY_POSTPROCESS_BITS = NHI_INCONSISTENT | (2**6) | (2**7) | (2**8)
    _ALL_POSTPROCESS_BITS_TO_CLEAR = _POSTPROCESS_MASK | _LEGACY_POSTPROCESS_BITS

    # All inference-time bits OR'd together
    _INFERENCE_MASK = POTENTIAL_BAL | BAD_ZFIT | BAD_NHIFIT
