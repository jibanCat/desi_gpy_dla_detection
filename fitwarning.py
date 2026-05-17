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
the bitmask. PDLA_SATURATED_FLAG is informational (high-confidence
detection, NOT a quality warning) and is NOT folded into DLAFLAG.

Schema history:
  prior to 2026-05-15: bits 0-2 reserved for template-DLA-fit boundary
    warnings (ZBOUNDARY_COARSE, ZBOUNDARY_REFINE, NHIBOUNDARY_REFINE)
    that were never actually set by the GP-DLA inference code. Removed
    in 2026-05-15 reshuffle; bits renumbered compactly.
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

    # ---- Postprocess catalog flags (tools/postprocess/add_dla_flags.py) ----
    LYBETA_MISID = 2**3       # likely Lyβ misidentification of a higher-z DLA
                              # on the same LOS
                              # (gpy_dla_detection.postprocess.lyb_veto)
    BAL_CAT_OVERLAP = 2**4    # TARGETID is in the mock/real bal_cat.fits.
                              # Stricter than POTENTIAL_BAL (which is the
                              # inference-time region overlap); this is the
                              # "drop-all-bal_cat-TIDs" molly recipe.
    NHI_INCONSISTENT = 2**5   # NHI - k * NHI_ERR < 20.3 (default k=0.5).
                              # The lower 1σ of the NHI estimate falls below
                              # the canonical NHI > 20.3 catalog floor — i.e.
                              # the DLA is not robustly above the floor.

    # All current postprocess bits OR'd together (handy for "clear
    # postprocess flags before re-running add_dla_flags.py")
    _POSTPROCESS_MASK = LYBETA_MISID | BAL_CAT_OVERLAP | NHI_INCONSISTENT

    # Bits known to have been used by ANY past schema for postprocess flags.
    # add_dla_flags.py clears these on every run so dlacats produced under an
    # older schema get cleaned up cleanly when re-postprocessed.
    # 2026-05-15 first schema used bits 6,7,8 for LYBETA / BAL / NHI; the
    # renumber to 3,4,5 left those bits set on already-postprocessed catalogs.
    _LEGACY_POSTPROCESS_BITS = (2**6) | (2**7) | (2**8)
    _ALL_POSTPROCESS_BITS_TO_CLEAR = _POSTPROCESS_MASK | _LEGACY_POSTPROCESS_BITS

    # All inference-time bits OR'd together
    _INFERENCE_MASK = POTENTIAL_BAL | BAD_ZFIT | BAD_NHIFIT
