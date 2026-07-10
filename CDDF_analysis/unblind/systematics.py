"""systematics.py -- the DLA-tier carried-systematics table, as DATA (not prose).

Downstream plots draw error bands / annotations from these rows.  Each row records
where its magnitude comes from and whether it was substantiated against a COMMITTED
routine or stamped artifact.  Anything traceable only to private notes / a superseded
estimand is marked ``UNVERIFIED`` rather than asserted.

Band relation semantics
-----------------------
``band_relation`` says whether the systematic is already inside the plotted Monte-Carlo
band or must be drawn separately:
  INSIDE  -- captured by the MC / independent statistical band on the artifact.
  OUTSIDE -- a calibration-transfer / model-choice / extrapolation term the MC band
             does NOT propagate; a plotter must add it explicitly.

The MC band on the headline is STATISTICAL only (C/rho calibration + real-sightline
bootstrap + NHI-measurement variance about a FROZEN calibration); every calibration-
transfer systematic below is therefore OUTSIDE it.
"""

from __future__ import annotations

from dataclasses import dataclass

VERIFIED = "VERIFIED"
UNVERIFIED = "UNVERIFIED"
# A number can be committed AND still not re-derivable: the generating artifact may name a
# commit that does not contain its routine. That is ORPHANED, and it is not VERIFIED.
ORPHANED = "ORPHANED (committed literal; artifact stamp does not contain its routine)"

INSIDE = "INSIDE"
OUTSIDE = "OUTSIDE"


@dataclass
class Systematic:
    name: str
    size: str                 # human-readable magnitude (kept as a string; may be a range)
    sign: str                 # 'one-sided (down)' / 'two-sided' / 'unknown'
    applies_to: str           # e.g. 'Omega(>=20.3)'
    source: str               # committed routine / stamped artifact it came from
    status: str               # VERIFIED / UNVERIFIED
    band_relation: str        # INSIDE / OUTSIDE the plotted MC band
    note: str = ""


def carried_systematics() -> list:
    """The carried DLA-tier systematics, verified against committed code where possible.

    Substantiation checked 2026-07-09 against the repo at HEAD d496f42."""
    rows = [
        Systematic(
            name="deep-tail / mean-flux transfer",
            size=("Omega: 12.8% at >=20.3 (london-0 R0=0.8715), 12.0% at >=20.0 (R0=0.8800). "
                  "dN/dX: only 1.9% at >=20.3 (R0=0.9810) -- do NOT draw a 12% band on dN/dX."),
            sign="ONE-SIDED, downward on Omega (Omega under-recovered). NOT +/-12-13%.",
            applies_to="Omega(>=20.0) and Omega(>=20.3); dN/dX only at the 1.9% level",
            source=("CDDF_analysis/hbi/track_c_tf_loa.py:831-834 is a PROSE STRING LITERAL in a "
                    "report function, not a computed number. Its R0 comes from the london-0 "
                    "artifact /scratch/.../tf_london0/track_c_tf_london0.json, stamped cff73cb -- "
                    "a real ancestor commit that does NOT contain "
                    "CDDF_analysis/hbi/track_c_tf_london0.py (first added at b057337). ORPHANED."),
            status=ORPHANED,
            band_relation=OUTSIDE,
            note=("DOMINANT term on Omega. Omega weights the high-N tail (N*f(N)) where forest "
                  "mean-flux + HCD prescription differ most between mock recipe and data -- which "
                  "is WHY Omega's transfer error (12.8%) dwarfs dN/dX's (1.9%). The MC band does "
                  "NOT propagate it. OPEN PI DECISION: apply as a correction (Omega x 1/R0 = "
                  "1.147 at >=20.3) or carry as a one-sided band? These give materially different "
                  "Omega. Re-run track_c_tf_london0.py to produce a clean in-repo stamped artifact "
                  "before publication."),
        ),
        Systematic(
            name="BAL false-positive residual",
            size="~2-6%",
            sign="one-sided (over-count; correcting lowers Omega)",
            applies_to="Omega(>=20.3)",
            source=("CDDF_analysis/diagnostics/bal_metal_fp/balfinder_validation.py "
                    "(committed; fig6 corrected estimand) -> figures/metrics.json "
                    "fp_bal_residual omega_residual_bi CI95 ~[0.027,0.038] (ge20.3), "
                    "[0.010,0.064] (deep) on the 2LPT-0 mock"),
            status=VERIFIED,
            band_relation=OUTSIDE,
            note=("Sub-dominant to the deep-tail term. Routine is committed; its output "
                  "metrics.json is UNTRACKED (mock-derived, public). Already folded into "
                  "the loa0 FP intensity; the ~2-6% is the residual band on top."),
        ),
        Systematic(
            name="metal contamination",
            size="~0.07%",
            sign="one-sided (over-count)",
            applies_to="Omega(>=20.3)",
            source=("mechanism: CDDF_analysis/diagnostics/bal_metal_fp/decompose_highn_fp.py "
                    "by-source flag split (committed). MAGNITUDE traces to private notes "
                    "2026-07-02_bal_metal_lyb_fp_plan.md (NOT a committed stamped artifact)"),
            status=UNVERIFIED,
            band_relation=OUTSIDE,
            note=("Mechanism committed but the specific 0.07% is not pinned to a committed "
                  "stamped output; consistent-in-order-of-magnitude with the tiny AI-BAL "
                  "residual (~0.03%). Negligible either way."),
        ),
        Systematic(
            name="Lyman-beta contamination",
            size="~0 (negligible)",
            sign="one-sided (over-count)",
            applies_to="Omega(>=20.3)",
            source=("mechanism: decompose_highn_fp.py Lyb flag split (committed). "
                    "MAGNITUDE traces to private notes 2026-07-02 plan (no committed stamp)"),
            status=UNVERIFIED,
            band_relation=OUTSIDE,
            note="Reported ~0; not pinned to a committed stamped size.",
        ),
        Systematic(
            name="mean-flux recipe (normalization)",
            size="~1-2%",
            sign="two-sided",
            applies_to="dN/dX (absolute normalization)",
            source=("CDDF_analysis/hbi/track_c_tf_loa.py committed literal (lines ~826-830); "
                    "london-0 overall mean-flux rescale s~1.01-1.02 restored R0 <1%"),
            status=VERIFIED,
            band_relation=OUTSIDE,
            note="Stated, not applied. Real LOA forest mean-flux differs from 2LPT-0.",
        ),
        Systematic(
            name="FP-model bracket (purity_mixture vs loa0)",
            size="report BOTH as a bracket (loa0 = headline)",
            sign="two-sided (discrete model choice)",
            applies_to="Omega, dN/dX",
            source=("two full artifacts: loa0 = run_loa0_headline_full.py @ d496f42 "
                    "(track_c_tf_loa_loa0_restamped.json); purity_mixture = "
                    "track_c_tf_loa.py @ f1784fc (track_c_tf_loa.json). Both routines committed."),
            status=VERIFIED,
            band_relation=OUTSIDE,
            note=("A modeling bracket, not a band: report both headline arms; loa0 is the "
                  "headline per the 4-lens verdict."),
        ),
        Systematic(
            name="z>4 extrapolation (highest z-bin)",
            size="UNBOUNDED",
            sign="unknown",
            applies_to="dN/dX, Omega in the z in [4.0,4.25] bin",
            source=("no committed routine or mock bounds it: 2LPT-0 truth caps at "
                    "z~3.5-3.79 (metadata.max_truth_z~3.786), so no mock constrains the "
                    "bias in the z>4 bin. Bin flagged metadata.z_extrapolated[-1]=True "
                    "(CDDF_analysis/hbi/track_c_tf_loa.py:481-494)."),
            status=UNVERIFIED,
            band_relation=OUTSIDE,
            note=("UNQUANTIFIED / UNBOUNDED error term, not a small correction. The frozen "
                  "completeness g(N,z) has no calibration support above the truth cap; NO "
                  "mock can constrain this bias. Must NOT be presented as if the MC band "
                  "covered it. Distinct from the z=3.75 bin, which is beyond the v2 fit "
                  "ceiling yet still partially truth-supported (see loader.ZBinFlags)."),
        ),
    ]
    return rows


def as_table(rows=None) -> str:
    """Fixed-width text table (magnitudes are systematic sizes, not real-LOA values)."""
    rows = rows or carried_systematics()
    hdr = ("name", "size", "sign", "status", "band")
    widths = (34, 22, 34, 11, 8)
    def fmt(vals):
        return " | ".join(str(v)[:w].ljust(w) for v, w in zip(vals, widths))
    lines = [fmt(hdr), "-+-".join("-" * w for w in widths)]
    for r in rows:
        lines.append(fmt((r.name, r.size, r.sign, r.status, r.band_relation)))
    return "\n".join(lines)
