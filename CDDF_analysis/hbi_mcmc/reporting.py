# -*- coding: utf-8 -*-
"""reporting.py — the ADOPTED reporting configuration (PI decisions 1, 3, 4, 8).

This module is the SINGLE home for four settled PI decisions.  It is pure numpy
(NO jax, NO scipy) on purpose: the extractor runs in the jax-free ``gpdla``
data-plane env and loads it file-directly, exactly like ``pack.py``.

DECISION 1 — the primary reporting window is 19.7 <= log N_HI <= 21.6.
    * the 19.7 FLOOR is not new.  It is the sub-DLA runner's ``NONIDENT_EDGE``
      and it is re-exported here with the SAME meaning (see
      ``NONIDENT_EDGE_SOURCE`` / ``NONIDENT_EDGE_REASON``), never re-derived.
    * the 21.6 CEILING is NEW.  It exists because the measured residual high-N
      excess (finding D2, 1.23-1.80x) sits ABOVE log N ~ 21.6 in every one of
      the 60 configurations of the D1 ladder.
      🔴 IT IS *NOT* WHERE THE RESPONSE STOPS BEING MEASURED.  The frozen
      response's top true-N anchor is at 21.0406-21.2164 depending on the cell
      (``RESPONSE_ANCHOR_MEASURED``), so 0.38-0.56 dex of EXTRAPOLATED response
      sits INSIDE [19.7, 21.6] — inside the one window where Omega_HI is
      authorized, carrying ~28% of the N-weighted Omega.  See
      ``extrapolated_response_inside_window``; this is a STATED LIMIT, not a
      resolved issue.
    * Omega_HI: a TOTAL Omega headline is NOT authorized.  ``omega_decision``
      REFUSES any Omega whose N window is not contained in [19.7, 21.6].  There
      is no tail extrapolation here, by design — inventing one is a PI decision.

DECISION 3 — the LATENT true-N basis is 0.2 dex.  The OBSERVED (n-hat) grid and
    the REPORTING grid stay 0.1 dex.  A 0.1-dex PLOTTING grid is permitted but
    every emitter must carry ``plotting_grid_disclosure`` IN THE SCHEMA, so a
    downstream reader cannot mistake it for independent 0.1-dex information
    resolution.  The merging convention is E4's, unchanged: "f is constant
    across the merged bin" (``basis_groups`` / ``merge_basis_columns`` /
    ``merged_truth`` LIVE HERE NOW and are re-exported by ``e4_probe`` so there
    is exactly one implementation).

DECISION 4 — pad floor 19.0 with the ``molly172`` sub-floor completeness.  The
    padded bins are a LATENT NUISANCE.  ``assert_no_subwindow_bins`` fails
    closed when a PAPER-FACING tier's integration weights reach below 19.7 —
    and NOT every tier is paper-facing: ``subdla_195_203`` and ``all_195_up``
    reach below 19.7 BY DEFINITION, so they are marked NOT_PAPER_FACING with
    the offending bins named rather than silently guarded.  The exact scope is
    enumerated in ``SUBWINDOW_GUARD_SCOPE`` — read it before quoting the guard.
    The clamp x completeness convention dependence is propagated as a NAMED
    per-bin + integrated systematic (``convention_systematic``), not as prose.

DECISION 8 — the malformed "|z| <= 5" criterion is restated with its exact
    mathematical definition in ``Z_CRITERION``.  The two ratio-span tolerances
    remain UNRATIFIED and are not ratified here.

MOCKS ONLY: nothing in this module reads data of any kind.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ReportingGuardError",
    # decision 1
    "NONIDENT_EDGE", "NONIDENT_EDGE_SOURCE", "NONIDENT_EDGE_REASON",
    "RESPONSE_ANCHOR_CEILING", "RESPONSE_ANCHOR_CEILING_REASON",
    "RESPONSE_ANCHOR_MEASURED", "extrapolated_response_inside_window",
    "REPORTING_WINDOW", "REPORTING_WINDOW_LABEL",
    "omega_decision", "OMEGA_RULE",
    "reported_tier_decision", "SUBWINDOW_GUARD_SCOPE",
    "truth_overlap_fractions",
    # decision 3
    "BASIS_WIDTH_DEFAULT", "BASIS_WIDTH_ADOPTED", "OBSERVED_STEP",
    "basis_groups", "merge_basis_columns", "merged_truth", "merged_edges",
    "plotting_grid_disclosure", "PLOTTING_GRID_NOTE",
    # decision 4
    "PAD_FLOOR_ADOPTED", "COMPLETENESS_ADOPTED", "ADOPTED_CONFIG",
    "assert_no_subwindow_bins", "bins_fully_inside", "window_overlap_weights",
    "CONVENTION_SYSTEMATIC", "convention_systematic",
    # decision 8
    "Z_CRITERION", "window_closure_metrics",
]


class ReportingGuardError(RuntimeError):
    """A reported/paper-facing quantity violated an adopted reporting rule."""


# ===========================================================================
# DECISION 1 — the primary reporting window
# ===========================================================================

# REUSED, not re-invented: the sub-DLA runner's non-identifiability edge.  The
# guard layer lives on branch `lls-subdla-cddf`, which is NOT merged here (that
# merge is PI-deferred), so the value cannot be imported.  It is re-declared
# with its source named, and `tests/test_adopted_reporting.py` reads the guard
# branch through `git show` and asserts the two agree — a real pin, not a
# comment.
NONIDENT_EDGE = 19.7
NONIDENT_EDGE_SOURCE = (
    "CDDF_analysis/diagnostics/subdla/run_subdla_headline_full.py:NONIDENT_EDGE "
    "on branch lls-subdla-cddf (the guard layer; NOT merged into this branch — "
    "that merge is PI-deferred). Same constant, same meaning, pinned by "
    "tests/test_adopted_reporting.py::test_nonident_edge_matches_the_subdla_"
    "runner_on_the_guard_branch.")
NONIDENT_EDGE_REASON = (
    "logN centers below 19.7 are NON-IDENTIFIABLE on a 19.5-floored catalog: "
    "the two lowest 0.1-dex bins straddle the fit floor and cannot be separated "
    "from edge migration (Track A closed). This is the sub-DLA runner's meaning "
    "of NONIDENT_EDGE and it is unchanged here.")

RESPONSE_ANCHOR_CEILING = 21.6

# MEASURED, not quoted: the per-cell min/max of the frozen forward response's
# empirical true-N anchors (`emp_N_anchors`), reduced through the committed
# routine `pack.resp_fit_range_from_forward_npz`. All 24 adopted-config packs
# carry BIT-IDENTICAL values (they share one frozen 2LPT-0 response NPZ).
# CORRECTION 2026-07-29 (referee): the previous prose said the anchors span
# "~19.336-21.503 down to 21.05-21.216". 21.503 appears in NO pack -- it is the
# BOTTOM anchor's 19.502988 mistyped -- and the top anchor's floor is 21.0406,
# not 21.05. Pinned to the NPZ by
# tests/test_adopted_reporting.py::test_response_anchor_measured_reproduces_
# from_the_frozen_npz.
RESPONSE_ANCHOR_MEASURED = dict(
    source=("/scratch/.../track_c/stage0/forward_response_2lpt0.npz "
            "-> pack.resp_fit_range_from_forward_npz -> emp_N_anchors"),
    n_response_cells=9,                 # (SR, ZR) = (3, 3)
    n_anchors_per_cell=7,
    bottom_anchor_min=19.336020,        # min over cells of the LOW anchor
    bottom_anchor_max=19.502988,        # max over cells of the LOW anchor
    top_anchor_min=21.040565,           # min over cells of the HIGH anchor
    top_anchor_max=21.216358,           # max over cells of the HIGH anchor
    measured_2026_07_29=True,
)

RESPONSE_ANCHOR_CEILING_REASON = (
    "NEW (PI decision 1, 2026-07-29). The measured residual excess of finding "
    "D2 (1.23x on 2LPT-0 to 1.80x on london-0) sits ABOVE logN ~ 21.6 and its "
    "per-bin digits are INVARIANT across every pad floor and both completeness "
    "conventions of the 60-config D1 ladder. The PI capped the primary "
    "reporting window at 21.6 rather than refit the poorly anchored high-N "
    "response. This is a REPORTING cap, not a fix: the excess is still there, "
    "it is just outside what is reported. "
    "CORRECTION (referee, 2026-07-29): 21.6 is NOT where the response stops "
    "being measured, and an earlier version of this string implied it was. The "
    "frozen response's moment polynomials were fitted at true-N anchors whose "
    "per-cell LOW end spans 19.336-19.503 and whose per-cell HIGH end spans "
    "21.041-21.216 (pack.resp_N_fit_range; measured, see "
    "RESPONSE_ANCHOR_MEASURED). Above the top anchor the degree-2 surfaces are "
    "EXTRAPOLATED, so 0.38-0.56 dex of EXTRAPOLATED response lies INSIDE the "
    "authorized window [19.7, 21.6] -- see "
    "extrapolated_response_inside_window(). The earlier string also quoted a "
    "top anchor above 21.5, which appears in no pack: it was the LOW range's "
    "19.503 mistyped with a 21 in front.")


def extrapolated_response_inside_window(top_anchor_min=None,
                                        top_anchor_max=None,
                                        ceiling=None):
    """How much EXTRAPOLATED response the authorized Omega window still contains.

    The window ceiling was set at 21.6 for a residual-excess reason (finding
    D2), NOT at the top of the calibrated covariate range.  The two are
    different numbers and conflating them is the defect this function exists to
    make un-conflatable: it returns the dex of ``[top_anchor, ceiling]`` for the
    BEST-anchored cell (``top_anchor_max``) and for the WORST (``top_anchor_min``).

    Defaults come from ``RESPONSE_ANCHOR_MEASURED`` (the frozen 2LPT-0 response).
    """
    lo = RESPONSE_ANCHOR_MEASURED["top_anchor_min"] if top_anchor_min is None \
        else float(top_anchor_min)
    hi = RESPONSE_ANCHOR_MEASURED["top_anchor_max"] if top_anchor_max is None \
        else float(top_anchor_max)
    c = RESPONSE_ANCHOR_CEILING if ceiling is None else float(ceiling)
    best = max(c - hi, 0.0)      # the most favourable cell
    worst = max(c - lo, 0.0)     # the least favourable cell
    return dict(
        ceiling=c,
        top_anchor_min=lo, top_anchor_max=hi,
        dex_extrapolated_best_cell=float(best),
        dex_extrapolated_worst_cell=float(worst),
        inside_the_authorized_omega_window=bool(worst > 0.0),
        statement=(
            f"the authorized Omega_HI window [{NONIDENT_EDGE}, {c}] CONTAINS "
            f"{best:.3f}-{worst:.3f} dex over which the forward response is "
            f"EXTRAPOLATED, not measured: the frozen response's top true-N "
            f"anchor sits at {lo:.4f}-{hi:.4f} depending on the response cell, "
            f"BELOW the ceiling {c}. The ceiling was chosen for a residual-"
            f"excess reason (finding D2), not because the response is measured "
            f"up to it. This is a STATED LIMIT of every Omega_HI emitted in "
            f"this window."
            if worst > 0.0 else
            f"the response is anchored to {lo:.4f}-{hi:.4f}, at or above the "
            f"ceiling {c}: no extrapolated response lies inside the window."),
        why_it_matters=(
            "Omega_HI is an N-WEIGHTED mass, so the top of the window "
            "dominates it. On the adopted 0.2-dex packs the N-weighted Omega "
            "share of [21.2, 21.6) alone is 27.5-29.6% of the in-window total "
            "(2LPT-0 / london-0 / saclay-0), and that whole sub-interval is "
            "above the best-anchored cell's top anchor. A reader must not take "
            "'window = [19.7, 21.6]' to mean 'measured response throughout'."),
    )

# closed on both ends: 19.7 <= log N_HI <= 21.6 (the PI's own wording)
REPORTING_WINDOW = (NONIDENT_EDGE, RESPONSE_ANCHOR_CEILING)
REPORTING_WINDOW_LABEL = "logNHI_19.7_to_21.6"

OMEGA_RULE = (
    "Omega_HI is emitted ONLY when its N window is CONTAINED in the primary "
    "reporting window [19.7, 21.6] and is LABELLED with that interval. An "
    "unqualified / total / open-topped Omega_HI is REFUSED: the PI ruled that "
    "restricting the reporting window does NOT automatically authorize a total "
    "Omega_HI headline, and that a defensible tail treatment must come back to "
    "the PI. No tail extrapolation is invented here.")


def omega_decision(lo, hi):
    """Is an Omega_HI over the true-N window [lo, hi) emittable?  (decision 1)

    Returns a dict that is meant to be written INTO the artifact verbatim:
    ``{"emit": bool, "label": str|None, "reason": str, "window_logN": [lo, hi]}``.

    Emittable iff ``lo >= 19.7`` and ``hi <= 21.6`` (both to 1e-9).  An
    open-topped window (``hi = inf``) is therefore always refused, which is
    exactly the "unqualified total Omega" the PI declined to authorize.
    """
    lo = float(lo)
    hi = float(hi)
    inside = (lo >= NONIDENT_EDGE - 1e-9) and (hi <= RESPONSE_ANCHOR_CEILING + 1e-9)
    if inside:
        return dict(
            emit=True,
            label=f"OMEGA_HI_LIMITED_{lo:g}_{hi:g}",
            window_logN=[lo, hi],
            reason=("window is contained in the primary reporting window "
                    f"[{NONIDENT_EDGE}, {RESPONSE_ANCHOR_CEILING}]; the value "
                    "is explicitly LIMITED to this N interval and is not a "
                    "total Omega_HI"),
            rule=OMEGA_RULE)
    why = []
    if lo < NONIDENT_EDGE - 1e-9:
        why.append(f"lower edge {lo:g} < {NONIDENT_EDGE} (non-identifiable; "
                   f"{NONIDENT_EDGE_REASON})")
    if hi > RESPONSE_ANCHOR_CEILING + 1e-9:
        why.append(
            f"upper edge {hi:g} > {RESPONSE_ANCHOR_CEILING}"
            + (" (OPEN-TOPPED: this is the unqualified total Omega the PI did "
               "NOT authorize)" if not np.isfinite(hi) else "")
            + f"; {RESPONSE_ANCHOR_CEILING_REASON}")
    return dict(emit=False, label=None, window_logN=[lo, hi],
                reason="REFUSED: " + " AND ".join(why), rule=OMEGA_RULE)


# ===========================================================================
# DECISION 3 — the 0.2-dex latent basis (observed + reporting grids stay 0.1)
# ===========================================================================

OBSERVED_STEP = 0.1          # the n-hat / reporting / plotting step. NEVER moves.
BASIS_WIDTH_DEFAULT = 0.1    # the SHIPPED default; unchanged until closure holds
BASIS_WIDTH_ADOPTED = 0.2    # PI decision 3 — explicit, opt-in, stamped

PLOTTING_GRID_NOTE = (
    "PLOTTING GRID ONLY — NOT INDEPENDENT INFORMATION RESOLUTION. The latent "
    "true-N basis of this measurement is {basis_width:g} dex. Any curve emitted on "
    "the {plot_step:g}-dex grid is the {basis_width:g}-dex basis value REPLICATED across "
    "the {n_sub:d} sub-bins of each basis bin under the adopted merging convention "
    "('f is constant across the merged bin'). Adjacent {plot_step:g}-dex points are "
    "therefore NOT independent and their scatter is NOT a {plot_step:g}-dex "
    "measurement uncertainty. PI decision 3 (2026-07-29) permits the plotting "
    "grid and forbids describing it as {plot_step:g}-dex resolution.")


def plotting_grid_disclosure(basis_width, plot_step=OBSERVED_STEP):
    """The MANDATORY schema block for anything emitted on the plotting grid.

    Returns a dict, not a sentence, so a downstream reader cannot miss it.  Any
    artifact block carrying a ``plot_step``-dex curve MUST carry this dict next
    to it.
    """
    bw = float(basis_width)
    ps = float(plot_step)
    n_sub = int(round(bw / ps))
    if n_sub < 1 or abs(n_sub * ps - bw) > 1e-8:
        raise ReportingGuardError(
            f"plotting_grid_disclosure: basis_width {bw} is not an integer "
            f"multiple of the plotting step {ps}")
    return dict(
        grid_role="PLOTTING_ONLY",
        is_independent_information_resolution=False,
        basis_width_dex=bw,
        plotting_step_dex=ps,
        n_plot_bins_per_basis_bin=n_sub,
        merging_convention="f is constant across the merged basis bin",
        note=PLOTTING_GRID_NOTE.format(basis_width=bw, plot_step=ps, n_sub=n_sub),
        pi_decision="decision 3 (2026-07-29)")


# --- E4's merging convention (MOVED here; e4_probe re-exports these names) ---
# Kept byte-for-byte in behaviour: `tests/test_e4_probe.py` exercises them
# through `e4_probe` and must keep passing unchanged.

def basis_groups(B: int, g: int) -> list:
    """Contiguous groups of ``g`` basis bins, remainder absorbed by the LAST."""
    if g < 1:
        raise ValueError("group size must be >= 1")
    groups, i = [], 0
    while i + 2 * g <= B:
        groups.append(list(range(i, i + g)))
        i += g
    groups.append(list(range(i, B)))
    return groups


def merge_basis_columns(M: np.ndarray, groups) -> np.ndarray:
    """Merge basis columns by SUMMING within each group.

    Summing columns is exactly the statement "f is constant across the merged
    bin": the fold weight of bin b already carries its own dN_b, so
    sum_{b in G} A[:, b] * f_G == sum_{b in G} A[:, b] * f_b when f_b == f_G.
    """
    return np.stack([M[:, list(gr)].sum(axis=1) for gr in groups], axis=1)


def merged_truth(f_true: np.ndarray, dN: np.ndarray, groups) -> np.ndarray:
    """dN-weighted mean of f within each group (the coarse-basis truth)."""
    return np.array([float(np.sum(f_true[list(gr)] * dN[list(gr)])
                           / np.sum(dN[list(gr)])) for gr in groups])


def merged_edges(edges, groups) -> np.ndarray:
    """Coarse edge array for a contiguous, exhaustive, ordered ``groups``.

    ``groups`` must partition ``range(len(edges) - 1)`` into contiguous blocks
    in increasing order (exactly what ``basis_groups`` returns); anything else
    is refused rather than silently producing a wrong grid.
    """
    e = np.asarray(edges, float)
    flat = [b for gr in groups for b in gr]
    if flat != list(range(len(e) - 1)):
        raise ReportingGuardError(
            "merged_edges: groups must be a contiguous, ordered, exhaustive "
            f"partition of range({len(e) - 1}); got {groups}")
    out = [e[gr[0]] for gr in groups] + [e[-1]]
    return np.round(np.asarray(out, float), 10)


# ===========================================================================
# DECISION 4 — pad floor 19.0 / molly172, and the pad as a LATENT NUISANCE
# ===========================================================================

PAD_FLOOR_ADOPTED = 19.0
COMPLETENESS_ADOPTED = "molly172"

ADOPTED_CONFIG = dict(
    basis_width_dex=BASIS_WIDTH_ADOPTED,
    basis_pad_floor=PAD_FLOOR_ADOPTED,
    completeness_below_floor=COMPLETENESS_ADOPTED,
    reporting_window_logN=[NONIDENT_EDGE, RESPONSE_ANCHOR_CEILING],
    observed_grid_step_dex=OBSERVED_STEP,
    reporting_grid_step_dex=OBSERVED_STEP,
    resp_clamp_adopted="both",
    resp_clamp_status=("NOT a PI decision. 'both' is build_consts' default and "
                       "is treated here as the adopted corner; the OTHER corner "
                       "('hi') is carried as half of the convention systematic, "
                       "never as an alternative headline."),
    pi_decisions=["1 (reporting window)", "3 (0.2-dex basis)",
                  "4 (pad floor 19.0 + molly172)"],
    default_is_still_0p1dex=True,
    default_note=("the SHIPPED default basis width remains 0.1 dex until "
                  "closure is demonstrated; 0.2 dex is explicit and opt-in "
                  "(--basis-width 0.2) and is stamped in every pack sidecar "
                  "and artifact."),
)


def window_overlap_weights(edges, lo=NONIDENT_EDGE, hi=RESPONSE_ANCHOR_CEILING):
    """dex of each basis bin that lies INSIDE the window [lo, hi].

    This is the ONLY sanctioned weight for an INTEGRATED reported quantity
    (dN/dX, Omega) on a basis coarser than the window edges.  It is exact under
    the adopted merging convention ("f is constant across the merged bin"): a
    bin straddling a window edge contributes exactly its overlapping dex, with
    no extrapolation and no new assumption.

    On a basis whose edges align with the window (every 0.1-dex pack) the result
    is IDENTICAL to "select bins by centre, weight by dN_b" — pinned by test.
    """
    e = np.asarray(edges, float)
    lo = float(lo)
    hi = float(hi)
    w = np.minimum(e[1:], hi) - np.maximum(e[:-1], lo)
    return np.clip(w, 0.0, None)


def truth_overlap_fractions(edges, lo=NONIDENT_EDGE, hi=RESPONSE_ANCHOR_CEILING):
    """FRACTION of each basis bin inside [lo, hi] — the COUNT-side counterpart
    of ``window_overlap_weights``.

    ``window_overlap_weights`` is the weight for a quantity carried as a DENSITY
    (f, per dex): weight = overlapping dex.  Truth-side closure quantities are
    carried as COUNTS already integrated over the whole bin, so their weight is
    the same overlap divided by the bin width.  Using the two together makes the
    posterior and the truth side integrate the SAME support:

        f_b * w_b            (density side)
        counts_b * frac_b    (count side),  counts_b = f_b * dN_b * dX

    are identical whenever ``counts_b == f_b * dN_b * dX``.

    🔴 THIS FUNCTION EXISTS BECAUSE OF THE PROJECT'S SIGNATURE BUG CLASS
    ([[one-sided support]], now 5 occurrences).  Selecting truth bins BY CENTRE
    while the posterior integrates BY OVERLAP produced a ~20% spurious deficit
    on ``dndx_20p0`` at the adopted 0.2-dex basis that was pure bookkeeping.  Any
    new truth-side reduction MUST route through here.
    """
    e = np.asarray(edges, float)
    dN = np.diff(e)
    if np.any(dN <= 0):
        raise ReportingGuardError(
            f"truth_overlap_fractions: non-increasing edges {e}")
    return window_overlap_weights(e, lo, hi) / dN


def bins_fully_inside(edges, lo=NONIDENT_EDGE, hi=RESPONSE_ANCHOR_CEILING):
    """Boolean mask of bins ENTIRELY inside [lo, hi] (differential reporting).

    A DIFFERENTIAL per-bin value may only be reported for a bin that is fully
    inside the window; a straddling bin is not a measurement of either side.
    (Integrated quantities use ``window_overlap_weights`` instead.)
    """
    e = np.asarray(edges, float)
    return (e[:-1] >= float(lo) - 1e-9) & (e[1:] <= float(hi) + 1e-9)


def assert_no_subwindow_bins(edges, weights_or_mask, *, where,
                             lo=NONIDENT_EDGE):
    """FAIL CLOSED if a reported quantity draws on any basis bin below ``lo``.

    ``weights_or_mask`` is whatever the reported quantity actually used to touch
    the basis: a per-bin weight vector or a boolean selection mask, length
    ``len(edges) - 1``.  Any nonzero entry on a bin whose LOWER edge is below
    ``lo`` raises ``ReportingGuardError`` naming the offending bins.

    This is the decision-4 guard: the padded (sub-19.5) bins and the
    non-identifiable [19.5, 19.7) bins exist ONLY so the fold can carry
    up-scattered sub-floor systems into the lowest observed bins.  They are
    latent nuisance support and must never enter a reported number.
    """
    e = np.asarray(edges, float)
    w = np.asarray(weights_or_mask)
    if w.shape != (len(e) - 1,):
        raise ReportingGuardError(
            f"assert_no_subwindow_bins({where}): expected a per-bin vector of "
            f"length {len(e) - 1}, got shape {w.shape}")
    below = e[:-1] < float(lo) - 1e-9
    bad = np.flatnonzero(below & (np.abs(w.astype(float)) > 0))
    if bad.size:
        raise ReportingGuardError(
            f"REPORTING GUARD ({where}): the reported quantity draws on "
            f"{bad.size} basis bin(s) below the reporting floor {lo}: "
            + ", ".join(f"[{e[b]:.3f},{e[b + 1]:.3f}) w={float(w[b]):.6g}"
                        for b in bad)
            + ". Those bins are LATENT NUISANCE support (the schema-v1.1 "
              "downward basis pad + the [19.5,19.7) non-identifiable edge) and "
              "may never appear in a reported/paper-facing quantity "
              "(PI decisions 1 and 4).")
    return True


def reported_tier_decision(lo, hi):
    """Is a tier's INTEGRATED dN/dX a PAPER-FACING (Paper-1 reportable) number?

    Decision 4: Paper 1 reports only log N_HI >= 19.7.  The schema-v1.1 downward
    pad AND the non-identifiable [19.5, 19.7) edge are LATENT NUISANCE support.
    A tier whose own window floor is below 19.7 therefore CANNOT produce a
    paper-facing dN/dX no matter how clean its weights are — its estimand
    includes nuisance support by definition.

    This is deliberately SEPARATE from ``omega_decision``:
      * ``omega_decision`` is two-sided (needs hi <= 21.6) because Omega_HI is an
        N-weighted mass whose top the extrapolated response dominates.
      * ``reported_tier_decision`` is ONE-sided (needs lo >= 19.7) because dN/dX
        is a line density: an open top does not import nuisance support, a floor
        below 19.7 does.
    Conflating them is what wired the decision-4 guard where it could not fire
    (referee defect 5, 2026-07-29): the guard ran only under
    ``omega_decision(...)['emit']``, i.e. only for the one tier whose weights are
    zero below 19.7 by construction.
    """
    lo = float(lo)
    hi = float(hi)
    if lo >= NONIDENT_EDGE - 1e-9:
        return dict(
            paper_facing=True, window_logN=[lo, hi],
            reason=("PAPER_FACING: the tier's own floor is at or above the "
                    f"reporting floor {NONIDENT_EDGE}, so its dN/dX draws on no "
                    "latent-nuisance basis support. Its integration weights are "
                    "additionally checked by assert_no_subwindow_bins."))
    return dict(
        paper_facing=False, window_logN=[lo, hi],
        reason=(f"NOT_PAPER_FACING: the tier floor {lo:g} is below the reporting "
                f"floor {NONIDENT_EDGE}, so its integrated dN/dX necessarily "
                "includes the non-identifiable [19.5, 19.7) edge and/or the "
                "schema-v1.1 downward pad. Those are LATENT NUISANCE support "
                "(PI decisions 1 and 4) and Paper 1 reports only "
                f">= {NONIDENT_EDGE}. This tier is retained as a rung-ladder / "
                "tier-coupling DIAGNOSTIC on mocks and may not be quoted as a "
                "measurement."))


# The exact scope of ``assert_no_subwindow_bins``.  Written out because the
# previous prose ("fails closed if a reported/paper-facing block carries a basis
# bin below 19.7") over-claimed: the guard cannot make a tier whose own window
# starts at 19.5 legitimate, and it was only ever CALLED for one tier.
SUBWINDOW_GUARD_SCOPE = dict(
    guard="reporting.assert_no_subwindow_bins",
    called_from=("model_a.reduce_f_posterior, for EVERY tier whose "
                 "reported_tier_decision is paper_facing (2026-07-29; it "
                 "previously ran only where omega_decision emitted, i.e. for "
                 "report_197_216 alone)"),
    guarded_tiers=["dla_20p0", "dla_20p3", "report_197_216"],
    refused_tiers=["subdla_195_203", "all_195_up"],
    what_IS_guarded=("for a paper-facing tier: any nonzero integration weight on "
                     "a basis bin whose LOWER edge is below 19.7 raises "
                     "ReportingGuardError. This is a tripwire against a future "
                     "weighting change, and it is now armed on every "
                     "paper-facing tier rather than on one."),
    what_is_NOT_guarded=(
        "the guard does NOT and CANNOT rescue subdla_195_203 or all_195_up: "
        "their windows START below 19.7, so their integrated dN/dX includes "
        "latent-nuisance support by definition. They are not guarded, they are "
        "REFUSED as paper-facing -- posterior_summary marks them "
        "paper_facing=False and attaches dndx_paper_facing_REFUSED naming the "
        "sub-19.7 basis bins they draw on. Nor does it cover the open-topped "
        "omega_20p0 / omega_20p3 DRAW arrays, which omega_decision refuses "
        "separately at the emission point."),
    pi_decisions=["1 (reporting window)", "4 (pad = latent nuisance)"],
)


# --- the convention systematic (decision 4: "as a SYSTEMATIC, not narratively")

CONVENTION_SYSTEMATIC = dict(
    estimator_name="half_span_of_convention_corners",
    definition=("sigma_conv(x) = ( max_corners x - min_corners x ) / 2, where "
                "the CORNERS are the full cross resp_clamp in {both, hi} x "
                "completeness_below_floor in {const_extrap, molly172} at the "
                "adopted pad floor and basis width. The fractional systematic "
                "is frac_conv(x) = sigma_conv(x) / x_adopted with x_adopted the "
                "(clamp=both, molly172) corner."),
    why_half_span=("the two conventions are a DISCRETE 2x2 bracket, not a "
                   "sampled random variable. The half-span is the smallest "
                   "symmetric interval about the bracket's midpoint that "
                   "contains every corner; it is a maximal-extent measure, "
                   "NOT a 1-sigma."),
    combination_rule="SEPARATE_LINEAR_ENVELOPE",
    combination_rule_definition=(
        "total_lo = stat_lo - sigma_conv ; total_hi = stat_hi + sigma_conv, "
        "with stat_lo/stat_hi the statistical (posterior) band edges. The "
        "systematic is ALSO carried separately in the artifact so it can be "
        "un-combined."),
    why_not_quadrature=(
        "QUADRATURE IS REFUSED, and this is a choice with a reason. Adding in "
        "quadrature is only defensible for independent zero-mean random errors "
        "with comparable coverage. Neither holds here: (i) the convention "
        "bracket is a discrete choice with no sampling distribution, so its "
        "half-span carries no coverage statement and cannot be read as a "
        "1-sigma; (ii) the SAME completeness surface and the SAME response "
        "clamp enter the likelihood that produces the statistical band, so the "
        "two are not independent by construction; (iii) quadrature would shrink "
        "a maximal-extent bracket (e.g. 5.5% + 5.5% -> 7.8% instead of 11%) "
        "exactly in the regime where the convention error is at its extreme. "
        "The linear envelope cannot understate the bracket."),
    scope_limit=(
        "MEASURED ON THE TRUTH-FOLD, NOT ON A POSTERIOR. The corners are "
        "measured as the convention dependence of the PREDICTED counts mu at "
        "the pack's own truth. Propagating them to f(N) assumes a "
        "multiplicative error in mu maps to the inverse multiplicative error in "
        "f, which is exact only for a diagonal kernel. It is reported as a "
        "systematic ON THE CLOSURE, and as an ESTIMATE of the systematic on the "
        "population, explicitly labelled as such."),
)


def convention_systematic(corners: dict, adopted_key: str):
    """Half-span systematic from a dict of corner values (scalar or per-bin).

    ``corners`` maps a corner label -> value (float, or 1-D array of per-bin
    values, all the same length).  ``adopted_key`` must be one of its keys.

    Returns a dict carrying the estimator name, the corner values, the half
    span, the fractional half span relative to the adopted corner, and the
    combination rule — i.e. a propagated systematic, not a sentence.
    """
    if adopted_key not in corners:
        raise ReportingGuardError(
            f"convention_systematic: adopted_key {adopted_key!r} is not one of "
            f"the corners {sorted(corners)}")
    keys = sorted(corners)
    vals = np.stack([np.atleast_1d(np.asarray(corners[k], float)) for k in keys])
    if vals.shape[0] < 2:
        raise ReportingGuardError(
            "convention_systematic: need >= 2 corners to form a bracket")
    adopted = np.atleast_1d(np.asarray(corners[adopted_key], float))
    span = vals.max(axis=0) - vals.min(axis=0)
    half = 0.5 * span
    frac = np.divide(half, np.abs(adopted),
                     out=np.full_like(half, np.nan), where=np.abs(adopted) > 0)
    frac_span = np.divide(span, np.abs(adopted),
                          out=np.full_like(span, np.nan),
                          where=np.abs(adopted) > 0)
    scalar = vals.shape[1] == 1

    def _out(a):
        return float(a[0]) if scalar else [float(x) for x in a]

    return dict(
        estimator=CONVENTION_SYSTEMATIC["estimator_name"],
        definition=CONVENTION_SYSTEMATIC["definition"],
        corners={k: _out(np.atleast_1d(np.asarray(corners[k], float)))
                 for k in keys},
        adopted_corner=adopted_key,
        adopted_value=_out(adopted),
        span=_out(span),
        sigma_conv=_out(half),
        frac_conv=_out(frac),
        # BOTH normalisations, because "the conventions span X%" is ambiguous
        # and has already been quoted both ways: frac_conv is the HALF-span over
        # the adopted corner (the propagated systematic); frac_span is the FULL
        # max-minus-min over the adopted corner (the bracket WIDTH). frac_span
        # == 2 * frac_conv by construction.
        frac_span=_out(frac_span),
        frac_conv_definition="sigma_conv / |adopted| = half-span, THE systematic",
        frac_span_definition="span / |adopted| = full bracket width = 2 x frac_conv",
        combination_rule=CONVENTION_SYSTEMATIC["combination_rule"],
        combination_rule_definition=CONVENTION_SYSTEMATIC[
            "combination_rule_definition"],
        why_half_span=CONVENTION_SYSTEMATIC["why_half_span"],
        why_not_quadrature=CONVENTION_SYSTEMATIC["why_not_quadrature"],
        scope_limit=CONVENTION_SYSTEMATIC["scope_limit"],
    )


# ===========================================================================
# DECISION 8 — the |z| criterion, restated exactly
# ===========================================================================

Z_CRITERION = dict(
    restatement_of=("the malformed criterion previously written as '|z| <= 5'. "
                    "That string named neither WHICH z, nor over WHICH SET of "
                    "bins, nor WHICH quantity sits in the denominator, nor how "
                    "the per-bin values were reduced to one number. All four "
                    "are pinned below."),
    residual_name="Poisson Pearson residual, model in the denominator",
    per_bin_formula="z_c = (obs_c - mu_c) / sqrt(max(mu_c, 1e-12))",
    total_formula="z_tot = (obs_tot - mu_tot) / sqrt(max(mu_tot, 1e-12))",
    denominator=("mu, the MODEL prediction — not obs, and not (obs+mu)/2. The "
                 "residual is therefore the Pearson residual of a Poisson(mu) "
                 "model, and it is finite for obs_c = 0."),
    sign_convention=("obs - mu. z > 0 means the model UNDER-predicts the "
                     "observed counts."),
    bin_set=("REPORTED n-hat bins c with obs_c > 0 whose interval [lo_c, hi_c) "
             "lies inside the evaluation window. Bins with obs_c == 0 are "
             "EXCLUDED (they carry no Poisson information about a ratio and "
             "would otherwise dominate a chi2/dof through a near-zero mu). "
             "Cells with dX == 0 are excluded upstream by ratio_tables."),
    reduction="z_bin_max = max over that bin set of |z_c|",
    chi2_dof="chi2_dof = ( sum over that bin set of z_c^2 ) / n_bins_in_set",
    criterion=("PASS iff |z_tot| <= z_total_max AND z_bin_max <= z_bin_max_tol "
               "AND chi2_dof <= chi2_dof_max, with the three tolerances read "
               "from run_posterior.GATE (5.0 / 5.0 / 3.0 — ratified, PI "
               "decision 8)."),
    not_ratified=("ratio_span_by_z_max (0.10) and ratio_span_by_snr_max (0.15) "
                  "are explicitly NOT ratified (PI decision 8) and are not "
                  "ratified here. They are to be defined and calibrated "
                  "prospectively; this module does not use them."),
    absolute_value=("the criterion is on |z|, two-sided. A large NEGATIVE z "
                    "(over-prediction) fails identically to a large positive "
                    "one."),
)


def window_closure_metrics(by_nhat, lo=None, hi=None, *, label=None):
    """Closure metrics over a window, using EXACTLY the ``Z_CRITERION`` above.

    ``by_nhat`` is ``forward_selftest.ratio_tables(...)["by_nhat"]``: a list of
    dicts with ``lo``, ``hi``, ``mu``, ``obs``.  ``lo``/``hi`` default to the
    FULL grid (no restriction) so the same routine produces the global and the
    windowed numbers with one code path — the restriction is a filter, never a
    different formula.

    The z-scores are RECOMPUTED here from (mu, obs) rather than read from the
    row, so the returned numbers cannot silently inherit a different
    normalisation.
    """
    rows = list(by_nhat)
    if lo is not None or hi is not None:
        _lo = -np.inf if lo is None else float(lo)
        _hi = np.inf if hi is None else float(hi)
        rows = [r for r in rows
                if float(r["lo"]) >= _lo - 1e-9 and float(r["hi"]) <= _hi + 1e-9]
    mu = np.array([float(r["mu"]) for r in rows], float)
    obs = np.array([float(r["obs"]) for r in rows], float)
    mu_tot = float(mu.sum())
    obs_tot = float(obs.sum())
    keep = obs > 0
    z = (obs[keep] - mu[keep]) / np.sqrt(np.maximum(mu[keep], 1e-12))
    n = int(keep.sum())
    return dict(
        label=label or ("full_grid" if lo is None and hi is None
                        else f"window_{lo}_{hi}"),
        window_logN=[None if lo is None else float(lo),
                     None if hi is None else float(hi)],
        n_bins_in_window=int(len(rows)),
        n_bins_in_z_set=n,
        total_mu=mu_tot,
        total_obs=obs_tot,
        total_ratio=(mu_tot / obs_tot) if obs_tot > 0 else float("nan"),
        z_total=float((obs_tot - mu_tot) / np.sqrt(max(mu_tot, 1e-12))),
        z_bin_max=float(np.abs(z).max()) if n else float("nan"),
        chi2_dof=float((z ** 2).sum() / max(n, 1)),
        per_bin=[dict(lo=float(r["lo"]), hi=float(r["hi"]), mu=float(r["mu"]),
                      obs=float(r["obs"]),
                      ratio=(float(r["mu"]) / float(r["obs"])
                             if float(r["obs"]) > 0 else float("nan")),
                      z=float((float(r["obs"]) - float(r["mu"]))
                              / np.sqrt(max(float(r["mu"]), 1e-12))))
                 for r in rows],
        z_criterion=Z_CRITERION,
    )
