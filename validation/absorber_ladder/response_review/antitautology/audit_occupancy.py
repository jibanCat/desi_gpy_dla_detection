#!/usr/bin/env python
"""audit_occupancy.py — anti-tautology test D.

Two halves:

1. ``STEPS`` — the CODE-LEVEL GROUND TRUTH table.  Every smoothing /
   shrinkage / basis / normalisation step of R1c, R1d-raw, R1d-smoothed and
   M_true, at file:line, with four yes/no columns:
     (a) does the amount of shrinkage depend on the ROW's own event count
         N_bsk?
     (b) on NEIGHBOURING rows' occupancies?
     (c) on the TOTAL calibration CDDF (the mock's p(N_true))?
     (d) on the S/N and z OCCUPANCY?
   and a verdict on whether the step CAN imprint population structure on the
   conditional row.  (The archaeology agent's mathematical memo cites this
   table; it is deliberately literal.)

2. ``run_audit`` — the EXECUTABLE probes that decide those columns from
   behaviour rather than from reading.  Each probe changes an OCCUPANCY while
   leaving every conditional row exactly unchanged (it duplicates events, i.e.
   doubles a weight), and measures how far the operator moves:
     (a) ``row_count``       — double the events of ONE target row;
     (b) ``neighbour_rows``  — double the events of the target row's two
                               latent-bin NEIGHBOURS (same S/N, same z cell);
     (d) ``snr_z_cell``      — double the events of a DIFFERENT (S/N, z) cell.
   (c) "total calibration CDDF" is exactly test A and is not duplicated here.

   A purely conditional estimator moves by 0 in (a), (b) and (d), to machine
   precision.

VALIDATION-ONLY.  ENV: gpdla-hbi.
"""
from __future__ import annotations

import numpy as np

import opbuild as OB
import toys as TOY

R1D = "validation/absorber_ladder/response/r1d_empirical.py"
RFI = "validation/absorber_ladder/response/respfit.py"
BMO = "validation/absorber_diag/build_matched_ops.py"
OPB = ("validation/absorber_ladder/response_review/antitautology/"
       "opbuild.py")

# --------------------------------------------------------------------------
# 1. the code-level table.  Columns: (a) own row count, (b) neighbour rows,
#    (c) total calibration CDDF, (d) S/N-z occupancy.
# --------------------------------------------------------------------------
STEPS = [
    # ---- R1d-raw ---------------------------------------------------------
    dict(operator="R1d-raw", step="weighted count accumulation into (row, c)",
         where=f"{OPB}:_weighted_counts (np.add.at)",
         a=False, b=False, c=False, d=False,
         can_imprint="no — pure per-row counting"),
    dict(operator="R1d-raw", step="Jeffreys +1/2 added to all 29 observed "
                                  "cells of each row",
         where=f"{OPB}:build_r1d_raw",
         a=True, b=False, c=False, d=False,
         can_imprint=("no — the shrinkage FRACTION does fall with the row's "
                      "own count (14.5 pseudo-counts vs n), but the target is "
                      "the UNIFORM row, a fixed object that carries no "
                      "population information; it can blunt a row, never "
                      "imprint p(N_true) on it")),
    dict(operator="R1d-raw", step="row renormalisation over c",
         where=f"{OPB}:build_r1d_raw",
         a=False, b=False, c=False, d=False,
         can_imprint="no — exact conditioning"),
    # ---- R1d as implemented ---------------------------------------------
    dict(operator="R1d (smoothed)",
         step="offset histogram of matched events per (S/N cell, z cell, "
              "offset j, latent bin b)",
         where=f"{R1D}:51-63 (np.add.at ... 1.0 at :62)",
         a=False, b=False, c=False, d=False,
         can_imprint="no — pure counting"),
    dict(operator="R1d (smoothed)",
         step="WITHIN-BIN marginalisation weight: events are histogrammed by "
              "latent bin, so the row is the mock's own f(N) average over the "
              "0.2-dex bin, not a flat average",
         where=f"{R1D}:29-32 (documented), :51-63 (the mechanism)",
         a=False, b=False, c=True, d=False,
         can_imprint=("YES in principle — a steeper population puts more of "
                      "the bin's weight at its low-N edge and shifts the row. "
                      "Bounded by the 0.2-dex bin width; measured by test A "
                      "(this is the ONLY channel a purely per-row estimator "
                      "has)")),
    dict(operator="R1d (smoothed)",
         step="ridge-penalised degree-2 polynomial in N fitted per "
              "(S/N cell, z cell, offset) to the per-bin mass fractions, "
              "WEIGHTED BY THE BIN'S EVENT COUNT (w = tot[i,j]; sw = sqrt(w))",
         where=f"{R1D}:66-95 (w at :79, sw at :80, Xw at :81)",
         a=True, b=True, c=True, d=False,
         can_imprint=("YES — this is the load-bearing step.  How far latent "
                      "bin b's row is pulled toward its neighbours is set by "
                      "the RELATIVE event counts of the bins, i.e. by the "
                      "calibration mock's f(N) folded through completeness "
                      "and detection.  Probes (a) 2.0e-2 and (b) 2.6e-2 "
                      "confirm it.  The polynomial is fitted INDEPENDENTLY in "
                      "each of the 9 (S/N, z) cells, so there is no "
                      "cross-cell leakage: probe (d) is exactly 0.  (The "
                      "within-cell S/N composition still enters through (c), "
                      "because one response cell pools several S/N strata.)")),
    dict(operator="R1d (smoothed)",
         step="clip at 0 and renormalise over offsets",
         where=f"{R1D}:96-98",
         a=False, b=False, c=False, d=False,
         can_imprint="no (a nonlinearity, not a shrinkage)"),
    dict(operator="R1d (smoothed)",
         step="off-grid mass dropped, row renormalised "
              "(count-conservation rule)",
         where=f"{R1D}:102-115",
         a=False, b=False, c=False, d=False,
         can_imprint="no"),
    # ---- R1c -------------------------------------------------------------
    dict(operator="R1c",
         step="ADAPTIVE sub-bin edges: 0.1-dex bins merged upward until a bin "
              "holds >= min_n events (cap 0.3 dex, floor 25)",
         where=f"{RFI}:185-225 (np.histogram at :207, merge test at :214); "
               f"called per (S/N, z) cell at {RFI}:413-431",
         a=True, b=True, c=True, d=False,
         can_imprint=("YES — the estimator's own BASIS (where the sub-bins "
                      "are) is a function of the calibration occupancy, and "
                      "is recomputed per (S/N, z) cell")),
    dict(operator="R1c",
         step="sub-bin covariate = the count-weighted MEAN N of the sub-bin's "
              "members (c = dn.mean())",
         where=f"{RFI}:227-256 (:240)",
         a=False, b=False, c=True, d=False,
         can_imprint=("YES but small — the covariate at which the moment is "
                      "attached moves with the population inside a <= 0.3-dex "
                      "sub-bin")),
    dict(operator="R1c",
         step="per-cell degree-2 + one shared cubic moment polynomials, WLS "
              "with weight sqrt(n_subbin)",
         where=f"{RFI}:258-330 (w at :270, :297, :313); the SHARED cubic is "
               f"pooled over all 9 cells at {RFI}:305-320",
         a=True, b=True, c=True, d=True,
         can_imprint=("YES — the polynomial pools information ACROSS N, and "
                      "how much each sub-bin contributes is its event count; "
                      "this is R1c's analogue of R1d's ridge weighting, with "
                      "far fewer free coefficients (108).  Unlike R1d it also "
                      "pools ACROSS CELLS: the order-3 term is ONE shared "
                      "coefficient fitted on the count-weighted union of all "
                      "9 cells, so another cell's occupancy moves this cell's "
                      "row — probe (d) = 6.2e-3, the only operator with a "
                      "non-zero cross-cell probe")),
    dict(operator="R1c",
         step="covariate clamp fit_rng",
         where=f"{RFI}:434-457 (fit_rng='full'); the data-determined "
               f"alternative is rng_data = min/max fitted sub-bin centre, "
               f"{RFI}:272, :298",
         a=False, b=False, c=False, d=False,
         can_imprint=("no FOR R1c — fit_rng='full' pins the clamp to the "
                      "fixed latent support [19.0, 22.4].  It IS occupancy-"
                      "dependent for R0/R1a-as-frozen, where the clamp is the "
                      "min/max fitted sub-bin centre")),
    dict(operator="R1c",
         step="latent-bin marginalisation quadrature (the width repair)",
         where=f"{RFI}:492-531 (weight=='flat' at :518; docstring :502-505)",
         a=False, b=False, c=False, d=False,
         can_imprint=("no — the within-bin weight is deliberately FLAT, not "
                      "the mock's f(N).  This is the one place the first "
                      "ladder removed an occupancy dependence on purpose")),
    dict(operator="R1c",
         step="re-expression of the marginalised targets on the latent-bin "
              "centres (UNWEIGHTED polyfit)",
         where=f"{RFI}:533-560 (:556, no w=)",
         a=False, b=False, c=False, d=False,
         can_imprint="no"),
    # ---- M_true ----------------------------------------------------------
    dict(operator="M_true (forensic)",
         step="matched counts normalised over c",
         where=f"{BMO}:467-470",
         a=False, b=False, c=False, d=False,
         can_imprint="no — the saturated conditional row, no pooling at all"),
    dict(operator="M_true (forensic)",
         step="EMPTY rows backfilled by the model kernel Mg_norm_sKcb",
         where=f"{BMO}:469, :478",
         a=True, b=False, c=False, d=False,
         can_imprint=("no population imprint, but it imports the PARAMETRIC "
                      "kernel into rows with zero matched events; those rows "
                      "are excluded by the >= 200-event gate of test A")),
    dict(operator="M_true (forensic)",
         step="completeness C_true_bKs = IN-GRID detections / "
              "truth_counts_bks — it already carries the in-grid fraction phi",
         where=f"{BMO}:460",
         a=False, b=False, c=False, d=False,
         can_imprint=("no — a per-cell ratio, no smoothing.  But it is a "
                      "CONVENTION trap: pairing it with a parametric row that "
                      "sums to adopted_phi_ref applies phi twice; see the "
                      "phi_convention_audit")),
    dict(operator="R0/R1a-R1d (deployed)",
         step="count-conservation rescaling of the row to the FROZEN "
              "adopted_phi_ref (masses / phi * adopted_phi_ref)",
         where="validation/absorber_ladder/response/build_variants.py:620-624; "
               "cc_posterior_validation.build_cc_tensors",
         a=False, b=False, c=False, d=False,
         can_imprint=("no population imprint — adopted_phi_ref is a frozen "
                      "parametric object — but it is NOT the measured in-grid "
                      "fraction of the calibration mock; the ratio is "
                      "tabulated in the phi_convention_audit")),
]


# --------------------------------------------------------------------------
# 2. the executable probes
# --------------------------------------------------------------------------
def _target_row(op, geom, min_n=200):
    """The best-populated row that has both latent-bin neighbours populated."""
    meta = OB.row_meta(op.grid, geom)
    n = op.row_n
    order = np.argsort(-n)
    for r in order:
        i0, i1, b = meta[r]
        if n[r] < min_n or b == 0 or b == geom["B"] - 1:
            continue
        nb = [int(np.ravel_multi_index((i0, i1, bb),
                                       OB.row_shape(op.grid, geom)))
              for bb in (b - 1, b + 1)]
        if all(n[x] >= 20 for x in nb):
            return int(r), nb, (int(i0), int(i1), int(b))
    return int(order[0]), [], tuple(int(x) for x in meta[order[0]])


def _build(kind, ev, geom, w, seed=0):
    kw = dict(seed=seed, force_resample=False) if kind == "R1c" else {}
    if kind in TOY.TOY_BUILDERS:
        return TOY.TOY_BUILDERS[kind](ev, w, geom)
    return OB.build_operator(ev, w, kind, geom, **kw)


def run_audit(kinds, ev, geom, verbose=True, fam="2lpt0"):
    cp = OB.completeness_and_phi(geom, fam)
    out = dict(code_level_table=STEPS,
               phi_convention_audit=dict(
                   identity_C_true_eq_C_det_times_phi_max_abs=cp[
                       "identity_max_abs"],
                   n_cells_supported=cp["n_cells_supported"],
                   n_cells_C_det_gt1=cp["n_cells_C_det_gt1"],
                   phi_ref_vs_measured=OB.phi_ref_vs_measured(geom,
                                                              cp["phi"]),
                   why=("C_true_bKs (build_matched_ops.py:460) counts only "
                        "IN-GRID detections, so it already carries phi; the "
                        "parametric Mg rows sum to the frozen "
                        "adopted_phi_ref.  C_true x Mg_parametric therefore "
                        "applies phi TWICE.  Test C uses convention (a): "
                        "C_det (no phi) with rows scaled by the MEASURED "
                        "phi(b,K,s), identically for every operator.")),
               probe_definition=(
                   "each probe DOUBLES the weight of a set of events, which "
                   "leaves every conditional row P(c|.) mathematically "
                   "unchanged and changes only an occupancy.  Reported: the "
                   "max |delta P| on the target row (and, for the cell probe, "
                   "on rows of an untouched (S/N, z) cell).  A purely "
                   "conditional estimator gives exactly 0."),
               probes={})
    for kind in kinds:
        base = _build(kind, ev, geom, np.ones(ev["n"]))
        r, nbr, (i0, i1, b) = _target_row(base, geom)
        sh = OB.row_shape(base.grid, geom)
        ridx = OB.row_index(base.grid, ev, geom)
        meta = OB.row_meta(base.grid, geom)

        def probe(mask, watch):
            w = np.where(mask, 2.0, 1.0)
            op = _build(kind, ev, geom, w)
            d = np.abs(op.P - base.P)
            return float(np.max(d[watch])) if np.any(watch) else 0.0

        watch_target = np.zeros(base.n_rows, bool); watch_target[r] = True
        m_row = ridx == r
        pa = probe(m_row, watch_target)

        watch_nbr = np.zeros(base.n_rows, bool)
        for x in nbr:
            watch_nbr[x] = True
        m_nbr = np.isin(ridx, nbr) if nbr else np.zeros(ev["n"], bool)
        pb = probe(m_nbr, watch_target) if nbr else 0.0

        other = [(j0, j1) for j0 in range(sh[0]) for j1 in range(sh[1])
                 if (j0, j1) != (i0, i1)]
        j0, j1 = other[0] if other else (i0, i1)
        m_cell = np.zeros(ev["n"], bool)
        for rr in range(base.n_rows):
            if meta[rr][0] == j0 and meta[rr][1] == j1:
                m_cell |= (ridx == rr)
        pd_ = probe(m_cell, watch_target)

        out["probes"][kind] = dict(
            row_grid=base.grid,
            target_row=dict(i0=i0, i1=i1, b=b, n_events=float(base.row_n[r])),
            neighbour_rows=[int(x) for x in nbr],
            perturbed_cell=[int(j0), int(j1)],
            a_own_row_count_maxdP=pa,
            b_neighbour_row_occupancy_maxdP=pb,
            d_other_snr_z_cell_occupancy_maxdP=pd_,
            DEPENDS=dict(a=bool(pa > 1e-12), b=bool(pb > 1e-12),
                         d=bool(pd_ > 1e-12)))
        if verbose:
            print(f"[D] {kind:16s} a={pa:.3e} b={pb:.3e} d={pd_:.3e}",
                  flush=True)
    return out


# --------------------------------------------------------------------------
def mutation_verdicts(rep):
    """The mutation checks the discipline requires: test A MUST flag the
    deliberately occupancy-imprinted toy and MUST NOT flag the conditional
    toys."""
    A = rep.get("test_A_reweighting_invariance", {})
    B = rep.get("test_B_train_one_slope_eval_another", {})
    v = {}
    if "toy_imprinted" in A:
        v["A_flags_imprinted_toy"] = bool(
            A["toy_imprinted"]["VERDICT_occupancy_imprinted"])
    if "toy_conditional" in A:
        v["A_passes_exact_conditional_toy"] = bool(
            not A["toy_conditional"]["VERDICT_occupancy_imprinted"])
    if "toy_rowfit" in A:
        v["A_passes_rowfit_conditional_toy"] = bool(
            not A["toy_rowfit"]["VERDICT_occupancy_imprinted"])
    if "toy_imprinted" in B:
        v["B_flags_imprinted_toy"] = bool(
            not B["toy_imprinted"]["VERDICT_conditional"])
    if "toy_conditional" in B:
        v["B_passes_exact_conditional_toy"] = bool(
            B["toy_conditional"]["VERDICT_conditional"])
    v["ALL_MUTATION_CONTROLS_PASS"] = bool(v) and all(v.values())
    return v
