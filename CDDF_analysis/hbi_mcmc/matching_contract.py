# -*- coding: utf-8 -*-
"""matching_contract.py — THE FROZEN FORWARD-MODEL AND MATCHING CONTRACT.

WHAT THIS FILE IS
-----------------
A machine-readable statement of *what the Model A forward fold counts, once
each*.  It is not a design document and it changes no model behaviour: it names
the six populations a DLA / sub-DLA candidate or truth row can belong to, gives
each an executable predicate, states the accounting identity those populations
jointly satisfy, and provides checkers that FAIL LOUDLY on an input that does
not satisfy it.

Everything here is DECLARATIVE plus CHECKS.  Nothing in this module is imported
by ``forward.py``, ``model_a.py`` or ``extract_pack.py``; it reads packs and the
committed fold and reports.

AUTHORITY (docs/RULES.md)
-------------------------
Nothing in this file is newly ratified.  The adopted baseline it encodes
(floor 19.7, ceiling 21.6, 0.2-dex latent basis, pad 19.0 + molly172, primary
window Lya-only 1025-1216 A, target dN/dX, no Omega_HI, 2LPT-0 = calibration)
is the PI-ratified/adopted point recorded in ``reporting.py`` and
``adopted_config.py``; this module re-uses those constants, never re-derives
them.  No field of this contract carries ``authority=PI``, ``RATIFIED`` or
``paper_facing``.  Items that would change the PHYSICAL definition of a
population are collected in ``PI_CHECKPOINT_ITEMS`` and are NOT decided here.

PRIVACY: mock-derived only (2LPT-0 / London-0 / Saclay-0 / loa-0).  This module
opens no real-DESI path and every number it can emit is mock-derived.

ENV: importable in the jax-free `gpdla` data-plane env (load it file-directly;
the ``hbi_mcmc`` package ``__init__`` imports jax).  Only
``check_accounting_identity``'s default row-mass path needs jax, and it imports
it lazily; inject ``row_mass=`` to run the ledger without jax at all.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Callable, Optional, Tuple

import numpy as np

from CDDF_analysis.hbi_mcmc import reporting as RP

__all__ = [
    "CONTRACT_VERSION", "ContractViolation",
    "SupportClass", "ParameterClass", "Side",
    "Axis", "Population", "Quantity",
    "AXES", "MATCHING", "POPULATIONS", "POPULATION_BY_ID", "QUANTITIES",
    "KNOWN_CONTRADICTIONS", "PI_CHECKPOINT_ITEMS",
    "contract_dict", "basis_partition", "classify_candidate", "classify_truth",
    "validate_pack_against_contract", "fp_normalisation_audit",
    "assert_forward_fp_normalisation", "check_accounting_identity",
]

CONTRACT_VERSION = "1.0"


class ContractViolation(ValueError):
    """An input, a pack or the committed fold violates this contract."""


# ---------------------------------------------------------------------------
# 0. vocabularies
# ---------------------------------------------------------------------------
class SupportClass:
    """How well a region of the (N_true, N_hat, z, SNR) support is determined."""

    MEASURED = "MEASURED"                    # calibrated from data at this point
    WEAKLY_MEASURED = "WEAKLY_MEASURED"      # calibrated, but from few events
    INTERPOLATED = "INTERPOLATED"            # between calibration anchors
    PRIOR_ASSISTED = "PRIOR_ASSISTED"        # data-informed only through a prior
    EXTRAPOLATED = "EXTRAPOLATED"            # outside every calibration anchor
    CLAMPED = "CLAMPED"                      # frozen at the nearest anchor
    UNSUPPORTED = "UNSUPPORTED"              # no basis / no term at all
    ALL = (MEASURED, WEAKLY_MEASURED, INTERPOLATED, PRIOR_ASSISTED,
           EXTRAPOLATED, CLAMPED, UNSUPPORTED)


class ParameterClass:
    """Is a quantity frozen input, sampled nuisance, target, or derived?"""

    DATA = "DATA"                                          # observed counts
    FIXED_CALIBRATION_PRODUCT = "FIXED_CALIBRATION_PRODUCT"  # frozen, not sampled
    INFERRED_NUISANCE = "INFERRED_NUISANCE"                # sampled, marginalized
    LATENT_TARGET = "LATENT_TARGET"                        # what is reported
    DERIVED = "DERIVED"                                    # a function of the above
    ALL = (DATA, FIXED_CALIBRATION_PRODUCT, INFERRED_NUISANCE, LATENT_TARGET,
           DERIVED)


class Side:
    """Which ledger a population is counted on."""

    CANDIDATE = "CANDIDATE"    # a row of cat_cut passing the op mask
    TRUTH = "TRUTH"            # a row of truth_cut
    RATE_SCALE = "RATE_SCALE"  # neither: a multiplicative factor on another slot
    ALL = (CANDIDATE, TRUTH, RATE_SCALE)


# ---------------------------------------------------------------------------
# 1. axes: edges, units, conditioning
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Axis:
    name: str
    role: str                 # OBSERVED | LATENT | CALIBRATION | CONDITIONING
    pack_key: Optional[str]   # where the edges live in a pack (None = derived)
    units: str
    bin_convention: str       # "[lo, hi)" everywhere in the pack
    measured_on: str          # which catalogue column produces the value
    note: str = ""

    def as_dict(self):
        return dataclasses.asdict(self)


#: the reporting floor / ceiling, RE-USED from reporting.py (never re-derived).
REPORT_FLOOR = RP.NONIDENT_EDGE            # 19.7
REPORT_CEILING = RP.RESPONSE_ANCHOR_CEILING  # 21.6

#: the observed detection grid, fixed by the pack schema.
OBSERVED_FLOOR = 19.5
OBSERVED_CEILING = 22.4
OBSERVED_STEP = 0.1

#: the adopted latent basis (PI decision 3) and its downward pad (finding D1).
ADOPTED_BASIS_WIDTH = 0.2
ADOPTED_PAD_FLOOR = 19.0
ADOPTED_COMPLETENESS_CONVENTION = "molly172"
ADOPTED_ANALYSIS_WINDOW = dict(name="lya_only", lam_rf_min=1025.0,
                               lam_rf_max=1216.0)

AXES = {
    "nhat": Axis(
        name="nhat", role="OBSERVED", pack_key="nhat_edges",
        units="dex of log10(N_HI / cm^-2)", bin_convention="[lo, hi)",
        measured_on="cat_cut['NHI'] — the finder's per-detection MAP column "
                    "density (NOT a truth column)",
        note=f"{OBSERVED_FLOOR}..{OBSERVED_CEILING} step {OBSERVED_STEP}, "
             "29 bins. The reporting grid never moves."),
    "ntrue": Axis(
        name="ntrue", role="LATENT", pack_key="ntrue_edges",
        units="dex of log10(N_HI / cm^-2)", bin_convention="[lo, hi)",
        measured_on="truth_cut['NHI'] for the truth histogram; the latent f is "
                    "carried on the SAME edges",
        note=f"adopted {ADOPTED_BASIS_WIDTH} dex, padded DOWN to "
             f"{ADOPTED_PAD_FLOOR}; every edge on the observed 0.1-dex grid; "
             f"top edge shared with nhat; {REPORT_FLOOR} MUST be an exact edge "
             f"(pad/report boundary); {REPORT_CEILING} STRADDLES a 0.2-dex bin "
             "and is split by overlap fraction, never by a centre test."),
    "zf": Axis(
        name="zf", role="OBSERVED", pack_key="zf_edges", units="redshift",
        bin_convention="[lo, hi)",
        measured_on="cat_cut['Z_DLA'] for counts; truth_cut['Z_DLA'] for the "
                    "truth histogram",
        note="2.0..3.5 step 0.1, 15 bins. The DETECTION z is the observed z; "
             "the truth histogram uses the TRUTH z. They are different axes "
             "that share edges — the fold never migrates in z."),
    "zc": Axis(
        name="zc", role="CONDITIONING", pack_key="zc_edges", units="redshift",
        bin_convention="[lo, hi)",
        measured_on="derived from zf via kz_to_K",
        note="[2.0, 2.5, 3.0, 3.5]. Conditions the response cell and the FP "
             "transfer factor t_K."),
    "snr": Axis(
        name="snr", role="CONDITIONING", pack_key="snr_edges",
        units="red-side S/N per sightline", bin_convention="[lo, hi)",
        measured_on="cat_cut['S2N_RED'] (== SNR_REDSIDE); the SIGHTLINE's SNR, "
                    "not the absorber's",
        note="[0,1,...,7,inf), 8 strata. Strata below the op cut (S2N_RED > 2 "
             "STRICT) are structurally empty (dX == 0) and masked out of the "
             "likelihood."),
    "molly": Axis(
        name="molly", role="CALIBRATION", pack_key="molly_nhi_edges",
        units="dex of log10(N_HI / cm^-2)", bin_convention="[lo, hi)",
        measured_on="the completeness matrix's own TRUE-N cell grid",
        note="COARSER and NON-UNIFORM: under molly172 "
             "[17.2,17.5,18,18.5,19,19.5,20,20.3,20.5,21,21.5,22,inf). Several "
             "latent basis bins share one completeness cell "
             "(forward.build_consts: clip(digitize(Nc, molly_nhi_edges)-1, 0, "
             "M-2)). This is a SUPPORT COARSENING, not an interpolation."),
    "resp": Axis(
        name="resp", role="CALIBRATION", pack_key="resp_snr_edges/resp_z_edges",
        units="(SNR stratum, z bin) response cells",
        bin_convention="[lo, hi)",
        measured_on="the frozen 2LPT-0 ForwardResponseModel NPZ",
        note="(SR, ZR) = (3, 3). Its calibrated TRUE-N covariate range is "
             "resp_N_fit_range; outside it the moment polynomials are "
             "EXTRAPOLATED and the fold CLAMPS (resp_clamp='both')."),
}

#: the spectral window that defines which absorbers and which pathlength count.
SPECTRAL_WINDOW = dict(
    kind="ANALYSIS window (HBIConfig.lam_rf_min / lam_rf_max), applied POST-HOC",
    adopted=ADOPTED_ANALYSIS_WINDOW,
    per_sightline_geometry=(
        "lambda_rest in [lam_rf_min, lam_rf_max] with a 3000 km/s collar on "
        "BOTH edges, a 3600 A observed-lambda floor, and z_qso in "
        "(z_qso_min, z_qso_max) — cddf_catalog_hbi.build_pathlength"),
    not_the_finder_window=(
        "Parameters.min_lambda = 911.75 / max_lambda = 1250 in production. "
        "This contract cannot change that; doing so would require re-running "
        "the GP."),
    consistency_rule=(
        "The SAME per-sightline window carves the detections (cat_cut), the "
        "truth (truth_cut) and the pathlength (dX). A pack whose cached bundle "
        "was cut at a different lam_rf_min than its frozen calibration is "
        "REFUSED by extract_pack's window GUARD."),
)

#: path-length normalisation.
PATHLENGTH = dict(
    quantity="dX[k, s] — absorption distance X summed over the SNR>2 sightline "
             "set, per fine-z bin and per SNR stratum",
    routine="cddf_catalog_hbi.build_pathlength (return_per_sl) -> build_M_b PX",
    bal="BAL targets are EXCLUDED from dX (cfg.no_bal=True) and from BOTH "
        "cat_cut and truth_cut (make_lambda_z_BAL_cuts with the same bal_tids).",
    rule="Every rate in this contract is per dX over the SAME sightline set "
         "that produced the candidates and the truth rows.",
)


# ---------------------------------------------------------------------------
# 2. matching rules: the exact definition of FOUND
# ---------------------------------------------------------------------------
MATCHING = dict(
    matcher=dict(
        routine="examples/molly_faithful_pc_plots.py:match_truth_to_cat_molly "
                "(:318), called from cddf_catalog_hbi.load_and_cut_catalog "
                "(:612)",
        kind="GREEDY 1-TO-1",
        candidate_iteration_order="descending NHI_pred (cat_iter_order="
                                  "'nhi_desc'); NaN NHI last",
        pairing_domain="the SAME TARGETID only — no cross-sightline matches",
        redshift_tolerance="|z_cand - z_truth| / (1 + z_truth) < dz_rel, "
                           "dz_rel = 0.01 (HBIConfig.dz_rel:110)",
        tie_break="minimum |NHI_pred - NHI_truth| among unmatched candidates",
        multiplicity=("1-to-1 by construction: a truth row is claimed at most "
                      "once (truth_matched flag), and a candidate claims at "
                      "most one truth row. A SECOND candidate on an already "
                      "claimed truth row is therefore UNMATCHED and lands in "
                      "P6_RESIDUAL_UNMATCHED, not in P4."),
    ),
    is_TP=dict(
        definition="is_TP = ~isnan(cat_cut['NHI_TRUE']) "
                   "(cddf_catalog_hbi.py:669)",
        bundle_dependence=(
            "is_TP is a property of the (catalogue, truth-floor) BUNDLE, not "
            "of a detection. The truth table is pre-floored at "
            "truth_nhi_floor = mm.nhi_edges[0] BEFORE matching "
            "(cddf_catalog_hbi.py:576), so a detection whose genuine absorber "
            "lies below the floor is labelled NOT is_TP."),
        floor_used_by_the_pack_detection_side=19.5,
        floor_used_by_the_molly172_sub_floor_cells=17.2,
        measured_2lpt0_19p5_floor_bundle=dict(
            n_cat_cut=582855, n_op=495553, n_on_pack_grid=88071,
            n_is_TP=63890, n_unmatched=24181,
            n_is_TP_with_nhi_true_below_19p5=0,
            window_nhat_19p7_to_21p6=dict(
                total=67086, true_N_in_window=53401, true_N_below_19p7=4521,
                true_N_above_21p6=70, unmatched=9094),
            measured="2026-08-05, extract_pack.load_mock_bundle. See "
                     "KNOWN_CONTRADICTIONS TRUTH_FLOOR_ASYMMETRY_IN_is_TP for "
                     "the 17.2-floor split of the same window."),
    ),
    found=dict(
        # THE definition the completeness NUMERATOR uses.
        completeness_numerator=(
            "examples/molly_faithful_pc_plots.py:completeness_snr_nhi_bins "
            "(:529): a CANDIDATE row with S2N_RED in (s_lo, s_hi) STRICT, "
            "NHI_TRUE in (n_lo, n_hi) STRICT (OPEN, not half-open), "
            "NHI > cmp_min_pred_nhi STRICT, P_DLA > 0.99 STRICT, and "
            "good_mask; counted via is_TP."),
        cmp_min_pred_nhi=(
            "defaults to the MATRIX'S OWN first edge "
            "(cddf_catalog_hbi.py:752 and :1563, used at :1581). No caller "
            "overrides it."),
        two_definitions_spliced=(
            "Under the adopted molly172 convention, extract_pack."
            "load_molly_counts_block (:555-557) concatenates the sub-19.5 "
            "cells of the floor-17.2 matrix (found <=> N_hat > 17.2) with the "
            ">= 19.5 cells of the canonical nhi195 matrix (found <=> N_hat > "
            "19.5). ONE array, TWO definitions of found, split exactly at the "
            "observed floor."),
        completeness_denominator=(
            "TRUTH rows of truth_cut with S2N_RED in (s_lo, s_hi) STRICT and "
            "NHI in (n_lo, n_hi) STRICT — the SAME open-interval convention as "
            "the numerator, and DIFFERENT from the pack's half-open [lo, hi) "
            "binning. A truth row exactly ON a molly edge falls in NO molly "
            "cell but DOES fall in a pack basis bin; "
            "track_c_tf_saclay._snap_off_molly_edges exists for that case."),
    ),
    op_mask=dict(
        definition="S2N_RED > 2 STRICT & P_DLA > 0.99 STRICT & DLAFLAG == 0, "
                   "inside a bundle that already applied the BAL veto, the "
                   "z_qso window and the lambda_rest window",
        applies_to="the CANDIDATE side only; the pack's `counts` are RAW "
                   "op-passing detections with NO FP subtraction on this route",
    ),
    detections_outside_the_reporting_interval=dict(
        rule=("A candidate reconstructed OUTSIDE the observed grid "
              f"[{OBSERVED_FLOOR}, {OBSERVED_CEILING}) x [2.0, 3.5) is DROPPED "
              "from `counts` (extract_pack.bin_counts_cks:384-388). On the "
              "model side the SAME event is the kernel row-mass deficit "
              "1 - sum_c K[c<-b]: it is neither a detection nor a miss, and it "
              "is carried by the P_OFFGRID slot of the truth ledger."),
        must_not=("It must NOT be folded into incompleteness. Doing so would "
                  "count the same absorber on both sides of the ledger."),
    ),
)


# ---------------------------------------------------------------------------
# 3. the six populations
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Population:
    pid: str
    name: str
    side: str
    predicate_text: str
    predicate: Callable
    forward_term: str
    prior: str
    support_class: str
    parameter_class: str
    note: str = ""

    def as_dict(self):
        d = dataclasses.asdict(self)
        d.pop("predicate")
        return d


def _is_num(x):
    return x is not None and np.isfinite(float(x))


def _p1(rec):
    """P1: genuine absorber BELOW the reporting floor, detected on the grid."""
    return (bool(rec.get("is_TP")) and _is_num(rec.get("nhi_true"))
            and float(rec["nhi_true"]) >= ADOPTED_PAD_FLOOR - 1e-9
            and float(rec["nhi_true"]) < REPORT_FLOOR - 1e-9)


def _p2(rec):
    """P2: genuine absorber INSIDE the reporting window."""
    return (bool(rec.get("is_TP")) and _is_num(rec.get("nhi_true"))
            and REPORT_FLOOR - 1e-9 <= float(rec["nhi_true"]) < REPORT_CEILING - 1e-9)


def _p3(rec):
    """P3: a TRUTH row with no candidate claiming it."""
    return not bool(rec.get("matched"))


def _p4(rec):
    """P4: a candidate with no genuine absorber, attributable to pure forest."""
    return (not bool(rec.get("is_TP"))) and bool(rec.get("forest_attributable"))


def _p6_candidate(rec):
    """P6: any candidate the other slots do not claim."""
    return not (_p1(rec) or _p2(rec) or _p4(rec))


POPULATIONS = (
    Population(
        pid="P1_SCATTER_IN", name="scatter-in from below the reporting floor",
        side=Side.CANDIDATE,
        predicate_text=f"is_TP AND {ADOPTED_PAD_FLOOR} <= NHI_TRUE < {REPORT_FLOOR}",
        predicate=_p1,
        forward_term="K[c<-b] . C[b,s] . g[b,k] . f[b,k] . dN_b . dX[k,s] "
                     "restricted to basis bins with ntrue_hi <= 19.7",
        prior="the SAME 2-D Gaussian random walk on log f as P2 "
              "(model_a.py:188-202). P1 has NO term and NO prior of its own.",
        support_class=SupportClass.PRIOR_ASSISTED,
        parameter_class=ParameterClass.LATENT_TARGET,
        note="Bins [19.0,19.2) and [19.2,19.5) are BELOW the observed floor: "
             "no observed bin is fed by them except through the kernel's upper "
             "tail (MEASURED row mass 0.120-0.437 and 0.512-0.825 on the "
             "adopted 2LPT-0 pack), and their completeness comes from the "
             "molly172 splice. Bin [19.5,19.7) is observed but is the "
             "NONIDENT_EDGE bin and is excluded from every reported value."),
    Population(
        pid="P2_IN_WINDOW", name="in-window migration between observed-N bins",
        side=Side.CANDIDATE,
        predicate_text=f"is_TP AND {REPORT_FLOOR} <= NHI_TRUE < {REPORT_CEILING}",
        predicate=_p2,
        forward_term="the SAME K.C.g.f.dN.dX restricted to basis bins inside "
                     "[19.7, 21.6); the [21.5,21.7) bin is SPLIT by overlap "
                     "fraction (reporting.truth_overlap_fractions)",
        prior="the 2-D Gaussian random walk on log f (shared with P1)",
        support_class=SupportClass.MEASURED,
        parameter_class=ParameterClass.LATENT_TARGET,
        note="This is the only population the reported dN/dX is about."),
    Population(
        pid="P3_INCOMPLETENESS", name="genuine absorbers that are missed",
        side=Side.TRUTH,
        predicate_text="a truth row of truth_cut (SNR > 2, inside the spectral "
                       "window, inside the basis support) that NO candidate "
                       "claims",
        predicate=_p3,
        forward_term="NO TERM OF ITS OWN. It is the complement (1 - C[b,s]) of "
                     "the completeness factor already in the P1/P2 term.",
        prior="psi_c ~ N(0, sigma_hat) around the Jeffreys-consistent molly "
              "surface eta_hat = log((n_det+1/2)/(n_tot-n_det+1/2)) "
              "(model_a.py:206-207)",
        support_class=SupportClass.MEASURED,
        parameter_class=ParameterClass.INFERRED_NUISANCE,
        note="Denominator = molly_n_tot (cmp_nfid): TRUTH rows per (SNR, "
             "true-N molly cell). NOT the pack's basis bins — the molly grid "
             "is coarser and non-uniform, and its interval convention is OPEN "
             "on both ends while the pack's is half-open."),
    Population(
        pid="P4_FOREST_FP", name="pure-forest false positives, measured on loa-0",
        side=Side.CANDIDATE,
        predicate_text="NOT is_TP AND attributable to the forest (the loa-0 "
                       "HCD-free twin: EVERY loa-0 detection is a forest FP by "
                       "construction)",
        predicate=_p4,
        forward_term="fp_w . exp(t_K(k)) . lam_fp[c,s] . fp_E[k,s] "
                     "(forward.py:452) — FORWARD-MODELLED, never subtracted on "
                     "this route",
        prior="single-Jeffreys TOTAL Gamma(1/2, eps) on the grid-independent "
              "total + a ZeroSumNormal logistic-normal shape "
              "(model_a.py:213-227); loa-0 likelihood "
              "fp_counts ~ Poisson(ell_eff . lam_fp)",
        support_class=SupportClass.WEAKLY_MEASURED,
        parameter_class=ParameterClass.INFERRED_NUISANCE,
        note="MEASURED on the adopted packs: 89 loa-0 detections on the "
             "(c=29, s=8) grid, in 25 of 232 cells, from 2255 searched "
             "sightlines. The FF route (ff_fp_estimator.py) SUBTRACTS instead "
             "— two different treatments cross-linked through t_sigma."),
    Population(
        pid="P5_TRANSFER", name="loa-0 -> absorber-bearing-mock transfer",
        side=Side.RATE_SCALE,
        predicate_text="NOT a population of objects: the per-coarse-z factor "
                       "exp(t_K) that multiplies P4 and nothing else",
        predicate=lambda rec: False,
        forward_term="exp(t[K(k)]) inside the P4 term (forward.py:451-453)",
        prior="t ~ N(0, t_sigma[K]) (model_a.py:210); t_sigma[K] = "
              "max(0.10, max over held-out mocks |ln(R_z/R_z^2lpt0)|) from the "
              "committed ff_fp_{mock}.json sub-DLA closure ratios "
              "(extract_pack.compute_t_sigma:451-472)",
        support_class=SupportClass.PRIOR_ASSISTED,
        parameter_class=ParameterClass.INFERRED_NUISANCE,
        note="t_sigma is a FIXED CALIBRATION PRODUCT; t itself is an INFERRED "
             "NUISANCE. t_sigma is built from an FF-route (FP-SUBTRACTED) "
             "closure ratio and applied to an FP-FORWARD-MODELLED term: that "
             "cross-link is stated, not resolved, here."),
    Population(
        pid="P6_RESIDUAL", name="residual requiring a substantive physical prior",
        side=Side.CANDIDATE,
        predicate_text="any candidate not claimed by P1, P2 or P4. Concretely: "
                       "(a) is_TP AND NHI_TRUE >= 21.6 (scatter-down from above "
                       "the ceiling); (b) is_TP AND NHI_TRUE < 19.0 (below the "
                       "basis floor — NO support at all); (c) NOT is_TP and not "
                       "forest-attributable: blends, a second candidate on an "
                       "already-claimed truth row, and matches beyond dz_rel",
        predicate=_p6_candidate,
        forward_term="NONE. P6 has no term in mu.",
        prior="NONE. Any nonzero P6 must be supplied by a substantive physical "
              "prior that this contract does not contain.",
        support_class=SupportClass.UNSUPPORTED,
        parameter_class=ParameterClass.DERIVED,
        note="P6 is measured as the ledger RESIDUAL, never fitted. Its "
             "above-ceiling sub-slot (a) DOES have a basis bin and is folded; "
             "sub-slots (b) and (c) do not."),
)

POPULATION_BY_ID = {p.pid: p for p in POPULATIONS}


def classify_candidate(rec) -> str:
    """Assign ONE population id to a candidate record. Fail-closed.

    ``rec`` needs keys: ``is_TP`` (bool), ``nhi_true`` (float or nan),
    ``forest_attributable`` (bool).  Exactly one slot must claim it.
    """
    hits = [p.pid for p in POPULATIONS
            if p.side == Side.CANDIDATE and p.predicate(rec)]
    if len(hits) != 1:
        raise ContractViolation(
            f"classify_candidate: record {rec!r} matched {len(hits)} candidate "
            f"populations {hits} — the contract requires EXACTLY one "
            "(no overlap, no gap).")
    return hits[0]


def classify_truth(rec) -> Optional[str]:
    """``P3_INCOMPLETENESS`` for an unmatched truth row, else None (it is
    already counted on the CANDIDATE side and must not be counted twice)."""
    return "P3_INCOMPLETENESS" if _p3(rec) else None


# ---------------------------------------------------------------------------
# 4. quantities: fixed calibration product vs inferred nuisance
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Quantity:
    key: str
    parameter_class: str
    support_class: str
    where: str
    note: str = ""

    def as_dict(self):
        return dataclasses.asdict(self)


QUANTITIES = {q.key: q for q in (
    Quantity("counts", ParameterClass.DATA, SupportClass.MEASURED,
             "pack.counts (C,Kf,S)",
             "RAW op-passing detections. extract_pack.py:874-879. NOT "
             "FP-subtracted on this route."),
    Quantity("fp_counts", ParameterClass.DATA, SupportClass.WEAKLY_MEASURED,
             "pack.fp_counts (C,S)",
             "89 loa-0 forest FPs in 25 of 232 cells."),
    Quantity("truth_counts / truth_counts_bks", ParameterClass.DATA,
             SupportClass.MEASURED, "pack.truth_counts (B,Kf) / (B,Kf,S)",
             "mock truth histogram, SNR > 2 strict, same spectral window."),
    Quantity("dX", ParameterClass.FIXED_CALIBRATION_PRODUCT,
             SupportClass.MEASURED, "pack.dX (Kf,S)",
             "analytic pathlength over the SNR>2, BAL-vetoed sightline set."),
    Quantity("molly_n_det / molly_n_tot", ParameterClass.FIXED_CALIBRATION_PRODUCT,
             SupportClass.MEASURED, "pack.molly_n_det / molly_n_tot (S,M)",
             "frozen 2LPT-0; identical in all three mock packs. Under molly172 "
             "the sub-19.5 cells use a DIFFERENT 'found' threshold."),
    Quantity("g_grid", ParameterClass.FIXED_CALIBRATION_PRODUCT,
             SupportClass.MEASURED, "pack.g_grid (M,Kf)",
             "frozen 2LPT-0 level-preserving z-shape; NOT sampled "
             "(model_a.py docstring, recorded deviation)."),
    Quantity("resp_*_coef", ParameterClass.FIXED_CALIBRATION_PRODUCT,
             SupportClass.MEASURED, "pack.resp_{mu,sig,skew}_coef (SR,ZR,D)",
             "ONE frozen 2LPT-0 kernel shared by all three mocks (VERIFIED "
             "bit-identical). Cross-mock spread therefore measures TRANSFER, "
             "never kernel uncertainty."),
    Quantity("resp_N_fit_range", ParameterClass.FIXED_CALIBRATION_PRODUCT,
             SupportClass.MEASURED, "pack.resp_N_fit_range (SR,ZR,2)",
             "the calibrated covariate range; outside it the fold CLAMPS."),
    Quantity("fp_ell_eff", ParameterClass.FIXED_CALIBRATION_PRODUCT,
             SupportClass.MEASURED, "pack.fp_ell_eff (scalar)",
             "N_sl_loa0 * (N_sl_loa0 / N_prod); the loa-0 Poisson exposure."),
    Quantity("fp_w_sightline_ratio", ParameterClass.FIXED_CALIBRATION_PRODUCT,
             SupportClass.MEASURED, "pack.fp_w_sightline_ratio (scalar)",
             "N_prod / N_sl_loa0. One-sided BAL veto — see "
             "KNOWN_CONTRADICTIONS."),
    Quantity("t_sigma", ParameterClass.FIXED_CALIBRATION_PRODUCT,
             SupportClass.WEAKLY_MEASURED, "pack.t_sigma (KK,)",
             "prior WIDTH of the transfer factor; not window-matched."),
    Quantity("theta_pop / f", ParameterClass.LATENT_TARGET,
             SupportClass.MEASURED, "model_a.py:193-203",
             "log f on the latent basis; the reported quantity."),
    Quantity("psi_c", ParameterClass.INFERRED_NUISANCE, SupportClass.MEASURED,
             "model_a.py:206-207", "completeness logit offsets."),
    Quantity("psi_k_delta", ParameterClass.INFERRED_NUISANCE,
             SupportClass.MEASURED, "model_a.py:208-209",
             "response order-0 mu/sig perturbations."),
    Quantity("t", ParameterClass.INFERRED_NUISANCE, SupportClass.PRIOR_ASSISTED,
             "model_a.py:210", "per-coarse-z log transfer factors."),
    Quantity("fp_lam_total / fp_shape_v", ParameterClass.INFERRED_NUISANCE,
             SupportClass.WEAKLY_MEASURED, "model_a.py:214-223",
             "FP intensity total + shape."),
)}


#: support classification of the TRUE-N axis, region by region, on the adopted
#: geometry.  Every claim here is MEASURED on the adopted 2LPT-0 pack unless it
#: says otherwise.
SUPPORT_MAP = (
    dict(region=[-np.inf, ADOPTED_PAD_FLOOR], name="below the basis floor",
         support=SupportClass.UNSUPPORTED,
         why="no basis bin exists; any genuine absorber here that produces a "
             "candidate lands in P6 and has no term."),
    dict(region=[19.0, 19.5], name="the downward pad",
         support=SupportClass.PRIOR_ASSISTED,
         why="below the observed floor: reached only through the response "
             "kernel's upper tail (MEASURED row mass into the observed grid "
             "0.1204-0.4366 on [19.0,19.2) and 0.5120-0.8247 on [19.2,19.5)); the "
             "response covariate is CLAMPED here (resp_N_fit_range bottom "
             "anchor 19.336); completeness comes from the molly172 splice."),
    dict(region=[19.5, 19.7], name="the non-identifiable edge",
         support=SupportClass.WEAKLY_MEASURED,
         why="observed, but straddles the fit floor and cannot be separated "
             "from edge migration (RP.NONIDENT_EDGE_REASON). Excluded from "
             "every reported value."),
    dict(region=[19.7, 21.04], name="the reporting window, inside the anchors",
         support=SupportClass.MEASURED,
         why="every response cell has calibration anchors here "
             "(RESPONSE_ANCHOR_MEASURED.top_anchor_min = 21.040565)."),
    dict(region=[21.04, 21.6], name="the reporting window, above some anchors",
         support=SupportClass.CLAMPED,
         why="above the LOWEST per-cell top anchor: resp_clamp='both' freezes "
             "the moment polynomials at the anchor. RP."
             "extrapolated_response_inside_window() records this."),
    dict(region=[21.6, 22.4], name="above the reporting ceiling",
         support=SupportClass.CLAMPED,
         why="folded (it feeds observed bins by scatter-down) but never "
             "reported; the D2 residual excess lives here."),
    dict(region=[22.4, np.inf], name="above the basis top edge",
         support=SupportClass.UNSUPPORTED,
         why="no basis bin; scatter-down from above 22.4 lands in P6."),
)


# ---------------------------------------------------------------------------
# 5. the accounting identity
# ---------------------------------------------------------------------------
ACCOUNTING_IDENTITY = dict(
    statement=(
        "TRUTH LEDGER (exact, per basis bin b, fine-z k, stratum s):\n"
        "    T[b,k,s] = FOUND_ON[b,k,s] + FOUND_OFF[b,k,s] + MISSED[b,k,s]\n"
        "  with  FOUND_ON  = T . C[b,s] . rho[b,K(k),s]\n"
        "        FOUND_OFF = T . C[b,s] . (1 - rho[b,K(k),s])\n"
        "        MISSED    = T . (1 - C[b,s])\n"
        "  rho[b,K,s] = sum_c K[c<-b](s,K) is the kernel ROW MASS landing on "
        "the observed grid.\n"
        "\n"
        "CANDIDATE LEDGER (the ledger that can fail):\n"
        "    N_obs = P1 + P2 + P6_ABOVE_CEILING + P4 + P6_UNSUPPORTED\n"
        "  where P1/P2/P6_ABOVE_CEILING partition sum_b FOUND_ON[b] by the "
        "OVERLAP FRACTION of basis bin b with [-inf,19.7), [19.7,21.6) and "
        "[21.6,inf) — the three fractions sum to EXACTLY 1 for every b, which "
        "is what forbids double counting across the reporting edges.\n"
        "\n"
        "CROSS-LINK: every genuine absorber appears exactly once on the truth "
        "ledger and, if and only if it is in FOUND_ON, exactly once on the "
        "candidate ledger. FOUND_OFF appears on the truth ledger ONLY. MISSED "
        "appears on the truth ledger ONLY. P4 appears on the candidate ledger "
        "ONLY."),
    checkable_residuals=dict(
        truth_ledger_residual=(
            "sum(T) - sum(FOUND_ON + FOUND_OFF + MISSED). ZERO by construction; "
            "a nonzero value means the implementation is broken."),
        candidate_ledger_residual=(
            "(P1 + P2 + P6_ABOVE_CEILING + P4) - N_obs, evaluated TRUTH-PINNED "
            "(f fixed so that dX.g.f.dN == the mock's own truth histogram, "
            "psi_c = 0, psi_k = 0, t = 0, lam_fp = fp_counts/ell_eff). ZERO "
            "free parameters. Any imbalance is a FINDING."),
        feasibility=(
            "Because 0 <= C <= 1 and 0 <= rho <= 1, sum_b FOUND_ON <= sum_b T. "
            "So N_obs - P4 <= sum_b T is a NECESSARY condition. Its violation "
            "is a COUNTING ARGUMENT: no parameter setting can close the model. "
            "Reported as efficiency_required = (N_obs - P4) / sum_b T."),
    ),
    truth_pinning=(
        "Pinning Lambda_intrinsic[b,k,s] := truth_counts_bks[b,k,s] bypasses "
        "dX, g and f together — they enter the fold only as their product. The "
        "pin is exact up to the mock's Poisson realisation."),
)


def basis_partition(ntrue_edges, lo=REPORT_FLOOR, hi=REPORT_CEILING):
    """Per-basis-bin fractional split into (below-floor, in-window, above-ceiling).

    Returns a dict of three (B,) arrays that sum to EXACTLY 1.0 in every bin.
    This is the ONLY sanctioned way to attribute a basis bin to P1 / P2 /
    P6_ABOVE_CEILING: a centre test would silently give a straddling bin to one
    side, and on the adopted 0.2-dex basis the ceiling 21.6 DOES straddle
    [21.5, 21.7) because 21.6 - 19.7 = 1.9 dex is an odd multiple of 0.1.
    """
    e = np.asarray(ntrue_edges, float)
    if e.ndim != 1 or len(e) < 2 or np.any(np.diff(e) <= 0):
        raise ContractViolation(f"basis_partition: bad edges {e}")
    f_in = RP.truth_overlap_fractions(e, lo, hi)
    dN = np.diff(e)
    f_below = np.clip(np.minimum(e[1:], lo) - e[:-1], 0.0, None) / dN
    f_above = np.clip(e[1:] - np.maximum(e[:-1], hi), 0.0, None) / dN
    tot = f_below + f_in + f_above
    if not np.allclose(tot, 1.0, atol=1e-12):
        raise ContractViolation(
            "basis_partition: the three fractions do not sum to 1 in every "
            f"bin (max deviation {np.max(np.abs(tot - 1.0)):.3e}) — this is the "
            "double-counting guard and it FAILED.")
    return dict(below_floor=f_below, in_window=f_in, above_ceiling=f_above)


# ---------------------------------------------------------------------------
# 6. FP normalisation
# ---------------------------------------------------------------------------
FP_NORMALISATION = dict(
    contract=("mu_FP[c,k,s] = (N_prod / N_sl_loa0) . N_FP_loa0[c,s] . "
              "exp(t_K) . E[k,s], i.e. in the pack's own scalars "
              "mu_FP = fp_w . fp_ell_eff . lam_fp . exp(t) . E, because "
              "lam_fp is defined by the loa-0 likelihood "
              "fp_counts ~ Poisson(fp_ell_eff . lam_fp)."),
    source=("CDDF_analysis/hbi/build_loa0_fp_product.py:35-39 — "
            "mu_FP = (N_prod/N_sl_loa0) . N_FP_loa0_total . (1 - eta_bar), "
            "ell_eff = N_sl_loa0 . (N_sl_loa0/N_prod)."),
    identity="fp_w . fp_ell_eff == N_sl_loa0 exactly (VERIFIED == 2255.0 on all "
             "three adopted packs).",
    eta="eta_DLA is FORCED to 0 (build_loa0_fp_product.py:DLA_ETA), so the "
        "(1 - eta_bar) factor is 1 on the pack's N >= 19.5 grid.",
)


def fp_normalisation_audit(pack) -> dict:
    """Compare the CONTRACT's FP normalisation with what ``forward.py`` folds.

    Pure arithmetic on pack scalars — no jax, no fold.  Returns both totals and
    their ratio.  ``check_accounting_identity`` reports the candidate ledger
    under BOTH so the finding is quantified rather than argued.
    """
    w = float(pack.fp_w_sightline_ratio)
    ell = float(pack.fp_ell_eff)
    n_fp = float(np.asarray(pack.fp_counts, float).sum())
    lam_tot = n_fp / ell
    implemented = w * lam_tot                    # forward.py:452
    contract = w * ell * lam_tot                 # == w * n_fp
    return dict(
        n_fp_loa0=n_fp, fp_w_sightline_ratio=w, fp_ell_eff=ell,
        lam_total_plugin=lam_tot,
        mu_fp_total_as_implemented=implemented,
        mu_fp_total_per_contract=contract,
        ratio_contract_over_implemented=(contract / implemented
                                         if implemented > 0 else np.inf),
        n_sl_loa0_implied=w * ell,
        site="CDDF_analysis/hbi_mcmc/forward.py:452 (mu_fp = consts.fp_w * "
             "exp_t_k * lam_fp * fp_E) — fp_ell_eff is absent from the "
             "expression although build_consts carries it.",
    )


def assert_forward_fp_normalisation(pack, *, rtol=1e-9):
    """FAIL LOUDLY if the committed fold's FP term is not the contract's.

    This is deliberately a check against the CODE, not against the pack: the
    pack's scalars are correct and the fold's use of them is not.
    """
    a = fp_normalisation_audit(pack)
    if abs(a["ratio_contract_over_implemented"] - 1.0) > rtol:
        raise ContractViolation(
            "FP NORMALISATION VIOLATION: the contract requires "
            "mu_FP = fp_w * fp_ell_eff * lam_fp * exp(t) * E, but "
            "forward.py:452 folds mu_FP = fp_w * lam_fp * exp(t) * E. On this "
            f"pack the contract total is {a['mu_fp_total_per_contract']:.4f} "
            f"and the implemented total is "
            f"{a['mu_fp_total_as_implemented']:.4f}: under-normalised by "
            f"exactly fp_ell_eff = {a['fp_ell_eff']:.6f} "
            f"(measured ratio {a['ratio_contract_over_implemented']:.9f}). "
            "See FP_NORMALISATION and KNOWN_CONTRADICTIONS[0].")
    return a


# ---------------------------------------------------------------------------
# 7. contract validation of an input pack
# ---------------------------------------------------------------------------
def validate_pack_against_contract(pack, *, require_pad=True,
                                   require_measured_sub_floor_completeness=True
                                   ) -> dict:
    """Fail-closed: raise ``ContractViolation`` on a pack this contract cannot
    describe.  Returns the geometry facts it verified.

    Rules (each is a real defect that has occurred in this project):
      1. the observed grid is the schema grid and never moves;
      2. the reporting floor 19.7 is an EXACT latent-basis edge, so no bin
         straddles the pad/report boundary;
      3. the latent basis extends BELOW the observed floor (the D1 pad).
         Without it the counting argument is refuted before any fit;
      4. the completeness cell grid covers the basis support FROM BELOW,
         otherwise ``forward.build_consts``'s clip silently applies a CONSTANT
         EXTRAPOLATION on the pad and P1's completeness is a convention;
      5. ``truth_counts_bks`` is present — the accounting identity is not
         checkable without it;
      6. ``counts`` is RAW (no FP subtraction can have produced a negative);
      7. the FP scalars are positive and finite.
    """
    ce = np.asarray(pack.nhat_edges, float)
    ne = np.asarray(pack.ntrue_edges, float)

    # 1
    if not (np.isclose(ce[0], OBSERVED_FLOOR) and np.isclose(ce[-1], OBSERVED_CEILING)
            and np.allclose(np.diff(ce), OBSERVED_STEP, atol=1e-8)):
        raise ContractViolation(
            f"observed grid moved: nhat_edges span [{ce[0]}, {ce[-1]}] with "
            f"steps {np.unique(np.round(np.diff(ce), 8))}; the contract fixes "
            f"[{OBSERVED_FLOOR}, {OBSERVED_CEILING}] step {OBSERVED_STEP}.")

    # 2
    if not np.any(np.isclose(ne, REPORT_FLOOR, atol=1e-8)):
        raise ContractViolation(
            f"the reporting floor {REPORT_FLOOR} is NOT an exact latent-basis "
            f"edge (ntrue_edges = {ne}). A bin straddling it would mix "
            "convention-dependent sub-floor support (P1) into an in-window "
            "bin (P2) — the two populations would not be separable.")

    # 3
    n_pad = int(np.sum(ne[:-1] < ce[0] - 1e-9))
    if require_pad and n_pad == 0:
        raise ContractViolation(
            "the latent basis is TRUNCATED at the observed floor (no D1 pad): "
            "P1_SCATTER_IN has no support, so every genuine absorber below "
            f"{OBSERVED_FLOOR} that scatters into the grid must be absorbed by "
            "P2 or P6. Pass require_pad=False only to DEMONSTRATE the "
            "resulting infeasibility.")

    # 4
    me = np.asarray(pack.molly_nhi_edges, float)
    sub_floor_measured = me[0] <= ne[0] + 1e-8
    if require_measured_sub_floor_completeness and not sub_floor_measured:
        raise ContractViolation(
            f"the completeness cell grid starts at {me[0]} but the latent "
            f"basis starts at {ne[0]}: forward.build_consts clips b_to_cell to "
            "0, so the pad's completeness is the CONSTANT EXTRAPOLATION of "
            "molly cell 0 (KNOWN TOO HIGH), not a measurement. That is the "
            "'const_extrap' convention; the adopted convention is "
            f"'{ADOPTED_COMPLETENESS_CONVENTION}'. Pass "
            "require_measured_sub_floor_completeness=False to accept it "
            "EXPLICITLY as a stated systematic.")

    # 5
    if pack.truth_counts_bks is None:
        raise ContractViolation(
            "pack.truth_counts_bks is None — the accounting identity is "
            "stratified by (b, k, s) because the completeness is a function of "
            "(true-N cell, SNR stratum). Re-extract the pack.")

    # 6
    c = np.asarray(pack.counts)
    if np.any(c < 0):
        raise ContractViolation(
            "pack.counts contains negative entries: `counts` must be RAW "
            "op-passing detections. On this route the FP is FORWARD-MODELLED "
            "(P4), never subtracted; a subtracted array is a different "
            "estimand and cannot enter this ledger.")

    # 7
    for k in ("fp_ell_eff", "fp_w_sightline_ratio"):
        v = float(getattr(pack, k))
        if not (np.isfinite(v) and v > 0):
            raise ContractViolation(f"pack.{k} = {v}: must be finite positive.")

    part = basis_partition(ne)
    return dict(
        contract_version=CONTRACT_VERSION,
        n_basis_bins=int(len(ne) - 1), n_pad_bins=n_pad,
        basis_floor=float(ne[0]), basis_top=float(ne[-1]),
        basis_widths=np.round(np.diff(ne), 8).tolist(),
        reporting_floor_is_a_basis_edge=True,
        ceiling_straddles_basis_bin=bool(
            np.any((part["in_window"] > 0) & (part["above_ceiling"] > 0))),
        sub_floor_completeness_measured=bool(sub_floor_measured),
        molly_floor=float(me[0]), n_molly_cells=int(len(me) - 1),
    )


# ---------------------------------------------------------------------------
# 8. THE CHECK
# ---------------------------------------------------------------------------
def _row_mass(pack, resp_clamp="both"):
    """rho[b, K, s] = sum_c K[c<-b](s, K): the fraction of FOUND systems from
    basis bin b that land on the OBSERVED grid.  Routed through the committed
    ``forward.build_K`` (jax) — never re-implemented here."""
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.forward import build_consts, build_K
    consts = build_consts(pack, resp_clamp=resp_clamp,
                          allow_unclamped_response=(resp_clamp == "off"))
    K = np.asarray(build_K(jnp.zeros((2, consts.n_sr, consts.n_zr)), consts))
    return K.sum(axis=2), consts        # (S, KK, B), consts


def _eta_hat(n_det, n_tot):
    """Jeffreys-consistent completeness logit — the SAME formula the fold uses
    (forward.eta_hat_sigma_hat), inlined so this module stays importable in the
    jax-free data-plane env."""
    d = np.asarray(n_det, float)
    t = np.asarray(n_tot, float)
    return np.log((d + 0.5) / (t - d + 0.5))


def _b_to_cell(pack):
    """b -> molly cell, EXACTLY forward.build_consts:338."""
    ne = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ne[:-1] + ne[1:])
    me = np.asarray(pack.molly_nhi_edges, float)
    return np.clip(np.digitize(Nc, me) - 1, 0, len(me) - 2).astype(int)


def check_accounting_identity(pack, *, resp_clamp="both",
                              validate=True, row_mass=None, b_to_cell=None,
                              **validate_kw) -> dict:
    """Evaluate the accounting identity on a REAL pack and return the residuals.

    TRUTH-PINNED and parameter-free: Lambda_intrinsic := truth_counts_bks,
    C := sigmoid(eta_hat) (psi_c = 0), psi_k_delta = 0, t = 0,
    lam_fp := fp_counts / fp_ell_eff.

    ``row_mass`` (S, KK, B) may be injected so the ledger is unit-testable
    without jax; by default it is computed through the committed
    ``forward.build_K``.

    Returns a JSON-serializable dict.  The two numbers a referee should read
    are ``truth_ledger.residual`` (MUST be 0) and
    ``candidate_ledger.residual_per_contract`` (a FINDING, never tuned).
    """
    geom = validate_pack_against_contract(pack, **validate_kw) if validate else {}

    if row_mass is None:
        rho, consts = _row_mass(pack, resp_clamp=resp_clamp)       # (S,KK,B)
        b2c = np.asarray(consts.b_to_cell)
    else:
        rho = np.asarray(row_mass, float)
        b2c = np.asarray(_b_to_cell(pack) if b_to_cell is None else b_to_cell)
    if np.any(rho < -1e-12) or np.any(rho > 1.0 + 1e-9):
        raise ContractViolation(
            f"row mass outside [0, 1] (min {rho.min()}, max {rho.max()}): the "
            "kernel is a probability distribution over N-hat and its mass on "
            "the observed grid cannot exceed 1. The feasibility bound depends "
            "on this.")
    C_cells = 1.0 / (1.0 + np.exp(-_eta_hat(pack.molly_n_det, pack.molly_n_tot)))
    C_bs = C_cells[:, b2c]                                        # (S, B)

    T = np.asarray(pack.truth_counts_bks, float)                  # (B, Kf, S)
    kz = np.asarray(pack.kz_to_K, int)                            # (Kf,)
    # rho aligned to (B, Kf, S)
    rho_bks = np.transpose(rho[:, kz, :], (2, 1, 0))              # (B, Kf, S)
    C_b_s = C_bs.T[:, None, :]                                    # (B, 1, S)

    found_on = T * C_b_s * rho_bks
    found_off = T * C_b_s * (1.0 - rho_bks)
    missed = T * (1.0 - C_b_s)
    truth_resid = float(T.sum() - (found_on.sum() + found_off.sum() + missed.sum()))

    part = basis_partition(np.asarray(pack.ntrue_edges, float))
    on_b = found_on.sum(axis=(1, 2))                              # (B,)
    p1 = float((on_b * part["below_floor"]).sum())
    p2 = float((on_b * part["in_window"]).sum())
    p6hi = float((on_b * part["above_ceiling"]).sum())

    fpa = fp_normalisation_audit(pack)
    n_obs = float(np.asarray(pack.counts, float).sum())
    sig = p1 + p2 + p6hi

    def _ledger(p4):
        pred = sig + p4
        return dict(P4_forest_fp=p4, predicted_total=pred,
                    residual=pred - n_obs,
                    rel_residual=(pred - n_obs) / n_obs if n_obs else np.nan)

    led_impl = _ledger(fpa["mu_fp_total_as_implemented"])
    led_contract = _ledger(fpa["mu_fp_total_per_contract"])

    t_tot = float(T.sum())
    return dict(
        contract_version=CONTRACT_VERSION,
        geometry=geom,
        resp_clamp=resp_clamp,
        truth_ledger=dict(
            n_truth_on_basis=t_tot,
            found_on_grid=float(found_on.sum()),
            found_off_grid=float(found_off.sum()),
            missed_P3=float(missed.sum()),
            residual=truth_resid,
            rel_residual=truth_resid / t_tot if t_tot else np.nan,
        ),
        candidate_ledger=dict(
            n_obs=n_obs,
            P1_scatter_in=p1, P2_in_window=p2, P6_above_ceiling=p6hi,
            signal_subtotal=sig,
            as_implemented=led_impl,
            per_contract=led_contract,
            residual_per_contract=led_contract["residual"],
            P6_unsupported_implied_per_contract=n_obs - led_contract["predicted_total"],
        ),
        fp_normalisation=fpa,
        feasibility=dict(
            max_attainable_signal=t_tot,
            required_signal_per_contract=n_obs - fpa["mu_fp_total_per_contract"],
            required_signal_as_implemented=n_obs - fpa["mu_fp_total_as_implemented"],
            efficiency_required_per_contract=(
                (n_obs - fpa["mu_fp_total_per_contract"]) / t_tot),
            efficiency_required_as_implemented=(
                (n_obs - fpa["mu_fp_total_as_implemented"]) / t_tot),
            # the efficiency the MEASURED calibration actually delivers:
            # sum_b C[b,s] . rho[b,K,s] weighted by the truth. The hard bound is
            # 1; THIS is the bound that matters, and it is always <= 1.
            efficiency_attainable_at_calibration=(sig / t_tot if t_tot else np.nan),
            feasible_per_contract=bool(
                (n_obs - fpa["mu_fp_total_per_contract"]) <= t_tot),
            feasible_at_calibration_per_contract=bool(
                (n_obs - fpa["mu_fp_total_per_contract"]) <= sig),
            feasible_at_calibration_as_implemented=bool(
                (n_obs - fpa["mu_fp_total_as_implemented"]) <= sig),
            note="C <= 1 and rho <= 1, so the signal can never exceed the "
                 "truth count on the basis: efficiency_required > 1 is a "
                 "COUNTING ARGUMENT and no parameter setting closes the model. "
                 "efficiency_attainable_at_calibration is the SHARPER bound — "
                 "what the frozen completeness and kernel actually deliver — "
                 "and closing above it requires moving a calibration product, "
                 "not the population.",
        ),
    )


# ---------------------------------------------------------------------------
# 9. where the code contradicts this contract
# ---------------------------------------------------------------------------
KNOWN_CONTRADICTIONS = (
    dict(
        id="FP_ELL_EFF_OMITTED",
        site="CDDF_analysis/hbi_mcmc/forward.py:452 (and the same expression in "
             "forward.py:607 fold_mu_reference and "
             "forward_selftest.py:163)",
        contract="mu_FP = fp_w . fp_ell_eff . lam_fp . exp(t) . E",
        code="mu_FP = fp_w . lam_fp . exp(t) . E",
        measured="under-normalised by exactly fp_ell_eff; MEASURED on the "
                 "adopted 2LPT-0 pack 1086.687 vs 14767.961, ratio "
                 "13.589891949531909 == fp_ell_eff. fp_w . fp_ell_eff == "
                 "2255.0 == N_sl_loa0 exactly on all three packs.",
        effect="P4 is ~13.6x too small; the fold's FP term is a rate the "
               "loa-0 likelihood does not define.",
        status="REPORTED, NOT FIXED — changing it changes model behaviour and "
               "is out of this contract's scope.",
    ),
    dict(
        id="FOUND_SPLICED_AT_19.5",
        site="CDDF_analysis/hbi_mcmc/extract_pack.py:555-557 "
             "(load_molly_counts_block, molly172); threshold defined at "
             "CDDF_analysis/hbi/cddf_catalog_hbi.py:752 and :1563, used at "
             ":1581",
        contract="ONE definition of 'found' across the whole completeness "
                 "surface, and it must be the event the response kernel is "
                 "conditional on (found AT ALL), because the fold multiplies "
                 "C by the kernel row mass rho.",
        code="cells < 19.5 use 'found <=> N_hat > 17.2'; cells >= 19.5 use "
             "'found <=> N_hat > 19.5'.",
        measured="On the adopted geometry the >= 19.5 half double-excludes "
                 "reconstructions below the observed floor, once in C and "
                 "again in rho. The size of that double-exclusion is bounded "
                 "by max over (s,K) of (1 - rho) and is MEASURED on the "
                 "adopted 2LPT-0 pack at 5.7089e-2 on the basis bin "
                 "[19.5,19.7), 9.1423e-4 on [19.7,19.9), 4e-8 on [19.9,20.1) "
                 "and <= 1.2e-6 on every bin up to [21.3,21.5).",
        effect="Confined to P1 and to the NONIDENT_EDGE bin; inside the "
               "reported window [19.7,21.6) the largest affected bin is "
               "[19.7,19.9) at 9.1e-4.",
        status="MEASURED AND BOUNDED. The splice is load-bearing exactly on "
               "the pad, which is P1.",
    ),
    dict(
        id="BAL_VETO_ONE_SIDED_IN_FP_W",
        site="CDDF_analysis/hbi/build_loa0_fp_product.py:231 ('loa-0 is "
             "BAL-free; no-bal is a no-op') and :262 "
             "count_searched_sightlines_loa0, vs "
             "cddf_catalog_hbi.build_pathlength:846-860 which DOES veto BAL",
        contract="P4's rate scale N_prod / N_sl_loa0 must divide a numerator "
                 "and a denominator defined on the SAME sightline set.",
        code="N_prod EXCLUDES BAL targets; N_sl_loa0 does NOT.",
        measured="MEASURED by cross-matching loa-0 target_ids against "
                 "loa-124 bal_cat.fits (byte-identical twins, same TARGETIDs): "
                 "of the 2255 searched loa-0 sightlines with SNR>2, 351 sit on "
                 "BAL targets, leaving 1904; of the 89 loa-0 FPs on the pack "
                 "grid, 19 sit on BAL targets, leaving 70. The support-matched "
                 "rate is 70/1904 = 0.0367647/sightline; the implemented rate "
                 "is 89/2255 = 0.0394678. Implemented / matched = 1.07352.",
        effect="P4 is 7.35% HIGH before the FP_ELL_EFF_OMITTED factor is "
               "considered. The two errors act in OPPOSITE directions and do "
               "not cancel (13.6x vs 1.07x).",
        status="REPORTED, NOT FIXED. The loa-0 mockdir carries no bal_cat.fits "
               "of its own; the veto set has to come from the loa-124 twin.",
    ),
    dict(
        id="MOLLY_INTERVAL_CONVENTION",
        site="examples/molly_faithful_pc_plots.py:529-548 (OPEN intervals) vs "
             "CDDF_analysis/hbi_mcmc/extract_pack.py:370-371 _idx "
             "(half-open [lo, hi))",
        contract="one interval convention per axis.",
        code="the completeness numerator and denominator use OPEN (n_lo, n_hi) "
             "and (s_lo, s_hi); the pack's counts and truth histograms use "
             "half-open [lo, hi).",
        measured="Affects only rows landing EXACTLY on an edge. Handled by "
                 "track_c_tf_saclay._snap_off_molly_edges, which the extractor "
                 "applies to every bundle; MEASURED as a no-op on 2lpt0 and "
                 "london0 and one NHI_TRUE == 20.0 row on saclay0 (the "
                 "committed tie-break).",
        status="BOUNDED AND HANDLED; recorded so it is not rediscovered.",
    ),
    dict(
        id="TRUTH_FLOOR_ASYMMETRY_IN_is_TP",
        site="CDDF_analysis/hbi_mcmc/extract_pack.py:715-720 "
             "(load_mock_bundle sets truth_floor = mm.nhi_edges[0] = 19.5 for "
             "the DETECTION side) vs :765-789 (load_truth_bundle re-cuts the "
             "TRUTH side at the pad floor)",
        contract="P1 and P6 are distinguished by the true N of the genuine "
                 "absorber, which requires the matcher to have SEEN absorbers "
                 "down to the basis floor.",
        code="the pack's detection side is matched against a truth table "
             "pre-floored at 19.5, so a candidate whose genuine absorber is at "
             "N = 19.2 is is_TP == False and is indistinguishable, in the "
             "pack, from a forest FP.",
        measured=(
            "MEASURED 2026-08-05 on the 19.5-floor 2LPT-0 DETECTION bundle "
            "(extract_pack.load_mock_bundle, 14 s): n_cat_cut 582855, "
            "n_op 495553, 88071 on the pack grid, of which 63890 is_TP and "
            "24181 unmatched. ZERO is_TP rows have NHI_TRUE < 19.5 -- the "
            "truth table was pre-floored there. Inside the observed reporting "
            "window N_hat in [19.7, 21.6) the 67086 candidates split "
            "53401 / 4521 / 70 / 9094 into (true N in window) / (true N < "
            "19.7) / (true N >= 21.6) / (unmatched). The SAME window cut on a "
            "17.2-floor bundle is reported elsewhere as 67078 = 53401 + 8591 + "
            "70 + 5016: the in-window and above-ceiling slots are IDENTICAL "
            "and 4070 candidates move from 'unmatched' to 'below 19.7' purely "
            "because the matcher can now see them. The 8-candidate difference "
            "in the totals is the known cat_cut perturbation "
            "(extract_pack.py:962-970, 88071 -> 88053 over the whole grid)."),
        effect="P1 and P6 are NOT separable at the per-candidate level without "
               "naming the truth floor the matcher ran at. Two defensible "
               "splits of the same 67086 candidates exist and differ by 4070.",
        status="STRUCTURAL. It is why check_accounting_identity works on the "
               "MODEL side (basis bins, where the truth histogram is cut at "
               "the basis floor) and not on the per-candidate side.",
    ),
)


PI_CHECKPOINT_ITEMS = (
    "Fixing FP_ELL_EFF_OMITTED changes the PHYSICAL size of P4 by ~13.6x and "
    "therefore the physical definition of what the DLA counts contain. "
    "Collected, not raised.",
    "Support-matching the BAL veto in fp_w (70/1904 rather than 89/2255) "
    "changes P4 by 1.0735x and requires deciding whether loa-0's BAL set is "
    "taken from its loa-124 twin. Collected, not raised.",
    "Adopting ONE definition of 'found' across the completeness surface would "
    "move the sub-floor completeness cells and therefore P1. Collected.",
    "Whether P6_ABOVE_CEILING (basis bins above 21.6, currently folded but "
    "never reported) should instead be given a prior is a physical question. "
    "Collected.",
    "P6's unsupported sub-slots (genuine absorbers below 19.0, and blends) "
    "have NO term. Giving them one is a new physical assumption. Collected.",
)


# ---------------------------------------------------------------------------
# 10. serialization
# ---------------------------------------------------------------------------
def contract_dict() -> dict:
    """The whole contract as a JSON-serializable dict."""
    return dict(
        contract_version=CONTRACT_VERSION,
        authority=dict(
            newly_ratified_here=[],
            reused_constants=dict(
                reporting_floor=dict(value=REPORT_FLOOR,
                                     source=RP.NONIDENT_EDGE_SOURCE),
                reporting_ceiling=dict(value=REPORT_CEILING,
                                       source="reporting.RESPONSE_ANCHOR_CEILING"),
            ),
            adopted_baseline=dict(
                basis_width_dex=ADOPTED_BASIS_WIDTH,
                pad_floor=ADOPTED_PAD_FLOOR,
                completeness_convention=ADOPTED_COMPLETENESS_CONVENTION,
                analysis_window=ADOPTED_ANALYSIS_WINDOW,
                target="dN/dX", omega_hi="NOT emitted", lls="out of Paper 1",
                calibration_mock="2LPT-0", transfer_mocks=["London-0", "Saclay-0"],
            ),
            note="No field of this contract asserts PI authority. Items that "
                 "would change a population's physical definition are in "
                 "PI_CHECKPOINT_ITEMS.",
        ),
        axes={k: v.as_dict() for k, v in AXES.items()},
        spectral_window=SPECTRAL_WINDOW,
        pathlength=PATHLENGTH,
        matching=MATCHING,
        populations=[p.as_dict() for p in POPULATIONS],
        quantities={k: v.as_dict() for k, v in QUANTITIES.items()},
        support_map=[dict(d, region=[float(d["region"][0]), float(d["region"][1])])
                     for d in SUPPORT_MAP],
        accounting_identity=ACCOUNTING_IDENTITY,
        fp_normalisation=FP_NORMALISATION,
        known_contradictions=list(KNOWN_CONTRADICTIONS),
        pi_checkpoint_items=list(PI_CHECKPOINT_ITEMS),
    )


def contract_json(indent=1) -> str:
    return json.dumps(contract_dict(), indent=indent, default=str)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", default=None,
                   help="run the accounting-identity check on this pack NPZ")
    p.add_argument("--resp-clamp", default="both", choices=["both", "hi", "off"])
    p.add_argument("--no-require-pad", action="store_true")
    p.add_argument("--allow-const-extrap", action="store_true")
    p.add_argument("--dump-contract", action="store_true")
    a = p.parse_args(argv)
    if a.dump_contract or a.pack is None:
        print(contract_json())
        if a.pack is None:
            return 0
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    pk = load_pack(a.pack, allow_nonstandard_grid=True)
    rep = check_accounting_identity(
        pk, resp_clamp=a.resp_clamp,
        require_pad=not a.no_require_pad,
        require_measured_sub_floor_completeness=not a.allow_const_extrap)
    print(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
