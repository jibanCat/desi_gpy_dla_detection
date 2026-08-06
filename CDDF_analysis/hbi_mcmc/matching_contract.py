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

WHAT THIS MODULE DOES **NOT** CLAIM (corrected 2026-08-05 after referee C3)
--------------------------------------------------------------------------
It does NOT emit a feasibility verdict on the adopted geometry.  An earlier
version evaluated the fold at ``psi_c = 0, psi_k_delta = 0`` and called the
resulting efficiency "the SHARPER bound", emitting
``feasible_at_calibration_per_contract``.  That was WRONG: ``psi_c`` and
``psi_k_delta`` are sample sites with UNBOUNDED support (model_a.py:206-209),
so a point value in that space is not an upper bound on anything.  Both the
boolean and the word "bound" are RETRACTED.  What replaces them is a
PRIOR-COST quantity — the minimum Mahalanobis distance in the DECLARED
nuisance space needed to reach the observed counts — and the reader judges.
The only genuine bound the module still asserts is the trivial ``efficiency
<= 1`` (C <= 1 and rho <= 1).  See ``prior_cost_audit`` and RETRACTIONS.

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

A GUARD THAT FIRES ON THE FIXED STATE (corrected 2026-08-05, second pass)
-------------------------------------------------------------------------
``assert_forward_fp_normalisation`` used to compare the contract against a
HARD-CODED reading of ``forward.py``'s FP expression.  When the fold was
repaired (7707c8e, 2b436df) the reading went stale in silence and the guard
raised on correct code, permanently.  It now obtains the folded total by
CALLING ``forward.fold_mu_fp``, so it tracks the code by construction and
fails only if the omission comes back.  The contradiction record itself is not
deleted — it moves to ``RESOLVED_CONTRADICTIONS`` with its fixing commits and
its measured before/after.  A description of the code is a claim like any
other and rots like any other; prefer measuring the code.

ENV: jax is REQUIRED.  ``check_accounting_identity`` and (since this commit)
``fp_normalisation_audit`` both reach the committed fold, importing it lazily;
injecting ``row_mass=`` now avoids only ``build_K``, not jax.

🔴 An earlier version of this paragraph said the module was "importable in the
jax-free `gpdla` data-plane env (load it file-directly)".  That was FALSE, and
was false before this commit: the top-level ``from CDDF_analysis.hbi_mcmc
import reporting`` executes the package ``__init__``, which imports jax, so a
file-direct ``importlib`` load raises ``ModuleNotFoundError: No module named
'jax'`` under ``gpdla`` — MEASURED 2026-08-05 against BOTH this revision and
its parent.  Use ``gpdla-hbi``.
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
    "KNOWN_CONTRADICTIONS", "CONTRADICTION_BY_ID", "PI_CHECKPOINT_ITEMS",
    "RESOLVED_CONTRADICTIONS", "RESOLVED_BY_ID",
    "RETRACTIONS", "FP_CEILING_MEASURED", "TRUTH_OUT_OF_BASIS_SUPPORT",
    "contract_dict", "basis_partition", "classify_candidate", "classify_truth",
    "assert_adopted_constants_agree_with_reporting", "ADOPTED_BASIS_TOP",
    "validate_pack_against_contract", "fp_normalisation_audit",
    "assert_forward_fp_normalisation", "check_accounting_identity",
    "prior_cost_audit", "min_prior_chi2_psi_c", "fp_ceiling_audit",
]

#: 1.1 — C3 retraction (no feasibility verdict), fail-closed classifiers, NaN
#: guards, per-slot value guards, FP ceiling check, BAL magnitude retracted.
#: 1.2 — FP_ELL_EFF_OMITTED moved to RESOLVED_CONTRADICTIONS (fixed in code by
#: 7707c8e / 2b436df); the FP normalisation guard now MEASURES the committed
#: fold instead of describing it, and fails on a re-introduced omission; FP
#: ceiling measured on all three mocks; FP z-shape one-sided support recorded.
CONTRACT_VERSION = "1.2"


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
#: RE-USED from ``reporting.ADOPTED_CONFIG`` — NOT re-declared here.  A referee
#: minor (2026-08-05): these were hard-coded literals while the docstring
#: claimed reuse, and ``ADOPTED_PAD_FLOOR`` is load-bearing as the P1/P6
#: boundary in ``_p1`` / ``_p6_candidate``, so a silent divergence from
#: reporting.py would move a population edge.
ADOPTED_BASIS_WIDTH = RP.ADOPTED_CONFIG["basis_width_dex"]
ADOPTED_PAD_FLOOR = RP.ADOPTED_CONFIG["basis_pad_floor"]
ADOPTED_COMPLETENESS_CONVENTION = RP.ADOPTED_CONFIG["completeness_below_floor"]

#: NOT in ``reporting.ADOPTED_CONFIG`` — reporting carries the logN reporting
#: window (``REPORTING_WINDOW``), not the SPECTRAL analysis window.  The
#: canonical spectral value lives in ``extract_pack.ANALYSIS_WINDOWS``
#: (lya_only: lam_rf_min = 1025.0), which this module deliberately does not
#: import (extract_pack pulls jax and this module must stay importable in the
#: jax-free data-plane env).  It is therefore DECLARED here, and
#: ``assert_adopted_constants_agree_with_reporting`` cannot check it.
ADOPTED_ANALYSIS_WINDOW = dict(name="lya_only", lam_rf_min=1025.0,
                               lam_rf_max=1216.0,
                               source="extract_pack.ANALYSIS_WINDOWS['lya_only']"
                                      " — NOT carried by reporting.py")


def assert_adopted_constants_agree_with_reporting():
    """FAIL LOUDLY if this module's adopted constants drift from reporting.py.

    Three of the four are taken BY REFERENCE above, so this can only fail if a
    future edit re-declares one as a literal.  The fourth (the spectral window)
    has no counterpart in reporting.py and is not checked — see above.
    """
    ac = RP.ADOPTED_CONFIG
    want = {
        "basis_width_dex": ADOPTED_BASIS_WIDTH,
        "basis_pad_floor": ADOPTED_PAD_FLOOR,
        "completeness_below_floor": ADOPTED_COMPLETENESS_CONVENTION,
    }
    bad = {k: (ac[k], v) for k, v in want.items() if ac[k] != v}
    if bad or list(ac["reporting_window_logN"]) != [REPORT_FLOOR, REPORT_CEILING]:
        raise ContractViolation(
            "the adopted baseline in matching_contract has DRIFTED from "
            f"reporting.ADOPTED_CONFIG: {bad}; window "
            f"{list(ac['reporting_window_logN'])} vs "
            f"[{REPORT_FLOOR}, {REPORT_CEILING}]. These constants are "
            "re-used, never re-derived (docs/RULES.md).")
    return True

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


#: the TOP of the adopted latent basis.  A truth row above it has no basis bin
#: (SUPPORT_MAP's last region) and therefore cannot be P3 — it is not in the
#: completeness denominator the fold applies.
ADOPTED_BASIS_TOP = 22.4

#: returned by ``classify_truth`` for an unmatched truth row that lies OUTSIDE
#: the basis support.  It is NOT P3: P3 is the (1 - C) complement of a term the
#: fold actually carries, and outside the basis there is no term.
TRUTH_OUT_OF_BASIS_SUPPORT = "OUT_OF_BASIS_SUPPORT"


def _is_num(x):
    """True only for a REAL, FINITE number.

    Fail-closed (referee minor, 2026-08-05): a ``str`` used to be silently
    coerced, so ``classify_candidate(nhi_true="20.5")`` returned P2 while the
    docstring promised fail-closed.  Strings, bytes and bools are now refused.
    """
    if x is None or isinstance(x, (str, bytes, bool, np.bool_)):
        return False
    if not isinstance(x, (int, float, np.integer, np.floating)):
        return False
    return bool(np.isfinite(float(x)))


def _require_keys(rec, keys, who):
    """Fail-closed on a record that does not DECLARE the fields the predicates
    read.  ``dict.get`` silently turns a missing key into False/None, which is
    how ``classify_candidate({})`` used to return P6_RESIDUAL — a real answer
    for a record that says nothing (referee minor, 2026-08-05)."""
    try:
        missing = [k for k in keys if k not in rec]
    except TypeError:
        raise ContractViolation(f"{who}: {rec!r} is not a mapping.")
    if missing:
        raise ContractViolation(
            f"{who}: record {rec!r} is missing required key(s) {missing}. "
            "The contract is fail-closed: an undeclared field is NOT a "
            "default, because the slot it would silently pick (P6_RESIDUAL / "
            "P3_INCOMPLETENESS) is exactly the slot with no forward term.")


def _p1(rec):
    """P1: genuine absorber BELOW the reporting floor, detected on the grid."""
    return (bool(rec.get("is_TP")) and _is_num(rec.get("nhi_true"))
            and float(rec["nhi_true"]) >= ADOPTED_PAD_FLOOR - 1e-9
            and float(rec["nhi_true"]) < REPORT_FLOOR - 1e-9)


def _p2(rec):
    """P2: genuine absorber INSIDE the reporting window."""
    return (bool(rec.get("is_TP")) and _is_num(rec.get("nhi_true"))
            and REPORT_FLOOR - 1e-9 <= float(rec["nhi_true"]) < REPORT_CEILING - 1e-9)


def _in_basis_support(rec):
    """The basis-support test P3's declared predicate names."""
    return (_is_num(rec.get("nhi_true"))
            and float(rec["nhi_true"]) >= ADOPTED_PAD_FLOOR - 1e-9
            and float(rec["nhi_true"]) < ADOPTED_BASIS_TOP - 1e-9)


def _p3(rec):
    """P3: an unmatched TRUTH row INSIDE the basis support.

    FIXED 2026-08-05 (referee M-C).  This used to be a bare
    ``not rec.get("matched")``, which does not implement the predicate the
    contract declares: ``classify_truth(dict(matched=False, nhi_true=18.0))``
    returned ``P3_INCOMPLETENESS`` for a truth row a full dex BELOW the 19.0
    basis floor.  That row has no basis bin, hence no completeness factor
    ``C[b,s]`` and no ``(1 - C)`` complement, so it cannot be the P3 the fold
    carries.  The support test is now executed, and ``nhi_true`` is REQUIRED.
    """
    _require_keys(rec, ("matched", "nhi_true"), "_p3/classify_truth")
    if bool(rec["matched"]):
        return False
    if not _is_num(rec["nhi_true"]):
        raise ContractViolation(
            f"classify_truth: unmatched truth row {rec!r} has a non-numeric "
            "nhi_true, so the basis-support test P3's predicate declares "
            "cannot be run. Fail-closed.")
    return _in_basis_support(rec)


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
        predicate_text=(
            "a truth row of truth_cut (SNR > 2, inside the spectral window) "
            f"with {ADOPTED_PAD_FLOOR} <= NHI_TRUE < {ADOPTED_BASIS_TOP} — "
            "i.e. INSIDE the adopted latent-basis support — that NO candidate "
            "claims. The support test is EXECUTED by _p3 (fixed 2026-08-05); "
            "an unmatched truth row outside the basis returns "
            f"'{TRUTH_OUT_OF_BASIS_SUPPORT}', not P3."),
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

#: The DECLARATION ORDER of ``POPULATIONS`` is pinned.  A referee minor
#: (2026-08-05): ``classify_candidate`` returns ``hits[0]``, so if two slots
#: ever claimed the same record the answer would be decided by tuple order.
#: The ``len(hits) != 1`` guard makes that unreachable, but the guard is the
#: load-bearing part and it must not be relaxed; the order is pinned so a
#: reorder is a visible diff rather than a silent re-ranking.
POPULATION_ORDER = ("P1_SCATTER_IN", "P2_IN_WINDOW", "P3_INCOMPLETENESS",
                    "P4_FOREST_FP", "P5_TRANSFER", "P6_RESIDUAL")
if tuple(p.pid for p in POPULATIONS) != POPULATION_ORDER:
    raise ContractViolation(
        f"POPULATIONS order {tuple(p.pid for p in POPULATIONS)} != the pinned "
        f"POPULATION_ORDER {POPULATION_ORDER}.")


def classify_candidate(rec) -> str:
    """Assign ONE population id to a candidate record. Fail-closed.

    ``rec`` MUST carry all three keys: ``is_TP`` (bool), ``nhi_true`` (a real
    finite float, or nan/None only when ``is_TP`` is False) and
    ``forest_attributable`` (bool).  Exactly one slot must claim it.

    Fail-closed hardening (2026-08-05): a missing key and a string ``nhi_true``
    both RAISE.  Before, ``classify_candidate({})`` returned ``P6_RESIDUAL``
    and ``nhi_true="20.5"`` returned ``P2_IN_WINDOW`` — the first invents the
    slot with no forward term, the second silently coerces.
    """
    _require_keys(rec, ("is_TP", "nhi_true", "forest_attributable"),
                  "classify_candidate")
    v = rec["nhi_true"]
    if bool(rec["is_TP"]) and not _is_num(v):
        raise ContractViolation(
            f"classify_candidate: record {rec!r} claims is_TP but its "
            f"nhi_true {v!r} is not a real finite number, so the P1/P2/P6 "
            "boundaries cannot be evaluated. Fail-closed.")
    if isinstance(v, (str, bytes)):
        raise ContractViolation(
            f"classify_candidate: nhi_true {v!r} is a string. The contract "
            "never coerces; pass a float.")
    hits = [p.pid for p in POPULATIONS
            if p.side == Side.CANDIDATE and p.predicate(rec)]
    if len(hits) != 1:
        raise ContractViolation(
            f"classify_candidate: record {rec!r} matched {len(hits)} candidate "
            f"populations {hits} — the contract requires EXACTLY one "
            "(no overlap, no gap).")
    return hits[0]


def classify_truth(rec) -> Optional[str]:
    """Classify a TRUTH row.  ``rec`` needs ``matched`` and ``nhi_true``.

    Returns
    -------
    ``None``
        the row is MATCHED — it is already counted on the CANDIDATE ledger and
        must not be counted twice.
    ``"P3_INCOMPLETENESS"``
        unmatched AND inside the basis support
        ``[ADOPTED_PAD_FLOOR, ADOPTED_BASIS_TOP)``.
    ``TRUTH_OUT_OF_BASIS_SUPPORT``
        unmatched but OUTSIDE the basis: no basis bin, hence no ``C[b,s]`` and
        no ``(1 - C)`` complement.  Counting it as P3 would put a system in an
        incompleteness slot the fold has no term for.
    """
    if _p3(rec):                       # also enforces the required keys
        return "P3_INCOMPLETENESS"
    return None if bool(rec["matched"]) else TRUTH_OUT_OF_BASIS_SUPPORT


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
             "N_sl_loa0 * (N_sl_loa0 / N_prod); the loa-0 Poisson exposure. "
             "It IS carried by the fold (forward.fold_mu_fp); the omission "
             "recorded here until 2026-08-05 is RESOLVED — see "
             "RESOLVED_BY_ID['FP_ELL_EFF_OMITTED']."),
    Quantity("fp_w_sightline_ratio", ParameterClass.FIXED_CALIBRATION_PRODUCT,
             SupportClass.MEASURED, "pack.fp_w_sightline_ratio (scalar)",
             "N_prod / N_sl_loa0. One-sided BAL veto — see "
             "KNOWN_CONTRADICTIONS."),
    Quantity("fp_E_alloc", ParameterClass.FIXED_CALIBRATION_PRODUCT,
             SupportClass.WEAKLY_MEASURED, "pack.fp_E_alloc (Kf, S)",
             "the FP's z-allocation, set to the PATHLENGTH shape "
             "dX[k,s]/sum_k dX[k,s]. It carries no FP z-information and the "
             "measured loa-0 FP z-shape differs across the observed floor — "
             "see KNOWN_CONTRADICTIONS["
             "'FP_Z_SHAPE_DIFFERS_ACROSS_THE_OBSERVED_FLOOR']."),
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
            "sum(T) - sum(FOUND_ON + FOUND_OFF + MISSED). *** THIS IS AN "
            "ALGEBRAIC TAUTOLOGY, NOT A TEST OF THE PHYSICS. *** "
            "T.C.rho + T.C.(1-rho) + T.(1-C) == T identically, for ANY C, ANY "
            "rho and ANY T. MEASURED 2026-08-05: injecting rho in {0.0, 0.8, "
            "1.0, U(0,1)} and molly_n_det in {0, 1, 99} leaves the residual at "
            "0.0 to +-2.3e-13 in all twelve combinations. It detects ONLY a "
            "shape / broadcast / dtype crash — which is worth having, because "
            "the (S,KK,B) -> (B,Kf,S) transpose is a real place to get an axis "
            "wrong — and it detects NOTHING about whether C or rho is right. "
            "The real per-slot content is in truth_ledger.value_guards, which "
            "bound FOUND_ON, FOUND_OFF and MISSED individually."),
        truth_ledger_value_guards=(
            "The NON-tautological half: 0 <= C <= 1, 0 <= rho <= 1, T >= 0 and "
            "finite, FOUND_ON <= T.C elementwise, FOUND_OFF >= 0, MISSED in "
            "[0, T], and an explicit index round-trip on the rho transpose "
            "(rho_bks[b,k,s] == rho[s, kz[k], b]) — the one failure mode the "
            "residual can see, made explicit instead of implicit."),
        candidate_ledger_residual=(
            "(P1 + P2 + P6_ABOVE_CEILING + P4) - N_obs, evaluated TRUTH-PINNED "
            "(f fixed so that dX.g.f.dN == the mock's own truth histogram, "
            "psi_c = 0, psi_k = 0, t = 0, lam_fp = fp_counts/ell_eff). ZERO "
            "free parameters. Any imbalance is a FINDING."),
        feasibility=(
            "Because 0 <= C <= 1 and 0 <= rho <= 1, sum_b FOUND_ON <= sum_b T. "
            "So N_obs - P4 <= sum_b T is a NECESSARY condition. Its violation "
            "is a COUNTING ARGUMENT: no parameter setting can close the model. "
            "Reported as efficiency_required = (N_obs - P4) / sum_b T. THIS IS "
            "THE ONLY BOUND THE CONTRACT ASSERTS. The efficiency evaluated at "
            "psi_c = psi_k_delta = 0 is a POINT, not a bound — see "
            "prior_cost_audit and RETRACTIONS[C3_CALIBRATION_POINT_IS_NOT_A_"
            "BOUND]."),
    ),
    truth_pinning=(
        "Pinning Lambda_intrinsic[b,k,s] := truth_counts_bks[b,k,s] bypasses "
        "dX, g and f together — they enter the fold only as their product. The "
        "pin is exact up to the mock's Poisson realisation."),
)


def basis_partition(ntrue_edges, lo=REPORT_FLOOR, hi=REPORT_CEILING):
    """Per-basis-bin fractional split into (below-floor, in-window, above-ceiling).

    Returns a dict of three (B,) arrays whose SUM is 1.0 in every bin — exactly,
    to the last bit, on the adopted basis (MEASURED max deviation 0.0), which is
    what forbids double counting across the reporting edges.  The individual
    fractions are NOT exact: the straddling bin's in-window share is
    0.5000000000000089, i.e. 8.9e-15 off the analytic 0.5.  Harmless, and
    recorded because the docstring used to claim "exactly 0.5" (referee minor,
    2026-08-05).

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
              "(1 - eta_c) . exp(t_K) . E[k,s], i.e. in the pack's own "
              "scalars mu_FP = fp_w . fp_ell_eff . (1 - fp_eta_c) . lam_fp . "
              "exp(t) . E, because lam_fp is defined by the loa-0 likelihood "
              "fp_counts ~ Poisson(fp_ell_eff . lam_fp) — the calibration "
              "side carries NO eta (loa-0 is HCD-free, nothing occludes)."),
    source=("CDDF_analysis/hbi/build_loa0_fp_product.py:35-39 — "
            "mu_FP = (N_prod/N_sl_loa0) . N_FP_loa0_total . (1 - eta_bar), "
            "ell_eff = N_sl_loa0 . (N_sl_loa0/N_prod)."),
    identity="fp_w . fp_ell_eff == N_sl_loa0 exactly (VERIFIED == 2255.0 on all "
             "three adopted packs).",
    eta=("host-occlusion survival, per observed bin (pack.fp_eta_c; RESTORED "
         "to the fold 2026-08-06, PI ruling 8). eta_DLA is FORCED to 0 "
         "(build_loa0_fp_product.py:DLA_ETA), so eta_c == 0 at and above "
         "N-hat 20.3; the [19.5, 20.3) bins carry eta_subdla = "
         "0.005756532459300326. CORRECTION: this entry previously claimed "
         "'the (1 - eta_bar) factor is 1 on the pack's N >= 19.5 grid' — "
         "WRONG (it read the DLA-band forcing as if it covered the whole "
         "grid); that claim rationalised the fold omitting the factor "
         "entirely (-85.01 counts on the adopted 2LPT-0 pack)."),
    implemented_at=(
        "CDDF_analysis/hbi_mcmc/forward.py:fold_mu_fp — ONE definition, "
        "`consts.fp_w * consts.fp_ell_eff * (1 - consts.fp_eta_c) * exp_t_k "
        "* lam_fp * fp_E`. "
        "``fold_mu`` calls it; ``forward_selftest.selftest`` calls it; "
        "``fold_mu_reference`` re-implements the same expression "
        "INDEPENDENTLY on purpose (it is the numpy oracle and must not share "
        "a helper); ``pack.synthetic_pack`` inverts it to place its FP mass. "
        "The contract AGREES with the code here — the disagreement recorded "
        "until 2026-08-05 is RESOLVED, see "
        "RESOLVED_BY_ID['FP_ELL_EFF_OMITTED']."),
    z_allocation=(
        "E[k,s] = pack.fp_E_alloc, which the pack schema constrains to "
        "sum_k E[k,s] == 1 on every POPULATED stratum (pack.py:545; MEASURED "
        "2026-08-05 on all three adopted packs the column sums are "
        "[0, 0, 1, 1, 1, 1, 1, 1] — the two zeros are the structurally empty "
        "SNR<=2 op-mask strata, which carry fp_counts == 0). The total "
        "therefore does NOT depend on the z-allocation, only on fp_w, "
        "fp_ell_eff and lam_fp — and neither does the ratified gate. See "
        "CONTRADICTION_BY_ID['FP_Z_SHAPE_DIFFERS_ACROSS_THE_OBSERVED_FLOOR']."),
)


def _fp_fold_total_through_forward(pack) -> float:
    """The FP total the COMMITTED fold actually produces on this pack.

    Calls ``forward.fold_mu_fp`` — the single site the FP term is defined at —
    with the pack's own scalars, ``log_t = 0`` and the pack's own
    ``fp_E_alloc``.  Nothing here re-types the expression: if the fold's
    expression changes, this number changes with it.  That is the whole point,
    and it is why the audit no longer hard-codes a reading of the source.
    """
    from CDDF_analysis.hbi_mcmc.forward import fold_mu_fp   # lazy: needs jax

    ell = float(pack.fp_ell_eff)
    fp_counts = np.asarray(pack.fp_counts, float)              # (C, S)
    kz = np.asarray(pack.kz_to_K, int)                         # (Kf,)
    E = np.asarray(pack.fp_E_alloc, float)                     # (Kf, S)
    if E.shape != (len(kz), fp_counts.shape[1]):
        raise ContractViolation(
            f"fp_E_alloc has shape {E.shape}, expected "
            f"{(len(kz), fp_counts.shape[1])} == (Kf, S).")
    # the exposure allocation must be a PROBABILITY over z on every stratum
    # that carries FP counts, or the fold's total is not fp_w.fp_ell_eff.lam_fp
    # and the comparison below would be testing the allocation, not the factor.
    col = E.sum(axis=0)
    populated = fp_counts.sum(axis=0) > 0
    bad = populated & (np.abs(col - 1.0) > 1e-9)
    if np.any(bad):
        raise ContractViolation(
            "fp_E_alloc: sum_k E[k,s] != 1 on stratum(s) "
            f"{np.flatnonzero(bad).tolist()} that carry loa-0 FP counts "
            f"(column sums {col[bad].tolist()}). The schema requires it "
            "(pack.py:545) and the FP normalisation is not checkable without "
            "it.")
    if getattr(pack, "fp_eta_c", None) is None:
        raise ContractViolation(
            "fp_eta_c: the pack does not carry the per-observed-bin "
            "host-occlusion vector (restoration 2026-08-06). Re-extract the "
            "pack or migrate it explicitly (pack.attach_fp_eta_bands).")
    consts = _FoldFPConsts(kz_to_K=kz, fp_w=float(pack.fp_w_sightline_ratio),
                           fp_ell_eff=ell, fp_E=E,
                           fp_eta_c=np.asarray(pack.fp_eta_c, float))
    log_t = np.zeros(int(kz.max()) + 1 if kz.size else 1)
    mu_fp = np.asarray(fold_mu_fp(log_t, fp_counts / ell, consts))
    return float(mu_fp.sum())


@dataclasses.dataclass(frozen=True)
class _FoldFPConsts:
    """The four fields ``forward.fold_mu_fp`` touches, and nothing else.

    Deliberately NOT ``forward.build_consts``: building the full consts needs
    the response fits and would make this guard depend on machinery it is not
    testing.  A duck-typed carrier keeps the guard about the FP expression.
    """
    kz_to_K: np.ndarray
    fp_w: float
    fp_ell_eff: float
    fp_E: np.ndarray
    fp_eta_c: np.ndarray   # (C,) host-occlusion survival (restored 2026-08-06)


def fp_normalisation_audit(pack) -> dict:
    """Compare the CONTRACT's FP normalisation with the one the fold APPLIES.

    🔴 2026-08-05, second correction.  Until this date the "implemented" total
    was hard-coded as ``fp_w * lam_tot`` from a READING of ``forward.py``'s
    source.  The reading was right when it was written and went silently STALE
    the moment the fold was repaired (7707c8e, 2b436df): the audit kept
    reporting a 13.59x disagreement against code that no longer had one, and
    ``assert_forward_fp_normalisation`` raised on correct code.  A number taken
    from a source reading goes stale without saying so; a number obtained by
    CALLING the code cannot.  ``mu_fp_total_as_folded`` is therefore measured
    through ``forward.fold_mu_fp``.

    ``mu_fp_total_if_ell_eff_omitted`` is kept as an explicitly labelled
    COUNTERFACTUAL — it is what the fold produced before the repair, and it is
    what the ledger's retracted "as implemented" column reported.  It is NOT a
    description of the committed code.

    Needs jax (through ``forward``); ``check_accounting_identity`` already does.
    """
    w = float(pack.fp_w_sightline_ratio)
    ell = float(pack.fp_ell_eff)
    if getattr(pack, "fp_eta_c", None) is None:
        raise ContractViolation(
            "fp_eta_c: the pack does not carry the per-observed-bin "
            "host-occlusion vector (restoration 2026-08-06). Re-extract the "
            "pack or migrate it explicitly (pack.attach_fp_eta_bands).")
    eta_c = np.asarray(pack.fp_eta_c, float)
    n_fp = float(np.asarray(pack.fp_counts, float).sum())
    lam_tot = n_fp / ell
    # the product's own definition (build_loa0_fp_product.py, restored
    # 2026-08-06): mu_FP = w * ell * sum_cs (1 - eta_c) * lam[c,s]
    #            == w * sum_c (1 - eta_c) * n0_row[c]
    n_fp_surv = float(((1.0 - eta_c)[:, None]
                       * np.asarray(pack.fp_counts, float)).sum())
    contract = w * n_fp_surv
    folded = _fp_fold_total_through_forward(pack)
    return dict(
        n_fp_loa0=n_fp, fp_w_sightline_ratio=w, fp_ell_eff=ell,
        lam_total_plugin=lam_tot,
        n_fp_eta_survived=n_fp_surv,
        mu_fp_total_as_folded=folded,
        mu_fp_total_per_contract=contract,
        ratio_contract_over_folded=(contract / folded if folded > 0 else np.inf),
        mu_fp_total_if_ell_eff_omitted=w * lam_tot * n_fp_surv / max(n_fp, 1e-300),
        mu_fp_total_if_eta_omitted=w * n_fp,
        n_sl_loa0_implied=w * ell,
        fold_site="CDDF_analysis/hbi_mcmc/forward.py:fold_mu_fp — mu_fp = "
                  "consts.fp_w * consts.fp_ell_eff * (1 - consts.fp_eta_c) "
                  "* exp_t_k * lam_fp * fp_E. "
                  "MEASURED by calling it, not by reading it.",
    )


def assert_forward_fp_normalisation(pack, *, rtol=1e-9):
    """FAIL LOUDLY if the COMMITTED fold's FP term is not the contract's.

    A check against the CODE, not against the pack: the pack's scalars are
    pushed through ``forward.fold_mu_fp`` and the total compared with
    ``fp_w . fp_ell_eff . (1 - eta_c) . lam_fp`` (the (1 - eta) host-occlusion
    survival restored 2026-08-06).  It PASSES on the repaired fold and it is
    the standing regression guard for
    ``RESOLVED_BY_ID['FP_ELL_EFF_OMITTED']``: dropping ``consts.fp_ell_eff``
    from the fold again makes the ratio equal ``fp_ell_eff`` exactly (13.59 on
    the adopted packs) and this raises.

    It was a PERMANENT FALSE ALARM between 7707c8e and this commit: it compared
    the contract against a hard-coded description of the pre-repair fold, so it
    raised on correct code.  A guard that fires on the fixed state is worse
    than no guard — it teaches its readers to skip it.
    """
    a = fp_normalisation_audit(pack)
    r = a["ratio_contract_over_folded"]
    if abs(r - 1.0) > rtol:
        regressed = abs(r - a["fp_ell_eff"]) <= 1e-6 * max(a["fp_ell_eff"], 1.0)
        raise ContractViolation(
            "FP NORMALISATION VIOLATION: the contract requires "
            "mu_FP = fp_w * fp_ell_eff * (1 - eta_c) * lam_fp * exp(t) * E "
            "((1 - eta) restored 2026-08-06). On this pack the "
            f"contract total is {a['mu_fp_total_per_contract']:.4f} and the "
            f"total the COMMITTED forward.fold_mu_fp produces is "
            f"{a['mu_fp_total_as_folded']:.4f} (measured ratio {r:.9f}). "
            + ("The ratio EQUALS fp_ell_eff = "
               f"{a['fp_ell_eff']:.6f}: the 2026-08-05 omission has been "
               "RE-INTRODUCED — see RESOLVED_BY_ID['FP_ELL_EFF_OMITTED'], "
               "fixed by 7707c8e and 2b436df. "
               if regressed else "")
            + "See FP_NORMALISATION.implemented_at.")
    return a


# ---------------------------------------------------------------------------
# 7. contract validation of an input pack
# ---------------------------------------------------------------------------
def _assert_finite(a, what):
    """FAIL CLOSED on NaN / inf.

    Every ``<``/``>`` comparison with NaN is False, so a NaN array walks
    through a range guard untouched.  This is the mechanism behind referee
    finding M-D and it is checked FIRST everywhere below.
    """
    a = np.asarray(a, float)
    bad = ~np.isfinite(a)
    if np.any(bad):
        raise ContractViolation(
            f"{what} contains {int(bad.sum())} non-finite entr{'y' if bad.sum()==1 else 'ies'} "
            f"of {a.size} (first at index "
            f"{tuple(int(x) for x in np.argwhere(bad)[0])}). NaN passes every "
            "range guard silently — the contract fails CLOSED on it.")
    return a


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
      5. ``truth_counts_bks`` is present AND finite AND non-negative — the
         accounting identity is not checkable without it;
      6. ``counts`` is RAW: finite (rule 6a) and non-negative (rule 6b);
      7. the FP scalars are positive and finite;
      8. the completeness counts are POSSIBLE: finite, non-negative, and
         ``0 <= molly_n_det <= molly_n_tot`` with ``molly_n_tot > 0``.

    Rules 5-8 were tightened on 2026-08-05 (referee M-D): the validator used to
    FAIL OPEN.  MEASURED before the fix, on the duck-typed pack:

      * ``counts`` all-NaN -> no raise; ``n_obs = nan``;
        ``feasible = bool(nan <= t_tot) = False`` — i.e. a NaN pack silently
        produced the very "INFEASIBLE" verdict this module exists to license.
      * ``molly_n_det = 200 > molly_n_tot = 100`` -> no guard at all;
        ``eta_hat = log(200.5 / -99.5) = nan``.
      * an all-NaN ``row_mass`` passed the ``rho < -1e-12 or rho > 1+1e-9``
        test, because every comparison with NaN is False.

    All four now raise ``ContractViolation``.
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
    T = np.asarray(pack.truth_counts_bks, float)
    _assert_finite(T, "pack.truth_counts_bks")
    if np.any(T < 0):
        raise ContractViolation(
            "pack.truth_counts_bks has negative entries: a truth histogram is "
            "a count.")

    # 6
    c = np.asarray(pack.counts, float)
    _assert_finite(c, "pack.counts")            # 6a — was FAIL-OPEN on NaN
    if np.any(c < 0):                           # 6b
        raise ContractViolation(
            "pack.counts contains negative entries: `counts` must be RAW "
            "op-passing detections. On this route the FP is FORWARD-MODELLED "
            "(P4), never subtracted; a subtracted array is a different "
            "estimand and cannot enter this ledger.")
    fpc = np.asarray(pack.fp_counts, float)
    _assert_finite(fpc, "pack.fp_counts")
    if np.any(fpc < 0):
        raise ContractViolation("pack.fp_counts has negative entries.")

    # 7
    for k in ("fp_ell_eff", "fp_w_sightline_ratio"):
        v = float(getattr(pack, k))
        if not (np.isfinite(v) and v > 0):
            raise ContractViolation(f"pack.{k} = {v}: must be finite positive.")

    # 8 — the completeness counts must describe a POSSIBLE experiment
    nd = np.asarray(pack.molly_n_det, float)
    nt = np.asarray(pack.molly_n_tot, float)
    _assert_finite(nd, "pack.molly_n_det")
    _assert_finite(nt, "pack.molly_n_tot")
    if nd.shape != nt.shape:
        raise ContractViolation(
            f"molly_n_det {nd.shape} and molly_n_tot {nt.shape} disagree.")
    if np.any(nd < 0) or np.any(nt <= 0):
        raise ContractViolation(
            "molly completeness counts must satisfy n_det >= 0 and n_tot > 0; "
            f"got n_det.min() = {nd.min()}, n_tot.min() = {nt.min()}.")
    over = nd > nt + 1e-9
    if np.any(over):
        i = np.argwhere(over)[0]
        raise ContractViolation(
            "IMPOSSIBLE COMPLETENESS: molly_n_det > molly_n_tot in "
            f"{int(over.sum())} cell(s); first at index {tuple(int(x) for x in i)} "
            f"with n_det = {nd[tuple(i)]} > n_tot = {nt[tuple(i)]}. More systems "
            "were found than exist. Unguarded this yields "
            "eta_hat = log(positive / negative) = nan and every downstream "
            "completeness becomes NaN silently (referee M-D, 2026-08-05).")

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


def _sigma_hat(n_det, n_tot):
    """Jeffreys completeness-logit WIDTH — the same formula the fold uses
    (forward.eta_hat_sigma_hat), inlined so this module stays jax-free."""
    d = np.asarray(n_det, float)
    t = np.asarray(n_tot, float)
    return np.sqrt(1.0 / (d + 0.5) + 1.0 / (t - d + 0.5))


# ---------------------------------------------------------------------------
# 7b. PRIOR COST — what replaces the retracted "sharper bound"
# ---------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _psi_c_lagrangian(lam, w, eta, sigma, n_grid=2001, n_refine=80):
    """Exact per-cell global minimiser of the separable Lagrangian

        h_j(psi) = (psi_j / sigma_j)^2 - lam * w_j * sigmoid(eta_j + psi_j)

    over ``psi_j >= 0``.  Two facts make this exact rather than a local search:
    for ``w_j >= 0`` the minimiser is non-negative (any ``psi < 0`` raises BOTH
    terms above their value at 0), and ``h_j(psi*) <= h_j(0)`` forces
    ``|psi*_j| <= sigma_j * sqrt(lam * w_j)``.  So the search interval is known
    in closed form and a dense sweep plus golden-section refinement finds the
    GLOBAL minimum, which is what makes the dual bound below valid.

    Returns ``(psi, chi2, signal)``.
    """
    w = np.maximum(np.asarray(w, float), 0.0)
    eta = np.asarray(eta, float)
    sigma = np.asarray(sigma, float)
    hi = sigma * np.sqrt(max(float(lam), 0.0) * w)
    u = np.linspace(0.0, 1.0, n_grid)[None, :]
    psi_g = hi[:, None] * u
    h_g = (psi_g / sigma[:, None]) ** 2 - lam * w[:, None] * _sigmoid(eta[:, None] + psi_g)
    j = np.argmin(h_g, axis=1)
    step = hi / (n_grid - 1)
    idx = np.arange(len(w))
    a = np.maximum(psi_g[idx, j] - step, 0.0)
    b = np.minimum(psi_g[idx, j] + step, hi)

    def h(p):
        return (p / sigma) ** 2 - lam * w * _sigmoid(eta + p)

    gr = 0.5 * (np.sqrt(5.0) - 1.0)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    hc, hd = h(c), h(d)
    for _ in range(n_refine):
        m = hc < hd
        b = np.where(m, d, b)
        a = np.where(m, a, c)
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        hc, hd = h(c), h(d)
    psi = 0.5 * (a + b)
    psi = np.where(h(psi) <= h(np.zeros_like(psi)), psi, 0.0)
    return psi, float(np.sum((psi / sigma) ** 2)), float(np.sum(w * _sigmoid(eta + psi)))


def min_prior_chi2_psi_c(w_cell, eta, sigma, target, *, n_lam=200) -> dict:
    """Minimum completeness-prior chi^2 needed to reach ``target`` counts.

    Solves  ``min sum_j (psi_j/sigma_j)^2   s.t.  sum_j w_j sigmoid(eta_j+psi_j)
    >= target``, where ``psi_c ~ Normal(0, sigma_hat)`` (model_a.py:206-207) is
    the DECLARED prior, ``eta`` is the Jeffreys point surface, and ``w_j`` is
    the truth-weighted, row-mass-weighted exposure of molly cell ``j``.

    Reports BOTH a primal ``upper_bound`` (an explicit witness psi that attains
    the target — always valid) and a Lagrangian ``lower_bound``
    ``max_lam [ lam*target + min_psi(chi2 - lam*S) ]`` (weak duality — always
    valid, and this is what licenses a LARGE cost being called large).  When
    the two agree the value is the minimum.

    ``upper_bound = lower_bound = inf`` when ``target`` exceeds ``sum_j w_j``,
    the SUPREMUM as every completeness goes to 1: no finite psi_c reaches it.
    """
    w = np.maximum(np.asarray(w_cell, float).ravel(), 0.0)
    eta = np.asarray(eta, float).ravel()
    sigma = np.asarray(sigma, float).ravel()
    _assert_finite(w, "prior-cost cell weights")
    _assert_finite(eta, "prior-cost eta_hat")
    _assert_finite(sigma, "prior-cost sigma_hat")
    sup = float(np.sum(w))
    s0 = float(np.sum(w * _sigmoid(eta)))
    target = float(target)
    out = dict(target_counts=target, signal_at_calibration=s0,
               supremum_as_C_to_one=sup)
    if target <= s0:
        out.update(upper_bound=0.0, lower_bound=0.0, achieved=s0,
                   note="the calibration point already reaches the target")
        return out
    if target > sup:
        out.update(upper_bound=np.inf, lower_bound=np.inf, achieved=None,
                   note="target EXCEEDS the supremum C -> 1: unreachable at "
                        "ANY finite prior cost")
        return out
    lo, hi = 0.0, 1.0
    while _psi_c_lagrangian(hi, w, eta, sigma)[2] < target and hi < 1e14:
        hi *= 4.0
    best_lower, best = 0.0, None
    for _ in range(int(n_lam)):
        mid = 0.5 * (lo + hi)
        _psi, chi2, s = _psi_c_lagrangian(mid, w, eta, sigma)
        best_lower = max(best_lower, mid * target + (chi2 - mid * s))
        if s >= target:
            hi, best = mid, (chi2, s)
        else:
            lo = mid
        if hi - lo < 1e-12 * max(hi, 1.0):
            break
    out.update(upper_bound=(best[0] if best else np.inf),
               achieved=(best[1] if best else None),
               lower_bound=float(best_lower))
    return out


def prior_cost_audit(pack, *, T, C_bs, rho_bks, b_to_cell, target,
                     mu_fp_per_contract, sig_at_calibration,
                     psi_k_signal=None, fitcov_sd=None) -> dict:
    """THE REPLACEMENT FOR THE RETRACTED FEASIBILITY VERDICT (referee C3).

    Reports how much PRIOR COST — Mahalanobis distance in the model's own
    declared nuisance space — it takes to reach the observed counts, and lets
    the reader judge.  It emits NO boolean and NO verdict.

    What is a bound and what is not
    -------------------------------
    * ``trivial_bound_efficiency = 1.0`` is the ONLY genuine bound: ``C <= 1``
      and ``rho <= 1``, so ``sum_b FOUND_ON <= sum_b T`` whatever the
      nuisances do.  ``efficiency_required > 1`` is a real counting argument.
    * ``sup_efficiency_psi_c_only`` (``C -> 1``, ``psi_c -> +inf``) and
      ``sup_efficiency_psi_k_only`` (``rho -> 1``) ARE suprema over their own
      one nuisance block, attained only in the limit.  A required efficiency
      ABOVE one of them means that block alone cannot close the counts at any
      finite prior cost — that is the ``inf`` case below.
    * ``efficiency_at_calibration`` (``psi_c = psi_k_delta = 0``) is a POINT,
      NOT a bound.  It used to be reported as "the SHARPER bound" and drove a
      ``feasible_at_calibration_per_contract`` boolean.  Both are RETRACTED:
      ``psi_c ~ Normal(0, sigma_hat)`` and ``psi_k_delta ~ Normal(0,
      fitcov_sd)`` are sample sites with UNBOUNDED support.

    Directions costed
    -----------------
    ``psi_c``      exact (primal witness + dual lower bound), see
                   ``min_prior_chi2_psi_c``.
    ``psi_k_delta`` a WITNESS only — the one-parameter uniform shift of the
                   order-0 sigma block through the committed ``build_K``.  Any
                   witness is an UPPER bound on the block's minimum; it is not
                   the minimum.  Requires ``psi_k_signal`` (a callable
                   ``alpha -> signal``), so it is skipped on the injected
                   row-mass path.
    ``fp_total``   the loa-0 Poisson width of P4's own total: sd of ``N_FP``
                   scaled by ``fp_w``.  This is likelihood, not prior, and is
                   labelled as such.
    ``transfer_t`` an EXACT witness: a UNIFORM shift ``t_K = delta`` scales the
                   whole FP term by ``exp(delta)`` regardless of the per-K
                   split, at cost ``delta^2 * sum_K 1/t_sigma_K^2``.

    M-F: ``fitcov_sd`` is a DOCUMENTED GUESS on every pack extracted so far —
    ``pack.resp_fitcov_diag`` is absent from all six, so ``build_consts`` falls
    back to ``_DEFAULT_FITCOV_DIAG = (0.02^2, 0.10^2)`` (forward.py:218). The
    psi_k witness cost scales as 1/fitcov_sd^2 and is reported WITH that
    provenance attached, never bare.
    """
    T = np.asarray(T, float)
    C_bs = np.asarray(C_bs, float)
    rho_bks = np.asarray(rho_bks, float)
    b2c = np.asarray(b_to_cell, int)
    eta = _eta_hat(pack.molly_n_det, pack.molly_n_tot)
    sg = _sigma_hat(pack.molly_n_det, pack.molly_n_tot)
    t_tot = float(T.sum())
    target = float(target)

    sup_c = float((T * rho_bks).sum())              # C -> 1
    sup_k = float((T * C_bs.T[:, None, :]).sum())   # rho -> 1

    # molly-cell exposure w_j = sum over the basis bins mapped to cell j
    W_bs = (T * rho_bks).sum(axis=1)                # (B, S)
    w_cell = np.zeros_like(eta, dtype=float)        # (S, M)
    for b, m in enumerate(b2c):
        w_cell[:, int(m)] += W_bs[b, :]
    psi_c = min_prior_chi2_psi_c(w_cell, eta, sg, target)
    n_psi_c = int(eta.size)

    out = dict(
        trivial_bound_efficiency=1.0,
        trivial_bound_is_the_only_bound=True,
        efficiency_required=target / t_tot if t_tot else np.nan,
        efficiency_at_calibration=(sig_at_calibration / t_tot if t_tot else np.nan),
        efficiency_at_calibration_is_a_bound=False,
        sup_efficiency_psi_c_only=sup_c / t_tot if t_tot else np.nan,
        sup_efficiency_psi_k_only=sup_k / t_tot if t_tot else np.nan,
        gap_counts=target - float(sig_at_calibration),
        psi_c=dict(
            n_free=n_psi_c,
            min_prior_chi2_upper_bound=psi_c["upper_bound"],
            min_prior_chi2_lower_bound=psi_c["lower_bound"],
            min_mahalanobis_sigma=(np.sqrt(psi_c["upper_bound"])
                                   if np.isfinite(psi_c["upper_bound"]) else np.inf),
            supremum_counts=psi_c["supremum_as_C_to_one"],
            detail=psi_c,
            prior="psi_c ~ Normal(0, sigma_hat), model_a.py:206-207 — "
                  "UNBOUNDED support",
        ),
    )

    # --- psi_k_delta witness (needs the jax kernel) --------------------------
    if psi_k_signal is not None:
        alphas = np.linspace(0.0, -4.0, 81)
        sigs = np.array([float(psi_k_signal(a)) for a in alphas])
        hit = np.flatnonzero(sigs >= target)
        if len(hit):
            j = int(hit[0])
            a_lo, a_hi = (alphas[j - 1], alphas[j]) if j else (0.0, 0.0)
            for _ in range(60):                     # bisect on the bracket
                if j == 0:
                    break
                mid = 0.5 * (a_lo + a_hi)
                if psi_k_signal(mid) >= target:
                    a_hi = mid
                else:
                    a_lo = mid
            alpha = a_hi
            n_free = int(np.asarray(fitcov_sd)[1].size) if fitcov_sd is not None else 9
            out["psi_k_delta"] = dict(
                witness_alpha_in_prior_sd=float(alpha),
                n_free_in_the_shifted_block=n_free,
                witness_prior_chi2=float(n_free * alpha ** 2),
                witness_mahalanobis_sigma=float(np.sqrt(n_free) * abs(alpha)),
                is_a_witness_not_the_minimum=True,
                direction="uniform shift of psi_k_delta[1] (the order-0 sigma "
                          "perturbation) across all (SR, ZR) response cells",
            )
        else:
            out["psi_k_delta"] = dict(
                witness_alpha_in_prior_sd=None,
                note="no uniform psi_k_delta[1] shift within 4 prior sd "
                     "reaches the target; the block's minimum is NOT bounded "
                     "by this witness.",
                signal_at_minus_4_sd=float(sigs[-1]),
                is_a_witness_not_the_minimum=True)
        out["psi_k_delta"]["fitcov_sd_provenance"] = _FITCOV_PROVENANCE(pack)

    # --- FP total: loa-0 Poisson width (likelihood, not prior) ---------------
    n_fp = float(np.asarray(pack.fp_counts, float).sum())
    w_sl = float(pack.fp_w_sightline_ratio)
    gap = target - float(sig_at_calibration)
    sd_counts = w_sl * np.sqrt(n_fp)
    out["fp_total_poisson"] = dict(
        n_fp_loa0=n_fp, mu_fp_per_contract=float(mu_fp_per_contract),
        one_sd_counts=sd_counts,
        gap_in_sd=(gap / sd_counts if sd_counts > 0 else np.inf),
        kind="LIKELIHOOD width (fp_counts ~ Poisson(ell_eff . lam_fp)), not a "
             "prior; the FP total is a free parameter with a single-Jeffreys "
             "Gamma(1/2, eps) prior (model_a.py:214).",
    )

    # --- transfer t: exact uniform-shift witness -----------------------------
    ts = np.asarray(pack.t_sigma, float)
    tr = dict(t_sigma=ts.tolist(), max_t_sigma=float(ts.max()))
    if 0 < gap < float(mu_fp_per_contract):
        delta = float(np.log(1.0 - gap / float(mu_fp_per_contract)))
        chi2 = float(delta ** 2 * np.sum(1.0 / ts ** 2))
        tr.update(uniform_shift_delta=delta, witness_prior_chi2=chi2,
                  witness_mahalanobis_sigma=float(np.sqrt(chi2)),
                  is_a_witness_not_the_minimum=True,
                  note="a uniform t_K = delta scales the WHOLE FP term by "
                       "exp(delta) exactly, independently of the per-coarse-z "
                       "split, so this witness needs no fp_E.")
    else:
        tr.update(uniform_shift_delta=None,
                  note="no uniform transfer shift closes the gap (gap <= 0 or "
                       "gap >= the whole FP total).")
    out["transfer_t"] = tr

    cands = [out["psi_c"]["min_mahalanobis_sigma"]]
    if "psi_k_delta" in out and out["psi_k_delta"].get("witness_mahalanobis_sigma"):
        cands.append(out["psi_k_delta"]["witness_mahalanobis_sigma"])
    if tr.get("witness_mahalanobis_sigma"):
        cands.append(tr["witness_mahalanobis_sigma"])
    if np.isfinite(out["fp_total_poisson"]["gap_in_sd"]):
        cands.append(abs(out["fp_total_poisson"]["gap_in_sd"]))
    out["cheapest_declared_direction_sigma"] = float(np.min(cands))
    out["reading"] = (
        "This is a COST, not a verdict. A cheapest direction of a fraction of "
        "a sigma means the declared nuisances close the counts comfortably and "
        "no infeasibility claim follows. A cost of order 1e2 sigma (prior chi2 "
        "~1e4) means closing requires a calibration excursion the stated "
        "priors do not contemplate. An INFINITE psi_c cost means that block "
        "cannot close it even in the C -> 1 limit.")
    return out


def _FITCOV_PROVENANCE(pack) -> dict:
    """M-F: where ``fitcov_sd`` came from.  Never report a psi_k cost without
    it — on every pack extracted so far it is a documented GUESS."""
    have = getattr(pack, "resp_fitcov_diag", None) is not None
    return dict(
        pack_carries_resp_fitcov_diag=bool(have),
        fallback="forward._DEFAULT_FITCOV_DIAG = (0.02^2, 0.10^2) "
                 "(forward.py:218)",
        status=("MEASURED from the pack" if have else
                "UNCALIBRATED HARD-CODED FALLBACK. MEASURED 2026-08-05: "
                "resp_fitcov_diag is absent from all six extracted packs "
                "(three adopted window-study, three v1.1), so build_consts "
                "uses the fallback. Every psi_k_delta prior cost below scales "
                "as 1/fitcov_sd^2 and is therefore a documented GUESS."),
    )


#: MEASURED per-candidate reference partition of the on-grid detections.  This
#: is what the hostless-census comparison reports against: the floor-17.2
#: `unmatched` class.  CORRECTION (Phase-A adversarial review, frozen verdict,
#: review/phaseA-adversarial-2026-08-05 @ a11dae0): this class is NOT a
#: physical forest-FP ceiling — ~92% of it is genuine sub-floor detections —
#: so "mu_FP cannot exceed it" was an estimand error, not a physical bound.
#: Evidence: review_phaseA/fp_normalization/findings.md.
_FP_CEILING_BUNDLE = ("truth floor 17.2, lya_only, op mask, on the pack grid "
                      "(N_hat in [19.5,22.4), z in [2.0,3.5))")
_FP_CEILING_MEASURED_ON = (
    "2026-08-05, load_and_cut_catalog(truth_nhi_floor=17.2, "
    "host_truth_floor=17.2) + _snap_off_molly_edges, 11 s per mock")
_FP_CEILING_NOTE = (
    "the four is_TP slots plus `unmatched` sum to the on-grid total EXACTLY. "
    "CORRECTION (Phase-A, 2026-08-06): `unmatched` is the floor-17.2 HOSTLESS "
    "class, not a forest-FP supply — ~92% of it is genuine sub-floor "
    "detections, and after chance-coincidence correction mu_FP/supply = 1.002 "
    "on 2LPT-0. It also holds blends, second candidates on an already-claimed "
    "truth row, and matches beyond dz_rel (P6 sub-slot (c)). "
    "The 19.5-floor DETECTION bundle the pack itself uses has (on 2LPT-0) "
    "88071 on-grid rows with 24181 unmatched, but 4070+ of those are genuine "
    "absorbers the 19.5-floored truth table hid — see "
    "TRUTH_FLOOR_ASYMMETRY_IN_is_TP. The 17.2-floor `unmatched` is the "
    "defensible comparator, as a CENSUS, not as a ceiling. Evidence: "
    "review_phaseA/fp_normalization/findings.md.")

#: 🔴 RE-MEASURED 2026-08-05 on ALL THREE mocks.  The earlier table carried
#: 2lpt0 only and ``fp_ceiling_audit`` returned "NOT MEASURED" for the other
#: two, i.e. the check was UNAVAILABLE exactly where the excess is largest.
#: The mu_fp_per_contract / excess / excess_frac columns carry the
#: (1-eta)-restored FP fold (2026-08-06), re-derived by running
#: ``fp_normalisation_audit`` on each adopted pack.
FP_CEILING_MEASURED = {
    "2lpt0": dict(
        bundle=_FP_CEILING_BUNDLE, measured=_FP_CEILING_MEASURED_ON,
        n_on_grid=88053,
        P1_true_19p0_to_19p7=15438, P2_true_19p7_to_21p6=55058,
        P6_true_above_21p6=497, P6_true_below_19p0=3200,
        unmatched=13860,
        # (1-eta) restoration 2026-08-06: was 14767.961419068737; x(1-0.005756532459300326) on the FP term
        mu_fp_per_contract=14682.949169806607,
        # (1-eta) restoration 2026-08-06: was 907.9614190687371 / 0.06551; x(1-0.005756532459300326) on the FP term
        excess=822.9491698066067, excess_frac=0.05938,
        note=_FP_CEILING_NOTE,
    ),
    "london0": dict(
        bundle=_FP_CEILING_BUNDLE, measured=_FP_CEILING_MEASURED_ON,
        n_on_grid=87831,
        P1_true_19p0_to_19p7=15834, P2_true_19p7_to_21p6=59186,
        P6_true_above_21p6=602, P6_true_below_19p0=2611,
        unmatched=9598,
        # (1-eta) restoration 2026-08-06: was 14716.376940133037; x(1-0.005756532459300326) on the FP term
        mu_fp_per_contract=14631.66163859828,
        # (1-eta) restoration 2026-08-06: was 5118.376940133037 / 0.53328; x(1-0.005756532459300326) on the FP term
        excess=5033.66163859828, excess_frac=0.52445,
        note=_FP_CEILING_NOTE,
    ),
    "saclay0": dict(
        bundle=_FP_CEILING_BUNDLE, measured=_FP_CEILING_MEASURED_ON,
        n_on_grid=86745,
        P1_true_19p0_to_19p7=15733, P2_true_19p7_to_21p6=57213,
        P6_true_above_21p6=539, P6_true_below_19p0=2668,
        unmatched=10592,
        # (1-eta) restoration 2026-08-06: was 14707.062527716187; x(1-0.005756532459300326) on the FP term
        mu_fp_per_contract=14622.400844898844,
        # (1-eta) restoration 2026-08-06: was 4115.062527716187 / 0.38851; x(1-0.005756532459300326) on the FP term
        excess=4030.4008448988443, excess_frac=0.38051,
        note=_FP_CEILING_NOTE,
    ),
}


def fp_ceiling_audit(pack, *, mu_fp_per_contract, n_unmatched_on_grid=None,
                     mock=None) -> dict:
    """ESTIMAND COMPARISON (referee M-B, corrected): mu_FP vs the floor-17.2
    hostless census.

    🔴 CORRECTION (Phase-A adversarial review, frozen verdict,
    review/phaseA-adversarial-2026-08-05 @ a11dae0).
    PREVIOUSLY CLAIMED: "a forest FP is an on-grid candidate with no genuine
    absorber, so mu_FP cannot exceed the mock's unmatched on-grid count" — a
    physical forest-FP CEILING, reported as "VIOLATED".
    WHY WRONG: the comparator is the floor-17.2 HOSTLESS class, ~92% of which
    is genuine sub-floor detections, not forest FPs; after chance-coincidence
    correction mu_FP/supply = 1.002 on 2LPT-0 (the calibration twin), so the
    on-twin excess is an ESTIMAND ARTIFACT, not a physical violation.
    Cross-mock, the excess reflects the unresolved transport systematic
    (Layer C), which is a different object again.
    REPLACED BY: this audit still runs the SAME numeric comparison (that is
    its job — it guards that the comparison is REPORTED), but the verdict
    string is "mu_fp_exceeds_hostless_census", never a physical-ceiling claim.
    EVIDENCE: review_phaseA/fp_normalization/findings.md.

    RE-MEASURED 2026-08-06 with the (1-eta)-restored fold:

        mock      on-grid   unmatched     mu_FP     excess    excess/census
        2lpt0       88053       13860   14682.95    +822.95      +5.94%
        london0     87831        9598   14631.66   +5033.66     +52.45%
        saclay0     86745       10592   14622.40   +4030.40     +38.05%

    An earlier version of this table held 2LPT-0 alone, so the check returned
    "NOT MEASURED" on london0 and saclay0 — i.e. it was UNAVAILABLE exactly
    where the excess is 5x larger.  Every partition sums to its on-grid
    total exactly; see FP_CEILING_MEASURED for the four is_TP slots.
    """
    if mock is None:
        prov = getattr(pack, "provenance", None)
        if isinstance(prov, dict):
            mock = prov.get("mock")
    ref = FP_CEILING_MEASURED.get(mock)
    if n_unmatched_on_grid is None and ref is not None:
        n_unmatched_on_grid = ref["unmatched"]
    mu = float(mu_fp_per_contract)
    if n_unmatched_on_grid is None:
        return dict(mock=mock, mu_fp_per_contract=mu, ceiling=None,
                    exceeds_ceiling=None,
                    status="NOT MEASURED for this mock — the per-candidate "
                           "17.2-floor bundle has not been run. The check is "
                           "UNAVAILABLE, not passed.")
    ceil_ = float(n_unmatched_on_grid)
    # Phase-A correction 2026-08-06: the numeric comparison is UNCHANGED; only
    # the verdict language moved from a physical-ceiling claim ("VIOLATED")
    # to the estimand statement. See the docstring's correction block.
    return dict(mock=mock, mu_fp_per_contract=mu, ceiling=ceil_,
                excess=mu - ceil_, ratio=mu / ceil_ if ceil_ else np.inf,
                exceeds_ceiling=bool(mu > ceil_),
                reference=ref,
                status=("mu_fp_exceeds_hostless_census: the contract's mu_FP "
                        "exceeds the mock's floor-17.2 hostless-class census "
                        "(unmatched on-grid candidates). NOT a physical "
                        "forest-FP ceiling: ~92% of that class is genuine "
                        "sub-floor detections; on the calibration twin the "
                        "excess is resolved by chance-coincidence correction "
                        "(mu_FP/supply = 1.002), and cross-mock it reflects "
                        "the unresolved transport systematic (Layer C). See "
                        "review_phaseA/fp_normalization/findings.md."
                        if mu > ceil_ else
                        "mu_fp_within_hostless_census"))


def _truth_ledger_value_guards(T, C_bs, rho, rho_bks, kz,
                               found_on, found_off, missed):
    """The NON-tautological half of the truth ledger (referee M-A).

    ``T.C.rho + T.C.(1-rho) + T.(1-C) == T`` for ANY C, rho and T, so the
    residual proves nothing about the values.  These guards bound the three
    slots INDIVIDUALLY, and pin the ``(S,KK,B) -> (B,Kf,S)`` transpose with an
    explicit index round-trip — the one real failure mode the residual can see
    and the reason the residual is still worth computing.
    """
    _assert_finite(rho, "row mass rho")            # BEFORE any range test
    if np.any(rho < -1e-12) or np.any(rho > 1.0 + 1e-9):
        raise ContractViolation(
            f"row mass outside [0, 1] (min {rho.min()}, max {rho.max()}): the "
            "kernel is a probability distribution over N-hat and its mass on "
            "the observed grid cannot exceed 1. The trivial efficiency bound "
            "depends on this.")
    _assert_finite(C_bs, "completeness C[b,s]")
    if np.any(C_bs < -1e-12) or np.any(C_bs > 1.0 + 1e-12):
        raise ContractViolation(
            f"completeness outside [0, 1] (min {C_bs.min()}, max {C_bs.max()}).")
    _assert_finite(T, "truth_counts_bks")
    if np.any(T < 0):
        raise ContractViolation("truth_counts_bks has negative entries.")

    if rho_bks.shape != T.shape:
        raise ContractViolation(
            f"rho aligned to {rho_bks.shape} but T is {T.shape}.")
    # explicit index round-trip on the transpose: rho_bks[b,k,s] == rho[s,kz[k],b]
    B, Kf, S = T.shape
    bb = np.arange(B)[:, None, None]
    kk = np.arange(Kf)[None, :, None]
    ss = np.arange(S)[None, None, :]
    want = rho[ss, np.asarray(kz, int)[kk], bb]
    if not np.array_equal(rho_bks, want):
        raise ContractViolation(
            "rho AXIS MISALIGNMENT: rho_bks[b,k,s] != rho[s, kz[k], b]. The "
            "truth-ledger residual is identically zero under any such "
            "permutation, so this check — not the residual — is what catches "
            "it.")

    TC = T * C_bs.T[:, None, :]
    tol = 1e-9 * max(1.0, float(T.sum()))
    checks = {
        "found_on_non_negative": bool(np.all(found_on >= -1e-12)),
        "found_on_le_T_times_C": bool(np.all(found_on <= TC + 1e-9)),
        "found_off_non_negative": bool(np.all(found_off >= -1e-12)),
        "found_on_plus_found_off_eq_T_times_C":
            bool(np.abs((found_on + found_off).sum() - TC.sum()) <= tol),
        "missed_non_negative": bool(np.all(missed >= -1e-12)),
        "missed_le_T": bool(np.all(missed <= T + 1e-9)),
    }
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise ContractViolation(
            f"truth-ledger VALUE guards failed: {bad}. These bound FOUND_ON, "
            "FOUND_OFF and MISSED individually; the additive residual cannot "
            "see any of them.")
    checks["n_cells_checked"] = int(T.size)
    checks["C_min"] = float(C_bs.min())
    checks["C_max"] = float(C_bs.max())
    checks["rho_min"] = float(rho.min())
    checks["rho_max"] = float(rho.max())
    checks["note"] = ("the additive residual is a TAUTOLOGY (see "
                      "ACCOUNTING_IDENTITY.checkable_residuals); THESE are the "
                      "checks with content.")
    return checks


def check_accounting_identity(pack, *, resp_clamp="both",
                              validate=True, row_mass=None, b_to_cell=None,
                              n_unmatched_on_grid=None, strict=False,
                              **validate_kw) -> dict:
    """Evaluate the accounting identity on a REAL pack and return the residuals.

    TRUTH-PINNED and parameter-free: Lambda_intrinsic := truth_counts_bks,
    C := sigmoid(eta_hat) (psi_c = 0), psi_k_delta = 0, t = 0,
    lam_fp := fp_counts / fp_ell_eff.

    ``row_mass`` (S, KK, B) may be injected so the ledger is unit-testable
    without jax; by default it is computed through the committed
    ``forward.build_K``.  ``strict=True`` turns any entry of the returned
    ``flags`` list into a ``ContractViolation``.

    Returns a JSON-serializable dict.  WHAT A REFEREE SHOULD READ:

      * ``flags`` — empty, or a list of loud structural problems.
      * ``truth_ledger.value_guards`` — the checks with CONTENT.
        ``truth_ledger.residual`` is an algebraic tautology and is reported
        only because it catches a shape / axis crash.
      * ``candidate_ledger.residual_per_contract`` — a FINDING, never tuned.
      * ``prior_cost`` — how many sigma of the model's own declared nuisances
        it takes to close the counts.  NOT a feasibility verdict; the module
        no longer emits one (referee C3).
      * ``fp_ceiling`` — the hostless-census comparison: mu_FP vs the mock's
        floor-17.2 unmatched class. An ESTIMAND comparison, not a physical
        forest-FP ceiling (Phase-A 2026-08-06 — see ``fp_ceiling_audit``).
    """
    geom = validate_pack_against_contract(pack, **validate_kw) if validate else {}

    consts = None
    if row_mass is None:
        rho, consts = _row_mass(pack, resp_clamp=resp_clamp)       # (S,KK,B)
        b2c = np.asarray(consts.b_to_cell)
    else:
        rho = np.asarray(row_mass, float)
        b2c = np.asarray(_b_to_cell(pack) if b_to_cell is None else b_to_cell)
    eta = _eta_hat(pack.molly_n_det, pack.molly_n_tot)
    _assert_finite(eta, "eta_hat (from molly_n_det / molly_n_tot)")
    C_cells = 1.0 / (1.0 + np.exp(-eta))
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
    value_guards = _truth_ledger_value_guards(
        T, C_bs, rho, rho_bks, kz, found_on, found_off, missed)

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

    led_folded = _ledger(fpa["mu_fp_total_as_folded"])
    led_contract = _ledger(fpa["mu_fp_total_per_contract"])
    # the pre-repair counterfactual, kept ONLY so the measured before/after in
    # RESOLVED_BY_ID['FP_ELL_EFF_OMITTED'] stays reproducible from this routine
    led_if_omitted = _ledger(fpa["mu_fp_total_if_ell_eff_omitted"])

    t_tot = float(T.sum())
    p6_unsup = n_obs - led_contract["predicted_total"]

    # --- prior cost (replaces the retracted feasibility verdict) -------------
    psi_k_signal = None
    if consts is not None:
        import jax.numpy as jnp
        from CDDF_analysis.hbi_mcmc.forward import build_K
        fitsd = np.asarray(consts.fitcov_sd)

        def _psi_k_signal(alpha, _c=consts, _f=fitsd):
            """signal(alpha) with psi_k_delta[1] = alpha * fitcov_sd[1] on every
            (SR, ZR) cell, through the COMMITTED jax build_K."""
            p = np.zeros((2, _c.n_sr, _c.n_zr))
            p[1] = float(alpha) * _f[1]
            r = np.asarray(build_K(jnp.asarray(p), _c)).sum(axis=2)
            rb = np.transpose(r[:, kz, :], (2, 1, 0))
            return float((T * C_b_s * rb).sum())

        psi_k_signal = _psi_k_signal

    prior_cost = prior_cost_audit(
        pack, T=T, C_bs=C_bs, rho_bks=rho_bks, b_to_cell=b2c,
        target=n_obs - fpa["mu_fp_total_per_contract"],
        mu_fp_per_contract=fpa["mu_fp_total_per_contract"],
        sig_at_calibration=sig, psi_k_signal=psi_k_signal,
        fitcov_sd=(consts.fitcov_sd if consts is not None else None))
    prior_cost["fitcov_sd_provenance"] = _FITCOV_PROVENANCE(pack)

    fp_ceil = fp_ceiling_audit(
        pack, mu_fp_per_contract=fpa["mu_fp_total_per_contract"],
        n_unmatched_on_grid=n_unmatched_on_grid)

    # --- loud structural flags ----------------------------------------------
    flags = []
    if p6_unsup < 0:
        flags.append(dict(
            id="NEGATIVE_IMPLIED_P6_UNSUPPORTED",
            value=float(p6_unsup),
            detail="P6_unsupported_implied_per_contract is the number of "
                   "on-grid candidates left over for P6's unsupported "
                   "sub-slots (genuine absorbers below the basis floor, "
                   "blends, second candidates). It is a COUNT and cannot be "
                   "negative. A negative value means the modelled slots "
                   "OVER-predict the observed total, i.e. the ledger closes "
                   "only by the modelled terms borrowing counts that do not "
                   "exist. MEASURED 2026-08-05 on the adopted 0.2-dex packs: "
                   "2lpt0 +882.30, london0 -3287.42, saclay0 -2079.08. It was "
                   "previously emitted with no comment at all (referee M-B)."))
    if fp_ceil.get("exceeds_ceiling"):
        # Phase-A correction 2026-08-06: same trigger condition and value;
        # the id/detail no longer assert a physical forest-FP-supply ceiling
        # (rejected interpretation — see fp_ceiling_audit's docstring).
        flags.append(dict(
            id="MU_FP_EXCEEDS_THE_HOSTLESS_CENSUS",
            value=float(fp_ceil["excess"]),
            detail=f"mu_FP_per_contract = {fp_ceil['mu_fp_per_contract']:.2f} "
                   f"exceeds the mock's {fp_ceil['ceiling']:.0f} unmatched "
                   "on-grid candidates (the floor-17.2 hostless census, "
                   "~92% genuine sub-floor detections — NOT a forest-FP "
                   "supply). On the calibration twin the excess is an "
                   "estimand artifact resolved by chance-coincidence "
                   "correction; cross-mock it reflects the unresolved "
                   "transport systematic (Layer C). See fp_ceiling.reference "
                   "(referee M-B; Phase-A verdict "
                   "review_phaseA/fp_normalization/findings.md)."))
    if flags and strict:
        raise ContractViolation(
            "check_accounting_identity(strict=True): "
            + "; ".join(f["id"] for f in flags))

    return dict(
        contract_version=CONTRACT_VERSION,
        geometry=geom,
        resp_clamp=resp_clamp,
        flags=flags,
        truth_ledger=dict(
            n_truth_on_basis=t_tot,
            found_on_grid=float(found_on.sum()),
            found_off_grid=float(found_off.sum()),
            missed_P3=float(missed.sum()),
            residual=truth_resid,
            rel_residual=truth_resid / t_tot if t_tot else np.nan,
            residual_is_a_tautology=True,
            residual_note="T.C.rho + T.C.(1-rho) + T.(1-C) == T identically; "
                          "this residual detects a shape / axis / dtype crash "
                          "and nothing else. Read value_guards.",
            value_guards=value_guards,
        ),
        candidate_ledger=dict(
            n_obs=n_obs,
            P1_scatter_in=p1, P2_in_window=p2, P6_above_ceiling=p6hi,
            signal_subtotal=sig,
            as_folded=led_folded,
            per_contract=led_contract,
            folded_equals_contract=bool(
                abs(fpa["ratio_contract_over_folded"] - 1.0) <= 1e-9),
            if_ell_eff_omitted=dict(
                led_if_omitted,
                note="COUNTERFACTUAL, not the committed code: what the fold "
                     "produced before 7707c8e. The key used to be called "
                     "`as_implemented` and it described the fold; it no "
                     "longer does. See RESOLVED_BY_ID['FP_ELL_EFF_OMITTED']."),
            residual_per_contract=led_contract["residual"],
            P6_unsupported_implied_per_contract=p6_unsup,
            P6_unsupported_implied_is_negative=bool(p6_unsup < 0),
        ),
        fp_normalisation=fpa,
        fp_ceiling=fp_ceil,
        prior_cost=prior_cost,
        feasibility=dict(
            max_attainable_signal=t_tot,
            required_signal_per_contract=n_obs - fpa["mu_fp_total_per_contract"],
            required_signal_as_folded=n_obs - fpa["mu_fp_total_as_folded"],
            efficiency_required_per_contract=(
                (n_obs - fpa["mu_fp_total_per_contract"]) / t_tot),
            efficiency_required_as_folded=(
                (n_obs - fpa["mu_fp_total_as_folded"]) / t_tot),
            # the pre-repair counterfactual; NOT the committed code
            efficiency_required_if_ell_eff_omitted=(
                (n_obs - fpa["mu_fp_total_if_ell_eff_omitted"]) / t_tot),
            # THE ONLY BOUND: C <= 1 and rho <= 1.
            trivial_bound_efficiency=1.0,
            feasible_per_contract=bool(
                (n_obs - fpa["mu_fp_total_per_contract"]) <= t_tot),
            feasible_as_folded=bool(
                (n_obs - fpa["mu_fp_total_as_folded"]) <= t_tot),
            feasible_if_ell_eff_omitted=bool(
                (n_obs - fpa["mu_fp_total_if_ell_eff_omitted"]) <= t_tot),
            # a POINT in an unbounded nuisance space — NOT a bound. Kept under
            # its honest name; the old key `efficiency_attainable_at_calibration`
            # and the two `feasible_at_calibration_*` booleans are RETRACTED.
            efficiency_at_calibration=(sig / t_tot if t_tot else np.nan),
            note="C <= 1 and rho <= 1, so the signal can never exceed the "
                 "truth count on the basis: efficiency_required > 1 is a "
                 "COUNTING ARGUMENT and no parameter setting closes the model. "
                 "That is the ONLY bound here. efficiency_at_calibration is "
                 "the value at psi_c = psi_k_delta = 0, which is a POINT in an "
                 "UNBOUNDED nuisance space and bounds nothing; see prior_cost "
                 "for what it actually costs to close, and RETRACTIONS for "
                 "what this module no longer claims.",
        ),
    )


# ---------------------------------------------------------------------------
# 9. where the code contradicts this contract
# ---------------------------------------------------------------------------
KNOWN_CONTRADICTIONS = (
    dict(
        id="FP_Z_SHAPE_DIFFERS_ACROSS_THE_OBSERVED_FLOOR",
        site="CDDF_analysis/hbi_mcmc/extract_pack.py:910-914 (fp_E_alloc[k,s] "
             "= dX[k,s] / sum_k dX[k,s]; empty strata -> 0), consumed by "
             "forward.fold_mu_fp; the loa-0 FP block is "
             "extract_pack.build_fp_block:594.",
        contract="P4's z-shape must be estimated on the SAME support the P4 "
                 "rate is estimated on. The fold spreads the loa-0 FP total "
                 "over coarse z by the PATHLENGTH allocation E[k,s], which is "
                 "a statement about where sightlines are, not about where "
                 "forest false positives are.",
        code="E[k,s] is dX-proportional and carries no FP z-information at "
             "all; the only FP z-information in the pack (which 0.1-dex n-hat "
             "bin each of the 89 on-grid loa-0 FPs sits in) is marginalised "
             "away before it reaches the fold.",
        measured=(
            "MEASURED 2026-08-05 on the committed loa-0 FP catalogue "
            "(gl_loa0_fp_v1_20260615/outputs, 3255 raw rows; op = "
            "SNR_REDSIDE>2 & P_DLA>0.99; lya = lam_rest >= 1025 A -> 2378 "
            "rows; z_DLA in [2.0,3.5) -> 2318):\n"
            "    on the pack grid N_hat in [19.5,22.4) :   89  (3.8%)\n"
            "    BELOW the observed floor N_hat < 19.5 : 2229  (96.2%)\n"
            "    at or above 22.4                     :    0\n"
            "so 96.2% of the op-passing loa-0 FP population is OFF the grid "
            "the FP rate is estimated on. The two halves do not share a "
            "z-shape. Coarse-z ([2.0,2.5) / [2.5,3.0) / [3.0,3.5)):\n"
            "    in-support (n=  89)  43 / 36 /  10  = 0.4831 / 0.4045 / 0.1124\n"
            "    below-floor (n=2229) 1497 / 588 / 144 = 0.6716 / 0.2638 / "
            "0.0646\n"
            "    2x3 homogeneity chi2(2) = 13.8066, p = 0.0010045 "
            "(scipy.stats.chi2_contingency, correction=False)\n"
            "The shape the FOLD imposes is neither: MEASURED on the adopted "
            "packs (fp_E_alloc, shipped dX allocation) 0.5985/0.2968/0.1048 "
            "(2lpt0), 0.5999/0.2962/0.1039 (london0), 0.6167/0.3019/0.0814 "
            "(saclay0)."),
        invisible_to_the_gate=(
            "🔴 The z-allocation CANCELS EXACTLY out of the statistic that "
            "decides closure. ``sum_k fp_E[k,s] == 1`` on every populated "
            "stratum (pack schema, pack.py:545; MEASURED column sums "
            "[0,0,1,1,1,1,1,1] on all three adopted packs, the zeros being "
            "the structurally empty SNR<=2 op-mask strata, which carry "
            "fp_counts == 0), so the FP term's n-hat marginal is "
            "fp_w.fp_ell_eff.lam_fp[c,s] whatever E is. MEASURED 2026-08-05 "
            "by re-folding each adopted pack with E replaced by the two "
            "shapes above (dX-proportional WITHIN each coarse block, so only "
            "the coarse masses move):\n"
            "    window [19.7,21.6) chi2/dof   2lpt0 22.2236  london0 28.3934 "
            " saclay0 25.7723\n"
            "    ... under BOTH alternatives, agreeing with the shipped "
            "allocation to <= 9.65e-16 RELATIVE (max |d mu| 1.8e-12 counts on "
            "per-bin mu of order 1e4 — float summation order, not signal). "
            "NOT bit-identical, and the distinction is only that.\n"
            "    by_nhat max|z|, by_snr max|z| and the total z are unchanged "
            "to every printed digit.\n"
            "    the ONLY arm that moves is by_z max|z|: on 2lpt0 7.6404 "
            "(shipped) -> 3.7459 (in-support shape) -> 10.2587 (below-floor "
            "shape).\n"
            "The leg that decides closure is chi2/dof <= 3 over the reported "
            "n-hat bins (run_posterior.GATE['chi2_dof_max']) — the only "
            "tolerance in that gate with a recorded deciding-authority "
            "decision behind it. The four |z| arms, by_z included, are marked "
            "restated-but-not-decided in forward_selftest._closure_verdict and "
            "ratification.py. So the one statistic that could separate "
            "sub-floor migration from genuine forest FP does not enter the leg "
            "that decides closure, and the leg it does enter has no deciding "
            "authority behind it. Closure FAILS on all three mocks under every "
            "allocation tried, so nothing here changes a verdict."),
        effect="This is occurrence #12 of the one-sided-support class "
               "(numerator/basis and denominator/target on different "
               "supports): the FP RATE is estimated on N_hat >= 19.5 while "
               "96.2% of the FP population — and a demonstrably different "
               "z-shape — lives below it. It bears directly on the FP ceiling "
               "violation: if the sub-floor FPs migrate up, the excess over "
               "`unmatched` is a mis-attribution of scatter, not a supply "
               "problem; the pack cannot tell the two apart.",
        status="MEASURED AND RECORDED, NO CHANGE PROPOSED. Re-allocating P4 in "
               "z would change the physical definition of the FP background "
               "and is a PI_CHECKPOINT item, not a contract decision.",
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
        measured="SUPPORT MISMATCH CONFIRMED, MAGNITUDE RETRACTED. Cross-"
                 "matching loa-0 target_ids against loa-124 bal_cat.fits "
                 "(193737 unique TIDs; byte-identical twins, same TARGETIDs): "
                 "of the 2255 searched loa-0 sightlines with SNR>2, 351 sit on "
                 "BAL targets and 1904 do not. Of the 89 loa-0 FPs on the pack "
                 "grid, 19 sit on BAL targets and 70 do not. Those numbers are "
                 "right. What they were used for was not — see 'retracted'.",
        retracted=dict(
            what="the claim that the site comment is FALSE, the 7.35% "
                 "magnitude, the direction ('P4 is HIGH'), and the associated "
                 "PI checkpoint item as it was worded.",
            why="The premise was tested on the FULL op+lya loa-0 FP catalogue "
                "(2378 events, 27x the statistics of the 89) and there is NO "
                "BAL signal in loa-0. MEASURED 2026-08-05 (raw 3255 FP rows; "
                "op = SNR_REDSIDE>2 & P_DLA>0.99; lya = lam_rest>=1025 A):\n"
                "    op     : N=2704  FP/sightline BAL 1.1880  nonBAL 1.2012  "
                "ratio 0.98908  z = -0.21\n"
                "    op+lya : N=2378  FP/sightline BAL 1.0570  nonBAL 1.0541  "
                "ratio 1.00274  z = +0.05\n"
                "    the 89 : N=  89  BAL 19 vs expected 13.853 (sd 3.420)     "
                "         z = +1.50\n"
                "The 19-of-89 excess is a 1.50-sigma fluctuation of the "
                "binomial split at p = 351/2255, and the 1.07352 ratio it "
                "produces is 1.40 sigma from 1.0 (sd 0.0525, propagating the "
                "same binomial). Adopting 70/1904 would DISCARD 351 valid "
                "sightlines and 19 valid FPs to chase that fluctuation: the FP "
                "total's Poisson variance goes as 1/n, so 89 -> 70 events "
                "inflates the relative variance by 89/70 = 1.271x (relative sd "
                "+12.7%), in exchange for advertising a 7.35% bias correction "
                "that is itself 1.40 sigma from zero.",
            measured="2026-08-05, this branch; loa-0 dlacat-*.fits under "
                     "gl_loa0_fp_v1_20260615/outputs + loa-124 bal_cat.fits.",
        ),
        effect="NO measured effect on P4. The one-sided veto is a real "
               "support-matching defect of the KIND that has bitten this "
               "project four times (numerator and denominator on different "
               "supports), and it is recorded for that reason; on loa-0 its "
               "measured size is consistent with zero.",
        status="OBSERVATION RECORDED, NO CHANGE PROPOSED. The comment at "
               ":231 is a statement about BAL *masking* being a no-op for the "
               "loa-0 FP rate, and at 27x the statistics that is what the data "
               "show. If a future FP set does show a BAL dependence the "
               "support mismatch becomes load-bearing and must be fixed then.",
    ),
    dict(
        id="FITCOV_SD_IS_AN_UNCALIBRATED_FALLBACK",
        site="CDDF_analysis/hbi_mcmc/forward.py:218 "
             "(_DEFAULT_FITCOV_DIAG = (0.02**2, 0.10**2)) used by "
             "build_consts when pack.resp_fitcov_diag is absent; consumed by "
             "model_a.py:208-209 as the psi_k_delta prior width.",
        contract="a nuisance PRIOR WIDTH that a quantitative argument hinges "
                 "on must be a measured calibration product, or be labelled a "
                 "guess wherever it is used.",
        code="the fallback is silent: build_consts substitutes it and stamps "
             "nothing that reaches the prior-cost report.",
        measured="MEASURED 2026-08-05: pack.resp_fitcov_diag is ABSENT from "
                 "all six extracted packs (2lpt0/london0/saclay0 adopted "
                 "window-study bw0p2, and the three v1.1), so consts.fitcov_sd "
                 "is (0.02, 0.10) broadcast over all (2, SR=3, ZR=3) entries "
                 "on every one of them.",
        effect="The psi_k_delta prior cost scales as 1/fitcov_sd^2. The "
               "witness that closes the adopted geometry's 882.30-count gap "
               "is a -0.7056 prior-sd uniform shift, prior chi2 = 9 x 0.7056^2 "
               "= 4.481, i.e. 2.117 sigma — a number computed entirely against "
               "a documented guess. It is reported ONLY with "
               "prior_cost.fitcov_sd_provenance attached.",
        status="REPORTED, NOT FIXED. Calibrating it means propagating the "
               "response fit covariance into the pack, which changes what the "
               "packs carry and is out of this contract's scope.",
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


CONTRADICTION_BY_ID = {d["id"]: d for d in KNOWN_CONTRADICTIONS}


# ---------------------------------------------------------------------------
# 9a. contradictions this contract recorded that the CODE has since fixed
# ---------------------------------------------------------------------------
# Kept, in full, on purpose.  A defect that was real, was measured and was
# repaired is the most useful kind of record there is; deleting it would leave
# the repair looking like an unexplained behaviour change.  What must NOT
# survive is the claim that it is LIVE — that belongs nowhere near
# KNOWN_CONTRADICTIONS, and the guard that enforced it must now enforce the
# repair instead (``assert_forward_fp_normalisation``).
RESOLVED_CONTRADICTIONS = (
    dict(
        id="FP_ELL_EFF_OMITTED",
        recorded="2026-08-05 (as a live KNOWN_CONTRADICTION)",
        resolved="2026-08-05, in the CODE",
        fixed_by=("7707c8e — C1 (BEHAVIOUR CHANGE): the FP fold must carry "
                  "fp_ell_eff, at every site, generator included",
                  "2b436df — C2 (behaviour-preserving): one FP term, "
                  "forward.fold_mu_fp; the re-typed copy had already drifted"),
        contract="mu_FP = fp_w . fp_ell_eff . lam_fp . exp(t) . E",
        was="mu_FP = fp_w . lam_fp . exp(t) . E — fp_ell_eff absent, although "
            "build_consts carried it.",
        now="mu_FP = consts.fp_w * consts.fp_ell_eff * exp_t_k * lam_fp * "
            "fp_E, defined ONCE in forward.fold_mu_fp.",
        sites=("forward.fold_mu (now delegates to fold_mu_fp)",
               "forward.fold_mu_reference (the independent numpy oracle; it "
               "does NOT call the helper, by design)",
               "forward_selftest.selftest (the re-typed mu_fp behind the "
               "mu_sig split — the copy had also dropped exp(log_t), inert "
               "only because that caller passes log_t = 0)",
               "pack.synthetic_pack (THE GENERATOR — it inverted the DEFECTIVE "
               "fold, so generator and model agreed while both were wrong and "
               "no synthetic rung or SBC replica could have caught this)"),
        why_it_survived=(
            "fp_ell_eff is INERT in the loa-0 SOURCE route — Gamma(a, 1/ell) "
            "scaled by ell is Gamma(a, 1) — so it cancels there. It does not "
            "cancel on the data side, where it is a live Poisson exposure: "
            "fp_counts ~ Poisson(fp_ell_eff . lam_fp) makes lam_fp an "
            "intensity PER UNIT loa-0 exposure, not a count."),
        measured_before_after=(
            "RE-MEASURED 2026-08-05 on this branch, adopted packs "
            "(bw 0.2 / pad 19.0 / molly172 / lya_only / resp_clamp both). The "
            "'before' column is reproduced WITHOUT reverting the code, by "
            "folding a pack whose fp_counts are divided by fp_ell_eff — "
            "algebraically identical to the omission:\n"
            "    folded mu_FP total (2LPT-0)  1086.6872  ->  14767.9614\n"
            "    ratio                        13.589891949531905 == "
            "fp_ell_eff, exactly\n"
            "    fp_w . fp_ell_eff == 2255.0 == N_sl_loa0 on all three packs\n"
            "    zero-sampling closure, window [19.7,21.6) chi2/dof:\n"
            "        2lpt0    56.5846 -> 22.2236\n"
            "        london0  40.1578 -> 28.3934\n"
            "        saclay0  44.2126 -> 25.7723\n"
            "    full observed grid chi2/dof:\n"
            "        2lpt0   441.4088 -> 19.0713\n"
            "        london0 224.0390 -> 29.0421\n"
            "        saclay0 305.7693 -> 20.3300\n"
            "    total mu/obs (full grid):\n"
            "        2lpt0   0.846207 -> 1.001551\n"
            "        london0 0.894839 -> 1.050090\n"
            "        saclay0 0.871053 -> 1.028140\n"
            "    truth-pinned candidate-ledger residual (adopted 2LPT-0): "
            "-14563.572 -> -882.298."),
        what_it_did_NOT_fix=(
            "CLOSURE STILL FAILS on all three mocks: the ratified leg is "
            "chi2/dof <= 3 and the repaired values are 22.2 / 28.4 / 25.8. "
            "This is a repair, not a pass. It also CREATED the ceiling "
            "violation on all three mocks (FP_CEILING_MEASURED): the "
            "correctly-normalised mu_FP now exceeds the mock's entire supply "
            "of unmatched on-grid candidates by +6.55% / +53.33% / +38.85%. "
            "The normalisation is forced by two committed definitions and was "
            "not chosen against the data; the ceiling violation is a FINDING "
            "about the FP model, not an argument for putting the factor back."),
        regression_guard=(
            "assert_forward_fp_normalisation(pack) — it CALLS "
            "forward.fold_mu_fp and compares the folded total with "
            "fp_w . fp_ell_eff . lam_fp. Re-introducing the omission makes the "
            "ratio equal fp_ell_eff exactly and it raises, naming this record. "
            "Pinned from the source side by "
            "tests/test_matching_contract.py::"
            "test_the_forward_fp_fold_carries_ell_eff_at_every_named_site."),
        status="RESOLVED IN CODE. NOT a live contradiction. The record is "
               "retained; the claim is withdrawn.",
    ),
)

RESOLVED_BY_ID = {d["id"]: d for d in RESOLVED_CONTRADICTIONS}


# ---------------------------------------------------------------------------
# 9b. what this module USED to claim and no longer does
# ---------------------------------------------------------------------------
RETRACTIONS = (
    dict(
        id="C3_CALIBRATION_POINT_IS_NOT_A_BOUND",
        date="2026-08-05",
        withdrawn=("`efficiency_attainable_at_calibration` described as 'the "
                   "SHARPER bound'; the emitted booleans "
                   "`feasible_at_calibration_per_contract` and "
                   "`feasible_at_calibration_as_implemented`; and the "
                   "INFEASIBLE reading of the adopted 0.2-dex/pad-19.0 "
                   "geometry that rested on them."),
        why=("The quantity was evaluated at psi_c = 0 and psi_k_delta = 0. "
             "Both are numpyro SAMPLE SITES with UNBOUNDED support "
             "(psi_c ~ Normal(0, sigma_hat), psi_k_delta ~ Normal(0, "
             "fitcov_sd), model_a.py:206-209). A point value in an unbounded "
             "nuisance space is not an upper bound on anything."),
        measured=("On the adopted 2LPT-0 pack (0.2-dex basis, pad 19.0, "
                  "molly172, lya_only), 2026-08-05:\n"
                  "    efficiency at calibration        0.7103624\n"
                  "    efficiency REQUIRED per contract 0.7190167\n"
                  "    sup over psi_c alone (C -> 1)    0.8572838\n"
                  "    sup over psi_k alone (rho -> 1)  0.8074237\n"
                  "The required efficiency sits far BELOW both suprema. Cost "
                  "to close, through the committed jax build_K: -0.7056 "
                  "prior-sd uniform shift on psi_k_delta[1], prior chi2 = "
                  "9 x 0.7056^2 = 4.481 = 2.117 sigma. Two further declared "
                  "directions close it inside 1 sigma: the FP total's own "
                  "loa-0 Poisson width (mu_FP = 14767.96 on N_FP = 89, so "
                  "1 sigma = 1565.40 counts against a gap of 882.30 => 0.564 "
                  "sigma) and a uniform transfer shift (delta = -0.0616031, "
                  "chi2 = 0.75467 => 0.86871 sigma). The exact minimum in psi_c "
                  "alone is prior chi2 = 107.87 (10.39 sigma; primal witness "
                  "and dual bound agree to 1e-4)."),
        what_survives=(
            "On the UNPADDED v1.1 packs the refutation is materially stronger, "
            "and for a reason the earlier text did not give — a PRIOR-COST "
            "argument, not a bound argument. MEASURED 2026-08-05:\n"
            "    pack          eff req (contract FP)  sup over psi_c alone   "
            "min prior chi2 in psi_c\n"
            "    2lpt0_v11        0.9958299            0.9938370 (< req)      "
            "infinity\n"
            "    london0_v11      0.9385533            0.9939876              "
            "10757.4  (103.7 sigma)\n"
            "    saclay0_v11      0.9559406            0.9937649              "
            "20085.8  (141.7 sigma)\n"
            "On 2lpt0_v11 even C == 1 at INFINITE prior cost cannot reach the "
            "required efficiency — that one IS a bound argument, because it "
            "compares against the supremum. On london0/saclay0 it is a cost of "
            "order 1e4 in prior chi2, which is a statement about how far the "
            "declared priors would have to be violated, not about "
            "impossibility."),
        replaced_by="prior_cost_audit / min_prior_chi2_psi_c; "
                    "check_accounting_identity(...)['prior_cost'].",
    ),
    dict(
        id="BAL_VETO_MAGNITUDE",
        date="2026-08-05",
        withdrawn=("the 7.35%-HIGH magnitude for P4, its direction, the claim "
                   "that build_loa0_fp_product.py:231 is FALSE, and the PI "
                   "checkpoint item proposing 70/1904."),
        why="tested on 2378 op+lya loa-0 FPs (27x the 89): no BAL signal.",
        measured="see KNOWN_CONTRADICTIONS BAL_VETO_ONE_SIDED_IN_FP_W."
                 "retracted.",
        what_survives="the support-matching OBSERVATION: N_prod excludes BAL "
                      "targets and N_sl_loa0 does not, so the ratio divides "
                      "quantities defined on different sightline sets. That is "
                      "a real defect of a class this project has hit four "
                      "times; its measured size on loa-0 is consistent with "
                      "zero.",
        replaced_by="the rewritten BAL_VETO_ONE_SIDED_IN_FP_W entry.",
    ),
    dict(
        id="TRUTH_LEDGER_RESIDUAL_ADVERTISED_AS_A_TEST",
        date="2026-08-05",
        withdrawn=("'a nonzero value means the implementation is broken' as "
                   "the FIRST of 'the two numbers a referee should read'."),
        why=("T.C.rho + T.C.(1-rho) + T.(1-C) == T is an algebraic identity in "
             "C, rho and T. MEASURED: rho in {0.0, 0.8, 1.0, U(0,1)} x "
             "molly_n_det in {0, 1, 99} — all twelve leave the residual at 0.0 "
             "to +-2.3e-13. It detects a shape / broadcast / dtype crash and "
             "nothing else."),
        what_survives="the residual still catches an axis or shape error, and "
                      "is kept for that; the CONTENT moved to "
                      "truth_ledger.value_guards, which bound FOUND_ON, "
                      "FOUND_OFF and MISSED individually and pin the "
                      "(S,KK,B) -> (B,Kf,S) transpose with an index "
                      "round-trip.",
        replaced_by="_truth_ledger_value_guards.",
    ),
)


PI_CHECKPOINT_ITEMS = (
    "FP_ELL_EFF_OMITTED IS FIXED IN CODE (7707c8e, 2b436df) and the PHYSICAL "
    "size of P4 changed by exactly fp_ell_eff ~ 13.6x. The item is no longer "
    "'should we fix it' — it is 'the FP background the corrected fold implies "
    "is larger than the mock can supply'. MEASURED 2026-08-05 on ALL THREE "
    "mocks, on-grid op-passing candidates against mu_FP_per_contract: "
    "2lpt0 13860 unmatched vs 14767.96 (+907.96, +6.55%); london0 9598 vs "
    "14716.38 (+5118.38, +53.33%); saclay0 10592 vs 14707.06 (+4115.06, "
    "+38.85%). `unmatched` over-counts true forest FP (it also holds blends "
    "and second candidates), so these are LOWER BOUNDS on the excess. Either "
    "the loa-0 rate does not transfer to the mocks at this normalisation, or "
    "the excess is sub-floor scatter mis-labelled as forest FP. Collected, "
    "not decided.",
    "The FP z-shape is a physical assumption the fold currently makes by "
    "accident: P4 is spread over z by the PATHLENGTH allocation, and the "
    "measured loa-0 FP z-shape differs significantly across the observed "
    "floor (chi2(2) = 13.81, p = 0.0010; 96.2% of op-passing loa-0 FPs sit "
    "below the grid). Re-allocating P4 in z changes what the FP background IS. "
    "Note before spending effort on it: the ratified chi2/dof leg of the "
    "closure gate is EXACTLY invariant to the allocation, so no measurement of "
    "the allocation can move the closure verdict. Collected.",
    "The one-sided BAL veto in fp_w is a SUPPORT-MATCHING observation with no "
    "measured effect: on 2378 op+lya loa-0 FPs the BAL/nonBAL rate ratio is "
    "1.0027 (z = +0.05). NO change to 70/1904 is proposed and the earlier "
    "'7.35% HIGH' item is RETRACTED (see RETRACTIONS[BAL_VETO_MAGNITUDE]). "
    "Recorded so the support mismatch is not rediscovered as new.",
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
        fp_ceiling_measured=FP_CEILING_MEASURED,
        known_contradictions=list(KNOWN_CONTRADICTIONS),
        resolved_contradictions=list(RESOLVED_CONTRADICTIONS),
        retractions=list(RETRACTIONS),
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
