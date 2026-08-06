# -*- coding: utf-8 -*-
"""Tests for the FROZEN FORWARD-MODEL AND MATCHING CONTRACT.

Every test is MUTATION-TESTED: the mutation that turns it red is named in the
test's own docstring, together with the MEASURED baseline it fails against, so
a reader can apply the mutation and check.

MOCK-DERIVED ONLY.  The pure-function tests read nothing; the integration tests
read the ADOPTED mock packs under the window-study scratch dir and SKIP when
they are absent.  No real-DESI path is opened.

2026-08-05: rewritten after an independent statistical referee.  The section-9
tests cover C3 (the retracted feasibility verdict), M-A (the tautological truth
residual), M-B (the FP ceiling and the negative implied population), M-C (_p3's
declared predicate), M-D (fail-open validation), M-E (the retracted BAL
magnitude) and M-F (the uncalibrated fitcov fallback).  Several of those
mutants die ONLY when the window-study packs are present; each such test says
so and there is a pack-free companion.
"""
import json
import os
import sys
import types

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from CDDF_analysis.hbi_mcmc import matching_contract as MC     # noqa: E402
from CDDF_analysis.hbi_mcmc import reporting as RP             # noqa: E402


# --- the ADOPTED geometry, written out once ---------------------------------
ADOPTED_NTRUE = np.array([19.0, 19.2, 19.5, 19.7, 19.9, 20.1, 20.3, 20.5, 20.7,
                          20.9, 21.1, 21.3, 21.5, 21.7, 21.9, 22.1, 22.4])
REAL_NHAT = np.round(np.arange(19.5, 22.4 + 1e-9, 0.1), 10)
MOLLY172 = np.array([17.2, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.3, 20.5,
                     21.0, 21.5, 22.0, np.inf])

PACKDIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
           "window_study/packs")
ADOPTED_PACK = os.path.join(
    PACKDIR, "modelA_pack_2lpt0_winlya_only_pad19p0_molly172_bw0p2.npz")
V11DIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs")
UNPADDED_PACK = os.path.join(V11DIR, "modelA_pack_2lpt0_v11.npz")


def _fake_pack(*, ntrue=None, nhat=None, molly=None, n_b=None, n_k=3, n_s=2,
               truth_bks=True, counts=None, fp_counts=None,
               fp_ell_eff=13.589891949531905, fp_w=165.93215077605322,
               molly_n_det=None, molly_n_tot=None, kz=None, t_sigma=None,
               mock=None, fp_E_alloc=None, fp_eta_c=None):
    """A minimal duck-typed pack.

    ``validate_pack_against_contract`` / ``fp_normalisation_audit`` /
    ``check_accounting_identity(row_mass=...)`` touch only these fields, so a
    namespace is enough and the tests stay fast and survey-free.

    2026-08-06 (fp_eta_c restoration): the schema now REQUIRES the
    per-observed-bin host-occlusion vector, so the fake pack carries it too —
    derived from the committed loa-0 band table exactly as the explicit
    legacy-pack migration (``pack.attach_fp_eta_bands``) would derive it.
    """
    ntrue = ADOPTED_NTRUE if ntrue is None else np.asarray(ntrue, float)
    nhat = REAL_NHAT if nhat is None else np.asarray(nhat, float)
    molly = MOLLY172 if molly is None else np.asarray(molly, float)
    if fp_eta_c is None:
        from CDDF_analysis.hbi_mcmc.pack import (
            FP_ETA_BANDS_COMMITTED, eta_from_intervals)
        fp_eta_c = eta_from_intervals(nhat,
                                      [b[0] for b in FP_ETA_BANDS_COMMITTED],
                                      [b[1] for b in FP_ETA_BANDS_COMMITTED],
                                      [b[2] for b in FP_ETA_BANDS_COMMITTED])
    B = len(ntrue) - 1 if n_b is None else n_b
    M = len(molly) - 1
    nd = np.full((n_s, M), 60.0) if molly_n_det is None else np.asarray(molly_n_det, float)
    nt = np.full((n_s, M), 100.0) if molly_n_tot is None else np.asarray(molly_n_tot, float)
    return types.SimpleNamespace(
        nhat_edges=nhat, ntrue_edges=ntrue, molly_nhi_edges=molly,
        kz_to_K=(np.zeros(n_k, dtype=int) if kz is None else np.asarray(kz, int)),
        counts=(np.zeros((len(nhat) - 1, n_k, n_s), dtype=np.int64)
                if counts is None else np.asarray(counts)),
        fp_counts=(np.zeros((len(nhat) - 1, n_s), dtype=np.int64)
                   if fp_counts is None else np.asarray(fp_counts)),
        fp_ell_eff=fp_ell_eff, fp_w_sightline_ratio=fp_w,
        # (Kf, S); the schema requires sum_k E[k,s] == 1 on populated strata
        fp_E_alloc=(np.full((n_k, n_s), 1.0 / n_k) if fp_E_alloc is None
                    else np.asarray(fp_E_alloc, float)),
        molly_n_det=nd, molly_n_tot=nt,
        fp_eta_c=np.asarray(fp_eta_c, float),
        t_sigma=(np.array([0.1, 0.1, 0.1]) if t_sigma is None
                 else np.asarray(t_sigma, float)),
        provenance=(dict(mock=mock) if mock else dict()),
        resp_fitcov_diag=None,
        truth_counts_bks=(np.zeros((B, n_k, n_s)) if truth_bks else None),
    )


# ===========================================================================
# 1. the partition: no overlap, no gap, no double counting
# ===========================================================================
def test_basis_partition_fractions_sum_to_one_in_every_bin():
    """MUTATION: in ``basis_partition``, drop the ``above_ceiling`` term from
    ``tot`` (or replace ``truth_overlap_fractions`` by a centre test). MEASURED
    baseline: on the ADOPTED basis the three fractions sum to 1.0 in all 16
    bins with max deviation 0.0; a centre test gives 0.0 or 2.0 on the bin
    [21.5, 21.7) which straddles the ceiling."""
    p = MC.basis_partition(ADOPTED_NTRUE)
    tot = p["below_floor"] + p["in_window"] + p["above_ceiling"]
    assert tot.shape == (len(ADOPTED_NTRUE) - 1,)
    assert np.max(np.abs(tot - 1.0)) == 0.0


def test_basis_partition_splits_the_straddling_ceiling_bin_by_overlap():
    """The ceiling 21.6 falls INSIDE the basis bin [21.5, 21.7): 21.6 - 19.7 =
    1.9 dex is an ODD multiple of 0.1, so no uniform 0.2-dex basis anchored at
    the reporting floor can carry both edges.

    MUTATION: use ``bins_fully_inside`` instead of ``truth_overlap_fractions``.
    MEASURED baseline: the straddling bin's in-window fraction is
    0.5000000000000089 and its above-ceiling fraction 0.4999999999999911 —
    8.9e-15 off the analytic 0.5, NOT exact (the docstring used to overstate
    this; referee minor 2026-08-05). ``bins_fully_inside`` gives 0.0 and would
    silently move a whole 0.2-dex bin out of P2."""
    p = MC.basis_partition(ADOPTED_NTRUE)
    j = int(np.flatnonzero(np.isclose(ADOPTED_NTRUE[:-1], 21.5))[0])
    assert p["in_window"][j] == pytest.approx(0.5, abs=1e-12)
    assert p["above_ceiling"][j] == pytest.approx(0.5, abs=1e-12)
    assert p["below_floor"][j] == 0.0
    # the deviation is real and bounded; pin it so "exactly 0.5" is never
    # re-asserted in the docstring.
    assert p["in_window"][j] != 0.5
    assert abs(p["in_window"][j] - 0.5) < 1e-13


def test_basis_partition_gives_the_pad_entirely_to_P1():
    """MUTATION: replace the ``below_floor`` overlap by an upper-edge test
    ``(e[1:] <= 19.7)``. On the ADOPTED basis that is an EQUIVALENT mutant —
    the floor is an exact edge, so nothing straddles it — which is exactly why
    the second half of this test evaluates the partition on a basis that DOES
    straddle 19.7. MEASURED baseline: on the adopted basis the three pad bins
    carry below_floor == 1.0; on a 0.2-dex basis anchored at 19.0 the bin
    [19.6, 19.8) carries below_floor == 0.5 and in_window == 0.5, while the
    upper-edge mutant reports 0.0 and 1.0 and hands half a bin of
    convention-dependent sub-floor support to P2."""
    p = MC.basis_partition(ADOPTED_NTRUE)
    pad = ADOPTED_NTRUE[1:] <= MC.REPORT_FLOOR + 1e-12
    assert np.all(p["below_floor"][pad] == 1.0)
    assert np.all(p["in_window"][pad] == 0.0)
    assert np.all(p["above_ceiling"][pad] == 0.0)

    straddling = np.round(np.concatenate([np.arange(19.0, 22.4, 0.2), [22.4]]), 10)
    j = int(np.flatnonzero(np.isclose(straddling[:-1], 19.6))[0])
    q = MC.basis_partition(straddling)
    assert q["below_floor"][j] == pytest.approx(0.5, abs=1e-12)
    assert q["in_window"][j] == pytest.approx(0.5, abs=1e-12)


def test_basis_partition_refuses_non_increasing_edges():
    """MUTATION: delete the edge check in ``basis_partition``. MEASURED
    baseline: a duplicated edge yields a zero-width bin and a 0/0 fraction; the
    guard must raise ContractViolation instead."""
    with pytest.raises(MC.ContractViolation):
        MC.basis_partition([19.0, 19.5, 19.5, 20.0])


# ===========================================================================
# 2. population predicates: exactly one slot per candidate
# ===========================================================================
@pytest.mark.parametrize("rec,expect", [
    (dict(is_TP=True, nhi_true=19.3, forest_attributable=False), "P1_SCATTER_IN"),
    (dict(is_TP=True, nhi_true=19.69, forest_attributable=False), "P1_SCATTER_IN"),
    (dict(is_TP=True, nhi_true=19.7, forest_attributable=False), "P2_IN_WINDOW"),
    (dict(is_TP=True, nhi_true=21.0, forest_attributable=False), "P2_IN_WINDOW"),
    (dict(is_TP=True, nhi_true=21.6, forest_attributable=False), "P6_RESIDUAL"),
    (dict(is_TP=True, nhi_true=18.5, forest_attributable=False), "P6_RESIDUAL"),
    (dict(is_TP=False, nhi_true=np.nan, forest_attributable=True), "P4_FOREST_FP"),
    (dict(is_TP=False, nhi_true=np.nan, forest_attributable=False), "P6_RESIDUAL"),
])
def test_classify_candidate_is_a_partition(rec, expect):
    """Every candidate lands in EXACTLY one population, and the boundaries are
    the reporting floor 19.7 (closed below), the ceiling 21.6 (open above) and
    the basis floor 19.0.

    MUTATION: change ``_p2``'s lower bound from ``>= REPORT_FLOOR`` to
    ``> REPORT_FLOOR``. MEASURED baseline: the record with nhi_true == 19.7
    then matches ZERO populations and ``classify_candidate`` raises — the
    fail-closed 'exactly one' check catches it."""
    assert MC.classify_candidate(rec) == expect


def test_classify_candidate_fails_closed_on_an_ambiguous_record():
    """MUTATION: relax the ``not (_p1 or _p2 or _p4)`` in ``_p6_candidate`` to
    a bare ``True``. MEASURED baseline: an in-window TP then matches BOTH
    P2_IN_WINDOW and P6_RESIDUAL and the guard raises; without the guard the
    same absorber is counted twice."""
    good = dict(is_TP=True, nhi_true=20.5, forest_attributable=False)
    assert MC.classify_candidate(good) == "P2_IN_WINDOW"
    # a record that claims to be BOTH a genuine absorber and forest-attributable
    # is still resolved to exactly one slot (is_TP wins; P4 requires not is_TP)
    assert MC.classify_candidate(
        dict(is_TP=True, nhi_true=20.5, forest_attributable=True)) == "P2_IN_WINDOW"


def test_classify_candidate_refuses_an_undeclared_or_coerced_record():
    """REFEREE MINOR (2026-08-05), fixed. The docstring promised fail-closed
    and the code was not.

    MEASURED before the fix: ``classify_candidate({})`` returned
    ``'P6_RESIDUAL'`` — a real answer, and specifically the slot with NO
    forward term, for a record that declares nothing; and
    ``classify_candidate(dict(is_TP=True, nhi_true='20.5',
    forest_attributable=False))`` returned ``'P2_IN_WINDOW'``, silently
    coercing a string through ``float()``.

    MUTATION: delete the ``_require_keys`` call in ``classify_candidate`` (or
    restore ``_is_num``'s old ``np.isfinite(float(x))`` body). MEASURED
    baseline: both calls below stop raising and return P6_RESIDUAL / P2."""
    with pytest.raises(MC.ContractViolation, match="missing required key"):
        MC.classify_candidate({})
    with pytest.raises(MC.ContractViolation, match="missing required key"):
        MC.classify_candidate(dict(is_TP=False, nhi_true=np.nan))
    with pytest.raises(MC.ContractViolation):
        MC.classify_candidate(
            dict(is_TP=True, nhi_true="20.5", forest_attributable=False))
    with pytest.raises(MC.ContractViolation, match="not a real finite number"):
        MC.classify_candidate(
            dict(is_TP=True, nhi_true=np.nan, forest_attributable=False))


def test_classify_truth_counts_a_matched_row_zero_times():
    """A matched truth row is already on the CANDIDATE ledger and must not be
    added to P3.

    MUTATION: make ``classify_truth`` return 'P3_INCOMPLETENESS' unconditionally.
    MEASURED baseline: a matched row then appears on both ledgers and the truth
    ledger residual (test below) goes from 0.0 to -sum(found)."""
    assert MC.classify_truth(dict(matched=True, nhi_true=20.5)) is None
    assert MC.classify_truth(
        dict(matched=False, nhi_true=20.5)) == "P3_INCOMPLETENESS"


def test_p3_executes_the_basis_support_test_its_predicate_declares():
    """REFEREE M-C (2026-08-05), fixed. ``P3_INCOMPLETENESS.predicate_text``
    says 'a truth row ... INSIDE THE BASIS SUPPORT ... that NO candidate
    claims', but ``_p3`` was a bare ``not rec.get('matched')``.

    MEASURED before the fix: ``classify_truth(dict(matched=False,
    nhi_true=18.0))`` returned ``'P3_INCOMPLETENESS'`` — for a truth row a full
    dex BELOW the 19.0 basis floor, which has no basis bin, hence no C[b,s] and
    no (1 - C) complement for the fold to carry. And
    ``classify_truth(dict(matched=False))`` — a row with no nhi_true at all —
    also returned P3.

    MUTATION: restore ``_p3 = lambda rec: not bool(rec.get('matched'))``.
    MEASURED baseline: the 18.0 row and the 22.9 row come back as P3 instead of
    OUT_OF_BASIS_SUPPORT, and the missing-key row stops raising."""
    assert MC.classify_truth(
        dict(matched=False, nhi_true=18.0)) == MC.TRUTH_OUT_OF_BASIS_SUPPORT
    assert MC.classify_truth(
        dict(matched=False, nhi_true=22.9)) == MC.TRUTH_OUT_OF_BASIS_SUPPORT
    # the boundaries the predicate now names, both closed-below / open-above
    assert MC.classify_truth(
        dict(matched=False, nhi_true=MC.ADOPTED_PAD_FLOOR)) == "P3_INCOMPLETENESS"
    assert MC.classify_truth(
        dict(matched=False, nhi_true=MC.ADOPTED_BASIS_TOP)
    ) == MC.TRUTH_OUT_OF_BASIS_SUPPORT
    # fail-closed on a row that cannot be tested
    with pytest.raises(MC.ContractViolation, match="missing required key"):
        MC.classify_truth(dict(matched=False))
    with pytest.raises(MC.ContractViolation, match="non-numeric nhi_true"):
        MC.classify_truth(dict(matched=False, nhi_true=np.nan))
    # and the DECLARED text names the executed bounds
    txt = MC.POPULATION_BY_ID["P3_INCOMPLETENESS"].predicate_text
    assert f"{MC.ADOPTED_PAD_FLOOR} <= NHI_TRUE < {MC.ADOPTED_BASIS_TOP}" in txt


def test_population_declaration_order_is_pinned():
    """REFEREE MINOR: ``classify_candidate`` returns ``hits[0]``, so tuple
    order would decide the answer if two slots ever claimed one record. The
    ``len(hits) != 1`` guard makes that unreachable; the order is pinned so a
    reorder is a visible diff, and the guard itself is exercised above.

    MUTATION: swap P1 and P2 in ``POPULATIONS``. MEASURED baseline: the module
    raises ContractViolation AT IMPORT."""
    assert tuple(p.pid for p in MC.POPULATIONS) == MC.POPULATION_ORDER
    assert MC.POPULATION_ORDER[0] == "P1_SCATTER_IN"
    assert MC.POPULATION_ORDER[-1] == "P6_RESIDUAL"


# ===========================================================================
# 3. contract validation fails loudly
# ===========================================================================
def test_validate_accepts_the_adopted_geometry():
    """MUTATION: change ``ADOPTED_PAD_FLOOR`` to 19.5 in the fake pack's edges.
    MEASURED baseline: the adopted geometry has 16 basis bins, 2 pad bins,
    widths [0.2, 0.3, 0.2 x 13, 0.3], molly floor 17.2, and the ceiling
    straddles a bin."""
    g = MC.validate_pack_against_contract(_fake_pack())
    assert g["n_basis_bins"] == 16
    assert g["n_pad_bins"] == 2
    assert g["basis_floor"] == 19.0
    assert g["reporting_floor_is_a_basis_edge"] is True
    assert g["ceiling_straddles_basis_bin"] is True
    assert g["sub_floor_completeness_measured"] is True
    assert g["molly_floor"] == 17.2


def test_validate_refuses_a_basis_that_straddles_the_reporting_floor():
    """MUTATION: delete rule 2 from ``validate_pack_against_contract``.
    MEASURED baseline: a basis anchored at 19.0 with a uniform 0.2-dex merge in
    ONE pass gives edges ... 19.4, 19.6 ... so 19.7 is not an edge and the bin
    [19.6, 19.8) mixes P1 (convention-dependent sub-floor completeness) with
    P2. The rule must raise; without it the pack validates and the P1/P2 split
    is silently wrong."""
    bad = np.round(np.concatenate([np.arange(19.0, 22.4, 0.2), [22.4]]), 10)
    assert not np.any(np.isclose(bad, 19.7))
    with pytest.raises(MC.ContractViolation, match="reporting floor"):
        MC.validate_pack_against_contract(_fake_pack(ntrue=bad))


def test_validate_refuses_an_unpadded_basis():
    """MUTATION: default ``require_pad=False``. MEASURED baseline: on the real
    unpadded 2LPT-0 pack the truth-pinned model needs an on-grid detection
    efficiency of 1.1817 (as the FP is currently coded) — impossible, since
    C <= 1 and rho <= 1. The refusal must be the DEFAULT, and opting out must
    be explicit."""
    with pytest.raises(MC.ContractViolation, match="TRUNCATED"):
        MC.validate_pack_against_contract(_fake_pack(ntrue=REAL_NHAT))
    # explicit opt-out still works, for demonstrating the infeasibility
    g = MC.validate_pack_against_contract(
        _fake_pack(ntrue=REAL_NHAT), require_pad=False)
    assert g["n_pad_bins"] == 0


def test_validate_refuses_constant_extrapolated_sub_floor_completeness():
    """MUTATION: delete rule 4. MEASURED baseline: with a molly grid starting at
    19.5, ``forward.build_consts``'s ``clip(digitize(Nc, molly_nhi_edges)-1, 0,
    M-2)`` sends BOTH pad bins to cell 0, i.e. P1's completeness becomes the
    constant extrapolation of [19.5, 20.0) — a convention, and a KNOWN TOO HIGH
    one. The contract must force that to be declared."""
    molly195 = np.array([19.5, 20.0, 20.3, 20.5, 21.0, 21.5, 22.0, np.inf])
    with pytest.raises(MC.ContractViolation, match="CONSTANT EXTRAPOLATION"):
        MC.validate_pack_against_contract(_fake_pack(molly=molly195))
    g = MC.validate_pack_against_contract(
        _fake_pack(molly=molly195),
        require_measured_sub_floor_completeness=False)
    assert g["sub_floor_completeness_measured"] is False


def test_validate_refuses_a_moved_observed_grid():
    """MUTATION: delete rule 1. MEASURED baseline: the reporting grid is fixed
    at [19.5, 22.4] step 0.1 in every pack ever extracted; a pack whose observed
    floor moved would silently redefine what 'scatter-in' means."""
    with pytest.raises(MC.ContractViolation, match="observed grid moved"):
        MC.validate_pack_against_contract(
            _fake_pack(nhat=np.round(np.arange(19.7, 22.4 + 1e-9, 0.1), 10)))


def test_validate_refuses_a_missing_truth_bks_and_negative_counts():
    """MUTATION: delete rules 5 and 6. MEASURED baseline: without
    ``truth_counts_bks`` the completeness (a function of the SNR stratum)
    cannot be applied and the ledger is not checkable; a NEGATIVE count can
    only come from an FP-SUBTRACTED array, which is the FF route's estimand and
    not this one's."""
    with pytest.raises(MC.ContractViolation, match="truth_counts_bks"):
        MC.validate_pack_against_contract(_fake_pack(truth_bks=False))
    c = np.zeros((len(REAL_NHAT) - 1, 3, 2), dtype=np.int64)
    c[0, 0, 0] = -1
    with pytest.raises(MC.ContractViolation, match="RAW"):
        MC.validate_pack_against_contract(_fake_pack(counts=c))


# ===========================================================================
# 4. the FP normalisation, pinned against the CODE (not against prose)
# ===========================================================================
def _fp_only_pack(n_fp=89):
    fp = np.zeros((len(REAL_NHAT) - 1, 2), dtype=np.int64)
    fp[0, 0] = n_fp
    return _fake_pack(fp_counts=fp)


def test_fp_normalisation_audit_reproduces_the_pack_scalars():
    """MEASURED on all three ADOPTED packs: fp_w * fp_ell_eff == 2255.0 exactly
    (== N_sl_loa0), and the contract total is fp_w * sum_c (1-eta_c) * N_FP[c]
    (the (1-eta) host-occlusion factor restored 2026-08-06, PI ruling 8).

    ``mu_fp_total_as_folded`` is obtained by CALLING ``forward.fold_mu_fp``,
    which is the whole point: the previous version hard-coded a reading of the
    fold's source and went stale the moment the fold was repaired.

    MUTATION: change ``contract = w * n_fp_surv`` to ``w * lam_tot`` in
    ``fp_normalisation_audit``. MEASURED baseline: the contract total drops
    14682.949169806607 -> 1080.431634359856, the ratio goes 1.0 ->
    0.07358272..., and ``assert_forward_fp_normalisation`` starts raising on
    the CORRECT fold."""
    a = MC.fp_normalisation_audit(_fp_only_pack())
    assert a["n_sl_loa0_implied"] == pytest.approx(2255.0, abs=1e-9)
    # (1-eta) restoration 2026-08-06: was 165.93215077605322 * 89.0; x(1-0.005756532459300326) on the FP term
    assert a["mu_fp_total_per_contract"] == pytest.approx(
        14682.949169806609, rel=1e-12)
    # the COMMITTED fold agrees with the contract, to float round-off
    # (1-eta) restoration 2026-08-06: was 14767.961419068737; x(1-0.005756532459300326) on the FP term
    assert a["mu_fp_total_as_folded"] == pytest.approx(
        14682.949169806609, rel=1e-12)
    assert a["ratio_contract_over_folded"] == pytest.approx(1.0, abs=1e-9)
    # the pre-restoration total survives ONLY as a labelled counterfactual
    assert a["mu_fp_total_if_eta_omitted"] == pytest.approx(
        14767.961419068737, rel=1e-12)
    # the pre-repair value survives ONLY as a labelled counterfactual
    # (1-eta) restoration 2026-08-06: was 1086.6871844096897; x(1-0.005756532459300326) on the FP term
    assert a["mu_fp_total_if_ell_eff_omitted"] == pytest.approx(
        1080.431634359856, rel=1e-9)
    # the omission ratio is UNCHANGED by the common (1-eta) factor
    assert (a["mu_fp_total_per_contract"]
            / a["mu_fp_total_if_ell_eff_omitted"]) == pytest.approx(
        13.589891949531905, rel=1e-9)


def test_assert_forward_fp_normalisation_passes_on_the_repaired_fold():
    """A guard that fires on the FIXED state is worse than no guard.

    Between 7707c8e and this commit ``assert_forward_fp_normalisation``
    compared the contract against a hard-coded description of the PRE-repair
    fold, so it raised ``ContractViolation`` on correct code, permanently. It
    now measures the committed ``forward.fold_mu_fp``.

    MUTATION: restore the hard-coded ``implemented = w * lam_tot``. MEASURED
    baseline: this call raises again on the repaired fold, with ratio
    13.589891949531905."""
    a = MC.assert_forward_fp_normalisation(_fp_only_pack())
    assert a["ratio_contract_over_folded"] == pytest.approx(1.0, abs=1e-9)
    assert a["mu_fp_total_as_folded"] == pytest.approx(
        a["mu_fp_total_per_contract"], rel=1e-12)
    assert "fold_mu_fp" in a["fold_site"]


def test_assert_forward_fp_normalisation_raises_if_the_omission_RETURNS():
    """THE regression guard. It has teeth only if re-introducing the defect
    turns it red, so re-introduce the defect.

    ``fp_ell_eff`` is dropped from the fold's expression by patching
    ``forward.fold_mu_fp`` — the single site the FP term is defined at since
    2b436df, and the site ``matching_contract`` imports lazily at call time.
    MEASURED baseline on this fake pack ((1-eta) restoration 2026-08-06: was
    14767.961419068737 -> 1086.6871844096897): the folded total falls
    14682.949169806607 -> 1080.431634359856, the ratio comes back exactly
    fp_ell_eff = 13.589891949531905 — UNCHANGED by the common (1-eta)
    factor — and the message names the resolved record.

    MUTATION: widen ``rtol`` to 1e2, or drop the ``ratio != 1`` test. MEASURED
    baseline: a 13.6x under-normalisation stops being reported. MUTATION:
    hard-code ``folded_equals_contract=True`` in ``check_accounting_identity``.
    MEASURED baseline: the ledger's own agreement flag stops responding to a
    13.6x disagreement."""
    from CDDF_analysis.hbi_mcmc import forward as FW
    good = FW.fold_mu_fp

    def omitted(log_t, lam_fp, consts):          # the PRE-7707c8e expression
        return good(log_t, lam_fp, consts) / consts.fp_ell_eff

    pk = _fp_only_pack()
    toy, _T, rho = _toy()
    toy.fp_counts = np.zeros((len(REAL_NHAT) - 1, 2), dtype=np.int64)
    toy.fp_counts[0, 0] = 1

    def _agrees():
        return MC.check_accounting_identity(
            toy, row_mass=rho)["candidate_ledger"]["folded_equals_contract"]

    assert _agrees() is True                     # the committed fold
    FW.fold_mu_fp = omitted
    try:
        with pytest.raises(MC.ContractViolation) as e:
            MC.assert_forward_fp_normalisation(pk)
        msg = str(e.value)
        assert "RE-INTRODUCED" in msg
        assert "FP_ELL_EFF_OMITTED" in msg
        # (1-eta) restoration 2026-08-06: was "1086.687" and "14767.961"; x(1-0.005756532459300326) on the FP term
        # (message strings verified by RUNNING the raise: "... contract total
        # is 14682.9492 and the total the COMMITTED forward.fold_mu_fp
        # produces is 1080.4316 ...")
        assert "1080.431" in msg and "14682.949" in msg
        a = MC.fp_normalisation_audit(pk)
        assert a["ratio_contract_over_folded"] == pytest.approx(
            13.589891949531905, rel=1e-9)
        # and the ledger's own agreement flag notices
        assert _agrees() is False
    finally:
        FW.fold_mu_fp = good
    # the guard is quiet again once the fold is back
    MC.assert_forward_fp_normalisation(pk)
    assert _agrees() is True


def test_the_forward_fp_fold_carries_ell_eff_at_every_named_site():
    """Pins the REPAIR to the source, at all four sites 7707c8e touched.

    The predecessor of this test asserted that ``fold_mu`` does NOT mention
    ``fp_ell_eff``. That was true when written and RED BY CONSTRUCTION after
    the repair; inverting it is the regression test the repair deserves.

    MUTATION: delete ``consts.fp_ell_eff`` from ``forward.fold_mu_fp``.
    MEASURED baseline: the first assertion below fails, and so do
    ``test_assert_forward_fp_normalisation_passes_on_the_repaired_fold`` and
    every integration test that reads the folded FP total."""
    def _read(rel):
        with open(os.path.join(REPO, rel)) as fh:
            return fh.read()

    fwd = _read("CDDF_analysis/hbi_mcmc/forward.py")

    # 1. THE definition
    body = fwd.split("def fold_mu_fp(", 1)[1].split("def fold_mu(", 1)[0]
    assert "consts.fp_ell_eff" in body and "consts.fp_w" in body
    # 2. the jitted fold delegates rather than re-typing the expression
    fold = fwd.split("def fold_mu(", 1)[1].split("def fold_mu_reference", 1)[0]
    assert "fold_mu_fp(" in fold
    # 3. the INDEPENDENT numpy oracle carries the factor in its own code
    ref = fwd.split("def fold_mu_reference(", 1)[1]
    assert "fp_ell_eff" in ref
    assert "fold_mu_fp(" not in ref, "the oracle must stay independent"
    # 4. the selftest split and 5. THE GENERATOR
    assert "fold_mu_fp(" in _read("CDDF_analysis/hbi_mcmc/forward_selftest.py")
    assert "fp_ell_eff" in _read("CDDF_analysis/hbi_mcmc/pack.py")

    # the record moved: RESOLVED, not live
    assert "FP_ELL_EFF_OMITTED" in MC.RESOLVED_BY_ID
    assert "FP_ELL_EFF_OMITTED" not in MC.CONTRADICTION_BY_ID
    r = MC.RESOLVED_BY_ID["FP_ELL_EFF_OMITTED"]
    assert any("7707c8e" in s for s in r["fixed_by"])
    assert any("2b436df" in s for s in r["fixed_by"])
    # the before/after is kept, and so is what the repair did NOT buy
    assert "1086.6872" in r["measured_before_after"]
    assert "14767.9614" in r["measured_before_after"]
    assert "22.2236" in r["measured_before_after"]
    assert "CLOSURE STILL FAILS" in r["what_it_did_NOT_fix"]


def test_the_live_contradiction_list_describes_only_live_defects():
    """The list a reader trusts to be CURRENT must not carry a fixed bug.

    MUTATION: put the ``FP_ELL_EFF_OMITTED`` dict back into
    ``KNOWN_CONTRADICTIONS``. MEASURED baseline: the id reappears in
    ``CONTRADICTION_BY_ID`` and this test fails on the second assertion."""
    live = {d["id"] for d in MC.KNOWN_CONTRADICTIONS}
    resolved = {d["id"] for d in MC.RESOLVED_CONTRADICTIONS}
    assert not (live & resolved), "a defect cannot be live AND resolved"
    assert "FP_ELL_EFF_OMITTED" not in live
    # nothing in the live list may advertise itself as fixed
    for d in MC.KNOWN_CONTRADICTIONS:
        assert "RESOLVED" not in d["status"].upper(), d["id"]
    # and the serialized contract carries the history
    ser = MC.contract_dict()
    assert [d["id"] for d in ser["resolved_contradictions"]] == \
        ["FP_ELL_EFF_OMITTED"]
    assert "FP_ELL_EFF_OMITTED" not in [d["id"]
                                        for d in ser["known_contradictions"]]


# ===========================================================================
# 5. the accounting identity, on an injected row mass (no jax, no scratch)
# ===========================================================================
def _toy(rho_val=0.8, n_det=50.0, n_obs_total=0):
    """A 3-bin latent basis on the adopted geometry's TOP three bins is not
    enough to satisfy the geometry rules, so the toy uses the FULL adopted
    edges with hand-set truth, completeness and row mass."""
    B, Kf, S = len(ADOPTED_NTRUE) - 1, 3, 2
    M = len(MOLLY172) - 1
    T = np.zeros((B, Kf, S))
    T[:, :, :] = 10.0                       # 16 * 3 * 2 * 10 = 960 truth systems
    nd = np.full((S, M), float(n_det))      # C = (n+.5)/(100+1)
    nt = np.full((S, M), 100.0)
    counts = np.zeros((len(REAL_NHAT) - 1, Kf, S), dtype=np.int64)
    if n_obs_total:
        counts[0, 0, 0] = int(n_obs_total)
    pk = _fake_pack(n_k=Kf, n_s=S, molly_n_det=nd, molly_n_tot=nt, counts=counts)
    pk.truth_counts_bks = T
    rho = np.full((S, 1, B), float(rho_val))   # KK = 1 (kz all zero)
    return pk, T, rho


def test_truth_ledger_closes_exactly():
    """found_on + found_off + missed == the truth total, to floating point.

    THIS IS A TAUTOLOGY (referee M-A) — see the dedicated test below. It is
    kept because it is still the cheapest detector of a shape/axis crash.

    MUTATION: in ``check_accounting_identity`` change ``missed = T * (1 - C)``
    to ``missed = T``. MEASURED baseline: the residual goes from 0.0 to
    -sum(T * C) = -480.0 on the toy (and to -82316.04 on the ADOPTED 2LPT-0
    pack)."""
    pk, T, rho = _toy()
    r = MC.check_accounting_identity(pk, row_mass=rho)
    t = r["truth_ledger"]
    assert t["residual"] == pytest.approx(0.0, abs=1e-9)
    assert t["n_truth_on_basis"] == pytest.approx(960.0)
    # C == 0.5 exactly by construction; rho == 0.8
    assert t["found_on_grid"] == pytest.approx(960 * 0.5 * 0.8, rel=1e-12)
    assert t["found_off_grid"] == pytest.approx(960 * 0.5 * 0.2, rel=1e-12)
    assert t["missed_P3"] == pytest.approx(960 * 0.5, rel=1e-12)
    assert t["residual_is_a_tautology"] is True


@pytest.mark.parametrize("rho_val", [0.0, 0.8, 1.0])
@pytest.mark.parametrize("n_det", [0.0, 1.0, 99.0])
def test_the_truth_ledger_residual_is_an_algebraic_tautology(rho_val, n_det):
    """REFEREE M-A. ``T.C.rho + T.C.(1-rho) + T.(1-C) == T`` for ANY C, rho, T.

    The residual was advertised in ``ACCOUNTING_IDENTITY.checkable_residuals``
    as 'a nonzero value means the implementation is broken' and as the FIRST of
    'the two numbers a referee should read'. It is neither: it detects a
    shape / broadcast / dtype crash and nothing else.

    MEASURED 2026-08-05: every combination of rho in {0.0, 0.8, 1.0, U(0,1)}
    and molly_n_det in {0, 1, 99} leaves the residual at 0.0 to +-2.3e-13,
    while found_on ranges over 0.000 .. 945.743 and missed over
    14.257 .. 955.248 on the same toy. Nine of those twelve are pinned here.

    MUTATION: none can turn this red by construction — that IS the finding.
    The mutation-tested content lives in
    ``test_the_truth_ledger_value_guards_have_teeth``, and the CLAIM this
    replaces is pinned by ``test_the_tautology_is_declared_in_the_contract``."""
    pk, T, rho = _toy(rho_val=rho_val, n_det=n_det)
    r = MC.check_accounting_identity(pk, row_mass=rho)
    assert abs(r["truth_ledger"]["residual"]) < 1e-9
    # ... while the slots themselves move a lot
    c = (n_det + 0.5) / 101.0
    assert r["truth_ledger"]["found_on_grid"] == pytest.approx(
        960.0 * c * rho_val, rel=1e-9, abs=1e-9)
    assert r["truth_ledger"]["missed_P3"] == pytest.approx(
        960.0 * (1.0 - c), rel=1e-9)


def test_the_tautology_is_declared_in_the_contract():
    """MUTATION: restore 'ZERO by construction; a nonzero value means the
    implementation is broken.' as ``truth_ledger_residual``. MEASURED
    baseline: the contract must SAY the residual is a tautology, must name what
    it does detect, and must point at the guards that carry the content."""
    ci = MC.ACCOUNTING_IDENTITY["checkable_residuals"]
    assert "TAUTOLOGY" in ci["truth_ledger_residual"]
    assert "shape" in ci["truth_ledger_residual"]
    assert "truth_ledger_value_guards" in ci
    assert "TRUTH_LEDGER_RESIDUAL_ADVERTISED_AS_A_TEST" in {
        d["id"] for d in MC.RETRACTIONS}


def test_the_truth_ledger_value_guards_have_teeth():
    """The NON-tautological half of M-A: guards on the VALUES of found_on /
    found_off / missed, not on their sum.

    MUTATION: delete the ``_truth_ledger_value_guards`` call in
    ``check_accounting_identity``. MEASURED baseline: each of the three inputs
    below then sails through — a completeness of 2.0 gives found_on = 1536.0 on
    a truth total of 960.0 with residual 0.0; a transposed rho gives a
    DIFFERENT per-bin split with residual 0.0; an all-NaN rho gives
    found_on = nan with residual = nan."""
    pk, T, rho = _toy()
    guards = MC.check_accounting_identity(
        pk, row_mass=rho)["truth_ledger"]["value_guards"]
    assert guards["n_cells_checked"] == T.size
    assert guards["found_on_le_T_times_C"] is True

    # 1. an impossible completeness (n_det > n_tot) is refused at validation
    with pytest.raises(MC.ContractViolation, match="IMPOSSIBLE COMPLETENESS"):
        MC.check_accounting_identity(
            _toy(n_det=200.0)[0], row_mass=rho)
    # 2. a row mass above 1 is refused
    with pytest.raises(MC.ContractViolation, match="row mass outside"):
        MC.check_accounting_identity(pk, row_mass=np.full_like(rho, 1.5))
    # 3. NaN is refused BEFORE the range test (NaN comparisons are all False)
    with pytest.raises(MC.ContractViolation, match="non-finite"):
        MC.check_accounting_identity(pk, row_mass=np.full_like(rho, np.nan))


def test_the_rho_transpose_is_pinned_by_an_index_round_trip():
    """The one real failure mode the additive residual CAN see: an axis error
    in ``rho[:, kz, :] -> (B, Kf, S)``.

    MUTATION: change the transpose in ``check_accounting_identity`` to
    ``(2, 1, 0)`` -> ``(0, 1, 2)`` (or delete the round-trip guard and feed a
    non-square rho). MEASURED baseline: with B = 16, Kf = 3, S = 2 the wrong
    permutation raises on shape; with a square-in-B-and-S toy it would NOT,
    which is why the guard compares element by element."""
    B, Kf, S = len(ADOPTED_NTRUE) - 1, 3, 2
    pk, T, _ = _toy()
    # a rho whose per-bin values differ, so a permutation is detectable
    rho = np.linspace(0.05, 0.95, S * 1 * B).reshape(S, 1, B)
    r = MC.check_accounting_identity(pk, row_mass=rho)
    g = r["truth_ledger"]["value_guards"]
    assert g["rho_min"] == pytest.approx(0.05)
    assert g["rho_max"] == pytest.approx(0.95)
    # the ledger used rho[s, kz[k], b] for cell (b, k, s)
    on = r["truth_ledger"]["found_on_grid"]
    c = 0.5
    assert on == pytest.approx(float((10.0 * c * rho[:, 0, :]).sum() * Kf),
                               rel=1e-12)


def test_candidate_ledger_partitions_the_on_grid_signal_without_gap_or_overlap():
    """P1 + P2 + P6_above_ceiling == found_on_grid, exactly.

    MUTATION: replace ``part['in_window']`` by ``RP.bins_fully_inside(...)``
    (a boolean). MEASURED baseline: the subtotal then loses half of the
    straddling [21.5,21.7) bin — 0.5 * 10 * 3 * 2 * 0.5 * 0.8 = 12.0 counts on
    the toy — and the sum no longer equals found_on_grid."""
    pk, T, rho = _toy()
    r = MC.check_accounting_identity(pk, row_mass=rho)
    c = r["candidate_ledger"]
    assert (c["P1_scatter_in"] + c["P2_in_window"] + c["P6_above_ceiling"]
            == pytest.approx(r["truth_ledger"]["found_on_grid"], rel=1e-12))
    # 3 pad/edge bins of the 16 are entirely below 19.7 -> P1 gets exactly 3/16
    assert c["P1_scatter_in"] == pytest.approx(3.0 / 16.0 * c["signal_subtotal"],
                                               rel=1e-12)


def test_row_mass_outside_zero_one_is_refused():
    """The trivial efficiency bound rests on rho <= 1.

    MUTATION: delete the row-mass guard. MEASURED baseline: with rho = 1.5 the
    reported found_on_grid (1440.0) EXCEEDS the truth total (960.0) and
    ``efficiency_at_calibration`` reads 1.5 — a nonsense the counting argument
    would then be built on."""
    pk, T, rho = _toy()
    with pytest.raises(MC.ContractViolation, match="row mass"):
        MC.check_accounting_identity(pk, row_mass=np.full_like(rho, 1.5))


# ===========================================================================
# 6. the contract serializes, and claims no authority it does not have
# ===========================================================================
def test_contract_dict_is_json_serializable_and_complete():
    """MUTATION: drop ``accounting_identity`` from ``contract_dict``. MEASURED
    baseline: the contract carries 6 populations, 7 support regions, 6 recorded
    contradictions and 3 recorded retractions."""
    d = MC.contract_dict()
    json.dumps(d, default=str)
    assert len(d["populations"]) == 6
    assert {p["pid"] for p in d["populations"]} == {
        "P1_SCATTER_IN", "P2_IN_WINDOW", "P3_INCOMPLETENESS", "P4_FOREST_FP",
        "P5_TRANSFER", "P6_RESIDUAL"}
    assert len(d["support_map"]) == 7
    assert len(d["known_contradictions"]) == 6
    assert len(d["retractions"]) == 3
    assert "statement" in d["accounting_identity"]
    for p in d["populations"]:
        assert p["support_class"] in MC.SupportClass.ALL
        assert p["parameter_class"] in MC.ParameterClass.ALL
        assert p["side"] in MC.Side.ALL


def test_contract_claims_no_ratified_authority():
    """docs/RULES.md: never write authority=PI / RATIFIED / paper_facing for
    anything not ratified in writing.

    MUTATION: add ``paper_facing=True`` anywhere in ``contract_dict``. MEASURED
    baseline: the serialized contract contains zero occurrences of 'RATIFIED',
    'authority=PI' and 'paper_facing'."""
    s = MC.contract_json()
    assert "RATIFIED" not in s
    assert "authority=PI" not in s
    assert "paper_facing" not in s
    assert MC.contract_dict()["authority"]["newly_ratified_here"] == []


def test_the_measured_candidate_decomposition_is_recorded_and_closes():
    """The per-candidate side, MEASURED 2026-08-05 on the 19.5-floor 2LPT-0
    detection bundle, is recorded in the contract and its four slots sum to the
    window total.

    MUTATION: change ``true_N_below_19p7`` from 4521 to the 17.2-floor bundle's
    8591. MEASURED baseline: 53401 + 4521 + 70 + 9094 == 67086 exactly; the
    17.2-floor split 53401 + 8591 + 70 + 5016 sums to 67078, a DIFFERENT
    detection set (8 rows), so the two must never be mixed."""
    m = MC.MATCHING["is_TP"]["measured_2lpt0_19p5_floor_bundle"]
    w = m["window_nhat_19p7_to_21p6"]
    assert (w["true_N_in_window"] + w["true_N_below_19p7"]
            + w["true_N_above_21p6"] + w["unmatched"]) == w["total"] == 67086
    assert m["n_is_TP"] + m["n_unmatched"] == m["n_on_pack_grid"] == 88071
    assert m["n_is_TP_with_nhi_true_below_19p5"] == 0


def test_the_reporting_window_is_reused_never_redeclared():
    """MUTATION: hardcode 19.7 / 21.6 in matching_contract instead of importing.
    MEASURED baseline: the two constants MUST be the same objects reporting.py
    exports, so a PI change there propagates here without an edit."""
    assert MC.REPORT_FLOOR == RP.NONIDENT_EDGE == 19.7
    assert MC.REPORT_CEILING == RP.RESPONSE_ANCHOR_CEILING == 21.6


def test_the_adopted_baseline_constants_are_reused_from_reporting():
    """REFEREE MINOR (2026-08-05), fixed. ``ADOPTED_BASIS_WIDTH``,
    ``ADOPTED_PAD_FLOOR`` and ``ADOPTED_COMPLETENESS_CONVENTION`` were
    hard-coded literals while the module docstring claimed reuse, and
    ``ADOPTED_PAD_FLOOR`` is LOAD-BEARING as the P1/P6 boundary in ``_p1`` and
    the basis-support test in ``_p3``.

    The fourth, ``ADOPTED_ANALYSIS_WINDOW``, is NOT in reporting.py — reporting
    carries the logN REPORTING_WINDOW, not the spectral window — so it is
    declared here with its real source named, and the docstring no longer
    claims otherwise.

    MUTATION: set ``ADOPTED_PAD_FLOOR = 19.5`` as a literal. MEASURED baseline:
    ``assert_adopted_constants_agree_with_reporting`` raises, and the P1/P6
    boundary silently moves by half a dex without it."""
    ac = RP.ADOPTED_CONFIG
    assert MC.ADOPTED_BASIS_WIDTH is ac["basis_width_dex"]
    assert MC.ADOPTED_PAD_FLOOR is ac["basis_pad_floor"]
    assert MC.ADOPTED_COMPLETENESS_CONVENTION is ac["completeness_below_floor"]
    assert MC.assert_adopted_constants_agree_with_reporting() is True
    assert MC.ADOPTED_ANALYSIS_WINDOW["source"].startswith("extract_pack")
    assert "reporting" not in MC.ADOPTED_ANALYSIS_WINDOW["source"].split("—")[0]


# ===========================================================================
# 7. C3: prior COST, never a feasibility verdict
# ===========================================================================
def test_the_calibration_point_verdict_is_retracted():
    """REFEREE C3 (BLOCKING). The contract used to compute the efficiency at
    ``psi_c = 0, psi_k_delta = 0``, call it 'the SHARPER bound', and emit
    ``feasible_at_calibration_per_contract``. ``psi_c ~ Normal(0, sigma_hat)``
    and ``psi_k_delta ~ Normal(0, fitcov_sd)`` are sample sites with UNBOUNDED
    support (model_a.py:206-209): a point value there is not an upper bound.

    MUTATION: re-add ``feasible_at_calibration_per_contract`` to the
    ``feasibility`` dict. MEASURED baseline: the three retracted keys are
    ABSENT and the retraction is recorded with its measurements."""
    pk, T, rho = _toy(n_obs_total=100)
    f = MC.check_accounting_identity(pk, row_mass=rho)["feasibility"]
    for k in ("feasible_at_calibration_per_contract",
              "feasible_at_calibration_as_implemented",
              "efficiency_attainable_at_calibration"):
        assert k not in f, f"{k} is RETRACTED and must not be emitted"
    assert f["trivial_bound_efficiency"] == 1.0
    assert "efficiency_at_calibration" in f
    assert "POINT" in f["note"] and "bounds nothing" in f["note"]
    r = {d["id"]: d for d in MC.RETRACTIONS}["C3_CALIBRATION_POINT_IS_NOT_A_BOUND"]
    assert "SHARPER bound" in r["withdrawn"]
    assert "UNBOUNDED support" in r["why"]
    assert "0.7190167" in r["measured"] and "0.8572838" in r["measured"]
    assert "infinity" in r["what_survives"] and "2lpt0_v11" in r["what_survives"]


def test_prior_cost_reports_a_cost_and_labels_what_is_a_bound():
    """The replacement quantity. MUTATION: set
    ``efficiency_at_calibration_is_a_bound=True`` in ``prior_cost_audit``, or
    drop ``trivial_bound_is_the_only_bound``. MEASURED baseline on the toy:
    the trivial bound is 1.0, the calibration point is explicitly NOT a bound,
    and both one-block suprema are reported."""
    pk, T, rho = _toy(n_obs_total=100)
    pc = MC.check_accounting_identity(pk, row_mass=rho)["prior_cost"]
    assert pc["trivial_bound_efficiency"] == 1.0
    assert pc["trivial_bound_is_the_only_bound"] is True
    assert pc["efficiency_at_calibration_is_a_bound"] is False
    # C = 0.5, rho = 0.8 on the toy: sup over psi_c alone is rho, over psi_k C
    assert pc["sup_efficiency_psi_c_only"] == pytest.approx(0.8, rel=1e-9)
    assert pc["sup_efficiency_psi_k_only"] == pytest.approx(0.5, rel=1e-9)
    assert pc["efficiency_at_calibration"] == pytest.approx(0.4, rel=1e-9)
    assert "COST, not a verdict" in pc["reading"]


def test_min_prior_chi2_psi_c_agrees_with_its_own_dual_bound():
    """The prior-cost solver reports a primal WITNESS (upper bound) and a
    Lagrangian DUAL (lower bound); when they meet, the value is the minimum.

    MUTATION: return only ``upper_bound`` from the last swept lambda without
    checking ``s >= target`` (i.e. drop the ``if s >= target`` branch).
    MEASURED baseline on this hand-built two-cell problem: upper and lower
    agree to < 1e-6 in relative terms, and the achieved signal equals the
    target; the mutant returns a witness that does NOT reach the target."""
    w = np.array([1000.0, 500.0])
    eta = np.array([0.0, -1.0])
    sigma = np.array([0.10, 0.20])
    s0 = float((w / (1.0 + np.exp(-eta))).sum())
    target = s0 + 80.0
    r = MC.min_prior_chi2_psi_c(w, eta, sigma, target)
    assert r["upper_bound"] == pytest.approx(r["lower_bound"], rel=1e-5)
    assert r["achieved"] >= target - 1e-6
    assert r["upper_bound"] > 0.0
    # below the calibration point it costs nothing
    assert MC.min_prior_chi2_psi_c(w, eta, sigma, s0 - 1.0)["upper_bound"] == 0.0
    # above the supremum it costs infinity — this IS a bound argument
    inf = MC.min_prior_chi2_psi_c(w, eta, sigma, float(w.sum()) + 1.0)
    assert inf["upper_bound"] == np.inf and inf["lower_bound"] == np.inf
    assert "EXCEEDS the supremum" in inf["note"]


def test_fitcov_provenance_travels_with_every_psi_k_cost():
    """REFEREE M-F. ``pack.resp_fitcov_diag`` is absent from all six extracted
    packs, so ``build_consts`` falls back to
    ``_DEFAULT_FITCOV_DIAG = (0.02^2, 0.10^2)`` (forward.py:218) — and the
    psi_k prior cost that closes the adopted geometry scales as 1/fitcov_sd^2.

    MUTATION: delete the ``prior_cost['fitcov_sd_provenance']`` assignment in
    ``check_accounting_identity``. MEASURED baseline: the flag reads False on
    every adopted pack and the status string says UNCALIBRATED."""
    pk, T, rho = _toy(n_obs_total=100)
    p = MC.check_accounting_identity(pk, row_mass=rho)["prior_cost"]
    prov = p["fitcov_sd_provenance"]
    assert prov["pack_carries_resp_fitcov_diag"] is False
    assert "UNCALIBRATED" in prov["status"]
    assert "0.02^2, 0.10^2" in prov["fallback"]
    c = MC.CONTRADICTION_BY_ID["FITCOV_SD_IS_AN_UNCALIBRATED_FALLBACK"]
    assert "all six extracted packs" in c["measured"]


# ===========================================================================
# 8. M-B (corrected): the hostless-census comparison and the negative implied
#    population
# ===========================================================================
def test_fp_hostless_audit_flags_a_mu_fp_above_the_hostless_census():
    """REFEREE M-B, CORRECTED by the Phase-A adversarial review (frozen
    verdict, review/phaseA-adversarial-2026-08-05 @ a11dae0).

    PREVIOUSLY CLAIMED here: "a forest FP is an on-grid candidate with NO
    genuine absorber, so mu_FP cannot exceed the mock's unmatched on-grid
    count" — a physical forest-FP ceiling, reported VIOLATED.
    WHY WRONG: the comparator is the floor-17.2 HOSTLESS class (~92% genuine
    sub-floor detections), not a forest-FP supply; after chance-coincidence
    correction mu_FP/supply = 1.002 on the calibration twin.
    REPLACED BY: the audit still runs the SAME numeric comparison and flags
    mu_FP > census — this test guards that the comparison is REPORTED — but
    the verdict string is "mu_fp_exceeds_hostless_census", never a physical
    claim. EVIDENCE: review_phaseA/fp_normalization/findings.md.

    MEASURED 2026-08-05 on the 2LPT-0 17.2-floor bundle
    (load_and_cut_catalog(truth_nhi_floor=17.2), 11 s): the 88053 on-grid
    op-passing candidates split 15438 / 55058 / 497 / 3200 / 13860 into
    (true N in [19.0,19.7)) / ([19.7,21.6)) / (>= 21.6) / (< 19.0) /
    (unmatched), summing to 88053 exactly. With the (1-eta)-restored fold the
    contract's mu_FP_per_contract = 14682.95 exceeds the 13860 by 822.95.

    MUTATION: change ``exceeds_ceiling`` to ``mu >= 2 * ceil_``. MEASURED
    baseline: the audit stops reporting the excess on 14682.95 vs 13860."""
    ref = MC.FP_CEILING_MEASURED["2lpt0"]
    assert (ref["P1_true_19p0_to_19p7"] + ref["P2_true_19p7_to_21p6"]
            + ref["P6_true_above_21p6"] + ref["P6_true_below_19p0"]
            + ref["unmatched"]) == ref["n_on_grid"] == 88053
    a = MC.fp_ceiling_audit(_fake_pack(mock="2lpt0"),
                            # (1-eta) restoration 2026-08-06: was 14767.961419068737; x(1-0.005756532459300326) on the FP term
                            mu_fp_per_contract=14682.949169806607)
    assert a["ceiling"] == 13860.0
    assert a["exceeds_ceiling"] is True
    # (1-eta) restoration 2026-08-06: was 907.961419; x(1-0.005756532459300326) on the FP term
    assert a["excess"] == pytest.approx(822.949170, abs=1e-4)
    assert "mu_fp_exceeds_hostless_census" in a["status"]
    assert "VIOLATED" not in a["status"]          # the rejected physical claim
    # a mock with no measured census reports UNAVAILABLE, never a silent pass
    b = MC.fp_ceiling_audit(_fake_pack(mock="not_a_mock"),
                            mu_fp_per_contract=1e9)
    assert b["exceeds_ceiling"] is None and "NOT MEASURED" in b["status"]


def test_the_hostless_comparison_is_measured_on_all_three_mocks():
    """The census used to exist for 2LPT-0 only, so ``fp_ceiling_audit``
    returned "NOT MEASURED" on london0 and saclay0 — UNAVAILABLE exactly where
    the excess is largest.

    CORRECTED (Phase-A adversarial review, frozen verdict,
    review/phaseA-adversarial-2026-08-05 @ a11dae0): this test previously
    asserted the excess as a physical forest-FP-ceiling VIOLATION. The
    comparator is the floor-17.2 hostless class (~92% genuine sub-floor
    detections); on the calibration twin the excess is an estimand artifact
    resolved by chance-coincidence correction (mu_FP/supply = 1.002), and
    cross-mock it reflects the unresolved transport systematic (Layer C).
    The COMPARISON stays measured and reported; the claim does not. Evidence:
    review_phaseA/fp_normalization/findings.md.

    Census RE-MEASURED 2026-08-05, 17.2-truth-floor bundle
    (load_and_cut_catalog(truth_nhi_floor=17.2, host_truth_floor=17.2) +
    _snap_off_molly_edges, ~11 s per mock), op mask, on the pack grid
    (N_hat in [19.5,22.4), z in [2.0,3.5)). Every partition sums EXACTLY to its
    on-grid total:

        mock     on-grid  P1[19.0,19.7) P2[19.7,21.6)  >=21.6  <19.0  unmatched
        2lpt0      88053      15438        55058         497    3200     13860
        london0    87831      15834        59186         602    2611      9598
        saclay0    86745      15733        57213         539    2668     10592

    mu_FP re-derived 2026-08-06 by running ``fp_normalisation_audit`` on each
    adopted pack with the (1-eta)-restored fold:

        mock      mu_FP_per_contract     excess        excess/census
        2lpt0        14682.949170        + 822.949         + 5.94%
        london0      14631.661639        +5033.662         +52.45%
        saclay0      14622.400845        +4030.401         +38.05%

    MUTATION: delete the london0 and saclay0 entries from
    ``FP_CEILING_MEASURED``. MEASURED baseline: both come back "NOT MEASURED"
    and the two largest excesses (+52% and +38%) stop being reported."""
    # (1-eta) restoration 2026-08-06: mu was 14767.961419068737 / 14716.376940133037
    # / 14707.062527716187; x(1-0.005756532459300326) on the FP term
    want = {"2lpt0": (88053, 15438, 55058, 497, 3200, 13860, 14682.949169806607),
            "london0": (87831, 15834, 59186, 602, 2611, 9598, 14631.66163859828),
            "saclay0": (86745, 15733, 57213, 539, 2668, 10592, 14622.400844898844)}
    assert set(MC.FP_CEILING_MEASURED) == set(want)
    for m, (tot, p1, p2, hi, lo, unm, mu) in want.items():
        r = MC.FP_CEILING_MEASURED[m]
        assert (r["n_on_grid"], r["P1_true_19p0_to_19p7"],
                r["P2_true_19p7_to_21p6"], r["P6_true_above_21p6"],
                r["P6_true_below_19p0"], r["unmatched"]) == \
            (tot, p1, p2, hi, lo, unm), m
        assert p1 + p2 + hi + lo + unm == tot, m
        assert r["mu_fp_per_contract"] == mu, m
        a = MC.fp_ceiling_audit(_fake_pack(mock=m), mu_fp_per_contract=mu)
        assert a["ceiling"] == float(unm), m
        assert a["exceeds_ceiling"] is True, m
        assert a["excess"] == pytest.approx(mu - unm, rel=1e-9), m
        assert "mu_fp_exceeds_hostless_census" in a["status"], m
        assert "VIOLATED" not in a["status"], m   # the rejected physical claim
    # the two transfer mocks carry the larger excesses — cross-mock this is
    # the Layer-C transport systematic, not a forest-FP statement
    # (1-eta) restoration 2026-08-06: ratios were 1.53328 / 1.38851; x(1-0.005756532459300326) on the FP term
    assert MC.fp_ceiling_audit(
        _fake_pack(mock="london0"),
        mu_fp_per_contract=14631.66163859828)["ratio"] == pytest.approx(
        1.52445, abs=1e-5)
    assert MC.fp_ceiling_audit(
        _fake_pack(mock="saclay0"),
        mu_fp_per_contract=14622.400844898844)["ratio"] == pytest.approx(
        1.38051, abs=1e-5)


def test_the_fp_z_shape_one_sided_support_is_recorded():
    """Occurrence #12 of the one-sided-support class, MEASURED 2026-08-05 on
    the committed loa-0 FP catalogue (3255 raw rows; op = SNR_REDSIDE > 2 &
    P_DLA > 0.99 -> 2704; + lam_rest >= 1025 A -> 2378; + z_DLA in [2.0,3.5)
    -> 2318):

        on the pack grid [19.5,22.4) :   89   ( 3.8%)
        BELOW the observed floor     : 2229   (96.2%)
        at or above 22.4             :    0

        coarse z            [2.0,2.5)  [2.5,3.0)  [3.0,3.5)
        in-support  n=  89     43         36         10    = .4831/.4045/.1124
        below-floor n=2229   1497        588        144    = .6716/.2638/.0646
        2x3 homogeneity chi2(2) = 13.8066, p = 0.0010045

    The fold imposes neither shape: the dX allocation gives
    .5985/.2968/.1048 on the adopted 2LPT-0 pack.

    AND the allocation is invisible to the ratified gate: sum_k fp_E[k,s] == 1
    on every populated stratum, so the window chi2/dof (22.2236 / 28.3934 /
    25.7723) is unchanged to <= 9.65e-16 RELATIVE under both measured
    alternatives, while the (unratified) by_z |z| arm moves 7.6404 -> 3.7459 /
    10.2587 on 2lpt0.

    MUTATION: delete the ``invisible_to_the_gate`` field. MEASURED baseline:
    the record still names a real support mismatch but stops saying that no
    measurement of it can move the closure verdict, which is the part that
    decides whether the work is worth doing."""
    c = MC.CONTRADICTION_BY_ID["FP_Z_SHAPE_DIFFERS_ACROSS_THE_OBSERVED_FLOOR"]
    m = c["measured"]
    for s in ("2318", "2229", "96.2%", "0.4831 / 0.4045 / 0.1124",
              "0.6716 / 0.2638 / 0.0646", "13.8066", "0.0010045"):
        assert s in m, s
    g = c["invisible_to_the_gate"]
    for s in ("22.2236", "28.3934", "25.7723", "9.65e-16", "7.6404",
              "3.7459", "10.2587", "restated-but-not-decided"):
        assert s in g, s
    # it is NOT claimed to be bit-identical, because it is not
    assert "NOT bit-identical" in g
    assert "one-sided-support" in c["effect"] and "#12" in c["effect"]
    assert "NO CHANGE PROPOSED" in c["status"]
    # and the fold's z-allocation is a named quantity, not an unlisted default
    assert "fp_E_alloc" in MC.QUANTITIES


def test_a_negative_implied_unsupported_population_is_flagged():
    """REFEREE M-B. ``P6_unsupported_implied_per_contract`` is a COUNT of
    on-grid candidates left over for P6's unsupported sub-slots. It came out
    NEGATIVE on two of the three adopted packs (MEASURED 2026-08-05: london0
    -3287.42, saclay0 -2079.08; 2lpt0 is +882.30) and was emitted with no
    comment at all.

    MUTATION: delete the ``NEGATIVE_IMPLIED_P6_UNSUPPORTED`` block in
    ``check_accounting_identity``. MEASURED baseline on this toy ((1-eta)
    restoration 2026-08-06: the FP slot was 165.93): the modelled slots
    predict 384.0 + 164.98 = 548.98 against n_obs = 100, so the implied
    population is -448.98 and the flag must fire; ``strict=True`` must raise."""
    pk, T, rho = _toy(n_obs_total=100)
    pk.fp_counts = np.zeros((len(REAL_NHAT) - 1, 2), dtype=np.int64)
    pk.fp_counts[0, 0] = 1
    r = MC.check_accounting_identity(pk, row_mass=rho)
    c = r["candidate_ledger"]
    assert c["P6_unsupported_implied_per_contract"] < 0
    assert c["P6_unsupported_implied_is_negative"] is True
    ids = [f["id"] for f in r["flags"]]
    assert "NEGATIVE_IMPLIED_P6_UNSUPPORTED" in ids
    with pytest.raises(MC.ContractViolation,
                       match="NEGATIVE_IMPLIED_P6_UNSUPPORTED"):
        MC.check_accounting_identity(pk, row_mass=rho, strict=True)
    # ... and a healthy pack raises no flag
    ok, _, rho_ok = _toy(n_obs_total=100000)
    assert MC.check_accounting_identity(ok, row_mass=rho_ok)["flags"] == []


# ===========================================================================
# 9. M-D: fail CLOSED, not open
# ===========================================================================
def test_validate_fails_closed_on_nan_counts():
    """REFEREE M-D. MEASURED before the fix: a pack whose ``counts`` are all
    NaN passed validation, produced ``n_obs = nan``, and yielded
    ``feasible = bool(nan <= t_tot) = False`` — a NaN pack silently licensed
    the very 'INFEASIBLE' verdict this module exists to support.

    MUTATION: delete the ``_assert_finite(c, 'pack.counts')`` call (rule 6a).
    MEASURED baseline: the call below stops raising and n_obs comes back nan."""
    pk = _fake_pack()
    pk.counts = np.full((len(REAL_NHAT) - 1, 3, 2), np.nan)
    with pytest.raises(MC.ContractViolation, match="pack.counts contains"):
        MC.validate_pack_against_contract(pk)
    # NaN really does slip past a bare range test — this is the mechanism
    assert not (np.nan < 0) and not (np.nan > 1)


def test_validate_fails_closed_on_impossible_completeness():
    """REFEREE M-D. MEASURED before the fix: ``molly_n_det = 200`` against
    ``molly_n_tot = 100`` had NO guard at all and gave
    ``eta_hat = log(200.5 / -99.5) = nan``, silently NaN-ing every downstream
    completeness.

    MUTATION: delete rule 8. MEASURED baseline: the call below stops raising
    and ``_eta_hat`` returns nan for every cell."""
    with pytest.raises(MC.ContractViolation, match="IMPOSSIBLE COMPLETENESS"):
        MC.validate_pack_against_contract(
            _fake_pack(molly_n_det=np.full((2, 12), 200.0),
                       molly_n_tot=np.full((2, 12), 100.0)))
    with pytest.raises(MC.ContractViolation, match="non-finite"):
        MC.validate_pack_against_contract(
            _fake_pack(molly_n_det=np.full((2, 12), np.nan)))
    with pytest.raises(MC.ContractViolation, match="n_det >= 0"):
        MC.validate_pack_against_contract(
            _fake_pack(molly_n_det=np.full((2, 12), -1.0)))


def test_validate_fails_closed_on_nan_row_mass_and_truth():
    """REFEREE M-D. MEASURED before the fix: an all-NaN ``row_mass`` passed the
    ``rho < -1e-12 or rho > 1+1e-9`` guard, because every comparison with NaN
    is False, and produced ``found_on = nan``, ``residual = nan``.

    MUTATION: move ``_assert_finite(rho, ...)`` AFTER the range test in
    ``_truth_ledger_value_guards``. MEASURED baseline: the first call below
    stops raising."""
    pk, T, rho = _toy()
    with pytest.raises(MC.ContractViolation, match="row mass"):
        MC.check_accounting_identity(pk, row_mass=np.full_like(rho, np.nan))
    bad = _fake_pack()
    bad.truth_counts_bks = np.full((16, 3, 2), np.nan)
    with pytest.raises(MC.ContractViolation, match="truth_counts_bks contains"):
        MC.validate_pack_against_contract(bad)


# ===========================================================================
# 10. M-E: the BAL magnitude is RETRACTED
# ===========================================================================
def test_the_bal_magnitude_is_retracted_and_the_null_is_recorded():
    """REFEREE M-E. The contract declared the comment at
    ``build_loa0_fp_product.py:231`` ('loa-0 is BAL-free') FALSE, booked a PI
    item to move P4 to 70/1904, and claimed a '7.35% HIGH' effect.

    The premise was tested on the FULL op+lya loa-0 FP catalogue. MEASURED
    2026-08-05 (3255 raw FP rows; op = SNR_REDSIDE>2 & P_DLA>0.99;
    lya = lam_rest >= 1025 A; BAL set = loa-124 bal_cat.fits, 193737 unique
    TARGETIDs; 351 BAL / 1904 non-BAL loa-0 sightlines with SNR>2):

        op      : N=2704  FP/sightline BAL 1.1880  nonBAL 1.2012  ratio 0.98908  z = -0.21
        op+lya  : N=2378  FP/sightline BAL 1.0570  nonBAL 1.0541  ratio 1.00274  z = +0.05
        the 89  : BAL 19 vs expected 13.853 (sd 3.420)                           z = +1.50

    There is no BAL signal in loa-0. The 19-of-89 excess is a 1.50-sigma
    fluctuation; adopting 70/1904 would discard 351 valid sightlines and 19
    valid FPs, inflating the FP Poisson variance ~18%, to chase it.

    MUTATION: restore ``effect='P4 is 7.35% HIGH ...'`` and the old PI item.
    MEASURED baseline: the entry must carry a ``retracted`` block naming the
    2378-event test, must NOT claim the site comment is FALSE, and the PI item
    must say no change is proposed."""
    c = MC.CONTRADICTION_BY_ID["BAL_VETO_ONE_SIDED_IN_FP_W"]
    assert "retracted" in c
    assert "2378" in c["retracted"]["why"]
    assert "1.00274" in c["retracted"]["why"]
    assert "1.50-sigma" in c["retracted"]["why"]
    assert "7.35" not in c["effect"]
    assert "NO measured effect" in c["effect"]
    assert "NO CHANGE PROPOSED" in c["status"]
    # the support-matching OBSERVATION survives, with its counts
    assert "351" in c["measured"] and "1904" in c["measured"]
    assert c["contract"].startswith("P4's rate scale")
    # the PI item no longer proposes 70/1904
    pi = "\n".join(MC.PI_CHECKPOINT_ITEMS)
    assert "RETRACTED" in pi and "NO change to 70/1904 is proposed" in pi
    assert "BAL_VETO_MAGNITUDE" in {d["id"] for d in MC.RETRACTIONS}


# ===========================================================================
# 11. integration on the REAL adopted mock packs
# ===========================================================================
@pytest.mark.skipif(not os.path.exists(ADOPTED_PACK),
                    reason="adopted window-study pack not on this filesystem")
def test_identity_on_the_adopted_2lpt0_pack():
    """The referee-facing run. MEASURED 2026-08-06 ((1-eta) restoration; the
    2026-08-05 values are noted inline) on
    modelA_pack_2lpt0_winlya_only_pad19p0_molly172_bw0p2.npz, resp_clamp=both:

        N_obs                       88071
        truth on basis             101949
        found_on_grid            72420.741
        found_off_grid            9895.303
        missed (P3)              19632.956
        truth-ledger residual        0.000     <- a TAUTOLOGY, not a test
        P1 scatter-in            18033.092
        P2 in-window             53847.300
        P6 above ceiling           540.349
        P4 as FOLDED             14682.949   <- was 14767.961; x(1-eta)
        P4 per contract          14682.949   <- was 14767.961; x(1-eta)
        P4 if ell_eff omitted     1080.432   <- COUNTERFACTUAL, pre-7707c8e
        candidate residual (folded)   -967.310   <- was -882.298
        candidate residual (contr)    -967.310   <- was -882.298
        candidate residual (ctf)   -14569.827   <- was -14563.572
        efficiency AT CALIBRATION    0.7103624   <- a POINT, not a bound;
                                                    signal-side, eta-UNCHANGED
        efficiency required (contr)  0.7198506   <- was 0.7190167

    Every signal-side row is bit-unchanged by the restoration; only the FP
    term and its downstream residuals moved, by x(1-0.005756532459300326).

    MUTATION: set ``rho_bks`` to 1 everywhere. MEASURED baseline: found_off
    collapses 9895.303 -> 0.0 and the candidate residual (contract) moves
    into surplus (was -882.298 -> +9013.005 pre-restoration)."""
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.pack import attach_fp_eta_bands as _aeta
    load_pack = (lambda _f: (lambda *a, **k: _aeta(_f(*a, **k))))(load_pack)
    pk = load_pack(ADOPTED_PACK, allow_nonstandard_grid=True)
    r = MC.check_accounting_identity(pk)
    t, c, f = r["truth_ledger"], r["candidate_ledger"], r["feasibility"]

    assert t["residual"] == pytest.approx(0.0, abs=1e-6)
    assert t["n_truth_on_basis"] == 101949.0
    assert c["n_obs"] == 88071.0
    assert t["found_on_grid"] == pytest.approx(72420.741, abs=1e-2)
    assert t["found_off_grid"] == pytest.approx(9895.303, abs=1e-2)
    assert t["missed_P3"] == pytest.approx(19632.956, abs=1e-2)
    assert c["P1_scatter_in"] == pytest.approx(18033.092, abs=1e-2)
    assert c["P2_in_window"] == pytest.approx(53847.300, abs=1e-2)
    assert c["P6_above_ceiling"] == pytest.approx(540.349, abs=1e-2)
    # the fold and the contract now AGREE; the pre-repair number survives only
    # as an explicitly labelled counterfactual
    # (1-eta) restoration 2026-08-06: was 14767.961; x(1-0.005756532459300326) on the FP term
    assert c["as_folded"]["P4_forest_fp"] == pytest.approx(14682.949, abs=1e-2)
    # (1-eta) restoration 2026-08-06: was -882.298; x(1-0.005756532459300326) on the FP term
    assert c["as_folded"]["residual"] == pytest.approx(-967.310, abs=1e-2)
    assert c["folded_equals_contract"] is True
    # (1-eta) restoration 2026-08-06: was -882.298; x(1-0.005756532459300326) on the FP term
    assert c["per_contract"]["residual"] == pytest.approx(-967.310, abs=1e-2)
    # (1-eta) restoration 2026-08-06: was -14563.572; x(1-0.005756532459300326) on the FP term
    assert c["if_ell_eff_omitted"]["residual"] == pytest.approx(-14569.827,
                                                               abs=1e-2)
    assert "COUNTERFACTUAL" in c["if_ell_eff_omitted"]["note"]
    # signal-side: eta-UNCHANGED by construction (the restoration touches
    # only the FP term)
    assert f["efficiency_at_calibration"] == pytest.approx(0.7103624, abs=1e-6)
    # (1-eta) restoration 2026-08-06: was 0.7190167; x(1-0.005756532459300326) on the FP term
    assert f["efficiency_required_per_contract"] == pytest.approx(0.7198506,
                                                                  abs=1e-6)


@pytest.mark.skipif(not os.path.exists(ADOPTED_PACK),
                    reason="adopted window-study pack not on this filesystem")
def test_the_adopted_geometry_closes_inside_one_sigma_of_prior_cost():
    """REFEREE C3, the positive half. On the ADOPTED geometry the counts are
    reached at a SMALL prior cost, so no infeasibility follows.

    MEASURED 2026-08-06 with the (1-eta)-restored FP fold on the adopted
    2LPT-0 pack (gap = 967.310 counts; was 882.298 — the gap grows by exactly
    the 85.012-count FP reduction; the two suprema and the calibration point
    are signal-side and eta-UNCHANGED):

        efficiency at calibration        0.7103624   (a POINT)
        efficiency REQUIRED per contract 0.7198506   (was 0.7190167)
        sup over psi_c alone (C -> 1)    0.8572838
        sup over psi_k alone (rho -> 1)  0.8074237
        min prior chi2 in psi_c          130.3258  (11.416 sigma, 96 free;
                                                    was 107.8667)
        psi_k_delta[1] uniform witness   -0.7577637 prior-sd (was -0.7056356)
                                         chi2 5.1679  ->  2.273 sigma
        loa-0 FP Poisson 1 sd            1565.401 counts -> gap = 0.6179 sigma
        uniform transfer shift           delta -0.0681502, chi2 0.92360
                                                          ->  0.9610 sigma
        cheapest declared direction      0.6179 sigma

    The required efficiency sits FAR BELOW both one-block suprema and three
    separate declared directions close it inside ~1 sigma. The retracted
    verdict said INFEASIBLE.

    MUTATION: in ``prior_cost_audit`` compute ``sup_c`` as ``(T*C*rho).sum()``
    (i.e. forget to send C -> 1). MEASURED baseline: sup_efficiency_psi_c_only
    collapses 0.8572838 -> 0.7103624 and this test fails on the first
    assertion."""
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.pack import attach_fp_eta_bands as _aeta
    load_pack = (lambda _f: (lambda *a, **k: _aeta(_f(*a, **k))))(load_pack)
    pk = load_pack(ADOPTED_PACK, allow_nonstandard_grid=True)
    p = MC.check_accounting_identity(pk)["prior_cost"]
    # the two suprema are signal-side: eta-UNCHANGED by construction
    assert p["sup_efficiency_psi_c_only"] == pytest.approx(0.8572838, abs=1e-6)
    assert p["sup_efficiency_psi_k_only"] == pytest.approx(0.8074237, abs=1e-6)
    assert p["efficiency_required"] < p["sup_efficiency_psi_c_only"]
    assert p["efficiency_required"] < p["sup_efficiency_psi_k_only"]
    # (1-eta) restoration 2026-08-06: was 882.298; x(1-0.005756532459300326) on the FP term
    assert p["gap_counts"] == pytest.approx(967.310, abs=1e-2)
    # (1-eta) restoration 2026-08-06: was 107.8667; x(1-0.005756532459300326) on the FP term (larger gap -> larger cost)
    assert p["psi_c"]["min_prior_chi2_upper_bound"] == pytest.approx(130.3258,
                                                                     abs=1e-3)
    assert p["psi_c"]["min_prior_chi2_lower_bound"] == pytest.approx(130.3258,
                                                                     abs=1e-3)
    k = p["psi_k_delta"]
    # (1-eta) restoration 2026-08-06: was -0.7056356; x(1-0.005756532459300326) on the FP term
    assert k["witness_alpha_in_prior_sd"] == pytest.approx(-0.7577637, abs=1e-5)
    # (1-eta) restoration 2026-08-06: was 4.4813; x(1-0.005756532459300326) on the FP term
    assert k["witness_prior_chi2"] == pytest.approx(5.1679, abs=1e-3)
    # (1-eta) restoration 2026-08-06: was 2.1169; x(1-0.005756532459300326) on the FP term
    assert k["witness_mahalanobis_sigma"] == pytest.approx(2.2733, abs=1e-3)
    assert k["is_a_witness_not_the_minimum"] is True
    assert k["fitcov_sd_provenance"]["pack_carries_resp_fitcov_diag"] is False
    assert p["fp_total_poisson"]["one_sd_counts"] == pytest.approx(1565.401,
                                                                   abs=1e-2)
    # (1-eta) restoration 2026-08-06: was 0.5636; x(1-0.005756532459300326) on the FP term
    assert p["fp_total_poisson"]["gap_in_sd"] == pytest.approx(0.6179, abs=1e-3)
    # (1-eta) restoration 2026-08-06: was 0.8687; x(1-0.005756532459300326) on the FP term
    assert p["transfer_t"]["witness_mahalanobis_sigma"] == pytest.approx(
        0.9610, abs=1e-3)
    # (1-eta) restoration 2026-08-06: was 0.5636; x(1-0.005756532459300326) on the FP term
    assert p["cheapest_declared_direction_sigma"] == pytest.approx(0.6179,
                                                                   abs=1e-3)
    assert p["cheapest_declared_direction_sigma"] < 1.0


@pytest.mark.skipif(not os.path.exists(ADOPTED_PACK),
                    reason="adopted window-study pack not on this filesystem")
def test_the_fp_hostless_comparison_is_reported_on_the_adopted_2lpt0_pack():
    """REFEREE M-B, on the real pack — CORRECTED interpretation.

    PREVIOUSLY (as ``test_the_fp_ceiling_is_violated_on_the_adopted_2lpt0_
    pack``) this asserted a "VIOLATED / EXCEEDS by 907.96 / +6.55%" reading:
    a physical forest-FP ceiling. The 2026-08-06 Phase-A adversarial review
    REJECTED that interpretation (frozen verdict,
    review/phaseA-adversarial-2026-08-05 @ a11dae0): the comparator is the
    floor-17.2 HOSTLESS class (~92% genuine sub-floor detections, not a
    forest-FP ceiling), and after chance-coincidence correction
    mu_FP/supply = 1.002 on 2LPT-0. The mu_FP > hostless excess on the twin
    is an ESTIMAND ARTIFACT resolved by that correction; cross-mock it
    reflects the unresolved transport systematic (Layer C). Evidence:
    review_phaseA/fp_normalization/findings.md.

    What this test still guards: the audit REPORTS these estimand
    comparisons (the census, the eta-restored mu_FP, the excess, the flag).

    MEASURED 2026-08-06 ((1-eta)-restored fold): mu_FP_per_contract =
    14682.949 vs the mock's 13860 unmatched on-grid candidates -> excess
    822.949, and the flag fires.

    MUTATION: drop the ``MU_FP_EXCEEDS_THE_HOSTLESS_CENSUS`` flag block.
    MEASURED baseline: ``flags`` goes from one entry to zero on this pack."""
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.pack import attach_fp_eta_bands as _aeta
    load_pack = (lambda _f: (lambda *a, **k: _aeta(_f(*a, **k))))(load_pack)
    pk = load_pack(ADOPTED_PACK, allow_nonstandard_grid=True)
    r = MC.check_accounting_identity(pk)
    a = r["fp_ceiling"]
    assert a["mock"] == "2lpt0"
    assert a["ceiling"] == 13860.0
    # (1-eta) restoration 2026-08-06: was 14767.9614; x(1-0.005756532459300326) on the FP term
    assert a["mu_fp_per_contract"] == pytest.approx(14682.9492, abs=1e-3)
    # (1-eta) restoration 2026-08-06: was 907.9614; x(1-0.005756532459300326) on the FP term
    assert a["excess"] == pytest.approx(822.9492, abs=1e-3)
    assert a["exceeds_ceiling"] is True
    assert "mu_fp_exceeds_hostless_census" in a["status"]
    assert "VIOLATED" not in a["status"]          # the rejected physical claim
    assert [f["id"] for f in r["flags"]] == ["MU_FP_EXCEEDS_THE_HOSTLESS_CENSUS"]


@pytest.mark.skipif(not os.path.exists(UNPADDED_PACK),
                    reason="unpadded v1.1 pack not on this filesystem")
def test_the_counting_argument_on_the_unpadded_pack():
    """The D1 counting argument, run through the contract. MEASURED 2026-08-06
    with the (1-eta)-restored fold (2026-08-05 values noted inline) on
    modelA_pack_2lpt0_v11.npz (0.1-dex basis, NO pad):

        N_obs 88071 > truth on basis 73610
        efficiency required, FP if ell_eff omitted : 1.18178  <- IMPOSSIBLE
                                                       (was 1.18168; the
                                                        trivial bound;
                                                        a COUNTERFACTUAL now)
        efficiency required, FP per contract       : 0.99698  (was 0.99583)
        sup over psi_c alone (C -> 1)          : 0.99384   <- BELOW the
                                                              requirement
        min prior chi2 in psi_c                : infinity

    So with the under-normalised FP the classic '88071 > 73610 therefore no
    parameter closes it' is exact against the TRIVIAL bound; with the
    contract's FP the refutation still holds, but as a SUPREMUM argument —
    even C == 1 at infinite prior cost falls short.

    MUTATION: pass ``require_pad=True``. MEASURED baseline: the pack is refused
    before any number is produced."""
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.pack import attach_fp_eta_bands as _aeta
    load_pack = (lambda _f: (lambda *a, **k: _aeta(_f(*a, **k))))(load_pack)
    pk = load_pack(UNPADDED_PACK, allow_nonstandard_grid=True)
    r = MC.check_accounting_identity(
        pk, require_pad=False, require_measured_sub_floor_completeness=False)
    f, p = r["feasibility"], r["prior_cost"]
    assert r["truth_ledger"]["residual"] == pytest.approx(0.0, abs=1e-6)
    assert r["truth_ledger"]["n_truth_on_basis"] == 73610.0
    assert r["candidate_ledger"]["n_obs"] == 88071.0
    # (1-eta) restoration 2026-08-06: was 1.18168; x(1-0.005756532459300326) on the FP term
    assert f["efficiency_required_if_ell_eff_omitted"] == pytest.approx(
        1.1817765, abs=1e-4)
    # (1-eta) restoration 2026-08-06: was 0.99583; x(1-0.005756532459300326) on the FP term
    # (recomputed by running this code path: 0.9969847959542641)
    assert f["efficiency_required_per_contract"] == pytest.approx(0.9969848, abs=1e-4)
    # (1-eta) restoration 2026-08-06: was 0.99583; x(1-0.005756532459300326) on the FP term
    assert f["efficiency_required_as_folded"] == pytest.approx(0.9969848, abs=1e-4)
    assert f["feasible_if_ell_eff_omitted"] is False   # the trivial bound
    assert f["feasible_per_contract"] is True
    assert f["feasible_as_folded"] is True
    # the supremum argument, which is what actually refutes it
    assert p["sup_efficiency_psi_c_only"] == pytest.approx(0.9938370, abs=1e-6)
    assert p["efficiency_required"] > p["sup_efficiency_psi_c_only"]
    assert p["psi_c"]["min_prior_chi2_upper_bound"] == np.inf
    assert p["psi_c"]["min_prior_chi2_lower_bound"] == np.inf
    assert "EXCEEDS the supremum" in p["psi_c"]["detail"]["note"]


@pytest.mark.skipif(
    not all(os.path.exists(os.path.join(V11DIR, f"modelA_pack_{m}_v11.npz"))
            for m in ("2lpt0", "london0", "saclay0")),
    reason="unpadded v1.1 packs not on this filesystem")
def test_the_unpadded_refutation_is_a_prior_cost_argument_on_all_three_mocks():
    """REFEREE C3, what SURVIVES. On the UNPADDED v1.1 packs the refutation is
    materially stronger than the adopted-geometry one — and it is a PRIOR-COST
    argument (chi2 ~ 1e4), not a bound argument, on two of the three.

    MEASURED 2026-08-06 with the (1-eta)-restored fold (the smaller mu_FP
    raises every required efficiency, so the refutation gets STRONGER; the
    suprema are signal-side and eta-UNCHANGED; 2026-08-05 values noted):

        pack          eff req (contract FP)  sup psi_c alone   min prior chi2 psi_c
        2lpt0_v11        0.9969848            0.9938370 (< req)   infinity
                         (was 0.9958299)
        london0_v11      0.9396406            0.9939876           11168.1 (105.68 s)
                         (was 0.9385533)                          (was 10757.4)
        saclay0_v11      0.9570638            0.9937649           20902.1 (144.58 s)
                         (was 0.9559406)                          (was 20085.8)

    MUTATION: return ``upper_bound=0.0`` whenever the bisection finds no
    feasible lambda. MEASURED baseline: the two finite costs collapse to 0 and
    the 1e4-scale statement disappears."""
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.pack import attach_fp_eta_bands as _aeta
    load_pack = (lambda _f: (lambda *a, **k: _aeta(_f(*a, **k))))(load_pack)
    want = {
        # (1-eta) restoration 2026-08-06 (all recomputed by running this code
        # path): req was 0.9958299 / 0.9385533 / 0.9559406, chi2 was
        # 10757.4 / 20085.8, mah was 103.72 / 141.72;
        # x(1-0.005756532459300326) on the FP term
        "2lpt0": (0.9969848, 0.9938370, np.inf, None),
        "london0": (0.9396406, 0.9939876, 11168.065, 105.68),
        "saclay0": (0.9570638, 0.9937649, 20902.128, 144.58),
    }
    for m, (req, sup, chi2, mah) in want.items():
        pk = load_pack(os.path.join(V11DIR, f"modelA_pack_{m}_v11.npz"),
                       allow_nonstandard_grid=True)
        p = MC.check_accounting_identity(
            pk, require_pad=False,
            require_measured_sub_floor_completeness=False)["prior_cost"]
        assert p["efficiency_required"] == pytest.approx(req, abs=1e-6), m
        assert p["sup_efficiency_psi_c_only"] == pytest.approx(sup, abs=1e-6), m
        if np.isinf(chi2):
            assert p["psi_c"]["min_prior_chi2_upper_bound"] == np.inf, m
            assert p["efficiency_required"] > p["sup_efficiency_psi_c_only"], m
        else:
            assert p["psi_c"]["min_prior_chi2_upper_bound"] == pytest.approx(
                chi2, rel=1e-4), m
            assert p["psi_c"]["min_prior_chi2_lower_bound"] == pytest.approx(
                chi2, rel=1e-4), m
            assert p["psi_c"]["min_mahalanobis_sigma"] == pytest.approx(
                mah, abs=1e-2), m
            assert p["psi_c"]["min_prior_chi2_upper_bound"] > 1e4, m


@pytest.mark.skipif(not os.path.exists(ADOPTED_PACK),
                    reason="adopted window-study packs not on this filesystem")
def test_the_negative_implied_population_shows_up_on_the_real_packs():
    """REFEREE M-B, on the real packs. MEASURED 2026-08-06 with the
    (1-eta)-restored fold on the adopted 0.2-dex/pad-19.0 packs:
    P6_unsupported_implied_per_contract is 2lpt0 +967.31, london0 -3202.71,
    saclay0 -1994.42 (each moved UP by exactly its mock's (1-eta) FP
    reduction; was +882.30 / -3287.42 / -2079.08). Two of the three still
    imply a NEGATIVE number of candidates and were once emitted with no
    comment.

    MUTATION: delete the negative-flag block. MEASURED baseline: london0 and
    saclay0 stop reporting the NEGATIVE_IMPLIED flag."""
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.pack import attach_fp_eta_bands as _aeta
    load_pack = (lambda _f: (lambda *a, **k: _aeta(_f(*a, **k))))(load_pack)
    # (1-eta) restoration 2026-08-06: was 882.298 / -3287.42 / -2079.08
    # (recomputed by running this code path); x(1-0.005756532459300326) on
    # the FP term
    want = {"2lpt0": 967.310, "london0": -3202.71, "saclay0": -1994.42}
    seen_negative = 0
    for m, v in want.items():
        p = os.path.join(
            PACKDIR, f"modelA_pack_{m}_winlya_only_pad19p0_molly172_bw0p2.npz")
        if not os.path.exists(p):
            pytest.skip(f"{m} adopted pack absent")
        r = MC.check_accounting_identity(load_pack(p, allow_nonstandard_grid=True))
        got = r["candidate_ledger"]["P6_unsupported_implied_per_contract"]
        assert got == pytest.approx(v, abs=1e-2), m
        if got < 0:
            seen_negative += 1
            assert "NEGATIVE_IMPLIED_P6_UNSUPPORTED" in [
                f["id"] for f in r["flags"]], m
    assert seen_negative == 2


@pytest.mark.skipif(not os.path.exists(ADOPTED_PACK),
                    reason="adopted window-study packs not on this filesystem")
def test_the_frozen_2lpt0_calibration_is_bit_identical_across_the_three_mocks():
    """Cross-mock spread therefore measures TRANSFER, never kernel uncertainty.

    MUTATION: compare ``counts`` instead of ``resp_mu_coef``. MEASURED
    baseline: every calibration block listed below is bit-identical across the
    three adopted packs, while counts (88071 / 87840 / 86763) and dX are not."""
    import numpy.lib.npyio  # noqa: F401
    keys = ("resp_mu_coef", "resp_sig_coef", "resp_skew_coef", "resp_N_ref",
            "resp_N_fit_range", "molly_n_det", "molly_n_tot",
            "molly_nhi_edges", "g_grid", "g_occupancy", "fp_counts", "t_sigma")
    paths = [os.path.join(
        PACKDIR, f"modelA_pack_{m}_winlya_only_pad19p0_molly172_bw0p2.npz")
        for m in ("2lpt0", "london0", "saclay0")]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("not all three adopted packs present")
    ds = [np.load(p) for p in paths]
    try:
        for k in keys:
            a = np.asarray(ds[0][k])
            for d in ds[1:]:
                assert np.array_equal(a, np.asarray(d[k])), k
        totals = sorted(int(np.asarray(d["counts"]).sum()) for d in ds)
        assert totals == [86763, 87840, 88071]
    finally:
        for d in ds:
            d.close()
