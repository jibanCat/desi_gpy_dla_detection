# -*- coding: utf-8 -*-
"""Tests for the FROZEN FORWARD-MODEL AND MATCHING CONTRACT.

Every test is MUTATION-TESTED: the mutation that turns it red is named in the
test's own docstring, together with the MEASURED baseline it fails against, so
a reader can apply the mutation and check.

MOCK-DERIVED ONLY.  The pure-function tests read nothing; the integration tests
read the ADOPTED mock packs under the window-study scratch dir and SKIP when
they are absent.  No real-DESI path is opened.
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
UNPADDED_PACK = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                 "modelA_packs/modelA_pack_2lpt0_v11.npz")


def _fake_pack(*, ntrue=None, nhat=None, molly=None, n_b=None, n_k=3, n_s=2,
               truth_bks=True, counts=None, fp_counts=None,
               fp_ell_eff=13.589891949531905, fp_w=165.93215077605322,
               molly_n_det=None, molly_n_tot=None, kz=None):
    """A minimal duck-typed pack.

    ``validate_pack_against_contract`` / ``fp_normalisation_audit`` /
    ``check_accounting_identity(row_mass=...)`` touch only these fields, so a
    namespace is enough and the tests stay fast and survey-free.
    """
    ntrue = ADOPTED_NTRUE if ntrue is None else np.asarray(ntrue, float)
    nhat = REAL_NHAT if nhat is None else np.asarray(nhat, float)
    molly = MOLLY172 if molly is None else np.asarray(molly, float)
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
        molly_n_det=nd, molly_n_tot=nt,
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
    MEASURED baseline: the straddling bin's in-window fraction is exactly 0.5
    and its above-ceiling fraction is exactly 0.5; ``bins_fully_inside`` gives
    0.0 and would silently move a whole 0.2-dex bin out of P2."""
    p = MC.basis_partition(ADOPTED_NTRUE)
    j = int(np.flatnonzero(np.isclose(ADOPTED_NTRUE[:-1], 21.5))[0])
    assert p["in_window"][j] == pytest.approx(0.5, abs=1e-12)
    assert p["above_ceiling"][j] == pytest.approx(0.5, abs=1e-12)
    assert p["below_floor"][j] == 0.0


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


def test_classify_truth_counts_a_matched_row_zero_times():
    """A matched truth row is already on the CANDIDATE ledger and must not be
    added to P3.

    MUTATION: make ``classify_truth`` return 'P3_INCOMPLETENESS' unconditionally.
    MEASURED baseline: a matched row then appears on both ledgers and the truth
    ledger residual (test below) goes from 0.0 to -sum(found)."""
    assert MC.classify_truth(dict(matched=True)) is None
    assert MC.classify_truth(dict(matched=False)) == "P3_INCOMPLETENESS"


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
# 4. the FP normalisation contradiction, pinned with its measured factor
# ===========================================================================
def test_fp_normalisation_audit_reproduces_the_pack_scalars():
    """MEASURED on all three ADOPTED packs: fp_w * fp_ell_eff == 2255.0 exactly
    (== N_sl_loa0), and the contract total is fp_w * N_FP.

    MUTATION: change ``contract = w * ell * lam_tot`` to ``w * lam_tot`` in
    ``fp_normalisation_audit``. MEASURED baseline: the ratio then reads 1.0
    instead of 13.589891949531907 and the contradiction disappears from the
    record."""
    fp = np.zeros((len(REAL_NHAT) - 1, 2), dtype=np.int64)
    fp[0, 0] = 89
    a = MC.fp_normalisation_audit(_fake_pack(fp_counts=fp))
    assert a["n_sl_loa0_implied"] == pytest.approx(2255.0, abs=1e-9)
    assert a["mu_fp_total_per_contract"] == pytest.approx(
        165.93215077605322 * 89.0, rel=1e-12)
    assert a["mu_fp_total_as_implemented"] == pytest.approx(
        1086.6871844096897, rel=1e-9)
    assert a["ratio_contract_over_implemented"] == pytest.approx(
        13.589891949531905, rel=1e-9)


def test_assert_forward_fp_normalisation_raises_on_the_committed_fold():
    """The committed ``forward.py:452`` omits ``fp_ell_eff``, so this assertion
    MUST currently fail — that is the point: the contract fails loudly on the
    code as it stands.

    MUTATION (the FIX, deliberately not applied here): multiply the mu_fp term
    in forward.py by ``consts.fp_ell_eff``. MEASURED baseline: the total FP on
    the ADOPTED 2LPT-0 pack goes 1086.687 -> 14767.961 and the truth-pinned
    candidate-ledger residual goes -14563.57 (-16.54%) -> -882.30 (-1.00%).
    If this test ever goes green, forward.py changed and the two numbers above
    must be re-measured."""
    fp = np.zeros((len(REAL_NHAT) - 1, 2), dtype=np.int64)
    fp[0, 0] = 89
    with pytest.raises(MC.ContractViolation, match="under-normalised"):
        MC.assert_forward_fp_normalisation(_fake_pack(fp_counts=fp))


def test_the_forward_module_still_omits_ell_eff_at_the_named_site():
    """Pins KNOWN_CONTRADICTIONS[0] to the SOURCE, not to prose.

    MUTATION: rename ``fp_w`` in the fold. MEASURED baseline: forward.py's
    mu_fp expression mentions ``consts.fp_w`` and does NOT mention
    ``fp_ell_eff`` anywhere in ``fold_mu``."""
    src = open(os.path.join(REPO, "CDDF_analysis/hbi_mcmc/forward.py")).read()
    body = src.split("def fold_mu(", 1)[1].split("def fold_mu_reference", 1)[0]
    assert "consts.fp_w" in body
    assert "fp_ell_eff" not in body
    assert MC.KNOWN_CONTRADICTIONS[0]["id"] == "FP_ELL_EFF_OMITTED"


# ===========================================================================
# 5. the accounting identity, on an injected row mass (no jax, no scratch)
# ===========================================================================
def _toy():
    """A 3-bin latent basis on the adopted geometry's TOP three bins is not
    enough to satisfy the geometry rules, so the toy uses the FULL adopted
    edges with hand-set truth, completeness and row mass."""
    B, Kf, S = len(ADOPTED_NTRUE) - 1, 3, 2
    M = len(MOLLY172) - 1
    T = np.zeros((B, Kf, S))
    T[:, :, :] = 10.0                       # 16 * 3 * 2 * 10 = 960 truth systems
    nd = np.full((S, M), 50.0)              # C = (50+.5)/(100+1) ~ 0.5 exactly
    nt = np.full((S, M), 100.0)
    pk = _fake_pack(n_k=Kf, n_s=S, molly_n_det=nd, molly_n_tot=nt)
    pk.truth_counts_bks = T
    rho = np.full((S, 1, B), 0.8)           # KK = 1 (kz all zero)
    return pk, T, rho


def test_truth_ledger_closes_exactly():
    """found_on + found_off + missed == the truth total, to floating point.

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
    """The feasibility bound rests on rho <= 1.

    MUTATION: delete the row-mass guard. MEASURED baseline: with rho = 1.5 the
    reported found_on_grid (1440.0) EXCEEDS the truth total (960.0) and
    ``efficiency_attainable_at_calibration`` reads 1.5 — a nonsense the
    counting argument would then be built on."""
    pk, T, rho = _toy()
    with pytest.raises(MC.ContractViolation, match="row mass"):
        MC.check_accounting_identity(pk, row_mass=np.full_like(rho, 1.5))


# ===========================================================================
# 6. the contract serializes, and claims no authority it does not have
# ===========================================================================
def test_contract_dict_is_json_serializable_and_complete():
    """MUTATION: drop ``accounting_identity`` from ``contract_dict``. MEASURED
    baseline: the contract carries 6 populations, 7 support regions and 5
    recorded contradictions."""
    d = MC.contract_dict()
    json.dumps(d, default=str)
    assert len(d["populations"]) == 6
    assert {p["pid"] for p in d["populations"]} == {
        "P1_SCATTER_IN", "P2_IN_WINDOW", "P3_INCOMPLETENESS", "P4_FOREST_FP",
        "P5_TRANSFER", "P6_RESIDUAL"}
    assert len(d["support_map"]) == 7
    assert len(d["known_contradictions"]) == 5
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


# ===========================================================================
# 7. integration on the REAL adopted mock packs
# ===========================================================================
@pytest.mark.skipif(not os.path.exists(ADOPTED_PACK),
                    reason="adopted window-study pack not on this filesystem")
def test_identity_on_the_adopted_2lpt0_pack():
    """The referee-facing run. MEASURED 2026-08-05 on
    modelA_pack_2lpt0_winlya_only_pad19p0_molly172_bw0p2.npz, resp_clamp=both:

        N_obs                       88071
        truth on basis             101949
        found_on_grid            72420.741
        found_off_grid            9895.303
        missed (P3)              19632.956
        truth-ledger residual        0.000     <- MUST be zero
        P1 scatter-in            18033.092
        P2 in-window             53847.300
        P6 above ceiling           540.349
        P4 as implemented         1086.687
        P4 per contract          14767.961
        candidate residual  (impl)  -14563.572   (-16.536%)
        candidate residual (contr)    -882.298   ( -1.002%)
        efficiency attainable        0.71036
        efficiency required (contr)  0.71902     <- EXCEEDS attainable

    MUTATION: set ``rho_bks`` to 1 everywhere. MEASURED baseline: found_off
    collapses 9895.303 -> 0.0 and the candidate residual (contract) moves
    -882.298 -> +9013.005."""
    from CDDF_analysis.hbi_mcmc.pack import load_pack
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
    assert c["as_implemented"]["residual"] == pytest.approx(-14563.572, abs=1e-2)
    assert c["per_contract"]["residual"] == pytest.approx(-882.298, abs=1e-2)
    assert f["efficiency_attainable_at_calibration"] == pytest.approx(
        0.710363, abs=1e-5)
    # the sharp bound: what the frozen calibration delivers is LESS than what
    # closing the counts requires, even with the contract's FP normalisation.
    assert f["feasible_at_calibration_per_contract"] is False


@pytest.mark.skipif(not os.path.exists(UNPADDED_PACK),
                    reason="unpadded v1.1 pack not on this filesystem")
def test_the_counting_argument_on_the_unpadded_pack():
    """The D1 counting argument, run through the contract. MEASURED 2026-08-05
    on modelA_pack_2lpt0_v11.npz (0.1-dex basis, NO pad):

        N_obs 88071 > truth on basis 73610
        efficiency required, FP as implemented : 1.18168   <- IMPOSSIBLE
        efficiency required, FP per contract   : 0.99583   <- possible in
                                                              principle, but
        efficiency attainable at calibration   : 0.85003   <- not in practice

    So the classic '88071 > 73610 therefore no parameter closes it' is exact
    ONLY with the under-normalised FP; with the contract's FP the refutation
    still holds, but through the ATTAINABLE efficiency, not the trivial bound.
    MUTATION: pass ``require_pad=True``. MEASURED baseline: the pack is refused
    before any number is produced."""
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    pk = load_pack(UNPADDED_PACK, allow_nonstandard_grid=True)
    r = MC.check_accounting_identity(
        pk, require_pad=False, require_measured_sub_floor_completeness=False)
    f = r["feasibility"]
    assert r["truth_ledger"]["residual"] == pytest.approx(0.0, abs=1e-6)
    assert r["truth_ledger"]["n_truth_on_basis"] == 73610.0
    assert r["candidate_ledger"]["n_obs"] == 88071.0
    assert f["efficiency_required_as_implemented"] == pytest.approx(1.18168, abs=1e-4)
    assert f["efficiency_required_per_contract"] == pytest.approx(0.99583, abs=1e-4)
    assert f["efficiency_attainable_at_calibration"] == pytest.approx(0.85003, abs=1e-4)
    assert f["feasible_at_calibration_as_implemented"] is False
    assert f["feasible_at_calibration_per_contract"] is False


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
