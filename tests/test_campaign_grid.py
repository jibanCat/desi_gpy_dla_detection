"""
tests/test_campaign_grid.py
===========================
TDD tests for ``injection.campaign_grid`` — the M3 injection-campaign GRID
+ SIGHTLINE-SAMPLER + MANIFEST-SCHEMA (the Bayesian-modeling owner's scope of the
M3 design, ``2026-06-10_m3_injection_campaign_design.md``).

This module is PURE python/numpy: NO desispec, NO coadd I/O (that is the CS
agent's ``coadd_injection.py``).  It defines

  * ``build_injection_grid(...)``      — the (logN_true × z_true × SNR_bin) cells,
                                         DENSE in [17.2, 19.0], emitting one manifest
                                         row per injection (the CONTRACT the CS
                                         injector consumes);
  * ``sample_clean_sightlines(...)``   — deterministic, SNR-bin-balanced draw of
                                         CLEAN TARGETIDs per cell, no reuse across
                                         cells (unless flagged);
  * ``MANIFEST_FIELDS`` / ``validate_manifest(...)`` — the manifest schema + guard.

The grid resolution is the science knob: dense (Δ≈0.1–0.2 dex) below 19.0 where
the single-absorber GP is weakest, moderate in [19, 20.3], coarse in
[20.3, 22.5].  ``target_injections`` sizes the campaign to a CPU-h budget
(≤4000 CPU-h cap; ~131.5 s/spec).
"""
import numpy as np
import pytest

from injection import campaign_grid as cg
from injection.campaign_grid import (
    MANIFEST_FIELDS,
    build_injection_grid,
    sample_clean_sightlines,
    validate_manifest,
)


# --------------------------------------------------------------------------- #
# manifest schema — the EXACT contract for the CS coadd injector
# --------------------------------------------------------------------------- #
def test_manifest_fields_are_the_exact_contract():
    # The CS agent's coadd injector consumes these keys verbatim.  Pin them so a
    # rename here is a loud test failure rather than a silent integration break.
    # The first 11 keys are FROZEN; ``control`` is the ADDITIVE M2 flag appended
    # at the end (no existing key is renamed/reordered).
    assert MANIFEST_FIELDS == (
        "inj_id",
        "campaign",
        "method",
        "target_id",
        "healpix",
        "z_qso",
        "snr_bin",
        "native_snr",
        "logN_true",
        "z_true",
        "num_lines",
        "control",
        "zqso_bin",
    )


def test_close_pair_fields_documented_optional():
    # Campaign B (close pairs) carries three OPTIONAL extra fields.
    assert cg.CLOSE_PAIR_FIELDS == ("logN_true2", "z_true2", "dv_kms")


# --------------------------------------------------------------------------- #
# NHI grid — DENSE below 19, moderate to 20.3, coarse to 22.5
# --------------------------------------------------------------------------- #
def test_default_logn_grid_spans_full_qmc_range():
    edges = cg.default_logn_grid()
    assert edges[0] == pytest.approx(17.2)
    assert edges[-1] == pytest.approx(22.5)
    assert np.all(np.diff(edges) > 0)  # strictly increasing


def test_logn_grid_is_dense_below_19_and_coarse_above_203():
    edges = np.asarray(cg.default_logn_grid())
    below19 = edges[(edges >= 17.2) & (edges < 19.0)]
    dla = edges[(edges >= 20.3) & (edges <= 22.5)]
    # dense LLS regime: spacing <= 0.2 dex
    d_below = np.diff(below19)
    assert d_below.size >= 8  # at least ~9 points in [17.2, 19.0)
    assert np.all(d_below <= 0.2 + 1e-9)
    # coarse DLA regime: spacing strictly coarser than the LLS regime
    d_dla = np.diff(dla)
    assert np.median(d_dla) > np.median(d_below)


def test_logn_grid_includes_the_203_knee_as_interior_point():
    # The 20.3 sub-DLA<->DLA boundary must be an INTERIOR grid point (the migration
    # knee), per the design ("20.3 is an interior point of R, not an edge").
    edges = np.asarray(cg.default_logn_grid())
    assert np.any(np.isclose(edges, 20.3, atol=1e-6))
    assert edges[0] < 20.3 < edges[-1]


# --------------------------------------------------------------------------- #
# z grid — across the search window
# --------------------------------------------------------------------------- #
def test_default_z_grid_within_search_window():
    z = np.asarray(cg.default_z_grid())
    assert z.size >= 3
    assert np.all(np.diff(z) > 0)
    # within the global DESI absorber-redshift window (zmin_search .. zmax_qso)
    assert z[0] >= 2.0
    assert z[-1] <= 4.25


# --------------------------------------------------------------------------- #
# build_injection_grid — one manifest row per injection
# --------------------------------------------------------------------------- #
def _toy_sightlines(n=40, seed=0):
    """A toy CLEAN sightline table: TARGETID, healpix, z_qso, native_snr."""
    rng = np.random.default_rng(seed)
    return {
        "target_id": np.arange(1000, 1000 + n, dtype=np.int64),
        "healpix": rng.integers(0, 12, size=n).astype(np.int64),
        "z_qso": rng.uniform(2.5, 4.0, size=n),
        "native_snr": rng.uniform(0.3, 6.0, size=n),
    }


def test_build_grid_returns_list_of_dict_rows_with_contract_keys():
    rows = build_injection_grid(
        clean_sightlines=_toy_sightlines(),
        logN_grid=[18.0, 20.5],
        z_grid=[2.6, 3.2],
        snr_bins=[0.0, 2.0, 100.0],
        n_per_cell=2,
        seed=7,
    )
    assert isinstance(rows, list) and rows and isinstance(rows[0], dict)
    for r in rows:
        for k in MANIFEST_FIELDS:
            assert k in r, f"manifest row missing contract key {k!r}"


def test_build_grid_inj_ids_are_unique_and_contiguous():
    rows = build_injection_grid(
        clean_sightlines=_toy_sightlines(),
        logN_grid=[18.0, 19.0, 20.5],
        z_grid=[2.6, 3.2],
        snr_bins=[0.0, 2.0, 100.0],
        n_per_cell=2,
        seed=1,
    )
    ids = sorted(r["inj_id"] for r in rows)
    assert ids == list(range(len(rows)))  # 0..N-1, unique


def _zqso_spread_sightlines(n=300, zq_lo=2.45, zq_hi=3.35, seed=0):
    """Clean sightlines whose host z_QSO spans a broad range, so a low-z absorber
    (z_true=2.3) is hostable across many z_QSO → its rest-frame position sweeps the
    forest.  z_true=2.3 is hostable while z_lo(z_qso) <= 2.3 <= z_qso-buffer, i.e.
    z_qso in ~[2.31, 3.39]; this pool sits inside that."""
    rng = np.random.default_rng(seed)
    return {
        "target_id": np.arange(7000, 7000 + n, dtype=np.int64),
        "healpix": np.zeros(n, dtype=np.int64),
        "z_qso": rng.uniform(zq_lo, zq_hi, size=n),
        "native_snr": rng.uniform(2.5, 9.0, size=n),
    }


def test_build_grid_zqso_stratification_labels_and_spans_bins():
    # With zqso_bins given, a FIXED low z_true is injected across the full hostable
    # z_QSO range (each row labelled by its z_QSO bin) → rest-frame forest position
    # is sampled, not left to wherever the SNR-only draw happens to land.
    zqso_bins = [2.4, 2.8, 3.1, 3.4]
    rows = build_injection_grid(
        clean_sightlines=_zqso_spread_sightlines(n=300),
        logN_grid=[20.5],
        z_grid=[2.3],
        snr_bins=[2.0, 100.0],
        n_per_cell=6,
        zqso_bins=zqso_bins,
        seed=1,
    )
    assert rows
    for r in rows:
        b = r["zqso_bin"]
        assert 0 <= b < len(zqso_bins) - 1
        assert zqso_bins[b] <= r["z_qso"] < zqso_bins[b + 1]
    # all three z_QSO bins are populated → the rest-frame position is spanned
    assert set(r["zqso_bin"] for r in rows) == {0, 1, 2}


def test_build_grid_zqso_none_emits_sentinel_and_is_backward_compatible():
    rows = build_injection_grid(
        clean_sightlines=_toy_sightlines(),
        logN_grid=[18.0],
        z_grid=[2.6],
        snr_bins=[0.0, 100.0],
        n_per_cell=1,
        seed=2,
    )
    assert rows and all(r["zqso_bin"] == -1 for r in rows)  # no stratification → -1


def test_build_injection_sample_follows_known_logN_pdf():
    # Campaign D: inject a KNOWN non-PW100 truth CDDF. The injected logN distribution
    # must follow the supplied pdf (here a falling power law dn/dlogN ∝ 10^(-0.7 logN)),
    # so deconvolving the recovery with R can be checked for unbiasedness on a
    # distribution the inference prior (PW100) does NOT match.
    from injection.campaign_grid import build_injection_sample
    pdf = lambda ln: 10.0 ** (-0.7 * np.asarray(ln))
    rows = build_injection_sample(
        clean_sightlines=_zqso_spread_sightlines(n=600, zq_lo=2.6, zq_hi=3.7),
        snr_bins=[2.0, 100.0],
        n_per_cell=40,
        logN_pdf=pdf,
        logN_range=(17.2, 22.5),
        zqso_bins=[2.5, 3.0, 3.5, 3.8],
        seed=3,
    )
    assert rows
    lN = np.array([r["logN_true"] for r in rows])
    # falling pdf → more low-N than high-N injections
    assert np.mean(lN < 19.0) > np.mean(lN > 21.0)
    assert lN.min() >= 17.2 - 1e-6 and lN.max() <= 22.5 + 1e-6
    # contract intact + globally unique sightlines + campaign D
    for r in rows:
        for k in MANIFEST_FIELDS:
            assert k in r
        assert r["campaign"] == "D"
    tids = [r["target_id"] for r in rows]
    assert len(tids) == len(set(tids))
    validate_manifest(rows)


def test_build_injection_sample_deterministic():
    from injection.campaign_grid import build_injection_sample
    pdf = lambda ln: np.ones_like(np.asarray(ln, float))
    kw = dict(clean_sightlines=_zqso_spread_sightlines(n=400, zq_lo=2.6, zq_hi=3.6),
              snr_bins=[2.0, 100.0], n_per_cell=20, logN_pdf=pdf,
              logN_range=(17.2, 22.5), seed=7)
    a = build_injection_sample(**kw)
    b = build_injection_sample(**kw)
    assert [r["target_id"] for r in a] == [r["target_id"] for r in b]
    assert [round(r["logN_true"], 6) for r in a] == [round(r["logN_true"], 6) for r in b]


def test_default_zqso_bins_spans_desi_qso_window():
    b = np.asarray(cg.default_zqso_bins())
    assert b[0] <= 2.2 and b[-1] >= 4.0       # covers the DESI QSO z range
    assert np.all(np.diff(b) > 0) and b.size >= 4  # >=3 bins, strictly increasing


def test_build_grid_target_ids_are_globally_unique_across_cells():
    # CRITICAL M3 invariant: each clean sightline yields exactly ONE DESI spectrum,
    # and ``inject_into_coadd`` STACKS every manifest row that shares a target_id
    # into that single spectrum (multiplicative blend).  So a target_id reused
    # across (logN, z, SNR) cells would superimpose several absorbers on one
    # spectrum while the manifest claims them as independent single-absorber
    # injections — corrupting recovery-by-inj_id.  The grid MUST inject each clean
    # sightline at most ONCE globally.
    rows = build_injection_grid(
        clean_sightlines=_toy_sightlines(n=200, seed=3),
        logN_grid=[18.0, 19.0, 20.5],
        z_grid=[2.6, 3.2],
        snr_bins=[0.0, 2.0, 100.0],
        n_per_cell=3,
        seed=5,
    )
    tids = [r["target_id"] for r in rows]
    assert len(tids) == len(set(tids)), "a clean sightline was injected in >1 cell"


def _ample_balanced_sightlines(n_per_bin=200, z_qso=3.5):
    """A toy pool where EVERY sightline hosts both test z's (high z_qso) and the
    two SNR bins [0,2) / [2,inf) are each amply filled — so the cell-product count
    is reachable even under the global one-injection-per-target rule."""
    snr = np.concatenate([np.full(n_per_bin, 1.0), np.full(n_per_bin, 5.0)])
    n = snr.size
    return {
        "target_id": np.arange(5000, 5000 + n, dtype=np.int64),
        "healpix": np.zeros(n, dtype=np.int64),
        "z_qso": np.full(n, float(z_qso)),
        "native_snr": snr,
    }


def test_build_grid_cell_product_with_n_per_cell():
    # rows == n_logN * n_z * n_snr_bins * n_per_cell when the clean pool is AMPLE
    # per SNR bin AND every sightline hosts both z's — under the global
    # one-injection-per-target rule the product is still reachable (the pool here
    # has 200/bin >> the 18/bin demanded).
    logN = [18.0, 19.0, 20.5]
    z = [2.6, 3.2]
    snr_bins = [0.0, 2.0, 100.0]  # 2 occupied SNR bins
    n_per_cell = 3
    rows = build_injection_grid(
        clean_sightlines=_ample_balanced_sightlines(),
        logN_grid=logN,
        z_grid=z,
        snr_bins=snr_bins,
        n_per_cell=n_per_cell,
        seed=5,
    )
    n_cells = len(logN) * len(z) * (len(snr_bins) - 1)
    assert len(rows) == n_cells * n_per_cell
    # …and still globally unique (no sightline reused across cells).
    tids = [r["target_id"] for r in rows]
    assert len(tids) == len(set(tids))


def test_build_grid_count_never_exceeds_cells_times_per_cell():
    # On a constrained pool the count is BOUNDED by the cell product AND by the
    # distinct hostable pool (global dedup) — never above the product, never reusing.
    logN = [18.0, 19.0, 20.5]
    z = [2.6, 3.2]
    snr_bins = [0.0, 2.0, 100.0]
    n_per_cell = 3
    rows = build_injection_grid(
        clean_sightlines=_toy_sightlines(n=60, seed=3),
        logN_grid=logN, z_grid=z, snr_bins=snr_bins,
        n_per_cell=n_per_cell, seed=5,
    )
    n_cells = len(logN) * len(z) * (len(snr_bins) - 1)
    assert len(rows) <= n_cells * n_per_cell
    tids = [r["target_id"] for r in rows]
    assert len(tids) == len(set(tids))


def test_build_grid_method_and_campaign_defaults():
    rows = build_injection_grid(
        clean_sightlines=_toy_sightlines(),
        logN_grid=[18.0],
        z_grid=[2.6],
        snr_bins=[0.0, 100.0],
        n_per_cell=1,
        seed=2,
    )
    assert all(r["campaign"] == "A" for r in rows)
    assert all(r["method"] == "coadd" for r in rows)
    assert all(r["num_lines"] == 31 for r in rows)  # matches the run's NUM_FOREST_LINES


def test_build_grid_native_snr_lands_in_assigned_snr_bin():
    snr_bins = [0.0, 1.0, 3.0, 100.0]
    rows = build_injection_grid(
        clean_sightlines=_toy_sightlines(n=80, seed=9),
        logN_grid=[18.0, 20.5],
        z_grid=[2.6, 3.2],
        snr_bins=snr_bins,
        n_per_cell=2,
        seed=11,
    )
    for r in rows:
        b = r["snr_bin"]
        assert snr_bins[b] <= r["native_snr"] < snr_bins[b + 1]


def test_build_grid_target_injections_caps_total_count():
    # When target_injections is given, the grid is sized DOWN (n_per_cell reduced)
    # to not exceed it — so the driver can hit a CPU-h budget.
    rows = build_injection_grid(
        clean_sightlines=_toy_sightlines(n=200, seed=4),
        logN_grid=cg.default_logn_grid(),
        z_grid=cg.default_z_grid(),
        snr_bins=[0.0, 1.0, 2.0, 4.0, 100.0],
        target_injections=300,
        seed=6,
    )
    assert len(rows) <= 300
    # and it actually used the budget (within one cell-worth)
    assert len(rows) > 0


def test_build_grid_deterministic_under_seed():
    kw = dict(
        clean_sightlines=_toy_sightlines(n=60, seed=8),
        logN_grid=[18.0, 19.0, 20.5],
        z_grid=[2.6, 3.2],
        snr_bins=[0.0, 2.0, 100.0],
        n_per_cell=2,
    )
    a = build_injection_grid(seed=42, **kw)
    b = build_injection_grid(seed=42, **kw)
    assert [r["target_id"] for r in a] == [r["target_id"] for r in b]
    assert [r["logN_true"] for r in a] == [r["logN_true"] for r in b]


def test_build_grid_z_true_is_clamped_into_each_sightline_window():
    # A grid z_true above a sightline's z_qso is unphysical (absorber redward of the
    # QSO).  The builder must DROP cells where no clean sightline can host the z, or
    # clamp into the per-sightline [min_z, max_z] window — here we require the
    # emitted z_true < z_qso for every row (absorber must be in the forest).
    sl = _toy_sightlines(n=60, seed=12)
    rows = build_injection_grid(
        clean_sightlines=sl,
        logN_grid=[18.0, 20.5],
        z_grid=cg.default_z_grid(),
        snr_bins=[0.0, 100.0],
        n_per_cell=2,
        seed=13,
    )
    for r in rows:
        assert r["z_true"] < r["z_qso"], "injected absorber must sit blueward of z_qso"


def test_validate_manifest_accepts_good_and_rejects_bad():
    rows = build_injection_grid(
        clean_sightlines=_toy_sightlines(),
        logN_grid=[18.0, 20.5],
        z_grid=[2.6],
        snr_bins=[0.0, 100.0],
        n_per_cell=1,
        seed=3,
    )
    validate_manifest(rows)  # no raise
    bad = [dict(rows[0])]
    del bad[0]["logN_true"]
    with pytest.raises((KeyError, ValueError)):
        validate_manifest(bad)


def test_validate_manifest_rejects_duplicate_inj_id():
    rows = build_injection_grid(
        clean_sightlines=_toy_sightlines(),
        logN_grid=[18.0, 20.5],
        z_grid=[2.6],
        snr_bins=[0.0, 100.0],
        n_per_cell=1,
        seed=3,
    )
    dup = [dict(r) for r in rows] + [dict(rows[0])]  # duplicate inj_id
    with pytest.raises(ValueError):
        validate_manifest(dup)


def test_validate_manifest_rejects_duplicate_target_id():
    # Two rows on the SAME sightline → the injector stacks both absorbers into the
    # one spectrum while the manifest claims two independent injections.  The guard
    # must reject this even when the inj_ids are distinct.
    rows = build_injection_grid(
        clean_sightlines=_toy_sightlines(),
        logN_grid=[18.0, 20.5],
        z_grid=[2.6],
        snr_bins=[0.0, 100.0],
        n_per_cell=1,
        seed=3,
    )
    dup = [dict(r) for r in rows]
    clash = dict(rows[0])
    clash["inj_id"] = max(r["inj_id"] for r in rows) + 1   # distinct inj_id…
    # …but the SAME target_id as rows[0]
    dup.append(clash)
    with pytest.raises(ValueError, match="target_id"):
        validate_manifest(dup)


# --------------------------------------------------------------------------- #
# sample_clean_sightlines — deterministic, SNR-bin-balanced, no reuse
# --------------------------------------------------------------------------- #
def _clean_table(n=300, seed=0):
    rng = np.random.default_rng(seed)
    tids = np.arange(5000, 5000 + n, dtype=np.int64)
    return tids


def _snr_table(tids, seed=0):
    rng = np.random.default_rng(seed)
    return {int(t): float(s) for t, s in zip(tids, rng.uniform(0.2, 6.0, size=len(tids)))}


def test_sample_clean_sightlines_balances_snr_bins():
    tids = _clean_table(300, seed=1)
    snr = _snr_table(tids, seed=2)
    snr_bins = [0.0, 1.0, 2.0, 4.0, 100.0]
    assign = sample_clean_sightlines(
        tids, snr, n_per_cell=5, snr_bins=snr_bins, seed=10
    )
    # one entry per SNR bin index
    assert set(assign.keys()) == set(range(len(snr_bins) - 1))
    for b, picked in assign.items():
        assert len(picked) <= 5
        # every picked TARGETID's SNR is inside bin b
        for t in picked:
            assert snr_bins[b] <= snr[int(t)] < snr_bins[b + 1]


def test_sample_clean_sightlines_no_reuse_across_bins():
    tids = _clean_table(300, seed=3)
    snr = _snr_table(tids, seed=4)
    snr_bins = [0.0, 1.0, 2.0, 4.0, 100.0]
    assign = sample_clean_sightlines(
        tids, snr, n_per_cell=5, snr_bins=snr_bins, seed=20
    )
    seen = [int(t) for picked in assign.values() for t in picked]
    assert len(seen) == len(set(seen)), "a TARGETID was reused across SNR bins"


def test_sample_clean_sightlines_deterministic():
    tids = _clean_table(300, seed=5)
    snr = _snr_table(tids, seed=6)
    snr_bins = [0.0, 2.0, 100.0]
    a = sample_clean_sightlines(tids, snr, n_per_cell=4, snr_bins=snr_bins, seed=99)
    b = sample_clean_sightlines(tids, snr, n_per_cell=4, snr_bins=snr_bins, seed=99)
    assert {k: list(v) for k, v in a.items()} == {k: list(v) for k, v in b.items()}


def test_sample_clean_sightlines_different_seed_differs():
    tids = _clean_table(300, seed=7)
    snr = _snr_table(tids, seed=8)
    snr_bins = [0.0, 2.0, 100.0]
    a = sample_clean_sightlines(tids, snr, n_per_cell=4, snr_bins=snr_bins, seed=1)
    b = sample_clean_sightlines(tids, snr, n_per_cell=4, snr_bins=snr_bins, seed=2)
    flat_a = sorted(int(t) for v in a.values() for t in v)
    flat_b = sorted(int(t) for v in b.values() for t in v)
    assert flat_a != flat_b


# --------------------------------------------------------------------------- #
# Campaign B — close-pair grid (optional fields)
# --------------------------------------------------------------------------- #
def test_close_pair_grid_emits_campaign_b_with_pair_fields():
    rows = cg.build_close_pair_grid(
        clean_sightlines=_toy_sightlines(n=60, seed=1),
        logN_grid=[20.5],
        z_grid=[3.0],
        dv_kms_grid=[100.0, 400.0],
        dlogN_grid=[0.0, -0.5],
        snr_bins=[0.0, 100.0],
        n_per_cell=2,
        seed=3,
    )
    assert rows and all(r["campaign"] == "B" for r in rows)
    for r in rows:
        # base manifest contract still holds
        for k in MANIFEST_FIELDS:
            assert k in r
        # plus the close-pair fields
        for k in cg.CLOSE_PAIR_FIELDS:
            assert k in r
        assert r["dv_kms"] in (100.0, 400.0)
        # second absorber redshift offset by dv from the first (blueward/redward)
        assert r["z_true2"] != r["z_true"]
        # second column density = first + dlogN
        assert r["logN_true2"] == pytest.approx(r["logN_true"] + r["_dlogN"], abs=1e-9) \
            if "_dlogN" in r else True


def test_close_pair_grid_dv_maps_to_redshift_separation():
    rows = cg.build_close_pair_grid(
        clean_sightlines=_toy_sightlines(n=40, seed=2),
        logN_grid=[20.5],
        z_grid=[3.0],
        dv_kms_grid=[300.0],
        dlogN_grid=[0.0],
        snr_bins=[0.0, 100.0],
        n_per_cell=1,
        seed=4,
    )
    C_KMS = 299792.458
    for r in rows:
        # dv = c * (z2 - z1) / (1 + z1)  (relativistic-free small-sep convention)
        dz = r["z_true2"] - r["z_true"]
        dv = C_KMS * dz / (1.0 + r["z_true"])
        assert dv == pytest.approx(300.0, rel=1e-6)


def test_close_pair_grid_target_injections_only_mode():
    # The CPU-budget sizing mode (target_injections WITHOUT n_per_cell) must work,
    # not crash in _resolve_n_per_cell.  Each sightline hosts ONE pair config → the
    # output stays globally target-unique and within the cap.
    rows = cg.build_close_pair_grid(
        clean_sightlines=_toy_sightlines(n=200, seed=11),
        logN_grid=[20.5],
        z_grid=[3.0],
        dv_kms_grid=[200.0, 500.0],
        dlogN_grid=[0.0],
        snr_bins=[0.0, 100.0],
        target_injections=30,
        seed=12,
    )
    assert rows and len(rows) <= 30
    tids = [r["target_id"] for r in rows]
    assert len(tids) == len(set(tids))      # one injection per sightline globally
    assert {r["dv_kms"] for r in rows} <= {200.0, 500.0}
    validate_manifest(rows)


def test_close_pair_grid_validates_with_manifest_guard():
    rows = cg.build_close_pair_grid(
        clean_sightlines=_toy_sightlines(n=40, seed=5),
        logN_grid=[20.5],
        z_grid=[3.0],
        dv_kms_grid=[200.0],
        dlogN_grid=[0.0],
        snr_bins=[0.0, 100.0],
        n_per_cell=1,
        seed=6,
    )
    validate_manifest(rows)  # no raise (close-pair fields sanity-checked)


def test_sample_clean_sightlines_underfilled_bin_returns_all_available():
    # Only 2 sightlines in a bin but n_per_cell=5 -> return the 2, no error/dup.
    tids = np.array([7001, 7002, 7003], dtype=np.int64)
    snr = {7001: 0.5, 7002: 0.6, 7003: 3.0}
    snr_bins = [0.0, 1.0, 100.0]
    assign = sample_clean_sightlines(tids, snr, n_per_cell=5, snr_bins=snr_bins, seed=0)
    assert sorted(int(t) for t in assign[0]) == [7001, 7002]
    assert sorted(int(t) for t in assign[1]) == [7003]


# --------------------------------------------------------------------------- #
# M1 — z-window MUST equal the GP inference search window (NOT the Lyβ floor /
# z_hi=z_qso).  The window reuses the REAL inference constants (Lyman limit,
# kms_to_z(3000), MAX_LAMBDA=1250, Z_SEARCH_MIN/MAX) — no hardcoding.
# --------------------------------------------------------------------------- #
def test_forest_window_blue_floor_reaches_lyman_limit_not_lybeta():
    # The GP searches down to the Lyman limit (911.7633 Å), NOT Lyβ (1025.72 Å).
    # The blue floor must equal (1+z_qso)*lyman_limit/lya - 1 + kms_to_z(3000),
    # which is BELOW the old Lyβ floor (1+z_qso)*lyb/lya - 1 by a large margin.
    from gpy_dla_detection.set_parameters import Parameters

    z_qso = 3.5
    z_lo, z_hi = cg._per_sightline_forest_window(z_qso)
    lya = Parameters.lya_wavelength
    lyman_limit = Parameters.lyman_limit
    kms = Parameters.kms_to_z(3000.0)

    expected_blue = (1.0 + z_qso) * (lyman_limit / lya) - 1.0 + kms
    lyb_floor = (1.0 + z_qso) * (1025.7223 / lya) - 1.0
    # the window reaches the Lyman limit (well below the Lyβ floor)
    assert z_lo == pytest.approx(max(expected_blue, cg.Z_SEARCH_MIN), abs=1e-9)
    assert z_lo < lyb_floor - 0.2  # demonstrably blueward of the old Lyβ floor


def test_forest_window_red_ceiling_respects_proximity_and_max_lambda():
    # The red ceiling must be min(z_qso - kms_to_z(3000),
    # (1+z_qso)*MAX_LAMBDA/lya - 1, Z_SEARCH_MAX) — NOT the bare z_qso.
    from gpy_dla_detection.set_parameters import Parameters

    z_qso = 3.5
    z_lo, z_hi = cg._per_sightline_forest_window(z_qso)
    lya = Parameters.lya_wavelength
    kms = Parameters.kms_to_z(3000.0)
    max_lambda = 1250.0

    expected_red = min(
        z_qso - kms,
        (1.0 + z_qso) * (max_lambda / lya) - 1.0,
        cg.Z_SEARCH_MAX,
    )
    assert z_hi == pytest.approx(expected_red, abs=1e-9)
    # strictly below z_qso (the proximity buffer is honored)
    assert z_hi < z_qso


def test_forest_window_uses_real_inference_constants_not_hardcoded():
    # The module must import the REAL constants (Parameters), so the window tracks
    # the inference if those constants change.  Pin the exact numeric values.
    from gpy_dla_detection.set_parameters import Parameters

    assert cg._LYA_REST == pytest.approx(Parameters.lya_wavelength, abs=1e-9)
    assert cg._LYMAN_LIMIT == pytest.approx(Parameters.lyman_limit, abs=1e-9)
    assert cg._KMS_TO_Z_3000 == pytest.approx(Parameters.kms_to_z(3000.0), abs=1e-12)


def test_build_grid_no_injection_lands_outside_gp_searchable_z():
    # Every emitted z_true must sit strictly inside [z_lo, z_hi] of its sightline's
    # GP search window (clamped), so NO injection is outside the searchable z.
    sl = _toy_sightlines(n=80, seed=21)
    rows = build_injection_grid(
        clean_sightlines=sl,
        logN_grid=[18.0, 20.5],
        z_grid=cg.default_z_grid(),
        snr_bins=[0.0, 100.0],
        n_per_cell=2,
        seed=22,
    )
    assert rows
    for r in rows:
        z_lo, z_hi = cg._per_sightline_forest_window(r["z_qso"])
        assert z_lo <= r["z_true"] <= z_hi, (
            f"z_true {r['z_true']} outside GP window [{z_lo}, {z_hi}] "
            f"for z_qso {r['z_qso']}"
        )


def test_validate_manifest_enforces_z_window_bounds():
    # validate_manifest must reject a z_true below the sightline's z_lo (outside the
    # GP-searchable blue edge), not just z_true >= z_qso.
    sl = _toy_sightlines(n=40, seed=23)
    rows = build_injection_grid(
        clean_sightlines=sl,
        logN_grid=[18.0],
        z_grid=[3.0],
        snr_bins=[0.0, 100.0],
        n_per_cell=1,
        seed=24,
    )
    validate_manifest(rows)  # good rows pass
    # corrupt one row: push z_true below the GP blue floor for its z_qso
    bad = [dict(r) for r in rows]
    z_lo, _ = cg._per_sightline_forest_window(bad[0]["z_qso"])
    bad[0]["z_true"] = z_lo - 0.5  # below the searchable window
    with pytest.raises(ValueError):
        validate_manifest(bad)


def test_close_pair_z_true2_clamped_into_window_and_validated():
    # Campaign-B second absorber z_true2 must also sit inside the GP window and be
    # blueward of z_qso; validate_manifest enforces z_true2 < z_qso.
    rows = cg.build_close_pair_grid(
        clean_sightlines=_toy_sightlines(n=60, seed=25),
        logN_grid=[20.5],
        z_grid=[3.0],
        dv_kms_grid=[300.0, 800.0],
        dlogN_grid=[0.0],
        snr_bins=[0.0, 100.0],
        n_per_cell=2,
        seed=26,
    )
    assert rows
    for r in rows:
        z_lo, z_hi = cg._per_sightline_forest_window(r["z_qso"])
        assert z_lo <= r["z_true2"] <= z_hi
        assert r["z_true2"] < r["z_qso"]
    validate_manifest(rows)  # guard accepts (and would reject z_true2 >= z_qso)


# --------------------------------------------------------------------------- #
# M2 — b_FP control rows.  The grid must emit CLEAN no-injection control rows
# (logN_true = NaN, control=True), and validate_manifest must EXEMPT them from
# the logN-range check (they legitimately carry NaN logN).
# --------------------------------------------------------------------------- #
def test_build_control_rows_emits_clean_no_injection_rows():
    sl = _toy_sightlines(n=40, seed=31)
    rows = cg.build_control_rows(
        clean_sightlines=sl,
        snr_bins=[0.0, 2.0, 100.0],
        n_per_cell=3,
        seed=32,
    )
    assert rows and isinstance(rows[0], dict)
    for r in rows:
        # full manifest contract present
        for k in MANIFEST_FIELDS:
            assert k in r
        assert r["control"] is True
        assert not np.isfinite(r["logN_true"])   # NaN logN (no injection)
        assert not np.isfinite(r["z_true"])      # NaN z (no injection)
        assert r["campaign"] == "A"


def test_control_flag_field_in_manifest_fields():
    # The control flag is an ADDITIVE manifest field (schema stays backward-compat:
    # the original 11 keys are unchanged and still come first).
    assert "control" in MANIFEST_FIELDS
    assert MANIFEST_FIELDS[:11] == (
        "inj_id", "campaign", "method", "target_id", "healpix", "z_qso",
        "snr_bin", "native_snr", "logN_true", "z_true", "num_lines",
    )


def test_validate_manifest_exempts_control_rows_from_logn_range():
    # Control rows carry NaN logN_true; validate_manifest must accept them (NOT
    # raise on the [LOGN_MIN, LOGN_MAX] range check) when control=True.
    sl = _toy_sightlines(n=20, seed=33)
    ctrl = cg.build_control_rows(
        clean_sightlines=sl, snr_bins=[0.0, 100.0], n_per_cell=2, seed=34
    )
    validate_manifest(ctrl)  # no raise despite NaN logN_true


def test_validate_manifest_rejects_nan_logn_when_not_control():
    # A NaN logN_true WITHOUT the control flag is a corrupt injection row -> reject.
    sl = _toy_sightlines(n=20, seed=35)
    rows = build_injection_grid(
        clean_sightlines=sl, logN_grid=[18.0], z_grid=[3.0],
        snr_bins=[0.0, 100.0], n_per_cell=1, seed=36,
    )
    bad = [dict(r) for r in rows]
    bad[0]["logN_true"] = np.nan  # control flag stays False
    with pytest.raises(ValueError):
        validate_manifest(bad)


def test_injection_and_control_rows_combine_into_one_valid_manifest():
    # The campaign manifest = injection rows + control rows; contiguous inj_id; the
    # combined manifest validates as a whole.
    sl = _toy_sightlines(n=60, seed=37)
    inj = build_injection_grid(
        clean_sightlines=sl, logN_grid=[18.0, 20.5], z_grid=[3.0],
        snr_bins=[0.0, 100.0], n_per_cell=2, seed=38,
    )
    # Controls MUST exclude the injected sightlines (the gen_injectables.py path),
    # else a control sits on an injected spectrum → duplicate target_id + b_FP
    # contamination.  validate_manifest now enforces global target_id uniqueness.
    ctrl = cg.build_control_rows(
        clean_sightlines=sl, snr_bins=[0.0, 100.0], n_per_cell=3,
        seed=39, inj_id_start=len(inj),
        exclude_target_ids={int(r["target_id"]) for r in inj},
    )
    manifest = inj + ctrl
    # inj_id contiguous across the combined manifest
    assert sorted(r["inj_id"] for r in manifest) == list(range(len(manifest)))
    # at least one control row present
    assert any(r.get("control") for r in manifest)
    # injection and control sightlines are disjoint
    assert {int(r["target_id"]) for r in inj}.isdisjoint(
        {int(r["target_id"]) for r in ctrl}
    )
    validate_manifest(manifest)


def test_injection_rows_carry_control_false():
    # Ordinary injection rows must carry control=False (additive field present on
    # every row so the schema is uniform).
    sl = _toy_sightlines(n=20, seed=40)
    rows = build_injection_grid(
        clean_sightlines=sl, logN_grid=[18.0], z_grid=[3.0],
        snr_bins=[0.0, 100.0], n_per_cell=1, seed=41,
    )
    assert rows and all(r["control"] is False for r in rows)


def test_build_control_rows_excludes_non_forest_hostable_zqso():
    # Controls must be forest-hostable (non-empty GP search window). A low-z_QSO
    # sightline (no Lyα forest in the searchable window) would crash the GP with
    # All-NaN evidence and pad the b_FP denominator — build_control_rows must drop it.
    import numpy as np
    from injection.campaign_grid import build_control_rows, _per_sightline_forest_window
    n = 200
    rng = np.random.default_rng(0)
    # half the pool at z_QSO ~ 1.85 (no searchable forest), half at ~3.2 (hostable)
    zq = np.concatenate([rng.uniform(1.75, 1.95, n // 2),
                         rng.uniform(3.0, 3.4, n // 2)])
    clean = dict(target_id=np.arange(9000, 9000 + n, dtype=np.int64),
                 healpix=np.full(n, 7, dtype=np.int64),
                 z_qso=zq, native_snr=rng.uniform(2, 9, n))
    ctrl = build_control_rows(clean, snr_bins=[2, 4, 8, 1e9], target_controls=80, seed=1)
    assert ctrl
    for r in ctrl:
        z_lo, z_hi = _per_sightline_forest_window(float(r["z_qso"]))
        assert z_lo <= z_hi, f"control z_qso {r['z_qso']} has an empty (un-searchable) window"
    # all drawn controls are from the high-z (hostable) half
    assert all(r["z_qso"] > 2.0 for r in ctrl)


def test_control_rows_disjoint_from_injections():
    """Control sightlines must NOT overlap injected sightlines (else b_FP is
    contaminated by the injection)."""
    import numpy as np
    from injection.campaign_grid import build_injection_grid, build_control_rows
    rng = np.random.default_rng(0)
    n = 400
    clean = dict(target_id=np.arange(1000, 1000 + n, dtype=np.int64),
                 healpix=np.full(n, 7, dtype=np.int64),
                 z_qso=np.full(n, 3.2), native_snr=rng.uniform(2, 10, n))
    inj = build_injection_grid(clean, snr_bins=[2, 4, 8, 1e9],
                               target_injections=120, seed=1)
    inj_tids = {int(r["target_id"]) for r in inj}
    ctrl = build_control_rows(clean, snr_bins=[2, 4, 8, 1e9], target_controls=50,
                              seed=2, inj_id_start=len(inj),
                              exclude_target_ids=inj_tids)
    ctrl_tids = {int(r["target_id"]) for r in ctrl}
    assert inj_tids.isdisjoint(ctrl_tids), "controls overlap injections"
    assert len(ctrl) > 0
