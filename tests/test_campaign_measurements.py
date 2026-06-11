"""
tests/test_campaign_measurements.py
===================================
TDD tests for ``injection.measurements`` — the three M3 recovered-vs-
injected estimators (the Bayesian-modeling owner's scope):

  1. ``detection_completeness`` — C_det(logN_true, z, SNR): fraction of injected
     absorbers recovered with p_dla > thresh, per cell, with a Beta-Binomial CI.
  2. ``nhi_bias``               — b_N(logN_true, SNR) = <logN_rec> - logN_true and
     scatter σ_N (the headline NHI<19 bias).
  3. ``response_matrix``        — R[(N_rec,z_rec),(N_true,z_true,SNR)] (the off-
     diagonal kernel from the per-injection posterior deposit) + b_FP (Poisson
     rate from no-injection control cells).

PURE python/numpy — operates on a STUBBED "recovered" structure keyed by the
manifest ``inj_id`` (mirroring the processed-HDF5 readout the CS deposit
produces).  No desispec, no file I/O.

Recovered-record schema (the seam the CS deposit fills, one per manifest row)::

    {inj_id:int,
     p_dla:float,                      # recovered P(>=1 DLA)
     logN_rec:float, z_rec:float,      # recovered MAP / posterior-mean (NaN if none)
     deposit:[(logN, z, weight), ...]} # per-(N,z) posterior probability mass

The deposit is the streaming-machinery output: a list of posterior probability
mass deposits in (logN, z) space (the same ``p_dla`` mass
``DiagonalSoftDeposit`` routes), used to build the response columns.
"""
import numpy as np
import pytest

from injection import measurements as M
from injection.measurements import (
    detection_completeness,
    gpdraw_spec,
    nhi_bias,
    response_matrix,
)


# --------------------------------------------------------------------------- #
# fixtures: a tiny manifest + matching recovered dict
# --------------------------------------------------------------------------- #
def _manifest(cells):
    """Build a manifest from (logN_true, z_true, snr_bin, n) cell specs.

    Each cell emits ``n`` rows with contiguous inj_id; target_id/healpix/z_qso are
    filler (the estimators key on inj_id + the truth columns).
    """
    rows = []
    iid = 0
    for (logN, z, snr_bin, n) in cells:
        for _ in range(n):
            rows.append({
                "inj_id": iid,
                "campaign": "A",
                "method": "coadd",
                "target_id": 1000 + iid,
                "healpix": 7,
                "z_qso": z + 0.3,
                "snr_bin": int(snr_bin),
                "native_snr": 1.0 + snr_bin,
                "logN_true": float(logN),
                "z_true": float(z),
                "num_lines": 31,
            })
            iid += 1
    return rows


def _recovered_detect(manifest, p_by_inj, logN_rec_by_inj=None):
    """A recovered dict where each row has a given p_dla (+ optional logN_rec)."""
    rec = {}
    for r in manifest:
        iid = r["inj_id"]
        p = p_by_inj[iid]
        lr = (logN_rec_by_inj or {}).get(iid, r["logN_true"])
        rec[iid] = {
            "inj_id": iid,
            "p_dla": float(p),
            "logN_rec": float(lr),
            "z_rec": float(r["z_true"]),
            "deposit": [(float(lr), float(r["z_true"]), float(p))],
        }
    return rec


# --------------------------------------------------------------------------- #
# 1. detection_completeness
# --------------------------------------------------------------------------- #
def test_completeness_all_recovered_is_one():
    man = _manifest([(18.0, 2.6, 0, 10)])
    rec = _recovered_detect(man, {i: 0.99 for i in range(10)})
    out = detection_completeness(rec, man, p_dla_thresh=0.5)
    # single cell, all 10 recovered -> C ~ 1
    assert out["C"].shape == (1,)
    assert out["C"][0] == pytest.approx(1.0, abs=1e-9)
    assert out["n_injected"][0] == 10
    assert out["n_recovered"][0] == 10


def test_completeness_none_recovered_is_zero():
    man = _manifest([(18.0, 2.6, 0, 8)])
    rec = _recovered_detect(man, {i: 0.01 for i in range(8)})
    out = detection_completeness(rec, man, p_dla_thresh=0.5)
    assert out["C"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["n_recovered"][0] == 0


def test_completeness_half_recovered_is_half():
    man = _manifest([(18.0, 2.6, 0, 10)])
    p = {i: (0.9 if i < 5 else 0.1) for i in range(10)}
    rec = _recovered_detect(man, p)
    out = detection_completeness(rec, man, p_dla_thresh=0.5)
    assert out["C"][0] == pytest.approx(0.5, abs=1e-9)
    assert out["n_recovered"][0] == 5


def test_completeness_threshold_is_strict_greater_than():
    man = _manifest([(18.0, 2.6, 0, 4)])
    # exactly at threshold -> NOT recovered (strict >)
    rec = _recovered_detect(man, {0: 0.5, 1: 0.5001, 2: 0.7, 3: 0.49})
    out = detection_completeness(rec, man, p_dla_thresh=0.5)
    assert out["n_recovered"][0] == 2  # ids 1 and 2


def test_completeness_binomial_ci_brackets_point():
    man = _manifest([(18.0, 2.6, 0, 20)])
    p = {i: (0.9 if i < 13 else 0.1) for i in range(20)}
    rec = _recovered_detect(man, p)
    out = detection_completeness(rec, man, p_dla_thresh=0.5)
    c = out["C"][0]
    assert out["C_lo68"][0] <= c <= out["C_hi68"][0]
    assert out["C_lo95"][0] <= out["C_lo68"][0]
    assert out["C_hi95"][0] >= out["C_hi68"][0]
    # CI strictly inside [0, 1]
    assert 0.0 <= out["C_lo95"][0] and out["C_hi95"][0] <= 1.0


def test_completeness_cells_keyed_by_logN_z_snr():
    man = _manifest([(18.0, 2.6, 0, 4), (21.0, 3.2, 1, 4)])
    rec = _recovered_detect(
        man,
        {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.99, 5: 0.99, 6: 0.99, 7: 0.99},
    )
    out = detection_completeness(rec, man, p_dla_thresh=0.5)
    assert out["C"].shape == (2,)
    # cells reported with their (logN_true, z_true, snr_bin) coordinates
    coords = list(zip(out["logN_true"], out["z_true"], out["snr_bin"]))
    assert (18.0, 2.6, 0) in [(float(a), float(b), int(c)) for a, b, c in coords]
    assert (21.0, 3.2, 1) in [(float(a), float(b), int(c)) for a, b, c in coords]
    # the weak-NHI low-SNR cell is incomplete; the strong-NHI cell is complete
    cell = {(float(a), float(b), int(c)): cc
            for a, b, c, cc in zip(out["logN_true"], out["z_true"], out["snr_bin"], out["C"])}
    assert cell[(18.0, 2.6, 0)] == pytest.approx(0.0, abs=1e-9)
    assert cell[(21.0, 3.2, 1)] == pytest.approx(1.0, abs=1e-9)


def test_completeness_missing_recovered_record_counts_as_not_recovered():
    # An inj_id in the manifest but absent from `recovered` (GP crashed / no output)
    # must count as a non-detection, NOT be silently dropped from the denominator.
    man = _manifest([(18.0, 2.6, 0, 4)])
    rec = _recovered_detect(man, {0: 0.99, 1: 0.99, 2: 0.99, 3: 0.99})
    del rec[3]  # drop one recovered record
    out = detection_completeness(rec, man, p_dla_thresh=0.5)
    assert out["n_injected"][0] == 4
    assert out["n_recovered"][0] == 3
    assert out["C"][0] == pytest.approx(0.75, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. nhi_bias
# --------------------------------------------------------------------------- #
def test_nhi_bias_zero_when_recovered_equals_truth():
    man = _manifest([(18.0, 2.6, 0, 6)])
    rec = _recovered_detect(man, {i: 0.99 for i in range(6)})  # logN_rec == truth
    out = nhi_bias(rec, man)
    assert out["b_N"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["sigma_N"][0] == pytest.approx(0.0, abs=1e-9)


def test_nhi_bias_positive_when_recovered_above_truth():
    man = _manifest([(18.0, 2.6, 0, 6)])
    logN_rec = {i: 18.4 for i in range(6)}  # +0.4 dex bias (prior-pull toward 20.3)
    rec = _recovered_detect(man, {i: 0.99 for i in range(6)}, logN_rec)
    out = nhi_bias(rec, man)
    assert out["b_N"][0] == pytest.approx(0.4, abs=1e-9)


def test_nhi_bias_scatter_is_std_of_residual():
    man = _manifest([(18.0, 2.6, 0, 4)])
    logN_rec = {0: 18.0, 1: 18.2, 2: 17.8, 3: 18.0}
    rec = _recovered_detect(man, {i: 0.99 for i in range(4)}, logN_rec)
    out = nhi_bias(rec, man)
    resid = np.array([0.0, 0.2, -0.2, 0.0])
    assert out["b_N"][0] == pytest.approx(resid.mean(), abs=1e-9)
    assert out["sigma_N"][0] == pytest.approx(resid.std(ddof=0), abs=1e-9)


def test_nhi_bias_only_over_recovered_absorbers():
    # Bias is conditioned on RECOVERY (p_dla > thresh): a non-detection has no
    # meaningful logN_rec, so it is excluded from the bias average.
    man = _manifest([(18.0, 2.6, 0, 4)])
    logN_rec = {0: 18.5, 1: 18.5, 2: np.nan, 3: np.nan}
    p = {0: 0.99, 1: 0.99, 2: 0.01, 3: 0.01}
    rec = _recovered_detect(man, p, logN_rec)
    out = nhi_bias(rec, man, p_dla_thresh=0.5)
    assert out["b_N"][0] == pytest.approx(0.5, abs=1e-9)
    assert out["n_used"][0] == 2


def test_nhi_bias_keyed_by_logN_and_snr_collapses_z():
    # b_N is defined over (logN_true, SNR) — z is marginalized (per the design).
    man = _manifest([(18.0, 2.6, 0, 3), (18.0, 3.4, 0, 3)])
    logN_rec = {i: 18.3 for i in range(6)}
    rec = _recovered_detect(man, {i: 0.99 for i in range(6)}, logN_rec)
    out = nhi_bias(rec, man)
    # one cell: (logN=18.0, snr_bin=0), pooled over both z
    assert out["b_N"].shape == (1,)
    assert out["n_used"][0] == 6
    assert out["b_N"][0] == pytest.approx(0.3, abs=1e-9)


def test_nhi_bias_standard_error_present():
    man = _manifest([(18.0, 2.6, 0, 9)])
    rng = np.random.default_rng(0)
    logN_rec = {i: 18.0 + rng.normal(0.0, 0.3) for i in range(9)}
    rec = _recovered_detect(man, {i: 0.99 for i in range(9)}, logN_rec)
    out = nhi_bias(rec, man)
    # SE of the mean = sigma / sqrt(n)
    assert out["b_N_se"][0] == pytest.approx(out["sigma_N"][0] / np.sqrt(9), rel=1e-9)


# --------------------------------------------------------------------------- #
# 3. response_matrix + b_FP
# --------------------------------------------------------------------------- #
def _recovered_with_deposit(manifest, deposit_by_inj):
    rec = {}
    for r in manifest:
        iid = r["inj_id"]
        dep = deposit_by_inj[iid]
        rec[iid] = {
            "inj_id": iid,
            "p_dla": float(sum(w for _, _, w in dep)) if dep else 0.0,
            "logN_rec": dep[0][0] if dep else np.nan,
            "z_rec": dep[0][1] if dep else np.nan,
            "deposit": dep,
        }
    return rec


def test_response_matrix_diagonal_when_recovered_at_truth():
    lnhi_edges = np.array([17.0, 19.0, 20.3, 22.5])
    z_edges = np.array([2.0, 3.0, 4.0])
    # one truth cell (logN=18.0 -> bin 0, z=2.5 -> bin 0); recovered AT truth.
    man = _manifest([(18.0, 2.5, 0, 5)])
    dep = {i: [(18.0, 2.5, 1.0)] for i in range(5)}
    rec = _recovered_with_deposit(man, dep)
    out = response_matrix(rec, man, lnhi_edges=lnhi_edges, z_edges=z_edges)
    R = out["R"]  # shape (n_rec_cells, n_true_cells)
    # the single true column has all its mass in the matching recovered cell.
    true_cols = out["true_cells"]
    rec_cells = out["rec_cells"]
    jt = true_cols.index((0, 0, 0))  # (ilnhi_true, iz_true, snr_bin)
    ir = rec_cells.index((0, 0))     # (ilnhi_rec, iz_rec)
    col = R[:, jt]
    assert col[ir] == pytest.approx(1.0, abs=1e-9)  # column-normalized to 1
    assert col.sum() == pytest.approx(1.0, abs=1e-9)


def test_response_matrix_off_diagonal_migration_across_203():
    # Inject sub-DLA (logN_true=20.0 -> bin 0 of [19,20.3,22.5]); the GP recovers
    # it MIGRATED above 20.3 (bin 1) half the time -> off-diagonal mass.
    lnhi_edges = np.array([19.0, 20.3, 22.5])
    z_edges = np.array([2.0, 4.0])
    man = _manifest([(20.0, 3.0, 0, 4)])
    dep = {i: [(20.0, 3.0, 0.5), (20.5, 3.0, 0.5)] for i in range(4)}
    rec = _recovered_with_deposit(man, dep)
    out = response_matrix(rec, man, lnhi_edges=lnhi_edges, z_edges=z_edges)
    R = out["R"]
    jt = out["true_cells"].index((0, 0, 0))
    ir_diag = out["rec_cells"].index((0, 0))  # recovered at 20.0 (bin 0)
    ir_mig = out["rec_cells"].index((1, 0))   # migrated to 20.5 (bin 1)
    assert R[ir_diag, jt] == pytest.approx(0.5, abs=1e-9)
    assert R[ir_mig, jt] == pytest.approx(0.5, abs=1e-9)


def test_response_matrix_columns_sum_to_recovered_fraction():
    # A column sums to the recovered probability fraction per true absorber
    # (<= 1; the missing mass is incompleteness, recorded separately).
    lnhi_edges = np.array([17.0, 22.5])
    z_edges = np.array([2.0, 4.0])
    man = _manifest([(18.0, 3.0, 0, 4)])
    dep = {i: [(18.0, 3.0, 0.3)] for i in range(4)}  # only 0.3 recovered mass each
    rec = _recovered_with_deposit(man, dep)
    out = response_matrix(rec, man, lnhi_edges=lnhi_edges, z_edges=z_edges, normalize=False)
    jt = out["true_cells"].index((0, 0, 0))
    # un-normalized: total deposited mass = 4 injections * 0.3 = 1.2
    assert out["R"][:, jt].sum() == pytest.approx(1.2, abs=1e-9)
    # n_true per column = 4
    assert out["n_true"][jt] == 4


def test_response_matrix_bfp_from_control_cells():
    # Control cells (no injection: logN_true = NaN / campaign tag) deposit FP mass.
    lnhi_edges = np.array([17.0, 20.3, 22.5])
    z_edges = np.array([2.0, 4.0])
    # 5 control sightlines, each with a spurious 0.2 deposit in rec cell (0,0).
    man = []
    for iid in range(5):
        man.append({
            "inj_id": iid, "campaign": "A", "method": "coadd",
            "target_id": 9000 + iid, "healpix": 1, "z_qso": 3.5,
            "snr_bin": 0, "native_snr": 1.0,
            "logN_true": np.nan, "z_true": np.nan, "num_lines": 31,
        })
    dep = {iid: [(18.0, 3.0, 0.2)] for iid in range(5)}
    rec = _recovered_with_deposit(man, dep)
    out = response_matrix(rec, man, lnhi_edges=lnhi_edges, z_edges=z_edges)
    bfp = out["b_FP"]  # per recovered (N,z) cell
    ir = out["rec_cells"].index((0, 0))
    # Raw FP mass = sum of FP deposit mass over control sightlines = 5 * 0.2 = 1.0.
    assert bfp["fp_mass"][ir] == pytest.approx(1.0, abs=1e-9)
    # The deposit POINT is the Gamma posterior MODE max(mass + a - 1, 0) (the SAME
    # O3 convention the M4 subtraction uses): max(1.0 + 0.5 - 1, 0) = 0.5.
    assert bfp["deposit"][ir] == pytest.approx(0.5, abs=1e-9)
    # n_control sightlines recorded
    assert bfp["n_control"] == 5


def test_response_matrix_bfp_ci_is_poisson_gamma():
    lnhi_edges = np.array([17.0, 22.5])
    z_edges = np.array([2.0, 4.0])
    man = [{
        "inj_id": i, "campaign": "A", "method": "coadd",
        "target_id": 9000 + i, "healpix": 1, "z_qso": 3.5,
        "snr_bin": 0, "native_snr": 1.0,
        "logN_true": np.nan, "z_true": np.nan, "num_lines": 31,
    } for i in range(10)]
    dep = {i: [(19.0, 3.0, 0.5)] for i in range(10)}  # 5.0 total FP mass
    rec = _recovered_with_deposit(man, dep)
    out = response_matrix(rec, man, lnhi_edges=lnhi_edges, z_edges=z_edges)
    bfp = out["b_FP"]
    ir = out["rec_cells"].index((0, 0))
    # Gamma(shape = mass + 0.5, scale = 1): CI brackets the mode, all >= 0.
    assert bfp["lo68"][ir] <= bfp["deposit"][ir] <= bfp["hi68"][ir] + 1e-9
    assert bfp["lo95"][ir] >= 0.0


def test_response_matrix_separates_truth_cells_by_snr():
    # Same (N_true, z_true) at two SNR bins -> two distinct true columns.
    lnhi_edges = np.array([17.0, 22.5])
    z_edges = np.array([2.0, 4.0])
    man = _manifest([(18.0, 3.0, 0, 3), (18.0, 3.0, 1, 3)])
    dep = {i: [(18.0, 3.0, 1.0)] for i in range(6)}
    rec = _recovered_with_deposit(man, dep)
    out = response_matrix(rec, man, lnhi_edges=lnhi_edges, z_edges=z_edges)
    assert (0, 0, 0) in out["true_cells"]
    assert (0, 0, 1) in out["true_cells"]
    assert out["R"].shape[1] == len(out["true_cells"]) == 2


def test_response_matrix_ignores_deposit_outside_grid():
    lnhi_edges = np.array([19.0, 22.5])
    z_edges = np.array([2.5, 4.0])
    man = _manifest([(20.0, 3.0, 0, 2)])
    # one deposit inside (20.0,3.0), one outside the logN grid (18.0) -> dropped
    dep = {0: [(20.0, 3.0, 0.5), (18.0, 3.0, 0.5)],
           1: [(20.0, 3.0, 0.5), (20.0, 4.5, 0.5)]}  # second z outside grid
    rec = _recovered_with_deposit(man, dep)
    out = response_matrix(rec, man, lnhi_edges=lnhi_edges, z_edges=z_edges, normalize=False)
    jt = out["true_cells"].index((0, 0, 0))
    # only the two in-grid deposits (0.5 + 0.5) survive
    assert out["R"][:, jt].sum() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 4. gpdraw spec (the GP+DLA generative cross-check method — DESIGN ONLY)
# --------------------------------------------------------------------------- #
def test_gpdraw_spec_documents_null_gp_pieces():
    spec = gpdraw_spec(z_qso=3.2, native_snr=2.5)
    # the spec names the EXACT null_gp attributes the CS draw needs.
    needed = set(spec["null_gp_pieces"])
    assert {"this_mu", "this_M", "this_omega2", "v"} <= needed
    # generative recipe: s = this_mu + this_M @ N(0, I_k) + sqrt(this_omega2+v)*N(0,I_n)
    assert "this_mu" in spec["draw_formula"]
    assert "this_M" in spec["draw_formula"]


def test_gpdraw_spec_parameterized_by_zqso_and_snr():
    spec = gpdraw_spec(z_qso=3.0, native_snr=1.5)
    assert spec["z_qso"] == 3.0
    assert spec["native_snr"] == 1.5
    # the absorber multiplies the drawn flux by the SAME Voigt as method (ii).
    assert "inject_voigt" in spec["absorber_step"]
    # measurement is IDENTICAL to the coadd method (same estimators).
    assert spec["measurement"] == "identical to coadd: detection_completeness / nhi_bias / response_matrix"


def test_gpdraw_spec_method_tag_matches_manifest():
    spec = gpdraw_spec(z_qso=3.0, native_snr=1.5)
    assert spec["method"] == "gpdraw"  # the manifest method tag for campaign (i)


def test_gpdraw_spec_does_not_implement_the_draw():
    # Design-only: the CS agent implements the actual draw (needs the GP model
    # files); this owner provides the spec, not the sampler.
    spec = gpdraw_spec(z_qso=3.0, native_snr=1.5)
    assert spec.get("implemented", False) is False


# --------------------------------------------------------------------------- #
# M2 — b_FP must come from CONTROL rows (data-driven), not the bare prior.
# --------------------------------------------------------------------------- #
def test_bfp_is_data_driven_when_controls_present():
    # With control rows depositing FP mass, n_control>0 and b_FP reflects the
    # measured deposit (NOT the bare prior, which would be 0 with no controls).
    lnhi_edges = np.array([17.0, 22.5])
    z_edges = np.array([2.0, 4.0])
    inj = _manifest([(20.5, 3.0, 0, 3)])
    inj_dep = {i: [(20.5, 3.0, 1.0)] for i in range(3)}
    # 4 control rows (logN_true NaN), each depositing 0.5 spurious mass at (0,0).
    ctrl = []
    for j in range(4):
        iid = 3 + j
        ctrl.append({
            "inj_id": iid, "campaign": "A", "method": "coadd",
            "target_id": 9000 + iid, "healpix": 1, "z_qso": 3.5,
            "snr_bin": 0, "native_snr": 1.0,
            "logN_true": np.nan, "z_true": np.nan, "num_lines": 31,
            "control": True,
        })
    ctrl_dep = {3 + j: [(19.0, 3.0, 0.5)] for j in range(4)}
    man = inj + ctrl
    rec = {**_recovered_with_deposit(inj, inj_dep),
           **_recovered_with_deposit(ctrl, ctrl_dep)}
    out = response_matrix(rec, man, lnhi_edges=lnhi_edges, z_edges=z_edges)
    bfp = out["b_FP"]
    ir = out["rec_cells"].index((0, 0))
    assert bfp["n_control"] == 4
    # raw FP mass = 4 * 0.5 = 2.0 -> data-driven, not the bare prior (mode 0)
    assert bfp["fp_mass"][ir] == pytest.approx(2.0, abs=1e-9)
    assert bfp["deposit"][ir] > 0.0  # a measured FP rate, not zero


def test_bfp_collapses_to_prior_without_controls():
    # No control rows -> n_control == 0 and the FP deposit is the bare prior (the
    # degenerate case the M2 control-row path fixes).
    lnhi_edges = np.array([17.0, 22.5])
    z_edges = np.array([2.0, 4.0])
    man = _manifest([(20.5, 3.0, 0, 3)])
    dep = {i: [(20.5, 3.0, 1.0)] for i in range(3)}
    rec = _recovered_with_deposit(man, dep)
    out = response_matrix(rec, man, lnhi_edges=lnhi_edges, z_edges=z_edges)
    assert out["b_FP"]["n_control"] == 0
    # mode of Gamma(0 + 0.5, 1) = max(0.5 - 1, 0) = 0 -> bare-prior collapse
    assert np.all(out["b_FP"]["deposit"] == 0.0)


# --------------------------------------------------------------------------- #
# M3 — single-absorber FILTER-off topology; read the ABSORBER posterior (col 1),
# not a DLA-only p_DLA.  A true LLS recovered as a LOW-N absorber must count.
# --------------------------------------------------------------------------- #
def test_topology_is_pinned_single_absorber_filter_off():
    # The campaign topology is pinned + documented loudly so a sub_dla=True run
    # (which would zero-out LLS recoveries) is caught.
    assert M.CAMPAIGN_TOPOLOGY["sub_dla"] is False
    assert M.CAMPAIGN_TOPOLOGY["absorber_posterior_col"] == 1
    assert M.CAMPAIGN_TOPOLOGY["logn_grid_min"] == pytest.approx(17.2)
    assert M.CAMPAIGN_TOPOLOGY["logn_grid_max"] == pytest.approx(22.5)


def test_lls_recovered_as_low_n_absorber_is_counted_not_zeroed():
    # A true LLS (NHI=18) recovered by the SINGLE-ABSORBER model as a low-N
    # absorber (col-1 p(absorber) > thresh, recovered logN=18) must be counted as a
    # detection + deposit mass — NOT zeroed (the sub_dla=True artifact M3 warns of).
    lnhi_edges = np.array([17.0, 19.0, 22.5])
    z_edges = np.array([2.0, 4.0])
    man = _manifest([(18.0, 3.0, 0, 4)])
    # recovered as a low-N absorber: p(absorber)=0.8 in col 1, logN_rec=18.0
    rec = _recovered_detect(man, {i: 0.8 for i in range(4)},
                            {i: 18.0 for i in range(4)})
    # completeness counts the low-N absorber recovery
    comp = detection_completeness(rec, man, p_dla_thresh=0.5)
    assert comp["n_recovered"][0] == 4
    assert comp["C"][0] == pytest.approx(1.0, abs=1e-9)
    # the deposit lands in the LLS recovered cell (bin 0), not dropped as "not a DLA"
    dep = {i: [(18.0, 3.0, 0.8)] for i in range(4)}
    rec2 = _recovered_with_deposit(man, dep)
    out = response_matrix(rec2, man, lnhi_edges=lnhi_edges, z_edges=z_edges,
                          normalize=False)
    ir = out["rec_cells"].index((0, 0))   # recovered NHI=18 -> lnhi bin 0
    jt = out["true_cells"].index((0, 0, 0))
    assert out["R"][ir, jt] == pytest.approx(4 * 0.8, abs=1e-9)  # mass kept, not zero


# --------------------------------------------------------------------------- #
# M5 — N_HI bias recovery-conditioning (Malmquist).  nhi_bias must ALSO return
# the deposit-weighted bias over ALL injections (recovered+not) and FLAG
# low-n_used cells as bias-unreliable.
# --------------------------------------------------------------------------- #
def test_nhi_bias_flags_low_n_cells_as_unreliable():
    # A cell with n_used below n_min must be flagged bias_unreliable=True.
    man = _manifest([(18.0, 2.6, 0, 5)])
    # only 2 of 5 recovered -> n_used=2, below the default n_min (~20)
    p = {i: (0.99 if i < 2 else 0.01) for i in range(5)}
    logN_rec = {0: 18.5, 1: 18.5, 2: np.nan, 3: np.nan, 4: np.nan}
    rec = _recovered_detect(man, p, logN_rec)
    out = nhi_bias(rec, man, n_min=20)
    assert "bias_unreliable" in out
    assert out["n_used"][0] == 2
    assert bool(out["bias_unreliable"][0]) is True


def test_nhi_bias_low_n_flag_clears_when_enough():
    # With n_used >= n_min the cell is reliable (flag False).
    man = _manifest([(18.0, 2.6, 0, 25)])
    rec = _recovered_detect(man, {i: 0.99 for i in range(25)},
                            {i: 18.1 for i in range(25)})
    out = nhi_bias(rec, man, n_min=20)
    assert out["n_used"][0] == 25
    assert bool(out["bias_unreliable"][0]) is False


def test_nhi_bias_returns_full_recovered_distribution_per_cell():
    # M5: nhi_bias must ALSO expose the FULL per-cell recovered-N distribution (the
    # list/array of recovered logN over survivors) so M4 can use the full
    # distribution rather than only the recovery-conditioned mean.
    man = _manifest([(18.0, 2.6, 0, 4)])
    logN_rec = {0: 18.0, 1: 18.4, 2: 18.2, 3: 17.9}
    rec = _recovered_detect(man, {i: 0.99 for i in range(4)}, logN_rec)
    out = nhi_bias(rec, man)
    assert "logN_rec_dist" in out
    dist = out["logN_rec_dist"][0]
    assert sorted(np.round(np.asarray(dist, dtype=float), 6)) == \
        sorted([18.0, 18.4, 18.2, 17.9])


def test_nhi_bias_deposit_weighted_bias_over_all_injections():
    # M5: the deposit-weighted bias over ALL injections (recovered AND not) is the
    # population-level quantity M4 needs; it differs from the recovery-conditioned
    # mean when completeness < 1 (the Malmquist under-statement).
    man = _manifest([(18.0, 2.6, 0, 4)])
    # 2 recovered high (b=+0.4), 2 NOT recovered (deposit ~ 0 mass) -> the
    # deposit-weighted bias is pulled toward 0 vs the survivor mean +0.4.
    p = {0: 0.99, 1: 0.99, 2: 0.01, 3: 0.01}
    logN_rec = {0: 18.4, 1: 18.4, 2: np.nan, 3: np.nan}
    rec = _recovered_detect(man, p, logN_rec)
    out = nhi_bias(rec, man)
    assert "b_N_deposit" in out
    # recovery-conditioned mean is +0.4 ...
    assert out["b_N"][0] == pytest.approx(0.4, abs=1e-9)
    # ... but the deposit-weighted bias over all 4 injections is strictly smaller
    # (the 2 non-recoveries carry ~0 recovered mass), i.e. < the survivor mean.
    assert out["b_N_deposit"][0] < out["b_N"][0]
