"""test_analyze_rung9.py — rung-9 closure/diagnostics reader (analyze_rung9).

Pure-synthetic, fixed seeds, no sampler: posterior draws are FABRICATED around
the pack truth (or deliberately biased) so the closure metrics are exercised
against known answers. Mirrors the run_rung9 result-JSON layout exactly
(reductions as plain lists, diagnostics dict, t_mean/t_sd), so the reader is
tested on the same object shape it will meet on scratch.

    conda run -n gpdla-hbi python -m pytest tests/test_analyze_rung9.py -v
"""
import json

import numpy as np
import pytest

from CDDF_analysis.hbi_mcmc.pack import synthetic_pack, small_test_grid, save_pack
from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior
from CDDF_analysis.hbi_mcmc.analyze_rung9 import (
    truth_tier_table, closure_report)

SEED = 1
N_DRAWS = 400


@pytest.fixture(scope="module")
def pack():
    # dx0 large enough that tier truth counts are ~1e3+ (Poisson error << the
    # closure tolerances asserted below)
    return synthetic_pack(seed=SEED, fp_frac=0.15, **small_test_grid())


def _f_truth_hat(pack):
    """truth_counts-implied f on the (b, k) grid (the closure estimand)."""
    dN = np.diff(np.asarray(pack.ntrue_edges))
    dX_k = np.asarray(pack.dX).sum(axis=1)
    return np.asarray(pack.truth_counts, float) / (dX_k[None, :] * dN[:, None])


def _fake_result(pack, f_center, sigma=0.05, seed=2, policy_pass=True,
                 t_mean=None, t_sd=None):
    """Assemble a dict with the exact run_rung9 output-JSON layout."""
    rng = np.random.default_rng(seed)
    draws = f_center[None, :, :] * np.exp(
        sigma * rng.standard_normal((N_DRAWS,) + f_center.shape))
    red = reduce_f_posterior(draws, pack)
    KK = pack.n_kk
    diagnostics = {
        "r_hat_max": 1.002, "ess_bulk_min": 900.0, "ess_tail_min": 850.0,
        "n_divergent": 0 if policy_pass else 37,
        "flags_fired": [] if policy_pass else ["flag_divergent"],
        "policy_pass": policy_pass,
    }
    reductions = {k: np.asarray(v).tolist() for k, v in red.items()}
    reductions["farr_ratio"] = 5.0
    reductions["t_mean"] = (np.zeros(KK) if t_mean is None
                            else np.asarray(t_mean, float)).tolist()
    reductions["t_sd"] = (np.full(KK, 0.05) if t_sd is None
                          else np.asarray(t_sd, float)).tolist()
    reductions["fp_lam_total_mean"] = 12.0
    reductions["fp_lam_total_sd"] = 3.0
    out = {
        "pack": "synthetic",
        "sampler": {"warmup": 1, "samples": N_DRAWS, "chains": 1, "seed": seed,
                    "wallclock_s": 0.0},
        "reductions": reductions,
        "diagnostics": diagnostics,
        "provenance": {"routine": "test", "code_commit": "test"},
    }
    # round-trip through JSON so the reader sees lists, exactly as on scratch
    return json.loads(json.dumps(out))


# --- truth tier table -----------------------------------------------------------


def test_truth_tier_table_matches_hand_sum(pack):
    tab = truth_tier_table(pack)
    ntrue = np.asarray(pack.ntrue_edges)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    dX_k = np.asarray(pack.dX).sum(axis=1)
    kz = np.asarray(pack.kz_to_K)
    tc = np.asarray(pack.truth_counts, float)
    for thr, tag in [(20.0, "20p0"), (20.3, "20p3")]:
        sel = Nc >= thr - 1e-9
        for K in range(pack.n_kk):
            ksel = kz == K
            n_tier = tc[np.ix_(sel, ksel)].sum()
            dX_K = dX_k[ksel].sum()
            assert tab[tag]["n_truth"][K] == pytest.approx(n_tier)
            assert tab[tag]["dndx_truth"][K] == pytest.approx(n_tier / dX_K)
            omega = (tc[np.ix_(sel, ksel)].sum(axis=1)
                     * 10.0 ** (Nc[sel] - 21.0)).sum() / dX_K
            assert tab[tag]["omega_truth"][K] == pytest.approx(omega)
    # f_truth_coarse: (B, KK), counts / (dX_K * dN_b)
    f_c = np.asarray(tab["f_truth_coarse"])
    assert f_c.shape == (pack.n_b, pack.n_kk)
    b, K = 2, 0
    ksel = kz == K
    dN = np.diff(ntrue)
    assert f_c[b, K] == pytest.approx(
        tc[b, ksel].sum() / (dX_k[ksel].sum() * dN[b]))


# --- closure on truth-centered draws ----------------------------------------------


def test_closure_report_truth_centered_draws_pass(pack):
    res = _fake_result(pack, _f_truth_hat(pack))
    rep = closure_report(res, pack)
    assert rep["policy_pass"] is True
    for tag in ("20p0", "20p3"):
        for entry in rep["tiers"][tag]:
            assert entry["ratio"] == pytest.approx(1.0, abs=0.15)
            assert entry["in95"]
            assert entry["truth"] > 0
            assert entry["post_sd"] > 0
    # t block: passthrough + shrinkage vs the pack prior
    t_sigma = np.asarray(pack.t_sigma)
    for K, tb in enumerate(rep["t"]):
        assert tb["prior_sigma"] == pytest.approx(t_sigma[K])
        assert tb["shrinkage"] == pytest.approx(0.05 / t_sigma[K])
        assert abs(tb["z0"]) < 1.0
    assert rep["fp"]["lam_total_mean"] == pytest.approx(12.0)
    # CDDF cell closure: mostly inside 95%, masked bins excluded
    for cz in rep["cddf"]:
        assert cz["frac_in95"] > 0.8
    n_masked_cells = int(np.sum(np.asarray(res["reductions"]["n_mask_bins"])))
    assert n_masked_cells > 0  # small grid starts at 19.5 -> mask hits
    assert rep["n_cddf_cells_masked_out"] > 0


def test_closure_report_biased_draws_flagged(pack):
    res = _fake_result(pack, 1.6 * _f_truth_hat(pack))
    rep = closure_report(res, pack)
    bad = [e for tag in ("20p0", "20p3") for e in rep["tiers"][tag]]
    for entry in bad:
        assert entry["ratio"] == pytest.approx(1.6, abs=0.2)
    assert sum(not e["in95"] for e in bad) >= len(bad) - 1


def test_closure_report_diagnostics_gate_first(pack):
    res = _fake_result(pack, _f_truth_hat(pack), policy_pass=False)
    rep = closure_report(res, pack)
    assert rep["policy_pass"] is False
    assert rep["diagnostics"]["flags_fired"] == ["flag_divergent"]
    # closure is still computed (report honestly) but the gate leads
    assert rep["tiers"]["20p3"]


# --- CLI ---------------------------------------------------------------------------


def test_main_writes_stamped_json(pack, tmp_path):
    from CDDF_analysis.hbi_mcmc import analyze_rung9 as mod
    res = _fake_result(pack, _f_truth_hat(pack))
    rj = tmp_path / "rung9_fake.json"
    rj.write_text(json.dumps(res))
    pk = tmp_path / "pack_fake.npz"
    save_pack(pack, pk, allow_nonstandard_grid=True)
    out = tmp_path / "closure.json"
    mod.main(["--result", str(rj), "--pack", str(pk), "--out", str(out),
              "--allow-nonstandard-grid"])
    rep = json.loads(out.read_text())
    assert rep["policy_pass"] is True
    assert rep["provenance"]["routine"].endswith("analyze_rung9.py")
    assert rep["provenance"]["code_commit"]
    assert rep["provenance"]["result_provenance"]["code_commit"] == "test"
