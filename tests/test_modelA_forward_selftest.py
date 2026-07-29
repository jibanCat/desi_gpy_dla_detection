# -*- coding: utf-8 -*-
"""test_modelA_forward_selftest.py — the rung-9 FORWARD-MODEL failure (2026-07-28).

Locks in the two defects the truth-fold self-test localized, and the fixes.

D1  BASIS-PAD / one-sided truth support (the DOMINANT, low-N defect).
    Schema v1 forced ``ntrue_edges == nhat_edges`` ("basis-pad decision
    deferred"). The forward response has a ~+0.27 dex up-bias and ~0.28 dex
    width at the reporting floor, so the lowest observed n-hat bins are fed
    overwhelmingly by TRUE systems BELOW the floor that the truncated basis
    cannot carry. On modelA_pack_2lpt0 the mock's own truth in [19.0, 19.5) is
    39% of the ENTIRE [19.5, 22.4) window, and the observed total (88071)
    EXCEEDS the whole in-window truth (73610) — with completeness <= 1 and
    kernel row mass <= 1 the truncated fold is arithmetically incapable of
    reproducing it. Same class as B16 (a mask applied to one side only). The
    committed estimator already carries the fix (cddf_catalog_hbi.py
    ``basis_pad_floor``, 2026-06-17).

D2  RESPONSE-POLYNOMIAL EXTRAPOLATION (the high-N defect).
    ``znz_kernel.ForwardResponseModel._eval_surface`` evaluates the per-cell
    degree-2 MOMENT polynomials at ANY N with no range guard. The frozen 2LPT-0
    response was fit at 7 empirical anchors spanning ~19.35-21.22 per cell; the
    fold reaches N = 22.35. The MEASURED mean up-bias at the top anchor is
    +0.0011 dex, while the quadratic extrapolates to +0.30 dex (cell 0,0) and
    +0.78 dex (cell 0,2) — which is the whole 1.5-3.5x high-N excess.

Both are UPSTREAM of NUTS: they are visible with zero sampling.
"""
import dataclasses
import json
import os
import pathlib

import numpy as np
import pytest

from CDDF_analysis.hbi_mcmc import forward as F
from CDDF_analysis.hbi_mcmc import forward_selftest as FS
from CDDF_analysis.hbi_mcmc.pack import (
    PackSchemaError, load_pack, synthetic_pack, small_test_grid, validate_pack,
)

_PACK_DIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
             "modelA_packs")
_PACK_V1 = os.path.join(_PACK_DIR, "modelA_pack_2lpt0.npz")
_PACK_V11 = os.path.join(_PACK_DIR, "modelA_pack_2lpt0_v11.npz")


@pytest.fixture(scope="module")
def spack():
    return synthetic_pack(seed=3, **small_test_grid())


# ---------------------------------------------------------------------------
# D2 — the covariate-range guard
# ---------------------------------------------------------------------------
def test_D2_fail_closed_without_fit_range(spack):
    """A pack with no calibrated covariate range is REFUSED (no silent
    extrapolation). Pre-fix code accepted it silently — that WAS the defect."""
    stripped = dataclasses.replace(spack, resp_N_fit_range=None)
    with pytest.raises(ValueError, match="resp_N_fit_range"):
        F.build_consts(stripped)
    with pytest.raises(ValueError, match="resp_N_fit_range"):
        F.fold_mu_reference(
            np.zeros((spack.n_b, spack.n_k)),
            np.zeros((spack.n_s, spack.n_molly)),
            np.zeros((2,) + np.asarray(spack.resp_mu_coef).shape[:2]),
            np.zeros(spack.n_kk), np.zeros((spack.n_c, spack.n_s)), stripped)
    # the explicit diagnostic escape hatch still works and stamps itself
    c = F.build_consts(stripped, allow_unclamped_response=True)
    assert c.resp_clamp == "off"


def test_D2_moments_are_frozen_outside_the_calibrated_range(spack):
    """THE fix, stated as behaviour: above N_hi the response moments must STOP
    moving. Pre-fix, the fitted polynomial kept running and (on the real 2LPT-0
    surfaces) turned a MEASURED +0.001 dex bias into +0.30..+0.78 dex."""
    import jax.numpy as jnp
    ntrue = np.asarray(spack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    # squeeze the calibrated range to the bottom third of the grid
    n_hi = float(Nc[len(Nc) // 3])
    SR, ZR = np.asarray(spack.resp_mu_coef).shape[:2]
    rng = np.broadcast_to(np.array([float(ntrue[0]), n_hi]), (SR, ZR, 2)).copy()
    # a response with a STRONG quadratic so the extrapolation is unmistakable
    mu_c = np.asarray(spack.resp_mu_coef, float).copy()
    mu_c[..., 1] = 0.30
    p = dataclasses.replace(spack, resp_N_fit_range=rng, resp_mu_coef=mu_c)

    zero = jnp.zeros((2, SR, ZR))
    K_off = np.asarray(F.build_K(zero, F.build_consts(p, resp_clamp="off")))
    K_on = np.asarray(F.build_K(zero, F.build_consts(p, resp_clamp="both")))

    above = Nc > n_hi + 1e-9
    assert above.sum() >= 2, "test grid too small to have an above-range region"
    # inside the calibrated range the two agree exactly
    assert np.allclose(K_off[..., ~above], K_on[..., ~above], atol=1e-12)
    # above it the clamped kernel is CONSTANT in the moment surface: every
    # above-range column equals the column at the clamp point, shifted by the
    # bin centre only -> its bin-mass profile must NOT keep drifting upward.
    first = np.argmax(above)
    drift_off = np.abs(K_off[:, :, :, above] - K_off[:, :, :, [first]]).max()
    drift_on = np.abs(K_on[:, :, :, above] - K_on[:, :, :, [first]]).max()
    assert drift_on < drift_off, (
        f"clamped kernel drifts as much as unclamped ({drift_on} vs {drift_off})"
        " — the covariate-range guard is not engaged")


def test_D2_jnp_and_numpy_oracle_agree_under_the_clamp(spack):
    """The clamp is implemented INDEPENDENTLY in the jnp fold and the numpy
    oracle; they must still agree to 1e-10 (the module's standing contract)."""
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    B, Kf, S, M = spack.n_b, spack.n_k, spack.n_s, spack.n_molly
    SR, ZR = np.asarray(spack.resp_mu_coef).shape[:2]
    ntrue = np.asarray(spack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    r = np.broadcast_to(np.array([float(Nc[1]), float(Nc[-3])]),
                        (SR, ZR, 2)).copy()
    p = dataclasses.replace(spack, resp_N_fit_range=r)
    theta = rng.normal(-1.0, 0.3, (B, Kf))
    psi_c = rng.normal(0, 0.05, (S, M))
    psik = rng.normal(0, 0.01, (2, SR, ZR))
    lt = rng.normal(0, 0.05, spack.n_kk)
    lam = np.abs(rng.normal(1.0, 0.2, (spack.n_c, S)))
    for mode in ("both", "hi", "off"):
        c = F.build_consts(p, resp_clamp=mode)
        a = np.asarray(F.fold_mu(jnp.asarray(theta), jnp.asarray(psi_c),
                                 jnp.asarray(psik), jnp.asarray(lt),
                                 jnp.asarray(lam), c))
        b = F.fold_mu_reference(theta, psi_c, psik, lt, lam, p,
                                resp_clamp=mode)
        assert np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)) < 1e-10, mode


# ---------------------------------------------------------------------------
# D1 — the basis pad
# ---------------------------------------------------------------------------
def test_D1_schema_accepts_a_downward_basis_pad_and_refuses_an_upward_one(spack):
    """Schema v1.1: ntrue_edges may EXTEND BELOW nhat_edges (unreported
    support), never above, never shrink. Pre-fix the schema hard-refused ANY
    ntrue != nhat — that deferral is defect D1."""
    ne = np.asarray(spack.ntrue_edges, float)
    step = float(np.diff(ne)[0])
    n_pad = 4
    padded = np.concatenate([ne[0] - step * np.arange(n_pad, 0, -1), ne])
    ok = dataclasses.replace(
        spack, ntrue_edges=padded,
        truth_counts=np.zeros((len(padded) - 1, spack.n_k), dtype=np.int64))
    validate_pack(ok, allow_nonstandard_grid=True)      # must not raise
    assert ok.n_b == spack.n_b + n_pad

    up = np.concatenate([ne, ne[-1] + step * np.arange(1, n_pad + 1)])
    bad = dataclasses.replace(
        spack, ntrue_edges=up,
        truth_counts=np.zeros((len(up) - 1, spack.n_k), dtype=np.int64))
    with pytest.raises(PackSchemaError, match="TAIL subset"):
        validate_pack(bad, allow_nonstandard_grid=True)


def test_D1_padded_basis_feeds_the_lowest_observed_bins(spack):
    """Bayesian coherence: sub-floor true-N bins must contribute to mu in the
    lowest OBSERVED bins (that up-scatter is the physical content of the pad)."""
    import jax.numpy as jnp
    ne = np.asarray(spack.ntrue_edges, float)
    step = float(np.diff(ne)[0])
    n_pad = 5
    padded = np.concatenate([ne[0] - step * np.arange(n_pad, 0, -1), ne])
    p = dataclasses.replace(
        spack, ntrue_edges=padded,
        truth_counts=np.zeros((len(padded) - 1, spack.n_k), dtype=np.int64))
    c = F.build_consts(p)
    assert c.n_b == spack.n_b + n_pad
    assert c.n_c == spack.n_c            # the OBSERVED grid is untouched

    B, Kf, S = c.n_b, c.n_k, c.n_s
    SR, ZR = c.n_sr, c.n_zr
    theta = np.full((B, Kf), -50.0)      # everything off ...
    theta[:n_pad, :] = 0.0               # ... except the PAD
    mu = np.asarray(F.fold_mu(jnp.asarray(theta), jnp.zeros((S, c.n_molly)),
                              jnp.zeros((2, SR, ZR)), jnp.zeros(c.n_kk),
                              jnp.zeros((c.n_c, S)), c))
    assert mu[0].sum() > 0, ("a purely sub-floor population produced ZERO "
                             "expected counts in the lowest observed bin — the "
                             "pad is not wired into the fold")


# ---------------------------------------------------------------------------
# archival reproduction on the real (mock) 2LPT-0 pack
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(_PACK_V1), reason="2LPT-0 pack absent")
def test_rung9_forward_failure_signature_is_reproduced_without_sampling():
    """The reported rung-9 signature, from the pure forward fold: 0.165x at
    n-hat [19.5,19.6) and 1.5-3.5x above 21.7. Zero MCMC involved."""
    pack = load_pack(_PACK_V1)
    res = FS.selftest(pack, resp_clamp="off")
    tab = FS.ratio_tables(res, pack)
    rows = tab["by_nhat"]
    assert rows[0]["lo"] == pytest.approx(19.5)
    assert rows[0]["ratio"] == pytest.approx(0.1655, abs=2e-3)
    assert tab["total"]["ratio"] == pytest.approx(0.7307, abs=2e-3)
    hi = [r["ratio"] for r in rows if r["lo"] > 21.65]
    assert min(hi) > 1.45 and max(hi) > 3.0
    # the arithmetic impossibility that proves D1 on its own
    assert float(np.asarray(pack.truth_counts).sum()) < \
        float(np.asarray(pack.counts).sum())


@pytest.mark.skipif(not os.path.exists(_PACK_V11), reason="v1.1 pack absent")
def test_D2_clamp_removes_the_high_N_excess_on_the_real_pack():
    """With the covariate-range guard ON, the high-N excess collapses. (The
    low-N deficit does NOT move — D1 needs a re-extracted, basis-padded pack.)"""
    pack = load_pack(_PACK_V11)
    off = FS.ratio_tables(FS.selftest(pack, resp_clamp="off"), pack)["by_nhat"]
    on = FS.ratio_tables(FS.selftest(pack, resp_clamp="both"), pack)["by_nhat"]
    hi_off = np.array([r["ratio"] for r in off if r["lo"] > 21.65])
    hi_on = np.array([r["ratio"] for r in on if r["lo"] > 21.65])
    assert hi_off.max() > 3.0
    assert hi_on.max() < 1.6, f"clamped high-N ratios still {hi_on}"
    assert np.all(hi_on < hi_off)


# ---------------------------------------------------------------------------
# THE PRE-FLIGHT GATE + THE STAMP (2026-07-29 audit)
# ---------------------------------------------------------------------------
_REPO = pathlib.Path(__file__).resolve().parents[1]
_RUNG9_JSON = _REPO / "CDDF_analysis/hbi_mcmc/rung9_forward_selftest.json"
_SBATCH_V3 = _REPO / "slurm/greatlakes/hbi_mcmc/rung9v3_2lpt0.sbatch"


def test_selftest_stamps_a_full_40_char_sha_not_an_abbreviation():
    """ITEM 5. ``_git`` used ``rev-parse --short HEAD`` and the committed
    artifact carries ``code_commit: 'b76ded7'`` -- 7 chars, and at that commit
    forward_selftest.py DID NOT EXIST (it was added at 85ddd95), which the
    repo's own provenance audit classifies ORPHANED."""
    sha = FS._git()
    base = sha.split("-")[0]
    assert len(base) == 40, f"stamped an abbreviated SHA: {sha!r}"
    assert all(c in "0123456789abcdef" for c in base)


def test_committed_rung9_selftest_artifact_carries_a_resolvable_full_sha():
    import json as _json
    import subprocess as _sp
    d = _json.loads(_RUNG9_JSON.read_text())
    sha = d["code_commit"].split("-")[0]
    assert len(sha) == 40, f"artifact stamp is abbreviated: {sha!r}"
    # PROVENANCE, not merely format: the routine must EXIST at that commit.
    r = _sp.run(["git", "cat-file", "-e",
                 f"{sha}:CDDF_analysis/hbi_mcmc/forward_selftest.py"],
                cwd=str(_REPO), capture_output=True)
    assert r.returncode == 0, (
        f"ORPHANED stamp: forward_selftest.py does not exist at {sha}")


def test_closure_verdict_chi2_arm_is_not_vacuous():
    """FOUND WHILE VERIFYING ITEM 6, not on the task list.

    ``_closure_verdict`` did ``float(tot.get("chi2_dof", 0.0))`` but
    ``ratio_tables``'s ``total`` has only mu/obs/ratio/z -- it has NEVER
    carried ``chi2_dof``.  So the chi2/dof arm read 0.0 always and could not
    fire: a table of many mildly-off bins (each |z| under the per-bin limit)
    passed.  10 bins at z=2 is chi2/dof = 4 > 3 and must REFUSE."""
    tab = {"total": {"mu": 1.0, "obs": 1.0, "ratio": 1.0, "z": 0.0},
           "by_nhat": [{"lo": 19.5, "hi": 19.6, "mu": 100.0, "obs": 100.0,
                        "ratio": 1.0, "z": 2.0} for _ in range(10)],
           "by_z": [], "by_snr": []}
    v = FS._closure_verdict(tab, 5.0, 5.0, 3.0)
    assert v["closes"] is False, v
    assert any("chi2" in r for r in v["reasons"]), v["reasons"]
    assert v["chi2_dof"] == pytest.approx(4.0)


def test_closure_verdict_chi2_arm_passes_a_clean_table():
    tab = {"total": {"mu": 1.0, "obs": 1.0, "ratio": 1.0, "z": 0.0},
           "by_nhat": [{"lo": 19.5, "hi": 19.6, "mu": 100.0, "obs": 100.0,
                        "ratio": 1.0, "z": 0.5} for _ in range(10)],
           "by_z": [], "by_snr": []}
    v = FS._closure_verdict(tab, 5.0, 5.0, 3.0)
    assert v["closes"] is True, v["reasons"]


def test_n_pad_bins_is_zero_on_an_unpadded_pack(spack):
    """ITEM 6, second half."""
    assert spack.n_pad_bins == 0
    padded = dataclasses.replace(
        spack, ntrue_edges=np.round(
            np.concatenate([np.arange(spack.nhat_edges[0] - 0.3,
                                      spack.nhat_edges[0] - 1e-9, 0.1),
                            np.asarray(spack.nhat_edges, float)]), 10))
    assert padded.n_pad_bins == 3


def test_require_basis_pad_refuses_an_unpadded_pack(spack, monkeypatch):
    """A pre-flight that cannot see the pad cannot enforce finding D1.

    (The pack is injected rather than round-tripped: ``small_test_grid`` is a
    non-standard grid that ``load_pack`` refuses, and the grid is irrelevant to
    what is under test.)"""
    from CDDF_analysis.hbi_mcmc import pack as PK
    monkeypatch.setattr(PK, "load_pack", lambda *a, **k: spack)
    assert spack.n_pad_bins == 0
    with pytest.raises(SystemExit) as e:
        FS.main(["--pack", "/tmp/unpadded_mock.npz", "--require-basis-pad"])
    assert e.value.code != 0
    assert "n_pad_bins=0" in str(e.value)
    # ... and it does NOT refuse when the flag is absent (it is a REPORT then)
    monkeypatch.setattr(FS, "structural_probes", lambda p: {"ntrue_lo": 19.5})
    monkeypatch.setattr(FS, "selftest", lambda *a, **k: {"mu": None})
    monkeypatch.setattr(FS, "ratio_tables", lambda *a, **k: {
        "total": {"mu": 1.0, "obs": 1.0, "ratio": 1.0, "z": 0.0},
        "by_nhat": [], "by_z": [], "by_snr": []})
    out = FS.main(["--pack", "/tmp/unpadded_mock.npz"])
    assert out["n_pad_bins"] == 0


@pytest.mark.skipif(not os.path.exists(_PACK_V11), reason="v1.1 pack absent")
def test_require_closure_exits_nonzero_on_the_v11_pack():
    """ITEM 6, first half, VERIFIED BY EXECUTION rather than by reading the
    commit message: the rung-9 v3 pre-flight must actually fail closed."""
    with pytest.raises(SystemExit) as e:
        FS.main(["--pack", _PACK_V11, "--require-closure"])
    assert e.value.code != 0


@pytest.mark.skipif(not os.path.exists(_PACK_V11), reason="v1.1 pack absent")
def test_without_require_closure_the_command_is_a_report_that_exits_0():
    """Which is exactly why --require-closure is load-bearing in the sbatch."""
    out = FS.main(["--pack", _PACK_V11])
    assert out["closure_verdict"]["closes"] is False


def test_rung9v3_sbatch_preflight_is_fail_closed_and_pad_guarded():
    txt = _SBATCH_V3.read_text()
    assert "--require-closure" in txt, "pre-flight is a REPORT, not a gate"
    assert "--require-basis-pad" in txt, (
        "the pre-flight does not refuse an UNPADDED pack (finding D1)")


# --------------------------------------------------------------------------
# run_rung9: the Farr-bypass stamp, tested BEHAVIOURALLY
#
# This used to be a source-text grep for two literal substrings, which a
# semantically broken implementation keeping those substrings would pass.
# These tests instead RUN ``main`` with a stubbed sampler and read the JSON it
# writes.
# --------------------------------------------------------------------------

def _stub_rung9(monkeypatch, tmp_path):
    """Stub out pack loading and NUTS; ``main`` is exercised for real."""
    from CDDF_analysis.hbi_mcmc import run_rung9 as R9

    class _Pack:
        provenance = {"code_commit": "deadbeef", "mock": "synthetic"}

    monkeypatch.setattr(R9, "load_pack", lambda path: _Pack())
    monkeypatch.setattr(
        R9, "run_model_a",
        lambda pack, cfg: (None, {"diagnostics": {"r_hat_max": 1.0,
                                                  "policy_pass": True},
                                  "some_reduction": np.zeros(3)}))
    return R9, str(tmp_path / "pack_2lpt0.npz"), str(tmp_path / "r9.json")


def _run_rung9(monkeypatch, tmp_path, extra):
    R9, pack, out = _stub_rung9(monkeypatch, tmp_path)
    R9.main(["--pack", pack, "--out", out, "--smoke"] + extra)
    return json.loads(pathlib.Path(out).read_text())


def test_run_rung9_records_the_farr_bypass_as_a_bypass(monkeypatch, tmp_path):
    """Every prepared rung-9/10 sbatch passes --allow-low-farr (7 files under
    slurm/greatlakes/hbi_mcmc/ as of 2026-07-29), so the artifact must say so
    in a MACHINE-READABLE field, not only in a free-text reason a reader has
    to notice."""
    REASON = "on-mock self-calibration: ratio 1.67 structural"
    j = _run_rung9(monkeypatch, tmp_path, ["--allow-low-farr", REASON])
    prov = j["provenance"]
    assert prov["bypasses"] == {"allow_low_farr": REASON}, prov
    assert prov["paper_facing"] is False, prov
    assert prov["farr_gate_override"] == REASON


def test_run_rung9_without_the_bypass_stamps_an_empty_bypass_dict(monkeypatch,
                                                                  tmp_path):
    """The field must be PRESENT and empty, never absent: a downstream reader
    must be able to tell 'no bypass' from 'this artifact predates the field'."""
    prov = _run_rung9(monkeypatch, tmp_path, [])["provenance"]
    assert prov["bypasses"] == {}
    assert prov["paper_facing"] is None      # decided downstream, not here
    assert prov["farr_gate_override"] is None


def test_run_rung9_bypass_actually_switches_the_farr_gate_off(monkeypatch,
                                                              tmp_path):
    """The stamp must describe what the run DID: --allow-low-farr must reach
    ModelAConfig.enforce_farr_gate, and its absence must leave the gate ON."""
    from CDDF_analysis.hbi_mcmc import run_rung9 as R9
    seen = {}
    _stub_rung9(monkeypatch, tmp_path)
    real_cfg = R9.ModelAConfig

    def _spy(**kw):
        seen.update(kw)
        return real_cfg(**kw)

    monkeypatch.setattr(R9, "ModelAConfig", _spy)
    R9.main(["--pack", str(tmp_path / "p.npz"), "--out", str(tmp_path / "a.json"),
             "--smoke"])
    assert seen["enforce_farr_gate"] is True
    R9.main(["--pack", str(tmp_path / "p.npz"), "--out", str(tmp_path / "b.json"),
             "--smoke", "--allow-low-farr", "documented reason"])
    assert seen["enforce_farr_gate"] is False


def test_run_rung9_bypass_flows_into_the_evidence_gate(monkeypatch, tmp_path):
    """End-to-end meaning of the field: a rung-9 artifact carrying a bypass
    must be un-stampable by the evidence gate, whatever its checks say."""
    pytest.importorskip("jax")
    from CDDF_analysis.hbi_mcmc import evidence as EV
    prov = _run_rung9(monkeypatch, tmp_path,
                      ["--allow-low-farr", "documented reason"])["provenance"]
    blocks = {b: {"checks": {b + "_ok": True}, "incomplete": []}
              for b in EV.REQUIRED_BLOCKS}
    g = EV.gate(blocks, bypasses=prov["bypasses"])
    assert g["stampable"] is False and g["paper_facing"] is False
    # and the control: same blocks, no bypass -> stampable
    assert EV.gate(blocks, bypasses={})["stampable"] is True


# --------------------------------------------------------------------------
# the forward-selftest stamp: "clean tree" was an overstatement
#
# The artifact carried only `code_commit`, and `_git()`'s dirty probe is
# PATH-SCOPED to CDDF_analysis/hbi_mcmc/ -- it says nothing about the rest of
# the tree.  So the stamp could not support the phrase "clean tree".  The
# stamp now carries an explicit `code_dirty` flag AND the scope that flag was
# measured over, so the claim is exactly as strong as the measurement.
# --------------------------------------------------------------------------

def test_selftest_stamp_declares_code_dirty_and_the_scope_it_measured():
    stamp = FS._stamp_fields([("m", "/nonexistent/pack.npz")])
    assert isinstance(stamp["code_dirty"], bool)
    assert "hbi_mcmc" in stamp["code_dirty_scope"]
    # the scope string must be an honest DISCLAIMER, not a boast
    low = stamp["code_dirty_scope"].lower()
    assert "not" in low and "tree" in low, stamp["code_dirty_scope"]


def test_selftest_stamp_code_dirty_tracks_the_probe(monkeypatch):
    """Discrimination: the flag must follow the probe, not be hard-coded."""
    monkeypatch.setattr(FS, "_git_dirty", lambda: True)
    assert FS._stamp_fields([("m", "/x.npz")])["code_dirty"] is True
    monkeypatch.setattr(FS, "_git_dirty", lambda: False)
    assert FS._stamp_fields([("m", "/x.npz")])["code_dirty"] is False


def test_selftest_stamp_commit_is_a_bare_40_char_sha_and_dirt_is_a_field():
    """A '-dirty' SUFFIX makes code_commit unusable with `git cat-file -e`,
    which is the whole point of the 40-char stamp.  Dirt is a FIELD now."""
    monkeypatch_free = FS._stamp_fields([("m", "/x.npz")])["code_commit"]
    assert "-dirty" not in monkeypatch_free
    assert monkeypatch_free == "unknown" or len(monkeypatch_free) == 40


def test_the_committed_artifact_carries_the_field():
    import json as _json
    art = _json.loads(
        (_REPO / "CDDF_analysis/hbi_mcmc/rung9_forward_selftest.json").read_text())
    assert "code_dirty" in art, "artifact predates the field; re-derive it"
    assert "code_dirty_scope" in art
    assert len(art["code_commit"]) == 40 and "-dirty" not in art["code_commit"]

# the gate itself — chi2 was FAIL-OPEN (2026-07-29)
# ---------------------------------------------------------------------------
def test_ratio_tables_emits_chi2_dof_so_the_gate_is_not_fail_open(spack):
    """``_closure_verdict`` reads ``tab["total"]["chi2_dof"]``, but
    ``ratio_tables`` never emitted that key — so ``tot.get("chi2_dof", 0.0)``
    silently evaluated 0.0 <= max_chi2_dof and the chi2 leg of
    ``--require-closure`` could NEVER fire. Same fail-open class the module's
    own commit history flags. ``ratio_tables`` now emits it with the SAME
    definition ``run_posterior.forward_closure_gate`` uses (Poisson z over the
    reported n-hat bins with obs > 0)."""
    res = FS.selftest(spack)
    tab = FS.ratio_tables(res, spack)
    assert "chi2_dof" in tab["total"]
    rows = [b for b in tab["by_nhat"]
            if b["obs"] > 0 and b["lo"] >= float(spack.nhat_edges[0]) - 1e-9]
    z = np.array([b["z"] for b in rows], float)
    assert np.isclose(tab["total"]["chi2_dof"], (z ** 2).sum() / len(z))
    assert tab["total"]["n_gate_bins"] == len(z)


def test_closure_verdict_fails_on_chi2_alone():
    """``_closure_verdict``'s chi2 leg, in isolation: a table whose totals and
    per-bin z are all tiny but whose chi2/dof is huge must FAIL.

    HISTORY (2026-07-29 integration). Two branches fixed the same fail-open in
    two different places, and the MERGED contract is the stronger one:

      * ``wip/d1-basis-pad`` made ``ratio_tables`` EMIT ``total["chi2_dof"]``,
        because ``_closure_verdict`` read ``tot.get("chi2_dof", 0.0)`` and the
        key was never present, so the arm read 0.0 and could not fire.
      * ``wip/gate-failopen`` instead made ``_closure_verdict`` COMPUTE chi2/dof
        itself from the ``by_nhat`` rows, so it can no longer depend on a
        producer remembering to supply a key.

    The computed version wins on merge, which retired this test's original
    form: it planted ``total["chi2_dof"] = 999`` and asserted the gate read it.
    Under the merged contract a planted total is correctly IGNORED, so the old
    assertion inverted. Rewritten here to exercise the arm as it now works —
    and to pin the scenario the arm actually exists for, which the planted-key
    version never reached: MANY MILDLY-OFF BINS, each individually inside the
    per-bin |z| limit, whose chi2/dof is nonetheless way over tolerance."""
    # every |z| = 2.0 < max_abs_z_bin 5.0, but mean(z^2) = 4.0 > max_chi2_dof 3.0
    rows = [{"z": 2.0, "obs": 100.0} for _ in range(12)]
    tab = {"total": {"z": 0.1}, "by_nhat": rows, "by_z": [{"z": 0.1}]}
    v = FS._closure_verdict(tab, 5.0, 5.0, 3.0)
    assert v["closes"] is False, v
    assert any("chi2/dof" in r for r in v["reasons"]), v["reasons"]
    assert v["chi2_dof"] == pytest.approx(4.0)
    assert v["n_bins"] == 12
    # the |z| legs must NOT be what failed it — the chi2 arm is doing the work
    assert not any("max|z|" in r for r in v["reasons"]), v["reasons"]
    # and a planted total["chi2_dof"] must be IGNORED, not trusted
    tab_planted = {"total": {"z": 0.1, "chi2_dof": 999.0},
                   "by_nhat": [{"z": 0.1, "obs": 100.0}], "by_z": [{"z": 0.1}]}
    assert FS._closure_verdict(tab_planted, 5.0, 5.0, 3.0)["closes"] is True


def test_closure_verdict_is_fail_closed_end_to_end(spack):
    """THE powered test for the 2026-07-29 fail-open fix: run the REAL producer
    (``ratio_tables``) and feed ITS output to ``_closure_verdict`` with the |z|
    legs opened wide, so the chi2 leg is the only thing that can fail. Remove
    the ``chi2_dof`` emission from ``ratio_tables`` and this goes red, because
    ``tot.get("chi2_dof", 0.0)`` silently reads 0.0 <= tolerance."""
    tab = FS.ratio_tables(FS.selftest(spack), spack)
    # read it EXACTLY the way _closure_verdict does, so a missing key surfaces
    # as the fail-open it is rather than as a KeyError
    chi2 = tab["total"].get("chi2_dof", 0.0)
    assert chi2 > 0.0, (
        "ratio_tables produced no usable chi2_dof — _closure_verdict's "
        "tot.get('chi2_dof', 0.0) would read 0.0 and the chi2 leg could never "
        "fire (this IS the 2026-07-29 fail-open bug)")
    # |z| legs wide open (nothing can trip them), chi2 tolerance well under the
    # measured value -> the ONLY route to closes=False is the chi2 leg.
    huge = 1e9
    v = FS._closure_verdict(tab, huge, huge, chi2 / 2.0)
    assert v["closes"] is False, (
        "the chi2 leg did not fire on a table produced by ratio_tables — the "
        "gate is fail-OPEN again (chi2_dof missing from tab['total']?)")
    assert any("chi2/dof" in r for r in v["reasons"])
    # and it PASSES when the tolerance is above the measured value, so the
    # assertion above is about chi2 and not about some unrelated leg
    v_ok = FS._closure_verdict(tab, huge, huge, chi2 * 2.0)
    assert v_ok["closes"] is True
