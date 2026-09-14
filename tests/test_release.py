"""Tests for the Zenodo release machinery (``validation/release/``).

VALIDATION-ONLY code: the provenance manifest, the completeness release and the
response release.  Every numeric comparison carries an EXPLICIT tolerance (never
``np.allclose``'s vacuous default), and the fail-closed tests are MUTATION
tests: a byte is flipped in a product and the builder must refuse, so a test
that merely "passes" on clean inputs cannot hide a disarmed gate.

The tests that need the frozen mock products skip cleanly when those products
are not mounted; the fail-closed and schema tests are self-contained.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REL = os.path.abspath(os.path.join(_HERE, "..", "validation", "release"))
for _p in (_REL, os.path.join(_REL, "standalone")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import completeness_release as CR                              # noqa: E402
import evaluate_completeness as EC                             # noqa: E402
import evaluate_response as ER                                 # noqa: E402
import hashutil as HU                                          # noqa: E402
import provenance_manifest as PM                               # noqa: E402
import response_release as RR                                  # noqa: E402

PRODUCTS = ("/scratch/cavestru_root/cavestru0/mfho/"
            "absorber_ladder_2026-09-13")
# the antitautology product was legitimately appended to after its SHA256SUMS
# was stamped; the override must be typed out, which is the point of the gate.
ACCEPT_STALE = {"antitautology_results.json":
                "85671f9bd0a0d9c099846c39837edf13f452c7e7fe50df7a13aa4628e9c"
                "3bfc9"}

_have = pytest.mark.skipif(not os.path.isdir(PRODUCTS),
                           reason="frozen mock products not mounted")


# ==========================================================================
# fixtures
# ==========================================================================
@pytest.fixture(scope="module")
def release(tmp_path_factory):
    """Build both release products once into a temporary tree."""
    if not os.path.isdir(PRODUCTS):
        pytest.skip("frozen mock products not mounted")
    out = str(tmp_path_factory.mktemp("release"))
    CR.build(PRODUCTS, out)
    RR.build(PRODUCTS, out, accept_stale=ACCEPT_STALE)
    return out


@pytest.fixture(scope="module")
def manifest_pair(tmp_path_factory):
    """(out_dir, manifest) -- the out dir is part of the environment lock path
    and therefore part of the manifest, so idempotence is only meaningful at a
    fixed out dir."""
    if not os.path.isdir(PRODUCTS):
        pytest.skip("frozen mock products not mounted")
    out = str(tmp_path_factory.mktemp("manifest"))
    b = PM.ManifestBuilder(PRODUCTS, out, strict=False,
                           accept_stale=ACCEPT_STALE)
    return out, b.build(runs_dir=os.path.join(PRODUCTS, "runs"))


@pytest.fixture(scope="module")
def manifest(manifest_pair):
    return manifest_pair[1]


def _mini_product(root, payload=b"mock-bytes-v1"):
    """A one-file product directory with a correct SHA256SUMS."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "thing.npz")
    with open(path, "wb") as fh:
        fh.write(payload)
    HU.write_sha256sums(root)
    return path


# ==========================================================================
# 1-5: fail-closed / integrity
# ==========================================================================
def test_sha_mismatch_is_fatal_mutation(tmp_path):
    """MUTATION: flip one byte and the verifier must refuse, not warn."""
    root = str(tmp_path / "prod")
    path = _mini_product(root)
    good, status = HU.verify_against_sums(path)
    assert status == "match"
    with open(path, "wb") as fh:
        fh.write(b"mock-bytes-v2")            # same length, different content
    assert HU.sha256_file(path) != good
    with pytest.raises(HU.ManifestIntegrityError) as err:
        HU.verify_against_sums(path)
    assert "MISMATCH" in str(err.value)


def test_missing_file_is_fatal_in_strict_mode(tmp_path):
    b = PM.ManifestBuilder(str(tmp_path / "nonexistent-products"),
                           str(tmp_path / "out"), strict=True)
    with pytest.raises(HU.ManifestIntegrityError) as err:
        b.add("cal:events", "calibration_events",
              str(tmp_path / "nope" / "calib_events_2lpt0.npz"))
    assert "FAIL CLOSED" in str(err.value)


def test_missing_file_is_recorded_not_swallowed_when_not_strict(tmp_path):
    b = PM.ManifestBuilder(str(tmp_path), str(tmp_path / "out"), strict=False)
    b.add("cal:events", "calibration_events", str(tmp_path / "nope.npz"))
    assert b.nodes["cal:events"]["exists"] is False
    assert b.nodes["cal:events"]["sha256sums_status"] == "MISSING"
    assert len(b.unresolved) == 1
    assert b.unresolved[0]["path"].endswith("nope.npz")


def test_accept_stale_requires_the_exact_observed_digest(tmp_path):
    root = str(tmp_path / "prod")
    path = _mini_product(root)
    with open(path, "wb") as fh:
        fh.write(b"appended-later")
    observed = HU.sha256_file(path)
    with pytest.raises(HU.ManifestIntegrityError):
        HU.verify_against_sums(path, None, {"thing.npz": "0" * 64})
    digest, status = HU.verify_against_sums(path, None,
                                            {"thing.npz": observed})
    assert status == "STALE-SUMS-ACCEPTED" and digest == observed


def test_sha256sums_roundtrip_and_nested_exclusion(tmp_path):
    root = str(tmp_path / "tree")
    os.makedirs(os.path.join(root, "sub"))
    for rel in ("a.txt", os.path.join("sub", "b.txt")):
        with open(os.path.join(root, rel), "w") as fh:
            fh.write(rel)
    HU.write_sha256sums(os.path.join(root, "sub"))
    HU.write_sha256sums(root)
    sums = HU.load_sha256sums(root)
    assert set(sums) == {"a.txt", "b.txt"}      # nested SHA256SUMS excluded
    assert sums["a.txt"] == HU.sha256_bytes("a.txt")


# ==========================================================================
# 6-8: manifest schema + idempotence
# ==========================================================================
@_have
def test_manifest_validates_against_its_own_schema(manifest):
    assert PM.validate(manifest) == []
    assert manifest["schema_version"] == PM.SCHEMA_VERSION
    assert manifest["summary"]["n_runs"] > 0
    assert manifest["summary"]["n_paper_numbers"] > 0


def test_schema_validator_has_power_it_rejects_bad_manifests():
    """A validator that cannot fail is worthless; show it fails."""
    bad = {"schema_version": "x"}
    assert any("missing required key" in e for e in PM.validate(bad))
    good_edge = {"from": "a", "to": "b", "kind": "consumed_by"}
    assert PM.validate(good_edge,
                       PM.MANIFEST_SCHEMA["properties"]["edges"]["items"]) == []
    bad_edge = dict(good_edge, kind="inspired_by")
    assert PM.validate(
        bad_edge,
        PM.MANIFEST_SCHEMA["properties"]["edges"]["items"]) != []
    bad_node = {"id": "n", "role": "r", "path": None, "sha256": None,
                "exists": True, "sha256sums_status": "probably-fine"}
    assert PM.validate(
        bad_node,
        PM.MANIFEST_SCHEMA["properties"]["nodes"]["items"]) != []


@_have
def test_manifest_is_idempotent(manifest_pair):
    out, manifest = manifest_pair
    b = PM.ManifestBuilder(PRODUCTS, out, strict=False,
                           accept_stale=ACCEPT_STALE)
    again = b.build(runs_dir=os.path.join(PRODUCTS, "runs"))
    assert again["content_digest"] == manifest["content_digest"]
    assert again["generated_utc"] >= manifest["generated_utc"]
    assert PM.content_digest(again) == again["content_digest"]


@_have
def test_manifest_chain_calibration_to_paper_number(manifest):
    """The graph really connects the five layers the PI asked for."""
    nodes = {n["id"]: n for n in manifest["nodes"]}
    roles = {n["role"] for n in nodes.values()}
    for role in ("calibration_events", "completeness_object",
                 "response_object", "support_pack", "hbi_run", "posterior",
                 "paper_number", "fp_product", "environment_lock",
                 "predeclaration"):
        assert role in roles, role
    kinds = {e["kind"] for e in manifest["edges"]}
    assert kinds <= set(PM.EDGE_KINDS) and kinds == set(PM.EDGE_KINDS)
    # completeness object <- calibration table; run -> posterior -> number
    assert any(e["kind"] == "fitted_from"
               and e["from"] == "fit:completeness:2lpt0" for e in
               manifest["edges"])
    runs = [n for n in nodes.values() if n["role"] == "hbi_run"]
    a_run = runs[0]["id"]
    post = [e["to"] for e in manifest["edges"]
            if e["from"] == a_run and e["kind"] == "reduced_to"]
    assert post, "run has no posterior"
    numbers = [e["to"] for e in manifest["edges"]
               if e["from"] in post and e["kind"] == "reduced_to"]
    assert numbers, "posterior reduces to no paper number"
    for nid in numbers:
        assert nodes[nid]["role"] == "paper_number"
        assert nodes[nid]["config"]["support_id"]
    assert any(nodes[nid]["config"].get("threshold") == "ge20.3"
               for nid in numbers)
    assert nodes[a_run]["config"]["seed"] is not None
    assert nodes[a_run]["code_commit"]


# ==========================================================================
# 9-12: the completeness product
# ==========================================================================
@_have
def test_standalone_evaluator_reproduces_the_fitted_table(release):
    m = np.load(os.path.join(release, "completeness", "completeness_model.npz"))
    coef, N0, U0 = m["coef"], float(m["N0"]), float(m["U0"])
    bcen, live = m["ntrue_centres"], m["live_strata"]
    logmed = m["log10_snr_median_stratum"]
    table = m["C_calibration_grid"]
    worst = 0.0
    for s in live:
        got = EC.evaluate_completeness(
            bcen, coef=coef, N0=N0, U0=U0, clamp=True,
            log10_snr_clamp_hi=float(m["log10_snr_clamp_hi"]),
            log10_snr=float(logmed[s]))
        worst = max(worst, float(np.max(np.abs(got - table[s]))))
    assert worst < 1e-10, worst


@_have
def test_clamp_is_active_above_the_cap_and_inert_below(release):
    m = np.load(os.path.join(release, "completeness", "completeness_model.npz"))
    coef, N0 = m["coef"], float(m["N0"])
    cap = float(m["snr_clamp_hi"])
    at_cap = EC.evaluate_completeness(20.3, cap, coef, N0=N0, clamp=True)
    above = EC.evaluate_completeness(20.3, 5.0 * cap, coef, N0=N0, clamp=True)
    assert abs(float(above) - float(at_cap)) < 1e-14      # clamped: identical
    above_free = EC.evaluate_completeness(20.3, 5.0 * cap, coef, N0=N0,
                                          clamp=False)
    assert abs(float(above_free) - float(at_cap)) > 1e-6  # and it MATTERS
    for snr in (2.5, 4.0, 8.0, cap * 0.999):
        a = EC.evaluate_completeness(20.3, snr, coef, N0=N0, clamp=True)
        b = EC.evaluate_completeness(20.3, snr, coef, N0=N0, clamp=False)
        assert float(a) == float(b)                        # inert below


@_have
def test_completeness_product_contents_and_domain(release):
    d = os.path.join(release, "completeness")
    for name in ("completeness_model.npz", "completeness_model.json",
                 "evaluate_completeness.py", "completeness_surface.csv",
                 "C_vs_N_at_representative_snr.csv",
                 "C_vs_snr_at_representative_N.csv",
                 "calibration_density.csv", "README.md"):
        assert os.path.isfile(os.path.join(d, name)), name
    meta = json.load(open(os.path.join(d, "completeness_model.json")))
    assert meta["basis"]["n_coef"] == 6
    assert meta["basis"]["N0"] == 20.0
    assert meta["production_clamp_rule"]["predeclared"] is True
    lo, hi = meta["valid_domain"]["snr_calibrated"]
    assert 2.43 < lo < 2.45 and 10.6 < hi < 10.7
    assert meta["evaluation_code"]["reproduces_calibration_table_to"] < 1e-10
    m = np.load(os.path.join(d, "completeness_model.npz"))
    for key in ("cov_fisher", "cov_bootstrap", "cov_halfsplit"):
        assert m[key].shape == (6, 6)
        assert np.all(np.linalg.eigvalsh(0.5 * (m[key] + m[key].T)) > -1e-12)
    assert m["beta_bootstrap"].shape[1] == 6
    assert m["truth_counts_bs"].sum() > 0


def test_evaluator_refuses_ambiguous_or_malformed_input():
    with pytest.raises(ValueError):
        EC.evaluate_completeness(20.3, coef=np.zeros(6))          # no S/N
    with pytest.raises(ValueError):
        EC.evaluate_completeness(20.3, 4.0, np.zeros(6), log10_snr=0.6)
    with pytest.raises(ValueError):
        EC.evaluate_completeness(20.3, 4.0, np.zeros(5))          # wrong dof
    with pytest.raises(ValueError):
        EC.evaluate_completeness(20.3, -1.0, np.zeros(6))         # S/N <= 0


# ==========================================================================
# 13-16: the response product
# ==========================================================================
@_have
@pytest.mark.parametrize("fam", ["E", "B"])
def test_released_Q_rows_sum_to_one(release, fam):
    m = np.load(os.path.join(release, "response",
                             "response_model_%s.npz" % fam))
    dev = ER.row_sum_check(m, atol=1e-12)
    assert dev < 1e-12, dev
    assert np.all(np.asarray(m["Q"]) >= 0.0)


@_have
@pytest.mark.parametrize("fam", ["E", "B"])
def test_released_Mg_equals_phi_times_Q(release, fam):
    m = np.load(os.path.join(release, "response",
                             "response_model_%s.npz" % fam))
    src = np.load(os.path.join(PRODUCTS, "response_review", "candidates",
                               "Mg_%s_2lpt0.npz" % fam), allow_pickle=True)
    Mg = np.asarray(src["Mg"], float)              # (S, kf, C, B)
    kz = np.asarray(m["kz_to_K"], int)
    M = ER.operator_M(m)                           # (B, S, K, C)
    rec = np.zeros_like(Mg)
    for kf in range(Mg.shape[1]):
        rec[:, kf, :, :] = np.transpose(M[:, :, kz[kf], :], (1, 2, 0))
    assert float(np.max(np.abs(rec - Mg))) == 0.0


@_have
def test_released_phi_smooth_rebuilds_from_its_six_coefficients(release):
    m = np.load(os.path.join(release, "response", "response_model_E.npz"))
    coef, bcen, vmid = m["phi_smooth_coef"], m["ntrue_centres"], m["vmid_s"]
    assert coef.shape == (6,)
    got = np.array([[[ER.phi_smooth(b, s, k, coef, bcen, vmid)
                      for k in range(m["phi_smooth"].shape[2])]
                     for s in range(m["phi_smooth"].shape[1])]
                    for b in range(m["phi_smooth"].shape[0])])
    assert float(np.max(np.abs(got - m["phi_smooth"]))) < 1e-12
    # and the smooth object is NOT the measured one (a tautological pass)
    assert float(np.max(np.abs(np.asarray(m["phi"]) - m["phi_smooth"]))) > 1e-3


@_have
def test_response_cell_lookup_agrees_with_the_stored_tensor(release):
    m = np.load(os.path.join(release, "response", "response_model_B.npz"))
    row = ER.response_row(20.35, 4.0, 2.6, m)
    b, s, k = ER.cell_index(20.35, 4.0, 2.6, m)
    assert (int(b), int(s), int(k)) == (6, 4, 1)
    assert float(np.max(np.abs(row - np.asarray(m["Q"])[b, s, k]))) == 0.0
    with pytest.raises(ValueError):
        ER.response_row(23.0, 4.0, 2.6, m)          # off the calibrated grid


@_have
def test_response_product_contents_and_candidate_status(release):
    d = os.path.join(release, "response")
    for name in ("response_model_E.npz", "response_model_B.npz",
                 "response_model.json", "evaluate_response.py",
                 "Q_rows_E.csv", "Q_rows_B.csv", "phi_table.csv",
                 "row_moments_E.csv", "row_moments_B.csv", "README.md"):
        assert os.path.isfile(os.path.join(d, name)), name
    meta = json.load(open(os.path.join(d, "response_model.json")))
    assert "PENDING PI" in meta["status"]
    assert set(meta["families"]) == {"E", "B"}
    assert meta["grid"]["shape"] == {"B": 16, "S": 8, "K": 3, "C": 29}
    for fam, chk in meta["checks"].items():
        assert chk["Mg_equals_phi_times_Q"] == 0.0, fam
        assert chk["max_abs_row_sum_minus_one"] < 1e-12, fam


def test_shipped_evaluators_import_nothing_but_numpy_and_stdlib():
    """The release must be usable without this repository."""
    for name in ("evaluate_completeness.py", "evaluate_response.py"):
        src = open(os.path.join(_REL, "standalone", name)).read()
        imports = [l.strip() for l in src.splitlines()
                   if l.startswith("import ") or l.startswith("from ")]
        allowed = {"import numpy as np", "from __future__ import annotations",
                   "import math"}
        assert set(imports) <= allowed, (name, imports)


# ==========================================================================
# 17-20: fitted coefficients recovered by deterministic refit
# ==========================================================================
_COEF = os.path.join(PRODUCTS, "release", "response")
_have_coef = pytest.mark.skipif(
    not os.path.isfile(os.path.join(_COEF, "response_coefficients.json")),
    reason="coefficient recovery not built")


@_have_coef
@pytest.mark.parametrize("fam", ["E", "B"])
def test_standalone_rebuild_matches_the_refit_it_came_from(fam):
    """The SHIPPED evaluator must reproduce the refit that produced the
    coefficients -- this is the test of the evaluator, independent of whether
    the refit itself reproduced the delivered tensors."""
    coef = np.load(os.path.join(_COEF, "response_coefficients_%s.npz" % fam))
    rebuilt = (ER.rebuild_Q_E(coef) if fam == "E" else ER.rebuild_Q_B(coef))
    refit = np.asarray(coef["%s_rows_refit" % fam], float)
    assert rebuilt.shape == refit.shape
    assert float(np.max(np.abs(rebuilt.sum(axis=-1) - 1.0))) < 1e-12
    assert float(np.max(np.abs(rebuilt - refit))) < 1e-10


@_have_coef
@pytest.mark.parametrize("fam", ["E", "B"])
def test_coefficient_reproduction_number_is_recorded_not_asserted_away(fam):
    """The refit did NOT reproduce the delivered tensors to 1e-8.  The release
    must SAY SO rather than silently relabel; the delivered rows stay the
    objects of record."""
    coef = np.load(os.path.join(_COEF, "response_coefficients_%s.npz" % fam))
    delivered = np.asarray(np.load(os.path.join(
        PRODUCTS, "response_review", "candidates", "rows_%s.npz" % fam),
        allow_pickle=True)["rows"], float)
    refit = np.asarray(coef["%s_rows_refit" % fam], float)
    dev = float(np.max(np.abs(refit - delivered)))
    meta = json.load(open(os.path.join(_COEF, "response_coefficients.json")))
    claimed = meta["variants"][fam]["reproduces_delivered_rows_to"]
    assert abs(dev - claimed) < 1e-12, (dev, claimed)
    assert meta["variants"][fam]["reproduces_within_tolerance"] is (
        dev <= meta["variants"][fam]["tolerance"])
    model = json.load(open(os.path.join(_COEF, "response_model.json")))
    assert model["families"][fam]["coefficients_reproduce_rows_to"] == claimed


@_have_coef
def test_coefficient_metadata_is_honest_about_provenance_and_dof():
    meta = json.load(open(os.path.join(_COEF, "response_coefficients.json")))
    assert meta["quadrature"]["snr_top_cap"] == 40.0
    assert meta["quadrature"]["n_quad_u"] == 17
    assert meta["quadrature"]["n_quad_v"] == 5
    e = meta["variants"]["E"]
    # the stored nominal dof was stale; the corrected one adds the base kernel
    assert e["nominal_dof_corrected"] == \
        int(meta["E_base"]["n_coef"]["total_moments"]) + 62
    assert e["nominal_dof_corrected"] == 197
    assert e["nominal_dof_corrected"] != e["nominal_dof_as_stored"]
    spec = meta["E_base"]["spec"]
    assert spec["estimator"] == "sample"
    assert spec["marginalise"]["deg"] == 4
    assert "NOT the ML" in meta["E_base"]["label_as_built"]


@_have_coef
def test_release_readme_records_the_post_hoc_recovery():
    txt = open(os.path.join(_COEF, "README.md")).read()
    assert "recovered" in txt.lower() and "refit" in txt.lower()
    assert "objects of record" in txt
    meta = json.load(open(os.path.join(_COEF, "response_model.json")))
    assert meta["coefficients"]["status"].startswith("recovered POST HOC")


@_have_coef
def test_delivered_candidate_files_were_not_modified():
    """The refit must never overwrite the objects of record."""
    sums = HU.load_sha256sums(os.path.join(PRODUCTS, "response_review",
                                            "candidates"))
    for name in ("rows_E.npz", "rows_B.npz", "Mg_E_2lpt0.npz",
                 "Mg_B_2lpt0.npz"):
        path = os.path.join(PRODUCTS, "response_review", "candidates", name)
        assert HU.sha256_file(path) == sums[name], name
