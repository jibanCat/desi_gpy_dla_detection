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


# ==========================================================================
# 21-28: the FROZEN configuration (PI ruling 2026-09-14)
#
# The model freeze turns three more objects into release products: the eight
# named systematics, the model-of-record configuration and the loa-0 FP
# template.  These tests (a) reproduce the published campaign numbers from the
# run JSONs, so the table can never drift from the record, (b) MUTATE the
# inputs to show the fail-closed gates are armed, and (c) check the shipped
# README is an inventory of what is actually there.
# ==========================================================================
import model_of_record as MOR                                  # noqa: E402
import systematics_table as ST                                 # noqa: E402
import fp_release as FPR                                       # noqa: E402
import release_readme as RRM                                   # noqa: E402

# the published record.  FINAL_CAMPAIGN_RETURN_2026-09-14.md section 3:
# baseline B and alternate E (ORACLE FP), seed-mean median bias %.
PUBLISHED_S3 = {
    ("2lpt0", "ge20.0"): (+0.07, +0.14, +0.07),
    ("london0", "ge20.0"): (+0.46, +0.46, 0.00),
    ("saclay0", "ge20.0"): (+0.55, +0.69, +0.13),
    ("2lpt0", "ge20.3"): (-0.15, +0.89, +1.04),
    ("london0", "ge20.3"): (-0.61, -0.04, +0.57),
    ("saclay0", "ge20.3"): (-0.14, +0.93, +1.07),
}
# section 4 / A0_BATTERY_TABLE.md: (>=20.0, >=20.3) median bias % per a0
PUBLISHED_S4 = {
    ("2lpt0", 0.0): (-0.31, -0.23), ("london0", 0.0): (+0.15, -0.70),
    ("saclay0", 0.0): (-0.06, -0.20),
    ("2lpt0", 0.0014368): (-0.35, -0.27),
    ("london0", 0.0014368): (+0.14, -0.72),
    ("saclay0", 0.0014368): (-0.09, -0.23),
    ("2lpt0", 0.0229885): (-0.54, -0.53),
    ("london0", 0.0229885): (+0.01, -0.86),
    ("saclay0", 0.0229885): (-0.23, -0.45),
    ("2lpt0", 0.5): (-1.18, -1.16), ("london0", 0.5): (-0.61, -1.57),
    ("saclay0", 0.5): (-0.90, -1.15),
}
# section 5 / caveat (B): the ORACLE -> M1CUT estimand shift, and caveat (E):
# the transported sub-floor term.
PUBLISHED_FP_SHIFT = {"ge20.0": (-0.66, -0.35), "ge20.3": (-0.18, -0.11)}
PUBLISHED_SUBFLOOR_MAX_PP = 0.06

# one published section-3 entry is the difference of two SEPARATELY rounded
# 2-dp numbers, so it can sit exactly on the 0.01 pp bound; the epsilon is
# float slack, not tolerance slack.
TOL_PP = 0.01 + 1e-9


@pytest.fixture(scope="module")
def systematics():
    if not os.path.isdir(PRODUCTS):
        pytest.skip("frozen mock products not mounted")
    import tempfile
    out = tempfile.mkdtemp(prefix="systematics-")
    _out, doc = ST.build(PRODUCTS, out)
    return doc


def _sysid(doc, sid):
    for s in doc["systematics"]:
        if s["id"] == sid:
            return s
    raise AssertionError("no systematic %r in the table" % sid)


@_have
def test_systematics_table_has_the_eight_named_effects_and_no_quadrature(
        systematics):
    """PI 2026-09-14 sec.18: eight DISTINCT named effects, no blind quadrature."""
    ids = [s["id"] for s in systematics["systematics"]]
    assert ids == ["S%d" % i for i in range(1, 9)]
    assert systematics["n_systematics"] == 8
    assert "NO blind quadrature" in systematics["combination_rule"]
    # S1 must stay a per-bin table, never one scalar (PI sec.5)
    s1 = _sysid(systematics, "S1")
    assert "do not collapse to one scalar" in s1["presentation"]
    assert len({r["bin"] for r in s1["rows"]}) == 5
    # S2 must never be halved into a 1 sigma (PI sec.13)
    assert "never |B - E| / 2" in _sysid(systematics, "S2")["treatment"] \
        or "|B - E| / 2 is NOT a 1 sigma" in _sysid(systematics, "S2")["treatment"]
    # S6 must never invent a number.  Before the propagation exists it says
    # PENDING; after it exists the result is PRIVATE (it is evaluated against
    # the real pooled posterior), so the RELEASE product records only that the
    # propagation exists and its digest -- still no number of its own unless
    # the result file declares an explicit release-safe block.
    s6 = _sysid(systematics, "S6")
    st = s6["summary"]["estimand_level_status"]
    assert st.startswith("PENDING") or st.startswith("PROPAGATED")
    if st.startswith("PENDING"):
        assert s6["summary"]["estimand_level_size_pp"] is None
        assert s6["summary"]["result_file_found"] is None
    else:
        assert s6["summary"]["result_file_sha256"]
        if not s6["summary"].get("release_safe_block_present"):
            assert s6["summary"]["estimand_level_size_pp"] is None


@_have
def test_systematics_reproduces_published_response_form_sensitivity(
        systematics):
    """Section 3 of FINAL_CAMPAIGN_RETURN_2026-09-14.md, to 0.01 pp."""
    rows = {(r["family"], r["threshold"]): r
            for r in _sysid(systematics, "S2")["rows"]
            if r["fp_configuration"] == "ORACLE"
            and r["threshold"] in ST.THRESHOLDS}
    assert set(rows) == set(PUBLISHED_S3)
    for key, (pub_b, pub_e, pub_d) in PUBLISHED_S3.items():
        r = rows[key]
        assert abs(r["baseline_B_bias_pct"] - pub_b) <= TOL_PP, key
        assert abs(r["alternate_E_bias_pct"] - pub_e) <= TOL_PP, key
        assert abs(r["signed_E_minus_B_pp"] - pub_d) <= TOL_PP, key
    # and the envelope is one-sided (E >= B everywhere under ORACLE)
    assert all(r["signed_E_minus_B_pp"] >= -TOL_PP for r in rows.values())


@_have
def test_systematics_reproduces_published_a0_battery_and_bracket(systematics):
    """Section 4 / A0_BATTERY_TABLE.md, to 0.01 pp, plus the bracket rule."""
    s4 = _sysid(systematics, "S4")
    rows = {(r["family"], round(r["a0"], 7)): r for r in s4["rows"]}
    for (fam, a0), (pub0, pub3) in PUBLISHED_S4.items():
        r = rows[(fam, round(a0, 7))]
        assert abs(r["bias_ge20.0_pct"] - pub0) <= TOL_PP, (fam, a0)
        assert abs(r["bias_ge20.3_pct"] - pub3) <= TOL_PP, (fam, a0)
    assert abs(s4["summary"]["bracket_max_abs_delta_ge20.0_pp"] - 0.16) <= TOL_PP
    assert abs(s4["summary"]["bracket_max_abs_delta_ge20.3_pp"] - 0.20) <= TOL_PP
    # Jeffreys is an OUTER envelope, never inside the bracket
    assert all(not r["in_factor4_bracket"] for k, r in rows.items()
               if k[1] == 0.5)
    assert abs(s4["summary"]["a0_record"] - 1.0 / 174) < 1e-12


@_have
def test_systematics_reproduces_published_fp_shift_and_subfloor(systematics):
    """Section 5 caveat (B) (ORACLE -> M1CUT) and caveat (E) (sub-floor)."""
    s3 = [r for r in _sysid(systematics, "S3")["rows"]
          if "shift_M1CUT_minus_ORACLE_pp" in r]
    for thr, (lo, hi) in PUBLISHED_FP_SHIFT.items():
        vals = [r["shift_M1CUT_minus_ORACLE_pp"] for r in s3
                if r["threshold"] == thr]
        assert abs(min(vals) - lo) <= TOL_PP, thr
        assert abs(max(vals) - hi) <= TOL_PP, thr
        assert all(v < 0 for v in vals), "the FP-rule shift is signed negative"
    s7 = _sysid(systematics, "S7")
    assert s7["summary"]["max_abs_shift_pp"] <= PUBLISHED_SUBFLOOR_MAX_PP + TOL_PP
    # the phi sensitivity is a real, signed shift -- not a rounding artefact
    s5 = _sysid(systematics, "S5")
    smooth = [r["shift_pp"] for r in s5["rows"]
              if r["variant"] == "phi_smooth_6coef"]
    assert max(smooth) < 0.0 and min(smooth) > -0.5


@_have
def test_model_of_record_fails_closed_on_a_missing_object(tmp_path):
    """MUTATION: hide ONE frozen object and the builder must refuse."""
    farm = tmp_path / "products"
    farm.mkdir()
    for name in os.listdir(PRODUCTS):
        os.symlink(os.path.join(PRODUCTS, name), str(farm / name))
    out, doc = MOR.build(str(farm), str(tmp_path / "ok"))
    assert doc["n_objects"] > 20 and doc["status"].startswith("ADOPTED")

    # now rebuild with the completeness directory shadowed, one file short
    farm2 = tmp_path / "products2"
    farm2.mkdir()
    for name in os.listdir(PRODUCTS):
        if name != "completeness":
            os.symlink(os.path.join(PRODUCTS, name), str(farm2 / name))
    comp = farm2 / "completeness"
    comp.mkdir()
    src = os.path.join(PRODUCTS, "completeness")
    for name in os.listdir(src):
        if name == "C_C1nsadd_saclay0.npz":
            continue                                   # <-- the mutation
        os.symlink(os.path.join(src, name), str(comp / name))
    with pytest.raises(HU.ManifestIntegrityError) as err:
        MOR.build(str(farm2), str(tmp_path / "bad"))
    assert "FAIL CLOSED" in str(err.value)
    assert "C_C1nsadd_saclay0.npz" in str(err.value)


@_have
def test_model_of_record_fails_closed_on_a_broken_predeclaration_seal(
        tmp_path, monkeypatch):
    """MUTATION: a sealed predeclaration whose sidecar no longer matches."""
    gov = tmp_path / "gov"
    for directory, stem in MOR.PREDECLARATIONS.values():
        d = gov / directory
        d.mkdir(parents=True, exist_ok=True)
        for ext in (".md", ".sha256", ".timestamp"):
            src = os.path.join(MOR.GOV, directory, stem + ext)
            if os.path.isfile(src):
                shutil.copy(src, str(d / (stem + ext)))
    monkeypatch.setattr(MOR, "GOV", str(gov))
    # clean copy: the seal verifies
    MOR._predeclaration(*MOR.PREDECLARATIONS["final_ladder"])
    directory, stem = MOR.PREDECLARATIONS["final_ladder"]
    doc = gov / directory / (stem + ".md")
    with open(str(doc), "a") as fh:
        fh.write("\nthis line was never sealed\n")
    with pytest.raises(HU.ManifestIntegrityError) as err:
        MOR._predeclaration(directory, stem)
    assert "BROKEN SEAL" in str(err.value)


@_have
def test_fp_template_csv_round_trips(tmp_path):
    """The long-format CSV must rebuild the 29 x 8 block exactly."""
    out, spec = FPR.build(PRODUCTS, str(tmp_path))
    z = np.load(os.path.join(out, "fp_template.npz"), allow_pickle=True)
    counts = np.asarray(z["fp_counts"], np.int64)
    live = np.asarray(z["live_stratum"], bool)
    share = np.asarray(z["perks_share"], float)
    assert counts.shape == (29, 8)
    assert int(counts.sum()) == spec["template"]["n_events"] == 89
    assert int(z["K_live_cells"]) == 29 * int(live.sum())
    assert abs(float(z["a0"]) - 1.0 / int(z["K_live_cells"])) < 1e-15
    # the shares are a probability distribution over the LIVE cells only
    assert abs(share[:, live].sum() - 1.0) < 1e-12
    assert share[:, ~live].sum() == 0.0

    import csv as _csv
    with open(os.path.join(out, "fp_counts.csv")) as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == 29 * 8
    back = np.zeros_like(counts)
    back_share = np.zeros_like(share)
    edges = np.asarray(z["nhat_edges"], float)
    for r in rows:
        c = int(np.argmin(np.abs(edges[:-1] - float(r["nhat_lo"]))))
        s = int(round(float(r["snr_lo"])))
        back[c, s] = int(r["fp_counts"])
        back_share[c, s] = float(r["perks_share"])
    assert np.array_equal(back, counts)
    assert np.max(np.abs(back_share - share)) < 1e-12
    # the 8 production imputations are the deterministic Gamma quantiles
    lam = spec["lambda_posterior"]["per_family"]["2lpt0"]
    imps = lam["lambda_imputations_J8"]
    assert len(imps) == 8 and imps == sorted(imps)
    assert abs(lam["lambda_gamma_shape"] - 89.5) < 1e-12


@_have
def test_release_readme_lists_every_file_present(tmp_path):
    """The README inventory is the release tree, not a hand-kept list."""
    root = str(tmp_path / "release")
    os.makedirs(root)
    CR.build(PRODUCTS, root)
    FPR.build(PRODUCTS, root)
    ST.build(PRODUCTS, root)
    MOR.build(PRODUCTS, root)
    HU.write_sha256sums(root)
    path, inv = RRM.build(root)
    text = open(path).read()

    present = set()
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name in ("SHA256SUMS", "README.md") and base == root:
                continue
            if name == "SHA256SUMS":
                continue
            present.add(os.path.relpath(os.path.join(base, name), root))
    listed = {rel for rel, _size in inv}
    assert listed == present, present ^ listed
    for rel in sorted(present):
        assert "`%s`" % rel in text, rel
    # and the binding wording constraints travel with the package
    assert "all mock closure tests passed" in text and "NOT \"all mock" in text
    assert "not a 1 sigma" in text
    assert "1 - C` is not contamination" in text


# ==========================================================================
# 29-36: the 2026-09-15 documentation / read-out cleanup
# (PI ruling 2026-09-14b sec.6, sec.10, sec.11, sec.17)
#
# These tests pin the four things the cleanup fixed and would silently
# regress: the S1 arm of record (J = 8 production, not the J = 1 arm that was
# mislabelled "model_of_record_M1CUT"), the Omega window (the run JSONs'
# thresholds.omega_allz is the SUB-DLA window and is NOT Paper-1 Omega), the
# S8 disclosure being built from the production runs, and the real half-widths
# never reaching the release tree.  Two are MUTATION tests.
# ==========================================================================
@_have
def test_S1_arm_of_record_is_the_J8_production_arm_and_J1_is_preserved(
        systematics):
    """PI 2026-09-14b sec.6: the production J = 8 read-out, not the J = 1 table.

    The J = 1 rows are HISTORY and must still be there under their own label:
    the correction is a relabelling plus a new arm, never a deletion.
    """
    s1 = _sysid(systematics, "S1")
    arms = {r["arm"] for r in s1["rows"]}
    assert s1["arm_of_record"] == "model_of_record_M1CUT_J8_production"
    assert "model_of_record_M1CUT_J8_production" in arms
    assert "M1CUT_J1_HISTORY" in arms, "the J = 1 rows were deleted, not preserved"
    assert "ORACLE_FP_diagnostic_F1" in arms
    # the old, ambiguous label must be gone so nothing can quote it by mistake
    assert "model_of_record_M1CUT" not in arms
    # history is documented in the product itself, not only in a commit message
    assert "J = 1" in s1["history_note"] and "PRESERVED" in s1["history_note"]
    # the production arm really is eight imputations at ONE seed
    rec = [r for r in s1["rows"]
           if r["arm"] == "model_of_record_M1CUT_J8_production"]
    assert {r["n_imputations"] for r in rec} == {8}
    assert s1["summary"]["arm_of_record_seeds"] == [ST.J8_SEED]
    assert len(rec) == 3 * len(ST.THRESHOLDS) * 5          # fam x thr x bin


@_have
def test_S1_J8_rows_equal_an_independent_read_of_the_production_runs():
    """The S1 arm of record is exactly the equal-weight imputation mean.

    Recomputed here straight from the run JSONs by a different code path, so a
    change in the roll-up (e.g. silently averaging seeds and imputations
    together, or dropping an imputation) cannot pass.
    """
    if not os.path.isdir(PRODUCTS):
        pytest.skip("frozen mock products not mounted")
    j8 = ST._load_j8(PRODUCTS)
    for fam, recs in j8.items():
        assert len(recs) == 8
        assert sorted(j for _s, j, _r, _p in recs) == list(range(8))
        for t in ST.THRESHOLDS:
            want = {}
            for _s, _j, run, _p in recs:
                for c in run["perz_recovery"]["estimand"][t]["paper1_bins"]:
                    if c.get("available", True):
                        want.setdefault(c["bin"], []).append(
                            float(c["median_bias_pct"]))
            for b, vals in want.items():
                assert len(vals) == 8
                mean = sum(vals) / 8.0
                rows = ST._sys1_zbin(j8, {}, {})["rows"]
                got = [r for r in rows if r["family"] == fam
                       and r["threshold"] == t and r["bin"] == b]
                assert len(got) == 1
                assert abs(got[0]["bias_pct"] - mean) < 1e-12, (fam, t, b)
                assert abs(got[0]["imputation_spread_pp"]
                           - (max(vals) - min(vals))) < 1e-12
            break                                   # one threshold is enough
        break                                       # one family is enough


@_have
def test_omega_column_is_the_paper_window_not_the_subdla_window(systematics):
    """PI 2026-09-14b sec.10.

    ``thresholds.omega_allz`` in every run JSON is
    ``omega_subdla_195_203_allz``.  The table's Omega must be the PAPER's
    [20.3, 21.6] window instead, and it must differ from the sub-DLA number --
    on two of three families it differs in SIGN, which is what made the
    mislabel dangerous.
    """
    j8 = ST._load_j8(PRODUCTS)
    mismatched_sign = 0
    for fam, recs in j8.items():
        subs, papers = [], []
        for _s, _j, run, pth in recs:
            sub = run["thresholds"]["omega_allz"]
            assert sub["key"] == "omega_subdla_195_203_allz"
            subs.append(float(sub["median_bias_pct"]))
            papers.append(ST.omega_bias_pct(pth))
        sub_m = sum(subs) / len(subs)
        paper_m = sum(papers) / len(papers)
        if paper_m * sub_m < 0:
            mismatched_sign += 1
    # the sign disagrees on two of the three families: quoting the sub-DLA
    # number as "Omega" would have reported the wrong sign for the Paper-1
    # quantity, which is exactly why the column had to be relabelled.
    assert mismatched_sign >= 2, "the two windows should disagree in sign"
    # every applicable systematic now carries an Omega read-out
    for sid, key in (("S2", "signed_E_minus_B_pp"),
                     ("S3", "shift_M1CUT_minus_ORACLE_pp"),
                     ("S5", "shift_pp"), ("S7", "shift_pp")):
        rows = [r for r in _sysid(systematics, sid)["rows"]
                if r.get("threshold") == "omega_20p3_21p6"]
        assert rows, "no Omega row in " + sid
        assert all(key in r and r[key] is not None for r in rows), sid
    s4 = [r for r in _sysid(systematics, "S4")["rows"]]
    assert all("delta_vs_record_omega_20p3_21p6_pp" in r for r in s4)
    s1hi = [r for r in _sysid(systematics, "S1")["rows"]
            if r["arm"] == "model_of_record_M1CUT_J8_production"
            and r["threshold"] == "ge20.3"]
    assert all("omega_20p3_21p6_bias_pct" in r for r in s1hi)


@_have
def test_B5_coverage_and_seed_spread_are_stated_not_asserted_away(systematics):
    """PI 2026-09-14b sec.6, sec.18: B5's 25 % coverage and its real seed noise.

    The superseded caption claimed "seed noise <= 0.05 pp".  That holds for the
    all-z headlines and FAILS in B5 by more than an order of magnitude; the
    table must carry the measured B5 seed spread instead of the old claim.
    """
    s1 = _sysid(systematics, "S1")
    b5 = [r for r in s1["rows"] if r["bin"] == "B5"]
    assert b5 and all(abs(r["nominal_coverage"] - 0.25) < 1e-9 for r in b5)
    others = [r for r in s1["rows"] if r["bin"] != "B5"]
    assert all(abs(r["nominal_coverage"] - 1.0) < 1e-9 for r in others)
    spread = s1["summary"]["B5_seed_spread_pp_J1_two_seed_arm"]
    assert len(spread) == 6                        # 3 families x 2 thresholds
    assert max(spread.values()) > 0.5, \
        "the B5 two-seed spread is being reported as if it were <= 0.05 pp"
    assert s1["summary"]["max_imputation_spread_pp"] is not None


@_have
def test_S8_is_built_from_the_production_runs_and_carries_rank_rhat(
        systematics):
    """PI 2026-09-14b sec.11, sec.17: the S8 disclosure must be CURRENT.

    The superseded block came from the J = 1 arm and understated both tails
    (E-BFMI min 0.0812, divergences max 10).
    """
    s8 = _sysid(systematics, "S8")
    assert s8["summary"]["arm_of_record"] == "M1CUT_J8_production"
    prod = [r for r in s8["rows"] if r.get("arm") == "M1CUT_J8_production"]
    assert len(prod) == 24
    assert s8["summary"]["ebfmi_min_over_production_runs"] < 0.0812
    assert s8["summary"]["divergences_max_production"] > 10
    assert "M1CUT_J1_HISTORY" in {r.get("arm") for r in s8["rows"]}
    assert s8["summary"]["history_J1_ebfmi_min"] == pytest.approx(0.0812,
                                                                 abs=1e-9)
    # the sealed rule's statistic is now read out, not only the runner's
    assert all(r["rank_rhat_ge20.3"] is not None for r in prod)
    assert s8["summary"]["headline_rank_rhat_max_production"] < 1.05
    # the t_K nuisance sites are NOT converged and the table must say so
    assert s8["summary"]["t_K_rank_rhat_max_production"] > 1.2
    assert s8["summary"]["t_K_ess_bulk_min_production"] < 20
    assert "never be quoted as a measured FP transfer" in s8["treatment"]


@_have
def test_release_table_carries_no_real_halfwidths(systematics, tmp_path):
    """Real-data privacy: the release product is in MOCK half-width units.

    The real pooled half-widths may only be written to the PRIVATE companion,
    and the CLI must refuse to put that companion inside the release tree.
    """
    assert "MOCK" in systematics["units"]["hw68_reference"]
    blob = json.dumps(systematics)
    # real pooled half-widths, and the real result of record, stay out
    assert "real_pooled_hw68" not in blob
    assert "size_in_real_hw68" not in blob
    # the S6 result is PRIVATE: only its basename and digest may appear
    s6sum = _sysid(systematics, "S6")["summary"]
    if s6sum.get("result_file_found"):
        assert os.sep not in s6sum["result_file_found"]
        assert "record_median" not in blob and "quoted_S6" not in blob
    # ... and the S8 real block, when present, is diagnostics only
    s8 = _sysid(systematics, "S8")
    for r in s8["rows"]:
        assert "median" not in r and "bias_pct" not in r
    rel = tmp_path / "release"
    rel.mkdir()
    with pytest.raises(SystemExit):
        ST.main(["--products", PRODUCTS, "--out", str(rel), "--no-sums",
                 "--real-pooled", os.path.join(PRODUCTS, "real_c1",
                                               "REAL_C1_POOLED.json"),
                 "--private-out", str(rel / "systematics" / "private")])
    with pytest.raises(SystemExit):       # the two flags must travel together
        ST.main(["--products", PRODUCTS, "--out", str(rel), "--no-sums",
                 "--private-out", str(tmp_path / "priv")])


@_have
def test_b5_presentation_refuses_to_write_into_the_release_tree(tmp_path):
    """MUTATION test: the B5 table carries REAL values and is notes-only."""
    import b5_presentation as B5
    rel = str(tmp_path / "release")
    os.makedirs(os.path.join(rel, "inside"), exist_ok=True)
    with pytest.raises(HU.ManifestIntegrityError):
        B5.build(PRODUCTS,
                 os.path.join(PRODUCTS, "release", "systematics",
                              "SYSTEMATICS_TABLE.json"),
                 os.path.join(PRODUCTS, "real_c1", "REAL_C1_POOLED.json"),
                 os.path.join(rel, "inside"), release_root=rel)


def test_manifest_git_stamp_names_the_uncommitted_files(tmp_path):
    """A dirty stamp must be actionable, not a bare boolean (SEC 12 B-10)."""
    b = PM.ManifestBuilder(str(tmp_path), str(tmp_path), strict=False)
    g = b.git_stamp()
    assert set(g) >= {"commit", "branch", "dirty",
                      "generated_with_uncommitted_edits",
                      "n_uncommitted_edits"}
    assert isinstance(g["generated_with_uncommitted_edits"], list)
    assert g["n_uncommitted_edits"] == len(
        g["generated_with_uncommitted_edits"])
    if g["dirty"]:
        assert g["generated_with_uncommitted_edits"], \
            "dirty tree but no file named"


@_have
def test_S6_placeholder_fills_itself_when_the_result_lands(tmp_path):
    """PI 2026-09-14b sec.9: S6 stays length-less until the propagation exists.

    POWER CHECK: the loader is shown to actually fill the fields when the
    result file is present, so 'still PENDING' cannot be a dead code path.
    """
    s6 = ST._sys6_completeness(PRODUCTS, None, search_defaults=False)
    assert s6["summary"]["estimand_level_size_pp"] is None
    assert s6["summary"]["result_file_found"] is None
    assert s6["summary"]["result_file_expected"] == ST.S6_RESULT_BASENAME
    # (a) a result that declares itself release-safe fills the release fields
    safe = tmp_path / ST.S6_RESULT_BASENAME
    safe.write_text(json.dumps({
        "classification": "release-safe summary",
        "release_safe": {"estimand_level_size_pp": 0.42,
                         "estimand_level_size_omega_20p3_21p6_pp": 0.37,
                         "estimand_level_status": "PROPAGATED (linearised)"}}))
    s6b = ST._sys6_completeness(PRODUCTS, str(safe), search_defaults=False)
    assert s6b["summary"]["estimand_level_size_pp"] == 0.42
    assert s6b["summary"]["estimand_level_size_omega_20p3_21p6_pp"] == 0.37
    assert s6b["summary"]["estimand_level_status"].startswith("PROPAGATED")
    assert s6b["summary"]["result_file_found"] == ST.S6_RESULT_BASENAME
    assert s6b["summary"]["result_file_sha256"]
    # (b) MUTATION: a PRIVATE result must be recorded but never transcribed
    priv = tmp_path / "priv" / ST.S6_RESULT_BASENAME
    priv.parent.mkdir()
    priv.write_text(json.dumps({
        "classification": "PRIVATE: contains real-data estimand values",
        "estimands": {"dndx_ge20p3_allz": {"record_median": 0.0643012345678}}}))
    s6c = ST._sys6_completeness(PRODUCTS, str(priv), search_defaults=False)
    assert s6c["summary"]["result_is_private"] is True
    assert s6c["summary"]["estimand_level_size_pp"] is None
    assert "0.0643012345678" not in json.dumps(s6c)
    assert "record_median" not in json.dumps(s6c)
