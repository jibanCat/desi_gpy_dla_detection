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
    # S6 has no estimand-level number yet and must say so, not invent one
    s6 = _sysid(systematics, "S6")
    assert s6["summary"]["estimand_level_size_pp"] is None
    assert s6["summary"]["estimand_level_status"].startswith("PENDING")


@_have
def test_systematics_reproduces_published_response_form_sensitivity(
        systematics):
    """Section 3 of FINAL_CAMPAIGN_RETURN_2026-09-14.md, to 0.01 pp."""
    rows = {(r["family"], r["threshold"]): r
            for r in _sysid(systematics, "S2")["rows"]
            if r["fp_configuration"] == "ORACLE"}
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
