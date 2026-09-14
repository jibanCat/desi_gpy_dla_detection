#!/usr/bin/env python
"""provenance_manifest.py -- the machine-readable provenance graph required by
PI ruling 2026-09-13d §17:

    calibration data -> fitted object -> HBI pack -> posterior -> paper number

with every hash, config, seed, environment and reduction listed.

Design rules
------------
* **Fail closed.**  A sha256 that disagrees with an existing ``SHA256SUMS``
  entry is fatal, always (an explicit, typed-out ``--accept-stale`` override is
  the only escape and is recorded in the manifest).  A referenced file that
  does not exist is fatal under ``--strict`` and recorded in ``unresolved``
  otherwise, because the final HBI ladder may still be running.
* **Idempotent.**  Everything except ``generated_utc`` is a pure function of
  the products on disk; ``content_digest`` is the sha256 of the manifest with
  the timestamp removed, so two builds of the same tree agree exactly.
* **Self-describing.**  The manifest carries its own schema and validates
  against it before being written (no jsonschema dependency).

VALIDATION-ONLY.  Mock/calibration products only; no real data; no sampler.

    python -m validation.release.provenance_manifest \
        --products /scratch/.../absorber_ladder_2026-09-13 \
        --runs-dir /scratch/.../absorber_ladder_2026-09-13/final/runs \
        --out      /scratch/.../release/PROVENANCE_MANIFEST.json [--strict]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hashutil import (sha256_bytes, sha256_file, verify_against_sums,
                      parse_accept_stale,
                      ManifestIntegrityError)                  # noqa: E402

SCHEMA_VERSION = "provenance_manifest/v1"
EDGE_KINDS = ("fitted_from", "consumed_by", "reduced_to")
FAMILIES = ("2lpt0", "london0", "saclay0")
FINALISTS = ("E", "B")
CONDA_ENV = "gpdla-hbi"

LOA0_FP = ("/nfs/turbo/lsa-cavestru/mfho/paper1_durable_inputs/"
           "loa0_fp_v1_20260615_outputs/loa0_fp_product.npz")
GOV = "/home/mfho/desi_gpy_dla_notes/governance"
PREDECL_DIRS = ("response_review_2026-09-13", "final_campaign_2026-09-13",
                "absorber_ladder_2026-09-13")

# --------------------------------------------------------------------------
# the manifest's own schema (a deliberately small JSON-Schema subset)
# --------------------------------------------------------------------------
MANIFEST_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "generated_utc", "generator", "products",
                 "strict", "git", "environment", "nodes", "edges",
                 "unresolved", "content_digest"],
    "properties": {
        "schema_version": {"type": "string"},
        "generated_utc": {"type": "string"},
        "generator": {"type": "string"},
        "products": {"type": "string"},
        "strict": {"type": "boolean"},
        "content_digest": {"type": "string"},
        "git": {"type": "object",
                "required": ["commit", "branch", "dirty"],
                "properties": {"commit": {"type": ["string", "null"]},
                               "branch": {"type": ["string", "null"]},
                               "dirty": {"type": ["boolean", "null"]}}},
        "environment": {"type": "object",
                        "required": ["conda_env", "lock_file", "lock_sha256",
                                     "python", "numpy"]},
        "nodes": {"type": "array", "items": {
            "type": "object",
            "required": ["id", "role", "path", "sha256", "exists",
                         "sha256sums_status"],
            "properties": {
                "id": {"type": "string"},
                "role": {"type": "string"},
                "path": {"type": ["string", "null"]},
                "sha256": {"type": ["string", "null"]},
                "exists": {"type": "boolean"},
                "sha256sums_status": {
                    "type": "string",
                    "enum": ["match", "absent-from-sums",
                             "STALE-SUMS-ACCEPTED", "MISSING", "n/a"]},
                "code_commit": {"type": ["string", "null"]},
                "config": {"type": ["object", "null"]},
                "predeclaration_sha256": {"type": ["string", "null"]},
                "note": {"type": ["string", "null"]},
            }}},
        "edges": {"type": "array", "items": {
            "type": "object",
            "required": ["from", "to", "kind"],
            "properties": {"from": {"type": "string"},
                           "to": {"type": "string"},
                           "kind": {"type": "string",
                                    "enum": list(EDGE_KINDS)},
                           "note": {"type": ["string", "null"]}}}},
        "unresolved": {"type": "array", "items": {"type": "object"}},
    },
}

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool,
          "number": (int, float), "integer": int, "null": type(None)}


def validate(instance, schema=None, path="$"):
    """Minimal JSON-Schema check; returns a list of human-readable errors."""
    schema = MANIFEST_SCHEMA if schema is None else schema
    errors = []
    types = schema.get("type")
    if types is not None:
        types = [types] if isinstance(types, str) else list(types)
        py = tuple(t for name in types for t in
                   (_TYPES[name] if isinstance(_TYPES[name], tuple)
                    else (_TYPES[name],)))
        ok = isinstance(instance, py)
        if ok and bool not in py and isinstance(instance, bool):
            ok = False                      # bool is not a number/string here
        if not ok:
            return ["%s: expected %s, got %s"
                    % (path, "/".join(types), type(instance).__name__)]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append("%s: %r not in %r" % (path, instance, schema["enum"]))
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append("%s: missing required key %r" % (path, key))
        for key, sub in schema.get("properties", {}).items():
            if key in instance:
                errors.extend(validate(instance[key], sub,
                                       "%s.%s" % (path, key)))
    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errors.extend(validate(item, schema["items"],
                                   "%s[%d]" % (path, i)))
    return errors


# --------------------------------------------------------------------------
class ManifestBuilder:
    def __init__(self, products, out_dir, strict=False, accept_stale=None,
                 repo=None):
        self.products = os.path.abspath(products)
        self.out_dir = os.path.abspath(out_dir)
        self.strict = bool(strict)
        self.accept_stale = accept_stale or {}
        self.repo = repo or os.path.abspath(os.path.join(_HERE, "..", ".."))
        self.nodes = {}
        self.order = []
        self.edges = []
        self.unresolved = []
        self._sums_cache = {}

    # ---- graph primitives -------------------------------------------------
    def add(self, node_id, role, path, code_commit=None, config=None,
            predeclaration_sha256=None, note=None, hash_it=True):
        if node_id in self.nodes:
            return node_id
        digest, status = None, "n/a"
        exists = True
        if path is None:
            exists = True
        elif os.path.isdir(path):
            exists, status, note = True, "n/a", (
                (note + " | " if note else "") + "directory: not hashed")
        else:
            if hash_it:
                digest, status = verify_against_sums(path, self._sums_cache,
                                                     self.accept_stale)
            else:
                status = "n/a"
                digest = sha256_file(path) if os.path.isfile(path) else None
            exists = os.path.isfile(path)
            if not exists:
                status = "MISSING"
                self.unresolved.append({"id": node_id, "role": role,
                                        "path": path,
                                        "reason": "file does not exist"})
                if self.strict:
                    raise ManifestIntegrityError(
                        "FAIL CLOSED (--strict): %s referenced by node %r does "
                        "not exist" % (path, node_id))
        self.nodes[node_id] = {
            "id": node_id, "role": role, "path": path, "sha256": digest,
            "bytes": (os.path.getsize(path)
                      if path and os.path.isfile(path) else None),
            "exists": bool(exists), "sha256sums_status": status,
            "code_commit": code_commit, "config": config,
            "predeclaration_sha256": predeclaration_sha256, "note": note}
        self.order.append(node_id)
        return node_id

    def link(self, src, dst, kind, note=None):
        if kind not in EDGE_KINDS:
            raise ValueError("unknown edge kind %r" % kind)
        self.edges.append({"from": src, "to": dst, "kind": kind, "note": note})

    # ---- ingest -----------------------------------------------------------
    def ingest_predeclarations(self):
        for d in PREDECL_DIRS:
            full = os.path.join(GOV, d)
            if not os.path.isdir(full):
                continue
            for name in sorted(os.listdir(full)):
                if not name.endswith(".sha256"):
                    continue
                stem = os.path.join(full, name[:-len(".sha256")])
                doc = stem if os.path.isfile(stem) else stem + ".md"
                sealed = open(os.path.join(full, name)).read().strip()
                nid = "predecl:%s/%s" % (d, os.path.basename(doc))
                self.add(nid, "predeclaration", doc, hash_it=False,
                         note="sealed: %s" % sealed)

    def ingest_calibration(self):
        ce = os.path.join(self.products, "response",
                          "calib_events_2lpt0.npz")
        prov = {}
        if os.path.isfile(ce):
            prov = json.loads(str(np.load(ce, allow_pickle=True)
                                  ["provenance"]))
        cat = prov.get("catalog_dir")
        truth = prov.get("truth_path")
        frozen = prov.get("frozen_npz")
        if cat:
            self.add("cal:catalogue", "calibration_catalogue", cat,
                     note="GP-DLA production catalogue of the 2LPT-0 mock")
        if truth:
            self.add("cal:truth", "calibration_truth", truth, hash_it=False,
                     note="mock HCD truth catalogue")
        if frozen:
            self.add("cal:frozen_response_npz", "calibration_reference",
                     frozen, hash_it=False,
                     note="frozen track-C forward-response reference; "
                          "recorded sha256 %s"
                          % prov.get("frozen_npz_sha256"))
        self.add("cal:events", "calibration_events", ce,
                 code_commit=prov.get("code_commit"),
                 config={k: prov.get(k) for k in
                         ("snr_min", "p_dla_min", "host_col", "xhat_floor",
                          "z_covariate", "snr_edges", "z_edges",
                          "N_ref_frozen", "deg_N_frozen", "n_events",
                          "n_uniq_tids")},
                 note="natural-pair matched truth/reported calibration events")
        for src in ("cal:catalogue", "cal:truth", "cal:frozen_response_npz"):
            if src in self.nodes:
                self.link("cal:events", src, "fitted_from")
        for fam in FAMILIES:
            p = os.path.join(self.products, "completeness",
                             "cal_table_%s.npz" % fam)
            nid = self.add("cal:table:%s" % fam, "calibration_table", p,
                           note="binned truth/detection counts, %s" % fam)
            if fam == "2lpt0":
                self.link(nid, "cal:truth", "fitted_from")

    def ingest_completeness(self):
        comp = os.path.join(self.products, "completeness")
        cov = self.add("fit:completeness:covariance",
                       "completeness_covariance",
                       os.path.join(comp, "C1nsadd_covariance_2lpt0.npz"),
                       note="Fisher + 200 bootstrap + half-split covariance")
        for fam in FAMILIES:
            p = os.path.join(comp, "C_C1nsadd_%s.npz" % fam)
            prov = {}
            if os.path.isfile(p):
                prov = json.loads(str(np.load(p, allow_pickle=True)
                                      ["provenance"].item()))
            nid = self.add("fit:completeness:%s" % fam, "completeness_object",
                           p, code_commit=prov.get("git", {}).get("commit"),
                           config={"variant": prov.get("variant"),
                                   "variant_id": prov.get("variant_id"),
                                   "meta": prov.get("meta"),
                                   "x_pivot": prov.get("x_pivot"),
                                   "u_pivot_log10_snr":
                                       prov.get("u_pivot_log10_snr"),
                                   "live_strata": prov.get("live_strata"),
                                   "cv": prov.get("cv"),
                                   "ridge": prov.get("ridge")},
                           note="FIXED completeness C1nsadd; fitted on 2LPT-0 "
                                "only; PI 2026-09-13d §6 baseline")
            self.link(nid, "cal:table:%s" % fam, "fitted_from")
            if fam == "2lpt0":
                self.link(nid, "cal:events", "fitted_from",
                          note="same calibration family")
            self.link(cov, nid, "fitted_from")

    def ingest_response(self):
        cand = os.path.join(self.products, "response_review", "candidates")
        cv = self.add("fit:response:cv_table", "response_cv",
                      os.path.join(cand, "candidate_cv_table.json"),
                      note="full-distribution 2-fold CV of every candidate")
        at = self.add("fit:response:antitautology", "response_antitautology",
                      os.path.join(self.products, "response_review",
                                   "antitautology",
                                   "antitautology_results.json"),
                      note="reweighting / train-one-slope / forward-fold / "
                           "occupancy tests + mutation controls")
        for var in FINALISTS:
            rows = self.add("fit:response:rows:%s" % var, "response_rows",
                            os.path.join(cand, "rows_%s.npz" % var),
                            note="Q[b,s,K,c], rows sum to one exactly")
            for fam in FAMILIES:
                p = os.path.join(cand, "Mg_%s_%s.npz" % (var, fam))
                prov = {}
                if os.path.isfile(p):
                    prov = json.loads(str(np.load(p, allow_pickle=True)
                                          ["provenance"]))
                nid = self.add(
                    "fit:response:%s:%s" % (var, fam), "response_object", p,
                    code_commit=prov.get("git", {}).get("commit"),
                    config={"variant": var,
                            "count_conservation":
                                prov.get("count_conservation"),
                            "phi_default": prov.get("phi", {}).get("default"),
                            "phi_smooth":
                                prov.get("phi", {}).get("smooth_alternative"),
                            "pack": prov.get("pack"),
                            "pack_sha256": prov.get("pack_sha256"),
                            "fitted_on": prov.get("fitted_on")},
                    predeclaration_sha256=prov.get("sealed_rule"),
                    note="Mg = phi * Q; CANDIDATE, model of record PENDING "
                         "PI adoption")
                self.link(nid, rows, "fitted_from")
                self.link(nid, "cal:events", "fitted_from")
                self.link(cv, nid, "consumed_by",
                          note="CV scores this candidate")
                self.link(at, nid, "consumed_by")

    def ingest_support(self, subdir="support_v3"):
        root = os.path.join(self.products, subdir)
        if not os.path.isdir(root):
            return
        for fam in FAMILIES:
            for kind, pat in (("support_pack", "scanpack_%s_b300_v3.npz"),
                              ("support_census", "fp_census_%s_v3.npz"),
                              ("support_ops", "empirical_ops_%s_v3.npz"),
                              ("support_fp_counts", "fp_counts_%s_v3.npz")):
                p = os.path.join(root, pat % fam)
                sup = p.replace(".npz", ".support.json")
                prv = p.replace(".npz", ".provenance.json")
                cfg, sid = None, None
                if os.path.isfile(sup):
                    try:
                        s = json.load(open(sup))
                        sid = s.get("support_id") or s.get("id")
                        cfg = {"support_id": sid,
                               "level": s.get("level"),
                               "fields": s.get("fields")}
                    except Exception:                        # pragma: no cover
                        cfg = None
                nid = self.add("pack:%s:%s:%s" % (subdir, kind, fam), kind, p,
                               config=cfg,
                               note="v3 corrected support (collar; Z_DLA-only; "
                                    "sentinel; row-level support_id)")
                for extra, role in ((sup, "support_contract"),
                                    (prv, "support_provenance")):
                    if os.path.isfile(extra):
                        eid = self.add("%s#%s" % (nid, role), role, extra)
                        self.link(eid, nid, "consumed_by")
                if kind == "support_pack":
                    self.link(nid, "cal:truth", "fitted_from",
                              note="mock truth of the same generation")

    def ingest_fp(self):
        nid = self.add("fp:loa0_template", "fp_product", LOA0_FP,
                       note="loa-0 false-positive calibration product; FIXED "
                            "shape, PI 2026-09-13d §11 (2,378-event product "
                            "CLOSED for Paper 1)")
        return nid

    def ingest_runs(self, runs_dir):
        if not os.path.isdir(runs_dir):
            self.unresolved.append({"id": "runs", "role": "hbi_run",
                                    "path": runs_dir,
                                    "reason": "runs directory does not exist "
                                              "(final ladder not finished)"})
            if self.strict:
                raise ManifestIntegrityError(
                    "FAIL CLOSED (--strict): runs dir %s missing" % runs_dir)
            return
        found = []
        for root, _d, files in os.walk(runs_dir):
            for name in sorted(files):
                if name.startswith("RUN_") and name.endswith(".json"):
                    found.append(os.path.join(root, name))
        found.sort()
        if not found:
            self.unresolved.append({"id": "runs", "role": "hbi_run",
                                    "path": runs_dir,
                                    "reason": "no RUN_*.json yet (final HBI "
                                              "ladder still running)"})
        for path in found:
            self._ingest_one_run(path, runs_dir)

    def _ingest_one_run(self, path, runs_dir):
        rel = os.path.relpath(path, runs_dir)
        tag = rel[:-len(".json")]
        try:
            R = json.load(open(path))
        except Exception as exc:                             # pragma: no cover
            self.unresolved.append({"id": "run:%s" % tag, "role": "hbi_run",
                                    "path": path,
                                    "reason": "unreadable: %s" % exc})
            if self.strict:
                raise
            return
        cfg = R.get("run_config", {})
        gate = R.get("support_gate", {})
        diag = R.get("diagnostics", {})
        rid = self.add(
            "run:%s" % tag, "hbi_run", path,
            code_commit=cfg.get("code_commit"),
            config={"seed": cfg.get("seed"), "chains": cfg.get("chains"),
                    "warmup": cfg.get("warmup"), "samples": cfg.get("samples"),
                    "target_accept": cfg.get("target_accept"),
                    "argv": cfg.get("argv"), "ladder": R.get("ladder"),
                    "stage": R.get("stage"), "role": R.get("role"),
                    "support_id": gate.get("support_id"),
                    "support_gate_status": gate.get("status"),
                    "divergences": R.get("divergences"),
                    "ebfmi_per_chain": diag.get("ebfmi_per_chain"),
                    "fixed_files": diag.get("fixed_files")},
            note="HBI ladder run")

        # fixed calibration objects actually consumed
        for key, val in (diag.get("fixed_files") or {}).items():
            if not val:
                continue
            nid = self._node_for_path(val, {"c": "completeness_object",
                                            "mg": "response_object",
                                            "extra": "fixed_extra_object"}
                                      .get(key, "fixed_object"))
            self.link(nid, rid, "consumed_by",
                      note="--%s-fixed-file" % ("c" if key == "c" else key))
        for key, role in (("pack", "support_pack"),
                          ("fp_census", "support_census"),
                          ("empirical_ops", "support_ops")):
            val = (gate.get("paths") or {}).get(key)
            if val:
                nid = self._node_for_path(val, role)
                self.link(nid, rid, "consumed_by")
        if "fp:loa0_template" in self.nodes and \
                str(R.get("ladder", "")).upper() != "ORACLE":
            self.link("fp:loa0_template", rid, "consumed_by",
                      note="FP template (non-ORACLE ladder)")

        # posterior draws
        fd = path[:-len(".json")] + "_fdraws.npz"
        bc = path[:-len(".json")] + "_bychain.npz"
        pid = self.add("posterior:%s" % tag, "posterior", fd,
                       code_commit=cfg.get("code_commit"),
                       config={"seed": cfg.get("seed"),
                               "chains": cfg.get("chains")},
                       note="posterior f-draws; the posterior hash of record")
        self.link(rid, pid, "reduced_to")
        if os.path.isfile(bc):
            bid = self.add("posterior_bychain:%s" % tag, "posterior_bychain",
                           bc)
            self.link(rid, bid, "reduced_to")

        # paper-number candidates
        for name, blob in (R.get("thresholds") or {}).items():
            if not isinstance(blob, dict):
                continue
            nid = "paper_number:%s:%s" % (tag, name)
            self.add(nid, "paper_number", None,
                     code_commit=cfg.get("code_commit"),
                     config={"threshold": name,
                             "estimand": blob.get("key", "dN/dX %s" % name),
                             "truth": blob.get("truth"),
                             "post_p16_50_84": blob.get("post_p16_50_84"),
                             "post_p2p5_97p5": blob.get("post_p2p5_97p5"),
                             "median_bias_pct": blob.get("median_bias_pct"),
                             "truth_in_68": blob.get("truth_in_68"),
                             "truth_in_95": blob.get("truth_in_95"),
                             "stage": R.get("stage"),
                             "support_id": gate.get("support_id")},
                     note="CANDIDATE paper number (mock closure); a real "
                          "headline requires PI adoption of the model of "
                          "record and the blind real C1")
            self.link(pid, nid, "reduced_to")
        for i, blob in enumerate(R.get("reporting_bins") or []):
            nid = "paper_number:%s:bin%d" % (tag, i)
            self.add(nid, "paper_number", None,
                     code_commit=cfg.get("code_commit"),
                     config={"estimand": "dN/dX reporting bin",
                             "bin": blob.get("bin"),
                             "median_bias_pct": blob.get("median_bias_pct"),
                             "truth_in_68": blob.get("truth_in_68"),
                             "truth_in_95": blob.get("truth_in_95"),
                             "stage": R.get("stage"),
                             "support_id": gate.get("support_id")},
                     note="CANDIDATE per-bin paper number (mock closure)")
            self.link(pid, nid, "reduced_to")

    def _node_for_path(self, path, role):
        """Reuse an existing node for ``path`` or register it on the fly."""
        for nid, node in self.nodes.items():
            if node["path"] and os.path.abspath(node["path"]) == \
                    os.path.abspath(path):
                return nid
        nid = "%s:%s" % (role, os.path.basename(path))
        return self.add(nid, role, path,
                        note="referenced by a run but not part of the "
                             "enumerated product set")

    def ingest_release_products(self, release_root):
        for sub, role in (("completeness", "release_product"),
                          ("response", "release_product")):
            d = os.path.join(release_root, sub)
            if not os.path.isdir(d):
                continue
            meta = os.path.join(d, "%s_model.json" % sub)
            if os.path.isfile(meta):
                nid = self.add("release:%s" % sub, role, meta, hash_it=False,
                               note="Zenodo %s product metadata" % sub)
                for src in list(self.nodes):
                    if sub == "completeness" and \
                            src.startswith("fit:completeness:"):
                        self.link(src, nid, "reduced_to")
                    if sub == "response" and src.startswith("fit:response:") \
                            and ":E:" not in src and ":B:" not in src:
                        continue
                    if sub == "response" and (":E:2lpt0" in src
                                              or ":B:2lpt0" in src):
                        self.link(src, nid, "reduced_to")

    # ---- environment ------------------------------------------------------
    def environment_lock(self):
        os.makedirs(self.out_dir, exist_ok=True)
        lock = os.path.join(self.out_dir, "conda_export_%s.txt" % CONDA_ENV)
        try:
            txt = subprocess.check_output(
                ["conda", "list", "--export", "-n", CONDA_ENV],
                stderr=subprocess.DEVNULL).decode()
            txt = "\n".join(l for l in txt.splitlines()
                            if not l.startswith("# created"))
            txt = txt.rstrip() + "\n"
        except Exception as exc:                             # pragma: no cover
            txt = "# conda list --export failed: %s\n" % exc
        prev = open(lock).read() if os.path.isfile(lock) else None
        if prev != txt:
            with open(lock, "w") as fh:
                fh.write(txt)
        env = {"conda_env": CONDA_ENV, "lock_file": lock,
               "lock_sha256": sha256_bytes(txt),
               "python": sys.version.split()[0], "numpy": np.__version__,
               "platform": sys.platform}
        self.add("env:conda_lock", "environment_lock", lock, hash_it=False,
                 note="conda list --export of %s" % CONDA_ENV)
        return env

    def git_stamp(self):
        def _run(args):
            try:
                return subprocess.check_output(
                    args, cwd=self.repo, stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:                                # pragma: no cover
                return None
        dirty = _run(["git", "status", "--porcelain"])
        return {"commit": _run(["git", "rev-parse", "HEAD"]),
                "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                "dirty": bool(dirty) if dirty is not None else None}

    # ---- assemble ---------------------------------------------------------
    def build(self, runs_dir=None, release_root=None):
        env = self.environment_lock()
        self.ingest_predeclarations()
        self.ingest_calibration()
        self.ingest_completeness()
        self.ingest_response()
        self.ingest_support("support_v3")
        self.ingest_fp()
        if release_root:
            self.ingest_release_products(release_root)
        self.ingest_runs(runs_dir or os.path.join(self.products, "final",
                                                  "runs"))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "schema": MANIFEST_SCHEMA,
            "generated_utc": _dt.datetime.now(_dt.timezone.utc)
                                .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "validation/release/provenance_manifest.py",
            "authority": "PI ruling 2026-09-13d §17 (provenance is a "
                         "Paper-lane deliverable) and §19 (Zenodo package)",
            "products": self.products,
            "strict": self.strict,
            "accept_stale": dict(self.accept_stale),
            "edge_kinds": list(EDGE_KINDS),
            "git": self.git_stamp(),
            "environment": env,
            "nodes": [self.nodes[n] for n in self.order],
            "edges": sorted(self.edges,
                            key=lambda e: (e["kind"], e["from"], e["to"])),
            "unresolved": self.unresolved,
            "summary": {},
            "content_digest": "",
        }
        roles = {}
        for n in manifest["nodes"]:
            roles[n["role"]] = roles.get(n["role"], 0) + 1
        manifest["summary"] = {
            "n_nodes": len(manifest["nodes"]),
            "n_edges": len(manifest["edges"]),
            "n_unresolved": len(manifest["unresolved"]),
            "nodes_by_role": dict(sorted(roles.items())),
            "n_paper_numbers": roles.get("paper_number", 0),
            "n_runs": roles.get("hbi_run", 0),
        }
        manifest["content_digest"] = content_digest(manifest)
        errors = validate(manifest)
        if errors:
            raise ManifestIntegrityError(
                "manifest fails its own schema:\n  " + "\n  ".join(errors))
        return manifest


def content_digest(manifest):
    """sha256 of the manifest with volatile fields removed (idempotence key)."""
    clone = {k: v for k, v in manifest.items()
             if k not in ("generated_utc", "content_digest")}
    return sha256_bytes(json.dumps(clone, sort_keys=True, default=str))


def write_manifest(manifest, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--release-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="fail closed on ANY missing referenced file")
    ap.add_argument("--accept-stale", action="append", default=[],
                    metavar="NAME=SHA256")
    ap.add_argument("--also-write", default=None,
                    help="second destination (e.g. the notes governance dir)")
    ap.add_argument("--max-bytes-also-write", type=int, default=2 * 1024 * 1024)
    a = ap.parse_args(argv)

    b = ManifestBuilder(a.products, os.path.dirname(os.path.abspath(a.out)),
                        strict=a.strict,
                        accept_stale=parse_accept_stale(a.accept_stale))
    man = b.build(runs_dir=a.runs_dir, release_root=a.release_root)
    write_manifest(man, a.out)
    size = os.path.getsize(a.out)
    print("manifest -> %s (%d bytes)" % (a.out, size))
    print("  nodes %(n_nodes)d  edges %(n_edges)d  runs %(n_runs)d  "
          "paper numbers %(n_paper_numbers)d  unresolved %(n_unresolved)d"
          % man["summary"])
    print("  content_digest %s" % man["content_digest"])
    if a.also_write:
        if size > a.max_bytes_also_write:
            ptr = a.also_write + ".POINTER.txt"
            with open(ptr, "w") as fh:
                fh.write("The provenance manifest is %d bytes (> %d) and is "
                         "kept on scratch:\n%s\nsha256 %s\ncontent_digest %s\n"
                         % (size, a.max_bytes_also_write, os.path.abspath(
                             a.out), sha256_file(a.out),
                            man["content_digest"]))
            print("  too large for the notes dir -> pointer %s" % ptr)
        else:
            write_manifest(man, a.also_write)
            print("  also ->", a.also_write)
    for u in man["unresolved"]:
        print("  UNRESOLVED: %s (%s)" % (u["path"], u["reason"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
