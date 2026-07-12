"""Addressable, keyed, write-once intermediate-results store + sqlite manifest
(implements ``CDDF_analysis/RESULTS_STORE_PLAN.md`` §1/§3/§4).

A notebook/script asks the store for a result by ``(dataset, stage, selection)``
instead of typing a fragile scratch path; the store resolves *which* immutable
leaf to open. Each leaf is a write-once directory
``$CDDF_STORE/{privacy}/{dataset}/{stage}/{slug}__{config_hash}/`` holding stable
filenames (``result.json``, ``result.h5``, ``provenance.json``, ``README.md``,
…). Any config change → a new hash → a new leaf; supersession is metadata, never
a rename.

The index is ``MANIFEST.sqlite`` at the store root (concurrent-write-safe for the
queued SLURM regen jobs), mirrored to a human-diffable ``MANIFEST.json`` and
**rebuildable** by scanning every leaf's ``provenance.json`` — the manifest is
never the sole copy of provenance.

Stdlib only (``sqlite3``, ``json``, ``os``, ``pathlib``); the provenance/slug/
privacy logic is delegated to ``CDDF_analysis.hbi.provenance``.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from CDDF_analysis.hbi.provenance import (
    config_hash,
    make_slug,
    write_provenance,
    privacy_class,
    git_stamp,
    _atomic_write,
)

__all__ = ["ResultStore", "ResultLeaf"]

_MANIFEST_SQLITE = "MANIFEST.sqlite"
_MANIFEST_JSON = "MANIFEST.json"

# the privacy-class -> top-level subdir map (§4).
_PRIVACY_SUBDIR = {"mock": "mock", "real-LOA": "real_loa"}

# Documents the manifest `results` table column order. Kept in sync with the
# CREATE TABLE in `_ensure_schema` / the row dict in `_row_from_prov` — the live
# column is `commit_stamp` (a serialized git_stamp record), NOT a bare `commit`.
_RESULTS_COLUMNS = [
    "id", "dataset", "stage", "producer", "config_hash", "config_json",
    "selection", "commit_stamp", "inputs_json", "outputs_json", "date", "status",
    "supersedes", "used_by",
]


# --------------------------------------------------------------------------- #
# ResultLeaf — the thin handle a notebook/producer holds                       #
# --------------------------------------------------------------------------- #
class ResultLeaf:
    """A single immutable result directory + its loaded provenance.

    Attributes
    ----------
    id : str            relative leaf path under the store root (the stable handle)
    dir : str           absolute leaf directory
    config : dict       the result-affecting config
    commit : dict       the code-commit stamp (git_stamp record)
    inputs : list       upstream leaf ids + external input descriptors
    status : str        current / superseded / draft
    privacy : dict      {"class": ..., "shareable": bool}
    provenance : dict   the full provenance.json record (None until committed)
    """

    def __init__(self, *, leaf_id: str, leaf_dir: str, provenance: dict | None = None):
        self.id = leaf_id
        self.dir = leaf_dir
        self.provenance = provenance

    # convenience path resolver ------------------------------------------------
    def path(self, fname: str) -> str:
        """Absolute path to a file inside the leaf (e.g. ``result.json``)."""
        return os.path.join(self.dir, fname)

    # provenance-backed accessors ---------------------------------------------
    @property
    def config(self) -> dict:
        return (self.provenance or {}).get("config", {})

    @property
    def commit(self):
        return (self.provenance or {}).get("code_commit")

    @property
    def inputs(self) -> list:
        return (self.provenance or {}).get("inputs", [])

    @property
    def status(self) -> str:
        return (self.provenance or {}).get("status", "draft")

    @property
    def privacy(self) -> dict:
        return (self.provenance or {}).get("privacy", {})

    def dataset_stage(self) -> tuple:
        p = self.provenance or {}
        return (p.get("dataset"), p.get("stage"))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ResultLeaf(id={self.id!r}, status={self.status!r})"


# --------------------------------------------------------------------------- #
# ResultStore                                                                  #
# --------------------------------------------------------------------------- #
class ResultStore:
    """The keyed store + manifest. Roots at ``$CDDF_STORE`` (or an explicit
    ``root=``). All path literals live here; relocating the store is one env var.
    """

    def __init__(self, root: str | None = None):
        root = root or os.environ.get("CDDF_STORE")
        if not root:
            raise ValueError(
                "ResultStore needs a root: pass root=... or set the CDDF_STORE "
                "environment variable to the store directory."
            )
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._db_path = self.root / _MANIFEST_SQLITE
        self._ensure_schema()

    # ----- sqlite schema -----------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id           TEXT PRIMARY KEY,
                    dataset      TEXT,
                    stage        TEXT,
                    producer     TEXT,
                    config_hash  TEXT,
                    config_json  TEXT,
                    selection    TEXT,
                    commit_stamp TEXT,
                    inputs_json  TEXT,
                    outputs_json TEXT,
                    date         TEXT,
                    status       TEXT,
                    supersedes   TEXT,
                    used_by      TEXT
                )
                """
            )
            conn.commit()

    # ----- leaf addressing ---------------------------------------------------
    def _privacy_subdir(self, privacy: str | None) -> str:
        """Map a privacy class to the top-level subdir. ``privacy`` may be a
        bare class string ('mock'/'real-LOA') or None (default mock)."""
        cls = privacy or "mock"
        return _PRIVACY_SUBDIR.get(cls, "mock")

    def leaf_path(
        self,
        dataset: str,
        stage: str,
        producer: str,
        config: dict,
        producer_defaults: dict | None = None,
        privacy: str | None = None,
    ) -> str:
        """Compute the absolute leaf directory for a (dataset, stage, producer,
        config) tuple: ``{privacy_subdir}/{dataset}/{stage}/{slug}__{hash8}/``.

        Deterministic: identical config → identical path (idempotent resolve).
        ``privacy`` is the *dataset* privacy class the caller declares; the final
        per-result privacy (after input contagion) is computed at commit time, but
        the subtree is fixed at allocation so real-LOA leaves never sit under
        ``mock/``.
        """
        defaults = producer_defaults or {}
        slug = make_slug(config, defaults)
        h = config_hash(config)
        sub = self._privacy_subdir(privacy)
        return str(self.root / sub / dataset / stage / f"{slug}__{h}")

    def _id_for_dir(self, leaf_dir: str) -> str:
        """Relative-to-root leaf id (the stable handle, used as the manifest PK)."""
        return os.path.relpath(leaf_dir, self.root)

    # ----- producer side: new + commit --------------------------------------
    def new(
        self,
        *,
        dataset: str,
        stage: str,
        producer: str,
        config: dict,
        inputs: list,
        privacy: str | None = None,
        producer_defaults: dict | None = None,
    ) -> ResultLeaf:
        """Allocate (mkdir) a fresh leaf and return its handle. The caller then
        writes ``result.*`` files into ``leaf.dir`` and calls ``commit_leaf``.

        ``privacy`` is the declared dataset privacy ('mock'/'real-LOA'); if not
        given it is derived from ``inputs`` (contagion) so an unflagged downstream
        of a real-LOA input still lands under ``real_loa/``.
        """
        # Determine the privacy subtree. Prefer an explicit declaration; else
        # derive from the inputs' privacy (contagious).
        if privacy is None:
            input_provs = self._resolve_input_provs(inputs)
            privacy = privacy_class(input_provs)["class"]

        leaf_dir = self.leaf_path(dataset, stage, producer, config,
                                  producer_defaults=producer_defaults,
                                  privacy=privacy)
        os.makedirs(leaf_dir, exist_ok=True)
        leaf = ResultLeaf(leaf_id=self._id_for_dir(leaf_dir), leaf_dir=leaf_dir)
        # stash allocation-time context the commit needs (not yet provenance).
        leaf._pending = {
            "dataset": dataset,
            "stage": stage,
            "producer": producer,
            "config": config,
            "inputs": inputs,
            "privacy_decl": privacy,
            "producer_defaults": producer_defaults or {},
        }
        return leaf

    def _resolve_input_provs(self, inputs: list) -> list:
        """Turn the mixed ``inputs`` list (leaf-id strings + external dicts) into a
        list of provenance-like dicts carrying ``privacy`` for contagion."""
        provs = []
        for inp in inputs or []:
            if isinstance(inp, str):
                # an upstream leaf id — load its provenance for privacy.
                try:
                    up = self.by_id(inp)
                    provs.append({"id": inp, "privacy": up.privacy})
                except Exception:
                    # unknown id (e.g. not yet committed): treat as mock-unknown.
                    provs.append({"id": inp, "privacy": {"class": "mock"}})
            elif isinstance(inp, dict):
                provs.append(inp)
            else:  # pragma: no cover - defensive
                provs.append({"id": str(inp), "privacy": {"class": "mock"}})
        return provs

    def commit_leaf(
        self,
        leaf: ResultLeaf,
        *,
        what: str,
        cli: str,
        outputs: list,
        regen_cmd: str,
        status: str = "current",
    ) -> None:
        """Stamp provenance into the leaf (README.md + provenance.json) and
        INSERT/REPLACE the manifest row + regenerate ``MANIFEST.json``."""
        pend = getattr(leaf, "_pending", None)
        if pend is None:
            raise ValueError(
                "commit_leaf requires a leaf from store.new(...); got one without "
                "allocation context."
            )

        input_provs = self._resolve_input_provs(pend["inputs"])
        privacy = privacy_class(input_provs)
        # Honor an explicit real-LOA declaration even if inputs look mock (a
        # producer that loads real spectra directly knows it is real-LOA).
        if pend["privacy_decl"] == "real-LOA":
            privacy = {"class": "real-LOA", "shareable": False}

        rec = write_provenance(
            leaf.dir,
            what=what,
            status=status,
            privacy=privacy,
            producer=pend["producer"],
            config=pend["config"],
            inputs=input_provs,
            cli=cli,
            outputs=outputs,
            regen_cmd=regen_cmd,
        )
        leaf.provenance = rec
        self._insert_row(rec)
        # Single-current invariant: a freshly-committed `current` leaf supersedes
        # any OTHER currently-`current` leaf at the same (dataset, store_stage). We
        # touch ONLY the manifest status (+ the supersedes/superseded_by columns when
        # present) — never the superseded leaf's write-once provenance.json payload.
        if status == "current":
            self._supersede_prior_current(
                dataset=pend["dataset"], stage=pend["stage"],
                new_id=self._leaf_id_from_prov(rec))
        self._write_json_mirror()

    def _supersede_prior_current(self, *, dataset: str, stage: str, new_id: str) -> None:
        """Mark every OTHER `current` leaf at the same (dataset, store_stage) as
        `superseded` (manifest only). The newest commit becomes the single current.

        Records the supersession link in the manifest's `supersedes`/`superseded_by`
        columns when they exist (back-compat: a schema without them just gets the
        status flip). The superseded leaf's on-disk provenance.json is intentionally
        left untouched — supersession is manifest metadata, not a payload rewrite.
        """
        with self._connect() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(results)")}
            prior = [r["id"] for r in conn.execute(
                "SELECT id FROM results WHERE dataset = ? AND stage = ? "
                "AND status = 'current' AND id != ?",
                [dataset, stage, new_id])]
            if not prior:
                return
            for old_id in prior:
                conn.execute(
                    "UPDATE results SET status = 'superseded' WHERE id = ?", [old_id])
                if "superseded_by" in cols:
                    conn.execute(
                        "UPDATE results SET superseded_by = ? WHERE id = ?",
                        [new_id, old_id])
            if "supersedes" in cols:
                # the new leaf supersedes the (newest) prior current leaf.
                conn.execute(
                    "UPDATE results SET supersedes = ? WHERE id = ?",
                    [prior[-1], new_id])
            conn.commit()

    # ----- manifest row IO ---------------------------------------------------
    def _row_from_prov(self, rec: dict) -> dict:
        """Project a provenance record onto the manifest row columns."""
        # id in the manifest is the leaf id (relative path); provenance.json's
        # `id` field is the leaf basename, so recompute from dataset/stage/basename.
        leaf_id = self._leaf_id_from_prov(rec)
        return {
            "id": leaf_id,
            "dataset": rec.get("dataset"),
            "stage": rec.get("stage"),
            "producer": rec.get("producer"),
            "config_hash": rec.get("config_hash"),
            "config_json": json.dumps(rec.get("config", {}), sort_keys=True),
            "selection": rec.get("slug"),
            "commit_stamp": json.dumps(rec.get("code_commit")),
            "inputs_json": json.dumps(rec.get("inputs", [])),
            "outputs_json": json.dumps(rec.get("outputs", [])),
            "date": rec.get("date_utc"),
            "status": rec.get("status"),
            "supersedes": rec.get("supersedes"),
            "used_by": json.dumps(rec.get("used_by", [])),
        }

    def _leaf_id_from_prov(self, rec: dict) -> str:
        """Reconstruct the relative leaf id from a provenance record. The record
        stores the basename as ``id`` and the dataset/stage; the privacy subtree
        is derived from ``privacy.class``."""
        sub = self._privacy_subdir((rec.get("privacy") or {}).get("class"))
        return os.path.join(sub, rec["dataset"], rec["stage"], rec["id"])

    def _insert_row(self, rec: dict) -> None:
        row = self._row_from_prov(rec)
        cols = list(row.keys())
        placeholders = ",".join("?" for _ in cols)
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO results ({','.join(cols)}) "
                f"VALUES ({placeholders})",
                [row[c] for c in cols],
            )
            conn.commit()

    def _write_json_mirror(self) -> None:
        """Regenerate the human-diffable MANIFEST.json from the sqlite table.

        Written via the same atomic tmp+os.replace helper the provenance writer
        uses, so a concurrent reader never sees a half-written mirror and two
        regen processes can't corrupt each other's mirror file."""
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM results ORDER BY id")]
        _atomic_write(self.root / _MANIFEST_JSON, json.dumps(rows, indent=2))

    # ----- consumer side: get / by_id / list ---------------------------------
    def _leaf_from_row(self, row) -> ResultLeaf:
        leaf_id = row["id"]
        leaf_dir = str(self.root / leaf_id)
        prov = None
        prov_path = os.path.join(leaf_dir, "provenance.json")
        if os.path.exists(prov_path):
            prov = json.loads(Path(prov_path).read_text())
            # The MANIFEST is authoritative for the LIVE status: supersession flips a
            # leaf's status in the manifest without rewriting its write-once
            # provenance.json, so overlay the manifest's status (+ supersedes link)
            # onto the loaded record. ResultLeaf.status then reflects supersession.
            try:
                row_status = row["status"]
            except (KeyError, IndexError):
                row_status = None
            if row_status is not None:
                prov = {**prov, "status": row_status}
                try:
                    if row["supersedes"] is not None:
                        prov["supersedes"] = row["supersedes"]
                except (KeyError, IndexError):
                    pass
        return ResultLeaf(leaf_id=leaf_id, leaf_dir=leaf_dir, provenance=prov)

    def get(self, *, dataset: str, stage: str, selection: str | None = None) -> ResultLeaf:
        """Resolve the unique leaf matching (dataset, stage[, selection]).

        STRICT: 0 or >1 matches raise ``LookupError`` listing the candidate ids
        (no silent wrong-twin binding). ``selection`` matches against the leaf's
        slug (``results.selection``).
        """
        query = "SELECT * FROM results WHERE dataset = ? AND stage = ?"
        params: list = [dataset, stage]
        if selection is not None:
            query += " AND selection = ?"
            params.append(selection)
        with self._connect() as conn:
            rows = list(conn.execute(query, params))

        # Prefer `current` leaves; a re-committed leaf supersedes its predecessor, so
        # the single current leaf is THE match. Only when zero current rows match do we
        # consider superseded ones (back-compat: a store that has only a superseded
        # leaf — e.g. the old config — still resolves if there is exactly one).
        current = [r for r in rows if r["status"] == "current"]
        superseded = [r for r in rows if r["status"] == "superseded"]
        if len(current) == 1:
            return self._leaf_from_row(current[0])
        if not current and len(superseded) == 1:
            return self._leaf_from_row(superseded[0])
        # otherwise (no current + 0/≥2 superseded, or ≥2 current) fall through to the
        # strict 0/>1 reporting below, scoped to the rows that would have matched.
        rows = current if current else rows

        if len(rows) == 1:
            return self._leaf_from_row(rows[0])

        sel_txt = f", selection={selection!r}" if selection is not None else ""
        if not rows:
            # show what *does* exist for this (dataset, stage) to aid the caller.
            with self._connect() as conn:
                near = [r["id"] for r in conn.execute(
                    "SELECT id FROM results WHERE dataset = ? AND stage = ?",
                    [dataset, stage])]
            near_txt = ("; existing leaves at this (dataset, stage): "
                        + ", ".join(near)) if near else ""
            raise LookupError(
                f"No result matched dataset={dataset!r}, stage={stage!r}{sel_txt}. "
                f"No candidates{near_txt}."
            )
        # >1 match: list every candidate id + its slug so the caller can pin one.
        cands = ", ".join(f"{r['id']} (selection={r['selection']!r})" for r in rows)
        raise LookupError(
            f"Ambiguous: {len(rows)} results matched dataset={dataset!r}, "
            f"stage={stage!r}{sel_txt}. Candidates: {cands}. "
            f"Pass selection=... to disambiguate, or pin one with by_id(...)."
        )

    def by_id(self, leaf_id: str) -> ResultLeaf:
        """Resolve a leaf by its exact id (the stable handle a TeX figure pins)."""
        with self._connect() as conn:
            rows = list(conn.execute(
                "SELECT * FROM results WHERE id = ?", [leaf_id]))
        if not rows:
            raise LookupError(f"No result with id={leaf_id!r}.")
        return self._leaf_from_row(rows[0])

    def list(self, *, producer: str | None = None, status: str | None = None,
             dataset: str | None = None) -> list:
        """All leaves, optionally filtered by producer / status / dataset."""
        clauses, params = [], []
        if producer is not None:
            clauses.append("producer = ?")
            params.append(producer)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if dataset is not None:
            clauses.append("dataset = ?")
            params.append(dataset)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = list(conn.execute(
                f"SELECT * FROM results{where} ORDER BY id", params))
        return [self._leaf_from_row(r) for r in rows]

    # ----- manifest rebuild from leaves --------------------------------------
    def rebuild_manifest(self) -> None:
        """Scan every leaf's ``provenance.json`` under the store root and rebuild
        the sqlite table + json mirror from scratch.

        This is the guarantee that the manifest is never the sole copy of
        provenance: a deleted/corrupt MANIFEST.sqlite is fully reconstructable
        from the immutable leaves alone.
        """
        # fresh table.
        if self._db_path.exists():
            self._db_path.unlink()
        self._ensure_schema()

        recs = []
        for prov_path in self.root.rglob("provenance.json"):
            try:
                rec = json.loads(prov_path.read_text())
            except Exception:
                continue
            if rec.get("schema_version", "").startswith("cddf-provenance/"):
                self._insert_row(rec)
                recs.append(rec)
        # Re-derive the single-current invariant from the leaves alone: supersession
        # is manifest-only metadata (the write-once provenance.json is never edited),
        # so a rebuild reconstructs it. Per (dataset, stage), if >1 leaf reports
        # status='current', keep the NEWEST (by date_utc) current and mark the rest
        # superseded — exactly what commit_leaf would have done in commit order.
        self._reconcile_current_status(recs)
        self._write_json_mirror()

    def _reconcile_current_status(self, recs: list) -> None:
        """Enforce one `current` leaf per (dataset, stage) in the manifest, choosing
        the newest by ``date_utc`` (commit order). Manifest-only; leaves untouched."""
        groups: dict = {}
        for rec in recs:
            if rec.get("status") != "current":
                continue
            key = (rec.get("dataset"), rec.get("stage"))
            groups.setdefault(key, []).append(rec)
        for key, group in groups.items():
            if len(group) < 2:
                continue
            # newest by date_utc is the survivor; ties broken by leaf id for stability.
            group.sort(key=lambda r: (r.get("date_utc") or "",
                                      self._leaf_id_from_prov(r)))
            survivor = group[-1]
            survivor_id = self._leaf_id_from_prov(survivor)
            with self._connect() as conn:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(results)")}
                for rec in group[:-1]:
                    old_id = self._leaf_id_from_prov(rec)
                    conn.execute(
                        "UPDATE results SET status = 'superseded' WHERE id = ?",
                        [old_id])
                    if "superseded_by" in cols:
                        conn.execute(
                            "UPDATE results SET superseded_by = ? WHERE id = ?",
                            [survivor_id, old_id])
                if "supersedes" in cols:
                    prior_id = self._leaf_id_from_prov(group[-2])
                    conn.execute(
                        "UPDATE results SET supersedes = ? WHERE id = ?",
                        [prior_id, survivor_id])
                conn.commit()
