# CDDF analysis — intermediate-results store: organization plan

> **Status:** design plan (2026-06-26), part of PR #21. Authored by a 3-lens
> data-science team (research-software-engineering / provenance / file-format) +
> integration. **Not yet implemented** — this is the agreed target; implementation
> is a follow-up PR.
>
> **Problem it solves:** the analysis writes many ad-hoc intermediate result files
> (JSON/npz/tsv) "floating" on scratch — `…/cddf_o3_realdata/track_c/` alone is
> **29 dirs / 104 files**: hand-named near-duplicate variants (`tf_loa`,
> `tf_loa_zext`, `tf_loa_5bin`, `tf_loa_calc_cddf`, `tf_loa_calc_cddf_snr2_nobal`,
> `…_regen`, `…_smoke`, `…_verify`), backup-by-rename (`*.preRECENTER.bak`,
> `*.9bin.bak.json`), producer scripts interleaved with outputs, and **no
> per-folder README, no commit stamp, no machine-readable index**. A notebook can
> only find a result by hand-typing a fragile scratch path (exactly what NB5
> originally did, and what the no-scratch-precomputed-table rule now forbids).
>
> **The good counter-example to generalize:** `CDDF_analysis/hbi/tutorial_data/`
> (a README provenance table + a regen script). This plan turns that hand-maintained
> pattern into an *addressable, self-documenting, version-stamped* store.

---

## 1. Layout — a keyed, write-once tree (kills the variant-dir zoo)

One configurable root, a 4-axis tree, one **immutable leaf per run**:

```
$CDDF_STORE/                                  # single env var; relocatable scratch↔turbo↔NERSC
├── MANIFEST.sqlite                           # the index (§3) — concurrent-write safe
├── MANIFEST.json                             # human-diffable mirror, regenerated from the db
├── mock/                                     # PRIVACY split (§4): committable/shareable
│   └── 2lpt0/ … 
└── real_loa/                                 # scratch-only; .gitignore '*' + DO_NOT_COMMIT
    ├── kernel/      remp__k-broaden012__a1b2c3d4/   # {slug}__{knobs}__{hash8}/
    │                ├── result.h5 · run.json · README.md · run.log
    ├── fit/         phase3d__pm_canonical__c0ffee12/
    ├── band/        perz__zdla_5bin__9c2200af/
    └── measurement/
        ├── tf_loa__snr2_nobal_z2-3.5__7f3e0091/     #  ← was tf_loa_calc_cddf_snr2_nobal
        │   ├── result.json · run.json · README.md · fig_*.png · run.log
        └── tf_loa__snr0_bal_z2-4.25__b1904c66/      #  ← was tf_loa_zext
```

**Rules that eliminate the zoo:**
- **Leaves are immutable / write-once.** Any config change → a new hash → a new leaf. No `_regen`/`_rerun`/`_finalize`/`.bak`. Supersession is recorded in metadata (`supersedes`/`superseded_by`), never by renaming.
- **Stable filenames *inside* a leaf** (`result.json`, `result.h5`, `run.json`, `README.md`, `fig_*.png`, `run.log`). All variability lives in the *leaf id*; a notebook always opens `result.json`, the resolver picks *which* leaf.
- **Producer scripts live in the repo, never in the store.** The store holds only results + provenance + logs.
- **`$CDDF_STORE` is the only path literal.** Relocating the store = one env var; no code edits.

**Producer change (mechanical):** drop the free-form `--out`; take `--store $CDDF_STORE --dataset … --stage …` + the existing config flags, and **compute the leaf path** from `{dataset}/{stage}/{slug}__{knobs}__{hash}/`. This is the one pivot that makes runs/notebooks resolve results by id, not by a typed path.

### Naming + config-hashing
The distinguishing config is already explicit in argparse — it just never reaches the name. Build a canonical config dict (only the knobs that affect results: `filter, snr_min, no_bal, p_dla_min, zbins, fit_floor/ceil, fp_estimator, kernel-id, resp_family`), then:
- **slug** = readable, from the salient knobs that differ from the producer default → `tf_loa__snr2_nobal_z2-3.5`.
- **hash8** = `sha1(json.dumps(config, sort_keys=True))[:8]` → `7f3e0091`. Idempotent (identical config re-resolves to the same leaf); collision-proof.

So `tf_loa`, `tf_loa_snr2_nobal`, `tf_loa_zext`, `tf_loa_5bin` become **four queryable leaves of one `producer=track_c_tf_loa` schema**, distinguished by `{snr_min, no_bal, zbins}` — not mystery dirs.

---

## 2. Provenance contract — "each folder has a README with commit stamp" (auto, not aspirational)

Every result leaf carries a `README.md` (human) **and** a `provenance.json` (machine), emitted together by one `write_provenance()` call so they cannot drift.

**Per-folder README required fields:** what-this-is (1 line) · status (`CANONICAL`/`superseded-by:…`/`scratch`/`smoke`) · **privacy** (`mock`/`real-LOA` + shareable y/n) · producing script (repo path) · **code commit (short+long + dirty flag)** · date (UTC) · inputs (each with its own id/commit/privacy — provenance is *transitive*) · exact CLI invocation · outputs (`file | what it is` table) · one-line regenerate command.

**Commit-stamping (generalize `track_c_tf_loa.py::_git_commit`):** a shared `CDDF_analysis/hbi/provenance.py::git_stamp()` returns `{commit_short, commit_long, branch, dirty, diff_sha256}`.
- Clean tree → trustworthy, fully reproducible from that SHA.
- **Dirty tree → loud, not fatal:** record `dirty:true` + `diff_sha256` (fingerprint of `git diff HEAD`), render **⚠ DIRTY** in the README, stderr-warn, optionally stash `uncommitted.diff`. Research moves fast; don't block, but never print a misleading clean SHA.
- `commit:"unknown"` is a hard smell → auto-tagged `scratch`, never promoted to CANONICAL.
- Backfill: `--code-commit <sha>` (mirrors `package_catalog.sh`) marks `source: backfilled`.

**`provenance.json`** is the superset of the `metadata` block already inside `track_c_tf_loa.json`, standardized so tools don't need per-result bespoke schemas. It answers: *was this built from the current code?* (commit vs tip), *are the inputs still the ones used?* (input sha256 vs recomputed), and walks the transitive DAG via `inputs[].produced_by`.

---

## 3. Manifest / index — `store.get(...)` instead of literal scratch paths

`MANIFEST.sqlite` at the store root (one `results` table + `tags`), mirrored to a human-diffable `MANIFEST.json`, and **rebuildable** by scanning every leaf's `run.json` (the index is never the sole copy of provenance). SQLite chosen over one big JSON for concurrent SLURM writes (the queued regen jobs each `INSERT` without clobbering) + indexed queries.

`results` columns: `id` (= relative leaf path, the stable handle) · `dataset` · `stage` · `producer` (schema id) · `config_hash` · `config_json` · `selection` (salient slug) · `commit` · `inputs` (upstream leaf ids + external paths = a DAG) · `outputs` · `date` · `status` (`current`/`superseded`/`draft`) · `supersedes` · `used_by` (back-ref to TeX figures).

### Resolver API — the thin layer notebooks/scripts import
`CDDF_analysis/results_store.py` reads the manifest and returns **paths** (downstream `np.load`/`json.load`/the existing dataclasses are unchanged):
```python
from CDDF_analysis.results_store import ResultStore
store = ResultStore()                       # roots at $CDDF_STORE; no path literal anywhere
r = store.get(dataset="real-loa", stage="measurement", selection="snr2_nobal")
#   strict: 0 or >1 match → raises, listing candidates (no silent wrong-twin binding)
data = json.load(open(r.path("result.json")));  print(r.commit, r.config["snr_min"], r.inputs)
r = store.by_id("real-loa/measurement/tf_loa__snr2_nobal_z2-3.5__7f3e0091")   # pin a TeX figure
store.list(producer="track_c_tf_loa", status="current")   # the 10 tf_loa_* dirs, as rows
leaf = store.new(dataset=…, stage=…, producer=…, config=config, inputs=[kernel_id, molly])  # producer side
```
**No notebook hardcodes a `/scratch/...` path** — it asks by `(dataset, stage, selection)`; relocating the store = one env var; `get()` is strict so a notebook can't bind the wrong twin.

---

## 4. Privacy enforced by the layout (not by vigilance)

The project rule — *real-LOA results never committed; only mock shareable* — becomes structural:
- Every leaf is exactly `mock` or `real-LOA` (`privacy.class`), **auto-derived** (a result is `real-LOA` iff any input in its chain is real-LOA — contagious downward; can't launder). `track_c_tf_loa.py` already knows this at runtime (it loads `loa_cat`).
- Top-level `mock/` (committable) vs `real_loa/` (scratch-only, `.gitignore '*'` + `DO_NOT_COMMIT` sentinel). Shareable `tutorial_data/`-style fixtures may only be sourced from `mock/`.
- **Pre-commit guard** (CI-enforceable): reject any staged path whose nearest `provenance.json` is `real-LOA`/`shareable:false`. The mechanical version of today's manual banner.
- Regen scripts assert `privacy==mock` on each source before copying (generalizes `regen_tutorial_fixtures.sh`).

---

## 5. File formats — JSON+schema for results, HDF5 for big intermediates

**Per-z CDDF results (dN/dX, Ω, f(N|z), band, config): versioned JSON + a JSON-Schema + a frozen dataclass.** Payloads are KB-scale → JSON wins on self-describing, language-agnostic, **git-diffable**, provenance-bearing; minimal migration (both producers already emit JSON). HDF5's chunking/compression don't pay off below a few MB and binary kills PR-reviewability of science numbers.

**ONE canonical result schema** (`schema_version: "cddf-result/1.0.0"`) unifies the two producers (the HBI `track_c_tf_loa.json` and the raw `loa_literal_calccddf_*.json` carry the *same observable* in incompatible ad-hoc schemas today). Key design: a reusable `BAND := {point, q16, q84, q025, q975, std|null}` everywhere a CI appears; `band_method` makes the convention explicit (`mc_quantile` vs `bootstrap_ci`); a first-class `zbins` block (`edges/centers/extrapolated/thin/truth_counts/dX`) so `result["dndx"]["perz"][i]` reads identically regardless of producer and z is no longer positional-only. Adapters `from_hbi_json()` / `from_calc_cddf_json()` convert the existing scratch files without re-running the jobs.

**Kernel / forward-response / band npz → HDF5.** The `kappa` array is `(537704,52,15) f32 ≈ 1.6 GB` — exactly where HDF5 earns its keep (gzip on f32 ≈ 3–5×, chunked one-z-slab partial reads, typed attrs for the scalars npz dumps as 0-d arrays, the *variant moved into an attr* `z_covariate="zdla"` instead of into the filename `_zdla`). One file per logical object, one group per concept, the same provenance block as root attrs.

**Shared IO + validation:** `CDDF_analysis/cddf_result_io.py` — `write_result/read_result` (jsonschema validate + frozen-dataclass invariants, fail loudly via `CDDFSchemaError`), `from_hbi_json/from_calc_cddf_json` adapters, `write_kernel/read_kernel` (HDF5, lazy/mmap, optional z-slab). Schemas at `CDDF_analysis/schemas/`; small round-trip fixtures + `tests/test_cddf_result_io.py` (round-trip, schema-reject, both-producer equivalence).

---

## 6. Migration — fold the zoo in without breaking the in-flight TeX draft

The draft + `tutorial_data/regen_tutorial_fixtures.sh` read literal scratch paths today, so migration is **non-destructive, copy-only until the end**:
- **Phase 0 — freeze + inventory:** snapshot the 104 files; write `LEGACY_MAP.tsv` (`old_path → proposed_id`).
- **Phase 1 — ingest by copy + symlink-back:** `migrate_legacy.py` infers each result's config from its dirname suffix + producer defaults (`_snr2_nobal`→snr/bal; `_zext`/`5bin`/`9bin`→zbins; `.bak`/`.preRECENTER`→`status:superseded`+`supersedes`), computes the §1 hash, `cp`s the canonical payload into the new leaf, `INSERT`s a record, and drops a **back-symlink** at the old path so legacy hardcoded paths keep resolving (the in-flight TeX/tutorial builds read through unchanged). Unknown config → `status:draft, config_inferred:true`.
- **Phase 2 — repoint at leisure:** update the regen script + draft figure scripts to `store.get(...)`/`store.by_id(...)` one figure at a time, byte-diffing each re-rendered PNG before removing that symlink. The queued regen jobs (`52266000/52266001`) should be the **first producers updated to write into the store natively** (cheapest to fix at source).
- **Phase 3 — retire:** once every draft figure resolves through the resolver and re-renders identically, delete legacy dirs + back-symlinks; `LEGACY_MAP.tsv` stays as the audit trail. Superseded results move to a `.gitignore`d `_attic/` (never `rm`), preserving their README + provenance.

**Deliverables of the implementation PR:** `CDDF_analysis/results_store.py` (resolver) · `CDDF_analysis/hbi/provenance.py` (`git_stamp`, `write_provenance`) · `CDDF_analysis/cddf_result_io.py` + `CDDF_analysis/schemas/` (canonical schema, IO, validation) · `tools/provenance/scan.py` (status table `fresh/stale-code/stale-input/dirty/unknown/superseded`, `--gc`→`_attic/`, `--code-commit` backfill) · `tools/migrate_legacy.py` · a pre-commit privacy guard. Committable artifacts only: the modules + regen/migrate scripts — never the GB-scale payloads.

---

## 7. Interim rule (in force now, before implementation)

Until the store exists: **tutorial/analysis notebooks must not read precomputed reduction tables floating on scratch.** Allowed inputs are (a) mock data, (b) GP-inference results (absorber catalogs), (c) committed repo fixtures (`tutorial_data/`). NB5 was reworked accordingly — it now *runs the reduction live* from a catalog + the committed frozen-calibration fixtures, rather than loading a scratch `track_c_tf_loa.json`. NB0–NB4 already comply (committed fixtures only).
