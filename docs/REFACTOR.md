# Repo refactor — 2026-07 (tag `v0.1.0`)

Packaging + structure cleanup so the repo installs and imports like a normal Python
package and a first-time user can go spectra → CDDF without insider knowledge. Done as
four small, separately-reviewed PRs.

**Pipeline behaviour change: none.** The Bayesian inference path
(`gpy_dla_detection/`, `run_bayes_select.py`) and the catalog-HBI CDDF estimator
(`CDDF_analysis/hbi/cddf_catalog_hbi.py`) are byte-identical before and after. Nothing was
renamed; every existing `import` string still resolves. The refactor only adds
installability, consolidates the SLURM dirs, and documents the entry path — it does not
touch what the code computes. The paper headline re-derives bit-for-bit across the tag (see
`CDDF_analysis/hbi/REPRODUCE_HEADLINE.md`).

## The four PRs

| PR | merge | what |
|----|-------|------|
| #24 | `2d4f343` | **Hygiene.** `git rm --cached` the Claude working artifacts (`CLAUDE.md`, `.claude/`) → gitignored, kept on disk. Privacy scrub of real-LOA systematics percentages out of tracked docstrings (→ private-notes pointers; mock values kept). Stale logs / `-Copy1` notebook removed. |
| #25 | `9112ded` | **Packaging.** Add `pyproject.toml`. `pip install -e .` makes `gpy_dla_detection`, `CDDF_analysis.*`, and the root modules (`run_bayes_select`, `constants`, `dlasearch`, `fitwarning`) import from **any** working directory — retires the run-from-repo-root / `sys.path.insert` requirement. Flat in-place layout: import names declared verbatim, no `src/` umbrella, no package renamed. |
| #26 | `347a7cb` | **SLURM consolidation.** `git mv slurm_{cddf,preload,train,vi}/` → `slurm/{cddf,preload,train,vi}/`. Pure path-prefix swap: 20 file renames + reference-string updates in 3 docs / 3 comment-only `.py` lines. No logic, no import impact. |
| #27 | `96943a6` | **Quickstart.** A four-step "run the finder → get the CDDF" block at the top of the README (install → `desi-DLAGP.py` → `combine_processed_h5.py` → `desi_cddf.py`), with every flag verified against the actual argparse. Distinct from — and pointing at — the calibrated paper pipeline in `REPRODUCE_HEADLINE.md`. |

Base for the whole series: `e507d38` (the PR #23 merge, immediately before this work).

## Hard constraints honoured (what did NOT move)

- **Frozen inference** — `gpy_dla_detection/*` (incl. `dla_gp.py`, the C-Voigt path) untouched. See the standing rule in the project notes: the NERSC-proven inference path stays byte-identical.
- **Frozen estimator** — `CDDF_analysis/hbi/cddf_catalog_hbi.py` untouched; the headline is a config-only variant of an archival SLURM job, still reproducible to `0.00e+00`.
- **Every import string preserved** — no module renamed or relocated. `import`s that worked from the repo root before still work; after `pip install -e .` they also work from anywhere.
- **Back-compat shims kept** — the seven top-level `CDDF_analysis/*.py` modules (`cddf_catalog_hbi.py` at root, etc.) have zero importers today but are retained as shims so older scripts / notebooks / handoffs don't break. `voigt.py` and `set_parameters.py` are **not** orphans (legacy Pathway-A imports them) — kept.

## Deferred on purpose (not done)

Diminishing value now, and each would churn user-visible invocation or add risk without a
current need. Listed so a future pass knows they were considered, not missed:

- **CLI entry scripts → `scripts/`** — would change how everyone invokes `desi-DLAGP.py` / `desi_cddf.py`. Not worth breaking muscle memory + docs for a cosmetic move.
- **`tests/conftest.py` path shim** — largely redundant once `pip install -e .` is the norm; revisit if the test tree grows.
- **Delete the back-compat shims** — only after confirming no external notebook / handoff imports them.
- **`examples/` reorg** — low traffic, low payoff.

## Verification

- `pytest tests/test_cddf_mock.py tests/test_cddf_calibration.py` → **67 passed** (the two CDDF suites) in the `gpdla` env.
- `pip install -e . --no-deps` succeeds; `import gpy_dla_detection`, `from CDDF_analysis.hbi import ...`, `import run_bayes_select` all resolve **from `/tmp`** (i.e. not the repo root) and still resolve the old CWD way.
- `git grep -nE 'slurm_(cddf|preload|train|vi)'` → no matches (no dangling old SLURM paths).
- Frozen files absent from every diff (`gpy_dla_detection/`, `hbi/cddf_catalog_hbi.py`).

## Trace it

```bash
# the whole refactor, before → after, in one diff:
git diff e507d38..v0.1.0

# just the structure move:
git diff e507d38..v0.1.0 -- slurm_cddf slurm/

# confirm the inference + estimator never changed across the tag:
git diff e507d38..v0.1.0 -- gpy_dla_detection CDDF_analysis/hbi/cddf_catalog_hbi.py   # empty
```

`v0.1.0` is `desi_y3` at the tip of this series (matches `pyproject.toml` `version`).
Pre-refactor state is `e507d38`.
