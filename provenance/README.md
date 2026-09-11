# `provenance/` — science-manifest tooling

**COMPONENT CLASS: PROVENANCE-ONLY / VALIDATION-ONLY. NOTHING HERE IS SCIENCE-PRODUCING.**

| file | class | what it does |
|---|---|---|
| `build_science_manifest.py` | **provenance-only** | emits `science_manifest.json` (validated against `SCIENCE_MANIFEST_SCHEMA.json`) for a freeze, plus a machine-readable `ledger_quantities.json` |
| `manifest_diff.py` | **validation-only** | reads two manifests and reports what is stale; computes no science value |
| `p1_gate.py` | **validation-only** | runs the P1 acceptance gate of `MANIFEST_DIFF_SPEC.md §3` against the diff output |

## The rule these tools obey

No component in this directory may open a spectrum, evaluate a posterior, re-run a
reduction, fit anything, or alter a printed value. Every number they emit is **transcribed**
from a file that some other lane produced, and every edge they record is **read from a
recorded source** (a provenance sidecar, a selection contract, a producer's own `emit()`
call) — never inferred. Where a link cannot be filled it is emitted as an explicit
`_GAP ...` marker with a matching entry in `gaps[]`; a gap is never silently dropped and is
never guessed.

`manifest_diff` walks only `products[].upstream` and `dependencies[].depends_on`. A product
with no recorded upstream and no producer commit is reported `UNTRACEABLE` — a finding, not
a pass. Absence on the old side is never evidence of no-change: it is reported as
`old_side_unrecorded`.

If a number in a manifest is wrong, the fix belongs in the science lane that produced it,
never here.

## What these tools do NOT do

- They do not write into the paper repository, the notes repository, or any frozen artifact.
- They do not relax any existing gate. `verify_frozen_status`, `common.pinned`,
  `table_common.require` and `tools/check_additions.py` keep their fail-closed behaviour;
  the manifest only supplies the expected value and makes the (correct) failure actionable.
- They do not adopt a freeze. A manifest with a blocking `open_items[]` entry makes the diff
  stamp itself `NOT-ADOPTABLE` and suppresses every re-pin recommendation.

## Usage

```bash
# the surgical C1 candidate
python3 provenance/build_science_manifest.py --mode candidate  --out science_manifest_C1_surgical.json

# the 2026-08-26 freeze, reconstructed (lossy; stamped previous_manifest_completeness=reconstructed)
python3 provenance/build_science_manifest.py --mode reconstruct-frozen --out science_manifest_2026-08-26_frozen.json

# what changed, and what the paper lane must re-emit
python3 provenance/manifest_diff.py OLD.json NEW.json --emit both --out-dir OUTDIR

# the acceptance gate for the diff machinery
python3 provenance/p1_gate.py --old OLD.json --new NEW.json --out-dir OUTDIR
```

Exit codes of `manifest_diff.py`: `0` no stale consumers, `1` stale consumers found (the
normal result after a freeze change), `2` schema-invalid input, `UNTRACEABLE` products on the
new side, a retired-but-referenced quantity, or an undeclared withholding.

Introduced 2026-09-11 under the PI ruling of that date, §4 (provenance-only tooling). No
science code was changed by this directory.
