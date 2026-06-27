# Real-LOA results: DO NOT COMMIT

This repository is **public**. Mock results are fine to commit; **real-LOA**
spectra and any per-object value derived from them (per-sightline N_HI/z,
results-store leaves, etc.) are **NOT**.

## The rule

- Mock results-store leaves (`provenance.json` with `privacy.class == "mock"`,
  `privacy.shareable == true`) **may** be committed.
- Real-LOA leaves live **only** under `$CDDF_STORE/real_loa/...` on scratch
  (outside this repo). They must **never** enter git history.
- There is intentionally **no** `real_loa/` directory inside this repo. If you
  ever see one staged, you are about to leak real data — stop.

## Why this file exists

It is a sentinel: a harmless marker in the tutorial-data tree that documents the
rule next to where someone might be tempted to drop a "just one" real-LOA
artifact. It is itself shareable (a doc, not a leaf).

## The automated guard

`tools/provenance/precommit_privacy_guard.py` enforces the rule on the staged
set. It **fails the commit** (exit 1, naming each offending path) if any staged
path:

1. sits under a `real_loa/` store partition, or
2. is a `provenance.json` with `privacy.class == "real-LOA"` or
   `privacy.shareable == false`, or
3. sits in the same leaf dir as such a `provenance.json`.

A `provenance.json` that cannot be parsed is treated as **suspect** and blocks
the commit (fail-closed). Wire it as a pre-commit hook, or run it manually:

```bash
python tools/provenance/precommit_privacy_guard.py            # scan git staged set
python tools/provenance/precommit_privacy_guard.py --paths .  # scan a tree (CI)
```

See the real-data-privacy note in the project memory for the underlying policy.
