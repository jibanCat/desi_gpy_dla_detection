# Artifact tombstones (`hbi-artifact-tombstone`, schema_version 1)

A **tombstone** is a small, committed, git-stamped record that retires a result artifact.

It exists because deletion is not retirement. Deleting a JSON removes the file but leaves
its *identity* (its repo path) free for something else to occupy, silently, later; and it
destroys the record of *why* the thing was untrustworthy. A tombstone keeps the identity
occupied by a refusal.

## Hard rules

1. **A tombstone carries NO retired science value.** No `dN/dX`, no `Omega`, no `f(N)`, no
   `ell(X)`, no `lambda_mfp`, no `tau_eff_LL`, no recovery ratio `R0`. Enforced
   mechanically by `tests/test_tombstones.py::test_tombstones_carry_no_science_values`,
   which rejects any float-valued leaf anywhere in a tombstone. Every number a tombstone
   is allowed to hold is an **integer** (byte counts, schema version, counts) or a **hex
   digest string**.
2. **Regeneration is a NEW MEASUREMENT with a NEW ARTIFACT IDENTITY.** Re-running the
   retired routine post-B16 does not repair the retired artifact's provenance; it produces
   a different measurement of a different (corrected) estimand. It must be written to a
   *different* path and stamped afresh. `successor_policy.must_not_reuse_identity` is
   `true` for every tombstone here, and the resurrection test enforces it.
3. **A tombstoned identity may never be silently resurrected.** If a file reappears at a
   tombstoned path *and is committed*,
   `tests/test_tombstones.py::test_tombstoned_identity_is_not_resurrected` goes RED.
   (Working-tree reappearance is tolerated — these artifacts were untracked scratch and
   still sit in the primary worktree — but committing one is the failure.)
4. **A tombstone may not be silently deleted.** The tombstone set is pinned by
   `TOMBSTONED` in the test module; a missing tombstone is RED, not a skip.

## Fields

| field | meaning |
| --- | --- |
| `schema` / `schema_version` | `"hbi-artifact-tombstone"` / `1` |
| `artifact.path` | the retired identity (repo-relative) |
| `artifact.sha256`, `artifact.bytes` | content fingerprint of the retired file **as read at retirement time**, from the primary worktree `/home/mfho/desi_gpy_dla_detection` |
| `artifact.stamped_code_commit` | the `metadata.code_commit` the retired file carried |
| `artifact.stamp_class` | `CLEAN` (a bare 40-char sha) or `DIRTY` (`<sha>-dirty`) |
| `artifact.was_tracked_at_git_head` | whether git ever held it (all four: `false`) |
| `artifact.what_it_was` | one-paragraph description of the product, no values |
| `artifact.rederive_command_as_stamped` | the command the retired file claimed would reproduce it |
| `retirement.retired_utc`, `retired_under` | when, and under which PI decision |
| `retirement.defects[]` | `{code, detail}`; codes below |
| `retirement.recoverable_from_git`, `recovery_note` | whether the producing tree can be recovered |
| `successor_policy` | statement + `requirements[]` + `must_not_reuse_identity` + `successor_identity_rule` |
| `values_policy` | explicit `carries_science_values: false` + rationale |
| `tripwire` | *optional*; present only where a live test depended on the retired file (see below) |
| `metadata` | `code_commit` (full 40 chars), `builder`, `paper_facing: false` |

### Defect codes

- `DIRTY_STAMP_NOT_REDERIVABLE` — `code_commit` ends in `-dirty`: the working tree that
  produced it was never committed, so git cannot recover it. Not a provenance *label*
  problem; an information-theoretic one.
- `LLS_POPULATION_CONTENT` — the leaves are LLS population measurements
  (`ell(X)`, `lambda_mfp`, `kappa_912`/`tau_eff_LL`). Standing project policy bars these
  as paper results; committing the values would be committing the retired result.
- `B16_LEAKY_TRUTH` — built from `tr['f_truth']`, which was integrated with **no z-mask**
  while the pathlength `Delta X` **was** masked. Anything derived from `f_truth` is biased
  regardless of what it is called; the correct response is RE-DERIVE, never rescale.

## The `tripwire` block

Some retired artifacts were load-bearing for a *test*, not for a result. Retiring them
naively converts a hard assertion into a `pytest.skip` — the exact silent-disarm this
project has been burned by. The `tripwire` block moves the dependency from *file presence*
to a *committed constant*:

```json
"tripwire": {
  "consumer": "tests/test_subdla_forward_headline.py::...",
  "commitments": [{"pointer": "/measurement/19.5/dndx/integrated/MAP",
                   "sha256_of_repr": "…"}]
}
```

`sha256_of_repr` is `sha256(repr(float(value)).encode())`. It is a **commitment, not a
value**: it cannot be plotted, quoted, or integrated, and it is not a number the schema's
no-float rule would admit. The corroborating artifact
(`crossmock_transfer_loa0.json`) is hashed the same way at test time and must match — so
the bit-for-bit head-vs-crossmock agreement the retired file used to certify is *still*
certified, with the retired file gone.

## Adding a tombstone

Edit `RETIRED` in `CDDF_analysis/hbi/tombstones/build_tombstones.py` and run it with
`--source-worktree` pointing at a tree that still holds the artifact. It stamps the full
40-char `HEAD` sha and refuses to run against a dirty tree.
