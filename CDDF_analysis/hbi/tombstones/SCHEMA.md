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

   ⚠️ **Known limit of that check, stated rather than papered over:** the no-float rule
   sees JSON *types*, so it cannot see a number written inside a prose string. The
   tombstones here do contain such numbers — log-N window edges (17.2, 17.5, 19.0, 19.2,
   19.5, 20.3), basis resolutions (0.1, 0.2 dex) and the B16 leak factors
   (1.0557–1.1818, already committed verbatim elsewhere in the repo). These are
   *configuration and defect-magnitude* descriptors, not the retired measurement, and the
   leak range is what makes "RE-DERIVE, never rescale" actionable. A reviewer adding a
   tombstone must still read the prose: the mechanical rule is a floor, not a proof.
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
| `metadata` | `code_commit` (clean full 40 chars), `routine` + `rederive` (the **tombstone's own** builder, so it audits `RE_DERIVABLE` rather than `NO_ROUTINE`), `builder`, `paper_facing: false` |

### Interaction with the repo-wide audit

A tombstone is itself a committed artifact, so `CDDF_analysis/unblind/audit.py` walks it.
Two consequences:

* it must declare its **own** generating routine in `metadata.routine` / `metadata.rederive`
  (the builder). The retired artifact's command lives separately and read-only at
  `/artifact/rederive_command_as_stamped` and is never used as the tombstone's provenance.
* `/artifact/stamped_code_commit` legitimately holds a `<sha>-dirty` string, which
  `tests/test_unblind_audit.py::test_no_committed_artifact_hides_dirty_substamps_under_a_clean_one`
  is designed to reject. It is exempted **only** at that exact path and **only** for docs
  declaring `schema == "hbi-artifact-tombstone"`. The exemption is itself guarded by
  `test_the_tombstone_exemption_is_narrow_and_cannot_be_abused`: a schema-blind or
  prefix-widened exemption fails it, a dirty sha anywhere else in a tombstone still fires,
  and a tombstone whose **own** `metadata.code_commit` is dirty still fires.

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
                   "corroborating_pointer": "/self_recovery_baseline_2lpt0/cumulative_map/dndx/19.5",
                   "sha256_of_repr": "…",
                   "corroborating_sha256_of_repr": "…"}],
  "derived_commitments": [{"from_committed_artifact": "…subdla_mock_validation_forward.json",
                           "sum_of_pointers": ["…/dndx_est_195_203", "…/dndx_est_203"],
                           "equals_pointer_in_retired_artifact": "/measurement/19.5/dndx/integrated/MAP",
                           "sha256_of_repr": "…"}]
}
```

`sha256_of_repr` is `sha256(repr(float(value)).encode())`. It is a **commitment, not a
value**: it cannot be plotted, quoted, or integrated, and it is not a number the schema's
no-float rule would admit.

**Why the commitment is TWO-SIDED.** The corroborating artifact
(`crossmock_transfer_loa0.json`) is *also* untracked scratch, so a one-sided commitment
would leave the agreement unverifiable the moment both files are gone. The builder
therefore hashes both sides and **refuses to write the tombstone unless the digests
match**. Because `repr()` of an IEEE-754 double is round-trip exact,
`sha256(repr(a)) == sha256(repr(b))` **iff** `a == b` bit-for-bit — so the committed pair
of digests is a permanent, value-free **record** of the agreement, and the consumer test
asserts it **unconditionally**, with no file present and no `pytest.skip` path.

**What that record is NOT (corrected 2026-07-29 after an adversarial re-run).** The
consumer assertion compares two hex strings that live in **the same committed file**, so it
verifies the *record* of the builder's stamp-time check, not the check itself. It is
therefore **self-referential and cannot detect a fabricated record**: replacing all four
`sha256_of_repr` *and* `corroborating_sha256_of_repr` with digests of invented values (four
distinct matching pairs) left the suites fully green — measured, 29 passed. Do not read
"certifies the bit-for-bit agreement" as "re-derives it".

The one place the endpoint block gains real teeth is the **cross-namespace anchor**: the
`/measurement/19.5/dndx/integrated/MAP` endpoint digest and the `derived_commitments`
digest for the same pointer are `sha256(repr(·))` of the *same* float by construction, and
the derived one **is** recomputed from committed data. Both suites now assert that
equality, so a fabricated or edited endpoint block is caught **at that one pointer**. The
other three endpoints (`omega` at both limits, `dndx` at 20.3) remain record-only: they
have no committed re-derivation, and no test can give them one until a committed artifact
carries those quantities.

**`derived_commitments`** re-anchor a *secondary* check that also read the retired file
(`band + DLA-tier == cum(19.5)`, i.e. the band is a difference-slice and not a mislabelled
cumulative). Both summands live in a **committed** artifact, so the tombstone commits the
digest of their sum. The consumer recomputes that sum and must reproduce the digest — this
is the one place with real arithmetic teeth, and it upgraded the original check from an
opportunistic `abs=1e-9` read to an unconditional bit-for-bit assertion.

**Commit-message correction.** `c158872` says "equality of the two digests IS bit-for-bit
equality of the two floats -- a value-free certificate that needs neither file present", and
`50e388d` says "its bit-for-bit certification now runs unconditionally". Both are too strong
without the qualification above: the *implication* about `repr()` is true, but a test that
reads both digests out of one committed record certifies the record, not the agreement.
History is not rewritten, so this paragraph is the correction of record.

Endpoint and derived commitments share a pointer, so consumers must keep them in
**separate namespaces**; merging them into one pointer-keyed dict silently shadows the
two-sided endpoint commitment with a one-sided derived one (this was a real bug, caught by
running it).

Net effect of the retirement on the tripwire, measured: naively deleting the record now
turns **3 tests RED**; before the retirement the same deletion produced a silent skip.

## Adding a tombstone

Edit `RETIRED` in `CDDF_analysis/hbi/tombstones/build_tombstones.py` and run it with
`--source-worktree` pointing at a tree that still holds the artifact. It stamps the full
40-char `HEAD` sha and refuses to run against a dirty tree.

## `--check`: what it compares, and what audits it

`build_tombstones.py --check` rebuilds every tombstone in memory from `RETIRED` and diffs it
against what is committed, writing nothing. It ignores exactly three **leaves**
(`build_tombstones.VOLATILE_LEAVES`): `retirement.retired_utc`, `metadata.generated_utc` and
`metadata.code_commit` — the two timestamps and the builder's own `HEAD` stamp, which move on
every legitimate re-stamp. **Everything else, including the whole `retirement` payload, is
compared.**

It did not always. Until 2026-07-29 the exclusion was the two whole blocks
`("metadata", "retirement")`, and the consequence was measured: flipping
`recoverable_from_git`, rewriting `recovery_note` to "FABRICATED: totally unrecoverable,
trust me." and swapping the defect code `DIRTY_STAMP_NOT_REDERIVABLE` →
`STALE_FP_RESAMPLE_SEMANTICS` printed `OK` for all four tombstones with exit 0, and the test
suite stayed green. **The tombstone's actual payload — which defect, and whether the source
is recoverable — was verified by nothing**, and nothing invoked `--check` at all.

Two tests now cover it, in `tests/test_tombstones.py`:

* `test_check_mode_treats_only_timestamps_and_the_head_stamp_as_volatile` pins
  `VOLATILE_LEAVES` from both sides (a re-stamp must not read as DRIFT; each falsified
  payload leaf must), and is host-independent.
* `test_check_mode_runs_clean_and_detects_a_falsified_payload` actually **runs** `--check`
  as a subprocess: clean → exit 0 with four `OK`s; then it falsifies the committed record on
  disk, requires `DRIFT` and a non-zero exit, and restores the bytes. It is a hard failure
  whenever a tree holding the four untracked retired artifacts is reachable, and skips with
  a stated reason where `--check` physically cannot rebuild them.

## The working-tree fingerprint guard is pinned, not self-certified

`test_working_tree_reappearance_must_match_the_recorded_fingerprint` used to read **both**
the expected digest and the path to compare against out of the record it was auditing.
Measured defeat: `artifact.sha256 = 'f'*64` plus
`artifact.read_from_worktree = '/nonexistent/tree'` left it passing (4 passed), because the
loop `continue`d when neither candidate path existed. Its teeth also depended on the literal
path `/home/mfho/desi_gpy_dla_detection` existing, so on any other host it was **silently**
vacuous.

Now the test module holds two independent pins — `EXPECTED_FINGERPRINT` (sha256 + byte count
per retired artifact) and `CANDIDATE_TREES` — separate from `RECORDING_WORKTREE`, which is
the historical worktree the record must *declare*. The guard (i) requires the record to match
the pinned fingerprint, (ii) requires `read_from_worktree` to be the pinned recording
worktree, and (iii) hashes every reachable candidate copy against the **pin**, not the
record. A falsified record, a redirected path, or a re-run into the retired identity are each
RED. When no copy is reachable the byte-level third of the guard is inert, and that is now an
explicit `pytest.skip` naming the absent trees plus a companion test
(`test_fingerprint_guard_host_dependence_is_declared`) that states the host-dependence — parts
(i) and (ii) run everywhere regardless.
