# Working rules for this repository

Binding on **automated agents** dispatched into this tree and on collaborators running
multi-stream work. These are operating rules, not style guidance: several exist because the
corresponding failure already happened here.

Scope note: this repository is the **DESI DR2 LoA** GP-DLA effort. Mock-derived values are
public-OK. Real-DESI result values — including in commit messages — belong in the private notes
repository, never here.

---

## 1. PI checkpoint stop — three triggers

**Stop and hand back to the PI. Do not resolve it yourself.** Escalating is a *successful*
outcome for a work stream, not a failure to deliver.

### 1.1 Budget changes

Any movement in the compute/cost picture: a new cost estimate, an existing estimate that moves
materially, approaching or exceeding the standing ~5,000 CPU-h cap, or discovering that a planned
run costs more than believed.

State the number against the remaining budget and stop. Do not decide to run it — and equally
**do not decide to skip it**. "Too expensive, so I dropped it" is also a PI decision, and folding
it into a report is not escalating it.

### 1.2 Science interpretation needed

Whenever the next step requires deciding what a result **means** rather than what it **is**:

- which estimand, window or floor to report;
- which configuration is primary versus a reported sensitivity;
- whether a residual is physical or bookkeeping;
- whether a null is a true null or merely underpowered;
- whether a systematic is propagated or absorbed.

Streams produce **measurements**; the PI produces **interpretations**. Present the measurement and
the plural candidate readings. Do not pre-announce which reading you expect to win — that biases
the very decision being escalated.

### 1.3 A gate failed

Any closure / ratification / audit / provenance gate that fails, **or** that you discover is
fail-open, vacuous, or unsatisfiable.

Forbidden self-resolutions: self-ratifying a tolerance, disarming an armed gate, relaxing a
threshold, re-baselining, narrowing the reported window, or reframing a failure as a pass. Report
the failure with its measured numbers and stop.

### 1.4 What this does *not* re-open

Mechanical work still runs end-to-end without asking: pytest, python probes, mutation batteries,
reductions, and commits on `wip/*` branches. The three triggers are narrow and specific.

---

## 2. Authority and provenance discipline

- **Never** write `authority=PI`, `status=RATIFIED`, or `paper_facing: true` for anything the PI
  has not ratified **in writing**. Prefer an explicit allow-list that fails closed over a
  convention you intend to honour.
- A tolerance does not inherit ratification by living in the same dict, commit, or hunk as a
  ratified one. Record each entry's true status and its true introducing commit.
- A headline number's provenance is a **committed routine plus a git stamp** — never an untracked
  scratch file.
- Quote no count, ratio or score you have not just seen in real output. If a remembered number and
  a measured one disagree, the measurement wins and the discrepancy gets recorded.

Why this is rule-shaped: a stream once recorded a criterion the PI had called *malformed and sent
back for restatement* as `status=RATIFIED, authority="PI"`, with `contributes_to_pass_fail=True`,
and then **pinned the false record with a test**. It had to be retracted. Another stream disarmed
two previously-armed production gates on a false-alarm rate measured at a toy grid geometry that
did not transfer to production scale.

---

## 3. Verification discipline

- **Mutation-test every new test.** Revert the fix (or inject the specific one-line mutant) and
  confirm the test goes red. A test green under both the fix and its revert has no power and is
  itself a defect. Report a per-test mutation table with the passing baseline count.
- **Coverage and containment claims need a measured power check.** Containment is monotone in band
  width, so it cannot fail an over-wide band — an over-dispersed posterior once passed every
  containment test. Always report the mis-scaling detection curve.
- **Check support-matching first.** The recurring defect class in this project is a numerator/basis
  and a denominator/target living on different supports. It has surfaced repeatedly. Check it
  *before* blaming the sampler, the prior, or false-positive subtraction. The decisive check is a
  counting argument.
- Check `.err` as well as `.log`. Trace the executed data path with prints; do not infer behaviour
  from docstrings.

---

## 4. Git worktree lifecycle

Parallel streams get one worktree each so they cannot collide. Worktrees are **temporary
scaffolding and must be cleaned up** — a stale worktree silently pins an old commit and invites
work against a superseded base.

**Creating.** One worktree per stream, branched off the correct parent. The guard layer and the
inference core are separate lineages and **must not be merged into each other**; branch off the
one you are actually changing.

**Removing — all three conditions must hold:**

1. `git -C <worktree> status --porcelain` is **empty, including untracked files**.
   `git worktree remove` preserves the branch and its commits, but untracked files are lost
   permanently.
2. The branch is an ancestor of its parent (`git merge-base --is-ancestor <branch> <parent>`),
   i.e. the work is genuinely integrated — not merely committed locally.
3. No open pull request depends on the worktree.

Then `git worktree remove <path>` followed by `git worktree prune`.

**Never remove** the primary checkout, or a worktree still under an active agent.

**Report the disposition.** At the end of a multi-stream run, list each worktree as removed or
retained *with the reason*. Unmerged-and-therefore-retained is a normal, expected outcome — say so
rather than cleaning up prematurely or staying silent.

---

## 5. Standing operational constraints

| Constraint | Rule |
|---|---|
| Push / history rewrite | Forbidden unless explicitly requested. |
| Cross-lineage merge | Never merge the guard layer with the inference core. |
| Compute | ~5,000 CPU-h cap. Estimate plus PI sign-off before any submission above ~500 CPU-h (cores × walltime × array size). |
| Login node | 8 CPUs. Anything beyond ~30 min goes to SLURM; do not parallelise heavy pilots on the login node. |
| BLAS | Export `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Oversubscription measured 3.7–8.9× slower for identical results. |
| Environments | `gpdla-hbi` for the HBI/inference modules (needs jax); `gpdla` for the finder, CDDF and unblind layers. |
| Privacy | No real-DESI result values in this repository, commit messages included. |

---

## 6. Reporting

- Report outcomes faithfully: if tests fail, say so with the output; if a step was skipped, say
  that. State plainly what is done and verified, without hedging.
- If a bounded scope was used (top-N, sampling, no-retry), log what was dropped. Silent truncation
  reads as complete coverage when it was not.
- Distinguish a plausibility argument from a hypothesis test. Reserve "confirmed" / "real" for
  claims backed by an experiment that could have falsified them.
