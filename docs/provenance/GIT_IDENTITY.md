# Git author identity: the `panel5@test.local` episode

**Status:** resolved by `.mailmap` + a config fix. **History deliberately NOT rewritten.**
**Decided:** PI decision 5, 2026-07-29.

## What happened

A repo-local override in `/home/mfho/desi_gpy_dla_detection/.git/config` set

```
user.name  = panel5
user.email = panel5@test.local
```

which shadowed the correct global identity in `/home/mfho/.gitconfig`
(`Ming-Feng Ho <mfho@umich.edu>` — recorded there as `jibanmich <mfho@umich.edu>`).
Because git worktrees share the common `.git/config`, **every** linked worktree inherited
the placeholder, so a run of agent-panel commits were recorded as
`panel5 <panel5@test.local>` in **both** the author and the committer field.

## Scope, as measured

Measured on 2026-07-29 at `a3abb19`:

| measurement | command | result |
| --- | --- | --- |
| commits with author email `panel5@test.local`, all refs | `git log --all --format='%ae' \| grep -c panel5@test.local` | **88** |
| same, committer field | `git log --all --format='%ce' \| grep -c panel5@test.local` | **88** |
| unpushed commits (all local branches, not on any remote) | `git rev-list --all --not --remotes \| wc -l` | **131** |
| of which `panel5@test.local` | — | **87** |
| on `hbi-mcmc-threeroute` vs `origin/desi_y3` | `git log --format='%ae' origin/desi_y3..hbi-mcmc-threeroute` | **60 of 63** |
| on `lls-subdla-cddf` vs `origin/desi_y3` | same, that branch | **27 of 63** |
| first / last affected | `20fb1e7` (2026-07-10) / `1b930d6` (2026-07-29) | — |

Every `wip/*` branch shares these commits by ancestry, so the per-branch counts overlap;
the non-overlapping figure is the 87 above. **No pushed commit is affected** — the
placeholder first appears at `20fb1e7`, which is not an ancestor of any remote ref.

## Why history was NOT rewritten

A `filter-branch` / `filter-repo` pass over 87 commits re-SHAs those commits **and every
descendant**. That is not cosmetic here: this project's provenance convention is that a
result artifact is trustworthy only if its `metadata.code_commit` names a real commit
(`git cat-file -e <sha>`), and several committed artifacts and tombstones stamp exactly
those SHAs. A rewrite would:

1. invalidate every `code_commit` stamp pointing into the rewritten range, converting
   clean-stamped artifacts into the `ORPHANED` provenance class wholesale;
2. break the DIRTY-vs-CLEAN distinction the artifact tombstones rest on
   (`CDDF_analysis/hbi/tombstones/SCHEMA.md`), since a CLEAN stamp is defined as a
   40-char sha that *exists in this repo*;
3. force a coordinated re-clone across nine live worktrees.

The defect being repaired is a **display** defect. `.mailmap` repairs display exactly, and
by construction cannot touch a commit object.

## The remedy, in three parts

1. **`.mailmap`** at the repo root maps `panel5 <panel5@test.local>` (and any name with
   that email) to `Ming-Feng Ho <mfho@umich.edu>`. `git log`, `git shortlog` and
   `git blame` honour it automatically; `git log --format=%ae` and the raw commit objects
   do not, which is the point — the record of what actually happened is preserved.
   It additionally folds the two same-email name aliases (`jibanmich`, `jibancat` at
   `mfho@umich.edu`). It deliberately does **not** fold the historical UCR/NTU addresses:
   mapping across emails is a stronger attribution claim than decision 5 authorised.

2. **The config override was removed** (`git config --local --unset user.email`,
   `--unset user.name`), so the correct global identity applies going forward in every
   worktree. Verified:

   ```
   $ git config --show-origin --get user.email
   file:/home/mfho/.gitconfig      mfho@umich.edu
   ```

3. **This note**, so the episode is documented rather than merely hidden.

## Verification (real output, 2026-07-29)

`git shortlog -sne --all` **before** (`.mailmap` absent):

```
   522	jibanmich <mfho@umich.edu>
   498	jibanmac <mho026@ucr.edu>
    88	panel5 <panel5@test.local>
    83	jibancat <mho026@ucr.edu>
    82	jibancat <mfho@umich.edu>
    26	Ming-Feng Ho <mfho@umich.edu>
    26	Ming-Feng Ho <mho026@ucr.edu>
    12	jibancat <r04244007@ntu.edu.tw>
    10	Roman Garnett <garnett@wustl.edu>
     4	Roman Garnett <romanempire@gmail.com>
     4	Simeon Bird <sbird@ucr.edu>
     2	copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>
     1	Roman Garnett <romangarnett@gmail.com>
```

**after** (`.mailmap` present):

```
   718	Ming-Feng Ho <mfho@umich.edu>
   498	jibanmac <mho026@ucr.edu>
    83	jibancat <mho026@ucr.edu>
    26	Ming-Feng Ho <mho026@ucr.edu>
    12	jibancat <r04244007@ntu.edu.tw>
    10	Roman Garnett <garnett@wustl.edu>
     4	Roman Garnett <romanempire@gmail.com>
     4	Simeon Bird <sbird@ucr.edu>
     2	copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>
     1	Roman Garnett <romangarnett@gmail.com>
     1	jibanCat <r04244007@ntu.edu.tw>
```

`522 + 88 + 82 + 26 = 718` — the four folded rows, and nothing else moved.

### Proof that no SHA changed

Hash the complete list of commit ids with the mailmap absent and present:

```
$ git rev-list --all | sha256sum      # .mailmap absent
85e4842c84f96046f44c08931ee631c812842c7e56ee0ec10ccd38982a656cb4  -
$ git rev-list --all | sha256sum      # .mailmap present
85e4842c84f96046f44c08931ee631c812842c7e56ee0ec10ccd38982a656cb4  -
$ git rev-list --all --count
1359
```

Bit-identical over all 1359 commits, and `HEAD` is `a3abb19bafb0b5249d5094d9661fd8f91eb220c4`
in both readings. This is a demonstration, not a discovery: `.mailmap` is consulted at
*display* time and is not an input to the commit hash. The demonstration is recorded
because "it cannot change a SHA" is exactly the kind of claim this project requires to be
shown rather than asserted.

## Going forward

Commit as `Ming-Feng Ho <mfho@umich.edu>`. If a tool or agent needs a distinguishable
identity, use a real address and a `Co-Authored-By:` trailer — never a `*.test.local`
placeholder in `user.email`, and never a repo-local override, which silently propagates to
every linked worktree.
