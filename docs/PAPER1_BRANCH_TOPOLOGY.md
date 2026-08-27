# Paper-1 branch topology and the finder / HBI dependency boundary (audit of 2026-08-26)

## 1. The authoritative line
**`hbi/forward-2026-08` is the Paper-1 science line of record.** The scientific history is one strictly linear spine rooted at `5b99163` (= local `desi_y3`, the merge-base of everything); every frozen artifact of 2026-08 was produced from the worktree `/home/mfho/wt_forward_2026_08` on this branch, and all twelve commits recorded inside frozen artifacts are on it (`4c95398` hz production tag, `14df2ce` H2 finder run, `1c02089`, `475c62b`, `70efc09`, `85bdba5`, `ebf3787`/`0babe21` CP-1, `b59e0b5`/`ea4c7bb`/`2d4035e` CP-3 + freeze, `5e26b35` BH ratify, `08504c0` recovery audit). At audit time it carried 34 unpushed commits (`ebf3787`…HEAD exist only locally) — **push before tagging**.

```
5b99163 (desi_y3 local; merge-base) ──2──▶ d5b306e = origin/desi_y3 (2026-07-10)
   │
   ├─128─▶ 9d73365 hbi-mcmc-threeroute ─▶ repair/phaseB ─▶ calibration/phaseC-highN-fp [85bdba5 C4]
   │        ─▶ calibration/phaseC-p1-coherent-ck ─▶ prov/p1-refold-2026-08-08 (3a65e2a) ─▶ prov/pre-nersc-highz-2026-08-12{,b,c}
   │        ─▶ prov/pre-gl-highz-2026-08-13 = 4c95398 ★hz catalogue production ─▶ 14df2ce ★H2 injection finder run
   │        ─▶ 1c02089 ─▶ 70efc09 ─▶ 475c62b ─▶ 686a3fc (origin/hbi/forward-2026-08) ─▶ ebf3787/0babe21 CP-1
   │        ─▶ b59e0b5/ea4c7bb CP-3 ─▶ 2d4035e freeze ─▶ 5e26b35 ratify ─▶ 08504c0 ─▶ … ★ hbi/forward-2026-08 HEAD
   │          └─ review/phaseA-adversarial-2026-08-05 (7 unmerged science commits; side branch)
   └─75──▶ 32e123f lls-subdla-cddf (LLS/sub-DLA tier, guard tests, tombstones, CDDF_analysis/unblind; never rejoined)

Upstream finder production (all ancestors of origin/desi_y3):
   b219996 ★2LPT-0 mock production (production_533, 2026-05-26) → the molly completeness matrices
   84fa654 ★real DESI Loa catalogue (nersc_production, 2026-06-06)
   d2ef1fc ★loa-0 FP companion (cddf_prod, 2026-06-15)
```

## 2. Branch roles and dispositions
| branch / tag | status | owns | required by Paper 1 | disposition |
|---|---|---|---|---|
| `hbi/forward-2026-08` (274 ahead of `5b99163`) | **downstream, live, authoritative** | HBI forward model, packs, CP-1/2/3, BH/H2 arm, hz finder production, code-review hardening | all 12 cited commits | push; tag; merge to `desi_y3` after PI approval |
| `origin/desi_y3` = `d5b306e` | upstream trunk | contains every finder production commit | `84fa654`, `b219996`, `d2ef1fc`, `c5f3211` | trunk; local `desi_y3` is 2 commits stale → fast-forward |
| `lls-subdla-cddf` (75 ahead) | parallel tier, independent | `gpy_dla_detection/lls/`, tombstone/provenance guards, `CDDF_analysis/unblind`, RULES.md | none (supplies `CDDF_analysis.unblind` for one skipped test) | keep independent; cherry-pick `docs/RULES.md` and the `generate_samples.py` arXiv-ID fix; reconcile at the merge checkpoint |
| `production_533`, `nersc_production`, `cddf_prod` | historical, fully merged | the three finder production states above | indirectly (frozen inputs) | archive-tag |
| `review/phaseA-adversarial-2026-08-05` | experimental, unmerged (7 commits) | Phase-A adversarial evidence | none | disposition at the merge checkpoint (cherry-pick or archive with a tombstone) |
| `hbi-mcmc-threeroute`, `repair/phaseB-*`, `calibration/phaseC-*`, `docs/terminology-*` | historical checkpoints (ancestors) | — | `85bdba5` lives in phaseC-highN-fp | archive |
| `wip/*` (9), `refactor/*`, `chore/*`, `loa0fp-resample-fix`, `crossmock-validation`, `cddf-analysis-reorg`, `bal-metal-fp-tests`, `clustering_prior` | integrated / merged / superseded | — | — | delete after the merge (integration commits exist) |
| tags `prov/*` (6) | provenance anchors | `prov/pre-gl-highz-2026-08-13` = the hz code of record | yes | keep permanently |
Worktrees: primary clone on `lls-subdla-cddf` (32e123f); `wt_forward_2026_08` on `hbi/forward-2026-08`; `hbi_mcmc_wt`, `hbi_tutorial_wt` historical (PR #21).

## 3. Finder isolation — source-level verdicts (production commit → `hbi/forward-2026-08` HEAD)
Architecture kept separate: **GP finder inference → catalogue/posterior products → HBI/selection/response/population inference.** The HBI engine imports nothing from the finder (`docs/HBI_ARCHITECTURE.md`); the finder imports nothing from the HBI code.

| production state | commit / branch | code stamp | verdict |
|---|---|---|---|
| (a) real catalogue `dlacat-loa-main-dark-v1.fits` | `84fa654` `nersc_production` | clean (`BASELINE.env`, FITS `CODECMT`) | **UNCHANGED.** Every module computing a likelihood/evidence/prior/sample/posterior is byte-identical `84fa654`→HEAD (`dla_gp.py` blob `0105697c…` identical at `84fa654`, `d2ef1fc`, `c5f3211`, `4c95398`, `14df2ce`, HEAD and `lls-subdla-cddf`). The six modified finder files: `constants.py` z-window as env-gated `GPDLA_ZMIN_QSO/ZMAX_QSO` with the historical defaults (input selection cut, not a likelihood); `dlasearch.py` opt-in archive I/O (`archive=None` ⇒ verbatim original path) + `np.in1d→np.isin` (mock path); `desi-DLAGP.py` opt-in CLI (`--spectra_archive`, `--pixel_col`, `--external_tid_list`; `UNIQPIX` fails loud); `run_bayes_select.py` one additive HDF5 root attribute after the results; two help-string typos. Re-running `84fa654`'s config at HEAD with no new env var reproduces the catalogue. |
| (b) 2LPT-0 mock production → molly matrices | `b219996` `production_533` | **unknown, backfilled** from a timestamp coincidence | **UNCHANGED AT PRODUCTION SETTINGS, source NOT identical.** `dla_gp.py` gained the gated clustering-prior hook (`pair_prior_mode == "clustering"`, `clustering_prior` merge `0154cc1`, 2026-05-29) — the only edit in the audit that adds a term to `log_likelihoods_dla`; doubly guarded, default `off`, and every production run records `PAIR_PRIOR_MODE=off`/unset. **Closed 2026-08-26 (PI ruling 1)**: one-healpix run at `b219996` vs HEAD with `PAIR_PRIOR_MODE=off` in one environment is BITWISE on every deterministic quantity; residuals are the finder's own unseeded multi-DLA resampling (matrix rows 5/5′). |
| (c) loa-0 FP companion | `d2ef1fc` `cddf_prod` | **`CODE_DIRTY=dirty`** | **UNCHANGED** (same hunk set as (a) minus two already-present files) — but the working-tree delta at launch is unrecoverable; the product is a frozen Paper-1 input (BH `--fp loa0`). Disclose (or re-run clean if the PI wants byte-level closure). |
| (d) high-z catalogue | `4c95398` = `prov/pre-gl-highz-2026-08-13` | clean | **UNCHANGED — zero hunks**, blob-identical finder tree; `slurm/greatlakes/production` changed only by additions. |
| (e) H2 injection finder run | `14df2ce` | clean | **UNCHANGED — zero hunks**; env file sources the hz env and overrides only QSOCAT/index/z-window. |
| `lls-subdla-cddf` vs production | `32e123f` | — | **no finder inference change**: lacks the 2026-08 archive/env work (fork state), adds the never-imported `gpy_dla_detection/lls/` package and `subdla_gp_lyman_break.py`, one docstring arXiv-ID fix. No contamination. |
Classification of every finder edit on `hbi/forward-2026-08`: genuinely required production change (z-window env gate; archive I/O; hz dispatch CLI), shared utility (`np.isin`; launcher forwarding; FILTER attribute), documentation only (help strings, env comments), scheduler plumbing (`launch_nersc.sh`, `LOA_BALANCE_COUNTS`), additive out-of-path packages (`injection/`, `inject_absorber.py`, `tools/h2_*`). **No accidental branch contamination found.** The standing rule "don't modify `dla_gp.py` behaviour" holds absolutely from `84fa654` onward and "GL speedups config-only" holds (verified: the hz env sources the NERSC flavour and overrides only I/O and allocation).

Track-C / cache-producing states: `forward_response_2lpt0.npz` ← `ecc06cb` (2026-07-29, `znz_kernel.build_forward_cache`); the quarantined Phase-C bridge ← `0b545cb`; the R_emp / znz caches: see the drift disposition in the pre-tag receipt.

## 4. Actions carried to the merge checkpoint
Push `hbi/forward-2026-08` (34 commits) and `lls-subdla-cddf` (1); fast-forward local `desi_y3`; archive-tag the three production branches; disposition `review/phaseA-adversarial`; delete integrated `wip/*`; reconcile `lls-subdla-cddf` ⊕ `hbi/forward-2026-08` → `desi_y3` under the pre/post-merge checks (manifest verify, profiles, byte-identical paper regeneration). Optional provenance closure before tagging: the (b) equivalence run and a clean re-run of (c).
