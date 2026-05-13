# BAL masking — what `--balmask` does today, and how to smoke-test it on 2LPT

> **Status**: read-only investigation, 2026-05-13. No jobs were run.
> All file/line references are against branch `production_533`
> (`/pscratch/sd/j/jibancat/desi_gpy_dla_detection`).

The user has never tested whether BAL masking helps DLA detection. Production
runs include BAL QSOs without any pixel masking (`BALMASK=false` in
`slurm/configs/_base.env:63`). Purity/completeness reporting in
`examples/molly_faithful_pc_plots.py` then excludes BAL TARGETIDs via
`--no-bal`, which simply drops those QSOs from both catalog and truth
*after* inference. The natural question is whether `--balmask` would let
us keep BAL QSOs in the catalog with usable P/C, rather than discarding
them at evaluation time.

---

## §1 What `--balmask` does today

`--balmask` is a CLI flag in `desi-DLAGP.py:96–101`. When set, it does
**three** distinct things:

### 1.1 Catalog read — pulls AI_CIV + velocity windows into the QSO catalog

- **Real LOA** (`desi-DLAGP.py:660 read_catalog`):
  - With `balmask=True`, reads extra columns from the QSO catalog:
    `AI_CIV, NCIV_450, VMIN_CIV_450, VMAX_CIV_450` (lines 681–693).
  - `VMIN_CIV_450` / `VMAX_CIV_450` are vector columns — one entry per BAL
    trough, padded with `-1`. `NCIV_450` is the count of valid troughs.
  - `constants.no_bal=False` (line 49) so BAL QSOs are **not removed** from
    the catalog; only the trough columns get loaded.

- **Mocks** (`desi-DLAGP.py:767 read_mock_catalog`):
  - With `balmask=True`, opens `<mockdir>/bal_cat.fits` (line 806) and
    pulls the same 4 columns: `TARGETID, AI_CIV, NCIV_450, VMIN_CIV_450, VMAX_CIV_450`.
  - For each QSO in `zcat`, looks up its row in `bal_cat` by TARGETID and
    appends the BAL columns to the catalog row (lines 816–827).
  - QSOs not in `bal_cat` get `AI_CIV=0`, `NCIV_450=0`, `VMIN/VMAX=-1`.
  - No QSO-level exclusion is performed at this step either.

### 1.2 Pixel masking — applied per-spectrum inside the worker

The actual masking happens in `dlasearch.py:438–467 process_spectra_group`:

```python
if "NCIV_450" in catalog.columns:
    nbal = catalog["NCIV_450"][entry]
    bal_locs = []
    for n in range(nbal):
        v_max = -catalog[entry]["VMAX_CIV_450"][n] / constants.c + 1.0
        v_min = -catalog[entry]["VMIN_CIV_450"][n] / constants.c + 1.0
        for line, lam in constants.bal_lines.items():
            mask = np.logical_and(wave_rf > lam * v_max, wave_rf < lam * v_min)
            ...
            pixel_mask[mask] = True
            ivar[mask] = 0
```

Mechanism per BAL trough:
- Convert blueshifted velocity edges `(VMIN, VMAX)` to rest-frame wavelength
  factors `(1 − v/c)`. Note the trough is *blueward* of the line center, so
  `VMAX > VMIN > 0` and the mask spans rest-frame
  `[lam*(1 − VMAX/c), lam*(1 − VMIN/c)]`.
- Apply to **17 BAL lines** simultaneously (`constants.bal_lines`, lines
  79–98): CIV, SiIV1/2, NV, Lyα, CIII (×2 — there is a duplicate-key bug
  for CIII at 1175 vs 977 Å; the second overwrites the first), PV1/2,
  SIV1/2, OIV, OVI, OI, Lyβ, Ly3, NIII, Ly4.
- `pixel_mask[mask] = True` excludes those pixels from the GP likelihood.
  `ivar[mask] = 0` is then propagated into `noise_variance = NaN` (line
  470–473) so the same pixels are also dropped from any downstream
  per-pixel computation.

### 1.3 Post-detection BAL flag (auxiliary)

Only the Lyα + NV trough windows in *observed* frame are also stored as
`bal_locs` (lines 458–462) and used after inference to set
`DLAFLAG.POTENTIAL_BAL` on any MAP DLA whose Lyα line falls inside a BAL
window (lines 547–551). This is a purity-flag annotation; it does NOT
change which spectra are included.

### 1.4 Summary

`--balmask` is **pixel-level masking**, not QSO-level exclusion. BAL QSOs
are still inferred on; we just zero out the BAL trough pixels (across 17
emission lines) before computing the GP likelihood. There is also a
companion catalog-level switch `constants.no_bal = True` (line 49) which,
in `read_catalog` only, drops every BAL QSO from the catalog entirely
(line 748). `read_mock_catalog` does NOT honour `constants.no_bal` —
this asymmetry is a small latent bug but not load-bearing for the
smoke test.

---

## §2 2LPT mock-0 BAL catalog

### 2.1 Path

The 2LPT mock-0 directory is

```
/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/
```

confirmed by `slurm/configs/2lpt0_y3.env:5` and reproduced in
`prod533_5k_20260511/2lpt0_y3/RUN_SETTINGS.md:11`.
By analogy with London (`jura-124/bal_cat.fits`, referenced in
`examples/_sweep_cuts.py:5`) and Saclay (`juraLy8-124/bal_cat.fits`,
referenced in `docs/runs/2026-05-12_saclay_v3_loa124_results.md:140`), the
2LPT BAL catalog is expected at

```
/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits
```

**Update 2026-05-13 (main thread): path verified.** `bal_cat.fits` exists
at the path above (108 504 000 bytes, modified 2026-01-15 by `hiramk`,
owner `desi`). No need to re-run the verification snippet below — left
here only for future-use of similar paths.

The sub-agent originally couldn't verify because its sandbox refuses
`ls`/`find` against `/global/cfs/cdirs/desicollab/`; the main thread
ran the check directly. For future sub-agents:

```bash
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
python - <<'PY'
import os
from astropy.table import Table
p = "/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"
print("exists:", os.path.exists(p))
if os.path.exists(p):
    t = Table.read(p)
    print("colnames:", t.colnames)
    print("rows:", len(t))
    print(t[:3])
PY
```

### 2.2 Expected schema

If the 2LPT bal_cat follows the same convention as London + Saclay (used
unchanged by `read_mock_catalog`), it must contain at minimum
`TARGETID, AI_CIV, NCIV_450, VMIN_CIV_450, VMAX_CIV_450`. Downstream
P/C evaluation in `examples/molly_faithful_pc_plots.py:572` also expects a
`BI_CIV` scalar column. The 2LPT BAL injection is generated by
`quickquasars` in the same release as London v5.9.5 and Saclay v4.7.5, so
all five columns should be present.

### 2.3 BAL fraction — what we know

- **2LPT mock-0 zcat total**: 1,213,217 rows (z ≥ 0 — pre-z-cut).
  Source: `prod533_5k_20260511/2lpt0_y3/logs/local_0_1.log` —
  `INFO:desi-DLAGP.py:778:read_mock_catalog: objects in catalog: 1213217`.
- After `z ∈ [2.0, 4.25]` cut the catalog is smaller; the same log
  doesn't print the post-z-cut size before z-filter is the only QSO cut
  in `read_mock_catalog`. From the file size of one spectra-16 (level2=0:
  1056 spectra), the processed slice is consistent with ~1.1–1.2M post-cut.
- **London BAL count**: `docs/notes/2026-04-27_london_production_pc.md:141`
  states "90,354 BAL targets identified" (BI_CIV>0). With the London zcat
  having ~750k QSOs at z≥2, this is ~12 % BAL fraction.
- The 2LPT BAL fraction is unmeasured but, given the same quickquasars
  BAL injection module, expected to be in the 8–15 % range.

### 2.4 Per spectra-16 file

A typical 2LPT mock-0 spectra-16 file contains **~1056 QSOs**
(observed value from `local_0_1.log`). At 12 % BAL fraction that is
~125 BAL QSOs per spectra-16 file.

---

## §3 Smoke-test design

Goal: a minimal-cost experiment that decides whether `--balmask`
produces a usable DLA catalog on BAL QSOs (vs the current "exclude BAL"
P/C convention). 5 of the 1150 spectra-16 files (~5300 QSOs) is the
right granularity — it matches the recent prod533_5k_20260511 production
slice size, and 4× the per-file wall time on jupyter (~27 min/file with
8 workers) is comfortable.

### 3.1 Configuration

Use the existing `slurm/configs/2lpt0_y3.env` flavour and the
`slurm/run_local.sh` driver. **One new file**: a 4-line wrapper that
exports `BALMASK=true` and points at a parallel OUTDIR. No code change.

```bash
# slurm/configs/2lpt0_y3_balmask.env (NEW — 4 lines)
source "$(dirname "${BASH_SOURCE[0]}")/2lpt0_y3.env"
BALMASK=true
OUTDIR="${OUTDIR:-/pscratch/sd/j/jibancat/prod533_5k_20260511/2lpt0_y3_balmask/}"
```

### 3.2 Run pair (baseline already exists)

Baseline (BALMASK=false) is **already done**:
`/pscratch/sd/j/jibancat/prod533_5k_20260511/2lpt0_y3/` (8 spectra-16
files, slices 0..8, MAX_DLAS=3, FILTER=1, τ-EB OFF).

Add **one** comparison run with the same shape but `BALMASK=true`:

```bash
cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
conda activate gpdla

# 5-file slice; matches the existing baseline by reusing its window/parallelism
bash slurm/run_local.sh slurm/configs/2lpt0_y3_balmask.env \
    --window 5 --parallel-files 5 --max-workers 8
```

5 files × ~27 min wall = **~30 min total** on a Perlmutter node with 5
parallel pythons × 8 inner workers (40 cores in flight; well within the
256-CPU node budget). Add ~10 % for the pixel-mask overhead — expect
**≤35 min wall**.

### 3.3 Comparison evaluation

Two P/C runs against the existing molly script — one **with** BAL
included (the new question), one **without** (the historical convention).

```bash
TRUTH=/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits
BAL=/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits

# Baseline OUTDIR, BAL-excluded (current convention) — should reproduce existing numbers
python examples/molly_faithful_pc_plots.py \
    --catalog-dir /pscratch/sd/j/jibancat/prod533_5k_20260511/2lpt0_y3 \
    --truth $TRUTH --bal-cat $BAL --no-bal \
    --snr-min 2.0 --nhi-min 20.3 --gp-conf 0.99 --lyb-veto \
    --out /pscratch/sd/j/jibancat/prod533_5k_20260511/2lpt0_y3/figures_molly_nobal

# Baseline OUTDIR, BAL-INCLUDED (untreated BAL spectra in P/C)
python examples/molly_faithful_pc_plots.py \
    --catalog-dir /pscratch/sd/j/jibancat/prod533_5k_20260511/2lpt0_y3 \
    --truth $TRUTH \
    --snr-min 2.0 --nhi-min 20.3 --gp-conf 0.99 --lyb-veto \
    --out /pscratch/sd/j/jibancat/prod533_5k_20260511/2lpt0_y3/figures_molly_withbal

# Masked OUTDIR, BAL-INCLUDED (this is the new measurement)
python examples/molly_faithful_pc_plots.py \
    --catalog-dir /pscratch/sd/j/jibancat/prod533_5k_20260511/2lpt0_y3_balmask \
    --truth $TRUTH \
    --snr-min 2.0 --nhi-min 20.3 --gp-conf 0.99 --lyb-veto \
    --out /pscratch/sd/j/jibancat/prod533_5k_20260511/2lpt0_y3_balmask/figures_molly_withbal

# Masked OUTDIR, BAL-excluded (sanity: pixel mask shouldn't hurt the non-BAL pop)
python examples/molly_faithful_pc_plots.py \
    --catalog-dir /pscratch/sd/j/jibancat/prod533_5k_20260511/2lpt0_y3_balmask \
    --truth $TRUTH --bal-cat $BAL --no-bal \
    --snr-min 2.0 --nhi-min 20.3 --gp-conf 0.99 --lyb-veto \
    --out /pscratch/sd/j/jibancat/prod533_5k_20260511/2lpt0_y3_balmask/figures_molly_nobal
```

### 3.4 Output metrics

The 2×2 grid is the deliverable:

| Inference \ Evaluation | BAL excluded (`--no-bal`) | BAL included |
|---|---|---|
| `BALMASK=false` (baseline) | already-reported P/C | NEW — BAL QSOs at face value |
| `BALMASK=true` (new run)   | sanity check (≈ baseline) | NEW — does pixel-masking salvage the BAL pop? |

Concrete numbers from each cell:
- Purity at `P_DLA ≥ {0.99, 0.999, 0.99999}` and `SNR_RED > 2`
- Completeness same cuts
- Per-NHI-bin breakdown for `[20.3, 20.6) / [20.6, 21.0) / [21.0, 21.5) / [21.5, 23.5]`
- BAL-only sub-population: split the catalog by `TARGETID ∈ BAL` and
  recompute P/C separately. Most informative single number — "BAL-only
  P/C under BALMASK=true" — is the new contribution.

### 3.5 Cost summary

- 5 new spectra-16 files × ~27 min each (parallel) = **~30 min wall**
- Storage: ~50 MB processed-h5 + ~10 MB dlacat per file → ~300 MB total
- No SLURM submission needed; runs on a jupyter compute node via `salloc`.

---

## §4 Expected outcomes

### 4.1 Literature priors

- **No DLA-specific BAL-masking measurements in this repo.** The
  Wang+2021 CNN paper and Garnett+2017 GP-DLA paper both elect to *exclude*
  BAL QSOs entirely rather than mask pixels. Neither reports P/C with
  pixel-masking enabled.
- The closest in-repo signal is `docs/notes/2026-04-27_london_production_pc.md`
  where the analyzer "load[s] bal_cat … but [does] not yet use [it] in any
  cut" (line 141). The user's prior assumption is that BAL pixel-masking
  was implemented but never validated.

### 4.2 Plausible P/C range

Three regimes:
- **Non-BAL QSOs**: pixel-masking should be a no-op (no BAL troughs to mask).
  Expect P/C identical to baseline within ~0.1 pp Monte-Carlo noise.
  *If not, the mask is leaking — surface that.*
- **BAL QSOs under BALMASK=false (face value)**: expect substantial
  false-positive inflation. The CIV trough at obs-frame
  ~1549 × (1 − v/c) × (1 + z_qso) overlaps the Lyα forest of the *next*
  high-redshift QSO behind it, and AI_CIV-class BALs frequently have
  troughs at v/c ~0.1 — i.e. ~150 Å blueward. These pixels look like
  absorption and the GP is happy to fit them as DLAs. From the London
  90k-BAL analysis, BAL QSOs are excluded specifically because they
  inflate the FP rate; including them at face value would drop purity
  by ~5–10 pp at the P_DLA ≥ 0.99 cut.
- **BAL QSOs under BALMASK=true (pixel-masked)**: the question is whether
  the GP can still constrain a DLA when ~10–30 % of the pixels in the
  Lyα forest are masked. The search-window quality cut at line 486
  (`dlasearch.py`) requires ≥ 20 % unmasked, so a BAL with 80%+
  trough coverage would be auto-skipped — that loss IS expected.
  Realistic BALs cover 5–30 % of the forest. Plausible outcome:
  recovery of 70–85 % of the BAL QSO sub-population at near-baseline
  purity (≥ 80 % at P_DLA ≥ 0.99). If true, net catalog size grows by
  ~8–12 % at near-constant purity. **That is the bullish hypothesis
  the smoke test should falsify.**

### 4.3 Failure modes the test should detect

1. **`bal_cat.fits` missing** → `read_mock_catalog` exits 1. Smoke test
   surfaces immediately at the first python launch (see §6).
2. **17-line BAL mask too aggressive** → all forest pixels masked, search
   window quality cut at `dlasearch.py:486–488` skips most BAL QSOs with
   `SEARCH WINDOW >80% MASKED`. Detectable by counting skipped TIDs in
   the log files vs the no-mask run.
3. **Bug: pixel mask interacts with τ-EB scan** — τ-EB null builds use
   `pixel_mask` (`tau_eb.py`), and with `noise_variance` set to NaN on
   masked pixels, a BAL-masked spectrum might destabilise the τ scan.
   Validate by running with `ENABLE_TAU_EB=0` first (the prod533_5k
   2lpt0_y3 baseline already has τ-EB off — good).

---

## §5 Scope of edits if positive

### 5.1 If pixel-masking works

The current `--balmask` plumbing is **complete** for mocks:
- CLI flag plumbed (`desi-DLAGP.py:96`).
- Catalog read with BAL columns (`desi-DLAGP.py:803–827`).
- Per-spectrum pixel masking (`dlasearch.py:438–467`).
- POTENTIAL_BAL flag for post-detection annotation (`dlasearch.py:547–551`).
- `BALMASK` knob in `slurm/configs/_base.env:63`, threaded through
  `slurm/run_local.sh:184–186`.

To make BAL masking the production default:

1. `slurm/configs/_base.env:63`: `BALMASK="${BALMASK:-true}"` (1-line flip).
2. `slurm/configs/BASELINE.md:62-64`: update the BAL handling row.
3. `docs/production_runbook.md:608-609`: update the "BAL included" note.
4. `examples/molly_faithful_pc_plots.py`: keep `--no-bal` as the default
   for headline P/C plots (so historical numbers stay comparable), but
   add a `--with-bal` companion figure that shows the catalog-side
   completeness over the BAL sub-population.
5. `CLAUDE.md` §6 ("Key Model Settings"): change `BAL treatment` line.

### 5.2 If pixel-masking is no help or hurts

No production change needed. Add a single-line follow-up note to
`docs/notes/2026-05-13_bal_masking_scope.md` and keep BAL exclusion
as the operating point. The smoke test artifacts (a parallel `_balmask`
OUTDIR + P/C panel) are sufficient evidence for future authors not to
revisit unless the model changes.

### 5.3 If pixel-masking shows mixed results

The CIV-line duplicate-key bug (`constants.py:79–98` — `"CIII"` is
defined at both 1175.0 and 977.0 Å; the second wins) merits attention.
A targeted second pass that masks only the CIV trough (a 1-line patch
in `dlasearch.py:455` to filter `constants.bal_lines` to `{"CIV"}`)
would isolate whether the 17-line mask is over-aggressive.

---

## §6 Open questions

1. **Does `bal_cat.fits` actually exist at the 2LPT path?** Cannot
   verify from this agent (CFS read sandbox-blocked). User should run
   the python verification block in §2.1 before launching the smoke test.
2. **Schema match.** If the 2LPT bal_cat lacks one of
   `{AI_CIV, NCIV_450, VMIN_CIV_450, VMAX_CIV_450}`,
   `read_mock_catalog` raises and exits 1. The MOCKID-vs-TARGETID
   column name (already a footgun in `pick_random_2lpt_targets.py:79`)
   could bite — verify with the §2.1 column print.
3. **BI_CIV vs AI_CIV.** `read_mock_catalog` uses `AI_CIV` (absorption
   index, requires a continuous trough of width > 450 km/s with
   normalized flux < 0.9); `molly_faithful_pc_plots.py` uses
   `BI_CIV` (Balnicity index, stricter — width > 2000 km/s).
   The smoke test will compare AI-masked inference against BI-defined
   evaluation sets. Whether to switch the evaluation to AI_CIV-based BAL
   exclusion (so the populations agree) is a follow-up decision.
4. **Does masking interact with the early-stop bug?**
   `docs/notes/2026-05-12_multidla_early_stop_bug.md` describes a
   multi-DLA early-stop where NaN log evidences abort the search.
   If pixel-masking pushes a BAL QSO into the "no valid regions"
   regime, we may falsely report no DLA where the unmasked search
   would have found one. Recommend running the smoke test with
   `EARLY_STOP_MODE=A` (early-stop OFF) as a sensitivity check.
5. **17-line mask vs CIV-only.** Some literature uses only CIV pixel
   masking. If the 17-line variant performs poorly, a CIV-only branch
   is a 1-line variant worth trying.
6. **Real LOA vs mock.** Mock BAL injection in quickquasars is more
   idealised than real BAL features (sharper troughs, narrower velocity
   distribution). A mock-positive result does NOT automatically extend
   to LOA. A follow-up "BALMASK=true on LOA 5k slice" run would be
   needed before adopting it as a production default. Cheap (~1 h wall
   on the existing LOA infrastructure under `slurm/configs/loa_y3.env`).

---

## Quick-reference: code touchpoints

| File:lines | What it does |
|---|---|
| `desi-DLAGP.py:96-101` | CLI flag definition |
| `desi-DLAGP.py:422-425` | "BALs will not be masked!" warning if `--balmask` not set |
| `desi-DLAGP.py:660-735` | `read_catalog` — real LOA, reads BAL columns from QSO catalog |
| `desi-DLAGP.py:767-833` | `read_mock_catalog` — mocks, joins `bal_cat.fits` by TARGETID |
| `dlasearch.py:438-467` | Per-spectrum pixel masking, 17 lines × N troughs |
| `dlasearch.py:486-488` | Search window quality cut (>20% unmasked required) |
| `dlasearch.py:545-551` | `DLAFLAG.POTENTIAL_BAL` post-detection annotation |
| `constants.py:79-98`   | The 17 BAL-line dictionary (`CIII` duplicate-key bug) |
| `constants.py:49`      | `no_bal = False` — alternative QSO-level exclusion (LOA only) |
| `slurm/configs/_base.env:62-63` | `BALMASK` default = `false` |
| `slurm/run_local.sh:184-186`    | Threads `BALMASK=true` → `--balmask` |
| `examples/molly_faithful_pc_plots.py:569-574` | `--no-bal` evaluation flag (drops BI_CIV>0) |
