# Test results overview

> Running record of correctness tests + perf benchmarks. Update this
> file when you land a new test, run a new benchmark, or supersede an
> earlier number. Easier than re-deriving from individual finding docs
> at write-up time.
>
> Linked from `docs/training_overview.md` ("see also: test results").

## How to use this file

When you add a test or benchmark:
1. Find the right table below.
2. Add a row with: scope (what was tested), result (the headline number),
   source (test file or commit/PR/notes path), date.
3. Keep entries sorted newest-first within each table.

When a test result changes (e.g. retrain → new endpoint scalars):
- Replace the row in place and bump the date.
- If the prior result is load-bearing for an older finding doc, leave
  a "(superseded by …)" cross-reference rather than deleting.

---

## 1. Correctness tests (PR #6 chain)

Frozen tests that gate on numerical equivalence to a reference (MATLAB,
v1, v3 vs v3-vectorized, etc.). Run via `pytest tests/`.

| Test | What it gates | Result | Source | Date |
|---|---|---|---|---|
| `test_v1_spectrum_loss_jacobian.py` | v1 hand-coded gradient ≡ FD numerical Jacobian | max rel_err ~1e-4 (one outlier pixel; math correct per A.2) | A.1 | 2026-05-07 |
| `test_v3_5_spectrum_loss_jacobian.py` | v3.5 hand-coded gradient ≡ FD | max rel_err 6.4e-10 on dlog_β | A.1 | 2026-05-07 |
| `test_v1_matches_matlab.py` | v1 Python `spectrum_loss` ≡ MATLAB `spectrum_loss.m` on 5 frozen 2lpt spectra | **max rel_err 5.30e-11** | A.2.a | 2026-05-07 |
| `test_v3_5_vs_matlab.py` | v3.5 (strict-dlog_β) vs MATLAB on 5 outputs | β diff 0.5–2.5% (chromatic-correction signature) | A.2.b | 2026-05-07 |
| `test_v3_objective_vectorized_parity.py` | `spectrum_loss_batch` ≡ per-spectrum loop on 6 fixtures | max rel 1e-10 / 6.4e-11 | B | 2026-05-08 |
| `test_v3_train_step_parity.py` | 3-iter Adam parity vec ≡ per-spec | max rel ~2e-10 | B | 2026-05-08 |
| `test_v3_objective_vectorized_jacobian.py` | Independent FD-vs-analytic on the batched function | max rel 4.01e-5 | B | 2026-05-08 |
| `test_objective_v2_parity.py` | trainer_v2 vectorized_nll ≡ legacy per-spectrum loop (v2 layer) | passes to ~1e-9 | (legacy) | pre-PR |
| `test_objective_v2_jitter.py` | Jitter keyword stability for ill-conditioned matrices | passes | (legacy) | pre-PR |
| `test_normalize_by_rest_median.py` | Preprocessing pipeline regression | passes | (legacy) | pre-PR |
| `test_train_gp_end_to_end.py` | v2 stack on synthetic preload schema | passes | (legacy) | pre-PR |
| `test_trainer_v2_smoke.py` | v2 stack wiring | passes | (legacy) | pre-PR |
| `test_preload_from_loa_archive.py` | LoaArchive→trainset.h5 adapter (9 sub-tests: schema, z filter, ZWARN filter, exclude_targetids, max_spectra cap with seed, rest-frame interpolation, mask propagation, downstream load_preprocessed_h5 compat, HCD NHI threshold) | **9/9 pass** | C / commits `0f3b643` + `30ca742` | 2026-05-11 |

## 2. Validation experiments (Step A → B)

Larger experiments comparing the new code to MATLAB / v1 / earlier
endpoints. Each row links to the finding doc with full numbers.

| Experiment | What | Result | Source | Date |
|---|---|---|---|---|
| **A.5 DR16 89k×200 vec full** | Phase 2 DR16 retrain on full 89408 train_ind, vectorized path | 8h03m wall, c_0=0.106 / τ_0=0.00449 / β=3.026, \|Δβ\| vs MATLAB = 2.13 (Adam-vs-L-BFGS) | `docs/notes/2026-05-09_phase2_vec_full_vs_matlab.md`, commit `c8116e3` | 2026-05-09 |
| **A.5 DR16 89k×200 per-spec full** | Same problem, per-spectrum loop reference | 21h31m wall, scalars match vec to ~3 sig figs | commit (vec/per-spec comparison series) | 2026-05-10 |
| **A.5 vec smoke 5k×50 vs Phase-1 baseline** | Vec 5k smoke vs per-spec 5k Phase-1 | scalars match, **28× speedup** | `docs/notes/2026-05-09_vec_smoke_vs_phase1_baseline.md` | 2026-05-09 |
| **B kernel comparison vec vs per-spec full** | M·M^T agreement at 89k×200 production scale | **1.7% Frobenius**; corr matrix to **0.95% Frobenius** | `docs/notes/2026-05-11_vec_vs_perspec_full_comparison.md`, commits `1a67b5b` + `e01ca54` | 2026-05-11 |
| **B inference consistency on canonical TID** | DLAHolder end-to-end on TID 120046865 with each model | p_DLA Δ 2.9e-3, log-evidence Δ ~0.05 nats — same qualitative verdict | `docs/notes/2026-05-11_vec_vs_perspec_inference_consistency.md`, commit `74907db` | 2026-05-11 |
| **C smoke 5k×50 on 2lpt loa-0 wide** | DESI trainer GPU smoke (validates GPU + per-iter rate) | 0.43 s/iter on A40, all 3 outputs (`.h5` + `.npz` + `README.md`) | `docs/notes/2026-05-11_desi_smoke/README.md`, SLURM 49913952 | 2026-05-11 |

## 3. Performance benchmarks

Wall-time + per-iter rate on production-scale problems. Use these for
walltime estimates and trainer-trainer comparisons.

### 3a. Trainer per-iter rate

Per-iter rate is the most stable measure (independent of total iter count).
"Eff. work per iter" = n_spectra × n_pix × k as a rough scaling factor.

| Trainer | Path | Hardware | n_spectra × n_pix × k | Per-iter rate | 1500-iter wall (extrapolated to 300k×5662) | Source / Date |
|---|---|---|---|---|---|---|
| **v1 per-spec** (`tests/short_retrain_2lpt.py`) | per-spectrum Python loop, hand-coded grad | CPU OMP=1 | 89k × 2281 × 20 (DR16) | **~387 s/iter** (measured) | extrapolated **~44 days** | DR16 49709974 / 2026-05-10 |
| **vec_v3** (`tests/phase2_train_dr16.py`) | vectorized chunked, hand-coded grad | CPU OMP=4 | 89k × 2281 × 20 (DR16) | **144 s/iter** (measured) | extrapolated **~17 days** | DR16 49700040 / 2026-05-09 |
| **vec_v3** (`tests/phase2_train_desi.py`, chunk=5000) | same as above + GPU support | A40 (44 GiB) | 236k × 5662 × 30 (DESI 2lpt loa-0) | **20 s/iter** (measured) | **~8 h** | SLURM 49921626 / 2026-05-11 |
| **vec_v3** (DESI, chunk=10000 — next default) | same | A40 | 236k × 5662 × 30 | ~10 s/iter (projected) | **~4 h** (projected) | next submission / 2026-05-11 |
| **trainer_v2** (`gpy_dla_detection/training/trainer_v2.py`) | autograd, randn-init M (BROKEN model) | A100 MIG 40GB | 118k × 3801 × 30 | **3.2 s/iter** (measured) | **~1.3 h** on its scale | jobs 49243842-49268620 / 2026-05-04 |

### 3b. Speedup ratios (apples-to-apples)

| Comparison | Speedup | Notes |
|---|---|---|
| vec_v3 GPU (chunk=5000) vs v1 per-spec CPU | **~130×** at production scale | extrapolated from DR16 measurements scaled to 300k×5662 |
| vec_v3 GPU (chunk=5000) vs vec_v3 CPU (vec_v3 same path) | **~50×** | pure GPU vs CPU win |
| vec_v3 GPU (chunk=10000 next) vs chunk=5000 | ~2× (projected) | better amortization of GPU kernel launches |
| trainer_v2 GPU vs vec_v3 GPU | ~3× faster (trainer_v2) | but trainer_v2 has the randn-init + autograd regression — speed vs correctness tradeoff |
| vec_v3 vs per-spec same path (DR16 89k×200) | **3.05× wall-time** | both correctness-equivalent (parity tests pass to 1e-10) |

### 3c. Storage / data sizes

| Asset | Size | Path |
|---|---|---|
| LoaArchive (compressed observed-frame coadds, 928k QSOs) | 75 GB | `/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5` |
| 2lpt loa-0 v2 wide preload (300k × 5662) | 9.5 GB | `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_wide_v2_*/trainset.h5` |
| 2lpt loa-124 nohcd-nobal v2 wide preload | 8.9 GB | `…/v2_runs/2lpt_loa124_nohcd_nobal_wide_v2_*/trainset.h5` |
| Legacy LOA trainset (narrow grid, [850.75, 1420.75]) | 5.4 GB | `/nfs/turbo/.../GP_trained/loa_no_dla_no_bal_52198069/trainset.h5` (don't use for new training — narrower grid) |
| MATLAB DR16 reference (`learned_qso_model_*.mat`) | ~few MB | `/home/mfho/MATLAB/gp_dla_detection_dr16q_public/data/dr16/MATLAB_Catalogue/` |
| Trained GP model (DESI schema) | ~1.5 MB | per-run `phase2_result.h5` |

## 4. SLURM job ledger (Step C)

| JobID | What | Hardware | State | Notes |
|---|---|---|---|---|
| 49913952 | Step C smoke (5k×50 on 2lpt loa-0 wide) | A40 spgpu | COMPLETED 7m | 0.43 s/iter validated GPU path |
| 49916028 | 2lpt loa-0 wide, 1500 iter, chunk=5000 | A40 spgpu | OOM (host RAM) | superseded by 49921626 (f32 fix) |
| 49916029 | 2lpt loa-124 wide, 1500 iter, chunk=5000 | A40 spgpu | OOM (host RAM) | superseded by 49921627 (f32 fix) |
| 49921626 | 2lpt loa-0 retrain (after f32 fix) | A40 spgpu | **COMPLETED 8h55m** | iter 1499 final: τ_0=0.000540, β=1.279, c_0=0.003964, loss=7.78e8 |
| 49921627 | 2lpt loa-124 retrain (after f32 fix) | A40 spgpu | **COMPLETED 7h37m** | iter 1499 final: τ_0=0.000694, β=1.451, c_0=0.006008, loss=6.28e8 |
| 49925097 | LoaArchive adapter (no-DLA + no-BAL) | standard CPU | FAILED (sys.path) | superseded by 49927767 |
| 49925098 | LoaArchive adapter (no-HCD + with-BAL) | standard CPU | FAILED (sys.path) | superseded by 49927768 |
| 49927767 | LoaArchive adapter (no-DLA + no-BAL), unchunked code | standard CPU | OOM at 2h21m | superseded by 49939506 (chunked-read fix) |
| 49927768 | LoaArchive adapter (no-HCD + with-BAL), unchunked code | standard CPU | OOM at 1h29m | superseded by 49936991 (chunked-read fix) |
| 49936991 | LoaArchive adapter (no-HCD + with-BAL), chunked-read fix | standard CPU | **COMPLETED 50.7 min** | 577,392 spectra → 36.6 GB trainset.h5 at `/scratch/.../loa_wide_v2/loa_no_hcd_with_bal_wide/` |
| 49939506 | LoaArchive adapter (no-DLA + no-BAL), chunked-read fix | standard CPU | **COMPLETED 53.8 min** | 639,419 spectra → 40.5 GB trainset.h5 at `/scratch/.../loa_wide_v2/loa_no_dla_no_bal_wide/` |
| 49947724 | LOA real (no-DLA + no-BAL, 638k × 1500 iter, chunk=10000) | A40 spgpu | OOM at 1m54s | host RAM at load_preprocessed_h5 (f64 preproc on 638k×5663 = 29 GB×N arrays, exceeded 96G); superseded by 49949799 |
| 49947725 | LOA real (no-HCD + with-BAL, 576k × 1500 iter, chunk=10000) | A40 spgpu | OOM at 1m54s | same — superseded by 49949800 |
| 49949799 | LOA real (no-DLA + no-BAL) — `working_dtype=f32` preproc + mem 192G + chunk=10000 | A40 spgpu | **GPU OOM at iter 0** | matmul C@M needed 6.3 GB on top of 42.7 GB used → 44 GB capacity exceeded; superseded by 49977782 (chunk=7500) |
| 49949800 | LOA real (no-HCD + with-BAL) — same | A40 spgpu | CANCELLED preemptively | would have hit same OOM; superseded by 49977783 |
| 49977782 | LOA real (no-DLA + no-BAL) — chunk=7500 | A40 spgpu | submitted | 638k spectra × 1500 iter ~23h; uses --max-walltime-sec=41000 + walltime-exit checkpoint, may need 2-3 chained jobs via --resume |
| 49977783 | LOA real (no-HCD + with-BAL) — chunk=7500 | A40 spgpu | submitted | 575k spectra × 1500 iter ~21h; same checkpoint+resume strategy |

When a job lands, replace its "RUNNING" / "PENDING" with "COMPLETED + outcome" or "FAILED + reason".

## 5. References

- `docs/training_overview.md` — every GP-training file in the repo with status
- `docs/notes/` — finding docs with full evidence (figures, JSONs)
- PR #6 — debugging PR rebuilding the trainer from v1 reference
- `MEMORY.md` — durable cross-session notes (in `~/.claude/projects/.../memory/`)
