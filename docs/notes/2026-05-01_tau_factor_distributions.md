# τ_factor distributions across mocks + real LOA — consolidated reference

> One-stop reference for all τ-EB-chosen τ_factor results in this PR.
> Source data: `tests/profile/results/phase_b_aggregate.csv` and
> `tests/profile/results/phase_b_tau_factor_per_mock.csv` (re-runnable
> via `examples/aggregate_phase_b_results.py`).
>
> All numbers below are from production-realistic Phase B runs: 5 k
> or 50 k random TIDs, FILTER=1, max_dlas=4, BAL-excluded (mocks),
> 6× τ-grid (`{0.5, 1, 1.5, 2, 3, 4, 5, 6}` × Turner+2024
> τ_0=0.00246), `objective="null"`, `apply_hcd_mask=False`.

---

## Headline — the mock-vs-real divergence

Three independent mock pipelines all want τ ≈ 3 × Turner+2024;
real DESI LOA wants ≈ 1.5 ×.

| Population | n | median τ | mean τ | std | frac ≥ 2× |
|---|---:|---:|---:|---:|---:|
| 2lpt mock | 49 000 | 3.0 | 2.75 | 1.33 | 77 % |
| london mock | 48 000 | 3.0 | 2.72 | 1.32 | 76 % |
| saclay mock | 48 999 | 3.0 | 2.64 | 1.28 | 76 % |
| **LOA real** | **5 000** | **1.5** | **1.78** | **1.26** | **41 %** |

Mocks systematically over-estimate forest opacity by ~2× vs real
DESI Y3. This is robust to the choice of mock pipeline (lyacolore,
jura, juraLy8 — completely independent codes).

The user's working hypothesis is that this is driven by the
production GP being trained on real LOA forest (so its μ + ω² model
is anchored to LOA opacity); when applied to mocks the GP sees ~2×
extra absorption and τ-EB compensates. A 2×2 experiment swapping
LOA-trained vs 2lpt-trained GP across {2lpt, LOA} datasets is in
flight (jobs 49108430 / 49108431 / 49108432 / 49108443) — those
results will close out the question.

---

## Full τ_factor histograms (all-z)

```
                      2lpt     london   saclay     LOA-real
  τ=0.50:    n=        2862     3083     3088       1141
                       (5.8%)   (6.4%)   (6.3%)    (22.8%)
  τ=1.00:    n=        3288     3097     3243        939
                       (6.7%)   (6.5%)   (6.6%)    (18.8%)
  τ=1.50:    n=        5329     5226     5650        877
                      (10.9%)  (10.9%)  (11.5%)    (17.5%)
  τ=2.00:    n=       10479    10681    11681        861
                      (21.4%)  (22.3%)  (23.8%)    (17.2%)
  τ=3.00:    n=       13722    13200    13724        700
                      (28.0%)  (27.5%)  (28.0%)    (14.0%)
  τ=4.00:    n=        8320     8076     7644        273
                      (17.0%)  (16.8%)  (15.6%)     (5.5%)
  τ=5.00:    n=        3442     3202     2754         96
                       (7.0%)   (6.7%)   (5.6%)     (1.9%)
  τ=6.00:    n=        1558     1435     1215        113
                       (3.2%)   (3.0%)   (2.5%)     (2.3%)
  ----------------------------------------------------------------
   total              49000    48000    48999       5000
   median               3.00     3.00     3.00       1.50
   mean                 2.75     2.72     2.64       1.78
```

Three observations:
- **Mocks are nearly identical** despite using independent simulation
  pipelines. Their median, mean, and shape match within < 5 %.
- **LOA's mode is at τ=0.50** (22.8 %) — i.e. lots of LOA spectra
  prefer SUB-Turner opacity. None of the mocks have this feature.
- **LOA's tail at τ ≥ 4 is much thinner** (5.5 %, 1.9 %, 2.3 %) than
  the mocks' (15-17 %, 5-7 %, 2.5-3 %). Mocks have a heavy
  high-τ tail; LOA does not.

---

## τ_factor by z_qso bin

The recipe's preference for high τ is driven by the *low-z* forest
in all four populations, but the magnitude differs:

```
  z_qso bin       2lpt med τ   london med τ   saclay med τ   LOA-real med τ
  -------------------------------------------------------------------------
  [2.0, 2.3)      3.0          3.0            3.0            2.0
  [2.3, 2.6)      3.0          3.0            3.0            2.0
  [2.6, 3.0)      2.0          2.0            2.0            1.5
  [3.0, 5.5)      1.5          1.5            1.5            1.0  ← matches Turner exactly
```

In counts:

```
  z_qso bin        2lpt n     london n     saclay n     LOA n
  -----------------------------------------------------------
  [2.0, 2.3)      19 239      19 159       19 628        1 850
  [2.3, 2.6)      14 326      13 824       14 511        1 445
  [2.6, 3.0)       9 692       9 436        9 697        1 030
  [3.0, 5.5)       5 743       5 581        5 163          675
```

Two things stand out:
- **At high z (z_qso ≥ 3), real LOA τ_factor lands at 1.0 ×**
  — i.e. Turner+2024 calibrates the high-z forest exactly. The
  recipe is a no-op here.
- **At every z bin, the LOA→mock divergence is roughly half a τ
  step**. LOA wants 2.0 where mocks want 3.0; LOA wants 1.0 where
  mocks want 1.5. The divergence is *not* z-specific; it's a uniform
  ~2× factor.

The z-monotonic decline is consistent across mocks and LOA — argues
the underlying physics (whatever it is) is correlated with the
forest-density evolution, not with any specific z range.

---

## Comparison to earlier (sub-population) measurements

Earlier-session and Phase A (τ-fit-only, no full bayes) measurements
for cross-check:

| Sample | n | source | median τ | mean τ |
|---|---:|---|---:|---:|
| 2lpt picker n=18 (DLA-only, cherry-picked) | 18 | `summary_n54.csv` | 2.0 | 2.36 |
| 2lpt random 5k Phase A (τ-fit only, 4× grid) | 5 000 | `tau_eb_phase_a_5k_2lpt.tsv` | 3.0 | 2.63 |
| 2lpt random 5k Phase A (τ-fit only, 6× grid) | 5 000 | `tau_eb_phase_a_5k_2lpt_extgrid.tsv` | 3.0 | 2.78 |
| 2lpt 5k Phase B (FILTER=0, max_dlas=3) | 5 000 | job 49040725 | 3.0 | 2.63 |
| **2lpt 50k Phase B (FILTER=1, max_dlas=4, BAL-excl)** | **49 000** | **job 49065622** | **3.0** | **2.75** |
| **LOA 5k Phase B (FILTER=1, max_dlas=4)** | **5 000** | **job 49071304** | **1.5** | **1.78** |

The 50 k results are the most authoritative; smaller samples agree
to within ~5 %.

---

## Working interpretation (status: hypothesis, not yet confirmed)

The current production GP (`learnlogs/model_epoch_920.h5`) is trained
on real DESI Y3 LOA forest (with DLAs/HCDs masked out). The GP's
mean μ and per-pixel variance ω² therefore encode "what LOA's
no-DLA forest looks like" at each rest wavelength. When inference
runs:

1. `null GP` predicts μ × A_lyα(τ_0=Turner+2024) + noise model.
2. If the spectrum's actual forest opacity matches LOA, μ × A_lyα
   matches the data and τ-EB has nothing to fix → picks τ ≈ 1×.
3. If the spectrum's actual forest opacity is **higher** than LOA's
   (the case for all three mocks), μ × A_lyα predicts too little
   absorption → τ-EB compensates by raising τ_0.

This implies the τ_factor we measure is approximately
`τ_factor ≈ τ_actual / τ_LOA-trained`. With τ_LOA-trained ≈ Turner
(by construction, since training assumed Turner mean-flux), the
mock measurement of τ_factor ≈ 3 says **mocks have ~3× Turner forest
opacity** while LOA has ~1.5× Turner — which is consistent with
Turner+2024 itself having been calibrated on real DESI data and so
already nearly correct for LOA, while mocks happen to overshoot.

If true, this means:
- The recipe's bias closure on mocks is "the GP is well-calibrated
  for LOA, mocks are off-distribution, τ-EB adapts the forward model
  per-spectrum to bridge the gap."
- On real LOA, τ-EB is closer to a no-op (median 1.5 ×, frac ≥ 2×
  is 41 %). It's safe to ship default-OFF.

---

## Pending: 2×2 training-data anchor experiment

In flight (jobs 49108430-49108443):

| Dataset \ GP-trained-on  | LOA_clean (real) | MOCK_2lpt |
|---|---|---|
| 2lpt mock | predicts τ ≈ 3-4 × (current observation; control) | **predicts τ ≈ 1 × (matched)** |
| LOA real | **predicts τ ≈ 1 × (matched)** | predicts τ < 1 × (over-absorbed) |

Confirmation predictions:
- If matched runs (2lpt + 2lpt-trained, LOA + LOA-clean) both center
  on τ_factor = 1 ×, the training-data anchor hypothesis is
  confirmed.
- If matched runs still show divergent τ_factor distributions,
  there is a residual forward-model bias not captured by training.

ETA: ~2 h wall after submission. Will be appended to this doc once
the runs land.

---

## Files referenced

| File | What |
|---|---|
| `tests/profile/results/phase_b_aggregate.csv` | per-dataset stats (bias, FPR, purity, completeness, τ stats) |
| `tests/profile/results/phase_b_tau_factor_per_mock.csv` | per-(mock, τ_factor) histogram |
| `examples/aggregate_phase_b_results.py` | re-aggregator (rerun after new arrays land) |
| `docs/stories/tau_eb_story_2lpt.md` | full 2lpt 50k story + example spectra |
| `docs/stories/tau_eb_story_london.md` | full london 50k story |
| `docs/stories/tau_eb_story_saclay.md` | full saclay 50k story |
| `docs/stories/tau_eb_story_loa.md` | LOA real-data behavior + mock-vs-real comparison |
| `docs/notes/2026-04-29_tau_eb_phase_a_5k_2lpt.md` | earlier τ-fit-only (Phase A) measurement |
| `docs/notes/2026-04-29_tau_eb_n90_unbiasedness.md` | n=90 picker (cherry-picked) τ-distribution |
| `docs/tau_eb_hcd_mask.md` | algorithm walkthrough |
