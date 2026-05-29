# Hybrid DLA detection on London 8f — Method A (null-quantile Δ_marg) + Method B (MAP over [17, 22])

**Date**: 2026-05-12 · prod533_5k_20260511 · London mock-0 (8 healpix files) · v3 `2lpt_loa124_nohcd_nobal_wide` GP model · per-spec hybrid catalog · n_initial=5000

This document is the headline write-up; artifacts in this directory:

```
method_a_per_spec.fits           per-spec table with truth annotations
method_a_summary.json            Method A null quantiles + sweep table
method_b_all.json                Method B (per-spec MAP) raw rows (3345 SNR>2 specs)
combined_summary.json            Hybrid sweep (9 A x B cells x veto on/off) + bookkeeping
dlacat_v3_loa124_combined.fits   THE catalog (458 rows; tid, z_qso, snr_red,
                                  z_dla_marg, log_nhi_marg, p_dla_marginal, delta_marg,
                                  z_dla_map, log_nhi_map, map_log_lr, lyb_flagged, method)
nsweep/nsweep_summary.json       n_initial sweep (5k/10k/50k)
nsweep/nsweep_per_target.tsv     per-target n_initial sweep data
nsweep/nsweep_table.tsv          compact n_initial decision table
figures/method_a_null_hist.png   Delta_marg histogram + p90/p95/p99 lines
figures/method_b_null_hist.png   log_LR distribution + A-vs-B scatter
logs/                            step1/2/3/4 stdout
raw/                             per-h5 Method B JSON (1 per worker)
```

---

## 1. n_initial decision: 5000

Sweep on 33-target calibration set (3 missed-cand TIDs in v3 + 10 strong_truth +
20 SNR>2 nulls). The MAP optimizer outputs (z_MAP, logN_MAP, log_LR) are
**identical to 4+ significant figures across n_initial ∈ {5000, 10000, 50000}**
on every single cell.

| n_initial | mean total (s/spec) | mean QMC (s) | mean opt (s) | missed_recov 3 | strong_recov 10 | null median log_LR |
|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 3.00 | 2.19 | 0.085 | 2 | 8 | 12.31 |
| 10000 | 4.89 | 4.37 | 0.079 | 2 | 8 | 12.31 |
| 50000 | 22.36 | 21.89 | 0.074 | 2 | 8 | 12.31 |

**Decision**: n_initial = 5000. 7x faster than 50000, scientifically indistinguishable.

### 5 prompt-listed missed candidates — Method B verdict (3 in v3 catalog scope)

2 of 5 (TIDs 80198262, 20115135) are in healpix files NOT in the v3 8-file processed set.

| TID | z_truth | logN_truth | z_MAP | logN_MAP | log_LR | Recovered? |
|---:|---:|---:|---:|---:|---:|---:|
| 105798 | 1.984 | 20.32 | 2.312 | 18.79 | +9.16 | **No** — logN<19, classified b_subdla_lls |
| 1798 | 2.048 | 20.54 | 2.045 | 20.66 | +23.30 | **Yes** |
| 64988 | 1.990 | 20.41 | 1.988 | 20.25 | +18.48 | **Yes** |

2 of 3 in-scope missed cands recovered. TID 105798's MAP finds a competing sub-DLA-strength peak rather than the truth peak.

---

## 2. Method A: null-quantile threshold

Δ_marg = log p(D|1 DLA) - log p(D|null) for the v3_loa124 prior-marginal evidence.

**Null anti-join**: row has NO truth absorber with NHI >= 17.2 in window. BAL excluded.

### Null quantile values (SNR > 2, BAL-excl, n_null=1683)

| Quantile | Δ_marg threshold |
|---:|---:|
| p90 | **6.306** |
| p95 | **15.862** |
| p99 | **35.701** |
| p99.9 | 76.091 |

### Method A per-spec sweep (SNR > 2, BAL-excl, MAP NHI >= 20.3)

Per-spec evaluation: spec is detected if (delta > thr) AND (MAP_LOGNHI_MARG >= 20.3).
Strict TP: matched truth NHI >= 20.3 within |Δz|/(1+z_truth) <= 0.01.

| Threshold | n_det | n_TP | Purity | Completeness |
|---:|---:|---:|---:|---:|
| Δ > p90 (6.31) | 255 | 208 | **81.57%** | **67.31%** |
| Δ > p95 (15.86) | 227 | 193 | **85.02%** | **62.46%** |
| Δ > p99 (35.70) | 179 | 156 | **87.15%** | **50.49%** |
| P_DLA > 0.99 | 229 | 196 | 85.59% | 63.43% |
| P_DLA > 0.999 | 214 | 187 | 87.38% | 60.52% |
| P_DLA > 0.99999 | 189 | 173 | 91.53% | 55.99% |

**Observation**: null-quantile p95 is statistically indistinguishable from P_DLA>0.99
on per-spec eval. The null-quantile recipe doesn't unlock new completeness on its own.

---

## 3. Method B: MAP detection with log NHI ∈ [17, 22]

For each SNR > 2 spec, QMC scan over `pw_samples_a3_172_220_50000.mat` (first 5000),
top-K=3 starts, Nelder-Mead optimize on (z, logNHI) with bounds (search_window, [17, 22]).
Classify:
- logN_MAP < 19 → **b_subdla_lls** (not a DLA detection)
- logN_MAP >= 19 AND log_LR > tau_LR → **B-detected**

### Method B null distribution (SNR > 2, BAL-excl, no truth NHI>=17.2 in window)

| | All MAPs (n=1683) | MAP NHI>=19 (n=377) |
|:---|---:|---:|
| p50 | 12.07 | 14.03 |
| p90 | 32.86 | 45.21 |
| p95 | 42.68 | **65.29** |
| p99 | 70.23 | **93.21** |

The **NHI>=19 nulls have a much higher log_LR distribution** than all nulls combined — the [17,22] optimizer over-fits on noise / truth-catalog-miss absorbers at logN ∈ [20.5, 21.0]. p95 = +65 is a very high bar.

### Method B sweep with NHI>=19 null quantiles

| τ_LR | n_det | n_TP | Purity | Completeness |
|---:|---:|---:|---:|---:|
| p90 (45.2) | 384 | 78 | 20.31% | 25.24% |
| p95 (65.3) | 234 | 50 | 21.37% | 16.18% |
| p99 (93.2) | 134 | 39 | 29.10% | 12.62% |

Method B alone is unusable as a detection score — ~20-30% purity at all thresholds (consistent with the HANDOFF's "30% null FP rate" note for MAP-based detection).

---

## 4. Hybrid catalog: A ∪ B + Lyβ veto

Per-spec attribution (catalog has 458 rows from the headline A=p95, B=p95 cell):

| method | count |
|:---|---:|
| A_only | 57 |
| A_and_B | 170 |
| B_only | 231 |
| B_filtered_lyb | 0 (only 2 specs Lyβ-flagged, neither was B-detected at p95) |

Lyβ within-spec self-flag: 2 rows (A vs B z-disagreement consistent with Lyβ apparent z).

### Full headline sweep (SNR > 2, BAL-excl, per-spec)

truth_denom (specs with NHI>=20.3 truth in window) = **309**

| A thr | B thr | kind | veto | n_det | n_TP | P% | C% |
|---|---|---|---|---:|---:|---:|---:|
| p90 | p90 | A_and_B | veto | 214 | 184 | 85.98 | 59.55 |
| p90 | p90 | **A_or_B** | veto | 603 | 263 | 43.62 | **85.11** |
| p95 | p90 | A_and_B | veto | 214 | 184 | 85.98 | 59.55 |
| p95 | p90 | A_or_B | veto | 575 | 247 | 42.96 | 79.94 |
| p95 | p95 | A_and_B | veto | 170 | 151 | **88.82** | 48.87 |
| p95 | p95 | A_or_B | veto | 458 | 245 | 53.49 | 79.29 |
| p95 | p99 | A_and_B | veto | 119 | 110 | **92.44** | 35.60 |
| p95 | p99 | A_or_B | veto | 361 | 235 | 65.10 | 76.05 |
| p99 | p99 | A_and_B | veto | 119 | 110 | 92.44 | 35.60 |
| baseline | n/a | P_DLA>0.99 | n/a | 229 | 196 | 85.59 | 63.43 |
| baseline | n/a | P_DLA>0.99999 | n/a | 189 | 173 | 91.53 | 55.99 |

**Headline answer to "does v3 + A∪B hit 85/85 at SNR > 2 (per-spec)?": NO.**

- **A∪B at p90/p90** reaches 85% completeness but only 44% purity — Method B's null tail dominates.
- **A∩B (A_and_B) at p95/p99** is the high-P regime: P=92%, C=36%.
- **No A∪B cell achieves >75% purity AND >75% completeness on per-spec eval.**

The fundamental trade-off: every Method B "null candidate" with logN ∈ [20.5, 21]
that lacks a truth-cat match is counted as FP. Either:
- a real over-fit by the 1-DLA model on forest noise → genuine FP;
- a truth-catalog miss (London dla_cat.fits is incomplete at NHI ~ 20 boundary).

---

## 5. Lyβ veto

Within-spec self-flag (since per-spec catalog has at most 1 A + 1 B candidate):
if A.z > B.z AND A.logN > B.logN AND |B.z - Lyβ_apparent(A.z)| ≤ 0.005, flag B.

**2 spectra flagged.** Neither was B-detected at p95, so the veto column showed
no impact on the headline (n_det differs by ≤1 between veto/noveto rows in the
sweep).

In a multi-DLA hybrid catalog the Lyβ veto would have more bite. The
within-spec single-DLA framing limits its scope here.

---

## 6. Per-spec vs per-DLA P/C — important caveat

The HANDOFF reports **v3_loa124 at P_DLA > 0.99**: P=84.52%, C=76.61% — these are
**per-DLA** numbers from `examples/molly_faithful_pc_plots.py` on the `dlacat-*.fits`
output (which has multi-DLA rows, MAX_DLAS=3). Each truth DLA is matched once per
detection candidate at any z, allowing multi-DLA spectra to contribute multiple TP/FN.

This document's **per-spec** numbers are structurally lower in completeness because:
- Multi-DLA truth spectra count ONCE in the denominator.
- The per-spec hybrid catalog proposes at most 1 detection per spec.
- ~13% of London truth-DLA spectra have multiple DLAs → ceiling on per-spec C is below 100%.

To compare apples-to-apples with the molly headline, run the molly script on this
catalog after expanding A_only/A_and_B/B_only into per-DLA rows.

---

## 7. Surprises / caveats

1. **n_initial = 5000 produces IDENTICAL results to 50000** at 7x lower cost.
   Output (z_MAP, logN_MAP, log_LR) matches to 4+ sig figs on all 33 calibration
   targets. Don't use 50000 in production.

2. **Method B has a huge null log_LR tail at MAP NHI ≥ 19**: p95(null, NHI>=19) = +65
   on the v3 model. The optimizer routinely finds [20.5, 21.0] absorbers on SNR>2
   spectra with no truth match in the window. These split into:
   - Genuine over-fits (forest noise gets a wide weak Voigt) — real FPs
   - Truth-catalog misses (the London dla_cat does not include every absorber)
   It is impossible to distinguish these from each other without truth-augmentation.

3. **Hybrid A∪B at p90/p90 reaches 85% C** but at 44% P. The completeness gain
   comes from B catching the Method-A-missed strong DLAs (e.g., 2 of the 3
   in-scope HANDOFF missed cands), but the cost is hundreds of B-only FPs.

4. **Hybrid A∩B (A_and_B) at p95/p99 is the high-purity regime**: P=92%, C=36%.
   The intersection mode filters out most of Method B's null tail because the
   prior-marginal Δ_marg never flags those nulls (their Δ_marg < 0).

5. **2/5 missed candidates are out of scope** (TIDs 80198262, 20115135 not in v3
   8-file set). Of the 3 in scope, B recovers 2 — same as the 5-cand n_sweep result.

6. **Lyβ veto in single-DLA framing has minimal impact** (2 spectra flagged total,
   neither in the p95 B-detected set). Would matter more in multi-DLA hybrid.

7. **No A∪B cell achieves both >85% P and >85% C.** The 85/85 target is not
   reachable with per-spec hybrid on London 8f via these two methods.

---

## 8. Reproduce

```bash
# Step 1 — Method A (Δ_marg quantiles + per-spec table)
python step1_method_a.py     # ~5s

# Step 2 — n_initial sweep on calibration set
python step2_nsweep.py       # ~16 min on 33 targets x 3 n_initials

# Step 3 — Method B optimizer over all SNR>2 specs, 8 workers
python step3_method_b.py --n-initial 5000 --k-starts 3 --logn-min 17.0 --logn-max 22.0 --snr-min 2.0 --workers 8
# ~30 min wall on Perlmutter

# Step 4 — Combine, Lyβ veto, FITS + eval
python step4_combine_eval.py # ~5s
```

All four steps run from a clean DESI subshell:
```bash
bash -c '
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
python <script>'
```
