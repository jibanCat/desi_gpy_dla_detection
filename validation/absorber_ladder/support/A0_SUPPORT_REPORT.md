# A0 — SUPPORT CONTRACT + COLLAR-MATCHED TRUTH / CENSUS REBUILD

**Task:** PI ruling `governance/PI_RULING_2026-09-13b_ABSORBER_LADDER_GO_CUT_FEEDBACK.md` §3
(A0, mandatory) — *"Rebuild mock truth on the same collar support as counts/path; rebuild
ORACLE/census on that support … Support consistency becomes a machine-enforced invariant:
a support_id such that support_id(N_truth) = support_id(dX) = support_id(counts) =
support_id(FP census), fail-closed."*

**Status:** A0 **and A0v2** built and gated for all three mock families; the row-level
support invariant holds across pack / truth / census / operators. **Validation-only; no
sampler ran; no tracked file was modified; nothing committed.**

| | |
|---|---|
| Worktree | `/home/mfho/wt_abs_diag_2026-09` (branch `absorber-diag-2026-09`) |
| New code | `validation/absorber_ladder/support/{support_contract.py, build_a0_support.py, rebuild_a0_products.py, verify_a0_products.py}`, `tests/test_support_contract.py` |
| Products | `/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/support/` — 47 files + `SHA256SUMS` |
| Envs | builds/tests `gpdla`; `load_pack` / `run_ladder` acceptance `gpdla-hbi` |
| Wall time | A0 26.4 / 24.6 / 24.4 s, A0v2 41.5 / 37.8 / 39.4 s per family, + 22 s verification — login node, no SLURM |
| Tests | `tests/test_support_contract.py` **50 passed**; `tests/test_absorber_diag.py` + `tests/test_fp_census_grid.py` still pass (82 total) |

> **Follow-up round (A0v2) folded in.** §5.1 closes support-mismatch instance #8 on the
> validation side; §2 records the commander's adjudication on the gate level. The row-level
> invariant now holds across **pack `counts` / `dX` / A0 `truth_counts` / census / operators**
> for all three families.

---

## 1. The defect, restated exactly

`CDDF_analysis/hbi_mcmc/build_scan_packs.py` rebuilds **`counts`, `dX`,
`dX_coarse_committed`, `fp_E_alloc`** at collar `3000 + b` km/s and copies *every other
array byte-identically* from the adopted (collar-3000) pack. Verified here key-by-key: in
`scanpack_<fam>_b300.npz` the arrays `truth_counts` and `truth_counts_bks` are
**byte-identical** to `adopted_packs_v2p2_20260821/modelA_pack_<fam>_bw0p2_pad19p0_molly172_v2.npz`
(`scanpack_truth_is_byte_identical_to_adopted = {truth_counts: true, truth_counts_bks: true}`,
all three families). So the packs of record divide a **collar-3000 numerator** by a
**collar-3300 denominator/exposure** — recurring bug class
`feedback_one_sided_support_bug_class`, instance #7.

The FP census (`validation/fp_ladder/build_fp_census.py`) was **gated against the adopted
collar-3000 pack**, not the pack of record: its own provenance records
`pack_gates[0].pack = …/adopted_packs_v2p2_20260821/…npz` (EQUAL, 88071 = 88071) and
`pack_gates[1] = pack_of_record_informational` (87844 vs 88071, **EQUAL: false**, 280 cells
differing). So `run_ladder.py:68`'s ORACLE `mu_FP` pin was a collar-3000 census against
collar-3300 counts.

---

## 2. The support contract (`support_contract.py`)

A **closed, 12-field schema**; `support_id(**fields)` returns a canonical JSON string
(sorted keys, `repr(float)` round-trip, schema version embedded) plus its `sha256`.

| # | field | meaning | in the PI's list |
|---|---|---|---|
| 1 | `collar_kms` | proximity collar on BOTH window edges | yes |
| 2 | `snr_min` | `S2N_RED > snr_min` (**strict >**) | yes |
| 3 | `z_window` | `(z_qso_min, z_qso_max)` | yes |
| 4 | `p_dla_min` | `P_DLA > p_dla_min` (**strict >**) | yes |
| 5 | `lya_only_lam_min` | blue rest-frame edge (Å) | yes |
| 6 | `lam_rf_max` | red rest-frame edge (Å) | **extension** |
| 7 | `z_cut_columns` | which z column(s) the λ/z window is applied to | **extension** |
| 8 | `quality_cut` | detection quality flag rule | **extension** |
| 9 | `truth_host_floor` | truth/host `N_HI` floor, or `'n/a'` | yes |
| 10 | `bal_policy` | BAL veto convention + `bal_cat` sha12 | yes |
| 11 | `catalogue_id` | detection-catalogue identity (sha manifest) | yes |
| 12 | `truth_catalogue_sha256` | truth-catalogue identity | yes |

Extension #7 is **not cosmetic**: it is the field that exposes a second, larger support
mismatch (§5). Extensions #6/#8 are cheap and close two otherwise-silent degrees of freedom.
`molly_tsv` is deliberately **excluded** — the molly matrix is a calibration object, not a row
selection — but it is carried in every provenance sidecar.

**Fail-closed behaviour** (all raise `SupportContractError`, all unit-tested):
unknown field · missing field · `None` · NaN/Inf · blank string · `bool` · empty sequence ·
unsupported type · empty object set · unstamped product · a stamp whose recorded
`support_id` disagrees with the sha256 of its own fields · a stamp from another schema
version · an empty or unknown comparison-field subset · a subset comparison against a
sha-only stamp.

### API delivered

* `support_id(**fields) -> SupportID` with `.canonical`, `.sha256`, `.row_sha256`, `.short`,
  `.subset_sha256(subset)`, `.to_json()`.
* `stamp(npz_path, support, extra=…)` → writes `<product>.support.json` (fields + canonical
  string + sha256 + **the product's own sha256** + UTC + optional `extra.planes`);
  `stamp_path`, `read_stamp(path, required=True)`, `stamp_array(support)`,
  `support_from_json`.
* `assert_same_support({name: SupportID|stamp-dict|sha256}, fields=SUPPORT_FIELDS)` →
  common sha256, else `SupportMismatch` with a per-field diff table naming every differing
  field and every object's value.
* **`check_support_consistency(pack_path, census_path=None, ops_path=None, fields=…)`** —
  the fail-closed gate for the ladder runner to call (wiring left to the caller, as
  instructed). It expands each product's **stamped planes** so `counts`, `dX`,
  `fp_E_alloc`, `truth_counts`, `truth_counts_bks` inside one pack are compared against
  *each other* — that pairing **is** defect #7.
* CLI: `python validation/absorber_ladder/support/support_contract.py --pack P [--census C]
  [--ops O] [--level full|row] [--json]`, exit 0 = one common support, exit 1 = mismatch.

### Gate level — **COMMANDER ADJUDICATION, PI RATIFICATION PENDING**

> **The ladder gate runs at the `row` level (11 fields), with `truth_host_floor`
> reported alongside every plane** — because the floor is a property of *which truth
> objects are used*, not of the row selection. Adjudicated by the commander 2026-09-13;
> **pending PI ratification.**

Two pieces of evidence now support the adjudication rather than merely motivating it:

* under the pack's own observable-only selection (§5.1) the floor is **provably inert on
  the rows**: `counts_all` built at floor 17.2 and `counts_obs` built at floor 19.0 are
  **elementwise identical** for all three families (87 840 / 87 574 / 86 500,
  `IDENTICAL: true`). The 18 / 9 / 18-row perturbation quoted below existed *only* because
  the floor leaked into the window through `Z_TRUE`; with that leak removed it is exactly
  zero;
* the gate never hides the floor: `check_support_consistency` returns a
  `truth_host_floor` map for every plane, and the `row`-level pass line prints it
  (`["17.2", "19.0", "n/a"]`).

The `full` (12-field) level remains available and is still run and recorded alongside;
it fails **only** on `truth_host_floor`, by design.

### The underlying design decision

The ruling's literal chain `support_id(N_truth) = support_id(counts) = support_id(FP census)`
**cannot hold with `truth_host_floor` inside the hash**, because the floor legitimately
differs *by object*: `n/a` (counts/dX/fp_E_alloc), `19.0` (the latent basis truth), `17.2`
(the census). So the module exposes two levels:

* **full** (default, strictest) — all 12 fields;
* **row** (`ROW_SELECTION_FIELDS`, 11 fields) — the row selection, i.e. *which rows of the
  universe* are counted, with `truth_host_floor` carried as a declared per-object attribute
  and always printed in the gate record.

The relaxation is **documented, never silent** (the raised/returned text names the level).
Under the *truth-aware* z cut the floor was not entirely inert on the detection side — it
perturbed the rows by **18 / 9 / 18** (2LPT-0 `counts_all` 88 071 @floor 19.5 → 88 053
@floor 17.2; London-0 87 840 → 87 831; Saclay-0 86 763 → 86 745), i.e. 0.02 %. **Under the
A0v2 observable-only selection that perturbation is exactly zero** (measured, §5.1), which
is what makes the `row` level clean rather than merely convenient.

---

## 3. Exact selection difference, collar 3000 vs collar 3300

Both collars use the *same* geometry — `make_lambda_z_BAL_cuts` /
`build_scan_packs.build_family`, with `coll = collar_kms / c`:

```
z_lo = max(3600/1215.67 - 1 ,  lam_rf_min*(1+z_qso)/1215.67 - 1 + coll)
z_hi = min(z_qso - coll      ,  lam_rf_max*(1+z_qso)/1215.67 - 1 - coll)
keep = (z > z_lo) & (z < z_hi) & (z_qso > 2.0) & (z_qso < 4.25) & ~BAL
```

with `lam_rf_min = 1025.0`, `lam_rf_max = 1216.0` (the `lya_only` window),
`snr_min = 2.0` (strict), `p_dla_min = 0.99` (strict), `quality = DLAFLAG == 0`.
**Nothing but `coll` changes between the two:** 3000 → `coll = 0.0100070`,
3300 → `coll = 0.0110077`; the window narrows by `Δz = 0.0010007` at **both** edges
(`z_lo` up, `z_hi` down), so the 3300 km/s selection is *strictly nested* inside the
3000 km/s one.

Two self-tests confirm the nesting and make "apply the 3300 mask on top of the already-cut
table" exact rather than approximate:

* **idempotence** — applying the collar-3000 mask to the collar-3000-cut truth table drops
  **0** of 203 533 / 214 932 / 210 668 rows (`NOOP: true`);
* **regression** — the `build_scan_packs` counts block, re-run here, reproduces the pack of
  record's `counts` **bit-exactly** at collar 3300 (87 844 / 87 579 / 86 505), so the
  selection this report attributes to the packs of record is provably theirs.

On the truth side `make_lambda_z_BAL_cuts` is called with `use_truth_z=False`, so the truth
histogram's window is applied to `Z_DLA` alone — the same column `build_scan_packs` uses for
`counts`. **The A0 truth is therefore z-column-consistent with the scan-pack counts.**

### Gates passed before anything was written

| gate | 2LPT-0 | London-0 | Saclay-0 |
|---|---|---|---|
| rebuilt `truth_counts_bks` @3000 == adopted pack (bit-exact) | ✅ 101 949 | ✅ 106 705 | ✅ 104 561 |
| rebuilt `truth_counts` @3000 == adopted pack (bit-exact) | ✅ | ✅ | ✅ |
| max abs cell difference | 0.0 | 0.0 | 0.0 |
| all 7 census blocks @3000 == census of record (bit-exact) | ✅ | ✅ | ✅ |

---

## 4. Results

### 4.1 Truth histogram, collar 3000 → 3300

| family | truth @3000 | truth @3300 | removed | **ratio** |
|---|---|---|---|---|
| 2LPT-0 | 101 949 | 101 494 | 455 | **0.995537** |
| London-0 | 106 705 | 106 232 | 473 | **0.995567** |
| Saclay-0 | 104 561 | 104 095 | 466 | **0.995543** |

(Expected ≈ 0.9955 — matches `OPERATOR_FORENSICS_REPORT.md` §7 to 5 decimals.)

**Ratio by coarse K** (`K0 = [2.0,2.5)`, `K1 = [2.5,3.0)`, `K2 = [3.0,3.5)`):

| family | K0 | K1 | K2 | (K0/K1/K2 totals @3000) |
|---|---|---|---|---|
| 2LPT-0 | 0.995414 | 0.995693 | 0.995653 | 54 944 / 33 202 / 13 803 |
| London-0 | 0.995740 | 0.995951 | 0.993975 | 57 518 / 34 580 / 14 607 |
| Saclay-0 | 0.995900 | 0.995389 | 0.994164 | 58 288 / 35 135 / 11 138 |

**Ratio by latent-N bin `b`** (`ntrue_edges` = 19.0, 19.2, 19.5, 19.7, 19.9, 20.1, 20.3,
20.5, 20.7, 20.9, 21.1, 21.3, 21.5, 21.7, 21.9, 22.1, 22.4):

| b (lo edge) | 19.0 | 19.2 | 19.5 | 19.7 | 19.9 | 20.1 | 20.3 | 20.5 | 20.7 | 20.9 | 21.1 | 21.3 | 21.5 | 21.7 | 21.9 | 22.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2LPT-0 | .995905 | .995794 | .995268 | .996008 | .995944 | .995061 | .995269 | .994847 | .996268 | .993859 | .994307 | .997338 | .992453 | .994898 | 1.0 | 1.0 |
| London-0 | .995089 | .995898 | .995568 | .995692 | .995705 | .995306 | .994704 | .995792 | .996822 | .994810 | .996063 | .996057 | .995320 | .991870 | 1.0 | 1.0 |
| Saclay-0 | .995865 | .995792 | .995369 | .995755 | .994772 | .994733 | .996346 | .996671 | .995987 | .995003 | .994009 | .993317 | .996534 | .995475 | .983051 | 1.0 |

**Flat in both axes** (0.992–0.998 over every populated cell; the `≥21.9` outliers are
single-digit-count bins). So the collar defect is uniform in `N` and `z` — it rescales every
mock bias by the same factor and explains neither the K1 concentration nor the zigzag, exactly
as the forensics concluded. Per-fine-`k` ratios are in
`A0_BUILD_SUMMARY.json:<fam>.truth_collar.ratio_by_fine_k`.

### 4.2 The mock estimand this moves

Computed with the **committed** reduction (`forward_selftest.truth_f` →
`model_a.reduce_f_posterior`), of-record pack vs A0 pack, identical `dX`:

| family | truth dN/dX(≥20.0) of record | A0 | truth dN/dX(≥20.3) of record | A0 | ratio ≥20.0 / ≥20.3 |
|---|---|---|---|---|---|
| 2LPT-0 | 0.086604 | **0.086191** | 0.054533 | **0.054268** | 0.995231 / 0.995147 |
| London-0 | 0.093729 | **0.093308** | 0.060025 | **0.059757** | 0.995502 / 0.995530 |
| Saclay-0 | 0.090070 | **0.089661** | 0.056990 | **0.056754** | 0.995453 / 0.995863 |

2LPT-0 `0.086604 → 0.086191` reproduces `OPERATOR_FORENSICS_REPORT.md` §7 **exactly**, from
an independently written path. Applying these truth ratios to the forensics' posterior
medians gives the corrected mock bias table (no sampler was re-run here):

| family | ≥20.0 of record → A0 | ≥20.3 of record → A0 |
|---|---|---|
| 2LPT-0 | +0.99 % → **+1.48 %** | +2.95 % → **+3.45 %** |
| London-0 | +1.19 % → **+1.65 %** | +2.17 % → **+2.63 %** |
| Saclay-0 | +1.42 % → **+1.88 %** | +3.29 % → **+3.71 %** |

The defect **understated** every mock bias by 0.45–0.49 pp.

### 4.3 FP-truth census, collar 3000 → 3300 (all blocks)

| block | 2LPT-0 3000 → 3300 (Δ) | London-0 3000 → 3300 (Δ) | Saclay-0 3000 → 3300 (Δ) |
|---|---|---|---|
| **`hostless`** (ORACLE `mu_FP` pin) | 13 860 → **13 844** (−16) | 9 598 → **9 586** (−12) | 10 592 → **10 574** (−18) |
| `host_17p2_19p0` (A0 sub-floor `mu_extra`) | 3 200 → 3 199 (−1) | 2 611 → 2 611 (−0) | 2 668 → 2 664 (−4) |
| `host_19p0_19p5` | 7 106 → 7 096 (−10) | 6 983 → 6 974 (−9) | 6 949 → 6 939 (−10) |
| `host_19p5_19p7` | 8 332 → 8 305 (−27) | 8 851 → 8 821 (−30) | 8 784 → 8 754 (−30) |
| `host_19p7_21p6` | 55 058 → 54 784 (−274) | 59 186 → 58 890 (−296) | 57 213 → 56 922 (−291) |
| `host_ge_21p6` | 497 → 494 (−3) | 602 → 594 (−8) | 539 → 530 (−9) |
| `counts_all` | 88 053 → 87 722 (−331) | 87 831 → 87 476 (−355) | 86 745 → 86 383 (−362) |

The **16 / 12 / 18** hostless deltas match the forensics' prediction exactly. The ORACLE arm
was over-pinned by ≈ 0.12 %, pushing the TP arm ≈ 0.02 % low.

**Proved en route:** `hostless` is **invariant** to the z-column convention — of
370 617 / 338 310 / 412 833 hostless op rows, **0** have a finite `Z_TRUE`, so
`min/max(Z_DLA, Z_TRUE) ≡ Z_DLA` for every one of them
(`hostless_is_z_column_invariant.INVARIANT: true`). That is why the A0 `hostless` block
lands on the pack's *exact* row support (§6), and why the ORACLE pin is now fully
support-matched.

---

## 5. 🔴 A SECOND, LARGER SUPPORT MISMATCH — and instance **#8**, in the adopted packs

> **Status: closed on the validation side by A0v2 (§5.1). The defect in the *adopted packs*
> themselves stands and is escalated.**

The `z_cut_columns` field (extension #7) exposed a difference **bigger than the collar one**:

* `build_scan_packs.py` applies the λ/z window to **`Z_DLA` only** (its docstring: *"Truth
  never enters the selection"* — the ckpt-10.8 observable-only predeclaration);
* `molly_faithful_pc_plots.make_lambda_z_BAL_cuts(use_truth_z=True)`, which
  `load_and_cut_catalog` calls for the *detection* catalogue, applies it to
  **`min/max(Z_DLA, Z_TRUE)`** (`Z_TRUE`→`Z_DLA` where NaN) — i.e. it **leaks truth into the
  selection**.

Measured at **fixed collar 3000** (so the collar cannot confound it):

| family | Z_DLA-only counts | truth-aware counts (adopted pack) | difference | % | (collar effect, for scale) |
|---|---|---|---|---|---|
| 2LPT-0 | 88 948 | 88 071 | **877** | **1.00 %** | 1 104 rows (3000→3300) |
| London-0 | 88 805 | 87 840 | **965** | **1.10 %** | 1 226 rows |
| Saclay-0 | 87 624 | 86 763 | **861** | **0.99 %** | 1 119 rows |

Two consequences:

1. **What A0 (round 1) did not close.** The census's five *host* slots, `counts_all`, and
   the `empirical_ops_*` operators all come through `load_and_cut_catalog` and were
   therefore on the **truth-aware** selection, while the scan-pack `counts` they are
   pinned/divided against are **observable-only**. The A0 `hostless` block escaped this
   (proved above); the **A0 sub-floor term `host_17p2_19p0` (`--extra-fixed-file` /
   `mu_extra`) did not** —
   `A0 sub-floor term: counts vs census.host_17p2_19p0 (row) -> FAIL, DIFFERS z_cut_columns`.
   **A0v2 (§5.1) closes this** by rebuilding both objects on the pack's own observable-only
   selection, without touching any frozen code.

2. **Instance #8 (new).** Inside the **adopted packs** `adopted_packs_v2p2_20260821/*`, the
   collars agree (3000/3000) but the **z columns do not**: `counts` is truth-aware
   (`load_mock_bundle` reproduces 88 071 bit-exactly, 0 cells differing) while
   `truth_counts` is `Z_DLA`-only (reproduced bit-exactly here from
   `use_truth_z=False`). Numerator on the *smaller* support, truth on the *larger* one →
   a ~1 % **understatement** of the mock bias, on top of, and independent of, the collar
   defect. The scan packs are accidentally z-column-*consistent* (both `Z_DLA`-only) and only
   collar-inconsistent; the adopted packs are the reverse. Any mock number taken off an
   **adopted** pack carries this ~1 pp. Real packs are built by a different path
   (`extract_pack_real.contract_row_mask`) and have no truth side — **not assessed here.**
   *This one is not fixed by A0v2 — it lives inside the adopted packs and needs a PI ruling.*

---

## 5.1 A0v2 — instance #8 closed on the validation side

`validation/absorber_ladder/support/rebuild_a0_products.py` (new, validation-only) rebuilds
the census **and** the empirical operators on the pack's own selection. It re-implements
**only step 6** of `load_and_cut_catalog` — the λ_rf + z_qso + BAL cut — parameterised on
`collar_kms` and `z_cut_columns ∈ {"minmax", "zdla_only"}`, and **calls the committed
machinery for everything else**: `load_catalog_dir`, `_build_qso_lookup`, the step-3 sentinel
filter, `match_truth_to_cat_molly` (steps 1–5), `good_mask`/`is_TP` (steps 7–8),
`track_c_tf_saclay._snap_off_molly_edges`, `extract_pack.bin_counts_cks`, and
`build_matched_ops`'s own `bin3` / `bin4` / `truth_hist_bks` / `_model_kernel_rows` /
`coarse_block_sum` / `safe_ratio` for the operator block.
`cddf_catalog_hbi.py`, `build_fp_census.py` and `build_matched_ops.py` are **untouched**
(all three are tracked on this branch).

### Fidelity is gated, not asserted

Run with the **committed** convention (`minmax`, collar 3000), the re-implementation must
reproduce the objects of record **bit-exactly**. It does, for every family:

| gate (collar 3000, `minmax`) | 2LPT-0 | London-0 | Saclay-0 |
|---|---|---|---|
| all 7 census blocks vs `fp_census_<fam>.npz` | PASS | PASS | PASS |
| `truth_counts_bks` vs the adopted pack | PASS | PASS | PASS |
| all 6 ops contract arrays (`C_true_bKs`, `C_true_bs`, `M_true_sKcb`, `E_true_cKsb`, `N_match_cksb`, `P6b_cks`) vs `empirical_ops_<fam>.npz` | PASS | PASS | PASS |

Nothing is written unless all of these pass.

### Primary gate — `counts_all` vs the A0 pack's `counts`

| family | A0v2 `counts_all` | A0 pack `counts` | Δ | cells differing | max abs Δ | relative |
|---|---|---|---|---|---|---|
| 2LPT-0 | 87 840 | 87 844 | **−4** | 4 | 1 | 4.6e−5 |
| London-0 | 87 574 | 87 579 | **−5** | 5 | 1 | 5.7e−5 |
| Saclay-0 | 86 500 | 86 505 | **−5** | 5 | 1 | 5.8e−5 |

**The residual is fully explained, and it is a defect on the *pack* side, not on this one.**
`load_and_cut_catalog` step 3 drops sentinel rows (`NHI_ERR == -1 OR Z_DLA_ERR == -1`)
*before* the truth match — 137 / 183 / 218 rows — while `build_scan_packs.py` applies **no
sentinel filter at all**. So the packs of record's `counts` contain **4 / 5 / 5
sentinel-flagged detections** that no truth-matched object can ever contain. Every residual
row sits in the **top SNR stratum `s = 7` (SNR ≥ 7)**: per-stratum differences are
`[0,0,0,0,0,0,0,−4]` / `[…,−5]` / `[…,−5]`; per-coarse-K `[0,−3,−1]` / `[−2,−1,−2]` /
`[−2,−3,0]`. Dropping the committed sentinel filter here would make the gate exact but would
feed fit-failure rows to the greedy one-to-one matcher, where they can steal truth systems
from good detections — so the filter was **kept** and the 0.005 % residual is reported
instead. **Flagged for the PI: `build_scan_packs.py` should apply the sentinel filter.**

### Deltas vs the truth-aware objects (combined collar 3000→3300 **and** z-column change)

| block | 2LPT-0 truth-aware → A0v2 (Δ) | London-0 (Δ) | Saclay-0 (Δ) |
|---|---|---|---|
| `counts_all` | 88 053 → 87 840 (**−213**) | 87 831 → 87 574 (**−257**) | 86 745 → 86 500 (**−245**) |
| **`hostless`** | 13 860 → 13 844 (**−16**) | 9 598 → 9 586 (**−12**) | 10 592 → 10 574 (**−18**) |
| **`host_17p2_19p0`** (`mu_extra`) | 3 200 → 3 208 (**+8**) | 2 611 → 2 612 (**+1**) | 2 668 → 2 668 (**0**) |
| `host_19p0_19p5` | 7 106 → 7 106 (0) | 6 983 → 6 980 (−3) | 6 949 → 6 947 (−2) |
| `host_19p5_19p7` | 8 332 → 8 315 (−17) | 8 851 → 8 823 (−28) | 8 784 → 8 764 (−20) |
| `host_19p7_21p6` | 55 058 → 54 864 (−194) | 59 186 → 58 970 (−216) | 57 213 → 57 005 (−208) |
| `host_ge_21p6` | 497 → 503 (+6) | 602 → 603 (+1) | 539 → 542 (+3) |
| op rows (floor 17.2) | 494 962 → 494 590 | 470 843 → 470 283 | 542 486 → 542 068 |
| `N_match_cksb` (ops) | 70 993 → 70 787 (−206) | 75 619 → 75 373 (−246) | 73 482 → 73 255 (−227) |
| **`truth_counts_bks`** used by `E` / `C` | 101 949 → **101 494** (−455) | 106 705 → **106 232** (−473) | 104 561 → **104 095** (−466) |

Three things to read off this table:

* **`hostless` is unchanged from A0v1** (−16 / −12 / −18): it is z-column invariant, so the
  ORACLE `mu_FP` pin moves only by the collar, as proved in §4.3.
* **`host_17p2_19p0` moves the *other* way (+8 / +1 / 0)** despite the collar tightening —
  the observable-only window *re-admits* rows the truth-aware window had dropped. This is
  the sub-floor term `mu_extra` the A0 generative contribution uses, so it matters: A0v1 gave
  3 199 / 2 611 / 2 664; **A0v2 gives 3 208 / 2 612 / 2 668**, and only the A0v2 value is on
  the counts' support.
* The **`truth_counts_bks` inside `E_true_cKsb` and `C_true_bKs` / `C_true_bs`** is now
  `equals_A0_pack_truth_counts_bks: true` — bit-identical to the A0 pack's truth plane, so
  the operators and the pack divide by literally the same array. (Crude aggregate: the mean
  per-`b` ratio of `C_true_bs` moves by ×1.018 / ×1.004 / ×1.005, but that mean runs over
  near-empty basis bins too and must not be quoted as a completeness shift.)

### Floor independence — a free by-product

Under `zdla_only` the truth floor can no longer reach the detection rows (that coupling ran
through `Z_TRUE` in the window). Measured: `counts_all` from the floor-17.2 pass and
`counts_obs` from the floor-19.0 pass are **elementwise identical** —
87 840 / 87 574 / 86 500, `IDENTICAL: true`. This is the evidence behind the §2 adjudication.

### Result

**Row-level support is now equal across pack `counts`, `dX`, `fp_E_alloc`, A0
`truth_counts` / `truth_counts_bks`, the A0v2 census (all 7 blocks) and the A0 operators
(all 6 contract arrays)** — 18 planes, one id, all three families
(`A0V2_SUPPORT_GATES.json`):

| family | common row `support_id` | planes | `truth_host_floor` reported alongside |
|---|---|---|---|
| 2LPT-0 | `1e6fe40c52e72fd0` | 18 | `n/a`, 19.0, 17.2 |
| London-0 | `a90d09704fe74c94` | 18 | `n/a`, 19.0, 17.2 |
| Saclay-0 | `af0ea7061f4ee4df` | 18 | `n/a`, 19.0, 17.2 |

The `full`-level run is recorded alongside and fails **only** on `truth_host_floor`, as
designed (§2).

`run_ladder.py`'s own `--fix` slicing was exercised on the A0 operators (dry, no sampling):
`C_true_bKs[:, kz, :] → (16,15,8)`, `C_true_bs.T → (8,16)`,
`M_true_sKcb[:, kz, :, :] → (8,15,29,16)` (equal to the model's own `Mg` shape),
`E_true_cKsb[:, kz, :, :] → (29,15,8,16)`; and the A0v2 census gives
`mu_FP = (29,15,8)` totalling 13 844 / 9 586 / 10 574 with `mu_extra` 3 208 / 2 612 / 2 668
— every shape equal to `counts`.

---

## 6. `support_id` of every object

Full 12-field ids (16-hex prefixes; full 64-hex in the `.support.json` sidecars and in
`A0_BUILD_SUMMARY.json:<fam>.support_ids`).

| object | 2LPT-0 | London-0 | Saclay-0 |
|---|---|---|---|
| scan pack `counts` / `dX` / `fp_E_alloc` @3300, `Z_DLA`, floor `n/a` | `9f0356c4cc3449ec` | `1eceebe91202a6f9` | `0c27e846db6ae39c` |
| **A0** `truth_counts` / `truth_counts_bks` @3300, `Z_DLA`, floor 19.0 | `4f78e8baf7c2edbf` | `1cbe4a1e5b5dfab6` | `92c3e51143ae80a1` |
| ~~of-record `truth_counts` @3000~~ (**the defect**) | `39da29b5ae44c203` | `2536c26e43213c8f` | `90737ef50d480c77` |
| **A0v1** census `hostless` @3300, `Z_DLA` (z-col invariant), floor 17.2 | `84a54a9f48b0e41d` | `8ed60aef84954f4d` | `910c3bfdccdaa4f3` |
| A0v1 census host slots + `counts_all` @3300, **truth-aware** z, floor 17.2 | `38ea411e33426a3a` | `bc9a36cde789a313` | `aaf40830c3d0b211` |
| ~~of-record census @3000~~ | `c4b3c2e2ad31d2f8` | `fb4dd23204e69ae2` | `d090ca3a0321d6ae` |
| **A0v2** census, **every block** @3300, `Z_DLA`, floor 17.2 | `84a54a9f48b0e41d` | `8ed60aef84954f4d` | `910c3bfdccdaa4f3` |
| **A0** operators, all 6 contract arrays @3300, `Z_DLA`, floor 19.0 | `4f78e8baf7c2edbf` | `1cbe4a1e5b5dfab6` | `92c3e51143ae80a1` |
| **row-selection id** (11 fields) shared by **every A0/A0v2 object above** | `1e6fe40c52e72fd0` | `a90d09704fe74c94` | `af0ea7061f4ee4df` |
| row-selection id of the *superseded* truth-aware host slots / `counts_all` | `fc49e6584a5d6b95` | `9be0cb7736f90a6b` | `3097eb5e607e2a76` |

Note the A0v2 census's full id equals the A0v1 **`hostless`** id: once the host slots are put
on the observable-only window, the whole census sits where `hostless` already sat.

### Gate outcomes (`A0_SUPPORT_GATES.json`, `A0V2_SUPPORT_GATES.json`, `A0_VERIFICATION.json`)

| check | 2LPT-0 | London-0 | Saclay-0 |
|---|---|---|---|
| pack planes (`counts`, `dX`, `fp_E_alloc`, `truth_counts`, `truth_counts_bks`), **row** level | **PASS** | **PASS** | **PASS** |
| pack planes, **full** level | FAIL — `truth_host_floor` only (`n/a` vs 19.0), by design (§2) | idem | idem |
| ORACLE pin: `counts` + `truth_counts` + A0v1 census `hostless`, row level | **PASS** | **PASS** | **PASS** |
| A0v1 sub-floor term: `counts` + A0v1 census `host_17p2_19p0`, row level | FAIL — `z_cut_columns` (§5) | idem | idem |
| pack + whole A0v1 census, row level | FAIL — `z_cut_columns` (§5) | idem | idem |
| **pack + A0v2 census (7 blocks) + A0 ops (6 arrays), row level — 18 planes** | **PASS** | **PASS** | **PASS** |
| pack + A0v2 census + A0 ops, **full** level | FAIL — `truth_host_floor` only, by design (§2) | idem | idem |

The invariant the PI asked for — `support_id(N_truth) = support_id(dX) = support_id(counts) =
support_id(FP census)`, now for the **whole** census and the operators too — **holds** at the
adjudicated `row` level for all three families.

---

## 7. `load_pack` / `run_ladder` acceptance (task 4)

Run under `gpdla-hbi`; see `A0_VERIFICATION.json`.

* `CDDF_analysis.hbi_mcmc.pack.load_pack` **accepts every A0 pack**
  (`load_pack_accepts_A0_pack: true` ×3); `truth_counts == truth_counts_bks.sum(axis=2)`
  holds; the `.provenance.json` sidecar is attached.
* **An embedded `support_id` NPZ key is REJECTED.** Probed directly:
  `PackSchemaError: pack scanpack_<fam>_b300_A0.npz: unknown keys ['support_id'] (schema v1
  is a closed contract)` — `pack.load_pack` computes
  `unknown = keys - _REQUIRED_KEYS - _OPTIONAL_KEYS` and fails closed (`pack.py:665`).
  **Therefore the A0 packs carry no `support_id` key; their support lives in the
  `.support.json` sidecar only**, which is recorded verbatim in each pack's provenance
  (`support_id_storage`). The census has no closed schema, so `fp_census_<fam>_A0.npz`
  carries **both** the embedded 0-d `support_id` string **and** the sidecar.
  *Consequence to note:* a pack's support is sidecar-resident, so it can be separated from
  the pack by a careless copy. `read_stamp` is fail-closed (a missing sidecar raises, never
  passes) and the sidecar records the pack's own sha256, so separation is detectable — but
  admitting `support_id` to `_OPTIONAL_KEYS` would be strictly better and needs a tracked
  one-line change (PI decision).
* `validation/fp_ladder/run_ladder.py` **imports cleanly** and everything it does before
  MCMC succeeds on the A0 products (dry, no sampling): the MOCK gate passes
  (`truth_counts.sum() > 0`), `build_cc_tensors(pk)` returns `Mg` of shape `(8, 15, 29, 16)`,
  and the ORACLE arm's census shape check passes — `mu_FP` totals 13 844 / 9 586 / 10 574 and
  `mu_extra` (`host_17p2_19p0`) 3 199 / 2 611 / 2 664 (A0v1) → **3 208 / 2 612 / 2 668**
  with the A0v2 census, which is the value on the counts' support.
* `run_ladder.py`'s `--fix` slicing was exercised on `empirical_ops_<fam>_A0.npz`: every
  contract array has the shape the runner expects (§5.1).
* Wiring `check_support_consistency` into `run_ladder.py` was **left to the caller** as
  instructed. Suggested call site: immediately after `load_pack`, before `build_cc_tensors`:
  `check_support_consistency(a.pack, a.census, a.ops, fields=ROW_SELECTION_FIELDS)` — the
  `row` level per the §2 commander adjudication, with `--census
  fp_census_<fam>_A0v2.npz` and `--ops empirical_ops_<fam>_A0.npz`.

---

## 8. Products

`/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/support/` — **47 files**,
`SHA256SUMS` covers all of them (regenerated after the A0v2 round). **The packs of record,
the census of record and the operators of record were not touched** — the builders only ever
read them.

**Use these for the ladder:** `scanpack_<fam>_b300_A0.npz` +
`fp_census_<fam>_A0v2.npz` + `empirical_ops_<fam>_A0.npz`. (`fp_census_<fam>_A0.npz` is the
first-round, truth-aware-host-slot census — kept for the audit trail; its `hostless` block is
identical to A0v2's, its host slots are not.)

| file (× 3 families) | contents |
|---|---|
| `scanpack_<fam>_b300_A0.npz` | the pack of record with **only** `truth_counts` + `truth_counts_bks` replaced by the collar-3300 rebuild. Asserted key-by-key: `keys_changed == ['truth_counts','truth_counts_bks']` and nothing else; dtypes preserved (float64). |
| `scanpack_<fam>_b300_A0.support.json` | the stamp: 12 fields, canonical string, sha256, product sha256, and `extra.planes` for `counts` / `dX` / `fp_E_alloc` / `truth_counts` / `truth_counts_bks` |
| `scanpack_<fam>_b300_A0.provenance.json` | what changed and why, source pack + sha256, gates, env, git, all findings of §3–§5 |
| `fp_census_<fam>_A0.npz` | all seven 17.2-floor blocks at collar **3300**, plus the collar-3000 arrays alongside under a `_collar3000` suffix, plus the embedded `support_id`; the three 19.5-floor accounting arrays copied unchanged (they belong to the adopted collar-3000 pack and are diagnostic only) |
| `fp_census_<fam>_A0.json` | totals @3300 and @3000, per-block deltas, `hostless` per coarse-K and per SNR stratum, `support_id`, full provenance |
| `fp_census_<fam>_A0.{support,provenance}.json` | stamp (with per-block planes) + provenance |
| **`fp_census_<fam>_A0v2.npz`** | **the census of use** — all seven 17.2-floor blocks on the pack's own **observable-only** (`Z_DLA`) window at collar 3300, plus the truth-aware collar-3000 arrays alongside under a `_truth_aware_collar3000` suffix, plus the embedded `support_id` |
| `fp_census_<fam>_A0v2.{json,provenance.json,support.json}` | totals + the `counts_all`-vs-pack gate; provenance (incl. the fidelity gates); stamp with per-block planes |
| **`empirical_ops_<fam>_A0.npz`** | **the operators of use** — the six contract arrays `C_true_bKs`, `C_true_bs`, `M_true_sKcb`, `E_true_cKsb`, `N_match_cksb`, `P6b_cks` plus the full `build_matched_ops` key set, on the same support; `truth_counts_bks` inside it is bit-identical to the A0 pack's |
| `empirical_ops_<fam>_A0.{provenance.json,support.json}` | provenance + stamp |
| `A0_BUILD_SUMMARY.json` / `A0V2_BUILD_SUMMARY.json` | every gate, ratio, delta and diagnostic, per family, per round |
| `A0_SUPPORT_GATES.json` / `A0V2_SUPPORT_GATES.json` | the gate run at both levels on our own products |
| `A0_VERIFICATION.json` | `load_pack` / `run_ladder` / estimand / invariant checks |

---

## 9. What was recoverable from provenance — and what was **not**

Support of the **existing** objects had to be reconstructed; here is the audit.

| source sidecar | fields it genuinely records | fields it does **not** |
|---|---|---|
| `scanpack_<fam>_b300.provenance.json` | recipe, `b = 300` ⇒ `collar_kms = 3300` (derivable), `src` pack, `data_plane_keys`, `code_commit` | **10 of 12**: `snr_min`, `p_dla_min`, `z_window`, `lam_rf_min/max`, `z_cut_columns`, `quality_cut`, `bal_policy`, `catalogue_id`, `truth_catalogue_sha256`, `truth_host_floor` |
| `modelA_pack_<fam>_…_v2.provenance.json` (adopted) | upgrade chain + `src_sha256`, adopted-response shas, `tp_convention_id`, `contract_id`, `code_commit` | **all 12 selection fields** — the adopted packs record *no* selection metadata |
| `fp_census_<fam>.json` (`provenance`) | `snr_min`, `p_dla_min`, `op_cut` ⇒ `quality_cut`, `census_truth_floor = 17.2`, `pack_truth_floor = 19.5`, catalogue **paths**, gated pack + sha | `collar_kms` (only *inferable* from the gated pack being the adopted one), `z_window`, `lam_rf_min/max`, `z_cut_columns`, catalogue **content hashes** |
| `empirical_ops_<fam>.provenance.json` | `snr_min`, `p_dla_min`, `lam_rf_min/max`, both floors, catalogue/truth/BAL/molly paths, pack + scanpack + census shas, and an explicit `collar` block (3000 adopted / 3300 scanpack) | `z_window`, `z_cut_columns`, `quality_cut`, catalogue **content hashes** |
| `modelA_pack_REAL_…selection_contract.json` (**real** pack, R-016) | essentially the whole schema: `collar_kms`, `snr_min` + `snr_strict`, `p_dla_min` + strictness, `quality`, `bal_policy`, `lambda_rf_window`, `z_qso_min/max` | catalogue content hashes; `z_cut_columns` |

**Not recoverable at all — re-derived, with the recovery argument:**

1. **Catalogue / truth-catalogue content hashes at build time.** No object on disk records
   them. The `catalogue_id` and `truth_catalogue_sha256` in every A0 stamp are hashes of the
   files **as they are today (2026-09-13)**. Recovery argument: the collar-3000 truth gate and
   the collar-3000 census gate reproduce the arrays on disk **bit-exactly** from today's
   catalogues — 101 949 / 106 705 / 104 561 truth systems and all seven census blocks, zero
   differing cells. That is essentially conclusive evidence the inputs are unchanged since
   2026-08-21 / 2026-09-12, and it is the only recovery mechanism available. Going forward
   the stamp closes the hole.
2. **`z_window` = (2.0, 4.25).** In no mock sidecar. Taken from the `HBIConfig` defaults —
   the values of record, independently corroborated by the real pack's
   `selection_contract.json` (`z_qso_min: 2.0`, `z_qso_max: 4.25`).
3. **`z_cut_columns`.** Recorded nowhere, by anything. Determined only by reading the two
   code paths — which is how §5 was found, and why it is in the schema.
4. **`quality_cut`** for the packs: recorded only as prose in the census/real sidecars
   (`DLAFLAG == 0`); re-derived from `build_scan_packs` (`flag_ok`) and
   `load_and_cut_catalog` (`good_mask`), which agree.
5. **`dX_coarse_committed` is numpy-version sensitive** at 1–3 ULP
   (`build_scan_packs.py` docstring: `gpdla` numpy 2.4.4 vs `gpdla-hbi` 2.2.6). Not a support
   field (it is a provenance carrier, optional in `pack.py`) and the A0 builder copies it
   byte-identically rather than recomputing it — but worth recording.
6. **`fp_counts` (the loa-0 FP calibration block)** is on **yet another support**: read
   directly in `extract_pack.build_fp_block`, its op cut is only
   `SNR > 2 & P_DLA > 0.99 & lam_rest ≥ 1025 Å` — **no collar, no `z_qso` window, no BAL
   veto**, and it comes from the loa-0 HCD-free twin catalogue, not the mock's. (The real
   pack's `selection_contract.json` records the collar consequence: 89 events on support vs
   87 under `c = 3300` — R-015, disclosure pending PI.) A0 copies the block unchanged and
   does **not** claim it matches; its calibration-vs-survey support difference is the
   modular-Λ cut-feedback question of PI ruling §12, not A0's. Flagged.

---

## 10. Ten-line summary

1. The support contract is built, closed (12 fields), hashed, fail-closed and unit-tested —
   `tests/test_support_contract.py`, **50 passed**: identical supports pass, each of the 12
   fields alone fails with the field named, every missing/unknown/None/NaN/blank/bool/empty
   case fails closed, an unstamped or self-inconsistent product fails closed.
2. Defect #7 confirmed at the array level and repaired: the scan packs' `truth_counts` /
   `truth_counts_bks` were byte-identical copies of the collar-3000 adopted packs, divided by
   collar-3300 `counts` / `dX`. Collars differ in **exactly one** number (`coll = collar/c`,
   `Δz = 0.0010007` per edge); the 3300 selection is strictly nested in the 3000 one (0 rows
   dropped by re-applying the 3000 mask), so the rebuild is exact.
3. Truth rebuilt at 3300 behind a **bit-exact collar-3000 gate**: ratios
   **0.995537 / 0.995567 / 0.995543**, flat in `b` (0.992–0.998) and `K` (0.9940–0.9960) —
   uniform, so it explains no shape defect. dN/dX(≥20.0) `0.086604 → 0.086191` (2LPT-0,
   reproducing the forensics exactly), `0.093729 → 0.093308`, `0.090070 → 0.089661`; every
   mock bias was understated by **0.45–0.49 pp** (≥20.0 → **+1.48 / +1.65 / +1.88 %**;
   ≥20.3 → **+3.45 / +2.63 / +3.71 %**).
4. Census rebuilt at 3300 behind a bit-exact collar-3000 gate on all seven blocks:
   `hostless` **13 860→13 844 / 9 598→9 586 / 10 592→10 574** (−16 / −12 / −18, exactly as
   predicted); the ORACLE arm had been over-pinned by ≈ 0.12 %. `hostless` is **proved**
   z-column invariant (0 of 370 617 / 338 310 / 412 833 hostless rows has a finite `Z_TRUE`).
5. **Instance #8, found here:** `build_scan_packs` selects on `Z_DLA` only while
   `load_and_cut_catalog` selects on `min/max(Z_DLA, Z_TRUE)` — truth leaking into the
   selection — **877 / 965 / 861 rows = 1.00 / 1.10 / 0.99 %** at fixed collar.
6. **A0v2 closes #8 on the validation side.** `rebuild_a0_products.py` re-implements *only*
   step 6 of `load_and_cut_catalog` (parameterised collar + z column) and calls the committed
   machinery for everything else; run with the committed convention it reproduces the census
   of record (7 blocks), the adopted pack's `truth_counts_bks`, and the ops of record
   (6 contract arrays) **bit-exactly** — nothing is written unless those gates pass.
7. **Row-level support is now equal across pack `counts` / `dX` / `fp_E_alloc` / A0
   `truth_counts` / `truth_counts_bks` / A0v2 census (7 blocks) / A0 operators (6 arrays)** —
   18 planes, one id (`1e6fe40c52e72fd0` / `a90d09704fe74c94` / `af0ea7061f4ee4df`), all three
   families. Deltas vs the truth-aware objects: `counts_all` −213 / −257 / −245,
   `hostless` −16 / −12 / −18, **`host_17p2_19p0` +8 / +1 / 0** (the sub-floor `mu_extra` moves
   *up*: 3 200→**3 208**, 2 611→**2 612**, 2 668→**2 668**), `N_match_cksb` −206 / −246 / −227,
   and the `truth_counts_bks` used by `E`/`C` is now bit-identical to the A0 pack's truth plane.
8. **Counts gate residual: −4 / −5 / −5 rows** (max 1 per cell, 4.6–5.8 × 10⁻⁵, *all* in SNR
   stratum `s = 7`). Fully explained and **a defect on the pack side**: `load_and_cut_catalog`
   drops 137 / 183 / 218 sentinel rows (`NHI_ERR == -1 OR Z_DLA_ERR == -1`) before matching,
   which `build_scan_packs.py` never does — so the packs of record's `counts` contain 4 / 5 / 5
   sentinel-flagged detections. Filter kept (removing it would let fit-failure rows steal truth
   systems in the greedy matcher). **Flagged: `build_scan_packs.py` should apply it.**
9. **Gate level — commander adjudication, PI ratification pending:** the ladder gate runs at
   the **`row`** level (11 fields) with `truth_host_floor` reported alongside every plane. Now
   independently supported: under the observable-only window the floor is **provably inert on
   the rows** (`counts_all` @17.2 ≡ `counts_obs` @19.0, elementwise, all three families) — the
   old 18 / 9 / 18-row perturbation existed only through the `Z_TRUE` leak. The `full` level is
   still run and recorded; it fails **only** on `truth_host_floor`, by design.
10. `load_pack` accepts all three A0 packs; an embedded `support_id` NPZ key is **rejected**
    (`PackSchemaError: unknown keys ['support_id']` — schema v1 is closed), so pack supports
    live in `.support.json` sidecars only (census and ops carry both); `run_ladder.py` imports,
    and its MOCK gate, `build_cc_tensors`, ORACLE shape check and `--fix` slicing all accept the
    A0/A0v2 products with no sampling. **Still open for the PI:** instance #8 *inside the
    adopted packs* (~1 pp on any mock number taken off them), the sentinel filter in
    `build_scan_packs.py`, admitting `support_id` to `pack._OPTIONAL_KEYS`, and the `fp_counts`
    block's own (third) support. 47 products + `SHA256SUMS` in
    `/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/support/`; nothing
    committed, no tracked file modified, no SLURM (~5 min total).
