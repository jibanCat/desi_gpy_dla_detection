# `validation/fp_ladder` — the MOCK false-positive truth census

**VALIDATION-ONLY. No sampler is run; nothing under `CDDF_analysis/` is modified.**

`build_fp_census.py` rebuilds, per mock family (2LPT-0 / London-0 / Saclay-0)
and per `(c, k, s)` cell of the pack grid (`c = 29` N̂ bins, `k = 15` fine-z
bins, `s = 8` S/N strata):

| array in `fp_census_<fam>.npz` | truth floor | meaning |
|---|---|---|
| `counts_19p5`, `counts_tp_19p5`, `counts_unmatched_19p5` | 19.5 | the pack's own accounting: `is_TP = isfinite(NHI_TRUE)` against the **19.5-floored** truth table |
| `counts_all` | 17.2 | the same catalogue and the same op cut, re-matched against the **17.2-floored** truth table |
| `hostless` | 17.2 | **the census**: no truth host even at 17.2 |
| `host_17p2_19p0`, `host_19p0_19p5`, `host_19p5_19p7`, `host_19p7_21p6`, `host_ge_21p6` | 17.2 | matched candidates split by their host's true `N_HI` |

Grid edges (`nhat_edges`, `zf_edges`, `snr_edges`, `zc_edges`, `kz_to_K`) and a
JSON `provenance` string are stored in the same NPZ; `fp_census_<fam>.json`
carries the totals, the per-coarse-z and per-S/N rollups, and the same
provenance.

## Why the second floor exists

`is_TP` is a property of the **(catalogue, truth-floor) bundle**, not of a
detection. The truth table is pre-floored at `truth_nhi_floor = mm.nhi_edges[0]
= 19.5` *before* matching (`cddf_catalog_hbi.py:576`), so a detection whose
genuine absorber sits below 19.5 is labelled NOT `is_TP`. That is the contract's
own `TRUTH_FLOOR_ASYMMETRY_IN_is_TP` contradiction. Re-cutting at the molly172
sub-floor 17.2 recovers those hosts; what is **still** hostless at 17.2 is the
only defensible FP-like census on disk.

On 2LPT-0, 10 321 of the 24 181 pack-level "unmatched" (43 %) are genuine
absorbers the 19.5-floored truth table simply could not see.

## The three "false positive" meanings — and which one the model means

Quoted verbatim from `CDDF_analysis/hbi_mcmc/matching_contract.py`
(`POPULATIONS`, `:562-620`):

**`P4_FOREST_FP`** — *pure-forest false positives, measured on loa-0*
> `predicate_text`: "NOT is_TP AND attributable to the forest (the loa-0 HCD-free twin: EVERY loa-0 detection is a forest FP by construction)"
> `forward_term`: "`fp_w . exp(t_K(k)) . lam_fp[c,s] . fp_E[k,s]` (forward.py:452) — **FORWARD-MODELLED, never subtracted on this route**"
> `support_class`: `WEAKLY_MEASURED`
> `note`: "MEASURED on the adopted packs: 89 loa-0 detections on the (c=29, s=8) grid, in 25 of 232 cells, from 2255 searched sightlines."

**`P6_RESIDUAL`** — *residual requiring a substantive physical prior*
> `predicate_text`: "any candidate not claimed by P1, P2 or P4. Concretely: (a) is_TP AND NHI_TRUE >= 21.6 (scatter-down from above the ceiling); (b) is_TP AND NHI_TRUE < 19.0 (below the basis floor — **NO support at all**); (c) NOT is_TP and not forest-attributable: **blends, a second candidate on an already-claimed truth row, and matches beyond dz_rel**"
> `forward_term`: "**NONE. P6 has no term in mu.**"
> `prior`: "**NONE.** Any nonzero P6 must be supplied by a substantive physical prior that this contract does not contain."

**`P3_INCOMPLETENESS`** is a **truth-side** population (`side=Side.TRUTH`) — it
is *missed absorbers*, not detections, so it is not a candidate for "FP" at all:
> `forward_term`: "NO TERM OF ITS OWN. It is the complement `(1 - C[b,s])` of the completeness factor already in the P1/P2 term."

**The HBI FP term models P4 and only P4.**

## What the census is, and how it may be used

`hostless` is **P4 ⊕ P6(c)**. It is not a measurement of P4, because nothing on
disk separates the forest FPs from P6 sub-slot (c) — blends, second candidates
on an already-claimed truth row, and matches beyond `dz_rel` — at the
per-candidate level. That separation does not exist in the data.

It is also **not a ceiling** on the forward FP term. The Phase-A adversarial
review corrected `_FP_CEILING_NOTE` (`matching_contract.py:1629-1647`) to say
exactly that: *"The 17.2-floor `unmatched` is the defensible comparator, as a
CENSUS, not as a ceiling."*

⇒ **Soft-flag use only.** A gap between the forward `mu_FP` and this census is a
flag to investigate (e.g. the z-tilt: on all three families the naive loa-0 FP
prediction over-predicts the lowest coarse-z block and under-predicts the
highest), never a subtraction, a bound, or a likelihood term.

The 17.2-floor `counts_all` is *not* the 19.5-floor `counts_19p5`: lowering the
truth floor moves `Z_TRUE`, hence a handful of rows through
`make_lambda_z_BAL_cuts(use_truth_z=True)` (`extract_pack.py:962-970`). The
measured perturbation is 18 / 9 / 18 rows (2LPT-0 / London-0 / Saclay-0) and is
reported as `derived.cat_cut_perturbation_19p5_minus_17p2`.

## Gates and controls (all hard, nothing is written if one fails)

1. **Pack gate.** The 19.5-floor `counts[c,k,s]` rebuilt from
   `extract_pack.load_mock_bundle` must equal a pack's `counts` **elementwise,
   exactly**. The pack is `--pack`; the default is the collar-scan pack of
   record `scanpack_<fam>_b300.npz`.
   ⚠️ **The scan packs do NOT satisfy this gate, by construction**: their
   `counts` are rebuilt at collar `3000 + b = 3300 km/s`
   (`build_scan_packs.py`), a different sightline selection. Measured deltas:
   2LPT-0 88071 vs 87844 (+227), London-0 87840 vs 87579 (+261), Saclay-0
   86763 vs 86505 (+258). The bundle reproduces the **adopted source** packs
   `adopted_packs_v2p2_20260821/modelA_pack_<fam>_bw0p2_pad19p0_molly172_v2.npz`
   — the `src` recorded in each scan pack's own provenance — with a total
   difference of 0. Point `--pack` at those. The failing comparison is still
   recorded in the provenance (`pack_gates[*].label == "pack_of_record_informational"`).
2. **Contract cross-read.** `CONTROLS` is checked against
   `matching_contract.py` by `ast` parsing (the module cannot be *imported*
   under `gpdla`: its top-level `from CDDF_analysis.hbi_mcmc import reporting`
   executes the package `__init__`, which imports jax).
3. **Control totals**, hard-asserted per family — the 19.5-floor
   `n_cat_cut / n_op / n_on_grid / n_is_TP / n_unmatched`, the minimum `is_TP`
   `NHI_TRUE` (must be ≥ 19.5), and all six 17.2-floor totals plus the fine
   `[19.0,19.5)` / `[19.5,19.7)` split.
4. **Slot partition.** Every op row must fall into exactly one of
   `hostless` + the five host slots, and the six arrays must sum to
   `counts_all` on the grid.
5. **Index convention.** The local pure `bin_index` must agree with the
   committed `extract_pack._idx` on the actual arrays, and `KZ_TO_K` must still
   be `repeat([0,1,2], 5)`.

Slot comparisons use the verified recipe's tolerance, `(n >= lo - 1e-9) &
(n < hi - 1e-9)` — note the sign: a value within 1e-9 *below* an edge is placed
in the *upper* slot. Unit-tested in `tests/test_fp_census_grid.py`.

## Running it

```bash
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh && conda activate gpdla
AD=/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/adopted_packs_v2p2_20260821
for f in 2lpt0 london0 saclay0; do
  python validation/fp_ladder/build_fp_census.py --family $f \
      --out /scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/census \
      --pack $AD/modelA_pack_${f}_bw0p2_pad19p0_molly172_v2.npz
done
pytest tests/test_fp_census_grid.py -q
```

≈ 22 s per family, single core (two catalogue re-cuts at ~10 s each).

## Provenance of the recipe

Reproduces, line for line, the recipe verified in `OPUSH_REPORT.md §Q1.4`
(2026-09-12), which reproduced every control total in `matching_contract.py`,
the 2026-08-05 kernel/FP audit table, and the 2026-08-07/08-12 per-detection
caches. `extract_pack.py` is loaded **file-directly** for the same jax reason as
above.
