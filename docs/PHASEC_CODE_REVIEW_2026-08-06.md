# INDEPENDENT REVIEW (PI rulings §21) — Phase-C calibration work

Reviewer: independent subagent (fresh context, no authorship of the
reviewed commits), 2026-08-06. Scope: `3802d27..8bfd842` on
`calibration/phaseC-highN-fp-2026-08-06`. The reviewer re-ran code
rather than reading it (per the project's referee discipline); every
re-run command wrote to session scratch only. Verbatim report below;
the disposition of each finding is recorded at the end (added by the
implementing session, clearly marked).

---

**Worktree:** `/home/mfho/wt_calib_phaseC`, branch `calibration/phaseC-highN-fp-2026-08-06`.
**Scope note:** HEAD advanced during the review — `git log 3802d27..HEAD` initially ended at `cd6044a` (12 commits); commit `8bfd842` (budget + checkpoint + handoff) landed mid-review and is included. Nothing was modified, committed, pushed, or submitted to SLURM; all re-runs wrote to the session scratchpad only.

## FINDINGS (ordered by severity)

**F1 — DEFECT (provenance): load-bearing `.npz` inputs are claimed "committed" but are gitignored and untracked.**
`diagnostics_phaseC/threshold_study/run_threshold_study.py:24` and `threshold_study/findings.md:13` describe `observed_tilt_shape.npz` as "committed"; it is caught by `.gitignore:52` (`*.npz`) and is NOT in git (`git ls-files` confirms), and no committed routine generates it (grep finds only the consumer). The committed `threshold_study.json` is therefore not reproducible from the repo alone. Same for `preimage_M_{mock}.npz`, listed in `preimage/findings.md`'s Files table as analysis outputs. Mitigation I verified: the npz content is exactly the twin's per-bin fractional residual — re-ran `FS.selftest` on the committed twin pack and compared: `max|npz − (obs−mu)/mu| = 0.000e+00` (29 bins). This violates the standing headline-provenance rule; fix is mechanical (`git add -f` + a small committed generation stanza).

**F2 — DEFECT (false evidence claim in a PI-facing document): the checkpoint asserts a committed review record that does not exist.**
`docs/PHASEC_CHECKPOINT_2026-08-06.md` §B: "Independent code review: committed as the review record alongside this checkpoint"; `docs/PHASEC_HANDOFF.md` work log: "independent §21 code review dispatched and its record committed alongside." No review record exists anywhere on the branch (searched docs/, diagnostics_phaseC/, git log), and the branch containing the claim is already pushed to origin. This review IS that review, and it was instructed not to commit. Written-before-true in the evidence section of a PI checkpoint is exactly the class the RULES discipline exists for; the checkpoint text must be corrected to point at wherever this record is actually filed.

**F3 — RISK: `injection/measure_phaseC_pairs.py` applies a SUBSET of the production op-mask while claiming exact fidelity.**
Lines 49–51/102 apply only `P_DLA > 0.99` and `SNR > 2`. Production (`CDDF_analysis/hbi/znz_kernel.py:571` `measure_znz_response` on `load_and_cut_catalog` output) additionally has: the sentinel filter (`NHI_ERR == −1 | Z_DLA_ERR == −1` dropped BEFORE matching, `cddf_catalog_hbi.py` step 3), `good_mask = DLAFLAG == 0`, and λ_rf/z_QSO/BAL cuts. The match-before-cuts ORDER is faithful; the mask CONTENT is not, contra the docstring's "exactly as measure_znz_response does". Measured impact at pilot scale (re-ran on the pilot dlacats): 0 op rows with DLAFLAG≠0, 0 SNR discrepancies, but **3/573 op rows carry sentinel `NHI_ERR/Z_DLA_ERR == −1`** that production would have dropped pre-match. Small now; must be closed before Stage-2 scoring (the handoff already lists the adjacent analysis-window refinement as pre-production work).

**F4 — RISK: the r5 deterministic bound [14,16] is guaranteed by this pack's counts, not by the test's stated exclusion rule.**
`tests/test_modelA_rungs.py::test_r5_deterministic_calibration_width_contract`. Algebra of the ratio itself is correct (`sig² = 1/(d+½)+1/(t−d+½)`, matches `forward.py:196`), and r ≤ 16 is analytically guaranteed. But r ≥ 14 requires min(d, t−d) ≥ 52, while the test's floor only excludes cells below 32 — at the admissible worst case (d = t−d = 32), r = 16·32.5/40 = **13.0 < 14**. Re-ran: the test PASSES (3.3 s), and at this pack r ∈ [15.935, 15.960] with min cell count 1531 over 6 ok cells — a huge margin, so no current defect; but a future `synthetic_pack` grid/count change could fail the test with healthy plumbing, and the docstring's rationale for the floor mis-states where the guarantee comes from. Either raise the floor to ≥ 52 effective counts or restate the docstring.

**F5 — RISK: bridge acceptance criterion 1's CI clause is vacuous as frozen.**
`docs/PHASEC_BRIDGE_DESIGN.md` §4 item 1: "< 75 counts, AND its 95% CI does not exclude values < 75" — a CI always contains its point estimate, so the second clause is implied by the first; an estimate of 74 ± 60 passes item 1 outright (item 2's coherent-offset z < 3 is the only dispersion protection). Since the criterion is FROZEN pre-data, any tightening later requires a written amendment + PI approval — surface this at the checkpoint now, before bridge data exist.

**F6 — RISK (minor, quantified): `sizing.py` sums σ²(G3) over PROD_BINS only** (`sizing.py:142-143`), omitting the tail (b14–b15) and bridge (b2–b6) bins' contributions to the predicted-G3 variance even though those bins are also re-measured. Recomputed from the committed JSONs: prod-only σ = 111.99 (json 111.96), +tail = 111.99, +tail+bridge = **113.07** — still under the 116.72 binding target, power 0.920 ≥ 0.90. Immaterial today; state or include it, since the frozen criterion is on the full response-induced σ(G3).

**F7 — RISK: role/holdout separation is design-only in committed code.** `measure_phaseC_pairs.py` aggregates every manifest row regardless of role (the sidecar is read only for a substrate label, taken from the first entry, lines 224–226). Fine for single-role pilot arms; the 25% whole-healpix holdout and role exclusion have no enforcement code yet — a Stage-2 blocker consistent with the handoff's own "labeled artifact builder" TODO, listed here so it cannot be forgotten.

**F8 — NIT (documentation overclaims):** (a) `run_threshold_study.py:29-33` claims the faithfulness guard reproduces committed "T_obs and p EXACTLY" "on the first replicate of every config"; the code checks T only (p cannot match — the reduced ensembles are seed-stream different, as the Gate docstring itself correctly states) and runs once per pack, not per config. Observed on re-run: |ΔT|/T ≈ 5e-16. (b) Docstring's "~90 events, ~15% μ_FP share" vs realized 94 / 12.3% (findings.md reports the realized values). (c) `threshold_study.json` `code_commit` = `13e0d46`, one commit BEFORE the script itself was committed (`90d5b91`) — generated from a then-dirty tree; content is nonetheless tied down (my smoke re-run with the committed code reproduces the guard exactly). (d) Type-I MC half-width ±0.0032 at α=0.01 vs the §15.5 ≤0.002 target — openly disclosed in findings as a limitation, not hidden. (e) `gen_phaseC_resp.py:86` dead variable `by_tid = {}`.

## Per-category verdicts (§21 list)

1. **preimage — CLEAN** (beyond F1's npz). Re-ran a spot check: for bins b∈{4,9,12}, folding a SINGLE-BIN truth through the committed `forward.fold_mu` reproduces `M_bc[b,:]` to rel ≲ 4e-16 — M[b,c] is a faithful column read of the committed fold with exactly the K_full·C_bs·g_bk·f·dN_b·dX weights. Per-bin decomposition G1+G2+G3+below+above+off_lo+off_hi − folded_live_total closes to 2.7e-12 abs (4e-16 rel). Committed values reproduce exactly: G3 total 5685.7536, migration 720.91/4775.31/189.53, clamped share 0.4725.
2. **truth_by_snr — CLEAN.** Only the allocation array swaps (`tc·dX-share → truth_counts_bks`); same K, completeness, g, FP, live mask; no fitting anywhere. Re-run reproduces the committed JSON bit-identically (L1 0.067311; group residuals [−1760.823, 130.464, 450.246] → [−1750.88, 117.917, 452.682]; χ² 22.0940 → 21.9875; max|Δ| = 0).
3. **design_sizing — arithmetic CORRECT** (F6 noted). Recomputed independently: z_.995+z_.90 = 3.8574 → σ_power = 116.72, σ_prec = 150.08; Neyman formula and n_uniform (230) reproduce; totals 2,007 pairs / 2,533 injections / 117.5 CPU-h reproduce; S/W unit conversions from the preimage verified at bin 9 (S=9313, W=436.8).
4. **gen_phaseC_resp — CLEAN.** Refusal verified by execution: `--anchors 19.4` → "REFUSED: anchor 19.4 < 19.5" at parse time, before any file IO; no low-N grid path exists (anchors always explicit, defaults min 19.6). `veto_hcd_neighbors` unit-tested with a synthetic table: km/s window correct, normalized by (1+z_inj) — the SAME convention as the matcher's (1+z_truth), and 5000 km/s comfortably exceeds the ~3000 km/s (dz_rel=0.01) matching window, so unambiguous truth ownership holds. One-injection-per-sightline enforced globally in `build_injection_grid` (`used_targets`, campaign_grid.py:472-479). Roles sidecar keyed by unique inj_id, records dv_excl and n_vetoed.
5. **measure_phaseC_pairs — CLEAN except F3.** The matcher is the production object (`match_truth_to_cat_molly`, dz_rel=0.01, nhi_desc), match-before-cuts exactly as `load_and_cut_catalog` step 5. The `j_by_key` round-6 inversion is collision-safe because TIDs are unique per manifest (one injection/sightline). Completeness denominator = every manifest row (per-anchor n_inj sums to 208/102; accounting 208 = 200 op + 2 sub-op + 6 unmatched verified in the JSON). No cross-arm leakage (arms scored independently). Full re-run on the prodlike arm reproduces the committed `pairs_prodlike.json` **bit-identically** (zero differing fields incl. per_anchor). Pilot input hashes all verify OK. The pilot findings' bridge claim (no anchor-cell |z_vs_old|>2 at n≥5, either arm) and the clamped-region pooled dx (+0.0113 / −0.0039 / −0.0501 at 21.2/21.6/22.0) re-verified from the committed JSONs.
6. **threshold_study — CLEAN except F1/F8.** Both exactness claims hold: (a) the FP-fold linearity M is probed from the committed `fold_mu_fp` on basis vectors with a 1e-9 random-probe guard (and the fold IS linear in n0 by its expression); (b) `group_aggregator` is 0/1 over disjoint contiguous groups (fail-loud on misaligned edges), so cell-level Poisson draws reduce to group-level draws exactly in distribution. The tilt multiplies the TRUE data mean only (`mu_alt`); the analyst's model (`g_mu_obs`) never sees it. Deployed B=2000 and fixed seeds 41001/43001 per replicate; p = (1+#exceed)/(B+1) with #{Tn ≥ T} (searchsorted left). Smoke re-run: guard reproduces committed T (25.984795595071947, |Δ| ≈ 1e-13) and p_committed 0.00049975; committed JSON's headline rates match findings.md exactly.
7. **gate_covariance — CLEAN.** The fallback now returns early: `p_value=None`, `p_mc_error=0.0`, `n_null_draws=0`, no null ensemble, correctly labeled ("NO p-value and NO pass/fail"). The non-fallback path is logic-identical; downstream consumers go through `report()` (closure_table), which handles None. `pytest tests/test_gate_covariance.py`: **8 passed** (17 s), including the updated fallback pin and the poisoned-provenance case. Produced Phase-B numbers unaffected: the preimage's closure sanity re-reproduces the committed closure-table residuals through the new code, and my spot run reproduced the twin's +450.246.
8. **test_modelA_rungs r5 — PASSES; F4 on the guarantee's provenance.** The old under-powered stochastic guard's demotion from non-strict xfail to an explained opt-in skip (with the governance doc as reason) removes no protection — the xfail never blocked — and the new deterministic contract adds a real per-commit blocker, including the `consts.sigma_hat ≡ sig1` plumbing assertion.
9. **docs — CONSISTENT** (F2, F5 noted). Design-doc per-bin counts (b7 108, b8 238, b9 359, b10–13 108, tails 60+60 = 1,257 production pairs; +750 bridge = 2,007; 2,533 injections) match sizing.json exactly; σ=112.0/power 0.926 match; the §9 criteria appear verbatim and unweakened (150.1; power ≥ 0.90 at two-sided α=0.01 ⇒ σ ≤ 116.7 binding; bridge 75-count tolerance); Stage-2 explicitly NOT authorized in design §8, budget, checkpoint, and the pilot launcher header; budget numbers consistent with the FP-expansion doc. The checkpoint's "354 passed + 1 skip" closure-suite figure was NOT independently re-run (full suite exceeds the review's per-check budget); the two suites I did run are green.
10. **independence/holdout — CLEAN except F7.** No committed code reads held-out data (no `mock-1` references outside design text). Pilot data are labeled engineering-validation in the pairs JSONs, findings, checkpoint, and handoff; the suggestive clamped-region observation is explicitly labeled NOT confirmation and no criterion/anchor/endpoint changed in response to it.

## Overall verdict

**PASS-WITH-FINDINGS** — the quantitative artifacts are faithful to the committed machinery (every re-run reproduced committed numbers exactly, several bit-identically), the PI's frozen criteria are nowhere weakened, and the prohibition boundary is enforced in code. The two defects are record-keeping, not science: (F1) untracked-but-claimed-committed npz inputs, and (F2) the checkpoint's premature claim of a committed review record — both must be corrected before the checkpoint is put in front of the PI; F3 and F7 must be closed before any Stage-2 scoring run.

---

## Disposition (added by the implementing session, same day — every item)

* **F1 FIXED:** the four npz force-added to git; committed generator
  `diagnostics_phaseC/threshold_study/make_observed_tilt_shape.py` added
  (regenerates the tilt shape from the committed pack + fold; verified
  identical).
* **F2 FIXED:** this file IS the record; the checkpoint and handoff now
  cite it by name.
* **F3 FIXED pre-Stage-2:** sentinel filter (pre-match) + DLAFLAG==0
  (op-mask) added to `measure_phaseC_pairs.py`; both pilot arms
  re-scored; docstring now enumerates the remaining analysis-window cut
  as an open pre-production item instead of claiming exact fidelity.
* **F4 FIXED:** floor raised to ≥ 3.5 post-shrink (= 56 pre-shrink
  effective counts, above the ≥ 52 bound); docstring states the
  guarantee's actual origin.
* **F5 AMENDED PRE-DATA (PI to ratify):** criterion 1's vacuous CI
  clause replaced by a real dispersion guard — 95% CI upper bound of the
  G3-projected difference < 116.7 counts (the binding production σ
  target). Recorded as a written amendment in the bridge doc; no bridge
  datum existed at amendment time.
* **F6 FIXED:** sizing now sums σ²(G3) over ALL re-measured bins
  (bridge + production + tail): σ = 113.1, power 0.920 — criteria still
  met; design/checkpoint numbers updated.
* **F7 ACKNOWLEDGED (Stage-2 blocker, tracked):** role-enforcement +
  labeled-artifact builder listed as pre-production implementation in
  the handoff's next-step block.
* **F8 FIXED:** (a) docstring corrected (T-only, once per pack);
  (b) realized-regime numbers in the docstring; (c) provenance note
  added to the study findings; (e) dead variable removed.
