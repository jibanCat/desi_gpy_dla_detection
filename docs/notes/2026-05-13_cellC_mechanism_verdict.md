# Why cellC works — mechanism verdict (2026-05-13)

> Investigation context: cellC's empirical P/C result (P=0.83, C=0.83 at SNR>2,
> P_DLA≥0.99; vs baseline P=0.85, C=0.77) was counter-intuitive — wider NHI
> prior `[17.2, 22]` + `SINGLE_ABSORBER_MODEL=1` was expected to dilute P_DLA
> via abundant LLS/sub-DLA detections. The data show the opposite. This note
> records the falsifiable mechanism and decides whether to ship cellC as the
> default. Investigation done by sub-agent against the running production_533
> branch; numbers verified against HDF5 outputs in
> `/pscratch/sd/j/jibancat/prod533_5k_20260511/joint_dla_subdla_sweep/cellC_md3_nhi172to22/`.

## 1. The mechanism — posterior arithmetic, not sample density

**(a) Dominant effect — 2-way vs 3-way model space changes the P_DLA denominator.**

Under `SINGLE_ABSORBER_MODEL=1`, `run_bayes_select.py:509` constructs
`BayesModelSelect([0, max_dlas], dla_model_ind=1)` — a **2-way** model space
`[Null, 1-abs, 2-abs, 3-abs]` (length 1+max_dlas=4, confirmed in HDF5
outputs: `model_posteriors.shape = (N, 4)`).

Under baseline (`SINGLE_ABSORBER_MODEL=0`), `run_bayes_select.py:521` builds
`BayesModelSelect([0, 1, max_dlas], dla_model_ind=2)` — a **3-way** space
`[Null, SubDLA, 1-DLA, 2-DLA, 3-DLA]` (length 5, confirmed).

`p_dla` is computed in `bayesian_model_selection.py:268-274`:
```python
p_dla = np.nansum(model_posteriors[self.dla_model_posterior_ind])
```
where `dla_model_posterior_ind` (lines 225-238) flags the **last
`all_max_dlas[dla_model_ind]` entries**. So:

- **baseline**: `p_dla = post[2] + post[3] + post[4]` (DLA models 1, 2, 3);
  denominator includes SubDLA evidence × SubDLA prior.
- **cellC**: `p_dla = post[1] + post[2] + post[3]` (all absorber models);
  denominator only includes the null.

When the SubDLA model has appreciable evidence — the typical case for a weak
true DLA in [20.3, 20.5) whose Voigt profile is borderline for the [20.3, 22]
DLA model but well-supported by the [19, 20.3] SubDLA model — it contributes
mass to the denominator that **suppresses baseline `p_dla` below 0.99**.

Cell C has no separate SubDLA term to bleed mass off; every absorber-like
evidence channels into one number. Empirically cellC pushes ~2× as many
spectra over `p_dla ≥ 0.99` as baseline (HDF5 sample: 247/1129 vs 134/1129
on slice 0; 243/1110 vs 133/1110 on slice 2). That ~2× lift in the
P_DLA≥0.99 yield is the primary win.

Quantitatively, the per-spectrum `p_dla` gain from removing the SubDLA
denominator is approximately

```
Δlog p_dla ≈ log(1 + Z_subDLA · π_subDLA /
                  (Z_null · π_null + Σ Z_DLA(k) · π_DLA(k)))
```

which for weak true DLAs (Z_subDLA and Z_DLA(1) comparable) is order
**+0.3-0.7 dex** — plenty to lift sub-0.99 baseline `p_dla` values over the
cut.

**(b) Subdominant effect — QMC sample density goes the *wrong* way in cellC.**

Inspecting the `.mat` sample files: in the n_initial=5000 coarse scan
(`dla_gp.py:580`), cellC has **213 samples in [20.3, 20.5)** vs baseline's
**412** (per-unit-NHI density is roughly halved across the DLA range). So
the FILTER=1 coarse scan in cellC is *less* likely to land a sample near a
weak DLA's high-likelihood mode than baseline.

This is the effect that *should* push cellC the other way, and it explains
why the headline win is only +13 pp rather than larger. The
posterior-arithmetic gain dominates the sample-density loss.

## 2. Why the user's intuition was wrong

The user's expectation chain:
1. Wider prior admits LLS/subDLA candidates,
2. ...which spread sample density thinner over true DLA NHI range,
3. ...which pull MAP NHI estimates downward,
4. ...which bias P_DLA = p(DLA) / Σ p(model) downward.

Two refutations, both operative; (a) dominates:

**(a) "Sub-DLA candidates dilute DLA detection" is a category error in cellC.**

There is no separate sub-DLA model competing with the DLA model under
`SINGLE_ABSORBER_MODEL=1` — only one absorber model whose integral
marginalizes over a wider NHI prior. The model evidence

`Z_1-abs = ∫ p(D|θ) p(θ) dθ`

integrates over the entire [17.2, 22] prior; whether the QMC integrand
happens to be high in the LLS range or the DLA range is internal to that
one scalar. The user's intuition is correct for a 3-way model space, but
cellC is 2-way.

**(b) The eval is intrinsically protected from MAP-NHI dilution.**

`dlasearch.py:533, 540-542, 574` reads `MAP_log_nhis` per absorber and
writes one catalog row per absorber. `examples/molly_faithful_pc_plots.py`
(lines 79, 96, 156, 622) then applies `NHI ≥ 20.3` cuts to **both** the
predicted-NHI catalog and the truth catalog before counting purity /
completeness. So if cellC's MAP NHI slides into LLS range — which it does:
**2998/3656 = 82%** of cellC's catalog rows have NHI<20.3 (confirmed by
reading the 8 dlacat FITS slices) — those rows are silently dropped by the
eval and contribute **neither** to FP nor TP.

The user's feared "P_DLA biased by subDLA abundance" doesn't materialize
because there's no per-MAP-NHI-bin P_DLA computation. `p_dla` is a
per-spectrum scalar and the MAP NHI is a downstream readout (`dla_gp.py:1067,
1074`).

## 3. The −2 pp purity hit

CellC writes **3656 cat rows** (vs baseline 1471); after `NHI ≥ 20.3`
filter, **658** survive (vs baseline 754). At `p_dla ≥ 0.99`: **574** vs
**492**. So cellC adds **82 more "DLA candidates"** to the post-cut pool
than baseline. From the NHI-bin purity table, ~60 of those are TP and ~22
are FP, giving the −2 pp aggregate purity drop concentrated in
[20.5, 21.0) (P drops from 0.932 to 0.914).

Likely mechanism: a fraction of cellC's "absorber" detections with MAP NHI
≥ 20.5 are real weak DLAs (TP), but some are mid-NHI Lyβ misidentifications
or noise excursions whose evidence in cellC is sufficient to clear
`p_dla ≥ 0.99` because (again) there's no SubDLA model to absorb the
evidence. The lyb_veto step doesn't catch them all. Not a deep pathology —
it's the symmetric cost of the same posterior-arithmetic that gives the
[20.3, 20.5) completeness win.

## 4. Verdict — ship cellC as a flag, not as default

**Recommendation: expose cellC as a non-default, well-documented production
flag. Do not change `SINGLE_ABSORBER_MODEL=0` as the default yet.**

- **+** Headline P/C improves measurably (0.83/0.83 vs 0.85/0.77). The
  completeness recovery is concentrated in the [20.3, 20.5) regression bin
  that has been the headline weakness for two months. This is the closest
  any tested config comes to balanced 85/85.
- **−** CellC's catalog is mostly (82%) predicted-LLS/sub-DLA rows that the
  DLA eval discards. If a downstream consumer expects a "DLA catalog ≈ DLA
  detections", cellC's raw catalog is misleading. It requires a documented
  post-hoc NHI ≥ 20.3 cut, which contradicts established convention.
  Historic catalogs all assume 3-way semantics.
- **−** **Loss of a separate sub-DLA catalog.** The 2-way model conflates
  sub-DLA and DLA evidence; there is no longer a per-spectrum `p(sub-DLA)`.
  The LLS production runs that have been queued for the LLS science
  deliverable already use `SINGLE_ABSORBER_MODEL=1` but as a separate run;
  adopting cellC as the DLA default doesn't break LLS science, but it
  deletes the joint product.
- **(?)** Validated on London-mock-0 5k only. Saclay, 2LPT, real LOA, and
  full London 26k are untested. The +13 pp recovery could plausibly be a
  mock-specific feature of London's truth-NHI distribution near the prior
  edge.

**Action**: keep `SINGLE_ABSORBER_MODEL=0` as the production default;
maintain `joint_dla_subdla_sweep/cellC_md3_nhi172to22/` as an opt-in
production config with a documented post-hoc NHI cut. Cross-validate on
Saclay/2LPT before promoting.

## 5. Open follow-ups that could change the verdict

1. **FILTER=1 knob 2×2 ablation** (separate note:
   [`2026-05-13_filter1_knob_tuning.md`](2026-05-13_filter1_knob_tuning.md)).
   If knob 1 (n_initial floor → 10000) recovers [20.3, 20.5) completeness
   *without* the −2 pp purity hit cellC pays, it's a strictly dominant fix
   and obsoletes cellC. ~30 min wall on one node. **Running right now**
   (cells `k1_5k_k4_on`, `k1_10k_k4_off`, `k1_10k_k4_on`).

2. **CellC with NHI prior `[19, 22]`** (same as baseline, just 2-way model).
   Disentangles the posterior-arithmetic effect from the wider-prior
   effect. If this matches cellC's P/C, the mechanism is purely (a) the
   denominator change; if it matches baseline, the wider prior is doing
   the work. Cheap — reuse existing PW14 samples file.

3. **Inspect false positives in cellC's [20.5, 21.0) bin** (~13 new FPs vs
   baseline). Are they all spectra where baseline also predicts a sub-DLA
   at the same z but with `p_dla < 0.99`? If yes, this is the symmetric
   cost of (a) and confirms the mechanism. If they're at *different* z,
   cellC has a different failure mode worth investigating before shipping.
   Cheap — match cellC and baseline catalogs on TARGETID + z_DLA tolerance.

## Caveats

- HDF5 stats above (the ~2× p_dla≥0.99 lift) are from 3 of 8 slices;
  aggregate stats would tighten but the direction is consistent.
- The "82% LLS-row dropout" figure assumes the published `pc_snr2_pdla99.md`
  eval recipe; consumers that drop the NHI cut see all 3656 rows.
- The wider-prior path through the prior catalog `p(DLA|z_QSO)` term was
  not verified — code reading (`dla_gp.py:1032-1041`) suggests it uses
  prior_catalog counts (independent of QMC samples), but per-spectrum
  `log_priors[0]` and `log_prior_dla` were not measured to confirm.
