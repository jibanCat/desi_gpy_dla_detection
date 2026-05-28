# Finding: v1 Python `objective.py:53` passes z_qso instead of (1 + z_qso) to spectrum_loss

> Discovered 2026-05-07 while preparing Step A.3 of PR #6
> (debug-trainer-from-v1). This is a **larger-impact** bug than the
> `dlog_β` chromatic-correction finding (`docs/notes/2026-05-07_dlog_beta_approximation_finding.md`),
> and it is **not present in MATLAB**. v1 Python production
> `model_epoch_920.h5` was trained with it.

## Summary

`gpy_dla_detection/objective.py:50–54` (v1 Python trainer wrapper) passes
the raw QSO redshift `z_qsos[i]` as the `zqso_1pz` argument of
`spectrum_loss`. The MATLAB sibling at
`/home/mfho/MATLAB/gp_dla_detection_dr16q_public/objective.m:47` does
the conversion correctly: `zqso_1pz = z_qsos(i) + 1`.

Inside `spectrum_loss` the indicator mask is
`indicator = lya_1pz <= zqso_1pz`. With the bug this becomes
`lya_1pz <= z_qso` instead of `lya_1pz <= 1 + z_qso`, which excludes
the bulk of the Lyα forest (any pixel with rest λ above
`λ_lya · z_qso/(1+z_qso) ≈ 0.75 · 1216 = 912 Å` for z_qso=3) from the
data-driven τ₀, β gradients.

Effect: `dlog_τ_0` and `dlog_β` get only the deepest-blue contribution
(below ~912 Å rest), so the priors dominate and τ₀, β stay near their
prior means.

## How the bug entered

MATLAB `objective.m` was historically refactored — see the comment at
line 45-46:

```matlab
% Apr 12: directly pass z_qsos in the argument since we don't want
% zeros in lya_1pzs to mess up the gradients in spectrum_loss
zqso_1pz = z_qsos(i) + 1;
```

The semantics: "rather than passing zeroed lya_1pzs (which has log(0)
issues), pass z_qso and let spectrum_loss apply the indicator". Inside
`spectrum_loss` the expression `lya_1pz <= zqso_1pz` is the indicator.

v1 Python lifted **the comment** without the `+ 1` arithmetic. The code
at `objective.py:50–54`:

```python
nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta = spectrum_loss(
    fluxes[i, valid_mask], lya_1pzs[i, valid_mask], noise_variances[i, valid_mask],
    M[valid_mask, :], omega2[valid_mask], c_0, tau_0, beta,
    num_forest_lines, all_transition_wavelengths, all_oscillator_strengths,
    z_qsos[i],   # ← BUG: should be z_qsos[i] + 1 to match MATLAB
)
```

vs MATLAB `objective.m:47–54`:

```matlab
zqso_1pz = z_qsos(i) + 1;   % ← correct conversion

[this_f, this_dM, this_dlog_omega, ...
 this_dlog_c_0, this_dlog_tau_0, this_dlog_beta] ...
   = spectrum_loss(centered_rest_fluxes(i, ind)', lya_1pzs(i, ind)', ...
                   rest_noise_variances(i, ind)', M(ind, :), omega2(ind), ...
                   c_0, tau_0, beta, num_forest_lines, all_transition_wavelengths, ...
                   all_oscillator_strengths, zqso_1pz);
```

## Why Step A.2 still passed

Step A.2 (`tests/test_v1_matches_matlab.py`) loaded our frozen fixture
where `zqso_1pz = z_qso + 1` was **explicitly precomputed** in
`build_2lpt_frozen_test_fixture.py` and stored as a fixture field.
Both the v1 Python test and the MATLAB driver loaded that field
directly — neither went through their respective `objective` wrappers.
So `spectrum_loss` itself agreed to ~1e-11.

The trainer-wrapper bug only matters at training time, when each
spectrum's `zqso_1pz` is derived per-iteration. That's where v1 Python
diverges from MATLAB.

## Empirical impact

For a typical 2lpt spectrum at `z_qso = 3`:
- Correct: `zqso_1pz = 4` → indicator passes for pixels with
  `lya_1pz ≤ 4`, i.e. rest λ ≤ 1216 Å. The full Lyα forest contributes.
- Buggy: `zqso_1pz = 3` → indicator passes only for pixels with
  `lya_1pz ≤ 3`, i.e. rest λ ≤ 912 Å (the lyman limit). Only the
  deepest blue ~5 % of forest pixels contribute.

Estimated gradient impact:
- `dlog_τ_0` shrinks by O(20×) (only deep-blue pixels contribute)
- `dlog_β` similarly
- `dM`, `dlog_ω` are NOT affected — those gradients don't use the
  indicator.

## Why v1 production trained anyway

The v1 production model was trained with this bug from `learn_qso_model.py`
+ `objective.py` (using DESI Y1 priors `τ₀ ~ N(0.00246, 0.00014²)`,
`β ~ N(3.62, 0.04²)`). With the bug:
- The data-driven gradient on `log_τ₀`, `log_β` is small relative to
  the Gaussian prior gradient.
- The optimizer settles near the prior mean `(τ₀, β) = (0.00246, 3.62)`.
- The Ω-kernel components (`M`, `log_ω`) train normally, since their
  gradients don't depend on `zqso_1pz`.

Functional outcome: v1 production has a "trained Ω-kernel + frozen
mean-flux at prior" model. DLA detection still works (Ω-kernel is the
main GP machinery) but `(τ₀, β)` are basically the Turner+2024 prior
values regardless of trainset. Several observations make sense in
this light:

1. v1 production's β = 3.62 ≈ Turner+2024 mean; minimal data shift.
2. The "_corrected" v2 retrains we audited 2026-05-06 used a
   `dataset.py` that processes the trainset differently — they
   may not have had this bug, but they had a different (also broken)
   trainer.
3. Cross-population mock-trained vs real-trained models look surprisingly
   similar at the (τ₀, β) layer, even though forest opacity differs by
   mock — because none of them were data-fit on (τ₀, β).

## Resolution path

PR #6 keeps `gpy_dla_detection/objective.py` **frozen** as a v1
reference (do not modify). For Step A.3 short-retrain experiments:
- The Python A.3 runner **bypasses** v1's `objective` wrapper and calls
  `spectrum_loss` directly in a loop with `zqso_1pz = z_qso + 1`
  (correct). The "v1 lane" of A.3 means "v1 spectrum_loss kernel used
  correctly in a fixed wrapper".
- v3.5 lane uses the same fixed wrapper but swaps the spectrum_loss
  for `gpy_dla_detection.training_v3_5.objective.spectrum_loss`.
- MATLAB lane uses the legacy `learn_qso_model.m` + `objective.m` + the
  installed minFunc, which has the correct conversion.

Future PR scope (after A.3/A.4 results land):
- Patch `objective.py:53` to `zqso_1pz=z_qsos[i] + 1` AND retrain v1
  production. **Compare to v1 production model_epoch_920.h5**: if the
  fixed model trains different `(τ₀, β)` and gives notably different
  DLA detection performance, this is a production-blocking bug to
  prioritize.
- If `_corrected` v2 retrains had a different trainer that *did*
  include the +1, those models would have learned τ₀, β from data;
  combined with the broken `randn` M init we already documented,
  they likely converged to different fixed points than v1 production.

## Status

- Documented here (this file).
- v1 source kept frozen.
- Step A.3 runs use the fixed trainer to get a meaningful comparison.
- Production fix is a separate follow-up PR after the new v3 trainer
  is validated end-to-end on SDSS DR16 against the MATLAB catalog at
  `/home/mfho/MATLAB/gp_dla_detection_dr16q_public/data/dr16/MATLAB_Catalogue`.
