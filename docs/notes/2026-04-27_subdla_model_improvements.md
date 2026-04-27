# Strong sub-DLA alternative model (NHI ∈ [19.5, 20.0]) — improvement notes

The user's question (paraphrased): *"the multi-DLA mode has an
alternative strong sub-DLA model with NHI=[19.5, 20.0] — why doesn't it
help resolve the Lyβ misidentification, and what improvements would I
suggest? On the CDDF this model biased low for log NHI=20.0–20.3 DLAs
in Ho+Bird+Garnett 2020 (arXiv:2003.11036), and was never tuned for the
best modelling choices."*

## Why the strong sub-DLA model can't fix Lyβ misidentification (as
currently implemented)

The strong-sub-DLA model is structured as an alternative to **Null OR
1-DLA**, not as a component of **N-DLA hypotheses with N ≥ 2**.
Looking at the production indexing in `dlasearch.py:38–43`:

```
DLA run (single_absorber_model=False), num_subdla=1:
    index = 1 + 1 + n  →  SubDLA at [1], DLA(n) at [2+n]
```

So the model competes for the SAME observed feature as the 1-DLA model
— a sub-DLA at z=2.7 vs a DLA at z=2.7. It does NOT contribute to the
multi-absorber hypothesis (1 DLA at z=2.7 + 1 sub-DLA at z=2.12).

The Lyβ misidentification failure mode is multi-absorber: M_DLA(2) is
favoured because *one* of the two QMC absorber slots can land near the
Lyβ-shifted z. The sub-DLA model has no such slot — it's single-
absorber, so it can't out-compete a multi-DLA hypothesis on a LOS that
genuinely has a strong DLA + a Lyβ residual.

## Why it might still bias the [20.0, 20.3] CDDF low

The user observed in Ho+2020 that this alternative model biased dN/dX
low for true DLAs with log NHI in [20.0, 20.3]. Two plausible mechanisms:

1. **Prior overlap.** The DLA-mode QMC sample grid uses NHI ∈ [20.0, 23.0]
   (with the Ho+2020 mixture α=0.97 prior). The strong-sub-DLA
   alternative covers [19.5, 20.0]. There's no overlap. A truth DLA at
   NHI=20.05 has its posterior ~equally well-supported by both models
   — but the 1-DLA Bayes factor will marginally prefer DLA(20.0) over
   sub-DLA(19.99) because the DLA prior is denser at the edge (the
   mixture pile-up at the prior boundary). Yet for the *catalog* this
   produces a 1-DLA-at-20.0 detection that the population statistics
   then count as a sub-DLA after the threshold cut at 20.3. The cut at
   20.3 — applied to a posterior centred at 20.05 — drops the entry,
   biasing dN/dX low in the [20.0, 20.3] bin.
2. **Sample-density imbalance.** The sub-DLA grid was historically
   coarser than the DLA grid (Ho+2020 used 10k QMC samples for both,
   but the sub-DLA prior is narrower in NHI so per-bin density is
   higher). Variance differences in the evidence integral can favour
   the better-sampled model in marginal cases.

Neither has been specifically tested with mocks at GreatLakes scale.

## Suggested improvements

### A. Extend the sub-DLA NHI range to overlap with the DLA prior

Use NHI ∈ [19.0, 20.3] for the sub-DLA model — matching the
conventional sub-DLA boundary, plus a 0.3-dex overlap with the DLA
prior. The overlap is then handled by the Bayesian model selection
explicitly: posterior probabilities at NHI=20.05 get split across the
two models in the right ratio rather than discontinuously assigned to
whichever model has the closer prior centre.

**Cost:** generate a new `subdla_samples_a03_190_203_*.mat`. Test on
the 200-target stratified sample before and after to confirm
[20.0, 20.3] dN/dX recovery.

### B. Make the alternative sub-DLA model multi-absorber

Allow `M_subdla(N)` for N ≥ 1 in the same way as `M_dla(N)`. This
introduces the sub-DLA hypothesis into the multi-absorber competition
where it belongs. A real LOS with a strong DLA at z=2.7 plus a
genuine LLS at z=2.12 would then be modelled by `M_dla(1) +
M_subdla(1)`, not forced into either pure-DLA or pure-sub-DLA.

**Cost:** larger architectural change. ~1 week of refactor in
`run_bayes_select.py` and `gpy_dla_detection/bayesian_model_selection.py`.
Worth it because it would also enable the **LLS-veto coupling** the
user asked about — automatic suppression of "DLA at Lyβ-of-real-DLA"
detections in favour of explicit "sub-DLA / null at Lyβ-position".

### C. Re-tune the alpha mixture weight on the sub-DLA prior

The Ho+2020 a=0.97 was inherited from the DLA prior fit. The sub-DLA
NHI distribution is genuinely shallower than the DLA NHI distribution
(Prochaska & Wolfe 2014 show this). A separate fit of the PW14 mixture
weight on a sub-DLA truth catalog should give a more honest prior. The
existing `gpy_dla_detection.generate_samples` script accepts `--alpha`
and the user can sweep this and check posterior calibration on the
truth catalog.

**Cost:** a one-day calibration run.

### D. Cross-couple the sub-DLA detection into the multi-DLA Lyβ veto

If `M_subdla(1)` posterior at z = z_lyb_apparent is high *while*
`M_dla(1)` posterior at z = z_lyb_apparent is moderate, accept the
sub-DLA hypothesis and reject the DLA. This is a postprocess —
implement in `gpy_dla_detection/postprocess/lls_cross_reference.py`
(already started — see the README in that directory).

The mechanism: a real DLA at z=2.7 makes a Lyβ feature at z_app=2.12.
That feature, modelled in *isolation*, looks like a sub-DLA more than
like a DLA (because the Lyα residual after subtracting the parent DLA
is shallow). So `M_subdla(1) at z=2.12` should naturally have higher
posterior than `M_dla(1) at z=2.12`. Use this to veto the M_DLA(2)
spurious detection at z=2.12.

This is the cleanest, most principled veto — it uses information that
the existing pipeline already produces. Implementation cost is low
because the LLS-mode catalog is already created in the production run.

## Summary table

| Improvement                                      | Cost | Expected gain |
|--------------------------------------------------|-----:|---------------|
| (A) Sub-DLA NHI prior [19.0, 20.3] (overlapping) | low  | Fix [20.0, 20.3] CDDF bias |
| (B) Multi-absorber sub-DLA hypothesis            | high | Right model structure for multi-DLA + LLS LOS |
| (C) Re-tune alpha mixture                        | low  | Better posterior calibration at the boundary |
| (D) Sub-DLA-driven Lyβ veto                      | low  | Catch ~20-30 % of multi-DLA spurious detections |

(A), (C), and (D) can be done independently and are all useful. (B) is
the right long-term refactor but it's more invasive and likely deserves
its own design pass — it touches the bayesian_model_selection module
which is a load-bearing piece of the pipeline.
