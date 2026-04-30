# HCD-masked empirical-Bayes τ_eff fit — step by step

> What the recipe does, why each step matters, and how to read the demo figure.
> Companion script: `examples/plot_tau_eb_hcd_mask_demo.py`. Validated on n=18
> DLA targets across 3 mocks; closes 81 % of the median DLA-regime N_HI bias.

## What problem this solves

The GP-DLA forward model uses a fixed mean-flux prior τ₀ ≈ 0.00246
(Turner+2024). On many DESI spectra the actual line-of-sight effective
optical depth differs from this; the fitter compensates by inflating
log N_HI on any DLA in the spectrum. On the historical canonical target
TID 120046865 (truth log N_HI = 21.263) production produces MAP = 21.60,
a +0.34 dex bias.

The recipe below fits τ_eff *per spectrum* on the forest pixels,
**masking out high-column-density (HCD) pixels first** so the fit isn't
biased low by DLA absorption looking like extra forest. This is the
standard Becker / Faucher-Giguère mean-flux convention. The corrected
τ_eff is then used as the production τ₀ for the actual (z_DLA, log N_HI)
inference. No change to the model or to the QMC samples is required.

The critical detail is the masking: a naive empirical-Bayes τ fit
*without* HCD masking only closes ~30 % of the bias. The masking step is
what makes the recipe work.

---

## The four steps

```text
                                  ┌───────────────────────────────────────────────┐
                                  │  Run normal GP-DLA inference at the new τ₀    │
                                  │  (NO HCD mask in this step — it's only for τ) │
                                  │                                               │
                                  │   ➜ MAP log N_HI now ≈ truth                  │
                                  └─────────────────────▲─────────────────────────┘
                                                        │
   τ_factor = τ_best × Turner_default ───────────────── │
                                                        │
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  Step 3 ── τ-grid scan with HCD pixels masked out                            │
   │                                                                              │
   │  for tf in {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}:                                 │
   │      build DLAGPMAT at τ₀ = tf × Turner_default,                             │
   │      pixel_mask = ORIGINAL ∪ HCD_FLAGGED                                     │
   │      max_log_l[tf] = max over an N_HI grid of log L(z_DLA, N_HI)             │
   │  τ_best = argmax(max_log_l)                                                  │
   └──────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │ HCD mask =
                                          │ {residual / σ < −1.5}
                                          │
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  Step 2 ── flag HCD pixels                                                   │
   │                                                                              │
   │  residuals_σ = (y − μ_pred) / sqrt(σ²_total)                                 │
   │  hcd_mask = residuals_σ < −N           ← N = 1.5 worked on canonical target  │
   └──────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  Step 1 ── build null GP at production τ₀ (Turner+2024 = 0.00246)            │
   │                                                                              │
   │  null_gp.set_data(pixel_mask = ORIGINAL)                                     │
   │  μ_pred = null_gp.this_mu                  ← model prediction (incl. A_lyα)  │
   │  σ²_total = null_gp.this_omega² + null_gp.v ← per-pixel variance + GP noise  │
   └──────────────────────────────────────────────────────────────────────────────┘
```

### Step 1 — build the null GP at production τ₀

We need a forward-model prediction of what each pixel should look like
*if there were no DLAs in the spectrum*. That's the null GP at the
production τ₀. Its `this_mu` array is μ × A_lyα — the model continuum
times the mean-flux Lyα absorption — and `this_omega² + v` is the total
per-pixel variance (GP noise + measurement noise).

This step uses the **original** pixel mask (no HCD masking yet); we
need a baseline before we know which pixels are "DLA-affected".

### Step 2 — flag HCD pixels by negative residuals

Compute the standardized residuals r_i = (y_i − μ_pred,i) / √σ²_total,i.
Pixels with r_i ≪ 0 are well below the model; on a clean forest most
pixels should have |r_i| ≲ 2. Pixels with r_i < −N are very likely
"sitting in" a DLA / sub-DLA / LLS trough — they are the HCD-affected
pixels we want to mask.

The threshold N is a tuning knob:
- **N = 1.5** worked on the canonical target. The 3 σ default is too
  conservative because GP per-pixel σ inflates inside saturated regions
  (the (1 − A_lyα) damping factor scales σ down where the mean is
  expected to be near zero), so 3 σ flags zero pixels.
- A separate sensitivity sweep across DLA strength × SNR is open work.

### Step 3 — τ-grid scan with HCD pixels masked

Pixel mask = original ∪ HCD-flagged. With those pixels excluded from
the log-likelihood sum, scan a small τ_factor grid (recipe default
{0.5, 0.75, 1.0, 1.25, 1.5, 2.0} → K = 6). At each τ:

1. Build a DLAGPMAT at `prev_tau_0 = τ_factor × Turner_default`.
2. Set its data with the **HCD-extended** mask.
3. Evaluate `sample_log_likelihood_k_dlas(z_DLA, N_HI)` on a fine N_HI
   grid at the candidate z_DLA (in production: scan over a small
   z-grid covering the absorption window).
4. Record max-over-N_HI log L.

Pick τ_best = arg max over τ. The HCD mask is critical here: without
it, the very pixels that *carry* the DLA information drag τ down — the
fitter sees a deeper-than-Lyα-typical absorption and "explains" it as
extra forest opacity rather than a DLA, locking τ to its production
value or below.

### Step 4 — re-run inference at the chosen τ, **without** the HCD mask

Build the production DLAGPMAT (and SubDLAGPMAT etc.) at the chosen τ,
this time with the **original** pixel mask. The HCD-masking trick was
only needed for the τ-fit step; the actual N_HI inference must see the
DLA pixels (otherwise it would have nothing to infer the column density
from).

The marginal posterior over (z_DLA, log N_HI) under the new τ now peaks
at (or very close to) the truth. On the canonical target the bias goes
from +0.337 dex to −0.063 dex — a flip of sign, which is what tells us
the recipe isn't just trimming the tail of a Gaussian.

---

## Reading the demo figure

> **2026-04-29 fix**: an earlier version of `plot_tau_eb_hcd_mask_demo.py`
> multiplied `null_gp.this_wavelengths` by `(1+z_qso)`, but that attribute
> is already in observed frame (set in `null_gp.py:188`). The double
> redshift made the panel-A x-axis nonsensical. Fixed now; the recipe
> itself was unaffected (numerics use index alignment, never plotted axes).


Run the script:

```bash
python examples/plot_tau_eb_hcd_mask_demo.py \
    --target-id 120046865 \
    --spec  /nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits \
    --zcat  /nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits \
    --truth-z 2.7730 \
    --truth-log-nhi 21.263 \
    --out-png docs/tau_eb_hcd_mask_demo.png
```

It produces a 4-panel PNG (≤ 2 minutes on one CPU core):

- **Panel A — spectrum + null GP**: The data (grey), the null-model
  prediction μ × A_lyα (blue), and the pixels flagged as HCD (red dots).
  Visually the red dots cluster inside the saturated DLA core. This is
  Step 1.
- **Panel B — residuals**: Standardized residuals r_i along the
  observed-wavelength axis, with the −1.5 σ threshold (red dashed)
  and the masked pixels highlighted. Useful for sanity-checking the
  threshold: on a clean forest most pixels should sit between ±2 σ;
  if a large fraction is below −1.5 σ outside the DLA, the threshold
  is too aggressive.
- **Panel C — τ-grid log-evidence**: The "naive" curve (no HCD mask,
  orange) and the "HCD-masked" curve (green). The peak (τ_best) shifts
  upward when HCD pixels are masked: HCDs were holding τ down. On the
  canonical target, naive peaks at τ_factor ≈ 1.5; masked peaks at 2.0.
- **Panel D — bias closure**: MAP log N_HI − truth for three cases:
  production (τ=1.0×, no mask), naive EB, HCD-masked EB. On the
  canonical target the bias goes +0.337 → +0.237 → −0.063 dex.

---

## Notes for production use

- **Cost**: K × inference cost when the recipe is on, where K is the τ
  grid size. K = 6 in the recipe as written. Empirically K = 4 (e.g.
  {1.0, 1.5, 2.0, 3.0}) is enough to bracket the optimum on most
  spectra; K = 3 might suffice on most LOA targets.
- **Where it fails**: One of the n=6 multi-target validation set
  (target 160089646) had production-bias −0.342 dex which τ-EB pushed
  to −0.442 dex. Targets where production is already biased *low* are
  the failure mode; the recipe assumes production is biased *high*
  (the common case). A safe-rollback heuristic is straightforward:
  compute |MAP − production| and if the τ-EB MAP moves the inference
  *further* from a sensible reference, fall back to production.
- **What it does NOT fix**: sub-DLA / LLS regime targets where the
  truth is below log N_HI = 20.0. Those are biased high by the DLA-
  prior boundary at 20.3, not by τ_eff. They are addressed by a
  separate prior-extension + post-cut design (see `feedback_dla_prior_edge_bias`
  in user memory; the current FILTER fix #5 is a partial mitigation).

## Code references

| File | Role |
|------|------|
| `examples/check_tau_eb_robust_mask.py` | Single-target diagnostic (this recipe) |
| `examples/plot_tau_eb_hcd_mask_demo.py` | Companion to this doc — generates the figure |
| `examples/run_multi_target_hcd_mask.sh` | n=6 multi-target driver |
| `slurm/greatlakes/hcd_mask_scale_out.sh` | n=54 SLURM driver (production scale) |
| `docs/notes/2026-04-29_voigt_lsf_sweep/scale_out/summary_n54.csv` | scale-out results table |
| `docs/notes/2026-04-29_bayesian_correctness_synthesis.md` | hypothesis-test summary that motivates this fix |

A production-path implementation under a `--enable-tau-eb-hcd-mask`
flag is in scope for PR #5; see the synthesis doc above for the
shipping plan.
