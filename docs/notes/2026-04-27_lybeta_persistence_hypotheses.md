# Why does the GP fit a "DLA" on Lyβ even though the Voigt model already includes Lyβ + Lyγ?

The user's question: every DLA's forward model already sums optical
depth from Lyα + Lyβ + Lyγ (`num_lines=3`). So a real DLA at z=2.7 is
modelled with Lyβ absorption at the correct apparent z=2.12, and the
spectrum *should* be fully explained without inventing a second DLA.

Yet, on the 200-target eBOSS sample, ~20–28 % of spurious multi-DLA
detections still match the Lyβ-shifted z of a real DLA on the same LOS.

## Working hypothesis

The forward model is correct. What goes wrong is the **evidence integral
over the M_DLA(2) prior**, not the per-sample fit.

### The mechanism

For the M_DLA(2) hypothesis at QMC sample j the model evaluates

    flux_pred_j = continuum × Voigt(z_dla1, NHI_dla1) × Voigt(z2_j, NHI2_j)

where each Voigt() carries Lyα + Lyβ + Lyγ, and (z_dla1, NHI_dla1) is the
already-known stronger DLA. For the subset of samples with z2_j very close
to `z_lyb_apparent(z_dla1)`, the second Voigt's Lyα coincides — *at the
same observed wavelength* — with the first Voigt's Lyβ. The summed model
optical depth at that wavelength is therefore *double* what's needed to
match the data. To compensate, the fitter pushes NHI2_j toward the prior
edge (~20.3) so that DLA2's Lyα contribution is small, and the redundant
absorption only inflates by a thin sliver.

The per-sample log-likelihood at those (z2 ≈ z_lyb_apparent, NHI2 ≈ 20.3)
samples is therefore *only marginally lower* than for a clean M_DLA(1)
fit — the over-prediction is small at low NHI2.

The Bayes factor M_DLA(2) / M_DLA(1) sums the log-likelihoods over the
QMC samples weighted by the prior. With N=10,000 samples spread over the
full (z, NHI) prior, the few hundred samples near (z_lyb_apparent, ~20.3)
that fit *almost as well* as the M_DLA(1) hypothesis can tip the
marginalised evidence above 1, especially if the rest of the (z2, NHI2)
prior gives uniformly low likelihoods (which depresses M_DLA(2) globally
but not relatively).

### Why FILTER=1 reduces this but doesn't eliminate it

FILTER=1 truncates QMC samples to those above a null-evidence threshold,
re-weighting the integral. The "almost-as-good" samples near
(z_lyb_apparent, 20.3) survive the threshold (their likelihood IS high),
but the rest of the (z2, NHI2) prior — where samples have very low
likelihood and would have lowered M_DLA(2) — gets cut. The net effect is
a *smaller* relative depression of M_DLA(2), i.e. **the relative ranking
of M_DLA(2) vs M_DLA(1) actually shifts the wrong way** for this failure
mode.

But empirically, FILTER=1 produces *fewer* spurious detections (90 vs
225 on the 200-target sample). So the dominant effect of FILTER=1 must
be on a *different* pool of spurious detections (sub-threshold absorbers,
noise spikes), not on Lyβ-confused ones. Indeed the Lyβ fraction of
spurious detections is similar between FILTER settings (19 % vs 22 %)
— FILTER=1 doesn't preferentially fix Lyβ misidentifications.

This is consistent with the recommendation: **catalog-time Lyβ veto is
needed regardless of FILTER**, see `gpy_dla_detection/postprocess/lyb_veto.py`.

## Alternative hypotheses, prioritised by testability

### H1 (most likely): num_lines=3 is enough to model the absorption but the marginal evidence integral still gets confused (above).

**Test:** rerun a few example LOS with `num_lines=6` (Lyα–Lyζ) using the
new `voigt_v2` injector, compare M_DLA(2) evidence at the Lyβ position.
If M_DLA(2) drops further with more lines, H1 is partly wrong (the issue
is residual under-modelling of the higher-order Lyman series). If the
drop is negligible, H1 stands.

### H2: The C-extension's instrument kernel mis-broadens the Lyβ feature, so the model under-predicts the true Lyβ trough depth and "needs" a second DLA to explain the data.

**Test:** apply the same comparison with `voigt_v2 kernel="desi-linear-r3000"`
vs `kernel="boss-log-r2000"`. If the DESI kernel reduces the spurious
M_DLA(2) at Lyβ position, H2 is at play. (H2 is the same physics as the
Y3 +0.37 dex N_HI bias hypothesis we've been investigating.)

### H3: The GP continuum prior is too tight at z_lyb_apparent, so the model "needs" extra absorption to fit a continuum dip there.

**Test:** compare per-pixel residuals (data − continuum) at z_lyb_apparent
between the M_DLA(1) and M_DLA(2) fits. If H3 is true, the residual
should be a continuum-shape error, not an absorption-line shape. The
plotter `plot_smoke_v2.py` already shows `this_mu` (the GP continuum) so
this can be inspected on a few example LOS. Reject H3 quickly if the
GP continuum looks fine through z_lyb_apparent.

### H4: The DLA prior at NHI=20.3 is so heavy (Ho+2020 a=0.97 mixture)
that the marginal weight at 20.3 inflates any near-edge M_DLA(2) evidence.

**Test:** rerun with PW14 prior (`alpha=0.3` or `--alpha 0.3`) on the
same 200-target sample. If the spurious-Lyβ rate drops, the prior shape
is the culprit, not the model.

H1 + H2 are the most likely to actually move the needle. H3 and H4 are
follow-up checks if H1+H2 don't fully resolve it.

## What we *can* claim already (data-backed)

- The Lyβ-explained fraction of spurious detections is 19–28 % across
  FILTER settings on the 200-target sample (≥3 conditions, consistent).
- A catalog-time Lyβ veto (`postprocess/lyb_veto.py`) flagging
  the lower-z, lower-NHI MAP DLA at z within 0.005 of
  0.844·(1+z_high) − 1 catches **all** of these without false-positive
  removals on the 5 unit-test cases.

What we *cannot* claim until tested:

- Whether num_lines or kernel changes reduce the Lyβ-confusion rate.
- Whether the strong-sub-DLA alternative model from Ho+2020 (NHI ∈
  [19.5, 20.0]) reduces this — it's a single-absorber model so by design
  it cannot replace the second DLA in a multi-DLA hypothesis. See
  separate notes on improving the sub-DLA model.
