# Real-data (truthless) closure criterion — PROPOSAL + control results (PI item 4, 2026-08-16)

STATUS: PROPOSED, NOT FROZEN. The freezing decision is returned to the PI
on CONTROL evidence alone (no real-data outcome has been computed or
inspected; the _REAL_TOKENS guards remain in force).

## Proposed statistic (fixed before any real-data application)

For a fitted model on real data (posterior or MAP + calibration priors):

    T/n  =  r^T ( diag(mu_bar) + Cov_psi )^{-1} r  /  n

    r        = obs_counts - mu_bar  on the observed N-hat marginal
    mu_bar   = predictive mean of the forward fold
    Cov_psi  = predictive covariance from the model's OWN calibration
               priors (psi_k_delta ~ N(0, fitcov_sd); psi_c ~ N(0,
               sigma_hat)) -- the kernel-fit + completeness-fit error;
               NO term is fit to the data under test
    binning  = the RATIFIED 0.2-dex reporting basis, edges 19.7:0.2:21.7,
               window [19.7, 21.6]
    PASS     = T/n <= 3   (the ONLY PI-ratified numerical closure number)

On mocks the truth-point version of this statistic is computed by
`kernel_uncertainty_closure.py` (committed, seed-pinned); on real data the
same statistic runs at the fitted point with psi integrated over the same
calibration priors (posterior-predictive form).

## Control results (truth-point, ADOPTED config, clamp=both, 400 draws)

| pack     | groups G1-G3 | 0.2-dex reporting | fine window | fine full grid |
|----------|--------------|-------------------|-------------|----------------|
| 2lpt0    | 1.36         | 4.03              | 2.96        | 4.81           |
| london0  | 3.40         | 4.38              | 2.10        | 7.18           |
| saclay0  | 3.43         | 6.25              | 3.48        | 4.55           |

(fixed-K Poisson-only reference: 19-86 across the same cells — the
kernel-error propagation is an order-of-magnitude effect.)

## Why freezing is a PI decision, not an implementation step

At the DESIGNATED grain (0.2-dex reporting) the proposed criterion FAILS
its own controls (4.0-6.3 vs 3). Freezing it now would predetermine a
real-data FAIL; selecting instead the grain that passes (fine-window, 2/3
mocks; or groups, 2lpt0 clean) BECAUSE it passes would be criterion
shopping. Additionally the propagated variance is a LOWER bound: the
model's fitcov covers only the ORDER-0 mu/sig response coefficients;
higher-order fit error is unrepresented. The PI options, on control
evidence alone:

  (a) keep the reporting-grain criterion; treat its control failure as the
      genuine science issue (response representation / higher-order
      kernel-fit error) to resolve BEFORE real data;
  (b) rule the GROUP-grain predictive criterion (G1-G3) the claim-relevant
      closure test (2lpt0 1.36 PASS; london0/saclay0 3.40/3.43 marginal);
  (c) authorize extending the calibration-uncertainty representation to
      the full kernel-fit covariance (touches the model's prior structure
      => PI matter), then re-run these controls.

NO real-data pack exists; no guard was lifted; POSTERIOR_MEDIAN_CI remains
mock/synthetic-only pending this ruling (item 5 conditionality).
