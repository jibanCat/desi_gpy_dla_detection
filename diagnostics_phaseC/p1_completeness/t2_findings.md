# Tier-2 findings (PARTIAL — session boundary; Tier 2 continues per the handoff)

Level-C/-A results on cached pairs; no reweighting performed yet; no
post-selection covariate used for balancing.

## 1. Blend composition: real, quantified, SECONDARY (Level C)

Natural matched pairs carrying a catalogued 17.2–19.5 truth neighbor
within 3,000 km/s (7.5–8.0% of pairs at every N) show dx elevated by
+0.03…+0.10 dex (e.g. +0.187 vs +0.091 isolated at [19.5,19.8)).
Removing them moves the natural mean by only ~0.005–0.008 — a
quantified composition class for the P1 operator (injections exclude it
by the 5,000 km/s rule), not the offset driver.

## 2. THE reframing result (Level A): the deployed SURFACE misfits its OWN pairs

Old-surface value at the deployed clamped covariate minus the raw
natural pair mean, pair-weighted per range:

| true N | raw pairs | surface | surface − pairs |
|---|---|---|---|
| [19.5,19.8) | +0.099 | +0.170 | **+0.071** |
| [19.8,20.1) | +0.069 | +0.089 | +0.020 |
| [20.1,20.4) | +0.054 | +0.037 | −0.017 |
| [20.4,20.7) | +0.050 | +0.015 | **−0.035** |
| [20.7,21.0) | +0.052 | +0.024 | −0.028 |
| [21.0,21.3) | +0.053 | +0.050 | −0.004 |

The degree-2 + edge-clamp parameterization overshoots at the low edge
and undershoots through the middle of its own calibration data. The
Stage-2A bridge compared injections against THIS surface (the estimand
production actually uses — the comparison was faithful and its FAIL
stands), so a large share of the low-N bridge Δ is the old surface's
representation error, not a pipeline estimand difference.

## 3. The surviving pair-level natural-vs-injected offset

Isolated naturals vs injected bridge pairs: +0.019 (19.6), −0.004
(19.8), ≈+0.03 (20.2), +0.017 (20.4), ≈+0.03 (20.8), **+0.059 (21.0)**
— a noisy +0.00…+0.06 dex offset, largest at 21.0 where BOTH selections
are ≥98% complete and blends are controlled. Remaining candidate
mechanisms (rulings §18, Tier-3 class): (a) host-environment coupling —
natural HCDs sit in correlated forest overdensities that inflate fitted
N̂, injections land at random z (physically expected in LyaCoLoRe, and
can grow with host N); (b) imprint realism — `inject_voigt` reproduces
the GP's OWN profile (M4-validated against the finder), while
quickquasars imprinted the natural absorbers with its own profile
machinery; a profile-shape difference at fixed catalog NHI shifts
fitted N̂. Both are truth-generation/transfer questions the P1 estimand
freeze must bound; neither is testable from catalogs alone
(spectrum-level Tier-3 diagnostics would need a small forced-fit /
stack budget → per rulings §21 that returns to the PI if required).

## Position vs the stopping rule

Criterion 3's condition ("the mechanism does not extend above 20.4")
is NOT met as hoped: the pair-level offset persists to 21.0 at
~+0.03–0.06. Projected on G3 via the preimage sensitivities, a +0.05
coherent pair-level offset over [20.7,21.1) is ~700 counts — MATERIAL.
The investigation therefore continues (Tier 2 completion: per-stratum
paired/common-substrate comparisons + prespecified reweighting on
pre-selection covariates; then the Tier-3 decision). The P1 estimand
cannot be frozen until this offset is attributed or bounded; if it is
host-environment coupling, the INJECTED operator would under-predict
natural N̂ at high N and P1 transfer would need an environment term —
exactly the "injections do not reproduce the dominant natural
mechanism" test the PI flagged (§22).
