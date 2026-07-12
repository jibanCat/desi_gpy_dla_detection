# 2LPT-0 DLA validation — combined recovery table

R0 = method / injected-2LPT-truth (1.000 = perfect).  z in [2.0, 3.5].

## Reporting conventions (4-lens review, 2026-06-19)
- HEADLINE = HBI **purity_mixture** point (MAP); **loa0** is the conservative
  cross-check. loa0 **Omega is NOT a central value** (42% MAP-vs-MC-median gap;
  Bayesian referee) — report loa0 Omega only as an upper-ceiling diagnostic.
- HEADLINE THRESHOLD = **>=20.3** (canonical DLA; matches Ho+2021/N12). >=20.0
  is shown only LABELLED (it folds in the 20.0-20.3 strong-subDLA octave that
  inflates dN/dX by ~+58% and is systematics-dominated).
- Omega is integrated to the fit ceiling; the f(N) histogram stops at logN=22.0,
  which truncates ~18%% of Omega(>=20.3) vs the literature 10^23 limit. Quote the
  Omega integration upper limit explicitly (literature referee).
- The nuisance-MC [q16,q84] band is a one-sided FP/ceiling DIAGNOSTIC, NOT sigma
  (it excludes the MAP). Statistical error is ~Poisson(N_DLA) ~ 0.6%% at >=20.3.
- Absolute values match Ho+2021 real-SDSS GP (dN/dX 0.058, Omega 7.3e-4) to
  ~10-15%% (literature referee); Omega prefactor audited CORRECT.

## dN/dX

| threshold | Raw feed-forward | HBI (loa0) | HBI (purity_mixture) | truth |
|---|---|---|---|---|
| >=20.0 | 0.818 | 1.101 | 1.049 | 0.08589 |
| >=20.3 | 0.904 | 1.159 | 1.090 | 0.05434 |
| >=20.6 | 1.038 | 1.141 | 1.120 | 0.0304 |

## Omega

| threshold | Raw feed-forward | HBI (loa0) | HBI (purity_mixture) | truth |
|---|---|---|---|---|
| >=20.0 | 1.395 | 1.100 | 1.021 | 0.0006942 |
| >=20.3 | 1.468 | 1.114 | 1.029 | 0.0006288 |
| >=20.6 | 1.614 | 1.115 | 1.035 | 0.0005307 |

## alpha(z)=1/R0 on-mock 'closure' — TAUTOLOGY CHECK (not a validation)

WARNING (Bayesian referee 2026-06-19): applying alpha=1/R0 measured on the
SAME mock forces residual->0 algebraically (calibrated = (truth/point)*point
= truth). This table only confirms the arithmetic is self-consistent; it is
NOT evidence of unbiasedness. The HONEST validation number is the PRE-alpha R0
above (HBI over-recovers ~5-16%). A non-circular test needs cross-mock
(London/Saclay) or held-out-N with fewer DOF than data bins -- documented as
the next step, NOT done here.

| q | threshold | alpha | calibrated | truth | residual |
|---|---|---|---|---|---|
| dndx | >=20.0 | 0.9080 | 0.08589 | 0.08589 | +0.0e+00 |
| dndx | >=20.3 | 0.8629 | 0.05434 | 0.05434 | -1.3e-16 |
| dndx | >=20.6 | 0.8764 | 0.0304 | 0.0304 | +0.0e+00 |
| omega | >=20.0 | 0.9093 | 0.0006942 | 0.0006942 | +0.0e+00 |
| omega | >=20.3 | 0.8974 | 0.0006288 | 0.0006288 | +0.0e+00 |
| omega | >=20.6 | 0.8972 | 0.0005307 | 0.0005307 | +0.0e+00 |

## OPEN ITEM: MAP-vs-MC-median skew (band diagnostic, >=20.0)

| FP | q | MAP | MC q50 | (MAP-q50)/MAP | std/MAP |
|---|---|---|---|---|---|
| purity_mixture | dndx | 9.0097e-02 | 8.1186e-02 | +9.9% | 0.7% |
| purity_mixture | omega | 7.0867e-04 | 6.3668e-04 | +10.2% | 2.2% |
| loa0 | dndx | 9.4585e-02 | 7.7213e-02 | +18.4% | 4.1% |
| loa0 | omega | 7.6336e-04 | 4.4326e-04 | +41.9% | 7.1% |

The nuisance-MC median sits below the MAP (worst: loa0 Omega, -42%).
The headline point estimate / R0 / alpha(z) use the MAP. Whether to
re-center on the MC median (which would flip loa0 Omega to UNDER-
recovery R0~0.64) is the #1 question for the Bayesian/stat referee.
