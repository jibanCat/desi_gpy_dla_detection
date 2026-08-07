# P1 FAILURE TAXONOMY + HOLDOUT ADJUDICABILITY GATE — PROPOSED FREEZE

**Status: PROPOSED with `P1_ESTIMAND_SPEC.md`; frozen only on PI
acceptance, and in any case BEFORE any holdout row is read.** Rulings:
the taxonomy is frozen pre-holdout with exactly one primary category
per verdict; the holdout is opened once or not at all; never tuned
after viewing; "no-go" vs "unadjudicated" vs "implementation-blocked"
are distinct outcomes.

## 1. Failure taxonomy (exactly one PRIMARY per verdict)

| # | category | definition (P1 form) | holdout-observable signature |
|---|---|---|---|
| F-est | estimand | the frozen (C_molly, K_natural) pair does not describe one jointly-conditioned object (e.g. matcher-conditioning leak between C and K) | R = C·K identity violated on calibration events; holdout C_inj consistent while joint per-group probabilities disagree |
| F-sup | support | a load-bearing region lies outside validated support (sparse bins, edge inheritance wrong) | failures concentrated in flagged bins / above last populated bin only |
| F-acc | accounting | denominator/eligibility error (live-support mask, dead-strata leak, blend-class mishandling) | holdout completeness off at ALL N coherently; C_inj vs C_molly cross-check breaks |
| F-tra | transition | the validation-overlap construction misleads (overlap-only agreement, disagreement outside) | overlap bins pass while adjacent out-of-overlap bins fail coherently |
| F-pre | predictive | the kernel is well-built but wrong out-of-sample (mean/width drift beyond tolerance) | holdout mean-dx / width tests fail in unflagged bins |
| F-pow | power-adjudicability | the gate lacked power for the defect that matters; holdout read would be uninformative | design-side: power table below threshold — holdout NOT opened |
| F-imp | implementation | builder/guard/versioning defect (double completeness, ID mismatch, non-atomic C/K) | guard tests fail pre-holdout; nothing scientific adjudicated |

Primary-selection rule: walk the table top-down; the first category
whose signature is established is PRIMARY; others are secondary flags.
"P1 failed" may never be reported without the primary category.

## 2. What the holdout can and cannot adjudicate

The 661 held-out injections (13 whole healpix; design counts below)
test the INJECTED operator out-of-sample: C_inj stability, injected
kernel mean/width stability, and — through the frozen transfer map
(spec §7) — the joint-operator overlap statement. They CANNOT
directly adjudicate: (i) the natural-pair kernel (naturals were never
blind within mock-0 — spec §14; its checks are the jackknife and the
overlap); (ii) the imprint-vs-near-field attribution (holdout
injections carry the same `inject_voigt` imprint); (iii) residual
kernel-mean defects ≲0.015 dex (power < ~0.5) — these are carried as
the quantified kernel-uncertainty term in the P1 covariance, NOT
certified absent.

## 3. Design-side power (computed, `p1_holdout_gate.json`, 2026-08-07)

Design counts (generation-time logN_true × roles; no outcomes read):
110/108/84/59/135/75/90 over 19.5–22.4 (all live-SNR); projected
op-matched n_eff 89/97/83/59/135/75/88; per-bin σ(mean dx)
0.0075–0.0116 dex from committed injected robust widths (outer bins:
Tier-1-completeness fallback yields, flagged). Power at α = 0.05
against D-mean-31 (+0.031 dex, the §9 G3-scale defect):
0.83/0.86/0.81/0.76/0.98/0.79/0.83; against D-mean-50 (clamp-scale):
≥0.99 everywhere. Pooled critical window [20.7, 21.1): n_eff = 164,
σ = 0.0069, **power 0.99 (D-mean-31), 1.00 (D-mean-50)**.
Completeness defect (5-point drop): detectable at the committed per-bin
n (binomial, table in the JSON).

**Adjudicability finding (proposed): the holdout IS adjudicable** for
material kernel-mean defects (≥ ~0.03 dex) and clamp-scale failures on
the full support, and for coherent completeness defects; it is NOT
adjudicable for ≤0.015-dex residuals (stated false-pass risk, carried
as systematic). F-pow is therefore NOT triggered at the frozen
materiality scale.

## 4. Frozen holdout test battery and tolerances (evaluated ONCE)

Open-conditions (all before any read): PI acceptance of the spec;
K artifact + guards built; R = C·K identity pass; jackknife stability
pass (spec §11).

Battery (families Holm-corrected together, family α = 0.01):

1. **Per-bin injected kernel mean:** holdout mean dx vs calibration
   injected mean, 7 bins, two-sided z.
2. **Pooled critical window [20.7, 21.1)** mean, two-sided z.
3. **Per-bin completeness:** holdout matched fraction vs calibration
   C_inj, binomial two-sided.
4. **Width:** holdout robust σ vs calibration per bin, F-style
   two-sided at α = 0.01 (diagnostic weight: width failures alone
   trigger review, not automatic no-go, unless > 25%).
5. **Joint-operator overlap** on [20.4, 21.1] via the frozen transfer
   map: per-group landing probabilities incl. miss; χ² at α = 0.01.

Verdict mapping: all pass → P1 holdout PASS (sufficiency within
validated support ONLY — no G3 claim until the gated refold; §28
conditions apply). Any failure → the taxonomy (§1) assigns ONE
primary; no re-runs, no tolerance changes, no reuse of the holdout
for redesign. Underpowered-in-hindsight claims are prohibited — power
was fixed here, pre-read.

## 5. False-pass / false-fail statement

False-fail: ≤ ~1% family-wise under the null (Holm at α = 0.01 over
the mean/completeness families). False-pass: quantified per defect
size in `p1_holdout_gate.json`; materially, defects ≥ 0.03 dex in the
critical window are missed with ≤ 1% probability; defects ≤ 0.015 dex
are likely passed and are carried in the covariance, never certified
absent.

## 6. Storage/read discipline for the eventual open

The one-time read uses `--role held-out-evaluation --evaluation-step`
via the committed measurement path only; no per-event inspection
before the battery statistics are computed and written; the full
battery output is committed in the same run that first touches the
rows.
