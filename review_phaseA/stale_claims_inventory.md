# Stale claims & artifacts inventory — Phase A (2026-08-05 adversarial review)

REVIEW-ONLY (Phase A). This inventory records where superseded, retracted, or
pre-repair claims still stand un-annotated. Nothing here has been edited yet;
corrections are Phase-B actions gated on the frozen Phase-A verdict.

Reviewed state: `hbi-mcmc-threeroute` @ `9d73365`, `lls-subdla-cddf` @ `1533333`,
notes repo @ `c7f1a0c`. The five formally retracted claims (checkpoint §3/§14):
(R1) the "sharper bound"/"attainable efficiency", (R2) the one-sided BAL effect,
(R3) the `a_fp` Δdeviance verdict (+2519/+3478/+3089 "on one dof"),
(R4) the cosine-similarity identifiability statistic, (R5) the 10–35× margin range
(superseded by post-repair 9.80×–35.53×).

## A. Public, committed locations

| # | location | stale content | status |
|---|---|---|---|
| A1 | GitHub issue #30 **body**, "The scientific blocker" §4 | R3 verbatim: "decisively prefers ... Δdeviance +2519/+3478/+3089 on one dof"; also "over-predicts the held-out mocks by 23–30%" (superseded by the §7 re-measurement) and "measured wrong by up to 35%" (superseded by the two-supports resolution, p=0.0010) | Correction exists only in a later issue comment; the body itself is uncorrected |
| A2 | `docs/CHECKPOINT_2026-08-05_kernel_fp.md:154` (hbi-mcmc-threeroute) | R3 verbatim, in a committed doc with no forward-pointer to the retraction | Live |
| A3 | `docs/PI_CHECKPOINT_2026-08-05_kernel_fp_identifiability.md` §8 | "Δdev = 41, i.e. 0.6σ against survey Poisson noise" and the "16 of 75 within 1°" framing — both under review this session | Pending Phase-A verdict |
| A4 | `CDDF_analysis/hbi_mcmc/reporting.py:159` + `RESPONSE_ANCHOR_MEASURED` | presents `resp_N_fit_range` as a measured physical boundary; established to be a binning knob (checkpoint §4, handoff §5.2) | Live documentation defect, known and deliberately held |
| A5 | Committed artifacts predating repair `7707c8e`: `adopted_config_closure.json` (427 chi2_dof entries), `spectral_window_study.json` + `_bw0p2.json` (147 each), `d1_basis_pad_ladder.json` (66), `rung9_forward_selftest.json` (6), `posterior_synthetic_smoke.json` (2) | every number in them predates the repaired fold | Deliberately not regenerated; any quotation from them is pre-repair |

## B. Private notes (jibanCat/desi_gpy_dla_notes)

| # | location | stale content |
|---|---|---|
| B1 | `notes/2026-07-30_PI_CHECKPOINT_window_study_and_closure.md` | pre-repair closure numbers throughout (χ²/dof 56.58/40.16/44.21 as current; 13–19× margins) |
| B2 | `notes/2026-08-05_kernel_fp_audit_findings.md` | mid-session snapshot; contains R3 (later retracted) and pre-referee FP claims |
| B3 | `notes/2026-07-02_session_handoff.md`, `notes/2026-07-02_real_loa_bal_arbiter_design.md` | early BAL-veto reasoning; R2 retracted it (no BAL signal at 27× statistics) |
| B4 | `notes/2026-05-17_pdla_cut_sweep.md` | quotes numbers in the R5 family (flagged by sweep; low priority, historical) |
| B5 | The 2026-08-05 handoffs themselves | quote "0.6σ" and the identifiability strength under review; each already flags its own doubt — annotate after verdict, do not rewrite |

## C. Manuscript

| # | location | finding |
|---|---|---|
| C1 | `hbi/main.tex` + `hbi/sections/S1–S9` (notes repo, last built 2026-06-22) | **Generation-level divergence, not line-level staleness.** The manuscript documents the predecessor catalog-HBI v3x pipeline: per-object rate likelihood (S2 eq:logL, eq:mu_fp_pm/eq:mu_fp_loa0), reporting limits (20.0, 20.3), 0.1-dex bins (S7), v3x SBC/validation claims (S9). The currently adopted Model-A forward fold (window [19.7,21.6], 0.2-dex latent basis, pack-level Poisson cell counts, loa-0 FP product with `ell_eff`) appears nowhere. Its equations are internally consistent for the model they describe — they are not "wrong equations" to patch, they are the wrong model generation. |
| C2 | consequence | Synchronizing the manuscript to the current model = adopting Model A as the paper's method — a PI-scope decision (Paper-1 scope). Pre-ruling, the only safe manuscript action is an explicit banner/inventory marking Methods as describing the v3x pipeline, plus marking claims invalidated regardless of model choice (e.g. any "validated on three mocks" nuisance-transfer language). |
| C3 | infrastructure | The `\correction`/`\claude`/`\unfinished`/ledger marker conventions and a Draft-vs-Submission split named in the session brief do **not exist** in either manuscript tree (`hbi/`, `notes/2026-06-29_hbi_cddf_draft/`). If Phase B proceeds to manuscript work, creating them is a mechanical prerequisite. |

## D. Verified-present load-bearing state (for completeness)

The six untracked load-bearing files documented in `.gitignore:130-152` are all
present in the primary checkout (four retired identities + two crossmock JSONs).
The two lineages remain unmerged. Probe archive
`session_2026-08-05_probe_code_and_outputs.tar.gz` extracted and readable.
