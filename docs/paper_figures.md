# Paper figures — production figure catalogue

The obvious place to find production figures for the GP-DLA papers:
what each figure shows, where it lives, and how to regenerate it.

**Organized by target paper.** The repo's figures span (at least) three
distinct papers plus a body of internal trainer/infrastructure
diagnostics. They are grouped accordingly below — do not treat the whole
catalogue as one paper's figure set.

> **Data policy.** Real-LOA-derived figures (the stacking composites) are
> **local only, gitignored** (DESI collaboration data policy); this
> committed `.md` is the index, the figures are not in git. Mock-derived
> figures (2LPT / London / Saclay) are **committed to git** unless noted.

> **Selection review (2026-05-19).** A figure-collection sweep + an
> independent verdict trimmed this catalogue: the trained-GP and kernel-
> comparison groups are internal diagnostics (not paper figures); the
> τ-EB story panels were over-included; the Voigt LSF sweep is a null
> result; and the main paper's headline figures (dN/dX, f(N,z), Ω_HI,
> α(z)) are **not yet exported** — see Paper 1 below.

---

## Paper 1 — GP-DLA population statistics (the main paper)

The repo's core deliverable: absorber catalogs → dN/dX, f(N,z), Ω_HI
(CLAUDE.md §1, §7). **Most of these figures do not exist as committed
PNGs yet** — they currently live inside the `CDDF_analysis` modules and
the `notebooks/CDDF_*.ipynb` cells and must be exported. This section is
the headline-figure plan, not a list of existing files.

### Headline figures needed (status: to be exported)

| Figure | Shows | Source |
|---|---|---|
| Method schematic | null GP vs DLA GP, Voigt absorption, Bayesian model selection | new — `gpy_dla_detection/plottings/` |
| Example detection | a spectrum + GP fit + (z_DLA, log N_HI) posterior + p_DLA | `plottings/plot_model.py` |
| Mock validation | recovered vs truth log N_HI on the population (n≈50k) | `CDDF_analysis/` + the n=50k runs |
| Completeness / α(z) | completeness & purity vs N_HI; the α(z) calibration curve | `CDDF_analysis/cddf_calibration.py` |
| dN/dX vs z | line density vs redshift (vs Noterdaeme+2012 etc.) | `CDDF_analysis/cddf_mock.py`, `notebooks/CDDF_dNdX_all.ipynb` |
| f(N,z) | column-density distribution function | `CDDF_analysis/calc_cddf.py`, `notebooks/CDDF_fN_z.ipynb` |
| Ω_HI vs z | cosmological H I mass density | `CDDF_analysis/cddf_mock.py` |

### The learned GP forward model

A method paper shows the trained GP itself — the learned mean
μ(λ_rest), the per-pixel variance ω(λ_rest), and the low-rank
covariance kernel K = M·Mᵀ (the emission-feature correlation matrix).
For the **production model only** (`model_epoch_920.h5`):

| Figure | Shows | Data | Committed |
|---|---|---|---|
| production μ(λ) + ω(λ) | the learned mean QSO continuum + per-pixel variance vs rest wavelength | real-LOA training (hyperparameters only — no spectra) | partial — see note |
| production K = M·Mᵀ correlation kernel | emission-feature covariance; block structure at Lyα + major emission lines | real-LOA training (hyperparameters only) | `docs/notes/2026-04-28_v2_3way_compare/correlation_y3_legacy.png` |
| production top-5 eigenspectra (columns of M) | the learned emission-line variability modes | real-LOA training | partial — in `eigenspectra.png` overlaid with v2 |

Note: the production model's μ/ω and eigenspectra currently appear only
*overlaid with v2 variants* (in `mu_omega_overlay.png` / `eigenspectra.png`
under the 3-way/5-way comparison folders) — a clean **single-model**
version should be exported for the paper. The correlation kernel
`correlation_y3_legacy.png` is already a single-(production-)model
figure and is paper-usable as-is.

Regenerate (single production model):
```bash
python examples/diagnose_trained_gp.py visualize \
    --model production:<path>/model_epoch_920.h5 \
    --out-dir docs/notes/<dated>_production_gp_model --n-eigenspectra 5
```

### Methods appendix

| Figure | Shows | Data | Committed |
|---|---|---|---|
| `docs/voigt_demo/voigt_kernel_demo_dl08.png` | bare Voigt vs BOSS-R2000 vs DESI-R3000 profiles at 4 log N_HI values, on the DESI observed-grid spacing (dλ = 0.8 Å) | synthetic | Yes |

Reassures that the LSF-kernel choice does not drive the DLA-regime N_HI
bias (damping wings ≫ kernel core). Regenerate:
`python examples/voigt_kernel_demo.py --dlambda-A 0.8 --out docs/voigt_demo/voigt_kernel_demo_dl08.png`.

---

## Paper 2 — τ-EB empirical-Bayes mean-flux recipe (PR #5)

A per-spectrum empirical-Bayes τ_eff fit that closes the DLA-regime N_HI
bias. Full methodology: `docs/tau_eb_hcd_mask.md`; story docs
`docs/stories/tau_eb_story_{2lpt,london,saclay}.md`.

### Paper-candidate figures

| Figure | Shows | Data | Committed |
|---|---|---|---|
| `docs/tau_eb_hcd_mask_demo.png` | 4-panel recipe walkthrough: spectrum + null GP, standardized residuals, τ-grid log-evidence (naive vs HCD-masked), bias-closure bar (TID 120046865) | 2LPT mock, single target | Yes |
| `docs/story_figures/2lpt_01_canonical_dla.png` | the canonical bias target — production +0.34 dex, τ-EB closes to +0.04 dex | 2LPT mock, single target | Yes |

Regenerate the demo:
`python examples/plot_tau_eb_hcd_mask_demo.py --target-id 120046865 --spec <2lpt-spectra>.fits --zcat <zcat>.fits --truth-z 2.7730 --truth-log-nhi 21.263 --out-png docs/tau_eb_hcd_mask_demo.png`.

### Headline figure needed (status: to be made)

The paper's evidential figure is **population-scale**, not anecdotal: a
Δlog N_HI distribution (baseline vs τ-EB) or a bias-vs-N_HI / bias-vs-z
curve on the n≈49k–50k Phase B runs, plus the τ_factor distribution and
the τ_factor-vs-z_qso trend. These exist only as tables in the story
docs / `docs/notes/2026-04-30_tau_eb_phase_b_5k_2lpt.md` — they must be
plotted.

### Supplementary (limitations)

The other `docs/story_figures/` per-spectrum panels are illustrative, not
evidential — keep them as supplementary, not headline:
- closure montage: `2lpt_02`, `london_01`, `london_02`, `saclay_01`, `saclay_02`;
- false-positive rescue: `2lpt_04_false_positive_rescue.png` (pair with a population FPR number, not alone);
- consolidated failure-mode panel: `2lpt_03_mid_dla`, `london_03_strong_dla`, `saclay_03_dla_persistent_bias` — combine into one "where τ-EB does not help" figure.

Generated by `examples/render_story_figures.sh`.

---

## Paper 3 — Real-LOA absorber metal-line stacking (PR #8)

Folder: `docs/notes/2026-05-15_stack_real_loa_dlas/` (see its `README.md`
for the full file inventory). Real-LOA-derived — **figures local only,
gitignored**. Figures tagged by `--purity` preset: `_high` (P_DLA > 0.97)
and `_marginal` (P_DLA ∈ [0.5,0.7]).

### Paper-candidate figures

| Figure | Shows | Science point |
|---|---|---|
| `stack_control_lls_high.png` | LLS real stack vs z-scrambled control, per metal line | **Decisive false-positive test** — CIV 1548/1551 coherent in the real stack (~35σ vs an empirical null), flat in the control ⇒ LLS detections are real absorbers. |
| `stack_control_subdla_high.png` | same, sub-DLA range | sub-DLA detections are real. |
| `stack_lyman_limit_high.png` | stacked flux around the 911.76 Å Lyman limit + the τ_LL ∝ (λ/912)³ recovery model | **LLS Lyman-limit break recovery** — coherent break at 912 Å, depth rising with N_HI ⇒ genuine optically-thick H I. |
| `stack_metal_zoom_prod_high.png` | per-metal-line zoom, production NHI bins, continuum-normalized | metal-line detections + their N_HI trend. |
| `stack_purity_comparison.png` | marginal vs high purity, pooled low-N_HI (marginal-real / marginal-control / high-real) | **Operating-point false-positive test.** *Pending — produced by `--compare-purity` once the marginal-purity run lands.* |

`stack_control_lls` and `stack_metal_zoom_prod` partly overlap in message
— one may suffice in the paper.

### Diagnostic / QC figures (not paper-headline)

`stack_pseudo_continuum_qc_<p>.png` (continuum-fit QC), `stack_bal_compare_<p>.png`
(non-BAL vs BAL), `stack_lls_diag_<p>.png` / `stack_metal_zoom_lls_diag_<p>.png`
(3 fine LLS bins), `stack_prod/subdla/dla_<p>.png` (raw overviews),
`zhist_<p>.png` (per-bin redshift).

### Data product

`stack_curves_<p>.npz` — cached composites + per-bin pseudo-continuum
`pcont` + provenance. The continuum-normalized stack is `curve / pcont`.

### Regenerate

```bash
sbatch slurm/greatlakes/stack_real_loa.sh                          # high
sbatch --export=ALL,PURITY=marginal slurm/greatlakes/stack_real_loa.sh
python examples/stack_real_loa_dlas.py --compare-purity
python examples/stack_real_loa_dlas.py --plot-only --purity <preset>  # fast re-render
```

Methodology + references: `docs/notes/2026-05-18_stacking_continuum_and_lls_literature.md`.

---

## Internal diagnostics — NOT paper figures

These exist on disk with science-style captions but are
trainer-debugging / infrastructure-validation artifacts. Kept in
`docs/notes/` for the record; **do not put them in a paper.**

- **Voigt LSF sweep** — `docs/notes/2026-04-29_voigt_lsf_sweep/delta_log_nhi_box_{2lpt,london,saclay}.png`. A **null result** on n=18 cherry-picked targets, superseded by the n=5000 Phase B runs; most box-plot cells are n=1–2. If the τ-EB / main paper wants a one-line LSF-null reassurance, make **one** combined panel — these three are not it. `voigt_kernel_demo.png` (dλ=0.15 grid) is the rest-grid twin of the methods-appendix figure.
- **`per_target_scatter.png`** — **dropped** from the catalogue: n=18, superseded, and visually a near-empty 2–3-point scatter with no diagnostic value.
- **Trained-GP model comparison** — `docs/story_figures/trained_gp_models_compare.png`, `trained_gp_models_hyperparameters.png`. Trainer normalization archaeology / β-gradient-bug visualization; 4 of the 5 models are known-buggy v2 variants not used in production. Ref: `docs/notes/2026-05-01_trained_gp_models_comparison.md`.
- **GP forward-model kernel comparison (3-way / 5-way)** — `docs/notes/2026-04-28_v2_3way_compare/` (6 figs) and `docs/notes/2026-04-28_v2_5way_compare/` (8 figs). Training-reproducibility checks ("truth-catalog anti-join works", "GreatLakes ≈ NERSC"); the two sets are mutually redundant. Refs: the `report.md` in each folder. *The **production-model** single-model μ/ω, eigenspectra, and correlation kernel belong under Paper 1 → "The learned GP forward model"; only the multi-model overlay/comparison versions here are diagnostics.*

---

## Adding to this catalogue

New paper-candidate figures: add them under the relevant **Paper N**
section (or start a new one) with a figure→shows→science-point table, the
regenerate command, and the data-policy note if real-data-derived. Keep
trainer / infrastructure diagnostics in the "Internal diagnostics"
section, not under a paper.
