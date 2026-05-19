# Paper figures — production figure catalogue

The obvious place to find production figures for the GP-DLA papers:
what each figure shows, where it lives, and how to regenerate it.

> **Data policy.** The real-LOA stacking figures are real-LOA-derived
> composites — **local only, gitignored** (DESI collaboration data
> policy). This committed `.md` is the *index*; the figures themselves
> are not in git. Regenerate them locally with the commands below.

---

## Real-LOA absorber metal-line stacking (PR #8)

Folder: `docs/notes/2026-05-15_stack_real_loa_dlas/` (see its `README.md`
for the full file inventory). Figures are tagged by `--purity` preset:
`_high` (P_DLA > 0.97) and `_marginal` (P_DLA ∈ [0.5,0.7]).

### Paper-candidate figures

| Figure | Shows | Science point |
|---|---|---|
| `stack_control_lls_high.png` | LLS real stack vs z-scrambled control, per metal line | **Decisive false-positive test** — CIV 1548/1551 coherent in the real stack (~35σ vs an empirical null), flat in the control ⇒ the LLS detections are real absorbers. |
| `stack_control_subdla_high.png` | same, sub-DLA range | sub-DLA detections are real. |
| `stack_lyman_limit_high.png` | stacked flux around the 911.76 Å Lyman limit + the τ_LL ∝ (λ/912)³ single-absorber recovery model | **LLS Lyman-limit break recovery** — a coherent break at 912 Å, depth rising with N_HI ⇒ genuine optically-thick H I. |
| `stack_purity_comparison.png` | marginal-purity vs high-purity, pooled low-N_HI (3 curves: marginal-real / marginal-control / high-real) | **Operating-point false-positive test** — does the marginal (P_DLA 0.5–0.7) tail show coherent CIV like the high-purity stack, or flat like its control? |
| `stack_metal_zoom_prod_high.png` | per-metal-line zoom, production NHI bins, continuum-normalized | metal-line detections + their N_HI trend. |

### Diagnostic / QC figures (not paper-headline)

| Figure | Shows |
|---|---|
| `stack_pseudo_continuum_qc_<p>.png` | masked-spline pseudo-continuum fit quality, per bin |
| `stack_bal_compare_<p>.png` | non-BAL vs BAL stacks per NHI bin |
| `stack_lls_diag_<p>.png`, `stack_metal_zoom_lls_diag_<p>.png` | LLS resolved into 3 fine bins |
| `stack_prod_<p>.png`, `stack_subdla_<p>.png`, `stack_dla_<p>.png` | raw overviews |
| `zhist_<p>.png` | per-NHI-bin redshift distributions |

### Data product

`stack_curves_<p>.npz` — cached composites + per-bin pseudo-continuum
`pcont` + provenance. The continuum-normalized stack (model overplots,
EW measurement) is `curve / pcont`. `numpy.load` it directly.

### Regenerate

```bash
sbatch slurm/greatlakes/stack_real_loa.sh                          # high
sbatch --export=ALL,PURITY=marginal slurm/greatlakes/stack_real_loa.sh
python examples/stack_real_loa_dlas.py --compare-purity
# re-render figures from a cached npz (fast):
python examples/stack_real_loa_dlas.py --plot-only --purity <preset>
```

Methodology + references: `docs/notes/2026-05-18_stacking_continuum_and_lls_literature.md`.
Scientific verdict on the LLS detections: see PR #8 description and the
session handoff.

---

## Adding to this catalogue

When a future analysis produces paper-candidate figures, add a section
here: the folder, a paper-candidate table (figure → shows → science
point), the regenerate command, and the data-policy note if the figures
are real-data-derived.
