# Real-LOA absorber metal-line stacking — output folder

Outputs of `examples/stack_real_loa_dlas.py` (PR #8). For the curated,
paper-oriented index of which figures go in a paper, see
[`docs/paper_figures.md`](../../paper_figures.md).

## ⚠ Data policy

Everything here except the three tracked files below is a **real-LOA-derived
composite** — kept **LOCAL ONLY**, gitignored per DESI collaboration data
policy. Do not commit figures / npz / count tables. See the repo memory
note on real-data privacy.

**Tracked (committed):** `README.md`, `LINE_LIST_REFERENCES.md`, `.gitignore`.
**Local only (gitignored):** everything else — `*.png`, `*.npz`, `counts*.txt`,
`zhist_summary*.txt`.

## What's here

Every per-run output is tagged with the `--purity` preset it came from
(`_high` = P_DLA > 0.97; `_marginal` = P_DLA ∈ [0.5,0.7]). Only
`stack_purity_comparison.png` spans both presets.

| File | What it is |
|---|---|
| `stack_curves_<p>.npz` | cached composites + per-bin pseudo-continuum `pcont` + provenance. The continuum-normalized stack is `curve / pcont`. The reusable data product — load with `numpy.load`. |
| `stack_prod_<p>.png` | production-bin overview (LLS merged + sub-DLA + DLA), raw normalized flux |
| `stack_metal_zoom_prod_<p>.png` | per-metal-line zoom, production bins, pseudo-continuum-normalized |
| `stack_lls_diag_<p>.png` / `stack_metal_zoom_lls_diag_<p>.png` | LLS resolved into 3 fine bins (diagnostic) |
| `stack_subdla_<p>.png` / `stack_dla_<p>.png` (+ `metal_zoom`) | sub-DLA / DLA focus |
| `stack_lyman_limit_<p>.png` | flux around the 911.76 Å Lyman limit + the τ_LL ∝ (λ/912)³ recovery model |
| `stack_control_{lls,subdla,lownhi}_<p>.png` | real vs z-scrambled control — the false-positive null test |
| `stack_bal_compare_<p>.png` | non-BAL vs BAL stacks per NHI bin (diagnostic) |
| `stack_pseudo_continuum_qc_<p>.png` | pseudo-continuum fit QC (diagnostic) |
| `stack_purity_comparison.png` | marginal vs high purity, pooled low-NHI — the operating-point false-positive test |
| `zhist_<p>.png` / `zhist_summary_<p>.txt` | per-NHI-bin redshift distributions (diagnostic) |
| `counts_<p>.txt` | per-bin candidate / non-BAL / BAL counts |
| `LINE_LIST_REFERENCES.md` | the 35-line metal-line list, vacuum λ cross-checked vs Morton 2003 / Mas-Ribas 2017 |

## Reproduce

```bash
# full stack, one preset (~18 min–5 h depending on /scratch I/O):
sbatch slurm/greatlakes/stack_real_loa.sh                          # high
sbatch --export=ALL,PURITY=marginal slurm/greatlakes/stack_real_loa.sh

# re-render figures from a cached npz (seconds, no archive reads):
python examples/stack_real_loa_dlas.py --plot-only --purity high

# the marginal-vs-high comparison figure (needs both npz):
python examples/stack_real_loa_dlas.py --compare-purity

# catalog-only redshift diagnostics (seconds):
python examples/stack_real_loa_dlas.py --zhist-only --purity high
```

Methodology + literature: `docs/notes/2026-05-18_stacking_continuum_and_lls_literature.md`;
design specs in `docs/superpowers/specs/2026-05-{18,19}-*`.
