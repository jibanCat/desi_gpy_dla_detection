# `tutorial_data/` — small mock-only fixtures for the HBI tutorial notebooks

These are the committable, machine-independent fixtures the HBI tutorial
notebooks (`notebooks/HBI_0*.ipynb`) load so they run end-to-end on a fresh
checkout **without** the scratch validation caches, the 1.7 GB posterior
kernel, or any GPFS spectra read.

**Everything here is 2LPT-0 *mock* (injected truth known). No real-survey
(LOA) values.** Safe to commit publicly.

| file | what it is | provenance |
|------|------------|------------|
| `compare_synthesis.json` | The 2LPT-0 mock injection-recovery summary: `table.truth` (injected dN/dX, Ω at logN≥{20.0, 20.3, 20.6}) and `table.methods` with three methods — `raw_feedforward`, `HBI_purity_mixture` (headline), `HBI_loa0` (cross-check) — each giving `dndx`/`omega` → threshold → `R0`/`value`/`ci68`/`std`/`q50`. | Copied from the validation run `hbi_validation_2lpt0/figures/compare_synthesis.json`. |
| `compare_R0_table.md` | The same numbers as a human-readable R0 table, with the reporting conventions (headline = purity_mixture, ≥20.3; band ≠ σ; Ω integration ceiling caveat). | Copied from `hbi_validation_2lpt0/figures/compare_R0_table.md`. |

Reference values at the **≥20.3 DLA headline** (so notebook self-checks can
assert against them): truth dN/dX = 0.05434, 10³Ω = 0.6288; R0 (method/truth)
— raw-FF 0.904 / 1.468; HBI_purity_mixture 1.090 / 1.029; HBI_loa0 1.159 / 1.114.

## Regenerating

Run `./regen_tutorial_fixtures.sh` to re-copy these from the scratch validation
cache (a copy, not a recompute — the provenance is the 2LPT-0 injection
validation run). The notebooks themselves are the live demonstration of the
pipeline; these fixtures only carry the *already-validated* mock summary so the
"destination" figure in NB0 is reproducible offline.
