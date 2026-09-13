# `validation/absorber_ladder/response_review/antitautology`

VALIDATION-ONLY. **Calibration / mock information only** — no HBI run, no MCMC, no real data, no mock
dN/dX closure number is read anywhere in this package. Nothing here adopts, selects or ranks a response
representation.

Governing authority: `governance/PI_RULING_2026-09-13c_RESPONSE_REPRESENTATION_REVIEW.md` §3/§4 (tests
A–D) and the SEALED `governance/response_review_2026-09-13/RESPONSE_FAMILY_OPENING_RULE_PREDECLARATION.md`
§6, whose thresholds are applied verbatim.

Read `figures/2026-09-13_response_review/antitautology/ANTITAUTOLOGY_REPORT.md` (notes repo) first.

| file | what |
|---|---|
| `opbuild.py` | the generic interface. `build_operator(events, weights, kind) -> Operator` (`.P` = conditional rows `P(c\|b,s,K)`, `.Mg` = the `(S,Kf,C,B)` tensor under the ratified count-conservation rule); the weightings; `operator_from_mg()` for a candidate delivered only as `Mg_<cand>_<fam>.npz`; `register_builder()` for a candidate that ships a fitting adapter |
| `metrics.py` | row KL, boundary-crossing / tail leakage, sightline-clustered paired held-out log-likelihood |
| `invert.py` | the forward fold and the deterministic unpenalised Poisson-MLE (EM) inversion — the update of `validation/absorber_diag/analyze_fold.py:535-560`, pinned element-wise by a test — plus the synthetic f(N) shapes |
| `toys.py` | the MUTATION controls: `toy_imprinted` (must be flagged), `toy_conditional` / `toy_rowfit` (must pass) |
| `audit_occupancy.py` | test D: the executable occupancy probes and the code-level file:line table |
| `run_antitautology.py` | the driver (tests A, B, C, D) |
| `make_report.py` | 4 figures + `ANTITAUTOLOGY_REPORT.md` + `SHA256SUMS` |
| `../../../../tests/test_antitautology.py` | 19 unit + mutation tests |

Reused UNMODIFIED from the first ladder: `r1d_empirical.smooth_along_N` / `to_masses`,
`respfit.fit_variant` and the whole R1c path, `opmetrics.model_masses` (which calls the COMMITTED
`count_conserving_fold.surface_masses` file-directly). `build_r1d` generalises only the count
accumulation of `r1d_empirical.raw_masses:62` (which hard-codes `1.0`) to carry event weights; with
`w == 1` it reproduces `fit_r1d` element-wise with `atol = 0` (tested).

`respfit` has no weighted path end-to-end, so R1c weightings are realised by **weighted resampling with
a fixed seed**; the native R1c arm is resampled the same way so both sides of every R1c comparison carry
matched Monte-Carlo noise, and test B reports a `native_mc` control arm that isolates it.

## Running

```bash
source ~/.bashrc; conda activate gpdla-hbi
export PYTHONPATH=/home/mfho/wt_abs_diag_2026-09 HDF5_USE_FILE_LOCKING=FALSE JAX_PLATFORMS=cpu \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/mfho/wt_abs_diag_2026-09
python -m pytest tests/test_antitautology.py -q
python validation/absorber_ladder/response_review/antitautology/run_antitautology.py \
    --kinds R1d_raw,R1d,R1c,M_true_emp --tests A,B,C,D --n-boot 8
python validation/absorber_ladder/response_review/antitautology/make_report.py
```

## Pushing a NEW low-DOF candidate through A–C

* **Tests A and B need a builder** (they rebuild the operator on reweighted calibration populations —
  an `Mg` file alone cannot be rebuilt). Ship an adapter `.py` defining
  `BUILDERS = {"<cand>": fn}` with `fn(events, w, geom, sel=None, **kw) -> opbuild.Operator`, then

  ```bash
  python .../run_antitautology.py --kinds <cand> --tests A,B,C,D \
      --candidate-builder /path/to/<cand>_adapter.py
  ```

* **Test C works from the delivered tensor alone** (schema `absorber_ladder/Mg_fixed/v1`, dropped into
  `/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/response_review/candidates/`):

  ```bash
  python .../run_antitautology.py --tests C --candidate-glob 'Mg_*_2lpt0.npz'
  ```

Products land in `.../absorber_ladder_2026-09-13/response_review/antitautology/`
(`antitautology_results.json`, `run.log`, `SHA256SUMS`); figures and the report go to the notes repo.
