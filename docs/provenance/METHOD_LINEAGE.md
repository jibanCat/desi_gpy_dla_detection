# Finder method lineage: the baseline is Ho, Bird & Garnett (2020), not the 2021 DR16Q method

**Status:** settled. **Decided:** PI decision 6, 2026-07-29.
**Every claim below is checked against source in this repository; file:line and commit ids
are given so each one can be re-verified.**

## The one-line version

The DLA finder in `gpy_dla_detection/` descends from the **multi-DLA MATLAB pipeline of
Ho, Bird & Garnett (2020)** ([arXiv:2003.11036](https://arxiv.org/abs/2003.11036), *Detecting
Multiple DLAs per Spectrum in SDSS DR12 with Gaussian Processes*). It is **not** the
published Ho, Bird & Garnett (2021) DR16Q method
([arXiv:2103.10964](https://arxiv.org/abs/2103.10964), MNRAS **507**, 704), and no commit in
this repository reproduces that method on the production path. Write
"**Ho et al. 2020 production lineage**", not "Ho 2021 reproduction".

## Fact 1 — the MATLAB history in this repo ends before the 2021 method, and cites only 2020

The MATLAB finder lived in `multi_dlas/`. Its history there:

```
$ git log --format='%h %ad %s' --date=short -- 'multi_dlas/*.m'
16f77c2 2024-10-07 remove .m files
121b4d2 2020-03-31 add the modified main process script; ...
...
$ git log --oneline -3 -- multi_dlas/
16f77c2 remove .m files
178f33b add arXiv number
30d3727 add a README
$ git ls-files multi_dlas/ | wc -l
0
```

The last MATLAB *code* change is `121b4d2` (2020-03-31). The last commit to touch the
directory before deletion is **`178f33b` (2020-04-01, "add arXiv number")**, and its entire
diff is 22 added lines of `multi_dlas/README.md` whose only arXiv identifier is
`2003.11036`:

```
$ git show 178f33b --stat
 multi_dlas/README.md | 23 ++++++++++++++++++++++-
$ git show 178f33b | grep arXiv
+> Spectrum in SDSS DR12 with Gaussian Processes. [arXiv:2003.11036
```

`178f33b` is an ancestor of `HEAD` (`git merge-base --is-ancestor 178f33b HEAD` → true).
The three `.m` files still tracked at `HEAD` are training-parity fixtures
(`tests/matlab/run_spectrum_loss_on_fixture.m`, `tests/matlab/short_retrain_2lpt.m`,
`tests/parity/matlab_parity_check.m`) — not the finder.

The Python package begins at a separate root, **`286b61d` (2020-07-17, "init commit")**.
The whole arXiv-citation surface of the repository is three ids, none of them 2103.10964:

```
$ grep -rIoE '(19|20)[0-9]{2}\.[0-9]{4,5}' --include=*.py --include=*.md --include=*.m .
README.md:2006.07343      # Leah/zQSO estimation
README.md:2003.11036      # Ho, Bird & Garnett (2020)  <- the finder lineage
constants.py:1904.01110
```

**Consequence:** there is no commit here that *is* the published 2021 method, so there is
nothing in this repository against which a like-for-like "Ho 2021 → current" delta could be
computed.

## Fact 2 — the 2021 paper's distinguishing module is NOT on the production path

The 2021 DR16Q paper's stated methodological advance over 2020 is *"an improved model for
marginalising uncertainty in the mean optical depth of each quasar"*. In this repository
that is `gpy_dla_detection/null_meanflux_gp.py` (`NullMFGP`) and its DLA extension
`gpy_dla_detection/dla_meanflux_gp.py`.

The production driver does not use them. `run_bayes_select.py:17-19`:

```python
from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.subdla_gp import SubDLAGPMAT
```

The only importer of the mean-flux module anywhere in the package is
`gpy_dla_detection/dla_meanflux_gp.py:44` (`from .null_meanflux_gp import NullMFGP`), which
is itself not imported by the driver:

```
$ grep -rn 'null_meanflux_gp\|NullMFGP' --include=*.py .
gpy_dla_detection/dla_meanflux_gp.py:12: Extends NullMFGP (null_meanflux_gp.py) ...
gpy_dla_detection/dla_meanflux_gp.py:44: from .null_meanflux_gp import NullMFGP
gpy_dla_detection/null_meanflux_gp.py:2:  null_meanflux_gp.py — Null GP model with mean-flux marginalization.
```

Correspondingly, the three production model modules cite 2020 and nothing later —
`gpy_dla_detection/null_gp.py:41`, `gpy_dla_detection/subdla_gp.py:36`,
`gpy_dla_detection/dla_gp.py:20,42`, `gpy_dla_detection/bayesian_model_selection.py:41`
(*"Reference: Ho, Bird & Garnett (2020), arXiv:2003.11036, Section 3.3"*) — which is the
lineage stated correctly.

**Consequence:** even if the 2021 method were present, describing the deployed configuration
as "the Ho 2021 model" would misname it, because the module that makes 2021 *different from*
2020 is off the path. Marking mean-flux marginalisation as "(Ho 2021 model)" in a list of
*optional, off-by-default* customisations was doubly misleading: the citation was wrong and
the "model" was not the model being run.

## Fact 3 — no like-for-like update claim is available

Per the decision-6 audit: **0 of the 15 finder-algorithm changes since the 2020 lineage are
ablated at the deployed operating point.** No such ablation artifact exists in this
repository. Therefore:

* **Do not write** "we update the Ho 2021 finder", "reproduction of Ho+2021", or any
  quantified "X% better than Ho 2021" statement.
* **Do write** that the finder is a descendant of the Ho et al. (2020) production lineage,
  and describe changes since it *as changes*, unquantified, until an ablation at the
  deployed operating point exists.
* A comparison to the **published** 2021 measurement is still legitimate as a *literature
  overlay* — plotting this measurement against the 2021 CDDF / dN/dX / Ω points, as
  `CDDF_analysis/hbi/track_c_tf_loa.py` does (labelled `Ho21`). That is a comparison of
  *results* against a published paper, not a claim about *method* lineage, and is left
  unchanged.

## Docs corrected under this decision

| file:line | was | now |
| --- | --- | --- |
| `README.md:456` | `- Marginalizing over meanflux for purity (Ho 2021 model)` | names arXiv:2103.10964 as the *paper* whose approach it resembles, and states it is **off the production path** (`run_bayes_select.py` does not import it) |
| `CDDF_analysis/README.md:5` | `## Pathway A — Bayesian posteriors (Bird/Ho+2021 reproduction)` | `## Pathway A — Bayesian posteriors (Bird 2017 / Ho et al. 2020 lineage)` |
| `CDDF_analysis/README.md:30` | `To reproduce Bird (2017) / Ho+2021 CDDF/dN/dX/OmegaDLA plots` | reworded to *plot in the style of* Bird (2017) and the SDSS-era GP-DLA papers, with the no-like-for-like caveat |

## References (web-verified 2026-07-29)

* Ho, Bird & Garnett (2020), *Detecting Multiple DLAs per Spectrum in SDSS DR12 with
  Gaussian Processes* — [arXiv:2003.11036](https://arxiv.org/abs/2003.11036). **The lineage
  of the code in this repository.**
* Ho, Bird & Garnett (2021), *Damped Lyman-α absorbers from Sloan Digital Sky Survey DR16Q
  with Gaussian processes* — [arXiv:2103.10964](https://arxiv.org/abs/2103.10964), MNRAS
  **507**, 704–719. **Not implemented on the production path here.** Its own method
  statement builds on Garnett et al. (2017) and Ho et al. (2020) plus the improved
  mean-optical-depth marginalisation.
* Garnett et al. (2017) — the original GP DLA detection method, upstream of both.

There is **no** standalone "Bird GP-DLA catalog paper"; do not cite one.
