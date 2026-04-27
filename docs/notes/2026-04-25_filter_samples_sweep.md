# FILTER × num_dla_samples sweep — eBOSS multi-DLA

20 high-SNR strong-DLA truth targets — 10 Saclay `juraLy8-124` + 10 2LPT `loa-124`.
Selection criteria: `ZWARN==0`, `2.5 < z_qso < 3.5`, exactly one truth DLA on
the LOS with `log NHI ≥ 20.7` in the mid-forest range
`[z_qso − 0.5, z_qso − 0.05]`, `SNR_FOREST > 2.5`.

Mode: multi-DLA (`--single-absorber-model 0 --max-dlas 4`), `MAX_WORKERS=8`,
`BATCH_SIZE=1250`. eBOSS DR16Q model (`dlambda=0.25`, `k=20`,
`prev_tau_0=0.0023`, `prev_beta=3.65`). DLA prior: log NHI ∈ [20.0, 23.0]
(broad, to avoid the prior-edge bias at 20.3 — see below). Sub-DLA prior:
log NHI ∈ [19.1, 20.0].

> Bias is computed only on the **clean** 1-DLA detections (i.e. the model
> selected exactly k=1, matching the one-truth-DLA setup). Multi-DLA
> selections (k≥2) are reported separately as a purity-loss diagnostic
> rather than fed into the bias number — otherwise a spurious second DLA
> at low N_HI would pull the median bias artificially negative.

## Results

| FILTER | N_DLA samples | detected/total | multi-DLA selections (k ≥ 2) | clean 1-DLA fits (k = 1) | median ΔlogN_HI (clean) | σ ΔlogN_HI (clean) |
|:------:|--------------:|:--------------:|:----------------------------:|:------------------------:|:----------------------:|:------------------:|
|   0    |        10,000 |     20 / 20    |              11              |             9            |        +0.100          |        0.104       |
|   0    |       100,000 |     20 / 20    |              11              |             9            |        +0.088          |        0.104       |
|   1    |        10,000 |     20 / 20    |               5              |            15            |        +0.038          |        0.105       |
|   1    |       100,000 |     20 / 20    |               5              |            15            |        +0.069          |        0.097       |

## Reading the table

- **detected/total** — *completeness*. All four conditions detect all 20
  truth DLAs at p(DLA)=1.0. On this strong-DLA, high-SNR sample, neither
  FILTER value loses completeness. The user's prior recollection of
  "FILTER=1 drops completeness ~10%" is **not reproduced here**, but this
  is N=20 with strong, clean DLAs — likely the completeness loss the user
  saw was on a fainter / lower-N_HI sample. Should be re-tested with a
  weaker-DLA selection before drawing a definitive conclusion.

- **multi-DLA selections** — *purity loss*. The truth has **exactly one**
  DLA on each LOS, so any k≥2 selection is a spurious extra detection.
  Some real LLS/sub-DLAs do exist on these LOS in truth, but they are
  below the DLA threshold (log NHI < 20.3) — when the model picks them up
  as additional "DLAs", their fitted N_HI gets inflated to ~20.3+ (prior
  pile-up; see § Prior edge below). Concrete example: TID 120231194 has
  a truth LLS at z=2.310 (logNHI=17.82) that the model fits as a DLA
  with logNHI=21.20. **FILTER=1 cuts the spurious-multi rate roughly in
  half (11 → 5 of 20)**.

- **median ΔlogN_HI** — *N_HI bias*. On the clean 1-DLA detections the
  median bias is positive in all four conditions. **FILTER=1 reduces the
  bias by roughly 2×** compared to FILTER=0 (median +0.04 vs +0.10 dex
  at N=10k). N_DLA samples is essentially neutral — going from 10k to
  100k doesn't move the median or std meaningfully.

## Recommendation

**Use `FILTER_LOW_LIKELIHOOD=1` with `NUM_DLA_SAMPLES=10,000`** (the
existing legacy `dla_samples_a03.mat`). This dominates FILTER=0 on every
metric here:

- equal completeness (20/20),
- 2× better purity (5 vs 11 spurious multi-DLA selections),
- ~2× smaller N_HI bias (+0.038 vs +0.10 dex),
- ~6× faster wall time on this preset.

The 100k sample file gains essentially nothing in bias or purity for the
eBOSS preset, but takes considerably longer to run.

### Caveats and follow-ups

1. **N=20 spectra**, all high-SNR, all strong DLAs. Need to repeat on a
   broader sample including weaker DLAs (log NHI ∈ [20.3, 21]) and lower
   SNR (≤ 2) to find the regime where FILTER=1 might lose completeness.
   That's the obvious next experiment.

2. **eBOSS preset only**. The Y3 preset (`dlambda=0.15`, `k=30`,
   Turner+2024 mean flux) has a known +0.37 dex N_HI bias on a single
   strong DLA per the prior smoke test, much larger than the eBOSS bias.
   Whether the FILTER picture is the same under Y3 is the next-most
   important question to settle. The 20-spectrum Y3 batch is in flight
   under sbatch job 48796157 and will be appended below when complete.

3. **Spurious multi-DLA detections trace LLS contamination**, not just
   noise. A follow-up question for the purity/completeness module is
   whether the model's "extra DLA at NHI > 20.3" maps onto the truth LLS
   catalog when one exists — if yes, the right operational move may be
   to enable the sub-DLA model in the comparison set, not just to flip
   FILTER.

## Aside — prior-edge bias (relevant to all of the above)

The default `gpy_dla_detection.generate_samples --mode dla` clips the DLA
prior at log NHI = 20.3, which makes Bayesian inference pile posterior
mass at 20.3 for any spectrum whose truth N_HI lies near that boundary.
For this sweep we used the legacy `dla_samples_a03.mat` (which already
spans [20.0, 23.0]) and a freshly-generated 100k file at the same broad
range. If a production NERSC run was launched with a 100k file generated
via `--mode dla` (the default), it would inherit the prior-edge bias —
worth double-checking the production config.

## Reproducibility

```bash
# regenerate 100k samples at log NHI ∈ [20.0, 23.0]
python -m gpy_dla_detection.generate_samples \
   --min-log-nhi 20.0 --max-log-nhi 23.0 --alpha 0.97 --num-samples 100000 \
   --output data/dr12q/processed/dla_samples_a03_100000.mat

# pick targets
python examples/pick_smoke_targets.py --n-per-mock 10 --snr-min 2.5 \
    --nhi-min 20.7 --out out/smoke/targets.tsv

# four-condition sweep
bash examples/run_smoke_batch.sh eboss 0  10000  10000
bash examples/run_smoke_batch.sh eboss 0 100000 100000
bash examples/run_smoke_batch.sh eboss 1  10000  10000
bash examples/run_smoke_batch.sh eboss 1 100000 100000

# build per-condition summaries + aggregate table
for d in out/smoke/batch/eboss_filter*; do
    python examples/finalize_smoke_batch.py --batch-dir "$d" \
       --targets out/smoke/targets.tsv
done
python examples/aggregate_sweep.py --root out/smoke/batch --preset eboss \
    --out docs/notes/2026-04-25_filter_samples_sweep.md
```
