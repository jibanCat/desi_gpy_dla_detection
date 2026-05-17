# 2026-05-16 — sub-DLA / DLA prior-boundary sweep (3-way, subdla_3way_sweep)

> **Status**: DONE (job 53018972, completed 2026-05-16 01:32).
> **Verdict: confirms the 2-way cellC family is the right production
> choice.** A clean non-overlapping 20.3 split (U3) reaches very high
> purity (0.946) but loses completeness (0.703) — the 3-way models all
> trade away recall, and none beats the 2-way C7/F2 operating point on
> the P/C frontier for the CDDF LLS use case.
>
> Sweep root:
> `/pscratch/sd/j/jibancat/prod533_5k_20260511/subdla_3way_sweep/`

## Design

Four 3-way cells (`SINGLE_ABSORBER_MODEL=0`, separate null / sub-DLA /
DLA channels, MAX_DLAS=3, PW 100k), varying where the sub-DLA prior ends
and the DLA prior begins. London-0 5k slice, fixed molly recipe
(SNR>2, p_DLA≥0.99, lyb-veto, no-BAL, λ_rf∈[911,1216], NHI≥20.3
truth+predicted, n_truth=581).

| Cell | sub-DLA prior | DLA prior | priors overlap? |
|---|---|---|---|
| U0 | [19.1, 20.0] | [19, 22] | yes (19.1–20.0) |
| U1 | [18.0, 20.0] | [19, 22] | yes (19.0–20.0) |
| U2 | [17.2, 20.0] | [19, 22] | yes (19.0–20.0) |
| U3 | [17.2, 20.3] | [20.3, 22] | **no — clean split at 20.3** |

## Results

| Cell | knob | P | C | n_cat |
|---|---|---:|---:|---:|
| U0 | subDLA [19.1,20.0] DLA [19,22] (baseline) | 0.8514 | 0.7276 | 1418 |
| U1 | subDLA [18.0,20.0] DLA [19,22] | 0.8464 | 0.7337 | 1148 |
| U2 | subDLA [17.2,20.0] DLA [19,22] | 0.8380 | 0.7368 | 1174 |
| U3 | subDLA [17.2,20.3] DLA [20.3,22] (clean 20.3 split) | **0.9458** | 0.7028 | 1002 |

## Interpretation

Widening the sub-DLA prior downward (U0→U1→U2) slightly *lowers* purity
and slightly *raises* completeness, but every 3-way cell sits well below
the 2-way frontier on completeness (0.70–0.74 vs C7 ≈ 0.81, F2 ≈ 0.83).
The separate sub-DLA channel siphons evidence away from the DLA channel,
so genuine DLAs near the 20.3 boundary are sometimes classified as
sub-DLAs and drop out of the NHI≥20.3 evaluation — a structural recall
cost of the 3-way design.

**U3 (clean 20.3 split)** is the standout on purity: a DLA prior whose
lower edge *exactly* equals the eval cut (20.3) produces a near-pure
catalog (P=0.946) — almost no NHI<20.3 contaminants survive because the
prior gives them no DLA-channel support. But it is the *worst* on
completeness (0.703) and smallest catalog (n_cat=1002): absorbers truly
near 20.3 straddle the prior boundary and are lost to the sub-DLA
channel.

## Production implication

This sweep reinforces the standing decision (memory
`project_subdla_dla_joint_design`): for a **joint** sub-DLA + DLA
catalog, do **not** use a 3-way [null, sub-DLA, DLA] model. Sub-DLAs and
DLAs can physically co-exist on the same sightline, and the hard
channel split forces an either/or classification that costs
completeness. The production choice remains the **2-way single-absorber
model over [17.2, 22]** (cellC family) with **post-hoc NHI cuts** to
separate the LLS / sub-DLA / DLA regimes — Option B. The 2-way model's
[17.2, 22] catalog is directly usable for CDDF LLS analysis without a
lossy channel gate.

U3's very high purity is worth remembering as a tool: if a *pure-DLA*
(NHI≥20.3) catalog is ever needed in isolation, a DLA prior pinned at
the desired cut is an effective purity lever — at a known recall cost.
