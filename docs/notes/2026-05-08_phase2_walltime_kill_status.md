# 2026-05-08 — SLURM 49671617 walltime-killed at iter 175

> Status snapshot for the per-spectrum 89k×200 production retrain
> (the run that motivated PR #6's checkpoint patch and Step B
> vectorization). **Vec full (49700040) is the authoritative Phase 2
> endpoint** — see `docs/notes/2026-05-09_phase2_vec_full_vs_matlab.md`.

## Outcome

| | value |
|---|---|
| SLURM job | 49671617 |
| State | **TIMEOUT** at exit code 0:0 |
| Walltime | 24h00m01s (1-00:00:01) |
| Started | 2026-05-08 13:07:35 |
| Killed  | 2026-05-09 13:07:36 |
| Last logged iter | **175 / 200** |
| Last logged loss | 110,601,826 (relative drop 2.65% from iter 0) |

Trainer pre-dates commit `916b5aa` (checkpoint/resume support) so the
in-flight state was lost on TIMEOUT. There's nothing on disk at iter
175 to resume from.

## Why we don't care anymore

Vec full 49700040 (PR #6 production retrain on `--vectorized=1`) ran
the same 89k×200 problem to completion in 8h03m on 2026-05-09 ~09:10.
It's now the authoritative Phase 2 endpoint:

| param | per-spec @ iter 175 (TIMEOUT) | vec full @ iter 200 (COMPLETED) |
|---|---:|---:|
| τ_0 | 0.004567 | 0.004488 |
| β | 3.0108 | 3.0263 |
| c_0 | 0.107640 | 0.106198 |
| loss | 110,601,826 | 110,596,952 |
| iter | 175 | 200 |

Both runs converge to the same valley in the GP loss; iter-175 vs
iter-200 is just the Adam trajectory continuing slightly. The
per-spectrum run was effectively at convergence already (loss
decreasing 0.001%/iter at iter 175). Detailed comparison in
`docs/notes/2026-05-09_phase2_vec_full_vs_matlab.md`.

## What this confirms about Step B

The fact that 49671617 walltime-killed exactly where the projection
said it would (iter ~178, hit at 175) **is itself the headline
production-time evidence for the 3.05× speedup**: vec full started 12
hours later than 49671617 and **finished 5 hours earlier**, on an
identical problem. Same data, same Adam, same priors, same loss
trajectory to ~1e-6 relative throughout — only the inner gradient loop
differs.

## Side effect: per-iter rate degraded over time

Per-iter rate on the per-spectrum path drifted from ~448s at iter 0
to ~697s by iter 175 — a 56% slowdown over the run. Not seen on the
vectorized path (vec full stayed at ~143s/iter throughout). Possible
cause: thread-storm contention with neighbour jobs on gl3025 as the
node filled up during the 24h window. Notable but not blocking — the
vectorized path naturally avoids this regime by using fewer, larger
BLAS calls.

## Loop termination

Task tracking: tasks #1 and #5 closing here. Tasks #3 and #4
(validation + findings doc) are covered by the vec full work
(`c8116e3`, `1c45986`).
