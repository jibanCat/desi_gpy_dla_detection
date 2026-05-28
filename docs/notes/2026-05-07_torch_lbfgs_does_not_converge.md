# Finding: `torch.optim.LBFGS` doesn't converge on this GP loss

> Discovered 2026-05-07 during Step A.3's torch-L-BFGS lane test.
> Confirms a prior user observation: "torch's L-BFGS doesn't quite work
> for me". MATLAB minFunc does work; the issue is torch's specific
> implementation, not L-BFGS as an algorithm.

## Symptom

Run with `tests/short_retrain_2lpt.py --lane lbfgs --n-iters 10` on
the 1300-spectrum 2lpt fixture, using:

```python
torch.optim.LBFGS(
    [M, log_omega, log_c_0, log_tau_0, log_beta],
    lr=1.0,
    max_iter=1,
    history_size=10,
    line_search_fn="strong_wolfe",
    tolerance_grad=0.0,
    tolerance_change=0.0,
)
```

Trajectory:

| iter | loss | τ_0 | β | c_0 |
|---:|---:|---:|---:|---:|
| init | 4,785,935 | 0.00246 | 3.62 | 0.1 |
| 0 | 4,359,153 | 0.00226 | 2.42 | 0.0899 |
| 1 | 4,321,346 | 0.00229 | 2.55 | 0.0877 |
| 2 | **6,376,229** ↑ | 0.00229 | 2.55 | 0.0877 |
| 3 | 6,376,229 | 0.00229 | 2.55 | 0.0877 |
| 4 | 6,376,229 | 0.00229 | 2.55 | 0.0877 |
| 5–9 | 6,376,229 | 0.00229 | 2.55 | 0.0877 |

After it=2, every parameter is frozen and the loss stays at 6,376,229.
The optimizer is fully stuck.

For comparison, MATLAB minFunc on the same fixture, also L-BFGS family
with strong-Wolfe-style line search:

| iter | MATLAB loss |
|---:|---:|
| 0 | 4,785,935 |
| 5 | 4,028,000 |
| 10 | 3,889,000 |
| 50 | 3,818,881 |

MATLAB descended monotonically and reached a 19% lower loss.

## Hypothesis

The GP-DLA loss landscape is highly anisotropic across parameter
scales:

- M:          O(100) magnitude per entry, ~114000 entries
- log_omega:  O(1), 3801 entries
- log_c_0:    O(-2), scalar
- log_tau_0:  O(-6), scalar
- log_beta:   O(1), scalar

`torch.optim.LBFGS` treats all parameters uniformly — no per-parameter
preconditioning. The L-BFGS Hessian approximation built from
(s_k, y_k) pairs across iterations gets mixed up by the wildly-
different parameter scales: a small step in M space looks like a
huge step in log_τ_0 space (and vice versa), so the Hessian curvature
estimate is dominated by whichever parameter has the largest scale,
leaving other parameters under-updated.

MATLAB minFunc handles this through a more sophisticated line search
+ progress checks, possibly including parameter rescaling. The torch
strong_wolfe line search appears to fail at it=2 — the line search
hits its max iterations without finding a step satisfying both Armijo
and curvature conditions, and the optimizer then accepts a degenerate
step that increases the loss. After that, the L-BFGS direction is
based on contradictory (s_k, y_k) history and produces no further
movement.

## Confirmed paths NOT to take in production

- ❌ `torch.optim.LBFGS` with `line_search_fn="strong_wolfe"` and
  default settings — confirmed gets stuck on this problem.

## Paths worth trying (future)

If we want L-BFGS in PyTorch for production, options:

1. **Pre-condition** parameters before passing to LBFGS:
   - Rescale M to unit-norm columns
   - Use `log_omega - mean(log_omega)` so log_omega has zero mean
   - Train (log_c_0, log_tau_0, log_beta) at a different LR than (M, log_omega)
2. **Two-phase training**: optimize (M, log_omega) first with LBFGS at
   fixed (log_c_0, log_tau_0, log_beta), then alternate.
3. **Different L-BFGS implementation**: try
   [pytorch_minimize](https://github.com/rfeinman/pytorch-minimize) or
   port minFunc to PyTorch.
4. **Stick with MATLAB minFunc** for this trainer specifically.
   PyTorch is fine for inference; MATLAB-as-trainer + PyTorch-as-inference
   is a reasonable architecture given (a) MATLAB minFunc is mature here,
   (b) MATLAB-trained .mat catalogs already exist for SDSS DR16
   (`/home/mfho/MATLAB/gp_dla_detection_dr16q_public/data/dr16/MATLAB_Catalogue`),
   (c) inference is the perf-critical path, not training.

## What this means for the v3 production trainer

The "v3 vectorization" Step B planned in the debug plan should NOT
default to torch L-BFGS based on this evidence. Two viable paths:

A. **Adam + many epochs** — known to converge (v1 production used this).
   Slower than L-BFGS in iteration count, but no convergence concerns.
   Add a Gaussian prior on log_c_0 (we found it can collapse).
B. **MATLAB minFunc as the trainer**, Python only for inference.
   Adds a MATLAB dependency for production retrains, but mirrors what
   v1 MATLAB pipeline has done successfully for ~5+ years.

A v1 production retrain comparison would inform this. For now, file
this as a known limitation; pick path A or B in the production
roadmap.
