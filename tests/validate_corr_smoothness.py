"""Quick validation gate: read a phase2_desi checkpoint .pt and report whether
the trained M's corr(M·M^T) is smooth (good) or noisy (broken trainer or
preload outliers).

Smoothness metric: mean |C[i,j] - C[i,j+1]| over off-diagonal pairs.
  - smooth (v1 production / DR16 corrected): ~0.005 (< 0.02 threshold)
  - noisy (v2 broken / outlier-corrupted PCA): ~0.20-0.30

Exit code:
  0  → smooth (PASS)
  1  → noisy  (FAIL — caller should `scancel` the running job)
  2  → file not found / error

Usage::

    python tests/validate_corr_smoothness.py PATH/TO/checkpoint.pt
    python tests/validate_corr_smoothness.py PATH --threshold 0.02
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def smoothness(M: np.ndarray) -> tuple[float, float]:
    """Return (mean_adj_pixel_diff_in_corr, top10_eigval_fraction)."""
    K = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    C = np.clip(K / np.outer(d, d), -1.0, 1.0)
    adj_diff = np.abs(np.diff(C, axis=1)).mean()
    eigvals = np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1]
    top10_frac = eigvals[:10].sum() / eigvals.sum()
    return float(adj_diff), float(top10_frac)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("ckpt", type=Path, help=".pt checkpoint to validate")
    p.add_argument("--threshold", type=float, default=0.02,
                   help="mean adj diff above this = NOISY (default 0.02)")
    args = p.parse_args()

    if not args.ckpt.exists():
        print(f"[validate] not found: {args.ckpt}", file=sys.stderr)
        sys.exit(2)
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        M = ckpt["M"].cpu().numpy() if hasattr(ckpt["M"], "cpu") else np.asarray(ckpt["M"])
    except Exception as e:
        print(f"[validate] error loading {args.ckpt}: {e}", file=sys.stderr)
        sys.exit(2)

    adj, top10 = smoothness(M)
    iter_done = int(ckpt.get("iter_completed", -1))
    verdict = "PASS" if adj < args.threshold else "FAIL"
    print(f"[validate] {args.ckpt.name}  iter={iter_done}  "
          f"M.shape={M.shape}  mean_adj_diff={adj:.4f}  top10_frac={top10:.4f}  "
          f"→ {verdict} (threshold {args.threshold})")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
