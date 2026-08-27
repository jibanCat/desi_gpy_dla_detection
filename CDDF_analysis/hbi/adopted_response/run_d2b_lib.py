"""Shared-surface builder: per-cell deg-2 + pooled shared orders 3..deg."""
import numpy as np


def shared_surfaces(rows, deg_shared, N_REF, n_iter=2):
    Ds = deg_shared + 1
    keys = ("mu", "sig", "skew")
    shared = np.zeros((3, Ds))
    for _ in range(n_iter):
        cell2 = {k: np.zeros((3, 3, 3)) for k in keys}
        rng = np.zeros((3, 3, 2))
        for i in range(3):
            for j in range(3):
                rr = [r for r in rows[i][j] if r["ok"]]
                c = np.array([r["c"] for r in rr]); u = c - N_REF
                w = np.sqrt([r["n"] for r in rr])
                rng[i, j] = (c.min(), c.max())
                up = u[:, None] ** np.arange(Ds)[None, :]
                for ki, k in enumerate(keys):
                    y = (np.array([r[k] for r in rr])
                         - up[:, 3:] @ shared[ki, 3:])
                    cell2[k][i, j] = np.polynomial.polynomial.polyfit(
                        u, y, 2, w=w)
        for ki, k in enumerate(keys):
            uu, rr_, ww = [], [], []
            for i in range(3):
                for j in range(3):
                    rrows = [r for r in rows[i][j] if r["ok"]]
                    c = np.array([r["c"] for r in rrows]); u = c - N_REF
                    up2 = u[:, None] ** np.arange(3)[None, :]
                    y = np.array([r[k] for r in rrows]) - up2 @ cell2[k][i, j]
                    uu.append(u); rr_.append(y)
                    ww.append(np.sqrt([r["n"] for r in rrows]))
            uu = np.concatenate(uu); rr_ = np.concatenate(rr_)
            ww = np.concatenate(ww)
            X = uu[:, None] ** np.arange(3, Ds)[None, :]
            beta, *_ = np.linalg.lstsq((ww[:, None] * X), ww * rr_,
                                       rcond=None)
            shared[ki, 3:] = beta
    surf = {}
    for ki, k in enumerate(keys):
        arr = np.zeros((3, 3, Ds))
        arr[..., :3] = cell2[k]
        arr[..., 3:] = shared[ki, 3:][None, None, :]
        surf[k] = arr
    return surf, rng, shared
