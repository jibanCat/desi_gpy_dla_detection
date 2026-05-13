"""
Var[Delta_marg] gating diagnostic — re-analysis of an existing inference run.

Δ_marg ≡ log p(D | 1 DLA) − log p(D | null) ≈ logsumexp(per-QMC-sample lik) − log N − log p(D|null).

We resample N ∈ {1k, 5k, 10k, 25k, 50k} from the 50k samples stored in
sample_log_likelihoods_dla[..., 0] (the k=1-DLA model column). For each N we
take ``n_seeds`` independent subsamples (non-overlapping when 4·N ≤ 50000;
bootstrap otherwise) and compute Δ_marg per (spectrum, seed, N).

Output: a parquet table (one row per (target_id, N, seed)) written
incrementally per input file so partial results survive jupyter expiry.

This script is read-only with respect to the inference outputs; it only writes
its own table + summary CSV/PNG.

Reference: docs/notes/2026-05-12_map_lr_failure.md §"Refined diagnostic
recommendation", docs/notes/2026-05-12_mlmc_design.md §Option A.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from time import time

import h5py
import numpy as np
from scipy.special import logsumexp


DEFAULT_N_LEVELS = (1000, 5000, 10000, 25000, 50000)
DEFAULT_N_SEEDS = 4


def _idx_for_seed(N_total: int, N: int, seed: int, n_seeds: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``N`` sample indices. Non-overlapping when ``n_seeds·N ≤ N_total``;
    bootstrap with replacement otherwise."""
    if n_seeds * N <= N_total:
        start = seed * N
        return np.arange(start, start + N)
    # bootstrap — re-seed per (N, seed) for reproducibility independent of file order
    sub_rng = np.random.default_rng((rng.integers(2**32 - 1), N, seed))
    return sub_rng.integers(0, N_total, size=N)


def process_file(
    h5_path: str,
    N_levels: tuple[int, ...],
    n_seeds: int,
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        # k=1-DLA samples (column 0 of axis 2).
        ll_1dla = f["sample_log_likelihoods_dla"][..., 0]   # (n_spec, N_total)
        ll_null = f["log_likelihoods_no_dla"][:]             # (n_spec,)
        p_dla_full = f["p_dlas"][:]
        tid = f["target_ids"][:]
        snr = f["snrs"][:]
        snr_b = f["snrs_blue"][:]
        z_qso = f["z_qsos"][:]
    n_spec, N_total = ll_1dla.shape
    for N in N_levels:
        if N > N_total:
            continue
        for seed in range(n_seeds):
            idx = _idx_for_seed(N_total, N, seed, n_seeds, rng)
            # logsumexp over the N selected samples, then subtract log N → log mean exp
            lme = logsumexp(ll_1dla[:, idx], axis=1) - np.log(N)
            delta_marg = lme - ll_null
            for i in range(n_spec):
                rows.append(
                    {
                        "target_id": int(tid[i]),
                        "z_qso": float(z_qso[i]),
                        "snr": float(snr[i]),
                        "snr_blue": float(snr_b[i]),
                        "p_dla_full": float(p_dla_full[i]),
                        "N": int(N),
                        "seed": int(seed),
                        "delta_marg": float(delta_marg[i]),
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--processed-dir",
        default="/pscratch/sd/j/jibancat/prod533_5k_20260511/london_v3_loa124_pw14_tau_eb/processed",
        help="Directory of processed-spectra-16-*.h5 files",
    )
    ap.add_argument(
        "--out-dir",
        default="/pscratch/sd/j/jibancat/prod533_5k_20260511/var_delta_marg",
        help="Output directory (created if missing)",
    )
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument(
        "--n-levels",
        nargs="+",
        type=int,
        default=list(DEFAULT_N_LEVELS),
        help="QMC sample-count levels to sweep",
    )
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.processed_dir, "processed-spectra-16-*.h5")))
    if not files:
        print(f"[var_delta_marg] no h5 files in {args.processed_dir}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    rows_path = os.path.join(args.out_dir, "var_delta_marg_rows.parquet")
    rows_npz_path = os.path.join(args.out_dir, "var_delta_marg_rows_partial.npz")
    print(f"[var_delta_marg] {len(files)} files; out={args.out_dir}")
    print(f"[var_delta_marg] N levels = {tuple(args.n_levels)}, n_seeds={args.n_seeds}")

    rng = np.random.default_rng(args.rng_seed)
    all_rows: list[dict] = []
    for k, fp in enumerate(files):
        t0 = time()
        rows = process_file(fp, tuple(args.n_levels), args.n_seeds, rng)
        all_rows.extend(rows)
        # Incremental .npz save so partial work survives a session crash.
        _save_npz(all_rows, rows_npz_path)
        print(
            f"[var_delta_marg] [{k+1}/{len(files)}] {os.path.basename(fp)}  "
            f"+{len(rows)} rows  ({time()-t0:.1f}s)  total={len(all_rows)}"
        )

    # Final outputs: parquet (if pandas available), summary CSV, plot.
    try:
        import pandas as pd

        df = pd.DataFrame(all_rows)
        df.to_parquet(rows_path, index=False)
        print(f"[var_delta_marg] wrote {rows_path} ({len(df)} rows)")
    except (ImportError, ValueError) as e:
        print(f"[var_delta_marg] parquet skipped ({e}); npz still on disk: {rows_npz_path}")
        df = None

    _write_summary_and_plot(all_rows, args.out_dir, df=df)
    return 0


def _save_npz(rows: list[dict], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    arrs = {k: np.asarray([r[k] for r in rows]) for k in keys}
    np.savez_compressed(path, **arrs)


def _write_summary_and_plot(rows: list[dict], out_dir: str, df=None):
    if df is None:
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
        except ImportError:
            print("[var_delta_marg] pandas missing — skip summary")
            return

    # Var across seeds per (target_id, N)
    g = df.groupby(["target_id", "N"])["delta_marg"]
    var_per_specN = g.var(ddof=1).rename("var").reset_index()
    mean_per_specN = g.mean().rename("mean_delta_marg").reset_index()
    summary = var_per_specN.merge(mean_per_specN, on=["target_id", "N"])

    # Bring p_dla_full back so we can stratify "borderline" vs "confident".
    pmap = df.drop_duplicates("target_id")[["target_id", "p_dla_full", "snr", "snr_blue", "z_qso"]]
    summary = summary.merge(pmap, on="target_id")

    summary_path = os.path.join(out_dir, "var_delta_marg_per_spec_N.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[var_delta_marg] wrote {summary_path}")

    # Aggregate: median, p25, p75, p95 of sqrt(var) across spectra, per N,
    # stratified by p_dla_full bucket.
    summary["bucket"] = "borderline"
    summary.loc[summary["p_dla_full"] >= 0.99, "bucket"] = "confident_pos"
    summary.loc[summary["p_dla_full"] <= 0.01, "bucket"] = "confident_neg"
    summary["snr_bucket"] = "snr_lt_2"
    summary.loc[summary["snr"] >= 2.0, "snr_bucket"] = "snr_2_to_4"
    summary.loc[summary["snr"] >= 4.0, "snr_bucket"] = "snr_ge_4"

    summary["std_delta_marg"] = np.sqrt(summary["var"].clip(lower=0))
    agg = (
        summary.groupby(["bucket", "snr_bucket", "N"])["std_delta_marg"]
        .agg(["count", "median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        .rename(columns={"<lambda_0>": "p25", "<lambda_1>": "p75"})
        .reset_index()
    )
    agg_path = os.path.join(out_dir, "var_delta_marg_aggregated.csv")
    agg.to_csv(agg_path, index=False)
    print(f"[var_delta_marg] wrote {agg_path}")

    # Signal-null typical separation per N (using mean_delta_marg across seeds):
    # |median(delta_marg | confident_pos) − median(delta_marg | confident_neg)|.
    sep_rows = []
    for N in sorted(summary["N"].unique()):
        sub = summary[summary["N"] == N]
        pos = sub[sub["bucket"] == "confident_pos"]["mean_delta_marg"]
        neg = sub[sub["bucket"] == "confident_neg"]["mean_delta_marg"]
        sep_rows.append(
            {
                "N": int(N),
                "n_pos": len(pos),
                "n_neg": len(neg),
                "median_pos": float(pos.median()) if len(pos) else float("nan"),
                "median_neg": float(neg.median()) if len(neg) else float("nan"),
                "separation": float(pos.median() - neg.median()) if len(pos) and len(neg) else float("nan"),
                "median_std_borderline": float(
                    summary[(summary["N"] == N) & (summary["bucket"] == "borderline")]["std_delta_marg"].median()
                ),
            }
        )
    try:
        import pandas as pd
        sep_df = pd.DataFrame(sep_rows)
    except ImportError:
        sep_df = None
    if sep_df is not None:
        sep_path = os.path.join(out_dir, "var_delta_marg_signal_vs_noise.csv")
        sep_df.to_csv(sep_path, index=False)
        print(f"[var_delta_marg] wrote {sep_path}")
        # Pretty-print the gating verdict
        print("\n=== gating verdict (signal-vs-noise vs N) ===")
        print(sep_df.to_string(index=False))

    # Plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for bucket, color in [("confident_neg", "C0"), ("borderline", "C2"), ("confident_pos", "C3")]:
            sub = summary[summary["bucket"] == bucket]
            if len(sub) == 0:
                continue
            byN = sub.groupby("N")["std_delta_marg"].median()
            ax.plot(byN.index, byN.values, "o-", color=color, label=f"{bucket} (n={len(sub.drop_duplicates('target_id'))})")
        if sep_df is not None:
            ax.plot(sep_df["N"], sep_df["separation"].abs(), "k--", label="signal–null gap (|med pos − med neg|)")
        ax.set_xscale("log")
        ax.set_xlabel("QMC samples N")
        ax.set_ylabel(r"median std$[\Delta_{\rm marg}]$ across seeds")
        ax.set_title("Var[Δ_marg] gating diagnostic — london v3 loa124 prod533")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        png_path = os.path.join(out_dir, "var_delta_marg.png")
        fig.savefig(png_path, dpi=140)
        print(f"[var_delta_marg] wrote {png_path}")
    except ImportError:
        print("[var_delta_marg] matplotlib missing — skip plot")


if __name__ == "__main__":
    raise SystemExit(main())
