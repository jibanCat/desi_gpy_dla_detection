#!/usr/bin/env python
"""Analyze a parallelism_sweep_nersc.sh run.

Reads the per-spectrum "time spent: XmYs" log lines from each sweep cell
and reports:
  - Phase A: per-spectrum compute vs MAX_WORKERS, on the set of QSOs
    common to all worker counts (apples-to-apples), plus the parallel
    speedup + efficiency.
  - Predicted node throughput for each N×W packing (N*W <= NCORES).
  - Phase B: measured aggregate throughput for the concurrency cells,
    so contention vs the Phase-A prediction is visible.

Usage:
    python slurm/nersc/production/analyze_sweep.py \
        /pscratch/sd/j/jibancat/nersc_parallelism_sweep_<date> [NCORES]
"""
import re
import sys
from pathlib import Path

TIME_RE = re.compile(r"Processed spectrum \d+/\d+ \(ID: (\d+)\), time spent: (\d+)m (\d+)s")


def parse_cell(run_log: Path, drop_warmup: int = 2):
    """Return {qso_id: seconds} for a single run.log, dropping warmup spectra."""
    rows = []
    if not run_log.is_file():
        return {}
    for line in run_log.read_text(errors="ignore").splitlines():
        m = TIME_RE.search(line)
        if m:
            qid, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
            rows.append((qid, mm * 60 + ss))
    rows = rows[drop_warmup:]
    return {qid: s for qid, s in rows}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    root = Path(sys.argv[1])
    ncores = int(sys.argv[2]) if len(sys.argv) > 2 else 36

    # --- Phase A: latency vs MAX_WORKERS --------------------------------------
    # NERSC layout: each latency cell is a 1-task srun, so its log lives at
    # latency_W{w}/srun_0/run.log (not latency_W{w}/run.log as on GL). W=1,2 are
    # skipped on NERSC (don't finish enough spectra in the debug wall at PW100k),
    # so efficiency is reported relative to the smallest W present.
    workers = [8, 16, 32]
    cells = {}
    for w in workers:
        d = parse_cell(root / f"latency_W{w}" / "srun_0" / "run.log")
        if d:
            cells[w] = d

    if not cells:
        print(f"No latency cells found under {root}")
        sys.exit(1)

    common = set.intersection(*[set(d) for d in cells.values()]) if len(cells) > 1 else set(next(iter(cells.values())))

    print(f"# Parallelism sweep analysis — {root.name}")
    print(f"Node cores assumed: {ncores}")
    print(f"QSOs common to all worker cells (matched set): {len(common)}\n")

    print("## Phase A — per-spectrum compute vs MAX_WORKERS (matched QSOs)")
    print(f"{'W':>3} {'n_meas':>7} {'mean_s':>8} {'mean_s(common)':>15} {'speedup':>8} {'efficiency':>11}")
    base = None
    rows_a = {}
    for w in sorted(cells):
        d = cells[w]
        all_mean = mean(d.values())
        common_mean = mean(d[q] for q in common if q in d)
        rows_a[w] = common_mean
        if w == 1 or base is None:
            base = common_mean
        speedup = base / common_mean if common_mean else float("nan")
        eff = speedup / w
        print(f"{w:>3} {len(d):>7} {all_mean:>8.1f} {common_mean:>15.1f} {speedup:>8.2f} {eff:>10.0%}")

    # --- Predicted node throughput per packing --------------------------------
    print("\n## Predicted node throughput per packing (N concurrent × W workers, N*W <= cores)")
    print(f"{'W':>3} {'N=cores/W':>10} {'per_spec_s':>11} {'node spec/min':>14} {'rel':>6}")
    best = None
    preds = {}
    for w in sorted(rows_a):
        per_spec = rows_a[w]
        n = max(1, ncores // w)
        node_rate = n / per_spec * 60.0  # spectra/min, assuming linear concurrency
        preds[w] = (n, node_rate)
        if best is None or node_rate > preds[best][1]:
            best = w
    for w in sorted(preds):
        n, rate = preds[w]
        rel = rate / preds[best][1]
        flag = "  <- predicted best" if w == best else ""
        print(f"{w:>3} {n:>10} {rows_a[w]:>11.1f} {rate:>14.1f} {rel:>6.0%}{flag}")

    # --- Phase B: measured concurrency throughput -----------------------------
    print("\n## Phase B — measured aggregate throughput (concurrency cells)")
    print(f"{'cell':>20} {'n_srun':>7} {'agg_spec':>9} {'note':>6}")
    for cell in sorted(root.glob("concurrency_N*_W*")):
        srun_logs = sorted(cell.glob("srun_*/run.log"))
        total = 0
        for sl in srun_logs:
            total += len(parse_cell(sl, drop_warmup=2))
        print(f"{cell.name:>20} {len(srun_logs):>7} {total:>9}")
    print("\n(Phase B note: divide agg_spec by the PHASE_B_SECS timebox for spectra/sec;\n"
          " compare against the Phase-A predicted node rate for the same W to see\n"
          " whether memory-bandwidth contention degrades real throughput.)")


if __name__ == "__main__":
    main()
