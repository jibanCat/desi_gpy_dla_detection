"""Drop a README into each `/scratch/.../phase2_desi/<run>/` so a runner
landing at the checkpoint folder can tell at a glance what this training
is, whether it's running / completed / superseded, and where the
authoritative output (phase2_result.h5) lives.

For each scratch dir:
  1. Find the latest SLURM log mentioning the run_name.
  2. Extract config from the log header (norm_band, n_iters, preload, ...).
  3. Read the latest checkpoint iter from the .pt filenames.
  4. Look up the corresponding output dir in docs/notes/.
  5. Determine status from sacct.
  6. Write `README.md` at the scratch dir root (NOT inside checkpoints/).

Re-run as new runs land or supersede.
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/mfho/desi_gpy_dla_detection")
SCRATCH_ROOT = Path("/scratch/cavestru_root/cavestru0/mfho/phase2_desi")
SLURM_LOGS = REPO / "slurm" / "greatlakes"
NOTES = REPO / "docs" / "notes"

HEADER_RE = re.compile(
    r"run_name\s*:\s*(\S+)\s*$.*?"
    r"preload\s*:\s*(\S+)\s*$.*?"
    r"out_dir\s*:\s*(\S+)\s*$.*?"
    r"n_iters\s*:\s*(\d+)\s*$.*?"
    r"norm_band\s*:\s*\[([\d.]+),\s*([\d.]+)\]",
    re.DOTALL | re.MULTILINE,
)


def find_latest_slurm_log(run_name: str) -> tuple[Path, int] | None:
    """Return (log_path, slurm_jobid) for the most recent log mentioning
    this run_name."""
    matches = []
    for log in SLURM_LOGS.glob("phase2_desi_*_*.log"):
        try:
            head = log.read_text(errors="ignore")[:4000]
        except Exception:
            continue
        m = re.search(rf"run_name\s*:\s*{re.escape(run_name)}\b", head)
        if m:
            jobid_match = re.search(r"_(\d+)\.log$", log.name)
            if jobid_match:
                matches.append((log, int(jobid_match.group(1))))
    if not matches:
        return None
    matches.sort(key=lambda t: t[1], reverse=True)
    return matches[0]


def parse_header(log_path: Path) -> dict:
    text = log_path.read_text(errors="ignore")[:4000]
    out = {}
    for key, pattern in [
        ("run_name", r"run_name\s*:\s*(\S+)"),
        ("preload",  r"preload\s*:\s*(\S+)"),
        ("out_dir",  r"out_dir\s*:\s*(\S+)"),
        ("n_iters",  r"n_iters\s*:\s*(\d+)"),
        ("lr",       r"lr\s*:\s*([\d.]+)"),
        ("chunk_size", r"chunk_size\s*:\s*(\d+)"),
        ("min_snr",  r"min_snr\s*:\s*([\d.]+)"),
        ("max_walltime_sec", r"--max-walltime-sec\s+(\d+)"),  # CLI invocation
    ]:
        m = re.search(pattern, text)
        if m:
            out[key] = m.group(1)
    band = re.search(r"norm_band\s*:\s*\[([\d.]+),\s*([\d.]+)\]", text)
    if band:
        out["norm_min"] = float(band.group(1))
        out["norm_max"] = float(band.group(2))
    c0 = re.search(r"--log-c-0-prior-sigma\s+(\S+)", text)
    if c0:
        out["log_c_0_prior_sigma"] = c0.group(1)
    return out


def latest_checkpoint_iter(scratch_dir: Path) -> tuple[int | None, Path | None]:
    """Highest iter number among phase2_desi_checkpoint_iter*.pt files."""
    ckpt_dir = scratch_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None, None
    best = -1
    best_path = None
    for p in ckpt_dir.glob("phase2_desi_checkpoint_iter*.pt"):
        m = re.search(r"iter(\d+)", p.name)
        if m:
            it = int(m.group(1))
            if it > best:
                best = it
                best_path = p
    if best < 0:
        return None, None
    return best, best_path


def sacct_state(jobid: int) -> str | None:
    try:
        out = subprocess.check_output(
            ["sacct", "-j", str(jobid), "--format=State", "-X", "-n", "-P"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode().strip()
        # Take the first line (one job per ID with -X)
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def supersession_for(run_name: str) -> str:
    """Hardcoded supersession map; update as new runs land. Returns a
    short markdown blurb pointing to the superseder, or empty string."""
    SUPERSEDED = {
        # Pre-reorder _m variants superseded by their _m_normmask retrains
        "2lpt_loa124_nohcd_nobal_wide_m": "Post-reorder retrain in flight: see `/scratch/.../2lpt_loa124_nohcd_nobal_wide_m_normmask/`.",
        "2lpt_loa0_wide_m":                "Post-reorder retrain in flight: see `/scratch/.../2lpt_loa0_wide_m_normmask/`.",
        # Pre-reorder _g variants superseded by _g_normmask retrains
        "2lpt_loa124_nohcd_nobal_wide_g":  "Post-reorder retrain in flight: see `/scratch/.../2lpt_loa124_nohcd_nobal_wide_g_normmask/`.",
        "2lpt_loa0_wide_g":                "Post-reorder retrain in flight: see `/scratch/.../2lpt_loa0_wide_g_normmask/`.",
        # Pre-reorder LOA variants superseded by 3000-iter post-reorder
        "loa_no_dla_no_bal_wide_g":   "Pre-reorder + TIMEOUT at iter ~700; post-reorder 3000-iter retrain in flight at `/scratch/.../loa_no_dla_no_bal_wide_m_normmask_3000iter/`.",
        "loa_no_dla_no_bal_wide_m":   "Pre-reorder + TIMEOUT at iter ~700; post-reorder 3000-iter retrain in flight at `/scratch/.../loa_no_dla_no_bal_wide_m_normmask_3000iter/`.",
        "loa_no_hcd_with_bal_wide_g": "Pre-reorder + TIMEOUT at iter ~770; post-reorder 3000-iter retrain in flight at `/scratch/.../loa_no_hcd_with_bal_wide_m_normmask_3000iter/`.",
        "loa_no_hcd_with_bal_wide_m": "Pre-reorder + TIMEOUT at iter ~800; post-reorder 3000-iter retrain in flight at `/scratch/.../loa_no_hcd_with_bal_wide_m_normmask_3000iter/`.",
        # Cancelled 1500-iter LOA submissions (now in 3000-iter form)
        "loa_no_dla_no_bal_wide_m_normmask":   "Cancelled 1500-iter submission; resubmitted at 3000 iter at `loa_no_dla_no_bal_wide_m_normmask_3000iter/`.",
        "loa_no_hcd_with_bal_wide_m_normmask": "Cancelled 1500-iter submission; resubmitted at 3000 iter at `loa_no_hcd_with_bal_wide_m_normmask_3000iter/`.",
        # Pre-reorder base wide-σ collapse runs — deprecated entirely
        "2lpt_loa0_wide":                   "Wide-σ prior caused β=1.28 collapse; deprecated, use post-reorder `_m_normmask` instead.",
        "2lpt_loa124_nohcd_nobal_wide":     "Wide-σ prior caused β=1.45 collapse; deprecated, use post-reorder `_m_normmask` instead.",
        # c0prior — kept for trail, but not preferred
        "2lpt_loa124_nohcd_nobal_wide_c0prior": "log_c_0 prior anchoring failed; equivalent to `_m` on 10-target sample. Prefer `_m_normmask`. See `docs/notes/2026-05-14_c0prior_failure_investigation/`.",
    }
    return SUPERSEDED.get(run_name, "")


def write_readme(scratch_dir: Path):
    run_name = scratch_dir.name
    log_info = find_latest_slurm_log(run_name)
    last_iter, last_ckpt = latest_checkpoint_iter(scratch_dir)
    state = None
    cfg = {}
    log_path = None
    jobid = None
    if log_info:
        log_path, jobid = log_info
        cfg = parse_header(log_path)
        state = sacct_state(jobid)

    # Determine pipeline (pre/post-reorder) from suffix
    is_post = "_normmask" in run_name
    is_smoke = run_name.endswith("_smoke")
    pipeline = "POST-reorder" if is_post else "PRE-reorder"

    # Out_dir mapping (output of `--out-dir`) — may be in cfg or guessable
    out_dir = cfg.get("out_dir", "(not found in log header)")

    # Tag the model's status header
    if state in ("RUNNING", "PENDING"):
        status_tag = "⏳ IN-FLIGHT"
    elif state == "COMPLETED":
        status_tag = "✓ COMPLETED" + (" (POST-reorder)" if is_post else " (PRE-reorder)")
    elif state == "TIMEOUT":
        status_tag = "⚠ TIMEOUT — last checkpoint preserved"
    elif state == "CANCELLED":
        status_tag = "🚫 CANCELLED"
    elif state is None:
        status_tag = "❓ UNKNOWN (no recent SLURM record)"
    else:
        status_tag = f"({state})"

    superseded_by = supersession_for(run_name)

    body_lines = [
        f"# `/scratch/.../phase2_desi/{run_name}/` — training checkpoint folder",
        "",
        f"> **STATUS: {status_tag}**.",
    ]
    if superseded_by:
        body_lines.append(f">")
        body_lines.append(f"> ⚠ {superseded_by}")
    body_lines.append(">")
    body_lines.append("> See `docs/CURRENT_MODELS.md` (in the repo) for the current top-pick model per use case.")
    body_lines.append("")
    body_lines.append("## What this is")
    body_lines.append("")
    body_lines.append(
        f"Adam-loop checkpoints from a `tests/phase2_train_desi.py` training run. "
        f"The pipeline is **{pipeline}** (`_normmask` suffix indicates post-2026-05-13 `dataset.py` reorder + `|med| < 1e-2` rejection threshold)."
    )
    body_lines.append("")
    body_lines.append("## Run metadata")
    body_lines.append("")
    body_lines.append("| Field | Value |")
    body_lines.append("|---|---|")
    body_lines.append(f"| run_name | `{run_name}` |")
    body_lines.append(f"| SLURM job | `{jobid}` |" if jobid else "| SLURM job | (no log) |")
    body_lines.append(f"| State (sacct) | `{state}` |" if state else "| State (sacct) | (unknown) |")
    body_lines.append(f"| Target n_iters | `{cfg.get('n_iters', '?')}` |")
    body_lines.append(f"| Latest checkpoint iter | `{last_iter}` |" if last_iter is not None else "| Latest checkpoint iter | (no .pt files) |")
    if "norm_min" in cfg:
        band_label = ("(MATLAB DR16 convention)"
                      if abs(cfg["norm_min"] - 1425.0) < 1
                      else "(Garnett+2017 convention)"
                      if abs(cfg["norm_min"] - 1310.0) < 1
                      else "(custom)")
        body_lines.append(f"| Norm band | `[{cfg['norm_min']:.2f}, {cfg['norm_max']:.2f}]` Å rest {band_label} |")
    body_lines.append(f"| lr | `{cfg.get('lr', '?')}` |")
    body_lines.append(f"| chunk_size | `{cfg.get('chunk_size', '?')}` |")
    body_lines.append(f"| min_snr | `{cfg.get('min_snr', '?')}` |")
    if "log_c_0_prior_sigma" in cfg:
        body_lines.append(f"| log_c_0 prior σ | `{cfg['log_c_0_prior_sigma']}` |")
    body_lines.append(f"| Preload | `{cfg.get('preload', '?')}` |")
    body_lines.append(f"| out_dir (repo) | `{out_dir}` |")
    body_lines.append(f"| SLURM log | `{log_path.relative_to(REPO) if log_path else '?'}` |")
    body_lines.append("")
    body_lines.append("## What lives here")
    body_lines.append("")
    body_lines.append(f"- `checkpoints/phase2_desi_checkpoint_iter*.pt` — Adam-state checkpoints (saved every ~25 iter).")
    body_lines.append(f"- `checkpoints/phase2_desi_checkpoint_final_iter*.pt` — final iteration's checkpoint if training completed.")
    body_lines.append("")
    body_lines.append("## Authoritative model artifact")
    body_lines.append("")
    if cfg.get("out_dir"):
        body_lines.append(
            f"The trained model lives at `{out_dir}/phase2_result.h5` "
            f"(production-loadable by `gpy_dla_detection.null_gp.NullGPMAT`)."
        )
        body_lines.append("")
        body_lines.append(
            "**Do not load .pt checkpoints directly for inference.** They lack the "
            "rest-grid + norm-band + prior metadata needed by the inference loader. "
            "Wait for / use `phase2_result.h5` instead."
        )
    else:
        body_lines.append("`phase2_result.h5` output location unknown (could not parse SLURM header).")
    body_lines.append("")
    body_lines.append(f"_README auto-generated by `examples/write_scratch_readmes.py` at {datetime.now(timezone.utc).isoformat()}._")

    readme_path = scratch_dir / "README.md"
    readme_path.write_text("\n".join(body_lines))
    return readme_path


def main():
    if not SCRATCH_ROOT.exists():
        raise SystemExit(f"scratch root not found: {SCRATCH_ROOT}")
    for d in sorted(SCRATCH_ROOT.iterdir()):
        if not d.is_dir():
            continue
        try:
            p = write_readme(d)
            print(f"[wrote] {p}")
        except Exception as e:
            print(f"[fail]  {d.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
