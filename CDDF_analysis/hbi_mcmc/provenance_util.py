"""provenance_util.py — the small shared helpers every Paper-1 producer stamps with
(Paper-1 code review 2026-08-26): the invocation, the seed/sampler configuration and
the code commit belong INSIDE the artifact, not only in a slurm log's first line."""
import hashlib, pathlib, subprocess, sys


def git_commit(repo=None) -> str:
    repo = pathlib.Path(repo) if repo else pathlib.Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "-uno"], capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_config(args, keys=("seed", "warmup", "samples", "chains", "target_accept", "fp_mode",
                            "rhat_max", "div_max")) -> dict:
    """argparse namespace -> the stamped configuration block."""
    cfg = {k: getattr(args, k) for k in keys if hasattr(args, k)}
    cfg.update(argv=list(sys.argv), code_commit=git_commit(), python=sys.version.split()[0])
    return cfg
