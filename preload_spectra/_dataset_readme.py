"""Helper: write a README.md + dataset_metadata.json next to a preloaded
gp_interp_trainset.h5 so a future user (or reviewer) can answer

    "what's in this file? where did it come from? how do I train on it?"

without reading the SLURM log or git-blaming the script.

Used by both ``preload_loa_real.py`` and ``preload_2lpt_simple.py``
at the end of their respective ``main()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_dataset_readme(
    output_h5: Path,
    *,
    dataset_kind: str,                    # "loa_real" | "2lpt_mock"
    n_spectra: int,
    n_pix: int,
    rest_min: float,
    rest_max: float,
    dlambda: float,
    z_min: float,
    z_max: float,
    filter_pipeline: list[str],           # one human-readable line per filter step
    sources: dict[str, str],              # e.g. {"qsocat": "/path/...", ...}
    cli_args: dict[str, Any],             # snapshot of argparse Namespace
    suggested_train_command: str,
):
    """Write README.md and dataset_metadata.json next to ``output_h5``.

    Both files are small (< 20 KB) and readable without h5py. The h5
    attributes still carry the same provenance for programmatic access;
    these companion files exist for human consumption.
    """
    output_h5 = Path(output_h5)
    parent = output_h5.parent
    parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "kind": dataset_kind,
        "n_spectra": int(n_spectra),
        "n_pix": int(n_pix),
        "rest_lambda_min_AA": float(rest_min),
        "rest_lambda_max_AA": float(rest_max),
        "dlambda_AA": float(dlambda),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "filter_pipeline": filter_pipeline,
        "sources": sources,
        "cli_args": cli_args,
        "schema": {
            "format": "legacy gp_interp_trainset",
            "datasets": {
                "tids": "(N,) int64 — TARGETID per spectrum",
                "rest_wavelengths": "(N, n_pix) float32 — duplicated rest grid; readers should take row 0",
                "fluxes": "(N, n_pix) float32 — observed-frame flux interpolated to rest grid",
                "noise_variance": "(N, n_pix) float32 — pipeline noise variance, NaN-padded for masked pixels",
                "zqso": "(N,) float32",
                "redsnr": "(N,) float32 — red-side SNR (median in rest 1425-1475 Å)",
                "bluesnr": "(N,) float32 — blue-side SNR if available, zeros otherwise",
            },
            "provenance_attrs": (
                "All filter parameters and source paths are also written as "
                "h5 root attributes — `h5py.File(path).attrs.items()` to read"
            ),
        },
        "compatible_loaders": [
            "gpy_dla_detection.training.dataset.load_preprocessed_h5",
            "gpy_dla_detection.learn_qso_model.GPTrainingSetLoader (legacy)",
        ],
        "next_step": "Train with: " + suggested_train_command,
    }

    md_path = parent / "README.md"
    json_path = parent / "dataset_metadata.json"

    json_path.write_text(json.dumps(metadata, indent=2))

    md = []
    md.append(f"# Preloaded training set — `{output_h5.name}`\n")
    md.append(f"**Kind**: `{dataset_kind}`  ")
    md.append(f"**N spectra**: {n_spectra}  ")
    md.append(f"**Rest grid**: {n_pix} pixels in [{rest_min:.2f}, {rest_max:.2f}] Å, dλ = {dlambda} Å  ")
    md.append(f"**z range**: [{z_min}, {z_max}]\n")

    md.append("## Filter pipeline (applied IN ORDER)\n")
    for line in filter_pipeline:
        md.append(f"- {line}")
    md.append("")

    md.append("## Sources\n")
    md.append("| key | path |")
    md.append("|---|---|")
    for k, v in sources.items():
        md.append(f"| `{k}` | `{v}` |")
    md.append("")

    md.append("## CLI arguments (full snapshot)\n")
    md.append("```json")
    md.append(json.dumps(cli_args, indent=2, default=str))
    md.append("```\n")

    md.append("## File schema\n")
    md.append("Legacy `gp_interp_trainset` HDF5. Top-level datasets:\n")
    md.append("| dataset | shape | dtype | meaning |")
    md.append("|---|---|---|---|")
    for name, desc in metadata["schema"]["datasets"].items():
        md.append(f"| `{name}` | — | — | {desc} |")
    md.append("\nAll filter parameters and source paths are also stored as ")
    md.append("h5 root attributes (`with h5py.File(path) as f: dict(f.attrs)`).\n")

    md.append("## How to train on this trainset\n")
    md.append("```bash")
    md.append(suggested_train_command)
    md.append("```\n")

    md.append("Or load programmatically:\n")
    md.append("```python")
    md.append("from gpy_dla_detection.training.dataset import load_preprocessed_h5")
    md.append(f"ts = load_preprocessed_h5('{output_h5.name}', z_min={z_min}, z_max={z_max})")
    md.append("```\n")

    md.append("## Companion files in this folder\n")
    md.append("- `README.md` — this document")
    md.append("- `dataset_metadata.json` — same content, JSON-readable for tooling")
    md.append(f"- `{output_h5.name}` — the actual HDF5")
    md.append(
        "- (after training) `model_epoch_*.h5`, `checkpoint_epoch_*.pt`, "
        "`config.json`, `loss_history.json`, `slurm.log`"
    )

    md_path.write_text("\n".join(md))
    print(f"[readme] wrote {md_path}")
    print(f"[readme] wrote {json_path}")
