"""privacy.py -- self-check that no real-LOA numerics leak into committed notebook outputs.

Real-LOA (DESI DR2 Loa) result values (dN/dX, Omega, f(N), per-z tables, over-count
percentages) are PRIVATE.  Executed notebook outputs contain them, so the committed
notebook MUST have zero outputs.  These helpers let a notebook self-audit before commit.

The strong guarantee is ``assert_no_outputs``: a committed notebook with zero code-cell
outputs cannot leak a value.  ``scan_notebook_outputs`` is a softer heuristic that flags
decimal numbers surviving in any output, for the case where a notebook was executed and
someone forgot to clear it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# decimals that look like a scientific result (a dot with digits either side, or sci-notation).
_NUM = re.compile(r"(?<![\w.])\d+\.\d+(?:[eE][+-]?\d+)?(?![\w.])")


@dataclass
class OutputHit:
    cell_index: int
    output_index: int
    kind: str            # 'stream' / 'execute_result' / 'display_data' / 'error'
    n_numbers: int
    sample: str          # a short, truncated excerpt (may itself contain numbers!)


def _iter_output_text(out):
    """Yield text payloads from a single nbformat output object."""
    t = out.get("output_type")
    if t == "stream":
        yield "".join(out.get("text", []) if isinstance(out.get("text"), list) else [out.get("text", "")])
    elif t in ("execute_result", "display_data"):
        data = out.get("data", {})
        txt = data.get("text/plain", "")
        yield "".join(txt) if isinstance(txt, list) else txt
    elif t == "error":
        yield "\n".join(out.get("traceback", []))


def assert_no_outputs(nb_path: str) -> int:
    """Hard guarantee: raise if ANY code cell carries outputs.  Returns 0 on success.

    A committed notebook that passes this cannot leak a real-LOA value."""
    with open(nb_path) as f:
        nb = json.load(f)
    offenders = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            offenders.append(i)
    if offenders:
        raise RuntimeError(
            f"{nb_path}: {len(offenders)} code cell(s) still carry outputs {offenders}. "
            "Run `jupyter nbconvert --clear-output --inplace <nb>` before committing -- "
            "executed outputs contain real-LOA values."
        )
    return 0


def scan_notebook_outputs(nb_path: str, max_sample: int = 60) -> list:
    """Heuristic: return OutputHit rows for every output containing decimal numbers.

    Use as a soft tripwire when a notebook was executed.  Empty list == clean/cleared."""
    with open(nb_path) as f:
        nb = json.load(f)
    hits = []
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for oi, out in enumerate(cell.get("outputs", []) or []):
            for text in _iter_output_text(out):
                nums = _NUM.findall(text or "")
                if nums:
                    hits.append(OutputHit(ci, oi, out.get("output_type", "?"),
                                          len(nums), (text or "")[:max_sample]))
    return hits


CLEAR_INSTRUCTION = (
    "Before committing this notebook, STRIP all outputs (they embed real-LOA values):\n"
    "    jupyter nbconvert --clear-output --inplace notebooks/UNBLIND_00_guards_and_data.ipynb\n"
    "and keep executed copies out of git (see the .gitignore note in the handoff)."
)
