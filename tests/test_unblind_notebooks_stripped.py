"""Privacy guard: the UNBLIND notebooks must never carry executed outputs.

`notebooks/UNBLIND_*.ipynb` render real-LoA (DESI DR2 Loa main-dark) CDDF results
-- dN/dX, Omega, f(N), per-z tables. Those values are private and must not reach the
public repository, including as cell outputs, embedded figures, or execution counts.

A .gitignore cannot enforce this: we *want* these notebooks committed, stripped. The
distinction between a stripped notebook and an executed one is content, not path.

Scope is deliberately narrow. Every other tracked notebook in `notebooks/` embeds its
figures on purpose (tutorials, plot galleries); this guard must not touch them.
"""

import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
UNBLIND_GLOB = "UNBLIND_*.ipynb"


def _unblind_notebooks():
    return sorted(NOTEBOOK_DIR.glob(UNBLIND_GLOB))


def test_unblind_notebooks_exist():
    """Guard against the glob silently matching nothing after a rename."""
    found = _unblind_notebooks()
    assert found, (
        f"no notebooks matched {NOTEBOOK_DIR}/{UNBLIND_GLOB}. If they were renamed, "
        "update UNBLIND_GLOB -- an empty glob makes every check below vacuous."
    )


@pytest.mark.parametrize("nb_path", _unblind_notebooks(), ids=lambda p: p.name)
def test_no_outputs_and_no_execution_counts(nb_path):
    nb = json.loads(nb_path.read_text())
    offenders = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            offenders.append(f"cell {i}: {len(cell['outputs'])} output(s)")
        if cell.get("execution_count") is not None:
            offenders.append(f"cell {i}: execution_count={cell['execution_count']}")

    assert not offenders, (
        f"{nb_path.name} carries executed state, which may embed real-LoA values.\n"
        + "\n".join(f"  {o}" for o in offenders)
        + "\n\nStrip before committing, e.g.:\n"
        f"  jupyter nbconvert --clear-output --inplace {nb_path}"
    )
