"""IO boundaries for the notebook generator.

V_notebook: validate Python sources, then exec each code cell.
Ouroboros uses patch-based approach: model returns only changed cells.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from compass.generators._types import (
    AskFn,
    Err,
    GenerationContext,
    Ok,
    Result,
)
from compass.generators._invoke import (
    build_system_prompt,
    build_user_message,
    resolve_ask_fn,
)
from compass.generators._validation import (
    validate_python_sources,
)

from compass.generators.notebook._types import (
    Cell,
    NotebookPatch,
    NotebookSpec,
    apply_notebook_patch,
)

logger = logging.getLogger(__name__)

_TYPES_MODULE = (Path(__file__).parent / "_types.py").read_text()


# ---------------------------------------------------------------------------
# Model invocation
# ---------------------------------------------------------------------------


def invoke_model(
    ctx: GenerationContext,
    model_id: str = "",
    ask_fn: AskFn | None = None,
) -> Result:
    """Call the model and return a NotebookSpec. G_notebook.

    Python-as-schema: model writes a Python constructor expression +
    content blocks. No JSON, no escaping.
    """
    fn = resolve_ask_fn(model_id, ask_fn)

    system = build_system_prompt(
        ctx,
        _TYPES_MODULE,
        role=(
            "You are an expert Jupyter notebook author. "
            "You generate well-structured, executable notebooks "
            "with clear explanations and correct Python code."
        ),
        contract_preamble=(
            "Respond with a Python expression constructing NotebookSpec.\n"
            "See the NotebookSpec docstring for the exact response format."
        ),
    )

    user = build_user_message(
        ctx,
        suffix_lines=(
            "Write a NotebookSpec(...) expression.",
            "Short metadata fields (cell_type, index) go inline in the constructor.",
            "No markdown fencing.",
            "",
            "Guidelines:",
            "- Start with a markdown cell containing the title as an H1 heading.",
            "- Interleave markdown explanation cells with code cells.",
            "- Each code cell should be focused and self-contained where possible.",
            "- Import all libraries in the first code cell.",
            "- Use matplotlib.use('Agg') before importing pyplot if generating plots,",
            "  and save figures with plt.savefig() rather than plt.show().",
            "- All code must be executable without user interaction.",
        ),
    )

    match fn(system, user):
        case Err() as e:
            return e
        case Ok(raw_text):
            pass

    from compass.core.python_schema import parse_typed_response
    try:
        spec = parse_typed_response(raw_text, NotebookSpec)
        return Ok(spec)
    except ValueError as e:
        return Err(str(e))


# ---------------------------------------------------------------------------
# Validation pipeline -- V_notebook
# ---------------------------------------------------------------------------


def validate_notebook(spec: NotebookSpec) -> Result[NotebookSpec, str]:
    """Full V_notebook pipeline. Cheapest first:

    1. Structural: validate_spec already done by parse step
    2. Semantic: ast.parse() every code cell
    3. Executive: exec() every code cell in sequence
    """
    code_sources = tuple(
        c.source for c in spec.cells if c.cell_type == "code"
    )

    # 2. Semantic: syntax check all code cells
    match validate_python_sources(code_sources, label="cells"):
        case Err(e):
            return Err(e)

    logger.info("All %d code cells parse cleanly", len(code_sources))

    # 3. Executive: exec each code cell in sequence
    match _exec_notebook(spec):
        case Err(e):
            return Err(e)
        case Ok(_):
            pass

    logger.info("All code cells executed successfully")
    return Ok(spec)


def _exec_notebook(spec: NotebookSpec) -> Result[None, str]:
    """Execute all code cells in sequence, sharing a namespace."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for exec

    namespace: dict[str, Any] = {"__name__": "__notebook__"}

    for i, cell in enumerate(spec.cells):
        if cell.cell_type != "code":
            continue
        if not cell.source.strip():
            continue

        try:
            exec(compile(cell.source, f"<cell {i}>", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            return Err(
                f"cells[{i}]: {type(exc).__name__}: {exc}"
            )

    return Ok(None)


# ---------------------------------------------------------------------------
# Ouroboros -- notebook-level (patch-based)
# ---------------------------------------------------------------------------


def ouroboros_notebook(
    spec: NotebookSpec,
    error: str,
    ctx: GenerationContext,
    model_id: str = "",
    ask_fn: AskFn | None = None,
) -> NotebookSpec | None:
    """G'_notebook: model sees its prior notebook + error, returns a NotebookPatch.

    The model returns a NotebookPatch (only the changed cells), not the
    full spec. The patch is applied to the original to produce the
    corrected spec. This keeps the output small -- the model doesn't
    have to reproduce unchanged cells.
    """
    fn = resolve_ask_fn(model_id, ask_fn)

    # Build a compact listing of all cells
    cell_listing: list[str] = []
    for i, cell in enumerate(spec.cells):
        cell_listing.append(f"### Cell {i} ({cell.cell_type})")
        cell_listing.append(cell.source)

    system_parts = [
        "You are correcting a Jupyter notebook you previously generated.",
        f"Notebook: {spec.title}",
        "",
        "You will receive your prior notebook cells and an error.",
        "Write a NotebookPatch(...) expression with targeted edits.",
        "Each patch targets a cell by index with either full replacement or search-and-replace edits.",
        "Only include the cells that need to change.",
        "",
        "NotebookPatch format (targeted edits -- preferred for surgical changes):",
        "NotebookPatch(",
        "    cells=(",
        '        CellPatch(index=2, edits=(',
        '            CellEdit(old="exact text to find", new="replacement text"),',
        "        )),",
        "    ),",
        ")",
        "",
        "NotebookPatch format (full cell replacement):",
        'CellPatch(index=2, cell=Cell(cell_type="code", source="..."))',
        "",
        "Rules:",
        "- Use edits for surgical changes. Use cell for full replacement.",
        "- 'old' must be an EXACT substring of the current cell source.",
        "- 'old' should include enough surrounding context to be unique.",
        "- 'new' is the complete replacement for that substring.",
        "- Each cell patch must have either 'edits' or 'cell', not both.",
        "- Use matplotlib.use('Agg') before pyplot imports if plotting.",
        "- Save figures with plt.savefig() not plt.show().",
        "- No markdown fencing, no explanation.",
    ]

    user_parts = [
        f"# Your notebook: {spec.title} ({len(spec.cells)} cells)",
        "",
        "\n".join(cell_listing),
        "",
        "# Error",
        "",
        error,
        "",
        "Write a NotebookPatch(...) expression with targeted edits to fix this error.",
    ]

    match fn("\n".join(system_parts), "\n".join(user_parts)):
        case Err(e):
            logger.warning("Ouroboros model error: %s", e)
            return None
        case Ok(raw_text):
            pass

    from compass.core.python_schema import parse_typed_response
    try:
        patch = parse_typed_response(raw_text, NotebookPatch)
    except ValueError as e:
        logger.warning(
            "Ouroboros parse error: %s\n--- raw response (first 1000 chars) ---\n%s",
            str(e), raw_text[:1000],
        )
        return None

    total_edits = sum(len(cp.edits) for cp in patch.cells)
    total_replacements = sum(1 for cp in patch.cells if cp.cell is not None)
    logger.info(
        "Ouroboros patch: %d cell(s), %d edit(s), %d replacement(s): indices %s",
        len(patch.cells), total_edits, total_replacements,
        [cp.index for cp in patch.cells],
    )

    match apply_notebook_patch(spec, patch):
        case Err(e):
            logger.warning("Ouroboros patch apply failed: %s", e)
            return None
        case Ok(corrected):
            return corrected


# ---------------------------------------------------------------------------
# Emit -- write notebook to disk as .ipynb
# ---------------------------------------------------------------------------


def _cell_to_ipynb(cell: Cell) -> dict:
    """Convert a Cell to Jupyter notebook cell format."""
    source_lines = cell.source.splitlines(keepends=True)
    if source_lines and not source_lines[-1].endswith("\n"):
        source_lines[-1] = source_lines[-1]  # last line no trailing newline

    base = {
        "cell_type": cell.cell_type,
        "metadata": {},
        "source": source_lines,
    }

    if cell.cell_type == "code":
        base["execution_count"] = None
        base["outputs"] = []

    return base


def emit_notebook(
    spec: NotebookSpec,
    output_dir: Path,
    version: int = 0,
) -> Path:
    """Write a NotebookSpec to disk as a .ipynb file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize title for filename
    safe_title = "".join(
        c if c.isalnum() or c in " _-" else "_"
        for c in spec.title
    ).strip().replace(" ", "_").lower()

    if version > 0:
        filename = f"{safe_title}_v{version}.ipynb"
    else:
        filename = f"{safe_title}.ipynb"

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
            "notebook_generator": {
                "title": spec.title,
                "claim": spec.claim,
            },
        },
        "cells": [_cell_to_ipynb(c) for c in spec.cells],
    }

    out_path = output_dir / filename
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    logger.info("Notebook written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Load -- read .ipynb back into NotebookSpec
# ---------------------------------------------------------------------------


def load_notebook(path: Path) -> Result[NotebookSpec, str]:
    """Load a .ipynb file into a NotebookSpec."""
    if not path.exists():
        return Err(f"File not found: {path}")

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return Err(f"JSON parse error: {e}")

    cells_raw = raw.get("cells", [])
    meta = raw.get("metadata", {}).get("notebook_generator", {})
    title = meta.get("title", path.stem)
    claim = meta.get("claim", "")

    cells: list[Cell] = []
    for c in cells_raw:
        source = c.get("source", [])
        if isinstance(source, list):
            source = "".join(source)
        cells.append(Cell(
            cell_type=c.get("cell_type", "code"),
            source=source,
        ))

    return Ok(NotebookSpec(
        title=title,
        claim=claim,
        cells=tuple(cells),
    ))
