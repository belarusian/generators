"""Notebook generator types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from compass.generators._types import Err, Ok, Result


@dataclass(frozen=True)
class Cell:
    """A single notebook cell.

    cell_type: 'code' or 'markdown'
    source: the cell content (Python code or markdown text)
    """

    cell_type: str   # 'code' | 'markdown'
    source: str | None = None


@dataclass(frozen=True)
class NotebookSpec:
    """Complete notebook specification.

    A notebook is a sequence of cells. Each cell is either code or markdown.
    The title appears as the first markdown cell (H1 heading).
    The claim is a one-sentence summary of what the notebook demonstrates.

    Response format -- Python constructor with content blocks:

    NotebookSpec(
        title="Sine Wave Analysis",
        claim="Demonstrates generating and plotting a sine wave.",
        cells=(
            Cell(cell_type="markdown", source="# Sine Wave Analysis"),
            Cell(cell_type="code", source=None),
            Cell(cell_type="code", source=None),
            Cell(cell_type="markdown", source="## Plot"),
            Cell(cell_type="code", source=None),
        ),
    )

    # === index=1 ===
    import numpy as np
    import matplotlib.pyplot as plt
    # === end ===

    # === index=2 ===
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    # === end ===

    # === index=4 ===
    plt.figure()
    plt.plot(x, y)
    plt.title('Sine Wave')
    plt.savefig('sine.png')
    plt.close()
    # === end ===
    """

    title: str
    claim: str
    cells: tuple[Cell, ...]


# ============================================================================
# Patch types for targeted ouroboros
# ============================================================================


@dataclass(frozen=True)
class CellEdit:
    """A targeted edit within a cell: find old text, replace with new."""

    old: str
    new: str


@dataclass(frozen=True)
class CellPatch:
    """Patch for a single cell by index.

    If cell is set, replaces the entire cell.
    If edits is set, applies search-and-replace edits to the cell source.
    Exactly one of cell or edits must be provided.
    """

    index: int
    cell: Cell | None = None
    edits: tuple[CellEdit, ...] = ()


@dataclass(frozen=True)
class NotebookPatch:
    """A partial update to a NotebookSpec.

    Ouroboros returns this instead of the full spec. Each CellPatch
    targets a single cell by index with either full replacement or
    search-and-replace edits. The model only returns the changes.

    NotebookPatch(
        cells=(
            CellPatch(index=2, edits=(
                CellEdit(old="text to find", new="replacement text"),
            )),
        ),
    )

    Or full cell replacement:
    CellPatch(index=2, cell=Cell(cell_type="code", source="new code"))
    """

    cells: tuple[CellPatch, ...]


# ============================================================================
# Validators
# ============================================================================


def validate_spec(raw: dict) -> Result[NotebookSpec, str]:
    """Validate a raw JSON dict into a NotebookSpec. [STRUCTURAL]"""
    if not isinstance(raw, dict):
        return Err(f"expected dict, got {type(raw).__name__}")

    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        return Err("title must be a non-empty string")
    title = title.strip()

    claim = raw.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        return Err("claim must be a non-empty string")
    claim = claim.strip()

    raw_cells = raw.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        return Err("cells must be a non-empty list")

    errors: list[str] = []
    validated: list[Cell] = []

    for i, c in enumerate(raw_cells):
        if not isinstance(c, dict):
            errors.append(f"cells[{i}]: expected dict")
            continue

        cell_type = c.get("cell_type")
        if cell_type not in ("code", "markdown"):
            errors.append(f"cells[{i}].cell_type: must be 'code' or 'markdown', got {cell_type!r}")
            continue

        source = c.get("source")
        if not isinstance(source, str):
            errors.append(f"cells[{i}].source: must be a string")
            continue

        validated.append(Cell(cell_type=cell_type, source=source))

    if errors:
        return Err("; ".join(errors))

    if not validated:
        return Err("cells must contain at least one cell")

    # Must have at least one code cell
    code_cells = [c for c in validated if c.cell_type == "code"]
    if not code_cells:
        return Err("notebook must contain at least one code cell")

    return Ok(NotebookSpec(
        title=title,
        claim=claim,
        cells=tuple(validated),
    ))


def validate_spec_instance(spec: NotebookSpec) -> Result[NotebookSpec, str]:
    """Validate a NotebookSpec instance (from Python-as-schema)."""
    errors: list[str] = []
    if not spec.title:
        errors.append("title must be non-empty")
    if not spec.claim:
        errors.append("claim must be non-empty")
    if not spec.cells:
        errors.append("cells must be non-empty")
    for i, c in enumerate(spec.cells):
        if c.cell_type not in ("code", "markdown"):
            errors.append(f"cells[{i}].cell_type: must be 'code' or 'markdown'")
        if c.source is None:
            errors.append(f"cells[{i}].source: content is empty")
    code_cells = [c for c in spec.cells if c.cell_type == "code"]
    if not code_cells:
        errors.append("notebook must contain at least one code cell")
    if errors:
        return Err("; ".join(errors))
    return Ok(spec)


def _validate_cell_dict(raw: dict, label: str) -> Result[Cell, str]:
    """Validate a raw dict into a Cell."""
    cell_type = raw.get("cell_type")
    if cell_type not in ("code", "markdown"):
        return Err(f"{label}.cell_type: must be 'code' or 'markdown', got {cell_type!r}")
    source = raw.get("source")
    if not isinstance(source, str):
        return Err(f"{label}.source: must be a string")
    return Ok(Cell(cell_type=cell_type, source=source))


def validate_notebook_patch(raw: dict) -> Result[NotebookPatch, str]:
    """Validate a raw JSON dict into a NotebookPatch. [STRUCTURAL]"""
    if not isinstance(raw, dict):
        return Err(f"expected dict, got {type(raw).__name__}")

    raw_cells = raw.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        return Err("cells must be a non-empty list")

    errors: list[str] = []
    validated: list[CellPatch] = []
    seen_indices: set[int] = set()

    for i, cp in enumerate(raw_cells):
        if not isinstance(cp, dict):
            errors.append(f"cells[{i}]: expected dict")
            continue

        index = cp.get("index")
        if not isinstance(index, int) or index < 0:
            errors.append(f"cells[{i}].index: must be a non-negative integer")
            continue

        if index in seen_indices:
            errors.append(f"cells[{i}].index: duplicate index {index}")
            continue
        seen_indices.add(index)

        raw_cell = cp.get("cell")
        raw_edits = cp.get("edits")
        has_cell = isinstance(raw_cell, dict)
        has_edits = isinstance(raw_edits, list) and len(raw_edits) > 0

        if not has_cell and not has_edits:
            errors.append(f"cells[{i}]: must have 'cell' or 'edits'")
            continue

        if has_cell and has_edits:
            errors.append(f"cells[{i}]: provide 'cell' or 'edits', not both")
            continue

        if has_cell:
            match _validate_cell_dict(raw_cell, f"cells[{i}].cell"):
                case Err(e):
                    errors.append(e)
                    continue
                case Ok(cell):
                    validated.append(CellPatch(index=index, cell=cell))
                    continue

        # has_edits
        edits: list[CellEdit] = []
        for j, e in enumerate(raw_edits):
            if not isinstance(e, dict):
                errors.append(f"cells[{i}].edits[{j}]: expected dict")
                continue
            old = e.get("old")
            new = e.get("new")
            if not isinstance(old, str) or not old:
                errors.append(f"cells[{i}].edits[{j}].old: must be a non-empty string")
                continue
            if not isinstance(new, str):
                errors.append(f"cells[{i}].edits[{j}].new: must be a string")
                continue
            edits.append(CellEdit(old=old, new=new))

        if edits:
            validated.append(CellPatch(index=index, edits=tuple(edits)))

    if errors:
        return Err("; ".join(errors))

    return Ok(NotebookPatch(cells=tuple(validated)))


def apply_notebook_patch(
    spec: NotebookSpec,
    patch: NotebookPatch,
) -> Result[NotebookSpec, str]:
    """Apply a NotebookPatch to a NotebookSpec.

    Each CellPatch targets a cell by index. Either replaces the entire
    cell or applies search-and-replace edits to the cell source.
    If an old string is not found, the patch fails with a precise error.
    """
    cells = list(spec.cells)
    errors: list[str] = []

    for cp in patch.cells:
        if cp.index < 0 or cp.index >= len(cells):
            errors.append(f"cells[{cp.index}]: index out of range (notebook has {len(cells)} cells)")
            continue

        if cp.cell is not None:
            # Full cell replacement
            cells[cp.index] = cp.cell
        else:
            # Targeted edits
            source = cells[cp.index].source
            for edit in cp.edits:
                if edit.old not in source:
                    errors.append(
                        f"cells[{cp.index}]: old text not found: {edit.old[:80]}..."
                    )
                    continue
                source = source.replace(edit.old, edit.new, 1)
            cells[cp.index] = Cell(
                cell_type=cells[cp.index].cell_type,
                source=source,
            )

    if errors:
        return Err("; ".join(errors))

    return Ok(NotebookSpec(
        title=spec.title,
        claim=spec.claim,
        cells=tuple(cells),
    ))
