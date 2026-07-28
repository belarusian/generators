"""
Fact dispatch - singledispatch for type-based fact presentation.

Separates the canonical value (raw data) from its presentation.
Same pattern as oracle-/compass/agents/neo/dispatch.py but for Facts.

    display_fact(fact) -> str    # formatted for REPL / model context
    resolve_fact(fact) -> Any    # raw value for $fact references
"""

from __future__ import annotations

import json
from functools import singledispatch
from typing import Any

from compass.generators.trinity._types import ErrorFact, Fact, FileFact

# -- read_file pagination constants (moved from _runtime.py) ----------------
_READ_FILE_LIMIT = 200
_HEAD_LINES = 120
_TAIL_LINES = 80


# ============================================================================
# display_fact -- formatted for human / model viewing
# ============================================================================


@singledispatch
def display_fact(fact) -> str:
    """Format a fact for display (REPL output, synthesis context).

    Default: return raw value unchanged.
    """
    return fact.value


@display_fact.register(FileFact)
def _display_file(fact: FileFact) -> str:
    """Line-numbered output with adaptive pagination for large files."""
    lines = fact.value.splitlines()
    total = len(lines)
    base = fact.line_offset  # 0-based offset in original file

    if total == 0:
        return "(empty file)"

    if total <= _READ_FILE_LIMIT:
        numbered = [f"line {base + i + 1}: {l}" for i, l in enumerate(lines)]
        return "\n".join(numbered)

    # Head + tail with omission separator
    head = [f"line {base + i + 1}: {lines[i]}" for i in range(_HEAD_LINES)]
    gap = total - _HEAD_LINES - _TAIL_LINES
    separator = f"    ... [{gap} lines omitted -- use offset/limit to read sections] ..."
    tail_start = total - _TAIL_LINES
    tail = [
        f"line {base + tail_start + i + 1}: {lines[tail_start + i]}"
        for i in range(_TAIL_LINES)
    ]
    header = f"[Lines {base + 1}-{base + _HEAD_LINES} + {base + tail_start + 1}-{base + total} of {base + total}]"
    return header + "\n" + "\n".join(head + [separator] + tail)


@display_fact.register(ErrorFact)
def _display_error(fact: ErrorFact) -> str:
    """Error message as-is."""
    return fact.value


# ============================================================================
# resolve_fact -- raw value for downstream $fact references
# ============================================================================


@singledispatch
def resolve_fact(fact) -> Any:
    """Resolve a fact to a Python value for $fact input references.

    Default: try JSON deserialization, fall back to raw string.
    """
    try:
        return json.loads(fact.value)
    except (json.JSONDecodeError, TypeError):
        return fact.value


@resolve_fact.register(FileFact)
def _resolve_file(fact: FileFact) -> str:
    """Raw file content -- no line numbers, no headers."""
    return fact.value


@resolve_fact.register(ErrorFact)
def _resolve_error(fact: ErrorFact) -> str:
    """Error message string."""
    return fact.value
