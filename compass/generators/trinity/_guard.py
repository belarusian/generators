"""Step guard -- pre-execution approval and post-execution file diffs.

Gated by TRINITY_GUARD=1 env var. When enabled:
  1. Before each step: show a preview of what will run, prompt Y/n
  2. After each step: detect changed files, show colored unified diffs

The guard is interactive-only. In non-TTY environments it auto-approves.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from compass.generators._ui import Colors
from compass.generators.trinity._types import Step


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

_guard_override: bool | None = None  # None = defer to env var


def is_guard_enabled() -> bool:
    if _guard_override is not None:
        return _guard_override
    return os.environ.get("TRINITY_GUARD") == "1"


def set_guard(enabled: bool | None) -> None:
    """Set guard state. None resets to env var."""
    global _guard_override
    _guard_override = enabled


# ---------------------------------------------------------------------------
# Step preview -- what is about to run
# ---------------------------------------------------------------------------

_SKIP_EXTENSIONS = frozenset({
    ".pyc", ".pyo", ".so", ".dylib", ".o", ".a",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
    ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".whl", ".egg",
})

_SKIP_DIRS = frozenset({
    "__pycache__", ".git", ".hg", ".svn", "node_modules",
    ".tox", ".venv", "venv", ".mypy_cache", ".pytest_cache",
})


def build_step_preview(step: Step, resolved_inputs: dict) -> str:
    """Build a human-readable preview of what a step will execute."""
    lines: list[str] = []

    if step.artifact_type == "inline_python":
        code = step.artifact_ref or ""
        code_lines = code.strip().splitlines()
        lines.append(f"Run inline Python ({len(code_lines)} lines):")
        lines.append("")
        for i, cl in enumerate(code_lines[:30], 1):
            lines.append(f"  {i:3d}  {cl}")
        if len(code_lines) > 30:
            lines.append(f"  ... ({len(code_lines) - 30} more lines)")

    elif step.artifact_type == "shell":
        cmd = step.artifact_ref or "(empty)"
        lines.append(f"Run shell command:")
        lines.append(f"  $ {cmd}")
        if resolved_inputs:
            lines.append("")
            lines.append("Env vars:")
            for k, v in resolved_inputs.items():
                val = str(v)
                if len(val) > 80:
                    val = val[:77] + "..."
                lines.append(f"  {k}={val}")

    elif step.artifact_type == "auto":
        ref = step.artifact_ref or "?"
        lines.append(f"Call artifact: {ref}")
        if resolved_inputs:
            lines.append("")
            lines.append("Arguments:")
            for k, v in resolved_inputs.items():
                val = str(v)
                if len(val) > 80:
                    val = val[:77] + "..."
                lines.append(f"  {k}={val}")

    elif step.artifact_type == "vision":
        ref = step.artifact_ref or "?"
        lines.append(f"Vision (read-only): send image to model")
        lines.append(f"  image: {ref}")

    else:
        lines.append(f"Unknown type: {step.artifact_type}")
        if step.artifact_ref:
            lines.append(f"  ref: {step.artifact_ref}")

    if step.expected_fact:
        lines.append("")
        lines.append(f"Produces: {step.expected_fact}")
        if step.extraction_expr and step.extraction_expr != "result":
            lines.append(f"  via: {step.extraction_expr}")

    return "\n".join(lines)


def format_preview_box(step: Step, preview: str) -> str:
    """Wrap a step preview in a bordered box for display."""
    tag = step.artifact_type
    header = f".--- {step.step_id} [{tag}] ---"
    desc = f"| {step.description}"

    lines = [
        Colors.cyan(header),
        Colors.cyan(desc),
        Colors.cyan("|"),
    ]
    for pl in preview.splitlines():
        lines.append(f"{Colors.cyan('|')} {Colors.dim(pl)}")
    lines.append(Colors.cyan("'---"))

    return "\n    ".join(lines)


# ---------------------------------------------------------------------------
# Approval prompt
# ---------------------------------------------------------------------------

def prompt_approve(step: Step, preview: str) -> bool:
    """Show preview, ask Y/n. Auto-approves in non-TTY."""
    if not sys.stdout.isatty():
        return True

    box = format_preview_box(step, preview)
    print(f"    {box}")
    print()
    try:
        answer = input(f"    {Colors.yellow('approve?')} [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    return answer in ("", "y", "yes")


# ---------------------------------------------------------------------------
# Workspace snapshot and diff
# ---------------------------------------------------------------------------

@dataclass
class FileDiff:
    """A single file change detected after step execution."""

    path: str                    # relative to workspace
    change_type: str             # "added" | "modified" | "deleted"
    unified_diff: list[str]      # diff lines (no color)
    size_before: int = 0
    size_after: int = 0


class WorkspaceSnapshot:
    """Captures file state for before/after diffing.

    Takes a hash + mtime of every text file in the workspace.
    After execution, call diff() to get a list of changes.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self._files: dict[str, tuple[float, int, str, str]] = {}  # rel -> (mtime, size, hash, content)
        self._scan()

    def _scan(self) -> None:
        if not self.workspace.is_dir():
            return
        for path in self.workspace.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in _SKIP_EXTENSIONS:
                continue
            if any(d in path.parts for d in _SKIP_DIRS):
                continue
            rel = str(path.relative_to(self.workspace))
            try:
                stat = path.stat()
                content = path.read_text(errors="replace")
                h = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
                self._files[rel] = (stat.st_mtime, stat.st_size, h, content)
            except (OSError, UnicodeDecodeError):
                continue

    def diff(self) -> list[FileDiff]:
        """Compare current workspace state against the snapshot."""
        diffs: list[FileDiff] = []
        current: dict[str, tuple[int, str, str]] = {}

        # Scan current state
        if self.workspace.is_dir():
            for path in self.workspace.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix in _SKIP_EXTENSIONS:
                    continue
                if any(d in path.parts for d in _SKIP_DIRS):
                    continue
                rel = str(path.relative_to(self.workspace))
                try:
                    stat = path.stat()
                    content = path.read_text(errors="replace")
                    h = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
                    current[rel] = (stat.st_size, h, content)
                except (OSError, UnicodeDecodeError):
                    continue

        # Detect changes
        for rel, (size, h, content) in current.items():
            if rel not in self._files:
                # New file
                diff_lines = list(difflib.unified_diff(
                    [], content.splitlines(keepends=True),
                    fromfile=f"a/{rel}", tofile=f"b/{rel}",
                ))
                diffs.append(FileDiff(
                    path=rel,
                    change_type="added",
                    unified_diff=diff_lines,
                    size_after=size,
                ))
            else:
                old_mtime, old_size, old_h, old_content = self._files[rel]
                if h != old_h:
                    diff_lines = list(difflib.unified_diff(
                        old_content.splitlines(keepends=True),
                        content.splitlines(keepends=True),
                        fromfile=f"a/{rel}", tofile=f"b/{rel}",
                    ))
                    diffs.append(FileDiff(
                        path=rel,
                        change_type="modified",
                        unified_diff=diff_lines,
                        size_before=old_size,
                        size_after=size,
                    ))

        for rel, (mtime, size, h, content) in self._files.items():
            if rel not in current:
                diff_lines = list(difflib.unified_diff(
                    content.splitlines(keepends=True), [],
                    fromfile=f"a/{rel}", tofile=f"b/{rel}",
                ))
                diffs.append(FileDiff(
                    path=rel,
                    change_type="deleted",
                    unified_diff=diff_lines,
                    size_before=size,
                ))

        return diffs


# ---------------------------------------------------------------------------
# Colored diff display
# ---------------------------------------------------------------------------

_MAX_DIFF_LINES = 50


def format_diff(diffs: list[FileDiff]) -> str:
    """Render file diffs with color.

    Aesthetic:
      ,--- modified: src/data.py (+20 bytes) ---
      | --- a/src/data.py
      | +++ b/src/data.py
      | @@ -12,3 +12,5 @@
      |    existing line
      |   -old line
      |   +new line
      `---
    """
    if not diffs:
        return ""

    parts: list[str] = []
    parts.append("")

    for fd in diffs:
        # Size annotation
        if fd.change_type == "added":
            tag_colored = Colors.green(fd.change_type)
            size_note = f" ({fd.size_after} bytes)"
        elif fd.change_type == "deleted":
            tag_colored = Colors.red(fd.change_type)
            size_note = f" ({fd.size_before} bytes)"
        else:
            tag_colored = Colors.yellow(fd.change_type)
            delta = fd.size_after - fd.size_before
            sign = "+" if delta >= 0 else ""
            size_note = f" ({sign}{delta} bytes)"

        # Top rule with label
        header = f"{fd.change_type}: {fd.path}{size_note}"
        parts.append(f"    {Colors.cyan(',---')} {tag_colored}: {Colors.bold(fd.path)}{Colors.dim(size_note)} {Colors.cyan('---')}")

        # Diff lines
        shown = 0
        total_lines = len(fd.unified_diff)
        for line in fd.unified_diff:
            line = line.rstrip("\n")
            if shown >= _MAX_DIFF_LINES:
                remaining = total_lines - shown
                parts.append(f"    {Colors.cyan('|')} {Colors.dim(f'... ({remaining} more lines)')}")
                break

            if line.startswith("---"):
                parts.append(f"    {Colors.cyan('|')} {Colors.bold(Colors.red(line))}")
            elif line.startswith("+++"):
                parts.append(f"    {Colors.cyan('|')} {Colors.bold(Colors.green(line))}")
            elif line.startswith("@@"):
                parts.append(f"    {Colors.cyan('|')} {Colors.magenta(line)}")
            elif line.startswith("+"):
                parts.append(f"    {Colors.cyan('|')} {Colors.green(line)}")
            elif line.startswith("-"):
                parts.append(f"    {Colors.cyan('|')} {Colors.red(line)}")
            else:
                parts.append(f"    {Colors.cyan('|')} {Colors.dim(line)}")
            shown += 1

        # Bottom rule
        parts.append(f"    {Colors.cyan('`---')}")

    return "\n".join(parts)
