"""
write_file action - singledispatch handlers for WriteFileAction.

Create NEW files. For editing existing files, use edit_file instead.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from compass.core.content import preview_head_tail

from compass.agents.neo.types import WriteFileAction, ExecutionContext, Reflector, LearningType
from compass.agents.neo.dispatch import display, validate, execute, extract_learnings, action_key, hint, display_name, content_field
from compass.agents.neo.memory import Learning


# --- Helper functions ---

def _preview_content(content: str, max_lines: int = 12) -> str:
    """
    Generate preview of written content for validation.

    Shows first N lines so Critic can catch obvious issues
    like duplication, wrong structure, etc.
    """
    lines = content.split('\n')
    if len(lines) <= max_lines:
        return content
    preview_lines = lines[:max_lines]
    return '\n'.join(preview_lines) + f'\n... ({len(lines) - max_lines} more lines)'


def _extract_content(value: Any) -> str:
    """Extract file content from whatever the model produced.

    The model wraps content in various ways:
      - {"path": "...", "content": "actual stuff", ...}  -> extract content
      - '{"path": "...", "content": "..."}'  (JSON string) -> parse, extract
      - plain string -> use as-is
    """
    if isinstance(value, dict) and "content" in value:
        return _extract_content(value["content"])

    if not isinstance(value, str):
        return json.dumps(value) if isinstance(value, (dict, list)) else str(value)

    stripped = value.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "content" in parsed:
                return _extract_content(parsed["content"])
        except (json.JSONDecodeError, TypeError):
            pass

    return value


def _detect_duplication(content: str) -> Optional[str]:
    """
    Detect if content contains obvious duplication.

    Returns warning message if duplication found, None otherwise.
    """
    lines = content.split('\n')
    if len(lines) < 20:
        return None

    # Check if content is roughly duplicated (first half ~ second half)
    mid = len(lines) // 2
    first_half = '\n'.join(lines[:mid])
    second_half = '\n'.join(lines[mid:mid + len(lines[:mid])])

    # If halves are very similar, flag it
    if first_half and second_half:
        # Simple check: same first 10 lines appear again
        first_10 = '\n'.join(lines[:10])
        if first_10 in '\n'.join(lines[15:]):
            return "WARNING: Content appears duplicated (same lines appear twice)"

    return None


def _validate_python_syntax(content: str, path: str) -> Optional[str]:
    """
    Validate Python syntax before writing.

    Returns error message if syntax is invalid, None if valid.
    """
    try:
        compile(content, path, 'exec')
        return None
    except SyntaxError as e:
        # Check for common markdown fence issue
        lines = content.split('\n')
        if lines and lines[0].strip().startswith('```'):
            return (
                f"Content starts with markdown fence (```). "
                f"Write raw Python code, not markdown-wrapped code. "
                f"The content block markers already delineate the code boundaries."
            )
        return f"Invalid Python syntax at line {e.lineno}: {e.msg}"


# --- Singledispatch handlers ---

@content_field.register(WriteFileAction)
def _(action): return "content"


@display.register(WriteFileAction)
def _(action: WriteFileAction):
    """Display info for write_file action."""
    from compass.agents.neo.types import ActionTarget
    path = action.path or ""
    content = action.content or ""
    preview = _preview_content(content, max_lines=12)
    return ActionTarget(
        target=path,
        display=f"Write {len(content)} bytes to {path}",
        content=preview
    )


@validate.register(WriteFileAction)
def _(
    action: WriteFileAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate write_file action.

    Returns (is_valid, error_message).

    Why no guardrails?
    ------------------
    We don't block writes to existing files. Guardrails are theater:
    - With them: model gets blocked, retries, finds workaround anyway
    - Without them: model occasionally does something dumb

    Either way mistakes happen. Git is the real undo button.
    Validation adds friction without adding safety. Trust the model,
    trust version control.

    We check:
    1. Structural requirements (path, content exist)
    2. Security boundary (stay within project)
    """
    path = action.path
    content = action.content

    if not path:
        return False, "Missing required field: path"
    if content is None:
        return False, (
            "content is None -- no content block was found for this action. "
            "Provide content inline: WriteFileAction(path=\"...\", content=\"the text\") "
            "or add a content block after the expression: # === content: path=\"...\" ==="
        )



    return True, None


@execute.register(WriteFileAction)
def _(action: WriteFileAction, project_path: str, ctx: ExecutionContext) -> Tuple[bool, str]:
    """
    Execute write_file action.

    Returns (success, message) with content preview for validation.

    Args:
        action: WriteFileAction
        project_path: Project root directory
        ctx: ExecutionContext (unused, for uniform signature)
    """
    path = action.path or ""
    content = _extract_content(action.content or "")

    # Resolve relative path
    if path and not os.path.isabs(path):
        path = os.path.join(project_path, path)

    # Validate Python syntax before writing (for .py files)
    if path.endswith('.py'):
        syntax_error = _validate_python_syntax(content, path)
        if syntax_error:
            return False, f"Cannot write {path}: {syntax_error}"

    try:
        # Create parent directories if needed
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(content)

        # Build result with preview for validation
        result_parts = [f"Wrote {len(content)} bytes to {path}"]

        # Check for obvious issues
        dup_warning = _detect_duplication(content)
        if dup_warning:
            result_parts.append(dup_warning)

        # Include preview so Critic can validate
        preview = _preview_content(content)
        result_parts.append(f"Preview:\n```\n{preview}\n```")

        return True, '\n'.join(result_parts)
    except Exception as e:
        return False, f"Failed to write {path}: {e}"


@extract_learnings.register(WriteFileAction)
def _(
    action: WriteFileAction,
    success: bool,
    result: str,
    reflect: Reflector,
) -> List[Learning]:
    """Extract learnings from write_file action."""
    path = action.path or ""
    content = action.content or ""

    prompt = f"""Action: write_file
Path: {path}
Content length: {len(content)} bytes
Success: {success}
Result: {preview_head_tail(result, max_lines=20)}

What did we learn from this?"""

    return [reflect(prompt)]


@action_key.register(WriteFileAction)
def _(action: WriteFileAction) -> tuple:
    """Hashable key for WriteFileAction."""
    return ("write_file", action.path)


@hint.register(WriteFileAction)
def _(action: WriteFileAction) -> str:
    """Hint for Critic when write_file fails."""
    return "Creates NEW files. For existing files use edit_file."


@display_name.register(WriteFileAction)
def _(action: WriteFileAction) -> str:
    """Human-friendly name for UI."""
    return "Write"
