"""
DeleteFileAction handlers - singledispatch registration.

Type-based dispatch for delete_file action: display, validate, execute, extract_learnings.
"""

import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from compass.core.content import preview_head_tail

from compass.agents.neo.types import DeleteFileAction
from compass.agents.neo.dispatch import display, validate, execute, extract_learnings, action_key, hint, display_name

if TYPE_CHECKING:
    from compass.agents.neo.types import ActionTarget, ExecutionContext
    from compass.agents.neo.memory import Learning


# --- Display ---

@display.register(DeleteFileAction)
def _(action: DeleteFileAction) -> "ActionTarget":
    """Get display info for DeleteFileAction."""
    from compass.agents.neo.types import ActionTarget

    path = action.path
    display_str = f"delete {path}"

    return ActionTarget(target=path, display=display_str, content=None)


# --- Validation ---

@validate.register(DeleteFileAction)
def _(
    action: DeleteFileAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate delete_file action.

    Returns (is_valid, error_message).

    Checks:
    1. Required fields present
    2. Path within project bounds

    Required fields:
    - path: File path to delete (relative to project root)

    Deletes a single file. Does not delete directories.
    Make sure the path is correct before deleting - this is not reversible.
    """
    # 1. Required fields
    path = action.path

    if not path:
        return False, "Missing required field: path"

    return True, None


# --- Execution ---

@execute.register(DeleteFileAction)
def _(action: DeleteFileAction, project_path: str, ctx: "ExecutionContext" = None) -> Tuple[bool, str]:
    """
    Execute delete_file action.

    Returns (success, message).

    Args:
        action: DeleteFileAction
        project_path: Project root directory
        ctx: ExecutionContext (unused, for uniform signature)
    """
    path = action.path or ""

    # Resolve relative path
    if path and not os.path.isabs(path):
        path = os.path.join(project_path, path)

    try:
        os.remove(path)
        return True, f"Deleted: {path}"
    except FileNotFoundError:
        return False, f"File not found: {path}"
    except Exception as e:
        return False, f"Failed to delete {path}: {e}"


# --- Learning Extraction ---

@extract_learnings.register(DeleteFileAction)
def _(
    action: DeleteFileAction,
    success: bool,
    result: str,
    reflect,
) -> List["Learning"]:
    """Extract learnings from delete_file action."""
    path = action.path or ""

    prompt = f"""Action: delete_file
Path: {path}
Success: {success}
Result: {preview_head_tail(result, max_lines=20)}

What did we learn from this?"""

    return [reflect(prompt)]


# --- Action Key ---

@action_key.register(DeleteFileAction)
def _(action: DeleteFileAction) -> tuple:
    """Hashable key for DeleteFileAction comparison."""
    return ("delete_file", action.path)


@hint.register(DeleteFileAction)
def _(action: DeleteFileAction) -> str:
    """Hint for Critic when delete_file fails."""
    return "Delete file. Check file exists and was read first."


@display_name.register(DeleteFileAction)
def _(action: DeleteFileAction) -> str:
    """Human-friendly name for UI."""
    return "Delete"
