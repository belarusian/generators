"""
create_dir action - singledispatch handlers.

Registers handlers for CreateDirAction.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from compass.core.content import preview_head_tail

from compass.agents.neo.types import CreateDirAction, ActionTarget, ExecutionContext, Reflector
from compass.agents.neo.dispatch import display, validate, execute, extract_learnings, action_key, hint, display_name
from compass.agents.neo.memory import Learning


@display.register(CreateDirAction)
def _(action: CreateDirAction) -> ActionTarget:
    """Get display info for create_dir."""
    return ActionTarget(
        target=action.path,
        display=f"mkdir -p {action.path}",
        content=None,
    )


@validate.register(CreateDirAction)
def _(
    action: CreateDirAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate create_dir action.

    Returns (is_valid, error_message).

    Checks:
    1. Required fields present
    2. Path within project bounds

    Required fields:
    - path: Directory path to create (relative to project root)

    Creates directory and any missing parent directories (like mkdir -p).
    """
    # 1. Required fields
    path = action.path

    if not path:
        return False, "Missing required field: path"

    return True, None


@execute.register(CreateDirAction)
def _(action: CreateDirAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """
    Execute create_dir action.

    Returns (success, message).
    """
    path = action.path or ""

    # Resolve relative path
    if path and not os.path.isabs(path):
        path = os.path.join(project_path, path)

    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True, f"Created directory: {path}"
    except Exception as e:
        return False, f"Failed to create directory {path}: {e}"


@extract_learnings.register(CreateDirAction)
def _(
    action: CreateDirAction,
    success: bool,
    result: str,
    reflect: Reflector,
) -> List[Learning]:
    """Extract learnings from create_dir action."""
    from dataclasses import asdict

    action_data = asdict(action) if hasattr(action, '__dataclass_fields__') else action

    prompt = f"""Action: create_dir
Input: {json.dumps(action_data)}
Success: {success}
Result:
{preview_head_tail(result, max_lines=23)}

What did we learn from this?"""

    return [reflect(prompt)]


@action_key.register(CreateDirAction)
def _(action: CreateDirAction) -> tuple:
    """Hashable key for create_dir comparison."""
    return ("create_dir", action.path)


@hint.register(CreateDirAction)
def _(action: CreateDirAction) -> str:
    """Hint for Critic when create_dir fails."""
    return "Create directory. Check path and permissions."


@display_name.register(CreateDirAction)
def _(action: CreateDirAction) -> str:
    """Human-friendly name for UI."""
    return "Mkdir"
