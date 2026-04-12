"""
Action dispatch - singledispatch for type-based routing.

No if/elif chains. Type-based dispatch.
"""

from functools import singledispatch
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from compass.agents.neo.types import ActionTarget, ExecutionContext
    from compass.agents.neo.memory import Learning


@singledispatch
def display(action) -> "ActionTarget":
    """Get display info. Returns ActionTarget.

    Fallback uses action_key() to extract target - avoids duplication.
    """
    from compass.agents.neo.types import ActionTarget
    key = action_key(action)
    # key is ("type_name", field1, field2, ...) - use first non-None field as target
    target = next((str(k) for k in key[1:] if k), "")
    return ActionTarget(target=target, display=str(action), content=None)


@singledispatch
def validate(action, project_path: str = ".", files_read: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
    """Validate action. Returns (is_valid, error_or_none)."""
    return True, None


@singledispatch
def execute(action, project_path: str, ctx: "ExecutionContext") -> Tuple[bool, str]:
    """Execute action. Returns (success, result)."""
    return False, f"No executor for {type(action).__name__}"


@singledispatch
def extract_learnings(action, success: bool, result: str, reflect) -> List["Learning"]:
    """Extract learnings from result."""
    return []


@singledispatch
def action_key(action) -> tuple:
    """Hashable key for action comparison."""
    return (type(action).__name__, getattr(action, "path", None))


@singledispatch
def content_field(action) -> str:
    """Default content field for this action type (when block has no explicit field=).

    Each type registers its own. Raises ValueError if no default is registered.
    """
    raise ValueError(
        f"{type(action).__name__} does not register a default content field. "
        f"Use explicit field= in the content block marker. "
        f"Or Put the value inline instead."
    )


@singledispatch
def hint(action) -> str:
    """Get hint for Critic when action fails. Colocated with action definition."""
    return ""


@singledispatch
def display_name(action) -> str:
    """Get human-friendly display name for action type."""
    # Default: derive from class name (ReadFileAction -> "Read")
    name = type(action).__name__.replace("Action", "")
    return name


@singledispatch
def format_result(action, result: str, max_len: int = 500) -> str:
    """Format action result for display. Each action controls its own presentation."""
    # Default: truncate long results
    if len(result) > max_len:
        return result[:max_len] + "..."
    return result


def get_registered_types() -> List[type]:
    """Get all registered action types."""
    return [t for t in display.registry.keys() if t is not object]


def ensure_registered():
    """Ensure all action handlers are registered.

    Call this before using dispatch functions if handlers might not be imported yet.
    Safe to call multiple times.
    """
    # Import actions package to trigger handler registration
    from compass.agents.neo import actions  # noqa: F401


# Auto-register handlers on module load
ensure_registered()
