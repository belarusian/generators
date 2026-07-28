"""
index action - singledispatch handlers.

Registers handlers for IndexAction.

Allows Actor to rebuild/refresh the RAG semantic search index.
"""

import json
from typing import Dict, List, Optional, Tuple

from compass.agents.neo.types import IndexAction, ActionTarget, ExecutionContext, Reflector
from compass.agents.neo.dispatch import display, validate, execute, extract_learnings, action_key, hint, display_name
from compass.agents.neo.memory import Learning


# =============================================================================
# Singledispatch handlers
# =============================================================================

@display.register(IndexAction)
def _(action: IndexAction) -> ActionTarget:
    """Get display info for index action."""
    mode = "rebuild" if action.force else "update"
    return ActionTarget(
        target=mode,
        display=f"index: {mode}",
        content=None,
    )


@validate.register(IndexAction)
def _(
    action: IndexAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate index action.

    Returns (is_valid, error_message).

    Optional fields:
    - force: If True, full rebuild. If False (default), incremental update.

    Use index action when:
    - Search results seem stale or incomplete
    - Many files have changed since last search
    - You want to ensure the semantic index is fresh

    The index is usually updated automatically, but this forces a refresh.
    """
    return True, None


@execute.register(IndexAction)
def _(action: IndexAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """
    Execute index action - rebuild/refresh the RAG index.

    Returns (success, message).
    """
    from compass.agents.neo.rag import get_embedder

    force = action.force or False

    try:
        embedder = get_embedder(project_path)
        count = embedder.build_index(force=force)

        mode = "Rebuilt" if force else "Updated"
        return True, f"{mode} index: {count} chunks indexed"

    except Exception as e:
        return False, f"Index failed: {e}"


@extract_learnings.register(IndexAction)
def _(
    action: IndexAction,
    success: bool,
    result: str,
    reflect: Reflector,
) -> List[Learning]:
    """Extract learnings from index action."""
    # Indexing doesn't typically produce learnings
    return []


@action_key.register(IndexAction)
def _(action: IndexAction) -> tuple:
    """Hashable key for index comparison."""
    return ("index", action.force)


@hint.register(IndexAction)
def _(action: IndexAction) -> str:
    """Hint for Critic when index fails."""
    return "Rebuild search index. Check project path."


@display_name.register(IndexAction)
def _(action: IndexAction) -> str:
    """Human-friendly name for UI."""
    return "Index"
