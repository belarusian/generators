"""
Action execution via singledispatch.

Type-based routing. The type IS the contract.
"""

from typing import Any, Dict, Optional

from compass.agents.neo.dispatch import (
    validate as dispatch_validate,
    execute as dispatch_execute,
    extract_learnings as dispatch_extract_learnings,
)
from compass.core.telemetry import record_action


def execute_action(
    action,  # Typed action dataclass
    project_path: str,
    files_read: Optional[Dict] = None,
    exec_globals: Optional[Dict] = None,
    oracle: Optional[Any] = None,
    memory: Optional[Any] = None,
    ctx: Optional[Any] = None,
) -> tuple:
    """
    Execute a typed action via singledispatch.

    Args:
        action: Typed action dataclass (ReadFileAction, WriteFileAction, etc.)
        project_path: Base project path
        files_read: Files already read (for validation)
        exec_globals: Python globals for exec actions
        oracle: LLM interface (for LLM-assisted actions)
        memory: Session memory
        ctx: Full ExecutionContext (if provided, other params merged in)

    Returns:
        (success, result) tuple
    """
    from compass.agents.neo.types import ExecutionContext

    # Build/update context
    if ctx is None:
        ctx = ExecutionContext(
            exec_globals=exec_globals,
            oracle=oracle,
            memory=memory,
            files_read=files_read,
        )
    else:
        # Merge explicitly passed params into existing ctx
        if exec_globals is not None and ctx.exec_globals is None:
            ctx.exec_globals = exec_globals
        if oracle is not None and ctx.oracle is None:
            ctx.oracle = oracle
        if memory is not None and ctx.memory is None:
            ctx.memory = memory
        if files_read is not None and ctx.files_read is None:
            ctx.files_read = files_read

    # Validate via singledispatch
    is_valid, error = dispatch_validate(action, project_path, files_read or {})
    if not is_valid:
        return False, error

    # Execute via singledispatch with timing
    import time
    start_time = time.time()
    result = dispatch_execute(action, project_path, ctx)
    duration = time.time() - start_time

    # Record to telemetry with duration
    record_action(action, duration)

    return result


def extract_learnings(action, success: bool, result: str, reflector) -> list:
    """Extract learnings via singledispatch."""
    return dispatch_extract_learnings(action, success, result, reflector)


