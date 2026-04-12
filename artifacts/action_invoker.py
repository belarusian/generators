"""
Action invoker - dynamically invoke actions from discovered modules.

This module provides utilities to:
- Discover and instantiate action classes from registered action modules
- Invoke actions with validation and execution support
- Integrate with the existing singledispatch-based action handling system

Trinity (generators): use artifact_type ``action_invoker`` with inputs::

    {"action": {"type": "ScreenshotAction", "region": "full"}}

Plan guide: inputs must include ``action`` (dict with Neo action ``type`` and
constructor fields). Optional ``validate_first`` (bool, default True) runs
validate before execute.
"""

import importlib
import inspect
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from compass.agents.neo.dispatch import (
    execute,
    validate,
    ensure_registered,
    get_registered_types,
)
from compass.agents.neo.types import ActionBase

CYCLE_BREAKING = True  # screen/UI state changes between steps


def run(step, resolved_inputs, workspace):
    """Trinity artifact contract: run(step, resolved_inputs, workspace) -> Result."""
    from compass.generators._types import Err, Ok
    from compass.generators.trinity._types import Fact

    raw = resolved_inputs.get("action")
    if not isinstance(raw, dict) or not raw.get("type"):
        return Err(
            f"step '{step.step_id}': action_invoker requires inputs['action'] "
            f"as a dict with 'type' and action fields (e.g. ScreenshotAction)"
        )

    validate_first = resolved_inputs.get("validate_first", True)
    if not isinstance(validate_first, bool):
        validate_first = bool(validate_first)

    project_path = str(workspace) if workspace is not None else "."
    action_payload = dict(raw)

    try:
        if validate_first:
            is_valid, verr, text = invoke_action_with_validation(
                action_payload, project_path=project_path, ctx=None
            )
            if not is_valid:
                return Err(
                    f"step '{step.step_id}': action validation failed: {verr}"
                )
            success, result = True, text
        else:
            success, result = invoke_action(
                action_payload, project_path=project_path, ctx=None
            )
    except Exception as e:
        return Err(f"step '{step.step_id}': action_invoker failed: {e}")

    if not success:
        return Err(f"step '{step.step_id}': action execution failed: {result}")

    value = result if isinstance(result, str) else repr(result)
    return Ok(
        Fact(
            step_id=step.step_id,
            name=step.expected_fact or "action_result",
            value=value,
            fact_type="text",
        )
    )


def _get_all_action_classes() -> List[Type[ActionBase]]:
    """Get all registered action classes from action modules."""
    ensure_registered()
    return get_registered_types()


def _create_action_instance(
    action_type_name: str,
    **kwargs,
) -> Optional[ActionBase]:
    """Create an action instance by type name and keyword arguments.

    Args:
        action_type_name: Name of the action class (e.g., 'ReadFileAction')
        **kwargs: Constructor arguments for the action

    Returns:
        Action instance or None if not found
    """
    for action_type in _get_all_action_classes():
        if action_type.__name__ == action_type_name:
            try:
                return action_type(**kwargs)
            except TypeError as e:
                raise ValueError(
                    f"Invalid arguments for {action_type_name}: {e}"
                )
    return None


def invoke_action(
    action: Union[ActionBase, Dict[str, Any]],
    project_path: str = ".",
    ctx: Optional[Any] = None,
) -> Tuple[bool, str]:
    """Invoke an action, executing it with the registered handler.

    Args:
        action: Either an action instance or a dict with 'type' and fields
        project_path: Project root path for file operations
        ctx: Execution context (optional)

    Returns:
        (success, result) tuple from the action execution
    """
    # Convert dict to action instance if needed
    if isinstance(action, dict):
        action_type = action.pop("type", None)
        if not action_type:
            raise ValueError("Action dict must include 'type' field")
        action_instance = _create_action_instance(action_type, **action)
        if not action_instance:
            raise ValueError(f"Unknown action type: {action_type}")
    else:
        action_instance = action

    # Ensure action is a registered type
    if type(action_instance) not in _get_all_action_classes():
        raise ValueError(
            f"Action type {type(action_instance).__name__} not registered"
        )

    # Execute with registered handler
    success, result = execute(action_instance, project_path, ctx)
    return success, result


def invoke_action_with_validation(
    action: Union[ActionBase, Dict[str, Any]],
    project_path: str = ".",
    ctx: Optional[Any] = None,
) -> Tuple[bool, Optional[str], str]:
    """Invoke an action with validation before execution.

    Args:
        action: Either an action instance or a dict with 'type' and fields
        project_path: Project root path for file operations
        ctx: Execution context (optional)

    Returns:
        (is_valid, error_message, execution_result) tuple
    """
    # Convert dict to action instance if needed
    if isinstance(action, dict):
        action_type = action.pop("type", None)
        if not action_type:
            raise ValueError("Action dict must include 'type' field")
        action_instance = _create_action_instance(action_type, **action)
        if not action_instance:
            raise ValueError(f"Unknown action type: {action_type}")
    else:
        action_instance = action

    # Validate action
    is_valid, error = validate(action_instance, project_path)
    if not is_valid:
        return False, error, ""

    # Execute with registered handler
    success, result = execute(action_instance, project_path, ctx)
    return True, None, result


def get_action_classes_by_module() -> Dict[str, List[Type[ActionBase]]]:
    """Get mapping of module names to their action classes.

    Returns:
        Dict mapping module names to lists of action classes
    """
    module_actions: Dict[str, List[Type[ActionBase]]] = {}
    for action_type in _get_all_action_classes():
        module_name = action_type.__module__
        if module_name not in module_actions:
            module_actions[module_name] = []
        module_actions[module_name].append(action_type)
    return module_actions


# Re-export key functions for convenience
__all__ = [
    "invoke_action",
    "invoke_action_with_validation",
    "get_all_action_classes",
    "create_action_instance",
    "get_action_classes_by_module",
]

# Re-export key functions from invoker_code
def get_all_action_classes() -> List[Type[ActionBase]]:
    """Alias for _get_all_action_classes."""
    return _get_all_action_classes()


def create_action_instance(
    action_type_name: str,
    **kwargs,
) -> Optional[ActionBase]:
    """Alias for _create_action_instance."""
    return _create_action_instance(action_type_name, **kwargs)
