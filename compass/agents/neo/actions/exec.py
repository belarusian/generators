"""
exec action - singledispatch handlers.

Registers handlers for ExecAction.
Execute Python code in memory. Variables persist across actions within a session.
"""

import io
import json
import os
import sys as _sys
from types import ModuleType
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from compass.core.content import preview_head_tail

from compass.agents.neo.types import ExecAction, ActionTarget, ExecutionContext, Reflector
from compass.agents.neo.dispatch import display, validate, execute, extract_learnings, action_key, hint, display_name, content_field
from compass.agents.neo.memory import Learning


class SafeExit(Exception):
    """Raised when exec code calls sys.exit()"""
    def __init__(self, code=0):
        self.code = code
        super().__init__(f"Script called sys.exit({code})")


def _safe_exit(code=0):
    """Safe replacement for sys.exit() that raises SafeExit instead."""
    raise SafeExit(code)


@content_field.register(ExecAction)
def _(action): return "code"


@display.register(ExecAction)
def _(action: ExecAction) -> ActionTarget:
    """Get display info for exec."""
    code = action.code or ""
    first_line = code.split("\n")[0][:60] if code else "(code)"
    return ActionTarget(
        target=first_line,
        display=f"exec: {first_line}",
        content=code,
    )


@validate.register(ExecAction)
def _(
    action: ExecAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate exec action.

    Returns (is_valid, error_message).

    Required fields:
    - code: Python code to execute

    Optional fields:
    - timeout: Seconds before timeout. Default 30.

    Variables defined in exec persist across actions within the session.
    Use exec for:
    - Quick calculations and type checks
    - API calls and data processing
    - Validation and verification

    For code worth keeping (logic you want to review, debug, or rerun),
    use write_file + run_command instead - code that exists can be improved.
    """
    code = action.code

    if not code:
        return False, (
            "code is None -- no content block was found for this action. "
            "Provide code inline: ExecAction(code=\"print(42)\") "
            "or add a content block: # === content: field=\"code\" ==="
        )

    return True, None


@execute.register(ExecAction)
def _(action: ExecAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """
    Execute Python code in memory.

    Returns (success, message).

    This allows the Actor to define custom actions that aren't built-in.
    The code runs in a context with:
    - os, sys, json, pathlib available
    - 'cwd' set to project path
    - 'result' variable to capture output

    Set 'result' to a string to return output.

    Args:
        action: ExecAction
        project_path: Working directory (project path)
        ctx: ExecutionContext with exec_globals for persisting variables across calls.
            If provided, variables set in one exec will be available in subsequent
            execs within the same step. The dict is updated in-place.
    """
    ctx = ctx or ExecutionContext()
    exec_globals = ctx.exec_globals

    code = action.code

    if code is None:
        return False, "No code provided for exec"

    # Create a safe sys module that doesn't allow exit()
    safe_sys = ModuleType("sys")
    for attr in dir(_sys):
        if not attr.startswith("_"):
            setattr(safe_sys, attr, getattr(_sys, attr))
    safe_sys.exit = _safe_exit

    # Use persistent globals if provided, otherwise create fresh
    if exec_globals is not None:
        globals_dict = exec_globals
        # Ensure builtins are always available (first exec sets them up)
        if "os" not in globals_dict:
            globals_dict.update({
                "os": os,
                "sys": safe_sys,
                "json": json,
                "Path": Path,
                "pathlib": Path,
                "cwd": project_path,
            })
    else:
        globals_dict = {
            "os": os,
            "sys": safe_sys,
            "json": json,
            "Path": Path,
            "pathlib": Path,
            "cwd": project_path,
        }

    # Change to project directory for execution
    original_cwd = os.getcwd()

    # Capture stdout/stderr
    old_stdout, old_stderr = _sys.stdout, _sys.stderr
    captured_out = io.StringIO()
    captured_err = io.StringIO()

    try:
        os.chdir(project_path)
        _sys.stdout = captured_out
        _sys.stderr = captured_err
        exec(code, globals_dict)

        # Build result with code and any output
        stdout_val = captured_out.getvalue().strip()
        stderr_val = captured_err.getvalue().strip()

        result_parts = [f"Code:\n{code}"]
        if stdout_val:
            result_parts.append(f"Output:\n{stdout_val}")
        if stderr_val:
            result_parts.append(f"Stderr:\n{stderr_val}")

        # Add computed variables so Planner can see values on replan
        builtins = {"os", "sys", "json", "Path", "pathlib", "cwd", "__builtins__"}
        user_vars = {k: v for k, v in globals_dict.items() if k not in builtins}
        if user_vars:
            computed = []
            for name, val in user_vars.items():
                val_str = repr(val)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                computed.append(f"  {name} = {val_str}")
            result_parts.append("Computed:\n" + "\n".join(computed))

        return True, "\n".join(result_parts)
    except SafeExit as e:
        # Script called our safe sys.exit() - exit(0) = success, non-zero = failure
        stdout_val = captured_out.getvalue().strip()
        stderr_val = captured_err.getvalue().strip()
        result_parts = [f"Code:\n{code}"]
        if stdout_val:
            result_parts.append(f"Output:\n{stdout_val}")
        if stderr_val:
            result_parts.append(f"Stderr:\n{stderr_val}")
        exit_code = e.code if e.code is not None else 0
        result_parts.append(f"(Script called sys.exit({exit_code}))")
        return exit_code == 0, "\n".join(result_parts)
    except SystemExit as e:
        # Script called real sys.exit() (from imported module) - treat as success
        stdout_val = captured_out.getvalue().strip()
        stderr_val = captured_err.getvalue().strip()
        result_parts = [f"Code:\n{code}"]
        if stdout_val:
            result_parts.append(f"Output:\n{stdout_val}")
        if stderr_val:
            result_parts.append(f"Stderr:\n{stderr_val}")
        exit_code = e.code if e.code is not None else 0
        result_parts.append(f"(Script called sys.exit({exit_code}))")
        # Treat exit(0) as success, non-zero as failure
        return exit_code == 0, "\n".join(result_parts)
    except Exception as e:
        # Include code in error so model can see what failed
        # IMPORTANT: Make clear that failed execs don't set any variables
        error_msg = f"Execution failed (no variables were set): {type(e).__name__}: {e}\nCode:\n{code}"
        # Show available user variables (helps debug)
        builtins = {"os", "sys", "json", "Path", "pathlib", "cwd", "__builtins__"}
        user_vars = [k for k in globals_dict.keys() if k not in builtins]
        if user_vars:
            error_msg += f"\nAvailable vars: {user_vars}"
        else:
            error_msg += "\nAvailable vars: (none yet)"
        return False, error_msg
    finally:
        _sys.stdout = old_stdout
        _sys.stderr = old_stderr
        os.chdir(original_cwd)


@extract_learnings.register(ExecAction)
def _(
    action: ExecAction,
    success: bool,
    result: str,
    reflect: Reflector,
) -> List[Learning]:
    """Extract learnings. LLM reflects and chooses learning type - no heuristics."""
    from dataclasses import asdict

    action_data = asdict(action) if hasattr(action, '__dataclass_fields__') else action

    prompt = f"""Action: exec
Input: {json.dumps(action_data)}
Success: {success}
Result:
{preview_head_tail(result, max_lines=23)}

What did we learn from this?"""

    return [reflect(prompt)]


@action_key.register(ExecAction)
def _(action: ExecAction) -> tuple:
    """Hashable key for exec comparison."""
    # Use first line of code as identifier
    code = action.code or ""
    first_line = code.split("\n")[0][:60] if code else ""
    return ("exec", first_line)


@hint.register(ExecAction)
def _(action: ExecAction) -> str:
    """Hint for Critic when exec fails."""
    return "Python code IN MEMORY. Fix = correct code/imports."


@display_name.register(ExecAction)
def _(action: ExecAction) -> str:
    """Human-friendly name for UI."""
    return "Exec"
