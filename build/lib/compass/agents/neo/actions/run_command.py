"""
run_command action - singledispatch handlers for RunCommandAction.

Registers handlers for:
- display: Get display info for UI
- validate: Validate action fields
- execute: Run shell command
- extract_learnings: Extract learnings from result
- action_key: Hashable key for comparison
"""

import json
import subprocess
from typing import Dict, List, Optional, Tuple

from compass.core.content import preview_head_tail

from compass.agents.neo.dispatch import (
    action_key,
    content_field,
    display,
    display_name,
    execute,
    extract_learnings,
    hint,
    validate,
)
from compass.agents.neo.memory import Learning
from compass.agents.neo.types import ActionTarget, ExecutionContext, RunCommandAction


@content_field.register(RunCommandAction)
def _(action): return "command"


@display.register(RunCommandAction)
def _(action: RunCommandAction) -> ActionTarget:
    """Display info for run_command action."""
    cmd_preview = (action.command or "")[:60]
    return ActionTarget(
        target=cmd_preview,
        display=f"run: {cmd_preview}",
        content=None,
    )


@validate.register(RunCommandAction)
def _(
    action: RunCommandAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate run_command action.

    Required fields:
    - command: Shell command to execute

    Optional fields:
    - timeout: Seconds before timeout. Default 120 (2 minutes).

    Use run_command for simple commands: ls, git status, npm install, pytest.

    For complex commands with $variables, quotes, or special characters,
    use shell_command instead - ShellBuilder handles escaping correctly.

    Security: Commands run in project directory with user permissions.
    Avoid commands that could damage the system.
    """
    if not action.command:
        return False, "Missing required field: command"
    return True, None


@execute.register(RunCommandAction)
def _(action: RunCommandAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """
    Execute run_command action.

    Runs command in project directory with specified timeout.
    Captures both stdout and stderr.
    """
    command = action.command or ""
    timeout = action.timeout or 120

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr

        if result.returncode == 0:
            return True, output or "(no output)"
        else:
            return False, f"Command failed (exit {result.returncode}):\nCommand: {command}\nOutput: {output}"

    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout} seconds:\nCommand: {command}"
    except Exception as e:
        return False, f"Failed to run command: {e}\nCommand: {command}"


@extract_learnings.register(RunCommandAction)
def _(
    action: RunCommandAction,
    success: bool,
    result: str,
    reflect,
) -> List[Learning]:
    """Extract learnings from run_command action."""
    from dataclasses import asdict

    action_data = asdict(action) if hasattr(action, '__dataclass_fields__') else action

    prompt = f"""Action: run_command
Input: {json.dumps(action_data)}
Success: {success}
Result:
{preview_head_tail(result, max_lines=23)}

What did we learn from this?"""

    return [reflect(prompt)]


@action_key.register(RunCommandAction)
def _(action: RunCommandAction) -> tuple:
    """Hashable key for run_command - based on command."""
    return ("run_command", action.command)


@hint.register(RunCommandAction)
def _(action: RunCommandAction) -> str:
    """Hint for Critic when run_command fails."""
    return "Shell command. Check path, permissions, syntax."


@display_name.register(RunCommandAction)
def _(action: RunCommandAction) -> str:
    """Human-friendly name for UI."""
    return "Run"
