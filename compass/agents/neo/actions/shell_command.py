"""
shell_command action - singledispatch handlers for ShellCommandAction.

Registers handlers for:
- display: Get display info for UI
- validate: Validate action fields
- execute: Run shell command via ShellBuilder
- extract_learnings: Extract learnings from result
- action_key: Hashable key for comparison
"""

import json
import subprocess
from typing import Dict, List, Optional, Tuple

from compass.core.content import preview_head_tail, preview_tail

from compass.agents.neo.dispatch import (
    action_key,
    display,
    display_name,
    execute,
    extract_learnings,
    hint,
    validate,
)
from compass.agents.neo.memory import Learning
from compass.agents.neo.types import ActionTarget, ExecutionContext, ShellCommandAction


@display.register(ShellCommandAction)
def _(action: ShellCommandAction) -> ActionTarget:
    """Display info for shell_command action."""
    intent_preview = (action.intent or "")[:60]
    return ActionTarget(
        target=intent_preview,
        display=f"shell: {intent_preview}",
        content=None,
    )


@validate.register(ShellCommandAction)
def _(
    action: ShellCommandAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate shell_command action.

    Required fields:
    - intent: Natural language description of what the command should do

    Optional fields:
    - context: Additional context (files, special characters involved)
    - timeout: Seconds before timeout. Default 120 (2 minutes).

    ShellBuilder generates the properly escaped command from your description.
    Use this when your command involves:
    - $variables or ${substitutions}
    - Quotes within quotes
    - Special characters like |, &, >, <
    - Complex pipelines

    Example:
        {"action_type": "shell_command",
         "intent": "Write 'Price: $100' to output.txt",
         "context": "Dollar sign must be literal, not variable"}

    For simple commands (ls, git status), use run_command directly.
    """
    if not action.intent:
        return False, "Missing required field: intent"
    return True, None


@execute.register(ShellCommandAction)
def _(action: ShellCommandAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """
    Execute shell_command action via ShellBuilder.

    Actor provides: intent + context (WHAT to do)
    ShellBuilder handles: proper escaping (HOW to write command)
    This function: executes the validated command

    Args:
        action: ShellCommandAction
        project_path: Base project path for execution
        ctx: ExecutionContext with oracle for LLM-assisted command building

    Returns:
        (success, result_message)
    """
    from compass.agents.neo.shell_builder import call_shell_builder
    from compass.llm.oracle import Oracle

    ctx = ctx or ExecutionContext()
    oracle = ctx.oracle

    intent = action.intent or ""
    context = action.context or ""
    timeout = action.timeout or 120

    if not intent:
        return False, "shell_command requires 'intent'"

    # Build context for ShellBuilder
    builder_context = f"Working directory: {project_path}"
    if context:
        builder_context += f"\n{context}"

    # Use provided oracle or create fresh one
    if oracle is None:
        oracle = Oracle()
    result = call_shell_builder(
        oracle=oracle,
        intent=intent,
        context=builder_context,
    )

    if not result.success:
        return False, f"ShellBuilder failed: {result.error}"

    # Show the generated command
    print(f"      $ {result.command}")
    if result.warnings:
        for warn in result.warnings:
            print(f"      [!] {warn}")

    # Execute the validated command
    try:
        proc = subprocess.run(
            result.command,
            shell=True,
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = proc.stdout + proc.stderr
        if proc.returncode == 0:
            # Return full output - _truncate_result handles truncation with registry
            return True, output if output else "(no output)"
        else:
            # Error: return full output with exit code
            return False, f"Exit {proc.returncode}:\n{output}"

    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s"
    except Exception as e:
        return False, f"Execution failed: {e}"


@extract_learnings.register(ShellCommandAction)
def _(
    action: ShellCommandAction,
    success: bool,
    result: str,
    reflect,
) -> List[Learning]:
    """Extract learnings from shell_command action. LLM reflects and chooses learning type."""
    from dataclasses import asdict

    action_data = asdict(action) if hasattr(action, '__dataclass_fields__') else action

    prompt = f"""Action: shell_command
Input: {json.dumps(action_data)}
Success: {success}
Result:
{preview_head_tail(result, max_lines=23)}

What did we learn from this?"""

    return [reflect(prompt)]


@action_key.register(ShellCommandAction)
def _(action: ShellCommandAction) -> tuple:
    """Hashable key for shell_command - based on intent."""
    return ("shell_command", action.intent)


@hint.register(ShellCommandAction)
def _(action: ShellCommandAction) -> str:
    """Hint for Critic when shell_command fails."""
    return "Complex shell command. ShellBuilder handles escaping. Check intent clarity."


@display_name.register(ShellCommandAction)
def _(action: ShellCommandAction) -> str:
    """Human-friendly name for UI."""
    return "Shell"
