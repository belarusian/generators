"""
edit_file action - singledispatch handlers for EditFileAction.

Modify existing files via FileEditor. Describe WHAT to change - FileEditor handles HOW.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from compass.core.content import preview_head_tail

from compass.agents.neo.types import EditFileAction, ExecutionContext, Reflector
from compass.agents.neo.dispatch import display, validate, execute, extract_learnings, action_key, hint, display_name, content_field
from compass.agents.neo.memory import Learning

# Import safeguards from write_file (DRY - same validation logic)
from compass.agents.neo.actions.write_file import (
    _detect_duplication,
    _preview_content,
    _validate_python_syntax,
)


@content_field.register(EditFileAction)
def _(action) -> str:
    return "instruction"


# =============================================================================
# Pure validation functions - extracted from executor.py heuristics
# =============================================================================

def _find_match_lines(content: str, target: str) -> List[int]:
    """
    Find all line numbers (1-based) where target appears.

    Pure function: content, target -> line numbers
    """
    return [
        i + 1
        for i, line in enumerate(content.split("\n"))
        if target in line
    ]


def _validate_unique_target(content: str, target: str) -> Tuple[bool, Optional[str]]:
    """
    Validate target appears exactly once in content.

    Pure function: content, target -> (valid, error_or_none)

    Returns:
        (True, None) if exactly one match
        (False, error_message) if 0 or >1 matches, with line numbers for >1
    """
    # Try exact match first
    count = content.count(target)

    if count == 1:
        return True, None

    if count == 0:
        # Try with tab expansion (whitespace tolerance)
        normalized_content = content.expandtabs()
        normalized_target = target.expandtabs()
        if normalized_content.count(normalized_target) == 1:
            return True, None  # Will need normalized replacement
        return False, (
            f"Target not found in file. "
            "Check whitespace/indentation matches exactly."
        )

    # count > 1: find line numbers for guidance
    lines = _find_match_lines(content, target)

    # Check if all occurrences are identical lines (bulk replace scenario)
    file_lines = content.split("\n")
    matching_lines = [file_lines[ln - 1] for ln in lines if ln <= len(file_lines)]
    all_identical = len(set(l.strip() for l in matching_lines)) == 1

    if all_identical:
        return False, (
            f"STOP: Target found {count} times (lines {lines}) and ALL are identical -- "
            "EditFileAction CANNOT fix this. "
            "Switch to: RunCommandAction(command=\"sed -i 's/old/new/g' file\") "
            "or ExecAction with content.replace()."
        )

    return False, (
        f"Target found {count} times (lines {lines}). "
        "Provide more context to make target unique."
    )


def _apply_edit_with_fallback(
    content: str,
    target: str,
    replacement: str,
    operation: str,
) -> Tuple[bool, str, str]:
    """
    Apply edit with whitespace-tolerant fallback.

    Pure function: content, target, replacement, operation -> (success, new_content, message)

    Tries exact match first, then tab-expanded match.
    """
    # INSERT needs newline before content (unless content already starts with one)
    def insert_with_newline(text: str, idx: int, new_content: str) -> str:
        separator = "" if new_content.startswith("\n") else "\n"
        return text[:idx] + separator + new_content + text[idx:]

    # Try exact match first
    if target in content:
        if operation == "replace":
            return True, content.replace(target, replacement, 1), "exact match"
        elif operation == "insert":
            idx = content.find(target) + len(target)
            return True, insert_with_newline(content, idx, replacement), "exact match"
        elif operation == "delete":
            return True, content.replace(target, "", 1), "exact match"

    # Try whitespace-tolerant (tab expansion)
    normalized_content = content.expandtabs()
    normalized_target = target.expandtabs()

    if normalized_target in normalized_content:
        normalized_replacement = replacement.expandtabs()
        if operation == "replace":
            new_content = normalized_content.replace(normalized_target, normalized_replacement, 1)
            return True, new_content, "whitespace-normalized match"
        elif operation == "insert":
            idx = normalized_content.find(normalized_target) + len(normalized_target)
            return True, insert_with_newline(normalized_content, idx, normalized_replacement), "whitespace-normalized match"
        elif operation == "delete":
            new_content = normalized_content.replace(normalized_target, "", 1)
            return True, new_content, "whitespace-normalized match"

    return False, content, "target not found"


# --- Singledispatch handlers ---

@display.register(EditFileAction)
def _(action: EditFileAction):
    """Display info for edit_file action."""
    from compass.agents.neo.types import ActionTarget
    path = action.path or ""
    instruction = action.instruction or ""
    preview = instruction[:80] + "..." if len(instruction) > 80 else instruction
    return ActionTarget(
        target=path,
        display=f"Edit {path}: {preview}",
        content=instruction
    )


@validate.register(EditFileAction)
def _(
    action: EditFileAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate edit_file action.

    Required fields:
    - path: File to edit (must exist)
    - instruction: Natural language description of what to change

    FileEditor reads the file and handles the mechanics - finding the exact
    code to modify and making the change. You describe WHAT, it handles HOW.

    Example:
        {"action_type": "edit_file",
         "path": "calc.py",
         "instruction": "Add multiply(a, b) function after the subtract function"}

    Tips:
    - If FILES READ already shows the file, proceed to edit (don't re-read)
    - If edit fails due to stale content, THEN re-read and retry
    - Be specific in instructions: "add after X", "replace Y with Z", "delete the Z block"
    """
    path = action.path
    instruction = action.instruction

    if not path:
        return False, "Missing required field: path"
    if not instruction:
        return False, "Missing required field: instruction"

    return True, None


@execute.register(EditFileAction)
def _(action: EditFileAction, project_path: str, ctx: ExecutionContext) -> Tuple[bool, str]:
    """
    Execute edit_file action via FileEditor.

    Actor provides: path + instruction (WHAT to change)
    FileEditor handles: validation + retries (finds target/content)
    This function: applies the edit (deterministic string replace)

    Args:
        action: EditFileAction
        project_path: Base project path
        ctx: ExecutionContext with oracle for LLM-assisted editing

    Returns:
        (success, result_message)
    """
    from compass.llm.oracle import Oracle
    from compass.agents.neo.file_editor import call_file_editor

    oracle = ctx.oracle if ctx else None

    path = action.path or ""
    instruction = action.instruction or ""

    if not path:
        return False, "edit_file requires 'path'"
    if not instruction:
        return False, "edit_file requires 'instruction'"

    rel_path = path
    abs_path = os.path.join(project_path, path)

    # Read current file content
    try:
        with open(abs_path, 'r') as f:
            file_content = f.read()
    except Exception as e:
        return False, f"Cannot read file: {e}"

    # Use provided oracle or create fresh one
    if oracle is None:
        oracle = Oracle()

    # Retry loop: FileEditor gets feedback on syntax errors
    max_edit_retries = 3
    last_error = None

    for attempt in range(max_edit_retries):
        # Build instruction with feedback from previous attempt
        effective_instruction = (
            f"{instruction}\n\nPREVIOUS ATTEMPT FAILED:\n{last_error}"
            if last_error else instruction
        )

        result = call_file_editor(
            oracle=oracle,
            file_path=rel_path,
            file_content=file_content,
            instruction=effective_instruction,
        )

        if not result.success:
            return False, f"FileEditor failed: {result.error}"

        # Validate target is unique BEFORE attempting edit
        valid, unique_error = _validate_unique_target(file_content, result.target)
        if not valid:
            last_error = f"Target not unique: {unique_error}"
            continue

        # Apply the edit with whitespace-tolerant fallback
        success, new_content, match_type = _apply_edit_with_fallback(
            file_content, result.target, result.content, result.operation
        )
        if not success:
            last_error = f"Failed to apply {result.operation}: {match_type}"
            continue

        # Validate Python syntax BEFORE writing (for .py files)
        if abs_path.endswith('.py'):
            syntax_error = _validate_python_syntax(new_content, path)
            if syntax_error:
                preview = _preview_content(new_content, max_lines=12)
                last_error = (
                    f"Would create invalid Python!\n"
                    f"{syntax_error}\n"
                    f"Preview:\n```\n{preview}\n```"
                )
                continue

        # Success - break out of retry loop
        break
    else:
        # All retries exhausted
        return False, (
            f"EDIT BLOCKED after {max_edit_retries} attempts!\n"
            f"Last error: {last_error}"
        )

    # Heuristic check: detect duplication BEFORE writing
    dup_warning = _detect_duplication(new_content)
    if dup_warning:
        # Don't write corrupted content - return error so Critic can revert
        preview = _preview_content(new_content, max_lines=12)
        return False, (
            f"EDIT BLOCKED - duplication detected!\n"
            f"{dup_warning}\n"
            f"Preview of would-be result:\n```\n{preview}\n```\n"
            f"Retry with a different approach to avoid duplication."
        )

    # Write the result
    try:
        with open(abs_path, 'w') as f:
            f.write(new_content)

        # Build result with preview for Critic validation
        result_parts = [f"[{result.operation}] {path}: {result.reasoning}"]
        preview = _preview_content(new_content, max_lines=12)
        result_parts.append(f"Result preview:\n```\n{preview}\n```")

        return True, "\n".join(result_parts)
    except Exception as e:
        return False, f"Write failed: {e}"


@extract_learnings.register(EditFileAction)
def _(
    action: EditFileAction,
    success: bool,
    result: str,
    reflect: Reflector,
) -> List[Learning]:
    """Extract learnings from edit_file action. LLM reflects and chooses learning type."""
    from dataclasses import asdict

    # Convert to dict for serialization (handles both dataclass and dict)
    action_data = asdict(action) if hasattr(action, '__dataclass_fields__') else action

    prompt = f"""Action: edit_file
Input: {json.dumps(action_data)}
Success: {success}
Result:
{preview_head_tail(result, max_lines=23)}

What did we learn from this?"""

    return [reflect(prompt)]


@action_key.register(EditFileAction)
def _(action: EditFileAction) -> tuple:
    """Hashable key for EditFileAction."""
    instruction = (action.instruction or "")[:60]
    return ("edit_file", action.path, instruction)


@hint.register(EditFileAction)
def _(action: EditFileAction) -> str:
    """Hint for Critic when edit_file fails."""
    return "Edit via FileEditor. FileEditor reads the file. Describe change clearly."


@display_name.register(EditFileAction)
def _(action: EditFileAction) -> str:
    """Human-friendly name for UI."""
    return "Edit"
