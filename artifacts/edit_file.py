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

    # Try with tab expansion (whitespace tolerance)
    normalized_content = content.expandtabs()
    normalized_target = target.expandtabs()
    normalized_replacement = replacement.expandtabs()

    if normalized_target in normalized_content:
        if operation == "replace":
            new_content = normalized_content.replace(normalized_target, normalized_replacement, 1)
            return True, new_content, "tab-expanded match"
        elif operation == "insert":
            idx = normalized_content.find(normalized_target) + len(normalized_target)
            return True, insert_with_newline(normalized_content, idx, normalized_replacement), "tab-expanded match"

    return False, content, "no match found"


# =============================================================================
# Validation handler
# =============================================================================

@validate.register(EditFileAction)
def validate_edit_file(action: EditFileAction, context: ExecutionContext) -> Tuple[bool, str]:
    """Validate edit_file action before execution."""
    # Check required fields
    if not action.path:
        return False, "path is required"
    if not action.instruction:
        return False, "instruction is required"

    # Check file exists
    abs_path = os.path.abspath(action.path)
    if not os.path.exists(abs_path):
        return False, f"File not found: {action.path}"

    # Read current content
    try:
        with open(abs_path, 'r') as f:
            content = f.read()
    except Exception as e:
        return False, f"Cannot read file: {e}"

    # Extract target and replacement from instruction
    # This is a simplified extraction - real implementation would parse the instruction
    target = action.instruction.split("replace")[0].strip() if "replace" in action.instruction.lower() else ""
    replacement = action.instruction.split("replace")[-1].strip() if "replace" in action.instruction.lower() else ""

    # Validate target uniqueness
    valid, error = _validate_unique_target(content, target)
    if not valid:
        return False, error

    # Check for Python syntax if file is Python
    if action.path.endswith('.py'):
        syntax_valid, syntax_error = _validate_python_syntax(content)
        if not syntax_valid:
            return False, f"Python syntax error: {syntax_error}"

    return True, "Validation passed"


# =============================================================================
# Execution handler
# =============================================================================

@execute.register(EditFileAction)
def execute_edit_file(action: EditFileAction, context: ExecutionContext) -> Tuple[bool, str]:
    """Execute edit_file action."""
    abs_path = os.path.abspath(action.path)

    # Read current content
    try:
        with open(abs_path, 'r') as f:
            content = f.read()
    except Exception as e:
        return False, f"Cannot read file: {e}"

    # Extract target and replacement from instruction
    # This is a simplified extraction - real implementation would parse the instruction
    target = action.instruction.split("replace")[0].strip() if "replace" in action.instruction.lower() else ""
    replacement = action.instruction.split("replace")[-1].strip() if "replace" in action.instruction.lower() else ""

    # Apply edit with fallback
    success, new_content, match_type = _apply_edit_with_fallback(content, target, replacement, "replace")

    if not success:
        return False, f"Edit failed: {match_type}"

    # Write the result
    try:
        with open(abs_path, 'w') as f:
            f.write(new_content)
        return True, f"File edited successfully ({match_type})"
    except Exception as e:
        return False, f"Write failed: {e}"


@extract_learnings.register(EditFileAction)
def _(
    action: EditFileAction,
    success: bool,
    result: str,
    reflect: Reflector,
) -> List[Learning]:
    """Extract learnings from edit_file action."""
    from dataclasses import asdict

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
