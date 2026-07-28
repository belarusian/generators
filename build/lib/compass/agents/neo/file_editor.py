"""
FileEditor - Specialized agent for file modifications.

Separation of concerns:
- Actor (big context): WHAT to change - understands problem, codebase, goals
- FileEditor (small context): HOW to change - just file + instruction

FileEditor uses oracle.ask(FileEditorResponse) with content blocks:
1. LLM outputs FileEditorResponse with operation, target, content
2. Content blocks handle multi-line target/content (zero escaping)
3. We validate (target exists, unique)
4. If invalid, oracle.ask retries with feedback
5. After max retries, fail the edit
"""

from dataclasses import dataclass
from typing import Optional

from compass.agents.neo.types import FileEditorResponse, FileEditOperation
from compass.llm.oracle import OracleSchemaError


@dataclass
class FileEditResult:
    """Result from the FileEditor agent.

    success: True if we validated and can apply the edit
    """
    success: bool
    operation: str  # "replace" | "insert" | "delete"
    target: str = ""  # Code to find in file
    content: str = ""  # New code (for replace/insert)
    reasoning: str = ""
    error: Optional[str] = None


# FileEditor prompt - focused on understanding instruction and finding target code
FILE_EDITOR_PROMPT = """You are FileEditor. Find the exact code to modify.

INSTRUCTION: {instruction}

FILE: {file_path}
```
{file_content}
```

OPERATIONS:
- replace: Find target, replace with content
- insert: Find target, add content AFTER it
- delete: Find target, remove it

Target must be actual CODE copied exactly from the file (not whitespace or blank lines).
For insert: target the last line of code you want to insert AFTER.
"""


def _find_target_by_lines(target: str, file_content: str) -> Optional[str]:
    """Find target in file by matching stripped lines.

    Content blocks add extra indentation -- models indent code relative
    to the block marker, not the file. This finds the file's actual version
    by comparing stripped lines, returning it only if the match is unique.

    Returns the file's version of the target, or None if not found/ambiguous.
    """
    target_lines = [line.strip() for line in target.strip().splitlines() if line.strip()]
    if not target_lines:
        return None

    file_lines = file_content.splitlines()
    matches = []

    for i in range(len(file_lines) - len(target_lines) + 1):
        file_window = [file_lines[i + j].strip() for j in range(len(target_lines))]
        if file_window == target_lines:
            matches.append('\n'.join(file_lines[i:i + len(target_lines)]))

    return matches[0] if len(matches) == 1 else None


def _validate_file_edit(response: FileEditorResponse, file_content: str) -> Optional[str]:
    """
    Validate a FileEditor response against the file.

    Returns error string if invalid, None if valid.
    Mutates response.target in-place when whitespace correction finds a match.
    """
    operation = response.operation
    target = response.target
    content = response.content

    # Validate operation
    valid_ops = {FileEditOperation.REPLACE, FileEditOperation.INSERT, FileEditOperation.DELETE}
    if operation not in valid_ops:
        return f"Invalid operation '{operation}'. Must be replace/insert/delete."

    # Validate target exists
    if not target:
        hint = " For INSERT, copy the line of code you want to insert AFTER." if operation == FileEditOperation.INSERT else ""
        return f"No target specified. Copy exact code from the file.{hint}"

    if target not in file_content:
        # Models consistently add trailing whitespace -- tolerate it
        stripped = target.rstrip()
        if stripped and stripped in file_content:
            response.target = stripped
            target = stripped
        else:
            # Content blocks add wrong indentation -- try line-by-line match
            found = _find_target_by_lines(target, file_content)
            if found:
                response.target = found
                target = found
            else:
                return "Target not found in file. Copy the exact text from the file."

    # Validate target is unique
    count = file_content.count(target)
    if count > 1:
        return f"Target appears {count} times. Include more surrounding code to make it unique."

    # Validate content for operations that need it
    if operation in (FileEditOperation.REPLACE, FileEditOperation.INSERT) and not content:
        return f"Operation '{operation.value}' requires content field."

    return None  # Valid


def _parse_file_edit(response: FileEditorResponse) -> FileEditResult:
    """Convert FileEditorResponse to FileEditResult."""
    return FileEditResult(
        success=True,
        operation=response.operation.value,  # Convert enum to string
        target=response.target,
        content=response.content or "",
        reasoning=response.reasoning,
    )


def call_file_editor(
    oracle,
    file_path: str,
    file_content: str,
    instruction: str,
    max_retries: int = 3,
) -> FileEditResult:
    """
    Call the FileEditor agent with retry.

    Explicit loop: ask -> validate -> feedback -> retry.
    Each step visible via telemetry and debug output.

    Args:
        oracle: Oracle instance for LLM calls
        file_path: Relative path to the file
        file_content: File content (raw text)
        instruction: What change to make (natural language)
        max_retries: Max attempts before giving up

    Returns:
        FileEditResult - success=True if edit is valid and can be applied
    """
    from compass.cli import ui
    from compass.core.debug import show_prompt
    from compass.core.telemetry import record_task_attempt, record_task_attempt_failure

    prompt = FILE_EDITOR_PROMPT.format(
        file_path=file_path,
        file_content=file_content,
        instruction=instruction,
    )

    show_prompt("editor", "FILE EDITOR PROMPT", prompt, ui.Colors.green)

    last_error = "Max retries exhausted"

    for attempt in range(max_retries):
        record_task_attempt("editor", attempt)

        # 1. Ask the oracle (content blocks handle multi-line target/content)
        try:
            response = oracle.ask(prompt, FileEditorResponse, task="editor")
        except OracleSchemaError as e:
            record_task_attempt_failure("editor", attempt, "parse")
            last_error = str(e)
            prompt = _feedback_prompt(instruction, str(e), file_content)
            show_prompt("editor", "FILE EDITOR RETRY (parse)", prompt, ui.Colors.yellow)
            continue

        # 2. Validate against file (pure)
        error = _validate_file_edit(response, file_content)
        if error:
            record_task_attempt_failure("editor", attempt, "validation")
            last_error = error
            prompt = _feedback_prompt(instruction, error, file_content)
            show_prompt("editor", "FILE EDITOR RETRY (validation)", prompt, ui.Colors.yellow)
            continue

        # 3. Valid -- convert and return
        return _parse_file_edit(response)

    return FileEditResult(success=False, operation="", error=last_error)


def _feedback_prompt(instruction: str, error: str, file_content: str) -> str:
    """Build feedback prompt for retry."""
    return (
        f"Your previous edit attempt failed validation.\n\n"
        f"INSTRUCTION (unchanged): {instruction}\n\n"
        f"ERROR: {error}\n\n"
        f"FILE CONTENT (unchanged):\n```\n{file_content}\n```\n\n"
        f"Fix the issue. Make sure target is copied EXACTLY from the file."
    )
