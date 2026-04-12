"""
ShellBuilder - Specialized agent for complex shell commands.

Separation of concerns:
- Actor (big context): WHAT to run - understands problem, decides to run command
- ShellBuilder (small context): HOW to write it - correct bash syntax

ShellBuilder uses the common retry loop from compass.core:
1. LLM outputs command
2. We validate (syntax check, dangerous patterns)
3. If invalid, feed error back to LLM and retry
4. After max retries, fail the command
"""

import re
import shlex
from dataclasses import dataclass
from typing import List, Optional

from compass.agents.neo.types import ShellBuilderResponse
from compass.llm.oracle import OracleSchemaError


@dataclass
class ShellBuildResult:
    """Result from the ShellBuilder agent.

    success: True if command is valid and safe to execute
    """
    success: bool
    command: str = ""
    explanation: str = ""
    warnings: List[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# ShellBuilder prompt - focused on correct shell commands
SHELL_BUILDER_PROMPT = """You are ShellBuilder. Write a shell command.

INTENT: {intent}

CONTEXT:
{context}

SHELL BASICS:
- Single quotes preserve literally: echo 'Price: $100'
- Double quotes allow expansion: echo "Home: $HOME"
- Quote paths with spaces: cat "my file.txt"

DANGEROUS PATTERNS TO AVOID:
- rm -rf / or rm -rf /* (catastrophic)
- Unquoted variables: $VAR (use "$VAR")

OUTPUT: ShellBuilderResponse with command, explanation, warnings."""


# Feedback prompt when validation fails
SHELL_BUILDER_FEEDBACK = """Your previous command failed validation.

ERROR: {error}

PREVIOUS ATTEMPT:
{previous_command}

INTENT (unchanged): {intent}

CONTEXT (unchanged):
{context}

Fix the issue and try again. Pay attention to:
- Use single quotes for literal $ (e.g., echo 'Price: $100')
- Avoid dangerous patterns
- Ensure command syntax is valid"""


def _validate_command(command: str) -> Optional[str]:
    """
    Validate shell command for common issues.

    Returns error message if invalid, None if valid.
    """
    if not command or not command.strip():
        return "Empty command"

    # Check for catastrophic patterns
    dangerous_patterns = [
        (r'rm\s+(-[rf]+\s+)*/', "Dangerous: rm on root directory"),
        (r'rm\s+(-[rf]+\s+)*\*', "Dangerous: rm with glob at root"),
        (r'>\s*/dev/sd[a-z]', "Dangerous: writing to block device"),
        (r'mkfs\.', "Dangerous: filesystem format command"),
        (r'dd\s+.*of=/dev/sd', "Dangerous: dd to block device"),
    ]

    for pattern, msg in dangerous_patterns:
        if re.search(pattern, command):
            return msg

    # Check for unescaped $ in double quotes (common mistake)
    # Pattern: "...$....." where $ is not escaped
    in_double_quote = False
    i = 0
    while i < len(command):
        c = command[i]
        if c == '"' and (i == 0 or command[i-1] != '\\'):
            in_double_quote = not in_double_quote
        elif c == "'" and not in_double_quote:
            # Skip single-quoted section
            end = command.find("'", i + 1)
            if end == -1:
                return "Unclosed single quote"
            i = end
        elif c == '$' and in_double_quote:
            # Check if it's escaped or a valid expansion
            if i > 0 and command[i-1] == '\\':
                pass  # Escaped, OK
            elif i + 1 < len(command) and command[i+1] in '({':
                pass  # Command substitution, OK
            elif i + 1 < len(command) and (command[i+1].isalpha() or command[i+1] == '_'):
                pass  # Variable expansion, probably intentional
            else:
                # Bare $ followed by digit or special char - likely mistake
                if i + 1 < len(command) and command[i+1].isdigit():
                    return f"$ followed by digit in double quotes expands as positional param. Use single quotes for literals: echo 'Price: $100'"
        i += 1

    # Basic syntax check using shlex
    try:
        shlex.split(command)
    except ValueError as e:
        return f"Shell syntax error: {e}"

    return None


def _parse_shell_build(response: ShellBuilderResponse) -> ShellBuildResult:
    """Convert ShellBuilderResponse to ShellBuildResult."""
    return ShellBuildResult(
        success=True,  # Will be overwritten if validation failed
        command=response.command,
        explanation=response.explanation,
        warnings=response.warnings or [],
    )


def call_shell_builder(
    oracle,
    intent: str,
    context: str = "",
    max_retries: int = 3,
) -> ShellBuildResult:
    """
    Call the ShellBuilder agent with retry.

    Uses oracle.ask() with external validation - full context preserved
    in message history, no arbitrary truncation.

    Args:
        oracle: Oracle instance for LLM calls
        intent: What the command should do (natural language)
        context: Additional context (current directory, files, etc.)
        max_retries: Max attempts before giving up

    Returns:
        ShellBuildResult - success=True if command is valid
    """
    from compass.cli import ui
    from compass.core.debug import show_prompt

    prompt = SHELL_BUILDER_PROMPT.format(
        intent=intent,
        context=context or "(no additional context)",
    )

    on_prompt = lambda p: show_prompt("shell", "SHELL BUILDER PROMPT", p, ui.Colors.yellow)

    try:
        response = oracle.ask(
            prompt,
            ShellBuilderResponse,
            max_retries=max_retries,
            task="shell-builder",
            validate=_validate_command_wrapper,
            feedback_suffix="Fix the issue. Use single quotes for literal $.",
            on_prompt=on_prompt,
        )
        return _parse_shell_build(response)

    except OracleSchemaError as e:
        # Return failure with context
        return ShellBuildResult(
            success=False,
            command="",
            error=str(e),
        )


def _validate_command_wrapper(response: ShellBuilderResponse) -> Optional[str]:
    """Wrapper to validate command from ShellBuilderResponse."""
    return _validate_command(response.command)


def _parse_builder_response(response: ShellBuilderResponse) -> ShellBuildResult:
    """Convert ShellBuilderResponse into ShellBuildResult.

    Note: This is a simple parser without validation.
    Use call_shell_builder() for the full retry loop with validation.
    """
    if not response.command:
        return ShellBuildResult(
            success=False,
            error="No command provided",
        )

    return ShellBuildResult(
        success=True,
        command=response.command,
        explanation=response.explanation,
        warnings=response.warnings or [],
    )
