"""
Bash-Transport Terminal State Detection for NFA Orchestration.

This module provides bash-transport terminal state detection for NFA orchestration.
The pattern: agents are forced to produce proper content by looping until the
model invokes the proper command via bash transport.

Mini-SWE-Agent Pattern:
1. Agent loops until it calls a terminal state tool or bash command
2. No free-text parsing - everything goes through real program invocations
3. Orchestrator detects mode and dispatches appropriately

Bash-Transport Terminal State:
- Model outputs bash command in specific format (e.g., ```mswea_bash_command ...)
- NFA detects bash command invocation and transitions to terminal state
- Bash command is executed as trusted code, not regex on LLM text
"""

import re
from typing import Any, Callable, Dict, Optional, Tuple


# Regex pattern for bash-transport command detection
# Matches: ```mswea_bash_command ... ``` or similar bash command blocks
BASH_TRANSPORT_PATTERN = re.compile(
    r'```(?:mswea_)?bash_command\s*\n(.*?)\n```',
    re.DOTALL | re.IGNORECASE
)


def detect_bash_transport_terminal_state(content: str) -> Optional[str]:
    """
    Detect if content contains a bash-transport terminal state command.

    Args:
        content: The LLM output content to check

    Returns:
        The bash command if detected, None otherwise
    """
    match = BASH_TRANSPORT_PATTERN.search(content)
    if match:
        return match.group(1).strip()
    return None


def is_bash_transport_terminal(content: str) -> bool:
    """
    Check if content represents a bash-transport terminal state.

    Args:
        content: The LLM output content to check

    Returns:
        True if content contains a bash-transport terminal command
    """
    return detect_bash_transport_terminal_state(content) is not None


def extract_bash_command(content: str) -> Optional[str]:
    """
    Extract the bash command from bash-transport content.

    Args:
        content: The LLM output content

    Returns:
        The extracted bash command, or None if not found
    """
    return detect_bash_transport_terminal_state(content)


class BashTransportTerminalDetector:
    """
    Bash-transport terminal state detector for NFA orchestration.

    This class provides stateful detection of bash-transport terminal states
    within NFA transitions.
    """

    def __init__(self, pattern: str = BASH_TRANSPORT_PATTERN.pattern):
        self.pattern = re.compile(pattern, re.DOTALL | re.IGNORECASE)

    def detect(self, content: str) -> Optional[str]:
        """
        Detect bash-transport terminal state in content.

        Args:
            content: The LLM output content to check

        Returns:
            The bash command if detected, None otherwise
        """
        match = self.pattern.search(content)
        if match:
            return match.group(1).strip()
        return None

    def is_terminal(self, content: str) -> bool:
        """
        Check if content represents a bash-transport terminal state.

        Args:
            content: The LLM output content to check

        Returns:
            True if content contains a bash-transport terminal command
        """
        return self.detect(content) is not None
