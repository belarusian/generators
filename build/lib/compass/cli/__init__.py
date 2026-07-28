"""
CLI - User interface layer.

- commands: Slash commands (/help, /session, etc.)
- driver: Approval flow (UserDriver, ClaudeDriver)
- input: User input parsing
- files: @file reference expansion
- ui: Output formatting
- main: Entry point for the `compass` command
"""

from .main import main

__all__ = ["main"]
