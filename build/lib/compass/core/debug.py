"""
Debug configuration for showing prompts from various NFA states.

Controls which state prompts are displayed when DEBUG=1 is set.

Usage:
    DEBUG=1 DEBUG_PROMPTS=all pytest ...           # show all prompts
    DEBUG=1 DEBUG_PROMPTS=actor,editor pytest ...  # show specific states
    DEBUG=1 pytest ...                             # default: actor only

Available states:
    actor      - Main Actor agent
    editor     - FileEditor (edit_file action)
    shell      - ShellBuilder (shell_command action)
    critic     - Critic evaluation
    answerer   - Final answer generation
    programmer - Programmer NFA states
    learning   - Learning extraction
    progress   - Progress assessment
    judge      - Loop detector (progress assessor model)
"""

import os
from typing import Set

# Default: only show actor prompts (backwards compatible)
DEFAULT_PROMPTS = "actor"

# All available prompt states
ALL_STATES = {
    "actor",
    "editor",
    "shell",
    "critic",
    "answerer",
    "programmer",
    "learning",
    "progress",
    "judge",  # Loop detector / progress assessor
}


def _get_enabled_prompts() -> Set[str]:
    """Get the set of enabled prompt states."""
    # Prompt dumps are developer-only, gated by DEBUG=1
    if not os.getenv("DEBUG"):
        return set()

    prompts = os.getenv("DEBUG_PROMPTS", DEFAULT_PROMPTS)

    if prompts == "all":
        return ALL_STATES
    if prompts == "none":
        return set()

    return set(p.strip() for p in prompts.split(","))


def should_show_prompt(state: str) -> bool:
    """
    Check if we should show the prompt for this state.

    Args:
        state: One of: actor, editor, shell, critic, answerer, programmer, learning, progress

    Returns:
        True if the prompt should be displayed
    """
    return state in _get_enabled_prompts()


def show_prompt(state: str, title: str, prompt: str, color_fn=None) -> None:
    """
    Display a prompt if enabled for this state.

    Args:
        state: The state identifier (actor, editor, etc.)
        title: Display title like "ACTOR PROMPT"
        prompt: The actual prompt content
        color_fn: Optional color function from ui.Colors
    """
    if not should_show_prompt(state):
        return

    from compass.cli import ui

    # Default to cyan if no color specified
    if color_fn is None:
        color_fn = ui.Colors.cyan

    print(f"\n{color_fn('='*60)}\n{color_fn(f'[{title}]')}\n{color_fn('='*60)}")
    print(ui.Colors.dim(prompt))
    print(f"{color_fn('='*60)}\n")
