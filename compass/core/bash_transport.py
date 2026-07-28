"""Bash-transport terminal state detection for NFA orchestration.

This module provides terminal state detection when the model invokes
proper bash commands via transport. The NFA loops until a bash command
is invoked via transport, with no constraints that prevent generations.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Tuple

from compass.core.nfa_types import State, Transition, TransitionFn


# Pattern to detect bash commands in model output
BASH_COMMAND_PATTERN = re.compile(
    r'^\s*(?:bash|sh|exec|run|\.sh|python|node|npm|pip|git|curl|wget|ssh|scp|rsync|make|docker|kubectl)\b',
    re.IGNORECASE
)

# Pattern to detect bash command execution transport markers
BASH_TRANSPORT_MARKER = re.compile(
    r'(?i)(bash-transport|bash_command|execute_bash|run_bash|shell_command|terminal_command)'
)


def is_bash_command(text: str) -> bool:
    """Detect if text contains a bash command invocation."""
    if not text or not isinstance(text, str):
        return False
    
    # Check for bash transport marker
    if BASH_TRANSPORT_MARKER.search(text):
        return True
    
    # Check for bash command pattern at start of line
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if BASH_COMMAND_PATTERN.match(line):
            return True
        
        # Check for common bash constructs
        if line.startswith('$ ') or line.startswith('# ') or line.startswith('cd ') or \
           line.startswith('export ') or line.startswith('source ') or line.startswith('. '):
            return True
            
    return False


def bash_transport_transition(
    context: Any,
    is_bash_invoked: Optional[Callable[[Any], bool]] = None,
) -> Tuple[str, Any]:
    """NFA transition that detects bash command invocation via transport.
    
    Returns ('bash_transport_invoked', context) if bash command is detected,
    otherwise returns ('continue', context).
    """
    # Check if context has bash_invoked flag or method
    if hasattr(context, 'bash_invoked') and context.bash_invoked:
        return ('bash_transport_invoked', context)
    
    if hasattr(context, 'get_bash_invoked'):
        if context.get_bash_invoked():
            return ('bash_transport_invoked', context)
    
    # Check if context has messages or output to analyze
    if hasattr(context, 'messages'):
        for msg in context.messages:
            content = msg.get('content', '') if isinstance(msg, dict) else str(msg)
            if is_bash_command(content):
                return ('bash_transport_invoked', context)
                
    if hasattr(context, 'output') and is_bash_command(context.output):
        return ('bash_transport_invoked', context)
        
    if is_bash_invoked and is_bash_invoked(context):
        return ('bash_transport_invoked', context)
        
    return ('continue', context)


def create_bash_transport_state_machine(
    initial_state: str = 'waiting_for_bash',
    bash_invoked_state: str = 'bash_transport_invoked',
    continue_state: str = 'continue_loop',
):
    """Create an NFA state machine for bash-transport terminal state detection.
    
    States:
    - waiting_for_bash: Initial state, waiting for bash command invocation
    - continue_loop: Loop state, continuing to wait for bash command
    - bash_transport_invoked: Terminal state, bash command was invoked
    
    Returns a tuple of (transitions, states, bash_invoked_state).
    """
    transitions: Dict[str, TransitionFn] = {
        initial_state: lambda ctx: bash_transport_transition(ctx),
        continue_state: lambda ctx: bash_transport_transition(ctx),
    }
    
    # Add terminal state
    states = {
        initial_state: State(id=initial_state),
        continue_state: State(id=continue_state),
        bash_invoked_state: State(id=bash_invoked_state),
    }
    
    return transitions, states, bash_invoked_state
