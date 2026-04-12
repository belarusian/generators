"""
Compass Agents - Composable NFA-based agents.

This module contains specialized agents that can be invoked as actions.
Each agent is its own NFA with bounded context, following the fractal
Actor-Critic pattern.

Available agents:
- Programmer: Abstract solution generation with Scribe (Critic) validation
"""

from compass.agents.programmer import call_programmer, ProgrammerResult

__all__ = [
    "call_programmer",
    "ProgrammerResult",
]
