"""
Programmer Agent - Abstract solution generation with Scribe validation.

The Programmer NFA implements its own internal Actor-Critic loop:
- Programmer (Actor): Generates abstract solution from problem statement
- Scribe (Critic): Validates against system constraints, requests patterns

Flow:
  UNDERSTAND -> DESIGN -> IMPLEMENT -> SCRIBE_REVIEW -> (feedback loop) -> DELIVER -> DONE

The Programmer sees only the problem and constraints - no file access.
The Scribe sees only the solution and system constraints - no problem.
"""

from compass.agents.programmer.context import (
    ProgrammerState,
    ProgrammerContext,
    ScribeView,
    ProgrammerResult,
)
from compass.agents.programmer.tool import call_programmer

__all__ = [
    "ProgrammerState",
    "ProgrammerContext",
    "ScribeView",
    "ProgrammerResult",
    "call_programmer",
]
