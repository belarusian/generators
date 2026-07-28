"""Critics infrastructure for NFA evolution.

Run diagnostic critics against an NFA to detect degradation,
loops, or opportunities for rewriting.
"""

from __future__ import annotations
from typing import Any, Callable, List
from .nfa_types import CriticResult


def run_critics(
    nfa: Any,
    goal: Any,
    critics: List[Callable[[Any, Any], CriticResult]],
) -> CriticResult:
    """
    Run all critics and aggregate their diagnoses.

    Each critic returns a CriticResult. If any critic detects a problem
    (loop, stuck state, error), the aggregate result reflects it.
    """
    aggregate = CriticResult(success=True)

    for critic in critics:
        result = critic(nfa, goal)
        if result.loop_detected:
            return CriticResult(
                success=False,
                loop_detected=True,
                stuck_state=result.stuck_state,
                should_rewrite=True,
            )
        if result.error_messages:
            aggregate = CriticResult(
                success=False,
                error_messages=list(aggregate.error_messages) + list(result.error_messages),
                should_rewrite=True,
            )

    return aggregate
