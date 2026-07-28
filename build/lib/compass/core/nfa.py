"""
Generic NFA Runner - The core state machine loop.

This module provides a reusable state machine that can be composed
to build any agent workflow. Each NFA defines its own states and
context, but shares this common execution infrastructure.

The key insight: NFAs can invoke other NFAs as actions, enabling
fractal composition where each level has bounded context.

Evolution: Neo added nfa_types.py (State, Transition, NFAConfig, CriticResult)
and nfa_evolution.py (self-rewriting kernel). The types are re-exported here
so the evolution infrastructure composes with the runner.
"""

import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Set, Tuple, TypeVar


# Generic type variables
S = TypeVar('S', bound=Enum)  # State type (must be an Enum)
C = TypeVar('C')              # Context type (any dataclass)

# Re-export evolution types so compass.core.nfa is the single import point
from compass.core.nfa_types import (  # noqa: F401
    State, Transition, NFAConfig, CriticResult, NestedFiniteAutomaton,
)


@dataclass
class NFAResult(Generic[C]):
    """
    Result from running an NFA.

    Attributes:
        success: True if NFA reached a success state
        final_state: The terminal state reached
        context: The final context after all transitions
        error: Error message if failed
        iterations: Number of transitions executed
        history: Tuple of prior NFAResults (for evolution traceability)
    """
    success: bool
    final_state: Any
    context: C
    error: Optional[str] = None
    iterations: int = 0
    history: Tuple['NFAResult', ...] = ()

    @property
    def value(self):
        """Alias for context -- Neo's evolution API uses .value."""
        return self.context


# Type alias for transition functions
# A transition takes context and returns (new_state, new_context)
TransitionFn = Callable[[C], Tuple[S, C]]


def with_cancellation(
    transition_fn: TransitionFn,
    is_cancelled: Callable[[], bool],
    cancelled_state: S,
) -> TransitionFn:
    """
    Compose a transition with cancellation check.

    FP pattern: wrap transitions instead of adding imperative checks.
    The cancellation check happens at transition boundaries, not mid-state.
    """
    def wrapped(ctx: C) -> Tuple[S, C]:
        if is_cancelled():
            return (cancelled_state, ctx)
        return transition_fn(ctx)
    return wrapped


class NFARunner(Generic[S, C]):
    """
    Generic NFA runner - executes state machines with any state/context types.

    Supports both the original call convention (transitions=, success_states=,
    max_iterations=, on_transition=) and Neo's simplified form (states=,
    terminal_states= only).
    """

    def __init__(
        self,
        transitions: Dict[S, TransitionFn[C, S]] = None,
        initial_state: S = None,
        terminal_states: Set[S] = None,
        success_states: Set[S] = None,
        max_iterations: int = 100,
        on_transition: Optional[Callable] = None,
        # Neo's simplified constructor
        states: Dict[S, TransitionFn[C, S]] = None,
    ):
        # Accept either 'transitions' or 'states' (Neo's name)
        self.transitions = transitions or states or {}
        self.initial_state = initial_state
        self.terminal_states = terminal_states or set()
        self.success_states = success_states or self.terminal_states
        self.max_iterations = max_iterations
        self.on_transition = on_transition

        if success_states and not success_states.issubset(self.terminal_states):
            invalid = success_states - self.terminal_states
            raise ValueError(
                f"success_states must be subset of terminal_states. "
                f"Invalid states: {invalid}"
            )

    def run(self, initial_context: C) -> NFAResult[C]:
        """Execute the NFA until a terminal state is reached."""
        state = self.initial_state
        ctx = initial_context
        iterations = 0

        while state not in self.terminal_states:
            if iterations >= self.max_iterations:
                return NFAResult(
                    success=False,
                    final_state=state,
                    context=ctx,
                    error=f"Max iterations ({self.max_iterations}) exceeded at state {state}",
                    iterations=iterations,
                )

            transition_fn = self.transitions.get(state)
            if not transition_fn:
                return NFAResult(
                    success=False,
                    final_state=state,
                    context=ctx,
                    error=f"No transition defined for state: {state}",
                    iterations=iterations,
                )

            prev_state = state
            start_time = time.time()
            try:
                state, ctx = transition_fn(ctx)
            except Exception as e:
                tb = traceback.format_exc()
                error_msg = (
                    f"NFA error: Transition from {prev_state} failed: {e}\n"
                    f"--- Traceback ---\n{tb}"
                )
                print(error_msg)
                return NFAResult(
                    success=False,
                    final_state=prev_state,
                    context=ctx,
                    error=error_msg,
                    iterations=iterations,
                )
            duration = time.time() - start_time
            iterations += 1

            if self.on_transition:
                self.on_transition(prev_state, state, ctx, iterations, duration)

        return NFAResult(
            success=(state in self.success_states),
            final_state=state,
            context=ctx,
            iterations=iterations,
        )

    def run_from(self, state: S, context: C) -> NFAResult[C]:
        """Execute the NFA starting from a specific state."""
        original_initial = self.initial_state
        self.initial_state = state
        try:
            return self.run(context)
        finally:
            self.initial_state = original_initial


def compose_nfas(
    outer_transitions: Dict[S, TransitionFn[C, S]],
    inner_runner: NFARunner,
    invoke_state: S,
    extract_inner_context: Callable[[C], Any],
    merge_inner_result: Callable[[C, NFAResult], C],
    next_state_on_success: S,
    next_state_on_failure: S,
) -> Dict[S, TransitionFn[C, S]]:
    """
    Compose two NFAs by embedding the inner NFA as a transition in the outer.

    This enables fractal composition: the outer NFA can invoke the inner NFA
    at a specific state, and the inner NFA runs to completion before the
    outer NFA continues.
    """
    def composed_transition(ctx: C) -> Tuple[S, C]:
        inner_ctx = extract_inner_context(ctx)
        result = inner_runner.run(inner_ctx)
        ctx = merge_inner_result(ctx, result)
        return (next_state_on_success if result.success else next_state_on_failure), ctx

    outer_transitions[invoke_state] = composed_transition
    return outer_transitions
