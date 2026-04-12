"""
Actor Loop - Immutable state machine for Actor execution.

This module provides the FP infrastructure for Actor iteration loops.
The pattern: immutable LoopState with transition methods, threaded
through iterations until a terminal ExecutionResult is produced.

    LoopState  = intermediate state (keep iterating)
    ExecutionResult = terminal state (halt with payload)

This is the inner NFA of an Actor - distinct from the outer NFA
(ACT -> REVIEW -> ANSWER) that orchestrates the full request flow.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, TYPE_CHECKING, Protocol, runtime_checkable
from enum import Enum

from compass.agents.neo.types import ExecutionResult, ExecutionStatus

if TYPE_CHECKING:
    from compass.agents.neo.trace import ActionTrace, ExecutionTrace

T = TypeVar('T')


# --- Protocol: what makes a response "actionable" ---

class ActionStatus(Enum):
    """Status for actionable responses."""
    CONTINUE = "continue"  # More work to do, you get another turn
    # ^^^ ---- you get to see results of your actions
    COMPLETE = "complete" # Request fulfilled
    DONE = "done" # Request fulfilled


@runtime_checkable
class ActionableResponse(Protocol):
    """Protocol for responses that can emit actions.

    Any response type with these fields can be used with make_actionable.
    The type IS the contract - no registration needed.
    """
    status: ActionStatus
    actions: Optional[List[Any]]  # List[Action] but avoid circular import


@dataclass(frozen=True)
class LoopState:

    """
    Immutable state threaded through Actor execution loop.

    FP pattern: instead of mutating variables, each transition
    returns a new LoopState with updated values.

    Fields:
        iteration: Current loop iteration (0-indexed)
        consecutive_failures: Failures since last success (for escalation)
        consecutive_syntactic_failures: Syntactic failures since last success (for escalation)
        creativity_iteration: Controls temperature/sampling (higher = more creative)
        current_provider: Active LLM provider (None = use default)
        action_results: Accumulated action result strings
        errors_content: Full error content for context [(target, error), ...]
        exec_globals: Python globals from exec actions
        files_read: {path: [(start, end), ...]} line ranges read
        files_read_content: {path: [(start, end, content), ...]} with content
        action_history: Actions executed (for loop detection)
        last_error: Most recent error message
    """
    iteration: int = 0
    consecutive_failures: int = 0
    consecutive_syntactic_failures: int = 0
    creativity_iteration: int = 0
    current_provider: Optional[Any] = None  # Provider type
    action_results: Tuple[str, ...] = ()
    errors_content: Tuple[Tuple[str, str], ...] = ()
    exec_globals: Dict[str, Any] = field(default_factory=dict)
    files_read: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    files_read_content: Dict[str, List[Tuple[int, int, str]]] = field(default_factory=dict)
    action_history: Tuple["ActionTrace", ...] = ()
    last_error: Optional[str] = None
    file_snapshots: Dict[str, str] = field(default_factory=dict)  # Pre-modification snapshots for REVERT
    progress_signal: Optional[str] = None  # Signal type: stalled, oscillating, stuck
    progress_feedback: Optional[str] = None  # Suggestion from progress assessor
    directive: Optional[str] = None  # Feedback from outer NFA (Critic/REPLAN)
    hesitation: Optional[str] = None  # Diagonal question at rewrite boundary

    # --- Transition methods (return new state) ---

    def fail(self, error: str) -> "LoopState":
        """Record a failure, increment failure count."""
        return replace(self,
            consecutive_failures=self.consecutive_failures + 1,
            last_error=error,
        )

    def fail_syntactic(self, error: str) -> "LoopState":
        """Record a syntactic failure (malformed output)."""
        return replace(self,
            consecutive_failures=self.consecutive_failures + 1,
            consecutive_syntactic_failures=self.consecutive_syntactic_failures + 1,
            last_error=error,
        )

    def reset_failures(self) -> "LoopState":
        """Reset failure count (after success)."""
        return replace(self, consecutive_failures=0)

    def adjust_creativity(self, delta: int) -> "LoopState":
        """Adjust creativity iteration (+1 = more creative, -1 = more deterministic)."""
        return replace(self, creativity_iteration=max(0, self.creativity_iteration + delta))

    def with_progress_feedback(self, signal: Optional[str], suggestion: Optional[str]) -> "LoopState":
        """Set progress feedback (signal + suggestion from assessor)."""
        return replace(self, progress_signal=signal, progress_feedback=suggestion)

    def with_directive(self, directive: Optional[str]) -> "LoopState":
        """Set directive from outer NFA (Critic/REPLAN feedback)."""
        return replace(self, directive=directive)

    def next_iteration(self) -> "LoopState":
        """Advance to next iteration. Clears one-shot signals (hesitation)."""
        return replace(self, iteration=self.iteration + 1, hesitation=None)

    def with_results(
        self,
        results: List[str],
        files_read: Dict,
        files_read_content: Dict,
        errors_content: List[Tuple[str, str]],
        traces: List["ActionTrace"] = None,
        file_snapshots: Dict[str, str] = None,
    ) -> "LoopState":
        """Add execution results from a batch of actions.

        IMPORTANT: files_read and files_read_content are MERGED, not replaced.
        This ensures file reads persist across model escalations.
        File snapshots are also merged (first snapshot wins - don't overwrite).
        """
        # Merge files_read: {path: [(start, end), ...]}
        merged_files_read = dict(self.files_read)
        for path, ranges in files_read.items():
            if path in merged_files_read:
                merged_files_read[path] = merged_files_read[path] + ranges
            else:
                merged_files_read[path] = ranges

        # Merge files_read_content: {path: [(start, end, content), ...]}
        merged_files_content = dict(self.files_read_content)
        for path, chunks in files_read_content.items():
            if path in merged_files_content:
                merged_files_content[path] = merged_files_content[path] + chunks
            else:
                merged_files_content[path] = chunks

        # Merge file_snapshots (first snapshot wins - preserve original state)
        merged_snapshots = dict(self.file_snapshots)
        for path, content in (file_snapshots or {}).items():
            if path not in merged_snapshots:
                merged_snapshots[path] = content

        return replace(self,
            action_results=self.action_results + tuple(results),
            files_read=merged_files_read,
            files_read_content=merged_files_content,
            errors_content=self.errors_content + tuple(errors_content),
            action_history=self.action_history + tuple(traces or []),
            file_snapshots=merged_snapshots,
        )

    def with_action(self, action: "ActionTrace") -> "LoopState":
        """Record an action in history (for loop detection)."""
        return replace(self, action_history=self.action_history + (action,))

def with_hesitation(
    pre: "LoopState",
    post: "LoopState",
    rule: str,
) -> "LoopState":
    """
    Diagonal hesitation: if a transition changed the grammar, inject a question.

    Pure function. Compares pre/post state to detect when the system
    rewrites its own behavior (creativity, escalation) and records WHY --
    so the next iteration can question the assumption, not just execute it.

    This is the metacognitive crack: the system pauses to ask
    "what did I just assume to be true, and why did I need to break it?"
    """
    # Detect grammar change: creativity shifted (strategy rewrite)
    creativity_changed = pre.creativity_iteration != post.creativity_iteration
    # Detect escalation boundary: crossed max failures threshold
    escalation_boundary = (
        pre.consecutive_failures < 3 <= post.consecutive_failures
    )

    if escalation_boundary:
        return replace(post, hesitation=(
            f"[{rule}] Consecutive failures reached {post.consecutive_failures}. "
            f"Escalating to Critic. The current strategy has exhausted its attempts."
        ))
    if creativity_changed:
        delta = post.creativity_iteration - pre.creativity_iteration
        direction = "more creative" if delta > 0 else "more deterministic"
        return replace(post, hesitation=(
            f"[{rule}] Strategy shifted {direction} "
            f"(creativity {pre.creativity_iteration}->{post.creativity_iteration}). "
            f"Assumption broken: previous approach was {'too rigid' if delta > 0 else 'too random'}."
        ))
    return post

def to_result(
    state: LoopState,
    status: ExecutionStatus,
    feedback: str = None,
    last_action: Optional[Dict] = None,
    last_error: Optional[str] = None,
    trace: Optional["ExecutionTrace"] = None,
) -> ExecutionResult:
    """
    Convert LoopState to terminal ExecutionResult.

    Pure function - extracts accumulated data from state into result.
    """
    return ExecutionResult(
        status=status,
        action_results=list(state.action_results),
        files_read=state.files_read,
        files_read_content=state.files_read_content,
        exec_globals=state.exec_globals,
        feedback=feedback,
        file_snapshots=state.file_snapshots,
        errors_content=list(state.errors_content),
        last_action=last_action,
        last_error=last_error,
        trace=trace,
    )


# Type alias for the Either pattern: continue or halt
LoopStep = Union[LoopState, ExecutionResult]


def run_loop(
    initial_state: LoopState,
    step_fn: Callable[[LoopState], LoopStep],
    max_iterations: int = 20,
) -> ExecutionResult:
    """
    Generic Actor loop runner.

    Threads state through step_fn until ExecutionResult is returned.

    Args:
        initial_state: Starting LoopState
        step_fn: (LoopState) -> LoopState | ExecutionResult
        max_iterations: Safety limit

    Returns:
        ExecutionResult (terminal state)
    """
    state = initial_state

    while state.iteration < max_iterations:
        result = step_fn(state)
        if isinstance(result, ExecutionResult):
            return result
        state = result

    # Max iterations reached
    return to_result(state, ExecutionStatus.DONE)


# --- Composition: make any response type actionable ---

def make_actionable(
    ask_fn: Callable[..., T],
    execute_fn: Callable[[List[Any], "LoopState"], "ActionBatchResult"],
    extract_response: Callable[[T], Tuple[ActionStatus, Optional[List[Any]], Any]] = None,
) -> Callable[["LoopState"], LoopStep]:
    """
    Lift any ask function into an actionable step function.

    This is the core composition: (AskFn, ExecuteFn) -> StepFn

    The Type (T) defines the shape - must have status + actions.
    The Function (step_fn) threads state through the loop.

    Args:
        ask_fn: Function that calls oracle and returns response of type T
        execute_fn: Function that executes actions and returns batch result
        extract_response: Optional extractor for (status, actions, payload)
                         Default assumes T has .status and .actions attributes

    Returns:
        step_fn suitable for run_loop

    Example:
        # Make Critic actionable
        critic_step = make_actionable(
            ask_fn=lambda state: oracle.ask(build_prompt(state), CriticOutputActionable),
            execute_fn=lambda actions, state: _execute_actions(actions, ...),
        )
        result = run_loop(LoopState(), critic_step)
    """
    def default_extract(response: T) -> Tuple[ActionStatus, Optional[List[Any]], T]:
        """Default extractor - assumes ActionableResponse protocol."""
        status = getattr(response, 'status', ActionStatus.COMPLETE)
        actions = getattr(response, 'actions', None)
        # Normalize status if it's a different enum
        if hasattr(status, 'value'):
            status = ActionStatus.COMPLETE if (status.value == 'complete' or status.value == "done") else ActionStatus.CONTINUE
        return status, actions, response

    extractor = extract_response or default_extract

    def step(state: LoopState) -> LoopStep:
        """The composed step function."""
        # Ask oracle
        response = ask_fn(state)

        # Extract status, actions, payload
        status, actions, payload = extractor(response)

        # No actions or complete -> terminal
        if not actions or status == ActionStatus.COMPLETE or status == ActionStatus.DONE:
            return to_result(state, ExecutionStatus.SUCCESS)

        # Execute actions
        batch = execute_fn(actions, state)

        # Handle failure
        if not batch.success:
            new_state = state.fail(batch.last_error or "Action failed")
            # Could add escalation logic here
            return to_result(new_state, ExecutionStatus.EVALUATE, feedback=batch.last_error)

        # Success - continue with updated state
        return state.with_results(
            batch.results,
            batch.files_read,
            batch.files_read_content,
            batch.errors_content,
            batch.traces,
            batch.file_snapshots,
        ).next_iteration()

    return step


# Type alias for batch result (avoid circular import)
class ActionBatchResult(Protocol):
    """Protocol for action batch results."""
    success: bool
    results: List[str]
    last_error: Optional[str]
    files_read: Dict
    files_read_content: Dict
    errors_content: List[Tuple[str, str]]
    traces: List[Any]
    file_snapshots: Dict[str, str]
