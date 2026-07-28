"""
Branch result types for parallel execution.

Pure data types capturing the outcome of a single branch execution.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from compass.agents.neo.types import ExecutionResult, ExecutionStatus
from compass.core.ui_events import UIEventStream

if TYPE_CHECKING:
    from compass.agents.neo.trace import ActionTrace


@dataclass(frozen=True)
class BranchResult:
    """
    Immutable result from executing a single branch.

    Captures execution outcome with full metadata for comparison
    and debugging. Used by race/parallel execution strategies.

    Fields:
        execution_result: The Actor's terminal result (success/replan/done)
        provider_name: Which LLM provider executed this branch
        started_at: ISO timestamp when branch started
        completed_at: ISO timestamp when branch completed
        duration_seconds: Wall-clock execution time
        ui_events: Stream of UI events from execution
        thinking: Accumulated thinking/reasoning text
        action_traces: Tuple of action traces for debugging
        actions_attempted: Total actions tried
        actions_succeeded: Actions that completed successfully
        retries_used: Number of retry attempts consumed
    """
    execution_result: ExecutionResult
    provider_name: str
    started_at: str
    completed_at: str
    duration_seconds: float
    ui_events: UIEventStream = field(default_factory=lambda: UIEventStream())
    thinking: str = ""
    action_traces: tuple["ActionTrace", ...] = ()
    actions_attempted: int = 0
    actions_succeeded: int = 0
    retries_used: int = 0

    @property
    def success_rate(self) -> float:
        """Ratio of successful actions to attempted actions."""
        return (
            self.actions_succeeded / self.actions_attempted
            if self.actions_attempted > 0
            else 0.0
        )

    @property
    def is_complete(self) -> bool:
        """Whether execution completed successfully."""
        return self.execution_result.status == ExecutionStatus.SUCCESS
