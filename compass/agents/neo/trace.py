"""
Execution trace - structured tracking of actions taken during plan execution.

Keeps the main execution logic clean while providing rich introspection
for tests, debugging, and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from compass.core.content import truncate_chars


@dataclass
class ActionTrace:
    """Structured record of a single action execution."""
    action_type: str
    target: str
    success: bool
    result: str
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_observation: bool = False  # True for reasoning-only entries
    key: tuple = ()  # Hashable key for action comparison (from singledispatch action_key)

    def __str__(self) -> str:
        icon = "PASS:" if self.success else "FAIL:"
        return f"{icon} {self.action_type} {self.target}: {self.result}"


@dataclass
class StepTrace:
    """Trace of a single step's execution."""
    step_index: int
    step_text: str
    actions: List[ActionTrace] = field(default_factory=list)
    observation: Optional[str] = None  # Final reasoning when no actions
    complete: bool = False

    @property
    def action_count(self) -> int:
        """Count of actual actions (excludes observations)."""
        return sum(1 for a in self.actions if not a.is_observation)

    @property
    def retry_count(self) -> int:
        """Number of retries (actions beyond the first)."""
        return max(0, self.action_count - 1)


@dataclass
class ExecutionTrace:
    """Full trace of plan execution."""
    steps: List[StepTrace] = field(default_factory=list)

    def add_action(self, step_index: int, action: ActionTrace) -> None:
        """Add an action to a step's trace."""
        while len(self.steps) <= step_index:
            self.steps.append(StepTrace(step_index=len(self.steps), step_text=""))
        self.steps[step_index].actions.append(action)

    def set_step_text(self, step_index: int, text: str) -> None:
        """Set the step text."""
        while len(self.steps) <= step_index:
            self.steps.append(StepTrace(step_index=len(self.steps), step_text=""))
        self.steps[step_index].step_text = text

    def set_observation(self, step_index: int, observation: str) -> None:
        """Set the final observation for a step."""
        while len(self.steps) <= step_index:
            self.steps.append(StepTrace(step_index=len(self.steps), step_text=""))
        self.steps[step_index].observation = observation

    @property
    def total_actions(self) -> int:
        """Total action count across all steps."""
        return sum(s.action_count for s in self.steps)

    @property
    def total_retries(self) -> int:
        """Total retry count across all steps."""
        return sum(s.retry_count for s in self.steps)

    def actions_for_step(self, step_index: int) -> List[ActionTrace]:
        """Get actions for a specific step."""
        if step_index < len(self.steps):
            return [a for a in self.steps[step_index].actions if not a.is_observation]
        return []

    def truncate_error(self, error: str, max_length: int = 200) -> str:
        """Truncate error message."""
        return truncate_chars(error, max_chars=max_length)

    def truncate_result(self, result: str, max_length: int = 200) -> str:
        """Truncate result."""
        return truncate_chars(result, max_chars=max_length)


def trace_from_action(
    action_type: str,
    target: str,
    success: bool,
    result: str,
    action,  # Typed action dataclass
) -> ActionTrace:
    """Create ActionTrace from execution context."""
    from dataclasses import asdict
    from compass.agents.neo.dispatch import action_key

    return ActionTrace(
        action_type=action_type,
        target=target,
        success=success,
        result=result,
        params=asdict(action),
        reasoning=action.reasoning or "",
        key=action_key(action),  # Store key for comparison
    )
