"""
Execution configuration - immutable context that flows through all calls.

FP pattern: instead of reading module state at various points,
create config once and thread it through the call chain.

Model is always the global COMPASS_MODEL. Config only carries think_level.
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from compass.llm.providers import ThinkLevel


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Immutable execution context.

    Created once from user overrides, then threaded through:
    execute_request -> call_actor -> oracle.ask
                   -> execute_action -> oracle.ask
                   -> critic_evaluate -> oracle.ask
                   -> generate_answer -> oracle.ask

    Model is always COMPASS_MODEL (set via /think or --model).
    This config only carries the think level override.
    """
    think_level: Optional["ThinkLevel"] = None
    max_consecutive_failures: int = 3

    @classmethod
    def from_overrides(cls) -> "ExecutionConfig":
        """Create config from current command overrides."""
        from compass.cli.commands import get_think_level_override
        from compass.llm.providers import ThinkLevel

        think_level = None
        level_str = get_think_level_override()
        if level_str:
            level_map = {
                "off": ThinkLevel.OFF,
                "low": ThinkLevel.LOW,
                "medium": ThinkLevel.MEDIUM,
                "high": ThinkLevel.HIGH,
            }
            think_level = level_map.get(level_str)

        # Support environment variable override for max_consecutive_failures
        import os
        max_failures = os.environ.get("COMPASS_MAX_CONSECUTIVE_FAILURES")
        if max_failures is not None:
            try:
                max_consecutive_failures = int(max_failures)
            except ValueError:
                # If conversion fails, keep default value of 3
                max_consecutive_failures = 3
        else:
            max_consecutive_failures = 3

        return cls(think_level=think_level, max_consecutive_failures=max_consecutive_failures)

    def with_think_level(self, level: "ThinkLevel") -> "ExecutionConfig":
        """Return new config with different think level."""
        return ExecutionConfig(think_level=level)
