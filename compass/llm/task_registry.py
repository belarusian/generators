"""Task registry - validates task names (catches typos).

Task strings are telemetry labels only -- they don't route to providers.
The registry catches typos: unregistered task= strings raise UnknownTaskError.
"""

from typing import Set


class UnknownTaskError(KeyError):
    """Raised when a task name is not registered."""


_registry: Set[str] = set()


def register(task: str) -> str:
    """Register a task name."""
    if task in _registry:
        raise ValueError(f"Task '{task}' is already registered")
    _registry.add(task)
    return task


def validate_task(task: str) -> None:
    """Validate a task name is registered."""
    if task not in _registry:
        raise UnknownTaskError(
            f"Task '{task}' not registered. Add it to compass/llm/task_registry.py"
        )


def registered_tasks() -> Set[str]:
    """Return the set of registered task names."""
    return set(_registry)


# Backward compat: old code calls get_ladder(task) which returned a ladder name.
# Now it just validates and returns the task name itself.
def get_ladder(task: str) -> str:
    """Validate task name. Returns task name (no routing)."""
    validate_task(task)
    return task


# -- Schema (structured JSON/Python output) --
register("ask")
register("actor")
register("actor:learning")
register("actor:progress-judge")
register("ask_for_field")
register("scribe-review")
register("scribe-continue")
register("scribe-deliver")
register("validator")
register("answerer")
register("cli:validate")
register("reflect")
register("travel:interpret-field")
register("travel:search-params")
register("telemetry-analysis")
register("test")

# -- Reasoning (evaluation, critique) --
register("critic")
register("critic-review")
register("critic-evaluate")
register("oracle-dream")
register("parent-critic")

# -- Coding (generation, editing) --
register("programmer-implement")
register("programmer-amend")
register("programmer-understand")
register("programmer-design")
register("shell-builder")

# -- Editing (structured file modifications) --
register("file-editor")
register("editor")

# -- Poetry (creative output) --
register("divine_essence")
register("divine_instruction")
register("divine_lodging")
register("plan_journey")
register("speak_greeting")
register("speak_farewell")
register("speak_no_path")
register("oracle_wisdom")
register("travel:generate-question")

# -- Extended consultation / summarization --
register("oracle")
register("compaction")

# -- Trinity (reflection, evolution) --
register("trinity_respond")
register("trinity_tone")
register("trinity_evolve")
register("trinity_reflect")
