"""Plan generator types."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from compass.generators._types import (
    DomainSection,
    Err,
    FileSpec,
    GenerationContext,
    GenerationReport,
    Ok,
    Result,
)


# ============================================================================
# Plan types
# ============================================================================


@dataclass(frozen=True)
class Step:
    """One step in a generation plan.

    Each step describes an artifact to produce. Steps execute in
    topological order respecting depends_on.
    """

    description: str
    artifact_type: str
    prompt: str | None = None
    depends_on: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlanSpec:
    """Complete plan specification from the model.

    The model writes a Python constructor expression::

        PlanSpec(
            goal="Build a data analysis toolkit",
            reasoning="Need notebook + CLI tool",
            steps=(
                Step(description="Create analysis notebook", artifact_type="notebook", prompt=None),
                Step(description="Create CLI tool", artifact_type="cli_tool", prompt=None, depends_on=(0,)),
            ),
        )

        # === index=0 ===
        Create a Jupyter notebook exploring pandas data analysis
        # === end ===

        # === index=1 ===
        Create a CLI tool that runs the analysis from the notebook
        # === end ===

    ``parse_typed_response`` fills each Step.prompt from the content blocks.

    A plan is a DAG of Steps. The executive validator processes them
    in topological order, resolving each artifact_type to a generator
    (or meta-generating one on the fly).
    """

    goal: str
    reasoning: str
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class StepResult:
    """Outcome of executing a single step."""

    step: Step
    step_index: int
    generator_used: str = "unknown"
    meta_generated: bool = False
    output_path: Optional[str] = None
    error: Optional[str] = None
    success: bool = False


@dataclass(frozen=True)
class ExecutedPlan:
    """A PlanSpec paired with execution results."""

    spec: PlanSpec
    results: tuple[StepResult, ...]


# ============================================================================
# Runtime config
# ============================================================================


@dataclass(frozen=True)
class PlanConfig:
    """Runtime configuration for the plan generation loop."""

    output_path: Path = Path("plans/neo_plan.json")
    max_rounds: int = 3
    max_fixes: int = 3
    model_id: str = ""
    verbose: bool = False
    dry_run: bool = False
    prompt: Optional[str] = None
    ask_fn: Optional[object] = None


# ============================================================================
# Known generator types
# ============================================================================

KNOWN_ARTIFACT_TYPES = frozenset({
    "notebook",
    "cli_tool",
    "test_suite",
    "python_module",
    "python_file",
    "config",
    "library",
})


# ============================================================================
# Structural validators
# ============================================================================


def _validate_step(raw: dict, index: int, total: int) -> Result[Step, str]:
    """Validate a single raw step dict."""
    errors: list[str] = []

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append(f"steps[{index}].description: must be a non-empty string")

    artifact_type = raw.get("artifact_type")
    if not isinstance(artifact_type, str) or not artifact_type.strip():
        errors.append(f"steps[{index}].artifact_type: must be a non-empty string")

    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append(f"steps[{index}].prompt: must be a non-empty string")

    depends_on_raw = raw.get("depends_on", [])
    if not isinstance(depends_on_raw, list):
        errors.append(f"steps[{index}].depends_on: must be a list")
        depends_on_raw = []

    depends_on: list[int] = []
    for j, dep in enumerate(depends_on_raw):
        if not isinstance(dep, int):
            errors.append(f"steps[{index}].depends_on[{j}]: must be an integer")
            continue
        if dep < 0 or dep >= total:
            errors.append(
                f"steps[{index}].depends_on[{j}]: index {dep} out of range [0, {total})"
            )
            continue
        if dep == index:
            errors.append(
                f"steps[{index}].depends_on[{j}]: self-reference not allowed"
            )
            continue
        depends_on.append(dep)

    if errors:
        return Err("; ".join(errors))

    return Ok(Step(
        description=description.strip(),
        artifact_type=artifact_type.strip(),
        prompt=prompt.strip(),
        depends_on=tuple(depends_on),
    ))


def _check_acyclic(steps: tuple[Step, ...]) -> Result[tuple[int, ...], str]:
    """Topological sort of step dependencies. Returns order or Err if cyclic."""
    n = len(steps)
    in_degree = [0] * n
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}

    for i, step in enumerate(steps):
        for dep in step.depends_on:
            adjacency[dep].append(i)
            in_degree[i] += 1

    queue: deque[int] = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != n:
        remaining = set(range(n)) - set(order)
        return Err(
            f"dependency cycle detected involving steps: {sorted(remaining)}"
        )

    return Ok(tuple(order))


def validate_spec(raw: dict) -> Result[PlanSpec, str]:
    """Validate a raw JSON dict into a PlanSpec. [STRUCTURAL]"""
    if not isinstance(raw, dict):
        return Err(f"expected dict, got {type(raw).__name__}")

    goal = raw.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        return Err("goal must be a non-empty string")

    reasoning = raw.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return Err("reasoning must be a non-empty string")

    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return Err("steps must be a non-empty list")

    errors: list[str] = []
    validated_steps: list[Step] = []
    total = len(raw_steps)

    for i, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            errors.append(f"steps[{i}]: expected dict")
            continue
        match _validate_step(raw_step, i, total):
            case Err(e):
                errors.append(e)
            case Ok(step):
                validated_steps.append(step)

    if errors:
        return Err("; ".join(errors))

    steps = tuple(validated_steps)

    # Check acyclic
    match _check_acyclic(steps):
        case Err(e):
            return Err(e)
        case Ok(_):
            pass

    return Ok(PlanSpec(
        goal=goal.strip(),
        reasoning=reasoning.strip(),
        steps=steps,
    ))


def validate_artifact_types(spec: PlanSpec) -> Result[list[str], str]:
    """Semantic validation: check artifact_types are recognized. [SEMANTIC]

    Warns on unknown types but does NOT fail -- unknown types will be
    meta-generated at execution time.
    """
    warnings: list[str] = []
    for i, step in enumerate(spec.steps):
        normalized = step.artifact_type.lower().replace("-", "_").replace(" ", "_")
        if normalized not in KNOWN_ARTIFACT_TYPES:
            warnings.append(
                f"steps[{i}].artifact_type '{step.artifact_type}' is not a "
                f"known generator -- will attempt meta-generation"
            )
    return Ok(warnings)


def topological_order(steps: tuple[Step, ...]) -> tuple[int, ...]:
    """Return topological order of steps. Assumes acyclic (already validated)."""
    match _check_acyclic(steps):
        case Ok(order):
            return order
        case Err(_):
            return tuple(range(len(steps)))


# ============================================================================
# Versioned output paths
# ============================================================================

_VERSION_SUFFIX_RE = re.compile(r"_v(\d+)$")


def detect_version(path: Path) -> int:
    """Extract version from _vN filename suffix."""
    m = _VERSION_SUFFIX_RE.search(path.stem)
    return int(m.group(1)) if m else 0


def versioned_path(base: Path, version: int) -> Path:
    """Return path with _vN suffix."""
    stem = base.stem
    m = _VERSION_SUFFIX_RE.search(stem)
    if m:
        stem = stem[:m.start()]
    return base.with_name(f"{stem}_v{version}{base.suffix}")


# ============================================================================
# Instance validator (for parse_typed_response path)
# ============================================================================


def validate_spec_instance(spec: PlanSpec) -> Result[PlanSpec, str]:
    """Validate a PlanSpec instance (from parse_typed_response). [STRUCTURAL]

    Mirrors validate_spec but operates on the already-constructed dataclass
    rather than a raw dict.
    """
    errors: list[str] = []
    if not spec.goal:
        errors.append("goal must be non-empty")
    if not spec.reasoning:
        errors.append("reasoning must be non-empty")
    if not spec.steps:
        errors.append("steps must be non-empty")
    for i, s in enumerate(spec.steps):
        if not s.description:
            errors.append(f"steps[{i}].description: must be non-empty")
        if not s.artifact_type:
            errors.append(f"steps[{i}].artifact_type: must be non-empty")
        if not s.prompt:
            errors.append(f"steps[{i}].prompt: must be non-empty")
        for dep in s.depends_on:
            if dep < 0 or dep >= len(spec.steps):
                errors.append(f"steps[{i}].depends_on: index {dep} out of range")
            if dep == i:
                errors.append(f"steps[{i}].depends_on: self-reference not allowed")
    if errors:
        return Err("; ".join(errors))
    return Ok(spec)
