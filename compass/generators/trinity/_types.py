"""Trinity generator types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from compass.generators._types import Err, Ok, Result


# ============================================================================
# Core types
# ============================================================================


@dataclass(frozen=True)
class Step:
    """A single artifact application step in the plan."""

    step_id: str
    description: str
    artifact_type: str       # built-in: 'read_file' | 'inline_python' | 'shell' | 'auto' | 'vision'
                             # dynamic: any name matching artifacts/<name>.py
    artifact_ref: str | None = None  # file path for read_file/auto/vision, command for shell, omit for inline_python
    inputs: dict = field(default_factory=dict)  # parameters passed to the artifact
    expected_fact: str = ""  # name of the fact this step produces
    extraction_expr: str = "result"  # Python expression to extract fact from locals/result
    depends_on: tuple[str, ...] = ()

    def validate(self, index: int) -> list[str]:
        """Return validation errors for this step. Subtypes override."""
        errors: list[str] = []
        if not self.step_id:
            errors.append(f"steps[{index}].step_id: must be non-empty")
        if not self.description:
            errors.append(f"steps[{index}].description: must be non-empty")
        if not self.artifact_type or not self.artifact_type.strip():
            errors.append(f"steps[{index}].artifact_type: must be non-empty")
        if not self.expected_fact:
            errors.append(f"steps[{index}].expected_fact: must be non-empty")
        # Base Step (unresolved / unknown type): artifact_ref optional
        return errors


# -- Step subtypes for singledispatch routing --------------------------------


@dataclass(frozen=True)
class InlinePythonStep(Step):
    """Step that executes inline Python code."""
    artifact_type: str = "inline_python"

    def validate(self, index: int) -> list[str]:
        errors = super().validate(index)
        if not self.artifact_ref:
            errors.append(
                f"steps[{index}].artifact_ref: must be non-empty "
                f"(write content after a ### {self.step_id} ### banner)"
            )
        return errors


@dataclass(frozen=True)
class ShellStep(Step):
    """Step that executes a shell command."""
    artifact_type: str = "shell"

    def validate(self, index: int) -> list[str]:
        errors = super().validate(index)
        if not self.artifact_ref:
            errors.append(f"steps[{index}].artifact_ref: must be non-empty")
        return errors


@dataclass(frozen=True)
class ReadFileStep(Step):
    """Step that reads a file."""
    artifact_type: str = "read_file"

    def validate(self, index: int) -> list[str]:
        errors = super().validate(index)
        if not self.artifact_ref:
            errors.append(f"steps[{index}].artifact_ref: must be non-empty")
        return errors


@dataclass(frozen=True)
class VisionStep(Step):
    """Step that sends an image to a vision model."""
    artifact_type: str = "vision"

    def validate(self, index: int) -> list[str]:
        errors = super().validate(index)
        if not self.artifact_ref:
            errors.append(f"steps[{index}].artifact_ref: must be non-empty")
        return errors


@dataclass(frozen=True)
class WriteFileStep(Step):
    """Step that writes banner content to a file."""
    artifact_type: str = "write_file"

    def validate(self, index: int) -> list[str]:
        errors = super().validate(index)
        if not self.artifact_ref and "content" not in self.inputs:
            errors.append(
                f"steps[{index}]: write_file needs either a ### {self.step_id} ### "
                f"banner or a 'content' input (e.g. {{\"content\": {{\"$fact\": \"name\"}}}})"
            )
        return errors


@dataclass(frozen=True)
class ProgrammerStep(Step):
    """Step that invokes the Programmer NFA for complex code generation."""
    artifact_type: str = "programmer"

    def validate(self, index: int) -> list[str]:
        errors = super().validate(index)
        if "problem" not in self.inputs and not self.artifact_ref:
            errors.append(
                f"steps[{index}]: programmer requires either a 'problem' input "
                f"or artifact_ref containing the problem statement"
            )
        return errors


@dataclass(frozen=True)
class DynamicStep(Step):
    """Step resolved to artifacts/{artifact_type}.py at promotion time."""
    module_path: str = ""


@dataclass(frozen=True)
class DiscoveredStep(Step):
    """Step resolved to a discovered workspace artifact at promotion time."""
    module_path: str = ""
    entry_point: str = "run"
    kind: str = "python_module"


_STEP_TYPE_MAP: dict[str, type] = {
    "inline_python": InlinePythonStep,
    "shell": ShellStep,
    "read_file": ReadFileStep,
    "write_file": WriteFileStep,
    "vision": VisionStep,
    "programmer": ProgrammerStep,
    "auto": Step,
}


def _resolve_artifact_ref(
    ref: str | None,
    artifacts: list["DiscoveredArtifact"],
) -> "DiscoveredArtifact | None":
    """Match a step's artifact_ref against discovered artifacts.

    Multi-pass: exact path, with .py suffix, by filename, dotted-to-slash.
    """
    if not ref:
        return None

    from pathlib import Path as P

    for a in artifacts:
        if a.path == ref:
            return a
    for a in artifacts:
        if a.path == ref + ".py":
            return a

    ref_name = P(ref).name
    for a in artifacts:
        if P(a.path).name == ref_name:
            return a
        if P(a.path).name == ref_name + ".py":
            return a

    ref_as_path = ref.replace(".", "/") + ".py"
    for a in artifacts:
        if a.path == ref_as_path:
            return a

    ref_as_pkg = ref.replace(".", "/")
    for a in artifacts:
        if a.path.startswith(ref_as_pkg):
            return a

    return None


def _promote_base_fields(step: Step) -> dict:
    """Extract base Step fields as a dict for subtype construction."""
    return dict(
        step_id=step.step_id,
        description=step.description,
        artifact_type=step.artifact_type,
        artifact_ref=step.artifact_ref,
        inputs=step.inputs,
        expected_fact=step.expected_fact,
        extraction_expr=step.extraction_expr,
        depends_on=step.depends_on,
    )


def promote_step(
    step: Step,
    artifacts: list["DiscoveredArtifact"] = (),
    workspace: "Path | None" = None,
) -> Step:
    """Promote a base Step to the right subtype based on artifact_type.

    Built-in types (inline_python, shell, etc.) are promoted from
    _STEP_TYPE_MAP. For auto types, the artifact_ref is resolved
    against discovered artifacts. For unknown types, we check for
    artifacts/{type}.py in the workspace.

    If already the right subtype, returns as-is.
    """
    # Already a resolved subtype?
    if isinstance(step, (DynamicStep, DiscoveredStep)):
        return step

    at = step.artifact_type

    # Built-in types (not auto)
    if at != "auto":
        cls = _STEP_TYPE_MAP.get(at)
        if cls is not None:
            return step if isinstance(step, cls) else cls(**_promote_base_fields(step))

        # Unknown type -- check for artifacts/{type}.py
        if workspace is not None:
            from pathlib import Path as P
            module_path = workspace / "artifacts" / f"{at}.py"
            if module_path.exists():
                return DynamicStep(
                    **_promote_base_fields(step),
                    module_path=str(module_path),
                )

        # No module found but has code in artifact_ref -- treat as inline_python
        if step.artifact_ref and step.artifact_ref.strip():
            return InlinePythonStep(**_promote_base_fields(step))

        # Truly unresolved -- stays as base Step
        return step

    # auto: resolve against discovered artifacts
    artifact = _resolve_artifact_ref(step.artifact_ref, artifacts)
    if artifact is not None:
        return DiscoveredStep(
            **_promote_base_fields(step),
            module_path=artifact.path,
            entry_point=artifact.entry_point,
            kind=artifact.kind,
        )

    # No discovered artifact -- stays as base Step
    return step


def promote_spec(
    spec: "Spec",
    artifacts: list["DiscoveredArtifact"] = (),
    workspace: "Path | None" = None,
) -> "Spec":
    """Promote all steps in a Spec to their proper subtypes."""
    promoted = tuple(promote_step(s, artifacts, workspace) for s in spec.steps)
    if promoted == spec.steps:
        return spec
    return replace(spec, steps=promoted)


@dataclass(frozen=True)
class Spec:
    """A plan of artifact applications to answer a question.

    The plan is a DAG of steps. Each step applies an artifact with
    inputs (possibly including facts from prior steps) and extracts
    a named fact. The synthesis field describes how to combine all
    facts into a final answer.

    Response format: Spec(...) constructor, then ### step_id ### banners
    for inline_python and write_file content. Write ONLY the constructor + banners.

    Example 1 (read + compute + shell):

    Spec(
        question="Read README.md, count its words, and post as a PR comment",
        steps=(
            Step(
                step_id="s1",
                description="read the target file",
                artifact_type="read_file",
                artifact_ref="README.md",
                inputs={},
                expected_fact="file_content",
                extraction_expr="result",
            ),
            Step(
                step_id="s2",
                description="count words in the document",
                artifact_type="inline_python",
                inputs={"content": {"$fact": "file_content"}},
                expected_fact="word_count",
                extraction_expr="result",
                depends_on=("s1",),
            ),
            Step(
                step_id="s3",
                description="post word count as PR comment",
                artifact_type="shell",
                artifact_ref='echo "$word_count words" | gh pr comment 1 --body-file -',
                inputs={"word_count": {"$fact": "word_count"}},
                expected_fact="comment_url",
                extraction_expr="result",
                depends_on=("s2",),
            ),
        ),
        synthesis="Report the word count and the comment URL.",
    )

    ### s2 ###
    lines = content.strip().split('\\n')
    words = sum(len(line.split()) for line in lines)
    result = str(words)

    Example 2 (inline_python chain -- passing data between steps):

    Spec(
        question="Define a helper class, build an instance, and summarize it",
        steps=(
            Step(
                step_id="s1",
                description="define a Point class and a factory function",
                artifact_type="inline_python",
                inputs={},
                expected_fact="point_tools",
                extraction_expr="result",
            ),
            Step(
                step_id="s2",
                description="build a point using the factory from s1",
                artifact_type="inline_python",
                inputs={"tools": {"$fact": "point_tools"}},
                expected_fact="my_point",
                extraction_expr="result",
                depends_on=("s1",),
            ),
            Step(
                step_id="s3",
                description="format the point as a string",
                artifact_type="inline_python",
                inputs={"pt": {"$fact": "my_point"}},
                expected_fact="summary",
                extraction_expr="result",
                depends_on=("s2",),
            ),
        ),
        synthesis="Show the point summary.",
    )

    ### s1 ###
    class Point:
        def __init__(self, x, y):
            self.x, self.y = x, y

    def make_point(x, y):
        return Point(x, y)

    result = {"Point": Point, "make_point": make_point}

    ### s2 ###
    # tools is injected from s1's fact via inputs
    make_point = tools["make_point"]
    result = make_point(3, 4)

    ### s3 ###
    # pt is injected from s2's fact via inputs
    result = f"Point({pt.x}, {pt.y})"

    Example 3 (programmer -- delegate code generation):

    Spec(
        question="Build a taskqueue package with add/pop/list_pending",
        steps=(
            Step(
                step_id="s1",
                description="generate the taskqueue package with tests",
                artifact_type="programmer",
                inputs={"problem": "Build a Python package called taskqueue with a Task dataclass and a TaskQueue class supporting add, pop (highest priority), and list_pending. Include tests."},
                expected_fact="code_result",
            ),
        ),
        synthesis="Report what was generated.",
    )
    """

    question: str
    steps: tuple[Step, ...]
    synthesis: str


@dataclass(frozen=True)
class Fact:
    """A structured fact extracted from an artifact execution."""

    step_id: str
    name: str
    value: str          # serialized value (always raw / canonical data)
    fact_type: str      # 'numeric' | 'text' | 'boolean' | 'json' | 'error'


@dataclass(frozen=True)
class FileFact(Fact):
    """Fact from reading a file. value = raw file content (no line numbers)."""

    path: str = ""
    line_offset: int = 0   # 0-based start line in original file (for display numbering)


@dataclass(frozen=True)
class ErrorFact(Fact):
    """Fact from a failed step."""

    fact_type: str = "error"


@dataclass(frozen=True)
class ExecutionResult:
    """The final result: collected facts and synthesized answer."""

    question: str
    facts: tuple[Fact, ...]
    answer: str
    success: bool


# ============================================================================
# Discovered artifact descriptor
# ============================================================================


@dataclass(frozen=True)
class DiscoveredArtifact:
    """An artifact discovered by scanning the workspace.

    Carries enough metadata for the model to plan against it
    and for the runtime to invoke it without hard-coded dispatch.
    """

    path: str                    # relative path from workspace root
    kind: str                    # 'python_module' | 'python_script' | 'notebook' | 'shell'
    entry_point: str             # function name or 'script' or 'notebook'
    parameters: tuple[str, ...]  # parameter names from signature inspection
    doc: str                     # first line of docstring or ''


# ============================================================================
# Patch types -- for ouroboros targeted edits
# ============================================================================


@dataclass(frozen=True)
class StepPatch:
    """A targeted edit to a single step.

    step_id identifies which step to patch.
    fields is a dict of field_name -> new_value for that step.
    Only the fields present in the dict are replaced; others are kept.
    """

    step_id: str
    fields: dict  # e.g. {"artifact_ref": "new code", "extraction_expr": "x"}


@dataclass(frozen=True)
class SpecPatch:
    """A partial update to a Spec.

    Ouroboros returns this instead of the full Spec. Each StepPatch
    targets a single step with field-level edits. Steps can also be
    added or removed.

    SpecPatch(
        step_patches=(
            StepPatch(step_id="step_1", fields={"artifact_ref": "new code"}),
        ),
    )
    """

    question: Optional[str] = None
    synthesis: Optional[str] = None
    step_patches: tuple[StepPatch, ...] = ()
    add_steps: tuple[Step, ...] = ()
    remove_steps: tuple[str, ...] = ()


# ============================================================================
# Valid artifact types
# ============================================================================

_VALID_ARTIFACT_TYPES = frozenset({
    "read_file",
    "write_file",
    "inline_python",
    "shell",
    "auto",
    "vision",
    "programmer",
})


# ============================================================================
# Structural validation -- V1
# ============================================================================


def _validate_step(raw: dict, index: int) -> Result[Step, str]:
    """Validate a single step dict into a Step. [STRUCTURAL]"""
    errors: list[str] = []

    step_id = raw.get("step_id")
    if not isinstance(step_id, str) or not step_id.strip():
        errors.append(f"steps[{index}].step_id: must be a non-empty string")
        step_id = f"step_{index}"

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append(f"steps[{index}].description: must be a non-empty string")
        description = ""

    artifact_type = raw.get("artifact_type")
    if not isinstance(artifact_type, str) or not artifact_type.strip():
        errors.append(f"steps[{index}].artifact_type: must be a non-empty string")
        artifact_type = "inline_python"

    artifact_ref = raw.get("artifact_ref")
    if artifact_type in ("inline_python", "write_file", "programmer") or artifact_type not in _VALID_ARTIFACT_TYPES:
        # artifact_ref can be None -- content comes from banners, or module from artifacts/
        if artifact_ref is not None and not isinstance(artifact_ref, str):
            errors.append(f"steps[{index}].artifact_ref: must be a string or None")
            artifact_ref = None
    else:
        if not isinstance(artifact_ref, str) or not artifact_ref.strip():
            errors.append(
                f"steps[{index}].artifact_ref: must be a non-empty string "
                f"for artifact_type '{artifact_type}'"
            )
            artifact_ref = ""

    inputs = raw.get("inputs", {})
    if not isinstance(inputs, dict):
        errors.append(f"steps[{index}].inputs: must be a dict")
        inputs = {}

    expected_fact = raw.get("expected_fact")
    if not isinstance(expected_fact, str) or not expected_fact.strip():
        errors.append(f"steps[{index}].expected_fact: must be a non-empty string")
        expected_fact = ""

    extraction_expr = raw.get("extraction_expr", "result")
    if not isinstance(extraction_expr, str) or not extraction_expr.strip():
        errors.append(f"steps[{index}].extraction_expr: must be a non-empty string")
        extraction_expr = "result"

    depends_on_raw = raw.get("depends_on", [])
    if not isinstance(depends_on_raw, list):
        errors.append(f"steps[{index}].depends_on: must be a list")
        depends_on_raw = []

    depends_on = tuple(
        d for d in depends_on_raw
        if isinstance(d, str) and d.strip()
    )

    if errors:
        return Err("; ".join(errors))

    cls = _STEP_TYPE_MAP.get(artifact_type, Step)
    return Ok(cls(
        step_id=step_id.strip(),
        description=description.strip(),
        artifact_type=artifact_type,
        artifact_ref=artifact_ref.strip() if artifact_type != "inline_python" else artifact_ref,
        inputs=inputs,
        expected_fact=expected_fact.strip(),
        extraction_expr=extraction_expr.strip(),
        depends_on=depends_on,
    ))


def validate_spec(raw: dict) -> Result[Spec, str]:
    """Validate a raw JSON dict into a Spec. [STRUCTURAL]

    Checks:
    - question is a non-empty string
    - steps is a non-empty list of valid Step dicts
    - synthesis is a non-empty string
    - step_ids are unique
    - depends_on references exist
    - no circular dependencies
    """
    if not isinstance(raw, dict):
        return Err(f"expected dict, got {type(raw).__name__}")

    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        return Err("question must be a non-empty string")

    synthesis = raw.get("synthesis", "Combine all facts to answer the question.")
    if not isinstance(synthesis, str) or not synthesis.strip():
        return Err("synthesis must be a non-empty string")

    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return Err("steps must be a non-empty list")

    errors: list[str] = []
    steps: list[Step] = []
    seen_ids: set[str] = set()

    for i, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            errors.append(f"steps[{i}]: expected dict")
            continue

        match _validate_step(raw_step, i):
            case Err(e):
                errors.append(e)
            case Ok(step):
                if step.step_id in seen_ids:
                    errors.append(f"steps[{i}]: duplicate step_id '{step.step_id}'")
                else:
                    seen_ids.add(step.step_id)
                    steps.append(step)

    if errors:
        return Err("; ".join(errors))

    # Validate dependency references
    all_ids = {s.step_id for s in steps}
    for step in steps:
        for dep in step.depends_on:
            if dep not in all_ids:
                errors.append(
                    f"step '{step.step_id}' depends on '{dep}' which does not exist"
                )

    # Check for circular dependencies
    if not errors:
        visited: set[str] = set()
        path: set[str] = set()
        dep_map = {s.step_id: s.depends_on for s in steps}

        def has_cycle(node: str) -> bool:
            if node in path:
                return True
            if node in visited:
                return False
            visited.add(node)
            path.add(node)
            for dep in dep_map.get(node, ()):
                if has_cycle(dep):
                    return True
            path.discard(node)
            return False

        for sid in all_ids:
            if has_cycle(sid):
                errors.append(f"circular dependency detected involving '{sid}'")
                break

    if errors:
        return Err("; ".join(errors))

    return Ok(Spec(
        question=question.strip(),
        steps=tuple(steps),
        synthesis=synthesis.strip(),
    ))


# ============================================================================
# Patch validation
# ============================================================================


def validate_spec_patch(raw: dict) -> Result[SpecPatch, str]:
    """Validate a raw JSON dict into a SpecPatch. [STRUCTURAL]"""
    if not isinstance(raw, dict):
        return Err(f"expected dict, got {type(raw).__name__}")

    errors: list[str] = []

    question = raw.get("question")
    if question is not None:
        if not isinstance(question, str) or not question.strip():
            errors.append("question: must be a non-empty string if provided")
            question = None
        else:
            question = question.strip()

    synthesis = raw.get("synthesis")
    if synthesis is not None:
        if not isinstance(synthesis, str) or not synthesis.strip():
            errors.append("synthesis: must be a non-empty string if provided")
            synthesis = None
        else:
            synthesis = synthesis.strip()

    # Step patches
    step_patches: list[StepPatch] = []
    raw_patches = raw.get("step_patches", [])
    if not isinstance(raw_patches, list):
        errors.append("step_patches: must be a list")
        raw_patches = []

    for i, rp in enumerate(raw_patches):
        if not isinstance(rp, dict):
            errors.append(f"step_patches[{i}]: expected dict")
            continue
        sid = rp.get("step_id")
        if not isinstance(sid, str) or not sid.strip():
            errors.append(f"step_patches[{i}].step_id: must be a non-empty string")
            continue
        fields = rp.get("fields")
        if not isinstance(fields, dict) or not fields:
            errors.append(f"step_patches[{i}].fields: must be a non-empty dict")
            continue
        step_patches.append(StepPatch(step_id=sid.strip(), fields=fields))

    # Add steps
    add_steps: list[Step] = []
    raw_add = raw.get("add_steps", [])
    if not isinstance(raw_add, list):
        errors.append("add_steps: must be a list")
        raw_add = []

    for i, ra in enumerate(raw_add):
        if not isinstance(ra, dict):
            errors.append(f"add_steps[{i}]: expected dict")
            continue
        match _validate_step(ra, i):
            case Err(e):
                errors.append(f"add_steps: {e}")
            case Ok(step):
                add_steps.append(step)

    # Remove steps
    remove_steps: list[str] = []
    raw_remove = raw.get("remove_steps", [])
    if not isinstance(raw_remove, list):
        errors.append("remove_steps: must be a list")
        raw_remove = []

    for i, rr in enumerate(raw_remove):
        if isinstance(rr, str) and rr.strip():
            remove_steps.append(rr.strip())
        else:
            errors.append(f"remove_steps[{i}]: must be a non-empty string")

    # Must have at least one change
    has_changes = (
        question is not None
        or synthesis is not None
        or step_patches
        or add_steps
        or remove_steps
    )
    if not has_changes and not errors:
        errors.append("patch must contain at least one change")

    if errors:
        return Err("; ".join(errors))

    return Ok(SpecPatch(
        question=question,
        synthesis=synthesis,
        step_patches=tuple(step_patches),
        add_steps=tuple(add_steps),
        remove_steps=tuple(remove_steps),
    ))


def apply_spec_patch(
    spec: Spec,
    patch: SpecPatch,
) -> Result[Spec, str]:
    """Apply a SpecPatch to a Spec."""
    errors: list[str] = []

    step_map: dict[str, Step] = {s.step_id: s for s in spec.steps}
    step_order: list[str] = [s.step_id for s in spec.steps]

    _STEP_FIELDS = frozenset({
        "description", "artifact_type", "artifact_ref",
        "inputs", "expected_fact", "extraction_expr", "depends_on",
    })

    for sp in patch.step_patches:
        # Coerce raw dicts to StepPatch (model sometimes emits dicts instead of constructors)
        if isinstance(sp, dict):
            sid = sp.get("step_id", "")
            flds = sp.get("fields", {})
            if not isinstance(sid, str) or not sid:
                errors.append(f"step_patches: dict missing valid 'step_id'")
                continue
            if not isinstance(flds, dict) or not flds:
                errors.append(f"step_patches['{sid}']: dict missing valid 'fields'")
                continue
            sp = StepPatch(step_id=sid, fields=flds)

        if sp.step_id not in step_map:
            errors.append(f"step_patches: step_id '{sp.step_id}' not found in spec")
            continue

        old_step = step_map[sp.step_id]
        kwargs: dict = {}
        for field_name, new_value in sp.fields.items():
            if field_name not in _STEP_FIELDS:
                errors.append(
                    f"step_patches['{sp.step_id}']: unknown field '{field_name}'"
                )
                continue
            if field_name == "depends_on":
                if isinstance(new_value, list):
                    new_value = tuple(new_value)
                elif not isinstance(new_value, tuple):
                    errors.append(
                        f"step_patches['{sp.step_id}'].depends_on: must be a list"
                    )
                    continue
            kwargs[field_name] = new_value

        if kwargs:
            step_map[sp.step_id] = replace(old_step, **kwargs)

    for sid in patch.remove_steps:
        if sid not in step_map:
            errors.append(f"remove_steps: step_id '{sid}' not found in spec")
            continue
        del step_map[sid]
        step_order = [s for s in step_order if s != sid]

    for step in patch.add_steps:
        # Coerce raw dicts to Step (model sometimes emits dicts instead of constructors)
        if isinstance(step, dict):
            validated = _validate_step(step, len(step_order))
            if isinstance(validated, Err):
                errors.append(f"add_steps: {validated.error}")
                continue
            step = validated.value
        if step.step_id in step_map:
            errors.append(f"add_steps: step_id '{step.step_id}' already exists")
            continue
        step_map[step.step_id] = step
        step_order.append(step.step_id)

    if errors:
        return Err("; ".join(errors))

    new_steps = tuple(step_map[sid] for sid in step_order if sid in step_map)
    new_question = patch.question if patch.question is not None else spec.question
    new_synthesis = patch.synthesis if patch.synthesis is not None else spec.synthesis

    return Ok(Spec(
        question=new_question,
        steps=new_steps,
        synthesis=new_synthesis,
    ))


# ============================================================================
# Serialization helpers
# ============================================================================


def serialize_spec(spec: Spec) -> str:
    """Serialize a Spec to a JSON string."""
    def step_to_dict(s: Step) -> dict:
        d: dict = {
            "step_id": s.step_id,
            "description": s.description,
            "artifact_type": s.artifact_type,
            "artifact_ref": s.artifact_ref,
            "inputs": s.inputs,
            "expected_fact": s.expected_fact,
            "extraction_expr": s.extraction_expr,
        }
        if s.depends_on:
            d["depends_on"] = list(s.depends_on)
        return d

    return json.dumps({
        "question": spec.question,
        "steps": [step_to_dict(s) for s in spec.steps],
        "synthesis": spec.synthesis,
    }, indent=2)


def serialize_result(result: ExecutionResult) -> str:
    """Serialize an ExecutionResult to JSON string."""
    return json.dumps({
        "question": result.question,
        "facts": [
            {
                "step_id": f.step_id,
                "name": f.name,
                "value": f.value,
                "fact_type": f.fact_type,
            }
            for f in result.facts
        ],
        "answer": result.answer,
        "success": result.success,
    }, indent=2)


# ============================================================================
# Instance validation (for parse_typed_response path)
# ============================================================================


def validate_spec_instance(spec: Spec) -> Result[Spec, str]:
    """Validate a Spec instance after banner code attachment. [STRUCTURAL]

    Checks that required fields are populated.
    Returns Ok(Spec) or Err(description).
    """
    errors: list[str] = []
    if not spec.question:
        errors.append("question must be non-empty")
    if not spec.synthesis:
        errors.append("synthesis must be non-empty")
    if not spec.steps:
        errors.append("steps must be non-empty")
    seen_ids: set[str] = set()
    for i, s in enumerate(spec.steps):
        errors.extend(s.validate(i))
        if s.step_id and s.step_id in seen_ids:
            errors.append(f"steps[{i}]: duplicate step_id '{s.step_id}'")
        seen_ids.add(s.step_id or "")
    # Check dependency references
    all_ids = {s.step_id for s in spec.steps if s.step_id}
    for s in spec.steps:
        for dep in s.depends_on:
            if dep not in all_ids:
                errors.append(f"step '{s.step_id}' depends on '{dep}' which does not exist")
    if errors:
        return Err("; ".join(errors))
    return Ok(spec)
