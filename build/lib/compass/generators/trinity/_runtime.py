"""IO boundaries for Trinity.

G_trinity   : Context -> Result[raw_dict]        (model invocation)
V2_trinity  : Spec -> Result[ExecutionResult]     (execute plan, collect facts)
G'_trinity  : (Spec, error, ctx) -> Spec | None   (ouroboros fix via patching)
IO_trinity  : (Spec, ExecutionResult, ...) -> Result[Path]  (emit)

Artifact dispatch is NOT hard-coded. Instead:
  1. discover_artifacts() scans the workspace for Python files, inspects
     their signatures (run(), main(), or top-level script), and builds
     DiscoveredArtifact descriptors.
  2. _resolve_artifact() maps a step's artifact_ref to a discovered
     artifact (or treats it as inline code / shell command).
  3. _execute_artifact() uses the DiscoveredArtifact's metadata to
     invoke the artifact, mapping step inputs to function parameters
     by inspecting the signature at call time.

Ouroboros patching:
  G'_trinity returns a SpecPatch (targeted step-level edits) instead of
  the full Spec. The patch is applied via apply_spec_patch(). This keeps
  the ouroboros output small -- the model only returns what changed.
  Falls back to full Spec replacement if patching fails.

  The ctx parameter is used to enrich the ouroboros system prompt with
  domain context (discovered artifacts, plan principles) so the fix
  model can make informed surgical patches.

Validation pipeline (cheapest first):
  1. Structural: validate_spec (in _types.py) -- pure dict -> Spec
  2. Semantic: validate inline Python syntax, check artifact refs
  3. Executive: actually run the steps and collect facts
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Optional

from compass.generators._types import (
    AskFn,
    DomainSection,
    Err,
    GenerationContext,
    Ok,
    Result,
)
from compass.generators._invoke import (
    build_system_prompt,
    build_user_message,
    resolve_ask_fn,
)
from compass.core.python_schema import parse_typed_response, parse_response_with_files

from compass.generators.trinity._types import (
    DiscoveredArtifact,
    ErrorFact,
    ExecutionResult,
    Fact,
    Spec,
    SpecPatch,
    Step,
    StepPatch,
    apply_spec_patch,
    promote_spec,
    promote_step,
    serialize_spec,
    serialize_result,
    validate_spec,
    validate_spec_patch,
)
from compass.generators.trinity.fact_dispatch import display_fact, resolve_fact
from compass.generators.trinity.step_dispatch import StepContext, execute_step

logger = logging.getLogger(__name__)

_SPEC_TYPE_SOURCE = inspect.getsource(Step) + "\n\n" + inspect.getsource(Spec)


# ---------------------------------------------------------------------------
# Artifact discovery -- scan workspace for runnable Python programs
# ---------------------------------------------------------------------------


def _inspect_python_file(filepath: Path, workspace: Path) -> DiscoveredArtifact | None:
    """Inspect a Python file and extract its entry point and signature.

    Looks for (in order of preference):
      1. A run() function -- the compass convention
      2. A main() function
      3. Top-level script (no entry function, but has executable code)

    Returns a DiscoveredArtifact or None if the file is not runnable.
    """
    try:
        source = filepath.read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    rel_path = str(filepath.relative_to(workspace))

    # Extract module-level docstring
    doc = ""
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        doc = tree.body[0].value.value.strip().split("\n")[0]

    # Find function definitions
    functions: dict[str, ast.FunctionDef] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node

    # Only discover files with explicit entry points (run() or main()).
    # Files with just top-level executable code are not artifacts --
    # they're framework modules, scripts, etc. The shape matters:
    # a real artifact has a callable entry point with inspectable params.
    for entry_name in ("run", "main"):
        if entry_name in functions:
            func_node = functions[entry_name]
            params = _extract_params_from_ast(func_node)
            return DiscoveredArtifact(
                path=rel_path,
                kind="python_module",
                entry_point=entry_name,
                parameters=tuple(params),
                doc=doc,
            )

    return None


def _extract_params_from_ast(func_node: ast.FunctionDef) -> list[str]:
    """Extract parameter names from an AST function definition."""
    params: list[str] = []
    args = func_node.args

    for arg in args.args:
        name = arg.arg
        if name not in ("self", "cls"):
            params.append(name)

    for arg in args.kwonlyargs:
        params.append(arg.arg)

    return params


def discover_artifacts(workspace: Path | None = None) -> list[DiscoveredArtifact]:
    """Scan workspace for runnable Python artifacts.

    Discovers files by shape (has run() or main() entry point),
    not by directory convention. Framework scripts without entry
    points are ignored by _inspect_python_file.
    """
    if workspace is None:
        workspace = Path(".")

    workspace = workspace.resolve()
    artifacts: list[DiscoveredArtifact] = []

    for py_file in sorted(workspace.rglob("*.py")):
        parts = py_file.relative_to(workspace).parts
        if any(p.startswith(".") or p == "__pycache__" for p in parts):
            continue

        name = py_file.name
        if name == "__init__.py":
            continue
        if name.startswith("test_") or name.endswith("_test.py"):
            continue

        artifact = _inspect_python_file(py_file, workspace)
        if artifact is not None:
            artifacts.append(artifact)

    return artifacts


def format_artifacts_for_context(artifacts: list[DiscoveredArtifact]) -> str:
    """Format discovered artifacts as a human/model-readable listing."""
    if not artifacts:
        return "No runnable artifacts discovered in workspace."

    lines = ["Discovered artifacts:"]
    for a in artifacts:
        param_str = ", ".join(a.parameters) if a.parameters else "(no parameters)"
        doc_str = f" -- {a.doc}" if a.doc else ""
        lines.append(
            f"  - {a.path} [{a.kind}] entry={a.entry_point}({param_str}){doc_str}"
        )

    return "\n".join(lines)




# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


def _topological_sort(steps: tuple[Step, ...]) -> list[Step]:
    """Sort steps in dependency order (topological sort)."""
    step_map = {s.step_id: s for s in steps}
    visited: set[str] = set()
    order: list[str] = []

    def visit(sid: str) -> None:
        if sid in visited:
            return
        visited.add(sid)
        step = step_map[sid]
        for dep in step.depends_on:
            if dep in step_map:
                visit(dep)
        order.append(sid)

    for s in steps:
        visit(s.step_id)

    return [step_map[sid] for sid in order]


# ---------------------------------------------------------------------------
# Fact resolution
# ---------------------------------------------------------------------------


def _resolve_inputs(
    inputs: dict,
    facts: dict[str, Fact],
) -> dict:
    """Resolve $fact references in inputs using collected facts."""
    resolved = {}
    for key, value in inputs.items():
        if isinstance(value, dict) and "$fact" in value:
            fact_name = value["$fact"]
            if fact_name in facts:
                resolved[key] = resolve_fact(facts[fact_name])
            else:
                resolved[key] = None
        else:
            resolved[key] = value
    return resolved


# ---------------------------------------------------------------------------
# Model invocation -- G_trinity
# ---------------------------------------------------------------------------


def _attach_banner_code(
    spec: Spec,
    sections: list[tuple[str, str]],
) -> Spec:
    """Attach banner content to steps by step_id.

    Only fills artifact_ref when the model left it empty (None).
    The prompt tells the model to omit artifact_ref for banner types
    (inline_python, write_file) and set it for path types (read_file,
    auto, shell, vision). So: empty = waiting for banner content.
    """
    from dataclasses import replace

    code_map = dict(sections)
    new_steps = []
    for step in spec.steps:
        if step.step_id in code_map and not step.artifact_ref:
            new_steps.append(replace(step, artifact_ref=code_map[step.step_id]))
        else:
            new_steps.append(step)
    return replace(spec, steps=tuple(new_steps))


def invoke_model(
    ctx: GenerationContext,
    model_id: str = "",
    ask_fn: AskFn | None = None,
) -> Result:
    """Call the model and return parsed Spec. G_trinity.

    Python-as-schema: model writes Spec constructor, then
    inline_python code after ### step_id ### banners.
    """
    fn = resolve_ask_fn(model_id, ask_fn)

    system = build_system_prompt(
        ctx,
        _SPEC_TYPE_SOURCE,
        contract_preamble=(
            "Respond as shown in the Spec docstring."
        ),
    )

    user = _build_user_message(ctx)

    logger.debug("--- G_trinity SYSTEM ---\n%s\n--- END SYSTEM ---", system)
    logger.debug("--- G_trinity USER ---\n%s\n--- END USER ---", user)

    match fn(system, user):
        case Err() as e:
            return e
        case Ok(raw_text):
            pass

    logger.debug("--- G_trinity RESPONSE ---\n%s\n--- END RESPONSE ---", raw_text)

    # Try banner format first (inline_python steps have ### step_id ### sections)
    try:
        spec, sections = parse_response_with_files(raw_text, Spec)
        spec = _attach_banner_code(spec, sections)
        return Ok(promote_spec(spec))
    except ValueError:
        pass

    # Fallback: no banners (all steps are auto/shell/vision, or code inline)
    try:
        spec = parse_typed_response(raw_text, Spec)
        return Ok(promote_spec(spec))
    except ValueError as e:
        return Err(str(e))


def _build_user_message(ctx: GenerationContext) -> str:
    """Build user message for Trinity."""
    primary = (
        ctx.user_prompt if ctx.user_prompt is not None else
        ctx.default_task if ctx.default_task is not None else
        "Answer a question by planning artifact applications."
    )
    parts = [primary]

    parts.extend([
        "",
        "Write a Spec(...) constructor followed by ### step_id ### banners for inline_python and write_file content.",
        "See the Spec docstring for the exact response format. No markdown fencing.",
        "",
        "Plan guidelines:",
        "- Each step should be self-contained with clear inputs and expected output.",
        "- For inline_python: omit artifact_ref. Write the code after a ### step_id ### banner.",
        "  The code should assign its result to a variable. extraction_expr names that variable.",
        "- For write_file: omit artifact_ref. Write the file content after a ### step_id ### banner.",
        "  inputs must include {\"path\": \"target/path\"} -- the banner content is written there.",
        "  Use this to persist reusable artifact modules to artifacts/.",
        "- For auto: artifact_ref is a file path to a discovered artifact.",
        "  Check the 'Discovered Artifacts' section for available files and their parameters.",
        "- For programmer: generates tested Python code.",
        "  inputs must include {\"problem\": \"what to build\"}.",
        "- For shell: artifact_ref is the shell command. inputs become env vars.",
        "- Steps execute in dependency order. Use depends_on for data dependencies.",
        "- In inputs, reference prior facts as {'$fact': 'fact_name'} to inject them.",
        "",
        "Prefer 'auto' for discovered artifacts. Use 'inline_python' for computation.",
        "Use 'programmer' for substantial code generation tasks.",
    ])

    if ctx.available_packages:
        parts.extend(["", f"Available packages: {ctx.available_packages}"])

    if ctx.feedback:
        parts.extend(["", "Your previous attempt had errors:", ""])
        for fb in ctx.feedback:
            parts.append(f"  {fb}")
        parts.extend(["", "Please fix these issues in your next attempt."])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Semantic validation
# ---------------------------------------------------------------------------


def validate_semantics(spec: Spec) -> Result[None, str]:
    """Semantic validation: check inline Python syntax, artifact refs.

    Pure (ast only for inline code). Cheaper than execution.
    """
    errors: list[str] = []

    for step in spec.steps:
        if step.artifact_type == "inline_python":
            try:
                ast.parse(step.artifact_ref)
            except SyntaxError as exc:
                loc = f" at line {exc.lineno}" if exc.lineno else ""
                errors.append(
                    f"step '{step.step_id}': SyntaxError in inline code{loc}: {exc.msg}"
                )

            try:
                ast.parse(step.extraction_expr, mode="eval")
            except SyntaxError:
                errors.append(
                    f"step '{step.step_id}': extraction_expr is not valid Python: "
                    f"{step.extraction_expr!r}"
                )

        elif step.artifact_type == "auto":
            try:
                ast.parse(step.extraction_expr, mode="eval")
            except SyntaxError:
                errors.append(
                    f"step '{step.step_id}': extraction_expr is not valid Python: "
                    f"{step.extraction_expr!r}"
                )

        elif step.artifact_type == "read_file":
            if not step.artifact_ref:
                errors.append(
                    f"step '{step.step_id}': read_file requires artifact_ref (file path)"
                )

        elif step.artifact_type == "shell":
            # Check that the step has inputs mapping to facts from depends_on.
            # The model often writes $VAR in the command but forgets the
            # {"VAR": {"$fact": "fact_name"}} input mapping.
            cmd = step.artifact_ref or ""
            input_keys = set(step.inputs.keys())
            dep_facts = set()
            for dep_id in step.depends_on:
                for other in spec.steps:
                    if other.step_id == dep_id and other.expected_fact:
                        dep_facts.add(other.expected_fact)

            # If the step depends on other steps but has no inputs,
            # it probably forgot the $fact mapping.
            if dep_facts and not input_keys:
                errors.append(
                    f"step '{step.step_id}': shell step depends on "
                    f"{step.depends_on} but has no inputs. "
                    f"Add inputs mapping to facts: "
                    + ", ".join(
                        f'"{f}": {{"$fact": "{f}"}}'
                        for f in sorted(dep_facts)
                    )
                )

    if errors:
        return Err("; ".join(errors))
    return Ok(None)


# ---------------------------------------------------------------------------
# Synthesis -- LLM combines facts into a final answer
# ---------------------------------------------------------------------------


def _synthesize_answer(
    spec: Spec,
    facts: list[Fact],
    model_id: str = "",
    ask_fn: AskFn | None = None,
) -> str:
    """Use the model to synthesize collected facts into a coherent answer.

    The spec.synthesis field is a natural language instruction describing
    how to combine the facts. The model reads the question, all facts,
    and the synthesis instruction, then produces the final answer.

    Falls back to mechanical concatenation if the model call fails.
    """
    fn = resolve_ask_fn(model_id, ask_fn)

    fact_listing = "\n\n".join(
        f"### {f.name} ({f.fact_type})\n{display_fact(f)}"
        for f in facts if f.fact_type != "error"
    )

    system = (
        "You are synthesizing research findings into a clear, direct answer. "
        "You will receive a question, collected facts, and a synthesis instruction. "
        "Produce the final answer only -- no preamble, no meta-commentary. "
    )

    user = (
        f"# Question\n{spec.question}\n\n"
        f"# Collected Facts\n{fact_listing}\n\n"
        f"# Synthesis Instruction\n{spec.synthesis}\n\n"
        f"Write the answer."
    )

    match fn(system, user):
        case Ok(answer):
            return answer.strip()
        case Err(e):
            logger.warning("Synthesis LLM call failed: %s; falling back to concatenation", e)
            fact_summary = "; ".join(f"{f.name}={f.value[:200]}" for f in facts)
            return f"{spec.synthesis}\n\nFacts: {fact_summary}"


# ---------------------------------------------------------------------------
# Dynamic artifact resolution -- unknown types
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Executive validation -- run the plan
# ---------------------------------------------------------------------------


def execute_plan(
    spec: Spec,
    workspace: Path | None = None,
    model_id: str = "",
    ask_fn: AskFn | None = None,
    on_step: Any = None,
) -> Result[ExecutionResult, str]:
    """Executive validation: run all steps, collect facts.

    on_step: optional callback(event, **kwargs) for progress display.
        Events: step_start(step, index, total), step_done(step, fact),
                step_error(step, error), synthesize().
    """
    if workspace is None:
        workspace = Path(".").resolve()
    else:
        workspace = workspace.resolve()

    artifacts = discover_artifacts(workspace)
    logger.info(
        "Discovered %d artifacts in %s",
        len(artifacts), workspace,
    )

    spec = promote_spec(spec, artifacts, workspace)
    sorted_steps = _topological_sort(spec.steps)
    total = len(sorted_steps)
    facts: dict[str, Fact] = {}
    all_facts: list[Fact] = []
    errors: list[str] = []

    for idx, step in enumerate(sorted_steps):
        # Lazy promotion: a prior step may have written new artifact files
        # (e.g. write_file persisting a module). Re-discover and promote
        # unresolved steps so they can find newly-created artifacts.
        if type(step) is Step:
            fresh = discover_artifacts(workspace)
            step = promote_step(step, fresh, workspace)

        if on_step:
            on_step("step_start", step=step, index=idx, total=total)

        missing_deps = [
            dep for dep in step.depends_on
            if dep not in {f.step_id for f in all_facts}
        ]
        if missing_deps:
            error_fact = ErrorFact(
                step_id=step.step_id,
                name=step.expected_fact,
                value=f"Skipped: missing dependencies {missing_deps}",
            )
            all_facts.append(error_fact)
            facts[step.expected_fact] = error_fact
            errors.append(f"step '{step.step_id}': missing dependencies {missing_deps}")
            if on_step:
                on_step("step_error", step=step, error=errors[-1])
            continue

        resolved_inputs = _resolve_inputs(step.inputs, facts)

        # Step guard: approval gate
        if on_step:
            approval = on_step("step_approve", step=step, resolved_inputs=resolved_inputs)
            if approval is False:
                skip_fact = Fact(
                    step_id=step.step_id,
                    name=step.expected_fact,
                    value="Skipped by user (step guard)",
                    fact_type="text",
                )
                all_facts.append(skip_fact)
                facts[step.expected_fact] = skip_fact
                if on_step:
                    on_step("step_skipped", step=step)
                continue

        ctx = StepContext(
            resolved_inputs=resolved_inputs,
            facts=facts,
            workspace=workspace,
        )
        result = execute_step(step, ctx)

        match result:
            case Ok(fact):
                all_facts.append(fact)
                facts[fact.name] = fact
                logger.info(
                    "Step '%s' produced fact '%s' = %s",
                    step.step_id, fact.name, display_fact(fact)[:100],
                )
                if on_step:
                    on_step("step_done", step=step, fact=fact)
            case Err(e):
                error_fact = ErrorFact(
                    step_id=step.step_id,
                    name=step.expected_fact,
                    value=str(e),
                )
                all_facts.append(error_fact)
                facts[step.expected_fact] = error_fact
                errors.append(e)
                logger.warning("Step '%s' failed: %s", step.step_id, e)
                if on_step:
                    on_step("step_error", step=step, error=e)

    error_facts = [f for f in all_facts if f.fact_type == "error"]
    success = len(error_facts) == 0

    if success:
        if on_step:
            on_step("synthesize")
        answer = _synthesize_answer(spec, all_facts, model_id, ask_fn)
    else:
        answer = f"Plan partially failed. Errors: {'; '.join(errors)}"

    execution_result = ExecutionResult(
        question=spec.question,
        facts=tuple(all_facts),
        answer=answer,
        success=success,
    )

    if not success:
        return Err(
            f"Plan execution had {len(error_facts)} error(s): "
            + "; ".join(errors)
        )

    return Ok(execution_result)




# ---------------------------------------------------------------------------
# Full validation pipeline -- V2_trinity
# ---------------------------------------------------------------------------


def validate_plan(
    spec: Spec,
    model_id: str = "",
    ask_fn: AskFn | None = None,
    workspace: Path | None = None,
    on_step: Any = None,
) -> Result[ExecutionResult, str]:
    """Full V2 pipeline. Cheapest first:

    1. Semantic: check inline Python syntax, artifact refs (pure, cheapest)
    2. Executive: run the plan and collect facts (side-effecting)

    on_step: optional callback(event, **kwargs) for progress display.
    """
    if on_step:
        on_step("validate_semantics")

    match validate_semantics(spec):
        case Err(e):
            return Err(e)

    logger.info("Plan semantics validated")

    match execute_plan(spec, workspace, model_id, ask_fn, on_step=on_step):
        case Err(e):
            return Err(e)
        case Ok(result):
            pass

    logger.info("Plan executed successfully: %d facts collected", len(result.facts))
    return Ok(result)


# ---------------------------------------------------------------------------
# Ouroboros -- G'_trinity with patching support
# ---------------------------------------------------------------------------


def _identify_failing_step(error: str, spec: Spec) -> str | None:
    """Extract the failing step_id from an error message."""
    for step in spec.steps:
        if f"step '{step.step_id}'" in error:
            return step.step_id
    return None


def ouroboros_fix(
    spec: Spec,
    error: str,
    ctx: GenerationContext,
    model_id: str = "",
    ask_fn: AskFn | None = None,
) -> Spec | None:
    """G'_trinity: model sees its prior plan + error, returns a SpecPatch.

    The model returns a SpecPatch (targeted step-level edits) instead of
    the full Spec. The patch is applied via apply_spec_patch(). This keeps
    the ouroboros output small -- the model only returns what changed.

    Falls back to full Spec replacement if:
    - The model returns a full Spec instead of a patch
    - The patch fails to apply

    ctx is used to enrich the system prompt with domain context:
    - Discovered artifacts listing helps the fix model know what
      artifacts are available and their signatures.
    - Plan construction principles guide the fix.
    - Feedback from prior rounds provides learning context.
    """
    fn = resolve_ask_fn(model_id, ask_fn)

    failing_step = _identify_failing_step(error, spec)

    system_parts = [
        "You are correcting a plan you previously generated.",
        f"Question: {spec.question}",
        "",
        "You will receive your prior plan and an error.",
        "Return a SpecPatch(...) Python expression with targeted edits to fix the error.",
        "",
    ]

    # Inject domain context from ctx so the fix model knows what
    # artifacts are available and what patterns to follow
    for ds in ctx.domain_context:
        if ds.content and not ds.content.startswith("No "):
            # Include a condensed version for the fix model
            preview = ds.content[:800]
            if len(ds.content) > 800:
                preview += "\n... (truncated)"
            system_parts.extend([
                f"## {ds.heading}",
                preview,
                "",
            ])

    system_parts.extend([
        "Respond with a SpecPatch(...) expression + banners. No markdown, no explanation.",
        "Use ### step_id ### banners for code -- both patched and new steps.",
        "",
        "Example:",
        "",
        "SpecPatch(",
        '    step_patches=(',
        '        StepPatch(step_id="s3", fields={"description": "fixed step"}),',
        "    ),",
        "    add_steps=(",
        '        Step(step_id="s_new", description="compute the answer",',
        '             artifact_type="inline_python",',
        '             expected_fact="answer", extraction_expr="result"),',
        '        Step(step_id="s_write", description="save the test file",',
        '             artifact_type="write_file",',
        '             inputs={"path": "tests/test_output.py"},',
        '             expected_fact="test_saved", depends_on=("s_new",)),',
        "    ),",
        '    remove_steps=("step_id_to_remove",),',
        ")",
        "",
        "### s3 ###",
        "x = fixed_value + 1",
        "result = str(x)",
        "",
        "### s_new ###",
        "y = 1 + 1",
        "result = str(y)",
        "",
        "### s_write ###",
        "import unittest",
        "from mymodule import compute",
        "",
        "class TestCompute(unittest.TestCase):",
        "    def test_basic(self):",
        "        self.assertEqual(compute(1), 2)",
    ])

    if failing_step:
        system_parts.append(f"\nThe failing step is: '{failing_step}'")
        system_parts.append("Focus your patch on this step.")

    user_parts = [
        "# Your prior plan",
        "",
        serialize_spec(spec),
        "",
        "# Error",
        "",
        error[:4000],
        "",
        "Return a SpecPatch(...) expression with targeted edits to fix this error.",
    ]

    # Include feedback from prior rounds if available in ctx
    if ctx.feedback:
        user_parts.extend([
            "",
            "# Prior round feedback (for context)",
            "",
        ])
        for fb in ctx.feedback[-3:]:
            user_parts.append(f"  {fb}")

    match fn("\n".join(system_parts), "\n".join(user_parts)):
        case Err(e):
            logger.warning("Ouroboros model error: %s", e)
            return None
        case Ok(raw_text):
            pass

    # Try as SpecPatch first -- with banner support
    banner_sections: list[tuple[str, str]] = []
    try:
        patch, banner_sections = parse_response_with_files(raw_text, SpecPatch)
    except ValueError:
        # No banners -- try plain parse
        try:
            patch = parse_typed_response(raw_text, SpecPatch)
        except ValueError as patch_err:
            logger.debug("Not a valid SpecPatch: %s", patch_err)
            patch = None

    if patch is not None:
        logger.info(
            "Ouroboros patch: %d step patch(es), %d add, %d remove",
            len(patch.step_patches),
            len(patch.add_steps),
            len(patch.remove_steps),
        )
        for sp in patch.step_patches:
            logger.info(
                "  step '%s' fields: %s",
                sp.step_id,
                {k: repr(v)[:120] for k, v in sp.fields.items()},
            )

        match apply_spec_patch(spec, patch):
            case Ok(corrected):
                if banner_sections:
                    corrected = _attach_banner_code(corrected, banner_sections)
                corrected = promote_spec(corrected)
                logger.info(
                    "Ouroboros patch applied: %d steps",
                    len(corrected.steps),
                )
                return corrected
            case Err(e):
                logger.warning("Ouroboros patch apply failed: %s", e)

    # Fallback: try as full Spec replacement
    try:
        corrected, sections = parse_response_with_files(raw_text, Spec)
        corrected = _attach_banner_code(corrected, sections)
        corrected = promote_spec(corrected)
        logger.info(
            "Ouroboros produced full replacement spec with %d steps",
            len(corrected.steps),
        )
        return corrected
    except ValueError:
        pass

    try:
        corrected = promote_spec(parse_typed_response(raw_text, Spec))
        logger.info(
            "Ouroboros produced full replacement spec (no banners) with %d steps",
            len(corrected.steps),
        )
        return corrected
    except ValueError as e:
        logger.warning("Ouroboros full-spec fallback also failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Emit -- write results to disk
# ---------------------------------------------------------------------------


def emit_result(
    spec: Spec,
    result: ExecutionResult,
    rounds: int,
    fixes: int,
    version: int,
    prompt: str | None,
    output_dir: Path | None = None,
) -> Result[Path, str]:
    """Write the execution result and plan to disk."""
    from compass.generators._types import GenerationReport

    if output_dir is None:
        output_dir = Path(".") / "trinity_output"

    output_dir.mkdir(parents=True, exist_ok=True)

    plan_path = output_dir / "plan.json"
    plan_path.write_text(serialize_spec(spec))

    result_path = output_dir / "result.json"
    result_path.write_text(serialize_result(result))

    report = GenerationReport(
        version=version,
        rounds=rounds,
        ouroboros_fixes=fixes,
        outcome="success" if result.success else "partial",
        claim=result.answer[:200] if result.answer else None,
        user_prompt=prompt,
    )
    report_path = output_dir / ".report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))

    summary_lines = [
        "# Trinity Result",
        "",
        "## Question",
        spec.question,
        "",
        "## Facts",
    ]
    for fact in result.facts:
        status = "\u2713" if fact.fact_type != "error" else "\u2717"
        summary_lines.append(
            f"- [{status}] {fact.name} ({fact.fact_type}): {display_fact(fact)}"
        )
    summary_lines.extend([
        "",
        "## Answer",
        result.answer,
        "",
        "## Plan",
        f"{len(spec.steps)} steps, {rounds} round(s), {fixes} fix(es)",
    ])
    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines))

    logger.info("Trinity result written to %s/", output_dir)
    return Ok(output_dir)


# ---------------------------------------------------------------------------
# Load -- read a Trinity result back
# ---------------------------------------------------------------------------


def load_result(path: Path) -> Result[Spec, str]:
    """Load a Trinity plan from disk."""
    plan_path = path / "plan.json" if path.is_dir() else path

    if not plan_path.exists():
        return Err(f"Plan file not found: {plan_path}")

    try:
        raw = json.loads(plan_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return Err(f"Cannot read plan: {exc}")

    return validate_spec(raw)
