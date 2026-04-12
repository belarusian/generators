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
  3. Runtime reality: workspace paths, artifact files on disk, git usable if shell uses git
  4. Executive: actually run the steps and collect facts
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from compass.generators._types import (
    AskFn,
    Cycle,
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
    DiscoveredStep,
    DynamicStep,
    ErrorFact,
    ExecutionResult,
    Fact,
    Spec,
    SpecPatch,
    Step,
    StepPatch,
    TRINITY_PARSE_EXTRA_TYPES,
    apply_spec_patch,
    promote_spec,
    promote_step,
    serialize_spec,
    serialize_result,
    validate_spec,
    validate_spec_patch,
)
from compass.generators.trinity.fact_dispatch import display_fact, resolve_fact
from compass.generators.trinity._paths import generators_repo_root
from compass.generators.trinity.step_dispatch import StepContext, execute_step

logger = logging.getLogger(__name__)

_SPEC_TYPE_SOURCE = inspect.getsource(Step) + "\n\n" + inspect.getsource(Spec)

def _is_cycle_breaking(step: Step) -> bool:
    """Non-deterministic steps whose output invalidates later speculative steps.

    Discovered/dynamic modules opt in by declaring CYCLE_BREAKING = True
    at module level -- detected during artifact discovery and carried
    through promotion.
    """
    if isinstance(step, (DiscoveredStep, DynamicStep)):
        return step.cycle_breaking
    return False


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
        doc = tree.body[0].value.value.strip()

    # Find function definitions and module-level flags
    functions: dict[str, ast.FunctionDef] = {}
    cycle_breaking = False
    return_type = "any"
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if (
                        target.id == "CYCLE_BREAKING"
                        and isinstance(node.value, ast.Constant)
                        and node.value.value is True
                    ):
                        cycle_breaking = True
                    elif (
                        target.id == "RETURN_TYPE"
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)
                    ):
                        return_type = node.value.value

    # Discover files whose run() starts with the Trinity artifact contract:
    #   run(step, resolved_inputs, workspace, *, ...) -> Result
    # Extra keyword args declare required/optional inputs.
    _ARTIFACT_CONTRACT = ("step", "resolved_inputs", "workspace")

    if "run" in functions:
        func_node = functions["run"]
        params = tuple(_extract_params_from_ast(func_node))
        if params[:3] == _ARTIFACT_CONTRACT:
            required = _extract_required_inputs(func_node)
            return DiscoveredArtifact(
                path=rel_path,
                kind="python_module",
                entry_point="run",
                parameters=params,
                doc=doc,
                cycle_breaking=cycle_breaking,
                required_inputs=tuple(required),
                return_type=return_type,
            )

    return None


def _extract_required_inputs(func_node: ast.FunctionDef) -> list[str]:
    """Extract required input keys from a run() function signature.

    Parameters beyond the base contract (step, resolved_inputs, workspace)
    declare what the artifact expects in resolved_inputs:
        def run(step, resolved_inputs, workspace, *, pre_push_oid, ref="main"):
            # pre_push_oid -> required (no default)
            # ref -> optional (has default)
    """
    args = func_node.args
    # kwonly args: everything after * in the signature
    kwonly = args.kwonlyargs
    defaults = args.kw_defaults  # parallel list, None = no default

    required: list[str] = []
    for arg, default in zip(kwonly, defaults):
        if default is None:
            required.append(arg.arg)
    return required


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

    Also merges ``artifacts/*.py`` from the generators checkout (same repo as
    ``compass/``) when the workspace is elsewhere, so bundled steps like
    ``screen`` appear in context and resolve at promotion time.
    """
    if workspace is None:
        workspace = Path(".")

    workspace = workspace.resolve()
    artifacts: list[DiscoveredArtifact] = []
    seen_paths: set[str] = set()

    def _try_file(py_file: Path, root: Path) -> None:
        parts = py_file.relative_to(root).parts
        if any(p.startswith(".") or p == "__pycache__" for p in parts):
            return

        name = py_file.name
        if name == "__init__.py":
            return
        if name.startswith("test_") or name.endswith("_test.py"):
            return

        artifact = _inspect_python_file(py_file, root)
        if artifact is not None and artifact.path not in seen_paths:
            seen_paths.add(artifact.path)
            artifacts.append(artifact)

    for py_file in sorted(workspace.rglob("*.py")):
        _try_file(py_file, workspace)

    bundled_root = generators_repo_root()
    if bundled_root is not None and bundled_root.resolve() != workspace:
        art_dir = bundled_root / "artifacts"
        if art_dir.is_dir():
            for py_file in sorted(art_dir.glob("*.py")):
                if py_file.name == "__init__.py":
                    continue
                _try_file(py_file, bundled_root)

    return artifacts


def format_artifacts_for_context(artifacts: list[DiscoveredArtifact]) -> str:
    """Format discovered artifacts as a human/model-readable listing."""
    if not artifacts:
        return "No runnable artifacts discovered in workspace."

    lines = ["Discovered artifacts:"]
    for a in artifacts:
        param_str = ", ".join(a.parameters) if a.parameters else "(no parameters)"
        first_line = a.doc.split("\n")[0] if a.doc else ""
        lines.append(
            f"  - {a.path} [{a.kind}] entry={a.entry_point}({param_str})"
            + (f" -- {first_line}" if first_line else "")
        )
        # Show required inputs so the model knows what to wire
        if a.required_inputs:
            inputs_str = ", ".join(a.required_inputs)
            lines.append(f"    required inputs: {inputs_str}")
        if a.return_type != "any":
            lines.append(f"    return_type: {a.return_type}")
        # Include plan guide lines from the docstring
        if a.doc and "Plan guide:" in a.doc:
            for paragraph in a.doc.split("\n"):
                stripped = paragraph.strip()
                if stripped.startswith("Plan guide:") or stripped.startswith("Falls back"):
                    lines.append(f"    {stripped}")

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
    """Resolve $fact references in inputs using collected facts.

    Handles three forms the model might produce:
      1. {"$fact": "name"}         -- canonical dict form
      2. "$name"                   -- string shorthand
      3. "{'$fact': 'name'}"       -- model wrote dict as string literal
    """
    resolved = {}
    for key, value in inputs.items():
        if isinstance(value, dict) and "$fact" in value:
            fact_name = value["$fact"]
            if fact_name in facts:
                resolved[key] = resolve_fact(facts[fact_name])
            else:
                resolved[key] = None
        elif isinstance(value, str) and value.startswith("$") and not value.startswith("${"):
            # "$test_output" -> look up "test_output"
            fact_name = value[1:]
            if fact_name in facts:
                resolved[key] = resolve_fact(facts[fact_name])
            else:
                resolved[key] = value  # leave as-is if not a known fact
        elif isinstance(value, str) and "$fact" in value:
            # Model wrote the dict reference as a string: "{'$fact': 'name'}"
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, dict) and "$fact" in parsed:
                    fact_name = parsed["$fact"]
                    if fact_name in facts:
                        resolved[key] = resolve_fact(facts[fact_name])
                    else:
                        resolved[key] = None
                else:
                    resolved[key] = value
            except (ValueError, SyntaxError):
                resolved[key] = value
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
    code_map = dict(sections)
    new_steps = []
    for step in spec.steps:
        if step.step_id in code_map and not step.artifact_ref:
            new_steps.append(replace(step, artifact_ref=code_map[step.step_id]))
        else:
            # Sub-banners: ### step_id:field ### injects into inputs.
            # Lets edit_file use ### s3:old ### / ### s3:new ### for raw content
            # without escaping inside the constructor.
            injected_inputs = dict(step.inputs)
            changed = False
            for banner_key, banner_content in code_map.items():
                if banner_key.startswith(step.step_id + ":"):
                    field = banner_key[len(step.step_id) + 1:]
                    injected_inputs[field] = banner_content
                    changed = True
            if changed:
                new_steps.append(replace(step, inputs=injected_inputs))
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
        spec, sections = parse_response_with_files(
            raw_text, Spec, extra_types=TRINITY_PARSE_EXTRA_TYPES
        )
        spec = _attach_banner_code(spec, sections)
        return Ok(promote_spec(spec))
    except ValueError:
        pass

    # Fallback: no banners (all steps are auto/shell/vision, or code inline)
    try:
        spec = parse_typed_response(raw_text, Spec, extra_types=TRINITY_PARSE_EXTRA_TYPES)
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

    from compass.generators.trinity._types import collect_plan_guides

    parts.extend([
        "",
        "Write a Spec(...) constructor followed by ### step_id ### banners for inline_python and write_file content.",
        "See the Spec docstring for the exact response format. No markdown fencing.",
        "",
        "Artifact types:",
        collect_plan_guides(),
        "",
        "General:",
        "- Steps execute in dependency order. Use depends_on for data dependencies.",
        "- In inputs, reference prior facts as {\"$fact\": \"fact_name\"} to inject them.",
        "- Use artifact_type matching the artifact name (e.g. 'onion' for artifacts/onion.py).",
        "  Use 'auto' only when artifact_ref points to a discovered file directly.",
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
            if not step.artifact_ref:
                errors.append(
                    f"step '{step.step_id}': inline_python has no code "
                    f"(write code after a ### {step.step_id} ### banner)"
                )
                continue
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


def _resolve_step_path(ref: str | None, workspace: Path) -> Path | None:
    if ref is None or not str(ref).strip():
        return None
    p = Path(ref.strip())
    if p.is_absolute():
        return p.resolve()
    return (workspace / p).resolve()


def _shell_command_invokes_git(step) -> bool:
    """True if shell artifact_ref runs the git CLI (workspace-relative check)."""
    if getattr(step, "artifact_type", "") != "shell":
        return False
    ref = step.artifact_ref or ""
    return bool(re.search(r"(?:^|[;&|])\s*git\b", ref))


def validate_runtime_reality(spec: Spec, workspace: Path) -> Result[None, str]:
    """Check plan against the actual workspace: paths, artifact modules, git.

    No user-prompt pattern matching — only ``spec`` steps and OS/git probes.
    Runs after ``validate_semantics``, before ``execute_plan``.
    """
    if not workspace.exists():
        return Err(f"workspace does not exist: {workspace}")
    if not workspace.is_dir():
        return Err(f"workspace is not a directory: {workspace}")

    artifacts = discover_artifacts(workspace)
    promoted = promote_spec(spec, artifacts, workspace)

    errors: list[str] = []

    for step in promoted.steps:
        at = step.artifact_type
        ref = step.artifact_ref

        if at in ("read_file", "vision", "edit_file") and ref and str(ref).strip():
            from compass.generators.trinity._types import VISION_ARTIFACT_REF_SCREEN
            if at == "vision" and str(ref).strip() == VISION_ARTIFACT_REF_SCREEN:
                pass  # captured at execution time; no pre-existing file
            else:
                p = _resolve_step_path(str(ref).strip(), workspace)
                if p is not None and not p.is_file():
                    errors.append(
                        f"step '{step.step_id}': {at} path does not exist or is not a file: {p}"
                    )

        if isinstance(step, (DynamicStep, DiscoveredStep)):
            mp = getattr(step, "module_path", "") or ""
            if mp:
                modp = Path(mp)
                if not modp.is_absolute():
                    modp = (workspace / modp).resolve()
                else:
                    modp = modp.resolve()
                if not modp.is_file():
                    errors.append(
                        f"step '{step.step_id}': artifact module not found: {modp}"
                    )

    if any(_shell_command_invokes_git(s) for s in promoted.steps):
        r = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip()
            tail = f": {err}" if err else ""
            errors.append(
                "shell steps invoke git but this workspace is not a git repository "
                f"(git rev-parse failed){tail}"
            )
        else:
            r2 = subprocess.run(
                ["git", "-C", str(workspace), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r2.returncode != 0:
                err = (r2.stderr or r2.stdout or "").strip()
                errors.append(
                    "git is not usable in this workspace (HEAD does not resolve). "
                    "Repair the repository (e.g. fsck, re-clone) before git shell steps. "
                    f"Detail: {err}"
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
    prior_facts: dict[str, Fact] | None = None,
) -> Result[ExecutionResult, str] | Cycle:
    """Executive validation: run all steps, collect facts.

    Returns Ok(ExecutionResult) on full success, Err on failure,
    or Cycle on partial success (non-deterministic step completed
    but remaining steps need re-planning).

    on_step: optional callback(event, **kwargs) for progress display.
        Events: step_start(step, index, total), step_done(step, fact),
                step_error(step, error), cycle_break(step, facts, remaining),
                step_cached(step, fact), synthesize().
    prior_facts: facts from previous session turns or prior cycle breaks,
        seeded into the facts dict so inline code and $fact refs can use them.
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
    if prior_facts:
        facts.update(prior_facts)
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

        # Skip steps whose fact was produced in a prior cycle break.
        # Prevents re-executing the non-deterministic step (which would
        # produce a different result, defeating the purpose of the break).
        if step.expected_fact and step.expected_fact in facts:
            existing = facts[step.expected_fact]
            if existing.step_id == step.step_id:
                logger.info(
                    "Step '%s': fact '%s' from prior cycle, skipping",
                    step.step_id, step.expected_fact,
                )
                if existing not in all_facts:
                    all_facts.append(existing)
                if on_step:
                    on_step("step_cached", step=step, fact=existing)
                continue

        failed_steps = {
            f.step_id for f in all_facts if isinstance(f, ErrorFact)
        }
        missing_deps = [
            dep for dep in step.depends_on
            if dep not in {f.step_id for f in all_facts}
        ]
        failed_deps = [
            dep for dep in step.depends_on
            if dep in failed_steps
        ]
        if missing_deps or failed_deps:
            reasons = []
            if missing_deps:
                reasons.append(f"missing dependencies {missing_deps}")
            if failed_deps:
                reasons.append(f"failed dependencies {failed_deps}")
            reason = "; ".join(reasons)
            error_fact = ErrorFact(
                step_id=step.step_id,
                name=step.expected_fact,
                value=f"Skipped: {reason}",
            )
            all_facts.append(error_fact)
            facts[step.expected_fact] = error_fact
            errors.append(f"step '{step.step_id}': {reason}")
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
            model_id=model_id,
            ask_fn=ask_fn,
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

                # Cycle break: non-deterministic step completed.
                # Its fact is a proper success -- but remaining steps
                # were planned speculatively. Halt and re-plan.
                if _is_cycle_breaking(step) and idx < total - 1:
                    remaining = [s.step_id for s in sorted_steps[idx + 1:]]

                    facts_desc = "\n".join(
                        f"  {f.name} ({f.fact_type}): "
                        f"{display_fact(f)[:200]}"
                        for f in all_facts if f.fact_type != "error"
                    )
                    logger.info(
                        "Cycle break after '%s' (%s) -- "
                        "%d facts collected, %d steps remaining",
                        step.step_id, type(step).__name__,
                        len(all_facts), len(remaining),
                    )
                    if on_step:
                        on_step(
                            "cycle_break", step=step,
                            facts=dict(facts), remaining=remaining,
                        )
                    return Cycle(
                        facts=dict(facts),
                        message=(
                            f"Step '{step.step_id}' "
                            f"({type(step).__name__}) completed.\n"
                            f"Collected facts:\n{facts_desc}\n"
                            f"Steps not executed: "
                            f"{', '.join(remaining)}\n"
                            f"These facts are available as "
                            f'{{\"$fact\": \"name\"}} references.'
                        ),
                    )

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
    prior_facts: dict[str, Fact] | None = None,
) -> Result[ExecutionResult, str] | Cycle:
    """Full V2 pipeline. Cheapest first:

    1. Semantic: check inline Python syntax, artifact refs (pure, cheapest)
    2. Runtime reality: filesystem + git (no user-text regex)
    3. Executive: run the plan and collect facts (side-effecting)

    Returns Ok(ExecutionResult), Err(str), or Cycle (partial success).
    on_step: optional callback(event, **kwargs) for progress display.
    prior_facts: facts from previous session turns or prior cycle breaks.
    """
    if on_step:
        on_step("validate_semantics")

    match validate_semantics(spec):
        case Err(e):
            return Err(e)

    logger.info("Plan semantics validated")

    ws = (workspace or Path(".")).resolve()
    if on_step:
        on_step("validate_runtime_reality")
    match validate_runtime_reality(spec, ws):
        case Err(e):
            return Err(e)

    logger.info("Plan runtime reality validated")

    match execute_plan(spec, workspace, model_id, ask_fn, on_step=on_step, prior_facts=prior_facts):
        case Cycle() as c:
            return c
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
        '        StepPatch(step_id="s3", fields={"artifact_type": "inline_python", "description": "verify with Python"}),',
        "    ),",
        "    add_steps=(",
        '        Step(step_id="s_new", description="compute the answer",',
        '             artifact_type="inline_python",',
        '             expected_fact="answer", extraction_expr="result"),',
        "    ),",
        '    remove_steps=("step_id_to_remove",),',
        ")",
        "",
        "### s3 ###",
        "count = int(word_count)",
        "result = count > 0",
        "",
        "### s_new ###",
        "y = 1 + 1",
        "result = str(y)",
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
        patch, banner_sections = parse_response_with_files(
            raw_text, SpecPatch, extra_types=TRINITY_PARSE_EXTRA_TYPES
        )
    except ValueError:
        # No banners -- try plain parse
        try:
            patch = parse_typed_response(
                raw_text, SpecPatch, extra_types=TRINITY_PARSE_EXTRA_TYPES
            )
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
        corrected, sections = parse_response_with_files(
            raw_text, Spec, extra_types=TRINITY_PARSE_EXTRA_TYPES
        )
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
        corrected = promote_spec(
            parse_typed_response(raw_text, Spec, extra_types=TRINITY_PARSE_EXTRA_TYPES)
        )
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
