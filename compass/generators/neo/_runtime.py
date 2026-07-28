"""
IO boundaries for plan generation.

All side effects live here: step execution, generator resolution,
meta-generation, ouroboros invocation, file I/O.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from compass.generators._types import (
    AskFn,
    Err,
    GenerationContext,
    GenerationReport,
    Ok,
    Result,
)
from compass.generators._invoke import (
    build_system_prompt,
    build_user_message,
    resolve_ask_fn,
)
from compass.core.python_schema import parse_typed_response

from compass.generators.neo._types import (
    ExecutedPlan,
    PlanConfig,
    PlanSpec,
    Step,
    StepResult,
    topological_order,
    validate_artifact_types,
    validate_spec,
)

logger = logging.getLogger(__name__)

_TYPES_MODULE = (Path(__file__).parent / "_types.py").read_text()


# ---------------------------------------------------------------------------
# Generator resolution
# ---------------------------------------------------------------------------


def _find_generators_root() -> Path:
    """Find the compass/generators directory."""
    import compass.generators
    return Path(compass.generators.__path__[0])


def _resolve_generator(artifact_type: str) -> Optional[Path]:
    """Try to find an existing generator for the given artifact type.

    Looks for compass/generators/<name>/generate.py where <name> is
    the artifact_type normalized to a Python identifier.

    Neo never delegates to himself -- that would be unbounded recursion.
    """
    root = _find_generators_root()
    normalized = artifact_type.lower().replace("-", "_").replace(" ", "_")

    if normalized == "neo":
        return None

    candidate = root / normalized / "generate.py"
    if candidate.exists():
        return candidate

    return None


def _meta_generate(artifact_type: str, config: PlanConfig) -> Result[Path, str]:
    """Meta-generate a new generator for the given artifact type.

    Calls compass.generators.meta.generate.run() in-process,
    then returns the path to its generate.py.
    """
    root = _find_generators_root()
    meta_generate = root / "meta" / "generate.py"

    if not meta_generate.exists():
        return Err(
            f"Meta-generator not found at {meta_generate}. "
            f"Cannot create generator for artifact_type '{artifact_type}'."
        )

    logger.info("Meta-generating generator for artifact_type: %s", artifact_type)

    try:
        gen_mod = importlib.import_module("compass.generators.meta.generate")
        result = gen_mod.run(
            prompt=f"Generate a {artifact_type} generator that produces {artifact_type} artifacts.",
            model_id=config.model_id,
        )
        match result:
            case Err(e):
                return Err(f"Meta-generation failed: {e}")
            case Ok(_):
                pass
    except Exception as e:
        return Err(f"Meta-generation error: {e}")

    # Check if the generator was created
    normalized = artifact_type.lower().replace("-", "_").replace(" ", "_")
    candidate = root / normalized / "generate.py"
    if candidate.exists():
        return Ok(candidate)

    return Err(
        f"Meta-generation completed but generator not found at {candidate}"
    )


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------


def _run_generator(
    generator_path: Path,
    prompt: str,
    config: PlanConfig,
    output: Path | None = None,
) -> Result[str, str]:
    """Run a generator's generate.py in-process via importlib.

    Returns the path to the generated artifact on success.
    Calls run() directly -- no sys.argv hacking, no SystemExit catching.
    """
    gen_name = generator_path.parent.name
    module_name = f"compass.generators.{gen_name}.generate"

    logger.info("Running generator in-process: %s", module_name)

    try:
        gen_mod = importlib.import_module(module_name)
        result = gen_mod.run(
            prompt=prompt,
            model_id=config.model_id,
            verbose=config.verbose,
            output=output,
        )
        match result:
            case Ok(path):
                return Ok(str(path))
            case Err(e):
                return Err(f"Generator '{gen_name}' failed: {e}")
            case _:
                return Err(f"Generator '{gen_name}' returned unexpected: {result}")
    except Exception as e:
        return Err(f"Generator execution error: {e}")


def _write_file_directly(
    step: Step,
    output_dir: Path,
    prior_results: dict[int, StepResult],
) -> Result[str, str]:
    """For simple file-producing steps, write the prompt content as a file.

    This is a fallback when no generator is available and meta-generation
    is not possible. The step's prompt should contain the file content.
    """
    # Determine filename from artifact_type and description
    normalized = step.artifact_type.lower().replace("-", "_").replace(" ", "_")
    ext_map = {
        "python_module": ".py",
        "python_file": ".py",
        "module": ".py",
        "test_suite": ".py",
        "test_file": ".py",
        "config": ".json",
    }
    ext = ext_map.get(normalized, ".txt")

    # Try to extract a filename from the description
    desc_lower = step.description.lower()
    name_match = re.search(r'[\w]+\.py', desc_lower)
    if name_match:
        filename = name_match.group(0)
    else:
        filename = f"step_output{ext}"

    out_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(step.prompt)
    return Ok(str(out_path))


def execute_plan(
    spec: PlanSpec,
    config: PlanConfig,
) -> Result[ExecutedPlan, str]:
    """Execute all steps in topological order. [EXECUTIVE]

    For each step:
    1. Resolve artifact_type to a generator
    2. If not found, attempt meta-generation
    3. Run the generator with step.prompt
    4. Accumulate results for later steps
    5. Stop on first failure
    """
    order = topological_order(spec.steps)
    results: dict[int, StepResult] = {}
    all_results: list[StepResult] = [None] * len(spec.steps)  # type: ignore

    # Snapshot existing generators so we can detect meta-generation
    # across validation retries within the same run.
    pre_existing = {
        p.parent.name
        for p in _find_generators_root().glob("*/generate.py")
    }

    # Create a temp directory for plan outputs
    plan_dir = config.output_path.parent / config.output_path.stem
    plan_dir.mkdir(parents=True, exist_ok=True)

    for step_idx in order:
        step = spec.steps[step_idx]
        logger.info(
            "Executing step %d/%d: %s (type=%s)",
            step_idx + 1, len(spec.steps), step.description, step.artifact_type,
        )

        # Check dependencies succeeded
        for dep in step.depends_on:
            if dep in results and not results[dep].success:
                error_msg = (
                    f"steps[{step_idx}]: dependency steps[{dep}] failed, "
                    f"cannot proceed"
                )
                sr = StepResult(
                    step=step,
                    step_index=step_idx,
                    error=error_msg,
                    success=False,
                )
                all_results[step_idx] = sr
                results[step_idx] = sr
                return Err(error_msg)

        step_dir = plan_dir / f"step_{step_idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Build augmented prompt with dependency context
        augmented_prompt = step.prompt
        dep_context_parts: list[str] = []
        for dep in step.depends_on:
            if dep in results and results[dep].success:
                dep_result = results[dep]
                dep_context_parts.append(
                    f"Step {dep} produced: {dep_result.output_path} "
                    f"({dep_result.step.description})"
                )
        if dep_context_parts:
            augmented_prompt = (
                augmented_prompt + "\n\nPrior step outputs:\n" +
                "\n".join(dep_context_parts)
            )

        # Try to resolve generator
        generator_path = _resolve_generator(step.artifact_type)
        meta_generated = (
            generator_path is not None
            and generator_path.parent.name not in pre_existing
        )

        if generator_path is None:
            logger.info(
                "No generator found for '%s', attempting meta-generation",
                step.artifact_type,
            )
            match _meta_generate(step.artifact_type, config):
                case Ok(path):
                    generator_path = path
                    meta_generated = True
                    logger.info("Meta-generated generator at %s", path)
                case Err(e):
                    logger.warning(
                        "Meta-generation failed for '%s': %s",
                        step.artifact_type, e,
                    )
                    # Fallback: write file directly from prompt
                    match _write_file_directly(step, step_dir, results):
                        case Ok(out_path):
                            sr = StepResult(
                                step=step,
                                step_index=step_idx,
                                generator_used="direct_write",
                                meta_generated=False,
                                output_path=out_path,
                                success=True,
                            )
                            all_results[step_idx] = sr
                            results[step_idx] = sr
                            continue
                        case Err(write_err):
                            error_msg = (
                                f"steps[{step_idx}]: no generator for "
                                f"'{step.artifact_type}' and meta-generation "
                                f"failed: {e}; direct write also failed: {write_err}"
                            )
                            sr = StepResult(
                                step=step,
                                step_index=step_idx,
                                error=error_msg,
                                success=False,
                            )
                            all_results[step_idx] = sr
                            results[step_idx] = sr
                            return Err(error_msg)

        # Run the generator
        if generator_path is not None:
            generator_name = generator_path.parent.name
            match _run_generator(generator_path, augmented_prompt, config, output=step_dir):
                case Ok(out_path):
                    sr = StepResult(
                        step=step,
                        step_index=step_idx,
                        generator_used=generator_name,
                        meta_generated=meta_generated,
                        output_path=out_path,
                        success=True,
                    )
                    all_results[step_idx] = sr
                    results[step_idx] = sr
                    logger.info(
                        "Step %d succeeded: %s -> %s",
                        step_idx, generator_name, out_path,
                    )
                case Err(e):
                    error_msg = f"steps[{step_idx}]: generator '{generator_name}' failed: {e}"
                    sr = StepResult(
                        step=step,
                        step_index=step_idx,
                        generator_used=generator_name,
                        meta_generated=meta_generated,
                        error=error_msg,
                        success=False,
                    )
                    all_results[step_idx] = sr
                    results[step_idx] = sr
                    return Err(error_msg)

    return Ok(ExecutedPlan(spec=spec, results=tuple(all_results)))


# ---------------------------------------------------------------------------
# Model invocation (plan-specific prompt)
# ---------------------------------------------------------------------------


def invoke_model(ctx: GenerationContext, config: PlanConfig) -> Result:
    """Call the model and return parsed PlanSpec or error."""
    ask_fn = resolve_ask_fn(config.model_id, config.ask_fn)

    system = build_system_prompt(
        ctx,
        _TYPES_MODULE,
        role=(
            "You are Neo, an expert plan generator. You decompose complex goals "
            "into a sequence of concrete steps, each producing a specific artifact. "
            "You understand software architecture, testing, and build systems."
        ),
        contract_preamble=(
            "Write a PlanSpec(...) Python constructor expression. "
            "See the type definition below for the schema."
        ),
    )

    user = build_user_message(
        ctx,
        suffix_lines=(
            "Write a PlanSpec(...) expression as defined in the Output Contract.",
            "See its docstring for the response format.",
            "No markdown fencing, no explanation.",
            "",
            "Each step's prompt must be a complete, self-contained instruction",
            "that a code generator can execute independently.",
            "Use the most natural artifact_type for each step. Unknown types will be meta-generated.",
        ),
    )

    if config.verbose:
        logger.info(
            "Prompt length: system=%d chars, user=%d chars",
            len(system), len(user),
        )

    match ask_fn(system, user):
        case Err() as e:
            return e
        case Ok(raw_text):
            pass

    try:
        spec = parse_typed_response(raw_text, PlanSpec)
        return Ok(spec)
    except ValueError as e:
        return Err(str(e))


# ---------------------------------------------------------------------------
# Ouroboros -- targeted step editing
# ---------------------------------------------------------------------------


def ouroboros(
    spec: PlanSpec,
    error: str,
    ctx: GenerationContext,
    config: PlanConfig,
) -> Result[PlanSpec, str]:
    """Ouroboros: the model consumes its own plan and produces a corrected version."""
    ask_fn = resolve_ask_fn(config.model_id, config.ask_fn)

    step_listing: list[str] = []
    for i, step in enumerate(spec.steps):
        marker = "  <<<< ERROR HERE" if f"steps[{i}]" in error else ""
        prompt_preview = (step.prompt or "")[:200]
        if step.prompt and len(step.prompt) > 200:
            prompt_preview += "..."
        step_listing.append(
            f"### steps[{i}]{marker}\n"
            f"  description: {step.description}\n"
            f"  artifact_type: {step.artifact_type}\n"
            f"  prompt: {prompt_preview}\n"
            f"  depends_on: {list(step.depends_on)}"
        )

    system_parts = [
        "You are correcting a plan you previously generated.",
        f"Original goal: {spec.goal}",
        "",
        "You will receive your prior plan and an error. Write a FULL corrected",
        "PlanSpec(...) Python constructor expression.",
        "",
        "Rules:",
        "- Focus your fix on the failing step. Minimise changes to other steps.",
        "- Make step prompts self-contained and complete.",
        "- Write ONLY the PlanSpec(...) expression. No explanation.",
    ]

    user_parts = [
        f"# Your plan: {spec.goal} ({len(spec.steps)} steps)",
        "",
        "\n\n".join(step_listing),
        "",
        "# Error",
        "",
        error,
        "",
        "Write a corrected PlanSpec(...) expression. Fix only what is broken.",
    ]

    match ask_fn("\n".join(system_parts), "\n".join(user_parts)):
        case Err() as e:
            return e
        case Ok(raw_text):
            pass

    try:
        corrected = parse_typed_response(raw_text, PlanSpec)
    except ValueError as e:
        return Err(f"Ouroboros parse error: {e}")

    from compass.generators.neo._types import validate_spec_instance
    match validate_spec_instance(corrected):
        case Err(e):
            return Err(f"Ouroboros spec validation: {e}")
        case Ok(corrected):
            return Ok(corrected)


def try_ouroboros(
    spec: PlanSpec,
    error: str,
    ctx: GenerationContext,
    config: PlanConfig,
) -> PlanSpec | None:
    """Attempt an ouroboros fix on a step-specific error."""
    # Always try ouroboros for plan errors since they reference steps[N]
    logger.info("Ouroboros: attempting plan fix for error: %s", error[:200])
    match ouroboros(spec, error, ctx, config):
        case Ok(corrected):
            logger.info(
                "Ouroboros: returned corrected plan (%d steps)",
                len(corrected.steps),
            )
            return corrected
        case Err(e):
            logger.warning("Ouroboros: failed -- %s", e)
            return None


# ---------------------------------------------------------------------------
# Plan I/O
# ---------------------------------------------------------------------------


def emit_plan(
    executed: ExecutedPlan,
    path: Path,
    *,
    report: GenerationReport | None = None,
) -> Path:
    """Serialize ExecutedPlan to a JSON report."""
    step_reports = []
    for sr in executed.results:
        if sr is None:
            step_reports.append({"error": "step not executed"})
            continue
        step_reports.append({
            "description": sr.step.description,
            "artifact_type": sr.step.artifact_type,
            "generator_used": sr.generator_used,
            "meta_generated": sr.meta_generated,
            "output_path": sr.output_path,
            "success": sr.success,
            "error": sr.error,
        })

    plan_report = {
        "goal": executed.spec.goal,
        "reasoning": executed.spec.reasoning,
        "steps": step_reports,
        "generation": report.to_dict() if report else {},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan_report, indent=2, ensure_ascii=False))

    if report:
        report_path = path.with_suffix(".report.json")
        report_path.write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("Report written to %s", report_path)

    return path


def load_plan(path: Path) -> Result[PlanSpec, str]:
    """Load a plan report JSON back into a PlanSpec."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return Err(f"Failed to read {path}: {e}")

    goal = raw.get("goal", "")
    reasoning = raw.get("reasoning", "")
    steps_raw = raw.get("steps", [])

    steps: list[Step] = []
    for sr in steps_raw:
        steps.append(Step(
            description=sr.get("description", ""),
            artifact_type=sr.get("artifact_type", "unknown"),
            prompt=sr.get("prompt", sr.get("description", "")),
            depends_on=tuple(sr.get("depends_on", [])),
        ))

    if not steps:
        return Err(f"No steps found in {path}")

    return Ok(PlanSpec(goal=goal, reasoning=reasoning, steps=tuple(steps)))
