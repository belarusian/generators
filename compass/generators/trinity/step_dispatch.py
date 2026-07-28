"""
Step execution dispatch - singledispatch for artifact type routing.

Same pattern as oracle-/compass/agents/neo/dispatch.py:
  - Each Step subtype gets a registered handler
  - Fallback handles unknown/dynamic artifact types
  - No if/elif chains in execute_plan

    execute_step(step, ctx) -> Result[Fact, str]
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import traceback
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path
from typing import Any

from compass.generators._types import Err, Ok, Result
from compass.generators.trinity._types import (
    DiscoveredStep,
    DynamicStep,
    Fact,
    FileFact,
    InlinePythonStep,
    ProgrammerStep,
    ReadFileStep,
    ShellStep,
    Step,
    VisionStep,
    WriteFileStep,
)
from compass.generators.trinity.fact_dispatch import resolve_fact

logger = logging.getLogger(__name__)


@dataclass
class StepContext:
    """Execution context passed to every step handler."""

    resolved_inputs: dict
    facts: dict[str, Fact]
    workspace: Path


# ============================================================================
# Shared helpers
# ============================================================================


def _extract_error_location(exc: Exception) -> str:
    """Extract the last traceback frame as 'File ..., line N' for error messages."""
    tb = exc.__traceback__
    if tb is None:
        return ""
    frames = traceback.extract_tb(tb)
    if not frames:
        return ""
    last = frames[-1]
    parts = [f"  at {last.filename}, line {last.lineno}"]
    if last.line:
        parts.append(f"    {last.line}")
    return "\n".join(parts)


def _classify_value(value: Any) -> str:
    """Classify a Python value into a fact_type."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float, complex)):
        return "numeric"
    if isinstance(value, str):
        try:
            json.loads(value)
            return "json"
        except (json.JSONDecodeError, TypeError):
            return "text"
    if isinstance(value, (dict, list, tuple)):
        return "json"
    return "text"


def _serialize_value(value: Any) -> str:
    """Serialize a Python value to a string for storage in a Fact."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _import_file_as_module(filepath: Path, module_name: str | None = None) -> Any:
    """Import a Python file as a module using importlib."""
    if module_name is None:
        module_name = filepath.stem

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {filepath}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def _inspect_signature_live(module: Any, entry_point: str) -> list[str]:
    """Inspect a live function's signature to get parameter names."""
    func = getattr(module, entry_point, None)
    if func is None or not callable(func):
        return []

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return []

    params: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        params.append(name)

    return params


def _map_inputs_to_params(
    inputs: dict,
    parameters: list[str],
    func: Any,
) -> dict:
    """Map step inputs to function parameters using signature inspection."""
    if not callable(func):
        return inputs

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return inputs

    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )

    if has_var_keyword:
        return dict(inputs)

    accepted = set(sig.parameters.keys()) - {"self", "cls"}
    mapped: dict[str, Any] = {}
    unmapped: list[str] = []

    for key, value in inputs.items():
        if key in accepted:
            mapped[key] = value
        else:
            unmapped.append(key)

    if unmapped:
        logger.debug(
            "Inputs not mapped to parameters (skipped): %s",
            unmapped,
        )

    return mapped


_READ_FILE_LIMIT = 200  # Lines; files under this are read in full


# ============================================================================
# Executor implementations
# ============================================================================


def _execute_python_code(
    step: Step,
    code: str,
    resolved_inputs: dict,
    facts: dict[str, Fact],
    source_label: str = "inline",
    workspace: Path | None = None,
) -> Result[Fact, str]:
    """Execute Python code and extract a fact."""
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    namespace.update(resolved_inputs)
    namespace["inputs"] = dict(resolved_inputs)
    namespace["workspace"] = workspace
    namespace["_facts"] = {
        name: resolve_fact(fact) for name, fact in facts.items()
    }

    try:
        exec(code, namespace)  # noqa: S102
    except Exception as exc:
        tb = _extract_error_location(exc)
        return Err(
            f"step '{step.step_id}': execution error in {source_label}: "
            f"{type(exc).__name__}: {exc}"
            + (f"\n{tb}" if tb else "")
        )

    try:
        value = eval(step.extraction_expr, namespace)  # noqa: S307
    except Exception as exc:
        return Err(
            f"step '{step.step_id}': extraction error for "
            f"'{step.extraction_expr}': {type(exc).__name__}: {exc}"
        )

    fact_type = _classify_value(value)
    serialized = _serialize_value(value)

    return Ok(Fact(
        step_id=step.step_id,
        name=step.expected_fact,
        value=serialized,
        fact_type=fact_type,
    ))


def _execute_read_file(
    step: Step,
    resolved_inputs: dict,
    workspace: Path,
) -> Result[Fact, str]:
    """Read a file with adaptive pagination.

    Small files (<= 200 lines) are read in full. Large files return
    head (120 lines) + tail (80 lines) with a gap marker. The model
    can request specific ranges via offset/limit in inputs.
    """
    ref = step.artifact_ref or ""
    path = Path(ref) if os.path.isabs(ref) else workspace / ref

    if not path.exists():
        return Err(f"step '{step.step_id}': file not found: {ref}")

    try:
        lines = path.read_text().splitlines()
    except Exception as exc:
        return Err(f"step '{step.step_id}': cannot read {ref}: {exc}")

    total = len(lines)
    offset = int(resolved_inputs.get("offset", 0))
    limit = int(resolved_inputs.get("limit", 0))

    # Extract the requested slice as raw content (no line-number formatting).
    # display_fact(FileFact) handles presentation with line numbers.
    actual_offset = 0
    if offset >= total:
        raw_content = ""
        actual_offset = total
    elif limit > 0:
        raw_content = "\n".join(lines[offset:offset + limit])
        actual_offset = offset
    elif offset > 0:
        cap = min(offset + _READ_FILE_LIMIT, total)
        raw_content = "\n".join(lines[offset:cap])
        actual_offset = offset
    else:
        raw_content = "\n".join(lines)

    return Ok(FileFact(
        step_id=step.step_id,
        name=step.expected_fact,
        value=raw_content,
        fact_type="text",
        path=ref,
        line_offset=actual_offset,
    ))


def _execute_shell(
    step: Step,
    resolved_inputs: dict,
    facts: dict[str, Fact] | None = None,
) -> Result[Fact, str]:
    """Execute a shell command step and extract a fact.

    All resolved inputs AND all accumulated facts are injected as
    environment variables. This means shell commands can reference
    any prior fact by its exact name: $formatted_comment, $pr_body, etc.
    No template engines, no regex -- just shell's own $VAR expansion.
    """
    cmd = step.artifact_ref or ""
    env = os.environ.copy()

    # Facts first (lower priority -- inputs override if names collide)
    if facts:
        for name, fact in facts.items():
            env[name] = str(resolve_fact(fact))

    # Resolved inputs on top
    for k, v in resolved_inputs.items():
        env[str(k)] = str(v)

    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,  # noqa: S602
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,  # own process group -- killable on timeout
        )
    except Exception as exc:
        return Err(
            f"step '{step.step_id}': shell error: {type(exc).__name__}: {exc}"
        )

    try:
        shell_timeout = int(step.inputs.get("timeout", 540))
        stdout, stderr = proc.communicate(timeout=shell_timeout)
    except subprocess.TimeoutExpired:
        # Kill the entire process group, not just the shell wrapper.
        # Without this, child processes (e.g. gh) survive as orphans
        # and can hold stdin, freezing the REPL.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=5)
        return Err(f"step '{step.step_id}': shell command timed out ({shell_timeout}s)")

    if proc.returncode != 0:
        detail = stderr.strip() or stdout.strip()
        return Err(
            f"step '{step.step_id}': shell command failed (exit {proc.returncode}): "
            f"{detail[:502]}"
        )

    output = stdout.strip()
    fact_type = _classify_value(output)

    return Ok(Fact(
        step_id=step.step_id,
        name=step.expected_fact,
        value=output,
        fact_type=fact_type,
    ))


def _execute_vision_step(
    step: Step,
    resolved_inputs: dict,
) -> Result[Fact, str]:
    """Execute a vision step by sending an image to a vision model."""
    image_path = Path(step.artifact_ref)
    if not image_path.exists():
        return Err(f"step '{step.step_id}': image not found: {step.artifact_ref}")

    try:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    except OSError as exc:
        return Err(f"step '{step.step_id}': cannot read image: {exc}")

    prompt = resolved_inputs.get("prompt", "Describe this image.")
    if not isinstance(prompt, str):
        prompt = str(prompt)

    vision_model = os.environ.get("VISION_MODEL", "")
    if not vision_model:
        return Err(
            f"step '{step.step_id}': VISION_MODEL env var not set. "
            "Set it to a vision-capable model (e.g. VISION_MODEL=qwen3-vl:30b@local)"
        )

    from compass.llm.providers import get_provider_by_id
    try:
        provider = get_provider_by_id(vision_model)
    except Exception as exc:
        return Err(f"step '{step.step_id}': cannot resolve vision model: {exc}")

    logger.info(
        "Vision step '%s': sending %s to %s",
        step.step_id, image_path.name, provider.name,
    )

    resp = provider.complete(
        [{"role": "user", "content": prompt, "images": [image_b64]}],
        max_tokens=-1,
        temperature=0.3,
    )

    raw_response = resp.text.strip()
    if not raw_response or raw_response.startswith("The oracle is silent"):
        return Err(f"step '{step.step_id}': vision model returned no response: {raw_response}")

    # Try to parse as JSON, fall back to raw string
    parsed = raw_response
    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            open_end = cleaned.index("\n") + 1
            close = cleaned.find("\n```", open_end)
            cleaned = cleaned[open_end:close] if close != -1 else cleaned[open_end:]
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    namespace = {
        "result": parsed,
        "raw_response": raw_response,
        "__builtins__": __builtins__,
    }
    namespace.update(resolved_inputs)

    try:
        value = eval(step.extraction_expr, namespace)  # noqa: S307
    except Exception as exc:
        return Err(
            f"step '{step.step_id}': extraction error for "
            f"'{step.extraction_expr}': {type(exc).__name__}: {exc}"
        )

    return Ok(Fact(
        step_id=step.step_id,
        name=step.expected_fact,
        value=_serialize_value(value),
        fact_type=_classify_value(value),
    ))



def _execute_programmer(
    step: Step,
    resolved_inputs: dict,
    workspace: Path,
) -> Result[Fact, str]:
    """Execute Programmer NFA as a Trinity step.

    Bridges Trinity's step context to call_programmer()'s interface.
    Oracle is constructed from environment (same as standalone usage).
    """
    from compass.llm.oracle import Oracle
    from compass.agents.programmer.tool import (
        call_programmer,
        create_pattern_fetcher,
        create_file_structure_getter,
        create_coding_standards_getter,
    )

    problem = resolved_inputs.get("problem", "") or step.artifact_ref or ""
    if not problem:
        return Err(f"step '{step.step_id}': programmer requires a 'problem' input")

    constraints = resolved_inputs.get("constraints", [])
    if isinstance(constraints, str):
        constraints = [c.strip() for c in constraints.split(",") if c.strip()]

    parent_feedback = resolved_inputs.get("feedback", None)

    oracle = Oracle()

    # Duck-type workspace ref for pattern fetcher (avoids importing CodeMemory)
    class _WsRef:
        def __init__(self, path):
            self.project_path = str(path)

    ws_ref = _WsRef(workspace)
    fetch_pattern = create_pattern_fetcher(oracle, ws_ref)
    get_file_structure = create_file_structure_getter(ws_ref)
    get_coding_standards = create_coding_standards_getter()

    def _apply_chunks(chunks):
        """Write chunks to workspace. Passed to Programmer so DELIVER writes files."""
        applied, failed = [], []
        for chunk in chunks:
            target = getattr(chunk, "target", None) or (chunk.get("target") if isinstance(chunk, dict) else "")
            content = getattr(chunk, "content", None) or (chunk.get("content") if isinstance(chunk, dict) else "")
            if not target or not content:
                failed.append(f"missing target or content")
                continue
            out = workspace / target
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content)
            applied.append(str(target))
        msg = f"Applied {len(applied)} chunks"
        if failed:
            msg += f", {len(failed)} failed: {failed}"
        return len(failed) == 0, msg

    try:
        result = call_programmer(
            oracle=oracle,
            problem=problem,
            constraints=constraints,
            fetch_pattern=fetch_pattern,
            get_file_structure=get_file_structure,
            get_coding_standards=get_coding_standards,
            apply_chunks=_apply_chunks,
            parent_feedback=parent_feedback,
        )
    except Exception as exc:
        return Err(
            f"step '{step.step_id}': programmer error: "
            f"{type(exc).__name__}: {exc}"
        )

    if not result.success:
        return Err(
            f"step '{step.step_id}': programmer failed: {result.error or 'unknown'}"
        )

    # Summary fact -- files are already on disk via _apply_chunks
    from dataclasses import asdict
    from enum import Enum

    def _default(o):
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    fact_value = json.dumps({
        "success": result.success,
        "solution_doc": result.solution_doc,
        "chunks": result.chunks or [],
        "reasoning": result.reasoning,
        "iterations": result.iterations,
    }, default=_default)

    return Ok(Fact(
        step_id=step.step_id,
        name=step.expected_fact,
        value=fact_value,
        fact_type="json",
    ))


# ============================================================================
# execute_step -- singledispatch on Step subtype
# ============================================================================


@singledispatch
def execute_step(step: Step, ctx: StepContext) -> Result:
    """Fallback: unresolved step type."""
    return Err(
        f"step '{step.step_id}': unresolved artifact_type '{step.artifact_type}' -- "
        f"no module found at artifacts/{step.artifact_type}.py"
    )


@execute_step.register(InlinePythonStep)
def _(step: InlinePythonStep, ctx: StepContext) -> Result:
    return _execute_python_code(
        step, step.artifact_ref, ctx.resolved_inputs, ctx.facts,
        workspace=ctx.workspace,
    )


@execute_step.register(ShellStep)
def _(step: ShellStep, ctx: StepContext) -> Result:
    return _execute_shell(step, ctx.resolved_inputs, ctx.facts)


@execute_step.register(ReadFileStep)
def _(step: ReadFileStep, ctx: StepContext) -> Result:
    return _execute_read_file(step, ctx.resolved_inputs, ctx.workspace)


@execute_step.register(WriteFileStep)
def _(step: WriteFileStep, ctx: StepContext) -> Result:
    content = ctx.resolved_inputs.get("content", "") or step.artifact_ref or ""
    target = ctx.resolved_inputs.get("path", "")
    if not target:
        return Err(f"step '{step.step_id}': write_file requires 'path' in inputs")

    target_path = Path(target) if os.path.isabs(target) else ctx.workspace / target
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        target_path.write_text(content)
    except Exception as exc:
        return Err(f"step '{step.step_id}': cannot write {target}: {exc}")

    return Ok(Fact(
        step_id=step.step_id,
        name=step.expected_fact,
        value=str(target_path),
        fact_type="text",
    ))


@execute_step.register(VisionStep)
def _(step: VisionStep, ctx: StepContext) -> Result:
    return _execute_vision_step(step, ctx.resolved_inputs)


@execute_step.register(ProgrammerStep)
def _(step: ProgrammerStep, ctx: StepContext) -> Result:
    return _execute_programmer(step, ctx.resolved_inputs, ctx.workspace)


@execute_step.register(DynamicStep)
def _(step: DynamicStep, ctx: StepContext) -> Result:
    module_path = Path(step.module_path)
    try:
        mod = _import_file_as_module(module_path, f"_trinity_artifact_{module_path.stem}")
    except Exception as exc:
        tb = _extract_error_location(exc)
        return Err(
            f"step '{step.step_id}': import error for '{module_path.name}': "
            f"{type(exc).__name__}: {exc}"
            + (f"\n{tb}" if tb else "")
        )

    run_fn = getattr(mod, "run", None)
    if run_fn is None or not callable(run_fn):
        return Err(f"step '{step.step_id}': '{module_path.name}' has no callable 'run'")

    try:
        return run_fn(step, ctx.resolved_inputs, ctx.workspace)
    except Exception as exc:
        tb = _extract_error_location(exc)
        return Err(
            f"step '{step.step_id}': run() error in '{module_path.name}': "
            f"{type(exc).__name__}: {exc}"
            + (f"\n{tb}" if tb else "")
        )


@execute_step.register(DiscoveredStep)
def _(step: DiscoveredStep, ctx: StepContext) -> Result:
    """Execute a discovered workspace artifact.

    Same contract as DynamicStep: import the module, call
    run(step, resolved_inputs, workspace). The python_script
    fallback executes the file as inline code.
    """
    filepath = ctx.workspace / step.module_path
    if not filepath.exists():
        return Err(f"step '{step.step_id}': artifact file not found: {step.module_path}")

    if step.kind == "python_script":
        return _execute_python_code(
            step, filepath.read_text(), ctx.resolved_inputs, ctx.facts,
            source_label=step.module_path, workspace=ctx.workspace,
        )

    try:
        mod = _import_file_as_module(filepath, f"_trinity_artifact_{step.step_id}")
    except Exception as exc:
        tb = _extract_error_location(exc)
        return Err(
            f"step '{step.step_id}': import error for '{step.module_path}': "
            f"{type(exc).__name__}: {exc}"
            + (f"\n{tb}" if tb else "")
        )

    run_fn = getattr(mod, step.entry_point, None)
    if run_fn is None or not callable(run_fn):
        return Err(
            f"step '{step.step_id}': '{step.module_path}' has no callable "
            f"'{step.entry_point}'"
        )

    logger.info("Calling %s.%s(step, resolved_inputs, workspace)", step.module_path, step.entry_point)

    try:
        return run_fn(step, ctx.resolved_inputs, ctx.workspace)
    except Exception as exc:
        tb = _extract_error_location(exc)
        return Err(
            f"step '{step.step_id}': {step.entry_point}() error in "
            f"'{Path(step.module_path).name}': {type(exc).__name__}: {exc}"
            + (f"\n{tb}" if tb else "")
        )


