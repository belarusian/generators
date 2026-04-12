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
import re
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
    AssertStep,
    DiscoveredStep,
    DynamicStep,
    EditFileStep,
    Fact,
    FileFact,
    InlinePythonStep,
    ReadFileStep,
    ShellStep,
    Step,
    VISION_ARTIFACT_REF_SCREEN,
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
    model_id: str = ""
    ask_fn: Any = None


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


def _coerce_to_type(value: Any, expected_type: str) -> Any:
    """Coerce a value to the expected Python type.

    Raises ValueError/TypeError on impossible coercions.
    """
    if expected_type == "any":
        return value
    if expected_type == "str":
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return value.get("summary", json.dumps(value))
        return str(value)
    if expected_type == "dict":
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            raise TypeError(f"JSON parsed to {type(parsed).__name__}, not dict")
        raise TypeError(f"cannot coerce {type(value).__name__} to dict")
    if expected_type == "list":
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            raise TypeError(f"JSON parsed to {type(parsed).__name__}, not list")
        raise TypeError(f"cannot coerce {type(value).__name__} to list")
    if expected_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
        return bool(value)
    if expected_type == "int":
        return int(value)
    if expected_type == "float":
        return float(value)
    return value


def _make_typed_fact(step: Step, value: Any) -> Result:
    """Create a Fact with type coercion and raw_value.

    Applies expected_type coercion, classifies, serializes.
    Replaces scattered Ok(Fact(...)) construction in handlers.
    """
    expected_type = getattr(step, "expected_type", "any")
    if expected_type != "any":
        try:
            value = _coerce_to_type(value, expected_type)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return Err(
                f"step '{step.step_id}': type contract '{expected_type}' "
                f"failed: {type(exc).__name__}: {exc}"
            )

    fact_type = _classify_value(value)
    serialized = _serialize_value(value)

    return Ok(Fact(
        step_id=step.step_id,
        name=step.expected_fact,
        value=serialized,
        fact_type=fact_type,
        raw_value=value,
    ))


def _import_file_as_module(filepath: Path, module_name: str | None = None) -> Any:
    """Import a Python file as a module using importlib."""
    if module_name is None:
        module_name = filepath.stem

    # Add parent dir so sibling imports work (e.g. artifacts/ importing each other)
    parent = str(Path(filepath).parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)

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


def _kwonly_args(func: Any, inputs: dict) -> dict:
    """Extract keyword-only args that func accepts from inputs dict."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return {}
    kwonly = {
        name for name, p in sig.parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    }
    return {k: v for k, v in inputs.items() if k in kwonly}


_READ_FILE_LIMIT = 200  # Lines; files under this are read in full


def _unwrap_string_assignment(text: Any) -> str:
    """Unwrap varname = '''...''' patterns the model writes in write_file banners.

    The model often wraps file content in a Python string assignment instead
    of writing it raw. Detect this and extract the inner string.
    """
    if not isinstance(text, str):
        return json.dumps(text) if isinstance(text, (dict, list)) else str(text)
    stripped = text.strip()
    # Match: <any_varname> = '''...''' or <any_varname> = \"\"\"...\"\"\"
    for quote in ("'''", '"""'):
        idx = stripped.find("= " + quote)
        if idx > 0 and stripped.endswith(quote) and idx + len("= " + quote) < len(stripped) - len(quote):
            inner = stripped[idx + len("= " + quote):-len(quote)]
            return inner
    return text


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

    # Inject accumulated fact values as top-level variables for convenience.
    # Allows inline code to reference prior step/session facts by name
    # without requiring explicit $fact input mappings.
    for _fn, _fv in namespace["_facts"].items():
        if _fn not in namespace:
            namespace[_fn] = _fv

    pre_exec_keys = set(namespace.keys())

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
        # Show what the code defined so the model can self-correct.
        # Structural feedback > prescriptive feedback (lesson 2):
        # show the topology, not the fix.
        user_vars = sorted(
            k for k in namespace
            if not k.startswith("_") and k not in pre_exec_keys
        )
        defined_summary = ", ".join(user_vars) if user_vars else "(none)"
        return Err(
            f"step '{step.step_id}': extraction error for "
            f"'{step.extraction_expr}': {type(exc).__name__}: {exc}\n"
            f"  variables defined by code: {defined_summary}\n"
            f"  extraction_expr must name a variable the code assigns to"
        )

    return _make_typed_fact(step, value)


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
    workspace: Path | None = None,
) -> Result[Fact, str]:
    """Execute a shell command step and extract a fact.

    All resolved inputs AND all accumulated facts are injected as
    environment variables. This means shell commands can reference
    any prior fact by its exact name: $formatted_comment, $pr_body, etc.
    No template engines, no regex -- just shell's own $VAR expansion.
    """
    cmd = step.artifact_ref or ""
    env = os.environ.copy()

    # Expose workspace so the model can reference it instead of guessing
    if workspace:
        env["WORKSPACE"] = str(workspace)

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
            cwd=str(workspace) if workspace else None,
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
        parts = []
        if stdout.strip():
            parts.append(f"stdout:\n{stdout.strip()}")
        if stderr.strip():
            parts.append(f"stderr:\n{stderr.strip()}")
        detail = "\n".join(parts) or "(no output)"
        return Err(
            f"step '{step.step_id}': shell command failed (exit {proc.returncode}):\n"
            f"{detail[:2000]}"
        )

    # Include stderr on success -- many tools (git, curl) write
    # useful output to stderr, not stdout.
    output = stdout.strip()
    if stderr.strip():
        output = output + "\n" + stderr.strip() if output else stderr.strip()

    return _make_typed_fact(step, output)


def _capture_screen_to_png(workspace: Path | None) -> Result[Path, str]:
    """Write a fresh screenshot to a temp path (neo-lab ``screen`` or pyautogui)."""
    import tempfile

    fd, path_str = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    out = Path(path_str)
    try:
        if workspace and (workspace / "neo" / "screen.py").is_file():
            root = str(workspace.resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            from neo import screen as neo_screen

            png = neo_screen.capture()
            out.write_bytes(png)
        else:
            import pyautogui

            pyautogui.screenshot().save(str(out))
    except Exception as exc:
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
        return Err(f"screen capture failed: {type(exc).__name__}: {exc}")
    return Ok(out)


def _execute_vision_step(
    step: Step,
    resolved_inputs: dict,
    workspace: Path | None = None,
) -> Result[Fact, str]:
    """Execute a vision step by sending an image to a vision model."""
    ref = (step.artifact_ref or "").strip()
    temp_capture = False
    if ref == VISION_ARTIFACT_REF_SCREEN:
        cap = _capture_screen_to_png(workspace)
        if isinstance(cap, Err):
            return Err(cap.error)
        image_path = cap.value
        temp_capture = True
    else:
        image_path = Path(ref) if os.path.isabs(ref) else (workspace / ref if workspace else Path(ref))
        if not image_path.exists():
            return Err(f"step '{step.step_id}': image not found: {step.artifact_ref}")

    try:
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

        return _make_typed_fact(step, value)
    finally:
        if temp_capture:
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                pass





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


@execute_step.register(AssertStep)
def _(step: AssertStep, ctx: StepContext) -> Result:
    inner = _execute_python_code(
        step, step.artifact_ref, ctx.resolved_inputs, ctx.facts,
        workspace=ctx.workspace,
    )
    if isinstance(inner, Err):
        return inner

    # Convention: {"ok": bool, "issues": [...]}
    raw = inner.value.value
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        value = raw

    if isinstance(value, dict):
        issues = value.get("issues", [])
        ok = value.get("ok", not issues)
        if not ok or issues:
            parts = [str(i) for i in issues] if issues else ["assertion failed"]
            return Err(f"step '{step.step_id}': {'; '.join(parts)}")
    elif not value:
        return Err(f"step '{step.step_id}': assertion failed (result was falsy)")

    return inner


@execute_step.register(ShellStep)
def _(step: ShellStep, ctx: StepContext) -> Result:
    return _execute_shell(step, ctx.resolved_inputs, ctx.facts, workspace=ctx.workspace)


@execute_step.register(ReadFileStep)
def _(step: ReadFileStep, ctx: StepContext) -> Result:
    return _execute_read_file(step, ctx.resolved_inputs, ctx.workspace)


def _extract_content(value: Any) -> str:
    """Extract file content from whatever the model produced.

    The model wraps content in various ways:
      - {"path": "...", "content": "actual stuff", ...}  -> extract content
      - '{"path": "...", "content": "..."}'  (JSON string) -> parse, extract
      - 'varname = \'\'\'...\'\'\'  -> unwrap string assignment
      - plain string -> use as-is
    """
    # Dict with a content key -- extract the payload
    if isinstance(value, dict) and "content" in value:
        return _extract_content(value["content"])

    if not isinstance(value, str):
        return json.dumps(value) if isinstance(value, (dict, list)) else str(value)

    # JSON string that parses to a dict with content key
    stripped = value.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "content" in parsed:
                return _extract_content(parsed["content"])
        except (json.JSONDecodeError, TypeError):
            pass

    return _unwrap_string_assignment(value)


@execute_step.register(WriteFileStep)
def _(step: WriteFileStep, ctx: StepContext) -> Result:
    content = ctx.resolved_inputs.get("content", "") or step.artifact_ref or ""
    content = _extract_content(content)
    target = ctx.resolved_inputs.get("path", "")
    if not target:
        return Err(f"step '{step.step_id}': write_file requires 'path' in inputs")

    target_path = Path(target) if os.path.isabs(target) else ctx.workspace / target

    # Syntax gate: refuse to write broken Python files.
    # Catches corrupted content (bad regex replacements, truncation)
    # before it overwrites a working file.
    if target_path.suffix == ".py":
        try:
            import ast as _ast
            _ast.parse(content)
        except SyntaxError as exc:
            loc = f" at line {exc.lineno}" if exc.lineno else ""
            return Err(
                f"step '{step.step_id}': refusing to write {target} -- "
                f"content has SyntaxError{loc}: {exc.msg}"
            )

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        target_path.write_text(content)
    except Exception as exc:
        return Err(f"step '{step.step_id}': cannot write {target}: {exc}")

    return _make_typed_fact(step, {"path": str(target_path), "content": content})


def _find_target_by_lines(target: str, file_content: str):
    """Find target in file by matching stripped lines.

    Models get indentation wrong (content blocks shift it) and add trailing
    whitespace.  This compares stripped lines and returns the file's actual
    version only when the match is unique.

    Ported from Neo's file_editor._find_target_by_lines.
    """
    target_lines = [ln.strip() for ln in target.strip().splitlines() if ln.strip()]
    if not target_lines:
        return None

    file_lines = file_content.splitlines()
    matches = []

    for i in range(len(file_lines) - len(target_lines) + 1):
        window = [file_lines[i + j].strip() for j in range(len(target_lines))]
        if window == target_lines:
            matches.append("\n".join(file_lines[i : i + len(target_lines)]))

    return matches[0] if len(matches) == 1 else None


@execute_step.register(EditFileStep)
def _(step: EditFileStep, ctx: StepContext) -> Result:
    ref = step.artifact_ref or ""
    path = Path(ref) if os.path.isabs(ref) else ctx.workspace / ref

    if not path.exists():
        return Err(f"step '{step.step_id}': file not found: {ref}")

    old_string = ctx.resolved_inputs.get("old_string", "")
    new_string = ctx.resolved_inputs.get("new_string", "")

    if not old_string:
        return Err(f"step '{step.step_id}': edit_file requires non-empty 'old_string'")

    # No-op: old and new are the same.  Treat as success (conditional edit
    # where upstream decided no change was needed).
    if old_string == new_string:
        return _make_typed_fact(step, {"path": str(path), "no_change": True})

    try:
        content = path.read_text()
    except Exception as exc:
        return Err(f"step '{step.step_id}': cannot read {ref}: {exc}")

    count = content.count(old_string)

    # Fallback 1: strip "line N: " prefix (model copies from display output)
    if count == 0:
        stripped = re.sub(r"^line \d+: ", "", old_string)
        if stripped != old_string and content.count(stripped) == 1:
            old_string = stripped
            count = 1

    # Fallback 2: rstrip trailing whitespace
    if count == 0:
        rstripped = old_string.rstrip()
        if rstripped != old_string and content.count(rstripped) == 1:
            old_string = rstripped
            count = 1

    # Fallback 3: line-by-line stripped match (handles indentation from content blocks)
    if count == 0:
        found = _find_target_by_lines(old_string, content)
        if found:
            old_string = found
            count = 1

    if count == 0:
        return Err(
            f"step '{step.step_id}': old_string not found in {ref}. "
            f"Searched for: {old_string[:200]!r}"
        )
    if count > 1:
        return Err(
            f"step '{step.step_id}': old_string matches {count} times in {ref} -- "
            f"must be unique. Provide more surrounding context."
        )

    updated = content.replace(old_string, new_string, 1)

    # Syntax gate for Python files
    if path.suffix == ".py":
        try:
            import ast as _ast
            _ast.parse(updated)
        except SyntaxError as exc:
            loc = f" at line {exc.lineno}" if exc.lineno else ""
            return Err(
                f"step '{step.step_id}': edit would produce SyntaxError{loc}: "
                f"{exc.msg} -- not writing"
            )

    try:
        path.write_text(updated)
    except Exception as exc:
        return Err(f"step '{step.step_id}': cannot write {ref}: {exc}")

    return _make_typed_fact(step, {"path": str(path), "old": old_string, "new": new_string})


@execute_step.register(VisionStep)
def _(step: VisionStep, ctx: StepContext) -> Result:
    return _execute_vision_step(step, ctx.resolved_inputs, workspace=ctx.workspace)




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
        return run_fn(step, ctx.resolved_inputs, ctx.workspace,
                      **_kwonly_args(run_fn, ctx.resolved_inputs))
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

    logger.info("Calling %s.%s(step, resolved_inputs, workspace, **kwargs)", step.module_path, step.entry_point)

    try:
        return run_fn(step, ctx.resolved_inputs, ctx.workspace,
                      **_kwonly_args(run_fn, ctx.resolved_inputs))
    except Exception as exc:
        tb = _extract_error_location(exc)
        return Err(
            f"step '{step.step_id}': {step.entry_point}() error in "
            f"'{Path(step.module_path).name}': {type(exc).__name__}: {exc}"
            + (f"\n{tb}" if tb else "")
        )
